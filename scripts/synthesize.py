# ---------------------------------------------------------------------------
# Produces: main image synthesis driver
#
# Provenance: diffusion-posterior-sampling/pixelbp_prodigy_depleted.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, CHECKPOINTS, DTD_IMAGES, DTD_LABELS# noqa: E402
# ==============================================================================
# Depletion runs with Prodigy, mirroring pixelbp_prodigy.py optimization config.
# Per-run config via env vars (so you can sweep without editing the file):
#   TOP_N_STATS=N        keep the top-N compressor-weighted stats   (default: full = 28929)
#   TOP_RAND_STATS=N     keep N random stats                        (mutually exclusive with TOP_N_STATS)
#   BOTTOM=1             flip TOP_N_STATS to use bottom-weighted instead
#   BATCH_SIZE=B         override per-GPU batch size                (default 36, matches baseline)
#   NUM_ITERATIONS=I     override iter count                        (default 20000, matches baseline)
#
# Examples:
#   TOP_N_STATS=10000 .venv/bin/python pixelbp_prodigy_depleted.py
#   TOP_N_STATS=1000  .venv/bin/python pixelbp_prodigy_depleted.py
#   TOP_RAND_STATS=1000 .venv/bin/python pixelbp_prodigy_depleted.py
#   TOP_N_STATS=1000 BOTTOM=1 .venv/bin/python pixelbp_prodigy_depleted.py
# ==============================================================================
import os

# Set BEFORE matplotlib import: non-interactive Agg backend, threadsafer.
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import time
import cv2
import csv
import sys
import threading
import traceback
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# --- Run config (env-overridable) ---
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 36))
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", 20000))
BOUNDARY_MODE = os.environ.get("BOUNDARY_MODE", "wrap")   # 'wrap' or 'black'
PRODIGY_LR = float(os.environ.get("PRODIGY_LR", "0.04"))
RUN_TAG = os.environ.get("RUN_TAG", "")                    # appended to output dir name
TOTAL_STATS = 28929
TOP_N_STATS = int(os.environ["TOP_N_STATS"]) if "TOP_N_STATS" in os.environ else None
TOP_RAND_STATS = int(os.environ["TOP_RAND_STATS"]) if "TOP_RAND_STATS" in os.environ else None
BOTTOM = os.environ.get("BOTTOM", "0") == "1"
COMPRESSED = os.environ.get("COMPRESSED", "0") == "1"   # match against trained compressor
COMPRESSOR_CKPT = os.environ.get(
    "COMPRESSOR_CKPT", str(CHECKPOINTS / "bottleneck100.pth"))
COMPRESSOR_DIM = int(os.environ.get("COMPRESSOR_DIM", "100"))   # for suffix
DROP_FAMILY = os.environ.get("DROP_FAMILY")              # single family name (legacy)
DROP_FAMILIES = os.environ.get("DROP_FAMILIES")          # comma-separated list of families to drop
KEEP_FAMILIES = os.environ.get("KEEP_FAMILIES")          # comma-separated list of families to keep (drop everything else)
SPLIT_FILE = os.environ.get("SPLIT_FILE", "val1_3sampled.txt")

modes_set = sum([
    TOP_N_STATS is not None, TOP_RAND_STATS is not None,
    COMPRESSED, DROP_FAMILY is not None,
    DROP_FAMILIES is not None, KEEP_FAMILIES is not None,
])
if modes_set > 1:
    raise SystemExit("Set at most one of TOP_N_STATS / TOP_RAND_STATS / COMPRESSED / DROP_FAMILY / DROP_FAMILIES / KEEP_FAMILIES")

# Held at 1.0 deliberately. Prodigy auto-tunes step size, so per-N loss-magnitude
# differences self-correct via its `d` adaptation. Don't reintroduce a stat-count
# scaling factor here — the previous MADGRAD run did, which conflated convergence
# dynamics across depletion levels.
LOSS_SCALER = 1.0

# Matplotlib pyparsing race guard — pyplot global state + log-axis mathtext
# blew up ~36% of plot saves under ThreadPoolExecutor in the main run.
_PLOT_LOCK = threading.Lock()

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# Disable thread oversubscription (keeps GPU workers from fighting for CPU cores).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(0)

# The original required the CWD to be the working-tree root and wrote synths
# beneath it. Outputs now go to the asset tree regardless of where you run from.
DPS_ROOT = ASSETS
ADJACENT = (DPS_ROOT / "../StatTexNetExtended").resolve()
if str(ADJACENT) not in sys.path:
    sys.path.insert(0, str(ADJACENT))
from utils import stattexgramnet as stn


# ==============================================================================
# Stat-selection: importance ranking from the trained compressor checkpoint
# ==============================================================================
def build_stat_indices():
    """
    Returns (indices_tensor_or_None, suffix_str).
    indices_tensor=None means: keep all stats (no depletion). For COMPRESSED
    mode the model itself returns the 100-d compressed vector, so we also
    return None here and skip selection downstream.
    """
    if COMPRESSED:
        return None, f"compressed{COMPRESSOR_DIM}"

    if DROP_FAMILY is not None or DROP_FAMILIES is not None or KEEP_FAMILIES is not None:
        # Pull family labels from a CPU-side instance (cheap; doesn't allocate filters at GPU).
        m = stn.StatTexGramNet(img_size=(256, 256), batch_size=1,
                               device=torch.device("cpu"), compress=False,
                               color=True, add_marginals=True, oris=4,
                               edge_levels=6, edge_start=1,
                               include_dphase=False, boundary_mode=BOUNDARY_MODE)
        groups = np.array(m.stat_groups_gram_label)
        uniq = set(groups.tolist())

        if KEEP_FAMILIES is not None:
            wanted = [f.strip() for f in KEEP_FAMILIES.split(",") if f.strip()]
            for f in wanted:
                if f not in uniq:
                    raise SystemExit(f"KEEP_FAMILIES entry '{f}' not in {sorted(uniq)}")
            keep_mask = np.isin(groups, wanted)
            tag = "keep_" + "_".join(wanted)
        else:
            drop_list = ([DROP_FAMILY] if DROP_FAMILY else
                         [f.strip() for f in DROP_FAMILIES.split(",") if f.strip()])
            for f in drop_list:
                if f not in uniq:
                    raise SystemExit(f"DROP entry '{f}' not in {sorted(uniq)}")
            keep_mask = ~np.isin(groups, drop_list)
            tag = "drop_" + "_".join(drop_list)

        idx = np.where(keep_mask)[0].astype(np.int64)
        suffix = tag
        print(f"[family-select] {suffix}: keeping {len(idx)}/{len(groups)}")
        return torch.from_numpy(idx), suffix

    if TOP_RAND_STATS is not None:
        rng = np.random.default_rng(0)  # deterministic across reruns
        idx = rng.choice(TOTAL_STATS, size=TOP_RAND_STATS, replace=False)
        return torch.from_numpy(np.sort(idx)), f"rand{TOP_RAND_STATS}"

    if TOP_N_STATS is None or TOP_N_STATS >= TOTAL_STATS:
        return None, "full"

    weights_path = str(CHECKPOINTS / "bottleneck100.pth")
    ckpt = torch.load(weights_path, map_location="cpu")
    w = ckpt["state_dict"]["w"].T.data.cpu().numpy()  # (stats, bottleneck)
    importance = np.sum(np.abs(w), axis=0)            # per-stat |w| sum

    if BOTTOM:
        order = np.argsort(importance)               # least → most
        suffix = f"bot{TOP_N_STATS}"
    else:
        order = np.argsort(importance)[::-1]         # most → least
        suffix = f"top{TOP_N_STATS}"

    idx = np.sort(order[:TOP_N_STATS].copy())
    return torch.from_numpy(idx), suffix


STAT_INDICES, RUN_SUFFIX = build_stat_indices()
print(f"[depleted-prodigy] selection={RUN_SUFFIX}  "
      f"kept={'all' if STAT_INDICES is None else len(STAT_INDICES)}/{TOTAL_STATS}")


# ==============================================================================
# Resume + split helpers (unchanged from baseline)
# ==============================================================================
def get_completed_images(csv_path):
    if not csv_path.is_file():
        return set()
    completed = set()
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "success" and "image_name" in row:
                    completed.add(row["image_name"])
    except Exception as e:
        print(f"⚠️ Could not parse results CSV {csv_path}. Starting fresh. Error: {e}")
        return set()
    return completed


def get_split_filenames(dataset_root, split_file_name=None):
    split_file_name = split_file_name or SPLIT_FILE
    split_file_path = Path(f"{DTD_LABELS}/{split_file_name}")
    if not split_file_path.is_file():
        raise FileNotFoundError(f"Split file not found at: {split_file_path}")
    with open(split_file_path) as f:
        relative_paths = [line.strip() for line in f if line.strip()]
    image_files = [dataset_root / p for p in relative_paths]
    existing = [p for p in image_files if p.is_file()]
    if len(existing) != len(image_files):
        print(f"Warning: {len(relative_paths)} paths in split, {len(existing)} on disk.")
    return existing


# ==============================================================================
# Synthesizer
# ==============================================================================
class StatTexSynthesizer:
    def __init__(self, device="cuda:0", stat_indices=None):
        print(f"Initializing StatTexSynthesizer on {device}")
        self.device = torch.device(device)
        self.img_size = 256
        self.stat_indices = (
            stat_indices.to(self.device) if stat_indices is not None else None
        )

        # NOTE on COMPRESSED: model.stats_size is stale (28923) so allocating the
        # built-in compressor with compress=True fails; the actual forward output
        # is 28929-d. We sidestep this by keeping compress=False here and applying
        # the loaded _w as an external matmul.
        self.statnet_model = stn.StatTexGramNet(
            img_size=(self.img_size, self.img_size),
            batch_size=1,
            device=self.device,
            compress=False,
            color=True,
            add_marginals=True,
            oris=4,
            edge_levels=6,
            edge_start=1,
            include_dphase=False,
            boundary_mode=BOUNDARY_MODE,
        ).to(self.device).eval()

        if COMPRESSED:
            ckpt = torch.load(COMPRESSOR_CKPT, map_location=self.device)
            self._compress_w = ckpt["state_dict"]["_w"].to(self.device)
            print(f"  loaded compressor _w from {COMPRESSOR_CKPT}: "
                  f"{tuple(self._compress_w.shape)}")
            self.compute_stats = lambda x: torch.matmul(
                self.statnet_model.forward(x), self._compress_w)
        else:
            self.compute_stats = self.statnet_model.forward
        print("StatTexSynthesizer init complete.")

    def _select(self, stats):
        if self.stat_indices is None:
            return stats
        return stats[:, self.stat_indices]

    def load_and_preprocess_image(self, image_path):
        ref = cv2.imread(str(image_path))
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        ref = (
            torch.tensor(ref).permute(2, 0, 1).contiguous().unsqueeze(0).float()
            .to(self.device) / 255.0
        )
        _, _, h, w = ref.shape
        size = min(h, w)
        s_h, s_w = (h - size) // 2, (w - size) // 2
        ref = ref[:, :, s_h:s_h + size, s_w:s_w + size]
        return F.interpolate(ref, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)

    def synthesize_texture_batch(self, reference_paths,
                                 num_iterations=NUM_ITERATIONS, report_every=100):
        """
        Optimization config matches pixelbp_prodigy.py exactly:
          - Prodigy(lr=0.04, betas=(0.99, 0.999), d0=1e-2,
                    safeguard_warmup=True, use_bias_correction=True)
          - clip_grad_norm_(x_batch, max_norm=10.0)
          - x_batch ~ U[0,1]; clamp_(0, 1) each step
          - per-image MSE = mean over (selected) stats; total = mean over batch
          - best_x tracked by the *optimized* loss (i.e., depleted MSE)
        """
        from prodigyopt import Prodigy

        B = len(reference_paths)
        print(f"Synthesizing batch of {B} images... (selection={RUN_SUFFIX})")

        # 1. Load refs and target stats; select once.
        ref_imgs = [self.load_and_preprocess_image(p) for p in reference_paths]
        ref_batch = torch.cat(ref_imgs, dim=0)
        with torch.no_grad():
            target_full = self.compute_stats(ref_batch)
            target_stats = self._select(target_full)
        print(f"  target_stats shape: {tuple(target_stats.shape)}")

        H, W = self.img_size, self.img_size
        # Uniform [0,1] init — matches the (sigmoid-removed) model's expected input
        # range and the optimizer A/B sweep that produced the baseline config.
        x_batch = torch.rand(B, 3, H, W, device=self.device, requires_grad=True)

        optimizer = Prodigy(
            [x_batch],
            lr=PRODIGY_LR,
            betas=(0.99, 0.999),
            d0=1e-2,
            safeguard_warmup=True,
            use_bias_correction=True,
        )

        best_x = x_batch.detach().clone()
        best_loss = torch.full((B,), float("inf"), device=self.device)
        loss_hist = [[] for _ in range(B)]

        def _per_img_losses(X):
            stats = self._select(self.compute_stats(X))
            per_img = F.mse_loss(stats, target_stats, reduction="none")
            return per_img.flatten(1).mean(1) * LOSS_SCALER

        print("Starting optimization...")
        for it in range(num_iterations):
            optimizer.zero_grad()
            per_img = _per_img_losses(x_batch)
            total_loss = per_img.mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(x_batch, max_norm=10.0)
            optimizer.step()

            with torch.no_grad():
                if torch.isnan(x_batch).any():
                    print(f"  Iter {it}: NaN in x_batch, resetting to best_x")
                    x_batch.data = best_x.clone()
                    continue
                x_batch.clamp_(0, 1)

            if it % report_every == 0 or it == num_iterations - 1:
                with torch.no_grad():
                    cur = _per_img_losses(x_batch)
                    for i in range(B):
                        loss_hist[i].append(cur[i].item())
                    improved = cur < best_loss
                    best_loss[improved] = cur[improved]
                    best_x[improved] = x_batch[improved]
                print(f"  Iter {it:5d}: Mean Loss = {cur.mean().item():.3e}", flush=True)

        finals = best_loss.cpu().tolist()
        return [
            {"tensor": best_x[i:i + 1].clone(),
             "losses": loss_hist[i],
             "final_loss": finals[i]}
            for i in range(B)
        ]

    def save_synthesized_image(self, tensor, output_path):
        with torch.no_grad():
            arr = tensor[0].permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr, 0, 1)
        Image.fromarray((arr * 255).astype(np.uint8)).save(output_path)

    def save_loss_plot(self, losses, output_path, image_name):
        # Lock + Agg + ScalarFormatter + OO API: avoids the pyparsing race that
        # killed ~36% of plot saves in the previous threaded run.
        with _PLOT_LOCK:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(losses, alpha=0.5, label="Raw Loss")
            if len(losses) > 20:
                window = min(20, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
                ax.plot(np.arange(window - 1, len(losses)), smoothed,
                        color="red", label="Smoothed")
            ax.set_title(f"Loss Curve - {image_name}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("MSE Loss")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())  # avoids mathtext path
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=100)
            plt.close(fig)


# ==============================================================================
# Per-GPU worker
# ==============================================================================
class BatchProcessor:
    def __init__(self, output_dir, gpu_device, gpu_id, stat_indices):
        self.output_dir = Path(output_dir)
        self.gpu_device = gpu_device
        self.gpu_id = gpu_id
        self.synthesizer = StatTexSynthesizer(device=gpu_device,
                                              stat_indices=stat_indices)

    def process_batch(self, image_paths, batch_num):
        print(f"\nGPU {self.gpu_id}: Batch {batch_num} ({len(image_paths)} images)")
        t0 = time.time()
        results = self.synthesizer.synthesize_texture_batch(image_paths)

        rows = []
        for img_path, r in zip(image_paths, results):
            try:
                category = img_path.parent.name
                prefix = f"{img_path.stem}_gpu{self.gpu_id}"
                synth_path = self.output_dir / f"{prefix}_synth.png"
                loss_path = self.output_dir / f"{prefix}_lossgraph.png"

                self.synthesizer.save_synthesized_image(r["tensor"], synth_path)
                self.synthesizer.save_loss_plot(r["losses"], loss_path, img_path.name)

                init_loss = r["losses"][0] if r["losses"] else 0.0
                rows.append({
                    "status": "success",
                    "image_name": img_path.name,
                    "category": category,
                    "original_path": str(img_path),
                    "synthesized_path": str(synth_path),
                    "loss_plot_path": str(loss_path),
                    "final_loss": r["final_loss"],
                    "initial_loss": init_loss,
                    "iterations": len(r["losses"]),
                    "gpu_device": self.gpu_device,
                    "selection": RUN_SUFFIX,
                    "error": "",
                })
                print(f"  ✓ [{self.gpu_device}] {img_path.name:<25} "
                      f"final={r['final_loss']:.3e}")
            except Exception as e:
                print(f"  ✗ [{self.gpu_device}] {img_path.name}: {e}")
                rows.append({
                    "status": "error",
                    "image_name": img_path.name,
                    "category": img_path.parent.name,
                    "original_path": str(img_path),
                    "synthesized_path": "",
                    "loss_plot_path": "",
                    "final_loss": "",
                    "initial_loss": "",
                    "iterations": 0,
                    "gpu_device": self.gpu_device,
                    "selection": RUN_SUFFIX,
                    "error": str(e),
                })

        dt = time.time() - t0
        print(f"GPU {self.gpu_id} batch {batch_num} done in {dt:.1f}s "
              f"({dt / max(len(image_paths), 1):.1f}s/img)")
        return rows


def process_on_single_gpu(config, image_files, gpu_device, gpu_id, stat_indices):
    print(f"\n🚀 GPU {gpu_id} ({gpu_device}) starting on {len(image_files)} images.")
    try:
        proc = BatchProcessor(config["output_dir"], gpu_device, gpu_id, stat_indices)
    except Exception:
        print(f"❌ GPU {gpu_id} init failed:\n{traceback.format_exc()}")
        return []

    all_rows = []
    bs = config["batch_size"]
    for i in range(0, len(image_files), bs):
        batch_paths = image_files[i:i + bs]
        rows = proc.process_batch(batch_paths, f"{gpu_id}-{i // bs + 1}")
        all_rows.extend(rows)
        save_results_to_csv(config["results_csv_path"], rows)
    return all_rows


# ==============================================================================
# CSV writer (fcntl + threading.Lock fallback, QUOTE_ALL, sanitized fields)
# ==============================================================================
FIELDNAMES = [
    "status", "image_name", "category", "original_path", "synthesized_path",
    "loss_plot_path", "final_loss", "initial_loss", "iterations", "gpu_device",
    "selection", "error",
]


def _clean_csv_field(v):
    if v is None:
        return ""
    s = str(v).replace("\r", " ").replace("\n", " ").replace('"', "'")
    if len(s) > 2000:
        s = s[:2000] + "…"
    return s


def save_results_to_csv(filepath, rows):
    if not rows:
        return
    write_header = not filepath.exists()
    lock = None
    if not HAS_FCNTL:
        if not hasattr(save_results_to_csv, "_lock"):
            save_results_to_csv._lock = threading.Lock()
        lock = save_results_to_csv._lock
    try:
        if lock:
            lock.acquire()
        with open(filepath, "a", newline="") as f:
            if HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.DictWriter(
                f, fieldnames=FIELDNAMES, extrasaction="ignore",
                quoting=csv.QUOTE_ALL, escapechar="\\",
                lineterminator="\n", doublequote=True,
            )
            if write_header:
                writer.writeheader()
            safe = [{k: _clean_csv_field(v) for k, v in r.items()} for r in rows]
            writer.writerows(safe)
            if HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"CSV write error: {e}")
    finally:
        if lock:
            lock.release()


# ==============================================================================
# Main
# ==============================================================================
def main():
    output_dir = Path(
        f"{ASSETS}/synths/"
        f"pixelbp_prodigy_depleted_{RUN_SUFFIX}{RUN_TAG}"
    )
    config = {
        "dataset_path": DTD_IMAGES,
        "output_dir": output_dir,
        "batch_size": BATCH_SIZE,
        "gpu_devices": ["cuda:0", "cuda:1"],
    }
    config["results_csv_path"] = config["output_dir"] / "synthesis_results.csv"
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    print(f"[depleted-prodigy] output_dir = {config['output_dir']}")
    print(f"[depleted-prodigy] BATCH_SIZE={BATCH_SIZE} NUM_ITERATIONS={NUM_ITERATIONS}")

    image_files = get_split_filenames(config["dataset_path"])   # respects SPLIT_FILE env-var
    print(f"Found {len(image_files)} images in {SPLIT_FILE} split.")
    if not image_files:
        return

    completed = get_completed_images(config["results_csv_path"])
    if completed:
        print(f"Resuming: {len(completed)} already complete in this output_dir.")
        image_files = [p for p in image_files if p.name not in completed]
        print(f"  → {len(image_files)} remaining.")
    if not image_files:
        print("🎉 All images already complete for this run.")
        return

    mid = len(image_files) // 2
    splits = {
        config["gpu_devices"][0]: image_files[:mid],
        config["gpu_devices"][1]: image_files[mid:],
    }

    all_rows = []
    with ThreadPoolExecutor(max_workers=len(config["gpu_devices"])) as ex:
        futures = [
            ex.submit(process_on_single_gpu, config, files, dev, i, STAT_INDICES)
            for i, (dev, files) in enumerate(splits.items())
        ]
        for fut in futures:
            all_rows.extend(fut.result())

    print(f"\n{'=' * 60}\nDONE. Processed {len(all_rows)} images "
          f"for selection={RUN_SUFFIX}.")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
    print("Done!")
