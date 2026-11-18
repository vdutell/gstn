# ---------------------------------------------------------------------------
# Produces: Figure 9
#
# Provenance: diffusion-posterior-sampling/make_figure9_lbfgs_madgrad_prodigy.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Figure 9: side-by-side qualitative comparison of L-BFGS, MADGRAD, Prodigy
syntheses against the original target across 8 DTD categories.

Picks one image per category from val1_3sampled (which is what the existing
MADGRAD-28929 run covered), so MADGRAD and Prodigy syntheses can be reused
from disk. L-BFGS is run on-demand here (no existing outputs).

Layout: 4 rows (target / L-BFGS / MADGRAD / Prodigy) x 8 cols (categories).

Output:
    lbfgs_figure9_synths/<stem>_lbfgs_synth.png    L-BFGS results (cached)
    fig_optimizer_comparison.png                   the figure
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, DTD_IMAGES, DTD_LABELS, FIGURES# noqa: E402
import csv
import os
import random
import sys
import time
from pathlib import Path

# Force Agg before pyplot import (we don't need a display, and the matplotlib
# pyparsing race we kept getting bit by likes Agg)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

DPS_ROOT = ASSETS
TARGETS_DIR = DPS_ROOT / "pixelbp_prodigy_targets"
PRODIGY_DIR = DPS_ROOT / "pixelbp_prodigy_synths"
MADGRAD_DIR = DPS_ROOT / "synths" / "pixelbp_madgrad_synths_28929"
LBFGS_DIR = DPS_ROOT / "lbfgs_figure9_synths"
OUT_FIG = FIGURES / "fig_optimizer_comparison.png"

CATEGORIES = ["bumpy", "veined", "fibrous", "scaly",
              "honeycombed", "lined", "cracked", "polka-dotted"]
SEED = 42
GPU = os.environ.get("GPU", "cuda:1")    # share with the running sweep

NUM_ITERATIONS_LBFGS = 100               # ab_test_optimizers used 1000; LBFGS converges fast
LBFGS_LR = 0.5                           # matches ab_test_optimizers.ipynb
LBFGS_MAX_ITER = 100
LBFGS_HISTORY = 20

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(0)

ADJACENT = (DPS_ROOT / "../StatTexNetExtended").resolve()
if str(ADJACENT) not in sys.path:
    sys.path.insert(0, str(ADJACENT))


# ---- Pick 8 images from val1_3sampled, one per category ----
def load_split():
    p = Path(str(DTD_LABELS / "val1_3sampled.txt"))
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    by_cat = {c: [] for c in CATEGORIES}
    for line in lines:
        cat = line.split("/", 1)[0]
        if cat in by_cat:
            by_cat[cat].append(line)
    return by_cat


def pick_one_per_category():
    rng = random.Random(SEED)
    by_cat = load_split()
    picks = {}
    for c in CATEGORIES:
        if not by_cat[c]:
            print(f"⚠️  no candidates for {c} in val1_3sampled")
            continue
        picks[c] = sorted(by_cat[c])[0]   # deterministic — first in sorted order
        # Re-roll if you want random: picks[c] = rng.choice(sorted(by_cat[c]))
    print("[fig9] picks:")
    for c, p in picks.items():
        print(f"  {c:14s} -> {p}")
    return picks


# ---- L-BFGS synthesis (single image at a time) ----
def load_and_preprocess(path, device, size=256):
    """Identical preprocessing to pixelbp_prodigy.py."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (
        torch.tensor(img).permute(2, 0, 1).contiguous().unsqueeze(0).float()
        .to(device) / 255.0
    )
    _, _, h, w = img.shape
    s = min(h, w)
    sh, sw = (h - s) // 2, (w - s) // 2
    img = img[:, :, sh:sh + s, sw:sw + s]
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)


def lbfgs_synthesize(ref_path, model, device, num_iters=NUM_ITERATIONS_LBFGS):
    """Run L-BFGS on a single image, return best-loss 256x256x3 ndarray in [0,1].

    Matches ab_test_optimizers.ipynb exactly:
      - x init: torch.rand_like(ref_img), requires_grad
      - target_stats from raw ref_img (no sigmoid)
      - LBFGS step on closure that returns MSE(stats(x), target)
      - in-place clamp x.clamp_(0,1) AFTER optimizer.step (outside closure)
      - track best_x by per-step current loss; return best
    """
    ref_img = load_and_preprocess(ref_path, device)
    with torch.no_grad():
        target_stats = model.forward(ref_img)

    torch.manual_seed(SEED)
    x = torch.rand_like(ref_img).requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x],
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=LBFGS_HISTORY,
        line_search_fn="strong_wolfe",
    )

    losses = []
    best_loss = float("inf")
    best_x = None

    def closure():
        optimizer.zero_grad()
        stats = model.forward(x)
        loss = F.mse_loss(stats, target_stats)
        if not torch.isfinite(loss):
            losses.append(float("nan"))
            return torch.tensor(1e10, device=x.device)
        loss.backward()
        if x.grad is not None and not torch.isfinite(x.grad).all():
            x.grad.zero_()
        losses.append(loss.item())
        return loss

    for _ in range(num_iters):
        optimizer.step(closure)
        with torch.no_grad():
            x.clamp_(0, 1)
        cur = losses[-1] if losses else float("inf")
        if cur < best_loss:
            best_loss = cur
            best_x = x.detach().clone()

    final = best_x if best_x is not None else x.detach().clone()
    with torch.no_grad():
        final_loss = F.mse_loss(model.forward(final), target_stats).item()
    arr = final[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(arr, 0, 1), final_loss


def run_lbfgs_for_picks(picks):
    LBFGS_DIR.mkdir(parents=True, exist_ok=True)
    import torch
    from utils import stattexgramnet as stn
    device = torch.device(GPU)
    print(f"[fig9] L-BFGS device={GPU}, {NUM_ITERATIONS_LBFGS} iters per image")

    model = stn.StatTexGramNet(
        img_size=(256, 256), batch_size=1, device=device,
        compress=False, color=True, add_marginals=True, oris=4,
        edge_levels=6, edge_start=1, include_dphase=False, boundary_mode="wrap",
    ).to(device).eval()

    for c, rel in picks.items():
        stem = Path(rel).stem
        out = LBFGS_DIR / f"{stem}_lbfgs_synth.png"
        if out.exists():
            print(f"  [{c}] cached -> {out.name}")
            continue
        ref_path = DTD_IMAGES / rel
        t0 = time.time()
        arr, final_loss = lbfgs_synthesize(ref_path, model, device)
        dt = time.time() - t0
        Image.fromarray((arr * 255).astype(np.uint8)).save(out)
        print(f"  [{c}] {stem} done in {dt:.1f}s  final_loss={final_loss:.3e}")


# ---- Resolve existing Prodigy/MADGRAD synth paths (file may have _gpuN_) ----
def find_synth(dir_, stem, optimizer_name):
    for p in sorted(dir_.glob(f"{stem}_gpu*_synth.png")):
        return p
    raise FileNotFoundError(f"no {optimizer_name} synth for {stem} in {dir_}")


# ---- Build the figure ----
def build_figure(picks):
    n = len(picks)
    rows = ["target", "L-BFGS", "MADGRAD", "Prodigy"]

    fig, axes = plt.subplots(len(rows), n, figsize=(2.0 * n, 2.0 * len(rows) + 0.4))
    for col, c in enumerate(CATEGORIES):
        if c not in picks:
            for r in range(len(rows)):
                axes[r, col].axis("off")
            continue
        stem = Path(picks[c]).stem

        target = TARGETS_DIR / f"{stem}.png"
        lbfgs = LBFGS_DIR / f"{stem}_lbfgs_synth.png"
        madgrad = find_synth(MADGRAD_DIR, stem, "MADGRAD")
        prodigy = find_synth(PRODIGY_DIR, stem, "Prodigy")

        for r, path in enumerate([target, lbfgs, madgrad, prodigy]):
            axes[r, col].imshow(Image.open(path))
            axes[r, col].axis("off")
        axes[0, col].set_title(c, fontsize=12)

    # Row labels on the leftmost column
    for r, label in enumerate(rows):
        axes[r, 0].text(-0.10, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.02,
                        wspace=0.04, hspace=0.06)
    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[fig9] saved {OUT_FIG}")


def main():
    picks = pick_one_per_category()
    run_lbfgs_for_picks(picks)
    build_figure(picks)


if __name__ == "__main__":
    main()
