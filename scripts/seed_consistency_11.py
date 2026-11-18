# ---------------------------------------------------------------------------
# Produces: five-seed synthesis run behind Table 2 and Figure 12
#
# Provenance: diffusion-posterior-sampling/seed_consistency_11.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Seed-consistency study over the 11 Figure-2 textures (NeurIPS rebuttal, 3Kqw).

Supersedes the 6-texture run. Synthesizes each of the 11 Figure-2 stems from 5
independent random initializations under the headline GSTN-zpad config
(ENCODER_MODE=mean_only, boundary_mode="black", Prodigy lr=0.02 with
betas=(0.99,0.999), d0=1e-2, safeguard_warmup=True, use_bias_correction=True,
grad-clip 10.0 — the actual production config from the run .sh scripts; 20000 iters),
in a single batch (B=55). Then computes, per synth, DISTS/LPIPS/SSIM/MS-SSIM vs the
256x256 target, plus mean pairwise synth-vs-synth DISTS across seeds per stem.

Outputs (synths/seed_consistency_11/):
  <stem>_seed<k>_synth.png, final_losses.csv, per_synth_metrics.csv, per_stem_summary.csv

Run: GPU=cuda:1 ENCODER_MODE=mean_only python -u seed_consistency_11.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, DTD_IMAGES, REPO  # noqa: E402
import os, sys, time, csv, itertools
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DPS = ASSETS
DTD = DTD_IMAGES
TARGETS = DPS / "pixelbp_prodigy_targets"
OUT_DIR = DPS / "synths/seed_consistency_11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO = REPO
sys.path.insert(0, str(REPO))
import utils.stattexgramnet as stn

DEVICE = torch.device(os.environ.get("GPU", "cuda:1"))
LR = float(os.environ.get("LR", "0.02"))   # actual production value (family-ablation + mean-only .sh scripts); Appendix A.3's 0.04 is a stale default
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "20000"))
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))

# The 11 Figure-2 textures (synth_stgn_meanonly_qualitative.py).
STEMS = [
    "chequered_0177", "striped_0050", "polka-dotted_0140",
    "honeycombed_0149", "cracked_0056", "lacelike_0004",
    "fibrous_0112", "bumpy_0077", "sprinkled_0042",
    "pleated_0136", "swirly_0065",
]
SEEDS = list(range(N_SEEDS))


def load_and_preprocess(stem, device, size=256):
    cat = stem.rsplit("_", 1)[0]
    img = cv2.imread(str(DTD / cat / f"{stem}.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (torch.tensor(img).permute(2, 0, 1).contiguous().unsqueeze(0).float()
           .to(device) / 255.0)
    _, _, h, w = img.shape
    s = min(h, w); sh, sw = (h - s) // 2, (w - s) // 2
    img = img[:, :, sh:sh + s, sw:sw + s]
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)


def load_t(p):
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    return transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)


def synthesize():
    slots = [(stem, seed) for stem in STEMS for seed in SEEDS]
    B = len(slots)
    print(f"== seed consistency: {len(STEMS)} stems x {len(SEEDS)} seeds = {B} syntheses",
          flush=True)
    print(f"   device={DEVICE} lr={LR} iters={NUM_ITERATIONS} boundary=black "
          f"ENCODER_MODE={os.environ.get('ENCODER_MODE','mean_only(default)')}", flush=True)

    model = stn.StatTexGramNet(
        img_size=(256, 256), batch_size=B, bottleneck_size=None, device=DEVICE,
        compress=False, color=True, add_marginals=True, oris=4, edge_levels=6,
        edge_start=1, include_dphase=False, boundary_mode="black").to(DEVICE).eval()

    ref_by_stem = {s: load_and_preprocess(s, DEVICE) for s in STEMS}
    ref_batch = torch.cat([ref_by_stem[stem] for stem, _ in slots], dim=0)
    with torch.no_grad():
        target_stats = model.forward(ref_batch)

    inits = []
    for i, (stem, seed) in enumerate(slots):
        g = torch.Generator(device="cpu").manual_seed(seed * 100003 + i)
        inits.append(torch.rand(1, 3, 256, 256, generator=g).to(DEVICE))
    x = torch.cat(inits, dim=0).clone().detach().requires_grad_(True)

    from prodigyopt import Prodigy
    # Exactly the paper / workhorse (pixelbp_prodigy_depleted.py) Prodigy config.
    optimizer = Prodigy([x], lr=LR, betas=(0.99, 0.999), d0=1e-2,
                        safeguard_warmup=True, use_bias_correction=True)

    best_x = x.detach().clone()
    best_loss = torch.full((B,), float("inf"), device=DEVICE)

    def save_best():
        arr = best_x.detach().cpu().clamp(0, 1).numpy()
        for i, (stem, seed) in enumerate(slots):
            a = (arr[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(a).save(OUT_DIR / f"{stem}_seed{seed}_synth.png")

    t0 = time.time()
    for it in range(NUM_ITERATIONS):
        optimizer.zero_grad()
        stats = model.forward(x)
        per_img = F.mse_loss(stats, target_stats, reduction="none").flatten(1).mean(1)
        per_img.mean().backward()
        torch.nn.utils.clip_grad_norm_(x, max_norm=10.0)   # workhorse setting
        optimizer.step()
        with torch.no_grad():
            if torch.isnan(x).any():
                print(f"NaN at iter {it}", flush=True); break
            x.clamp_(0, 1)
            improved = per_img.detach() < best_loss
            best_loss[improved] = per_img.detach()[improved]
            best_x[improved] = x.detach()[improved]
        if it % 2000 == 0 or it == NUM_ITERATIONS - 1:
            print(f"  iter {it:5d}  best_mean={best_loss.mean().item():.3e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            save_best()
    save_best()

    with open(OUT_DIR / "final_losses.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["stem", "seed", "final_best_loss"])
        for i, (stem, seed) in enumerate(slots):
            wr.writerow([stem, seed, f"{best_loss[i].item():.6e}"])
    print(f"== synth done in {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/B:.1f}s/image amortized)", flush=True)
    return {(s, k): best_loss[i].item() for i, (s, k) in enumerate(slots)}


def analyze(loss_by):
    import piq, lpips as lpips_mod
    dists = piq.DISTS(reduction="none").to(DEVICE)
    lp = lpips_mod.LPIPS(net="vgg").to(DEVICE)
    targets = {s: load_t(TARGETS / f"{s}.png") for s in STEMS}
    synth = {(s, k): load_t(OUT_DIR / f"{s}_seed{k}_synth.png") for s in STEMS for k in SEEDS}

    rows = []
    for s in STEMS:
        for k in SEEDS:
            r, sy = targets[s], synth[(s, k)]
            with torch.no_grad():
                rows.append(dict(stem=s, seed=k, final_loss=loss_by[(s, k)],
                    dists=dists(r, sy).mean().item(),
                    lpips=lp(r*2-1, sy*2-1).mean().item(),
                    ssim=piq.ssim(r, sy, data_range=1.0).item(),
                    ms_ssim=piq.multi_scale_ssim(r, sy, data_range=1.0).item()))
    with open(OUT_DIR / "per_synth_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["stem", "seed", "dists", "lpips",
                                           "ssim", "ms_ssim", "final_loss"])
        wr.writeheader(); wr.writerows(rows)

    def col(s, m): return np.array([r[m] for r in rows if r["stem"] == s])
    summ = []
    for s in STEMS:
        with torch.no_grad():
            pw = [dists(synth[(s, a)], synth[(s, b)]).mean().item()
                  for a, b in itertools.combinations(SEEDS, 2)]
        d = col(s, "dists")
        rec = dict(stem=s, pairwise_synth_dists_mean=float(np.mean(pw)))
        for m in ["dists", "lpips", "ssim", "ms_ssim"]:
            v = col(s, m)
            rec[f"{m}_mean"] = float(v.mean()); rec[f"{m}_sd"] = float(v.std(ddof=1))
            rec[f"{m}_cv"] = float(v.std(ddof=1) / abs(v.mean())) if v.mean() else float("nan")
        summ.append(rec)
    fields = (["stem"] + [f"{m}_{q}" for m in ["dists", "lpips", "ssim", "ms_ssim"]
              for q in ["mean", "sd", "cv"]] + ["pairwise_synth_dists_mean"])
    with open(OUT_DIR / "per_stem_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(summ)

    # Console summary + across-seed vs across-stem spread (DISTS).
    print("\n== per-stem DISTS (mean +/- sd, across-seed CV) ==", flush=True)
    for rec in sorted(summ, key=lambda r: r["dists_mean"]):
        print(f"  {rec['stem']:20s} {rec['dists_mean']:.3f} +/- {rec['dists_sd']:.3f}  "
              f"CV={rec['dists_cv']*100:4.1f}%  loss_worst_seed="
              f"{max(loss_by[(rec['stem'],k)] for k in SEEDS):.2e}", flush=True)
    within = np.mean([r["dists_sd"] for r in summ])
    across = np.std([r["dists_mean"] for r in summ], ddof=1)
    print(f"\n  mean within-stem (across-seed) sd = {within:.3f}", flush=True)
    print(f"  across-stem sd of means           = {across:.3f}", flush=True)
    print(f"  ratio (texture matters vs seed)   = {across/within:.1f}x", flush=True)


def main():
    torch.cuda.set_device(DEVICE)
    loss_by = synthesize()
    analyze(loss_by)


if __name__ == "__main__":
    main()
