# ---------------------------------------------------------------------------
# Produces: Table 2
#
# Provenance: diffusion-posterior-sampling/seed_full_metrics.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Recompute the 11-texture seed study on the PAPER'S FULL 9-METRIC SUITE.

The earlier seed table reported only DISTS/LPIPS/SSIM/MS-SSIM. The paper's own
per-category analysis (Appendix Figures 9-10, and the Pareto comparison) uses nine:
DISTS, LPIPS, DreamSim, FSIM, SSIM, VIF, GMSD, HaarPSI, PSD-L2. Reporting a four-metric
subset for the seed study -- one of which (MS-SSIM) is not in the paper's nine at all --
invites a cherry-picking objection. This computes all nine, with the same
implementations used in extend_bench_to_141.py so the numbers are commensurable:
  - 256x256 inputs, bicubic-resized if needed (load_t)
  - LPIPS with the VGG backbone
  - DreamSim pretrained
  - PSD-L2 = RMS difference of log1p power spectra on the luminance channel

Run: GPU=cuda:1 python -u seed_full_metrics.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS  # noqa: E402
import os, csv, itertools
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision import transforms

DEV = torch.device(os.environ.get("GPU", "cuda:1"))
torch.cuda.set_device(DEV)
import piq, lpips as lpips_mod
from dreamsim import dreamsim

DPS = ASSETS
OUT_DIR = DPS / "synths/seed_consistency_11"
TARGETS = DPS / "pixelbp_prodigy_targets"
STEMS = ["chequered_0177", "striped_0050", "polka-dotted_0140", "honeycombed_0149",
         "cracked_0056", "lacelike_0004", "fibrous_0112", "bumpy_0077",
         "sprinkled_0042", "pleated_0136", "swirly_0065"]
SEEDS = range(5)
# (key, display, higher-is-better)
SUITE = [("dists", "DISTS", False), ("lpips", "LPIPS", False), ("dreamsim", "DreamSim", False),
         ("fsim", "FSIM", True), ("ssim", "SSIM", True), ("vif", "VIF", True),
         ("gmsd", "GMSD", False), ("haarpsi", "HaarPSI", True), ("psd_l2", "PSD-L2", False)]


def load_t(p):
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    return transforms.ToTensor()(img).unsqueeze(0).to(DEV)


def psd_l2(r, s):
    def psd(x):
        return torch.log1p(torch.fft.fft2(x.mean(1)).abs() ** 2)
    return (psd(r) - psd(s)).pow(2).mean().sqrt().item()


def main():
    dists = piq.DISTS(reduction="none").to(DEV)
    lp = lpips_mod.LPIPS(net="vgg").to(DEV)
    ds_model, _ = dreamsim(pretrained=True, device=DEV)

    tgt = {s: load_t(TARGETS / f"{s}.png") for s in STEMS}
    syn = {(s, k): load_t(OUT_DIR / f"{s}_seed{k}_synth.png") for s in STEMS for k in SEEDS}

    rows = []
    for s in STEMS:
        for k in SEEDS:
            r, y = tgt[s], syn[(s, k)]
            with torch.no_grad():
                rows.append(dict(stem=s, seed=k,
                    dists=dists(r, y).mean().item(),
                    lpips=lp(r * 2 - 1, y * 2 - 1).mean().item(),
                    dreamsim=float(ds_model(r, y).item()),
                    fsim=piq.fsim(y, r, data_range=1.0).item(),
                    ssim=piq.ssim(y, r, data_range=1.0).item(),
                    vif=piq.vif_p(y, r, data_range=1.0).item(),
                    gmsd=piq.gmsd(y, r, data_range=1.0).item(),
                    haarpsi=piq.haarpsi(y, r, data_range=1.0).item(),
                    psd_l2=psd_l2(r, y)))
        print(f"  {s} done", flush=True)

    with open(OUT_DIR / "per_synth_metrics_full9.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "seed"] + [k for k, _, _ in SUITE])
        w.writeheader(); w.writerows(rows)

    def col(s, m): return np.array([r[m] for r in rows if r["stem"] == s])

    order = sorted(STEMS, key=lambda s: col(s, "dists").mean())
    L = ["| Texture | " + " | ".join(f"{d} {'↑' if hi else '↓'}" for _, d, hi in SUITE) + " |",
         "|---|" + "---|" * len(SUITE)]
    for s in order:
        cells = [f"{col(s,k).mean():.3f} ± {col(s,k).std(ddof=1):.3f}" for k, _, _ in SUITE]
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    md = "\n".join(L)
    print("\n" + md)

    L2 = ["\n**Texture-vs-seed variance ratio** (across-texture sd of means ÷ mean "
          "within-texture across-seed sd):\n",
          "| | " + " | ".join(d for _, d, _ in SUITE) + " |", "|---|" + "---:|" * len(SUITE)]
    for tag, sub in [("all 11 textures", order),
                     ("excluding polka-dotted_0140", [s for s in order if "polka" not in s])]:
        v = []
        for k, _, _ in SUITE:
            w_ = np.mean([col(s, k).std(ddof=1) for s in sub])
            a_ = np.std([col(s, k).mean() for s in sub], ddof=1)
            v.append(f"{a_/w_:.1f}×")
        L2.append(f"| {tag} | " + " | ".join(v) + " |")
    md2 = "\n".join(L2)
    print(md2)
    (DPS / "seed_results_full9.md").write_text(md + "\n" + md2 + "\n")
    print("\n[written seed_results_full9.md]")


if __name__ == "__main__":
    main()
