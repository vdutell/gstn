# ---------------------------------------------------------------------------
# Produces: resolution-scaling metrics (Table 5)
#
# Provenance: diffusion-posterior-sampling/native_quality_metrics.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Native-resolution quality metrics vs synthesis resolution (11 Figure-2 textures).

All metrics computed with synthesis AND target at the synthesis resolution -- i.e.
what a viewer actually sees. This is the fair comparison across resolutions; scoring
everything upsampled to a common size instead flatters small resolutions, because it
compares two heavily blurred images.

Metric suite matches the paper's (DISTS, LPIPS, SSIM, MS-SSIM, FSIM, GMSD, HaarPSI,
VIF) so the numbers are directly relatable to Figure 3 and Appendix Figures 9/10.

Metrics that cannot be evaluated at a given size (MS-SSIM needs enough pixels for its
downsampling pyramid; some piq metrics have minimum window sizes) are reported as n/a
rather than silently dropped. VGG-based metrics (DISTS, LPIPS) are flagged below 64px:
they downsample 5x, so their deepest features are degenerate, and a 16px target carries
so little texture information that any roughly-correct synthesis scores well.

Run: GPU=cuda:1 python -u native_quality_metrics.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, DTD_IMAGES  # noqa: E402
import os, sys, csv
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image

DEV = torch.device(os.environ.get("GPU", "cuda:1"))
torch.cuda.set_device(DEV)
import piq, lpips as lpips_mod

DTD = DTD_IMAGES
DPS = ASSETS
RUN = os.environ.get("RUN", "resolution_scaling")
R = DPS / "synths" / RUN
SIZES = [int(x) for x in os.environ.get("SIZES", "16,32,64,128,256").split(",")]
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
STEMS = os.environ.get("STEMS", ",".join([
    "chequered_0177", "striped_0050", "polka-dotted_0140", "honeycombed_0149",
    "cracked_0056", "lacelike_0004", "fibrous_0112", "bumpy_0077",
    "sprinkled_0042", "pleated_0136", "swirly_0065"])).split(",")

STATS = 28_929
HIGHER_BETTER = {"SSIM", "MS-SSIM", "FSIM", "HaarPSI", "VIF"}


def target(stem, size):
    im = cv2.cvtColor(cv2.imread(str(DTD / stem.rsplit("_", 1)[0] / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]; s = min(h, w)
    im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False).clamp(0, 1)


def synth(stem, size, k):
    a = np.array(Image.open(R / f"size{size:04d}" / f"{stem}_seed{k}_synth.png").convert("RGB"))
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255


def main():
    dists = piq.DISTS(reduction="none").to(DEV)
    lp = lpips_mod.LPIPS(net="vgg").to(DEV)

    def metrics(t, s):
        out = {}
        with torch.no_grad():
            for name, fn in [
                ("DISTS",   lambda: dists(t, s).mean().item()),
                ("LPIPS",   lambda: lp(t * 2 - 1, s * 2 - 1).mean().item()),
                ("SSIM",    lambda: piq.ssim(s, t, data_range=1.0).item()),
                ("MS-SSIM", lambda: piq.multi_scale_ssim(s, t, data_range=1.0).item()),
                ("FSIM",    lambda: piq.fsim(s, t, data_range=1.0).item()),
                ("GMSD",    lambda: piq.gmsd(s, t, data_range=1.0).item()),
                ("HaarPSI", lambda: piq.haarpsi(s, t, data_range=1.0).item()),
                ("VIF",     lambda: piq.vif_p(s, t, data_range=1.0).item()),
            ]:
                try:
                    v = fn()
                    out[name] = v if np.isfinite(v) else np.nan
                except Exception:
                    out[name] = np.nan
        return out

    names = ["DISTS", "LPIPS", "SSIM", "MS-SSIM", "FSIM", "GMSD", "HaarPSI", "VIF"]
    table = {}
    for size in SIZES:
        acc = {n: [] for n in names}
        for stem in STEMS:
            t = target(stem, size)
            for k in range(N_SEEDS):
                m = metrics(t, synth(stem, size, k))
                for n in names:
                    acc[n].append(m[n])
        table[size] = {n: (np.nanmean(acc[n]) if np.isfinite(acc[n]).any() else np.nan)
                       for n in names}
        print(f"  {size:4d}px  " + "  ".join(
            f"{n}={table[size][n]:.3f}" if np.isfinite(table[size][n]) else f"{n}=n/a"
            for n in names), flush=True)

    # markdown
    arrows = {n: ("↑" if n in HIGHER_BETTER else "↓") for n in names}
    L = []
    L.append(f"| Resolution | Image values (3·N²) | Values per statistic | "
             + " | ".join(f"{n} {arrows[n]}" for n in names) + " |")
    L.append("|---|---:|---:|" + "---:|" * len(names))
    for size in SIZES:
        px = 3 * size * size
        flag = " \\*" if size < 64 else ""
        cells = []
        for n in names:
            v = table[size][n]
            cells.append("n/a" if not np.isfinite(v) else f"{v:.3f}")
        L.append(f"| {size}×{size}{flag} | {px:,} | {px/STATS:.3f}× | " + " | ".join(cells) + " |")
    md = "\n".join(L)
    print("\n" + md)
    (DPS / f"native_quality_{RUN}.md").write_text(md + "\n")
    with open(DPS / f"native_quality_{RUN}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["size", "img_values", "values_per_stat"] + names)
        for size in SIZES:
            w.writerow([size, 3 * size * size, 3 * size * size / STATS]
                       + [table[size][n] for n in names])
    print(f"\n[written native_quality_{RUN}.md / .csv]")


if __name__ == "__main__":
    main()
