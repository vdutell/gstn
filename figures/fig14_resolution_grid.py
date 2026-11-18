# ---------------------------------------------------------------------------
# Produces: Figure 14
#
# Provenance: StatTexNetExtended/rebuttal/camera_ready/make_fig_resolution_paired.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Resolution sweep with a RESOLUTION-MATCHED target in every column.

The earlier version put the 256 px target beside synths upsampled from 16/32/64 px, which
made low resolutions look bad for a reason that has nothing to do with synthesis quality --
the reference was sharper than anything the cell could represent. The metrics never had this
problem: native_quality_metrics.target() center-crops the original DTD jpg and resizes it to
the SYNTHESIS resolution, so synth and target are always matched. This figure now does the
same, so what you see matches what Table tab:resolution scores. It is why DISTS is *best* at
16 px (0.101): against a 16 px reference, a roughly-correct synthesis scores well.

Target construction is deliberately identical to the metric script -- centre crop to square,
then plain bilinear to N (no antialiasing) -- so the figure shows the reference the numbers
actually used. All six sources are >=512 px native, so every column is a genuine downsample.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, DTD_IMAGES, FIGURES  # noqa: E402
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DPS = ASSETS
DTDIMG = DTD_IMAGES
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
NSTATS = 28_929
SIZES = [16, 32, 64, 128, 256, 512]
RSTEMS = ["bubbly_0064", "dotted_0078", "grooved_0054",
          "polka-dotted_0140", "scaly_0168", "swirly_0065"]
BOX = 256


def dtd_target(stem, s):
    p = DTDIMG/stem.rsplit("_", 1)[0]/f"{stem}.jpg"
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]; m = min(h, w)
    im = im[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
    return cv2.resize(im, (s, s), interpolation=cv2.INTER_LINEAR)


def synth(stem, s):
    p = DPS/"synths/highres_final6"/f"size{s:04d}"/f"{stem}_seed0_synth.png"
    im = cv2.imread(str(p))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None


def show(a, im):
    a.set_xticks([]); a.set_yticks([])
    if im is None:
        a.set_facecolor("#eee")
    else:
        a.imshow(cv2.resize(im, (BOX, BOX), interpolation=cv2.INTER_NEAREST))
    for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999")


def main():
    GAP, nc = 0.015, len(SIZES)
    CELL = 0.85
    heights = []
    for i, _ in enumerate(RSTEMS):
        heights += [1, 1]
        if i < len(RSTEMS)-1: heights += [GAP]
    fig = plt.figure(figsize=(nc*CELL, sum(heights)*CELL + 0.40))
    gs = gridspec.GridSpec(len(heights), nc, figure=fig, height_ratios=heights,
                           wspace=0.022, hspace=0.004)
    pairs, r = [], 0
    for gi, stem in enumerate(RSTEMS):
        axs = []
        for sub, lab in enumerate(["Target", "GSTN"]):
            row = []
            for c, s in enumerate(SIZES):
                a = fig.add_subplot(gs[r+sub, c])
                show(a, dtd_target(stem, s) if sub == 0 else synth(stem, s))
                if gi == 0 and sub == 0:
                    # Two lines. 6.6pt is the largest that keeps the widest string
                    # ("27.18x stats", 60.6pt) inside a 61.2pt column; measured, not
                    # estimated -- at 7.2pt it is 66.1pt and adjacent headers touch.
                    a.set_title(f"{s}$\\times${s}\n{3*s*s/NSTATS:.2f}$\\times$ stats",
                                fontsize=6.6, pad=2, linespacing=1.2)
                if c == 0:
                    a.set_ylabel(lab, fontsize=10, labelpad=3)
                row.append(a)
            axs.append(row)
        pairs.append((stem, axs))
        r += 2 + (1 if gi < len(RSTEMS)-1 else 0)

    fig.subplots_adjust(left=0.135, right=0.999,
                        top=1-0.30/(sum(heights)*CELL+0.40), bottom=0.0015)
    # Texture name centred across each Target/GSTN pair, outside the per-row labels.
    for stem, axs in pairs:
        y0 = axs[1][0].get_position().y0
        y1 = axs[0][0].get_position().y1
        fig.text(0.030, (y0+y1)/2, stem.rsplit("_", 1)[0], rotation=90,
                 va="center", ha="center", fontsize=13)
    for ext in ("pdf", "png"):
        p = OUT/f"fig_resolution_grid_paired.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.012, dpi=350); print("wrote", p)
    plt.close(fig)


main()
