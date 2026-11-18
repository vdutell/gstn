# ---------------------------------------------------------------------------
# Produces: Figures 12 and 13
#
# Provenance: StatTexNetExtended/rebuttal/camera_ready/make_fig_seed_and_resolution.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Two promised camera-ready figures, both from synths already on disk.

  A. Seed grid       -- 3Kqw Q2. "This result holds qualitatively in the texture images,
                        a figure we will include in the camera ready."
                        11 Figure-2 textures x 5 random initializations, GSTN-zpad.
  B. Resolution grid -- 3Kqw Q3 / 4HA9 W1. "a new figure documenting the relative model size
                        compared to image resolution and the resulting qualitative performance."
                        6 textures x 6 resolutions, fixed 28,929-statistic budget.
  C. Resolution curves - compression factor and quality vs. resolution; the quantitative
                        companion to B, and the peripheral-falloff argument of Sec. 6.1.

Source of B/C is highres_final6 (6 stems x 3 seeds, sizes 16-512), which is the sweep the
posted table's numbers come from -- NOT synths/resolution_scaling (11 stems, stops at 256).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, FIGURES, RESULTS# noqa: E402
import csv
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPS = ASSETS
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = DPS / "pixelbp_prodigy_targets"
NSTATS = 28_929


def load(p, s=256):
    im = cv2.imread(str(p))
    if im is None: return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (s, s), interpolation=cv2.INTER_NEAREST if im.shape[0] < s
                      else cv2.INTER_AREA)


def frame(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999")


# ---------------- A. seed grid ----------------
def seed_grid():
    d = DPS / "synths/seed_consistency_11"
    stems = ["pleated_0136", "cracked_0056", "striped_0050", "fibrous_0112", "bumpy_0077",
             "lacelike_0004", "sprinkled_0042", "honeycombed_0149", "swirly_0065",
             "chequered_0177", "polka-dotted_0140"]          # ordered by mean DISTS
    cols = ["Target"] + [f"Seed {i}" for i in range(5)]
    fig, ax = plt.subplots(len(stems), 6, figsize=(6*1.1, len(stems)*1.1+0.25),
                           squeeze=False, gridspec_kw=dict(wspace=0.035, hspace=0.035))
    for r, stem in enumerate(stems):
        for c in range(6):
            a = ax[r][c]; frame(a)
            p = TARGETS/f"{stem}.png" if c == 0 else d/f"{stem}_seed{c-1}_synth.png"
            im = load(p) if p.exists() else None
            a.imshow(im) if im is not None else a.set_facecolor("#eee")
            if r == 0: a.set_title(cols[c], fontsize=8, pad=3)
            if c == 0: a.set_ylabel(stem.rsplit("_", 1)[0], fontsize=7, labelpad=3)
    plt.subplots_adjust(left=0.075, right=0.999, top=1-0.25/(len(stems)*1.1+0.25), bottom=0.002)
    for ext in ("pdf", "png"):
        p = OUT/f"fig_seed_grid.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.012, dpi=350); print("wrote", p)
    plt.close(fig)


# ---------------- B. resolution grid ----------------
SIZES = [16, 32, 64, 128, 256, 512]
RSTEMS = ["bubbly_0064", "dotted_0078", "grooved_0054",
          "polka-dotted_0140", "scaly_0168", "swirly_0065"]


def resolution_grid():
    root = DPS/"synths/highres_final6"
    nc = len(SIZES)+1
    fig, ax = plt.subplots(len(RSTEMS), nc, figsize=(nc*1.1, len(RSTEMS)*1.1+0.45),
                           squeeze=False, gridspec_kw=dict(wspace=0.035, hspace=0.035))
    for r, stem in enumerate(RSTEMS):
        for c in range(nc):
            a = ax[r][c]; frame(a)
            if c == 0:
                im = load(TARGETS/f"{stem}.png"); head = "Target\n(1.00$\\times$ img)"
            else:
                s = SIZES[c-1]
                im = load(root/f"size{s:04d}"/f"{stem}_seed0_synth.png")
                head = f"{s}$\\times${s}\n{3*s*s/NSTATS:.2f}$\\times$ stats"
            a.imshow(im) if im is not None else a.set_facecolor("#eee")
            if r == 0: a.set_title(head, fontsize=7, pad=3, linespacing=1.25)
            if c == 0: a.set_ylabel(stem.rsplit("_", 1)[0], fontsize=7, labelpad=3)
    plt.subplots_adjust(left=0.085, right=0.999,
                        top=1-0.45/(len(RSTEMS)*1.1+0.45), bottom=0.002)
    for ext in ("pdf", "png"):
        p = OUT/f"fig_resolution_grid.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.012, dpi=350); print("wrote", p)
    plt.close(fig)


# ---------------- C. resolution curves ----------------
def resolution_curves():
    rows = list(csv.DictReader(open(RESULTS/"native_quality_highres_final6.csv")))
    n = np.array([int(r["size"]) for r in rows])
    vps = np.array([float(r["values_per_stat"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

    a = ax[0]
    a.loglog(n, vps, "o-", color="#2f6f9f", lw=1.6, ms=5)
    a.axhline(1.0, color="#999", ls="--", lw=0.9)
    a.axvline(98, color="#c0504d", ls=":", lw=1.1)
    a.text(100, 0.035, "unity at\n$\\sim$98 px", fontsize=7, color="#c0504d", va="bottom")
    a.set_xlabel("image side (px)", fontsize=8)
    a.set_ylabel("image values per statistic", fontsize=8)
    a.set_title("Compression grows as $3N^2/28{,}929$", fontsize=8.5)
    a.set_xticks(n); a.set_xticklabels([str(x) for x in n], fontsize=7)
    a.tick_params(labelsize=7); a.grid(alpha=0.25, which="both", lw=0.4)

    a = ax[1]
    for key, lab, col, up in [("DISTS", "DISTS $\\downarrow$", "#c0504d", False),
                              ("LPIPS", "LPIPS $\\downarrow$", "#e08214", False),
                              ("FSIM", "FSIM $\\uparrow$", "#2f6f9f", True),
                              ("HaarPSI", "HaarPSI $\\uparrow$", "#4b9b4b", True)]:
        v = np.array([float(r[key]) for r in rows])
        a.semilogx(n, v, "o-", lw=1.5, ms=4, color=col, label=lab,
                   ls="-" if up else "--")
    a.set_xlabel("image side (px)", fontsize=8); a.set_ylabel("metric value", fontsize=8)
    a.set_title("Quality falls as the fixed budget is spread thinner", fontsize=8.5)
    a.set_xticks(n); a.set_xticklabels([str(x) for x in n], fontsize=7)
    a.tick_params(labelsize=7); a.grid(alpha=0.25, lw=0.4)
    a.legend(fontsize=6.5, ncol=2, loc="center left", frameon=True, framealpha=0.9,
             edgecolor="none", facecolor="white", borderpad=0.4)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        p = OUT/f"fig_resolution_curves.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.015, dpi=350); print("wrote", p)
    plt.close(fig)


import sys
_only = sys.argv[1] if len(sys.argv) > 1 else "all"
if _only in ("all","seed"): seed_grid()
if _only in ("all","grid"): resolution_grid()
if _only in ("all","curves"): resolution_curves()
