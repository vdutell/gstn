# ---------------------------------------------------------------------------
# Produces: Figure 8, left panel (x-t slices)
#
# Provenance: StatTexNetExtended/rebuttal/make_fig_xt_transposed.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Figure 8 x-t slices, transposed layout: all 5 videos.

Changes from make_fig_motion.xt_slices():
  * time on the HORIZONTAL axis, space on the vertical (the conventional space-time /
    kymograph orientation, and it matches the "time ->" reading direction).
  * rows = Target / GSTN / DyNCA, columns = the five videos, so a reader scans DOWN a
    column to compare methods on one texture -- which is the comparison being made.
  * all five DynTex clips instead of three.

With time horizontal, STATIC structure reads as horizontal streaks, MOTION as slanted
streaks, and frame-wide temporal change (fire's flicker) as vertical bands.

Rendered at exactly \textwidth = 5.5in with NO bbox_inches="tight", so
\includegraphics[width=\textwidth] does not rescale and the stated point sizes hold.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import FIGURES, VIDEO  # noqa: E402
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

V   = VIDEO
OUT = FIGURES
TEXTWIDTH_IN = 5.5
NT, CKPT = 20, "3000"

COLS = [("54pi110","grass"), ("55fc210","fire"), ("6440110","fence"),
        ("54pb110","blooms"), ("6441210","tree")]
ROWS = [("Target","target"), ("GSTN","gstn"), ("DyNCA","dynca")]


def stack(kind, code):
    if kind == "target":
        c = cv2.VideoCapture(str(V/"targets"/f"{code}.mp4")); out = []
        while len(out) < NT:
            ok, fr = c.read()
            if not ok: break
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        c.release(); return np.stack(out)
    d = V/"gstn_hm50_raw"/code/f"e{CKPT}" if kind == "gstn" else V/"dynca_synths_raw"/code
    return np.stack([cv2.cvtColor(cv2.imread(str(d/f"frame_{t:03d}.png")), cv2.COLOR_BGR2RGB)
                     for t in range(NT)])


def build(fig_h, out_stem, fs=7.0, up=8, fig_w=TEXTWIDTH_IN, left=0.072):
    fig, ax = plt.subplots(3, 5, figsize=(fig_w, fig_h), squeeze=False,
                           gridspec_kw=dict(wspace=0.05, hspace=0.07))
    for c, (code, name) in enumerate(COLS):
        for r, (rl, kind) in enumerate(ROWS):
            s  = stack(kind, code)                 # [T,H,W,3]
            y  = s.shape[1] // 2
            xt = s[:, y, :, :].transpose(1, 0, 2)  # [W,T,3] -> space vertical, time horizontal
            big = cv2.resize(xt, (xt.shape[1]*up, xt.shape[0]*up),
                             interpolation=cv2.INTER_NEAREST)
            a = ax[r][c]
            a.imshow(big, aspect="auto")           # auto: fill the cell, axes are different units
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#999")
            if r == 0: a.set_title(name, fontsize=fs, pad=2.5)
            if c == 0: a.set_ylabel(rl, fontsize=fs, labelpad=2)

    fig.text(0.5, 0.012, "time $t$ $\\rightarrow$", ha="center", fontsize=fs-0.6)
    fig.text(0.010, 0.55, "space $x$", va="center", ha="center",
             rotation=90, fontsize=fs-0.6)
    fig.subplots_adjust(left=left, right=0.998, top=0.915, bottom=0.075)

    for ext in ("pdf", "png"):
        p = OUT/f"{out_stem}.{ext}"
        fig.savefig(p, dpi=400)                    # no bbox_inches="tight"
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    # full \textwidth, table would move below
    build(1.55, "video_xt_all5_h155")
    build(1.85, "video_xt_all5_h185")
    # narrow: drops into the EXISTING slot beside the metrics table (~3.5in)
    build(1.30, "video_xt_all5_narrow_h130", fs=6.0, fig_w=3.50, left=0.105)
    build(1.55, "video_xt_all5_narrow_h155", fs=6.0, fig_w=3.50, left=0.105)
    build(1.85, "video_xt_all5_narrow_h185", fs=6.0, fig_w=3.50, left=0.105)
