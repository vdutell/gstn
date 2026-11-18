# ---------------------------------------------------------------------------
# Produces: Figure 8, right panel (frame grid)
#
# Provenance: StatTexNetExtended/rebuttal/make_fig_video_synths.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Static figure for the paper: target / GSTN / DyNCA frames for the five DynTex videos.

Replaces figures/video_synths_3x.png. Sources are the LOSSLESS frames the metrics were
computed on:
    target : dynamic_eval_videos/targets/<code>.mp4           (only stored as mp4)
    GSTN   : dynamic_eval_videos/gstn_hm50_raw/<code>/e3000/frame_XXX.png
    DyNCA  : dynamic_eval_videos/dynca_synths_raw/<code>/frame_XXX.png

DyNCA's 25->15 fps resampling is already applied in dynca_synths_raw (rerender_dynca_lossless.py
saves 20 frames on the target's grid), so frame k of all three sources is the same instant.
Do not resample again.

Frames are 64x64; upscaled with NEAREST so no detail is invented. Video names verified from the
target frames -- Calc_quality.ipynb mislabels 6440110 as "tree1".

Run: python make_fig_video_synths.py            # 5 videos as columns (wide, fits beside a table)
     N_VIDEOS=3 python make_fig_video_synths.py # 3 videos, matching the old video_synths_3x
     FRAME=10 python make_fig_video_synths.py   # pick a different time point
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import FIGURES, VIDEO  # noqa: E402
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

V = VIDEO
OUT = FIGURES
OUT.mkdir(exist_ok=True)

ORDER = [("54pi110", "grass"), ("55fc210", "fire"), ("6440110", "fence"),
         ("54pb110", "blooms"), ("6441210", "tree")]
N = int(os.environ.get("N_VIDEOS", "5"))
FRAME = int(os.environ.get("FRAME", "0"))
CKPT = os.environ.get("CKPT", "3000")
UP = 4                                   # nearest upscale so the PNG prints at ~400 dpi
ROWS = [("Target", "target"), ("GSTN (ours)", "gstn"), ("DyNCA", "dynca")]


def target_frame(code, t):
    c = cv2.VideoCapture(str(V / "targets" / f"{code}.mp4"))
    for _ in range(t + 1):
        ok, fr = c.read()
    c.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def png(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def get(kind, code, t):
    if kind == "target":
        return target_frame(code, t)
    if kind == "gstn":
        return png(V / "gstn_hm50_raw" / code / f"e{CKPT}" / f"frame_{t:03d}.png")
    return png(V / "dynca_synths_raw" / code / f"frame_{t:03d}.png")


codes = ORDER[:N]
fig, axes = plt.subplots(len(ROWS), len(codes),
                         figsize=(1.15 * len(codes), 1.15 * len(ROWS) + 0.18),
                         squeeze=False,
                         gridspec_kw=dict(wspace=0.04, hspace=0.04))
for r, (rlabel, kind) in enumerate(ROWS):
    for c, (code, name) in enumerate(codes):
        im = get(kind, code, FRAME)
        h, w = im.shape[:2]
        ax = axes[r][c]
        ax.imshow(cv2.resize(im, (w * UP, h * UP), interpolation=cv2.INTER_NEAREST))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#999999")
        if r == 0:
            ax.set_title(name, fontsize=8, pad=3)
        if c == 0:
            ax.set_ylabel(rlabel, fontsize=8, labelpad=4)

plt.subplots_adjust(left=0.055, right=0.998, top=0.93, bottom=0.004)
for ext, kw in [("png", dict(dpi=400)), ("pdf", {})]:
    p = OUT / f"video_synths_{len(codes)}x_target_gstn_dynca.{ext}"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.01, **kw)
    print(f"wrote {p}")
plt.close(fig)
