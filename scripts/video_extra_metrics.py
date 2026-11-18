# ---------------------------------------------------------------------------
# Produces: video diversity and spectrum metrics
#
# Provenance: StatTexNetExtended/rebuttal/compute_video_extra.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Two extra dynamic-texture axes (NeurIPS rebuttal), independent of DyNCA's losses:

1) Temporal power-spectrum distance: does the synth reproduce the TARGET's motion
   rhythm (temporal-frequency content)?  Per-pixel temporal FFT, spatially averaged,
   normalized to a distribution over temporal frequencies; L1 distance target-vs-synth.
   Lower = better. Measured over a matched 20-frame span (DyNCA: first 20 consecutive,
   whose per-frame motion is trained to match the target's adjacent frames).

2) Frame diversity: mean pairwise LPIPS between distinct frames within a video (matched
   20-frame span). This is the axis where the paradigms differ: GSTN samples the full
   statistical distribution; an NCA can loop / mode-collapse. We report each method's
   diversity and its ratio to the TARGET's own diversity (ratio ~1 = reproduces the
   texture's intrinsic variability; <1 = too static/repetitive). We also report DyNCA's
   diversity over its FULL long rollout (400 frames) as a looping/collapse check.

Run: python compute_video_extra.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import VIDEO  # noqa: E402
import os, itertools, csv
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
import lpips as lpips_mod

DEV = torch.device("cuda:0")
BASE = VIDEO
NAMES = {"54pi110": "grass", "54pb110": "blooms", "55fc210": "fire",
         "6440110": "fence", "6441210": "tree"}
CODES = list(NAMES)
SPAN = 20  # matched temporal span (frames)


def read_frames(p):
    c = cv2.VideoCapture(str(p)); fr = []; fps = c.get(5)
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    c.release()
    return np.stack(fr), fps


def resample_to_target(frames, src_fps, n_out=SPAN, out_fps=15.0):
    """Resample to the target's temporal grid (out_fps, n_out frames) so per-frame
    motion is comparable across methods regardless of a method's render fps."""
    idx = [min(len(frames) - 1, int(round(k * src_fps / out_fps))) for k in range(n_out)]
    return frames[idx]


def temporal_spectrum(frames):
    """(T,H,W,3)uint8 -> normalized temporal power spectrum over freqs 1..T//2."""
    g = frames.astype(np.float32).mean(3) / 255.0        # (T,H,W) luminance
    g = g - g.mean(0, keepdims=True)                      # remove per-pixel DC
    P = np.abs(np.fft.rfft(g, axis=0)) ** 2               # (F,H,W)
    P = P.mean((1, 2))                                    # (F,)
    P = P[1:]                                             # drop DC
    s = P.sum()
    return P / s if s > 0 else P


def frame_diversity(frames, lp, npairs=190):
    """mean pairwise LPIPS between frames (uint8 (T,H,W,3))."""
    T = len(frames)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(DEV) / 255.0
    x = x * 2 - 1
    pairs = list(itertools.combinations(range(T), 2))
    if len(pairs) > npairs:
        idx = np.linspace(0, len(pairs) - 1, npairs).round().astype(int)
        pairs = [pairs[i] for i in idx]
    with torch.no_grad():
        d = [lp(x[i:i+1], x[j:j+1]).item() for i, j in pairs]
    return float(np.mean(d))


def main():
    lp = lpips_mod.LPIPS(net="alex").to(DEV)
    rows, div_rows = [], []
    for code in CODES:
        name = NAMES[code]
        tf, tfps = read_frames(BASE / "targets" / f"{code}.mp4")
        gf, gfps = read_frames(BASE / os.environ.get("GSTN_DIR","gstn_synths") / f"{code}.mp4")
        dyn_full, dfps = read_frames(BASE / "dynca_synths" / code / code / "video.mp4")
        tgt = tf[:SPAN]                                    # target 15fps, 20 frames
        gstn = gf[:SPAN]                                   # GSTN 15fps, 20 frames (aligned)
        dyn = resample_to_target(dyn_full, dfps, SPAN, tfps)  # DyNCA 25fps -> target grid

        # --- temporal spectrum distance ---
        pt = temporal_spectrum(tgt)
        d_gstn = float(np.abs(temporal_spectrum(gstn) - pt).sum())
        d_dyn = float(np.abs(temporal_spectrum(dyn) - pt).sum())
        rows.append(dict(name=name, GSTN_specL1=d_gstn, DyNCA_specL1=d_dyn))

        # --- frame diversity ---
        dt = frame_diversity(tgt, lp)
        dg = frame_diversity(gstn, lp)
        dd = frame_diversity(dyn, lp)
        dd_long = frame_diversity(dyn_full[np.linspace(0, len(dyn_full)-1, SPAN).round().astype(int)], lp)
        div_rows.append(dict(name=name, target_div=dt, GSTN_div=dg, DyNCA_div=dd,
                             GSTN_ratio=dg/dt if dt else float('nan'),
                             DyNCA_ratio=dd/dt if dt else float('nan'),
                             DyNCA_div_longrollout=dd_long))
        print(f"{name:7s} | specL1  GSTN={d_gstn:.3f} DyNCA={d_dyn:.3f} | "
              f"div tgt={dt:.3f} GSTN={dg:.3f} DyNCA={dd:.3f} (long={dd_long:.3f})", flush=True)

    def mean(rs, k): return float(np.mean([r[k] for r in rs]))
    print("\n== Temporal spectrum L1 distance to target (lower=better) ==")
    print(f"  GSTN  mean = {mean(rows,'GSTN_specL1'):.3f}")
    print(f"  DyNCA mean = {mean(rows,'DyNCA_specL1'):.3f}")
    print("\n== Frame diversity (mean pairwise LPIPS; target = reference) ==")
    print(f"  target mean div        = {mean(div_rows,'target_div'):.3f}")
    print(f"  GSTN   mean div        = {mean(div_rows,'GSTN_div'):.3f}  "
          f"(ratio {mean(div_rows,'GSTN_ratio'):.2f})")
    print(f"  DyNCA  mean div        = {mean(div_rows,'DyNCA_div'):.3f}  "
          f"(ratio {mean(div_rows,'DyNCA_ratio'):.2f})")
    print(f"  DyNCA  long-rollout div= {mean(div_rows,'DyNCA_div_longrollout'):.3f}  "
          f"(vs target {mean(div_rows,'target_div'):.3f})")

    with open(BASE / "video_extra_spectrum.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(BASE / "video_extra_diversity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(div_rows[0])); w.writeheader(); w.writerows(div_rows)
    print(f"\nwrote video_extra_spectrum.csv, video_extra_diversity.csv")


if __name__ == "__main__":
    main()
