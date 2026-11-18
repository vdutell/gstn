# ---------------------------------------------------------------------------
# Produces: FVMD metric (Table 7)
#
# Provenance: StatTexNetExtended/rebuttal/fvmd_run.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""FVMD (Frechet Video Motion Distance) for the dynamic-texture comparison.

PIPs++ point tracking -> velocity/acceleration histograms -> Frechet distance.
We SANITY-CHECK trajectory coherence on textures before trusting the number:
dynamic textures have no persistent keypoints, so tracking may be degenerate.

Run: FVMD_METHOD=gstn python fvmd_run.py   (or dynca)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import FVMD_REPO, VIDEO  # noqa: E402
import os, sys, shutil
from pathlib import Path
import numpy as np, cv2
from PIL import Image

sys.path.insert(0, str(FVMD_REPO))
BASE = VIDEO
WORK = BASE / "fvmd_data"
NAMES = {"54pi110": "grass", "54pb110": "blooms", "55fc210": "fire",
         "6440110": "fence", "6441210": "tree"}
METHOD = os.environ.get("FVMD_METHOD", "gstn")


def read_frames(p):
    c = cv2.VideoCapture(str(p)); fr = []; fps = c.get(5)
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    c.release()
    return np.stack(fr), fps


def resample(frames, src_fps, n=20, out_fps=15.0):
    idx = [min(len(frames) - 1, int(round(k * src_fps / out_fps))) for k in range(n)]
    return frames[idx]


def dump(frames, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(folder / f"f{i:03d}.png")


def setup():
    if WORK.exists(): shutil.rmtree(WORK)
    for code, name in NAMES.items():
        tgt, _ = read_frames(BASE / "targets" / f"{code}.mp4")
        dump(tgt[:20], WORK / "target" / name)
        gstn, _ = read_frames(BASE / "gstn_synths" / f"{code}.mp4")
        dump(gstn[:20], WORK / "gstn" / name)
        dyn, dfps = read_frames(BASE / "dynca_synths" / code / code / "video.mp4")
        dump(resample(dyn, dfps), WORK / "dynca" / name)
    print("dumped frames for target/gstn/dynca", flush=True)


def coherence(log_dir):
    """Diagnose whether PIPs++ trajectories are meaningful on textures."""
    tr = np.load(Path(log_dir) / "trajctorys_gt.npy")   # (Nclips*?, S, N, 2) or similar
    ve = np.load(Path(log_dir) / "velocity_gt.npy")
    print(f"\n== trajectory sanity (target) ==  traj{tr.shape} velo{ve.shape}", flush=True)
    speed = np.linalg.norm(ve.reshape(-1, ve.shape[-1]), axis=-1)
    print(f"  point speed (px/frame @256): mean={speed.mean():.3f} median={np.median(speed):.3f} "
          f"p95={np.percentile(speed,95):.3f} max={speed.max():.3f}", flush=True)
    print(f"  frac near-zero (<0.5px): {(speed<0.5).mean():.2%}", flush=True)
    # trajectory smoothness: jerk = |d(acc)| relative to speed
    if tr.ndim >= 3:
        t = tr.reshape(tr.shape[0], tr.shape[1], -1, 2) if tr.ndim == 4 else tr
        disp = np.diff(t, axis=1)
        jerk = np.linalg.norm(np.diff(disp, n=2, axis=1), axis=-1)
        print(f"  mean jerk (lower=smoother): {jerk.mean():.4f}", flush=True)


def main():
    setup()
    from fvmd.fvmd import fvmd as fvmd_score
    log_dir = str(WORK / f"log_{METHOD}")
    score = fvmd_score(log_dir=log_dir, gen_path=str(WORK / METHOD), gt_path=str(WORK / "target"))
    print(f"\nFVMD ({METHOD} vs target) = {score}", flush=True)
    coherence(log_dir)


if __name__ == "__main__":
    main()
