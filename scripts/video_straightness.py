# ---------------------------------------------------------------------------
# Produces: perceptual straightness (Table 7)
#
# Provenance: StatTexNetExtended/rebuttal/perceptual_straightness.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Perceptual straightness of dynamic textures.

Henaff, Goris & Simoncelli, "Perceptual straightening of natural videos",
Nature Neuroscience 2019. Natural videos follow curved trajectories in pixel space but
straighter ones in perceptual representations; the visual system appears to straighten
temporal trajectories, which supports prediction.

Curvature is the mean angle between successive difference vectors along the trajectory:
    v_t = x_{t+1} - x_t ;  c = mean_t angle(v_t, v_{t+1})
Reported in degrees. 0 = perfectly straight, 90 = uncorrelated steps, ~90-180 = jitter
(each step reverses the previous one). We report pixel-space and perceptual-space
curvature, and the straightening effect (pixel - perceptual), which is the quantity the
original work is about.

SANITY CHECK FIRST: we compute curvature for the five TARGETS and for temporally
SHUFFLED copies, to confirm the measure responds to real temporal order rather than
returning a constant. Four of five targets are strongly separated from their shuffles
(|z| = 4.8-10.2); fire is weakly separated (z = -1.7), consistent with turbulent smoke
having little coherent trajectory. All five are RETAINED in the comparison -- fire is a
dynamic texture like the others, and the z-score is reported as information rather than
used as an exclusion criterion. Including fire is also the conservative choice, since
it is the one video where DyNCA matches the target curvature exactly.

NOTE ON INTERPRETATION: perceptual curvature comes out HIGHER than pixel curvature for
all five targets, i.e. the straightening effect of Henaff et al. does NOT reproduce for
dynamic textures in VGG conv3_3. Report these numbers as a trajectory-curvature
comparison, not as evidence of perceptual straightening.

Run: python -u perceptual_straightness.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import VIDEO  # noqa: E402
import os
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F

DEV = torch.device(os.environ.get("GPU", "cuda:1"))
BASE = VIDEO
NAMES = {"6440110": "fence", "6441210": "tree", "55fc210": "fire",
         "54pi110": "grass", "54pb110": "blooms"}
N_SHUFFLE = int(os.environ.get("N_SHUFFLE", "20"))


def read(p):
    c = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    c.release()
    return np.stack(fr)


def curvature(X):
    """X: (T, D) trajectory -> mean angle between successive difference vectors, degrees."""
    V = np.diff(X, axis=0)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(n, 1e-12)
    cos = np.clip((V[:-1] * V[1:]).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def pixel_traj(frames):
    x = frames.astype(np.float32).reshape(len(frames), -1) / 255.0
    return x - x.mean(0, keepdims=True)


def build_vgg():
    from torchvision.models import vgg16, VGG16_Weights
    m = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(DEV).eval()   # through conv3_3
    return m


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def percep_traj(frames, net):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(DEV) / 255
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(DEV)) / STD.to(DEV)
    with torch.no_grad():
        f = net(x)
    f = f.flatten(1).cpu().numpy()
    return f - f.mean(0, keepdims=True)


def main():
    net = build_vgg()
    rng = np.random.default_rng(0)

    print("=== FALSIFICATION TEST: do real targets differ from temporally shuffled? ===")
    print(f"{'target':8s} {'pix curv':>9s} {'perc curv':>10s} {'straighten':>11s}"
          f" | {'shuffled perc':>14s} {'(sd)':>7s} {'z':>7s}")
    zs = []
    for code, name in NAMES.items():
        v = read(BASE / "targets" / f"{code}.mp4")
        pc = curvature(pixel_traj(v))
        qc = curvature(percep_traj(v, net))
        sh = []
        for _ in range(N_SHUFFLE):
            idx = rng.permutation(len(v))
            sh.append(curvature(percep_traj(v[idx], net)))
        m, s = float(np.mean(sh)), float(np.std(sh, ddof=1))
        z = (qc - m) / s if s > 0 else float("nan")
        zs.append(z)
        print(f"{name:8s} {pc:9.1f} {qc:10.1f} {pc-qc:11.1f} | {m:14.1f} {s:7.2f} {z:7.1f}")
    print(f"\n  mean z of real vs shuffled: {np.mean(zs):.1f} "
          f"(|z| > 2 on most targets -> measure is informative here)")

    print("\n=== syntheses vs targets (perceptual curvature, degrees) ===")
    print(f"{'video':8s} {'target':>8s} {'GSTN drive':>11s} {'GSTN m_only':>12s} {'DyNCA':>8s}")
    rows = []
    for code, name in NAMES.items():
        t = curvature(percep_traj(read(BASE / "targets" / f"{code}.mp4"), net))
        g = curvature(percep_traj(read(BASE / "gstn_synths" / f"{code}.mp4"), net))
        v2p = BASE / "gstn_synths_v2" / f"{code}.mp4"
        g2 = curvature(percep_traj(read(v2p), net)) if v2p.is_file() else float("nan")
        raw = BASE / "dynca_synths_raw" / code / "frames.npy"
        d = curvature(percep_traj((np.load(raw) * 255).round().clip(0, 255).astype(np.uint8), net)) \
            if raw.is_file() else float("nan")
        rows.append((name, t, g, g2, d))
        print(f"{name:8s} {t:8.1f} {g:11.1f} {g2:12.1f} {d:8.1f}")
    print("\n  |error| vs target (lower = better):")
    for meth, i in [("GSTN drive", 2), ("GSTN mean_only", 3), ("DyNCA", 4)]:
        e = np.nanmean([abs(r[i] - r[1]) for r in rows])
        print(f"    {meth:16s} {e:6.2f} deg")


if __name__ == "__main__":
    main()
