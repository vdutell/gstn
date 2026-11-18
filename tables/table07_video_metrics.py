# ---------------------------------------------------------------------------
# Produces: Table 7
#
# Provenance: StatTexNetExtended/rebuttal/compute_video_metrics.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Dynamic-texture comparison metrics: GSTN vs DyNCA vs target (NeurIPS rebuttal).

Paper Table 3 metrics (per video, appearance/alignment-based):
    LPIPS(Alex), LPIPS(VGG), SSIM, FID, MSE, PSNR
Added video metrics (dataset-level, motion-aware, I3D spatiotemporal features):
    FVD (Frechet Video Distance)  +  KVD (Kernel Video Distance, unbiased at small n)

All three sources measured with IDENTICAL code -> GSTN-vs-DyNCA is apples-to-apples.
Recomputed GSTN numbers may differ slightly from Table 3 (implementation), which we note.

Run: python compute_video_metrics.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import I3D_MODEL, VIDEO  # noqa: E402
import os, sys, glob
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
from scipy import linalg
import lpips as lpips_mod
import piq

DEV = torch.device("cuda:0")
BASE = VIDEO
CODES = ["54pi110", "54pb110", "55fc210", "6441210", "6440110"]
NAMES = {"54pi110": "grass", "54pb110": "blooms", "55fc210": "fire",
         "6440110": "fence", "6441210": "tree"}
# Table 3 evaluates codes 6440110, 6441210, 55fc210 (per Calc_quality.ipynb's
# synth_names) = fence, tree, fire. Note that notebook labels 6440110 "tree1", which
# is wrong for its content (it is a chain-link fence); the NAMES above are correct.
IN_PAPER = {"fence", "tree", "fire"}
I3D_PT = str(I3D_MODEL)


def read_frames(path, rgb=True):
    c = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if rgb else f)
    c.release()
    return np.stack(fr)  # (T,H,W,3) uint8


def to_t(frames):  # (T,H,W,3) uint8 -> (T,3,H,W) float [0,1] on DEV
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(DEV) / 255.0


def sample_n(frames, n):  # evenly sample n frames -- DO NOT use for DyNCA, see below
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def sample_on_target_grid(frames, n=20, src_fps=25.0, out_fps=15.0):
    """Resample onto the TARGET's temporal grid.

    DyNCA renders 400 frames at 25fps (16 s); the target is 20 frames at 15fps
    (1.33 s). Spreading 20 samples evenly across the whole rollout -- which sample_n()
    does -- makes consecutive DyNCA frames nearly independent, so it measures FID over
    a 12x longer temporal window than GSTN, aliases the temporal spectrum, and inflates
    frame diversity. Measured effect on fence: FID 118.5 (16 s window) vs 173.9 (target
    grid), a 47% error that systematically flattered DyNCA. Matches resample_to_target()
    in compute_video_extra.py and the sampling in rerender_dynca_lossless.py.
    """
    idx = [min(len(frames) - 1, int(round(k * src_fps / out_fps))) for k in range(n)]
    return frames[idx]


# ---------- appearance / per-frame paper metrics ----------
def paper_metrics(tgt, syn, lp_a, lp_v):
    """tgt, syn: (T,H,W,3) uint8, same T. Per-frame paired, averaged."""
    T = min(len(tgt), len(syn))
    t = to_t(tgt[:T]); s = to_t(syn[:T])
    with torch.no_grad():
        mse = F.mse_loss(s, t).item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        ssim = piq.ssim(s, t, data_range=1.0).item()
        la = lp_a(s * 2 - 1, t * 2 - 1).mean().item()
        lv = lp_v(s * 2 - 1, t * 2 - 1).mean().item()
    return dict(LPIPS_Alex=la, LPIPS_VGG=lv, SSIM=ssim, MSE=mse, PSNR=psnr)


# ---------- FID (inception) ----------
def build_inception():
    from torchvision.models import inception_v3, Inception_V3_Weights
    m = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(DEV).eval()
    m.fc = torch.nn.Identity()  # -> 2048-d pool features
    return m


def inception_feats(frames, incep):  # (T,H,W,3)uint8 -> (T,2048)
    x = to_t(frames)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    with torch.no_grad():
        return incep(x).cpu().numpy()


# ---------- I3D features for FVD/KVD ----------
def build_i3d():
    return torch.jit.load(I3D_PT).eval().to(DEV)


def i3d_feats(frames, det, clip_len=16, stride=1, max_clips=None):
    """frames (T,H,W,3)uint8 -> (Nclips,400). Consecutive windows, resized 224, [-1,1]."""
    x = to_t(frames)  # (T,3,H,W)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x * 2 - 1
    T = x.shape[0]
    starts = list(range(0, max(1, T - clip_len + 1), stride))
    if max_clips and len(starts) > max_clips:
        starts = list(np.linspace(0, starts[-1], max_clips).round().astype(int))
    feats = []
    for s0 in starts:
        clip = x[s0:s0 + clip_len]
        if clip.shape[0] < clip_len:  # pad short
            clip = torch.cat([clip, clip[-1:].repeat(clip_len - clip.shape[0], 1, 1, 1)], 0)
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1,3,T,224,224)
        with torch.no_grad():
            feats.append(det(clip, rescale=False, resize=False, return_features=True).cpu().numpy())
    return np.concatenate(feats, 0)


# ---------- distances ----------
def frechet(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False); s2 = np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def poly_mmd(f1, f2):  # KID/KVD cubic-kernel unbiased MMD^2 (x1000)
    d = f1.shape[1]
    def k(a, b): return (a @ b.T / d + 1) ** 3
    m, n = len(f1), len(f2)
    kxx, kyy, kxy = k(f1, f1), k(f2, f2), k(f1, f2)
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    mmd = kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean()
    return float(mmd * 1000)


def main():
    lp_a = lpips_mod.LPIPS(net="alex").to(DEV)
    lp_v = lpips_mod.LPIPS(net="vgg").to(DEV)
    incep = build_inception(); det = build_i3d()

    methods = {"GSTN": BASE / os.environ.get("GSTN_DIR", "gstn_synths"), "DyNCA": BASE / "dynca_synths"}
    have = [c for c in CODES if (BASE / "dynca_synths" / c / c / "video.mp4").exists()]
    print(f"videos with DyNCA output: {have}", flush=True)

    # per-video appearance metrics + FID
    rows = []
    feat_bank = {"target": {"incep": [], "i3d": []},
                 "GSTN": {"incep": [], "i3d": []}, "DyNCA": {"incep": [], "i3d": []}}
    for code in have:
        tgt = read_frames(BASE / "targets" / f"{code}.mp4")
        srcs = {"GSTN": read_frames(methods["GSTN"] / f"{code}.mp4"),
                "DyNCA": sample_on_target_grid(
                    read_frames(methods["DyNCA"] / code / code / "video.mp4"), 20)}
        tfi = inception_feats(tgt, incep); t3 = i3d_feats(tgt, det)
        feat_bank["target"]["incep"].append(tfi); feat_bank["target"]["i3d"].append(t3)
        for meth, syn in srcs.items():
            pm = paper_metrics(tgt, syn, lp_a, lp_v)
            sfi = inception_feats(syn, incep)
            pm["FID"] = frechet(tfi, sfi)
            pm.update(video=code, name=NAMES.get(code, code), method=meth)
            rows.append(pm)
            feat_bank[meth]["incep"].append(sfi)
            feat_bank[meth]["i3d"].append(i3d_feats(syn, det, max_clips=8))
            print(f"  {code:9s} {meth:6s} "
                  f"LPIPS_A={pm['LPIPS_Alex']:.3f} SSIM={pm['SSIM']:.3f} "
                  f"FID={pm['FID']:.1f} PSNR={pm['PSNR']:.1f}", flush=True)

    # dataset-level FVD / KVD (pooled clips across videos)
    def pool(m, k): return np.concatenate(feat_bank[m][k], 0)
    tgt_i3d = pool("target", "i3d")
    print("\n== dataset-level video metrics (I3D) ==", flush=True)
    for meth in ["GSTN", "DyNCA"]:
        m_i3d = pool(meth, "i3d")
        fvd = frechet(tgt_i3d, m_i3d); kvd = poly_mmd(tgt_i3d, m_i3d)
        print(f"  {meth:6s}  FVD={fvd:8.1f}   KVD(x1e3)={kvd:8.3f}   "
              f"(clips: tgt={len(tgt_i3d)} {meth}={len(m_i3d)})", flush=True)

    # per-video table
    print("\n== per-video appearance metrics ==", flush=True)
    hdr = ["name", "video", "method", "LPIPS_Alex", "LPIPS_VGG", "SSIM", "FID", "MSE", "PSNR"]
    print("  " + "  ".join(f"{h:>10s}" for h in hdr), flush=True)
    for r in rows:
        print("  " + "  ".join(f"{r[h]:>10.3f}" if isinstance(r[h], float) else f"{r[h]:>10s}"
                               for h in hdr), flush=True)
    import csv
    with open(BASE / "video_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {BASE/'video_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
