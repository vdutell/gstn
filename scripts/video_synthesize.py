# ---------------------------------------------------------------------------
# Produces: video synthesis driver
#
# Provenance: MoTexNet/regenerate_video_synths.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Regenerate all 5 evaluation video synths with the current code.

Stage 1 (A/B, fence only, 200 iters): does porting the image model's winning encoder
normalization to the video model help?
    no_norm    -- current video behaviour (all normalization commented out, raw [0,1])
    mean_only  -- vid - vid.mean(); DC removal. This is the image model's winning config
                  (mean_only + zero-pad beat z_score: DISTS -11%, LPIPS -10%,
                  SSIM +35%, MS-SSIM +74%) and was never ported to video. The line
                  exists in stattexgramnetvideo.py:428 but is commented out.
Stage 2: regenerate all 5 videos at 500 iters with whichever won, save mp4 + FID.

Marginals stay OFF -- already A/B'd (367.7 off vs 369.1 on, no effect; histogram
matching at weight 10000 already pins the marginals).

FID via torcheval FrechetInceptionDistance == the paper's Table 3 pipeline.
Reference (drive synths, 500 iters): fence 383, fire 79, grass 190, 54pi110 95, 54pb110 82.

Run: python -u regenerate_video_synths.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import REPO, VIDEO  # noqa: E402
import sys, types, csv, os
import numpy as np, cv2, torch
import torch.nn as nn, torch.optim as optim
import torchvision
sys.path.insert(0, str(REPO))
import utils.stattexgramnetvideo as stn
from pytorch_histogram_matching import Histogram_Matching
from torcheval.metrics import FrechetInceptionDistance

DEV = "cuda:0"; T, H, W = 20, 64, 64
AB_ITERS, FULL_ITERS = 200, 500
TGT = str(VIDEO / "targets")
OUT = str(VIDEO / "gstn_synths_v2")
os.makedirs(OUT, exist_ok=True)
FENCE = "6440110"
# code -> label, and the drive-synth FID for reference
ALL = [("6441210", "grass", 190), ("6440110", "fence", 383), ("55fc210", "fire", 79),
       ("54pi110", "54pi110", 95), ("54pb110", "54pb110", 82)]
FILTERS = "./filter_banks_normFalse_64px_20fr/filters_torch.pkl"


def load(code):
    c = cv2.VideoCapture(f"{TGT}/{code}.mp4"); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    c.release()
    return torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).unsqueeze(0).float().to(DEV) / 255


def enc_mean_only(self, vid):
    """The image model's winning normalization, ported: DC removal before the pyramid."""
    vid = vid - torch.mean(vid)
    x = self.pyr_encoder(vid)
    return self.calcgramm(x, triu_idxes=self.triu_idxes)


def synth(code, mode, n_iters, seed=10, save=False):
    target = load(code)
    m = stn.StatTexGramNetVideo(
        movie_size=(H, W, T), scales=[1, 2, 4, 8, 16],
        oris=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4],
        heights=[1, 2, 4, 8, 16], batch_size=1, bottleneck_size=None, device=DEV,
        compress=False, color=True, add_marginals=False, filter_location=FILTERS).to(DEV)
    if mode == "mean_only":
        m.encoder = types.MethodType(enc_mean_only, m)

    with torch.no_grad():
        tgt_act = m(target)
    torch.manual_seed(seed)
    s = (torch.randn([1, 3, T, H, W], device=DEV) * 0.1).requires_grad_(True)
    opt = optim.LBFGS([s], line_search_fn="strong_wolfe")
    HM = Histogram_Matching(differentiable=False); mse = nn.MSELoss()

    def hmatch(x):
        return torch.stack([HM(x[:, i].reshape(1, 1, T, -1),
                               target[:, i].reshape(1, 1, T, -1)).view(1, T, H, W)
                            for i in range(3)], dim=1)

    def closure():
        opt.zero_grad()
        with torch.no_grad():
            s.sub_(s.min()).div_(s.max() - s.min() + 1e-8); s.copy_(hmatch(s))
        loss = mse(m(s), tgt_act); loss.backward(); return loss

    for _ in range(n_iters):
        opt.step(closure)
    with torch.no_grad():
        s.sub_(s.min()).div_(s.max() - s.min() + 1e-8)
        final = hmatch(s).clamp(0, 1)

    fid = FrechetInceptionDistance().to(DEV); fid.reset()
    fid.update(final[0].permute(1, 0, 2, 3), is_real=False)
    fid.update(target[0].permute(1, 0, 2, 3), is_real=True)
    f = fid.compute().item()
    if save:
        vid = (final[0].permute(1, 2, 3, 0) * 255).byte().cpu()
        torchvision.io.write_video(f"{OUT}/{code}.mp4", vid, fps=15,
                                   video_codec="h264", options={"crf": "16"})
    return f


print(f"=== STAGE 1: encoder-normalization A/B on fence ({AB_ITERS} iters) ===", flush=True)
ab = {}
for mode in ["no_norm", "mean_only"]:
    ab[mode] = synth(FENCE, mode, AB_ITERS)
    print(f"  {mode:10s} FID={ab[mode]:.1f}", flush=True)
win = min(ab, key=ab.get)
delta = 100 * (ab["mean_only"] - ab["no_norm"]) / ab["no_norm"]
print(f"  -> mean_only is {delta:+.1f}% vs no_norm; winner = {win}", flush=True)
if abs(delta) < 2.0:
    win = "no_norm"
    print("  (within 2% = tie; keeping no_norm, the config the paper used)", flush=True)

print(f"\n=== STAGE 2: regenerating all 5 with mode={win} ({FULL_ITERS} iters) ===", flush=True)
rows = []
for code, label, ref in ALL:
    f = synth(code, win, FULL_ITERS, save=True)
    d = 100 * (f - ref) / ref
    rows.append((code, label, win, f, ref, d))
    print(f"  {label:8s} ({code})  FID={f:6.1f}   drive={ref:5.0f}  {d:+6.1f}%   -> {OUT}/{code}.mp4",
          flush=True)

with open(f"{OUT}/fids_v2.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["code", "label", "encoder_mode", "fid_new", "fid_drive", "pct_change"])
    w.writerows(rows)
mean_new = np.mean([r[3] for r in rows]); mean_ref = np.mean([r[4] for r in rows])
print(f"\n  mean FID: new {mean_new:.1f}  vs  drive {mean_ref:.1f}  "
      f"({100*(mean_new-mean_ref)/mean_ref:+.1f}%)", flush=True)
print("DONE", flush=True)
