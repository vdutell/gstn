# ---------------------------------------------------------------------------
# Produces: per-image metric CSVs the figures read
#
# Provenance: diffusion-posterior-sampling/recompute_metrics_clean.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Re-compute DISTS, LPIPS, FSIM, SSIM, MS-SSIM cleanly: compare 256x256 synth
directly against the matched 256x256 target (from pixelbp_prodigy_targets/),
with no CenterCrop/Resize step.

The original calc_img_quality.py does CenterCrop(512)+Resize(256) on both
images. The 256x256 synths get padded with zeros to 512 then squeezed back to
256 — the actual synthesis content lives in the central ~128 with black
borders. This silently inflates every per-image perceptual distance.

Output:
    For each bench dir: writes per_image_metrics_<stem>_clean.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, METRICS# noqa: E402
import json
from pathlib import Path

import numpy as np
import piq
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

DPS = ASSETS
OUT = METRICS
TARGETS = DPS / "pixelbp_prodigy_targets"

METHODS = [
    ("Gaussian",       "gaussian",          "bench_gaussian"),
    ("P&S",            "ps",                "bench_ps"),
    ("Gatys",          "gatys",             "bench_gatys"),
    ("StatTexGramNet", "stgn_prodigy_full", "bench_stgn_prodigy_full"),
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.ToTensor()


def load_t(p):
    """Load a 256x256 PNG/JPG as [1,3,256,256] in [0,1]."""
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        # Synth or DTD-target should both be 256; fall back to resize if not.
        img = img.resize((256, 256), Image.BICUBIC)
    return to_tensor(img).unsqueeze(0).to(DEVICE)


def main():
    dists  = piq.DISTS(reduction="none").to(DEVICE)
    lpips_net = __import__("lpips").LPIPS(net="vgg").to(DEVICE)
    print(f"Models loaded on {DEVICE}")

    for disp, stem, dir_name in METHODS:
        in_csv = OUT / dir_name / f"per_image_metrics_{stem}.csv"
        out_csv = OUT / dir_name / f"per_image_metrics_{stem}_clean.csv"
        df = pd.read_csv(in_csv)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=disp):
            target_p = TARGETS / f"{row['key']}.png"
            if not target_p.exists():
                # fall back to whatever the CSV pointed at
                target_p = Path(row["real"])
            r = load_t(target_p)
            s = load_t(row["synth"])
            with torch.no_grad():
                d_dists  = dists(r, s).mean().item()
                d_lpips  = lpips_net(r * 2 - 1, s * 2 - 1).mean().item()
                d_psnr   = piq.psnr(r, s, data_range=1.0).item()
                d_ssim   = piq.ssim(r, s, data_range=1.0).item()
                d_msssim = piq.multi_scale_ssim(r, s, data_range=1.0).item()
                d_fsim   = piq.fsim(r, s).item()
                d_mse    = ((r - s) ** 2).mean().item()
            rows.append({
                "key": row["key"], "real": str(target_p), "synth": row["synth"],
                "dists": d_dists, "lpips": d_lpips,
                "ssim": d_ssim, "ms_ssim": d_msssim, "fsim": d_fsim,
                "psnr": d_psnr, "mse": d_mse,
            })
        clean_df = pd.DataFrame(rows)
        clean_df.to_csv(out_csv, index=False)
        means = {k: float(clean_df[k].mean()) for k in
                 ["dists", "lpips", "ssim", "ms_ssim", "fsim", "psnr", "mse"]}
        print(f"  {disp:18s}  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))


if __name__ == "__main__":
    main()
