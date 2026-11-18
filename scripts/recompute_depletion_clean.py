# ---------------------------------------------------------------------------
# Produces: per-image metrics for the depletion conditions
#
# Provenance: diffusion-posterior-sampling/recompute_depletion_clean.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Same as recompute_metrics_clean.py but for the 13 depletion conditions."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, METRICS# noqa: E402
from pathlib import Path
import numpy as np, piq, torch, pandas as pd, lpips
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

DPS = ASSETS
OUT = METRICS
TARGETS = DPS / "pixelbp_prodigy_targets"

CONDITIONS = [
    "full",
    "top100", "top1000", "top5000", "top10000",
    "rand100", "rand1000", "rand5000", "rand10000",
    "bot100", "bot1000", "bot5000", "bot10000",
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()


def load_t(p):
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    return to_tensor(img).unsqueeze(0).to(DEVICE)


def main():
    dists = piq.DISTS(reduction="none").to(DEVICE)
    lpips_net = lpips.LPIPS(net="vgg").to(DEVICE)
    print(f"Models loaded on {DEVICE}")

    for suf in CONDITIONS:
        d = OUT / f"depletion_{suf}"
        base = f"pixelbp_prodigy_depleted_{suf}"
        in_csv = d / f"per_image_metrics_{base}.csv"
        out_csv = d / f"per_image_metrics_{base}_clean.csv"
        if not in_csv.is_file():
            print(f"[skip] {suf}")
            continue
        df = pd.read_csv(in_csv)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=suf):
            target_p = TARGETS / f"{row['key']}.png"
            if not target_p.exists():
                target_p = Path(row["real"])
            r = load_t(target_p)
            s = load_t(row["synth"])
            with torch.no_grad():
                rows.append({
                    "key": row["key"], "real": str(target_p), "synth": row["synth"],
                    "dists":  dists(r, s).mean().item(),
                    "lpips":  lpips_net(r * 2 - 1, s * 2 - 1).mean().item(),
                    "ssim":   piq.ssim(r, s, data_range=1.0).item(),
                    "ms_ssim": piq.multi_scale_ssim(r, s, data_range=1.0).item(),
                    "fsim":   piq.fsim(r, s).item(),
                    "psnr":   piq.psnr(r, s, data_range=1.0).item(),
                    "mse":    ((r - s) ** 2).mean().item(),
                })
        cdf = pd.DataFrame(rows)
        cdf.to_csv(out_csv, index=False)
        print(f"  {suf:10s}  DISTS = {cdf['dists'].mean():.4f}  "
              f"LPIPS = {cdf['lpips'].mean():.4f}  N={len(cdf)}")


if __name__ == "__main__":
    main()
