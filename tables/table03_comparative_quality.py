# ---------------------------------------------------------------------------
# Produces: Table 3
#
# Provenance: diffusion-posterior-sampling/quant_method_comparison_munca.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Method-level quantitative comparison matching fig_method_comparison.png.

For each method, compute DISTS / LPIPS / SSIM / MS-SSIM / FSIM / PSNR / MSE /
SIFID / DreamSim on whichever stems are available. Two summaries:

  (A) 11-stem qualitative subset (matches the figure; zpad has full coverage here)
  (B) Full-141 bench (methods missing data are skipped)

Output:
    fig_method_quant_munca_perstem.csv   per-method per-stem
    fig_method_quant_munca_means.csv     per-method means at each subset
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, BASELINES, DTD, DTD_LABELS, MUNCA, TABLES# noqa: E402
import os
import re
from pathlib import Path

import numpy as np
import torch
import lpips
import piq
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_extra_metrics import sifid_pair
import add_extra_metrics as _aem

DEVICE = torch.device(os.environ.get("GPU", "cuda:0")
                      if torch.cuda.is_available() else "cpu")
_aem.DEVICE = DEVICE

DPS = ASSETS
DTD = DTD
TARGETS = DPS / "pixelbp_prodigy_targets"

METHODS = [
    ("GSTN-wrap",     DPS / "synths/pixelbp_prodigy_depleted_full_meanonly_wrap",
     re.compile(r"^(.+)_gpu[01]_synth\.png$")),
    ("GSTN-zpad",     DPS / "synths/stgn_meanonly_black_lr002_qualitative",
     re.compile(r"^(.+)_synth\.png$")),
    ("P&S",           BASELINES / "dtd_121_ps_synths",
     re.compile(r"^\d+_psynth_\d+_(.+)\.jpg$")),
    ("Gatys",         BASELINES / "dtd_121_gatys",
     re.compile(r"^(.+)_gen_[\d\-]+\.png$")),
    ("Ustyuzhaninov", DPS / "synths/bench_ustyuzhaninov_multiscale_pytorch",
     re.compile(r"^(.+)_synth\.png$")),
    # muNCA: 11 Figure-2 textures only, trained at 256 with the authors' canonical
    # notebook. polka-dotted_0140 collapsed to a constant image -- it is INCLUDED here
    # deliberately, because excluding a method's failures inflates its mean. Report the
    # failure alongside the number.
    ("muNCA", Path(str(MUNCA / "fig2_8k_256")),
     re.compile(r"^(.+)_munca\.png$")),
]

SUBSET_11 = [
    "chequered_0177", "striped_0050", "polka-dotted_0140", "honeycombed_0149",
    "cracked_0056", "lacelike_0004", "fibrous_0112", "bumpy_0077",
    "sprinkled_0042", "pleated_0136", "swirly_0065",
]

VAL_LIST = Path(str(DTD_LABELS / "val1_3sampled.txt"))


def psd_l2(real_t, synth_t):
    """L2 distance between log power spectral densities. Same definition as
    augment_metrics_vif_gmsd_haarpsi_psd.py so the numbers are comparable to the
    existing CSVs."""
    def psd(x):
        gray = x.mean(dim=1, keepdim=False)
        F = torch.fft.fft2(gray)
        return torch.log1p(F.abs() ** 2)
    return (psd(real_t) - psd(synth_t)).pow(2).mean().sqrt().item()


def load_t(p):
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    return transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)


def index_dir(d, regex):
    out = {}
    if not d.is_dir():
        return out
    for p in d.iterdir():
        m = regex.match(p.name)
        if m:
            out[m.group(1)] = p
    return out


def main():
    print(f"device: {DEVICE}")
    dists = piq.DISTS(reduction="none").to(DEVICE)
    lpips_net = lpips.LPIPS(net="vgg").to(DEVICE)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inc = inception_v3(weights=weights).eval().to(DEVICE)
    inc_extractor = create_feature_extractor(inc, return_nodes={"avgpool": "pool_3"})
    inc_tf = weights.transforms()
    from dreamsim import dreamsim
    ds_model, ds_preprocess = dreamsim(pretrained=True, device=DEVICE)

    # Build the 141-stem set from the val file
    val_stems_141 = []
    for line in VAL_LIST.read_text().splitlines():
        line = line.strip()
        if line:
            val_stems_141.append(line.split("/", 1)[1].rsplit(".", 1)[0])
    print(f"141-bench stems: {len(val_stems_141)}")

    rows = []
    for name, dir_, regex in METHODS:
        idx = index_dir(dir_, regex)
        if not idx:
            print(f"[skip] {name}: no synths in {dir_}")
            continue
        print(f"\n=== {name}: {len(idx)} synths in {dir_.name} ===")
        stems = sorted(idx.keys())
        if os.environ.get("ONLY_11", "1") == "1":
            stems = [t for t in stems if t in SUBSET_11]
        for stem in tqdm(stems, desc=name):
            target_p = TARGETS / f"{stem}.png"
            if not target_p.is_file():
                continue
            synth_p = idx[stem]
            r = load_t(target_p)
            s = load_t(synth_p)
            with torch.no_grad():
                row = {
                    "method": name, "stem": stem,
                    "in_subset_11":  stem in SUBSET_11,
                    "in_full_141":   stem in val_stems_141,
                    "dists":   dists(r, s).mean().item(),
                    "lpips":   lpips_net(r * 2 - 1, s * 2 - 1).mean().item(),
                    "ssim":    piq.ssim(r, s, data_range=1.0).item(),
                    "ms_ssim": piq.multi_scale_ssim(r, s, data_range=1.0).item(),
                    "fsim":    piq.fsim(r, s).item(),
                    "psnr":    piq.psnr(r, s, data_range=1.0).item(),
                    "mse":     ((r - s) ** 2).mean().item(),
                    "vif":     piq.vif_p(r, s, data_range=1.0).item(),
                    "gmsd":    piq.gmsd(r, s, data_range=1.0).item(),
                    "haarpsi": piq.haarpsi(r, s, data_range=1.0).item(),
                    "psd_l2":  psd_l2(r, s),
                }
                rpil = Image.open(target_p).convert("RGB").resize((256, 256), Image.BICUBIC)
                spil = Image.open(synth_p).convert("RGB").resize((256, 256), Image.BICUBIC)
                row["sifid"] = sifid_pair(inc, inc_extractor, inc_tf, rpil, spil)
                rt = ds_preprocess(rpil).to(DEVICE)
                st = ds_preprocess(spil).to(DEVICE)
                row["dreamsim"] = float(ds_model(rt, st).item())
            rows.append(row)

    if not rows:
        print("no rows; nothing to report")
        return

    df = pd.DataFrame(rows)
    out_csv = TABLES / "fig_method_quant_munca_perstem.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    metric_cols = ["dists", "lpips", "ssim", "ms_ssim", "fsim",
                   "psnr", "mse", "sifid", "dreamsim"]

    summaries = []
    for subset_name, mask_col in [("11-stem", "in_subset_11"),
                                  ("141-stem", "in_full_141")]:
        sub = df[df[mask_col]]
        if sub.empty:
            continue
        means = (sub.groupby("method")[metric_cols].mean()
                    .assign(n=sub.groupby("method").size()))
        means["subset"] = subset_name
        summaries.append(means.reset_index())
        print(f"\n=== {subset_name} means ===")
        print(means.to_string(float_format="%.4f"))

    if summaries:
        all_means = pd.concat(summaries)
        means_csv = TABLES / "fig_method_quant_munca_means.csv"
        all_means.to_csv(means_csv, index=False)
        print(f"\nwrote {means_csv}")


if __name__ == "__main__":
    main()
