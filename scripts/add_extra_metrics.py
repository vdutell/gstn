# ---------------------------------------------------------------------------
# Produces: SIFID and DreamSim helpers
#
# Provenance: diffusion-posterior-sampling/add_extra_metrics.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Compute SIFID, DreamSim, CMMD on the 4 bench methods (108-stem intersection)
and the compressed-100d run, all paired against the 256x256 matched targets.

Outputs: appends columns 'sifid', 'dreamsim' to each per_image_metrics_*_clean.csv,
and a separate cmmd_summary.json (set-level, no per-image CIs).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, METRICS# noqa: E402
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from scipy import linalg as la

from dreamsim import dreamsim
import open_clip

DPS = ASSETS
OUT = METRICS
TARGETS = DPS / "pixelbp_prodigy_targets"

METHODS = [
    ("Gaussian",       "gaussian",          "bench_gaussian"),
    ("P&S",            "ps",                "bench_ps"),
    ("Gatys",          "gatys",             "bench_gatys"),
    ("StatTexGramNet", "stgn_prodigy_full", "bench_stgn_prodigy_full"),
    ("STGN-100d",      "bench_stgn_compressed100", "bench_stgn_compressed100"),
    ("Ustyuzhaninov",  "ustyuzhaninov",     "bench_ustyuzhaninov"),
]

DEVICE = torch.device("cuda:0")
to_tensor = transforms.ToTensor()


def load_t(p):
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    return to_tensor(img).unsqueeze(0).to(DEVICE)


# ---------- SIFID ----------
@torch.no_grad()
def sifid_pair(model, extractor, tf_inception, real_pil, synth_pil,
               n_patches=64, patch_size=128):
    """Single-Image FID: extract Inception pool3 features from many patches
    of real vs synth, compute FID between the two patch sets."""
    rng = np.random.default_rng(0)
    H, W = 256, 256
    coords = [(rng.integers(0, H - patch_size), rng.integers(0, W - patch_size))
              for _ in range(n_patches)]

    def feats(pil_img):
        t = transforms.ToTensor()(pil_img)   # [3, 256, 256]
        patches = []
        for y, x in coords:
            p = t[:, y:y + patch_size, x:x + patch_size].unsqueeze(0)
            p = F.interpolate(p, size=(299, 299), mode="bilinear", antialias=True)
            patches.append(p)
        x = torch.cat(patches, dim=0).to(DEVICE)
        # tf_inception expects PIL; we pre-resize ourselves and only do the
        # ImageNet normalize:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        x = norm(x)
        out = extractor(x)["pool_3"]
        return torch.flatten(out, 1).cpu().numpy()

    Fr = feats(real_pil)
    Fs = feats(synth_pil)
    mu1, mu2 = Fr.mean(0), Fs.mean(0)
    C1, C2 = np.cov(Fr, rowvar=False), np.cov(Fs, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = la.sqrtm(C1.dot(C2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(C1 + C2 - 2 * covmean))


# ---------- CMMD (CLIP-MMD) ----------
@torch.no_grad()
def get_clip_features(paths, model, preprocess):
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def cmmd(x, y, sigma=10.0):
    """Compute MMD^2 with a Gaussian kernel between feature sets x, y."""
    def k(a, b):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2 * sigma ** 2))
    Kxx = k(x, x); Kyy = k(y, y); Kxy = k(x, y)
    n, m = len(x), len(y)
    # unbiased
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()


def main():
    # SIFID setup
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inc = inception_v3(weights=weights).eval().to(DEVICE)
    inc_extractor = create_feature_extractor(inc, return_nodes={"avgpool": "pool_3"})
    inc_tf = weights.transforms()
    print("inception loaded for SIFID")

    # DreamSim setup
    ds_model, ds_preprocess = dreamsim(pretrained=True, device=DEVICE)
    print("DreamSim loaded")

    # CLIP for CMMD
    clip_model, _, clip_pp = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model = clip_model.to(DEVICE).eval()
    print("CLIP ViT-B/32 loaded for CMMD")

    cmmd_results = {}
    for disp, stem, dir_name in METHODS:
        in_csv = OUT / dir_name / f"per_image_metrics_{stem}_clean.csv"
        if not in_csv.is_file():
            print(f"[skip] {disp}: csv not found at {in_csv}")
            continue
        df = pd.read_csv(in_csv)
        if "sifid" in df.columns and "dreamsim" in df.columns:
            print(f"[skip] {disp}: already done")
            # still load for CMMD
        else:
            sifids, dsims = [], []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"SIFID+DS {disp}"):
                tgt = TARGETS / f"{row['key']}.png"
                if not tgt.exists():
                    tgt = Path(row["real"])
                rpil = Image.open(tgt).convert("RGB").resize((256, 256), Image.BICUBIC)
                spil = Image.open(row["synth"]).convert("RGB").resize((256, 256), Image.BICUBIC)
                sifids.append(sifid_pair(inc, inc_extractor, inc_tf, rpil, spil))
                rt = ds_preprocess(rpil).to(DEVICE)
                st = ds_preprocess(spil).to(DEVICE)
                with torch.no_grad():
                    dsims.append(float(ds_model(rt, st).item()))
            df["sifid"] = sifids
            df["dreamsim"] = dsims
            df.to_csv(in_csv, index=False)
            print(f"  {disp:18s} SIFID mean = {np.mean(sifids):.3f}  DreamSim mean = {np.mean(dsims):.4f}")

        # CMMD: set-level CLIP MMD between real-set and synth-set
        real_paths = [str(TARGETS / f"{k}.png") if (TARGETS / f"{k}.png").exists()
                      else r for k, r in zip(df["key"], df["real"])]
        synth_paths = list(df["synth"])
        rf = get_clip_features(real_paths, clip_model, clip_pp)
        sf = get_clip_features(synth_paths, clip_model, clip_pp)
        d = cmmd(rf, sf)
        cmmd_results[disp] = float(d)
        print(f"  {disp:18s} CMMD = {d:.5f}")

    (DPS / "cmmd_summary.json").write_text(json.dumps(cmmd_results, indent=2))
    print(f"\nwrote {DPS / 'cmmd_summary.json'}")


if __name__ == "__main__":
    main()
