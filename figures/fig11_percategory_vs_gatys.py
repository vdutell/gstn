# ---------------------------------------------------------------------------
# Produces: Figure 11
#
# Provenance: diffusion-posterior-sampling/make_per_category_bars_vs_gatys.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Per-category STGN-zpad vs Gatys comparison across perceptual metrics.

Mirror of make_per_category_bars_vs_ps.py but with Gatys as the baseline.
Gatys CSV doesn't include SIFID/DreamSim, so those panels are dropped.

Output: fig_per_category_bars_vs_gatys.png + fig_per_category_table_vs_gatys.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, FIGURES, METRICS, TABLES# noqa: E402
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPS = ASSETS
OUT = METRICS

CSVS = {
    "Gatys":          OUT / "bench_gatys/per_image_metrics_gatys_clean.csv",
    "StatTexGramNet": OUT / "bench_stgn_zpad_141/per_image_metrics_stgn_zpad_141_clean.csv",
}

# (col, panel-label, lower-is-better?)
METRICS = [
    ("dists",    "DISTS",    True),
    ("lpips",    "LPIPS",    True),
    ("fsim",     "FSIM",     False),
    ("ssim",     "SSIM",     False),
    ("dreamsim", "DreamSim", True),
    ("vif",      "VIF",      False),
    ("gmsd",     "GMSD",     True),
    ("haarpsi",  "HaarPSI",  False),
    ("psd_l2",   "PSD-L2",   True),
]


def load(method, metrics):
    df = pd.read_csv(CSVS[method])
    keep = ["key"] + [m for m, *_ in metrics if m in df.columns]
    return df[keep].set_index("key").sort_index()


def main():
    gatys = load("Gatys", METRICS)
    stgn = load("StatTexGramNet", METRICS)

    df = stgn.join(gatys, how="inner", lsuffix="_stgn", rsuffix="_gatys")
    df["category"] = df.index.str.split("_").str[0]

    diff_frames = {}
    for m, _, _ in METRICS:
        s = f"{m}_stgn"
        g = f"{m}_gatys"
        if s not in df.columns or g not in df.columns:
            continue
        if df[s].isna().all() or df[g].isna().all():
            print(f"  skipping {m}: all-NaN on one side")
            continue
        diff_frames[m] = df.groupby("category").apply(
            lambda gdf: gdf[s].mean() - gdf[g].mean(), include_groups=False)

    n_per_cat = df.groupby("category").size().rename("n")
    diff_df = pd.DataFrame(diff_frames)
    diff_df = diff_df.join(n_per_cat)

    # Drop categories with n<3 (Gatys has incomplete coverage on some cats)
    n_dropped = (diff_df["n"] < 3).sum()
    if n_dropped:
        print(f"  dropping {n_dropped} categories with n<3")
    diff_df = diff_df[diff_df["n"] >= 3]

    diff_df = diff_df.sort_values("dists", ascending=True)
    diff_df.to_csv(TABLES / "fig_per_category_table_vs_gatys.csv")
    print(f"saved {DPS / 'fig_per_category_table_vs_gatys.csv'} ({len(diff_df)} cats)")

    n_cats = len(diff_df)
    n_metrics = sum(1 for m, *_ in METRICS if m in diff_df.columns)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(3.4 * n_metrics, max(7.0, 0.22 * n_cats)),
        sharey=True, gridspec_kw=dict(wspace=0.06))
    if n_metrics == 1:
        axes = [axes]

    y = np.arange(n_cats)
    ylabels = list(diff_df.index)

    for ax, (m, label, lower_is_better) in zip(
            axes, [m for m in METRICS if m[0] in diff_df.columns]):
        diffs = diff_df[m].to_numpy()
        stgn_wins = diffs < 0 if lower_is_better else diffs > 0
        colors = ["#2ca02c" if w else "#d62728" for w in stgn_wins]

        ax.barh(y, diffs, color=colors, edgecolor="black", lw=0.4, alpha=0.9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="x", ls=":", alpha=0.5)

        xmax = max(np.abs(diffs).max() * 1.05, 1e-9)
        ax.set_xlim(-xmax, xmax)

        n_wins = int(stgn_wins.sum())
        ax.text(0.99, 0.01, f"GSTN {n_wins}/{n_cats}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    plt.tight_layout()
    out = FIGURES / "fig_per_category_bars_vs_gatys.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[per-cat-vs-gatys] saved {out}")

    print("\n=== STGN-zpad beats Gatys per metric ===")
    for m, label, lower_is_better in METRICS:
        if m in diff_df.columns:
            n_w = int(((diff_df[m] < 0) if lower_is_better else (diff_df[m] > 0)).sum())
            print(f"  {label:10s}: {n_w}/{n_cats}")


if __name__ == "__main__":
    main()
