# ---------------------------------------------------------------------------
# Produces: Figure 16
#
# Provenance: StatTexNetExtended/make_family_weight_mass.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Per-family weight-mass for the trained contrastive compressors (100-d / 500-d /
1000-d). For each compressor:
   - load _w (shape [28929, N])
   - sum |_w| across the N output dims → per-stat importance (28929,)
   - bucket by family (model.stat_groups_gram_label) → per-family mass
   - normalize as fraction of total |_w|

Outputs a grouped bar chart: each family has up to 3 bars (one per dim),
showing the fraction of weight mass the contrastive task assigned.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import CHECKPOINTS, FIGURES, REPO# noqa: E402
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# (the original computed this from __file__; after moving into figures/ that
# would resolve to figures/, so paths now come from gstn_paths)
sys.path.insert(0, str(REPO))
import utils.stattexgramnet as stn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CKPT_PATHS = {
    "100":  CHECKPOINTS / "bottleneck100.pth",
    "500":  CHECKPOINTS / "bottleneck500.pth",
    "1000": CHECKPOINTS / "bottleneck1000.pth",
}

OUT_PNG = FIGURES / "fig_family_weight_mass.png"


def build_model(bn):
    return stn.StatTexGramNet(
        img_size=(128, 128),
        batch_size=1,
        bottleneck_size=int(bn),
        device=DEVICE,
        compress=True,
        color=True,
        add_marginals=True,
        oris=4,
        edge_levels=6,
        edge_start=1,
        include_dphase=False,
        boundary_mode="wrap",
    ).to(DEVICE)


def main():
    # Family labels are identical across compressors (same stats vector), so
    # we only need them once.
    model_ref = build_model(100)
    fam_labels = list(model_ref.stat_groups_gram_label)
    n_stats = len(fam_labels)
    print(f"n_stats = {n_stats}")

    # Compute per-stat importance for each compressor
    fam_mass = {}     # bn_name -> {family -> normalized mass}
    for name, ckpt_path in CKPT_PATHS.items():
        if not ckpt_path.is_file():
            print(f"[skip] {name}-d: checkpoint missing at {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        w = ckpt["state_dict"]["_w"]
        assert w.shape[0] == n_stats, f"shape mismatch: {w.shape[0]} vs {n_stats}"
        per_stat = w.abs().sum(dim=1).cpu().numpy()   # (n_stats,)
        per_stat = per_stat / per_stat.sum()
        bucket = {}
        for f, s in zip(fam_labels, per_stat):
            bucket[f] = bucket.get(f, 0.0) + float(s)
        fam_mass[name] = bucket
        print(f"  {name}-d top-3 families:",
              sorted(bucket.items(), key=lambda kv: -kv[1])[:3])

    # Fraction-of-stats baseline (uniform).
    fam_count = {}
    for f in fam_labels:
        fam_count[f] = fam_count.get(f, 0) + 1
    fam_uniform = {f: c / n_stats for f, c in fam_count.items()}

    # Concentration ratio = (compressor mass / fraction of stats). >1 means
    # the compressor weights this family MORE than its share of stats;
    # <1 means LESS. Drops out the family-size dominance.
    ratios = {}
    for name, bucket in fam_mass.items():
        ratios[name] = {f: (bucket[f] / fam_uniform[f]) if fam_uniform[f] > 0 else 0.0
                         for f in fam_uniform}

    # Order families by 100-d ratio (descending). Drop zero-count families.
    nonzero = [f for f in fam_uniform if fam_uniform[f] > 0]
    if "100" in ratios:
        ordering = sorted(nonzero, key=lambda f: -ratios["100"][f])
    else:
        ordering = nonzero

    fig, (ax_ratio, ax_share) = plt.subplots(
        2, 1, figsize=(11.5, 8.0),
        gridspec_kw={"height_ratios": [1.6, 1.0]})

    x = np.arange(len(ordering))
    bar_colors = {"100": "#1b9e77", "500": "#7570b3", "1000": "#d95f02"}

    # --- top panel: concentration ratio (log y, anchored at 1)
    ratio_bars = [(n, [ratios[n][f] for f in ordering])
                  for n in ["100", "500", "1000"] if n in ratios]
    n_bars = len(ratio_bars)
    width = 0.78 / max(n_bars, 1)
    for i, (name, vals) in enumerate(ratio_bars):
        offset = (i - (n_bars - 1) / 2) * width
        ax_ratio.bar(x + offset, vals, width=width, color=bar_colors[name],
                     edgecolor="black", linewidth=0.4,
                     label=f"{name}-d compressor")

    ax_ratio.axhline(1.0, color="black", lw=1.0, ls="--",
                     label="uniform (= proportional to family size)")
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylim(0.05, 6.0)
    ax_ratio.set_ylabel("concentration ratio  (mass / family-size share)",
                        fontsize=11)
    ax_ratio.set_title(
        "Compressor concentration ratio per family\n"
        "Bars >1 = task weights this family MORE than its share of stats; "
        "<1 = LESS",
        fontsize=11)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(ordering, rotation=45, ha="right", fontsize=9)
    ax_ratio.legend(loc="upper right", fontsize=9, frameon=True)
    ax_ratio.grid(True, ls=":", alpha=0.4, axis="y")
    ax_ratio.set_axisbelow(True)

    # --- bottom panel: family-size shares (context)
    ax_share.bar(x, [fam_uniform[f] for f in ordering], color="#999999",
                 edgecolor="black", linewidth=0.4)
    ax_share.set_xticks(x)
    ax_share.set_xticklabels(ordering, rotation=45, ha="right", fontsize=9)
    ax_share.set_ylabel("fraction of stats\n(family size / 28929)", fontsize=10)
    ax_share.grid(True, ls=":", alpha=0.4, axis="y")
    ax_share.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {OUT_PNG}")
    print("\nRatios for 100-d compressor (sorted):")
    for f in ordering:
        if "100" in ratios:
            print(f"  {f:25s}  ratio = {ratios['100'][f]:5.2f}   "
                  f"size = {fam_uniform[f]:6.3%}")


if __name__ == "__main__":
    main()
