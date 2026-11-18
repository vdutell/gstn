# ---------------------------------------------------------------------------
# Produces: Figure 4
#
# Provenance: diffusion-posterior-sampling/analyze_texture_fingerprints.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Per-category texture-family fingerprint analysis.

Loads the precomputed (5640, 28929) stat tensor + manifest, groups stats into
the families defined by StatTexGramNet.assign_gram_groups, builds a per-image
family profile, then produces:

    fig_fingerprints.png   - clustered category x family heatmap
                             + within/between distance KDE inset
    fig_per_image.png      - per-image profile spaghetti plot for a few
                             chosen categories (within-category consistency)
    family_profiles.npy    - (5640, num_families) per-image family profile
    category_means.csv     - one row per category, columns = families
    fingerprints_summary.txt - top-3 families per category

CPU-only. Self-contained except for needing the model class to read
`stat_groups_gram_label`.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, CHECKPOINTS# noqa: E402
import csv
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

# --- Project paths ---
DPS_ROOT = ASSETS
STATS_DIR = DPS_ROOT / "dtd_stats"
OUT_DIR = STATS_DIR / "fingerprints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADJACENT = (DPS_ROOT / "../StatTexNetExtended").resolve()
if str(ADJACENT) not in sys.path:
    sys.path.insert(0, str(ADJACENT))


def load_family_labels():
    """
    Build the model on CPU just to get the 28929-element family-label list
    (no forward pass, no weights — purely structural).
    """
    import torch
    from utils import stattexgramnet as stn
    model = stn.StatTexGramNet(
        img_size=(256, 256),
        batch_size=1,
        device=torch.device("cpu"),
        compress=False,
        color=True,
        add_marginals=True,
        oris=4,
        edge_levels=6,
        edge_start=1,
        include_dphase=False,
        boundary_mode="wrap",
    )
    return list(model.stat_groups_gram_label)


def main():
    print("[analysis] loading stats + manifest")
    stats = np.load(STATS_DIR / "stats.npy")           # (N, 28929)
    with open(STATS_DIR / "manifest.csv") as f:
        manifest = list(csv.DictReader(f))
    categories = np.array([r["category"] for r in manifest])
    image_names = np.array([r["image_name"] for r in manifest])

    N, D = stats.shape
    print(f"[analysis] stats shape={stats.shape} categories={len(set(categories))}")

    # ---- Family assignment per stat ----
    print("[analysis] building family labels (instantiating model on CPU)")
    fam_labels = load_family_labels()
    assert len(fam_labels) == D, f"family labels {len(fam_labels)} != stat dim {D}"

    families_in_order = list(OrderedDict.fromkeys(fam_labels))   # preserve first-seen order
    fam_to_idx = {f: i for i, f in enumerate(families_in_order)}
    fam_idx = np.array([fam_to_idx[f] for f in fam_labels])
    F = len(families_in_order)
    print(f"[analysis] {F} families: {families_in_order}")

    fam_sizes = np.array([(fam_idx == i).sum() for i in range(F)])
    for f, s in zip(families_in_order, fam_sizes):
        print(f"    {f:24s}  {s:6d} stats")

    # ---- Per-stat z-score across all images, then |z| → family-binned ----
    print("[analysis] z-scoring stats (per-stat across images)")
    mu = stats.mean(axis=0, keepdims=True)
    sd = stats.std(axis=0, keepdims=True) + 1e-8
    z = (stats - mu) / sd
    abs_z = np.abs(z)                                            # (N, D)

    # We need cat_idx for AUC scoring; build it now.
    cats = sorted(set(categories))
    C = len(cats)
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    cat_idx = np.array([cat_to_idx[c] for c in categories])

    # ---- Try multiple profile variants; pick the one with best AUC ----
    print("[analysis] computing candidate profiles + AUCs")

    def compute_profile(per_stat_score):
        """per_stat_score: (N, D); reduce to (N, F) by mean within family, normalize per image."""
        prof = np.zeros((N, F), dtype=np.float32)
        for i in range(F):
            prof[:, i] = per_stat_score[:, fam_idx == i].mean(axis=1)
        prof = prof / (prof.sum(axis=1, keepdims=True) + 1e-12)
        return prof

    def auc_within_lt_between(prof):
        pn = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - pn @ pn.T
        np.fill_diagonal(d, np.nan)
        same = cat_idx[:, None] == cat_idx[None, :]
        w = d[same & np.isfinite(d)]
        b = d[(~same) & np.isfinite(d)]
        rng = np.random.default_rng(0)
        sw = rng.choice(w, size=min(20000, len(w)), replace=False)
        sb = rng.choice(b, size=min(20000, len(b)), replace=False)
        return (sw[:, None] < sb[None, :]).mean(), w, b

    candidates = {}

    # A: mean |z| per family
    candidates["mean_abs_z"] = compute_profile(abs_z)

    # B: fraction of |z|>2 per family ("extremeness count")
    candidates["frac_extreme"] = compute_profile((abs_z > 2.0).astype(np.float32))

    # C: importance-weighted mean |z| per family (using trained compressor weights)
    # Checkpoint stores w as (D=28929, K=100); compress with stats @ w.
    try:
        import torch
        ckpt = torch.load(CHECKPOINTS / "bottleneck100.pth", map_location="cpu")
        compressor_w = ckpt["state_dict"]["w"].data.cpu().numpy()  # (D, K)
        assert compressor_w.shape == (D, 100), f"unexpected w shape {compressor_w.shape}"
        importance = np.sum(np.abs(compressor_w), axis=1)          # (D,)
        importance = importance / (importance.max() + 1e-12)
        weighted_abs_z = abs_z * importance[None, :]
        candidates["imp_weighted_z"] = compute_profile(weighted_abs_z)
    except Exception as e:
        print(f"[analysis]   (skipping compressor-weighted: {e})")
        compressor_w = None

    # D: per-stat (raw - global_mean) magnitude (no z-scoring) — let natural scale dominate
    raw_abs_dev = np.abs(stats - mu)
    candidates["mean_abs_dev"] = compute_profile(raw_abs_dev)

    cat_idx_local = cat_idx  # alias for local use in loops below
    aucs = {}
    for name, prof in candidates.items():
        auc, _, _ = auc_within_lt_between(prof)
        aucs[name] = auc
        print(f"    {name:20s}  AUC(within<between) = {auc:.3f}")

    # ---- Bonus: compressor 100-d direct similarity (skip family aggregation) ----
    if compressor_w is not None:
        z100 = stats @ compressor_w  # (N, K)
        z100n = z100 / (np.linalg.norm(z100, axis=1, keepdims=True) + 1e-12)
        d100 = 1.0 - z100n @ z100n.T
        np.fill_diagonal(d100, np.nan)
        same = cat_idx[:, None] == cat_idx[None, :]
        w100 = d100[same & np.isfinite(d100)]
        b100 = d100[(~same) & np.isfinite(d100)]
        rng = np.random.default_rng(0)
        sw = rng.choice(w100, size=min(20000, len(w100)), replace=False)
        sb = rng.choice(b100, size=min(20000, len(b100)), replace=False)
        auc_100d = (sw[:, None] < sb[None, :]).mean()
        print(f"    {'compressor_100d':20s}  AUC(within<between) = {auc_100d:.3f}  "
              f"(no family agg — sanity check that category structure exists at all)")

    best_name = max(aucs, key=aucs.get)
    print(f"[analysis] BEST family profile: {best_name}  (AUC={aucs[best_name]:.3f})")
    profile = candidates[best_name]
    np.save(OUT_DIR / "family_profiles.npy", profile)

    # ---- Per-category mean profile (cats/cat_idx already built above) ----
    # The category-mean signature: |mean z per stat for category c|, averaged
    # within family. This directly measures how much each category's family
    # signal deviates from the population — strongest interpretable readout.
    print("[analysis] computing category-mean deviation signatures")
    centroid_z = np.zeros((C, D), dtype=np.float32)
    for i in range(C):
        centroid_z[i] = z[cat_idx == i].mean(axis=0)        # per-category z-centroid
    abs_centroid_z = np.abs(centroid_z)                     # |deviation from pop mean|
    cat_mean = np.zeros((C, F), dtype=np.float32)
    for i in range(F):
        cat_mean[:, i] = abs_centroid_z[:, fam_idx == i].mean(axis=1)
    # Optionally row-normalize for the heatmap; keep both views.
    cat_mean_raw = cat_mean.copy()
    cat_mean = cat_mean / (cat_mean.sum(axis=1, keepdims=True) + 1e-12)

    # Save category-mean CSV
    with open(OUT_DIR / "category_means.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category"] + families_in_order)
        for i, c in enumerate(cats):
            w.writerow([c] + [f"{x:.4f}" for x in cat_mean[i]])

    # ---- Top-3 families per category (text summary) ----
    summary_lines = []
    for i, c in enumerate(cats):
        order = np.argsort(cat_mean[i])[::-1]
        tops = ", ".join(f"{families_in_order[j]} ({cat_mean[i, j]:.2f})" for j in order[:3])
        summary_lines.append(f"{c:18s}  top: {tops}")
    (OUT_DIR / "fingerprints_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n[analysis] top-3 families per category (sample):")
    for line in summary_lines[:8]:
        print(" ", line)
    print("  ...")

    # ---- Within vs between distance distribution ----
    print("[analysis] computing pairwise distances (cosine)")
    profile_norm = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-12)
    cos_dist = 1.0 - profile_norm @ profile_norm.T
    np.fill_diagonal(cos_dist, np.nan)

    same = cat_idx[:, None] == cat_idx[None, :]
    within_d = cos_dist[same & np.isfinite(cos_dist)]
    between_d = cos_dist[(~same) & np.isfinite(cos_dist)]

    # AUC of within < between
    rng = np.random.default_rng(0)
    sub_w = rng.choice(within_d, size=min(20000, len(within_d)), replace=False)
    sub_b = rng.choice(between_d, size=min(20000, len(between_d)), replace=False)
    auc = (sub_w[:, None] < sub_b[None, :]).mean()
    print(f"[analysis] within median={np.median(within_d):.4f}  "
          f"between median={np.median(between_d):.4f}  "
          f"AUC(within<between)={auc:.3f}")

    # ---- Hierarchical clustering on category-mean profiles ----
    print("[analysis] clustering categories")
    cat_dist = pdist(cat_mean, metric="cosine")
    cat_linkage = hierarchy.linkage(cat_dist, method="average")
    leaves = hierarchy.leaves_list(cat_linkage)

    # Optional: cluster columns too (correlation between family profiles across categories)
    fam_dist = pdist(cat_mean.T, metric="correlation")
    fam_linkage = hierarchy.linkage(fam_dist, method="average")
    fam_leaves = hierarchy.leaves_list(fam_linkage)

    # ---- The Killer Plot ----
    print("[analysis] rendering fig_fingerprints.png")
    fig = plt.figure(figsize=(15, 13))
    gs = GridSpec(2, 2, width_ratios=[1.0, 0.18], height_ratios=[0.18, 1.0],
                  wspace=0.02, hspace=0.02)

    # Top dendrogram (column clustering)
    ax_top = fig.add_subplot(gs[0, 0])
    hierarchy.dendrogram(fam_linkage, no_labels=True, color_threshold=-1,
                         above_threshold_color="#777777", ax=ax_top)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    # Left dendrogram (row clustering)
    ax_left = fig.add_subplot(gs[1, 1])
    hierarchy.dendrogram(cat_linkage, orientation="right", no_labels=True,
                         color_threshold=-1, above_threshold_color="#777777",
                         ax=ax_left)
    ax_left.invert_yaxis()
    ax_left.set_xticks([]); ax_left.set_yticks([])
    for s in ax_left.spines.values():
        s.set_visible(False)

    # Heatmap
    ax_hm = fig.add_subplot(gs[1, 0])
    cm = cat_mean[leaves][:, fam_leaves]
    im = ax_hm.imshow(cm, aspect="auto", cmap="magma_r")
    ax_hm.set_xticks(np.arange(F))
    ax_hm.set_xticklabels([families_in_order[i] for i in fam_leaves],
                          rotation=45, ha="right", fontsize=9)
    ax_hm.set_yticks(np.arange(C))
    ax_hm.set_yticklabels([cats[i] for i in leaves], fontsize=7)
    ax_hm.set_xlabel("statistical family")
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.01)
    cb.set_label("normalized |z| share")

    fig.suptitle(
        f"Texture fingerprints: per-category contribution of each statistical family\n"
        f"(N={N} images, {C} DTD categories, {F} families;  "
        f"within<between AUC = {auc:.3f})",
        y=0.995, fontsize=12,
    )
    plt.savefig(OUT_DIR / "fig_fingerprints.png", dpi=180, bbox_inches="tight")
    plt.close()

    # ---- Within vs between KDE plot (separate) ----
    print("[analysis] rendering fig_within_between.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(within_d.max(), between_d.max()), 80)
    ax.hist(between_d, bins=bins, density=True, alpha=0.5,
            label=f"different-category (n={len(between_d):,})", color="#888")
    ax.hist(within_d, bins=bins, density=True, alpha=0.6,
            label=f"same-category (n={len(within_d):,})", color="#c0392b")
    ax.set_xlabel("cosine distance between family profiles")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ax.set_title(f"Within-category vs between-category profile distance "
                 f"(AUC within<between = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_within_between.png", dpi=180)
    plt.close()

    # ---- Per-image profiles for a few categories (within-category spread) ----
    print("[analysis] rendering fig_per_image.png")
    showcase = [c for c in
                ["banded", "marbled", "honeycombed", "polka-dotted",
                 "swirly", "zigzagged"] if c in cat_to_idx]
    if not showcase:
        showcase = cats[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    axes = axes.flatten()
    x = np.arange(F)
    for ax, c in zip(axes, showcase):
        sel = cat_idx == cat_to_idx[c]
        for row in profile[sel]:
            ax.plot(x, row, color="#888", alpha=0.18, linewidth=0.7)
        ax.plot(x, profile[sel].mean(axis=0), color="#c0392b", linewidth=2.0,
                label="category mean")
        ax.set_title(f"{c}  (n={sel.sum()})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(families_in_order, rotation=45, ha="right", fontsize=7)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-image family profiles (gray) vs category mean (red) — "
                 "tight gray bands = strong within-category conservation",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_per_image.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\n[analysis] done. outputs in {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
