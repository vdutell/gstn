# ---------------------------------------------------------------------------
# Produces: Figure 3 (and the full-metric appendix variant)
#
# Provenance: StatTexNetExtended/rebuttal/camera_ready/make_fig_pareto_munca.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Figure 3 (main text) regenerated WITH muNCA on the Pareto frontier.

x: representation size -- number of statistics for the matching methods, number
   of LEARNED PARAMETERS for muNCA (588). Log scale.
y: perceptual metric, paired bootstrap mean + 95% CI (1000 resamples).

Sample-size honesty
-------------------
Every other method is benchmarked on the full 141-stem DTD val1_3sampled set.
muNCA exists only for the 11 Figure-2 textures (~1-2 h/image on one RTX 4090),
so its point is an 11-stem mean. The 11-stem subset is systematically HARDER
than the full 141 for every method (e.g. GSTN-zpad MS-SSIM 0.172 vs 0.217), so
dropping an 11-stem point onto a 141-stem panel silently penalises muNCA.

Convention used here:  FILLED marker = 141 stems,  OPEN marker = 11 stems.
GSTN-zpad is plotted BOTH ways, joined by a thin dotted line, so the reader can
read the subset offset straight off each panel and shift muNCA accordingly.

muNCA is scored at 256x256 (the resolution every other method is evaluated at),
matching Table 2; the qualitative figure shows the 128x128 synths for space.
The collapsed `polka-dotted_0140` run is INCLUDED -- excluding a method's
failures inflates its mean.

Outputs (into camera_ready/figures/):
    fig_pareto_munca.pdf / .png        5 panels, main text, drop-in for Fig. 3
    fig_pareto_munca_full.pdf / .png   all 10 per-image metrics, appendix option
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, FIGURES, METRICS, RESULTS# noqa: E402
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPS = ASSETS
OUT = FIGURES
MUNCA_CSV = RESULTS / "fig_method_quant_munca_perstem.csv"

TEXTWIDTH_IN = 5.5          # NeurIPS \textwidth -- render native, never downscale

# ---------------------------------------------------------------- methods ---
# label -> (representation size, colour, marker)
METHODS = {
    "P&S":            (  8_400, "#1f77b4", "^"),
    "Gatys":          (825_000, "#d62728", "o"),
    "Ustyuzhaninov":  (132_000, "#9467bd", "D"),
    "GSTN-wrap":      ( 28_929, "#2ca02c", "P"),
    "GSTN-zpad":      ( 28_929, "#2ca02c", "*"),
    "muNCA":          (    588, "#ff7f0e", "h"),   # 588 LEARNED PARAMETERS
}

# 141-stem bench sources: label -> (file stem, directory)
BENCH_141 = {
    "P&S":           ("ps_141",                        "bench_ps_141"),
    "Gatys":         ("gatys",                         "bench_gatys"),
    "Ustyuzhaninov": ("ustyuzhaninov",                 "bench_ustyuzhaninov"),
    "GSTN-wrap":     ("pixelbp_prodigy_depleted_full", "depletion_full"),
    "GSTN-zpad":     ("stgn_zpad_141",                 "bench_stgn_zpad_141"),
}

# methods shown as OPEN markers from the 11-stem muNCA-coverage CSV
SUBSET_11 = ["muNCA", "GSTN-zpad"]

DEPLETION = [
    ("top",  "top-|w|",    "#0e5e1f", "-"),
    ("rand", "random",     "#4cae4c", "--"),
    ("bot",  "bottom-|w|", "#a8d8a8", ":"),
]
DEPLETION_N = [100, 1000, 5000, 10000]

# (metric key, direction, panel title)
PANELS_MAIN = [
    ("lpips",   "down", "LPIPS"),
    ("ms_ssim", "up",   "MS-SSIM"),
    ("haarpsi", "up",   "HaarPSI"),
    ("fsim",    "up",   "FSIM"),
    ("ssim",    "up",   "SSIM"),
]
# Alternative main-five: drops the SSIM-family panels, which reward smooth
# low-contrast output and are the only ones on which muNCA scores well (its
# COLLAPSED polka-dotted run earns its best SSIM of all 11 textures). On this
# set muNCA is last of six on every panel, matching the qualitative figure.
PANELS_ALT = [
    ("lpips",    "down", "LPIPS"),
    ("dists",    "down", "DISTS"),
    ("dreamsim", "down", "DreamSim"),
    ("fsim",     "up",   "FSIM"),
    ("haarpsi",  "up",   "HaarPSI"),
]

PANELS_FULL = PANELS_MAIN + [
    ("dists",    "down", "DISTS"),
    ("dreamsim", "down", "DreamSim"),
    ("vif",      "up",   "VIF"),
    ("gmsd",     "down", "GMSD"),
    ("psd_l2",   "down", "PSD-L2"),
]

ARROW = {"up": "$\\uparrow$", "down": "$\\downarrow$"}


# ------------------------------------------------------------------ stats ---
def boot_ci(arr, B=1000, seed=42):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        boots[b] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


_cache = {}


def bench_141(metric):
    """{label: (mean, lo, hi)} over the full 141-stem set."""
    out = {}
    for label, (stem, dir_) in BENCH_141.items():
        key = (stem, dir_)
        if key not in _cache:
            csv = METRICS / dir_ / f"per_image_metrics_{stem}_clean.csv"
            _cache[key] = pd.read_csv(csv) if csv.is_file() else None
        df = _cache[key]
        if df is None or metric not in df.columns:
            continue
        ci = boot_ci(df[metric].to_numpy())
        if ci:
            out[label] = ci
    return out


def bench_11(metric):
    """{label: (mean, lo, hi)} over the 11 stems muNCA covers."""
    if "munca" not in _cache:
        _cache["munca"] = pd.read_csv(MUNCA_CSV)
    df = _cache["munca"]
    if metric not in df.columns:
        return {}
    out = {}
    for label in SUBSET_11:
        sub = df[df["method"] == label]
        if len(sub) == 0:
            continue
        ci = boot_ci(sub[metric].to_numpy())
        if ci:
            out[label] = ci
    return out


def depletion(metric):
    """[(prefix, label, colour, ls, xs, ys, los, his)]"""
    series = []
    for prefix, label, colour, ls in DEPLETION:
        xs, ys, los, his = [], [], [], []
        for N in DEPLETION_N:
            csv = (METRICS / f"depletion_{prefix}{N}"
                   / f"per_image_metrics_pixelbp_prodigy_depleted_{prefix}{N}_clean.csv")
            if not csv.is_file():
                continue
            df = pd.read_csv(csv)
            if metric not in df.columns:
                continue
            ci = boot_ci(df[metric].to_numpy())
            if ci is None:
                continue
            xs.append(N); ys.append(ci[0]); los.append(ci[1]); his.append(ci[2])
        if xs:
            series.append((prefix, label, colour, ls, xs, ys, los, his))
    return series


# ------------------------------------------------------------------ panel ---
def draw_panel(ax, metric, direction, title, fs):
    d141 = bench_141(metric)
    d11 = bench_11(metric)
    dep = depletion(metric)

    lo_vals, hi_vals = [], []

    for _, label, colour, ls, xs, ys, los, his in dep:
        ax.plot(xs, ys, ls, color=colour, lw=1.0, marker="o", markersize=2.6,
                mec="black", mew=0.3, alpha=0.95, label=label, zorder=3)
        ax.fill_between(xs, los, his, color=colour, alpha=0.12, lw=0)
        lo_vals += list(los); hi_vals += list(his)

    # filled = 141 stems
    for label, (size, colour, marker) in METHODS.items():
        if label not in d141:
            continue
        m, lo, hi = d141[label]
        ms = 6.2 if marker == "*" else 4.2
        ax.errorbar(size, m, yerr=[[m - lo], [hi - m]], color=colour, marker=marker,
                    markersize=ms, capsize=1.3, lw=0.7, mec="black", mew=0.4,
                    zorder=5, label=label)
        lo_vals.append(lo); hi_vals.append(hi)

    # open = 11 stems (muNCA's coverage)
    for label in SUBSET_11:
        if label not in d11:
            continue
        size, colour, marker = METHODS[label]
        m, lo, hi = d11[label]
        ms = 6.2 if marker == "*" else 4.2
        ax.errorbar(size, m, yerr=[[m - lo], [hi - m]], color=colour, marker=marker,
                    markersize=ms, capsize=1.3, lw=0.7, mec=colour, mew=0.8,
                    mfc="white", zorder=6,
                    label=f"{label} (11 stems)" if label == "muNCA" else None)
        lo_vals.append(lo); hi_vals.append(hi)
        # tie the open point to its filled twin so the subset offset is readable
        if label in d141:
            ax.plot([size, size], [d141[label][0], m], ls=":", color="0.35",
                    lw=0.7, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(60, 3e6)
    if lo_vals:
        lo, hi = min(lo_vals), max(hi_vals)
        pad = 0.08 * (hi - lo) if hi > lo else 0.01
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{title} {ARROW[direction]}", fontsize=fs + 0.7, pad=2.5)
    ax.tick_params(axis="both", labelsize=fs - 0.7, length=2, pad=1.2)
    ax.grid(True, ls=":", alpha=0.4, lw=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


# ------------------------------------------------------------------ build ---
def build(panels, cols, out_stem, fs, fig_h, top, bottom, hspace=0.50):
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(TEXTWIDTH_IN, fig_h))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, (metric, direction, title) in enumerate(panels):
        r, c = divmod(i, cols)
        draw_panel(axes[r, c], metric, direction, title, fs)
    for j in range(len(panels), rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    # shared x label, once, under the bottom row
    fig.text(0.5, 0.012, "representation size  (statistics; learned parameters for muNCA)",
             ha="center", fontsize=fs - 0.4)

    # ------- legend -------
    handles, labels = [], []
    for prefix, label, colour, ls in DEPLETION:
        handles.append(plt.Line2D([], [], color=colour, ls=ls, lw=0.9,
                                  marker="o", markersize=2.2, mec="black", mew=0.3))
        labels.append(label)
    for label, (size, colour, marker) in METHODS.items():
        ms = 6.0 if marker == "*" else 4.2
        filled = label in BENCH_141
        handles.append(plt.Line2D([], [], color=colour, marker=marker, lw=0,
                                  markersize=ms, mec="black" if filled else colour,
                                  mew=0.4 if filled else 0.8,
                                  mfc=colour if filled else "white"))
        labels.append(label if filled else f"{label}†")
    # the open-marker convention, stated in the legend itself
    handles.append(plt.Line2D([], [], color="0.35", ls=":", lw=0.8, marker="o",
                              markersize=3.2, mfc="white", mec="0.35", mew=0.8))
    labels.append("† open = 11-stem subset")

    fig.legend(handles, labels, loc="upper center", ncol=10,
               bbox_to_anchor=(0.5, 1.005), fontsize=fs - 1.0, frameon=True,
               handletextpad=0.3, columnspacing=0.7, borderpad=0.25,
               handlelength=1.4, borderaxespad=0.0)

    fig.subplots_adjust(left=0.055, right=0.995, top=top, bottom=bottom,
                        wspace=0.36, hspace=hspace)

    OUT.mkdir(parents=True, exist_ok=True)
    # NO bbox_inches="tight": that would crop to the drawn extent and make the
    # saved width != TEXTWIDTH_IN, so \includegraphics[width=\textwidth] would
    # rescale and shrink every label away from its stated point size.
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{out_stem}.{ext}", dpi=400)
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"[pareto] {out_stem}: {w:.2f} x {h:.2f} in  ({rows}x{cols})")


def report():
    """Print the numbers that go in the caption."""
    print("\n--- subset offset: GSTN-zpad, 141 stems vs the 11 muNCA stems ---")
    print(f"{'metric':>10} {'141':>8} {'11':>8} {'shift':>8}")
    for metric, direction, title in PANELS_MAIN:
        a = bench_141(metric).get("GSTN-zpad")
        b = bench_11(metric).get("GSTN-zpad")
        if a and b:
            print(f"{title:>10} {a[0]:8.3f} {b[0]:8.3f} {b[0]-a[0]:+8.3f}")

    print("\n--- muNCA (11 stems, 256px), all 11 vs polka-dotted excluded ---")
    df = pd.read_csv(MUNCA_CSV)
    mu = df[df["method"] == "muNCA"]
    mu_ok = mu[mu["stem"] != "polka-dotted_0140"]
    print(f"{'metric':>10} {'all 11':>8} {'n=10':>8}")
    for metric, direction, title in PANELS_MAIN:
        print(f"{title:>10} {mu[metric].mean():8.3f} {mu_ok[metric].mean():8.3f}")


SUBSET_11_STEMS = [
    "chequered_0177", "striped_0050", "polka-dotted_0140", "honeycombed_0149",
    "cracked_0056", "lacelike_0004", "fibrous_0112", "bumpy_0077",
    "sprinkled_0042", "pleated_0136", "swirly_0065",
]


def paired_vs_depletion(N=1000):
    """muNCA vs GSTN depleted to top-N statistics, PAIRED on the same 11 stems.

    This is the comparison the captions quote: both sides restricted to the
    stems muNCA covers, so the subset-difficulty offset cancels.
    """
    mu = pd.read_csv(MUNCA_CSV)
    mu = mu[mu["method"] == "muNCA"]
    csv = (METRICS / f"depletion_top{N}"
           / f"per_image_metrics_pixelbp_prodigy_depleted_top{N}_clean.csv")
    dep = pd.read_csv(csv).set_index("key")
    keep = [s for s in SUBSET_11_STEMS if s in dep.index]
    assert len(keep) == 11, f"only {len(keep)}/11 stems in depletion_top{N}"
    dep = dep.loc[keep]

    print(f"\n--- PAIRED on the 11 muNCA stems: muNCA (588 params) vs "
          f"GSTN top-{N} stats ---")
    print(f"{'metric':>10} {'muNCA':>8} {f'top-{N}':>8}  winner")
    wins = 0
    for metric, direction, title in PANELS_FULL:
        a, b = mu[metric].mean(), dep[metric].mean()
        better = a < b if direction == "down" else a > b
        wins += better
        print(f"{title:>10} {a:8.4f} {b:8.4f}  {'muNCA' if better else 'GSTN'}")
    print(f"muNCA wins {wins}/{len(PANELS_FULL)}")


if __name__ == "__main__":
    build(PANELS_MAIN, cols=5, out_stem="fig_pareto_munca",
          fs=5.4, fig_h=1.46, top=0.775, bottom=0.205)
    build(PANELS_ALT, cols=5, out_stem="fig_pareto_munca_alt",
          fs=5.4, fig_h=1.46, top=0.775, bottom=0.205)
    build(PANELS_FULL, cols=5, out_stem="fig_pareto_munca_full",
          fs=5.4, fig_h=2.95, top=0.885, bottom=0.098, hspace=0.55)
    report()
    paired_vs_depletion(1000)
