# ---------------------------------------------------------------------------
# Produces: family ablation registry (imported by fig05)
#
# Provenance: diffusion-posterior-sampling/make_family_ablation_grids.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Qualitative grids for family ablation results.

For each of 14 family-group conditions (drop and keep modes), display
syntheses on 3 stems (chequered_0177, bumpy_0077, pleated_0136).

Two layouts:
  fig_family_ablation_drop.png   16 rows (target + full + 14 conditions) x 3 stems
  fig_family_ablation_keep.png   same layout for KEEP
  fig_family_ablation_combined.png  same rows, 6 cols (3 drop + 3 keep)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, FIGURES  # noqa: E402
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DPS = ASSETS
SYNTHS = DPS / "synths"
TARGETS = DPS / "pixelbp_prodigy_targets"

STEMS = ["chequered_0177", "bumpy_0077", "pleated_0136"]

# (display_label, tag_used_in_RUN_TAG, n_stats)
# Display labels are plain English, matching the Table 1 naming promised to 4HA9; the
# second field is the RUN_TAG the synth dirs were written with and must NOT change.
# dead_eq is the three same-filter cross-phase families (72*3 = 216), so it is named for
# what it is rather than left as an opaque code.
CONDITIONS = [
    ("Marginal",                      "marginal",            9),
    ("Highpass",                      "highpass",            6),
    ("Lowpass",                       "lowpass",           231),
    ("Non-oriented $\\times$ band",     "pass_multi",       5247),
    ("Cross-orientation",             "sub_Xori",          324),
    ("Cross-color + band energy",     "sub_Xcolor",        432),
    ("Cross-scale",                   "sub_Xlevel",        540),
    ("Cross orient.+color+scale",     "Xori_Xlevel_Xcolor", 1296),
    ("Single-phase, multi-attribute", "sub_X_struct",     6588),
    ("Mixed-phase, multi-attribute",  "sub_multi",       15336),
    ("Non-orient.$\\times$band + mixed", "PassMulti_subXmulti", 20583),
    ("All structured families",       "Structured",       1758),
    ("Cross-phase (same filter)",     "dead_eq",           216),
    ("Cross-phase + highpass",        "dead_eq_hp",        222),
]

FULL_GRAM_DIR = SYNTHS / "pixelbp_prodigy_depleted_full_meanonly_wrap"
FULL_GRAM_RX = re.compile(r"^(.+)_gpu[01]_synth\.png$")
ABLATION_RX = re.compile(r"^(.+)_gpu[01]_synth\.png$")


def find_in(dir_, regex, stem):
    if not dir_.is_dir():
        return None
    for p in dir_.iterdir():
        m = regex.match(p.name)
        if m and m.group(1) == stem:
            return p
    return None


def find_drop_dir(tag):
    """Returns the synth dir for DROP_FAMILIES=<list>, RUN_TAG=_drop_<tag>."""
    candidates = sorted(SYNTHS.glob(f"pixelbp_prodigy_depleted_drop_*_drop_{tag}"))
    return candidates[0] if candidates else None


def find_keep_dir(tag):
    candidates = sorted(SYNTHS.glob(f"pixelbp_prodigy_depleted_keep_*_keep_{tag}"))
    return candidates[0] if candidates else None


def load_synth(dir_, stem):
    if dir_ is None:
        return None
    return find_in(dir_, ABLATION_RX, stem)


def build_grid(rows, n_cols, cols_for_row, out_path, col_titles=None,
               group_separator_after=None, cell=1.0, label_w=1.7,
               title_h=0.30, label_fs=10, title_fs=10, group_titles=None):
    """
    group_titles: optional list of (label, start_col, end_col) for super-titles
                  spanning multiple columns above the per-column titles.
    """
    n_rows = len(rows)
    extra_top = 0.35 if group_titles else 0.0
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell * n_cols + label_w,
                                      cell * n_rows + title_h + extra_top),
                             squeeze=False,
                             gridspec_kw=dict(wspace=0.05, hspace=0.0))

    for r, (row_label, picker) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            p = picker(c)
            if p is None or not Path(p).is_file():
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="red", fontsize=8, transform=ax.transAxes)
            else:
                ax.imshow(Image.open(p))
            ax.axis("off")
            if r == 0 and col_titles and c < len(col_titles):
                ax.set_title(col_titles[c], fontsize=title_fs, pad=3)
        axes[r, 0].text(-0.05, 0.5, row_label,
                        transform=axes[r, 0].transAxes,
                        rotation=0, va="center", ha="right",
                        fontsize=label_fs, fontweight="bold",
                        linespacing=1.05)

    fig_h = cell * n_rows + title_h + extra_top
    plt.subplots_adjust(left=label_w / (cell * n_cols + label_w),
                        right=0.995,
                        top=1.0 - (title_h + extra_top) / fig_h,
                        bottom=0.0, wspace=0.05, hspace=0.0)

    if group_titles:
        fig.canvas.draw()
        for label, start_col, end_col in group_titles:
            x0 = axes[0, start_col].get_position().x0
            x1 = axes[0, end_col].get_position().x1
            x_mid = (x0 + x1) / 2
            y_top = axes[0, start_col].get_position().y1 + 0.025
            fig.text(x_mid, y_top, label, ha="center", va="bottom",
                     fontsize=title_fs + 4, fontweight="bold")

    if group_separator_after is not None:
        # Compute the actual gap between col_separator and col_separator+1 in
        # FIGURE coordinates, then draw one vertical line spanning all rows.
        fig.canvas.draw()
        x_left = axes[0, group_separator_after].get_position().x1
        x_right = axes[0, group_separator_after + 1].get_position().x0
        x_fig = (x_left + x_right) / 2
        y0 = axes[-1, 0].get_position().y0
        y1 = axes[0, 0].get_position().y1
        fig.add_artist(plt.Line2D([x_fig, x_fig], [y0, y1],
                                  transform=fig.transFigure,
                                  color="black", linewidth=1.0))

    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    _cr = FIGURES
    if _cr.exists() and "combined" in Path(out_path).stem:
        _q = _cr / "fig_family_ablation_combined.pdf"
        plt.savefig(_q, dpi=220, bbox_inches="tight", pad_inches=0.05)
        print(f"also saved {_q}")
    plt.close()
    print(f"saved {out_path}")


# ==== build 3 figures ====
def main():
    # Pre-resolve all dirs
    drop_dirs = {tag: find_drop_dir(tag) for _, tag, _ in CONDITIONS}
    keep_dirs = {tag: find_keep_dir(tag) for _, tag, _ in CONDITIONS}

    missing = [t for t, d in {**drop_dirs, **keep_dirs}.items() if d is None]
    if missing:
        print(f"⚠️  missing dirs for tags: {missing}")

    # ---- DROP-only grid (3 cols = stems) ----
    rows_drop = [
        ("Target",   lambda c: TARGETS / f"{STEMS[c]}.png"),
        ("Full Gram\n(28929)", lambda c: find_in(FULL_GRAM_DIR, FULL_GRAM_RX, STEMS[c])),
    ] + [
        (f"DROP {label}\n(n={n})",
         (lambda c, t=tag: load_synth(drop_dirs[t], STEMS[c])))
        for label, tag, n in CONDITIONS
    ]
    build_grid(rows_drop, n_cols=3, cols_for_row=None,
               out_path=FIGURES / "fig_family_ablation_drop.png",
               col_titles=[s.split("_", 1)[0] for s in STEMS])

    # ---- KEEP-only grid (3 cols = stems) ----
    rows_keep = [
        ("Target",   lambda c: TARGETS / f"{STEMS[c]}.png"),
        ("Full Gram\n(28929)", lambda c: find_in(FULL_GRAM_DIR, FULL_GRAM_RX, STEMS[c])),
    ] + [
        (f"KEEP {label}\n(n={n})",
         (lambda c, t=tag: load_synth(keep_dirs[t], STEMS[c])))
        for label, tag, n in CONDITIONS
    ]
    build_grid(rows_keep, n_cols=3, cols_for_row=None,
               out_path=FIGURES / "fig_family_ablation_keep.png",
               col_titles=[s.split("_", 1)[0] for s in STEMS])

    # ---- Combined grid: drop on left (cols 0-2), keep on right (cols 3-5) ----
    def combined_picker(tag, mode):
        def picker(c):
            stem = STEMS[c % 3]
            d = (drop_dirs if mode == "drop" else keep_dirs)[tag]
            return load_synth(d, stem)
        return picker

    rows_combined = [
        ("Target",
         lambda c: TARGETS / f"{STEMS[c % 3]}.png"),
        ("Full Gram\n(28929)",
         lambda c: find_in(FULL_GRAM_DIR, FULL_GRAM_RX, STEMS[c % 3])),
    ]
    for label, tag, n in CONDITIONS:
        def make_picker(tag_=tag):
            def picker(c):
                stem = STEMS[c % 3]
                d = drop_dirs[tag_] if c < 3 else keep_dirs[tag_]
                return load_synth(d, stem)
            return picker
        rows_combined.append((f"{label}\n(n={n})", make_picker()))

    col_titles = [s.split("_", 1)[0] for s in STEMS] * 2
    build_grid(rows_combined, n_cols=6, cols_for_row=None,
               out_path=FIGURES / "fig_family_ablation_combined.png",
               col_titles=col_titles, group_separator_after=2,
               group_titles=[("DROP", 0, 2), ("KEEP ONLY", 3, 5)])


if __name__ == "__main__":
    main()
