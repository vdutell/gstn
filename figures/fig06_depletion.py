# ---------------------------------------------------------------------------
# Produces: Figure 6 (stat-depletion grid)
#
# Provenance: diffusion-posterior-sampling/make_depletion_full_figure.py from the
# original working tree. Copied verbatim except that hardcoded absolute paths now
# resolve through gstn_paths, and the output goes to the figures build directory
# instead of back into the working tree. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""
Full depletion comparison figure: top / rand / bot depletions across N levels
for 3 textures.

Layout (rotated 90° from prior version):
  rows   = 5: target, N=10000, N=5000, N=1000, N=100      (left-side labels)
  cols   = 9: 3 categories × 3 conditions (top, rand, bot)
  group titles at top: category names spanning their 3 cols
  inner col titles:    cond (top/rand/bot)

Output: fig_depletion_grid.png
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, FIGURES  # noqa: E402
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DPS = ASSETS
TARGETS = DPS / "pixelbp_prodigy_targets"
SYNTHS = DPS / "synths"
OUT = FIGURES / "fig_depletion_grid.png"

CATEGORIES = ["banded", "cracked", "polka-dotted", "bumpy"]
CONDITIONS = ["top", "rand", "bot"]
N_LEVELS = [10000, 5000, 1000, 100]   # high → low

# Per-category stem overrides (for visual clarity).
# Default is sorted-first; override here if a specific exemplar is preferred.
STEM_OVERRIDE = {
    "cracked": "cracked_0079",   # 0056 looked like a failure; 0079 cleaner exemplar
}

ROW_LABELS = ["target"] + [f"N={n}" for n in N_LEVELS]   # 5 rows
N_ROWS = len(ROW_LABELS)
N_COLS = len(CATEGORIES) * len(CONDITIONS)               # 9


def find_synth(cat, cond, N):
    d = SYNTHS / f"pixelbp_prodigy_depleted_{cond}{N}"
    override = STEM_OVERRIDE.get(cat)
    if override:
        cand = sorted(d.glob(f"{override}_gpu*_synth.png"))
        if cand:
            return cand[0]
    matches = sorted(d.glob(f"{cat}_*_synth.png"))
    if not matches:
        raise FileNotFoundError(f"no {cat} in {d}")
    return matches[0]


def main():
    cell = 1.4
    fig_w = cell * N_COLS + 0.7    # narrower left margin since labels rotate
    fig_h = cell * N_ROWS + 0.9    # +0.9 for top titles + supertitles

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(fig_w, fig_h),
                             squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    # Resolve a stem per category from the top-10000 dir (they should match).
    cat_stem = {}
    for cat in CATEGORIES:
        sample = find_synth(cat, "top", 10000)
        cat_stem[cat] = sample.name.split("_gpu")[0]

    for ci, cat in enumerate(CATEGORIES):
        target_img = Image.open(TARGETS / f"{cat_stem[cat]}.png")
        for cj, cond in enumerate(CONDITIONS):
            col = ci * len(CONDITIONS) + cj

            # Row 0: target (same image for each cond inside a category, but
            # repeated so visual scanning aligns with the 3 cols of that category)
            axes[0, col].imshow(target_img)
            axes[0, col].set_aspect("equal")

            # Rows 1..: each N level for this cond
            for ri, N in enumerate(N_LEVELS):
                ax = axes[1 + ri, col]
                ax.imshow(Image.open(find_synth(cat, cond, N)))
                ax.set_aspect("equal")

            # cond title at the top of this column
            axes[0, col].text(0.5, 1.04, cond, transform=axes[0, col].transAxes,
                              ha="center", va="bottom", fontsize=11,
                              fontweight="bold")

        # Category supertitle spanning the 3 cols of this category
        cs, ce = ci * len(CONDITIONS), ci * len(CONDITIONS) + len(CONDITIONS) - 1

    # Row labels on the LEFT, rotated 90° to save horizontal space
    for r, label in enumerate(ROW_LABELS):
        axes[r, 0].text(-0.10, 0.5, label,
                        transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.04, right=0.995, top=0.92, bottom=0.01,
                        wspace=0.04, hspace=0.04)

    # Add category supertitles (spanning 3 cols each) above the cond titles
    fig.canvas.draw()
    for ci, cat in enumerate(CATEGORIES):
        cs = ci * len(CONDITIONS)
        ce = cs + len(CONDITIONS) - 1
        x0 = axes[0, cs].get_position().x0
        x1 = axes[0, ce].get_position().x1
        x_mid = (x0 + x1) / 2
        y_top = axes[0, cs].get_position().y1 + 0.06
        fig.text(x_mid, y_top, cat, ha="center", va="bottom",
                 fontsize=14, fontweight="bold")

    # Thin vertical separators between category groups
    for ci in range(1, len(CATEGORIES)):
        cs = ci * len(CONDITIONS)
        x_left = axes[0, cs - 1].get_position().x1
        x_right = axes[0, cs].get_position().x0
        x = (x_left + x_right) / 2
        y0 = axes[-1, 0].get_position().y0
        y1 = axes[0, 0].get_position().y1
        fig.add_artist(plt.Line2D([x, x], [y0, y1],
                                   transform=fig.transFigure,
                                   color="0.85", lw=0.8))

    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[depletion-grid] saved {OUT}")


if __name__ == "__main__":
    main()
