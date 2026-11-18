# ---------------------------------------------------------------------------
# Produces: Figure 5
#
# Provenance: StatTexNetExtended/rebuttal/camera_ready/make_fig_ablation_wide.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Family ablation, TRANSPOSED so it fits the main text without taking a whole page.

The portrait grid is 0.50 aspect -- it only fits by filling a page, and a sidewaysfigure
does not help because that also occupies a full page. Transposing (conditions along the
x-axis, textures down) turns it into a wide, short band that sits as an ordinary [t] float.

Layout: 16 columns (target, full Gram, 14 ablation conditions) x 6 rows
(3 textures under DROP, then the same 3 under KEEP ONLY). Column labels are rotated so
they fit a narrow column. Row labels carry the texture; the two blocks are labelled at left.

Reuses make_family_ablation_grids.py for the directory resolution, so the RUN_TAG lookups
and the plain-English condition names stay in one place.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, FIGURES  # noqa: E402
import importlib.util, sys
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DPS = ASSETS
OUT = FIGURES
spec = importlib.util.spec_from_file_location(
    "fag", _Path(__file__).resolve().parent.parent / "scripts" / "family_ablation_grids.py")
fag = importlib.util.module_from_spec(spec); sys.modules["fag"] = fag; spec.loader.exec_module(fag)

STEMS = fag.STEMS
# Table 1 code names. Short, and the table (now in the main text) carries the plain-English
# definition, so the figure can stay compact without costing the reader anything. Only
# dead_eq is renamed -- it is the three same-filter cross-phase families (72*3 = 216) and
# the response promised that label would go.
CODE = {
    "marginal": "marginal", "highpass": "highpass", "lowpass": "lowpass",
    "pass_multi": "pass_multi", "sub_Xori": "sub_Xori", "sub_Xcolor": "sub_Xcolor",
    "sub_Xlevel": "sub_Xlevel", "Xori_Xlevel_Xcolor": "Xori+Xcolor+Xlevel",
    "sub_X_struct": "sub_X_struct", "sub_multi": "sub_Xmulti",
    "PassMulti_subXmulti": "pass_multi+sub_Xmulti", "Structured": "Structured",
    "dead_eq": "sub_X*_eq", "dead_eq_hp": "sub_X*_eq+highpass",
}
SHORT = {"chequered_0177": "chequered", "bumpy_0077": "bumpy", "pleated_0136": "pleated"}
# Render at the FINAL printed size. The figure was 10.5 in wide and \includegraphics scaled
# it into a 5.5 in \textwidth, halving every label: 5.6 pt became 2.9 pt on the page. Sizing
# the canvas to \textwidth instead means 1:1, so the type is the size it says it is. With 16
# columns that leaves ~0.31 in per image -- small, but that is the budget a 16-column figure
# has on a NeurIPS page however it is drawn.
TEXTWIDTH_IN = 5.5
LABEL_IN = 0.34          # left gutter for the texture row labels
SIZE = 180


def wrap_to_col(label, fs, col_pt, renderer, probe):
    """Break a code name onto lines that each fit one column.

    Horizontal labels need only a few line-heights of headroom, instead of the whole string
    length as a rotated label does -- that is where the vertical saving comes from. Break at
    '+' first (the join between family names) and only fall back to '_' inside a chunk that
    is still too wide, so "pass_multi+sub_Xmulti" becomes two lines rather than four.
    """
    import re as _re

    def w(t):
        return probe.text(0, 0, t, fontsize=fs).get_window_extent(renderer=renderer).width

    def greedy(toks):
        out, cur = [], ""
        for t in toks:
            if cur and w(cur + t) > col_pt:
                out.append(cur); cur = t
            else:
                cur += t
        if cur: out.append(cur)
        return out

    if w(label) <= col_pt:
        return label
    lines = greedy(_re.split(r"(?<=\+)", label))          # '+' boundaries first
    final = []
    for ln in lines:
        if w(ln) <= col_pt:
            final.append(ln); continue
        for sub in greedy(_re.split(r"(?<=_)", ln)):       # then '_' inside that chunk
            while w(sub) > col_pt and len(sub) > 1:        # last resort: hard split
                k = len(sub)
                while k > 1 and w(sub[:k]) > col_pt: k -= 1
                final.append(sub[:k]); sub = sub[k:]
            if sub: final.append(sub)
    return "\n".join(final)


def load(p):
    if p is None or not Path(p).is_file(): return None
    im = cv2.imread(str(p))
    if im is None: return None
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (SIZE, SIZE),
                      interpolation=cv2.INTER_AREA)


def main():
    drop = {t: fag.find_drop_dir(t) for _, t, _ in fag.CONDITIONS}
    keep = {t: fag.find_keep_dir(t) for _, t, _ in fag.CONDITIONS}
    miss = [t for t, d in {**drop, **keep}.items() if d is None]
    if miss: print(f"  !! missing dirs: {miss}")

    cols = [("Target", None, None), ("Full Gram", "FULL", 28929)]
    cols += [(CODE[tag], tag, n) for _, tag, n in fag.CONDITIONS]
    nc, nr = len(cols), 6
    GAP = 0.18
    FS = 5.6
    CELL = (TEXTWIDTH_IN - LABEL_IN) / nc
    # Rotated column titles sit ABOVE the grid, so the top margin has to be the length of
    # the longest label, not a guess. Measure it.
    _pr = plt.figure(figsize=(1, 1)); _r = _pr.canvas.get_renderer()
    # Horizontal headers are impossible here: 16 columns across \textwidth gives ~22 pt
    # columns, narrower than any of these words, so they wrap to 8 lines and collide.
    # Rotated it is -- on two lines (name / n) so the headroom is the longest LINE.
    lines = [x for l, t, n in cols for x in ([l] if n is None else [l, f"$n$={n:,}"])]
    TOPIN = max(_pr.text(0, 0, x, fontsize=FS).get_window_extent(renderer=_r).width
                for x in lines) / 72.0 + 0.06
    plt.close(_pr)
    print(f"  longest column title needs {TOPIN:.2f} in of headroom")

    fig = plt.figure(figsize=(nc*CELL + LABEL_IN, (nr + GAP)*CELL + TOPIN))
    gs = gridspec.GridSpec(nr + 1, nc, figure=fig,
                           height_ratios=[1, 1, 1, GAP, 1, 1, 1],
                           wspace=0.03, hspace=0.03)

    for ci, (lab, tag, n) in enumerate(cols):
        for block, (panel, dirs) in enumerate([("DROP", drop), ("KEEP ONLY", keep)]):
            for si, stem in enumerate(STEMS):
                r = si if block == 0 else si + 4
                a = fig.add_subplot(gs[r, ci]); a.set_xticks([]); a.set_yticks([])
                if tag is None:
                    p = fag.TARGETS / f"{stem}.png"
                elif tag == "FULL":
                    p = fag.find_in(fag.FULL_GRAM_DIR, fag.FULL_GRAM_RX, stem)
                else:
                    p = fag.load_synth(dirs[tag], stem)
                im = load(p)
                a.imshow(im) if im is not None else a.set_facecolor("#eee")
                for sp in a.spines.values(): sp.set_linewidth(0.3); sp.set_color("#999")
                if ci == 0:
                    a.set_ylabel(SHORT[stem], fontsize=4.6, labelpad=1.2)
                if r == 0:
                    head = lab if n is None else f"{lab}\n$n$={n:,}"
                    a.set_title(head, fontsize=FS, rotation=90, ha="left", va="center",
                                rotation_mode="anchor", x=0.5, y=1.02, linespacing=1.18)
    # block labels at the far left
    fig.canvas.draw()
    for block, name in enumerate(["DROP", "KEEP ONLY"]):
        axs = [fig.axes[i] for i in range(len(fig.axes))]
        ys = [gs[r, 0].get_position(fig) for r in ([0, 2] if block == 0 else [4, 6])]
        y = (ys[0].y1 + ys[1].y0) / 2
        fig.text(0.004, y, name, rotation=90, va="center", ha="left",
                 fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=LABEL_IN/(nc*CELL + LABEL_IN), right=0.998,
                        top=1 - TOPIN/((nr+GAP)*CELL + TOPIN), bottom=0.004)
    for ext in ("pdf", "png"):
        p = OUT / f"fig_ablation_wide.{ext}"
        fig.savefig(p, dpi=320, bbox_inches="tight", pad_inches=0.015); print("wrote", p)
    plt.close(fig)


main()
