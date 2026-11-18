# ---------------------------------------------------------------------------
# Produces: Figure 2
#
# Provenance: diffusion-posterior-sampling/make_method_comparison_munca.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, BASELINES, DTD, FIGURES, MUNCA# noqa: E402
"""
Qualitative cross-method comparison: target + 4 baselines + 2 STGN variants,
for ~11 example textures. Layout: methods as ROWS, categories as COLUMNS.

Adds muNCA (Mordvintsev, Niklasson & Randazzo 2021) as a row: 588 learned
parameters, trained per texture with the authors' canonical notebook at 8,000
steps on the same 256x256 targets every other method uses.

Rows are ordered by representation size so the figure reads as a compactness
axis: muNCA (588) < P&S (8.4K) < GSTN (28.9K) < Ustyuzhaninov (132K) < Gatys (825K).

muNCA trains one CA per texture, so a missing cell means that texture has not
finished training yet -- the run is ~70 min/texture.

Output: fig_method_comparison_munca.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DPS = ASSETS
DTD = DTD

TARGETS = DPS / "pixelbp_prodigy_targets"
DIRS = {
    "P&S":           BASELINES / "dtd_121_ps_synths",
    "Gatys":         BASELINES / "dtd_121_gatys",
    "GSTN-wrap":     DPS / "synths/pixelbp_prodigy_depleted_full_meanonly_wrap",
    "GSTN-zero-pad": DPS / "synths/stgn_meanonly_black_lr002_qualitative",
    "Ustyuzhaninov": DPS / "synths/bench_ustyuzhaninov_multiscale_pytorch",
    "muNCA-256":     MUNCA / "fig2_8k_256",
    "muNCA-128":     MUNCA / "fig2_8k_128",
}
# Methods as ROWS, our methods after target; categories are COLUMNS.
# Only muNCA's native 128 is shown: the two resolutions look near-identical except on
# swirly, and dropping one row keeps the figure from overflowing the page. NOTE the quality
# table scores muNCA at 256 (the common evaluation resolution), so that row is labelled
# "muNCA (256)" there -- keep the two labels explicit rather than implying one number.
ROWS = ["target", "GSTN-zero-pad", "GSTN-wrap", "muNCA-128",
        "P&S", "Ustyuzhaninov", "Gatys"]
import os as _os
_SAMPLE_PRE = _os.environ.get("SAMPLE", "1")
OUT = FIGURES / (f"fig_method_comparison_munca_sample{_SAMPLE_PRE}.png"
             if _SAMPLE_PRE != "1" else "fig_method_comparison_munca.png")

import os
SAMPLE = os.environ.get("SAMPLE", "1")
if SAMPLE == "2":
    CATEGORIES = ["chequered", "scaly", "lined", "paisley", "studded", "pitted"]
else:
    # Span the scope of texture types rather than picking by performance:
    # chequered    - regular periodic grid
    # striped      - oriented periodic
    # polka-dotted - dot lattice
    # honeycombed  - cellular repeating tiles
    # fibrous      - fine oriented / directional
    # bumpy        - stochastic
    # swirly       - organic curvilinear
    # crystalline  - faceted / piecewise-flat
    CATEGORIES = ["chequered", "striped", "polka-dotted",
                  "honeycombed", "cracked", "lacelike",
                  "fibrous", "bumpy", "sprinkled",
                  "pleated", "swirly"]

# Per-category stem overrides for visual clarity (bench-intersection only).
STEM_OVERRIDE = {
    "chequered": "chequered_0177",  # B&W high-contrast check; default 0064 is subtle
    "pleated":   "pleated_0136",    # pink pleats with clear ridges; default 0067 is faint
}

# Representation size (number of statistics / params per method).
# Target = raw pixel count = 256*256*3 = 196608.
PARAM_COUNTS = {
    "target":         196_608,
    "P&S":              8_400,
    # Gatys 2015 report BOTH: ~852k using all layers up to pool4, and ~177k using one layer
    # per scale (conv1_1, pool1-4) "while hardly loosing any quality". Label shows both, since
    # quoting 852k alone omits their own reduction result.
    "Gatys":          852_128,
    "GSTN-wrap":       28_929,
    "GSTN-zero-pad":   28_929,
    "Ustyuzhaninov":  524_800,
    "muNCA-256":          588,   # dense 4*CHN->CHN + bias, CHN=12; kernels are fixed
    "muNCA-128":          588,   # identical count: the CA is resolution-independent
}

PATTERNS = {
    "P&S":           re.compile(r"^\d+_psynth_\d+_(.+)\.jpg$"),
    "Gatys":         re.compile(r"^(.+)_gen_[\d\-]+\.png$"),
    "GSTN-wrap":     re.compile(r"^(.+)_gpu[01]_synth\.png$"),
    "GSTN-zero-pad": re.compile(r"^(.+)_synth\.png$"),
    "Ustyuzhaninov": re.compile(r"^(.+)_synth\.png$"),
    "muNCA-256":     re.compile(r"^(.+)_munca\.png$"),
    "muNCA-128":     re.compile(r"^(.+)_munca\.png$"),
}


def index_dir(d, regex):
    out = {}
    for p in d.iterdir():
        m = regex.match(p.name)
        if m:
            out[m.group(1)] = p
    return out


def main():
    indices = {name: index_dir(d, PATTERNS[name]) for name, d in DIRS.items()}

    # Pick: sorted-first stem per category that exists in all method dirs.
    # Usty has partial coverage; GSTN-zero-pad is the qualitative-only 11-stem
    # set (no full DTD coverage), so neither is required for stem selection.
    REQUIRED_DIRS = [name for name in indices
                     if name not in ("Ustyuzhaninov", "GSTN-zero-pad",
                                     "muNCA-256", "muNCA-128")]
    cols = []
    for cat in CATEGORIES:
        candidates = sorted([s for s in indices["Gatys"] if s.startswith(cat + "_")])
        override = STEM_OVERRIDE.get(cat)
        if override and all(override in indices[name] for name in REQUIRED_DIRS):
            chosen = override
        else:
            chosen = None
            for stem in candidates:
                if all(stem in indices[name] for name in REQUIRED_DIRS):
                    chosen = stem
                    break
        if chosen is None:
            print(f"⚠️  no full-coverage stem for {cat}; using Gatys-first")
            chosen = candidates[0]
        target = TARGETS / f"{chosen}.png"
        if not target.exists():
            raise FileNotFoundError(f"missing target {target}")
        col = {"cat": cat, "stem": chosen, "target": target,
               **{name: indices[name].get(chosen) for name in indices}}
        cols.append(col)
        print(f"  {cat:14s}  {chosen}")

    # Build figure: METHODS are rows, CATEGORIES are columns.
    n_rows = len(ROWS)
    n_cols = len(cols)
    cell = 0.6     # ≤6.75" wide; rows compress vertically to remove gaps
    label_w = 0.30  # narrow because row labels are now rotated 90°
    title_h = 0.18  # only the topmost row has column titles; tight after that
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell * n_cols + label_w,
                                      cell * n_rows + title_h),
                             squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=0.0))

    for r, key in enumerate(ROWS):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            path = col["target"] if key == "target" else col.get(key)
            if path is None or not Path(path).exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")
            else:
                ax.imshow(Image.open(path))
            ax.axis("off")
            if r == 0:
                ax.set_title(col["cat"], fontsize=6, pad=2)
        # Row label: rotated 90°, two-line — method/(ours) on line 1,
        # statistic count on line 2.
        if key == "target":
            display_key = "Target"
        elif key == "GSTN-wrap":
            display_key = "GSTN-wrap"
        elif key == "GSTN-zero-pad":
            display_key = "GSTN-zpad"
        else:
            display_key = key
        n = PARAM_COUNTS.get(key)
        count_str = (f"({round(n / 1000)}K)" if n and n >= 1000
                     else (f"({n})" if n else ""))
        if key == "Gatys":
            count_str = "(852K/177K)"      # full / reduced, both from Gatys 2015 Fig. 3A
        row_label = f"{display_key}\n{count_str}" if count_str else display_key
        if key == "muNCA-256":
            display_key = "muNCA 256"
        elif key == "muNCA-128":
            display_key = "muNCA 128"
        is_ours = key in ("GSTN-wrap", "GSTN-zero-pad")  # bold = our method
        axes[r, 0].text(-0.18, 0.5, row_label,
                        transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=6,
                        fontweight=("bold" if is_ours else "normal"),
                        linespacing=1.0)

    plt.subplots_adjust(left=0.05, right=0.995, top=1.0 - title_h / (cell * n_rows + title_h),
                        bottom=0.0, wspace=0.04, hspace=0.0)
    plt.savefig(OUT, dpi=220, bbox_inches="tight", pad_inches=0.02)
    # also emit a vector copy into the camera-ready tree for \includegraphics
    _cr = FIGURES
    if _cr.exists() and SAMPLE == "1":
        for _ext in ("pdf", "png"):
            _q = _cr / f"fig2_method_comparison_munca.{_ext}"
            plt.savefig(_q, dpi=220, bbox_inches="tight", pad_inches=0.02)
            print(f"[method-comparison] also saved {_q}")
    plt.close()
    print(f"[method-comparison] saved {OUT}")


if __name__ == "__main__":
    main()
