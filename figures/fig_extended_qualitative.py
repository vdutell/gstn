# ---------------------------------------------------------------------------
# Produces: Appendix extended qualitative grids
#
# Provenance: StatTexNetExtended/rebuttal/camera_ready/make_fig_extended_qualitative.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Extended qualitative comparison over a WIDER sample of DTD categories.

Requested by 3Kqw Q1 ("I can't tell if GSTN uniquely produces better quality textures") and
n7wj W1 ("the qualitative examples are not uniformly strong"); promised in both rebuttals as an
appendix extension of Figure 2.

No new synthesis: every image already exists on disk. GSTN-zpad comes from the DTD-141 run
(pixelbp_prodigy_depleted_full_meanonly_black), not the 11-stem qualitative run that Figure 2
uses, because only the former covers the wider sample.

PANEL=A|B selects which half of the category sample to render.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, BASELINES, DTD, FIGURES# noqa: E402
import os, re
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPS = ASSETS
DTD = DTD
TARGETS = DPS / "pixelbp_prodigy_targets"
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)

DIRS = {
    "GSTN-zpad":     (DPS/"synths/pixelbp_prodigy_depleted_full_meanonly_black", r"^(.+)_gpu[01]_synth\.png$"),
    "GSTN-wrap":     (DPS/"synths/pixelbp_prodigy_depleted_full_meanonly_wrap",  r"^(.+)_gpu[01]_synth\.png$"),
    "P&S":           (BASELINES / "dtd_121_ps_synths",                                   r"^\d+_psynth_\d+_(.+)\.jpg$"),
    "Ustyuzhaninov": (DPS/"synths/bench_ustyuzhaninov_multiscale_pytorch",       r"^(.+)_synth\.png$"),
    "Gatys":         (BASELINES / "dtd_121_gatys",                                       r"^(.+)_gen_[\d\-]+\.png$"),
}
ROWS = ["Target", "GSTN-zpad", "GSTN-wrap", "P&S", "Ustyuzhaninov", "Gatys"]
NSTAT = {"Target": "196,608 px", "GSTN-zpad": "28,929", "GSTN-wrap": "28,929",
         "P&S": "~8,400", "Ustyuzhaninov": "524,800", "Gatys": "852k"}

# 16 categories NOT in Figure 2, chosen to span the DTD taxonomy rather than by performance.
PANELS = {
    # Figure 1 of 2: 16 categories disjoint from the 11 in Figure 2.
    "A": ["banded", "braided", "bubbly", "dotted", "grid", "grooved", "knitted", "marbled"],
    "B": ["meshed", "paisley", "perforated", "scaly", "stained", "studded", "veined", "woven"],
    # Figure 2 of 2: 16 more, disjoint from both of the above. Drawn from the 20 DTD
    # categories still unused; the four omitted (freckled, potholed, stratified, wrinkled)
    # are the closest duplicates of ones already shown (flecked, pitted, lined, smeared).
    "C": ["blotchy", "cobwebbed", "crosshatched", "crystalline",
          "flecked", "frilly", "gauzy", "interlaced"],
    "D": ["lined", "matted", "pitted", "porous",
          "smeared", "spiralled", "waffled", "zigzagged"],
    # Figure 3 of 3: every DTD category is already shown, so this panel uses 16 DIFFERENT
    # EXEMPLARS -- the 4 categories never shown (freckled, potholed, stratified, wrinkled)
    # plus a second image from 12 categories that appeared earlier. Entries are explicit
    # stems, not categories, and are labelled with the full stem so the reader can see these
    # are different source images rather than repeats.
    "E": ["freckled_0102", "potholed_0046", "stratified_0098", "wrinkled_0060",
          "banded_0114", "bubbly_0130", "cobwebbed_0089", "crystalline_0141"],
    "F": ["flecked_0122", "grid_0023", "interlaced_0115", "lined_0094",
          "meshed_0165", "pitted_0081", "porous_0110", "spiralled_0071"],
    # Figure 4 of 4: 16 further exemplars, drawn from the categories shown only once so far
    # so the four figures spread evenly over the taxonomy. Disjoint from everything above.
    "G": ["blotchy_0062", "braided_0145", "bumpy_0166", "chequered_0064",
          "cracked_0079", "crosshatched_0081", "dotted_0128", "fibrous_0164"],
    "H": ["freckled_0140", "frilly_0049", "gauzy_0105", "grooved_0055",
          "honeycombed_0150", "knitted_0171", "lacelike_0073", "marbled_0114"],
}
PAIRS = {"1": ("A", "B"), "2": ("C", "D"), "3": ("E", "F"), "4": ("G", "H")}
STEM_RX = __import__("re").compile(r"_\d{4}$")
SIZE = 256


def index(d, pat):
    rx = re.compile(pat); out = {}
    if d.exists():
        for p in d.iterdir():
            m = rx.match(p.name)
            if m: out[m.group(1)] = p
    return out


def load(p):
    im = cv2.imread(str(p))
    if im is None: return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (SIZE, SIZE):
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return im


def combined():
    """All 16 categories in ONE figure: two stacked blocks of 8 columns, 6 methods each.
    8 cols x 12 rows gives aspect ~0.63, which is almost exactly a NeurIPS page text area
    (5.5 x 9 in, 0.61), so it fills a single page with \\includegraphics[height=...]."""
    import matplotlib.gridspec as gridspec
    idx = {k: index(d, p) for k, (d, p) in DIRS.items()}
    tgt = {q.stem: q for q in TARGETS.glob("*.png")}

    which = os.environ.get("PAIR", "1")
    blocks = []
    for key in PAIRS[which]:
        stems = []
        for c in PANELS[key]:
            if STEM_RX.search(c):                      # explicit stem, not a category
                if all(c in idx[k] for k in DIRS) and c in tgt: stems.append(c)
                else: print(f"  !! incomplete stem {c}")
                continue
            cand = sorted(s for s in idx["GSTN-zpad"] if s.rsplit("_", 1)[0] == c)
            pick = next((s for s in cand if all(s in idx[k] for k in DIRS) and s in tgt), None)
            if pick is None: print(f"  !! no complete stem for {c}"); continue
            stems.append(pick)
        blocks.append(stems)

    full_lab = True                  # full DTD stem in every panel, so a reader can tell
                                     # exemplars apart across the three figures
    nr, nc = len(ROWS), max(len(b) for b in blocks)
    SPACER = 0.42
    # Column labels are the binding width constraint once the 4-digit id is included
    # ("crosshatched_0046" is 92pt at 6.8pt, against a 79pt column). Measure the widest
    # label actually used and size the font to it rather than hard-coding a number.
    # Size from the widest label across ALL THREE figures, not just this one, so the three
    # sibling pages share a label size instead of drifting (6.6 / 5.7 / 6.3pt).
    _all = []
    for _v in PANELS.values():
        for _c in _v:
            if STEM_RX.search(_c):
                _all.append(_c)
            else:
                _cand = sorted(x for x in idx["GSTN-zpad"] if x.rsplit("_", 1)[0] == _c)
                _pk = next((x for x in _cand if all(x in idx[k] for k in DIRS) and x in tgt), None)
                if _pk: _all.append(_pk)
    _probe = plt.figure(figsize=(1, 1)); _r = _probe.canvas.get_renderer()
    _w1 = max(_probe.text(0, 0, l, fontsize=10).get_window_extent(renderer=_r).width
              for l in _all) / 10.0
    TITLE_FS = min(7.5, 0.97 * (1.1 * 72) / _w1)
    plt.close(_probe)
    print(f"  widest label overall {max(_all, key=lambda x: len(x))!r} -> title font {TITLE_FS:.2f}pt")
    fig = plt.figure(figsize=(nc*1.1, (2*nr + SPACER)*1.1 + 0.62))
    gs = gridspec.GridSpec(2*nr + 1, nc, figure=fig,
                           height_ratios=[1]*nr + [SPACER] + [1]*nr,
                           wspace=0.035, hspace=0.035)
    for b, stems in enumerate(blocks):
        r0 = 0 if b == 0 else nr + 1
        for r, row in enumerate(ROWS):
            for c, stem in enumerate(stems):
                a = fig.add_subplot(gs[r0+r, c]); a.set_xticks([]); a.set_yticks([])
                path = tgt[stem] if row == "Target" else idx[row].get(stem)
                im = load(path) if path else None
                a.imshow(im) if im is not None else a.set_facecolor("#eee")
                for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999")
                if r == 0:
                    # No LaTeX escaping: matplotlib's default renderer prints "\_"
                    # literally, and a bare "_" is only special inside mathtext.
                    lab = stem if full_lab else stem.rsplit("_", 1)[0]
                    a.set_title(lab, fontsize=TITLE_FS, pad=3)
                if c == 0: a.set_ylabel(f"{row}\n({NSTAT[row]})", fontsize=7, labelpad=3)
    fig.subplots_adjust(left=0.058, right=0.999,
                        top=1-0.34/((2*nr+SPACER)*1.1+0.62), bottom=0.0015)
    for ext in ("pdf", "png"):
        tag = "" if which == "1" else f"_{which}"
        q = OUT/f"fig_extended_qualitative{tag}.{ext}"
        fig.savefig(q, bbox_inches="tight", pad_inches=0.012, dpi=350); print("wrote", q)
    plt.close(fig)


def main():
    idx = {k: index(d, p) for k, (d, p) in DIRS.items()}
    tgt = {p.stem: p for p in TARGETS.glob("*.png")}
    panel = os.environ.get("PANEL", "A")
    cats = PANELS[panel]

    # One stem per category: the first that every method has.
    stems = []
    for c in cats:
        cand = sorted(s for s in idx["GSTN-zpad"] if s.rsplit("_", 1)[0] == c)
        pick = next((s for s in cand
                     if all(s in idx[k] for k in DIRS) and s in tgt), None)
        if pick is None:
            print(f"  !! no complete stem for {c}"); continue
        stems.append(pick)

    nr, nc = len(ROWS), len(stems)
    fig, ax = plt.subplots(nr, nc, figsize=(nc*1.15, nr*1.15+0.30), squeeze=False,
                           gridspec_kw=dict(wspace=0.035, hspace=0.035))
    for r, row in enumerate(ROWS):
        for c, stem in enumerate(stems):
            a = ax[r][c]; a.set_xticks([]); a.set_yticks([])
            path = tgt[stem] if row == "Target" else idx[row].get(stem)
            im = load(path) if path else None
            if im is None:
                a.set_facecolor("#eee")
            else:
                a.imshow(im)
            for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999")
            if r == 0:
                a.set_title(stem.rsplit("_", 1)[0], fontsize=7, pad=3)
            if c == 0:
                a.set_ylabel(f"{row}\n({NSTAT[row]})", fontsize=6.5, labelpad=3)
    plt.subplots_adjust(left=0.055, right=0.999, top=1-0.30/(nr*1.15+0.30), bottom=0.002)
    for ext in ("pdf", "png"):
        p = OUT/f"fig_extended_qualitative_{panel}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.012, dpi=350); print("wrote", p)
    plt.close(fig)


import sys
if os.environ.get("COMBINED") or "combined" in sys.argv:
    combined()
else:
    main()
