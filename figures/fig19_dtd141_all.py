# ---------------------------------------------------------------------------
# Produces: Figure 19 (nine-page DTD-141 grid)
#
# Provenance: StatTexNetExtended/rebuttal/camera_ready/make_fig_dtd141_all.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""The ENTIRE DTD-141 evaluation set as one multi-page PDF.

16 stems per page (two blocks of 8), 6 method rows each -> 9 pages for 141 stems. Drop the
whole thing into the appendix with pdfpages:

    \\usepackage{pdfpages}
    \\includepdf[pages=-]{figures/fig_dtd141_all.pdf}

Nothing is hand-picked and nothing repeats: stems are every image in the 141 set for which
all five methods and the target exist, in alphabetical order.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, BASELINES, DTD, FIGURES, MUNCA# noqa: E402
import re
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

DPS = ASSETS
DTD = DTD
TARGETS = DPS / "pixelbp_prodigy_targets"
OUT = FIGURES
DIRS = {
    "GSTN-zpad":     (DPS/"synths/pixelbp_prodigy_depleted_full_meanonly_black", r"^(.+)_gpu[01]_synth\.png$"),
    "GSTN-wrap":     (DPS/"synths/pixelbp_prodigy_depleted_full_meanonly_wrap",  r"^(.+)_gpu[01]_synth\.png$"),
    "P&S":           (BASELINES / "dtd_121_ps_synths",                                   r"^\d+_psynth_\d+_(.+)\.jpg$"),
    "Ustyuzhaninov": (DPS/"synths/bench_ustyuzhaninov_multiscale_pytorch",       r"^(.+)_synth\.png$"),
    "Gatys":         (BASELINES / "dtd_121_gatys",                                       r"^(.+)_gen_[\d\-]+\.png$"),
}
ROWS = ["Target", "GSTN-zpad", "GSTN-wrap", "P&S", "Ustyuzhaninov", "Gatys"]
# muNCA exists only for the 11 Figure-2 stems: it costs 1-2 h per image on one 4090 without
# TPU access, which is why DTD-141 was out of reach for it. Rather than pad 130 columns with
# blanks, those 11 stems get page 1 to themselves WITH the two muNCA rows, and are then not
# repeated on the remaining pages. Every image in the set still appears exactly once.
MUNCA = MUNCA
ROWS_MUNCA = ["Target", "GSTN-zpad", "GSTN-wrap", "muNCA-128",
              "P&S", "Ustyuzhaninov", "Gatys"]
NSTAT = {"Target": "196,608 px", "GSTN-zpad": "28,929", "GSTN-wrap": "28,929",
         "muNCA-256": "588", "muNCA-128": "588",
         "P&S": "~8,400", "Ustyuzhaninov": "524,800", "Gatys": "852k"}
NCOL, SIZE, DPI = 8, 256, 220
CELL = 1.0          # was 1.1 in; trims the page without touching the layout
PAD  = 0.22         # inches of white border baked into each page


def index(d, pat):
    rx = re.compile(pat)
    return {m.group(1): p for p in d.iterdir() if (m := rx.match(p.name))} if d.exists() else {}


def load(p):
    im = cv2.imread(str(p))
    if im is None: return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] == (SIZE, SIZE): return im
    # muNCA-128 is genuinely 128 px; upsample smoothly so it matches how Figure 2 renders it
    interp = cv2.INTER_AREA if im.shape[0] > SIZE else cv2.INTER_LINEAR
    return cv2.resize(im, (SIZE, SIZE), interpolation=interp)


def cell_path(row, stem, idx, tgt):
    if row == "Target":      return tgt.get(stem)
    if row == "muNCA-256":   return MUNCA / "fig2_8k_256" / f"{stem}_munca.png"
    if row == "muNCA-128":   return MUNCA / "fig2_8k_128" / f"{stem}_munca.png"
    return idx[row].get(stem)


def main():
    idx = {k: index(d, p) for k, (d, p) in DIRS.items()}
    tgt = {p.stem: p for p in TARGETS.glob("*.png")}
    stems = sorted(s for s in idx["GSTN-zpad"]
                   if all(s in idx[k] for k in DIRS) and s in tgt)
    print(f"  {len(stems)} complete stems")

    probe = plt.figure(figsize=(1, 1)); r = probe.canvas.get_renderer()
    w1 = max(probe.text(0, 0, s, fontsize=10).get_window_extent(renderer=r).width
             for s in stems) / 10.0
    FS = min(7.5, 0.97 * (CELL * 72) / w1); plt.close(probe)
    print(f"  widest label {max(stems, key=len)!r} -> label font {FS:.2f}pt")

    munca_stems = sorted(st for st in stems
                         if (MUNCA / "fig2_8k_128" / f"{st}_munca.png").exists())
    rest = [st for st in stems if st not in set(munca_stems)]
    print(f"  {len(munca_stems)} stems have muNCA -> own first page; {len(rest)} on the rest")

    # (stems, rows, ncol) per page. Page 1 is the muNCA page at 6 columns so the taller
    # 8-row block stays close to the other pages' proportions.
    # Even chunks, so the last page is not left with a stem or two rattling around
    # (130 split 16-at-a-time leaves a final page of 2).
    import math
    npg = math.ceil(len(rest) / (2 * NCOL))
    sizes = [len(rest)//npg + (1 if i < len(rest) % npg else 0) for i in range(npg)]
    plan, k = [(munca_stems, ROWS_MUNCA, 6)], 0
    for sz in sizes:
        plan.append((rest[k:k+sz], ROWS, NCOL)); k += sz

    SPACER = 0.16
    out = OUT / "fig_dtd141_all.pdf"
    with PdfPages(out) as pdf:
        for pi, (page, rows, ncol) in enumerate(plan, 1):
            nr = len(rows)
            half = (len(page) + 1) // 2
            blocks = [page[:half], page[half:]]
            fig = plt.figure(figsize=(ncol*CELL, (2*nr + SPACER)*CELL + 0.52))
            gs = gridspec.GridSpec(2*nr + 1, ncol, figure=fig,
                                   height_ratios=[1]*nr + [SPACER] + [1]*nr,
                                   wspace=0.006, hspace=0.006)
            for b, blk in enumerate(blocks):
                r0 = 0 if b == 0 else nr + 1
                for ri, row in enumerate(rows):
                    for c in range(ncol):
                        a = fig.add_subplot(gs[r0+ri, c]); a.set_xticks([]); a.set_yticks([])
                        if c >= len(blk):
                            a.axis("off"); continue
                        stem = blk[c]
                        pth = cell_path(row, stem, idx, tgt)
                        im = load(pth) if pth and Path(pth).exists() else None
                        a.imshow(im) if im is not None else a.set_facecolor("#eee")
                        for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("#999")
                        if ri == 0: a.set_title(stem, fontsize=FS, pad=3)
                        if c == 0: a.set_ylabel(f"{row}\n({NSTAT[row]})", fontsize=7, labelpad=3)
            head = ("DTD-141 full qualitative comparison \u2014 page 1 of %d: the 11 stems with muNCA"
                    % len(plan)) if pi == 1 else (
                    "DTD-141 full qualitative comparison \u2014 page %d of %d" % (pi, len(plan)))
            fig.suptitle(head, fontsize=9, y=0.998)
            fig.subplots_adjust(left=0.058 if ncol >= 8 else 0.075, right=0.999,
                                top=1-0.36/((2*nr+SPACER)*CELL+0.52), bottom=0.0015)
            pdf.savefig(fig, dpi=DPI, bbox_inches="tight", pad_inches=PAD)
            plt.close(fig)
            print(f"  page {pi}/{len(plan)}: {len(page)} stems, {nr} rows", flush=True)
    print("wrote", out)


main()
