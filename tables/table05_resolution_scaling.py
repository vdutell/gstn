# ---------------------------------------------------------------------------
# Produces: Table 5
#
# Provenance: diffusion-posterior-sampling/make_efficiency_table.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Efficiency-vs-quality trade-off table as synthesis resolution increases.

The representation size is FIXED at 28,929 statistics regardless of image size, while
the image itself grows as 3*N^2 values. So the compression ratio is entirely a function
of resolution, and crosses unity at 3*N^2 = 28,929 -> N ~= 98px. Below that the
"representation" is larger than the image it describes (over-complete); above it, the
model is genuinely compressing.

Quality columns:
  rel_stat_err  scale-free objective error ||s_syn - s_tgt||^2 / ||s_tgt||^2.
                Comparable ACROSS resolutions -- this is the one to read down a column.
  DISTS_256     perceptual distance with synth and target both resampled to 256.
                Comparable WITHIN a resolution only: at low resolution it compares two
                heavily blurred images, which flatters small sizes. Do not read it as
                a quality-vs-resolution curve.
  CV            mean across-seed coefficient of variation of DISTS (3 seeds).

Emits markdown to stdout and writes efficiency_quality_table.md.
Re-run as more resolutions finish; missing sizes are reported as pending.

Run: python make_efficiency_table.py [RUN]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS  # noqa: E402
import csv, sys
from pathlib import Path
import numpy as np

DPS = ASSETS
RUN = sys.argv[1] if len(sys.argv) > 1 else "highres_final6"
ROOT = DPS / "synths" / RUN
SIZES = [16, 32, 64, 128, 256, 512]
STATS = 28_929
ORDER6 = ["polka-dotted_0140", "swirly_0065", "bubbly_0064",
          "grooved_0054", "dotted_0078", "scaly_0168"]
SHORT = {"polka-dotted_0140": "polka", "swirly_0065": "swirly", "bubbly_0064": "bubbly",
         "grooved_0054": "grooved", "dotted_0078": "dotted", "scaly_0168": "scaly",
         "chequered_0177": "chequer", "striped_0050": "striped",
         "honeycombed_0149": "honeyc", "cracked_0056": "cracked",
         "lacelike_0004": "lacelike", "fibrous_0112": "fibrous", "bumpy_0077": "bumpy",
         "sprinkled_0042": "sprinkl", "pleated_0136": "pleated"}


def load(size):
    f = ROOT / f"size{size:04d}" / "per_stem_summary.csv"
    if not f.is_file():
        return None
    return {r["stem"]: r for r in csv.DictReader(open(f))}


def fmt(v, sig=2):
    return f"{v:.{sig}e}" if v < 0.01 else f"{v:.3f}"


def main():
    data = {s: load(s) for s in SIZES}
    have = [s for s in SIZES if data[s]]
    if not have:
        print("no completed sizes yet"); return
    present = list(data[have[0]])
    ordered = [s for s in ORDER6 if s in present]
    # keep the ORDER6 ordering when this IS the 6-texture set; otherwise take every
    # stem in the file (earlier bug: the ORDER6 filter silently dropped 9 of 11 stems)
    stems = ordered if len(ordered) == len(present) else present

    lines = []
    lines.append(f"### Efficiency vs quality as resolution increases ({RUN})\n")
    lines.append("Representation is fixed at **28,929 statistics** at every resolution; "
                 "the image grows as 3·N². Compression crosses unity at ~98×98 px.\n")

    # --- efficiency table ---
    lines.append("| Resolution | Image values (3·N²) | Statistics | Values per statistic | Regime |")
    lines.append("|---|---:|---:|---:|---|")
    for s in SIZES:
        px = 3 * s * s
        r = px / STATS
        regime = "over-complete" if r < 1 else "compressing"
        mark = "" if data[s] else "  *(pending)*"
        lines.append(f"| {s}×{s} | {px:,} | {STATS:,} | **{r:.3f}×** | {regime}{mark} |")
    lines.append("")

    # --- quality per texture ---
    for metric, label, note in [
        ("rel_stat_err", "rel_stat_err ↓ (scale-free objective error)",
         "Comparable across resolutions."),
        ("dists_256", "DISTS₂₅₆ ↓ (perceptual)",
         "Comparable **within** a row only — low-resolution rows compare two blurred "
         "images and are flattered."),
    ]:
        lines.append(f"#### {label}\n\n_{note}_\n")
        hdr = "| Resolution | Values/stat | " + " | ".join(SHORT.get(x, x) for x in stems) + " | **mean** |"
        lines.append(hdr)
        lines.append("|---|---:|" + "---:|" * (len(stems) + 1))
        for s in SIZES:
            d = data[s]
            px = 3 * s * s
            if not d:
                lines.append(f"| {s}×{s} | {px/STATS:.3f}× | " +
                             " | ".join(["–"] * len(stems)) + " | *pending* |")
                continue
            vals = [float(d[x][f"{metric}_mean"]) for x in stems]
            lines.append(f"| {s}×{s} | {px/STATS:.3f}× | " +
                         " | ".join(fmt(v) for v in vals) +
                         f" | **{fmt(float(np.mean(vals)))}** |")
        lines.append("")

    # --- variability ---
    lines.append("#### Across-seed CV of DISTS (%) — 3 seeds\n")
    lines.append("| Resolution | " + " | ".join(SHORT.get(x, x) for x in stems) + " | **mean** |")
    lines.append("|---|" + "---:|" * (len(stems) + 1))
    for s in SIZES:
        d = data[s]
        if not d:
            lines.append(f"| {s}×{s} | " + " | ".join(["–"] * len(stems)) + " | *pending* |")
            continue
        cv = [100 * float(d[x]["dists_256_cv"]) for x in stems]
        lines.append(f"| {s}×{s} | " + " | ".join(f"{v:.1f}" for v in cv) +
                     f" | **{np.mean(cv):.1f}** |")
    lines.append("")

    # --- the trade-off summary ---
    lines.append("#### Trade-off summary\n")
    lines.append("| Resolution | Values/stat | mean rel_stat_err | quality per unit compression |")
    lines.append("|---|---:|---:|---|")
    for s in SIZES:
        d = data[s]
        px = 3 * s * s; r = px / STATS
        if not d:
            lines.append(f"| {s}×{s} | {r:.3f}× | *pending* | – |")
            continue
        e = float(np.mean([float(d[x]["rel_stat_err_mean"]) for x in stems]))
        lines.append(f"| {s}×{s} | {r:.3f}× | {fmt(e)} | {'compressing ' if r>=1 else 'over-complete '}"
                     f"at err {fmt(e)} |")
    out = "\n".join(lines)
    print(out)
    (DPS / f"efficiency_quality_table_{RUN}.md").write_text(out)
    print(f"\n[written to efficiency_quality_table_{RUN}.md]")


if __name__ == "__main__":
    main()
