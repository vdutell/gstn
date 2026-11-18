# ---------------------------------------------------------------------------
# Produces: Table 4
#
# Provenance: diffusion-posterior-sampling/make_final_efficiency_table.py from the original working tree.
# Copied verbatim except that hardcoded absolute paths now resolve through
# gstn_paths. See docs/REPRODUCE.md.
# Run from the repository root.
# ---------------------------------------------------------------------------
"""Final efficiency-vs-quality table: representation size against PERCEPTUAL quality.

Deliberately excludes rel_stat_err. That quantity measures how well the OPTIMIZER met
the objective, not how good the model's representation is -- a synthesis with a tiny
statistic error can still look nothing like the target (swirly is the standing example:
error ~3e-06 at every resolution, appearance never close). It is reported once at the
bottom purely as a convergence VALIDITY CHECK: if it were large, the quality numbers
above it would be uninterpretable because the optimizer, not the model, would be the
limiting factor.

Also excludes DISTS_256-style metrics that resample everything to a common size: at low
resolution those compare two heavily blurred images, which flatters small sizes.

Quality is measured at NATIVE resolution -- synthesis and target both at the synthesis
size, i.e. what a viewer sees.

Run: python make_final_efficiency_table.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from gstn_paths import ASSETS, TABLES# noqa: E402
import csv
from pathlib import Path
import numpy as np

DPS = ASSETS
STATS = 28_929
# measured wall-clock for the whole batch, 20k iters (from run logs)
WALL = {
    "resolution_scaling": ({16: 3770, 32: 4280, 64: 5330, 128: 6676, 256: 7967}, 33),
    "highres_final6": ({16: 1725, 32: 1846, 64: 2781, 128: 4459, 256: 5639, 512: 12059}, 18),
}
LOWER = {"DISTS", "LPIPS", "GMSD"}


def build(run, title, note):
    q = DPS / f"native_quality_{run}.csv"
    if not q.is_file():
        return f"_(native_quality_{run}.csv missing)_\n"
    rows = list(csv.DictReader(open(q)))
    wall, B = WALL[run]
    names = ["DISTS", "LPIPS", "FSIM", "GMSD", "HaarPSI", "VIF"]
    L = [f"### {title}\n", note, ""]
    L.append("| Resolution | Image values (3·N²) | Values per statistic | "
             + " | ".join(f"{n} {'↓' if n in LOWER else '↑'}" for n in names)
             + " | Wall-clock / synthesis |")
    L.append("|---|---:|---:|" + "---:|" * (len(names) + 1))
    for r in rows:
        s = int(r["size"]); px = 3 * s * s
        flag = " \\*" if s < 64 else ""
        cells = []
        for n in names:
            v = float(r[n]) if r[n] not in ("", "nan") else float("nan")
            cells.append("n/a" if not np.isfinite(v) else f"{v:.3f}")
        w = wall.get(s)
        wc = f"{w/B:.0f} s" if w else "–"
        L.append(f"| {s}×{s}{flag} | {px:,} | {px/STATS:.3f}× | " + " | ".join(cells)
                 + f" | {wc} |")
    L.append("")
    L.append("\\* *Below 64 px, DISTS and LPIPS use VGG features and are outside their "
             "validated regime (5 downsampling stages); at these sizes the target itself "
             "carries little texture information, so distances are small for trivial "
             "reasons. VIF is undefined below 64 px.*")
    return "\n".join(L)


def convergence_note():
    L = ["\n---\n", "### Convergence validity check (not a quality measure)\n",
         "Median relative statistic error `‖s_syn − s_tgt‖²/‖s_tgt‖²` per resolution. "
         "Reported only to confirm the optimizer met its objective at every size, so the "
         "quality numbers above reflect the representation rather than optimizer failure. "
         "It is **not** a quality metric: swirly attains ~3e−06 at every resolution while "
         "never resembling its target.\n",
         "| Resolution | 11-texture | 6-texture |", "|---|---:|---:|"]
    for s in [16, 32, 64, 128, 256, 512]:
        vals = []
        for run in ["resolution_scaling", "highres_final6"]:
            f = DPS / f"synths/{run}/size{s:04d}/per_stem_summary.csv"
            if f.is_file():
                v = np.median([float(r["rel_stat_err_mean"]) for r in csv.DictReader(open(f))])
                vals.append(f"{v:.1e}")
            else:
                vals.append("–")
        L.append(f"| {s}×{s} | {vals[0]} | {vals[1]} |")
    return "\n".join(L)


out = "\n\n".join([
    build("resolution_scaling", "11 Figure-2 textures × 3 seeds (16→256)",
          "_All targets natively ≥305 px, so every size is a genuine downsample._"),
    build("highres_final6", "6 high-resolution textures × 3 seeds (16→512)",
          "_All targets natively ≥522 px; 512 is the largest size reachable without "
          "upsampling any target._"),
    convergence_note(),
])
print(out)
(TABLES / "efficiency_vs_quality_FINAL.md").write_text(out + "\n")
print("\n[written efficiency_vs_quality_FINAL.md]")
