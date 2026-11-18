#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Produces: Table 1 (statistic families in GSTN)
#
# Provenance: written for this release. The published table was transcribed by
# hand from the model's own family assignment; this script derives it directly
# from `StatTexGramNet.stat_groups_gram_label`, so the numbers in the paper can
# be checked against the code rather than taken on trust.
#
# Runs on CPU in about a minute. No GPU, no dataset, no asset bundle needed.
# ---------------------------------------------------------------------------
"""Derive Table 1 from the model's own statistic-family labels.

    python tables/table01_families.py              # print the table
    python tables/table01_families.py --latex      # emit a LaTeX tabular

Each Gram entry is labelled with the pair of pyramid channels it correlates (for
example `real_L3_O2,imag_L3_O2`), and `assign_gram_groups` buckets those labels
into 14 families by which filter attributes differ between the two channels. This
script instantiates the production configuration and counts them.

Production configuration: colour, 4 orientations, 6 edge levels starting at 1,
no phase difference, magnitude not collapsed. That gives 240 pyramid channels
(216 sub-band + 24 non-sub-band), C(240,2) + 240 = 28,920 Gram entries, plus 9
marginals, for 28,929 statistics.

The structured families marked with a dagger in the paper are the ones whose
members differ in exactly one attribute, plus the three non-sub-band families.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import utils.stattexgramnet as stn  # noqa: E402

EXPECTED_TOTAL = 28929

# plain-English names, as used in the paper's text and Table 1
NAMES = {
    "marginal":        "Marginal",
    "highpass":        "Highpass",
    "lowpass":         "Lowpass",
    "pass_multi":      "Pass-band, multiple attributes",
    "sub_Xori":        "Sub-band, orientation",
    "sub_Xcolor":      "Sub-band, colour",
    "sub_Xlevel":      "Sub-band, level",
    "sub_Xri_eq":      "Sub-band, real-imaginary",
    "sub_Xrm_eq":      "Sub-band, real-magnitude",
    "sub_Xim_eq":      "Sub-band, imaginary-magnitude",
    "sub_real_Xmulti": "Sub-band, real, multiple attributes",
    "sub_imag_Xmulti": "Sub-band, imaginary, multiple attributes",
    "sub_magn_Xmulti": "Sub-band, magnitude, multiple attributes",
    "sub_Xmulti":      "Sub-band, multiple attributes",
}

# families the paper marks as structured (single varying attribute, or non-sub-band)
STRUCTURED = {"marginal", "highpass", "lowpass", "sub_Xori", "sub_Xcolor",
              "sub_Xlevel", "sub_Xri_eq", "sub_Xrm_eq", "sub_Xim_eq"}

ORDER = ["marginal", "highpass", "lowpass", "sub_Xori", "sub_Xcolor", "sub_Xlevel",
         "sub_Xri_eq", "sub_Xrm_eq", "sub_Xim_eq", "pass_multi", "sub_real_Xmulti",
         "sub_imag_Xmulti", "sub_magn_Xmulti", "sub_Xmulti"]


def build_model():
    return stn.StatTexGramNet(
        img_size=(256, 256), color=True, oris=4, edge_levels=6, edge_start=1,
        mag_only=False, include_dphase=False, batch_size=1, bottleneck_size=None,
        device="cpu", compress=False, use_pooling=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    args = ap.parse_args()

    torch.manual_seed(0)
    m = build_model()
    counts = collections.Counter(m.stat_groups_gram_label)

    missing = set(counts) - set(NAMES)
    if missing:
        print(f"WARNING: unnamed families present in the model: {sorted(missing)}")

    total = sum(counts.values())
    structured = sum(v for k, v in counts.items() if k in STRUCTURED)

    print(f"pyramid channels (npyrs)   {m.npyrs}")
    print(f"Gram entries               {m.gram_size}")
    print(f"marginals                  {m.marginal_size}")
    print(f"statistics (stats_size)    {m.stats_size}\n")

    if args.latex:
        print(r"\begin{tabular}{llr}")
        print(r"\toprule")
        print(r"Family & Code & \#Stats \\")
        print(r"\midrule")
        for k in ORDER:
            if k not in counts:
                continue
            dag = r"$^\dagger$" if k in STRUCTURED else ""
            print(rf"{NAMES[k]}{dag} & \texttt{{{k.replace('_', '-')}}} & {counts[k]:,} \\")
        print(r"\midrule")
        print(rf"Total & & {total:,} \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print(f"{"Family":<44}{"Code":<20}{"#Stats":>9}")
        print("-" * 73)
        for k in ORDER:
            if k not in counts:
                continue
            dag = " (structured)" if k in STRUCTURED else ""
            print(f"{NAMES[k] + dag:<44}{k:<20}{counts[k]:>9,}")
        print("-" * 73)
        print(f"{"Total":<64}{total:>9,}")
        print(f"{"of which structured":<64}{structured:>9,}")

    ok = total == EXPECTED_TOTAL == m.stats_size
    print(f"\ncheck: total {total:,} == stats_size {m.stats_size:,} "
          f"== published {EXPECTED_TOTAL:,}  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
