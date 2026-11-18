#!/usr/bin/env bash
# Assemble the asset bundle that fetch_assets.sh downloads.
#
# MAINTAINER SCRIPT. Users do not run this; it is here so the bundle's contents
# are reproducible and auditable rather than a hand-assembled tarball.
#
# It reads from the original working tree (three sibling repos) and writes a
# staging directory plus a tarball and its sha256.
set -euo pipefail

DPS=${DPS:-/home/mer/projects/diffusion-posterior-sampling}
STX=${STX:-/home/mer/projects/StatTexNetExtended}
DTD=${DTD:-/home/mer/projects/datasets/dtd}
STAGE=${STAGE:-/home/mer/projects/gstn-assets}
STEMS=${STEMS:-/tmp/claude-1001/-home-mer-projects/88ce048d-993e-48fc-9dc0-70b3ae4059de/scratchpad/needed_stems.txt}

# Figure 6 uses only these categories, so the twelve depletion dirs are pruned to them
FIG6_CATS="banded cracked polka-dotted bumpy"

rm -rf "$STAGE"; mkdir -p "$STAGE"

echo "== targets (only the stems any figure references) =="
mkdir -p "$STAGE/pixelbp_prodigy_targets"
n=0
while read -r s; do
  [ -f "$DPS/pixelbp_prodigy_targets/$s.png" ] && \
    cp "$DPS/pixelbp_prodigy_targets/$s.png" "$STAGE/pixelbp_prodigy_targets/" && n=$((n+1))
done < "$STEMS"
echo "   $n targets"

echo "== synth directories used by the qualitative figures =="
mkdir -p "$STAGE/synths"
for d in pixelbp_prodigy_depleted_full_meanonly_black \
         pixelbp_prodigy_depleted_full_meanonly_wrap \
         stgn_meanonly_black_lr002_qualitative \
         bench_ustyuzhaninov_multiscale_pytorch \
         highres_final6 seed_consistency_11; do
  [ -d "$DPS/synths/$d" ] && cp -r "$DPS/synths/$d" "$STAGE/synths/" && echo "   $d" || echo "   (absent: $d)"
done

echo "== family ablation dirs (drop/keep) =="
for d in "$DPS"/synths/pixelbp_prodigy_depleted_drop_*_drop_* \
         "$DPS"/synths/pixelbp_prodigy_depleted_keep_*_keep_*; do
  [ -d "$d" ] && cp -r "$d" "$STAGE/synths/"
done
echo "   $(ls -d "$STAGE"/synths/*_drop_* "$STAGE"/synths/*_keep_* 2>/dev/null | wc -l) dirs"

echo "== depletion dirs, pruned to the Figure 6 categories =="
for cond in top rand bot; do
  for N in 100 1000 5000 10000; do
    src="$DPS/synths/pixelbp_prodigy_depleted_${cond}${N}"
    [ -d "$src" ] || continue
    dst="$STAGE/synths/pixelbp_prodigy_depleted_${cond}${N}"
    mkdir -p "$dst"
    for cat in $FIG6_CATS; do
      cp "$src/${cat}_"*_synth.png "$dst/" 2>/dev/null || true
    done
  done
done
echo "   $(ls "$STAGE"/synths/pixelbp_prodigy_depleted_top10000 2>/dev/null | wc -l) files per depletion dir"

echo "== baseline syntheses we redistribute =="
mkdir -p "$STAGE/baselines"
for d in dtd_121_ps_synths dtd_121_gatys; do
  [ -d "$DTD/$d" ] && cp -r "$DTD/$d" "$STAGE/baselines/" && echo "   $d" || echo "   (absent: $d)"
done

echo "== muNCA syntheses =="
mkdir -p "$STAGE/munca_synths"
for d in fig2_8k_128 fig2_8k_256; do
  [ -d "$STX/rebuttal/munca_synths/$d" ] && cp -r "$STX/rebuttal/munca_synths/$d" "$STAGE/munca_synths/"
done
echo "   $(ls "$STAGE/munca_synths" | tr '\n' ' ')"

echo "== video sources =="
mkdir -p "$STAGE/video"
# gstn_hm50_raw / dynca_synths_raw hold the per-frame PNGs Figure 8's x-t slices read
for d in targets gstn_synths dynca_synths gstn_hm50_raw dynca_synths_raw; do
  [ -d "$STX/dynamic_eval_videos/$d" ] && cp -r "$STX/dynamic_eval_videos/$d" "$STAGE/video/" && echo "   $d"
done

echo "== compressor checkpoints (stripped) =="
# The originals also carry Adam optimizer moments and the pyramid filter buffers.
# The filter buffers are rebuilt deterministically at model init, and the only
# tensors any figure reads are state_dict['_w'] and state_dict['w'] (fig16 and
# fig04 index them directly; nothing calls load_state_dict). Dropping the rest
# cuts the checkpoints from 574 MB to about 370 MB with no loss of function.
mkdir -p "$STAGE/checkpoints"
"${PYTHON:-/home/mer/.venv/bin/python}" - "$STX" "$STAGE" <<'PYSTRIP'
import sys, torch
from pathlib import Path
STX, STAGE = Path(sys.argv[1]), Path(sys.argv[2])
SRC = {100: STX/"models/final_model_checkpoint.pth"}
for n in (500, 1000):
    SRC[n] = STX/f"saved_outputs/run_flipflip_btch45_crp128_size_bns{n}_lr0.01_normTrue/final_model_checkpoint.pth"
for n, src in SRC.items():
    if not src.is_file():
        print(f"   (absent: bottleneck{n})"); continue
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    keep = {k: sd[k] for k in ("_w", "w") if k in sd}
    assert set(keep) == {"_w", "w"}, f"{src}: missing _w/w, got {sorted(sd)[:4]}"
    for k, v in keep.items():
        assert v.shape[0] == 28929, f"{src}: {k} has {v.shape[0]} rows, expected 28929"
        assert v.shape[1] == n,     f"{src}: {k} has {v.shape[1]} cols, expected {n}"
    out = STAGE/"checkpoints"/f"bottleneck{n}.pth"
    torch.save({"state_dict": keep}, out)
    print(f"   bottleneck{n}.pth  {src.stat().st_size/1e6:.0f} MB -> {out.stat().st_size/1e6:.0f} MB")
PYSTRIP

echo "== pruning backup/scratch subdirectories =="
# e.g. synths/seed_consistency_11/_pre_redo_backup -- superseded outputs kept during
# a re-run. They are not what any table reports and should not ship.
for d in $(find "$STAGE" -type d \( -name '_*backup*' -o -name '*_old*' -o -name '*_buggy*' \)); do
  echo "   removed $(realpath --relative-to="$STAGE" "$d")"; rm -rf "$d"
done

echo "== pruning stray metric files =="
# Drop loose FID files left inside synthesis run directories. These are partial
# artifacts from interrupted runs and they disagree with the published tables:
# video/gstn_hm50_raw/fids_raw_cuda1.csv held a single row, fire FID 186.4, where
# Table 7 reports 120. The published numbers come from the scripts in tables/,
# computed over all five videos; a one-row file from one GPU mid-run does not.
# Shipping it next to the code invites someone to read it as a result.
#
# Scoped deliberately NARROW. These are all kept, because scripts read them or
# they are complete run outputs rather than partial result tables:
#   per_stem_summary.csv    read by tables/table04 and table05
#   per_synth_metrics.csv   complete per-run output of seed_consistency_11
#   losscurve.csv           optimization traces, not results
#   synthesis_results.csv   the resume ledger synthesize.py reads
removed=0
while IFS= read -r f; do
  rm -f "$f"; removed=$((removed+1)); echo "   removed $(realpath --relative-to="$STAGE" "$f")"
done < <(find "$STAGE" -type f \( -name 'fids*.csv' -o -name 'marg_fids.csv' -o -name 'video_metrics.csv' \) )
echo "   $removed stray metric file(s) removed"

# Fail loudly if anything that looks like a results table is still in there.
if stray=$(find "$STAGE" -type f -name '*.csv' \
             ! -name 'losscurve.csv' ! -name 'synthesis_results.csv' \
             ! -name 'per_stem_summary.csv' ! -name 'per_synth_metrics.csv' ! -name 'final_losses.csv' ! -name 'per_synth_metrics_full9.csv' \
             -print -quit) && [ -n "$stray" ]; then
  echo "WARNING: unexpected CSV still in the bundle: $stray" >&2
  echo "         check it is not a results table before publishing." >&2
fi

echo
echo "== bundle =="
du -sh "$STAGE"
du -sh "$STAGE"/* | sort -rh
cd "$(dirname "$STAGE")"
tar czf gstn-assets-v1.tar.gz -C "$STAGE" .
sha256sum gstn-assets-v1.tar.gz | tee gstn-assets-v1.tar.gz.sha256
ls -lh gstn-assets-v1.tar.gz
