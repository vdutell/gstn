# Reproducing the figures and tables

Every script here runs from the **repository root**:

```bash
python figures/fig03_pareto.py
```

Outputs land in `build/figures/` and `build/tables/`. Nothing is written back into
the inputs. Override locations with `GSTN_FIGURES` / `GSTN_TABLES`.

## What you need, by column

- **git only** — works from a bare clone. The per-image metric CSVs live in
  `data/results/`, so every curve and bar figure works with no download.
- **assets** — needs the bundle: `bash data/fetch_assets.sh` (372 MB). See
  [ASSETS.md](ASSETS.md).
- **DTD** — needs the Describable Textures Dataset. Set `GSTN_DTD`.
- **GPU** — re-runs synthesis or recomputes perceptual metrics on a GPU.

## Status column

"verified" means the script was executed in a clean checkout while preparing this
release and its output compared against the published figure. Ten figures came out
**pixel-identical** to the camera-ready versions; the two per-category tables came
out **byte-identical** CSVs. "runs" means it executed cleanly but there was no
published raster to diff against. "not executed" means exactly that: the code is
here and its inputs are documented, but it was not run during packaging, so treat
it as untested.

## Figures

| Paper | Script | Needs | Status |
|---|---|---|---|
| Figure 1 | `figures/static/` | — | static, see [PROVENANCE](figures/static/PROVENANCE.md) |
| Figure 2 | `figures/fig02_qualitative.py` | assets | **verified, pixel-identical** |
| Figure 3 | `figures/fig03_pareto.py` | git only | **verified, pixel-identical** |
| Figure 4 | `figures/fig04_fingerprints.py` | assets, GPU | not executed |
| Figure 5 | `figures/fig05_ablation.py` | assets | **verified, pixel-identical** |
| Figure 6 | `figures/fig06_depletion.py` | assets | runs |
| Figure 7 | `figures/static/` | — | static; see the pooling warning in PROVENANCE |
| Figure 8 (left) | `figures/fig08a_video_xt.py` | assets | **verified, pixel-identical** |
| Figure 8 (right) | `figures/fig08b_video_frames.py` | assets | **verified, pixel-identical** |
| Figure 9 | `figures/fig09_optimizers.py` | assets, DTD, GPU | not executed; runs L-BFGS synthesis on demand |
| Figure 10 | `figures/fig10_percategory_vs_ps.py` | git only | **verified, byte-identical CSV** |
| Figure 11 | `figures/fig11_percategory_vs_gatys.py` | git only | **verified, byte-identical CSV** |
| Figures 12, 13 | `figures/fig12_13_seed_and_resolution.py` | assets | **verified, pixel-identical** |
| Figure 14 | `figures/fig14_resolution_grid.py` | assets, DTD | **verified, pixel-identical** |
| Figure 15 | `figures/static/` | — | static |
| Figure 16 | `figures/fig16_family_weight_mass.py` | assets | runs |
| Figure 17 | `figures/static/` | — | static |
| Figure 18 | `figures/static/` | — | static; the bank it depicts **is** rebuildable, below |
| Figure 19 | `figures/fig19_dtd141_all.py` | assets | runs |
| Appendix grids | `figures/fig_extended_qualitative.py` | assets | **verified, pixel-identical** |

## Tables

| Paper | Script | Needs | Status |
|---|---|---|---|
| Table 1 | `tables/table01_families.py` | nothing | **verified**, reproduces all 14 family counts and the 28,929 total |
| Table 2 | `tables/table02_seed_sensitivity.py` | assets, GPU | not executed |
| Table 3 | `tables/table03_comparative_quality.py` | assets, DTD, GPU | not executed |
| Table 4 | `tables/table04_efficiency.py` | git only | runs |
| Table 5 | `tables/table05_resolution_scaling.py` | assets | runs |
| Table 6 | — | — | **not reproducible from this code**, see below |
| Table 7 | `tables/table07_video_metrics.py` | assets, GPU | not executed |

### Table 6 is not reproducible from this code

Table 6 groups the 735,078 spatiotemporal statistics into families by which of five
filter attributes vary. Two of its entries check out exactly: the total is
C(1212,2) + 1212 = 735,078, and AUTO = 1,212 equals the filter-response count.

The per-family breakdown does not, because the machinery that would produce it is
commented out. In `utils/stattexgramnetvideo.py`, both `stat_names_gram_label` and
the `assign_gram_groups` call are disabled (around lines 97–111), so the video model
computes no family labels at all. `assign_gram_groups` exists but is never called
and has nothing to operate on.

Reconstructing the convention analytically did not land on the published numbers
either. Table 6 reports LPS = LPT = 7,233. Counting unordered pairs involving the
six spatial-lowpass responses gives 7,257 including the diagonal and 7,251 without
it; excluding LPS×LPT cross terms gives 7,215. None is 7,233. The gap is small and
most likely a convention in the original computation that is not recorded anywhere
in the code, but we could not recover it, so rather than ship a script whose numbers
disagree with the paper we are flagging it here. If you know the intended
convention, this is a twenty-line script.

## Rebuilding the video filter bank

The video model loads a precomputed bank of spatiotemporal Gabor filters. It is
about 7 GB, so it is **generated locally** rather than distributed:

```bash
python scripts/build_video_filterbank.py            # faithful, ~7 GB, a few minutes
python scripts/build_video_filterbank.py --minimal  # ~4.2 GB, 21 s
```

Generation is deterministic, with no RNG. The extracted generator was verified
against the original bank: all 200 oriented filter keys present, all 400 oriented
tensors and all four lowpass tensors agreeing to a worst relative error of 8e-16,
which is float64 round-off.

`--minimal` writes only the `fourier_pad` branch. That is the only branch
`utils/stpyramidbuilder.py` reads, so the model behaves identically, but the file is
then not a byte-for-byte match for the original, which is why the faithful build is
the default.

## Re-running the heavy work

The cached syntheses in the asset bundle came from these. All are GPU jobs; the full
141-image sweeps are hours to days.

```bash
# image synthesis -- configured entirely through environment variables
BOUNDARY_MODE=black ENCODER_MODE=mean_only PRODIGY_LR=0.02 \
  python scripts/synthesize.py

# one family ablation condition
DROP_FAMILY=sub_Xcolor python scripts/synthesize.py

# depletion by learned importance
TOP_N_STATS=1000 python scripts/synthesize.py

# the five-seed study behind Table 2 and Figure 12
python scripts/seed_consistency_11.py

# contrastive compressor (produces the checkpoints Figures 4 and 16 read)
BOTTLENECK_SIZE=100 python scripts/train_compressor.py

# per-image metric CSVs that the quantitative figures read
python scripts/recompute_metrics_clean.py
python scripts/recompute_depletion_clean.py

# video
python scripts/build_video_filterbank.py
python scripts/video_synthesize.py
python tables/table07_video_metrics.py
```

At most one of `TOP_N_STATS`, `TOP_RAND_STATS`, `COMPRESSED`, `DROP_FAMILY`,
`DROP_FAMILIES`, `KEEP_FAMILIES` may be set; `synthesize.py` exits if more are. With
none set it runs the full representation. Runs resume by reading
`synthesis_results.csv` and skipping rows marked `success`, so re-invoking a finished
condition is cheap.

## Naming

The code calls the model `StatTexGramNet` / `STGN`; the paper calls it
`GramStatTexNet` / `GSTN`. They are the same model. Figure labels use `GSTN-wrap`
and `GSTN-zpad` for the two boundary modes (`BOUNDARY_MODE=wrap` and `black`).
