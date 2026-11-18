# GramStatTexNet (GSTN)

Code for **GramStatTexNet: Efficient, Interpretable, and Neuro-Inspired Texture
Analysis-by-Synthesis** (NeurIPS 2026, spotlight).

GSTN is a texture analysis-by-synthesis model built from a steerable pyramid and
pairwise correlations between its sub-bands. It represents a 256×256 colour texture
in **28,929 statistics**, every one of which names the two filters it correlates, so
the representation can be decomposed by *statistic family* and asked which parts of
a texture each family carries. Swapping the spatial filter bank for a bank of 3D
spatiotemporal Gabor filters extends the same pipeline to video with no other change.

Project page: <https://vdutell.github.io/gstn>

## Quick start

```bash
git clone <this repo> && cd gstn
pip install -r requirements.txt

# Table 1, derived from the model itself. No data, no GPU, about a minute.
python tables/table01_families.py

# Figures 3, 10, 11 and Table 4. Also no download: the metric CSVs are in git.
python figures/fig03_pareto.py
python figures/fig10_percategory_vs_ps.py
python tables/table04_efficiency.py
```

Everything else needs the asset bundle (372 MB):

```bash
bash data/fetch_assets.sh
python figures/fig02_qualitative.py      # the qualitative comparison grid
python figures/fig08a_video_xt.py        # video space-time slices
```

Outputs go to `build/figures/` and `build/tables/`. Run scripts from the repository
root.

## How faithful is this?

Ten of the rebuilt figures are **pixel-identical** to the camera-ready versions, and
both per-category tables come out as **byte-identical** CSVs. Table 1 reproduces all
fourteen family counts and the 28,929 total from the model's own labels.
[REPRODUCE.md](REPRODUCE.md) gives per-figure status, and says plainly
which scripts were not executed during packaging and what Table 6 does not
reproduce.

## Layout

```
utils/            the model. Package layout preserved from the original working
                  tree rather than refactored, so this is the code that produced
                  the published numbers.
  stattexgramnet.py       image model (StatTexGramNet)
  stattexgramnetvideo.py  video model
  BNPyramids.py, spyramid.py, ...  steerable pyramid
  poolingregions.py, gazewarp.py   spatial pooling (peripheral-vision branch)
gstn_paths.py     every path in one place, all overridable by environment variable
figures/          one script per paper figure, named for it
  static/         the five figures this repo does not regenerate, with provenance
tables/           one script per paper table
scripts/          synthesis drivers, compressor training, metrics, filter bank
data/results/     per-image metric CSVs, in git, so quantitative figures need no download
data/splits/      the 141- and 108-stem evaluation sets
docs/             the project page served at vdutell.github.io/gstn
REPRODUCE.md      per-figure reproduction status and commands
ASSETS.md         what data lives where, and the baseline sources
```

## Model configuration

The published configuration is colour, 4 orientations, 6 edge levels starting at 1,
no phase difference, magnitudes kept:

- **240 pyramid channels** = 216 sub-band (4 orientations × 6 levels × 3 colours ×
  3 phase types) + 24 non-sub-band (3 highpass + 21 lowpass)
- **28,920 Gram entries** = C(240,2) + 240
- **+ 9 marginals** = **28,929 statistics**, grouped into **14 families**

`python tables/table01_families.py` prints the breakdown and checks the total.

Two configuration choices carry the published results and are easy to get wrong:

- `ENCODER_MODE=mean_only` (the default). Subtracts the image mean. This beat
  `z_score` on every metric: DISTS −11%, LPIPS −10%, SSIM +35%, MS-SSIM +74%. An
  earlier sigmoid preprocessing was removed in favour of it.
- `BOUNDARY_MODE` is `wrap` or `black` (zero-pad, labelled "zpad" in the figures).
  `mean_only` + `black` is the headline configuration.

## Known limitations

**Desaturation.** GSTN mildly desaturates chromatic targets; `pleated_0136` is the
standing example. The cause is the encoder's global normalization: marginals computed
on a normalized image cannot constrain absolute saturation, and cross-channel Gram
entries capture only *relative* chroma. Three fixes were attempted and all failed.
Disclosed as a limitation in the paper.

**The pooled branch has a bug.** In `calcgramm`, the pooled path indexes `[0]` and
silently keeps only batch element 0. Every published result used
`use_pooling=False`, so nothing in the paper is affected, but fix that indexing
before enabling pooling. This is why Figure 7 ships static.

## Naming

The code says `StatTexGramNet` / `STGN`; the paper says `GramStatTexNet` / `GSTN`.
Same model. The code's names were kept so that this is a faithful copy rather than a
rename that might have changed behaviour.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE). Files in `utils/` marked as such are vendored from the
PooledStatisticsMetamers project and carry their own MIT headers. Baseline methods
are not redistributed as code; [ASSETS.md](ASSETS.md) records each source
and the exact configuration used.
