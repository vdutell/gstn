# GramStatTexNet (GSTN)

Texture analysis-by-synthesis from a steerable pyramid and pairwise correlations
between its sub-bands.

**[Paper](docs/static/gstn_neurips2026.pdf)** &nbsp;·&nbsp;
**[Project page](https://vdutell.github.io/gstn)** &nbsp;·&nbsp;
**[Reproduction guide](REPRODUCE.md)** &nbsp;·&nbsp;
**[Data](ASSETS.md)**

*GramStatTexNet: Efficient, Interpretable, and Neuro-Inspired Texture
Analysis-by-Synthesis.* NeurIPS 2026 (spotlight).

GSTN represents a 256×256 colour texture in 28,929 statistics. Every statistic
names the two filters it correlates, so the representation can be decomposed by
statistic family and asked which parts of a texture each family carries. Swapping
the spatial filter bank for a bank of 3D spatiotemporal Gabor filters extends the
same pipeline to video with no other change.

## Installation

```bash
git clone https://github.com/vdutell/gstn.git && cd gstn
pip install -r requirements.txt
```

A GPU build of PyTorch is needed to run synthesis. The figure and table scripts
that read cached metrics run on CPU.

## Usage

Synthesize a texture from an image, configured through environment variables:

```bash
BOUNDARY_MODE=black ENCODER_MODE=mean_only PRODIGY_LR=0.02 \
  python scripts/synthesize.py
```

Encode an image to its 28,929 statistics:

```python
import torch
import utils.stattexgramnet as gstn

model = gstn.GramStatTexNet(
    img_size=(256, 256), color=True, oris=4, edge_levels=6, edge_start=1,
    batch_size=1, compress=False, boundary_mode="black", device="cuda:0",
).to("cuda:0")

stats = model(image)          # (B, 28929)
families = model.stat_groups_gram_label   # family label per statistic
```

Ablate or deplete the representation by family or by learned importance:

```bash
DROP_FAMILY=sub_Xcolor python scripts/synthesize.py   # drop one family
KEEP_FAMILIES=sub_Xori,sub_Xlevel python scripts/synthesize.py
TOP_N_STATS=1000 python scripts/synthesize.py         # keep the top 1000
```

At most one of `TOP_N_STATS`, `TOP_RAND_STATS`, `COMPRESSED`, `DROP_FAMILY`,
`DROP_FAMILIES`, `KEEP_FAMILIES` may be set. Runs resume from
`synthesis_results.csv`, skipping rows marked `success`.

For video, build the spatiotemporal filter bank once, then synthesize:

```bash
python scripts/build_video_filterbank.py --minimal
python scripts/video_synthesize.py
```

## Reproducing the paper

```bash
python tables/table01_families.py      # no data, no GPU
python figures/fig03_pareto.py         # metric CSVs are in git
bash data/fetch_assets.sh              # 372 MB, for everything else
python figures/fig02_qualitative.py
```

Outputs go to `build/figures/` and `build/tables/`. Run scripts from the
repository root. See **[REPRODUCE.md](REPRODUCE.md)** for the per-figure table of
scripts, requirements and status, and **[ASSETS.md](ASSETS.md)** for what data
lives where and how each baseline was run.

## Repository layout

```
utils/                   the model
  stattexgramnet.py        image model
  stattexgramnetvideo.py   video model
  BNPyramids.py            steerable pyramid
  spyramid.py              pyramid internals
  poolingregions.py        spatial pooling, for foveated synthesis
gstn_paths.py            all paths, each overridable by environment variable
figures/                 one script per paper figure
  static/                  figures this repo does not regenerate, with provenance
tables/                  one script per paper table
scripts/                 synthesis, compressor training, metrics, filter bank
data/results/            per-image metric CSVs, so most figures need no download
data/splits/             the 141- and 108-stem evaluation sets
docs/                    the project page
tests/                   smoke tests
```

## Model configuration

The published configuration is colour, 4 orientations, 6 edge levels starting at
1, no phase difference, magnitudes kept:

- **240 pyramid channels** = 216 sub-band (4 orientations × 6 levels × 3 colours ×
  3 phase types) + 24 non-sub-band (3 highpass + 21 lowpass)
- **28,920 Gram entries** = C(240,2) + 240
- **+ 9 marginals** = **28,929 statistics**, in **14 families**

`python tables/table01_families.py` prints the breakdown and checks the total.

Two settings carry the published results:

| Variable | Value | Effect |
|---|---|---|
| `ENCODER_MODE` | `mean_only` (default) | Subtracts the image mean. Beat `z_score` on every metric: DISTS −11%, LPIPS −10%, SSIM +35%, MS-SSIM +74%. |
| `BOUNDARY_MODE` | `wrap` or `black` | `black` is zero-padding, labelled `zpad` in the figures. `mean_only` + `black` is the headline configuration. |

## Tests

```bash
python tests/test_smoke.py      # or: python -m pytest tests/ -q
```

Checks that the model builds and produces 28,929 statistics in 14 families
matching Table 1. No dataset, asset bundle or GPU required.

## Citation

```bibtex
@inproceedings{gstn2026,
  title     = {GramStatTexNet: Efficient, Interpretable, and Neuro-Inspired
               Texture Analysis-by-Synthesis},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE). Files in `utils/` marked as such are vendored from
the PooledStatisticsMetamers project and carry their own MIT headers. Baseline
methods are not redistributed as code; [ASSETS.md](ASSETS.md) records each source
and the configuration used.
