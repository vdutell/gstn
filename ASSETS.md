# Data and assets

Three tiers, by size.

## 1. In this repository (no download)

`data/results/` holds the per-image metric CSVs (`synth_metrics_out/`, 39
condition directories) and the summary CSVs the quantitative figures read. About
4 MB. Because these are in git, **every curve and bar figure works from a bare
clone**: Figures 3, 10 and 11 and Table 4 need nothing else.

`data/splits/` holds the evaluation split lists:

- `val1_3sampled.txt` — 141 stems, 3 per DTD category. The standard evaluation set.
- `bench_intersection_108.txt` — 108 stems, the subset where the P&S, Gatys and
  Gaussian baselines all have outputs. Older Pareto plots use this.

## 2. The asset bundle (372 MB download, 409 MB unpacked)

```bash
bash data/fetch_assets.sh
```

Or, if you have it already, skip the download: `export GSTN_ASSETS=/path/to/assets`.

| Directory | Size | Contents |
|---|---|---|
| `checkpoints/` | 177 MB | Contrastive compressor weights at bottleneck 100, 500 and 1000 |
| `synths/` | 142 MB | GSTN and baseline syntheses: full, 28 family-ablation conditions, 12 depletion conditions, seeds, high-resolution |
| `video/` | 38 MB | DynTex targets, GSTN and DyNCA video syntheses, and the per-frame PNGs Figure 8 reads |
| `baselines/` | 28 MB | P&S and Gatys syntheses |
| `pixelbp_prodigy_targets/` | 18 MB | The 145 target images any figure references, 256×256 |
| `munca_synths/` | 9 MB | µNCA syntheses at 128 and 256 px |

Three size decisions worth knowing about, since they make the bundle smaller than
the working tree it came from:

- **Targets** are pruned from 5,640 files (660 MB, all of DTD) to the 145 stems
  some figure actually references.
- **Depletion directories** are pruned to the four categories Figure 6 displays
  (banded, cracked, polka-dotted, bumpy), 12 files each instead of 141.
- **Checkpoints** are stripped from 574 MB to 177 MB. The originals also carried
  Adam optimizer moments and the steerable-pyramid filter buffers; the buffers are
  rebuilt deterministically at model init, and the only tensors any figure reads
  are `state_dict["_w"]` and `state_dict["w"]`, which Figures 4 and 16 index
  directly. Nothing calls `load_state_dict`, so nothing is lost.

`data/make_asset_bundle.sh` is the maintainer script that assembles the bundle. It
is included so the contents are auditable rather than a hand-made tarball.

## 3. Fetched separately

**DTD (Describable Textures Dataset).** Not redistributed. Download from
<https://www.robots.ox.ac.uk/~vgg/data/dtd/> and point at it:

```bash
export GSTN_DTD=/path/to/dtd     # expects images/ and labels/ underneath
```

Needed by Figures 9 and 14 and Table 3, and by any new synthesis run.

**The video filter bank.** About 7 GB, so it is generated rather than downloaded:
`python scripts/build_video_filterbank.py`. See [REPRODUCE.md](REPRODUCE.md).

**I3D weights**, only for the FVD column of Table 7: set `GSTN_I3D` to an
`i3d_torchscript.pt`. **FVMD**, only for the FVMD column: clone
<https://github.com/ljh0v0/FMD-frechet-motion-distance> and set `GSTN_FVMD`.

## Baselines

Per the paper's comparisons. We redistribute the **outputs** of these methods in
the bundle, not their code; each entry below is where the implementation came from
and how it was run.

| Method | Source | Configuration |
|---|---|---|
| Portilla & Simoncelli | <https://www.cns.nyu.edu/pub/eero/colorTextureSynth.tgz> | MATLAB reference implementation, defaults |
| Gatys et al. | VGG19 Gram-matrix texture loss | Reimplemented; see `deeptextures/` lineage in the paper's appendix |
| Ustyuzhaninov et al. | multiscale random-feature texture synthesis | PyTorch port, multiscale configuration |
| µNCA | `google-research/self-organising-systems`, `notebooks/μNCA_jax.ipynb` | 588 learned parameters (4×144 + 12), STEPS=8000, run at both 128 and 256 px |
| DyNCA | <https://github.com/IVRL/DyNCA> | `fit_video_motion.py --img_size 64 64 --motion_img_size 64 64 --iterations 5000` |

Two details that matter for reading the numbers:

**µNCA resolution.** Table 3 reports the 256 px run; Figure 2 shows the 128 px run,
because the two look near-identical except on `swirly` and dropping a row kept the
figure on the page. One µNCA synthesis, `polka-dotted_0140`, collapsed to a constant
image. It is **included** in the reported means deliberately: excluding a method's
failures inflates its average.

**DyNCA frame matching.** DyNCA is a cellular automaton that rolls out indefinitely;
its native output is 400 frames at 25 fps, a 16-second window, against GSTN's
20 frames at 15 fps. Comparing those directly is not like for like, because spread
over 16 seconds consecutive DyNCA frames are nearly independent, which aliases the
temporal spectrum and inflates frame diversity. Every DyNCA video is therefore
resampled onto the target's own temporal grid:

```python
idx = [min(len(frames) - 1, round(k * 25.0 / 15.0)) for k in range(20)]
```

So the DyNCA frames in Figure 8 and Table 7 are exactly the frames the metrics were
computed on. The alternative, taking the first 20 frames at face value, inflated
fence FID by 47% (118.5 against 173.9) in DyNCA's favour.
