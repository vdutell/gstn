#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Produces: the spatiotemporal filter bank the video model loads
#           (data/assets/filter_banks_64px_20fr/filters_torch.pkl)
#
# Provenance: extracted from MoTexNet/SpatiotemporalMochinutFilters.ipynb in the
# original working tree (code cells 0, 10, 15 and 16, in that order). The notebook
# is exploratory and most of its cells only plot; this script is the subset that
# actually constructs and saves the bank, with the cell-to-cell globals made
# explicit. Figure 18 in the paper visualizes this bank.
#
# Why this exists: the generated bank is ~7 GB, too large to distribute, so it is
# rebuilt locally rather than downloaded. Generation is deterministic -- no RNG --
# so every machine gets the same bank.
# ---------------------------------------------------------------------------
"""Build the spatiotemporal Gabor filter bank used by the GSTN video model.

    python scripts/build_video_filterbank.py                 # faithful, ~7 GB
    python scripts/build_video_filterbank.py --minimal        # ~4.2 GB, see below

The bank is a nested dict with four top-level branches: `fourier`,
`spatiotemporal`, `fourier_pad` and `spatiotemporal_pad`. Only `fourier_pad` is
ever read by utils/stpyramidbuilder.py, which accesses exactly:

    filterbank['fourier_pad']['ori']['re'][key]        200 oriented filters
    filterbank['fourier_pad']['ori']['im'][key]        200 oriented filters
    filterbank['fourier_pad']['lowpass_spatial']['re']
    filterbank['fourier_pad']['lowpass_spatial']['im']
    filterbank['fourier_pad']['lowpass_temporal']['re']
    filterbank['fourier_pad']['lowpass_temporal']['im']

giving 400 + 4 = 404 filter responses per colour channel, so 1,212 across RGB and
C(1212,2) + 1212 = 735,078 Gram statistics. That is the Table 6 total.

`--minimal` writes only the `fourier_pad` branch. The model behaves identically
because nothing reads the other three, but the file is no longer a byte-for-byte
match for the original, so the default is the faithful build.

Filter keys are '{scale}-{orientation_in_degrees}-{height}'.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gstn_paths import FILTERBANK  # noqa: E402

PI = np.pi

# --- bank geometry (cell 15's values, which override cell 0's wider sweep) ----
SCALES = [1, 2, 4, 8, 16]
ORIENTATIONS = [0, PI / 4, PI / 2, 3 * PI / 4, PI, 5 * PI / 4, 3 * PI / 2, 7 * PI / 4]
HEIGHTS = [1, 2, 4, 8, 16]

# `norm` is the notebook's own flag: it is what distinguishes the bank directories
# named filter_banks_normFalse_* from filter_banks_normTrue_*. Every published
# video result used norm=False, so that is the default here. Passing --norm applies
# Parseval normalization to each filter instead, which is NOT what the paper used.
NORM_DEFAULT = False

# --- sampling grid (cell 0) ---------------------------------------------------
FILT_SIZE_XY = 25
FILT_SIZE_T = 12
NUM_PIXELS = 64
NUM_FRAMES = 20

XYSPACE = np.linspace(-FILT_SIZE_XY, FILT_SIZE_XY, num=NUM_PIXELS)
TSPACE = np.linspace(-FILT_SIZE_T, FILT_SIZE_T, num=NUM_FRAMES)


# --- cell 0 -------------------------------------------------------------------
def gauss_filter_3d(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, norm=False):
    return np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2)
                    + (y - mu_y) ** 2 / (2 * sigma_y ** 2)
                    + (z - mu_z) ** 2 / (2 * sigma_z ** 2)))


def pol2cart(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)


def radial_gaussian_3d(xs, ys, zs, scale, orientation, height,
                       scaling_factor=0.7, norm=False):
    mu_x, mu_y = pol2cart(scale, orientation)
    sigma_scale_space, sigma_scale_time = scaling_factor, 0.4
    filt_sigma_xy = scale * sigma_scale_space
    filt_sigma_z = height * sigma_scale_time
    return gauss_filter_3d(xs, ys, zs, mu_x, mu_y, height,
                           filt_sigma_xy, filt_sigma_xy, filt_sigma_z, norm=norm)


def quadrature_gaussian_filter_3d(xs, ys, zs, scale, orientation, height,
                                  scaling_factor=0.7, norm=False):
    """Quadrature pair, as a single complex array. The imaginary member is the
    antipodal lobe: orientation + pi and -height."""
    filter_real = radial_gaussian_3d(xs, ys, zs, scale, orientation, height,
                                     scaling_factor=scaling_factor, norm=norm)
    filter_imag = radial_gaussian_3d(xs, ys, zs, scale, orientation + PI, -height,
                                     scaling_factor=scaling_factor, norm=norm)
    return filter_real + 1j * filter_imag


def normalize_filter(filt):
    """Parseval normalization: power is conserved between the spatial and Fourier
    domains, so normalizing in Fourier and inverse-transforming later is valid."""
    return filt / np.sqrt(np.sum(np.abs(filt) ** 2))


# --- cell 10 ------------------------------------------------------------------
def inverse_fourier_transform(filt):
    filt = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(filt)))
    return np.real(filt), np.imag(filt)


# --- cell 15 ------------------------------------------------------------------
def format_filter(fourier_filter, norm):
    fourier_filter_re = np.real(fourier_filter).astype(np.float32)
    fourier_filter_im = np.imag(fourier_filter).astype(np.float32)
    st_filter_re, st_filter_im = inverse_fourier_transform(np.real(fourier_filter))

    # standard shape: frames, height, width
    fourier_filter_re = fourier_filter_re.transpose((2, 0, 1))
    fourier_filter_im = fourier_filter_im.transpose((2, 0, 1))
    st_filter_re = st_filter_re.transpose((2, 0, 1))
    st_filter_im = st_filter_im.transpose((2, 0, 1))

    if norm:
        fourier_filter_re = normalize_filter(fourier_filter_re)
        fourier_filter_im = normalize_filter(fourier_filter_im)
        st_filter_re = normalize_filter(st_filter_re)
        st_filter_im = normalize_filter(st_filter_im)

    return fourier_filter_re, fourier_filter_im, st_filter_re, st_filter_im


def _pad_spatial(arr):
    pad_width = [(d // 2, d // 2) for d in arr.shape]
    return np.pad(arr, pad_width, mode="constant")


def _empty_bank(minimal: bool) -> dict:
    branches = ["fourier_pad"] if minimal else [
        "fourier", "spatiotemporal", "fourier_pad", "spatiotemporal_pad"]
    bank = {}
    for b in branches:
        bank[b] = {"ori": {"re": {}, "im": {}}}
    return bank


def build(minimal: bool = False, norm: bool = NORM_DEFAULT,
          verbose: bool = True) -> dict:
    xs, ys, zs = np.meshgrid(XYSPACE, XYSPACE, TSPACE)
    bank = _empty_bank(minimal)

    n = 0
    for scale in SCALES:
        for orientation in ORIENTATIONS:
            for height in HEIGHTS:
                ordeg = int(math.degrees(orientation))
                key = f"{scale}-{ordeg}-{height}"

                fourier_filter = quadrature_gaussian_filter_3d(
                    xs, ys, zs, scale, orientation, height, norm=norm)
                f_re, f_im, st_re, st_im = format_filter(fourier_filter, norm)

                if not minimal:
                    bank["fourier"]["ori"]["re"][key] = torch.from_numpy(f_re)
                    bank["fourier"]["ori"]["im"][key] = torch.from_numpy(f_im)
                    bank["spatiotemporal"]["ori"]["re"][key] = torch.from_numpy(st_re)
                    bank["spatiotemporal"]["ori"]["im"][key] = torch.from_numpy(st_im)

                st_re_pad = _pad_spatial(st_re)
                st_im_pad = _pad_spatial(st_im)
                f_re_pad = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(st_re_pad)))
                f_im_pad = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(st_im_pad)))

                bank["fourier_pad"]["ori"]["re"][key] = torch.from_numpy(f_re_pad)
                bank["fourier_pad"]["ori"]["im"][key] = torch.from_numpy(f_im_pad)
                if not minimal:
                    bank["spatiotemporal_pad"]["ori"]["re"][key] = torch.from_numpy(st_re_pad)
                    bank["spatiotemporal_pad"]["ori"]["im"][key] = torch.from_numpy(st_im_pad)

                n += 1
                if verbose and n % 25 == 0:
                    print(f"  {n}/{len(SCALES)*len(ORIENTATIONS)*len(HEIGHTS)} oriented filters",
                          flush=True)

    # --- cell 16: the four lowpass filters ---------------------------------
    x, y, z = np.meshgrid(XYSPACE, XYSPACE, TSPACE)

    # temporal lowpass: two lobes at mu_z = +/-1, wide in space, narrow in time
    lp_t = (gauss_filter_3d(x, y, z, 0, 0, 1,
                            FILT_SIZE_XY / 1.25, FILT_SIZE_XY / 1.25, 0.5, norm=False)
            + gauss_filter_3d(x, y, z, 0, 0, -1,
                              FILT_SIZE_XY / 1.25, FILT_SIZE_XY / 1.25, 0.5, norm=False))
    # spatial lowpass: narrow in space, two lobes far out in time
    tmax = max(TSPACE)
    lp_s = (gauss_filter_3d(x, y, z, 0, 0, tmax / 2, 1, 1, tmax / 4, norm=False)
            + gauss_filter_3d(x, y, z, 0, 0, -tmax / 2, 1, 1, tmax / 4, norm=False))

    for name, lp in (("lowpass_temporal", lp_t), ("lowpass_spatial", lp_s)):
        _, _, st_re, st_im = format_filter(lp, norm)
        st_re_pad, st_im_pad = _pad_spatial(st_re), _pad_spatial(st_im)
        # note: unlike the oriented filters, these are cast to real/imag parts
        f_re_pad = np.real(np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(st_re_pad))))
        f_im_pad = np.imag(np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(st_im_pad))))
        bank["fourier_pad"][name] = {"re": torch.from_numpy(f_re_pad),
                                     "im": torch.from_numpy(f_im_pad)}
        if not minimal:
            bank["spatiotemporal_pad"][name] = {"re": torch.from_numpy(st_re_pad),
                                                "im": torch.from_numpy(st_im_pad)}

    if verbose:
        n_ori = len(bank["fourier_pad"]["ori"]["re"])
        print(f"built {n_ori} oriented quadrature pairs + 2 lowpass pairs "
              f"= {n_ori * 2 + 4} responses per colour channel")
    return bank


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=FILTERBANK,
                    help=f"output .pkl (default: {FILTERBANK})")
    ap.add_argument("--minimal", action="store_true",
                    help="write only the fourier_pad branch the model reads (~4.2 GB)")
    ap.add_argument("--norm", action="store_true",
                    help="Parseval-normalize each filter (the notebook's normTrue "
                         "variant). NOT what the published results used.")
    args = ap.parse_args()

    print(f"building filter bank -> {args.out}")
    print(f"  scales={SCALES} orientations={len(ORIENTATIONS)} heights={HEIGHTS} "
          f"norm={args.norm} grid={NUM_PIXELS}px x {NUM_FRAMES}fr")
    if args.norm:
        print("  WARNING: --norm does not reproduce the published video results")
    if args.minimal:
        print("  --minimal: skipping the three branches nothing reads")

    bank = build(minimal=args.minimal, norm=args.norm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bank, f)
    size_gb = args.out.stat().st_size / 1e9
    print(f"wrote {args.out}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
