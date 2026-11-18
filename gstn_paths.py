"""Central path configuration for the GSTN release.

Every script in this repository imports its paths from here instead of hardcoding
them. Nothing below is baked in: each location can be overridden with an environment
variable, so you can keep the large asset bundle on a different disk from the code.

    GSTN_DTD        the DTD dataset root (not redistributed; see docs/ASSETS.md)
    GSTN_ASSETS     the downloaded asset bundle (synth images, per-image metrics,
                    checkpoints, video sources)
    GSTN_FIGURES    where figure scripts write
    GSTN_TABLES     where table scripts write
    GSTN_I3D        I3D TorchScript weights, for the video FVD metric
    GSTN_FVMD       a clone of the FVMD repository, for the video FVMD metric

Run scripts from the repository root, so that `import utils.stattexgramnet` resolves.

The asset bundle deliberately mirrors the directory layout of the original working
tree (`synths/`, `synth_metrics_out/`, `pixelbp_prodigy_targets/`, ...). That is why
the figure scripts can reference it with their original relative paths: it keeps this
published code a faithful copy of the code that produced the published numbers,
rather than a reorganization of it.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


# --- inputs ---------------------------------------------------------------
DTD    = _env("GSTN_DTD",    REPO / "data" / "dtd")
ASSETS = _env("GSTN_ASSETS", REPO / "data" / "assets")

DTD_IMAGES = DTD / "images"
DTD_LABELS = DTD / "labels"

SYNTHS      = ASSETS / "synths"               # GSTN + baseline syntheses
TARGETS     = ASSETS / "pixelbp_prodigy_targets"
# per_image_metrics_*_clean.csv -- small enough to keep in git, so every
# quantitative figure works from a bare clone with no download
METRICS     = REPO / "data" / "results" / "synth_metrics_out"
CHECKPOINTS = ASSETS / "checkpoints"          # compressor checkpoints (100/500/1000-dim)
MUNCA       = ASSETS / "munca_synths"
# Baseline syntheses (P&S, Gatys) that we redistribute. These are outputs we
# produced, not part of the DTD dataset, which is why they are not under DTD.
BASELINES   = ASSETS / "baselines"
VIDEO       = ASSETS / "video"                # targets/, gstn_synths/, dynca_synths/
FILTERBANK  = ASSETS / "filter_banks_64px_20fr" / "filters_torch.pkl"

# --- small results kept in git -------------------------------------------
RESULTS = REPO / "data" / "results"

# --- outputs --------------------------------------------------------------
FIGURES = _env("GSTN_FIGURES", REPO / "build" / "figures")
TABLES  = _env("GSTN_TABLES",  REPO / "build" / "tables")
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# --- optional external tools (video metrics only) -------------------------
I3D_MODEL = _env("GSTN_I3D",  ASSETS / "fvd_models" / "i3d_torchscript.pt")
FVMD_REPO = _env("GSTN_FVMD", REPO.parent / "FVMD")


def require(path: Path, what: str = "") -> Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    p = Path(path)
    if p.exists():
        return p
    hint = f" ({what})" if what else ""
    raise FileNotFoundError(
        f"Missing{hint}: {p}\n"
        f"If this is part of the asset bundle, fetch it with:\n"
        f"    bash data/fetch_assets.sh\n"
        f"If it is the DTD dataset, see docs/ASSETS.md and set GSTN_DTD.\n"
        f"Current settings: GSTN_ASSETS={ASSETS}  GSTN_DTD={DTD}"
    )


__all__ = [
    "REPO", "DTD", "DTD_IMAGES", "DTD_LABELS", "ASSETS", "SYNTHS", "TARGETS",
    "METRICS", "CHECKPOINTS", "MUNCA", "BASELINES", "VIDEO", "FILTERBANK", "RESULTS",
    "FIGURES", "TABLES", "I3D_MODEL", "FVMD_REPO", "require",
]
