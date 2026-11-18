"""Smoke tests: import the model, check the statistic count, check the config.

    python -m pytest tests/ -q        (or just: python tests/test_smoke.py)

These need no dataset, no asset bundle and no GPU. They are here to catch the
failure mode that matters most for a release: the repository is checked out and
the model does not even build, or builds with the wrong number of statistics so
that no checkpoint will load.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_STATS = 28929
EXPECTED_FAMILIES = 14
EXPECTED_CHANNELS = 240
# Table 1, as published
EXPECTED_COUNTS = {
    "marginal": 9, "highpass": 6, "lowpass": 231, "pass_multi": 5247,
    "sub_Xori": 324, "sub_Xcolor": 432, "sub_Xlevel": 540,
    "sub_Xri_eq": 72, "sub_Xrm_eq": 72, "sub_Xim_eq": 72,
    "sub_real_Xmulti": 2196, "sub_imag_Xmulti": 2196, "sub_magn_Xmulti": 2196,
    "sub_Xmulti": 15336,
}


def _model():
    import utils.stattexgramnet as stn
    return stn.StatTexGramNet(
        img_size=(256, 256), color=True, oris=4, edge_levels=6, edge_start=1,
        mag_only=False, include_dphase=False, batch_size=1, bottleneck_size=None,
        device="cpu", compress=False, use_pooling=False)


def test_paths_import():
    import gstn_paths
    assert gstn_paths.REPO.is_dir()
    for name in ("DTD", "ASSETS", "FIGURES", "TABLES", "METRICS", "RESULTS"):
        assert hasattr(gstn_paths, name), f"gstn_paths missing {name}"


def test_metrics_shipped_in_git():
    """The quantitative figures must work from a bare clone."""
    import gstn_paths
    assert gstn_paths.METRICS.is_dir(), f"missing {gstn_paths.METRICS}"
    assert any(gstn_paths.METRICS.rglob("*_clean.csv")), "no per-image metric CSVs"


def test_model_statistic_count():
    m = _model()
    assert m.npyrs == EXPECTED_CHANNELS, f"npyrs {m.npyrs}"
    assert m.gram_size == 28920, f"gram_size {m.gram_size}"
    assert m.marginal_size == 9, (
        f"marginal_size {m.marginal_size}; the colour path must multiply by 3, "
        "otherwise stats_size is 28,923 and no checkpoint loads")
    assert m.stats_size == EXPECTED_STATS, f"stats_size {m.stats_size}"


def test_family_counts_match_table1():
    import collections
    m = _model()
    counts = dict(collections.Counter(m.stat_groups_gram_label))
    assert len(counts) == EXPECTED_FAMILIES, f"{len(counts)} families"
    assert counts == EXPECTED_COUNTS, (
        "family counts differ from Table 1: "
        f"{ {k: (v, EXPECTED_COUNTS.get(k)) for k, v in counts.items() if EXPECTED_COUNTS.get(k) != v} }")
    assert sum(counts.values()) == EXPECTED_STATS


def test_video_filterbank_geometry():
    """404 responses per colour channel, so 1,212 total and 735,078 Gram stats."""
    from math import comb
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import build_video_filterbank as B
    n_ori = len(B.SCALES) * len(B.ORIENTATIONS) * len(B.HEIGHTS)
    assert n_ori == 200, n_ori
    per_colour = n_ori * 2 + 4
    assert per_colour == 404, per_colour
    total_filters = per_colour * 3
    assert total_filters == 1212
    assert comb(total_filters, 2) + total_filters == 735078


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except Exception as e:
                fails += 1; print(f"FAIL  {name}\n        {e}")
    print(f"\n{'all passed' if not fails else f'{fails} failed'}")
    raise SystemExit(1 if fails else 0)
