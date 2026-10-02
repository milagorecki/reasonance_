"""Tests for the reasoning-length x-axis variants in reasonance.utils."""

import numpy as np
import pandas as pd
import pytest
from reasonance.utils import LENGTH_VARIANTS, build_length_variants


@pytest.fixture
def token_matrix():
    """One run whose lengths vary a lot and one that writes the same amount for everyone."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"varying": rng.gamma(2, 300, 400), "flat": rng.normal(450, 8, 400)})


def test_build_length_variants_returns_one_series_per_variant(token_matrix):
    variants = build_length_variants(token_matrix)
    assert list(variants) == list(LENGTH_VARIANTS)
    for spec in variants.values():
        assert len(spec["series"]) == len(token_matrix)
        assert spec["label"]


def test_variant_scales(token_matrix):
    variants = build_length_variants(token_matrix)
    assert variants["raw"]["series"].min() > 1  # token counts
    assert abs(variants["zscore"]["series"].median()) < 1  # centred
    ranks = variants["rank"]["series"]
    assert 0 <= ranks.min() and ranks.max() <= 1  # bounded, unlike z-scores


def test_rank_is_bounded_where_zscore_explodes(token_matrix):
    # the flat run's own z-scores span a wide range despite a ~2% spread in tokens
    flat = token_matrix[["flat"]]
    assert build_length_variants(flat)["zscore"]["series"].abs().max() > 2
    assert build_length_variants(flat)["rank"]["series"].abs().max() <= 1


def test_variant_selection_and_unknown_name(token_matrix):
    assert list(build_length_variants(token_matrix, variants=["rank"])) == ["rank"]
    with pytest.raises(ValueError, match="unknown length variant"):
        build_length_variants(token_matrix, variants=["nonsense"])
