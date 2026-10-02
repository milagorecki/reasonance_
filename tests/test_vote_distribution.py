"""Tests for the vote-distribution helpers in reasonance.agreement."""

import numpy as np
import pandas as pd
import pytest
from reasonance.agreement import (
    _poisson_binomial_pmf,
    vote_distribution,
    vote_distribution_null,
)
from scipy.stats import binom


def vote_excess(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Observed minus expected vote distribution; positive = more clustered than independence."""
    return np.asarray(observed, dtype=float) - np.asarray(null, dtype=float)


def vote_overdispersion(observed: np.ndarray, null: np.ndarray) -> float:
    """Variance of the vote count, observed / expected under independence."""
    levels = np.arange(len(observed))

    def _var(pmf: np.ndarray) -> float:
        pmf = np.asarray(pmf, dtype=float)
        mean = float(np.sum(levels * pmf))
        return float(np.sum(pmf * (levels - mean) ** 2))

    return _var(observed) / _var(null)


@pytest.fixture
def unanimous():
    """Four runs that always agree, half of them right."""
    preds = pd.DataFrame({f"m{j}": [1, 1, 0, 0] for j in range(4)})
    labels = pd.Series([1, 0, 0, 1])
    return preds, labels


def test_vote_distribution_unanimous_sits_at_the_ends(unanimous):
    preds, labels = unanimous
    dist = vote_distribution(preds, list(preds.columns), labels=labels, mode="correct")
    assert dist[0] == pytest.approx(0.5)  # all wrong on half the individuals
    assert dist[-1] == pytest.approx(0.5)  # all correct on the other half
    assert dist[1:-1].sum() == pytest.approx(0.0)


def test_vote_distribution_class_mode_ignores_labels():
    preds = pd.DataFrame({"a": [1, 0], "b": [1, 0]})
    dist = vote_distribution(preds, ["a", "b"], mode="class")
    assert dist[0] == pytest.approx(0.5)  # both predict 0
    assert dist[2] == pytest.approx(0.5)  # both predict 1


def test_vote_distribution_sums_to_one_and_needs_labels():
    preds = pd.DataFrame({"a": [1, 0, 1], "b": [0, 0, 1]})
    assert vote_distribution(preds, ["a", "b"], mode="class").sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        vote_distribution(preds, ["a", "b"], mode="correct")
    with pytest.raises(ValueError):
        vote_distribution(preds, ["a", "b"], mode="nonsense")


def test_poisson_binomial_matches_binomial_for_equal_probabilities():
    pmf = _poisson_binomial_pmf(np.repeat(0.3, 5))
    assert pmf == pytest.approx(binom.pmf(np.arange(6), 5, 0.3))


def test_null_matches_observed_for_independent_runs():
    rng = np.random.default_rng(0)
    preds = pd.DataFrame({f"m{j}": rng.binomial(1, 0.7, 4000) for j in range(4)})
    labels = pd.Series(np.ones(4000, dtype=int))
    observed = vote_distribution(preds, list(preds.columns), labels=labels)
    null = vote_distribution_null(preds, list(preds.columns), labels=labels)
    assert np.abs(vote_excess(observed, null)).max() < 0.02
    assert vote_overdispersion(observed, null) == pytest.approx(1.0, abs=0.1)


def test_correlated_runs_are_overdispersed():
    rng = np.random.default_rng(1)
    shared = rng.binomial(1, 0.7, 4000)
    preds = pd.DataFrame(
        {f"m{j}": np.where(rng.random(4000) < 0.9, shared, rng.binomial(1, 0.7, 4000)) for j in range(4)}
    )
    labels = pd.Series(np.ones(4000, dtype=int))
    observed = vote_distribution(preds, list(preds.columns), labels=labels)
    null = vote_distribution_null(preds, list(preds.columns), labels=labels)
    excess = vote_excess(observed, null)
    assert excess[-1] > 0.1 and excess[0] > 0  # mass piles up at both ends
    assert vote_overdispersion(observed, null) > 2
