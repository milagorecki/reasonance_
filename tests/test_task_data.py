"""Tests for the task-loading helpers in reasonance.utils."""

import numpy as np
import pandas as pd
import pytest
from reasonance.utils import binarize_predictions, common_individuals


@pytest.fixture
def ragged_predictions():
    """Three runs; run_c is missing two individuals, run_b one."""
    return pd.DataFrame(
        {
            "run_a": [1, 0, 1, 0, 1],
            "run_b": [1, 0, np.nan, 0, 1],
            "run_c": [np.nan, 0, np.nan, 1, 1],
        },
        index=[10, 11, 12, 13, 14],
    )


def test_common_individuals_keeps_only_fully_scored_rows(ragged_predictions):
    assert list(common_individuals(ragged_predictions)) == [11, 13, 14]


def test_common_individuals_respects_the_run_subset(ragged_predictions):
    # run_c is the one with holes; without it more individuals survive
    assert list(common_individuals(ragged_predictions, ["run_a", "run_b"])) == [10, 11, 13, 14]
    assert list(common_individuals(ragged_predictions, ["run_a"])) == [10, 11, 12, 13, 14]


def test_common_individuals_can_be_empty():
    preds = pd.DataFrame({"run_a": [1.0, np.nan], "run_b": [np.nan, 1.0]}, index=[0, 1])
    assert len(common_individuals(preds)) == 0


def test_binarize_predictions_keeps_missing_scores_missing():
    """A run that never scored an individual must not look like it predicted class 0.

    NaN >= threshold is False, so a direct cast would fabricate confident negatives for every
    individual a partially finished run has not reached.
    """
    risk_scores = pd.DataFrame(
        {"Qwen--Qwen3-4B_abc": [0.9, 0.1, np.nan], "gpt-5.1_def": [0.2, np.nan, 0.8]},
        index=[0, 1, 2],
    )
    aggregate = pd.DataFrame(
        [
            {"model": "Qwen/Qwen3-4B", "bench_hash": "abc", "threshold": 0.5},
            {"model": "gpt-5.1", "bench_hash": "def", "threshold": 0.5},
        ]
    )

    preds = binarize_predictions(risk_scores, aggregate)

    assert preds.isna().sum().sum() == 2, "missing risk scores must stay missing"
    assert preds.loc[0].tolist() == [1.0, 0.0]
    # only the individual both runs scored survives the common-set filter
    assert list(common_individuals(preds)) == [0]
