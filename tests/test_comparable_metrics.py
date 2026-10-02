"""Tests for the accuracy-adjusted agreement metric and its components in reasonance.agreement."""

import numpy as np
import pandas as pd
import pytest
from reasonance.agreement import (
    METRIC_COMPONENT_FUNCTIONS,
    METRIC_FUNCTIONS,
    METRICS_NEEDING_LABELS,
    build_pairwise_metric_df,
    combine_metric_components,
)


@pytest.fixture
def unequal_accuracy():
    """Different accuracies with overlapping errors: agreement is capped below 1 and kappa > 0."""
    rng = np.random.default_rng(1)
    y = pd.Series(rng.binomial(1, 0.4, 500))
    a, b = y.copy(), y.copy()
    a[:25] = 1 - a[:25]  # 5% errors
    b[:150] = 1 - b[:150]  # 30% errors, including all of a's
    return pd.DataFrame({"a": a, "b": b}), y


def test_label_metrics_are_selectable_and_require_labels(unequal_accuracy):
    preds, labels = unequal_accuracy
    run_meta = {c: {"model": c, "family": "F", "reasoning_effort": "none", "category": "x"} for c in preds.columns}
    for metric in ("acc_adjusted_agree", "abc"):
        assert metric in METRIC_FUNCTIONS
        assert metric in METRICS_NEEDING_LABELS
        df = build_pairwise_metric_df(preds, run_meta, metric=metric, labels=labels, exclude_same_model=False)
        assert len(df) == 1 and df["value"].notna().all()
        with pytest.raises(ValueError, match="requires labels"):
            build_pairwise_metric_df(preds, run_meta, metric=metric, labels=None)


def test_combine_metric_components_matches_direct_computation():
    """Recombining averaged components reproduces the metric when there is nothing to average."""
    rng = np.random.default_rng(0)
    preds = pd.DataFrame(rng.integers(0, 2, size=(200, 3)), columns=["a", "b", "c"])
    labels = pd.Series(rng.integers(0, 2, size=200))

    for metric in ["kappa", "acc_adjusted_agree", "abc", "acc_baseline", "pred_baseline", "observed"]:
        direct = METRIC_FUNCTIONS[metric](preds, labels)
        p_o, p_base = METRIC_COMPONENT_FUNCTIONS[metric](preds, labels)
        recombined = combine_metric_components(metric, p_o.to_numpy(), p_base.to_numpy())
        np.testing.assert_allclose(recombined, direct.to_numpy(), atol=1e-12)


def test_ratio_of_means_differs_from_mean_of_ratios():
    """The averaging order matters: this is why seed averaging works on the components."""
    p_o = np.array([0.90, 0.80])
    p_base = np.array([0.50, 0.80])  # baselines differ between the two "seeds"

    mean_of_ratios = np.mean((p_o - p_base) / (1 - p_base))
    ratio_of_means = combine_metric_components("kappa", p_o.mean(), p_base.mean())

    assert not np.isclose(mean_of_ratios, ratio_of_means)
    assert np.isclose(ratio_of_means, (0.85 - 0.65) / (1 - 0.65))
