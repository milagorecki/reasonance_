"""Agreement metrics for binary predictions."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import poisson_binom

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

MetricFn = Callable[[pd.DataFrame, pd.Series | None], pd.DataFrame]


def pairwise_positive_agreement_matrix(preds: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Pairwise counts of individuals on which both runs predict 1.

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.

    Returns
    -------
    np.ndarray
        run x run matrix: positive predictions per run on the diagonal, shared positives off it.
    """
    if isinstance(preds, pd.DataFrame):
        preds = preds.values.astype(float)

    if not np.all((preds == 0) | (preds == 1)):
        raise ValueError("preds must be binary (0/1) for pairwise agreement counts to be meaningful")

    # matrix will have number of positive predictions per model on the diagonal, count of agreements on off-diagonals
    pairwise_agree_matrix = preds.T @ preds
    return pairwise_agree_matrix


def observed_agreement_matrix(preds: pd.DataFrame | np.ndarray, return_fraction: bool = False) -> np.ndarray:
    """Pairwise observed agreement: individuals on which two runs give the same prediction.

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    return_fraction : bool
        Return the fraction of individuals (``p_o``) instead of the count.

    Returns
    -------
    np.ndarray
        run x run matrix of agreement counts, or fractions with ``return_fraction``.
    """
    if isinstance(preds, pd.DataFrame):
        preds = preds.values.astype(float)
    # agree in 1s + in 0s
    number_agreed = pairwise_positive_agreement_matrix(preds) + pairwise_positive_agreement_matrix(1 - preds)
    fraction_agreed = number_agreed / len(preds)

    if return_fraction:
        return fraction_agreed
    else:
        return number_agreed


def _baseline_agreement_from_rates(rate: np.ndarray) -> np.ndarray:
    """Baseline agreement ``p_base`` implied by per-run rates.

    How often two runs would match if nothing linked them beyond those rates -- i.e. if they placed
    their errors at random (``rate`` = accuracy, used by the accuracy-adjusted metrics) or their
    positive predictions at random (``rate`` = positive prediction rate, used by plain kappa).
    Agreement above this is what the two runs share beyond the baseline.
    """
    return np.outer(rate, rate) + np.outer(1 - rate, 1 - rate)


def _kappa(
    p_o: np.ndarray, p_base: np.ndarray, columns: list[str] | None = None, return_components: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(p_o - p_base) / (1 - p_base) with DataFrame wrapping.

    If ``return_components`` is True, returns ``(kappa, p_o, p_base)`` as DataFrames.

    Parameters
    ----------
    p_o, p_base : np.ndarray
        Observed and baseline agreement, run x run.
    columns : list of str, optional
        Run ids used as index and columns of the returned frames.
    return_components : bool
        Also return ``p_o`` and ``p_base`` as DataFrames.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (p_o - p_base) / (1 - p_base)
    np.fill_diagonal(kappa, 1.0)  # avoid floating point noise
    # p_o == p_base == 1 → 0/0 → NaN, but this is perfect agreement → 1.0
    perfect = np.isclose(p_o, 1.0) & np.isclose(p_base, 1.0)
    np.fill_diagonal(perfect, False)
    if perfect.any():
        warnings.warn(
            f"{int(perfect.sum())} entries have p_o=p_base=1 (perfect agreement, kappa undefined). Setting to 1.0.",
            stacklevel=3,
        )
        kappa[perfect] = 1.0

    kappa_df = pd.DataFrame(kappa, index=columns, columns=columns)
    if return_components:
        return (
            kappa_df,
            pd.DataFrame(p_o, index=columns, columns=columns),
            pd.DataFrame(p_base, index=columns, columns=columns),
        )
    return kappa_df


def compute_kappa_matrix(
    preds: pd.DataFrame | np.ndarray, return_components: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise Cohen's kappa matrix for binary predictions.

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    return_components : bool
        Also return ``p_o`` and ``p_base`` (from positive prediction rates).

    Returns
    -------
    pd.DataFrame or tuple of pd.DataFrame
        run x run kappa, or ``(kappa, p_o, p_base)`` with ``return_components``.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("preds must be binary (0/1)")

    p_o = observed_agreement_matrix(p, return_fraction=True)
    # expected agreement by chance given each model's positive prediction rate
    pos_pred_rate = p.mean(axis=0)
    p_base = _baseline_agreement_from_rates(pos_pred_rate)
    return _kappa(p_o, p_base, columns, return_components=return_components)


def compute_acc_adjusted_agreement_matrix(
    preds: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    return_components: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Kappa-style agreement with chance level based on per-model accuracy.

    p_o is observed pairwise agreement on raw predictions (same as kappa).
    p_base is chance-level agreement given accuracies:
        acc_i * acc_j + (1-acc_i) * (1-acc_j).

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    labels : pd.Series or np.ndarray
        Ground-truth labels; a Series is aligned to ``preds.index`` when ``preds`` is a DataFrame.
    return_components : bool
        Also return ``p_o`` and ``p_base``.

    Returns
    -------
    pd.DataFrame or tuple of pd.DataFrame
        run x run accuracy-adjusted kappa, or ``(kappa, p_o, p_base)`` with ``return_components``.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("preds must be binary (0/1)")
    if isinstance(labels, pd.Series) and isinstance(preds, pd.DataFrame):
        # align labels to preds index
        y = labels.loc[preds.index].values.astype(float)
    else:
        y = np.asarray(labels, dtype=float)

    # observed agreement on predictions (same as standard kappa)
    p_o = observed_agreement_matrix(p, return_fraction=True)
    # expected agreement by chance given each model's accuracy
    acc = (p == y[:, None]).mean(axis=0)
    p_base = _baseline_agreement_from_rates(acc)
    return _kappa(p_o, p_base, columns, return_components=return_components)


def agreement_beyond_chance(
    preds: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Excess pairwise agreement beyond accuracy-based chance: p_o - p_base.

    Same p_o and p_base as ``compute_acc_adjusted_agreement_matrix``, but
    returns the raw difference instead of normalising by (1 - p_base).

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    labels : pd.Series or np.ndarray
        Ground-truth labels; a Series is aligned to ``preds.index`` when ``preds`` is a DataFrame.

    Returns
    -------
    pd.DataFrame
        run x run matrix of ``p_o - p_base``.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("preds must be binary (0/1)")
    if isinstance(labels, pd.Series) and isinstance(preds, pd.DataFrame):
        # align labels to preds index
        y = labels.loc[preds.index].values.astype(float)
    else:
        y = np.asarray(labels, dtype=float)

    p_o = observed_agreement_matrix(p, return_fraction=True)
    acc = (p == y[:, None]).mean(axis=0)
    p_base = _baseline_agreement_from_rates(acc)

    return pd.DataFrame(p_o - p_base, index=columns, columns=columns)


def acc_baseline_agreement(
    preds: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """The baseline itself: the agreement the two runs' accuracies alone would produce.

    ``p_base = a_i a_j + (1 - a_i)(1 - a_j)`` -- the agreement two runs of these accuracies would reach
    if they were independent. Plotting it next to ``observed`` and ``acc_adjusted_agree`` shows how
    much of a group difference is the baseline moving rather than the agreement.

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    labels : pd.Series or np.ndarray
        Ground-truth labels; a Series is aligned to ``preds.index`` when ``preds`` is a DataFrame.

    Returns
    -------
    pd.DataFrame
        run x run matrix of ``p_base``.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    y = (
        labels.loc[preds.index].values.astype(float)
        if isinstance(labels, pd.Series) and isinstance(preds, pd.DataFrame)
        else np.asarray(labels, dtype=float)
    )
    acc = (p == y[:, None]).mean(axis=0)
    return pd.DataFrame(_baseline_agreement_from_rates(acc), index=columns, columns=columns)


def pred_baseline_agreement(
    preds: pd.DataFrame | np.ndarray, labels: pd.Series | np.ndarray | None = None
) -> pd.DataFrame:
    """Baseline agreement from per-run POSITIVE PREDICTION rates -- the one plain kappa corrects by.

    ``p_base = r_i r_j + (1 - r_i)(1 - r_j)`` with ``r`` the rate at which a run predicts 1. It asks
    what two runs would agree at if they placed their positive predictions at random, ignoring
    whether those predictions are right. ``acc_baseline`` is the same construction on accuracies.
    Labels are accepted and ignored, so the metric registry can call every entry the same way.

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    labels : pd.Series or np.ndarray, optional
        Ignored.

    Returns
    -------
    pd.DataFrame
        run x run matrix of ``p_base``.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    return pd.DataFrame(_baseline_agreement_from_rates(p.mean(axis=0)), index=columns, columns=columns)


def agree_correct_matrix(preds: pd.DataFrame | np.ndarray, labels: pd.Series | np.ndarray) -> pd.DataFrame:
    """Fraction of individuals both runs got RIGHT.

    With binary predictions, two runs that are both wrong have necessarily given the same answer,
    and so have two runs that are both right -- so observed agreement splits exactly into this and
    ``agree_wrong_matrix``: ``p_o = agree_correct + agree_wrong``.

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    labels : pd.Series or np.ndarray
        Ground-truth labels; a Series is aligned to ``preds.index`` when ``preds`` is a DataFrame.

    Returns
    -------
    pd.DataFrame
        run x run fraction of individuals both runs predicted correctly.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    if isinstance(labels, pd.Series) and isinstance(preds, pd.DataFrame):
        y = labels.loc[preds.index].values.astype(float)
    else:
        y = np.asarray(labels, float)
    correct = (p == y[:, None]).astype(float)
    return pd.DataFrame(correct.T @ correct / len(p), index=columns, columns=columns)


def agree_wrong_matrix(preds: pd.DataFrame | np.ndarray, labels: pd.Series | np.ndarray) -> pd.DataFrame:
    """Fraction of individuals both runs got WRONG (and therefore agreed on).

    Parameters
    ----------
    preds : pd.DataFrame or np.ndarray
        Binary (0/1) predictions, individuals x run.
    labels : pd.Series or np.ndarray
        Ground-truth labels; a Series is aligned to ``preds.index`` when ``preds`` is a DataFrame.

    Returns
    -------
    pd.DataFrame
        run x run fraction of individuals both runs predicted incorrectly.
    """
    columns = list(preds.columns) if isinstance(preds, pd.DataFrame) else None
    p = preds.values.astype(float) if isinstance(preds, pd.DataFrame) else np.asarray(preds, dtype=float)
    if isinstance(labels, pd.Series) and isinstance(preds, pd.DataFrame):
        y = labels.loc[preds.index].values.astype(float)
    else:
        y = np.asarray(labels, float)
    wrong = (p != y[:, None]).astype(float)
    return pd.DataFrame(wrong.T @ wrong / len(p), index=columns, columns=columns)


# metrics built from correctness (an accuracy-based chance level, or the right/wrong split), so they
# cannot be computed without labels
METRICS_NEEDING_LABELS = (
    "agree_correct",
    "agree_wrong",
    "acc_adjusted_agree",
    "abc",
    "acc_baseline",
)

METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "kappa": lambda preds, labels: compute_kappa_matrix(preds),
    "acc_adjusted_agree": compute_acc_adjusted_agreement_matrix,
    "abc": agreement_beyond_chance,
    # the chance level on its own, for showing the components of the adjusted metric
    "acc_baseline": acc_baseline_agreement,
    # the same baseline built from positive prediction rates: what plain kappa corrects by
    "pred_baseline": pred_baseline_agreement,
    # observed agreement split by outcome: p_o = agree_correct + agree_wrong
    "agree_correct": agree_correct_matrix,
    "agree_wrong": agree_wrong_matrix,
    "observed": lambda preds, labels: pd.DataFrame(
        observed_agreement_matrix(preds, return_fraction=True),
        index=preds.columns,
        columns=preds.columns,
    ),
}


# Metrics of the form f(p_o, p_base): kept separately so that averaging (e.g. over seed pairs) can
# average the COMPONENTS and recombine them, instead of averaging ratios. Mean-of-ratios and
# ratio-of-means differ whenever p_base varies between the averaged runs.
def _observed_and_acc_baseline(
    preds: pd.DataFrame | np.ndarray, labels: pd.Series | np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(p_o, p_base)`` of the accuracy-adjusted kappa (baseline from per-run accuracy)."""
    _, p_o, p_base = compute_acc_adjusted_agreement_matrix(preds, labels, return_components=True)
    return p_o, p_base


def _observed_and_pred_baseline(
    preds: pd.DataFrame | np.ndarray, labels: pd.Series | np.ndarray | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(p_o, p_base)`` of plain kappa (baseline from positive prediction rates); ``labels`` is ignored."""
    _, p_o, p_base = compute_kappa_matrix(preds, return_components=True)
    return p_o, p_base


METRIC_COMPONENT_FUNCTIONS: dict[str, Callable] = {
    "kappa": _observed_and_pred_baseline,
    "acc_adjusted_agree": _observed_and_acc_baseline,
    "abc": _observed_and_acc_baseline,
    "acc_baseline": _observed_and_acc_baseline,
    "pred_baseline": _observed_and_pred_baseline,
    # p_base is unused for "observed"; the label-free variant is used so it works without labels
    "observed": _observed_and_pred_baseline,
}


def combine_metric_components(metric: str, p_o: float | np.ndarray, p_base: float | np.ndarray) -> float | np.ndarray:
    """Value of ``metric`` from its components, elementwise.

    Parameters
    ----------
    metric : str
        A key of ``METRIC_COMPONENT_FUNCTIONS``.
    p_o, p_base : float or array-like
        Observed agreement and baseline agreement, already averaged if averaging is wanted.

    Returns
    -------
    float or np.ndarray
        Same type as the inputs.
    """
    if metric not in METRIC_COMPONENT_FUNCTIONS:
        raise ValueError(f"metric {metric!r} does not decompose into (p_o, p_base)")
    if metric == "observed":
        return p_o
    if metric in ("acc_baseline", "pred_baseline"):
        return p_base
    if metric == "abc":
        return p_o - p_base
    # kappa and the accuracy-adjusted variant. p_base == 1 means the baseline already predicts
    # perfect agreement (a bin where both runs are always right, say): the ratio is 0/0 there, and
    # perfect observed agreement is defined as 1.0, matching _kappa's handling of the matrix case.
    numerator, denominator = np.asarray(p_o) - np.asarray(p_base), 1 - np.asarray(p_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(
            np.isclose(denominator, 0.0),
            np.where(np.isclose(p_o, 1.0), 1.0, np.nan),
            numerator / np.where(np.isclose(denominator, 0.0), np.nan, denominator),
        )
    return float(value) if np.ndim(value) == 0 else value


def build_pairwise_metric_df(
    preds: pd.DataFrame,
    run_meta: Mapping[str, Mapping[str, Any]],
    metric: str = "kappa",
    labels: pd.Series | None = None,
    exclude_same_model: bool = True,
    return_components: bool = False,
) -> pd.DataFrame:
    """Build a long DataFrame of pairwise metric values with run metadata.

    Parameters
    ----------
    preds : pd.DataFrame
        Wide binary predictions (individuals x run_id).
    run_meta : dict
        Mapping from run_id to metadata dict (must have keys:
        ``model``, ``reasoning_effort``, ``family``, ``category``).
    metric : str
        One of ``"kappa"``, ``"acc_adjusted_agree"``, ``"abc"``,
        ``"observed"``.
    labels : pd.Series, optional
        Ground-truth labels, required for accuracy-based metrics.
    exclude_same_model : bool
        If True, skip pairs from the same (model, effort) combination.
    return_components : bool
        Also return the ``p_o`` and ``p_base`` behind each value, so that callers averaging over
        runs (e.g. seeds) can average the components and recombine them with
        ``combine_metric_components`` instead of averaging the metric itself. Only available for
        metrics in ``METRIC_COMPONENT_FUNCTIONS``.

    Returns
    -------
    pd.DataFrame
        Columns: ``run_a``, ``run_b``, ``family_a``, ``family_b``,
        ``effort_a``, ``effort_b``, ``pair_type``, ``value``, and with
        ``return_components`` also ``p_o`` and ``p_base``.
    """
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric {metric!r}, choose from {list(METRIC_FUNCTIONS)}")
    if metric in METRICS_NEEDING_LABELS and labels is None:
        raise ValueError(f"metric={metric!r} requires labels")

    metric_fn = METRIC_FUNCTIONS[metric]
    mat = metric_fn(preds, labels)
    p_o_mat = p_base_mat = None
    if return_components:
        if metric not in METRIC_COMPONENT_FUNCTIONS:
            raise ValueError(f"metric {metric!r} has no (p_o, p_base) decomposition")
        p_o_mat, p_base_mat = METRIC_COMPONENT_FUNCTIONS[metric](preds, labels)

    run_ids = list(preds.columns)
    rows = []
    for i, a in enumerate(run_ids):
        for b in run_ids[i + 1 :]:
            ma, mb = run_meta[a], run_meta[b]
            if exclude_same_model:
                if ma["model"] == mb["model"] and str(ma["reasoning_effort"]) == str(mb["reasoning_effort"]):
                    continue
            row = {
                "run_a": a,
                "run_b": b,
                "family_a": ma["family"],
                "family_b": mb["family"],
                "effort_a": str(ma["reasoning_effort"]),
                "effort_b": str(mb["reasoning_effort"]),
                "pair_type": pair_type_from_categories(ma["category"], mb["category"]),
                "value": mat.loc[a, b],
            }
            if return_components:
                assert p_o_mat is not None and p_base_mat is not None
                row["p_o"] = p_o_mat.loc[a, b]
                row["p_base"] = p_base_mat.loc[a, b]
            rows.append(row)

    return pd.DataFrame(rows)


def pair_type_from_categories(cat_a: str, cat_b: str) -> str:
    """Classify a pair of runs by their reasoning categories.

    Parameters
    ----------
    cat_a, cat_b : str
        Categories of the two runs (e.g. ``"reasoning"``, ``"non-reasoning"``).

    Returns
    -------
    str
        ``"both <category>"`` if the categories match, else ``"cross"``.
    """
    if cat_a == cat_b:
        return f"both {cat_a}"
    return "cross"


def _model_pair_mean(
    sub: pd.DataFrame,
    sub_labels: pd.Series | np.ndarray | None,
    metric: str,
    run_ids: Sequence[str],
    model_of_run: Mapping[str, str],
) -> tuple[np.ndarray, float]:
    """Mean metric over MODEL pairs, averaging a pair's cross-seed runs before combining.

    Seeds of one model are several runs of the same thing: their pairs with another model all
    estimate the same quantity, so their components are averaged and the metric rebuilt from the
    averages (the metric is a ratio -- averaging the ratios is a different, biased quantity).
    Run pairs within one model are dropped: they measure seed reproducibility, not agreement
    between models.

    Parameters
    ----------
    sub : pd.DataFrame
        Binary predictions of one bin, individuals x run, columns in the order of ``run_ids``.
    sub_labels : pd.Series or np.ndarray, optional
        Ground-truth labels of the bin's individuals.
    metric : str
        A key of ``METRIC_COMPONENT_FUNCTIONS``.
    run_ids : sequence of str
        Runs in the column order of ``sub``.
    model_of_run : mapping
        ``{run_id: model}``.

    Returns
    -------
    p_o_mat : np.ndarray
        Run x run observed agreement of the bin.
    value : float
        Mean metric over model pairs (NaN if there are none).
    """
    p_o_mat, p_base_mat = METRIC_COMPONENT_FUNCTIONS[metric](sub, sub_labels)
    p_o_mat, p_base_mat = np.asarray(p_o_mat), np.asarray(p_base_mat)
    models = [model_of_run[run_id] for run_id in run_ids]
    by_pair: dict[tuple, list[tuple[float, float]]] = {}
    for i, model_i in enumerate(models):
        for j in range(i + 1, len(models)):
            if model_i == models[j]:
                continue  # same model, different seed
            key = tuple(sorted((model_i, models[j])))
            by_pair.setdefault(key, []).append((p_o_mat[i, j], p_base_mat[i, j]))
    values = []
    for combos in by_pair.values():
        p_o = float(np.nanmean([component[0] for component in combos]))
        p_base = float(np.nanmean([component[1] for component in combos]))
        values.append(combine_metric_components(metric, p_o, p_base))
    # floats in practice (scalar components), but typed float | ndarray, which numpy's stubs reject in a list
    return p_o_mat, float(np.nanmean(values)) if values else float("nan")  # type: ignore[arg-type]


def agreement_metric_by_length(
    run_ids: list[str],
    preds: pd.DataFrame,
    length: pd.Series,
    metric: str = "kappa",
    labels: pd.Series | np.ndarray | None = None,
    bins: pd.Series | pd.Categorical | None = None,
    n_bins: int = 10,
    min_bin: int = 30,
    model_of_run: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean pairwise agreement metric among ``run_ids``, binned by length.

    Parameters
    ----------
    run_ids : list of str
        Runs whose pairwise agreement is measured within each bin.
    preds : pd.DataFrame
        Wide binary predictions (individuals x run_id).
    length : pd.Series
        Per-individual reasoning length used both for binning and for the x
        position of each bin.
    metric : str
        One of ``"kappa"``, ``"acc_adjusted_agree"``, ``"abc"``, ``"observed"``.
    labels : pd.Series or np.ndarray, optional
        Ground-truth labels, required for accuracy-based metrics.
    bins : Series/Categorical, optional
        Precomputed shared binning aligned to ``length``. If None, ``length``
        is binned into ``n_bins`` equal-count quantiles.
    n_bins : int
        Number of quantile bins when ``bins`` is None.
    min_bin : int
        Minimum number of individuals in a bin to compute the metric.
    model_of_run : dict, optional
        ``{run_id: model}``. Given, the bin's value is the mean over MODEL pairs, each pair's
        components averaged over its cross-seed run pairs first, and run pairs within one model
        dropped. Without it every run pair counts once, which lets a model with more seeds weigh
        more and counts seed-to-seed agreement as cross-model agreement.

    Returns
    -------
    curve : pd.DataFrame
        Columns: ``median_tokens``, ``n``, ``mean_value``, ``bin_left``, ``bin_right``.
    last_mat : np.ndarray
        Metric matrix from the last bin (useful for debugging). With ``model_of_run`` it is the
        bin's observed agreement ``p_o`` matrix instead.
    """
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric {metric!r}, choose from {list(METRIC_FUNCTIONS)}")
    if metric in METRICS_NEEDING_LABELS and labels is None:
        raise ValueError(f"metric={metric!r} requires labels")
    metric_fn = METRIC_FUNCTIONS[metric]
    if model_of_run is not None and metric not in METRIC_COMPONENT_FUNCTIONS:
        raise ValueError(f"metric={metric!r} does not decompose, so seeds cannot be averaged")

    p = preds[run_ids]
    # One bin of easy individuals where every run is right gives p_o = p_base = 1, which _kappa
    # reports as a warning. Over all bins and pairs that floods the output with identical lines, so they
    # are collected and summarised once per curve instead.
    saturated = 0
    if bins is None:
        bins = pd.qcut(length, q=n_bins, duplicates="drop")
    records = []
    mat = np.eye(len(run_ids))
    for _interval, idx in length.groupby(bins, observed=True).groups.items():
        sub = p.loc[idx]
        if len(sub) < min_bin:
            continue
        sub_labels = (
            labels.loc[idx]
            if isinstance(labels, pd.Series)
            else (labels[p.index.get_indexer(idx)] if labels is not None else None)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model_of_run is None:
                mat = metric_fn(sub, sub_labels).values
                iu = np.triu_indices_from(mat, k=1)
                value = float(np.nanmean(mat[iu]))
            else:
                mat, value = _model_pair_mean(sub, sub_labels, metric, run_ids, model_of_run)
        saturated += sum(int(str(w.message).split()[0]) for w in caught if "p_o=p_base=1" in str(w.message))
        records.append(
            {
                "median_tokens": float(length.loc[idx].median()),
                "n": len(sub),
                "mean_value": value,
                # bin edges, so a plot can show the value as constant across the bin it was
                # computed on instead of interpolating between bin centres
                "bin_left": float(getattr(_interval, "left", np.nan)),
                "bin_right": float(getattr(_interval, "right", np.nan)),
            }
        )
    if saturated:
        logging.info(
            "%d pair-bin entries had p_o = p_base = 1 (every run right on those individuals), set to 1.0",
            saturated,
        )
    return pd.DataFrame(records).sort_values("median_tokens").reset_index(drop=True), mat


def vote_distribution(
    preds: pd.DataFrame,
    run_ids: list[str],
    labels: pd.Series | None = None,
    mode: str = "correct",
) -> np.ndarray:
    """Distribution of per-individual consensus over a group of runs.

    For each individual, counts how many of the ``run_ids`` vote the same way, and returns how the
    individuals spread over the possible counts. Mass at the ends means the group is unanimous;
    mass in the middle means it is split.

    Parameters
    ----------
    preds : pd.DataFrame
        Binary predictions, individuals x run_id.
    run_ids : list of str
        Runs forming the group.
    labels : pd.Series, optional
        True labels, aligned to ``preds``. Required for ``mode="correct"``.
    mode : {"correct", "class"}
        "correct" counts runs predicting the *true* label, so index 0 is "all wrong" and index N
        "all correct". "class" counts runs predicting 1, regardless of the truth.

    Returns
    -------
    np.ndarray
        Fractions of individuals for counts ``0..N``, summing to 1 (all-NaN input gives zeros).
    """
    if mode not in ("correct", "class"):
        raise ValueError(f"unknown mode: {mode!r}")
    group = preds[run_ids].dropna()
    n_runs = len(run_ids)
    if group.empty:
        return np.zeros(n_runs + 1)

    votes = group.to_numpy().astype(int)
    if mode == "correct":
        if labels is None:
            raise ValueError('mode="correct" requires `labels`')
        y = labels.loc[group.index].to_numpy().astype(int)[:, None]
        votes = (votes == y).astype(int)
    counts = np.bincount(votes.sum(axis=1), minlength=n_runs + 1)
    return counts / counts.sum()


# copied
def _poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """pmf of a sum of independent Bernoullis with different probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Success probability per trial.

    Returns
    -------
    np.ndarray
        ``pmf[k]`` = P(exactly k successes), length ``len(probs) + 1``.
    """
    rates = np.atleast_1d(np.asarray(probs, dtype=float).squeeze())
    pmf = np.array([poisson_binom.pmf(k=k, p=rates) for k in np.arange(rates.size + 1)])
    return np.asarray(pmf, dtype=float)


def vote_distribution_null(
    preds: pd.DataFrame,
    run_ids: list[str],
    labels: pd.Series | None = None,
    mode: str = "correct",
) -> np.ndarray:
    """Vote distribution expected from *independent* runs with the same marginal rates.

    The multi-run generalisation of the chance level used by
    :func:`compute_acc_adjusted_agreement_matrix`: each run is right (``mode="correct"``) or votes 1
    (``mode="class"``) independently, at its own observed rate, so the vote count follows a
    Poisson-binomial distribution. Comparing the observed distribution against this null separates
    "the models agree because they are accurate" from "the models agree beyond that".

    Parameters
    ----------
    preds : pd.DataFrame
        Binary predictions, individuals x run_id.
    run_ids : list of str
        Runs forming the group.
    labels : pd.Series, optional
        True labels, aligned to ``preds``. Required for ``mode="correct"``.
    mode : {"correct", "class"}
        As in :func:`vote_distribution`.

    Returns
    -------
    np.ndarray
        Poisson-binomial pmf over counts ``0..N`` (all-NaN input gives zeros).

    Notes
    -----
    Assumes every individual is equally hard for a given run. Real individuals differ, and
    independent runs still cluster at the extremes on easy ones, so an excess over this null mixes
    item difficulty with shared model behaviour -- see ``difficulty`` in
    :func:`vote_distribution_null_stratified`.
    """
    group = preds[run_ids].dropna()
    if group.empty:
        return np.zeros(len(run_ids) + 1)
    votes = group.to_numpy().astype(int)
    if mode == "correct":
        if labels is None:
            raise ValueError('mode="correct" requires `labels`')
        votes = (votes == labels.loc[group.index].to_numpy().astype(int)[:, None]).astype(int)
    elif mode != "class":
        raise ValueError(f"unknown mode: {mode!r}")
    return _poisson_binomial_pmf(votes.mean(axis=0))
