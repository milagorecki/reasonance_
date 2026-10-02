"""Analysis that produces the numbers behind the paper's figures.

This is the single implementation: ``reasonance.paper_figures`` calls it to draw a figure, and the
notebooks call it to look at the same values interactively. Nothing here depends on a notebook
having been run first -- everything starts from the aggregate CSVs and the per-run predictions.

Results are cached to ``results/pair-values/`` because loading predictions for a task takes about a
minute; pass ``refresh=True`` (or delete the file) to recompute.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from reasonance.agreement import (
    METRIC_COMPONENT_FUNCTIONS,
    METRICS_NEEDING_LABELS,
    build_pairwise_metric_df,
    combine_metric_components,
)
from reasonance.utils import (
    EXCLUDED_MODELS,
    MATCHED_PAIR_SPECS,
    balance_seeds,
    baseline_accuracy,
    common_individuals,
    effort_display,
    load_aggregate,
    load_task_data,
    matched_models,
    matched_sides,
    model_sort_key,
    pretty_model_name,
    runs_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping
    from typing import Any

    from reasonance.utils import TaskData

# Where the per-run predictions live: the directory that contains the benchmark `results/` folder
# (the paths in the aggregate CSVs start with `results/`). Defaults to the working directory, i.e.
# the repository root; set REASONANCE_RESULTS_ROOT to point elsewhere.
RESULTS_MODELS_ROOT = Path(os.environ.get("REASONANCE_RESULTS_ROOT", "."))
PAIR_VALUES_DIR = Path("results/pair-values")
AGGREGATE_TEMPLATE = "results/{task}-aggregated-0-bullet-is.csv"


def _newest_input_time(aggregate_path: Path, results_root: Path) -> float:
    """Modification time of the newest thing a cached value derives from.

    The aggregate changes when runs are added, but a run's predictions can also be rewritten (a
    re-run, a retry) without the aggregate moving, so both are checked.
    """
    aggregate_path = Path(aggregate_path)
    if not aggregate_path.exists():
        return 0.0
    newest = aggregate_path.stat().st_mtime
    try:
        paths = pd.read_csv(aggregate_path, usecols=["predictions_path"])["predictions_path"]
    except (ValueError, KeyError):
        return newest
    for relative in paths.dropna().unique():
        file_path = Path(results_root) / str(relative)
        if file_path.exists():
            newest = max(newest, file_path.stat().st_mtime)
    return newest


def _cache_is_current(cache: Path, aggregate_path: Path, results_root: Path = RESULTS_MODELS_ROOT) -> bool:
    """True when ``cache`` was written after every input it derives from.

    Existence alone is not enough: adding runs and rebuilding the aggregate, or re-running a single
    model, leaves a cache that silently describes the old data.
    """
    if not cache.exists():
        return False
    return cache.stat().st_mtime >= _newest_input_time(aggregate_path, results_root)


def label_tag(label_filter: int | None) -> str:
    """File-name tag for the individuals a value was computed on.

    Parameters
    ----------
    label_filter : int or None
        True label the individuals were restricted to; None for all individuals.

    Returns
    -------
    str
        ``"all_individuals"`` or ``"label<label_filter>"``.
    """
    return "all_individuals" if label_filter is None else f"label{label_filter}"


# Pairs where both runs reason, or neither; a mixed pair answers a different question.
PAIR_TYPE_TO_GROUP = {"both reasoning": "reasoning", "both non-reasoning": "non-reasoning"}
# OpenAI models exist at several effort levels; the off/on comparison uses these two.
OPENAI_EFFORT_KEEP = {"none", "high"}


def available_tasks(aggregate_template: str = AGGREGATE_TEMPLATE) -> list[str]:
    """Tasks that have an aggregate CSV, so a figure can include whatever exists.

    Parameters
    ----------
    aggregate_template : str
        Path template of the per-task aggregate CSV, with a ``{task}`` placeholder.

    Returns
    -------
    list of str
        Task names, sorted.
    """
    pattern = Path(aggregate_template.format(task="*"))
    prefix, suffix = pattern.name.split("*")
    return sorted(p.name[len(prefix) : -len(suffix)] for p in pattern.parent.glob(pattern.name))


def matched_run_ids(meta: pd.DataFrame) -> set[str]:
    """Runs belonging to a matched (model, effort) cell, i.e. one with a reasoning counterpart.

    Parameters
    ----------
    meta : pd.DataFrame
        Run metadata with ``run_id``, ``model`` and ``reasoning_effort`` columns.

    Returns
    -------
    set of str
        The run ids of every cell whose matched counterpart is present in ``meta``.
    """
    matched_cells = set()
    for m_nr, e_nr, m_r, e_r in MATCHED_PAIR_SPECS:
        if runs_for(meta, model=m_nr, effort=e_nr) and runs_for(meta, model=m_r, effort=e_r):
            matched_cells.add((m_nr, str(e_nr)))
            matched_cells.add((m_r, str(e_r)))
    by_run = meta.set_index("run_id")
    return {
        run_id
        for run_id in meta["run_id"]
        if (by_run.loc[run_id, "model"], str(by_run.loc[run_id, "reasoning_effort"])) in matched_cells
    }


def _average_over_seeds(pairs: pd.DataFrame, run_meta: Mapping[str, Mapping[str, Any]], metric: str) -> pd.DataFrame:
    """One row per model pair: components averaged over balanced cross-seed combinations.

    Both sides are truncated to the ``min(n_a, n_b)`` lowest seed numbers, so neither model
    contributes more runs than the other, and ``p_o``/``p_base`` are averaged over all k x k
    combinations before the metric is rebuilt from them.

    Balancing is PER PAIR, not pool-wide: the unit of analysis here is the model pair and every pair
    enters the mean once, so a pair with more seeds gets a more precise value rather than more weight
    -- there is no bias to remove, and equalising k across pairs would discard most of the cross-seed
    combinations for nothing. The length curves balance pool-wide instead
    (``reasonance.utils.balance_seeds``) because there the RUNS are the unit inside a bin, so unequal
    seed counts do become unequal weight. Averaging ratios instead would give a different (and wrong)
    answer whenever ``p_base`` differs between seeds. Truncation is by seed number rather than
    sampled, so repeated runs give identical numbers.

    Parameters
    ----------
    pairs : pd.DataFrame
        One row per run pair, from ``build_pairwise_metric_df`` (with components if ``metric``
        decomposes).
    run_meta : mapping
        ``{run_id: {"model": ..., "reasoning_effort": ..., "generation_seed": ...}}``.
    metric : str
        Agreement metric the pairs hold.

    Returns
    -------
    pd.DataFrame
        One row per model pair, with ``n_seeds_per_side`` and ``n_seed_pairs`` added.
    """
    decomposes = metric in METRIC_COMPONENT_FUNCTIONS
    pairs = pairs.assign(
        model_a=pairs["run_a"].map(lambda r: run_meta[r]["model"]),
        model_b=pairs["run_b"].map(lambda r: run_meta[r]["model"]),
    )
    pairs["_pair_key"] = [
        tuple(sorted([(row.model_a, row.effort_a), (row.model_b, row.effort_b)])) for row in pairs.itertuples()
    ]

    rows = []
    for key, group in pairs.groupby("_pair_key"):
        (m1, e1), (m2, e2) = key
        all_runs = sorted(set(group["run_a"]) | set(group["run_b"]))

        def _side(model: str, effort: str, runs: list[str] = all_runs) -> list[str]:
            """The runs of one (model, effort) side of the pair, in seed order."""
            ids = [r for r in runs if run_meta[r]["model"] == model and str(run_meta[r]["reasoning_effort"]) == effort]
            return sorted(ids, key=lambda r: run_meta[r]["generation_seed"])

        side_1, side_2 = _side(m1, e1), _side(m2, e2)
        k = min(len(side_1), len(side_2))
        keep = set(side_1[:k]) | set(side_2[:k])
        sub = group[group["run_a"].isin(keep) & group["run_b"].isin(keep)]
        if sub.empty:
            continue

        row = sub.iloc[0].copy()
        if decomposes:
            row["p_o"] = sub["p_o"].mean()
            row["p_base"] = sub["p_base"].mean()
            row["value"] = combine_metric_components(metric, row["p_o"], row["p_base"])
        else:
            row["value"] = sub["value"].mean()
        row["n_seeds_per_side"] = k
        row["n_seed_pairs"] = len(sub)
        rows.append(row)

    return pd.DataFrame(rows).drop(columns=["model_a", "model_b", "_pair_key"], errors="ignore")


def build_effort_pairs(
    task_data: TaskData,
    metric: str = "acc_adjusted_agree",
    matched: bool = True,
    average_seeds: bool = True,
    openai_effort_keep: Collection[str] | None = OPENAI_EFFORT_KEEP,
    include_pooled: bool = True,
) -> pd.DataFrame:
    """Pairwise agreement with reasoning off vs on, one row per model pair and family group.

    Parameters
    ----------
    task_data : TaskData
        From ``reasonance.utils.load_task_data``.
    metric : str
        Agreement metric; accuracy-based ones use the labels in ``task_data``.
    matched : bool
        Keep only runs whose (model, effort) cell has a reasoning counterpart, so the two groups
        hold the same base models.
    average_seeds : bool
        Average balanced cross-seed combinations into one value per model pair.
    openai_effort_keep : set of str or None
        Effort levels used for OpenAI's off/on comparison; None keeps every level.
    include_pooled : bool
        Append a copy of every pair with ``family="All"``, which the per-task figures use as their
        pooled group. Set False to get one row per pair, e.g. to filter by pair scope yourself.

    Returns
    -------
    pd.DataFrame
        Columns ``run_a``, ``run_b``, ``family`` ("All" duplicates the pooled pairs), ``group``
        (reasoning / non-reasoning), the components, ``value`` and the seed counts.
    """
    meta = task_data.meta
    run_meta = meta.set_index("run_id").to_dict("index")
    decomposes = metric in METRIC_COMPONENT_FUNCTIONS

    # Agreement is measured on the individuals every run IN THE FIGURE answered. Unmatched models
    # (o1, o3, o3-mini, o4-mini) never enter a pair, so their gaps must not shrink the set -- with
    # them included the intersection loses individuals no reported number depends on.
    keep_runs: set[str] | list[str] = matched_run_ids(meta) if matched else set(task_data.run_ids)
    keep_runs = [r for r in task_data.run_ids if r in keep_runs]
    index = common_individuals(task_data.predictions, keep_runs)
    predictions = task_data.predictions.loc[index, keep_runs]
    labels = task_data.labels.loc[index]

    pairs = build_pairwise_metric_df(
        predictions,
        run_meta,
        metric=metric,
        labels=labels if metric in METRICS_NEEDING_LABELS else None,
        exclude_same_model=True,
        return_components=decomposes,
    )
    if average_seeds:
        pairs = _average_over_seeds(pairs, run_meta, metric)

    pairs = pairs[pairs["pair_type"].isin(PAIR_TYPE_TO_GROUP)].copy()
    pairs["group"] = pairs["pair_type"].map(PAIR_TYPE_TO_GROUP)
    # OpenAI has several effort levels; only the off/on pair belongs in the off-vs-on comparison.
    # Pass openai_effort_keep=None to keep every level, e.g. for a by-effort-level figure.
    if openai_effort_keep is not None:
        drop = ((pairs["family_a"] == "OpenAI") & ~pairs["effort_a"].isin(openai_effort_keep)) | (
            (pairs["family_b"] == "OpenAI") & ~pairs["effort_b"].isin(openai_effort_keep)
        )
        pairs = pairs[~drop]

    pairs["model_a"] = pairs["run_a"].map(lambda r: run_meta[r]["model"])
    pairs["model_b"] = pairs["run_b"].map(lambda r: run_meta[r]["model"])
    # the two base models identify one pair across the reasoning-off and reasoning-on groups, which
    # is what lets a figure connect them with a line
    pairs["pair_key"] = [" + ".join(sorted([a, b])) for a, b in zip(pairs["model_a"], pairs["model_b"])]

    # "family" names the single family of a within-family pair, and the pair of families otherwise
    pairs["family"] = np.where(
        pairs["family_a"] == pairs["family_b"], pairs["family_a"], pairs["family_a"] + "-" + pairs["family_b"]
    )
    if include_pooled:
        within = pairs[pairs["family_a"] == pairs["family_b"]]
        pooled = pairs.assign(family="All")  # the same pairs again, pooled across families
        out = pd.concat([pooled, within], ignore_index=True)
    else:
        out = pairs.copy()
    out["metric"] = metric
    out["task"] = task_data.task
    out["qa_mode"] = task_data.qa_mode
    out["n_individuals"] = len(index)  # travels with the values, so a figure can state it
    return out


def effort_pairs(
    task: str,
    metric: str = "acc_adjusted_agree",
    qa_mode: str = "numeric",
    label_filter: int | None = None,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
    **build_kwargs: Any,
) -> pd.DataFrame:
    """``build_effort_pairs`` for one task, computing from raw data or reading the cache.

    The cache exists only to save the ~1 minute of loading predictions; deleting it changes nothing
    but the runtime, and ``refresh=True`` recomputes.

    Parameters
    ----------
    task : str
        Task name.
    metric : str
        Agreement metric; part of the cache file name.
    qa_mode : str
        QA mode to load.
    label_filter : int, optional
        Keep only the individuals with this true label.
    results_root : Path
        Root of the per-run prediction files.
    cache_dir : Path
        Directory of the cached pair values.
    refresh : bool
        Recompute even when the cache is newer than its inputs.
    **build_kwargs
        Passed on to :func:`build_effort_pairs`.

    Returns
    -------
    pd.DataFrame
        As :func:`build_effort_pairs`.
    """
    from reasonance.paper_figures import save_pair_values  # local import: paper_figures imports this module

    # The metric is part of the file name: without it, computing another metric would overwrite the
    # cache of the first, and alternating between the two would recompute (and rewrite) every time.
    tag = label_tag(label_filter)
    # one row per model pair, or one per cross-seed combination: two different files, or the second
    # would silently overwrite the first (and every figure would then read whichever ran last)
    seed_tag = "" if build_kwargs.get("average_seeds", True) else "-allseeds"
    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    cache = Path(cache_dir) / f"{task}-{qa_mode}-{tag}-{metric}{seed_tag}-effort-pairs.csv"
    if not refresh and _cache_is_current(cache, aggregate_path, results_root):
        logging.info(f"using cached {cache.name} (newer than every input)")
        return pd.read_csv(cache)
    if cache.exists():
        logging.info(f"{cache.name} is older than its inputs; recomputing")

    task_data = load_task_data(
        task,
        results_root=results_root,
        qa_mode=qa_mode,
        aggregate_path=AGGREGATE_TEMPLATE.format(task=task),
        drop_models=EXCLUDED_MODELS,
        dedup_subset=["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode", "generation_seed"],
        label_filter=label_filter,
        common_only=False,  # build_effort_pairs restricts to the runs it actually uses
    )
    pairs = build_effort_pairs(task_data, metric=metric, **build_kwargs)
    save_pair_values(pairs, task, qa_mode, kind=f"{tag}-{metric}{seed_tag}-effort", root=Path(cache_dir))
    return pairs


def effort_pairs_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """``effort_pairs`` for several tasks, concatenated. ``tasks=None`` uses every task available.

    Tasks that fail to load (missing files, no matched runs) are skipped with a message.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`effort_pairs`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    frames = []
    for task in tasks:
        try:
            frames.append(effort_pairs(task, **kwargs))
        except (FileNotFoundError, IndexError, KeyError) as err:
            print(f"skipping {task}: {type(err).__name__}: {err}")
    if not frames:
        raise RuntimeError(f"no usable tasks in {tasks}")
    return pd.concat(frames, ignore_index=True)


def within_level_pairs(
    task: str,
    metric: str = "acc_adjusted_agree",
    qa_mode: str = "numeric",
    family: str = "OpenAI",
    matched: bool = True,
    label_filter: int | None = None,
    results_root: Path = RESULTS_MODELS_ROOT,
) -> pd.DataFrame:
    """Agreement between runs of one family AT THE SAME effort level, one row per pair and level.

    Only OpenAI exposes graded effort; "none" is its reasoning-off reference. Not cached: every call
    recomputes from the predictions, so there is no ``refresh``.

    ``matched=True`` keeps only models that also have a reasoning-off run (the gpt-5.x family here).
    Without it the graded levels pull in o3/o3-mini/o4-mini while "none" stays gpt-5.x only, so the
    levels would be compared across different sets of models.

    Parameters
    ----------
    task : str
        Task name.
    metric : str
        Agreement metric.
    qa_mode : str
        QA mode to load.
    family : str
        Family whose within-level pairs are kept.
    matched : bool
        Keep only runs of models that also have a reasoning-off run.
    label_filter : int, optional
        Keep only the individuals with this true label.
    results_root : Path
        Root of the per-run prediction files.

    Returns
    -------
    pd.DataFrame
        One row per model pair and level, seed-averaged, with ``level``, ``family``, ``metric``,
        ``task`` and ``qa_mode`` added.
    """
    task_data = load_task_data(
        task,
        results_root=results_root,
        qa_mode=qa_mode,
        aggregate_path=AGGREGATE_TEMPLATE.format(task=task),
        drop_models=EXCLUDED_MODELS,
        dedup_subset=["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode", "generation_seed"],
        label_filter=label_filter,
        common_only=True,
    )
    run_meta = task_data.meta.set_index("run_id").to_dict("index")
    decomposes = metric in METRIC_COMPONENT_FUNCTIONS
    pairs = build_pairwise_metric_df(
        task_data.predictions,
        run_meta,
        metric=metric,
        labels=task_data.labels if metric in METRICS_NEEDING_LABELS else None,
        exclude_same_model=True,
        return_components=decomposes,
    )
    pairs = _average_over_seeds(pairs, run_meta, metric)
    if matched:
        keep = matched_run_ids(task_data.meta)
        pairs = pairs[pairs["run_a"].isin(keep) & pairs["run_b"].isin(keep)]
    pairs = pairs[
        (pairs["family_a"] == family) & (pairs["family_b"] == family) & (pairs["effort_a"] == pairs["effort_b"])
    ].copy()
    pairs["level"] = pairs["effort_a"]
    pairs["family"] = family
    pairs["metric"] = metric
    pairs["task"] = task
    pairs["qa_mode"] = qa_mode
    return pairs


def within_level_pairs_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """``within_level_pairs`` for several tasks, concatenated.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`within_level_pairs`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    frames = []
    for task in tasks:
        try:
            frames.append(within_level_pairs(task, **kwargs))
        except (FileNotFoundError, IndexError, KeyError) as err:
            print(f"skipping {task}: {type(err).__name__}: {err}")
    return pd.concat(frames, ignore_index=True)


np.seterr(invalid="ignore")


# Effort level that represents "reasoning" for OpenAI in the vote figures; Qwen has a plain on/off.
VOTE_OPENAI_EFFORT = "high"


def vote_group_draws(task_data: TaskData, openai_effort: str = VOTE_OPENAI_EFFORT) -> dict[str, list[list[str]]]:
    """``{group: [draw, ...]}`` for the matched models, reasoning off vs on; a draw is one run per model.

    A vote count is a number of MODELS, so one distribution holds exactly one run per matched base
    model -- and the two groups hold the same number, since unanimity is mechanically easier in a
    smaller group. Every seed is used all the same: draw r takes each model's r-th seed (seeds
    balanced to the lowest count any cell of either side has), and the figures average over the
    draws. Averaging distributions over draws equals pooling over (individual, draw); merging a
    model's seeds into one vote instead would give fractional counts and remove seed noise before
    counting. Draw r of the off side and draw r of the on side pair up, for the transitions.

    Parameters
    ----------
    task_data : TaskData
        From ``reasonance.utils.load_task_data``.
    openai_effort : str
        The one effort level that stands for "reasoning on" for gpt-5.x.

    Returns
    -------
    dict
        ``{"non-reasoning": draws, "reasoning": draws}``, each draw a list of run ids (one per
        model); groups without any draw are left out.
    """
    meta = task_data.meta
    by_run = meta.set_index("run_id")

    cells: dict[str, dict[str, list[str]]] = {"non-reasoning": {}, "reasoning": {}}
    for m_nr, e_nr, m_r, e_r in MATCHED_PAIR_SPECS:
        if str(m_r).startswith("gpt-5") and str(e_r) != openai_effort:
            continue  # one effort per OpenAI model, else that side holds three settings per model
        off = [r for r in runs_for(meta, model=m_nr, effort=e_nr) if r in task_data.predictions.columns]
        on = [r for r in runs_for(meta, model=m_r, effort=e_r) if r in task_data.predictions.columns]
        if off and on:
            cells["non-reasoning"][m_nr] = off
            cells["reasoning"][m_r] = on
    # one seed count for every cell of BOTH sides, so each draw is complete on both
    (balanced,), _ = balance_seeds(meta, [r for side in cells.values() for runs in side.values() for r in runs])
    keep = set(balanced)
    draws = {}
    for group, models in cells.items():
        per_model = [
            sorted((r for r in runs if r in keep), key=lambda r: (by_run.loc[r, "generation_seed"], str(r)))
            for runs in models.values()
        ]
        draws[group] = [[runs[rank] for runs in per_model] for rank in range(min(map(len, per_model)))]
    return {group: runs for group, runs in draws.items() if runs}


def vote_distributions(
    task: str,
    mode: str = "correct",
    qa_mode: str = "numeric",
    label_filter: int | None = None,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
    **group_kwargs: Any,
) -> pd.DataFrame:
    """Observed vote distribution, the independent-errors null, and their difference.

    One row per (group, vote count): the share of individuals with exactly ``k`` of the matched models
    predicting the true label, what independent errors at each model's own accuracy would give
    (Poisson-binomial), and the excess of one over the other. Both are averaged over the seed draws of
    :func:`vote_group_draws` -- one run per model per draw, every seed used across the draws.

    Parameters
    ----------
    task : str
        Task name.
    mode : {"correct", "class"}
        What a vote counts: runs predicting the true label, or runs predicting 1.
    qa_mode : str
        QA mode to load.
    label_filter : int, optional
        Keep only the individuals with this true label.
    results_root : Path
        Root of the per-run prediction files.
    cache_dir : Path
        Directory of the cached values.
    refresh : bool
        Recompute even when the cache is newer than its inputs.
    **group_kwargs
        Passed on to :func:`vote_group_draws`.

    Returns
    -------
    pd.DataFrame
        One row per (group, k): ``observed``, ``null``, ``value`` (their difference), the run, draw
        and individual counts, ``task`` and ``qa_mode``.
    """
    from reasonance.agreement import vote_distribution, vote_distribution_null
    from reasonance.paper_figures import save_pair_values

    tag = label_tag(label_filter)
    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    cache = Path(cache_dir) / f"{task}-{qa_mode}-{tag}-vote-{mode}-pairs.csv"
    if not refresh and _cache_is_current(cache, aggregate_path, results_root):
        logging.info(f"using cached {cache.name} (newer than every input)")
        return pd.read_csv(cache)
    if cache.exists():
        logging.info(f"{cache.name} is older than its inputs; recomputing")

    task_data = load_task_data(
        task,
        results_root=results_root,
        qa_mode=qa_mode,
        aggregate_path=AGGREGATE_TEMPLATE.format(task=task),
        drop_models=EXCLUDED_MODELS,
        dedup_subset=["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode", "generation_seed"],
        label_filter=label_filter,
        common_only=False,
    )
    groups = vote_group_draws(task_data, **group_kwargs)
    # the individuals every run of every draw answered, so all draws count the same people
    index = common_individuals(task_data.predictions, [r for draws in groups.values() for ids in draws for r in ids])
    predictions, labels = task_data.predictions.loc[index], task_data.labels.loc[index]

    rows = []
    for group, draws in groups.items():
        observed = np.mean([vote_distribution(predictions, ids, labels=labels, mode=mode) for ids in draws], axis=0)
        null = np.mean([vote_distribution_null(predictions, ids, labels=labels, mode=mode) for ids in draws], axis=0)
        for k, (obs, exp) in enumerate(zip(observed, null)):
            rows.append(
                {
                    "group": group,
                    "k": k,
                    "n_runs": len(draws[0]),  # runs per distribution: one per model
                    "n_draws": len(draws),
                    "observed": obs,
                    "null": exp,
                    "value": obs - exp,  # the excess; "value" keeps the shared save/load contract
                    "metric": f"vote-{mode}",
                    "n_individuals": len(index),
                }
            )
    frame = pd.DataFrame(rows)
    # the identifying columns belong on the returned frame too, not only on the saved copy:
    # with refresh=True the caller gets this frame directly
    frame["task"] = task
    frame["qa_mode"] = qa_mode
    if label_filter is not None:
        frame["label_filter"] = label_filter
    # the label tag has to reach the file name, or a subgroup run overwrites the pooled cache
    save_pair_values(frame, task, qa_mode, kind=f"{tag}-vote-{mode}", root=Path(cache_dir))
    return frame


def vote_distributions_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """``vote_distributions`` for several tasks, concatenated.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`vote_distributions`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    frames = []
    for task in tasks:
        try:
            frames.append(vote_distributions(task, **kwargs))
        except (FileNotFoundError, IndexError, KeyError) as err:
            print(f"skipping {task}: {type(err).__name__}: {err}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Coverage — how many individuals the figures actually use, and what was lost
# ---------------------------------------------------------------------------
# Every figure restricts to the individuals EVERY run scored (`common_individuals`), so one run
# failing on a person removes that person from all runs. Reasoning models fail by spending the whole
# budget on the trace and never stating an answer, which is not missing at random -- it is the
# individuals the model deliberates longest about. The functions below report the size of that loss
# so it can be stated next to the figures.

ANSWER_SOURCE_COLUMN = "answer_source"
SOURCE_FORCED = "forced"
SOURCE_RETRY = "retry"
# Why a row was retried, written by folktexts' retry tool. "truncated" means the original run ran
# out of budget mid-thinking, never closed `</think>`, and the parser scraped a number out of the
# unfinished trace -- so the score the retry replaces was wrong, not missing, which is why these
# rows are reported separately from an ordinary gap-fill.
RETRY_REASON_COLUMN = "retry_reason"
REASON_TRUNCATED = "truncated"


def _predictions_pair(results_root: Path, rel_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """A run's predictions as the FIGURES load them, plus the pre-retry version for comparison.

    Both come from `utils.load_run_frame`, the same helper `load_predictions` uses -- coverage that
    re-implemented the read could report a different dataset from the one being plotted, which is
    worse than not reporting at all.

    Returns
    -------
    (pd.DataFrame or None, pd.DataFrame or None)
        The original predictions (None when the file is missing) and the retried ones (None when
        no retry has run).
    """
    from reasonance.utils import load_run_frame, retry_sibling

    path = Path(results_root) / rel_path
    if not path.exists():
        return None, None
    original = load_run_frame(path, kind="predictions", use_retries=False, index_col=0)
    retried = load_run_frame(path, kind="predictions", use_retries=True, index_col=0) if retry_sibling(path) else None
    return original, retried


# What the analysis itself keeps: every generation seed, not one representative per cell.
# `load_aggregate`'s own default collapses seeds, so coverage computed with it would describe a
# smaller run set than the figures use -- and a seed whose run has gaps would not show up at all.
ALL_SEEDS_DEDUP = ["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode", "generation_seed"]


def coverage_summary(
    tasks: Iterable[str] | None = None,
    qa_mode: str = "numeric",
    results_root: Path = RESULTS_MODELS_ROOT,
    aggregate_template: str = AGGREGATE_TEMPLATE,
    drop_models: Iterable[str] | None = EXCLUDED_MODELS,
    all_seeds: bool = True,
) -> pd.DataFrame:
    """Per task: the individuals the figures are computed on, and what was lost getting there.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    qa_mode : str
        QA mode to restrict the aggregate to.
    results_root : Path
        Root of the per-run prediction files.
    aggregate_template : str
        Path template for the per-task aggregate CSV.
    drop_models : iterable of str, optional
        Models left out, as in the figures; None keeps every model.
    all_seeds : bool
        Count every generation seed (True) or one run per cell (False).

    Returns
    -------
    pd.DataFrame
        One row per task, with the columns below.

    Columns
    -------
    runs, individuals
        Runs included, and individuals appearing in any of them. ``all_seeds=True`` (the default)
        counts every generation seed, matching what `effort_pairs` and `vote_distributions` load;
        ``False`` keeps one run per cell, as a figure that picks a single seed would.
    covered
        Individuals every run actually evaluated -- the effective size of the task. Smaller than
        ``individuals`` when runs used different test-set sizes: a 5000-row run beside a 16181-row
        one restricts everything to the 5000 (the subsamples are nested). That is the experiment
        design, not a loss, so completeness is measured against this number.
    used
        Individuals every run has an *answer* for -- the set every figure is computed on.
    complete_pct
        ``used`` as a percentage of ``covered``: how complete the task actually is.
    no_answer
        Dropped although every run covered them, because at least one run returned nothing. This is
        the failure mode: a reasoning model spends its whole budget on the trace and never states an
        answer, on the individuals it deliberates longest about -- so these are not missing at random.
    cells, retry_cells, truncated_cells, forced_cells
        Counted in CELLS -- one individual in one run -- not individuals. ``cells`` is the total
        (``used`` x ``runs``), so these are comparable against it and not against ``used``: a task
        with 112 runs has 112 cells per individual, which is why the raw counts exceed ``used``.
    truncated_cells
        Answers within ``used`` that replaced a score scraped from an unfinished thinking trace
        (``retry_reason == "truncated"``). Those originals were wrong rather than missing, so this
        counts corrections, not gap-fills -- and a run that still shows truncated rows has not had
        them corrected yet.
    retry_cells, forced_cells
        Answers **within ``used``** that came from a retry: re-queried under the original
        conditions (``retry_cells``), or obtained by closing the thinking trace and asking the model
        to commit (``forced_cells``). Counted in cells, so an individual recovered in two runs counts
        twice. ``forced`` is a different condition from the original run, so a figure about thinking
        length should exclude it; the count is reported so that choice is explicit.
    retry_pct, truncated_pct
        ``retry_cells`` and ``truncated_cells`` as a percentage of ``cells``.
    forced_individuals
        Distinct individuals in ``used`` whose answer in any run came from the forced pass. Below
        ``forced_cells`` when someone was forced in more than one run.
    without_retries
        What ``used`` would be if the retried answers were ignored -- the cost of not using them.

    Every column from ``used`` onwards describes the analysed set only; answers recovered for
    individuals that the intersection drops are not counted, since no figure sees them.
    """
    tasks = available_tasks(aggregate_template) if tasks is None else list(tasks)
    rows = []
    for task in tasks:
        aggregate = load_aggregate(
            Path(aggregate_template.format(task=task)),
            qa_mode=qa_mode,
            dedup_subset=ALL_SEEDS_DEDUP if all_seeds else None,
        )
        for model in drop_models or ():
            aggregate = aggregate.drop(aggregate[aggregate["model"] == model].index)

        covered = answered = answered_orig = None  # individuals a run saw / answered
        index = None
        n_runs = 0
        # Collected per run, tallied only once `used` is known: every count after `used`
        # describes the set the analysis actually runs on, not the whole results tree.
        retry_index: list = []
        forced_index: list = []
        truncated_index: list = []
        for _, row in aggregate.iterrows():
            original, retried = _predictions_pair(results_root, row["predictions_path"])
            if original is None:
                continue
            n_runs += 1
            index = original.index if index is None else index.union(original.index)

            seen = set(original.index)
            covered = seen if covered is None else covered & seen

            ok_orig = set(original.index[original["risk_score"].notna()])
            answered_orig = ok_orig if answered_orig is None else answered_orig & ok_orig

            latest = original if retried is None else retried
            ok = set(latest.index[latest["risk_score"].notna()])
            answered = ok if answered is None else answered & ok

            if retried is not None and ANSWER_SOURCE_COLUMN in retried.columns:
                source = retried[ANSWER_SOURCE_COLUMN]
                retry_index.append(retried.index[source == SOURCE_RETRY])
                forced_index.append(retried.index[source == SOURCE_FORCED])
                if RETRY_REASON_COLUMN in retried.columns:
                    replaced = (retried[RETRY_REASON_COLUMN] == REASON_TRUNCATED) & source.isin(
                        (SOURCE_RETRY, SOURCE_FORCED)
                    )
                    truncated_index.append(retried.index[replaced])

        if not n_runs:
            continue
        used = answered or set()
        # Rows, so an individual recovered in two runs counts twice -- that is how much of the
        # analysed data came from a retry. `forced_individuals` is the distinct-people version.
        retry_rows = sum(len([i for i in idx if i in used]) for idx in retry_index)
        forced_rows = sum(len([i for i in idx if i in used]) for idx in forced_index)
        truncated_rows = sum(len([i for i in idx if i in used]) for idx in truncated_index)
        forced_individuals = {i for idx in forced_index for i in idx if i in used}
        assert index is not None  # set by every counted run (n_runs > 0)
        rows.append(
            {
                "task": task,
                "runs": n_runs,
                "individuals": len(index),
                "covered": len(covered or ()),
                "used": len(used),
                "no_answer": len(covered or ()) - len(used),
                "complete_pct": round(100 * len(used) / max(1, len(covered or ())), 1),
                # `used` counts individuals; these count CELLS (one individual in one run), so
                # they are naturally larger than `used` and belong against `cells`, not against it.
                "cells": len(used) * n_runs,
                "retry_cells": retry_rows,
                "truncated_cells": truncated_rows,
                "forced_cells": forced_rows,
                "retry_pct": round(100 * retry_rows / max(1, len(used) * n_runs), 2),
                "truncated_pct": round(100 * truncated_rows / max(1, len(used) * n_runs), 2),
                "forced_individuals": len(forced_individuals),
                "without_retries": len(answered_orig or ()),
            }
        )
    return pd.DataFrame(rows)


def print_coverage(
    summary: pd.DataFrame | None = None, printer: Callable[[str], Any] = print, **kwargs: Any
) -> pd.DataFrame:
    """Print `coverage_summary` with the caveats a reader of the figures needs.

    Parameters
    ----------
    summary : pd.DataFrame, optional
        Output of :func:`coverage_summary`; computed from ``kwargs`` when None.
    printer : callable
        Called with each block of text, e.g. ``print`` or a logger method.
    **kwargs
        Passed on to :func:`coverage_summary` when ``summary`` is None.

    Returns
    -------
    pd.DataFrame
        The summary that was printed.
    """
    summary = coverage_summary(**kwargs) if summary is None else summary
    if summary.empty:
        printer("coverage: no tasks loaded")
        return summary

    printer("Individuals behind every figure in this notebook")
    printer(summary.to_string(index=False))
    printer(
        "\n  covered    = individuals every run evaluated -- the effective task size. Below\n"
        "               `individuals` when runs used different test-set sizes (a 5000-row run\n"
        "               beside a 16181-row one); the subsamples are nested, so this is the design,\n"
        "               not a loss. Completeness is measured against it.\n"
        "  used       = individuals every run has an ANSWER for; the figures use only these.\n"
        "  no_answer  = dropped although covered, because a run returned nothing. Reasoning models\n"
        "               run out of budget on the individuals they deliberate longest about, so\n"
        "               these are NOT missing at random.\n"
        "  *_cells    = one individual in ONE RUN, so they count against `cells` (= used x runs),\n"
        "               never against `used`: with 112 runs there are 112 cells per individual.\n"
        "  retry_cells= answers re-queried under the original conditions with a larger budget.\n"
        "  forced_cells = answers obtained by closing the thinking trace and asking the model to\n"
        "               commit. Counted in `used`, but a different condition from the original run\n"
        "               -- exclude them from any analysis of thinking length.\n"
        "  truncated_cells = scores that CORRECTED a value scraped from an unfinished trace, not\n"
        "               gap-fills. The originals were wrong (0.74 vs 0.84 accuracy where measured).\n"
        "  *_pct      = the matching *_cells as a percentage of `cells`.\n"
        "  forced_individuals = distinct individuals with a forced answer in any run."
    )
    gained = summary["used"] - summary["without_retries"]  # individuals, not cells
    if gained.gt(0).any():
        printer(
            f"\nRetried answers add {int(gained.sum())} individual(s) overall "
            f"({', '.join(f'{t}: +{g}' for t, g in zip(summary['task'], gained) if g)}). "
            f"Pass use_retries=False to load_task_data to reproduce figures made without them."
        )
    return summary


# One run per matched model on each side; gpt-5.x exists at three effort levels, so the length
# curve picks one rather than letting OpenAI contribute three runs. It is "high" -- the same level
# the effort and vote figures use -- so "reasoning on" means the same thing across the paper.
LENGTH_OPENAI_EFFORT = "high"


def _length_matched_runs(
    meta: pd.DataFrame, pool: Collection[str], openai_effort: str | None = LENGTH_OPENAI_EFFORT
) -> tuple[list[str], list[str]]:
    """``(reasoning run_ids, non-reasoning run_ids)`` holding the same base models, one run each.

    Thin wrapper over :func:`reasonance.utils.matched_sides`: both sides are drawn from ``pool``,
    which the caller has already restricted to the runs that have a length and a prediction.
    """
    reasoning = [run_id for run_id in pool if meta.set_index("run_id").loc[run_id, "reasoning"]]
    non_reasoning = [run_id for run_id in pool if run_id not in set(reasoning)]
    runs_on, runs_off, _ = matched_sides(meta, reasoning, non_reasoning, openai_effort=openai_effort)
    return runs_on, runs_off


def cached_reasoning_tokens(
    task: str,
    aggregate: pd.DataFrame,
    qa_mode: str = "numeric",
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Reasoning token counts (individuals x run), cached per task.

    Reading them means parsing one responses CSV per run -- about 23 seconds for a task -- and the
    length figures ask for the same matrix once per length definition and once per pair figure. The
    matrix itself is small (a few MB), so it is worth keeping on disk; the usual mtime check against
    the aggregate and the prediction files invalidates it.

    Parameters
    ----------
    task : str
        Task name; part of the cache file name.
    aggregate : pd.DataFrame
        Aggregate rows of the runs to read.
    qa_mode : str
        QA mode; part of the cache file name.
    results_root : Path
        Root of the per-run response files.
    cache_dir : Path
        Directory of the cached matrix.
    refresh : bool
        Re-read the responses even when the cache is newer than its inputs.

    Returns
    -------
    pd.DataFrame
        Reasoning token counts, individuals x run_id.
    """
    from reasonance.utils import load_reasoning_tokens

    cache = Path(cache_dir) / f"{task}-{qa_mode}-reasoning-tokens.parquet"
    if not refresh and _cache_is_current(cache, Path(AGGREGATE_TEMPLATE.format(task=task)), results_root):
        logging.info(f"using cached {cache.name} (newer than every input)")
        return pd.read_parquet(cache)
    tokens = load_reasoning_tokens(aggregate, results_root, question_idx=0)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(cache)
    return tokens


def _length_setup(
    task: str,
    qa_mode: str,
    openai_effort: str | None,
    levels: Iterable[str] | None,
    results_root: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Everything the length figures share before a panel is drawn: runs, individuals, tokens.

    Shared by :func:`length_curves` and :func:`vote_distributions_by_length`, so every length figure
    bins the SAME individuals on the SAME length -- the vote distribution by length must fall on
    exactly the bins of the agreement curve.

    Parameters
    ----------
    task, qa_mode : str
        What to load.
    openai_effort : str or None
        Effort level that stands for "reasoning on" for gpt-5.x in the matched groups.
    levels : iterable of str or None
        Effort levels whose runs of the matched models are kept for the level curves.
    results_root, cache_dir : Path
        Root of the per-run files, and directory of the cached token matrix.

    Returns
    -------
    dict
        ``task``, ``predictions``, ``labels``, ``tokens`` (restricted to the individuals every kept
        run answered), ``token_meta``, ``by_run``, ``family_of``, ``model_of``, ``setting`` (the
        run -> (model, effort) function), ``reasoning_ids``, ``non_reasoning_ids``, ``pool_on``,
        ``pool_off`` and ``pool_levels``.
    """
    from reasonance.utils import build_run_metadata

    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    task_data = load_task_data(
        task,
        results_root=results_root,
        qa_mode=qa_mode,
        aggregate_path=str(aggregate_path),
        drop_models=EXCLUDED_MODELS,
        dedup_subset=ALL_SEEDS_DEDUP,
        common_only=False,
    )
    tokens = cached_reasoning_tokens(task, task_data.aggregate, qa_mode, results_root, cache_dir)
    token_meta = build_run_metadata(task_data.aggregate, tokens)
    # the curve needs a prediction AND a length for the same individual
    index = task_data.predictions.index.intersection(tokens.index)
    predictions, labels = task_data.predictions.loc[index], task_data.labels.loc[index]
    tokens = tokens.loc[index]

    reasoning_ids, non_reasoning_ids = _length_matched_runs(
        token_meta[token_meta["run_id"].isin(predictions.columns)], set(tokens.columns), openai_effort
    )
    by_run = token_meta.set_index("run_id")
    family_of, model_of = by_run["family"], by_run["model"]

    # matched_sides picks one run per model; the curves want every SEED of those settings, because
    # they average the seeds themselves (a model pair's components over its cross-seed run pairs)
    def _setting(run_id: str) -> tuple[str, str] | None:
        """(model, effort) for a run, or None for one the token matrix does not cover.

        `predictions` carries every run of the task, including models with no reasoning traces
        (Claude), which are absent from the token metadata.
        """
        if run_id not in by_run.index:
            return None
        return (model_of[run_id], str(by_run.loc[run_id, "reasoning_effort"]))

    settings_on = {_setting(run_id) for run_id in reasoning_ids}
    settings_off = {_setting(run_id) for run_id in non_reasoning_ids}
    pool_on = [r for r in predictions.columns if r in set(tokens.columns) and _setting(r) in settings_on]
    pool_off = [r for r in predictions.columns if _setting(r) in settings_off]  # None never matches

    # Runs of the matched models at the other effort levels: they are not part of the matched
    # groups (which hold one level per model), but the level curves need them, so they must survive
    # the restriction below.
    # reasoning_ids come from token_meta, so _setting is never None for them
    models_on = {model for model, _effort in settings_on}  # type: ignore[misc]
    pool_levels = [
        run_id
        for run_id in predictions.columns
        if run_id in set(tokens.columns)
        and model_of.get(run_id) in models_on
        and str(by_run.loc[run_id, "reasoning_effort"]) in set(levels or ())
        and run_id not in set(pool_on)
    ]

    # the agreement metrics need a 0/1 prediction from every run in the figure, so the curves are
    # computed on the individuals ALL matched runs answered -- the same restriction as elsewhere
    used = pool_on + pool_off + pool_levels
    index = common_individuals(predictions, used)
    predictions, labels, tokens = predictions.loc[index, used], labels.loc[index], tokens.loc[index]

    return {
        "task": task,
        "predictions": predictions,
        "labels": labels,
        "tokens": tokens,
        "token_meta": token_meta,
        "by_run": by_run,
        "family_of": family_of,
        "model_of": model_of,
        "setting": _setting,
        "reasoning_ids": reasoning_ids,
        "non_reasoning_ids": non_reasoning_ids,
        "pool_on": pool_on,
        "pool_off": pool_off,
        "pool_levels": pool_levels,
    }


def _length_panel(setup: Mapping[str, Any], family: str, variant: str, n_bins: int) -> dict[str, Any] | None:
    """One panel's balanced runs, length axis and bins; None when a side has fewer than two models.

    The single definition of a length bin: :func:`length_curves` and
    :func:`vote_distributions_by_length` both call it, so their bins are identical by construction.

    Parameters
    ----------
    setup : mapping
        Output of :func:`_length_setup`.
    family : str
        Family to restrict to, or "All" for every family.
    variant : str
        Length definition, a key of ``reasonance.utils.LENGTH_VARIANTS``.
    n_bins : int
        Quantile bins of individuals.

    Returns
    -------
    dict or None
        ``r_ids``, ``nr_ids`` (seed-balanced run ids), ``n_seeds``, ``spec`` (the length variant),
        ``length`` (per-individual series) and ``bins`` (its quantile bins).
    """
    from reasonance.utils import build_length_variants

    token_meta, tokens = setup["token_meta"], setup["tokens"]
    family_of, model_of = setup["family_of"], setup["model_of"]
    pool_on, pool_off = setup["pool_on"], setup["pool_off"]
    keep = lambda run_id: family == "All" or family_of[run_id] == family  # noqa: E731
    r_ids = [r for r in pool_on if keep(r)]
    nr_ids = [r for r in pool_off if keep(r)]
    if len({model_of[r] for r in r_ids}) < 2 or len({model_of[r] for r in nr_ids}) < 2:
        logging.warning("%s/%s: fewer than two models on a side, skipped", setup["task"], family)
        return None
    # Balanced WITHIN this comparison: every model contributes the same number of seeds, the
    # lowest any of them has here -- so an OpenAI-only panel can use more than a pooled one.
    (r_ids, nr_ids), n_seeds = balance_seeds(token_meta, r_ids, nr_ids)

    # the length axis is built from THIS panel's reasoning runs: a z-score is only meaningful
    # against the runs it is computed over
    spec = build_length_variants(tokens, r_ids, [variant])[variant]
    length = spec["series"].dropna()
    bins = pd.qcut(length, q=n_bins, duplicates="drop")
    return {"r_ids": r_ids, "nr_ids": nr_ids, "n_seeds": n_seeds, "spec": spec, "length": length, "bins": bins}


def length_curves(
    task: str,
    metric: str = "acc_adjusted_agree",
    qa_mode: str = "numeric",
    variant: str = "zscore",
    n_bins: int = 20,
    min_bin: int = 30,
    families: Iterable[str] = ("All", "OpenAI", "Qwen"),
    levels: Collection[str] = ("low", "medium", "high"),
    seed_draws: bool = True,
    openai_effort: str = LENGTH_OPENAI_EFFORT,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Agreement as a function of how long the models actually thought.

    Individuals are binned by their reasoning length (``variant``: raw median tokens, median
    z-score within run, or median percentile rank), and within each bin the mean pairwise agreement
    is computed -- once over the matched reasoning runs and once over their non-reasoning
    counterparts. The non-reasoning runs have no length of their own, so they are measured on the
    SAME individuals per bin and simply borrow the x: the vertical gap in a bin is then the
    reasoning effect at that difficulty level, not a comparison of different people.

    Parameters
    ----------
    task : str
        Task name.
    metric : str
        Agreement metric, as in the effort figures.
    qa_mode : str
        QA mode to load.
    variant : {"raw", "zscore", "rank"}
        Definition of "reasoning length" on the x axis. Raw tokens are not comparable between
        families (Qwen thinks in thousands, OpenAI in hundreds), which is what the z-score fixes.
    n_bins : int
        Quantile bins of individuals.
    min_bin : int
        Bins holding fewer individuals than this are dropped.
    families : iterable of str
        Panels to compute: "All" pools the developers, the others restrict to one.
    levels : iterable of str
        Effort levels to emit as their own curves, for the models that have ALL of them (only the
        OpenAI models do). They come from a wider pool than the matched groups -- which hold one
        level per model -- so they are labelled by level and sit on the panel's own bins, which
        makes them comparable at a fixed realized length.
    seed_draws : bool
        Also emit one curve per seed rank (``draw`` 1..k, each holding every model once), which a
        figure can draw as thin lines behind the main curve (``draw = 0``).
    openai_effort : str
        Effort level that stands for "reasoning on" for gpt-5.x in the matched groups.
    results_root : Path
        Root of the per-run files.
    cache_dir : Path
        Directory of the cached curves and token matrix.
    refresh : bool
        Recompute even when the cache is newer than its inputs.

    Returns
    -------
    pd.DataFrame
        One row per (family, group, bin): ``task``, ``family``, ``group``, ``x`` (the bin's median
        length), ``bin_left``, ``bin_right``, ``n``, ``value``, plus ``metric``, ``variant``,
        ``x_label`` (the variant's short axis label, for exploratory plots -- the paper figures
        label the axis themselves) and the run counts.
    """
    from reasonance.agreement import agreement_metric_by_length

    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    cache = Path(cache_dir) / f"{task}-{qa_mode}-{metric}-{variant}-length-curves.csv"
    if not refresh and _cache_is_current(cache, aggregate_path, results_root):
        logging.info(f"using cached {cache.name} (newer than every input)")
        return pd.read_csv(cache)

    setup = _length_setup(task, qa_mode, openai_effort, levels, results_root, cache_dir)
    predictions, labels = setup["predictions"], setup["labels"]
    token_meta, by_run = setup["token_meta"], setup["by_run"]
    family_of, model_of, _setting = setup["family_of"], setup["model_of"], setup["setting"]
    pool_on, pool_levels = setup["pool_on"], setup["pool_levels"]

    rows: list[dict[str, Any]] = []
    for family in families:
        panel = _length_panel(setup, family, variant, n_bins)
        if panel is None:
            continue
        r_ids, nr_ids, n_seeds = panel["r_ids"], panel["nr_ids"], panel["n_seeds"]
        spec, length, bins = panel["spec"], panel["length"], panel["bins"]

        def _bin_accuracy(ids: list[str], bins: pd.Series = bins) -> dict[float, tuple[float, float]]:
            """Per bin: the mean over the runs of their accuracy on that bin's individuals."""
            correct = predictions.loc[bins.index, ids].eq(labels.loc[bins.index], axis=0)
            per_run = correct.groupby(bins, observed=True).mean()  # bin x run
            return {
                interval.left: (float(mean), float(std))
                for interval, mean, std in zip(per_run.index, per_run.mean(axis=1), per_run.std(axis=1))
            }

        def _emit(
            ids: list[str],
            group: str,
            draw: int,
            length: pd.Series = length,
            bins: pd.Series = bins,
            spec: dict[str, Any] = spec,
            family: str = family,
            n_seeds: int = n_seeds,
        ) -> None:
            """Append one curve's rows (agreement and accuracy per bin) for the runs ``ids``."""
            # accuracy on the SAME bins, so it can be overlaid on the agreement curve: the two
            # questions -- do they agree, are they right -- are only comparable on one binning
            accuracy = _bin_accuracy(ids)
            curve, _ = agreement_metric_by_length(
                run_ids=ids,
                preds=predictions,
                length=length,
                metric=metric,
                labels=labels,
                bins=bins,
                min_bin=min_bin,
                # seeds of one model are runs of the same thing: average a pair's cross-seed
                # combinations, and never count a same-model pair as cross-model agreement
                model_of_run={run_id: model_of[run_id] for run_id in ids},
            )
            rows.extend(
                {
                    "task": task,
                    "qa_mode": qa_mode,
                    "metric": metric,
                    "variant": variant,
                    "x_label": spec["short_label"],
                    "family": family,
                    "group": group,
                    "draw": draw,  # 0 = every seed together, 1..k = one seed per model
                    "x": row.median_tokens,
                    "bin_left": row.bin_left,
                    "bin_right": row.bin_right,
                    "n": row.n,
                    "value": row.mean_value,
                    "accuracy": accuracy.get(row.bin_left, (np.nan, np.nan))[0],
                    "accuracy_std": accuracy.get(row.bin_left, (np.nan, np.nan))[1],
                    "n_runs": len(ids),
                    "n_models": len({model_of[run_id] for run_id in ids}),
                    "n_seeds": n_seeds,
                }
                for row in curve.itertuples()
            )

        # Effort levels of the models that have every one of them: a different population from the
        # matched groups, drawn on the SAME bins so a level can be read at a fixed realized length.
        level_ids = {}
        if levels:
            candidates: dict[str, dict[str, list[str]]] = {}
            for run_id in [*pool_on, *pool_levels]:
                effort = str(by_run.loc[run_id, "reasoning_effort"])
                if effort in set(levels) and (family == "All" or family_of.get(run_id) == family):
                    candidates.setdefault(model_of[run_id], {}).setdefault(effort, []).append(run_id)
            complete = [model for model, by_level in candidates.items() if set(by_level) >= set(levels)]
            if len(complete) >= 2:
                level_ids = {
                    level: [run_id for model in complete for run_id in candidates[model][level]] for level in levels
                }
                (balanced := list(balance_seeds(token_meta, *level_ids.values())[0]))
                level_ids = dict(zip(level_ids, balanced))

        for group, ids in [("reasoning", r_ids), ("non-reasoning", nr_ids), *level_ids.items()]:
            _emit(ids, group, draw=0)
            if not seed_draws:
                continue
            # Draw r takes each model's r-th seed, so every draw holds ALL the models and the draws
            # are disjoint replicates. Grouping by seed NUMBER instead would give draws of different
            # composition, since the models were run at different seed numbers.
            by_setting: dict[tuple, list[str]] = {}
            for run_id in ids:
                by_setting.setdefault(_setting(run_id), []).append(run_id)
            for runs in by_setting.values():
                runs.sort(key=lambda r: (by_run.loc[r, "generation_seed"], str(r)))
            for rank in range(min((len(runs) for runs in by_setting.values()), default=0)):
                draw_ids = [runs[rank] for runs in by_setting.values()]
                if len({model_of[run_id] for run_id in draw_ids}) >= 2:
                    _emit(draw_ids, group, draw=rank + 1)

    frame = pd.DataFrame(rows)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def pair_length_curves(
    task: str,
    metric: str = "acc_adjusted_agree",
    qa_mode: str = "numeric",
    variant: str = "rank",
    n_bins: int = 20,
    min_bin: int = 50,
    include_non_reasoning: bool = True,
    levels: Iterable[str] = (),
    openai_effort: str = LENGTH_OPENAI_EFFORT,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Agreement vs length for EACH MODEL PAIR, binned by the length that pair itself produced.

    The group curves bin individuals by a median length over many runs, which is a consensus
    difficulty proxy. Here the x is the pair's own doing: an individual's length is the mean tokens
    the pair's two reasoning runs spent on it, and the bins are quantiles of that. The matched
    non-reasoning pair (same two models, reasoning off) is measured on the SAME individuals per bin,
    so each pair is one clean matched comparison along its own length.

    Parameters
    ----------
    task : str
        Task name.
    metric : str
        Agreement metric.
    qa_mode : str
        QA mode to load.
    variant : {"rank", "raw"}
        Only where the bins are DRAWN; the binning itself is always per pair. ``"raw"`` puts each
        bin at its median token count, so pairs sit at their own token scale and the lines pull
        apart horizontally. ``"rank"`` puts it at the mean percentile rank of the individuals in
        it, so every pair covers 0-1 and the bin positions coincide -- the positions, not the
        individuals: bin b is each pair's own b-th length quantile, over its own people.
    include_non_reasoning : bool
        Also emit each pair's reasoning-off counterpart on the same bins.
    levels : iterable of str
        Extra effort levels to emit for the models that have them (e.g. ``("low", "medium")``), each
        binned by the length THAT level produced. Leaving them out keeps one line per pair.
    n_bins : int
        Quantile bins of the pair's own length.
    min_bin : int
        A single pair is noisier than a group, so bins below this many individuals are dropped.
    openai_effort : str
        Effort level that stands for "reasoning on" for gpt-5.x.
    results_root : Path
        Root of the per-run files.
    cache_dir : Path
        Directory of the cached curves and token matrix.
    refresh : bool
        Recompute even when the cache is newer than its inputs.

    Returns
    -------
    pd.DataFrame
        One row per (pair, group, bin): ``task``, ``family``, ``pair``, ``model_a``, ``model_b``,
        ``group``, ``x``, ``bin_left``, ``bin_right``, ``n``, ``value``, ``n_runs``.
    """
    from reasonance.agreement import agreement_metric_by_length
    from reasonance.utils import build_run_metadata

    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    cache = Path(cache_dir) / f"{task}-{qa_mode}-{metric}-{variant}-pair-length-curves.csv"
    if not refresh and _cache_is_current(cache, aggregate_path, results_root):
        logging.info(f"using cached {cache.name} (newer than every input)")
        return pd.read_csv(cache)

    task_data = load_task_data(
        task,
        results_root=results_root,
        qa_mode=qa_mode,
        aggregate_path=str(aggregate_path),
        drop_models=EXCLUDED_MODELS,
        dedup_subset=ALL_SEEDS_DEDUP,
        common_only=False,
    )
    tokens = cached_reasoning_tokens(task, task_data.aggregate, qa_mode, results_root, cache_dir)
    meta = build_run_metadata(task_data.aggregate, tokens)
    by_run = meta.set_index("run_id")
    keep_models = matched_models()

    def runs_for_cell(model: str, effort: str | None, reasoning: bool) -> list[str]:
        """Runs of one (model, effort) cell that have both predictions and token counts."""
        ids = [
            run_id
            for run_id in meta.loc[meta["reasoning"] == reasoning, "run_id"]
            if by_run.loc[run_id, "model"] == model
            and str(by_run.loc[run_id, "reasoning_effort"]) == str(effort)
            and run_id in task_data.predictions.columns
            and run_id in tokens.columns
        ]
        return ids

    # the reasoning cell each model is compared at, and its reasoning-off counterpart
    cells = {}
    for model_off, effort_off, model_on, effort_on in MATCHED_PAIR_SPECS:
        if str(model_on).startswith("gpt-5") and str(effort_on) != openai_effort:
            continue
        on, off = runs_for_cell(model_on, effort_on, True), runs_for_cell(model_off, effort_off, False)
        if on and off and model_on in keep_models:
            cells[model_on] = {"on": on, "off": off, "family": by_run.loc[on[0], "family"]}

    rows: list[dict[str, Any]] = []
    for family in sorted({cell["family"] for cell in cells.values()}):
        models = sorted([m for m, cell in cells.items() if cell["family"] == family], key=model_sort_key)
        for model_a, model_b in itertools.combinations(models, 2):
            groups = {"reasoning": (cells[model_a]["on"], cells[model_b]["on"])}
            if include_non_reasoning:
                groups["non-reasoning"] = (cells[model_a]["off"], cells[model_b]["off"])
            for level in levels:
                runs_a, runs_b = runs_for_cell(model_a, level, True), runs_for_cell(model_b, level, True)
                if runs_a and runs_b:
                    groups[level] = (runs_a, runs_b)

            # the pair's own length defines the bins; the off counterpart borrows them, so a bin
            # holds the same individuals on both sides
            (runs_a, runs_b), _ = balance_seeds(meta, *groups["reasoning"])
            length = tokens[list(runs_a) + list(runs_b)].mean(axis=1).dropna()
            index = common_individuals(task_data.predictions, [r for pair in groups.values() for g in pair for r in g])
            index = index.intersection(length.index)
            if len(index) < min_bin * 2:
                continue
            length = length.loc[index]
            bins = pd.qcut(length, q=n_bins, duplicates="drop")
            # percentile position of each bin, MEASURED: ties merge bins and small bins are dropped,
            # so the nominal (k + 0.5) / n_bins position can be wrong
            rank_of_bin = length.rank(pct=True).groupby(bins, observed=True).mean()
            rank_by_left = {interval.left: value for interval, value in rank_of_bin.items()}

            for group, (runs_a, runs_b) in groups.items():
                (runs_a, runs_b), _ = balance_seeds(meta, runs_a, runs_b)
                ids = list(runs_a) + list(runs_b)
                curve, _ = agreement_metric_by_length(
                    run_ids=ids,
                    preds=task_data.predictions.loc[index],
                    length=length,
                    metric=metric,
                    labels=task_data.labels.loc[index],
                    bins=bins,
                    min_bin=min_bin,
                    model_of_run={run_id: by_run.loc[run_id, "model"] for run_id in ids},
                )
                rows.extend(
                    {
                        "task": task,
                        "qa_mode": qa_mode,
                        "metric": metric,
                        "variant": variant,
                        "family": family,
                        "pair": f"{pretty_model_name(model_a)} + {pretty_model_name(model_b)}",
                        "model_a": model_a,
                        "model_b": model_b,
                        "group": group,
                        "x": row.median_tokens if variant == "raw" else rank_by_left.get(row.bin_left, np.nan),
                        "bin_left": row.bin_left,
                        "bin_right": row.bin_right,
                        "n": row.n,
                        "value": row.mean_value,
                        "n_runs": len(ids),
                    }
                    for row in curve.itertuples()
                )

    frame = pd.DataFrame(rows)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def pair_length_curves_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """``pair_length_curves`` for several tasks, concatenated.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`pair_length_curves`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    return pd.concat([pair_length_curves(task, **kwargs) for task in tasks], ignore_index=True)


def length_curves_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """``length_curves`` for several tasks, concatenated.

    A task whose files are missing is skipped with a warning; none usable raises ``RuntimeError``.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`length_curves`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    frames = []
    for task in tasks:
        try:
            frames.append(length_curves(task, **kwargs))
        except FileNotFoundError as err:  # a task whose files are not there yet, not a bug
            logging.warning("skipping %s: %s", task, err)
    if not frames:
        raise RuntimeError(f"no usable tasks in {tasks}")
    if len(frames) < len(tasks):  # a figure built on fewer tasks must not look like the full one
        logging.warning("length curves cover %d of %d tasks", len(frames), len(tasks))
    return pd.concat(frames, ignore_index=True)


def reasoning_length_histograms(
    tasks: Iterable[str] | None = None,
    qa_mode: str = "numeric",
    bin_width: int = 50,
    max_tokens: int = 10000,
    matched: bool = True,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Realised reasoning length per (task, model, effort setting), as a histogram.

    The effort setting is what we ASK for; the number of reasoning tokens is what we GET. They are
    not the same thing -- a model can ignore the setting, saturate at its budget, or return an
    empty trace -- so every statement about "more reasoning" needs this check.

    Only the binned counts are kept, not the raw token counts: histograms need nothing else, and
    the result caches to one CSV per task (reading the traces takes ~15s per task). The bins are
    fine and fixed, so a figure can still truncate or rebin without recomputing.

    Parameters
    ----------
    tasks : list of str, optional
        Defaults to every task with an aggregate.
    qa_mode : str
        QA mode to load.
    bin_width, max_tokens : int
        Bin width and upper edge of the binned range; anything above lands in the last bin, and
        ``max_observed`` reports how far the tail actually reached.
    matched : bool
        Restrict to the models that take part in a matched pair, as the other figures do.
    results_root : Path
        Root of the per-run response files.
    cache_dir : Path
        Directory of the cached histograms and token matrix.
    refresh : bool
        Recompute the curves even when their cache is newer than its inputs. The token matrix keeps
        its own mtime check, so this does not force the slow re-read of every responses file.

    Returns
    -------
    pd.DataFrame
        Columns ``task``, ``model``, ``model_label``, ``family``, ``reasoning_effort``,
        ``effort_label``, ``bin_left``, ``fraction``, ``n_runs``, ``n_values``, ``zero_share``,
        ``median``, ``max_observed``.
    """
    from reasonance.utils import build_run_metadata

    tasks = available_tasks() if tasks is None else list(tasks)
    keep_models = matched_models()  # shared definition, see reasonance.utils
    edges = np.arange(0, max_tokens + bin_width, bin_width)

    frames = []
    for task in tasks:
        aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
        cache = Path(cache_dir) / f"{task}-{qa_mode}-reasoning-length-histograms.csv"
        if not refresh and _cache_is_current(cache, aggregate_path, results_root):
            logging.info(f"using cached {cache.name} (newer than every input)")
            frames.append(pd.read_csv(cache))
            continue

        aggregate = load_aggregate(aggregate_path, qa_mode=qa_mode, dedup_subset=ALL_SEEDS_DEDUP)
        if matched:
            aggregate = aggregate[aggregate["model"].isin(keep_models)]
        tokens = cached_reasoning_tokens(task, aggregate, qa_mode, results_root, cache_dir)
        meta = build_run_metadata(aggregate, tokens)

        rows: list[dict[str, Any]] = []
        for (model, effort), group in meta.groupby(["model", "reasoning_effort"], dropna=False):
            run_ids = [run_id for run_id in group["run_id"] if run_id in tokens.columns]
            values = tokens[run_ids].to_numpy().ravel()
            values = values[~np.isnan(values)]
            if not len(values):
                continue
            # everything above the last edge is folded into the last bin, so the fractions still sum to 1
            counts, _ = np.histogram(np.clip(values, 0, edges[-1] - 1e-9), bins=edges)
            shared = {
                "task": task,
                "qa_mode": qa_mode,
                "model": model,
                "model_label": pretty_model_name(model),
                "family": group["family"].iloc[0],
                "reasoning_effort": str(effort),
                "effort_label": effort_display(effort),
                "n_runs": len(run_ids),
                "n_values": int(len(values)),
                # asked to think and did not: a trace of length zero
                "zero_share": float((values == 0).mean()),
                "median": float(np.median(values)),
                "max_observed": float(values.max()),
            }
            rows.extend(
                {**shared, "bin_left": float(left), "fraction": float(count / counts.sum())}
                for left, count in zip(edges[:-1], counts)
            )

        frame = pd.DataFrame(rows)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["task", "model"], key=lambda col: col.map(model_sort_key) if col.name == "model" else col)


def risk_score_histograms(
    tasks: Iterable[str] | None = None,
    qa_mode: str = "numeric",
    n_bins: int = 20,
    matched: bool = True,
    openai_effort_keep: Collection[str] | None = OPENAI_EFFORT_KEEP,
    include_pooled: bool = True,
    results_root: Path = RESULTS_MODELS_ROOT,
) -> pd.DataFrame:
    """Distribution of the predicted risk scores per task and model, reasoning off and on.

    Numeric QA asks for a probability rather than a class, so the risk scores are the raw material
    of every downstream number: how confident the models are, and whether reasoning changes that.
    Each run is turned into a histogram normalised to fractions and the runs behind one cell are
    then AVERAGED -- over the seeds of a model, or over every run of a group for ``model="All"`` --
    so a run with more answered individuals does not weigh more than the others.

    Parameters
    ----------
    tasks : list of str, optional
        Defaults to every task with an aggregate.
    qa_mode : str
        QA mode to load.
    n_bins : int
        Bins over [0, 1].
    matched : bool
        Keep only runs whose (model, effort) cell has a reasoning counterpart, so both groups hold
        the same base models -- the same restriction the agreement figures use.
    openai_effort_keep : set of str or None
        Effort levels used for OpenAI's off/on comparison, as in the agreement figures: keeping
        every level would put three times as many reasoning runs as non-reasoning ones in the
        average. None keeps them all.
    include_pooled : bool
        Also emit ``model="All"``, the average over every matched run of the group.
    results_root : Path
        Root of the per-run prediction files.

    Returns
    -------
    pd.DataFrame
        Columns ``task``, ``model`` ("All" pools them), ``model_label``, ``family``, ``group``,
        ``bin_left``, ``bin_centre``, ``fraction``, ``n_runs``.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_data = load_task_data(
            task,
            results_root=results_root,
            qa_mode=qa_mode,
            aggregate_path=AGGREGATE_TEMPLATE.format(task=task),
            drop_models=EXCLUDED_MODELS,
            dedup_subset=ALL_SEEDS_DEDUP,
            common_only=False,
        )
        meta = task_data.meta
        if openai_effort_keep is not None:
            other_levels = meta[
                (meta["family"] == "OpenAI") & ~meta["reasoning_effort"].astype(str).isin(openai_effort_keep)
            ]
            meta = meta.drop(other_levels.index)
        keep = matched_run_ids(meta) if matched else set(meta["run_id"])
        by_run = meta.set_index("run_id")
        keep &= set(meta["run_id"])
        models = sorted(set(meta.loc[meta["run_id"].isin(keep), "model"]), key=model_sort_key)
        for group in ("reasoning", "non-reasoning"):
            # one distribution per MODEL (averaged over its seeds), and optionally one pooling them
            for model in [*models, "All"] if include_pooled else models:
                run_ids = [
                    run_id
                    for run_id in task_data.risk_scores.columns
                    if run_id in keep
                    and ("reasoning" if by_run.loc[run_id, "reasoning"] else "non-reasoning") == group
                    and (model == "All" or by_run.loc[run_id, "model"] == model)
                ]
                shares = []
                for run_id in run_ids:
                    scores = task_data.risk_scores[run_id].dropna()
                    if scores.empty:
                        continue
                    counts, _ = np.histogram(scores, bins=edges)
                    shares.append(counts / counts.sum())
                if not shares:
                    continue
                mean_share = np.mean(shares, axis=0)
                family = "All" if model == "All" else by_run.loc[run_ids[0], "family"]
                rows.extend(
                    {
                        "task": task,
                        "qa_mode": qa_mode,
                        "model": model,
                        "model_label": "All" if model == "All" else pretty_model_name(model),
                        "family": family,
                        "group": group,
                        "bin_left": float(left),
                        "bin_centre": float(centre),
                        "fraction": float(value),
                        "n_runs": len(shares),
                    }
                    for left, centre, value in zip(edges[:-1], centres, mean_share)
                )
    return pd.DataFrame(rows)


def vote_transitions(
    task: str,
    qa_mode: str = "numeric",
    label_filter: int | None = None,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
    **group_kwargs: Any,
) -> pd.DataFrame:
    """Where each individual sits on the unanimity ladder with reasoning off vs on.

    For every individual, count how many of the matched models predict the true label with reasoning
    off (``k_off``) and with reasoning on (``k_on``); the same models on both sides, one run each.
    This is done per seed draw (draw r off against draw r on, see :func:`vote_group_draws`) and the
    shares are averaged over the draws. Returns one row per (k_off, k_on) cell, so the mass above the
    diagonal is the individuals the ecosystem got right more unanimously, and the mass below it is
    those it got wrong more unanimously.

    Parameters
    ----------
    task : str
        Task name.
    qa_mode : str
        QA mode to load.
    label_filter : int, optional
        Keep only the individuals with this true label.
    results_root : Path
        Root of the per-run prediction files.
    cache_dir : Path
        Directory of the cached table.
    refresh : bool
        Recompute even when the cache is newer than its inputs.
    **group_kwargs
        Passed on to :func:`vote_group_draws`.

    Returns
    -------
    pd.DataFrame
        One row per (``k_off``, ``k_on``): ``n_individuals_in_cell`` (mean over draws), ``value``
        (its share of ``n_individuals``), and the run, draw and individual counts.
    """
    tag = label_tag(label_filter)
    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    cache = Path(cache_dir) / f"{task}-{qa_mode}-{tag}-vote-transitions-pairs.csv"
    if not refresh and _cache_is_current(cache, aggregate_path, results_root):
        return pd.read_csv(cache)

    task_data = load_task_data(
        task,
        results_root=results_root,
        qa_mode=qa_mode,
        aggregate_path=str(aggregate_path),
        drop_models=(),
        dedup_subset=["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode", "generation_seed"],
        label_filter=label_filter,
        common_only=False,
    )
    groups = vote_group_draws(task_data, **group_kwargs)
    index = common_individuals(task_data.predictions, [r for draws in groups.values() for ids in draws for r in ids])
    predictions, labels = task_data.predictions.loc[index], task_data.labels.loc[index]

    def correct_count(run_ids: list[str]) -> pd.Series:
        """Per individual: how many of ``run_ids`` predict the true label."""
        return predictions[run_ids].eq(labels, axis=0).sum(axis=1).astype(int)

    # one move table per draw (the off side's r-th draw against the on side's r-th), then averaged:
    # a cell's count is the mean number of individuals making that move across the draws
    per_draw = [
        pd.DataFrame({"k_off": correct_count(off), "k_on": correct_count(on)}).groupby(["k_off", "k_on"]).size()
        for off, on in zip(groups["non-reasoning"], groups["reasoning"])
    ]
    cells = pd.concat(per_draw, axis=1).fillna(0).mean(axis=1)
    table = (
        cells.rename("n_individuals_in_cell")
        .reset_index()
        .assign(
            task=task,
            qa_mode=qa_mode,
            n_runs=len(groups["reasoning"][0]),
            n_draws=len(per_draw),
            n_individuals=len(index),
            metric="vote-transitions",
            value=lambda d: d["n_individuals_in_cell"] / len(index),
        )
    )
    save_path = Path(cache_dir) / cache.name
    table.to_csv(save_path, index=False)
    return table


def vote_transitions_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """``vote_transitions`` for several tasks, concatenated.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`vote_transitions`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    return pd.concat([vote_transitions(task, **kwargs) for task in tasks], ignore_index=True)


def model_accuracy(
    tasks: Iterable[str] | None = None,
    qa_mode: str = "numeric",
    aggregate_template: str = AGGREGATE_TEMPLATE,
    results_root: Path = RESULTS_MODELS_ROOT,
) -> pd.DataFrame:
    """Accuracy of every matched model on every task, averaged over generation seeds.

    One row per (task, model, effort): the mean over seeds, the spread across them, and the number
    of seeds behind it. Only models that take part in a matched pair are kept -- the others have no
    reasoning counterpart, so they say nothing about what reasoning does.

    Each task also carries its XGBoost baseline, computed on the individuals that task's runs were
    actually scored on, so the reference is comparable to the model dots rather than to the full
    test set.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    qa_mode : str
        QA mode to restrict the aggregate to.
    aggregate_template : str
        Path template for the per-task aggregate CSV.
    results_root : Path
        Root of the per-run prediction files, used for the baseline's individual index.

    Returns
    -------
    pd.DataFrame
        Columns ``task``, ``model``, ``model_label``, ``family``, ``reasoning_effort``,
        ``effort_label``, ``accuracy_mean``, ``accuracy_std``, ``n_seeds``, ``baseline_accuracy``,
        ``qa_mode``, sorted by family and release order within a task.
    """
    tasks = available_tasks(aggregate_template) if tasks is None else list(tasks)
    keep_models = matched_models()  # shared definition, see reasonance.utils

    frames = []
    for task in tasks:
        aggregate = load_aggregate(
            Path(aggregate_template.format(task=task)), qa_mode=qa_mode, dedup_subset=ALL_SEEDS_DEDUP
        )
        aggregate = aggregate[aggregate["model"].isin(keep_models)]
        if aggregate.empty:
            logging.warning("%s: no matched models for qa_mode=%r, skipped", task, qa_mode)
            continue
        # std over seeds, not over models: the groupby key is one run configuration
        frame = (
            aggregate.groupby(["model", "reasoning_effort", "family"], dropna=False)["accuracy"]
            .agg(accuracy_mean="mean", accuracy_std="std", n_seeds="count")
            .reset_index()
        )
        # the same individuals the runs were scored on, so the line sits in the dots' world
        index = pd.read_csv(results_root / aggregate.iloc[0]["predictions_path"], index_col=0).index
        try:
            frame["baseline_accuracy"] = baseline_accuracy(task, index=index)
        except FileNotFoundError as err:
            logging.warning("%s: no XGBoost baseline (%s)", task, err)
            frame["baseline_accuracy"] = np.nan
        frames.append(frame.assign(task=task, qa_mode=qa_mode))

    data = pd.concat(frames, ignore_index=True)
    data["model_label"] = data["model"].map(pretty_model_name)
    data["effort_label"] = data["reasoning_effort"].map(effort_display)
    # release order within a family: get_size returns the same placeholder for every closed model
    data = data.sort_values(
        ["task", "family", "model"],
        key=lambda col: col.map(model_sort_key) if col.name == "model" else col,
    )
    return data.reset_index(drop=True)


def vote_distributions_by_length(
    task: str,
    mode: str = "correct",
    qa_mode: str = "numeric",
    variant: str = "rank",
    n_bins: int = 20,
    min_bin: int = 30,
    levels: Iterable[str] | None = ("low", "medium", "high"),
    label_filter: int | None = None,
    openai_effort: str = LENGTH_OPENAI_EFFORT,
    results_root: Path = RESULTS_MODELS_ROOT,
    cache_dir: Path = PAIR_VALUES_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """The vote distribution of :func:`vote_distributions`, on the bins of :func:`length_curves`.

    :func:`vote_distributions` asks how the whole ecosystem votes; this asks whether that shape is
    the same for the individuals the models dispatch quickly and the ones they chew on. The bins are
    the pooled ("All") panel's bins of the length curves -- same individuals, same length, same
    edges -- because both figures call :func:`_length_panel`. The defaults (``n_bins``, ``min_bin``,
    ``levels``) are those of :func:`length_curves` for the same reason: ``levels`` decides which runs
    must have answered an individual, so it changes the individuals.

    Seeds: the length axis uses every balanced seed of the matched models, so the votes do too. A
    vote count is a number of MODELS, though, so one distribution takes one run per model -- two
    seeds of one model agree with each other more than two models do, and counting both would
    inflate unanimity. The distribution is therefore computed per seed draw (draw r = every
    model's r-th seed, as for the length curves' faint seed lines) and averaged over the draws, which
    equals pooling over (individual, draw). Inside each bin the independent-models null is computed
    the same way, from that bin's own accuracies.

    Parameters
    ----------
    task : str
        Task name.
    mode : {"correct", "class"}
        ``"correct"`` counts the runs predicting the true label (ends: all wrong / all right),
        ``"class"`` the runs predicting 1 (ends: all no / all yes).
    qa_mode : str
        QA mode to load.
    variant : str
        Length definition, as in :func:`length_curves`.
    n_bins : int
        Quantile bins of individuals.
    min_bin : int
        Bins holding fewer individuals than this (after ``label_filter``) are dropped.
    levels : iterable of str, optional
        Effort levels kept for the level curves; they decide which runs must have answered.
    label_filter : int, optional
        Keep only the individuals with this true label, AFTER binning, so the conditioned figure
        still sits on the pooled figure's bins.
    openai_effort : str
        Effort level that stands for "reasoning on" for gpt-5.x.
    results_root : Path
        Root of the per-run files.
    cache_dir : Path
        Directory of the cached values and token matrix.
    refresh : bool
        Recompute even when the cache is newer than its inputs.

    Returns
    -------
    pd.DataFrame
        One row per (group, bin, k): ``bin_left``/``bin_right``/``x``, ``k``, ``observed`` (share of
        the bin's individuals), ``null``, ``value`` (observed - null), ``n``, ``n_runs``.
    """
    from reasonance.agreement import vote_distribution, vote_distribution_null

    tag = label_tag(label_filter)
    aggregate_path = Path(AGGREGATE_TEMPLATE.format(task=task))
    cache = Path(cache_dir) / f"{task}-{qa_mode}-{tag}-vote-{mode}-{variant}-{n_bins}-by-length.csv"
    if not refresh and _cache_is_current(cache, aggregate_path, results_root):
        logging.info(f"using cached {cache.name} (newer than every input)")
        return pd.read_csv(cache)

    setup = _length_setup(task, qa_mode, openai_effort, levels, results_root, cache_dir)
    panel = _length_panel(setup, "All", variant, n_bins)
    if panel is None:
        raise ValueError(f"{task}: fewer than two matched models on a side")
    length, bins = panel["length"], panel["bins"]
    predictions, labels = setup["predictions"], setup["labels"]
    by_run, model_of = setup["by_run"], setup["model_of"]

    def seed_draws(run_ids: Iterable[str]) -> list[list[str]]:
        """Draw r = every model's r-th seed, as for the faint seed lines of the length curves.

        The x axis rests on ALL the panel's balanced runs, so the votes must too; but a vote count
        is a number of MODELS, so each distribution takes one run per model. Every draw holds all
        the models, and together the draws use every run the length axis uses.
        """
        by_model: dict[str, list[str]] = {}
        for run_id in run_ids:
            by_model.setdefault(model_of[run_id], []).append(run_id)
        for runs in by_model.values():
            runs.sort(key=lambda r: (by_run.loc[r, "generation_seed"], str(r)))
        n_draws = min(len(runs) for runs in by_model.values())
        return [[runs[rank] for runs in by_model.values()] for rank in range(n_draws)]

    rows = []
    for group, run_ids in (("reasoning", panel["r_ids"]), ("non-reasoning", panel["nr_ids"])):
        draws = seed_draws(run_ids)
        for interval, idx in length.groupby(bins, observed=True).groups.items():
            if label_filter is not None:
                idx = idx[labels.loc[idx].to_numpy() == label_filter]
            if len(idx) < min_bin:
                continue
            bin_preds, bin_labels = predictions.loc[idx], labels.loc[idx]
            # The same individuals in every draw, so the mean of the draws' shares is the share
            # pooled over (individual, draw): every individual counts once per draw.
            observed = np.mean(
                [vote_distribution(bin_preds, ids, labels=bin_labels, mode=mode) for ids in draws], axis=0
            )
            null = np.mean(
                [vote_distribution_null(bin_preds, ids, labels=bin_labels, mode=mode) for ids in draws], axis=0
            )
            for k, (obs, exp) in enumerate(zip(observed, null)):
                rows.append(
                    {
                        "task": task,
                        "qa_mode": qa_mode,
                        "metric": f"vote-{mode}",
                        "variant": variant,
                        "group": group,
                        "bin_left": interval.left,
                        "bin_right": interval.right,
                        "x": float(length.loc[idx].median()),
                        "k": k,
                        "observed": obs,
                        "null": exp,
                        "value": obs - exp,
                        "n": len(idx),
                        "n_runs": len(draws[0]),  # runs per distribution: one per model
                        "n_draws": len(draws),
                    }
                )
    frame = pd.DataFrame(rows)
    if label_filter is not None:
        frame["label_filter"] = label_filter
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def vote_distributions_by_length_for_tasks(tasks: Iterable[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    """:func:`vote_distributions_by_length` for several tasks, concatenated.

    Parameters
    ----------
    tasks : iterable of str, optional
        Tasks to include; defaults to every task with an aggregate.
    **kwargs
        Passed on to :func:`vote_distributions_by_length`.

    Returns
    -------
    pd.DataFrame
        The per-task frames, concatenated.
    """
    tasks = available_tasks() if tasks is None else list(tasks)
    return pd.concat([vote_distributions_by_length(task, **kwargs) for task in tasks], ignore_index=True)
