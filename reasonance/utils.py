"""Core data layer: model registry, results aggregation, and loading of predictions, responses and labels.

Run identity throughout is the ``run_id`` string ``{model_key}_{bench_hash}``; wide tables are
individuals x run_id.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Sequence
    from typing import Any

# ---------------------------------------------------------------------------
# Model registry — grouped by family, then reasoning / non-reasoning
# ---------------------------------------------------------------------------
# Qwen models reason by default; reasoning can be switched on/off via enable_thinking.
QWEN_MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
]

# OpenAI models with reasoning capabilities, reasoning effort can be controlled via reasoning_effort
# ('low', 'medium', 'high' for all models, 'none' is possible for 5.x models)
OPENAI_REASONING_MODELS = ["o1", "o3-mini", "o3", "o4-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4"]
OPENAI_LLM_MODELS = ["gpt-4.1", "gpt-3.5-turbo-0125", "gpt-4o-mini"]
OPENAI_MODELS = OPENAI_REASONING_MODELS + OPENAI_LLM_MODELS

DEEPSEEK_REASONING_MODELS = ["DeepSeek-R1"]
DEEPSEEK_LLM_MODELS = ["DeepSeek-V3.2"]
DEEPSEEK_MODELS = DEEPSEEK_REASONING_MODELS + DEEPSEEK_LLM_MODELS

KIMI_MODELS = ["Kimi-K2-Thinking", "Kimi-K2.5"]

CLAUDE_MODELS = ["claude-opus-4-5"]

# Aggregate lists
REASONING_MODELS = tuple(
    QWEN_MODELS + OPENAI_REASONING_MODELS + DEEPSEEK_REASONING_MODELS + KIMI_MODELS + CLAUDE_MODELS
)
LLM_MODELS = tuple(OPENAI_LLM_MODELS + DEEPSEEK_LLM_MODELS)
MODELS = list(REASONING_MODELS) + list(LLM_MODELS)

_REASONING_MODELS_SET = set(m.replace("Qwen/", "Qwen--").replace("/", "--") for m in REASONING_MODELS) | set(
    REASONING_MODELS
)


# Matched pairs for reasoning vs non-reasoning comparison.
# Each tuple: (model_nonreasoning, effort_nonreasoning, model_reasoning, effort_reasoning).
# Use None for effort when the value is NaN (e.g. DeepSeek-V3.2).
MATCHED_PAIR_SPECS = [
    ("DeepSeek-V3.2", None, "DeepSeek-R1", "1"),
    ("Qwen/Qwen3-4B", "0", "Qwen/Qwen3-4B", "1"),
    ("Qwen/Qwen3-14B", "0", "Qwen/Qwen3-14B", "1"),
    ("Qwen/Qwen3-32B", "0", "Qwen/Qwen3-32B", "1"),
    ("Qwen/Qwen3.5-27B", "0", "Qwen/Qwen3.5-27B", "1"),
    ("Qwen/Qwen3.5-4B", "0", "Qwen/Qwen3.5-4B", "1"),
    ("Qwen/Qwen3.5-9B", "0", "Qwen/Qwen3.5-9B", "1"),
    ("gpt-5.1", "none", "gpt-5.1", "low"),
    ("gpt-5.1", "none", "gpt-5.1", "medium"),
    ("gpt-5.1", "none", "gpt-5.1", "high"),
    ("gpt-5.2", "none", "gpt-5.2", "low"),
    ("gpt-5.2", "none", "gpt-5.2", "medium"),
    ("gpt-5.2", "none", "gpt-5.2", "high"),
    ("gpt-5.4", "none", "gpt-5.4", "low"),
    ("gpt-5.4", "none", "gpt-5.4", "medium"),
    ("gpt-5.4", "none", "gpt-5.4", "high"),
]


# Models left out of every analysis: aggregates, coverage counts and figures.
# The analysis restricts to the individuals EVERY run answered, so a model excluded here
# does not shrink that set with its missing predictions.
# Comment a line out to bring that model back into the stats.
EXCLUDED_MODELS = (
    "o1",  # reasoning-only: no reasoning-off arm, so no matched pair
    "o3",  # reasoning-only
    "o3-mini",  # reasoning-only
    "o4-mini",  # reasoning-only
)


_UNSET = object()


# Generation seed to keep when a configuration was run several times and only one run is wanted.
# 42 is the default generation seed of the benchmark runs.
PREFERRED_GENERATION_SEED = 42


def runs_for(meta_df: pd.DataFrame, model: str | None = None, effort: object = _UNSET) -> list[str]:
    """Select run_ids from meta_df matching a model and/or effort level.

    Parameters
    ----------
    meta_df : DataFrame
        Run metadata (as returned by ``build_run_metadata``).
    model : str, optional
        Filter to this model name.
    effort : str, int, or None
        Filter to this effort level. ``None`` matches rows where
        reasoning_effort is NaN. Omit (default) to skip effort filtering.

    Returns
    -------
    list of str
        Matching run ids, in ``meta_df`` row order.
    """
    mask = pd.Series(True, index=meta_df.index)
    if model is not None:
        mask &= meta_df["model"] == model
    if effort is not _UNSET:
        if effort is None:
            mask &= meta_df["reasoning_effort"].isna()
        else:
            mask &= meta_df["reasoning_effort"].astype(str) == str(effort)
    return meta_df.loc[mask, "run_id"].tolist()


def matched_models(specs: Iterable[tuple[str, str | None, str, str]] | None = None) -> set[str]:
    """Base models that take part in any matched pair, both sides pooled.

    A model outside this set has no reasoning/non-reasoning counterpart, so it cannot contribute to
    a comparison of the two -- keeping it would make the two sides different sets of models.

    Parameters
    ----------
    specs : iterable of tuple, optional
        ``(model_off, effort_off, model_on, effort_on)`` pair specs. Defaults to
        :data:`MATCHED_PAIR_SPECS`.

    Returns
    -------
    set of str
        Model names appearing on either side of any pair.
    """
    specs = MATCHED_PAIR_SPECS if specs is None else specs
    return {model for spec in specs for model in (spec[0], spec[2])}


def pick_run(
    meta_df: pd.DataFrame,
    model: str,
    effort: str | int | None,
    pool: Iterable[str] | None = None,
    prefer_seed: int = PREFERRED_GENERATION_SEED,
) -> str | None:
    """One run_id for (model, effort), preferring ``prefer_seed``; None if there is none.

    Deterministic: the preferred seed first, then the lowest run_id, so repeated calls agree.

    Parameters
    ----------
    meta_df : DataFrame
        Run metadata (``build_run_metadata``).
    model, effort : str
        The cell to resolve; ``effort`` is compared as a string. ``effort=None`` matches a NaN
        reasoning_effort (see :func:`runs_for`).
    pool : iterable of str, optional
        Restrict to these run ids (e.g. one side of the reasoning switch).
    prefer_seed : int
        Generation seed to take when the cell was run several times.

    Returns
    -------
    str or None
        The chosen run id, or None when no run matches.
    """
    ids = runs_for(meta_df, model=model, effort=effort)
    if pool is not None:
        keep = set(pool)
        ids = [run_id for run_id in ids if run_id in keep]
    if not ids:
        return None
    seeds = meta_df.set_index("run_id")["generation_seed"]
    return sorted(ids, key=lambda run_id: (seeds[run_id] != prefer_seed, str(run_id)))[0]


def balance_seeds(
    meta_df: pd.DataFrame, *pools: Collection[str], prefer_low: bool = True
) -> tuple[list[list[str]], int]:
    """Truncate every (model, effort) to the same number of seeds -- the lowest any of them has.

    A model that happens to have been run at five seeds otherwise carries more weight than one run
    at three, which is a fact about how much compute it got rather than about the data. Truncation
    is by seed number, not sampled, so repeated calls give the same runs.

    Parameters
    ----------
    meta_df : DataFrame
        Run metadata (``build_run_metadata``).
    *pools : iterable of str
        One or more run-id pools. The seed count is the minimum over the settings of ALL pools, so
        the two sides of a comparison stay comparable, and each pool is returned truncated.
    prefer_low : bool
        Keep the lowest seed numbers (default). False keeps the highest.

    Returns
    -------
    (list of list, int)
        The truncated pools in the order given, and the number of seeds kept per setting.
    """
    by_run = meta_df.set_index("run_id")
    settings: dict[tuple, list[str]] = {}
    for pool in pools:
        for run_id in pool:
            key = (by_run.loc[run_id, "model"], str(by_run.loc[run_id, "reasoning_effort"]))
            settings.setdefault(key, []).append(run_id)
    if not settings:
        return [list(pool) for pool in pools], 0
    n_seeds = min(len(runs) for runs in settings.values())
    keep = set()
    for runs in settings.values():
        ordered = sorted(runs, key=lambda r: (by_run.loc[r, "generation_seed"], str(r)), reverse=not prefer_low)
        keep.update(ordered[:n_seeds])
    return [[run_id for run_id in pool if run_id in keep] for pool in pools], n_seeds


def matched_sides(
    meta_df: pd.DataFrame,
    reasoning_pool: Collection[str],
    non_reasoning_pool: Collection[str],
    openai_effort: str | None = None,
    specs: Iterable[tuple[str, str | None, str, str]] | None = None,
    prefer_seed: int = PREFERRED_GENERATION_SEED,
) -> tuple[list[str], list[str], dict[str, tuple[str, str]]]:
    """The two sides of the reasoning switch, one run per matched pair.

    Walks the pair specs rather than matching on the model name, because a pair may join two
    different models (DeepSeek: V3.2 with reasoning off, R1 with it on).

    Parameters
    ----------
    meta_df : DataFrame
        Run metadata (``build_run_metadata``).
    reasoning_pool, non_reasoning_pool : iterable of str
        Candidate run ids on each side.
    openai_effort : str, optional
        Keep only this effort level for gpt-5.x, which is matched at several; without it that side
        gets one run per level and outweighs the models that have a single on state.
    specs : list of tuple, optional
        Defaults to :data:`MATCHED_PAIR_SPECS`.
    prefer_seed : int
        Generation seed to take when a cell was run several times (see :func:`pick_run`).

    Returns
    -------
    (list, list, dict)
        Reasoning run ids, non-reasoning run ids, and ``{reasoning model: (run_on, run_off)}``.
    """
    specs = MATCHED_PAIR_SPECS if specs is None else specs
    reasoning, non_reasoning, by_model = {}, {}, {}
    for model_off, effort_off, model_on, effort_on in specs:
        if openai_effort is not None and str(model_on).startswith("gpt-5") and str(effort_on) != openai_effort:
            continue
        run_on = pick_run(meta_df, model_on, effort_on, pool=reasoning_pool, prefer_seed=prefer_seed)
        run_off = pick_run(meta_df, model_off, effort_off, pool=non_reasoning_pool, prefer_seed=prefer_seed)
        if run_on is None or run_off is None:
            continue  # spec not available in this task / qa mode
        reasoning[model_on], non_reasoning[model_off] = run_on, run_off
        by_model[model_on] = (run_on, run_off)
    return list(reasoning.values()), list(non_reasoning.values()), by_model


def is_reasoning(reasoning_effort: Any) -> bool:
    """Check if a reasoning_effort value indicates active reasoning.

    Non-reasoning = reasoning_effort is NaN, 0, 'none', or 'suppressed'.

    Parameters
    ----------
    reasoning_effort : str, int, float or None
        Raw effort value from the aggregate (a level name, a 0/1 flag, or NaN).

    Returns
    -------
    bool
        True when the run reasons.
    """
    if pd.isna(reasoning_effort) or reasoning_effort in ("none", "suppressed"):
        return False
    try:
        return int(reasoning_effort) != 0
    except (ValueError, TypeError):
        return True  # string levels like 'low', 'medium', 'high'


def reasoning_category(reasoning_effort: Any) -> str:
    """Classify reasoning_effort into 'suppressed', 'reasoning', or 'non-reasoning'.

    Parameters
    ----------
    reasoning_effort : str, int, float or None
        Raw effort value, as for :func:`is_reasoning`.

    Returns
    -------
    str
        ``"suppressed"``, ``"reasoning"`` or ``"non-reasoning"``.
    """
    if reasoning_effort == "suppressed":
        return "suppressed"
    return "reasoning" if is_reasoning(reasoning_effort) else "non-reasoning"


# copied
PRETTY_MODEL_NAMES = {
    "Qwen/Qwen3-4B": "Qwen 3 4B",
    "Qwen/Qwen3-14B": "Qwen 3 14B",
    "Qwen/Qwen3-32B": "Qwen 3 32B",
    "Qwen/Qwen3.5-4B": "Qwen 3.5 4B",
    "Qwen/Qwen3.5-9B": "Qwen 3.5 9B",
    "Qwen/Qwen3.5-27B": "Qwen 3.5 27B",
    "gpt-5.1": "GPT 5.1",
    "gpt-5.2": "GPT 5.2",
    "gpt-5.4": "GPT 5.4",
}


# copied
def model_to_key(name: str) -> str:
    """Turn a HuggingFace model name into a path-safe key (``/`` becomes ``--``).

    Parameters
    ----------
    name : str
        Model name, e.g. ``"Qwen/Qwen3-4B"``.

    Returns
    -------
    str
        The key, e.g. ``"Qwen--Qwen3-4B"``.
    """
    return name.replace("/", "--")


# copied
def get_size(model_key: str) -> int | float | None:
    """Model size in billions of parameters, read from the name (``Qwen3-4B`` -> 4).

    API models do not publish their size; they get a placeholder, used only to order models.

    Parameters
    ----------
    model_key : str
        Model key or name.

    Returns
    -------
    int, float or None
        Size in billions of parameters; None if the name carries no size.
    """
    if "gpt" in model_key or model_key in ["o1", "o3", "o4", "o3-mini", "o4-mini"]:
        if "mini" in model_key:
            return 100
        return 1000
    regex = re.search(r"((?P<times>\d+)[xX])?(?P<size>(\d\.)?\d+)[bB]", model_key)
    if regex:
        size = regex.group("size")
        return (float(size) if "." in size else int(size)) * int(regex.group("times") or 1)
    return None


# Release order within each family. `get_size` returns placeholders for closed models (gpt-5.1 and
# o1 both report 1000), so size cannot order them; version order is used as the release proxy.
MODEL_RELEASE_ORDER = [
    # Qwen: version, then size within a version
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    # OpenAI
    "o1",
    "o3-mini",
    "o3",
    "o4-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    # others
    "DeepSeek-R1",
    "DeepSeek-V3.2",
    "Kimi-K2-Thinking",
    "Kimi-K2.5",
    "claude-opus-4-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
]
_RELEASE_INDEX = {name: i for i, name in enumerate(MODEL_RELEASE_ORDER)}


def model_sort_key(model_name: str) -> tuple[str, int, float | None, str]:
    """Sort key ordering models by family, then release (falling back to size, then name).

    Parameters
    ----------
    model_name : str
        Model name (HuggingFace style or API id).

    Returns
    -------
    tuple of (str, int, float or None, str)
        ``(family, release index, size, model_name)``; unknown models sort after known ones. The
        size is None when ``get_size`` cannot infer it from the name.
    """
    release = _RELEASE_INDEX.get(model_name, _RELEASE_INDEX.get(model_name.split("/")[-1], len(MODEL_RELEASE_ORDER)))
    try:
        size = get_size(model_to_key(model_name))
    except Exception:
        size = float("inf")
    return (get_model_family(model_name), release, size, model_name)


# Qwen exposes reasoning as a 0/1 flag, OpenAI as named levels; "0" and "none" mean the same thing,
# as do "1" and "high", so they share a label (and hence a color) in figures.
EFFORT_DISPLAY_LABELS = {
    "0": "off / none",
    "none": "off / none",
    "1": "on / high",
    "high": "on / high",
    "low": "low",
    "medium": "medium",
    "suppressed": "suppressed",
}


def effort_display(effort: object) -> str:
    """Effort label used in legends: NaN counts as no reasoning, equivalent levels are merged.

    Parameters
    ----------
    effort : str, int, float or None
        Raw reasoning_effort value.

    Returns
    -------
    str
        Display label from :data:`EFFORT_DISPLAY_LABELS`, or ``str(effort)`` when not listed.
    """
    if pd.isna(effort):
        return EFFORT_DISPLAY_LABELS["none"]
    return EFFORT_DISPLAY_LABELS.get(str(effort), str(effort))


def pretty_model_name(model_name: str) -> str:
    """Display name for a model; the name itself for a model without one.

    Parameters
    ----------
    model_name : str
        Model name (HuggingFace style or API id).

    Returns
    -------
    str
        Human-readable model name.
    """
    return PRETTY_MODEL_NAMES.get(model_name, model_name)


def get_model_family(model_id: str) -> str:
    """
    Get the family of a model based on its ID.

    Parameters
    ----------
    model_id : str
        Model ID (key, model_id or name)

    Returns
    -------
    str
        Model family.
    """
    model_id = model_id.lower()
    if "qwen" in model_id:
        return "Qwen"
    if "deepseek" in model_id:
        return "DeepSeek"
    if "kimi" in model_id:
        return "Kimi"
    if "claude" in model_id:
        return "Claude"
    if model_id.startswith("gpt") or any(
        n in model_id
        for n in [
            "o4-mini",
            "o3",
            "o3-mini",
            "o1",
        ]
    ):
        return "OpenAI"
    else:
        logging.warning(f"Model family not found for model {model_id}, returning 'Unknown'.")
        return "Unknown"


# Folder inside the results root holding superseded executions of cells that were run more
# than once (one cell with several `bench-*` folders). Skipped when aggregating.
QUARANTINE_DIR_NAME = "_superseded"

_RESULT_COLUMNS = [
    "task",
    "model",
    "is_inst",
    "is_reasoning_model",
    "threshold_fitted",
    "threshold",
    "threshold_obj",
    "accuracy",
    "balanced_accuracy",
    "bench_hash",
    "num_shots",
    "prompt_format",
    "prompt_connector",
    "prompt_granularity",
    "prompt_feature_order",
    "prompt_example_order",
    "prompt_example_composition",
    "eval_results_path",
    "predictions_path",
    "correct_order_bias",
    "reasoning_effort",
    "generation_seed",
    "qa_mode",
    "responses_path",
    "run_time",
]

_INSTRUCT_API_MODELS = frozenset(
    {
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-3.5-turbo-0125",
        "o4-mini",
        "o3",
        "o3-mini",
        "o1",
        "claude-opus-4-5",
        "Kimi-K2-Thinking",
        "Kimi-K2.5",
        "DeepSeek-R1",
        "DeepSeek-V3.2",
    }
)


def _is_instruction_tuned(model_key: str) -> bool:
    """Return True if the model is instruction/chat fine-tuned.

    Parameters
    ----------
    model_key : str
        Directory-style model identifier (e.g. ``Qwen--Qwen3-4B`` or ``o3-mini``).

    Returns
    -------
    bool
        True for known API instruction models and any model whose name contains
        ``instruct``, ``chat``, or ``-it``.
    """
    canonical = model_key.replace("--", "/")
    if canonical in _INSTRUCT_API_MODELS or model_key in _INSTRUCT_API_MODELS:
        return True
    return bool(re.search(r"(?i)(instruct|chat|it\b)", model_key))


def _model_key_to_name(model_key: str) -> str:
    """Convert a model key to a HuggingFace model name.

    Parameters
    ----------
    model_key : str
        Model key, e.g. ``Qwen--Qwen3-4B``.

    Returns
    -------
    str
        HuggingFace model name, e.g. ``Qwen/Qwen3-4B``.
    """
    return model_key.replace("--", "/", 1)


def _parse_result_file(file_path: Path, task: str) -> dict[str, Any] | None:
    """Parse a single ``results.bench-*.json`` into a result row.

    Parameters
    ----------
    file_path : Path
        Path to the ``results.bench-{hash}.json`` file.
    task : str
        Task name used to strip the task suffix from the model directory name.

    Returns
    -------
    dict or None
        Flat dict with all result columns, or None if the file cannot be loaded.
    """
    try:
        with file_path.open() as f:
            data = json.load(f)
    except Exception as e:
        logging.warning(f"Could not load {file_path}: {e}")
        return None

    config = data.get("config", {})
    prompt_variation = config.get("prompt_variation", {})

    # Model key lives two levels up: model-{key}_task-{task}/
    model_key = file_path.parent.parent.name
    model_key = model_key.replace("model-", "").replace(f"_task-{task}", "")
    canonical_name = _model_key_to_name(model_key)

    # Metrics
    threshold_fitted = int(bool(data.get("threshold_fitted_on", 0)))
    threshold_obj = data.get("threshold_obj") if threshold_fitted else None

    # Prompt config
    num_shots = config.get("few_shot") or 0
    prompt_example_order = None
    prompt_example_composition = None
    if num_shots:
        prompt_example_order = prompt_variation.get("example_order", "default") or "default"
        comp = config.get("compose_few_shot_examples")
        prompt_example_composition = ",".join(map(str, comp)) if isinstance(comp, list) else comp

    # Paths — store relative to the "results/" root so the CSV is portable
    def _rel(p: str) -> str:
        """Cut ``p`` to start at its ``results/`` component; unchanged if it has none."""
        idx = p.find("results/")
        return p[idx:] if idx != -1 else p

    predictions_path = _rel(data.get("predictions_path", ""))
    responses_path = predictions_path.replace(".test_predictions.csv", ".test_responses.csv")

    return {
        "task": task,
        "model": canonical_name,
        "is_inst": int(_is_instruction_tuned(model_key)),
        "is_reasoning_model": int(canonical_name in REASONING_MODELS or model_key in _REASONING_MODELS_SET),
        "threshold_fitted": threshold_fitted,
        "threshold": data.get("threshold", 0.5),
        "threshold_obj": threshold_obj,
        "accuracy": data.get("accuracy"),
        "balanced_accuracy": data.get("balanced_accuracy"),
        "bench_hash": file_path.parent.name.split("_bench-")[1],
        "num_shots": num_shots,
        "prompt_format": prompt_variation.get("format", "bullet"),
        "prompt_connector": prompt_variation.get("connector", "is"),
        "prompt_granularity": prompt_variation.get("granularity", "original"),
        "prompt_feature_order": prompt_variation.get("order", "default") or "default",
        "prompt_example_order": prompt_example_order,
        "prompt_example_composition": prompt_example_composition,
        "eval_results_path": _rel(file_path.as_posix()),
        "predictions_path": predictions_path,
        "correct_order_bias": int(bool(config.get("correct_order_bias", False))),
        "reasoning_effort": config.get("reasoning"),
        # A result file without a `generation_seed` was generated at the global `seed`,
        # so that is its generation seed, not a seed of its own. `.get(key, default)`
        # only covers an absent key, so an explicit null is normalised too.
        "generation_seed": (
            config.get("generation_seed") if config.get("generation_seed") is not None else config.get("seed", 42)
        ),
        "qa_mode": "numeric" if config.get("numeric_risk_prompting", False) else "mcq",
        "responses_path": responses_path,
        # When the run finished; the tiebreaker when one cell was executed twice.
        "run_time": data.get("current_time"),
    }


def build_aggregate(
    results_dir: str | Path,
    tasks: list[str] | str = "ACSIncome",
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate benchmark result JSONs into a single DataFrame.

    Parameters
    ----------
    results_dir : str or Path
        Root results directory (e.g. ``.../reasoning/0-bullet-is/``).
    tasks : list of str or str
        Task name(s) to include.
    save_path : str or Path, optional
        If given, save the DataFrame as a CSV at this path.

    Returns
    -------
    pd.DataFrame
        One row per benchmark run with columns: task, model, is_inst,
        is_reasoning_model, threshold_fitted, threshold, threshold_obj,
        accuracy, balanced_accuracy, bench_hash, num_shots, prompt_format,
        prompt_connector, prompt_granularity, prompt_feature_order,
        prompt_example_order, prompt_example_composition, eval_results_path,
        predictions_path, correct_order_bias, reasoning_effort,
        generation_seed, qa_mode, responses_path.
    """
    results_dir = Path(results_dir)
    if isinstance(tasks, str):
        tasks = [tasks]

    pattern = re.compile(r"^results\.bench-\d+\.json$")
    rows = []

    for task in tasks:
        for json_file in sorted(results_dir.rglob("results.bench-*.json")):
            if task not in json_file.as_posix():
                continue
            # Older executions of a cell that was run more than once are quarantined
            # here rather than deleted. They sit inside
            # the results root, so `rglob` still reaches them -- and including them
            # both duplicates the cell and, because their files have moved, leaves a
            # row whose `predictions_path` no longer resolves. Dedup then picks that
            # dead row and the seed silently disappears from the analysis.
            if QUARANTINE_DIR_NAME in json_file.parts:
                continue
            if not pattern.match(json_file.name):
                continue
            row = _parse_result_file(json_file, task)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    logging.info(f"Aggregated {len(df)} results from {results_dir}")

    if save_path is not None:
        df.to_csv(save_path, index=False)
        logging.info(f"Saved results to {save_path}")

    return df


CELL_KEYS = ("model", "reasoning_effort", "qa_mode")


def seed_coverage(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Seeds per ``(model, reasoning_effort, qa_mode)``, and whether any are duplicated.

    ``runs`` above ``seeds`` means some seed was executed more than once. The analysis
    de-duplicates on ``generation_seed`` and keeps one arbitrarily, so a repeated seed is
    a silent choice between two different sets of predictions.

    A run whose result file has no ``generation_seed`` counts as the global ``seed``
    (normalised in `_parse_result_file`), so re-running such a cell with the seed stated
    explicitly shows up here as a duplicate -- which is what it is.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Aggregate as returned by :func:`build_aggregate` (not de-duplicated).

    Returns
    -------
    pd.DataFrame
        One row per cell with ``runs``, ``seeds``, ``seed_list``, ``repeated`` and
        ``repeated_seed``; empty when the aggregate lacks the cell or seed columns.
    """
    keys = [k for k in CELL_KEYS if k in aggregate.columns]
    if not keys or "generation_seed" not in aggregate.columns:
        return pd.DataFrame()

    coverage = (
        aggregate.groupby(keys, dropna=False)["generation_seed"]
        .agg(
            runs="size",
            seeds="nunique",
            seed_list=lambda s: ",".join(str(x) for x in sorted(set(s.dropna()))),
            repeated=lambda s: ",".join(str(x) for x in sorted(set(s[s.duplicated()].dropna()))),
        )
        .reset_index()
    )
    # Count repeated seeds, not `runs > seeds`: `nunique` skips nulls while `size` counts
    # them, so the latter fires on a cell that merely has a seedless run and no duplicate.
    # Not named `duplicated`: that shadows `DataFrame.duplicated` on attribute access.
    coverage["repeated_seed"] = coverage["repeated"].astype(bool)
    return coverage


def duplicate_runs(aggregate: pd.DataFrame) -> pd.DataFrame:
    """The individual rows of every cell whose seed was executed more than once.

    Adds ``supersedes``: ``keep`` for the most recent execution of that (cell, seed),
    ``replace`` for the earlier ones. Recency is `run_time`; an aggregate without that column
    keeps the later row in file order, which is arbitrary (:func:`build_aggregate` records it).

    Parameters
    ----------
    aggregate : pd.DataFrame
        Aggregate as returned by :func:`build_aggregate` (not de-duplicated).

    Returns
    -------
    pd.DataFrame
        The duplicated rows (identifying columns only) plus ``supersedes``; empty when there are
        none or the aggregate lacks the cell or seed columns.
    """
    keys = [k for k in CELL_KEYS if k in aggregate.columns]
    if not keys or "generation_seed" not in aggregate.columns:
        return pd.DataFrame()

    group = keys + ["generation_seed"]
    dupes = aggregate[aggregate.duplicated(group, keep=False)].copy()
    if dupes.empty:
        return dupes

    sort_cols = group + (["run_time"] if "run_time" in dupes.columns else [])
    dupes = dupes.sort_values(sort_cols)
    newest = ~dupes.duplicated(group, keep="last")
    dupes["supersedes"] = np.where(newest, "keep", "replace")
    shown = [c for c in (*group, "run_time", "bench_hash", "num_samples", "predictions_path") if c in dupes]
    return dupes[[*shown, "supersedes"]]


def report_seed_coverage(aggregate: pd.DataFrame, printer: Callable[[str], object] = print) -> pd.DataFrame:
    """Print `seed_coverage`, and list the runs to replace when a seed was run twice.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Aggregate as returned by :func:`build_aggregate` (not de-duplicated).
    printer : callable
        Called with each block of text; ``print`` by default (pass e.g. ``logging.info``).

    Returns
    -------
    pd.DataFrame
        The :func:`seed_coverage` table.
    """
    coverage = seed_coverage(aggregate)
    if coverage.empty:
        printer("seed coverage: not available (aggregate lacks model/effort/seed columns)")
        return coverage

    printer("\nSeeds per (model, effort):")
    printer(coverage.to_string(index=False))

    duplicated = coverage[coverage["repeated_seed"]]
    if not len(duplicated):
        return coverage

    printer(
        f"\nWARNING: {len(duplicated)} cell(s) ran the same seed more than once. "
        f"De-duplication keeps one of them arbitrarily, so these are silent choices "
        f"between different predictions. Rows marked 'replace' are the older executions:"
    )
    rows = duplicate_runs(aggregate)
    printer(rows.to_string(index=False) if len(rows) else duplicated.to_string(index=False))
    return coverage


RETRY_SUFFIX = ".retry.csv"


def retry_sibling(path: Path) -> Path | None:
    """The ``*.retry.csv`` written beside `path`, if a retry has run for it.

    `folktexts.cli.retry_failed_predictions` never modifies a run's own files; it writes its
    results alongside. Every reader therefore has to opt in, and a reader that forgets silently
    reports the pre-retry state.

    Parameters
    ----------
    path : Path
        A run's own predictions or responses CSV.

    Returns
    -------
    Path or None
        The retry file, or None when no retry exists.
    """
    retried = Path(path).with_suffix(RETRY_SUFFIX)
    return retried if retried.exists() else None


def load_run_frame(
    path: Path,
    *,
    kind: str,
    use_retries: bool = True,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Read one run's predictions or responses, optionally folding in its retry.

    The two file kinds need different handling, which is the whole reason for a shared helper:

    * ``kind="predictions"`` -- the retry file is a COMPLETE copy of the run (every individual,
      with `answer_source` added), so it simply replaces the original.
    * ``kind="responses"``   -- the retry file holds ONLY the individuals it re-queried, so it is
      overlaid onto the original row by row. Columns the retry does not carry (`prompt`,
      `question_idx`, and notably `reasoning_tokens`) keep the original value, except that a
      retried row's `reasoning_tokens` is set to NaN: the count belongs to the trace that was
      replaced, and the retry does not record a new one.

    Parameters
    ----------
    path : Path
        The run's own file, as recorded in the aggregate.
    kind : {"predictions", "responses"}
        Which file kind ``path`` is; decides how the retry is folded in (see above).
    use_retries : bool
        ``False`` reproduces the pre-retry state, which is what an analysis of the ORIGINAL
        experiment wants -- retried traces were generated at a larger output budget, so their
        lengths are not drawn from the same distribution as the run's own.
    **read_kwargs
        Passed to ``pd.read_csv`` for the run's own file (and, for predictions, the retry file).

    Returns
    -------
    pd.DataFrame
        The run's frame with the retry applied, or the original when there is none.
    """
    if kind not in ("predictions", "responses"):
        raise ValueError(f"kind must be 'predictions' or 'responses', got {kind!r}")
    path = Path(path)
    retried_path = retry_sibling(path) if use_retries else None

    if kind == "predictions":
        return pd.read_csv(retried_path or path, **read_kwargs)

    frame = pd.read_csv(path, **read_kwargs)
    if retried_path is None:
        return frame
    try:
        patch = pd.read_csv(retried_path, index_col=0)
    except Exception as error:
        logging.warning(f"could not read '{retried_path}': {error}; using the original responses")
        return frame
    if "row_idx" not in frame.columns or patch.empty:
        return frame

    indexed = frame.set_index("row_idx")
    shared = [c for c in patch.columns if c in indexed.columns]
    rows = indexed.index.intersection(patch.index)
    if len(rows) and shared:
        indexed.loc[rows, shared] = patch.loc[rows, shared]
        if "reasoning_tokens" in indexed.columns:
            # the stored count describes the trace that was just replaced; the retry records none
            indexed.loc[rows, "reasoning_tokens"] = np.nan
        logging.info(f"folded {len(rows)} retried response(s) into '{path.name}'")
    return indexed.reset_index()


def load_predictions(
    aggregate: pd.DataFrame,
    results_root: str | Path,
    use_retries: bool = True,
) -> pd.DataFrame:
    """Load risk scores for all runs in an aggregate DataFrame as a wide table.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Subset of an aggregate DataFrame (one or more rows).
    results_root : str or Path
        Directory that contains the ``results/`` subtree; used to resolve
        relative ``predictions_path`` values.
    use_retries : bool
        Prefer a run's ``*.retry.csv`` when it exists. Recovered answers include rows obtained by
        closing the thinking trace and asking the model to commit (``answer_source == "forced"``),
        which is a different condition from the original run -- `analysis.coverage_summary` reports
        how many. Pass ``False`` to use only the original run's answers.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with one column per run named ``{model}_{bench_hash}``,
        containing the ``risk_score`` values aligned on the sample index.
    """
    results_root = Path(results_root)
    series = {}
    n_retried = 0
    for _, row in aggregate.iterrows():
        path = results_root / row["predictions_path"]
        if not path.exists():
            logging.warning(f"Predictions file not found: {path}")
            continue
        n_retried += bool(use_retries and retry_sibling(path))
        preds = load_run_frame(path, kind="predictions", use_retries=use_retries, index_col=0)
        col_name = f"{model_to_key(row['model'])}_{row['bench_hash']}"
        series[col_name] = preds["risk_score"]
    if n_retried:
        logging.info(f"Using retried answers for {n_retried} of {len(aggregate)} runs")
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series)


def _drop_empty_traces(df: pd.DataFrame, cols: list[str], model_id: str = "") -> pd.DataFrame:
    """Drop rows where any column in ``cols`` is NaN or whitespace-only."""

    def is_valid(t: object) -> bool:
        """True for a string with at least one non-whitespace character."""
        return isinstance(t, str) and bool(t.strip())

    mask = df[cols].apply(lambda c: c.map(is_valid)).all(axis=1)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logging.warning(f"{model_id}: dropping {n_dropped} rows with empty traces in {cols}.")
    return df[mask]


def iter_responses(
    aggregate: pd.DataFrame,
    results_root: str | Path,
    cols: tuple[str, ...] = ("reasoning", "response"),
    question_idx: int = 0,
    drop_empty: bool = True,
    use_retries: bool = True,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Iterate over reasoning traces for all runs in an aggregate DataFrame.

    Yields one model at a time so callers can process and discard each chunk
    without materialising all traces in memory at once.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Subset of an aggregate DataFrame (one or more rows).
    results_root : str or Path
        Directory that contains the ``results/`` subtree; used to resolve
        relative ``responses_path`` values.
    cols : tuple of str
        Columns to load from each responses CSV. Defaults to
        ``("reasoning", "response")``.
    question_idx : int
        Which question permutation to include. Defaults to 0.
    drop_empty : bool
        If True, drop rows where any column in ``cols`` is NaN or
        whitespace-only before yielding. Defaults to True.
    use_retries : bool
        Overlay each run's ``*.retry.csv`` onto its responses (see :func:`load_run_frame`).
        Defaults to True.

    Yields
    ------
    model_id : str
        Run identifier ``{model_key}_{bench_hash}``.
    df : pd.DataFrame
        DataFrame with columns ``row_idx``, ``model_id``, and the requested
        ``cols``, filtered to ``question_idx``.
    """
    results_root = Path(results_root)
    for _, row in aggregate.iterrows():
        path = results_root / row["responses_path"]
        if not path.exists():
            logging.warning(f"Responses file not found: {path}")
            continue
        # Read only the columns asked for: these files carry the full traces, so parsing the text
        # only to drop it is several times slower than the slim read.
        wanted = {"row_idx", "question_idx", *cols}
        responses = load_run_frame(path, kind="responses", use_retries=use_retries, usecols=lambda name: name in wanted)
        missing = wanted - set(responses.columns) - {"question_idx"}
        if missing:
            logging.warning(f"{path}: missing column(s) {sorted(missing)}, skipped")
            continue
        if "question_idx" in responses:
            responses = responses[responses["question_idx"] == question_idx]
        model_id = f"{model_to_key(row['model'])}_{row['bench_hash']}"
        chunk = responses[["row_idx", *cols]].copy()
        chunk.insert(1, "model_id", model_id)
        if drop_empty:
            chunk = _drop_empty_traces(chunk, list(cols), model_id)
        yield model_id, chunk


_DEFAULT_DEDUP_SUBSET = ["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode"]


def load_aggregate(
    path: str | Path,
    qa_mode: str | None = None,
    dedup_subset: list[str] | None = None,
    preferred_seed: int | None = PREFERRED_GENERATION_SEED,
) -> pd.DataFrame:
    """Load an aggregated results CSV and annotate with reasoning metadata.

    Adds ``reasoning``, ``category``, and ``family`` columns, then
    deduplicates on ``dedup_subset``. Optionally filters to a single QA mode.

    Parameters
    ----------
    path : str or Path
        Path to the aggregated CSV file.
    qa_mode : str, optional
        If given, keep only rows with this ``qa_mode`` value (e.g. ``"mcq"``).
    dedup_subset : list of str, optional
        Columns to deduplicate on. Defaults to
        ``["model", "reasoning_effort", "reasoning", "category", "family", "qa_mode"]``.
        Note this default does *not* include ``generation_seed``, so a configuration run at
        several seeds is collapsed to a single row -- see ``preferred_seed``. Pass a subset
        containing ``generation_seed`` to keep every seed.
    preferred_seed : int, optional
        Generation seed to keep when deduplication collapses several seeds of the same
        configuration, by default ``PREFERRED_GENERATION_SEED``. Configurations that were never
        run at this seed fall back to their first row. Pass None to let CSV row order decide.

    Returns
    -------
    pd.DataFrame
    """
    if dedup_subset is None:
        dedup_subset = _DEFAULT_DEDUP_SUBSET
    agg = pd.read_csv(path)
    agg["reasoning"] = agg["reasoning_effort"].apply(is_reasoning)
    agg["category"] = agg["reasoning_effort"].apply(reasoning_category)
    agg["family"] = agg["model"].apply(get_model_family)

    # `drop_duplicates` keeps the first row of each group, which would otherwise be whichever run
    # happens to come first in the CSV. Sort the preferred seed to the front first (stably, so
    # configurations lacking it keep their original order) to make the surviving run deterministic.
    if preferred_seed is not None and "generation_seed" in agg.columns:
        agg = agg.sort_values(
            "generation_seed",
            key=lambda s: pd.to_numeric(s, errors="coerce").ne(preferred_seed),
            kind="stable",
        )

    agg.drop_duplicates(subset=dedup_subset, inplace=True)
    agg = agg.sort_index()  # undo the preference sort; only *which* row survived should change
    if qa_mode is not None:
        agg = agg[agg["qa_mode"] == qa_mode]
    return agg


def binarize_predictions(
    risk_scores: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-run thresholds to risk scores to get binary predictions.

    Parameters
    ----------
    risk_scores : pd.DataFrame
        Wide DataFrame of risk scores (individuals x run_id), as returned
        by ``load_predictions``.
    aggregate : pd.DataFrame
        Aggregate DataFrame with ``model``, ``bench_hash``, and ``threshold``
        columns.

    Returns
    -------
    pd.DataFrame
        Binary predictions (0/1) as floats, with NaN kept where the run has no risk score.

    Notes
    -----
    A missing risk score must stay missing: ``NaN >= threshold`` is ``False``, so casting the
    comparison directly would turn every individual a run never scored into a confident class 0.
    A run covering part of a task would then look like it agreed with whoever else predicts 0.
    Callers restrict to the individuals every run scored (see ``common_individuals``).
    """
    thresholds = pd.Series(
        {f"{model_to_key(row['model'])}_{row['bench_hash']}": row["threshold"] for _, row in aggregate.iterrows()}
    )
    thresholds = thresholds[thresholds.index.isin(risk_scores.columns)]
    return risk_scores.apply(
        lambda col: (col >= thresholds[col.name]).astype(float).where(col.notna()),
        axis=0,
    )


def build_run_metadata(
    aggregate: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build a run metadata DataFrame from aggregate and prediction columns.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Aggregate DataFrame with model, family, reasoning, etc. columns.
    predictions : pd.DataFrame
        Wide predictions DataFrame whose columns are run_ids to keep.

    Returns
    -------
    pd.DataFrame
        One row per run with columns: ``run_id``, ``model``, ``family``,
        ``reasoning`` (bool), ``category``, ``reasoning_effort``,
        ``qa_mode``, ``generation_seed``.
    """
    run_meta = {
        f"{model_to_key(row['model'])}_{row['bench_hash']}": {
            "model": row["model"],
            "family": row["family"],
            "reasoning": row["reasoning"],
            "category": row["category"],
            "reasoning_effort": row["reasoning_effort"],
            "qa_mode": row["qa_mode"],
            "generation_seed": row["generation_seed"],
        }
        for _, row in aggregate.iterrows()
    }
    run_meta = {k: v for k, v in run_meta.items() if k in predictions.columns}
    meta_df = pd.DataFrame(run_meta).T.reset_index().rename(columns={"index": "run_id"})
    meta_df["reasoning"] = meta_df["reasoning"].astype(bool)
    return meta_df


def load_reasoning_tokens(
    aggregate: pd.DataFrame,
    results_root: str | Path,
    question_idx: int = 0,
) -> pd.DataFrame:
    """Load reasoning token counts as a wide matrix (individuals x run_id).

    Parameters
    ----------
    aggregate : pd.DataFrame
        Aggregate DataFrame selecting which runs to include.
    results_root : str or Path
        Root directory containing the ``results/`` subtree.
    question_idx : int
        Which question permutation to load. Defaults to 0.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with individuals as rows and run_ids as columns,
        values are reasoning token counts.
    """
    tokens_by_run = {}
    for model_id, chunk in iter_responses(
        aggregate, results_root, cols=("reasoning_tokens",), question_idx=question_idx, drop_empty=False
    ):
        tokens_by_run[model_id] = chunk.set_index("row_idx")["reasoning_tokens"]
    return pd.DataFrame(tokens_by_run)


# Baseline (XGBoost, GBM, ...) predictions are looked up in `results/baselines` inside this
# repository; set REASONANCE_BASELINE_ROOT to use another directory.
BASELINE_ROOTS = (Path(__file__).parent.parent / "results/baselines",)


def find_baseline_predictions(
    task: str,
    model: str = "XGBoost",
    roots: Iterable[str | Path] | None = None,
) -> Path:
    """Locate a baseline model's test predictions for one task.

    Parameters
    ----------
    task : str
        Task name, e.g. "ACSIncome".
    model : str
        Baseline model name, as in the ``model-{model}`` directories of the baseline results.
    roots : iterable of path, optional
        Directories to search instead of ``BASELINE_ROOTS``. ``REASONANCE_BASELINE_ROOT`` in the
        environment takes precedence over both.

    Returns
    -------
    Path
        The ``*.test_predictions.csv`` file (the first one, if a task has several benchmark runs).

    Raises
    ------
    FileNotFoundError
        If no matching file exists under any root; the message lists what was searched.
    """
    if roots is None:
        env_root = os.environ.get("REASONANCE_BASELINE_ROOT")
        roots = [env_root, *BASELINE_ROOTS] if env_root else BASELINE_ROOTS
    searched = []
    for root in roots:
        root = Path(root)
        searched.append(str(root))
        matches = sorted(root.glob(f"model-{model}/{model}_task-{task}/*/*.test_predictions.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"no {model} predictions for task {task!r}; searched: {', '.join(searched)}. "
        "Set REASONANCE_BASELINE_ROOT to the baselines directory."
    )


def zscore_within_run(col: pd.Series) -> pd.Series:
    """Z-score one run's values; a run with no spread contributes zeros, not division by ~0.

    Parameters
    ----------
    col : pd.Series
        One run's per-individual values (e.g. reasoning token counts).

    Returns
    -------
    pd.Series
        ``(col - mean) / std``, or all zeros when the run has no spread.
    """
    std = col.std()
    return col * 0.0 if not std else (col - col.mean()) / std


# How to turn the per-run reasoning token counts into ONE per-individual x coordinate. Each entry
# maps the (individuals x run) matrix to a per-individual series, plus the axis label to use.
#   raw     - median tokens; comparable across individuals, dominated by the most verbose runs
#   zscore  - median of per-run z-scores; scale-free, but a run whose lengths barely vary has a tiny
#             std, so its noise is inflated to the same range as a genuinely varying run
#   rank    - median of per-run percentile ranks; scale-free like zscore and bounded, so a flat run
#             yields near-uniform ranks instead of amplified noise
LENGTH_VARIANTS: dict[str, dict[str, Any]] = {
    "raw": {
        "transform": lambda matrix: matrix,
        "label": "median reasoning tokens per individual",
        "short_label": "median tokens",
    },
    "zscore": {
        "transform": lambda matrix: matrix.apply(zscore_within_run),
        "label": "median z-scored reasoning length per individual",
        "short_label": "median z-score",
    },
    "rank": {
        "transform": lambda matrix: matrix.rank(pct=True),
        "label": "median percentile rank of reasoning length per individual",
        "short_label": "median percentile rank",
    },
}


def build_length_variants(
    token_matrix: pd.DataFrame, run_ids: Sequence[str] | None = None, variants: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Per-individual length series for each x-axis variant.

    Parameters
    ----------
    token_matrix : pd.DataFrame
        Reasoning token counts, individuals x run_id.
    run_ids : list of str, optional
        Runs to summarise; defaults to every column.
    variants : iterable of str, optional
        Subset of :data:`LENGTH_VARIANTS` keys, in order. Defaults to all of them.

    Returns
    -------
    dict
        ``{name: {"series": pd.Series, "label": str, "short_label": str}}``, ready to hand to the
        binning helpers.
    """
    matrix = token_matrix if run_ids is None else token_matrix[run_ids]
    names = list(LENGTH_VARIANTS) if variants is None else list(variants)
    unknown = set(names) - set(LENGTH_VARIANTS)
    if unknown:
        raise ValueError(f"unknown length variant(s) {sorted(unknown)}; choose from {list(LENGTH_VARIANTS)}")
    return {
        name: {
            "series": LENGTH_VARIANTS[name]["transform"](matrix).median(axis=1),
            "label": LENGTH_VARIANTS[name]["label"],
            "short_label": LENGTH_VARIANTS[name]["short_label"],  # for multi-panel axes
        }
        for name in names
    }


def baseline_accuracy(
    task: str,
    model: str = "XGBoost",
    threshold: float = 0.5,
    index: pd.Index | None = None,
    roots: Iterable[str | Path] | None = None,
) -> float:
    """Accuracy of a baseline model on a task, read from its saved predictions.

    Avoids hardcoding a reference line in a notebook: the number comes from the same predictions the
    difficulty proxy uses, so it stays correct when the task changes.

    Parameters
    ----------
    task : str
        Task name, e.g. "ACSIncome".
    model : str
        Baseline model name, as in the ``model-{model}`` directories of the baseline results.
    threshold : float
        Risk score above which the prediction counts as positive.
    index : pd.Index, optional
        Restrict to these individuals (e.g. the subsample the LLMs were evaluated on). Defaults to
        every row of the baseline's test predictions.
    roots : iterable of path, optional
        Passed to :func:`find_baseline_predictions`.

    Returns
    -------
    float
        Fraction of individuals the baseline classifies correctly.
    """
    frame = pd.read_csv(find_baseline_predictions(task, model=model, roots=roots), index_col=0)
    if index is not None:
        frame = frame.loc[frame.index.intersection(index)]
    return float(((frame["risk_score"] > threshold).astype(int) == frame["label"]).mean())


def load_labels(aggregate: pd.DataFrame, results_root: str | Path, index: pd.Index | None = None) -> pd.Series:
    """Ground-truth labels for a task, read from the runs' predictions files.

    Files are read in aggregate order until every requested individual has a label.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Aggregate rows for the task; their ``predictions_path`` files are read in order.
    results_root : str or Path
        Directory the ``predictions_path`` values are relative to.
    index : pd.Index, optional
        Restrict (and order) the labels to these individuals.

    Returns
    -------
    pd.Series
        Labels indexed by individual.
    """
    # Runs can cover different individuals (some were run on a subsample), so the first file is not
    # necessarily the widest. Read further files until every requested individual has a label.
    results_root = Path(results_root)
    labels = pd.Series(dtype=float)
    for path in aggregate["predictions_path"]:
        file_path = results_root / str(path)
        if not file_path.exists():
            continue
        found = pd.read_csv(file_path, index_col=0)["label"]
        labels = found if labels.empty else labels.combine_first(found)
        if index is not None and not len(pd.Index(index).difference(labels.index)):
            break
    return labels.loc[labels.index.intersection(index)] if index is not None else labels


def common_individuals(predictions: pd.DataFrame, run_ids: Iterable[str] | None = None) -> pd.Index:
    """Individuals scored by EVERY run, i.e. the rows with no missing prediction.

    Agreement between two runs can only be measured where both predicted, and comparing different
    pairs on different individuals makes their values incomparable. Restricting once, up front, to
    the common set keeps every pair on the same individuals.

    Parameters
    ----------
    predictions : pd.DataFrame
        Wide predictions or risk scores, individuals x run_id.
    run_ids : list of str, optional
        Runs that must all have a prediction. Defaults to every column.

    Returns
    -------
    pd.Index
        Index of the individuals all those runs scored.
    """
    frame = predictions if run_ids is None else predictions[list(run_ids)]
    return frame.dropna(axis=0, how="any").index


class TaskData:
    """Everything the analysis notebooks load for one task, loaded once.

    Attributes
    ----------
    task, qa_mode : str
        What was loaded.
    aggregate : pd.DataFrame
        Aggregate rows after model filtering.
    risk_scores, predictions : pd.DataFrame
        Wide risk scores and their binarized counterpart, individuals x run_id.
    labels : pd.Series
        Ground truth, aligned to ``predictions``.
    meta : pd.DataFrame
        Run metadata (run_id, model, family, reasoning, category, reasoning_effort, seed).
    label_filter : int or None
        The subgroup this data is restricted to, kept here so figures and file names can read it
        from the data instead of a separate notebook variable.
    n_dropped : int
        Individuals removed when restricting to the common set and applying ``label_filter``.
    """

    def __init__(
        self,
        task: str,
        qa_mode: str,
        aggregate: pd.DataFrame,
        risk_scores: pd.DataFrame,
        predictions: pd.DataFrame,
        labels: pd.Series,
        meta: pd.DataFrame,
        label_filter: int | None,
        n_dropped: int,
    ) -> None:
        """Store the loaded objects; see the class docstring for each attribute."""
        self.task = task
        self.qa_mode = qa_mode
        self.aggregate = aggregate
        self.risk_scores = risk_scores
        self.predictions = predictions
        self.labels = labels
        self.meta = meta
        self.label_filter = label_filter
        self.n_dropped = n_dropped

    @property
    def run_ids(self) -> list[str]:
        """Run ids, i.e. the columns of ``predictions``."""
        return list(self.predictions.columns)

    @property
    def label_filter_tag(self) -> str | None:
        """Figure-name tag for the subgroup, None when all individuals are kept."""
        from reasonance.plotting import label_filter_tag

        return label_filter_tag(self.label_filter)

    def __repr__(self) -> str:
        """Task, QA mode, subgroup and sizes in one line."""
        subgroup = "" if self.label_filter is None else f", label={self.label_filter}"
        return (
            f"TaskData({self.task}, {self.qa_mode}{subgroup}: {len(self.predictions)} individuals, "
            f"{len(self.run_ids)} runs, {self.n_dropped} dropped)"
        )


def load_task_data(
    task: str,
    results_root: str | Path,
    qa_mode: str = "numeric",
    aggregate_path: str | Path | None = None,
    drop_models: Iterable[str] = EXCLUDED_MODELS,
    label_filter: int | None = None,
    common_only: bool = True,
    dedup_subset: list[str] | None = None,
    use_retries: bool = True,
) -> TaskData:
    """Load one task end to end: aggregate, risk scores, predictions, labels, run metadata.

    Every analysis notebook needs the same five objects in the same order, and getting the label
    alignment wrong is silent, so it happens once here.

    Parameters
    ----------
    task : str
        Task name, e.g. ``"ACSIncome"``.
    results_root : str or Path
        Root holding the per-run prediction files.
    qa_mode : str
        QA mode to keep.
    aggregate_path : str or Path, optional
        Aggregate CSV. Defaults to ``results/{task}-aggregated-0-bullet-is.csv``.
    drop_models : iterable of str
        Models to exclude; `EXCLUDED_MODELS` by default. Pass ``()`` to keep every model.
    label_filter : int, optional
        Keep only individuals whose true label is this value.
    common_only : bool
        Restrict to individuals every run scored (see ``common_individuals``).
    dedup_subset : list of str, optional
        Passed to ``load_aggregate``; ``None`` uses its default, which keeps one generation seed
        per configuration (see its ``preferred_seed``).
    use_retries : bool
        Prefer each run's ``*.retry.csv`` (see ``load_predictions``). ``False`` uses only each run's
        original answers.

    Returns
    -------
    TaskData
        With ``predictions`` and ``labels`` sharing one index.
    """
    aggregate_path = Path(aggregate_path or f"results/{task}-aggregated-0-bullet-is.csv")
    aggregate = load_aggregate(aggregate_path, qa_mode=qa_mode, dedup_subset=dedup_subset)
    for model in drop_models or ():
        aggregate = aggregate.drop(aggregate[aggregate["model"] == model].index)

    risk_scores = load_predictions(aggregate=aggregate, results_root=results_root, use_retries=use_retries)
    predictions = binarize_predictions(risk_scores, aggregate)
    meta = build_run_metadata(aggregate, predictions)
    labels = load_labels(aggregate, results_root, index=predictions.index)

    index = predictions.index
    n_before = len(index)
    if common_only:
        index = common_individuals(predictions)
    if label_filter is not None:
        index = index.intersection(labels[labels == label_filter].index)
    # Labels come from the runs' predictions files (see `load_labels`); individuals none of them
    # covers have no label, so keep what is covered and say how much was lost.
    unlabelled = index.difference(labels.index)
    if len(unlabelled):
        logging.warning(
            f"{task}: {len(unlabelled)} of {len(index)} individuals have no label "
            f"(labels read from {aggregate.iloc[0]['predictions_path']}); dropping them"
        )
        index = index.intersection(labels.index)
    # one index for predictions, risk scores and labels, so nothing can silently misalign
    predictions = predictions.loc[index]
    risk_scores = risk_scores.loc[index]
    labels = labels.loc[index]

    return TaskData(
        task=task,
        qa_mode=qa_mode,
        aggregate=aggregate,
        risk_scores=risk_scores,
        predictions=predictions,
        labels=labels,
        meta=meta,
        label_filter=label_filter,
        n_dropped=n_before - len(index),
    )
