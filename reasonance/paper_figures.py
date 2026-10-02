"""Figures that appear in the paper, one function per figure.

Each function computes what it needs through ``reasonance.analysis`` and draws it, so a figure
never depends on another notebook cell having run first; the notebooks in ``notebooks/`` call these
functions. Values are cached in ``results/pair-values/`` purely to skip reloading predictions;
``refresh=True`` recomputes them.

Adding or removing a task needs no change here: the task list is discovered from the aggregates that
exist, and every figure lays itself out for however many tasks it is given.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator, NullFormatter

from reasonance.plotting import (
    ACCURACY_COLOR,
    CATEGORY_COLORS,
    EFFORT_LABEL_PALETTE,
    EFFORT_PALETTE,
    LINE_WIDTH,
    METRIC_LABELS,
    POINT_COLOR,
    REFERENCE_COLOR,
    SCATTER_SIZE,
    STEP_LINE_WIDTH,
    STEP_MARKER_SIZE,
    VOTE_CORRECT_COLOR,
    VOTE_EDGE_COLOR,
    VOTE_EDGE_WIDTH,
    VOTE_SPLIT_COLOR,
    VOTE_WRONG_COLOR,
    add_reference_line,
    count_prefix,
    family_color,
    family_display,
    fraction_tick_labels,
    get_figsize,
    get_figure_prefix,
    light_box_kwargs,
    light_grid,
    pin_axes_left,
    place_legend,
    plot_model_dumbbell,
    plot_vote_movement,
    plot_vote_stairs,
    savefig,
    set_figure_prefix,
    sort_tasks,
    tex_escape,
    vote_colormap,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import Transform
    from matplotlib.typing import ColorType

PAIR_VALUES_DIR = Path("results/pair-values")
PAPER_FIGURES_DIR = Path("results/figures/paper")
# Where each paper figure goes: the paper directory is split so the two sets stay apart when the
# paper is assembled. A figure names its own place; nothing else has to know.
PAPER_MAIN, PAPER_APPENDIX = "main", "appendix"
# Variants drawn from a notebook (one task, another metric, ...) land in the ordinary figure tree,
# so they never overwrite the version that goes in the paper.
FIGURES_ROOT = Path("results/figures")
# Stands in for the task in a file name when the figure's panels ARE the tasks.
ALL_TASKS_PREFIX = "all-tasks"
LENGTH_SECTION = "agreement-by-reasoning-length"

# The knobs the paper figures are produced with. A notebook can override them per call; the file
# name then picks up a suffix, so the paper version on disk stays the paper version.
PAPER_METRIC = "acc_adjusted_agree"

# Task order lives in plotting, so every figure -- paper or exploratory -- uses the same one.
FAMILY_ORDER = ["All", "OpenAI", "Qwen"]
GROUP_ORDER = ["non-reasoning", "reasoning"]
# What the two groups mean in the runs: the open models expose reasoning as a flag, OpenAI as named
# levels, so "reasoning" IS the on/high condition and needs no separate "high" curve beside it.
# The panels do not hold the same things: the pooled panel mixes both developers, only the OpenAI
# models have effort levels, and the open models have a single on state. The legend therefore names
# WHO each curve is, so that one legend is true of all three panels.
GROUP_DISPLAY = {
    "reasoning": "all reasoning (on / high)",
    "non-reasoning": "non-reasoning (off / none)",
    "low": "low (OpenAI only)",
    "medium": "medium (OpenAI only)",
    "high": "high (OpenAI only)",
}
# order the legend as the story reads: the pooled curve, then the levels, then the reference
LEGEND_ORDER = ["reasoning", "low", "medium", "high", "non-reasoning"]
# Effort levels are shades of one colour, which a reader cannot separate in a dense figure, so the
# line style carries them too. One definition for every figure that draws levels.
LEVEL_DASHES = {"low": ":", "medium": "--", "high": "-."}
EFFORT_LEVEL_ORDER = ["none", "low", "medium", "high"]

# Boxes narrower than light_box_kwargs' default: the message is the offset between the two groups,
# and thin boxes leave room for the per-pair dots without widening the panel.
BOX_WIDTH = 0.45
# Pairs of models from different developers: no single family color, so they are drawn in grey
CROSS_FAMILY_LABEL = "cross-family"
CROSS_FAMILY_COLOR = "0.6"
# Tick steps to try, finest first, capped at Y_TICKS_MAX ticks per panel
# finest first: a panel spanning 0.03 needs 0.005, one spanning 0.65 needs 0.1
Y_TICK_STEPS = (0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5)
# An axis with one or two labels cannot be read: the reader cannot tell the spacing, so a value
# between the labels is a guess. Every axis gets at least this many.
MIN_TICKS = 3
# Head room above and below the data in a panel, as a fraction of the data range
PANEL_MARGIN = 0.08
# completed by count_prefix(): "number of correct predictions" / "fraction of correct predictions"
_VOTE_XLABEL = {"correct": "correct predictions", "class": "predictions of class 1"}
# What the vote excess is measured against: each model erring independently at its own accuracy
# (Poisson-binomial). "independent errors" names that assumption; "independent models" would read as
# models built independently of each other.
EXCESS_LABEL = "excess over\nindependent errors"
MAX_GRID_COLS = 3


# Display-label order: the open models' single on state and OpenAI's "high" are one condition.
_EFFORT_LABEL_RANK = {"off / none": -1, "low": 0, "suppressed": 0.5, "medium": 1, "on / high": 2}


def _sorted_tasks(tasks: Collection[str]) -> list[str]:
    """Tasks in the canonical plotting order, see :func:`reasonance.plotting.sort_tasks`."""
    return sort_tasks(tasks)


def save_pair_values(
    frame: pd.DataFrame, task: str, qa_mode: str, kind: str = "effort", root: Path = PAIR_VALUES_DIR
) -> Path:
    """Write the per-pair values a paper figure is drawn from, with provenance.

    The schema is defined in this module, next to its reader, so the writer and the reader cannot
    drift apart. ``metric``, ``written_at`` and ``git_sha`` travel with the values; ``load_pair_values``
    uses them to refuse a set of files that disagree.

    Parameters
    ----------
    frame : pd.DataFrame
        Per-pair rows. Must carry ``value`` and ``metric``; other columns are kept as they are.
    task, qa_mode : str
        Identify the run; they become columns and the file name.
    kind : str
        ``"effort"`` or ``"within-level"``.
    root : Path
        Directory to write into.

    Returns
    -------
    Path
        The file written.
    """
    missing = {"value", "metric"} - set(frame.columns)
    if missing:
        raise ValueError(f"pair values need {sorted(missing)}")
    root.mkdir(parents=True, exist_ok=True)
    out = frame.assign(
        task=task,
        qa_mode=qa_mode,
        written_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        git_sha=_git_sha(),
    )
    path = root / f"{task}-{qa_mode}-{kind}-pairs.csv"
    out.to_csv(path, index=False)
    return path


def _git_sha() -> str:
    """Short SHA of the working tree, or "unknown" outside a repo."""
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return f"{sha}-dirty" if dirty.strip() else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _set_y_ticks(ax: Axes, steps: Sequence[float] = Y_TICK_STEPS, max_ticks: int | None = None) -> None:
    """Round tick steps: the finest of ``steps`` that fits the panel.

    The cap follows the panel's height (about four labels per inch) unless one is given, so a short
    panel -- e.g. a row whose height is proportional to a narrow data range -- does not end up with
    overlapping labels.

    Parameters
    ----------
    ax : Axes
        Panel whose y axis gets the locator.
    steps : sequence of float
        Candidate tick steps, finest first.
    max_ticks : int, optional
        Cap on the number of ticks; derived from the panel's height when None.
    """
    if max_ticks is None:
        fig = ax.get_figure(root=True)
        assert fig is not None
        height_in = ax.get_position().height * fig.get_figheight()
        max_ticks = max(MIN_TICKS, int(height_in * 6))
    max_ticks = max(max_ticks, MIN_TICKS)
    low, high = sorted(ax.get_ylim())
    span = high - low

    def _n_ticks(step: float) -> int:
        """How many multiples of ``step`` actually fall inside the limits."""
        return int(np.floor(high / step)) - int(np.ceil(low / step)) + 1

    # the FINEST round step that still fits within max_ticks: more labels make a value easier to
    # read off, while MIN_TICKS guards the other end -- an axis with two labels cannot be read
    usable = [step for step in steps if _n_ticks(step) >= MIN_TICKS and span / step <= max_ticks]
    if usable:
        ax.yaxis.set_major_locator(MultipleLocator(min(usable)))
    else:  # a span too narrow even for the finest round step
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, min_n_ticks=MIN_TICKS))


def _apply_y_ticks(fig: Figure, axes: np.ndarray, **kwargs: Any) -> None:
    """Set y ticks once the layout has settled.

    The tick step depends on the panel's height, and constrained_layout only fixes the geometry on
    draw: choosing ticks before that uses the placeholder positions of an equal-height grid, which
    gives a short row too few ticks.
    """
    fig.canvas.draw()
    for ax in np.ravel(axes):
        if ax.get_visible():
            _set_y_ticks(ax, **kwargs)


def _save_section_figure(
    fig: Figure,
    name: str,
    tasks: Sequence[str] | None,
    qa_mode: str | None = None,
    figures_dir: str | Path | None = None,
    paper_section: str = PAPER_APPENDIX,
    expected_tasks: Iterable[str] | None = None,
) -> None:
    """Save a figure where it belongs, under the project-wide naming pattern.

    Directory: the notebook's own figure directory (``figures_dir``), with a ``numeric``/``mcq``
    subdirectory inside, since exploratory output piles up and it must stay clear which run produced
    a file. The paper directory stays flat -- it holds few figures and each names its QA mode.

    ``paper_section`` picks the subdirectory of the paper directory: "main" or "appendix".

    ``expected_tasks`` is what the caller ASKED for. A figure that ended up covering fewer tasks --
    because one failed to compute and was skipped -- is not the figure the paper claims, so it goes
    to the exploratory section directory with a warning instead of landing in ``paper/``.

    Name: ``{task}-{qa_mode}-{name}``, where a figure spanning several tasks uses ``all-tasks`` in
    place of the task. The prefix is set for the duration of the save so a notebook's own prefix
    (its single task) cannot leak into a multi-task figure, and restored afterwards.
    """
    qa_mode = qa_mode or "numeric"
    if expected_tasks is not None and set(tasks or ()) != set(expected_tasks):
        logging.warning(
            "%s covers %s, missing %s -- saving to the exploratory directory, not paper/",
            name,
            sorted(tasks or ()),
            sorted(set(expected_tasks) - set(tasks or ())),
        )
        figures_dir = figures_dir or FIGURES_ROOT / LENGTH_SECTION
    task_part = tasks[0] if tasks is not None and len(tasks) == 1 else ALL_TASKS_PREFIX
    previous = get_figure_prefix()
    set_figure_prefix(task_part, qa_mode)
    try:
        # a notebook's FIGURES_DIR usually already ends in its QA mode; do not nest a second one
        if figures_dir is None:
            target = PAPER_FIGURES_DIR / paper_section
        else:
            figures_dir = Path(figures_dir)  # also accept a str
            target = figures_dir if figures_dir.name == qa_mode else figures_dir / qa_mode
        savefig(fig, name, target)
    finally:
        set_figure_prefix(*previous)


def _family_line_color(family: str) -> str:
    """Family color for a pair line, grey for a pair that spans two families."""
    return CROSS_FAMILY_COLOR if family == CROSS_FAMILY_LABEL else family_color(family)


def _variant_name(
    base: str, metric: str, tasks: Sequence[str] | None = None, default_tasks: Sequence[str] | None = None
) -> str:
    """File name for a figure, with a suffix only for settings the name does not already carry.

    The task part of the prefix already names which tasks are in the figure, so only a non-paper
    metric needs spelling out here; ``tasks`` and ``default_tasks`` are accepted but not used.
    """
    return base if metric == PAPER_METRIC else f"{base}-{metric}"


def _task_panels(
    tasks: Sequence[str], height_to_width_ratio: float = 1.0, sharey: bool = False
) -> tuple[Figure, np.ndarray]:
    """A row of panels, one per task, wrapped onto a grid when there are many.

    Returns the figure and a flat array of the visible axes, one per task.
    """
    n_cols = min(MAX_GRID_COLS, len(tasks))
    n_rows = -(-len(tasks) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=get_figsize(nrows=n_rows, ncols=n_cols, height_to_width_ratio=height_to_width_ratio),
        sharey=sharey,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[len(tasks) :]:
        ax.set_visible(False)
    return fig, flat[: len(tasks)]


def effort_main(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    metric: str = PAPER_METRIC,
    metric_label: str | None = None,
    seed_data: pd.DataFrame | None = None,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """MAIN TEXT: agreement with reasoning off vs on, one panel per task.

    Each panel holds the family groups (All pools the families) with one box per reasoning group and
    one dot per matched model pair. Values are seed-averaged in the components: p_o and p_base are
    averaged over seed combinations and the metric is rebuilt from those averages.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed effort pairs (``effort_pairs_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    metric : str
        Pairwise agreement metric.
    metric_label : str, optional
        Y axis label; defaults to the metric's registry label.
    seed_data : pd.DataFrame, optional
        Per seed-combination values; when given, each pair gets a whisker of their standard deviation.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its flat array of task panels.
    """
    from reasonance.analysis import available_tasks, effort_pairs_for_tasks

    _check_data(data, tasks, metric=metric)
    if data is None:
        data = effort_pairs_for_tasks(tasks, metric=metric, refresh=refresh)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    metric_label = metric_label or METRIC_LABELS.get(data["metric"].iloc[0], data["metric"].iloc[0])
    is_paper_version = metric == PAPER_METRIC and set(tasks) == set(available_tasks())

    # spread of a pair's cross-seed combinations, looked up per (task, pair, group) below
    seed_spread = (
        seed_data.groupby(["task", "pair_key", "group"])["value"].std().to_dict() if seed_data is not None else None
    )

    fig, axes = _task_panels(tasks, height_to_width_ratio=1.15, sharey=False)
    for ax, task in zip(axes, tasks):
        panel = data[data["task"] == task]
        families = [f for f in FAMILY_ORDER if f in set(panel["family"])]
        sns.boxplot(
            data=panel,
            x="family",
            y="value",
            hue="group",
            order=families,
            hue_order=GROUP_ORDER,
            palette=CATEGORY_COLORS,
            ax=ax,
            legend=False,
            **light_box_kwargs(width=BOX_WIDTH),
        )
        # Dots at the box centres: seaborn dodges a stripplot across the full categorical width
        # (0.8) while the boxes use BOX_WIDTH, so stripplot(dodge=True) would sit beside its box.
        rng = np.random.default_rng(0)
        for hue_index, group in enumerate(GROUP_ORDER):
            # seaborn centres each hue box at (i - (n-1)/2) * width / n within the category slot
            offset = (hue_index - (len(GROUP_ORDER) - 1) / 2) * BOX_WIDTH / len(GROUP_ORDER)
            for x_index, family in enumerate(families):
                rows = panel[(panel["family"] == family) & (panel["group"] == group)]
                if rows.empty:
                    continue
                # jitter inside the box, never past its edges
                box_width = BOX_WIDTH / len(GROUP_ORDER)
                jitter = rng.uniform(-0.3, 0.3, len(rows)) * box_width
                x = x_index + offset + jitter
                if seed_spread is not None:
                    # One whisker per pair instead of a cloud of its k x k seed combinations: same
                    # information about how far the seeds scatter, but one mark per pair, so the
                    # unit of observation stays the pair and the boxes keep their weight. It is
                    # centred on the pair's OWN value -- the metric rebuilt from the averaged
                    # components -- not on the mean of the per-combination values, which differs
                    # from it by Cov(value, p_base) / mean(1 - p_base).
                    for x_pair, key, value in zip(x, rows["pair_key"], rows["value"]):
                        sd = seed_spread.get((task, key, group))
                        if not sd or not np.isfinite(sd):
                            continue
                        ax.plot(
                            [x_pair, x_pair],
                            [value - sd, value + sd],
                            color=CATEGORY_COLORS[group],
                            lw=0.6 * LINE_WIDTH,
                            alpha=0.45,
                            solid_capstyle="butt",
                            zorder=2,
                        )
                ax.scatter(
                    x,
                    rows["value"],
                    s=SCATTER_SIZE,
                    color=CATEGORY_COLORS[group],
                    alpha=0.9,
                    linewidths=0,
                    zorder=3,
                )
        ax.set_title(task)
        ax.set_xlabel("")
        ax.set_ylabel(metric_label if ax is axes[0] else "")
        # the boxes are grouped by family, but a reader sees the developer behind it
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([family_display(f) for f in families])
    # One SCALE for every panel: the same span of kappa per unit of height, each panel centred on its
    # own data (tasks sit at different levels, so identical limits would leave most of a panel
    # empty). A gap between two boxes then has the same length on the page in every task, and one
    # tick step serves them all.
    spans = [np.subtract(*ax.dataLim.intervaly[::-1]) for ax in axes]
    span = max(spans) * (1 + 2 * PANEL_MARGIN)
    for ax in axes:
        centre = float(np.mean(ax.dataLim.intervaly))
        ax.set_ylim(centre - span / 2, centre + span / 2)
    _set_y_ticks(axes[0])
    step = float(np.diff(axes[0].get_yticks())[0])  # the step chosen for the first panel, for all
    for ax in axes[1:]:
        ax.yaxis.set_major_locator(MultipleLocator(step))

    sns.despine(fig=fig)

    handles = [plt.Line2D([], [], marker="s", ls="none", color=CATEGORY_COLORS[g], label=g) for g in GROUP_ORDER]
    place_legend(axes, handles=handles, labels=GROUP_ORDER, ncol=2)
    if save:
        name = file_name or _variant_name(
            "effort-agreement", metric, None if is_paper_version else tasks, available_tasks()
        )
        _save_section_figure(
            fig,
            name,
            tasks,
            data["qa_mode"].iloc[0] if "qa_mode" in data else None,
            figures_dir,
            paper_section=PAPER_MAIN,
        )
    return fig, axes


def effort_levels(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    metric: str = PAPER_METRIC,
    metric_label: str | None = None,
    save: bool = True,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """APPENDIX: agreement between runs AT THE SAME effort level, one panel per task.

    Only OpenAI exposes graded effort, so each box is the pairs of OpenAI models run at that level;
    "none" is the reasoning-off reference.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed within-level pairs (``within_level_pairs_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    metric : str
        Pairwise agreement metric.
    metric_label : str, optional
        Y axis label; defaults to the metric's registry label.
    save : bool
        Write the figure to disk.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its flat array of task panels.
    """
    from reasonance.analysis import available_tasks, within_level_pairs_for_tasks

    _check_data(data, tasks, metric=metric)
    if data is None:
        data = within_level_pairs_for_tasks(tasks, metric=metric)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    metric_label = metric_label or METRIC_LABELS.get(data["metric"].iloc[0], data["metric"].iloc[0])
    is_paper_version = metric == PAPER_METRIC and set(tasks) == set(available_tasks())

    fig, axes = _task_panels(tasks, height_to_width_ratio=1.15, sharey=False)
    for ax, task in zip(axes, tasks):
        panel = data[data["task"] == task].copy()
        levels = [level for level in EFFORT_LEVEL_ORDER if level in set(panel["level"].astype(str))]
        panel["level"] = panel["level"].astype(str)
        sns.boxplot(
            data=panel,
            x="level",
            y="value",
            hue="level",
            order=levels,
            hue_order=levels,
            palette={level: EFFORT_PALETTE.get(level, "gray") for level in levels},
            ax=ax,
            legend=False,
            **light_box_kwargs(width=BOX_WIDTH),
        )
        # seaborn draws the jitter from numpy's global RNG and takes no seed: seed it for this call
        # only, so the dot positions are the same on every run, and restore the global state after
        rng_state = np.random.get_state()
        np.random.seed(0)
        sns.stripplot(data=panel, x="level", y="value", order=levels, ax=ax, color="0.25", size=2.5, jitter=0.12)
        np.random.set_state(rng_state)
        # OpenAI-only figure, so the raw effort names are the right labels: effort_display() adds
        # the Qwen on/off wording ("off / none", "on / high"), which does not apply here
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_title(task)
        ax.set_xlabel("reasoning effort" if ax is axes[len(axes) // 2] else "")  # one label, centred
        ax.set_ylabel(metric_label if ax is axes[0] else "")
    sns.despine(fig=fig)
    if save:
        name = file_name or _variant_name(
            "effort-levels", metric, None if is_paper_version else tasks, available_tasks()
        )
        _save_section_figure(fig, name, tasks, data["qa_mode"].iloc[0] if "qa_mode" in data else None, figures_dir)
    return fig, axes


def effort_components(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    metric: str = PAPER_METRIC,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
    share_scale: bool = True,
) -> tuple[Figure, np.ndarray]:
    """APPENDIX: the metric taken apart, components as rows and tasks as columns.

    Top row is observed agreement p_o, middle the baseline p_base implied by the pair's accuracies,
    bottom the adjusted metric. Each thin line is ONE model pair with reasoning off and on -- the
    same two base models on both sides -- colored by its family, so the slope is that pair's effect
    free of the spread between different pairs. A rise in p_o that p_base matches is agreement that
    accuracy explains.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed effort pairs (``effort_pairs_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    metric : str
        Pairwise agreement metric shown in the bottom row.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.
    share_scale : bool
        Give the agreement-unit rows one common span (and the metric row its own), each panel
        centred on its data.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes (components x tasks).
    """
    from reasonance.analysis import available_tasks, effort_pairs_for_tasks

    _check_data(data, tasks, metric=metric)
    if data is None:
        data = effort_pairs_for_tasks(tasks, metric=metric, refresh=refresh)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    is_paper_version = metric == PAPER_METRIC and set(tasks) == set(available_tasks())
    # Within-family pairs keep their family color; cross-family pairs (which live only in the
    # pooled "All" rows) are shown too, in grey, since they have no single family.
    within = data[data["family"] != "All"]
    cross = data[(data["family"] == "All") & (data["family_a"] != data["family_b"])].assign(family=CROSS_FAMILY_LABEL)
    data = pd.concat([within, cross], ignore_index=True)
    # abc = p_o - p_base: the agreement left once the accuracy-implied baseline is subtracted,
    # still in agreement units (the adjusted metric divides it by 1 - p_base instead)
    if {"p_o", "p_base"} <= set(data):
        data = data.assign(abc=data["p_o"] - data["p_base"])
    components = [c for c in ["p_o", "p_base", "abc", "value"] if c in data]
    # two-line labels: what the row is, then the symbol, so a tall four-row figure stays readable
    labels = {
        "p_o": "observed agreement\n" + r"$p_o$",
        "p_base": "baseline agreement\n" + r"$p_\mathrm{base}$",
        "abc": "agreement beyond chance\n" + r"$p_o - p_\mathrm{base}$",
        "value": "accuracy-adjusted\n" + r"$\kappa$",
    }
    if metric != PAPER_METRIC:  # a variant metric keeps its registry label
        labels["value"] = METRIC_LABELS.get(data["metric"].iloc[0], data["metric"].iloc[0])

    # All panels are the same size, so an equal span means an equal data-to-page ratio.
    # p_o, p_base and abc are all in agreement units and share ONE span across rows AND tasks, so
    # 0.1 covers the same distance in every one of those panels. kappa is a ratio over a much wider
    # range and shares a span only along its own row (across tasks).
    agreement_components = [c for c in components if c in ("p_o", "p_base", "abc")]

    def _span(members: Sequence[str]) -> float:
        """Widest per-task range of ``members``, padded by the panel margin (1.0 when empty)."""
        spans = [
            float(np.ptp(data.loc[data["task"] == task, member]))
            for member in members
            for task in tasks
            if len(data.loc[data["task"] == task, member])
        ]
        return max(spans) * (1 + 2 * PANEL_MARGIN) if spans else 1.0

    row_spans = {c: _span(agreement_components if c in agreement_components else [c]) for c in components}

    fig, axes = plt.subplots(
        len(components),
        len(tasks),
        # appendix figure: four rows are allowed to take most of a page, so each panel stays legible
        figsize=get_figsize(nrows=len(components), ncols=len(tasks), height_to_width_ratio=1.0),
        squeeze=False,
    )
    # cross-family last in the legend: it is the context, not the message
    families = sorted(f for f in data["family"].unique() if f != CROSS_FAMILY_LABEL)
    if CROSS_FAMILY_LABEL in set(data["family"]):
        families = families + [CROSS_FAMILY_LABEL]
    for row, component in enumerate(components):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            panel = data[data["task"] == task]
            sns.boxplot(
                data=panel,
                x="group",
                y=component,
                hue="group",
                order=GROUP_ORDER,
                hue_order=GROUP_ORDER,
                palette=CATEGORY_COLORS,
                ax=ax,
                legend=False,
                **light_box_kwargs(width=BOX_WIDTH),
            )
            # one line per model pair, off -> on, in its family color
            wide = panel.pivot_table(index=["pair_key", "family"], columns="group", values=component, aggfunc="mean")
            if {"non-reasoning", "reasoning"} <= set(wide.columns):
                wide = wide.dropna(subset=GROUP_ORDER)
                for (_pair, family), values in wide.iterrows():
                    ax.plot(
                        [0, 1],
                        [values["non-reasoning"], values["reasoning"]],
                        color=_family_line_color(family),
                        lw=0.7 * LINE_WIDTH,
                        alpha=0.5 if family == CROSS_FAMILY_LABEL else 0.8,
                        zorder=1 if family == CROSS_FAMILY_LABEL else 2,
                    )
                for pos, group in enumerate(GROUP_ORDER):
                    ax.scatter(
                        np.full(len(wide), pos),
                        wide[group],
                        s=SCATTER_SIZE,
                        color=CATEGORY_COLORS[group],
                        zorder=3,
                    )
            ax.set_title(task if row == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(labels[component] if col == 0 else "")
            if row < len(components) - 1:
                ax.set_xticklabels([])
    # p_o, p_base and abc live at different levels but the SIZE of a change is comparable, so those
    # rows get the same span (same kappa units per axis length), each centred on its own data.
    # Identical limits would flatten the baseline row, since p_base moves in a much narrower band;
    # this way a slope in one row can be compared with a slope in the other by eye.
    if share_scale:
        # each panel gets its ROW's span, centred on its own data
        for row, component in enumerate(components):
            for col, task in enumerate(tasks):
                values = data.loc[data["task"] == task, component]
                if not len(values):
                    continue
                centre = (float(values.min()) + float(values.max())) / 2
                axes[row][col].set_ylim(centre - row_spans[component] / 2, centre + row_spans[component] / 2)

    sns.despine(fig=fig)

    handles = [
        plt.Line2D([], [], color=_family_line_color(f), lw=LINE_WIDTH, label=family_display(f)) for f in families
    ]
    place_legend(
        np.ravel(axes),
        handles=handles,
        labels=[family_display(f) for f in families],
        title="model pair family",
        ncol=len(families),
    )
    if save:
        name = file_name or _variant_name(
            "effort-components", metric, None if is_paper_version else tasks, available_tasks()
        )
        _save_section_figure(fig, name, tasks, data["qa_mode"].iloc[0] if "qa_mode" in data else None, figures_dir)
    return fig, axes


def effort_outcome_split(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """APPENDIX: agreement split by outcome, one line per model pair.

    Pairwise agreement splits exactly into agreement on correct answers and agreement on errors
    (with binary predictions, two runs that are both wrong have given the same answer):
    ``p_o = agree_correct + agree_wrong``. The bottom row is the share of individuals who receive
    the same wrong decision from both models.

    Descriptive, not a test of clustering: for a PAIR the excess over an independent-models null is
    the same at both ends by construction, so any asymmetry here follows from the accuracies. The
    clustering question needs three or more models (see the vote-distribution figures).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Effort pairs with ``value_correct`` and ``value_wrong`` columns; computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes (rows x tasks).
    """
    from reasonance.analysis import effort_pairs_for_tasks

    _check_data(data, tasks)
    if data is None:
        correct = effort_pairs_for_tasks(tasks, metric="agree_correct", refresh=refresh)
        wrong = effort_pairs_for_tasks(tasks, metric="agree_wrong", refresh=refresh)
        keys = ["task", "family", "group", "pair_key"]
        data = correct.merge(wrong[keys + ["value"]], on=keys, suffixes=("_correct", "_wrong"))
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    # within-family pairs keep their family colour; cross-family pairs (only present in the pooled
    # "All" rows) are drawn in grey, as in the components figure
    within = data[data["family"] != "All"]
    cross = data[(data["family"] == "All") & (data["family_a"] != data["family_b"])].assign(family=CROSS_FAMILY_LABEL)
    data = pd.concat([within, cross], ignore_index=True)

    rows = [
        ("value_correct", "observed agreement\n" + r"$P(\mathrm{both\ correct})$"),
        ("value_wrong", "observed agreement\n" + r"$P(\mathrm{both\ incorrect})$"),
    ]
    # Both rows are probabilities of an outcome and share one data-to-page ratio, but P(both incorrect)
    # varies over a much narrower range than P(both correct): giving them equal heights would leave
    # the lower row mostly empty. The row heights are therefore proportional to the ranges, which
    # keeps 0.1 the same distance in both rows without wasting space.
    row_spans = {
        column: max(
            float(np.ptp(data.loc[data["task"] == task, column]))
            for task in tasks
            if len(data.loc[data["task"] == task, column])
        )
        * (1 + 2 * PANEL_MARGIN)
        for column, _label in rows
    }

    # Taller than a plain two-row grid: the rows' heights are proportional to their data ranges, so
    # the narrow P(both incorrect) row is short, and the whole figure has to grow for it to carry enough
    # ticks to be read off the axis.
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=get_figsize(nrows=len(rows), ncols=len(tasks), height_to_width_ratio=1.45),
        squeeze=False,
        gridspec_kw={"height_ratios": [row_spans[column] for column, _label in rows]},
    )
    families = sorted(f for f in data["family"].unique() if f != CROSS_FAMILY_LABEL)
    if CROSS_FAMILY_LABEL in set(data["family"]):
        families = families + [CROSS_FAMILY_LABEL]
    for row, (column, label) in enumerate(rows):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            panel = data[data["task"] == task]
            sns.boxplot(
                data=panel,
                x="group",
                y=column,
                hue="group",
                order=GROUP_ORDER,
                hue_order=GROUP_ORDER,
                palette=CATEGORY_COLORS,
                ax=ax,
                legend=False,
                **light_box_kwargs(width=BOX_WIDTH),
            )
            wide = panel.pivot_table(index=["pair_key", "family"], columns="group", values=column, aggfunc="mean")
            if {"non-reasoning", "reasoning"} <= set(wide.columns):
                wide = wide.dropna(subset=GROUP_ORDER)
                for (_pair, family), values in wide.iterrows():
                    ax.plot(
                        [0, 1],
                        [values["non-reasoning"], values["reasoning"]],
                        color=_family_line_color(family),
                        lw=0.7 * LINE_WIDTH,
                        alpha=0.5 if family == CROSS_FAMILY_LABEL else 0.8,
                        zorder=1 if family == CROSS_FAMILY_LABEL else 2,
                    )
                for pos, group in enumerate(GROUP_ORDER):
                    ax.scatter(
                        np.full(len(wide), pos), wide[group], s=SCATTER_SIZE, color=CATEGORY_COLORS[group], zorder=3
                    )
            ax.set_title(task if row == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(label if col == 0 else "")
            values = panel[column]
            if len(values):
                centre = (float(values.min()) + float(values.max())) / 2
                ax.set_ylim(centre - row_spans[column] / 2, centre + row_spans[column] / 2)
            if row < len(rows) - 1:
                ax.set_xticklabels([])
    sns.despine(fig=fig)
    handles = [
        plt.Line2D([], [], color=_family_line_color(f), lw=LINE_WIDTH, label=family_display(f)) for f in families
    ]
    place_legend(
        np.ravel(axes),
        handles=handles,
        labels=[family_display(f) for f in families],
        title="model pair family",
        ncol=len(families),
    )
    _apply_y_ticks(fig, axes)
    if save:
        name = file_name or "effort-outcome-split"
        _save_section_figure(fig, name, tasks, data["qa_mode"].iloc[0] if "qa_mode" in data else None, figures_dir)
    return fig, axes


def effort_vote_main(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    mode: str = "correct",
    observed: bool = False,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """MAIN TEXT (excess) / APPENDIX (``observed=True``): where the extra agreement goes.

    Per individual, count how many of the nine matched models predict the TRUE label, reasoning off
    and on. The main text shows the excess over independent errors at each model's own accuracy
    (Poisson-binomial), so zero means "accuracy alone explains this". ``observed=True`` draws the
    distribution itself instead -- the context for the excess, in the appendix.

    Nine models is what makes the shape informative: for a PAIR the excess at "all wrong" and "all
    correct" is equal by construction, so only three or more models can show whether the extra
    agreement is concentrated on correct answers or on shared errors.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed vote distributions (``vote_distributions_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    mode : {"correct", "class"}
        Count models predicting the true label, or predicting class 1.
    observed : bool
        Draw the observed distribution instead of the excess.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes (rows x tasks).
    """
    from reasonance.analysis import vote_distributions_for_tasks

    _check_data(data, tasks, metric=f"vote-{mode}")
    if data is None:
        data = vote_distributions_for_tasks(tasks, mode=mode, refresh=refresh)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)

    # the excess carries the message and goes in the main text; the raw distribution is its context
    rows = [("observed", "fraction of\nindividuals")] if observed else [("value", EXCESS_LABEL)]
    # sharey="row": the two rows are different quantities, but within a row the tasks are on the
    # same scale, so their bars can be compared directly and only the left panel needs tick labels.
    # No sharex: tasks can differ in how many matched models they have, and a shared x would
    # silently drop the top vote count of the tasks that have more.
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=get_figsize(nrows=len(rows), ncols=len(tasks), height_to_width_ratio=0.85),
        squeeze=False,
        sharey="row",
    )
    width = 0.4
    for row, (column, label) in enumerate(rows):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            panel = data[data["task"] == task]
            for i, group in enumerate(GROUP_ORDER):
                series = panel[panel["group"] == group].sort_values("k")
                offset = (i - (len(GROUP_ORDER) - 1) / 2) * width
                ax.bar(
                    series["k"] + offset,
                    series[column],
                    width=width,
                    color=CATEGORY_COLORS[group],
                    label=group if (row == 0 and col == 0) else None,
                )
            if column == "value":
                ax.axhline(0, color=POINT_COLOR, lw=0.6 * LINE_WIDTH, zorder=1)
            ax.set_title(task if row == 0 else "")
            ax.set_ylabel(label if col == 0 else "")
            # centred under the grid, but on an axes: a figure-level label collides with the
            # legend, which place_legend puts below the panels
            middle = len(tasks) // 2
            ax.set_xlabel(f"{count_prefix()} {_VOTE_XLABEL[mode]}" if (row == len(rows) - 1 and col == middle) else "")
            if not panel.empty:  # a task with no vote data would give NaN here
                counts = list(range(int(panel["k"].max()) + 1))
                # a tick per bar, labelled as the SHARE of models: tasks can differ in how many
                # models they have, and then a raw count means different things per panel
                ax.set_xticks(counts)
                ax.set_xticklabels(fraction_tick_labels(counts[-1], counts))
                ax.set_xlim(-0.5 - width, counts[-1] + 0.5 + width)
            _set_y_ticks(ax)
            light_grid(ax)  # bar heights are read off the axis, as in the movement figure
    sns.despine(fig=fig)
    place_legend(np.ravel(axes), ncol=2)
    if save:
        name = file_name or f"effort-vote-{mode}{'-observed' if observed else ''}"
        _save_section_figure(
            fig,
            name,
            tasks,
            data["qa_mode"].iloc[0] if "qa_mode" in data else None,
            figures_dir,
            paper_section=PAPER_APPENDIX if observed else PAPER_MAIN,
        )
    return fig, axes


def effort_vote_observed(**kwargs: Any) -> tuple[Figure, np.ndarray]:
    """APPENDIX: the vote distribution behind the main-text excess, see :func:`effort_vote_main`.

    Parameters
    ----------
    **kwargs
        Passed on to :func:`effort_vote_main` (with ``observed=True``).

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes.
    """
    return effort_vote_main(observed=True, **kwargs)


def effort_vote_by_label(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    mode: str = "correct",
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """APPENDIX: the vote excess computed separately for each true label.

    Asks whether the unanimity beyond accuracy is carried by one class -- e.g. models agreeing on
    "below threshold" for everyone who is in fact below it, while disagreeing about the rest. Rows
    are the true label, columns the tasks.

    The null is the same accuracy-based one as in the pooled figure; conditioned on a label the
    accuracy null and the prediction-rate null simply coincide (a run's accuracy on the subgroup IS
    its rate of predicting that class), which is not a problem -- it is still "what independent
    models with these rates would produce".

    Parameters
    ----------
    data : pd.DataFrame, optional
        Vote distributions with a ``label`` column; computed per label when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    mode : {"correct", "class"}
        Count models predicting the true label, or predicting class 1.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes (labels x tasks).
    """
    from reasonance.analysis import vote_distributions_for_tasks

    _check_data(data, tasks, metric=f"vote-{mode}")
    if data is None:
        frames = []
        for label in (0, 1):
            frame = vote_distributions_for_tasks(tasks, mode=mode, label_filter=label, refresh=refresh)
            frames.append(frame.assign(label=label))
        data = pd.concat(frames, ignore_index=True)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    labels = sorted(data["label"].unique())

    fig, axes = plt.subplots(
        len(labels),
        len(tasks),
        figsize=get_figsize(nrows=len(labels), ncols=len(tasks), height_to_width_ratio=0.85),
        squeeze=False,
    )
    width = 0.4
    for row, label in enumerate(labels):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            panel = data[(data["task"] == task) & (data["label"] == label)]
            for i, group in enumerate(GROUP_ORDER):
                series = panel[panel["group"] == group].sort_values("k")
                offset = (i - (len(GROUP_ORDER) - 1) / 2) * width
                ax.bar(
                    series["k"] + offset,
                    series["value"],
                    width=width,
                    color=CATEGORY_COLORS[group],
                    label=group if (row == 0 and col == 0) else None,
                )
            ax.axhline(0, color=POINT_COLOR, lw=0.6 * LINE_WIDTH, zorder=1)
            ax.set_title(task if row == 0 else "")
            # short label: the two-line version collides with the tick labels at this panel size,
            # and the row is already identified by the title
            ax.set_ylabel(f"true label = {label}\nexcess" if col == 0 else "")
            if not panel.empty:
                counts = list(range(int(panel["k"].max()) + 1))
                ax.set_xticks(counts)
                ax.set_xticklabels(fraction_tick_labels(counts[-1], counts))
                ax.set_xlim(-0.5 - width, counts[-1] + 0.5 + width)
            _set_y_ticks(ax)
            light_grid(ax)
    sns.despine(fig=fig)
    axes[-1][len(tasks) // 2].set_xlabel(f"{count_prefix()} {_VOTE_XLABEL[mode]}")
    place_legend(np.ravel(axes), ncol=2)
    if save:
        _save_section_figure(
            fig,
            file_name or f"effort-vote-{mode}-by-label",
            tasks,
            data["qa_mode"].iloc[0] if "qa_mode" in data else None,
            figures_dir,
        )
    return fig, axes


def effort_vote_movement(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """APPENDIX: who moves on the unanimity ladder when reasoning is switched on.

    The vote figure compares two distributions; this one follows the individuals between them. Top
    row: how many models get an individual right with reasoning off (x) and on (y), the diagonal
    blanked because it is the unmoved mass. Bottom row: the same as one step per individual,
    stacked by the consensus level they END at, so a move is read together with its destination.

    Drawing lives in :func:`reasonance.plotting.plot_vote_movement`; this wrapper only supplies the
    numbers and the paper's file name and directory.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed vote transitions (``vote_transitions_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        What :func:`reasonance.plotting.plot_vote_movement` returns.
    """
    from reasonance.analysis import vote_transitions_for_tasks

    _check_data(data, tasks, metric="vote-transitions")
    if data is None:
        data = vote_transitions_for_tasks(tasks, refresh=refresh)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)

    fig, axes = plot_vote_movement(data, tasks=tasks, save=False)
    if save:
        _save_section_figure(
            fig,
            file_name or "effort-vote-movement-nr-to-r",
            tasks,
            data["qa_mode"].iloc[0] if "qa_mode" in data else None,
            figures_dir,
        )
    return fig, axes


# How much accuracy a model may give up and still count as an alternative worth considering.
RASHOMON_EPS = 0.05
# Labels on the agreement axis of the length figures: enough to interpolate between, few enough
# to stay legible in a 2-inch panel.
LENGTH_Y_TICKS_MAX = 8
# Where a median length of zero tokens is drawn on a logarithmic token axis. Small enough to sit
# clearly left of the real data, large enough to stay on the axis.
RAW_ZERO_EPSILON = 1.0
# Written ONCE under each figure (spelled out per panel they overlap between the columns), so they
# are short.
#
# What an individual's "length" is differs between the two figures -- that is the whole difference
# between them. Group figure: the MEDIAN over the panel's runs (one number per individual, the same
# for every curve in the panel). Per-pair figure: the MEAN over the two models' runs of that pair.
# Within a figure, "rank" and "raw" are the SAME quantity, only shown as a percentile within the
# run/pair or in tokens. The dot of a bin sits at that bin's median length (raw) or at the mean
# percentile of the individuals in it (rank).
GROUP_X_LABELS = {
    "rank": "median percentile rank",
    "raw": "median reasoning tokens",
    "zscore": "median z-scored length",
}
PAIR_X_LABELS = {
    "rank": "percentile rank within pair",
    "raw": "mean reasoning tokens within pair",
}
# Both length figures pin their axes here, so the developer grid and the per-pair figure line up
# when they are stacked as subfigures. It must be at least what the widest of them needs: the
# developer grid carries a row label AND the shared y label, and needs 0.124 on its own.
LENGTH_AXES_LEFT = 0.13
# Height one model row contributes to a panel, as a fraction of the text width.
DUMBBELL_ROW_ASPECT = 0.075


VOTE_LENGTH_XLABELS = {
    "zscore": "median z-scored reasoning length",
    "rank": "median percentile rank",
    "raw": "median reasoning tokens",
}


def length_vote_distribution(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    mode: str = "correct",
    variant: str = "rank",
    by_label: bool = False,
    groups: Iterable[str] | None = None,
    value: str = "observed",
    qa_mode: str = "numeric",
    n_bins: int = 20,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> Figure:
    """APPENDIX: the vote distribution of the effort section, cut by realized reasoning length.

    The vote figures ask how the ecosystem votes; the length figures ask where in the population the
    disagreement sits. This is the two together: one column per length bin, the individuals of that
    bin stacked over the vote counts. As everywhere in the length section the x sorts INDIVIDUALS,
    so a shape changing to the right describes the people, not an effect of thinking longer.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed vote distributions by length bin, with a ``label_filter`` column; computed when
        None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    mode : {"class", "correct"}
        ``"class"`` stacks by how many runs say 1 (ends: all no / all yes), ``"correct"`` by how
        many get the true label right (ends: all wrong / all right).
    variant : {"rank", "raw", "zscore"}
        Length definition the individuals are binned on.
    by_label : bool
        One row per true label, i.e. the conditioned version. Then only the reasoning runs are
        drawn by default, since four rows of two groups stop being readable.
    groups : iterable of str, optional
        Which sides to draw as rows. Defaults to both when ``by_label`` is off, reasoning only when
        it is on.
    value : {"observed", "null"}
        ``"null"`` stacks the independent-models distribution instead, for comparison.
    qa_mode : str
        QA mode of the runs.
    n_bins : int
        Number of length bins.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    Figure
        The figure.
    """
    from reasonance.analysis import available_tasks, vote_distributions_by_length_for_tasks

    requested = list(tasks) if tasks is not None else available_tasks()
    groups = list(groups) if groups is not None else (["reasoning"] if by_label else list(GROUP_ORDER))
    labels = [0, 1] if by_label else [None]
    if data is None:
        data = pd.concat(
            [
                vote_distributions_by_length_for_tasks(
                    tasks,
                    mode=mode,
                    variant=variant,
                    qa_mode=qa_mode,
                    n_bins=n_bins,
                    label_filter=label,
                    refresh=refresh,
                ).assign(label_filter=label)
                for label in labels
            ],
            ignore_index=True,
        )
    _check_data(data, tasks, metric=f"vote-{mode}", variant=variant, qa_mode=qa_mode)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    n_levels = int(data["n_runs"].max()) + 1

    # rows are the thing being conditioned on (a label, or a side of the reasoning switch), columns
    # are tasks -- the layout of every other figure in the paper
    rows = [(label, group) for label in labels for group in groups]
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=get_figsize(nrows=len(rows), ncols=len(tasks), height_to_width_ratio=0.75),
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    cmap = vote_colormap(n_levels)
    # every panel of a task spans ALL its length bins: a bin dropped in one row (too few individuals
    # of that label) must show as an empty column, not shift the shared x axis of the whole column
    x_range = data.groupby("task").agg(lo=("bin_left", "min"), hi=("bin_right", "max"))
    for row, (label, group) in enumerate(rows):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            panel = data[(data["task"] == task) & (data["group"] == group)]
            if label is not None and "label_filter" in panel:
                panel = panel[panel["label_filter"] == label]
            if panel.empty:
                ax.set_axis_off()
                continue
            plot_vote_stairs(ax, panel, value=value, mode=mode, n_levels=n_levels, cmap=cmap)
            ax.set_xlim(x_range.loc[task, "lo"], x_range.loc[task, "hi"])
            ax.set_ylim(0, 1)
            ax.set_title(task if row == 0 else "")
            # the effort section's by-label style: what the row is, then the quantity; a column
            # is one length bin, and its stack is the share of THAT bin's individuals
            row_label = f"true label = {label}" if label is not None else group
            ax.set_ylabel(f"{row_label}\nshare in length bin" if col == 0 else "")
            ax.set_xlabel("")  # one label for the figure, on the middle bottom panel below
    axes[-1][len(tasks) // 2].set_xlabel(VOTE_LENGTH_XLABELS.get(variant, variant))
    sns.despine(fig=fig)
    bar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, n_levels - 0.5)),
        ax=axes,
        ticks=range(0, n_levels, max(n_levels // 5, 1)),
    )
    # the effort section's wording, so the same count is named the same way in every vote figure
    bar.set_label(f"{count_prefix()} {_VOTE_XLABEL[mode]}")
    if save:
        suffix = "-by-label" if by_label else ""
        suffix += "" if value == "observed" else f"-{value}"
        _save_section_figure(
            fig,
            file_name or f"length-vote-distribution-{mode}{suffix}-{variant}",
            tasks,
            qa_mode,
            figures_dir,
            paper_section=PAPER_APPENDIX,
            expected_tasks=requested,
        )
    return fig


# The three vote levels the excess line panel follows: the two unanimous ends, and a split vote
# (4 or 5 of 9 correct, i.e. as close to half as nine votes get). Named by the number of correct
# predictions, the unit of every vote figure's axis and colourbar.
EXCESS_LEVELS: dict[str, dict[str, Any]] = {
    "0 correct predictions": {"k": [0]},
    "{n} correct predictions": {"k": "max"},  # n = the number of models, filled in when drawn
    "4 or 5 correct predictions": {"k": [4, 5]},
}


def _excess_level_style(
    ks: Sequence[int], n_levels: int, cmap: Colormap
) -> tuple[tuple[float, float, float, float], bool]:
    """Colour of a vote level's excess line: the colour of its band in the stacked distribution.

    The split band is off-white and would vanish on a white page, so that level's segments are drawn
    in the dark tone that separates the bands of the stack, and its dots keep the band's off-white
    with a thin dark edge.

    Returns the RGBA colour and whether the level is a pale (non-unanimous) one.
    """
    colour = cmap(float(np.mean(ks)) / (n_levels - 1))
    pale = float(np.mean(ks)) not in (0, n_levels - 1)
    return colour, pale


def length_vote_excess(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    variant: str = "rank",
    qa_mode: str = "numeric",
    n_bins: int = 20,
    joint: bool = False,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> Figure:
    """MAIN TEXT / APPENDIX (``joint``): agreement beyond accuracy along the length-sorted population.

    Reasoning runs only, one column per task, in the layout of the effort section's vote figure:
    the excess first, the raw distribution below it as context.

    - Excess row: the observed share of a bin's individuals at a vote count, minus what nine
      independent models with that bin's own accuracies would give, for three levels -- all wrong,
      all correct, and a split vote (4 or 5 of 9 correct) -- as stairs, one value per bin.
    - Distribution row: the observed vote distribution itself, stacked by number of correct
      predictions. Each excess line has the colour of its band here.
    - ``joint=True`` (appendix) adds a heatmap row on top: the excess at EVERY vote count. Within a
      bin it sums to zero over the counts (both distributions sum to one), so it shows which vote
      counts the mass at the unanimous ends comes from.

    The bins are length_main's bins and the votes are averaged over seed draws, as in
    :func:`length_vote_distribution`. The null is recomputed inside each bin, so "the models are more
    accurate on the people they dispatch quickly" is already absorbed. As everywhere in this section,
    the x sorts individuals; it is not a budget.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed "correct" vote distributions by length bin; computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    variant : {"rank", "raw", "zscore"}
        Length definition the individuals are binned on.
    qa_mode : str
        QA mode of the runs.
    n_bins : int
        Number of length bins.
    joint : bool
        Add the heatmap row of the excess at every vote count (appendix version).
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    Figure
        The figure.
    """
    from matplotlib.colors import TwoSlopeNorm

    from reasonance.analysis import available_tasks, vote_distributions_by_length_for_tasks

    requested = list(tasks) if tasks is not None else available_tasks()
    if data is None:
        data = vote_distributions_by_length_for_tasks(
            tasks, mode="correct", variant=variant, qa_mode=qa_mode, n_bins=n_bins, refresh=refresh
        )
    _check_data(data, tasks, metric="vote-correct", variant=variant, qa_mode=qa_mode)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    data = data[data["group"] == "reasoning"]
    n_levels = int(data["n_runs"].max()) + 1
    cmap = vote_colormap(n_levels)
    pale_edge = 0.25  # the dot edge of the off-white split level: just enough to see it

    rows = ["heatmap", "excess", "distribution"] if joint else ["excess", "distribution"]
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=get_figsize(nrows=len(rows), ncols=len(tasks), height_to_width_ratio=0.75),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    row_of = {name: axes[i] for i, name in enumerate(rows)}
    # one colour scale for every heatmap, symmetric around no excess
    limit = float(data["value"].abs().max())
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    for col, task in enumerate(tasks):
        panel = data[data["task"] == task]
        rights = panel.groupby("bin_left")["bin_right"].first().sort_index()
        lefts = rights.index.to_numpy(dtype=float)
        edges = np.append(lefts, rights.to_numpy(dtype=float)[-1])
        axes[0][col].set_title(task)

        if joint:
            ax = row_of["heatmap"][col]
            grid = panel.pivot_table(index="k", columns="bin_left", values="value").sort_index()
            mesh = ax.pcolormesh(
                edges, np.arange(-0.5, n_levels), grid.to_numpy(), cmap="RdBu_r", norm=norm, shading="flat"
            )
            # two lines: the heatmap row is short, and one line would run past the top of the panel
            ax.set_ylabel(f"{count_prefix()}\n{_VOTE_XLABEL['correct']}" if col == 0 else "")
            ax.set_yticks(range(0, n_levels, 3))

        ax = row_of["excess"][col]
        by_k = panel.groupby(["bin_left", "k"])["value"].first().unstack("k").sort_index()
        x = panel.groupby("bin_left")["x"].first().sort_index()
        for spec in EXCESS_LEVELS.values():
            ks = [n_levels - 1] if spec["k"] == "max" else spec["k"]
            colour, pale = _excess_level_style(ks, n_levels, cmap)
            values = by_k[ks].sum(axis=1).to_numpy()
            # stairs, as in length_main: a value is constant across the bin it was computed on, and
            # nothing is drawn between bins that the binning never measured
            ax.hlines(
                values,
                lefts,
                rights.to_numpy(dtype=float),
                color=VOTE_EDGE_COLOR if pale else colour,
                lw=STEP_LINE_WIDTH,
                zorder=2,
            )
            ax.plot(
                x,
                values,
                ls="none",
                marker="o",
                markersize=STEP_MARKER_SIZE,
                color=colour,
                markeredgecolor=VOTE_EDGE_COLOR if pale else colour,
                markeredgewidth=pale_edge if pale else 0.4,
                zorder=3,
            )
        add_reference_line(ax, 0)  # independence: no excess
        light_grid(ax)
        _set_y_ticks(ax, max_ticks=6)  # never fewer than MIN_TICKS, whatever the row's height
        ax.set_ylabel(EXCESS_LABEL if col == 0 else "")

        ax = row_of["distribution"][col]
        plot_vote_stairs(ax, panel, value="observed", mode="correct", n_levels=n_levels, cmap=cmap)
        ax.set_ylim(0, 1)
        ax.set_ylabel("share in\nlength bin" if col == 0 else "")
        ax.set_xlim(edges[0], edges[-1])
    axes[-1][len(tasks) // 2].set_xlabel(GROUP_X_LABELS.get(variant, variant))
    # attached to every row, so all rows keep the same width and the bins line up vertically;
    # shrunk and anchored to sit beside the distribution it describes
    bar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, n_levels - 0.5)),
        ax=np.ravel(axes).tolist(),
        ticks=range(0, n_levels, 3),
        shrink=1 / len(rows) - 0.05,
        anchor=(0.0, 0.0),
    )
    bar.set_label(f"{count_prefix()}\n{_VOTE_XLABEL['correct']}")
    sns.despine(fig=fig)
    handles = []
    for spec in EXCESS_LEVELS.values():
        ks = [n_levels - 1] if spec["k"] == "max" else spec["k"]
        colour, pale = _excess_level_style(ks, n_levels, cmap)
        handles.append(
            plt.Line2D(
                [],
                [],
                color=VOTE_EDGE_COLOR if pale else colour,
                lw=STEP_LINE_WIDTH,
                marker="o",
                markersize=STEP_MARKER_SIZE,
                markerfacecolor=colour,
                markeredgecolor=VOTE_EDGE_COLOR if pale else colour,
                markeredgewidth=pale_edge if pale else 0.4,
            )
        )
    labels = [name.format(n=n_levels - 1) for name in EXCESS_LEVELS]
    place_legend(row_of["excess"].tolist(), handles=handles, labels=labels, ncol=len(EXCESS_LEVELS))
    if joint:
        # The heatmap's colourbar goes in the SAME right-hand column as the distribution's, beside its
        # own row. A second colourbar through constrained_layout would open a column of its own (or
        # narrow one row only), so the layout is settled first and the bar placed in that column.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        column = bar.ax.get_position()
        heat_row = row_of["heatmap"][-1].get_position()
        heat_bar = fig.colorbar(mesh, cax=fig.add_axes((column.x0, heat_row.y0, column.width, heat_row.height)))
        heat_bar.set_label(EXCESS_LABEL)
        # the colours keep the symmetric scale (0 white, equal steps both ways); the bar itself only
        # spans the values that occur, so its dark blue end is not advertised when nothing is there
        heat_bar.ax.set_ylim(float(data["value"].min()), float(data["value"].max()))
        # Its tick labels (-0.5) are wider than the count bar's (0-9), and the layout only reserved
        # room for those: the label would run past the right edge of the page. Move everything left
        # by the difference in tick-label width (the extent of a rotated two-line LaTeX label is not
        # measured reliably, the tick labels are).
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas; missing from the base-class stubs

        def tick_width(cbar: Colorbar) -> float:
            """Width in pixels of the widest drawn tick label of ``cbar``."""
            # ticks outside the bar's range are not drawn and have no extent (NaN): skip them
            widths = [t.get_window_extent(renderer).width for t in cbar.ax.get_yticklabels() if t.get_text()]
            return float(np.nanmax(widths))

        overflow = (tick_width(heat_bar) - tick_width(bar)) / fig.bbox.width
        if overflow > 0:
            for cax in (bar.ax, heat_bar.ax):
                pos = cax.get_position()
                cax.set_position((pos.x0 - overflow, pos.y0, pos.width, pos.height))
            panels = [ax for ax in np.ravel(axes)]
            start = min(ax.get_position().x0 for ax in panels)
            end = max(ax.get_position().x1 for ax in panels)
            scale = (end - overflow - start) / (end - start)
            for ax in panels:
                pos = ax.get_position()
                ax.set_position((start + (pos.x0 - start) * scale, pos.y0, pos.width * scale, pos.height))
    if save:
        _save_section_figure(
            fig,
            file_name or f"length-vote-excess{'-joint' if joint else ''}-{variant}",
            tasks,
            qa_mode,
            figures_dir,
            paper_section=PAPER_APPENDIX if joint else PAPER_MAIN,
            expected_tasks=requested,
        )
    return fig


def accuracy_by_task(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    qa_mode: str = "numeric",
    orient: str = "h",
    ncols: int | None = None,
    save: bool = True,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """APPENDIX: how accurate the matched models are, per task.

    One panel per task, one dumbbell per model, one dot per effort level with the spread across
    seeds. The agreement figures say nothing about whether these models are any good; this one
    places them against the XGBoost baseline (solid) and the Rashomon cutoff (dashed),
    ``RASHOMON_EPS`` below the best model in that task.

    The model axis is shared, so a model sits in the same place in every panel and an absent model
    leaves an empty slot. In the flipped layout accuracy is shared too: the tasks land in a similar
    range, and one scale buys the width that a single row of panels needs.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed per-model accuracies (``model_accuracy``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    qa_mode : str
        QA mode of the runs.
    orient : {"h", "v"}
        ``"h"`` puts the models on the y axis and accuracy on x -- the accuracy axis then gets the
        panel's full width, which is why the panels are laid out as a grid. ``"v"`` flips them.
    ncols : int, optional
        Panels per row. Defaults to 2 for ``"h"`` (a grid: accuracy keeps half the text width
        instead of a third) and one row for ``"v"``.
    save : bool
        Write the figure to disk.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes.
    """
    from reasonance.analysis import model_accuracy

    _check_data(data, tasks, qa_mode=qa_mode)
    if data is None:
        data = model_accuracy(tasks, qa_mode=qa_mode)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    # every model that appears in any task, so an absent model leaves an empty slot instead of
    # shifting the models of the panel it is missing from
    models = list(dict.fromkeys(data["model"]))
    horizontal = orient == "h"
    # The OpenAI models run at four effort levels, the Qwen models at two. An equal slot per model
    # would squeeze four dots into the room two get; weighting by the number of levels (taken over
    # ALL tasks, so every panel of the shared axis places the models identically) spaces them alike.
    slot_weights = data.groupby("model")["reasoning_effort"].nunique().to_dict()

    # Two columns either way: half the text width per panel is what gives the axis its resolution --
    # accuracy when horizontal, and room for a model's effort dots side by side when vertical (the
    # OpenAI models have four of them, twice what a Qwen model has).
    ncols = ncols or 2
    nrows = math.ceil(len(tasks) / ncols)
    if horizontal:
        # the models stack, so the height follows their number; each panel holds them all
        ratio = DUMBBELL_ROW_ASPECT * len(models) * ncols
    else:
        # the models sit side by side; the panel needs height for the accuracy axis AND for the
        # turned model labels below it, which take roughly a third of it
        ratio = 1.05
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=get_figsize(nrows=nrows, ncols=ncols, height_to_width_ratio=ratio),
        # horizontal: the shared y axis is the models. vertical: it is accuracy -- the tasks then
        # read on ONE scale and only the left panel spends width on tick labels, which is what makes
        # a single row of panels wide enough.
        sharey=True,
        sharex=not horizontal,
        squeeze=False,
    )
    flat = np.ravel(axes)
    for ax in flat[len(tasks) :]:
        ax.set_visible(False)  # empty slot, waiting for the next task
    # a shared axis hides the tick labels of every panel that has one below it -- but a panel above
    # an EMPTY slot has nothing below, so it would lose its labels for no reason
    if not horizontal:
        for index in range(len(tasks)):
            if index + ncols >= len(tasks):  # nothing below this panel
                flat[index].tick_params(labelbottom=True)

    handles: list[Patch] | list[Line2D | Patch]  # the dumbbell's patches, then the reference lines
    for ax, task in zip(flat, tasks):
        panel = data[data["task"] == task]
        _, _, handles = plot_model_dumbbell(
            panel,
            value_col="accuracy_mean",
            err_col="accuracy_std",
            model_col="model",
            label_col="model_label",
            category_col="effort_label",
            order=models,
            palette=EFFORT_LABEL_PALETTE,
            connector=False,
            row_separator="band",
            separator_group_col="family",
            orient=orient,
            slot_weights=slot_weights,
            ax=ax,
            xlabel="accuracy",
            title=task,
        )
        reference_axis = "x" if horizontal else "y"
        baseline = panel["baseline_accuracy"].iloc[0]
        if pd.notna(baseline):
            add_reference_line(ax, baseline, axis=reference_axis, linestyle="-")
        add_reference_line(ax, panel["accuracy_mean"].max() - RASHOMON_EPS, axis=reference_axis, linestyle="--")

    handles = handles + [
        plt.Line2D([], [], color=REFERENCE_COLOR, linestyle="-", label="XGBoost"),
        plt.Line2D([], [], color=REFERENCE_COLOR, linestyle="--", label=rf"best in task $-$ {RASHOMON_EPS}"),
    ]
    if not horizontal:
        # accuracy is on y here: round tick steps, and only the left panel spends width on the
        # label, since the scale is shared
        _apply_y_ticks(fig, flat[: len(tasks)])
        for ax in flat[1 : len(tasks)]:
            ax.set_ylabel("")
    # Artist.get_label is typed as returning object in the stubs; these labels are str
    place_legend(flat[: len(tasks)], handles=handles, labels=[h.get_label() for h in handles], ncol=3)  # type: ignore[misc]
    if save:
        # the panels ARE the tasks and the name already starts with "all-tasks", so "by-task" would
        # say it a third time
        name = file_name or ("accuracy" if horizontal else "accuracy-flipped")
        _save_section_figure(fig, name, tasks, qa_mode, figures_dir)
    return fig, axes


def risk_scores(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    qa_mode: str = "numeric",
    ncols: int = 3,
    save: bool = True,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> dict[str, tuple[Figure, np.ndarray]]:
    """APPENDIX (general performance): what the models actually answer, before any thresholding.

    ONE FIGURE PER TASK, one panel per model: the mean over that model's seeds of its risk-score
    histogram, with reasoning off and on. Mass at 0 and 1 is a model answering with near-certainty,
    mass in the middle one that hedges, and spikes on round values are the answers models like to
    give. Pooling the models would hide what the figure is for -- a population-level distribution
    can sit still while individual models move in opposite directions.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed risk-score histograms (``risk_score_histograms``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    qa_mode : str
        QA mode of the runs.
    ncols : int
        Panels per row.
    save : bool
        Write the figure to disk.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    dict
        ``{task: (fig, axes)}``.
    """
    from reasonance.analysis import risk_score_histograms

    _check_data(data, tasks, qa_mode=qa_mode)
    if data is None:
        data = risk_score_histograms(tasks, qa_mode=qa_mode)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    per_model = data[data["model"] != "All"]  # the pooled row is available, but this figure splits

    drawn = {}
    for task in tasks:
        task_rows = per_model[per_model["task"] == task]
        models = list(dict.fromkeys(task_rows["model"]))
        nrows = math.ceil(len(models) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=get_figsize(nrows=nrows, ncols=ncols, height_to_width_ratio=0.55),
            sharex=True,
            squeeze=False,
        )
        flat = np.ravel(axes)
        for ax in flat[len(models) :]:
            ax.set_visible(False)

        for index, (ax, model) in enumerate(zip(flat, models)):
            panel = task_rows[task_rows["model"] == model]
            centres = panel["bin_centre"].unique()
            width = float(centres[1] - centres[0]) if len(centres) > 1 else 0.05
            for group in GROUP_ORDER:
                series = panel[panel["group"] == group].sort_values("bin_centre")
                if series.empty:
                    continue
                # Outline plus a light fill: two filled histograms would hide each other. Drawn on
                # the bin EDGES with where="post" (last edge repeated to close the final bin), so a
                # step sits over its bin instead of half a bin to the right.
                edges = list(series["bin_left"]) + [float(series["bin_left"].iloc[-1]) + width]
                values = list(series["fraction"]) + [float(series["fraction"].iloc[-1])]
                ax.fill_between(edges, values, step="post", color=CATEGORY_COLORS[group], alpha=0.15, lw=0)
                ax.step(edges, values, where="post", color=CATEGORY_COLORS[group], lw=0.8 * LINE_WIDTH, label=group)
            ax.set_title(tex_escape(panel["model_label"].iloc[0]))
            ax.set_xlim(0, 1)
            # bottom row and left column only: a label per panel does not fit at this grid size
            ax.set_xlabel("risk score" if index >= len(models) - ncols else "")
            ax.set_ylabel("mean fraction\nof individuals" if index % ncols == 0 else "")
            _set_y_ticks(ax, max_ticks=4)
        sns.despine(fig=fig)
        place_legend(flat[: len(models)], ncol=2)
        if save:
            _save_section_figure(fig, file_name or "risk-scores", [task], qa_mode, figures_dir)
        drawn[task] = (fig, axes)
    return drawn


def effort_vs_length(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    qa_mode: str = "numeric",
    quantile: float = 0.99,
    min_x_max: float | None = None,
    ncols: int = 3,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> dict[str, tuple[Figure, np.ndarray]]:
    """APPENDIX (general performance): the effort we ASK for vs the reasoning we GET.

    ONE FIGURE PER TASK, one panel per model, one histogram per effort setting over the realised
    reasoning tokens -- the layout of the exploratory length histograms. A claim that "more
    reasoning" was tested rests on the setting actually moving the length, which only the OpenAI
    models expose as levels; the open models have a single on state, so their panel holds one
    histogram and shows how long they think at all.

    The off setting is left out (no trace by construction). Each panel truncates at the
    ``quantile`` of its own mass and reports the longest trace it holds, as a handful of
    20k-token traces would otherwise flatten every distribution.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed reasoning-length histograms (``reasoning_length_histograms``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    qa_mode : str
        QA mode of the runs.
    quantile : float
        Share of the pooled mass the x axis must cover.
    min_x_max : float, optional
        Lower bound on the x axis cutoff, in tokens; defaults to the second largest per-model
        maximum.
    ncols : int
        Panels per row.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    dict
        ``{task: (fig, axes)}``.
    """
    from reasonance.analysis import reasoning_length_histograms

    _check_data(data, tasks, qa_mode=qa_mode)
    if data is None:
        data = reasoning_length_histograms(tasks, qa_mode=qa_mode, refresh=refresh)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    # the off setting is left out: it produces no trace by construction, so it is a spike at zero
    reasoning = data[~data["reasoning_effort"].isin(["none", "0", "nan"])]
    width = float(np.diff(sorted(data["bin_left"].unique())[:2])[0])
    # ONE cutoff for every panel of every task, so a distribution's width means the same thing
    # everywhere: the bin where the average model-and-setting series has `quantile` of its mass.
    pooled = reasoning.groupby("bin_left")["fraction"].mean()
    share = pooled.cumsum() / pooled.sum()
    cutoff = float(share.index[min(share.searchsorted(quantile), len(share) - 1)])
    if min_x_max is None:
        # second largest per-model maximum: the models that run into their generation cap (8000
        # tokens for the open ones) stay inside the axis, while the one model with a 20k tail does
        # not stretch it for everyone
        per_model_max = sorted(reasoning.groupby("model")["max_observed"].max(), reverse=True)
        min_x_max = per_model_max[1] if len(per_model_max) > 1 else 0
    cutoff = max(cutoff, float(min_x_max))
    logging.info("effort_vs_length: x axis truncated at %.0f tokens (quantile %.3f)", cutoff, quantile)

    drawn = {}
    for task in tasks:
        task_rows = reasoning[reasoning["task"] == task]
        models = list(dict.fromkeys(task_rows["model"]))
        nrows = math.ceil(len(models) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            # sized so the three task figures fit on one A4 page: a 3 x 3 grid of short panels.
            # The x axis is shared, so only the bottom row spends height on it.
            figsize=get_figsize(nrows=nrows, ncols=ncols, height_to_width_ratio=0.52),
            sharex=True,
            squeeze=False,
        )
        flat = np.ravel(axes)
        for ax in flat[len(models) :]:
            ax.set_visible(False)

        for ax, model in zip(flat, models):
            thinking = panel = task_rows[task_rows["model"] == model]
            # one entry per DISPLAY label: the open models' single on state and OpenAI's "high" are
            # the same condition, so they share a label and a colour
            for label, series in sorted(panel.groupby("effort_label"), key=lambda kv: _EFFORT_LABEL_RANK.get(kv[0], 0)):
                series = series.sort_values("bin_left")
                colour = EFFORT_LABEL_PALETTE[label]
                edges = list(series["bin_left"]) + [float(series["bin_left"].iloc[-1]) + width]
                values = list(series["fraction"]) + [float(series["fraction"].iloc[-1])]
                ax.fill_between(edges, values, step="post", color=colour, alpha=0.2, lw=0)
                ax.step(edges, values, where="post", color=colour, lw=0.8 * LINE_WIDTH, label=label)
            # Log x: the settings span about three decades of reasoning length (the open models sit
            # near their generation cap), and on a linear axis the short settings collapse into the
            # first pixel column. Empty traces cannot be drawn on it, which is why
            # their share is annotated instead.
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_minor_formatter(NullFormatter())
            # a little headroom past the last bin, so the capped models' spike is not on the frame
            ax.set_xlim(max(width, 30), 1.15 * max(cutoff, width * 2))
            ax.set_title(tex_escape(panel["model_label"].iloc[0]))
            # short panels: the axis is labelled once for the figure, and the per-model facts (the
            # longest trace, and how often the model was asked to think and did not) go inside
            note = [rf"max {int(thinking['max_observed'].max())}"]
            empty = thinking["zero_share"].max()  # off is zero by construction and does not count
            if empty > 0.01:
                note.append(rf"{empty:.0%} empty".replace("%", r"\%"))
            # top right, at tick-label size and in grey: a remark on the panel, not part of the data.
            # The open models peak at the right, so the panel gets headroom for the note to sit above
            # the curve instead of on it.
            ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
            ax.annotate(
                ", ".join(note),
                xy=(0.97, 0.93),
                xycoords="axes fraction",
                ha="right",
                va="top",
                size=plt.rcParams["xtick.labelsize"],
                color=POINT_COLOR,
            )
            _set_y_ticks(ax, max_ticks=3)
        for index, ax in enumerate(flat[: len(models)]):
            ax.set_xlabel("reasoning tokens" if index == len(models) - 2 else "")
            ax.set_ylabel("")  # one shared label for the figure, see supylabel below
        fig.supylabel("fraction of answers")
        sns.despine(fig=fig)
        # one legend for the figure: every panel draws the settings its model has
        handles, labels = {}, []
        for ax in flat[: len(models)]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in handles:
                    handles[label] = handle
                    labels.append(label)
        order = sorted(labels, key=lambda label: _EFFORT_LABEL_RANK.get(label, 0))
        place_legend(
            flat[: len(models)],
            handles=[handles[label] for label in order],
            labels=order,
            title="effort",
            ncol=len(order),
        )
        if save:
            _save_section_figure(fig, file_name or "effort-vs-length", [task], qa_mode, figures_dir)
        drawn[task] = (fig, axes)
    return drawn


def task_ylims(data: pd.DataFrame, margin: float = 0.04) -> dict[str, tuple[float, float]]:
    """Per-task y limits of a length figure, for stacking two of them as subfigures.

    Matplotlib autoscales each figure to its own data, so two figures of the same quantity land on
    different scales. Passing these limits to both pins them to the same range per task.

    Parameters
    ----------
    data : pd.DataFrame
        Length curves with ``task`` and ``value`` columns.
    margin : float
        Padding on each side, as a fraction of the task's value range.

    Returns
    -------
    dict of str to tuple of (float, float)
        ``{task: (low, high)}``, ready for the ``ylim`` argument of the length figures.
    """
    limits = {}
    for task, part in data.groupby("task"):
        low, high = float(part["value"].min()), float(part["value"].max())
        pad = margin * (high - low)
        limits[task] = (low - pad, high + pad)
    return limits


def _check_data(data: pd.DataFrame | None, tasks: Iterable[str] | None = None, **expected: Any) -> None:
    """Refuse a precomputed ``data`` frame that disagrees with the arguments describing it.

    Every figure takes an optional ``data`` -- the values -- next to arguments like ``metric``,
    ``mode``, ``variant`` or ``qa_mode`` that drive the axis labels and the file name. Those
    arguments do NOT select rows: the frame holds what it was computed with. Passing a frame of one
    metric under the name of another would draw those values with the wrong label and save them
    under the wrong file, so the mismatch is an error and the caller recomputes instead.

    ``tasks`` is checked as a subset: a figure may draw part of what the frame holds, but never a
    task the frame has no rows for.
    """
    if data is None:
        return
    for column, wanted in expected.items():
        if wanted is None or column not in data:
            continue
        present = sorted({str(value) for value in data[column].dropna()})
        if present != [str(wanted)]:
            raise ValueError(
                f"data holds {column} {present}, not {str(wanted)!r}; drop data= (or pass "
                "refresh=True) to recompute it for this figure"
            )
    if tasks is not None and "task" in data:
        missing = sorted(set(tasks) - set(data["task"]))
        if missing:
            raise ValueError(f"data has no rows for task(s) {missing}; drop data= to recompute them")


def length_main(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    variant: str = "rank",
    metric: str = PAPER_METRIC,
    metric_label: str | None = None,
    families: Sequence[str] = ("All", "OpenAI", "Qwen"),
    include_non_reasoning: bool = False,
    levels: Sequence[str] = ("low", "medium", "high"),
    ylim: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    accuracy_overlay: bool = False,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """SECTION: agreement against how long the models actually thought.

    READ IT AS A SORT OF THE POPULATION, NOT AS A TREATMENT. The x orders INDIVIDUALS by the length
    they drew; nobody's budget is raised anywhere in this figure. A falling curve therefore says
    "the people the models think about longest are the people they split on", not "thinking longer
    makes models disagree". The manipulated question -- same person, bigger budget -- is the effort
    ladder, and there agreement is flat to rising (see :func:`effort_main` and the effort
    comparison in ``notebooks/2-reasoning-effort.ipynb``).

    One row per task, one column per developer group. Individuals are binned by their reasoning
    length and each bin carries two points: the matched reasoning runs and their non-reasoning
    counterparts on the SAME individuals -- the non-reasoning runs have no length of their own, so
    they borrow the x and the vertical gap in a bin is the reasoning effect at that difficulty.

    Thin lines behind each curve are the seed draws: draw r takes every model's r-th seed, so they
    are disjoint replicates of the whole curve and their spread is run-to-run noise.

    ``variant`` picks the x axis: "rank" (percentile within a run) is the paper version, because it
    is the only one comparable across models with different token scales and immune to the output
    cap; "raw" and "zscore" go to the appendix.

    ``levels`` adds the effort-level curves of the models that have every level (only the OpenAI
    ones), on the same bins, so a level can be read at a fixed realized length. "high" is left out
    of the default: the matched "reasoning" group already IS the on/high condition, so drawing both
    would put the same runs on the panel twice.

    ``accuracy_overlay`` adds the mean accuracy of the same runs on the same bins on a right-hand
    axis. It answers the obvious objection to a falling agreement curve: the models could simply be
    getting everything wrong on the long individuals. Only the matched groups are overlaid -- the
    effort levels would triple the lines.

    The reasoning-off reference is off by default: this figure is about how agreement moves with
    length, while the matched off/on comparison is the effort section's claim.
    ``include_non_reasoning=True`` draws it.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed length curves (``length_curves_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    variant : {"rank", "raw", "zscore"}
        Length definition on the x axis.
    metric : str
        Pairwise agreement metric.
    metric_label : str, optional
        Y axis label; defaults to "mean pairwise" and the metric's registry label.
    families : sequence of str
        Developer groups to draw, one row (or column) each; those absent from ``data`` are skipped.
    include_non_reasoning : bool
        Draw the reasoning-off reference.
    levels : sequence of str
        Effort levels to add as their own curves.
    ylim : tuple of (float, float) or dict, optional
        One ``(low, high)`` for every panel, or ``{task: (low, high)}`` (see :func:`task_ylims`).
    accuracy_overlay : bool
        Add the mean accuracy of the matched groups on a right-hand axis.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes.
    """
    from reasonance.analysis import available_tasks, length_curves_for_tasks

    requested = list(tasks) if tasks is not None else available_tasks()
    _check_data(data, tasks, metric=metric, variant=variant)
    if data is None:
        data = length_curves_for_tasks(tasks, metric=metric, variant=variant, refresh=refresh)
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    families = [f for f in families if f in set(data["family"])]
    # each point is the mean over the MODEL PAIRS in the group (seeds averaged inside a pair), so
    # the axis says so; the per-pair figure, where a line IS one pair, keeps the plain label
    metric_label = metric_label or f"mean pairwise {METRIC_LABELS.get(metric, metric)}"
    groups = [*GROUP_ORDER, *levels]
    # With the levels on the panel the pooled curve is the darkest shade of the same ramp
    # (EFFORT_PALETTE already defines "all reasoning" that way); on its own it keeps the plain
    # reasoning colour, so it matches the other figures.
    shades = bool(set(levels) & set(data["group"]))

    def group_colour(group: str) -> ColorType:
        """Line colour of a group: the effort ramp when levels are drawn, else the category colour."""
        if group == "reasoning":
            return EFFORT_PALETTE["all reasoning"] if shades else CATEGORY_COLORS["reasoning"]
        return CATEGORY_COLORS.get(group) or EFFORT_PALETTE[group]

    # The outermost bins are quantiles of individuals, so their outer edge is the most extreme
    # individual -- on the z-score axis that is several units past the last dot and would leave a
    # third of the panel empty. The segments are clipped to just past the extreme dot instead.
    dots = data[data["draw"] == 0]["x"]
    bin_width = (data["bin_right"] - data["bin_left"]).median()
    clip_lo, clip_hi = dots.min() - 0.5 * bin_width, dots.max() + 0.5 * bin_width

    twins, drawn_accuracy = [], set()
    zero_length = data[data["x"] <= 0] if variant == "raw" else data.iloc[0:0]
    if len(zero_length):  # a log axis has no place for zero; shift it just off the left instead
        data = data.assign(x=data["x"].where(data["x"] > 0, RAW_ZERO_EPSILON))
    if not include_non_reasoning:
        groups = [group for group in groups if group != "non-reasoning"]
        data = data[data["group"] != "non-reasoning"]

    # A single row can afford taller panels, which the kappa axis needs. With one developer group
    # the row is the tasks -- the panel logic of the other main figures -- with one task, the groups.
    single_row = len(tasks) == 1 or len(families) == 1
    # A column is always a TASK and a row a developer group: agreement levels differ by task, so a
    # column can share its y and the developers of one task are read down it.
    grid = (1, max(len(tasks), len(families))) if single_row else (len(families), len(tasks))
    fig, axes = plt.subplots(
        *grid,
        # a single row of three panels is only ~1.8in wide each: without extra height the panel ends
        # up shorter than the legend and axis labels it has to carry
        figsize=get_figsize(nrows=grid[0], ncols=grid[1], height_to_width_ratio=1 if single_row else 0.71),
        sharex=True,  # the same length axis in every panel
        # One kappa scale per COLUMN, i.e. per task: the developers of one task are compared down a
        # column, while tasks reach different agreement levels and should not be forced onto one
        # scale.
        sharey=False if single_row else "col",
        squeeze=False,
    )
    drawn_groups = set()
    for row, task in enumerate(tasks):
        for col, family in enumerate(families):
            ax = axes[0][max(row, col)] if single_row else axes[col][row]  # [developer][task]
            panel = data[(data["task"] == task) & (data["family"] == family)]
            for group in groups:
                series = panel[panel["group"] == group]
                if series.empty:
                    continue
                # the levels are shades of the reasoning colour, with the dash telling them apart
                colour = group_colour(group)
                dashes = LEVEL_DASHES.get(group, "-")
                for draw, draw_rows in series[series["draw"] > 0].groupby("draw"):
                    draw_rows = draw_rows.sort_values("x")
                    ax.plot(
                        draw_rows["x"],
                        draw_rows["value"],
                        color=colour,
                        lw=0.6 * LINE_WIDTH,
                        alpha=0.3,
                        zorder=1,
                        label=None,
                    )
                main = series[series["draw"] == 0].sort_values("x")
                if group in LEVEL_DASHES:
                    # A level sits on the SAME bins as the pooled curve, so it gets the same per-bin
                    # horizontal segment. The dashed connector on top separates the levels where the
                    # bins are narrow and the shades are close.
                    for bin_row in main.itertuples():
                        ax.plot(
                            [max(bin_row.bin_left, clip_lo), min(bin_row.bin_right, clip_hi)],
                            [bin_row.value, bin_row.value],
                            color=colour,
                            lw=STEP_LINE_WIDTH,
                            alpha=0.9,
                            zorder=2,
                        )
                    ax.plot(
                        main["x"],
                        main["value"],
                        color=colour,
                        ls=dashes,
                        lw=0.7 * STEP_LINE_WIDTH,
                        alpha=0.6,
                        zorder=2,
                        marker="o",
                        markersize=0.6 * STEP_MARKER_SIZE,
                    )
                    drawn_groups.add(group)
                    continue
                # the stair spans each bin, so a value is shown as constant across the bin it was
                # computed on; a gap is a bin dropped for holding too few individuals
                for bin_row in main.itertuples():
                    ax.plot(
                        [max(bin_row.bin_left, clip_lo), min(bin_row.bin_right, clip_hi)],
                        [bin_row.value, bin_row.value],
                        color=colour,
                        ls=dashes,
                        lw=STEP_LINE_WIDTH,
                        alpha=0.7,
                        zorder=2,
                    )
                ax.plot(
                    main["x"], main["value"], marker="o", markersize=STEP_MARKER_SIZE, ls="none", color=colour, zorder=3
                )
                drawn_groups.add(group)
            if accuracy_overlay:
                if "accuracy" not in panel:
                    raise ValueError("data has no accuracy column; pass refresh=True to recompute the curves")
                twin = ax.twinx()
                twins.append(twin)
                for group in [g for g in GROUP_ORDER if g in set(panel["group"])]:
                    main = panel[(panel["group"] == group) & (panel["draw"] == 0)].sort_values("x")
                    twin.plot(
                        main["x"],
                        main["accuracy"],
                        color=ACCURACY_COLOR,
                        ls="-" if group == "reasoning" else "--",
                        lw=0.7 * STEP_LINE_WIDTH,
                        marker="D",
                        markersize=0.5 * STEP_MARKER_SIZE,
                        alpha=0.9,
                        zorder=4,
                    )
                    drawn_accuracy.add(group)
                twin.tick_params(axis="y", colors=ACCURACY_COLOR, labelsize=plt.rcParams["ytick.labelsize"])
                twin.set_ylabel("")
                twin.spines["right"].set_color(ACCURACY_COLOR)
            # the task names the column once, the developer names the row on the left
            if single_row:
                ax.set_title(family_display(family) if len(tasks) == 1 else task)
            else:
                ax.set_title(task if col == 0 else "")
            ax.set_xlabel("")  # one label for the figure, on the middle bottom panel below
            if single_row:
                # two lines: "mean pairwise <metric>" is longer than a single-row panel is tall
                ax.set_ylabel(metric_label.replace("pairwise ", "pairwise\n", 1) if max(row, col) == 0 else "")
            else:
                # the row is the developer and the metric is written once for the figure: the two
                # together on every left panel are long enough to collide between rows
                ax.set_ylabel(family_display(family) if row == 0 else "")
            if variant == "rank":
                ax.xaxis.set_major_locator(MultipleLocator(0.25))  # 0 to 1 in quarters: five labels
            if variant == "raw":
                # Token counts span two decades or more (the GPTs think in hundreds, the open models
                # in thousands), so a linear axis packs the short end into a few pixels.
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10))
                ax.xaxis.set_minor_formatter(NullFormatter())
            if ylim is not None:  # pinned, e.g. to match the per-pair figure as a subfigure
                ax.set_ylim(*(ylim[task] if isinstance(ylim, dict) else ylim))
            light_grid(ax)
            # kappa spans most of [0, 1] here and the panels are tall: a handful of labels is too
            # coarse to read a gap off, so this axis asks for more than the height-based default
            _set_y_ticks(ax, max_ticks=LENGTH_Y_TICKS_MAX)
    if variant == "raw" and len(zero_length):
        # A log axis cannot draw a zero, and dropping those bins would hide them. They are shifted
        # to RAW_ZERO_EPSILON instead, which is a lie the figure has to own -- hence the print and
        # the reminder to say so in the caption.
        print(
            f"raw axis: {len(zero_length)} bin(s) with a median length of 0 tokens were drawn at "
            f"x = {RAW_ZERO_EPSILON} so the log axis can show them "
            f"({', '.join(sorted({f'{row.task}/{row.family}/{row.group}' for row in zero_length.itertuples()}))})"
            " -- say so in the caption"
        )
    bottom = np.ravel(axes)[-len(tasks) if not single_row else 0 : None]
    bottom[len(bottom) // 2].set_xlabel(GROUP_X_LABELS.get(variant, variant))
    if not single_row:
        shared_label = fig.supylabel(metric_label)
    sns.despine(fig=fig)
    if twins:
        # one accuracy scale for the whole figure, and the right spine back that despine removed
        lo = min(twin.get_ylim()[0] for twin in twins)
        hi = max(twin.get_ylim()[1] for twin in twins)
        for twin in twins:
            twin.set_ylim(lo, hi)
            twin.spines["right"].set_visible(True)
            twin.spines["top"].set_visible(False)
            _set_y_ticks(twin, max_ticks=LENGTH_Y_TICKS_MAX)
        # one accuracy scale, so only the rightmost twin needs its numbers; on the others they
        # would sit in the next panel
        for twin in twins[:-1]:
            twin.set_yticklabels([])
        twins[-1].set_ylabel("mean accuracy across runs", color=ACCURACY_COLOR)
    shown = [group for group in LEGEND_ORDER if group in drawn_groups] + [
        group for group in groups if group in drawn_groups and group not in LEGEND_ORDER
    ]
    labels = [GROUP_DISPLAY.get(group, group) for group in shown]
    handles = [
        plt.Line2D(
            [], [], marker="o", ls=LEVEL_DASHES.get(g, "-"), lw=STEP_LINE_WIDTH, color=group_colour(g), label=label
        )
        for g, label in zip(shown, labels)
    ]
    for group in [g for g in GROUP_ORDER if g in drawn_accuracy]:
        labels.append("accuracy" if group == "reasoning" else f"accuracy ({group})")
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="D",
                ls="-" if group == "reasoning" else "--",
                lw=STEP_LINE_WIDTH,
                color=ACCURACY_COLOR,
                label=labels[-1],
            )
        )
    place_legend(np.ravel(axes), handles=handles, labels=labels, ncol=min(len(labels), 4))
    pin_axes_left(fig, LENGTH_AXES_LEFT)
    if not single_row:
        # The layout put the shared y label at the figure's edge, and pinning the axes moved the
        # developer names further right: close the gap to half a line.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas; missing from the base-class stubs
        names_left = min(ax.yaxis.label.get_window_extent(renderer).x0 for ax in axes[:, 0]) / fig.bbox.width
        line = plt.rcParams["axes.labelsize"] / 72 / fig.get_figwidth()  # one line of label text
        shared_label.set_x(max(names_left - 1.5 * line, 0.0))
    if save:
        # the default (reasoning only) keeps the plain name; the version carrying the reasoning-off
        # reference says so, as does the developer grid
        suffix = "-with-non-reasoning" if include_non_reasoning else ""
        suffix += "-by-developer" if len(families) > 1 else ""
        # a metric other than the paper's is a variant, not a replacement: it must not overwrite it
        suffix += "" if metric == PAPER_METRIC else f"-{metric}"
        suffix += "-with-accuracy" if accuracy_overlay else ""
        _save_section_figure(
            fig,
            file_name or f"length-agreement-{variant}{suffix}",
            tasks,
            data["qa_mode"].iloc[0] if "qa_mode" in data else None,
            figures_dir,
            # Main text: the paper metric on the rank axis, pooled over developers so the panels
            # are a single row of tasks. Every other combination -- the developer grid, the other
            # length definitions, the reasoning-off reference, the accuracy overlay -- is appendix.
            paper_section=(
                PAPER_MAIN
                if (variant == "rank" and single_row and not include_non_reasoning and metric == PAPER_METRIC)
                else PAPER_APPENDIX
            ),
            expected_tasks=requested,
        )
    return fig, axes


def length_pairs(
    data: pd.DataFrame | None = None,
    tasks: Sequence[str] | None = None,
    variant: str = "rank",
    metric: str = PAPER_METRIC,
    metric_label: str | None = None,
    include_non_reasoning: bool = False,
    levels: Sequence[str] = ("low", "medium"),
    color_by: str | None = None,
    merge_families: bool | None = None,
    ylim: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    show_mean: bool = False,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
    file_name: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """SECTION: the same question per MODEL PAIR, each binned by the length it produced itself.

    One thin line per pair. The group figure bins by a median length over many runs, a consensus
    difficulty proxy; here the x is the pair's own doing, so nothing is borrowed from models outside
    the pair. The reasoning-off counterpart of a pair sits on the same bins, which makes the
    vertical gap within a line a matched comparison -- across lines it is not, since each pair bins
    its own individuals.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Precomputed per-pair length curves (``pair_length_curves_for_tasks``); computed when None.
    tasks : sequence of str, optional
        Tasks to draw; defaults to every task in ``data``.
    variant : {"rank", "raw"}
        ``"rank"`` aligns the bins across pairs and is the paper version; ``"raw"`` keeps token
        units and goes to the appendix.
    metric : str
        Pairwise agreement metric.
    metric_label : str, optional
        Y axis label; defaults to the metric's registry label.
    include_non_reasoning : bool
        Draw each pair's reasoning-off counterpart. Dropping it frees the colour channel.
    levels : iterable of str
        Effort levels to add as their own lines, for the models that have them.
    color_by : {"group", "developer"}, optional
        Defaults to "group" when the reasoning-off lines are drawn, "developer" when they are not.
        ``"group"`` colours reasoning vs non-reasoning. ``"developer"`` colours by the developer of
        the pair, which only reads if the non-reasoning lines are left out -- then the developers
        can share one panel per task instead of one panel each.
    ylim : tuple or dict, optional
        One ``(low, high)`` for every panel, or ``{task: (low, high)}`` -- e.g. from
        :func:`task_ylims` on the group figure's data, so the two stack as subfigures.
    merge_families : bool, optional
        One panel per task with every developer in it. Defaults to True when the non-reasoning lines
        are dropped and the colour carries the developer.
    show_mean : bool
        Draw the mean over the pairs of a colour group as a thick line. Only on the rank axis (True
        is refused on the raw one): every pair is cut into the same 20 quantiles of its OWN length,
        so in rank units all pairs land on one x grid (0.025 ... 0.975) and the mean is over equally
        sized slices at a matched relative length; in token units the pairs sit at different x and
        there is nothing to average vertically. The mean is still over different individuals per
        pair -- it is the average pair, not a population curve.
    save : bool
        Write the figure to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper directory.
    file_name : str, optional
        File name to save under instead of the default.

    Returns
    -------
    tuple of (Figure, np.ndarray)
        The figure and its 2-D array of axes (developers x tasks, or one row when merged).
    """
    from reasonance.analysis import available_tasks, pair_length_curves_for_tasks

    requested = list(tasks) if tasks is not None else available_tasks()
    _check_data(data, tasks, metric=metric, variant=variant)
    if data is None:
        data = pair_length_curves_for_tasks(
            tasks,
            metric=metric,
            variant=variant,
            include_non_reasoning=include_non_reasoning,
            levels=tuple(levels),
            refresh=refresh,
        )
    if not include_non_reasoning:
        data = data[data["group"] != "non-reasoning"]
    tasks = _sorted_tasks(data["task"].unique() if tasks is None else tasks)
    metric_label = metric_label or METRIC_LABELS.get(metric, metric)
    if color_by is None:  # with no reasoning-off lines the colour is free for the developer
        color_by = "group" if include_non_reasoning else "developer"
    if merge_families is None:
        merge_families = not include_non_reasoning and color_by == "developer"
    if show_mean and variant != "rank":
        raise ValueError("show_mean needs the rank axis: on raw tokens the pairs share no x grid")
    families = sorted(set(data["family"]))
    columns = [None] if merge_families else families

    def line_style(row_family: str, group: str) -> tuple[ColorType, str]:
        """Colour and dash for one line: by group, or by developer with the level as the dash."""
        if color_by == "developer":
            colour: ColorType = family_color(row_family)
        else:
            colour = CATEGORY_COLORS.get(group, EFFORT_PALETTE.get(group, CATEGORY_COLORS["reasoning"]))
        dashes = LEVEL_DASHES.get(group, "-")
        return colour, dashes

    # A column is always a TASK: agreement levels differ by task, so a column can share its y and
    # the three columns read like the other paper figures. Merged: one row. Split: a row per
    # developer.
    rows = [None] if merge_families else columns
    fig, axes = plt.subplots(
        len(rows),
        len(tasks),
        figsize=get_figsize(nrows=len(rows), ncols=len(tasks), height_to_width_ratio=0.9),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    seen = {}
    for row, family in enumerate(rows):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            panel = data[data["task"] == task]
            if family is not None:
                panel = panel[panel["family"] == family]
            for (pair, group), line in panel.groupby(["pair", "group"]):
                line = line.sort_values("x")
                colour, dashes = line_style(line["family"].iloc[0], group)
                label = family_display(line["family"].iloc[0]) if color_by == "developer" else group
                label = f"{label} ({group})" if (color_by == "developer" and group != "reasoning") else label
                ax.plot(
                    line["x"],
                    line["value"],
                    color=colour,
                    ls=dashes,
                    lw=0.7 * LINE_WIDTH,
                    alpha=0.6,
                    zorder=2,
                    label=None if label in seen else label,
                )
                seen[label] = (colour, dashes)
            if show_mean:
                # The pairs of one colour group, averaged bin by bin. Averaged by BIN RANK, not by
                # x: every pair is cut into the same quantiles, but its x is the MEASURED mean
                # percentile of that bin, so the values differ in the third decimal and grouping on
                # x would put every pair in its own group.
                keys = ["family", "group"] if color_by == "developer" else ["group"]
                for key, block in panel.groupby(keys):
                    group = key[-1] if isinstance(key, tuple) else key
                    block = block.sort_values(["pair", "x"])
                    block = block.assign(bin_rank=block.groupby("pair").cumcount())
                    mean_curve = block.groupby("bin_rank")[["x", "value"]].mean()
                    # a bin only one pair reached is that pair, not a mean
                    mean_curve = mean_curve[block.groupby("bin_rank").size() > 1]
                    colour, dashes = line_style(block["family"].iloc[0], group)
                    ax.plot(
                        mean_curve["x"], mean_curve["value"], color=colour, ls=dashes, lw=1.8 * LINE_WIDTH, zorder=3
                    )
            # the task names the column once, the developer names the row on the left
            ax.set_title(task if row == 0 else "")
            ax.set_xlabel("")  # one label for the figure, on the middle bottom panel below
            label = metric_label if family is None else f"{family_display(family)}\n{metric_label}"
            ax.set_ylabel(label if col == 0 else "")
            if variant == "rank":
                ax.xaxis.set_major_locator(MultipleLocator(0.25))  # 0 to 1 in quarters: five labels
            if ylim is not None:  # pinned to the group figure's scale, for use as its subfigure
                ax.set_ylim(*(ylim[task] if isinstance(ylim, dict) else ylim))
            light_grid(ax)
            _set_y_ticks(ax, max_ticks=LENGTH_Y_TICKS_MAX)
    axes[-1][len(tasks) // 2].set_xlabel(PAIR_X_LABELS.get(variant, variant))
    sns.despine(fig=fig)
    handles = [
        plt.Line2D([], [], color=colour, ls=dashes, lw=LINE_WIDTH, label=label)
        for label, (colour, dashes) in seen.items()
    ]
    place_legend(np.ravel(axes), handles=handles, labels=list(seen), ncol=min(len(seen), 4))
    pin_axes_left(fig, LENGTH_AXES_LEFT)
    if save:
        suffix = "-with-non-reasoning" if include_non_reasoning else ""
        suffix += "" if metric == PAPER_METRIC else f"-{metric}"
        _save_section_figure(
            fig,
            file_name or f"length-pairs-{variant}{suffix}",
            tasks,
            data["qa_mode"].iloc[0] if "qa_mode" in data else None,
            figures_dir,
            # appendix throughout: it sits beside the group figure as a subfigure
            paper_section=PAPER_APPENDIX,
            expected_tasks=requested,
        )
    return fig, axes


# ---- Figure 1: enabling reasoning (a treatment) vs observing reasoning length (a sort) --------------
# One task, chosen for readability; the section figures carry all three.
FIG1_TASK = "ACSIncome"
FIG1_NAME = "fig1-treatment-behavior"
FIG1_Y_LABEL = METRIC_LABELS[PAPER_METRIC]
# the pair segments are context behind the medians, so they stay neutral and faint
FIG1_PAIR_COLOR = "0.72"
# horizontal spread of the pair endpoints around x = 0 / 1, in data units; deterministic (by rank)
FIG1_JITTER = 0.04
# rows of the grid: sketches, kappa (left) and the bars under it (relative heights); the right plot
# spans the last two
FIG1_HEIGHT_RATIOS = (1.0, 1.3, 0.6)
# The size of all-tasks-numeric-effort-agreement (5.5 x 2.11 in), so it is scaled like
# the other main-text figures wherever the paper includes them at \textwidth. The height may never
# exceed the tallest main-text figure, all-tasks-numeric-length-vote-excess-rank (2.75 in).
FIG1_REL_WIDTH = 1.0
FIG1_HEIGHT_TO_WIDTH = 2.11 / 5.5
FIG1_MAX_HEIGHT = 2.75
# height of the left sketch's row of marks above the sketch's bottom edge, in points
FIG1_SKETCH_CENTRE = 9.0
FIG1_NO_RECOURSE_TITLE = "individuals misclassified by all 9 models"
# the one fact each column holds fixed or lets vary, under its title
FIG1_SUBTITLES = ("same individuals, models, prompts", "fixed setting, different individuals")
# spines and ticks a notch lighter than the style's, so the data are the strongest marks
FIG1_FRAME_WIDTH = 0.6
# vertical step between the strokes of a trace glyph, in points
FIG1_TRACE_STEP = 2.3
FIG1_BAR_WIDTH = 0.4
# the column titles name the kind of question, then what is varied or observed
FIG1_TITLES = ("Intervention: enable reasoning", "Behavior: observe reasoning length")
# x range of the left (intervention) plots: the two settings at 0 and 1, with room for the jitter and the labels
FIG1_A_XLIM = (-0.4, 1.4)
# Width of every panel, in inches, each centred in its column: the kappa panels are the
# narrow ones, so they are less flat and the off -> on increase reads at a steeper angle. In the left
# column the sketch and the bars are wider but keep the kappa panel's inches per x unit (see
# _fig1_set_panel_widths), so off / on stay stacked above each other.
FIG1_PANEL_WIDTHS = {
    "sketch left": 2.1,
    "kappa left": 0.85,
    "no recourse": 1.7,
    "sketch right": 2.5,
    "kappa right": 1.75,
    "longest outcomes": 2.5,
}
# left-column panels drawn on the kappa panel's x scale
FIG1_LEFT_X_SCALED = ("sketch left", "no recourse")
# the left sketch's model blocks sit just outside off / on: between them the people and arrows need
# more room than the narrow panel leaves
FIG1_SKETCH_GRID_X = (-0.55, 1.55)


def _fig1_pct(value: float) -> str:
    """A percentage for a figure label, escaped for usetex (a bare '%' drops the rest of the text)."""
    return tex_escape(f"{value:.1f}%")


def figure1_values(task: str = FIG1_TASK, qa_mode: str = "numeric", refresh: bool = False) -> dict[str, Any]:
    """Everything Figure 1 draws, from the same analysis calls as the section figures.

    Nothing is recomputed differently for this figure: the pairs are ``effort_main``'s, the curve is
    ``length_main``'s pooled reasoning curve, and the no-recourse shares are the 0-correct level of
    the vote figures (effort and length). Only one task is used.

    Parameters
    ----------
    task : str
        The one task the figure shows.
    qa_mode : str
        QA mode of the runs.
    refresh : bool
        Recompute the values instead of reading the cache.

    Returns
    -------
    dict
        ``pairs`` (one row per model pair and setting, pooled over developers), ``medians`` (per
        setting), ``median_gap`` (difference of the medians, not the median paired difference),
        ``curve`` (draw 0 = every seed, 1..k = one seed per model), ``no_recourse`` (share of
        individuals no matched model gets right, off / on, averaged over the seed draws) and
        ``no_recourse_by_length`` (the same share per length bin, reasoning on) and
        ``longest_outcomes`` (the longest bin's shares with all nine correct, mixed, all nine wrong),
        plus ``task``, ``qa_mode`` and ``individuals`` (the individual count of each source).
    """
    from reasonance.analysis import effort_pairs, length_curves, vote_distributions, vote_distributions_by_length

    pairs = effort_pairs(task, metric=PAPER_METRIC, qa_mode=qa_mode, refresh=refresh)
    pairs = pairs[pairs["family"] == "All"]
    medians = pairs.groupby("group")["value"].median()

    curves = length_curves(task, metric=PAPER_METRIC, qa_mode=qa_mode, variant="rank", refresh=refresh)
    curve = curves[(curves["family"] == "All") & (curves["group"] == "reasoning")].sort_values(["draw", "x"])

    votes = vote_distributions(task, qa_mode=qa_mode, refresh=refresh)
    no_recourse = votes[votes["k"] == 0].set_index("group")["observed"]

    by_length = vote_distributions_by_length(task, qa_mode=qa_mode, variant="rank", refresh=refresh)
    by_length = by_length[by_length["group"] == "reasoning"]
    # the longest bin split into unanimous right, split, unanimous wrong (averaged over the draws)
    last = by_length[by_length["bin_left"] == by_length["bin_left"].max()].set_index("k")["observed"]
    top = int(last.index.max())
    longest_outcomes = pd.Series(
        {"all 9 correct": last[top], "mixed": last.loc[1 : top - 1].sum(), "all 9 wrong": last[0]}
    )
    by_length = by_length[by_length["k"] == 0].sort_values("x")

    # the four sources must describe the same individuals, or the panels are about different people
    counts = {
        "effort pairs": int(pairs["n_individuals"].iloc[0]),
        "votes": int(votes["n_individuals"].iloc[0]),
        "length bins": int(curve.loc[curve["draw"] == 0, "n"].sum()),
        "vote bins": int(by_length["n"].sum()),
    }
    if len(set(counts.values())) > 1:
        logging.warning("Figure 1 sources cover different individuals: %s", counts)
    return {
        "task": task,
        "qa_mode": qa_mode,
        "pairs": pairs,
        "medians": medians,
        "median_gap": float(medians["reasoning"] - medians["non-reasoning"]),
        "curve": curve,
        "no_recourse": no_recourse,
        "no_recourse_by_length": by_length,
        "longest_outcomes": longest_outcomes,
        "individuals": counts,
    }


def _fig1_people(
    ax: Axes, xs: Sequence[float] | np.ndarray, y: float, color: str = "0.35", size: float = 10.0, y_offset: float = 0.0
) -> None:
    """Neutral person glyphs (head + shoulders) at data ``(x, y)``; sizes in points, so the aspect
    of the schematic axes does not distort them."""
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import offset_copy

    shoulders = MplPath.wedge(0, 180)  # half disc, flat side at the anchor point
    # ``y_offset`` (points) places the glyph relative to a data height independent of the axes' size
    body_offset = offset_copy(ax.transData, fig=ax.figure, y=y_offset, units="points")  # type: ignore[arg-type]  # stubs reject a SubFigure
    head_offset = offset_copy(ax.transData, fig=ax.figure, y=y_offset + 0.55 * np.sqrt(size) + 1.2, units="points")  # type: ignore[arg-type]  # stubs reject a SubFigure
    ax.scatter(
        xs, [y] * len(xs), s=2.4 * size, marker=shoulders, color=color, lw=0, transform=body_offset, clip_on=False
    )
    ax.scatter(xs, [y] * len(xs), s=size, marker="o", color=color, lw=0, transform=head_offset, clip_on=False)


def _fig1_models(
    ax: Axes,
    x: float,
    y: float,
    color: str,
    n_side: int = 3,
    size: float = 9.0,
    gap: float = 4.2,
    y_offset: float = 0.0,
) -> None:
    """A block of ``n_side**2`` identical model symbols centred on data ``(x, y)``, spaced in points.

    ``y_offset`` (points) moves the block's centre, to place it relative to a baseline.
    """
    from matplotlib.transforms import offset_copy

    for row in range(n_side):
        for col in range(n_side):
            shift = offset_copy(
                ax.transData,
                fig=ax.figure,  # type: ignore[arg-type]  # stubs reject a SubFigure
                x=(col - (n_side - 1) / 2) * gap,
                y=y_offset + ((n_side - 1) / 2 - row) * gap,
                units="points",
            )
            ax.scatter(
                [x], [y], s=size, marker="s", facecolor="white", edgecolor=color, lw=0.7, transform=shift, clip_on=False
            )


def _fig1_trace(ax: Axes, x: float, y: float, lines: int, y_offset: float = 0.0) -> None:
    """A trace glyph: ``lines`` short neutral strokes stacked down from data ``(x, y)``, in points.

    ``y_offset`` (points) shifts the first stroke, so the glyph can be placed relative to a baseline
    independently of the axes' height.
    """
    from matplotlib.transforms import offset_copy

    for i in range(lines):
        shift = offset_copy(ax.transData, fig=ax.figure, y=y_offset - FIG1_TRACE_STEP * i, units="points")  # type: ignore[arg-type]  # stubs reject a SubFigure
        ax.plot([x], [y], marker="_", markersize=7, mew=0.9, color="0.55", transform=shift, clip_on=False)


def _fig1_points(ax: Axes, y_offset: float) -> Transform:
    """Data coordinates shifted up by ``y_offset`` points: places sketch marks relative to the
    axes' bottom edge (data y = 0) at any row height."""
    from matplotlib.transforms import offset_copy

    return offset_copy(ax.transData, fig=ax.figure, y=y_offset, units="points")  # type: ignore[arg-type]  # stubs reject a SubFigure


def _fig1_schematic_treatment(ax: Axes) -> None:
    """Left sketch (intervention): the same individuals go to the same nine models; only the setting flips.

    One row, laid out in points from the bottom edge so it fits a short row: off models <- people ->
    on models, each model block labelled with its setting.
    """
    ax.set_xlim(*FIG1_A_XLIM)
    ax.set_ylim(0, 1)
    ax.axis("off")
    small = plt.rcParams["xtick.labelsize"]
    ax.text(0.5, 1.0, FIG1_SUBTITLES[0], ha="center", va="top", size=small, color=POINT_COLOR)
    centre = FIG1_SKETCH_CENTRE
    _fig1_people(ax, [0.25, 0.5, 0.75], 0, y_offset=centre - 2.5)
    at_centre = _fig1_points(ax, centre)
    left_x, right_x = FIG1_SKETCH_GRID_X
    for x, group, label, shift, ha in (
        (left_x, "non-reasoning", "off", -12, "right"),
        (right_x, "reasoning", "on", 12, "left"),
    ):
        side = np.sign(x - 0.5)
        start = 0.5 + 0.4 * side
        ax.annotate(
            "",
            xy=(x - 0.22 * side, 0),
            xycoords=at_centre,
            xytext=(start, 0),
            textcoords=at_centre,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "0.6",
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 5,
            },
        )
        _fig1_models(ax, x, 0, CATEGORY_COLORS[group], y_offset=centre)
        # the setting named on the model set itself, not left to the colour and the plot below
        ax.annotate(
            label,
            xy=(x, 0),
            xycoords=at_centre,
            xytext=(shift, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            size=small,
        )


def _fig1_schematic_behavior(ax: Axes) -> None:
    """Right sketch (behavior): one setting; individuals ordered and grouped by the length they drew.

    Illustrative only: the stacked lines stand for trace length, not for trace content, and the
    groups at different x hold different individuals. Laid out in points from the bottom edge.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    small = plt.rcParams["xtick.labelsize"]
    ax.text(0.5, 1.0, FIG1_SUBTITLES[1], ha="center", va="top", size=small, color=POINT_COLOR)
    xs = np.linspace(0.06, 0.94, 10)
    n_lines = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # sorted: the grouping, not a trajectory
    top_stroke = FIG1_TRACE_STEP * (max(n_lines) - 1) + 1.0  # the longest trace ends just above y = 0
    _fig1_people(ax, xs, 0, y_offset=top_stroke + 3.0)
    for x, lines in zip(xs, n_lines):
        _fig1_trace(ax, x, 0, lines, y_offset=top_stroke)
    # group boundaries, echoing the bins of the plot below
    for boundary in (xs[1] + xs[2]) / 2, (xs[3] + xs[4]) / 2, (xs[5] + xs[6]) / 2, (xs[7] + xs[8]) / 2:
        ax.annotate(
            "",
            xy=(boundary, 0),
            xytext=(boundary, 0),
            textcoords=_fig1_points(ax, top_stroke + 9.0),
            arrowprops={"arrowstyle": "-", "color": "0.8", "lw": 0.5, "ls": ":", "shrinkA": 0, "shrinkB": 0},
        )


def _fig1_panel_treatment(ax: Axes, values: Mapping[str, Any]) -> None:
    """Left, top: each model pair off vs on (light grey), the medians on top."""
    pairs, medians = values["pairs"], values["medians"]
    wide = pairs.pivot_table(index="pair_key", columns="group", values="value").sort_values("non-reasoning")
    # deterministic jitter, unrelated to the values (a spread by rank would tilt the column of dots)
    offsets = np.random.default_rng(0).permutation(np.linspace(-FIG1_JITTER, FIG1_JITTER, len(wide)))
    for offset, (_, row) in zip(offsets, wide.iterrows()):
        ax.plot(
            [0 + offset, 1 + offset],
            [row["non-reasoning"], row["reasoning"]],
            color=FIG1_PAIR_COLOR,
            lw=0.45,
            alpha=0.6,
            zorder=1,
        )
    for x, group in ((0, "non-reasoning"), (1, "reasoning")):
        ax.scatter(
            x + offsets, wide[group], s=0.4 * SCATTER_SIZE, color=CATEGORY_COLORS[group], alpha=0.35, lw=0, zorder=2
        )
    ax.plot([0, 1], [medians["non-reasoning"], medians["reasoning"]], color="0.15", lw=1.2 * LINE_WIDTH, zorder=3)
    for x, group in ((0, "non-reasoning"), (1, "reasoning")):
        ax.scatter(
            [x], [medians[group]], s=5 * SCATTER_SIZE, color=CATEGORY_COLORS[group], edgecolor="white", lw=0.8, zorder=4
        )
    # the difference of the two medians (not the median paired difference), next to the "on" median,
    # in the gutter: the narrow panel has no room for it
    ax.annotate(
        f"median\n${values['median_gap']:+.2f}$",
        xy=(1, medians["reasoning"]),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        annotation_clip=False,
        size=plt.rcParams["xtick.labelsize"],
    )
    ax.set_xticks([0, 1], ["off", "on"])
    ax.set_xlabel("reasoning")
    ax.set_xlim(*FIG1_A_XLIM)


def _fig1_bars_no_recourse(ax: Axes, values: Mapping[str, Any]) -> None:
    """Left, bottom: share of individuals all nine models get wrong, off and on, as zero-based bars.

    On the same x as the kappa panel above, so off/on read down the column; the values are written on
    the bars, so the panel needs no axis of its own.
    """
    shares = 100 * values["no_recourse"]
    small = plt.rcParams["xtick.labelsize"]
    for x, group in ((0, "non-reasoning"), (1, "reasoning")):
        ax.bar(x, shares[group], width=FIG1_BAR_WIDTH, color=CATEGORY_COLORS[group], lw=0)
        ax.annotate(
            _fig1_pct(shares[group]),
            xy=(x, shares[group]),
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=small,
        )
    # from the full-precision shares, not the rounded labels
    change = shares["reasoning"] - shares["non-reasoning"]
    # next to the "on" bar, in the gutter: the narrow panel has no room between the bars
    ax.annotate(
        f"${change:+.1f}$ pp",
        xy=(1 + FIG1_BAR_WIDTH / 2, shares["reasoning"] / 2),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        size=small,
        annotation_clip=False,
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)  # off / on are named right above
    ax.set_title(FIG1_NO_RECOURSE_TITLE, size=small)
    sns.despine(ax=ax, left=True)


def _fig1_panel_behavior(ax: Axes, values: Mapping[str, Any]) -> None:
    """Right plot (behavior): ``length_main``'s pooled reasoning curve, stairs over each bin plus the seed draws."""
    curve = values["curve"]
    colour = CATEGORY_COLORS["reasoning"]
    for _, draw_rows in curve[curve["draw"] > 0].groupby("draw"):
        ax.plot(draw_rows["x"], draw_rows["value"], color=colour, lw=0.6 * LINE_WIDTH, alpha=0.3, zorder=1)
    main = curve[curve["draw"] == 0]
    for bin_row in main.itertuples():
        ax.plot(
            [bin_row.bin_left, bin_row.bin_right],
            [bin_row.value, bin_row.value],
            color=colour,
            lw=STEP_LINE_WIDTH,
            alpha=0.7,
            zorder=2,
        )
    ax.plot(main["x"], main["value"], marker="o", markersize=1.3 * STEP_MARKER_SIZE, ls="none", color=colour, zorder=3)
    last = main.iloc[-1]
    small = plt.rcParams["xtick.labelsize"]
    ax.annotate(
        rf"last bin: $\kappa \approx {last['value']:.2f}$",
        xy=(last["x"], last["value"]),
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
        size=small,
    )
    handles = [
        plt.Line2D([], [], color=colour, marker="o", markersize=1.3 * STEP_MARKER_SIZE, lw=STEP_LINE_WIDTH),
        plt.Line2D([], [], color=colour, lw=0.6 * LINE_WIDTH, alpha=0.3),
    ]
    ax.legend(
        handles,
        ["mean over\nmodel pairs", "resamples"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # beside the narrow panel, not on it
        frameon=False,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    ax.set_xlim(0, 1)
    # the ends in words instead of 0 and 1, so the ticks stay on one line
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["shorter", "0.25", "0.5", "0.75", "longer"])
    ax.set_xlabel("reasoning length (percentile rank)")


def _fig1_longest_outcomes(ax: Axes, values: Mapping[str, Any]) -> None:
    """Right, bottom: how the individuals of the longest length bin split over the three outcomes.

    One horizontal stacked bar -- all nine correct, mixed, all nine wrong -- in the vote figures'
    colours, led by the sketch's long-trace glyph instead of a text label. Shares are the observed
    vote distribution of the curve's last bin (reasoning on, averaged over the seed draws).
    """
    shares = 100 * values["longest_outcomes"]
    colours = {"all 9 correct": VOTE_CORRECT_COLOR, "mixed": VOTE_SPLIT_COLOR, "all 9 wrong": VOTE_WRONG_COLOR}
    small = plt.rcParams["xtick.labelsize"]
    # the bar sits on the axes' bottom edge (y = 0), the baseline of the left bars in the same row
    height = 0.5
    left = 0.0
    for name, share in shares.items():
        ax.barh(
            height / 2,
            share,
            left=left,
            height=height,
            color=colours[name],
            edgecolor=VOTE_EDGE_COLOR,
            lw=VOTE_EDGE_WIDTH,
        )
        # the wide middle segment holds its own label; the narrow ends are labelled above the bar, like
        # the values on the left bars, so nothing in the row reaches below the shared baseline
        if name == "mixed":
            ax.text(left + share / 2, height / 2, f"{name} {_fig1_pct(share)}", ha="center", va="center", size=small)
        else:
            at_left = left == 0
            ax.annotate(
                f"{name} {_fig1_pct(share)}",
                xy=(left if at_left else left + share, height),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="left" if at_left else "right",
                va="bottom",
                size=small,
            )
        left += share
    # who the bar is about: the sketch's individual with the longest trace, standing on the baseline:
    # placed in points from y = 0, so the lowest stroke sits just above it at any row height
    n_lines = 5
    top_stroke = FIG1_TRACE_STEP * (n_lines - 1) + 1.0
    _fig1_trace(ax, -7, 0, n_lines, y_offset=top_stroke)
    _fig1_people(ax, [-7], 0, y_offset=top_stroke + 3.0)
    ax.set_xlim(-14, 100)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    # the lowest trace stroke must not hang below the baseline the left bars stand on
    ax.figure.canvas.draw()
    # a marker's window extent is padded by its full size in every direction, which overstates how
    # low a flat stroke reaches; measure the stroke centres minus half the stroke width instead
    points_to_pixels = ax.figure.dpi / 72
    lowest = min(
        line.get_transform().transform(line.get_xydata())[:, 1].min()
        - line.get_markeredgewidth() / 2 * points_to_pixels
        for line in ax.lines
    )
    if lowest < ax.get_window_extent().y0 - 0.5:
        logging.warning("Figure 1: the long-trace icon reaches below the bottom row's baseline")


def _draw_figure1(values: Mapping[str, Any]) -> tuple[Figure, dict[str, Axes]]:
    """Figure 1 with every text at most the size of the tick labels and subtitles.

    The sizes stay the bundle's own -- only the larger ones (titles, axis labels) are brought down to
    its tick-label size, for this figure alone.
    """
    small = plt.rcParams["xtick.labelsize"]
    with plt.rc_context({"axes.titlesize": small, "axes.labelsize": small, "legend.fontsize": small}):
        return _draw_figure1_body(values)


def _draw_figure1_body(values: Mapping[str, Any]) -> tuple[Figure, dict[str, Axes]]:
    """Lay out Figure 1 on the ICLR text width; returns the figure and its axes by role.

    Three aligned rows: titles with the sketches, the two kappa plots (same limits, same height and
    position on the page), and a strip under each: the share of individuals all nine models get
    wrong, off and on (left), and how the longest bin's individuals split over all correct / mixed /
    all wrong (right).
    """
    fig = plt.figure(figsize=get_figsize(rel_width=FIG1_REL_WIDTH, height_to_width_ratio=FIG1_HEIGHT_TO_WIDTH))
    if fig.get_figheight() > FIG1_MAX_HEIGHT:
        logging.warning("Figure 1 is %.2f in high, above the %.2f in limit", fig.get_figheight(), FIG1_MAX_HEIGHT)
    grid = fig.add_gridspec(3, 2, height_ratios=FIG1_HEIGHT_RATIOS, width_ratios=(1, 1.3))
    axes = {
        "sketch left": fig.add_subplot(grid[0, 0]),
        "sketch right": fig.add_subplot(grid[0, 1]),
        "kappa left": fig.add_subplot(grid[1, 0]),
        "kappa right": fig.add_subplot(grid[1, 1]),
    }
    axes["no recourse"] = fig.add_subplot(grid[2, 0])  # its x range is set with the panel widths
    axes["longest outcomes"] = fig.add_subplot(grid[2, 1])

    _fig1_schematic_treatment(axes["sketch left"])
    _fig1_schematic_behavior(axes["sketch right"])
    _fig1_panel_treatment(axes["kappa left"], values)
    _fig1_panel_behavior(axes["kappa right"], values)
    _fig1_bars_no_recourse(axes["no recourse"], values)
    _fig1_longest_outcomes(axes["longest outcomes"], values)

    for key, title in (("sketch left", FIG1_TITLES[0]), ("sketch right", FIG1_TITLES[1])):
        # usetex ignores fontweight, the editable SVG ignores \textbf
        bold = rf"\textbf{{{title}}}" if plt.rcParams["text.usetex"] else title
        axes[key].set_title(bold, fontweight="bold")  # centred over its column, like the subtitle
    small = plt.rcParams["xtick.labelsize"]
    ax_left, ax_right = axes["kappa left"], axes["kappa right"]
    axes["no recourse"].set_ylim(0, 1.45 * axes["no recourse"].dataLim.y1)
    for ax in (ax_left, ax_right):
        # the same kappa range on both sides, from the independent-errors baseline to full agreement
        ax.set_ylim(-0.04, 1.06)
        ax.yaxis.set_major_locator(MultipleLocator(0.25))  # labelled quarters
        light_grid(ax)
        for level in (0, 1):
            add_reference_line(ax, level, linestyle=":", linewidth=0.6)
        sns.despine(ax=ax)
    for level, text in ((1, "full agreement"), (0, "independent errors")):
        ax_left.annotate(
            text,
            xy=(FIG1_A_XLIM[0], level),
            xytext=(2, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            size=small,
            color="0.5",
        )
    # once: the right plot shows the same quantity on the same range. One line, even where it is
    # longer than the short panel
    ax_left.set_ylabel(FIG1_Y_LABEL)
    for ax in axes.values():
        for spine in ax.spines.values():
            spine.set_linewidth(FIG1_FRAME_WIDTH)
        ax.tick_params(width=FIG1_FRAME_WIDTH)
    _fig1_set_panel_widths(fig, axes)
    _fig1_check_kappa_alignment(fig, ax_left, ax_right)
    return fig, axes


def _fig1_set_panel_widths(
    fig: Figure, axes: Mapping[str, Axes], widths: Mapping[str, float] = FIG1_PANEL_WIDTHS
) -> None:
    """Give every panel its width (inches), centred in its column.

    Done through the box aspect, so constrained_layout still places everything: after one layout
    pass each panel's height is known, and its aspect is set to height / width. The rows keep their
    heights, so the two kappa panels stay exactly as high as each other. The left sketch and bars
    are wider than the kappa panel but get its inches per x unit, centred on the same x, so their
    x = 0 / 1 sit exactly above / below off / on.
    """
    fig.canvas.draw()
    height = fig.get_figheight()
    for key, ax in axes.items():
        ax.set_box_aspect(ax.get_position().height * height / widths[key])
    centre = sum(FIG1_A_XLIM) / 2
    for key in FIG1_LEFT_X_SCALED:
        half_span = (FIG1_A_XLIM[1] - FIG1_A_XLIM[0]) * widths[key] / widths["kappa left"] / 2
        axes[key].set_xlim(centre - half_span, centre + half_span)


def _fig1_check_kappa_alignment(fig: Figure, ax_left: Axes, ax_right: Axes, tolerance: float = 1e-4) -> float:
    """Warn unless kappa = 0, 0.5 and 1 sit at the same figure height in both columns.

    The two kappa axes share limits and must share their geometry too, or equal kappa differences
    get different lengths on the page (the right curve would look steeper than it is). Returns the
    largest misalignment, as a fraction of the figure height.
    """
    fig.canvas.draw()  # the positions are only final once the layout has run

    def heights(ax: Axes) -> np.ndarray:
        """Figure-fraction heights of kappa = 0, 0.5 and 1 on ``ax``."""
        points = ax.transData.transform([(ax.get_xlim()[0], y) for y in (0, 0.5, 1)])
        return fig.transFigure.inverted().transform(points)[:, 1]

    gap = float(np.max(np.abs(heights(ax_left) - heights(ax_right))))
    if gap > tolerance:
        logging.warning("Figure 1: the two kappa axes are misaligned by %.4f of the figure height", gap)
    return gap


def figure1_table(values: Mapping[str, Any]) -> pd.DataFrame:
    """The plotted values in one long table, with what each statistic is -- no individual-level data.

    Parameters
    ----------
    values : mapping
        What :func:`figure1_values` returns.

    Returns
    -------
    pd.DataFrame
        One row per plotted element, with ``panel``, ``element``, ``value`` and ``statistic``
        columns plus ``task`` and ``qa_mode``.
    """
    rows = []
    for row in values["pairs"].itertuples():
        rows.append(
            {
                "panel": "a",
                "element": "pair",
                "pair": row.pair_key,
                "setting": row.group,
                "value": row.value,
                "p_o": row.p_o,
                "p_base": row.p_base,
                "statistic": "accuracy-adjusted kappa, p_o and p_base averaged over the 5x5 cross-seed run pairs",
                "n": row.n_individuals,
            }
        )
    for group, median in values["medians"].items():
        rows.append(
            {"panel": "a", "element": "median", "setting": group, "value": median, "statistic": "median over 36 pairs"}
        )
    rows.append(
        {"panel": "a", "element": "median gap", "value": values["median_gap"], "statistic": "difference of medians"}
    )
    for group, share in values["no_recourse"].items():
        rows.append(
            {
                "panel": "a",
                "element": "no recourse",
                "setting": group,
                "value": share,
                "statistic": "share of individuals no matched model gets right, mean over seed draws",
            }
        )
    for row in values["curve"].itertuples():
        rows.append(
            {
                "panel": "b",
                "element": "length bin" if row.draw == 0 else "length bin, one draw",
                "setting": "reasoning",
                "draw": row.draw,
                "x": row.x,
                "bin_left": row.bin_left,
                "bin_right": row.bin_right,
                "n": row.n,
                "value": row.value,
                "statistic": "mean over model pairs of accuracy-adjusted kappa, recomputed within the bin",
            }
        )
    for name, share in values["longest_outcomes"].items():
        rows.append(
            {
                "panel": "b",
                "element": f"longest bin: {name}",
                "setting": "reasoning",
                "value": share,
                "statistic": "share of the longest bin's individuals, mean over seed draws",
            }
        )
    for row in values["no_recourse_by_length"].itertuples():
        rows.append(
            {
                "panel": "b",
                "element": "no recourse by length",
                "setting": "reasoning",
                "x": row.x,
                "bin_left": row.bin_left,
                "bin_right": row.bin_right,
                "n": row.n,
                "value": row.observed,
                "statistic": "share of individuals no matched model gets right, mean over seed draws",
            }
        )
    return pd.DataFrame(rows).assign(task=values["task"], qa_mode=values["qa_mode"])


def figure1(
    tasks: Sequence[str] | None = None,
    values: Mapping[str, Any] | None = None,
    save: bool = True,
    refresh: bool = False,
    figures_dir: str | Path | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """MAIN TEXT, Figure 1: enabling reasoning vs observing longer reasoning, on one task.

    Left, an intervention: the same individuals, models and prompts with reasoning off (Qwen off,
    OpenAI none) and on (Qwen on, OpenAI high); one grey segment per model pair, the medians on
    top, and below on the same x the share of individuals all nine models get wrong. Right,
    observed behavior: reasoning on, individuals SORTED by the length they drew --
    different individuals at different x, nobody's budget raised. It is ``length_main``'s pooled
    curve: per bin, the mean over model pairs of kappa recomputed within the bin. The left plot
    summarises by the median, the right one by the mean; both are labelled so.

    Saves PDF/PNG (LaTeX text) to the paper's main directory, an SVG and an ``-editable.pdf`` with
    plain text elements for editing in Inkscape, and the plotted values to ``results/tables``. The
    SVG uses the non-LaTeX fonts of the same tueplots ICLR 2024 bundle (Times-family text, Computer Modern math, as
    LaTeX's times package gives in the PDF) at the same sizes.

    Parameters
    ----------
    tasks : sequence of str, optional
        Only the first is used; defaults to ``FIG1_TASK``.
    values : mapping, optional
        Precomputed :func:`figure1_values`; computed when None.
    save : bool
        Write the figure, its editable copies and the values table to disk.
    refresh : bool
        Recompute the values instead of reading the cache.
    figures_dir : str or Path, optional
        Exploratory figure directory; None saves to the paper's main directory.

    Returns
    -------
    tuple of (Figure, dict of str to Axes)
        The figure and its axes by role.
    """
    task = tasks[0] if tasks else FIG1_TASK
    values = values if values is not None else figure1_values(task, refresh=refresh)
    fig, axes = _draw_figure1(values)
    if save:
        qa_mode = values["qa_mode"]
        _save_section_figure(fig, FIG1_NAME, [task], qa_mode, figures_dir, paper_section=PAPER_MAIN)
        # the same directory _save_section_figure picked
        if figures_dir is None:
            target = PAPER_FIGURES_DIR / PAPER_MAIN
        else:
            figures_dir = Path(figures_dir)  # also accept a str
            target = figures_dir if figures_dir.name == qa_mode else figures_dir / qa_mode
        name = f"{task}-{qa_mode}-{FIG1_NAME}"
        # The editable copy: usetex writes glyphs as paths, which Inkscape cannot edit as text.
        from tueplots import fonts

        # the bundle's own non-LaTeX fonts; text stays <text> elements, not glyph paths
        # (pdf.fonttype 42: TrueType, so the PDF's text also stays text when opened in Inkscape)
        editable = {**fonts.iclr2024(), "svg.fonttype": "none", "pdf.fonttype": 42}
        # embedding matplotlib's Computer Modern TrueType files makes fontTools warn about their
        # (harmless) zero timestamps, once per font; that noise is kept out of the notebook
        font_log = logging.getLogger("fontTools")
        level = font_log.level
        font_log.setLevel(logging.ERROR)
        try:
            with plt.rc_context(editable):
                svg_fig, _ = _draw_figure1(values)
                svg_fig.savefig(Path(target) / f"{name}.svg")
                svg_fig.savefig(Path(target) / f"{name}-editable.pdf")
                plt.close(svg_fig)
        finally:
            font_log.setLevel(level)
        table = figure1_table(values)
        tables_dir = Path("results/tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(tables_dir / f"{name}-values.csv", index=False)
        print(f"Saved {target / name}.svg / -editable.pdf and {tables_dir / name}-values.csv")
    return fig, axes
