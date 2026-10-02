"""Plotting functions.

All figures use the tueplots ICLR 2024 bundle (LaTeX text, Times font, ICLR font sizes and
column width). The style is applied on import. Conventions: figure sizes come from
:func:`get_figsize` and text sizes from the bundle, so no function passes ``figsize=(w, h)`` or
``fontsize=``; layout is ``constrained_layout`` (never ``tight_layout`` or ``bbox_inches="tight"``,
which would break the exact text width); labels built from data go through :func:`tex_escape`
while LaTeX rendering is on.
"""

from __future__ import annotations

import colorsys
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
from tueplots import bundles, figsizes

from reasonance.utils import get_model_family

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from typing import Any, Literal

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType

_FIGURES_DIR = Path(__file__).parent.parent / "results/figures"

GOLDEN_RATIO = (5**0.5 - 1) / 2  # 0.618

# consistent mark sizes across all figures (tuned for the 5.5in ICLR text width)
LINE_WIDTH = 1.0
MARKER_SIZE = 3.0  # for ax.plot(markersize=...)
SCATTER_SIZE = 8.0  # for ax.scatter(s=...), in points^2
# Stair plots (a value per bin, drawn as a dot over the bin's span) carry many more marks than a
# line plot, so they get smaller dots and a slightly heavier stair. One definition, so every
# stair figure looks the same.
STEP_MARKER_SIZE = 0.6 * MARKER_SIZE
STEP_LINE_WIDTH = 0.9 * LINE_WIDTH
FILL_ALPHA = 0.3
# points drawn on top of boxes/categories that already encode the grouping: coloring them by the
# same variable would repeat what the axis says, so they stay neutral
POINT_COLOR = "0.25"


def _fix_tex_font_lookup() -> None:
    """Make matplotlib's TeX font lookup work inside a conda environment.

    Matplotlib locates TeX fonts by running the system ``luatex``. Conda's activation hook
    (``activate.d/libstdcxx.sh``) puts the environment's ``lib/`` on ``LD_LIBRARY_PATH``, so
    ``luatex`` loads conda's newer ``libz`` and aborts; matplotlib then reports the font as missing
    ("searched for a file named 'ptmr7t.tfm' ... but could not find it").

    Dropping the conda entries from ``os.environ`` only affects processes started from here (the
    TeX subprocesses), not libraries already loaded in this one. Modules imported afterwards look
    for their shared libraries on the shortened path, so this runs after the imports at the top of
    this module -- do imports first, then call :func:`set_plot_style`.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH")
    if not ld_path:
        return
    prefixes = tuple(p for p in (os.environ.get("CONDA_PREFIX"), sys.prefix) if p)
    kept = [entry for entry in ld_path.split(os.pathsep) if entry and not entry.startswith(prefixes)]
    if len(kept) == len(ld_path.split(os.pathsep)):
        return  # nothing conda-related on the path
    if kept:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    else:
        del os.environ["LD_LIBRARY_PATH"]

    # drop the cached (crashed) luatex process and any negative lookups made before the path changed
    from matplotlib import dviread

    # private/lru_cache-wrapped APIs that matplotlib's stubs do not describe
    dviread.find_tex_file.cache_clear()  # type: ignore[attr-defined]
    dviread._LuatexKpsewhich.__new__.cache_clear()  # type: ignore[attr-defined]


def set_plot_style(usetex: bool = True, seaborn_style: str = "ticks", rc: dict[str, Any] | None = None) -> None:
    """Apply the tueplots ICLR 2024 bundle to matplotlib's rcParams.

    Called once on import of this module. rcParams are reset first so the result does not depend
    on import order -- importing ``folktexts``, for example, applies a seaborn theme
    with a grid and grey spines. Call this again after any import or ``sns.set_theme`` that
    happens later in a session.

    Parameters
    ----------
    usetex : bool
        Render text with LaTeX (matches the paper font). Labels coming from data must then be
        passed through :func:`tex_escape`.
    seaborn_style : str
        Seaborn base style applied before the bundle; "ticks" matches the ``sns.despine()`` calls
        in this module.
    rc : dict of str to Any, optional
        Extra rcParams applied on top of the bundle (avoid font sizes here).
    """
    if usetex:
        _fix_tex_font_lookup()
    plt.rcdefaults()
    sns.set_theme(style=seaborn_style)
    plt.rcParams.update(bundles.iclr2024(usetex=usetex))
    # the bundle leaves these at matplotlib defaults ("large"/"medium"), which stand out next to
    # the ICLR sizes -- tie them to the bundle's own sizes so no plotting function sets a fontsize.
    # Layout stays as the bundle sets it: constrained_layout on, autolayout (tight_layout) off, and
    # no savefig bbox="tight" -- cropping on save would break the exact ICLR text width.
    plt.rcParams.update(
        {
            "legend.title_fontsize": plt.rcParams["legend.fontsize"],
            "figure.titlesize": plt.rcParams["axes.titlesize"],
            "figure.labelsize": plt.rcParams["axes.labelsize"],
            "savefig.dpi": 300,
            "lines.linewidth": LINE_WIDTH,
        }
    )
    # Ticks: sns.set_theme applies seaborn's "notebook" context (6 pt ticks, 1.25 wide), sized for a
    # screen. At paper size that puts ~10 pt between the tick labels and the axis, so a y label sits
    # far from the axis it names. Shorter ticks and tighter pads, set once for every figure.
    plt.rcParams.update(
        {
            **{f"{axis}tick.major.size": 3.5 for axis in "xy"},  # type: ignore[misc]
            **{f"{axis}tick.major.width": 0.8 for axis in "xy"},  # type: ignore[misc]
            **{f"{axis}tick.minor.size": 2.0 for axis in "xy"},  # type: ignore[misc]
            **{f"{axis}tick.minor.width": 0.6 for axis in "xy"},  # type: ignore[misc]
            **{f"{axis}tick.major.pad": 2.0 for axis in "xy"},  # type: ignore[misc]
            **{f"{axis}tick.minor.pad": 2.0 for axis in "xy"},  # type: ignore[misc]
            "axes.labelpad": 2.0,
        }
    )
    if rc is not None:
        plt.rcParams.update(rc)  # type: ignore[arg-type]  # stubs type rcParams keys as a closed Literal set


def get_figsize(
    nrows: int = 1,
    ncols: int = 1,
    rel_width: float = 1.0,
    height_to_width_ratio: float = GOLDEN_RATIO,
) -> tuple[float, float]:
    """Figure size fitting the ICLR 2024 text width.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid; height scales with ``nrows / ncols``.
    rel_width : float
        Fraction of the text width (e.g. 0.5 for a half-width figure).
    height_to_width_ratio : float
        Height/width ratio of a single subplot. For a figure with one row per model, raise it until
        the rows have room; the bundle then scales the total height by ``nrows``.

    Returns
    -------
    tuple of (float, float)
        ``(width, height)`` in inches, ready for ``figsize=``.
    """
    return figsizes.iclr2024(
        rel_width=rel_width, nrows=nrows, ncols=ncols, height_to_width_ratio=height_to_width_ratio
    )["figure.figsize"]


# Only the characters matplotlib's usetex pipeline does not escape on its own. It handles "_", so
# escaping that here would render a literal backslash. It does NOT handle "%" (a LaTeX comment: the
# rest of the label silently disappears) or "~" (a non-breaking space: the tilde renders as a gap).
_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "&": r"\&",
    "$": r"\$",
    "#": r"\#",
    "^": r"\textasciicircum{}",
    "{": r"\{",
    "}": r"\}",
}


def tex_escape(text: object) -> str:
    """Escape LaTeX special characters in a data-derived label.

    No-op when ``text.usetex`` is off. Do not apply to labels that contain intended math
    (e.g. ``METRIC_LABELS``). Non-strings are converted with ``str``.

    Parameters
    ----------
    text : object
        Label to escape; converted with ``str`` first.

    Returns
    -------
    str
        The label, with LaTeX special characters escaped when ``text.usetex`` is on.
    """
    text = str(text)
    if not plt.rcParams["text.usetex"]:
        return text
    return "".join(_TEX_SPECIAL.get(c, c) for c in text)


set_plot_style()


# Parts put in front of every saved figure name (task, QA mode), so a file is identifiable on its
# own -- a figure often travels without its directory. Notebooks set this once via set_figure_prefix.
_FIGURE_PREFIX_PARTS: tuple[str, ...] = ()


def set_figure_prefix(*parts: str | None) -> tuple[str, ...]:
    """Prefix every :func:`savefig` name with these parts, e.g. ``set_figure_prefix(TASK, QA_MODE)``.

    Call once per notebook, after the task and QA mode are chosen. Call with no arguments (or None)
    to clear it. Parts already at the start of a file name are not repeated, so names that already
    carry the QA mode stay as they are.

    Parameters
    ----------
    *parts : str or None
        Name parts, in order; None and empty parts are skipped, surrounding dashes stripped.

    Returns
    -------
    tuple of str
        The prefix parts now in effect.
    """
    global _FIGURE_PREFIX_PARTS
    _FIGURE_PREFIX_PARTS = tuple(str(p).strip("-") for p in parts if p)
    return _FIGURE_PREFIX_PARTS


def get_figure_prefix() -> tuple[str, ...]:
    """The prefix parts currently applied to saved figure names.

    Returns
    -------
    tuple of str
        The parts set by the last :func:`set_figure_prefix` call (empty when none).
    """
    return _FIGURE_PREFIX_PARTS


def figure_name(file_name: str) -> str:
    """File name with the configured prefix parts applied (idempotent, order-preserving).

    Empty name components (e.g. an optional suffix that is "") would leave "--" in the file name,
    so runs of dashes are collapsed and stray leading/trailing ones dropped.

    Parameters
    ----------
    file_name : str
        Base file name, without extension.

    Returns
    -------
    str
        The prefixed file name.
    """
    rest = file_name
    for part in _FIGURE_PREFIX_PARTS:
        if rest.startswith(f"{part}-"):
            rest = rest[len(part) + 1 :]  # already there; re-added below in the canonical order
    name = "-".join([*_FIGURE_PREFIX_PARTS, rest]) if _FIGURE_PREFIX_PARTS else rest
    return re.sub(r"-{2,}", "-", name).strip("-")


def label_filter_tag(label_filter: int | bool | None) -> str | None:
    """Name token for a ground-truth label filter, for use as a :func:`set_figure_prefix` part.

    Parameters
    ----------
    label_filter : int, bool or None
        1 = positives only, 0 = negatives only, None = no filter.

    Returns
    -------
    str or None
        ``"pos_label_only"``, ``"neg_label_only"``, or None when nothing is filtered (the prefix
        then simply omits the part).
    """
    if label_filter is None:
        return None
    return "pos_label_only" if int(label_filter) == 1 else "neg_label_only"


def pin_axes_left(fig: Figure, left: float) -> None:
    """Put the axes of ``fig`` at a fixed distance from the left edge, after the layout has run.

    Two figures meant to stack as subfigures must share the x of their axes, but constrained_layout
    sizes the left margin from whatever decorations a figure happens to carry -- a row label in one
    and none in the other is enough to shift the panels against each other. This lets the layout do
    its work, then rescales the axes horizontally so the leftmost one starts at ``left`` (a figure
    fraction) while the right edge stays put.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure whose axes are moved; its layout engine is switched off afterwards.
    left : float
        Figure-fraction x of the leftmost axes. Too small clips the tick labels, so check the
        figure after changing it.
    """
    fig.canvas.draw()  # let constrained_layout place everything first, then freeze it
    fig.set_layout_engine("none")
    positions = [(ax, ax.get_position()) for ax in fig.axes]
    if not positions:
        return
    start = min(pos.x0 for _, pos in positions)
    end = max(pos.x1 for _, pos in positions)
    if left < start - 1e-3:
        # moving the axes left of where the layout put them eats the space it reserved for the tick
        # and axis labels, which then collide -- pass a larger ``left`` instead
        logging.warning("pin_axes_left(%.3f) squeezes the left labels: the layout needs %.3f", left, start)
    if end <= left or abs(start - left) < 1e-4:
        return
    scale = (end - left) / (end - start)
    for ax, pos in positions:
        ax.set_position((left + (pos.x0 - start) * scale, pos.y0, pos.width * scale, pos.height))


def savefig(fig: Figure, file_name: str, figures_dir: str | Path | None = None) -> None:
    """Save figure as .png and .pdf at the figure's own size (constrained_layout, no tight bbox).

    The name is prefixed with whatever :func:`set_figure_prefix` was given, so every file of a task
    is identifiable without its directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    file_name : str
        Base name, without extension; the prefix is applied by :func:`figure_name`.
    figures_dir : str or Path, optional
        Defaults to ``results/figures``.
    """
    if figures_dir is None:
        figures_dir = _FIGURES_DIR
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    file_name = figure_name(file_name)
    for ext in (".png", ".pdf"):
        fig.savefig(figures_dir / f"{file_name}{ext}")
    print(f"Saved figure to {figures_dir / file_name}.png/.pdf")


METRIC_LABELS = {
    "kappa": r"Cohen's $\kappa$",
    "acc_adjusted_agree": r"accuracy-adjusted $\kappa$",
    "abc": "agreement beyond chance",
    "acc_baseline": r"baseline agreement $p_\mathrm{base}$ (accuracy)",
    "pred_baseline": r"baseline agreement $p_\mathrm{base}$ (prediction rate)",
    "observed": "observed agreement",
}

CATEGORY_COLORS = {
    "reasoning": "#4C72B0",
    "non-reasoning": "#DD8452",
    "suppressed": "#F2C9A8",
}


# create distinct shades of the base color
def shades_by_hsv(
    base_color: ColorType,
    n: int = 5,
    sat_range: tuple[float, float] = (0.3, 1.0),
    val_range: tuple[float, float] = (1.0, 0.5),
) -> list[tuple[float, float, float]]:
    """Distinct shades of one hue, varying saturation and value in HSV space.

    Parameters
    ----------
    base_color : color
        Any matplotlib color; only its hue is kept.
    n : int
        Number of shades.
    sat_range : tuple of (float, float)
        Saturation of the first and the last shade.
    val_range : tuple of (float, float)
        HSV value (brightness) of the first and the last shade.

    Returns
    -------
    list of tuple of (float, float, float)
        RGB colors, from the first to the last shade.
    """
    r, g, b = mcolors.to_rgb(base_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    sats = np.linspace(*sat_range, n)
    vals = np.linspace(*val_range, n)
    return [colorsys.hsv_to_rgb(h, s_, v_) for s_, v_ in zip(sats, vals)]


# Effort shades: light → dark from the reasoning base color
_effort_shades = shades_by_hsv(CATEGORY_COLORS["reasoning"], n=4)
EFFORT_PALETTE: dict[str, ColorType] = {
    "none": CATEGORY_COLORS["non-reasoning"],
    "suppressed": CATEGORY_COLORS["suppressed"],
    "low": _effort_shades[0],
    "medium": _effort_shades[1],
    "high": _effort_shades[2],
    "all reasoning": _effort_shades[3],
    # models whose reasoning is a flag, not a graded level (Qwen 0/1)
    "reasoning": CATEGORY_COLORS["reasoning"],
}

# Same colors keyed by the merged display labels (see utils.EFFORT_DISPLAY_LABELS), in legend order
EFFORT_LABEL_PALETTE: dict[str, ColorType] = {
    "off / none": EFFORT_PALETTE["none"],
    "low": EFFORT_PALETTE["low"],
    "medium": EFFORT_PALETTE["medium"],
    "on / high": EFFORT_PALETTE["high"],
    "suppressed": EFFORT_PALETTE["suppressed"],
}


# Reference lines are distinguished from data by linestyle
# (grey if no specific group, otherwise color of group)
REFERENCE_COLOR = "#8C8C8C"
# Accuracy drawn alongside an agreement curve: a different quantity, so it gets a color outside the
# reasoning/non-reasoning pair rather than a shade of one of them.
ACCURACY_COLOR = "#55A868"
REFERENCE_LINESTYLE = "--"


def add_reference_line(
    ax: Axes,
    value: float,
    *,
    label: str | None = None,
    group: str | None = None,
    axis: str = "y",
    **kwargs: Any,
) -> Line2D:
    """Draw a baseline/chance-level line in the convention above.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw in.
    value : float
        Position of the line.
    label : str, optional
        Legend label; escaped for LaTeX.
    group : str, optional
        Category the baseline belongs to (a key of ``CATEGORY_COLORS``, e.g. "reasoning"). Its color
        is used, so the baseline matches the series it refers to. Neutral grey when omitted.
    axis : {"y", "x"}
        Whether ``value`` is a y (horizontal line) or x (vertical line) position.
    **kwargs
        Passed to ``axhline``/``axvline`` (e.g. ``alpha``), overriding the defaults.

    Returns
    -------
    matplotlib.lines.Line2D
        The line drawn.
    """
    kwargs.setdefault("color", CATEGORY_COLORS.get(group, REFERENCE_COLOR) if group else REFERENCE_COLOR)
    kwargs.setdefault("linestyle", REFERENCE_LINESTYLE)
    kwargs.setdefault("linewidth", LINE_WIDTH)
    kwargs.setdefault("zorder", 0)
    if label is not None:
        kwargs["label"] = tex_escape(label)
    line_fn = ax.axhline if axis == "y" else ax.axvline
    return line_fn(value, **kwargs)


# Fixed color per developer/family
FAMILY_COLORS = {
    "OpenAI": "#1B9E77",  # teal
    "Qwen": "#6A3D9A",  # violet
    "Claude": "#C57316",  # orange
    "DeepSeek": "#305CB5",  # blue
    "Kimi": "#E7298A",  # magenta
    "Unknown": "#949494",
}


# Figures name the DEVELOPER, not the model line: "Qwen" is Alibaba's series of models, whereas
# "OpenAI" already is a developer, so the two read as different kinds of thing side by side. The
# data keeps the family key ("Qwen"); only what a reader sees changes.
FAMILY_DISPLAY = {"Qwen": "Alibaba"}


def family_display(family: str) -> str:
    """Developer name shown for a model family.

    Parameters
    ----------
    family : str
        Family key, e.g. ``"Qwen"``.

    Returns
    -------
    str
        The display name from ``FAMILY_DISPLAY``, or ``family`` itself when it has none.
    """
    return FAMILY_DISPLAY.get(family, family)


def _family_of(name: str) -> str:
    """Family of a model, tolerating prettified names.

    ``get_model_family`` matches ids like "o4-mini"; figures often carry prettified names
    ("o4 mini"), which would otherwise fall back to "Unknown" and get the grey color.
    """
    family = get_model_family(name)
    if family == "Unknown":
        family = get_model_family(str(name).replace(" ", "-"))
    return family


def family_color(model_or_family: str) -> str:
    """Color for a model family, accepting either a family name or any model id/key.

    Parameters
    ----------
    model_or_family : str
        A family name ("Qwen"), or a model id/key ("Qwen/Qwen3-4B", "Qwen--Qwen3-4B"), which is
        mapped to its family via :func:`reasonance.utils.get_model_family`.

    Returns
    -------
    str
        Hex color from ``FAMILY_COLORS``; the "Unknown" grey when no family matches.
    """
    if model_or_family in FAMILY_COLORS:
        return FAMILY_COLORS[model_or_family]
    return FAMILY_COLORS.get(_family_of(model_or_family), FAMILY_COLORS["Unknown"])


def light_grid(ax: Axes, axis: Literal["both", "x", "y"] = "y") -> None:
    """A grid faint enough to read values against without competing with the data.

    One definition, so every figure that needs one looks the same; drawn behind the data.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw the grid in.
    axis : {"y", "x", "both"}
        Which grid lines to draw.
    """
    ax.set_axisbelow(True)
    ax.grid(visible=True, axis=axis, color=POINT_COLOR, lw=0.4, alpha=0.15)


def light_box_kwargs(alpha: float = FILL_ALPHA, width: float = 0.5, show_fliers: bool = False) -> dict[str, Any]:
    """Keyword arguments for a light, unobtrusive boxplot.

    Boxes are a faint tint of the series color with thin edges and a slightly stronger median, so
    the box reads as context behind the data points rather than as the main mark. Spread into
    ``sns.boxplot(...)``.

    Parameters
    ----------
    alpha : float
        Opacity of box fill, whiskers and caps.
    width : float
        Box width in categorical units.
    show_fliers : bool
        Draw outlier markers; off by default since the raw points are usually plotted on top.

    Returns
    -------
    dict of str to Any
        Keyword arguments for ``sns.boxplot``.
    """
    line_width = 0.8 * LINE_WIDTH
    return dict(
        width=width,
        linewidth=line_width,
        boxprops=dict(alpha=alpha, linewidth=line_width),
        whiskerprops=dict(alpha=alpha, linewidth=line_width),
        capprops=dict(alpha=alpha, linewidth=line_width),
        medianprops=dict(color="0.3", linewidth=LINE_WIDTH, alpha=min(alpha * 3, 1.0)),
        flierprops=dict(marker="o" if show_fliers else "none", markersize=MARKER_SIZE, alpha=alpha),
    )


def _data_points_in_display(axis: Axes) -> np.ndarray:
    """Display-coordinate points of the data drawn in an axes, for overlap tests.

    Returns an ``(n, 2)`` array: lines sampled along their length, scatter offsets, and the corners
    of every patch.
    """
    points = []
    for line in axis.lines:
        data = np.asarray(line.get_xydata(), dtype=float)
        if len(data) >= 2:
            # sample ALONG the line, not just its vertices: a two-point line has nothing between
            # its ends, and a legend sitting over the middle of it would otherwise look "clear"
            steps = np.linspace(0, len(data) - 1, max(len(data), 200))
            index = np.clip(steps.astype(int), 0, len(data) - 2)
            fraction = (steps - index)[:, None]
            data = data[index] * (1 - fraction) + data[index + 1] * fraction
        if len(data):
            points.append(axis.transData.transform(data))
    for collection in axis.collections:
        offsets = collection.get_offsets()
        if offsets is not None and len(offsets):  # type: ignore[arg-type]  # stubs: ArrayLike, runtime: ndarray
            points.append(axis.transData.transform(np.asarray(offsets)))
    for patch in axis.patches:
        bbox = patch.get_extents()  # already display coords
        points.append(np.array([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1], [bbox.x0, bbox.y1], [bbox.x1, bbox.y0]]))
    return np.vstack(points) if points else np.empty((0, 2))


def _legend_clears_data(axis: Axes, legend: Legend, pad: float = 4.0) -> bool:
    """True when nothing drawn in ``axis`` lies under ``legend``.

    ``pad`` is a margin around the legend box, in display units (pixels).
    """
    figure = axis.get_figure()
    assert figure is not None
    figure.canvas.draw()  # extents are only known once drawn
    box = legend.get_window_extent(figure.canvas.get_renderer()).expanded(1.0, 1.0)  # type: ignore[attr-defined]
    points = _data_points_in_display(axis)
    if not len(points):
        return True
    inside = (
        (points[:, 0] > box.x0 - pad)
        & (points[:, 0] < box.x1 + pad)
        & (points[:, 1] > box.y0 - pad)
        & (points[:, 1] < box.y1 + pad)
    )
    return not inside.any()


def _try_legend_inside(
    axes: Sequence[Axes],
    handles: Sequence[Artist],
    labels: Sequence[str],
    title: str | None = None,
    loc: str = "best",
    **kwargs: Any,
) -> Legend | None:
    """Put one legend inside a panel if it fits clear of the data.

    Tried in order: the caller's ``loc`` in the first panel (default "best", which matplotlib
    already places in the emptiest corner), then the top left of the first panel, then the bottom
    right of the last one -- the corners a reader looks at first and last. Returns the legend, or
    None when every attempt would cover data, in which case the caller puts it below the figure.
    """
    attempts = [(axes[0], loc), (axes[0], "upper left"), (axes[-1], "lower right")]
    for axis, location in attempts:
        legend = axis.legend(handles, labels, title=title, loc=location, **kwargs)  # type: ignore[call-overload]
        if _legend_clears_data(axis, legend):
            return legend
        legend.remove()
    return None


def place_legend(
    ax: Axes | list[Axes] | tuple[Axes, ...] | np.ndarray,
    *,
    title: str | None = None,
    handles: Sequence[Artist] | None = None,
    labels: Sequence[str] | None = None,
    max_inside: int = 4,
    ncol: int | None = None,
    loc: str = "best",
    per_axes: bool | str = "auto",
    **kwargs: Any,
) -> Legend | list[Legend] | None:
    """Add a legend, keeping it close to what it describes and out of the data.

    One axes: the legend goes inside when it can be placed without covering drawn data, and below
    the figure when it cannot.

    Several axes: if the panels label *different* series (e.g. one model family per panel), each
    panel keeps its own legend -- a merged legend would force the reader to map colors back to
    panels. Those per-panel legends sit inside the axes while they are short, and drop below their
    own panel when they are long. Panels that share series get one merged legend below the figure,
    with each label listed once.

    Parameters
    ----------
    ax : matplotlib Axes, or a list/tuple/array of Axes
        Axes to take handles from. Pass the whole ``axes`` array once *after* the panel loop, not
        inside it.
    title : str, optional
        Legend title.
    handles, labels : list, optional
        Explicit handles/labels; taken from ``ax`` when omitted. Forces a single legend.
    max_inside : int
        Only for the per-panel case below: a panel keeps its own legend inside while it has at most
        this many entries, otherwise the per-panel legends move under the panels. A single shared
        legend ignores it -- that placement is measured against the drawn data.
    ncol : int, optional
        Columns for a legend placed below. Defaults to at most 4, filling one row.
    loc : str
        Location used for a legend inside the axes.
    per_axes : {"auto", True, False}
        "auto" gives each panel its own legend when the panels' label sets are disjoint.
    **kwargs
        Passed to the underlying ``legend()`` call.

    Returns
    -------
    Legend, list of Legend, or None
        The single legend, the per-panel legends, or None when there is nothing to label.
    """
    # numpy's stubs do not accept a list/tuple of Axes (object elements) in np.ravel
    axes: list[Axes] = list(np.ravel(ax)) if isinstance(ax, (list, tuple, np.ndarray)) else [ax]  # type: ignore[arg-type]
    axes = [axis for axis in axes if axis.get_visible()]
    if not axes:
        return None
    fig = axes[0].get_figure(root=True)
    explicit = handles is not None and labels is not None

    per_axes_items: list[tuple[Axes, list[Artist], list[str]]] = []
    if not explicit:
        for axis in axes:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            if axis_handles:
                per_axes_items.append((axis, axis_handles, axis_labels))
        merged: dict[str, Artist] = {}
        for _axis, axis_handles, axis_labels in per_axes_items:
            for handle, label in zip(axis_handles, axis_labels):
                merged.setdefault(label, handle)
        labels, handles = list(merged), list(merged.values())
    if not handles:
        return None

    if per_axes == "auto":
        # disjoint label sets -> the panels describe different series
        label_sets = [set(axis_labels) for _axis, _h, axis_labels in per_axes_items]
        per_axes = (
            not explicit and len(label_sets) > 1 and sum(len(ls) for ls in label_sets) == len(set().union(*label_sets))
        )

    for axis in axes:
        if axis.get_legend() is not None:
            axis.get_legend().remove()  # type: ignore[union-attr]  # not None: checked on the line above

    if per_axes:
        if max(len(axis_labels) for _axis, _h, axis_labels in per_axes_items) <= max_inside:
            return [
                axis.legend(axis_handles, axis_labels, title=title, loc=loc, **kwargs)  # type: ignore[call-overload]
                for axis, axis_handles, axis_labels in per_axes_items
            ]
        assert fig is not None
        return _legends_below_panels(fig, per_axes_items, title=title, ncol=ncol, **kwargs)

    # A legend that fits inside a panel without covering data belongs there: it costs no figure
    # height and stays next to what it describes. Whether it fits is measured, not configured --
    # the drawn geometry decides, so no call site has to guess.
    assert labels is not None  # set above whenever handles are
    inside = _try_legend_inside(axes, handles, labels, title=title, loc=loc, **kwargs)
    if inside is not None:
        return inside

    # one shared legend under the whole figure; constrained_layout reserves the space
    kwargs.setdefault("frameon", False)
    assert fig is not None
    return fig.legend(
        handles,
        labels,
        title=title,
        loc="outside lower center",
        ncol=ncol or min(len(handles), 4),
        **kwargs,
    )


def _legends_below_panels(
    fig: Figure,
    per_axes_items: Sequence[tuple[Axes, list[Artist], list[str]]],
    *,
    title: str | None = None,
    ncol: int | None = None,
    max_passes: int = 3,
    **kwargs: Any,
) -> list[Legend]:
    """One legend under each panel, sized and positioned by measuring the drawn legends.

    constrained_layout does not reserve room for legends anchored outside the axes, and a guessed
    margin either clips them or wastes space, so the margin and the column count are measured:
    columns shrink until a legend fits its panel's width, then the bottom margin grows until no
    legend hangs below the canvas.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure the legends are added to.
    per_axes_items : sequence of (Axes, handles, labels)
        One entry per panel that gets a legend.
    title : str, optional
        Title of every legend.
    ncol : int, optional
        Starting column count; defaults to at most 3. Shrunk until each legend fits its panel.
    max_passes : int
        Upper bound on the column-shrinking and the margin-growing passes, each.
    **kwargs
        Passed to ``fig.legend()``.

    Returns
    -------
    list of Legend
        The legends, in the order of ``per_axes_items``.
    """
    kwargs.setdefault("frameon", False)
    margin = 0.0
    legends: list = []

    anchors: list[float] = []  # panel centres in figure coords; never read back from the legend,
    # because Legend.get_bbox_to_anchor() returns display coordinates

    def _draw_legends(columns: Mapping[int, int], anchor_y: float = 0.0) -> None:
        """(Re)draw one legend per panel, ``columns`` keyed by ``id(axis)``, anchored at ``anchor_y``."""
        for legend in legends:
            legend.remove()
        legends.clear()
        anchors.clear()
        for axis, handles, labels in per_axes_items:
            position = axis.get_position()
            centre = position.x0 + position.width / 2
            legend = fig.legend(
                handles,
                labels,
                title=title,
                loc="upper center",
                bbox_to_anchor=(centre, anchor_y),
                bbox_transform=fig.transFigure,
                ncol=columns.get(id(axis), 1),
                **kwargs,
            )
            # NOTE: in_layout stays True. It must, because Figure.get_tightbbox() ignores artists
            # with in_layout=False, and the notebook inline backend renders with bbox_inches="tight"
            # -- the legend would be cropped out of the displayed figure. The space it needs is
            # reserved manually below by shrinking the layout engine's rect.
            legends.append(legend)
            anchors.append(centre)
        fig.canvas.draw()

    columns = {id(axis): ncol or min(len(labels), 3) for axis, _handles, labels in per_axes_items}
    _draw_legends(columns)

    # shrink columns until each legend fits the width of its own panel
    for _pass in range(max_passes):
        shrunk = False
        for (axis, _handles, _labels), legend in zip(per_axes_items, legends):
            width = legend.get_window_extent().transformed(fig.transFigure.inverted()).width
            if width > axis.get_position().width * 1.05 and columns[id(axis)] > 1:
                columns[id(axis)] -= 1
                shrunk = True
        if not shrunk:
            break
        _draw_legends(columns)

    # grow the bottom margin until nothing hangs below the canvas
    for _pass in range(max_passes):
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # missing from FigureCanvasBase stubs
        top = min(
            axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0  # type: ignore[union-attr]
            for axis, _h, _l in per_axes_items
        )
        for legend, centre in zip(legends, anchors):
            legend.set_bbox_to_anchor((centre, top - 0.02), transform=fig.transFigure)
        fig.canvas.draw()
        lowest = min(lg.get_window_extent().transformed(fig.transFigure.inverted()).y0 for lg in legends)
        if lowest >= 0.01:
            break
        margin = min(margin + 0.01 - lowest + 0.02, 0.45)
        engine = fig.get_layout_engine()
        if engine is not None and hasattr(engine, "set"):
            engine.set(rect=(0, margin, 1, 1 - margin))  # type: ignore[call-arg]  # base LayoutEngine.set has no rect
        fig.canvas.draw()

    return legends


# Consensus direction: brown = models agree on the wrong answer (or on class 0), teal = on the
# correct answer (or class 1); pale in between = the group is split. Shared by every vote
# distribution figure.
VOTE_WRONG_COLOR = "#8b5e3c"
VOTE_CORRECT_COLOR = "#2a9d8f"  # teal, paired with the brown above
VOTE_SPLIT_COLOR = "#f5f5f5"
# Fine dark separators between neighbouring vote levels, so adjacent shades stay distinguishable.
VOTE_EDGE_COLOR = "0.25"
VOTE_EDGE_WIDTH = 0.3
# Bin edges in a binned stack: the separators' dark tone, so the figure has one line colour, but
# half as wide -- they mark where each bin starts, the level separators carry the quantity.
VOTE_BIN_EDGE_COLOR = VOTE_EDGE_COLOR
VOTE_BIN_EDGE_WIDTH = 0.15


# Order tasks appear in, wherever several are shown; unknown tasks are appended.
TASK_ORDER = ["ACSIncome", "SIPP", "ACSEmployment"]


def sort_tasks(tasks: Collection[str]) -> list[str]:
    """Tasks in the canonical order, with anything unknown after them.

    Parameters
    ----------
    tasks : collection of str
        Task names; iterated more than once, so not a one-shot iterator.

    Returns
    -------
    list of str
        Known tasks in ``TASK_ORDER``, then the rest sorted alphabetically.
    """
    known = [task for task in TASK_ORDER if task in set(tasks)]
    return known + sorted(set(tasks) - set(known))


def vote_colormap(n_levels: int) -> mcolors.LinearSegmentedColormap:
    """Diverging colormap for consensus levels, wrong/negative -> split -> correct/positive.

    Parameters
    ----------
    n_levels : int
        Number of discrete colors (at least 2 are used).

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The brown -> pale -> teal ramp.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "vote", [VOTE_WRONG_COLOR, VOTE_SPLIT_COLOR, VOTE_CORRECT_COLOR], N=max(n_levels, 2)
    )


def plot_vote_stairs(
    ax: Axes,
    frame: pd.DataFrame,
    value: str = "observed",
    mode: str = "correct",
    n_levels: int | None = None,
    cmap: mcolors.Colormap | None = None,
) -> mcolors.Colormap:
    """Stacked stairs: one column per length bin, the vote distribution stacked inside it.

    A bar chart of the vote distribution cannot be repeated per bin without one panel per bin; an area
    plot would interpolate between bin centres, implying values the binning never measured. Stairs
    keep a bin's value constant across the bin it was computed on, and stacking them fills the
    column, so the picture is "how the individuals of this length spread over the vote counts".

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to draw in; its x limits are set to the outer bin edges.
    frame : pd.DataFrame
        Rows of one panel: ``bin_left``, ``bin_right``, ``k`` and the column named by ``value``.
    value : str
        Which column to stack, e.g. ``"observed"`` or ``"null"``.
    mode : {"correct", "class"}
        Only used for the colorbar/legend wording of the caller; the colors are the same ramp.
    n_levels : int, optional
        Number of vote levels, so several panels share one color scale even if a panel is missing
        an empty level.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the levels; defaults to :func:`vote_colormap` with ``n_levels`` colors.

    Returns
    -------
    matplotlib.colors.Colormap
        The colormap used, for a colorbar.
    """
    table = frame.pivot_table(index="bin_left", columns="k", values=value, aggfunc="mean").sort_index()
    rights = frame.groupby("bin_left")["bin_right"].first().sort_index()
    # Each bin spans ITS OWN [left, right]. A bin dropped upstream (too few individuals) must stay a
    # gap: taking the next bin's left edge as this bin's right would silently widen its neighbour.
    lefts = table.index.to_numpy(dtype=float)
    ends = rights.to_numpy(dtype=float)
    edges: list[float] = [lefts[0]]
    keep: list[int | None] = []
    for i, (left, right) in enumerate(zip(lefts, ends)):
        if left > edges[-1] + 1e-12:  # a gap before this bin: an empty stair
            edges.append(left)
            keep.append(None)
        edges.append(right)
        keep.append(i)
    edges = np.array(edges)  # type: ignore[assignment]  # the name is reused for the array of edges
    levels = n_levels if n_levels is not None else table.shape[1]
    cmap = cmap or vote_colormap(levels)
    lower = np.zeros(len(keep))
    for position, k in enumerate(table.columns):
        column = table[k].fillna(0).to_numpy()
        upper = lower + np.array([column[i] if i is not None else np.nan for i in keep])
        colour = cmap(position / max(levels - 1, 1))
        ax.stairs(upper, edges, baseline=lower, fill=True, color=colour, lw=0, zorder=1)
        if position < len(table.columns) - 1:  # the top level's upper edge is the axis itself
            # horizontal segments only: the step's vertical risers would double the bin edges
            drawn = ~np.isnan(upper)
            ax.hlines(
                upper[drawn], edges[:-1][drawn], edges[1:][drawn], color=VOTE_EDGE_COLOR, lw=VOTE_EDGE_WIDTH, zorder=3
            )
        lower = upper
    # every bin edge, full height: a column of equal colour then still reads as several bins
    ax.vlines(edges[1:-1], 0, 1, color=VOTE_BIN_EDGE_COLOR, lw=VOTE_BIN_EDGE_WIDTH, zorder=2)
    ax.set_xlim(edges[0], edges[-1])
    return cmap


def plot_model_dumbbell(
    data: pd.DataFrame,
    value_col: str,
    model_col: str = "model",
    category_col: str = "category",
    ax: Axes | None = None,
    *,
    err_col: str | None = None,
    order: Sequence[str] | None = None,
    label_col: str | None = None,
    palette: Mapping[str, ColorType] | None = None,
    xlabel: str | None = None,
    title: str = "",
    connector: bool = True,
    connector_color: ColorType | None = None,
    dodge: float = 0.18,
    row_separator: str | None = "line",
    separator_group_col: str | None = None,
    orient: str = "h",
    slot_weights: Mapping[str, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes, list[mpatches.Patch]]:
    """One row per model, a dot per run, a line spanning that model's runs.

    The horizontal layout is what makes a many-model figure narrow: model names sit on the y axis
    unrotated instead of eating half the height as slanted x labels, and the runs of one model
    (effort levels, reasoning on/off) collapse onto a single row instead of spreading along x.

    Parameters
    ----------
    data : pd.DataFrame
        One row per run.
    value_col : str
        Column plotted on the x axis (e.g. ``"accuracy_mean"``).
    model_col : str
        Column identifying the model; one y row per distinct value.
    category_col : str
        Column coloring the dots, keyed into ``palette`` (defaults to ``CATEGORY_COLORS``).
    ax : matplotlib Axes, optional
        Axes to draw in; a new figure is created when omitted.
    err_col : str, optional
        Symmetric x error bars (e.g. the std across seeds).
    order : list, optional
        Model order, top to bottom. Defaults to order of appearance in ``data``.
    label_col : str, optional
        Column holding the y tick label for each model; defaults to ``model_col``.
    palette : dict of str to color, optional
        Color per category; defaults to ``CATEGORY_COLORS``. Its key order sets the dodge order.
    xlabel : str, optional
        Label of the value axis (the x axis when horizontal); defaults to ``value_col``.
    title : str
        Axes title; escaped for LaTeX.
    connector : bool
        Draw a line spanning a model's runs. Off leaves the dots alone, which reads more cleanly when
        the runs are close together.
    connector_color : color, optional
        Color of that line; neutral by default.
    dodge : float
        Vertical offset between categories within a row, so their error bars do not overlap. 0
        places every dot on the row's centre line.
    row_separator : {"line", "band", None}
        Separate the rows visually: a hairline between models, alternating background bands, or
        nothing. With dodged dots and error bars, rows otherwise run into each other.
    separator_group_col : str, optional
        Draw separators only where this column changes (e.g. "family"), instead of between every
        model -- fewer lines, and the boundary that carries meaning.
    slot_weights : dict, optional
        Relative width of each model's slot, e.g. its number of runs: a model with four effort
        levels then gets twice the room of one with two, and their dots are equally far apart
        instead of equally crowded. Missing models count as 1; the weights are rescaled so the
        axis keeps its total extent. Pass the SAME weights to every panel of a shared axis.
    orient : {"h", "v"}
        ``"h"`` puts the models on the y axis and the value on x (the default, and what keeps model
        names horizontal). ``"v"`` flips the two, so the value is read up the y axis -- worth it
        when the value axis is the interesting one and several panels should share it.
    figsize : tuple of (float, float), optional
        Size of a newly created figure; ignored when ``ax`` is given.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure holding ``ax``.
    ax : matplotlib Axes
        The axes drawn in.
    handles : list of matplotlib.patches.Patch
        One legend patch per category present, in ``palette`` order.
    """
    palette = CATEGORY_COLORS if palette is None else palette
    label_col = label_col or model_col
    # an explicit order is kept as given, including models absent from `data`: their row stays empty
    # so several panels sharing a y axis line up
    models = list(dict.fromkeys(data[model_col])) if order is None else list(order)
    if orient not in ("h", "v"):
        raise ValueError(f"orient must be 'h' or 'v', got {orient!r}")
    horizontal = orient == "h"

    # Slot per model, in axis units. Equal weights reproduce the plain 0, 1, 2, ... layout; unequal
    # ones keep the total extent but hand more of it to the models that have more to show.
    widths = {model: float((slot_weights or {}).get(model, 1) or 1) for model in models}
    scale = len(models) / sum(widths.values())
    widths = {model: width * scale for model, width in widths.items()}
    centres, cursor = {}, -0.5
    for model in models:
        centres[model] = cursor + widths[model] / 2
        cursor += widths[model]
    # horizontal: first model on top, counting down; vertical: first model on the left
    positions = {m: (len(models) - 1 - centres[m] if horizontal else centres[m]) for m in models}

    if ax is None:
        default_size = (
            get_figsize(rel_width=0.75, height_to_width_ratio=0.09 * len(models))
            if horizontal
            else get_figsize(rel_width=0.75)
        )
        fig, ax = plt.subplots(figsize=figsize or default_size)
    else:
        fig = ax.figure  # type: ignore[assignment]  # stubs: Figure | SubFigure; no subfigures are used

    # spread the categories slightly within a row, so overlapping error bars stay readable
    categories_present = [c for c in palette if c in set(data[category_col])]
    offsets = (
        {c: (i - (len(categories_present) - 1) / 2) * dodge for i, c in enumerate(categories_present)}
        if dodge
        else dict.fromkeys(categories_present, 0.0)
    )

    # dodged dots and their error bars blur the boundary between neighbouring models
    if separator_group_col is not None:
        group_of: dict[str, Any] = {}
        for model in models:
            rows = data[data[model_col] == model]
            # a model missing from THIS panel keeps the previous row's group, so a shared row order
            # does not open a band boundary where there is no group change
            group_of[model] = rows[separator_group_col].iloc[0] if len(rows) else group_of.get(models[0])
        # halfway between the two rows, taken from the positions themselves: they count DOWN (first
        # model on top), so subtracting half a row would put the edge on the wrong side of the row
        boundaries = [
            (positions[models[i - 1]] + positions[models[i]]) / 2
            for i in range(1, len(models))
            if group_of[models[i]] != group_of[models[i - 1]]
        ]
    else:
        boundaries = list(np.arange(len(models) - 1) + 0.5)

    # the separators run across the value axis, so they swap with the orientation
    draw_line, draw_span = (ax.axhline, ax.axhspan) if horizontal else (ax.axvline, ax.axvspan)
    if row_separator == "line":
        for boundary in boundaries:
            draw_line(boundary, color=POINT_COLOR, lw=0.4, alpha=0.25, zorder=0)
    elif row_separator == "band":
        # shade alternating blocks (whole groups when grouped, single rows otherwise)
        edges = [-0.5, *sorted(boundaries), len(models) - 0.5]
        for i in range(0, len(edges) - 1, 2):
            draw_span(edges[i], edges[i + 1], color=POINT_COLOR, alpha=0.06, lw=0, zorder=0)

    for model in models:
        rows = data[data[model_col] == model]
        y = positions[model]
        values = rows[value_col].to_numpy(dtype=float)
        span = [np.nanmin(values), np.nanmax(values)]
        if connector and len(values) > 1:  # the line only means something with several runs
            line_x, line_y = (span, [y, y]) if horizontal else ([y, y], span)
            ax.plot(line_x, line_y, color=connector_color or POINT_COLOR, lw=0.8 * LINE_WIDTH, alpha=0.6, zorder=1)
        for _, row in rows.iterrows():
            color = palette.get(row[category_col], POINT_COLOR)
            # the dodge is a fraction of THIS model's slot, so a wide slot spreads its dots
            offset = y + offsets.get(row[category_col], 0.0) * widths[model]
            dot_x, dot_y = (row[value_col], offset) if horizontal else (offset, row[value_col])
            if err_col and np.isfinite(row.get(err_col, np.nan)):
                err = {"xerr": row[err_col]} if horizontal else {"yerr": row[err_col]}
                ax.errorbar(
                    dot_x,
                    dot_y,
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.8 * LINE_WIDTH,
                    capsize=1.5,
                    alpha=0.7,
                    zorder=2,
                    **err,
                )
            ax.scatter(dot_x, dot_y, s=SCATTER_SIZE * 2, color=color, edgecolors="none", zorder=3)

    labels = [
        tex_escape(rows[label_col].iloc[0]) if len(rows := data[data[model_col] == m]) else tex_escape(m)
        for m in models
    ]
    model_axis, value_axis = (ax.yaxis, ax.xaxis) if horizontal else (ax.xaxis, ax.yaxis)
    model_axis.set_ticks([positions[m] for m in models])
    # vertical: the model names do not fit side by side, so they are turned
    model_axis.set_ticklabels(labels, **({} if horizontal else {"rotation": 90, "ha": "center"}))  # type: ignore[arg-type]
    (ax.set_ylim if horizontal else ax.set_xlim)(-0.7, len(models) - 0.3)
    value_axis.set_label_text(value_col if xlabel is None else xlabel)
    ax.set_title(tex_escape(title) if title else "")

    categories = [c for c in palette if c in set(data[category_col])]
    handles = [mpatches.Patch(color=palette[c], label=tex_escape(c)) for c in categories]
    sns.despine(ax=ax, left=horizontal, bottom=not horizontal)
    ax.tick_params(axis="y" if horizontal else "x", length=0)
    return fig, ax, handles


# One switch for every axis that counts models: False labels them as the raw count, which is what
# the figures use -- with nine models a share reads 0, 0.33, 0.67, 1 and is too coarse to see where
# the mass sits. True labels the same ticks as a SHARE of the models, which is what tasks with a
# different number of models would need. Read at call time, so a notebook can flip it with
# ``reasonance.plotting.SHOW_FRACTION = True``.
SHOW_FRACTION = False


def count_prefix() -> str:
    """ "number of" or "fraction of", following :data:`SHOW_FRACTION`.

    Returns
    -------
    str
        The prefix for a count axis label.
    """
    return "fraction of" if SHOW_FRACTION else "number of"


def count_tick_step(n_models: int, max_labels: int = 6) -> int:
    """Label every ``step``-th count so the labels stay readable AND land on round fractions.

    The step divides ``n_models``, so the labelled counts are exactly the ones whose share is a
    round number (for nine models: 0, 3, 6, 9 -> 0, 1/3, 2/3, 1), and the endpoint -- unanimity,
    the count that carries the story -- is always labelled.

    Parameters
    ----------
    n_models : int
        Number of models, i.e. the largest count on the axis.
    max_labels : int
        Most labels allowed on the axis, endpoints included.

    Returns
    -------
    int
        The smallest divisor of ``n_models`` that keeps the labels within ``max_labels``.
    """
    for step in range(1, n_models + 1):
        if n_models % step == 0 and n_models / step + 1 <= max_labels:
            return step
    return max(1, n_models)


def fraction_tick_labels(
    n_models: int,
    counts: Iterable[int] | None = None,
    signed: bool = False,
    max_labels: int = 6,
    limit: int | None = None,
) -> list[str]:
    """Labels turning a count of models into the FRACTION of them, blank between the labelled ones.

    Tasks can differ in how many models they have, so a count is not comparable across panels while
    a share is. The ticks stay at the counts -- one per bar or cell -- only their labels change.

    With :data:`SHOW_FRACTION` off (the default) the labels are the counts themselves.

    Parameters
    ----------
    n_models : int
        Number of models; a count is divided by it to give the share.
    counts : iterable of int, optional
        Tick positions to label; defaults to ``0, ..., n_models``.
    signed : bool
        Label a signed axis (changes in a count): positive labels get a "+".
    max_labels : int
        Most labelled ticks when showing shares, see :func:`count_tick_step`.
    limit : int, optional
        Blank the share labels of counts beyond ``+-limit`` (shares only).

    Returns
    -------
    list of str
        One label per count, "" where the tick stays unlabelled.
    """
    counts = range(n_models + 1) if counts is None else list(counts)
    # Shares are three characters wider than counts, so they are thinned to the round ones. Counts
    # fit at every tick, except on a signed axis where the minus sign runs into its neighbour.
    step = count_tick_step(n_models, max_labels) if SHOW_FRACTION else (2 if signed else 1)
    labels = []
    for count in counts:
        # `limit` blanks the outer labels of a signed axis of SHARES, where the tails hold almost
        # no data and two labels three steps apart still run into each other
        if count % step or (SHOW_FRACTION and limit is not None and abs(count) > limit):
            labels.append("")
            continue
        if not SHOW_FRACTION:
            labels.append(f"+{count}" if signed and count > 0 else str(count))
            continue
        share = count / n_models
        text = f"{share:.2f}".rstrip("0").rstrip(".") if share not in (0, 1, -1) else f"{share:.0f}"
        labels.append(f"+{text}" if signed and share > 0 else text)
    return labels


def plot_vote_movement(
    transitions: pd.DataFrame,
    tasks: Collection[str] | None = None,
    figures_dir: str | Path | None = None,
    file_name: str = "effort-vote-movement-nr-to-r",
    save: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Who moves on the unanimity ladder when reasoning is switched on.

    The function only plots: ``reasonance.analysis.vote_transitions_for_tasks`` computes the frame.

    Top row: for every individual, how many of the matched models predict the true label with
    reasoning off (x) and on (y). Mass above the diagonal is individuals the ecosystem gets right
    more unanimously, mass below it those it gets wrong more unanimously, the diagonal itself
    everyone who did not move. Bottom row: the same as one number per individual, ``k_on - k_off``.

    Parameters
    ----------
    transitions : pd.DataFrame
        One row per (task, k_off, k_on) with a ``value`` share of individuals.
    tasks : list of str, optional
        Tasks to draw, in order; defaults to those present.
    figures_dir : str or Path, optional
        Directory to save into; defaults to ``results/figures`` (see :func:`savefig`).
    file_name : str
        Base name, given the usual task/QA prefix by :func:`savefig`.
    save : bool
        Write the figure to disk.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : np.ndarray
        ``2 x len(tasks)`` array of Axes: heatmaps on top, step histograms below.
    """

    tasks = sort_tasks(transitions["task"].unique() if tasks is None else tasks)
    # square heatmaps in the top row, so the figure needs height to keep them legible
    fig, axes = plt.subplots(
        2, len(tasks), figsize=get_figsize(nrows=2, ncols=len(tasks), height_to_width_ratio=1.15), squeeze=False
    )

    # Linear diverging scale in the project's vote palette: brown below the diagonal (moved toward
    # wrong), teal above it (moved toward correct). The diagonal is excluded, so the largest cell is
    # a move, not the unmoved mass, and a linear scale is readable -- very small cells stay close to
    # white, which is fine: they are the moves that hardly anyone made.
    off_diagonal = transitions[transitions["k_on"] != transitions["k_off"]]
    largest = float(off_diagonal["value"].max()) if len(off_diagonal) else 1.0
    cmap = vote_colormap(256)
    norm = mcolors.Normalize(vmin=-largest, vmax=largest)

    # One x range for the lower row: the tasks have different tails, and per-panel ranges would put
    # the same step at a different place in each panel.
    step_limit = max(int((transitions["k_on"] - transitions["k_off"]).abs().max()), 1)

    for col, task in enumerate(tasks):
        panel = transitions[transitions["task"] == task]
        n_runs = int(panel["n_runs"].iloc[0])
        grid = (
            panel.pivot_table(index="k_on", columns="k_off", values="value", aggfunc="sum")
            .reindex(index=range(n_runs + 1), columns=range(n_runs + 1))
            .fillna(0.0)
            .to_numpy()
        )
        direction = np.sign(np.subtract.outer(np.arange(n_runs + 1), np.arange(n_runs + 1)))
        # blank on the diagonal: those individuals did not move, and their mass would swamp the scale
        signed = np.where((grid > 0) & (direction != 0), grid * direction, np.nan)

        ax = axes[0][col]
        edges = np.arange(n_runs + 2) - 0.5
        mesh = ax.pcolormesh(edges, edges, signed, cmap=cmap, norm=norm)
        ax.plot([-0.5, n_runs + 0.5], [-0.5, n_runs + 0.5], color="0.4", lw=0.8 * LINE_WIDTH, ls="--")
        ax.set(xlim=(-0.5, n_runs + 0.5), ylim=(-0.5, n_runs + 0.5), title=task)
        ax.set_aspect("equal")
        # a tick per cell -- only whole models exist -- labelled as the share of them, so tasks with
        # a different number of models still read on the same scale
        ax.set_xticks(range(n_runs + 1))
        ax.set_yticks(range(n_runs + 1))
        ax.set_xticklabels(fraction_tick_labels(n_runs))
        ax.set_yticklabels(fraction_tick_labels(n_runs))
        # one label under the middle panel: three copies of it touch at this panel width
        # worded as "correct predictions" (one per model), like the other vote figures
        ax.set_xlabel("correct predictions, reasoning off" if col == len(tasks) // 2 else "")
        ax.set_ylabel("correct predictions, reasoning on" if col == 0 else "")
        ax.grid(visible=False)
        if col == len(tasks) - 1:
            bar = fig.colorbar(mesh, ax=ax, pad=0.12)
            bar.set_label("share of individuals")
            # Ticks and labels are set together and symmetrically about zero. Labelling
            # bar.get_ticks() instead slides the labels against the colours -- that list can hold
            # ticks outside the visible range -- which can put "0%" in the brown half.
            ticks = [
                tick
                for tick in MaxNLocator(nbins=5, symmetric=True).tick_values(-largest, largest)
                if -largest <= tick <= largest
            ]
            bar.set_ticks(ticks)
            # the % is escaped because a bare one starts a LaTeX comment
            bar.set_ticklabels([r"0\%" if tick == 0 else rf"{tick * 100:+g}\%" for tick in ticks])
            # unlabelled ticks halfway between the labelled ones, to read off intermediate cells
            bar.ax.yaxis.set_minor_locator(FixedLocator([(lo + hi) / 2 for lo, hi in zip(ticks[:-1], ticks[1:])]))

        ax = axes[1][col]
        moves = panel.assign(step=panel["k_on"] - panel["k_off"])
        unchanged = float(moves.loc[moves["step"] == 0, "value"].sum())
        movers = moves[moves["step"] != 0]
        # Stacked by DESTINATION: a bar says not only how many moved one step, but where they
        # ended up -- the palette is the consensus level after reasoning, brown (all wrong) to
        # teal (all correct), so a bar dominated by teal is people who moved into unanimity.
        levels = vote_colormap(n_runs + 1)
        bottoms: dict[Any, float] = {}
        for k_on, group in movers.groupby("k_on"):
            shares = group.groupby("step")["value"].sum()
            offsets = np.array([bottoms.get(step, 0.0) for step in shares.index])
            # thin edges between the segments: neighbouring destination levels differ by one step
            # of the ramp, and without a separator a stack reads as one block
            ax.bar(
                shares.index,
                shares.to_numpy(),
                bottom=offsets,
                color=levels(k_on / n_runs),
                width=0.85,
                edgecolor=VOTE_EDGE_COLOR,
                linewidth=VOTE_EDGE_WIDTH,
            )
            for step, share in shares.items():
                bottoms[step] = bottoms.get(step, 0.0) + share

        up = float(movers.loc[movers["step"] > 0, "value"].sum())
        down = float(movers.loc[movers["step"] < 0, "value"].sum())
        # percent signs must be escaped: under usetex a bare % comments out the rest of the title
        ax.set_title(rf"unchanged {unchanged * 100:.0f}\%" + "\n" + rf"up {up * 100:.0f}\%, down {down * 100:.0f}\%")
        # a tick per bar over the same range in every panel, labelled as a share of the models:
        # a move of one model means something different when there are more of them
        ax.set_xlim(-step_limit - 0.6, step_limit + 0.6)
        steps_shown = [step for step in range(-step_limit, step_limit + 1) if step != 0]
        ax.set_xticks(steps_shown)
        ax.set_xticklabels(fraction_tick_labels(n_runs, steps_shown, signed=True, limit=n_runs // 2))
        # "recourse" in the sense of Gorecki and Hardt, Monoculture or Multiplicity: how many of the
        # models give an individual the correct prediction. The bar is the change in that count.
        # Only under the middle panel: repeated, the label is too long and runs into its neighbours.
        ax.set_xlabel(f"change in recourse ({count_prefix()} correct predictions)" if col == len(tasks) // 2 else "")
        ax.set_ylabel("share of individuals" if col == 0 else "")
        # the bars are stacked, so a reader needs help carrying a height across the panel
        light_grid(ax)
        if col == len(tasks) - 1:
            # discrete scale: the destination is a count of models, so every level gets a tick
            bounds = np.arange(-0.5, n_runs + 1.5)
            levels_bar = fig.colorbar(
                plt.cm.ScalarMappable(norm=mcolors.BoundaryNorm(bounds, levels.N), cmap=levels),
                ax=ax,
                pad=0.12,
                ticks=range(n_runs + 1),
            )
            levels_bar.ax.set_yticklabels(fraction_tick_labels(n_runs))
            levels_bar.set_label("correct predictions after reasoning")
            # a count of models: nothing sits between two levels, and nothing outside 0..n
            levels_bar.ax.minorticks_off()

    sns.despine(fig=fig)
    if save:
        savefig(fig, file_name, figures_dir)
    return fig, axes
