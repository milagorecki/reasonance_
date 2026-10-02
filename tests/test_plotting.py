"""Tests for the tueplots-based style helpers in reasonance.plotting."""

import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib import dviread  # noqa: E402
from matplotlib import pyplot as plt
from reasonance.plotting import get_figsize, set_plot_style, tex_escape  # noqa: E402


def test_style_applied_on_import():
    # ICLR 2024 bundle: 5.5in text width, 9pt base font, LaTeX text
    assert plt.rcParams["figure.figsize"][0] == 5.5
    assert plt.rcParams["font.size"] == 9
    assert plt.rcParams["text.usetex"]


def test_constrained_layout_not_tight_layout():
    assert plt.rcParams["figure.constrained_layout.use"]
    assert not plt.rcParams["figure.autolayout"]  # autolayout == tight_layout on every draw
    # cropping on save would defeat the exact ICLR text width
    assert plt.rcParams["savefig.bbox"] != "tight"


def test_no_oversized_default_fontsizes():
    # legend titles and suptitles must not fall back to matplotlib's "large"/"medium"
    assert plt.rcParams["legend.title_fontsize"] == plt.rcParams["legend.fontsize"]
    assert plt.rcParams["figure.titlesize"] == plt.rcParams["axes.titlesize"]


def test_get_figsize_scales_with_grid():
    width, height = get_figsize()
    assert width == 5.5
    # two rows of subplots are twice as tall as one
    assert get_figsize(nrows=2)[1] == pytest.approx(2 * height)
    # half width is half as wide
    assert get_figsize(rel_width=0.5)[0] == 0.5 * width


def test_tex_escape_escapes_only_what_matplotlib_does_not():
    # "&" is a LaTeX alignment character and crashes the figure unless escaped
    assert tex_escape("agreement & accuracy") == r"agreement \& accuracy"
    assert tex_escape("a#b$c^d{e}") == r"a\#b\$c\textasciicircum{}d\{e\}"
    # "%" would start a LaTeX comment and drop the rest of the label; "~" would render as a space
    assert tex_escape("100% of ~x") == r"100\% of \textasciitilde{}x"
    # matplotlib escapes "_" itself; escaping again would render a literal backslash
    assert tex_escape("gpt_5") == "gpt_5"
    assert tex_escape(3) == "3"


def test_tex_escape_is_noop_without_usetex():
    try:
        set_plot_style(usetex=False)
        assert tex_escape("gpt_5 100%") == "gpt_5 100%"
    finally:
        set_plot_style()


def test_style_survives_a_late_seaborn_theme():
    # folktexts applies its own theme on import; re-applying the style must win
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    set_plot_style()
    assert not plt.rcParams["axes.grid"]
    assert plt.rcParams["font.size"] == 9


def test_tex_font_lookup_works_with_conda_ld_library_path(monkeypatch):
    # the conda lib dir makes the system luatex crash, which matplotlib reports as a missing font
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{sys.prefix}/lib:")
    set_plot_style()
    assert sys.prefix not in os.environ.get("LD_LIBRARY_PATH", "")
    assert dviread.find_tex_file("ptmr7t.tfm")


def test_family_color_accepts_model_ids_and_family_names():
    from reasonance.plotting import FAMILY_COLORS, family_color

    assert family_color("Qwen/Qwen3-4B") == FAMILY_COLORS["Qwen"]
    assert family_color("Qwen--Qwen3-4B") == FAMILY_COLORS["Qwen"]
    assert family_color("gpt-5.1") == FAMILY_COLORS["OpenAI"]
    assert family_color("OpenAI") == FAMILY_COLORS["OpenAI"]


def test_place_legend_goes_inside_only_when_it_clears_the_data():
    """Placement is decided by measuring the drawn data, not by the number of entries."""
    from reasonance.plotting import place_legend

    # data confined to the lower left: the upper right is free, so the legend belongs inside
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot([0, 0.2], [0, 0.1 * i], label=f"s{i}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend = place_legend(ax)
    assert legend in ax.get_children()
    assert not fig.legends

    # lines spanning the whole axes leave no clear corner, so it drops below the figure
    fig, ax = plt.subplots()
    for i in range(8):
        ax.plot([0, 1], [i, i], label=f"s{i}")
    legend = place_legend(ax)
    assert ax.get_legend() is None
    assert legend in fig.legends


def test_reference_line_uses_group_color_when_given():
    from reasonance.plotting import CATEGORY_COLORS, REFERENCE_COLOR, add_reference_line

    fig, ax = plt.subplots()
    grouped = add_reference_line(ax, 0.5, label="chance", group="reasoning")
    neutral = add_reference_line(ax, 0.4, label="XGBoost")
    assert grouped.get_color() == CATEGORY_COLORS["reasoning"]
    assert neutral.get_color() == REFERENCE_COLOR


def test_figure_prefix_is_applied_and_idempotent(tmp_path):
    from reasonance.plotting import figure_name, savefig, set_figure_prefix

    try:
        set_figure_prefix("ACSIncome", "numeric")
        assert figure_name("vote-stacks") == "ACSIncome-numeric-vote-stacks"
        # a part already at the front is not repeated, in any combination
        assert figure_name("numeric-vote-stacks") == "ACSIncome-numeric-vote-stacks"
        assert figure_name("ACSIncome-numeric-vote-stacks") == "ACSIncome-numeric-vote-stacks"
        # the part is only stripped from the front, not from inside a name
        assert figure_name("numeric-qa-thing") == "ACSIncome-numeric-qa-thing"

        fig, _ax = plt.subplots()
        savefig(fig, "demo", tmp_path)
        assert {f.name for f in tmp_path.iterdir()} == {"ACSIncome-numeric-demo.png", "ACSIncome-numeric-demo.pdf"}
    finally:
        set_figure_prefix()


def test_prefix_accepts_one_part_or_none(tmp_path):
    from reasonance.plotting import figure_name, savefig, set_figure_prefix

    try:
        set_figure_prefix("ACSIncome")  # notebooks without a QA mode
        assert figure_name("corr-heatmap") == "ACSIncome-corr-heatmap"
        set_figure_prefix("ACSIncome", None)  # a missing part is dropped, not rendered as "None"
        assert figure_name("corr-heatmap") == "ACSIncome-corr-heatmap"
    finally:
        set_figure_prefix()

    assert figure_name("plain") == "plain"
    fig, _ax = plt.subplots()
    savefig(fig, "plain", tmp_path)
    assert (tmp_path / "plain.png").exists()


def test_label_filter_tag_maps_filters_to_name_tokens():
    from reasonance.plotting import label_filter_tag

    assert label_filter_tag(None) is None  # no filter -> the prefix omits the part entirely
    assert label_filter_tag(1) == "pos_label_only"
    assert label_filter_tag(0) == "neg_label_only"
    assert label_filter_tag(True) == "pos_label_only"


def test_label_filter_appears_in_the_figure_name():
    from reasonance.plotting import figure_name, label_filter_tag, set_figure_prefix

    try:
        set_figure_prefix("ACSIncome", "numeric", label_filter_tag(1))
        assert figure_name("kappa") == "ACSIncome-numeric-pos_label_only-kappa"
        set_figure_prefix("ACSIncome", "numeric", label_filter_tag(None))
        assert figure_name("kappa") == "ACSIncome-numeric-kappa"
    finally:
        set_figure_prefix()


def test_empty_name_parts_do_not_leave_double_dashes():
    from reasonance.plotting import figure_name, set_figure_prefix

    try:
        set_figure_prefix("ACSIncome", "numeric")
        # an optional suffix that renders empty must not leave a dangling or doubled dash
        assert figure_name("tokens-per-individual-") == "ACSIncome-numeric-tokens-per-individual"
        assert figure_name("a--b") == "ACSIncome-numeric-a-b"
    finally:
        set_figure_prefix()
