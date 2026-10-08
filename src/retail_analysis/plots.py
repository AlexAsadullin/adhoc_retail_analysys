"""Charts in one shared style. Titles are passed in and state the conclusion."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter  # noqa: E402

from retail_analysis.config import PlotParams  # noqa: E402

_THOUSANDS = FuncFormatter(lambda value, _: f"{value / 1000:,.0f}".replace(",", " "))
_LABEL_WIDTH = 14
_LABEL_MARGIN = 1.15


def _wrap(labels: pd.Series) -> list[str]:
    return [textwrap.fill(str(label), _LABEL_WIDTH) for label in labels]


def set_style(params: PlotParams) -> None:
    """Apply the project chart style.

    Args:
        params: Plot settings with the colour palette.
    """
    sns.set_theme(style="whitegrid", palette=params.palette)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
        }
    )


def save(figure: Figure, path: Path, dpi: int) -> Path:
    """Save a figure as PNG and close it.

    Args:
        figure: Figure to save.
        path: Output file.
        dpi: Resolution.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def analysis_window(weekly: pd.DataFrame, window: dict[str, float], title: str) -> Figure:
    """Weekly active households with the threshold and the chosen analysis window.

    Args:
        weekly: Columns `week_no` and `active_households`.
        window: `start_week`, `end_week` and `active_households_threshold`.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.axvspan(
        window["start_week"] - 0.5, window["end_week"] + 0.5, alpha=0.12, label="Окно анализа"
    )
    axis.plot(
        weekly["week_no"],
        weekly["active_households"],
        linewidth=1.6,
        label="Активные домохозяйства",
    )
    axis.axhline(
        window["active_households_threshold"],
        linestyle="--",
        color="grey",
        label="Порог стабильности",
    )
    axis.set_xlabel("Неделя")
    axis.set_ylabel("Домохозяйств с покупкой, шт.")
    axis.set_ylim(bottom=0)
    axis.legend(loc="lower right")
    axis.set_title(title)
    return figure


def weekly_kpi(weekly: pd.DataFrame, title: str) -> Figure:
    """Four small charts of weekly revenue, baskets, active households and basket value.

    Args:
        weekly: Output of sql/01_weekly_kpi.sql.
        title: Figure title stating the conclusion.

    Returns:
        Figure.
    """
    panels = [
        ("revenue", "Выручка, тыс. $", _THOUSANDS),
        ("baskets", "Чеки, шт.", None),
        ("active_households", "Активные домохозяйства", None),
        ("avg_basket_value", "Средний чек, $", None),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, (column, label, formatter) in zip(axes.flat, panels, strict=True):
        axis.plot(weekly["week_no"], weekly[column], linewidth=1.6)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        if formatter is not None:
            axis.yaxis.set_major_formatter(formatter)
    for axis in axes[1]:
        axis.set_xlabel("Неделя")
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def category_shares(categories: pd.DataFrame, level: str, title: str) -> Figure:
    """Horizontal bars of revenue share for the top categories of one level.

    Args:
        categories: Output of sql/02_category_share.sql.
        level: "department" or "commodity".
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    data = categories[categories["category_level"] == level].sort_values("revenue_share")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(data["category"], data["revenue_share"])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Доля выручки, %")
    axis.set_ylabel("")
    for y, share in enumerate(data["revenue_share"]):
        axis.text(share, y, f" {share:.1%}", va="center", fontsize=9)
    axis.set_xlim(0, data["revenue_share"].max() * _LABEL_MARGIN)
    axis.set_title(title)
    return figure


def rfm_segments(summary: pd.DataFrame, title: str) -> Figure:
    """Share of households versus share of revenue for every RFM segment.

    Args:
        summary: Output of `rfm.segment_summary`.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    data = summary.melt(
        id_vars="segment",
        value_vars=["household_share", "revenue_share"],
        var_name="metric",
        value_name="share",
    )
    data["metric"] = data["metric"].map(
        {"household_share": "Доля клиентов", "revenue_share": "Доля выручки"}
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=data, x="segment", y="share", hue="metric", ax=axis)
    axis.set_xticks(range(len(summary)), _wrap(summary["segment"]))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("RFM-сегмент")
    axis.set_ylabel("Доля, %")
    axis.legend(title="")
    axis.set_title(title)
    return figure


def cohort_heatmap(matrix: pd.DataFrame, labels: list[str], title: str) -> Figure:
    """Heatmap of the retention matrix.

    Args:
        matrix: Output of `cohorts.retention_matrix`.
        labels: Cohort labels in matrix row order.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    figure, axis = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        matrix * 100,
        ax=axis,
        cmap="Blues",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Удержание, %"},
        yticklabels=labels,
    )
    axis.set_xlabel("Периодов (по 4 недели) с первой покупки")
    axis.set_ylabel("Когорта первой покупки")
    axis.set_title(title)
    return figure


def cohort_lines(matrix: pd.DataFrame, labels: list[str], title: str) -> Figure:
    """Retention curves of all cohorts.

    Args:
        matrix: Output of `cohorts.retention_matrix`.
        labels: Cohort labels in matrix row order.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    figure, axis = plt.subplots(figsize=(10, 5))
    for (_, row), label in zip(matrix.iterrows(), labels, strict=True):
        axis.plot(row.index, row.to_numpy(), marker="o", markersize=3, label=label)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Периодов (по 4 недели) с первой покупки")
    axis.set_ylabel("Доля активных клиентов когорты, %")
    axis.legend(title="Когорта")
    axis.set_title(title)
    return figure


def loyal_activity(summary: pd.DataFrame, labels: dict[str, str], title: str) -> Figure:
    """Baskets per household before and in the recent block for loyal groups.

    Args:
        summary: Output of `churn.group_summary`.
        labels: Display names of the groups to show, keyed by group code.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    data = summary[summary["household_group"].isin(labels)].melt(
        id_vars="household_group",
        value_vars=["baseline_baskets", "recent_baskets"],
        var_name="period",
        value_name="baskets",
    )
    data["household_group"] = data["household_group"].map(labels)
    data["period"] = data["period"].map(
        {"baseline_baskets": "Раньше (в среднем за блок)", "recent_baskets": "Последний блок"}
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="household_group", y="baskets", hue="period", ax=axis)
    axis.set_xlabel("")
    axis.set_ylabel("Чеков на домохозяйство за 12 недель, шт.")
    axis.legend(title="")
    axis.set_title(title)
    return figure


def department_differences(mix: pd.DataFrame, title: str) -> Figure:
    """Difference in department share between churning and stable loyal households.

    Args:
        mix: Output of `churn.department_mix`.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    data = mix.sort_values("difference_pp")
    palette = sns.color_palette()
    colors = [palette[1] if significant else palette[4] for significant in data["significant"]]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(data["department"], data["difference_pp"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Разница доли в выручке: уходящие минус стабильные, п.п.")
    axis.set_ylabel("")
    axis.legend(
        handles=[
            Patch(color=palette[1], label="Значимо (с поправкой Бонферрони)"),
            Patch(color=palette[4], label="Незначимо"),
        ],
        loc="lower right",
    )
    axis.set_title(title)
    return figure


def promo_trends(means: pd.DataFrame, event_study: pd.DataFrame, title: str) -> Figure:
    """Group means and event-study coefficients around the promo week.

    Args:
        means: Output of `promo.group_means`.
        event_study: Output of `promo.estimate_event_study`.
        title: Figure title stating the conclusion.

    Returns:
        Figure.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    left.plot(means.index, means["treated"], marker="o", label="В каталоге")
    left.plot(means.index, means["control"], marker="o", label="Контроль")
    for axis in (left, right):
        axis.xaxis.set_major_locator(MultipleLocator(1))
    left.set_xlabel("Неделя относительно промо")
    left.set_ylabel("Среднее log(1 + штук в неделю)")
    left.set_title("Средние продажи групп")
    left.legend()
    errors = [
        event_study["coefficient"] - event_study["ci_low"],
        event_study["ci_high"] - event_study["coefficient"],
    ]
    right.errorbar(
        event_study["rel_week"], event_study["coefficient"], yerr=errors, fmt="o", capsize=4
    )
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xlabel("Неделя относительно промо (база: -1)")
    right.set_ylabel("Разница с контролем, log-пункты")
    right.set_title("Event study, 95% ДИ")
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def promo_effects(effects: pd.DataFrame, title: str) -> Figure:
    """Point estimates and confidence intervals of promo effects in percent.

    Args:
        effects: Columns `label`, `effect_pct`, `effect_pct_ci_low`, `effect_pct_ci_high`.
        title: Chart title stating the conclusion.

    Returns:
        Figure.
    """
    data = effects.iloc[::-1].reset_index(drop=True)
    errors = [
        data["effect_pct"] - data["effect_pct_ci_low"],
        data["effect_pct_ci_high"] - data["effect_pct"],
    ]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.errorbar(data["effect_pct"], data.index, xerr=errors, fmt="o", capsize=4)
    axis.set_yticks(data.index, data["label"])
    axis.set_ylim(-0.5, len(data) - 0.3)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.xaxis.set_major_formatter(PercentFormatter(100))
    axis.set_xlabel("Изменение продаж в неделю промо, % (95% ДИ)")
    for y, value in zip(data.index, data["effect_pct"], strict=True):
        axis.annotate(
            f"{value:+.1f}%", (value, y), textcoords="offset points", xytext=(0, 8), ha="center"
        )
    axis.set_title(title)
    return figure
