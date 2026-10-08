"""Run every analysis, save figures and write reports/results.json."""

import json
import logging
from dataclasses import asdict
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from retail_analysis import churn, cohorts, plots, promo, rfm
from retail_analysis.config import Config, load_config
from retail_analysis.queries import connect, run_query

logger = logging.getLogger(__name__)

GROUP_LABELS = {churn.CHURNING: "Уходящие лояльные", churn.STABLE: "Стабильные лояльные"}


def period_change(values: pd.Series, weeks: int) -> float:
    """Relative change between the mean of the last and the first `weeks` values.

    Args:
        values: Weekly values in time order.
        weeks: Number of weeks averaged at each end.

    Returns:
        Mean of the last `weeks` divided by the mean of the first `weeks`, minus one.
    """
    return float(values.iloc[-weeks:].mean() / values.iloc[:weeks].mean() - 1)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _figure(config: Config, figure: Any, name: str) -> str:
    path = plots.save(figure, config.paths.figures_dir / f"{name}.png", config.plots.dpi)
    return path.relative_to(config.paths.figures_dir.parent.parent).as_posix()


def _window_params(window: dict[str, int]) -> dict[str, int]:
    return {"start_week": window["start_week"], "end_week": window["end_week"]}


def analyze_business(
    con: duckdb.DuckDBPyConnection, config: Config, window: dict[str, int]
) -> dict[str, Any]:
    """Weekly KPIs and top categories (question 1).

    Args:
        con: DuckDB connection.
        config: Project configuration.
        window: Analysis window with `start_week` and `end_week`.

    Returns:
        Section results.
    """
    weekly = run_query(con, config.paths.sql_dir, "01_weekly_kpi", _window_params(window))
    categories = run_query(
        con,
        config.paths.sql_dir,
        "02_category_share",
        {**_window_params(window), "top_n": config.analysis.top_categories},
    )
    trend_weeks = config.analysis.trend_weeks
    revenue_change = period_change(weekly["revenue"], trend_weeks)
    baskets_change = period_change(weekly["baskets"], trend_weeks)
    basket_value_change = (1 + revenue_change) / (1 + baskets_change) - 1
    households_change = period_change(weekly["active_households"], trend_weeks)
    totals = (
        con.execute(
            """
        SELECT
            SUM(sales_value) AS revenue,
            COUNT(DISTINCT basket_id) AS baskets,
            COUNT(DISTINCT household_key) AS households
        FROM transaction_data
        WHERE week_no BETWEEN $start_week AND $end_week
        """,
            _window_params(window),
        )
        .df()
        .iloc[0]
    )
    departments = categories[categories["category_level"] == "department"]
    commodities = categories[categories["category_level"] == "commodity"]
    top3_share = float(departments["revenue_share"].head(3).sum())
    direction = "выросла" if revenue_change > 0 else "снизилась"
    figures = {
        "weekly_kpi": _figure(
            config,
            plots.weekly_kpi(
                weekly,
                f"Выручка {direction} на {abs(revenue_change):.0%} (последние {trend_weeks} нед. "
                f"к первым): средний чек {basket_value_change:+.0%}, "
                f"число чеков {baskets_change:+.0%}",
            ),
            "weekly_kpi",
        ),
        "departments": _figure(
            config,
            plots.category_shares(
                categories,
                "department",
                f"{departments['category'].iloc[0]} дает "
                f"{departments['revenue_share'].iloc[0]:.0%} выручки, "
                f"топ-3 отдела — {top3_share:.0%}",
            ),
            "departments",
        ),
        "commodities": _figure(
            config,
            plots.category_shares(
                categories,
                "commodity",
                f"Выручка распределена широко: крупнейшая категория дает "
                f"{commodities['revenue_share'].iloc[0]:.1%}, следующая — "
                f"{commodities['revenue_share'].iloc[1]:.1%}",
            ),
            "commodities",
        ),
    }
    return {
        "total_revenue": float(totals["revenue"]),
        "total_baskets": int(totals["baskets"]),
        "active_households": int(totals["households"]),
        "avg_weekly_revenue": float(weekly["revenue"].mean()),
        "avg_weekly_baskets": float(weekly["baskets"].mean()),
        "avg_weekly_active_households": float(weekly["active_households"].mean()),
        "avg_basket_value": float(totals["revenue"] / totals["baskets"]),
        "trend_weeks": trend_weeks,
        "revenue_change_last_vs_first": revenue_change,
        "baskets_change_last_vs_first": baskets_change,
        "basket_value_change_last_vs_first": basket_value_change,
        "active_households_change_last_vs_first": households_change,
        "top3_department_share": top3_share,
        "weekly": _records(weekly),
        "top_departments": _records(departments),
        "top_commodities": _records(commodities),
        "figures": figures,
    }


def analyze_rfm(
    con: duckdb.DuckDBPyConnection, config: Config, window: dict[str, int]
) -> dict[str, Any]:
    """RFM segmentation (question 2).

    Args:
        con: DuckDB connection.
        config: Project configuration.
        window: Analysis window.

    Returns:
        Section results.
    """
    scores = run_query(
        con,
        config.paths.sql_dir,
        "03_rfm",
        {**_window_params(window), "buckets": config.rfm.score_buckets},
    )
    segmented = rfm.assign_segments(scores, config.rfm.segments)
    summary = rfm.segment_summary(segmented, config.rfm.segments)
    top_share = config.rfm.top_customers_share
    concentration = rfm.top_customers_revenue_share(scores["monetary"], top_share)
    best = summary.iloc[0]
    figure = plots.rfm_segments(
        summary,
        f"«{best['segment']}» — {best['household_share']:.0%} клиентов и "
        f"{best['revenue_share']:.0%} выручки",
    )
    return {
        "households": len(scores),
        "top_customers_share": top_share,
        "top_customers_revenue_share": concentration,
        "share_bought_in_last_week": float((scores["recency_weeks"] == 0).mean()),
        "segments": _records(summary),
        "figures": {"rfm_segments": _figure(config, figure, "rfm_segments")},
    }


def analyze_cohorts(
    con: duckdb.DuckDBPyConnection, config: Config, window: dict[str, int]
) -> dict[str, Any]:
    """Cohort retention (question 3).

    Args:
        con: DuckDB connection.
        config: Project configuration.
        window: Analysis window; cohorts use all weeks up to its end.

    Returns:
        Section results.
    """
    params = config.cohorts
    activity = run_query(
        con,
        config.paths.sql_dir,
        "04_cohorts",
        {"end_week": window["end_week"], "period_weeks": params.period_weeks},
    )
    last_period = cohorts.last_full_period(window["end_week"], params.period_weeks)
    matrix = cohorts.retention_matrix(activity, params.min_cohort_size, last_period)
    sizes = cohorts.cohort_sizes(activity)
    average = cohorts.average_retention(matrix, sizes)
    labels = [cohorts.cohort_label(period, params.period_weeks) for period in matrix.index]
    kept_households = int(sizes[matrix.index].sum())
    report = {str(period): float(average[period]) for period in params.report_periods}
    first, *_, last = params.report_periods
    title = (
        f"Через {first * params.period_weeks} недели покупает {average[first]:.0%} когорты, "
        f"через {last * params.period_weeks} недель — {average[last]:.0%}"
    )
    figures = {
        "cohort_heatmap": _figure(
            config, plots.cohort_heatmap(matrix, labels, title), "cohort_heatmap"
        ),
        "cohort_lines": _figure(
            config,
            plots.cohort_lines(
                matrix,
                labels,
                f"Кривые удержания всех когорт выходят на плато "
                f"{average.iloc[1:].min():.0%}-{average.iloc[1:].max():.0%}",
            ),
            "cohort_lines",
        ),
    }
    return {
        "period_weeks": params.period_weeks,
        "min_cohort_size": params.min_cohort_size,
        "cohorts_kept": len(matrix),
        "households_in_kept_cohorts": kept_households,
        "households_in_dropped_cohorts": int(sizes.sum() - kept_households),
        "cohort_sizes": {
            label: int(sizes[period]) for label, period in zip(labels, matrix.index, strict=True)
        },
        "average_retention_by_period": report,
        "retention_plateau_min": float(average.iloc[1:].min()),
        "retention_plateau_max": float(average.iloc[1:].max()),
        "retention_matrix": {
            label: [None if np.isnan(value) else float(value) for value in row]
            for label, row in zip(labels, matrix.to_numpy(), strict=True)
        },
        "figures": figures,
    }


def analyze_churning_loyal(
    con: duckdb.DuckDBPyConnection, config: Config, window: dict[str, int]
) -> dict[str, Any]:
    """Churning loyal households (separate finding).

    Args:
        con: DuckDB connection.
        config: Project configuration.
        window: Analysis window.

    Returns:
        Section results.
    """
    params = config.churning_loyal
    mid_week = (window["start_week"] + window["end_week"]) // 2
    rows = run_query(
        con,
        config.paths.sql_dir,
        "05_churning_loyal",
        {
            **_window_params(window),
            "mid_week": mid_week,
            "recent_weeks": params.recent_weeks,
            "loyal_top_share": params.loyal_top_share,
            "activity_drop": params.activity_drop,
        },
    )
    household_table = churn.households(rows)
    summary = churn.group_summary(household_table)
    basket = churn.compare_basket_value(household_table)
    mix = churn.department_mix(rows, params.top_departments, params.alpha)
    demographics = churn.demographic_tests(household_table, params.alpha, params.min_expected_count)
    churning = summary.set_index("household_group").loc[churn.CHURNING]
    drop = 1 - churning["recent_baskets"] / churning["baseline_baskets"]
    significant = mix[mix["significant"]]
    if significant.empty:
        mix_title = "Структура корзины уходящих лояльных значимо не отличается"
    else:
        top = significant.loc[significant["difference_pp"].abs().idxmax()]
        sign = "больше" if top["difference_pp"] > 0 else "меньше"
        mix_title = (
            f"Уходящие лояльные тратят на {top['department']} на "
            f"{abs(top['difference_pp']):.1f} п.п. {sign}"
        )
    figures = {
        "loyal_activity": _figure(
            config,
            plots.loyal_activity(
                summary,
                GROUP_LABELS,
                f"{int(churning['households'])} уходящих лояльных "
                f"сократили число чеков на {drop:.0%}",
            ),
            "loyal_activity",
        ),
        "churning_departments": _figure(
            config, plots.department_differences(mix, mix_title), "churning_departments"
        ),
    }
    return {
        "mid_week": mid_week,
        "recent_weeks": params.recent_weeks,
        "loyal_top_share": params.loyal_top_share,
        "activity_drop": params.activity_drop,
        "churning_households": int(churning["households"]),
        "loyal_households": int(
            summary.loc[summary["household_group"].isin(GROUP_LABELS), "households"].sum()
        ),
        "churning_baseline_revenue_share": float(churning["baseline_revenue_share"]),
        "churning_recent_revenue_share": float(churning["recent_revenue_share"]),
        "churning_window_revenue_share": float(churning["revenue_share"]),
        "churning_basket_drop": float(drop),
        "groups": _records(summary),
        "basket_value_comparison": basket,
        "department_mix": _records(mix),
        "demographic_tests": _records(demographics),
        "figures": figures,
    }


def _effects_table(results: dict[str, promo.DidResult]) -> pd.DataFrame:
    labels = {
        "units": "Каталог: штуки",
        "revenue": "Каталог: выручка",
        "units_display_control": "Каталог: штуки,\nс учетом выкладки",
        "cannibalization_units": "Соседние товары:\nштуки",
    }
    return pd.DataFrame(
        [
            {
                "label": labels[key],
                "effect_pct": result.effect_pct,
                "effect_pct_ci_low": result.effect_pct_ci_low,
                "effect_pct_ci_high": result.effect_pct_ci_high,
            }
            for key, result in results.items()
        ]
    )


def analyze_promo(config: Config, window: dict[str, int]) -> dict[str, Any]:
    """Mailer effect and cannibalisation (question 4).

    Args:
        config: Project configuration.
        window: Analysis window.

    Returns:
        Section results.
    """
    params = config.promo
    tables = promo.to_wide(
        promo.load_product_weeks(config.paths.processed_dir, params.excluded_departments)
    )
    start, end = window["start_week"], window["end_week"]
    panel = promo.expand_to_panel(
        promo.promo_events(tables, params, start, end), tables, params.pre_weeks
    )
    cannibal_panel = promo.expand_to_panel(
        promo.cannibalization_events(tables, params, start, end), tables, params.pre_weeks
    )
    level = params.confidence_level
    estimates = {
        "units": promo.estimate_did(panel, "units", level),
        "revenue": promo.estimate_did(panel, "revenue", level),
        "units_display_control": promo.estimate_did(panel, "units", level, ("display_share",)),
        "cannibalization_units": promo.estimate_did(cannibal_panel, "units", level),
    }
    event_study, pretrend_p = promo.estimate_event_study(panel, "units", level)
    cannibal_study, cannibal_pretrend_p = promo.estimate_event_study(cannibal_panel, "units", level)
    pre = event_study[event_study["rel_week"] < -1]
    max_pre_gap = float(100 * np.expm1(pre["coefficient"].abs().max()))
    display = promo.concurrent_display_share(panel)
    alpha = 1 - level
    units = estimates["units"]
    with_display = estimates["units_display_control"]
    cannibal = estimates["cannibalization_units"]
    cannibal_text = (
        "соседние товары значимо не меняются"
        if cannibal.p_value >= alpha
        else f"соседние товары: {cannibal.effect_pct:+.1f}%"
    )
    figures = {
        "promo_trends": _figure(
            config,
            plots.promo_trends(
                promo.group_means(panel, "units"),
                event_study,
                f"До промо разница с контролем не больше {max_pre_gap:.0f}%, "
                f"в неделю промо {units.effect_pct:+.0f}%",
            ),
            "promo_trends",
        ),
        "promo_effects": _figure(
            config,
            plots.promo_effects(
                _effects_table(estimates),
                f"Каталог: {units.effect_pct:+.0f}% штук, с учетом выкладки "
                f"{with_display.effect_pct:+.0f}%;\n{cannibal_text}",
            ),
            "promo_effects",
        ),
    }
    return {
        "pre_weeks": params.pre_weeks,
        "treated_min_store_share": params.treated_min_store_share,
        "confidence_level": level,
        "estimates": {key: asdict(result) for key, result in estimates.items()},
        "event_study_units": _records(event_study),
        "pretrend_joint_p_value": pretrend_p,
        "max_pre_period_gap_pct": max_pre_gap,
        "cannibalization_event_study_units": _records(cannibal_study),
        "cannibalization_pretrend_joint_p_value": cannibal_pretrend_p,
        "concurrent_display_share": display,
        "figures": figures,
    }


def data_summary(config: Config, quality: dict[str, Any]) -> dict[str, Any]:
    """Row counts from preparation, the data quality report and the window chart.

    Args:
        config: Project configuration.
        quality: Content of the quality report written by `quality`.

    Returns:
        Section results.
    """
    prepare_stats = json.loads(config.paths.prepare_stats_file.read_text(encoding="utf-8"))
    transactions = prepare_stats["transaction_data"]
    removed = transactions["zero_quantity_removed"] + transactions["non_positive_sales_removed"]
    window = quality["analysis_window"]
    weekly = pd.DataFrame(quality["weekly_activity"])
    excluded = [f"1-{window['start_week'] - 1} (набор панели)"]
    if weekly["week_no"].max() > window["end_week"]:
        excluded.append(f"{window['end_week'] + 1}-{weekly['week_no'].max()} (неполные)")
    figure = plots.analysis_window(
        weekly,
        window,
        f"Исключены недели {' и '.join(excluded)}: "
        f"анализ по неделям {window['start_week']}-{window['end_week']}",
    )
    return {
        "source": config.source.name,
        "source_url": config.source.url,
        "tables": prepare_stats,
        "removed_transaction_share": removed / transactions["rows_raw"],
        "quality_checks": quality["checks"],
        "analysis_window": quality["analysis_window"],
        "weekly_activity": quality["weekly_activity"],
        "stable_activity_ratio": config.quality.stable_activity_ratio,
        "figures": {"analysis_window": _figure(config, figure, "analysis_window")},
    }


def dataset_totals(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Size of the clean transaction table over all weeks.

    Args:
        con: DuckDB connection.

    Returns:
        Households, baskets, products, stores and weeks.
    """
    row = (
        con.execute(
            """
        SELECT
            COUNT(DISTINCT household_key) AS households,
            COUNT(DISTINCT basket_id) AS baskets,
            COUNT(DISTINCT product_id) AS products,
            COUNT(DISTINCT store_id) AS stores,
            COUNT(DISTINCT week_no) AS weeks
        FROM transaction_data
        """
        )
        .df()
        .iloc[0]
    )
    return {key: int(value) for key, value in row.items()}


def main() -> None:
    """Entry point for `make analyze`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    plots.set_style(config.plots)
    quality = json.loads(config.paths.quality_file.read_text(encoding="utf-8"))
    window = quality["analysis_window"]
    con = connect(config.paths.processed_dir)
    results: dict[str, Any] = {"data": {**data_summary(config, quality), **dataset_totals(con)}}
    sections = {
        "business": lambda: analyze_business(con, config, window),
        "rfm": lambda: analyze_rfm(con, config, window),
        "cohorts": lambda: analyze_cohorts(con, config, window),
        "churning_loyal": lambda: analyze_churning_loyal(con, config, window),
        "promo": lambda: analyze_promo(config, window),
    }
    for name, run in sections.items():
        logger.info("Running %s analysis", name)
        results[name] = run()
    config.paths.results_file.parent.mkdir(parents=True, exist_ok=True)
    config.paths.results_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved %s", config.paths.results_file)


if __name__ == "__main__":
    main()
