"""Assign RFM segments and summarise them."""

import pandas as pd

from retail_analysis.config import SegmentRule


def _in_range(score: int, bounds: tuple[int, int]) -> bool:
    low, high = bounds
    return low <= score <= high


def assign_segment(r_score: int, f_score: int, m_score: int, rules: list[SegmentRule]) -> str:
    """Return the first segment whose score ranges contain the household's scores.

    Args:
        r_score: Recency score, 5 = most recent.
        f_score: Frequency score, 5 = most baskets.
        m_score: Monetary score, 5 = highest revenue.
        rules: Ordered segment rules.

    Returns:
        Segment name.

    Raises:
        ValueError: If no rule matches the scores.
    """
    for rule in rules:
        if (
            _in_range(r_score, rule.recency)
            and _in_range(f_score, rule.frequency)
            and _in_range(m_score, rule.monetary)
        ):
            return rule.name
    raise ValueError(f"No RFM segment rule matches scores R={r_score} F={f_score} M={m_score}")


def assign_segments(rfm: pd.DataFrame, rules: list[SegmentRule]) -> pd.DataFrame:
    """Add a `segment` column to the household RFM table.

    Args:
        rfm: Output of sql/03_rfm.sql with `r_score`, `f_score`, `m_score`.
        rules: Ordered segment rules.

    Returns:
        Copy of `rfm` with the segment name of every household.
    """
    segments = [
        assign_segment(r, f, m, rules)
        for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"], strict=True)
    ]
    return rfm.assign(segment=segments)


def segment_summary(segmented: pd.DataFrame, rules: list[SegmentRule]) -> pd.DataFrame:
    """Summarise segments: size, basket value, frequency and revenue share.

    Args:
        segmented: Output of `assign_segments`.
        rules: Segment rules, used for the row order.

    Returns:
        One row per segment in rule order with `households`, `household_share`,
        `avg_basket_value`, `avg_frequency`, `avg_recency_weeks`, `revenue` and
        `revenue_share`.
    """
    summary = segmented.groupby("segment").agg(
        households=("household_key", "count"),
        baskets=("frequency", "sum"),
        avg_frequency=("frequency", "mean"),
        avg_recency_weeks=("recency_weeks", "mean"),
        revenue=("monetary", "sum"),
    )
    summary["household_share"] = summary["households"] / summary["households"].sum()
    summary["avg_basket_value"] = summary["revenue"] / summary["baskets"]
    summary["revenue_share"] = summary["revenue"] / summary["revenue"].sum()
    order = [rule.name for rule in rules if rule.name in summary.index]
    columns = [
        "households",
        "household_share",
        "avg_basket_value",
        "avg_frequency",
        "avg_recency_weeks",
        "revenue",
        "revenue_share",
    ]
    return summary.loc[order, columns].reset_index()


def top_customers_revenue_share(monetary: pd.Series, top_share: float) -> float:
    """Share of revenue brought by the top customers.

    Args:
        monetary: Revenue per household.
        top_share: Share of households with the highest revenue, e.g. 0.2.

    Returns:
        Revenue share of the top `top_share` households.
    """
    top_count = max(1, round(len(monetary) * top_share))
    return float(monetary.nlargest(top_count).sum() / monetary.sum())
