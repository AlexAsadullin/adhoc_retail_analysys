"""Build the cohort retention matrix."""

import numpy as np
import pandas as pd


def last_full_period(end_week: int, period_weeks: int) -> int:
    """Index of the last period that ends on or before `end_week`.

    Args:
        end_week: Last week of the data used.
        period_weeks: Weeks per period; period 0 covers weeks 1..period_weeks.

    Returns:
        Zero-based period index.
    """
    return end_week // period_weeks - 1


def cohort_sizes(cohort_activity: pd.DataFrame) -> pd.Series:
    """Number of households in every cohort.

    Args:
        cohort_activity: Output of sql/04_cohorts.sql.

    Returns:
        Households active in the first period, indexed by `cohort_period`.
    """
    first_period = cohort_activity[cohort_activity["periods_since_first"] == 0]
    return first_period.set_index("cohort_period")["active_households"]


def retention_matrix(
    cohort_activity: pd.DataFrame, min_cohort_size: int, last_period: int
) -> pd.DataFrame:
    """Share of each cohort active in every period since the first purchase.

    Args:
        cohort_activity: Output of sql/04_cohorts.sql.
        min_cohort_size: Cohorts with fewer households are dropped.
        last_period: Last observed period, see `last_full_period`.

    Returns:
        Matrix indexed by `cohort_period` with one column per `periods_since_first`
        from 0 to the longest observable offset. Values are shares in [0, 1]; periods
        after `last_period` are NaN, observed periods without purchases are 0.
    """
    sizes = cohort_sizes(cohort_activity)
    kept = sizes[sizes >= min_cohort_size].index
    activity = cohort_activity[cohort_activity["cohort_period"].isin(kept)]
    offsets = range(last_period - int(kept.min()) + 1)
    counts = activity.pivot(
        index="cohort_period", columns="periods_since_first", values="active_households"
    ).reindex(columns=offsets, fill_value=0)
    observed = np.add.outer(counts.index.to_numpy(), np.array(offsets)) <= last_period
    return counts.fillna(0).where(observed).div(sizes[kept], axis=0)


def average_retention(matrix: pd.DataFrame, sizes: pd.Series) -> pd.Series:
    """Household-weighted average retention by period since the first purchase.

    Args:
        matrix: Output of `retention_matrix`.
        sizes: Cohort sizes indexed by `cohort_period`.

    Returns:
        Weighted retention for each `periods_since_first`, using only observed cells.
    """
    weights = matrix.notna().mul(sizes.reindex(matrix.index), axis=0)
    return matrix.fillna(0).mul(weights).sum() / weights.sum()


def cohort_label(cohort_period: int, period_weeks: int) -> str:
    """Human-readable week range of a cohort.

    Args:
        cohort_period: Zero-based period index.
        period_weeks: Weeks per period.

    Returns:
        Label like "нед. 1-4".
    """
    first_week = cohort_period * period_weeks + 1
    return f"нед. {first_week}-{first_week + period_weeks - 1}"
