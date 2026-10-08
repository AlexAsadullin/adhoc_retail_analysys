"""Describe churning loyal households and compare them with stable loyal ones."""

import pandas as pd
from scipy import stats

CHURNING = "churning_loyal"
STABLE = "stable_loyal"
DEMOGRAPHIC_COLUMNS = [
    "classification_1",
    "classification_2",
    "classification_3",
    "classification_4",
    "classification_5",
    "homeowner_desc",
    "kid_category_desc",
]


def households(churn_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per household from the household x department output of sql/05.

    Args:
        churn_rows: Output of sql/05_churning_loyal.sql.

    Returns:
        Household-level table without department columns.
    """
    return (
        churn_rows.drop(columns=["department", "department_revenue"])
        .drop_duplicates("household_key")
        .reset_index(drop=True)
    )


def group_summary(household_table: pd.DataFrame) -> pd.DataFrame:
    """Size, revenue and activity of every household group.

    Args:
        household_table: Output of `households`.

    Returns:
        One row per group with household counts, revenue shares for the whole window and
        for the baseline weeks, average basket value and baskets per block before and in
        the recent weeks.
    """
    summary = household_table.groupby("household_group").agg(
        households=("household_key", "count"),
        revenue=("revenue", "sum"),
        baseline_revenue=("baseline_revenue", "sum"),
        recent_revenue=("recent_revenue", "sum"),
        baskets=("baskets", "sum"),
        baseline_baskets=("baseline_baskets", "mean"),
        recent_baskets=("recent_baskets", "mean"),
    )
    summary["household_share"] = summary["households"] / summary["households"].sum()
    summary["revenue_share"] = summary["revenue"] / summary["revenue"].sum()
    summary["baseline_revenue_share"] = (
        summary["baseline_revenue"] / summary["baseline_revenue"].sum()
    )
    summary["recent_revenue_share"] = summary["recent_revenue"] / summary["recent_revenue"].sum()
    summary["avg_basket_value"] = summary["revenue"] / summary["baskets"]
    return summary.drop(columns=["baskets"]).reset_index()


def compare_basket_value(household_table: pd.DataFrame) -> dict[str, float]:
    """Compare average basket value of churning and stable loyal households.

    Args:
        household_table: Output of `households`.

    Returns:
        Medians of both groups and the two-sided Mann-Whitney U test p-value.
    """
    churning = household_table.loc[household_table["household_group"] == CHURNING]
    stable = household_table.loc[household_table["household_group"] == STABLE]
    test = stats.mannwhitneyu(
        churning["avg_basket_value"], stable["avg_basket_value"], alternative="two-sided"
    )
    return {
        "churning_median": float(churning["avg_basket_value"].median()),
        "stable_median": float(stable["avg_basket_value"].median()),
        "p_value": float(test.pvalue),
    }


def department_mix(churn_rows: pd.DataFrame, top_n: int, alpha: float) -> pd.DataFrame:
    """Compare department shares of baseline revenue between churning and stable loyal.

    Shares are computed per household, then averaged within a group, so that heavy
    spenders do not dominate. Each department is tested with a two-sided Mann-Whitney U
    test; significance uses the Bonferroni-corrected level `alpha / top_n`.

    Args:
        churn_rows: Output of sql/05_churning_loyal.sql.
        top_n: Number of departments with the largest revenue among loyal households.
        alpha: Family-wise significance level.

    Returns:
        One row per department with mean shares of both groups, their difference in
        percentage points, the p-value and a significance flag.
    """
    loyal = churn_rows[churn_rows["household_group"].isin([CHURNING, STABLE])]
    shares = loyal.pivot_table(
        index=["household_key", "household_group"],
        columns="department",
        values="department_revenue",
        aggfunc="sum",
        fill_value=0,
    )
    shares = shares.div(shares.sum(axis=1), axis=0)
    departments = loyal.groupby("department")["department_revenue"].sum().nlargest(top_n).index
    group = shares.index.get_level_values("household_group")
    rows = []
    for department in departments:
        churning = shares.loc[group == CHURNING, department]
        stable = shares.loc[group == STABLE, department]
        p_value = stats.mannwhitneyu(churning, stable, alternative="two-sided").pvalue
        rows.append(
            {
                "department": department,
                "churning_share": churning.mean(),
                "stable_share": stable.mean(),
                "difference_pp": 100 * (churning.mean() - stable.mean()),
                "p_value": p_value,
                "significant": p_value < alpha / top_n,
            }
        )
    return pd.DataFrame(rows)


def demographic_tests(
    household_table: pd.DataFrame, alpha: float, min_expected_count: float
) -> pd.DataFrame:
    """Chi-square tests of demographic differences between churning and stable loyal.

    Only households with demographic data are used. A difference is significant if the
    p-value is below the Bonferroni-corrected level `alpha / number of attributes` and
    every expected cell count reaches `min_expected_count`, so that the chi-square
    approximation holds.

    Args:
        household_table: Output of `households`.
        alpha: Family-wise significance level.
        min_expected_count: Smallest acceptable expected cell count.

    Returns:
        One row per demographic attribute with sample sizes, the chi-square p-value, the
        smallest expected cell count, a reliability flag and a significance flag.
    """
    loyal = household_table[
        household_table["household_group"].isin([CHURNING, STABLE])
        & household_table[DEMOGRAPHIC_COLUMNS[0]].notna()
    ]
    rows = []
    for column in DEMOGRAPHIC_COLUMNS:
        table = pd.crosstab(loyal[column], loyal["household_group"])
        test = stats.chi2_contingency(table)
        p_value = test.pvalue
        reliable = bool(test.expected_freq.min() >= min_expected_count)
        rows.append(
            {
                "attribute": column,
                "churning_with_demographics": int(table[CHURNING].sum()),
                "stable_with_demographics": int(table[STABLE].sum()),
                "p_value": p_value,
                "min_expected_count": float(test.expected_freq.min()),
                "reliable": reliable,
                "significant": reliable and p_value < alpha / len(DEMOGRAPHIC_COLUMNS),
            }
        )
    return pd.DataFrame(rows)
