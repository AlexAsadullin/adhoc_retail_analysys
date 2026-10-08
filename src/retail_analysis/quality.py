"""Data quality checks and the choice of the analysis window."""

import json
import logging
from dataclasses import asdict, dataclass

import duckdb
import pandas as pd

from retail_analysis.config import QualityParams, load_config
from retail_analysis.queries import connect

logger = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    """Raised when a critical data quality check fails."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one data quality check."""

    name: str
    passed: bool
    critical: bool
    details: str


@dataclass(frozen=True)
class AnalysisWindow:
    """Weeks used by the analysis, both inclusive."""

    start_week: int
    end_week: int
    active_households_threshold: float


def check_no_nulls(con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> CheckResult:
    """Check that key columns of a table have no missing values.

    Args:
        con: Connection with the table available as a view.
        table: Table name.
        columns: Key columns to check.

    Returns:
        Critical check result listing columns with nulls.
    """
    counts = ", ".join(f'COUNT(*) FILTER (WHERE "{column}" IS NULL)' for column in columns)
    null_counts = dict(
        zip(columns, con.execute(f"SELECT {counts} FROM {table}").fetchone(), strict=True)
    )
    with_nulls = {column: count for column, count in null_counts.items() if count > 0}
    details = f"nulls: {with_nulls}" if with_nulls else f"no nulls in {', '.join(columns)}"
    return CheckResult(f"no_nulls_{table}", not with_nulls, critical=True, details=details)


def check_products_in_reference(con: duckdb.DuckDBPyConnection) -> CheckResult:
    """Check that every product_id in transactions exists in the product reference.

    Args:
        con: Connection with `transaction_data` and `product` views.

    Returns:
        Critical check result with the number of unknown products.
    """
    missing = con.execute(
        """
        SELECT COUNT(DISTINCT t.product_id)
        FROM transaction_data AS t
        ANTI JOIN product AS p USING (product_id)
        """
    ).fetchone()[0]
    return CheckResult(
        "products_in_reference",
        missing == 0,
        critical=True,
        details=f"{missing} product_id values from transactions are missing in product",
    )


def check_product_hierarchy(
    con: duckdb.DuckDBPyConnection, columns: list[str], sold_only: bool
) -> CheckResult:
    """Check that products have department and category descriptions.

    Products that never sold do not affect the analysis, so a gap there is not critical.

    Args:
        con: Connection with `transaction_data` and `product` views.
        columns: Hierarchy columns that must be filled.
        sold_only: Check only products present in transactions (critical) or the whole
            reference (informational).

    Returns:
        Check result with the number of products missing any hierarchy column.
    """
    any_null = " OR ".join(f'p."{column}" IS NULL' for column in columns)
    scope = "SEMI JOIN transaction_data AS t USING (product_id)" if sold_only else ""
    missing = con.execute(f"SELECT COUNT(*) FROM product AS p {scope} WHERE {any_null}").fetchone()[
        0
    ]
    label = "sold" if sold_only else "reference"
    return CheckResult(
        f"product_hierarchy_{label}",
        missing == 0,
        critical=sold_only,
        details=f"{missing} {label} products without {', '.join(columns)}",
    )


def check_week_range(
    con: duckdb.DuckDBPyConnection, table: str, min_week: int, max_week: int
) -> CheckResult:
    """Check that week_no stays within the documented range.

    Args:
        con: Connection with the table available as a view.
        table: Table name.
        min_week: Smallest allowed week.
        max_week: Largest allowed week.

    Returns:
        Critical check result with the observed range.
    """
    low, high = con.execute(f"SELECT MIN(week_no), MAX(week_no) FROM {table}").fetchone()
    return CheckResult(
        f"week_range_{table}",
        min_week <= low and high <= max_week,
        critical=True,
        details=f"observed weeks {low}-{high}, allowed {min_week}-{max_week}",
    )


def weekly_activity(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Count active households and distinct days per week.

    Args:
        con: Connection with the `transaction_data` view.

    Returns:
        Columns `week_no`, `active_households`, `days`, ordered by week.
    """
    return con.execute(
        """
        SELECT
            week_no,
            COUNT(DISTINCT household_key) AS active_households,
            COUNT(DISTINCT day) AS days
        FROM transaction_data
        GROUP BY week_no
        ORDER BY week_no
        """
    ).df()


def find_analysis_window(
    weekly: pd.DataFrame, stable_activity_ratio: float, full_week_days: int
) -> AnalysisWindow:
    """Choose the weeks with complete data and stable household activity.

    The window ends at the last complete week. It starts at the earliest week from which
    every week up to the end has at least `stable_activity_ratio` of the median number
    of weekly active households.

    Args:
        weekly: Output of `weekly_activity`.
        stable_activity_ratio: Required share of the median weekly active households.
        full_week_days: Number of distinct days in a complete week.

    Returns:
        Analysis window.
    """
    weekly = weekly.sort_values("week_no")
    end_week = int(weekly.loc[weekly["days"] >= full_week_days, "week_no"].max())
    threshold = stable_activity_ratio * weekly["active_households"].median()
    in_range = weekly[weekly["week_no"] <= end_week]
    below = in_range.loc[in_range["active_households"] < threshold, "week_no"]
    start_week = int(below.max() + 1) if not below.empty else int(in_range["week_no"].min())
    return AnalysisWindow(start_week, end_week, float(threshold))


def run_checks(con: duckdb.DuckDBPyConnection, params: QualityParams) -> list[CheckResult]:
    """Run all data quality checks.

    Args:
        con: Connection with the processed tables.
        params: Quality settings.

    Returns:
        One result per check.
    """
    results = [check_no_nulls(con, table, columns) for table, columns in params.key_columns.items()]
    results.append(check_products_in_reference(con))
    for sold_only in (True, False):
        results.append(check_product_hierarchy(con, params.hierarchy_columns, sold_only))
    for table in ("transaction_data", "causal_data"):
        results.append(check_week_range(con, table, params.min_week, params.max_week))
    return results


def raise_on_critical(results: list[CheckResult]) -> None:
    """Stop the pipeline if any critical check failed.

    Args:
        results: Check results.

    Raises:
        DataQualityError: If at least one critical check failed.
    """
    failed = [result for result in results if result.critical and not result.passed]
    if failed:
        lines = "; ".join(f"{result.name}: {result.details}" for result in failed)
        raise DataQualityError(f"Critical data quality checks failed: {lines}")


def main() -> None:
    """Entry point for `make quality`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    con = connect(config.paths.processed_dir)
    results = run_checks(con, config.quality)
    for result in results:
        status = "PASS" if result.passed else ("FAIL" if result.critical else "WARN")
        logger.info("%s %s: %s", status, result.name, result.details)
    weekly = weekly_activity(con)
    window = find_analysis_window(
        weekly, config.quality.stable_activity_ratio, config.quality.full_week_days
    )
    logger.info(
        "Analysis window: weeks %d-%d (active households threshold %.0f)",
        window.start_week,
        window.end_week,
        window.active_households_threshold,
    )
    report = {
        "checks": [asdict(result) for result in results],
        "analysis_window": asdict(window),
        "weekly_activity": weekly.to_dict(orient="records"),
    }
    config.paths.quality_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise_on_critical(results)


if __name__ == "__main__":
    main()
