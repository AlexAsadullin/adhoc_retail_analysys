"""Run SQL files from sql/ with DuckDB over the processed Parquet tables."""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

TABLES = ("transaction_data", "product", "hh_demographic", "causal_data", "promo_product_week")


def connect(processed_dir: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with one view per processed table.

    Args:
        processed_dir: Directory with Parquet tables written by `prepare`.

    Returns:
        Connection where each table from `TABLES` is available as a view.
    """
    con = duckdb.connect()
    for table in TABLES:
        pattern = (processed_dir / table / "**" / "*.parquet").as_posix()
        con.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)"
        )
    return con


def run_query(
    con: duckdb.DuckDBPyConnection, sql_dir: Path, name: str, params: dict[str, Any]
) -> pd.DataFrame:
    """Execute one SQL file with named parameters.

    Args:
        con: DuckDB connection with the source views.
        sql_dir: Directory with SQL files.
        name: File stem, e.g. "03_rfm".
        params: Values for `$name` placeholders in the query.

    Returns:
        Query result.
    """
    query = (sql_dir / f"{name}.sql").read_text(encoding="utf-8")
    return con.execute(query, params).df()
