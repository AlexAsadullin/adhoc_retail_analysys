"""Small synthetic tables for unit tests. No real data and no Spark."""

from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from retail_analysis.config import PROJECT_ROOT, Config, load_config

PROMO_EFFECT = 0.4


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def sql_dir() -> Path:
    return PROJECT_ROOT / "sql"


@pytest.fixture
def make_con() -> Callable[..., duckdb.DuckDBPyConnection]:
    """Register DataFrames as DuckDB views named like the processed tables."""

    def factory(**tables: pd.DataFrame) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        for name, frame in tables.items():
            con.register(name, frame)
        return con

    return factory


def _lines(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """Build transactions from (household, basket, week, sales_value) tuples."""
    frame = pd.DataFrame(rows, columns=["household_key", "basket_id", "week_no", "sales_value"])
    frame["day"] = frame["week_no"] * 7
    frame["product_id"] = 1
    frame["quantity"] = 1
    frame["store_id"] = 1
    return frame


@pytest.fixture
def rfm_transactions() -> pd.DataFrame:
    """Household k (1..5) has k baskets of 10*k each, the last one in week 5 + k.

    With end_week = 10 its recency is 5 - k weeks, so every RFM score equals k.
    """
    rows = []
    basket = 0
    for household in range(1, 6):
        for week in range(6, 6 + household):
            basket += 1
            rows.append((household, basket, week, 10.0 * household))
    return _lines(rows)


@pytest.fixture
def cohort_transactions() -> pd.DataFrame:
    """Hand-made activity with 2-week periods and data up to week 6.

    Cohort 0 (weeks 1-2): households 1, 2, 3. Cohort 1 (weeks 3-4): households 4, 5.
    Household 1 also buys in week 7, after the end week, which must be ignored.
    """
    weeks = {1: [1, 3, 5, 7], 2: [2, 6], 3: [1], 4: [3, 4], 5: [4, 5]}
    rows = []
    basket = 0
    for household, active_weeks in weeks.items():
        for week in active_weeks:
            basket += 1
            rows.append((household, basket, week, 5.0))
    return _lines(rows)


@pytest.fixture
def product_weeks() -> pd.DataFrame:
    """Product-week panel with a known mailer effect on log(1 + units) and no spillover.

    Two commodities with two sub-commodities each, 12 products per sub-commodity.
    In sub-commodities "A1" and "B1" products 0-3 enter the mailer once, in weeks
    12, 15, 18 and 21. log(1 + units) = product effect + week effect + noise, plus
    `PROMO_EFFECT` for a product in its mailer week.
    """
    rng = np.random.default_rng(7)
    weeks = np.arange(1, 31)
    week_effect = rng.normal(0, 0.2, len(weeks))
    rows = []
    product_id = 0
    for commodity in ("A", "B"):
        for sub in (f"{commodity}1", f"{commodity}2"):
            for index in range(12):
                product_id += 1
                level = rng.normal(2.5, 0.5)
                promo_week = 12 + 3 * index if sub.endswith("1") and index < 4 else None
                for week, effect in zip(weeks, week_effect, strict=True):
                    in_mailer = week == promo_week
                    log_units = level + effect + rng.normal(0, 0.05) + PROMO_EFFECT * in_mailer
                    rows.append(
                        {
                            "product_id": product_id,
                            "week_no": int(week),
                            "department": "GROCERY",
                            "commodity_desc": commodity,
                            "sub_commodity_desc": sub,
                            "units": float(np.expm1(log_units)),
                            "revenue": float(np.expm1(log_units)) * 2.0,
                            "mailer_store_share": 1.0 if in_mailer else 0.0,
                            "display_store_share": 0.0,
                        }
                    )
    return pd.DataFrame(rows)
