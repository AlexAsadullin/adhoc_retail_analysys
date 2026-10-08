import pandas as pd
import pytest

from retail_analysis import quality


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "household_key": [1, 2, None],
            "basket_id": [10, 11, 12],
            "product_id": [100, 101, 999],
            "week_no": [1, 2, 103],
            "day": [1, 8, 715],
        }
    )


@pytest.fixture
def products():
    return pd.DataFrame(
        {
            "product_id": [100, 101, 102],
            "department": ["GROCERY", None, None],
        }
    )


def test_null_check_flags_missing_values(make_con, transactions):
    con = make_con(transaction_data=transactions)
    failed = quality.check_no_nulls(con, "transaction_data", ["household_key", "basket_id"])
    passed = quality.check_no_nulls(con, "transaction_data", ["basket_id"])
    assert not failed.passed
    assert "household_key" in failed.details
    assert passed.passed


def test_products_missing_from_reference(make_con, transactions, products):
    con = make_con(transaction_data=transactions, product=products)
    result = quality.check_products_in_reference(con)
    assert not result.passed
    assert result.details.startswith("1 ")


def test_hierarchy_check_is_critical_only_for_sold_products(make_con, transactions, products):
    con = make_con(transaction_data=transactions, product=products)
    sold = quality.check_product_hierarchy(con, ["department"], sold_only=True)
    reference = quality.check_product_hierarchy(con, ["department"], sold_only=False)
    assert not sold.passed and sold.critical
    assert sold.details.startswith("1 ")
    assert not reference.passed and not reference.critical
    assert reference.details.startswith("2 ")


def test_week_range(make_con, transactions):
    con = make_con(transaction_data=transactions)
    assert not quality.check_week_range(con, "transaction_data", 1, 102).passed
    assert quality.check_week_range(con, "transaction_data", 1, 103).passed


def test_raise_on_critical_failure():
    results = [
        quality.CheckResult("ok", True, True, ""),
        quality.CheckResult("warning", False, False, ""),
    ]
    quality.raise_on_critical(results)
    with pytest.raises(quality.DataQualityError, match="broken"):
        quality.raise_on_critical([*results, quality.CheckResult("broken", False, True, "x")])


def test_analysis_window_cuts_ramp_up_and_incomplete_weeks():
    weekly = pd.DataFrame(
        {
            "week_no": range(1, 9),
            "active_households": [10, 50, 95, 100, 90, 105, 100, 40],
            "days": [5, 7, 7, 7, 7, 7, 7, 3],
        }
    )
    window = quality.find_analysis_window(weekly, stable_activity_ratio=0.9, full_week_days=7)
    # Median is 92.5, threshold 83.25: week 2 is the last week below it before week 7.
    assert (window.start_week, window.end_week) == (3, 7)
    assert window.active_households_threshold == pytest.approx(83.25)
