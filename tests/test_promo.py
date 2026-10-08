import pytest
from conftest import PROMO_EFFECT

from retail_analysis import promo
from retail_analysis.config import PromoParams

START_WEEK = 1
END_WEEK = 30


@pytest.fixture
def params():
    return PromoParams(
        pre_weeks=4,
        treated_min_store_share=0.5,
        min_pre_weeks_with_sales=4,
        excluded_departments=[],
        confidence_level=0.95,
    )


@pytest.fixture
def tables(product_weeks):
    return promo.to_wide(product_weeks)


@pytest.fixture
def panel(tables, params):
    events = promo.promo_events(tables, params, START_WEEK, END_WEEK)
    return promo.expand_to_panel(events, tables, params.pre_weeks)


def test_events_use_same_sub_commodity_controls(panel):
    # 4 promo weeks x 2 sub-commodities, each event with exactly one treated product
    # (5 panel rows) and controls only from the same sub-commodity.
    assert panel["event_id"].nunique() == 8
    assert (panel.groupby("event_id")["treated"].sum() == 5).all()
    assert panel["event_id"].str.split("|").str[0].isin(["A1", "B1"]).all()


def test_panel_is_balanced(panel, params):
    rows_per_unit = panel.groupby(["event_id", "product_id"]).size()
    assert (rows_per_unit == params.pre_weeks + 1).all()


def test_did_recovers_known_effect(panel):
    result = promo.estimate_did(panel, "units", 0.95)
    assert result.coefficient == pytest.approx(PROMO_EFFECT, abs=0.05)
    assert result.ci_low < PROMO_EFFECT < result.ci_high
    assert result.p_value < 0.01


def test_event_study_has_flat_pre_period(panel):
    table, pretrend_p = promo.estimate_event_study(panel, "units", 0.95)
    pre = table[table["rel_week"] < -1]
    assert pre["coefficient"].abs().max() < 0.05
    assert pretrend_p > 0.05
    promo_week = table.set_index("rel_week").loc[0, "coefficient"]
    assert promo_week == pytest.approx(PROMO_EFFECT, abs=0.05)


def test_no_cannibalization_when_siblings_unaffected(tables, params):
    events = promo.cannibalization_events(tables, params, START_WEEK, END_WEEK)
    panel = promo.expand_to_panel(events, tables, params.pre_weeks)
    result = promo.estimate_did(panel, "units", 0.95)
    assert result.ci_low < 0 < result.ci_high
    assert result.coefficient == pytest.approx(0, abs=0.05)


def test_two_way_demean_removes_fixed_effects(panel):
    demeaned = promo.two_way_demean(panel.assign(y=panel["product_id"] + panel["rel_week"]), ["y"])
    assert demeaned["y"].abs().max() == pytest.approx(0, abs=1e-9)
