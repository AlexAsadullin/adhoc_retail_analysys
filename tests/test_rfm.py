import pytest

from retail_analysis import rfm
from retail_analysis.queries import run_query


@pytest.fixture
def scores(make_con, sql_dir, rfm_transactions):
    con = make_con(transaction_data=rfm_transactions)
    return run_query(con, sql_dir, "03_rfm", {"start_week": 1, "end_week": 10, "buckets": 5})


def test_rfm_metrics(scores):
    household_5 = scores.set_index("household_key").loc[5]
    assert household_5["recency_weeks"] == 0
    assert household_5["frequency"] == 5
    assert household_5["monetary"] == pytest.approx(250.0)


def test_rfm_scores_follow_quintiles(scores):
    for column in ("r_score", "f_score", "m_score"):
        assert scores[column].tolist() == scores["household_key"].tolist()


def test_segment_assignment(config):
    rules = config.rfm.segments
    assert rfm.assign_segment(5, 5, 5, rules) == "Чемпионы"
    assert rfm.assign_segment(3, 4, 1, rules) == "Лояльные"
    assert rfm.assign_segment(5, 1, 3, rules) == "Перспективные"
    assert rfm.assign_segment(1, 5, 5, rules) == "Под угрозой ухода"
    assert rfm.assign_segment(1, 1, 1, rules) == "Спящие"
    assert rfm.assign_segment(3, 1, 2, rules) == "Середняки"


def test_unmatched_scores_raise(config):
    with pytest.raises(ValueError):
        rfm.assign_segment(9, 9, 9, config.rfm.segments[:1])


def test_segment_summary_shares_sum_to_one(scores, config):
    summary = rfm.segment_summary(
        rfm.assign_segments(scores, config.rfm.segments), config.rfm.segments
    )
    assert summary["household_share"].sum() == pytest.approx(1.0)
    assert summary["revenue_share"].sum() == pytest.approx(1.0)
    assert summary["households"].sum() == 5


def test_top_customers_revenue_share(scores):
    # The top 20% of five households is household 5 with 250 of 550 total revenue.
    assert rfm.top_customers_revenue_share(scores["monetary"], 0.2) == pytest.approx(250 / 550)
