import numpy as np
import pandas as pd
import pytest

from retail_analysis import cohorts
from retail_analysis.queries import run_query

PERIOD_WEEKS = 2
END_WEEK = 6


@pytest.fixture
def activity(make_con, sql_dir, cohort_transactions):
    con = make_con(transaction_data=cohort_transactions)
    return run_query(
        con, sql_dir, "04_cohorts", {"end_week": END_WEEK, "period_weeks": PERIOD_WEEKS}
    )


def test_last_full_period():
    assert cohorts.last_full_period(END_WEEK, PERIOD_WEEKS) == 2
    assert cohorts.last_full_period(101, 4) == 24


def test_retention_matrix_matches_hand_calculation(activity):
    last_period = cohorts.last_full_period(END_WEEK, PERIOD_WEEKS)
    matrix = cohorts.retention_matrix(activity, min_cohort_size=1, last_period=last_period)
    expected = pd.DataFrame(
        [[1.0, 1 / 3, 2 / 3], [1.0, 0.5, np.nan]],
        index=pd.Index([0, 1], name="cohort_period"),
        columns=pd.Index([0, 1, 2], name="periods_since_first"),
    )
    pd.testing.assert_frame_equal(matrix, expected, check_dtype=False)


def test_small_cohorts_are_dropped(activity):
    matrix = cohorts.retention_matrix(activity, min_cohort_size=3, last_period=2)
    assert matrix.index.tolist() == [0]


def test_average_retention_is_weighted_by_cohort_size(activity):
    matrix = cohorts.retention_matrix(activity, min_cohort_size=1, last_period=2)
    average = cohorts.average_retention(matrix, cohorts.cohort_sizes(activity))
    assert average[0] == pytest.approx(1.0)
    assert average[1] == pytest.approx((1 + 1) / (3 + 2))
    assert average[2] == pytest.approx(2 / 3)


def test_cohort_label():
    assert cohorts.cohort_label(0, 4) == "нед. 1-4"
    assert cohorts.cohort_label(4, 4) == "нед. 17-20"
