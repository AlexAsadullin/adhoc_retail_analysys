-- Question 3: how well are customers retained?
-- Returns active households per cohort and number of periods since the first purchase.
-- A cohort is the period of the household's first purchase; a period is $period_weeks
-- consecutive weeks counted from week 1. Only periods that end by $end_week are kept.
WITH household_weeks AS (
    SELECT DISTINCT
        household_key,
        week_no
    FROM transaction_data
    WHERE week_no <= $end_week
),

with_first_week AS (
    SELECT
        household_key,
        week_no,
        MIN(week_no) OVER (PARTITION BY household_key) AS first_week
    FROM household_weeks
),

periods AS (
    SELECT
        household_key,
        (first_week - 1) // $period_weeks AS cohort_period,
        (week_no - 1) // $period_weeks AS activity_period
    FROM with_first_week
)

SELECT
    cohort_period,
    activity_period - cohort_period AS periods_since_first,
    COUNT(DISTINCT household_key) AS active_households
FROM periods
WHERE activity_period < $end_week // $period_weeks
GROUP BY
    cohort_period,
    activity_period
ORDER BY
    cohort_period,
    periods_since_first;
