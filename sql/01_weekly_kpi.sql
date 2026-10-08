-- Question 1: how does the business develop week to week?
-- Returns one row per week of the analysis window: revenue, baskets, active households,
-- average basket value and week-over-week revenue change.
WITH weekly AS (
    SELECT
        week_no,
        SUM(sales_value) AS revenue,
        COUNT(DISTINCT basket_id) AS baskets,
        COUNT(DISTINCT household_key) AS active_households
    FROM transaction_data
    WHERE week_no BETWEEN $start_week AND $end_week
    GROUP BY week_no
)

SELECT
    week_no,
    revenue,
    baskets,
    active_households,
    revenue / baskets AS avg_basket_value,
    revenue / LAG(revenue) OVER (ORDER BY week_no) - 1 AS revenue_wow_change
FROM weekly
ORDER BY week_no;
