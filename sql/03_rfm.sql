-- Question 2: which customer segments exist?
-- Returns one row per household active in the analysis window with recency (weeks since
-- the last purchase to the window end), frequency (baskets), monetary (revenue) and
-- quintile scores 1-5, where 5 is the best value. Ties are broken by the last purchase
-- day and household key so that scores are deterministic.
WITH household_activity AS (
    SELECT
        household_key,
        $end_week - MAX(week_no) AS recency_weeks,
        MAX(day) AS last_purchase_day,
        COUNT(DISTINCT basket_id) AS frequency,
        SUM(sales_value) AS monetary
    FROM transaction_data
    WHERE week_no BETWEEN $start_week AND $end_week
    GROUP BY household_key
)

SELECT
    household_key,
    recency_weeks,
    frequency,
    monetary,
    monetary / frequency AS avg_basket_value,
    NTILE($buckets) OVER (
        ORDER BY recency_weeks DESC, last_purchase_day, household_key
    ) AS r_score,
    NTILE($buckets) OVER (ORDER BY frequency, household_key) AS f_score,
    NTILE($buckets) OVER (ORDER BY monetary, household_key) AS m_score
FROM household_activity
ORDER BY household_key;
