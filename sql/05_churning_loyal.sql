-- Separate finding: who are the churning loyal households?
-- Loyal: top $loyal_top_share of households by revenue in the first half of the window
-- (weeks $start_week-$mid_week). Churning: baskets in the last $recent_weeks weeks fell by
-- more than $activity_drop versus the household's own average for a block of the same
-- length in the earlier weeks of the window.
-- Returns one row per household and department with household-level metrics repeated,
-- so that one result covers group sizes, revenue, category mix and demographics.
WITH first_half AS (
    SELECT
        household_key,
        SUM(sales_value) AS first_half_revenue
    FROM transaction_data
    WHERE week_no BETWEEN $start_week AND $mid_week
    GROUP BY household_key
),

loyalty AS (
    SELECT
        household_key,
        first_half_revenue,
        CUME_DIST() OVER (ORDER BY first_half_revenue DESC) <= $loyal_top_share AS is_loyal
    FROM first_half
),

activity AS (
    SELECT
        household_key,
        SUM(sales_value) AS revenue,
        SUM(sales_value) FILTER (WHERE week_no <= $end_week - $recent_weeks) AS baseline_revenue,
        SUM(sales_value) FILTER (WHERE week_no > $end_week - $recent_weeks) AS recent_revenue,
        COUNT(DISTINCT basket_id) AS baskets,
        COUNT(DISTINCT basket_id) FILTER (WHERE week_no <= $end_week - $recent_weeks)
            * $recent_weeks / ($end_week - $recent_weeks - $start_week + 1) AS baseline_baskets,
        COUNT(DISTINCT basket_id) FILTER (WHERE week_no > $end_week - $recent_weeks)
            AS recent_baskets
    FROM transaction_data
    WHERE week_no BETWEEN $start_week AND $end_week
    GROUP BY household_key
),

households AS (
    SELECT
        a.household_key,
        CASE
            WHEN l.is_loyal AND a.recent_baskets < (1 - $activity_drop) * a.baseline_baskets
                THEN 'churning_loyal'
            WHEN l.is_loyal THEN 'stable_loyal'
            ELSE 'other'
        END AS household_group,
        COALESCE(l.first_half_revenue, 0) AS first_half_revenue,
        a.revenue,
        COALESCE(a.baseline_revenue, 0) AS baseline_revenue,
        COALESCE(a.recent_revenue, 0) AS recent_revenue,
        a.baskets,
        a.baseline_baskets,
        a.recent_baskets,
        a.revenue / a.baskets AS avg_basket_value
    FROM activity AS a
    LEFT JOIN loyalty AS l
        ON a.household_key = l.household_key
),

department_revenue AS (
    SELECT
        t.household_key,
        p.department,
        SUM(t.sales_value) AS department_revenue
    FROM transaction_data AS t
    INNER JOIN product AS p
        ON t.product_id = p.product_id
    WHERE t.week_no BETWEEN $start_week AND $end_week - $recent_weeks
    GROUP BY
        t.household_key,
        p.department
)

SELECT
    h.household_key,
    h.household_group,
    h.first_half_revenue,
    h.revenue,
    h.baseline_revenue,
    h.recent_revenue,
    h.baskets,
    h.baseline_baskets,
    h.recent_baskets,
    h.avg_basket_value,
    d.classification_1,
    d.classification_2,
    d.classification_3,
    d.classification_4,
    d.classification_5,
    d.homeowner_desc,
    d.kid_category_desc,
    dr.department,
    dr.department_revenue
FROM households AS h
LEFT JOIN hh_demographic AS d
    ON h.household_key = d.household_key
LEFT JOIN department_revenue AS dr
    ON h.household_key = dr.household_key
ORDER BY
    h.household_key,
    dr.department;
