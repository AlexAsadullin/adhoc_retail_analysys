-- Question 1: which categories drive revenue?
-- Returns the top departments and commodities by revenue in the analysis window
-- with their revenue share and rank within each level.
WITH line_items AS (
    SELECT
        p.department,
        p.commodity_desc,
        t.sales_value
    FROM transaction_data AS t
    INNER JOIN product AS p
        ON t.product_id = p.product_id
    WHERE t.week_no BETWEEN $start_week AND $end_week
),

category_revenue AS (
    SELECT
        'department' AS category_level,
        department AS category,
        SUM(sales_value) AS revenue
    FROM line_items
    GROUP BY department

    UNION ALL

    SELECT
        'commodity' AS category_level,
        commodity_desc AS category,
        SUM(sales_value) AS revenue
    FROM line_items
    GROUP BY commodity_desc
),

ranked AS (
    SELECT
        category_level,
        category,
        revenue,
        revenue / SUM(revenue) OVER (PARTITION BY category_level) AS revenue_share,
        ROW_NUMBER() OVER (PARTITION BY category_level ORDER BY revenue DESC) AS revenue_rank
    FROM category_revenue
)

SELECT
    category_level,
    category,
    revenue,
    revenue_share,
    revenue_rank
FROM ranked
WHERE revenue_rank <= $top_n
ORDER BY
    category_level,
    revenue_rank;
