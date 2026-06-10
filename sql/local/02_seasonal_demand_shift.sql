-- sql/local/02_seasonal_demand_shift.sql
-- Purpose: Measure quarter-over-quarter and year-over-year demand shifts by product category.
-- Expected runtime: < 5 seconds on the project parquet data.

WITH sales_history AS (
    SELECT *
    FROM read_parquet('data/raw/sales_history.parquet')
),
products AS (
    SELECT *
    FROM read_parquet('data/raw/products.parquet')
),
category_quarters AS (
    SELECT
        p.category,
        DATE_TRUNC('quarter', s.week_start_date) AS quarter_start,
        SUM(s.units_sold) AS total_units
    FROM sales_history AS s
    INNER JOIN products AS p
        ON s.sku_id = p.sku_id
    GROUP BY p.category, DATE_TRUNC('quarter', s.week_start_date)
),
with_lags AS (
    SELECT
        category,
        quarter_start,
        total_units,
        LAG(total_units) OVER (
            PARTITION BY category
            ORDER BY quarter_start
        ) AS prior_quarter_units,
        LAG(total_units, 4) OVER (
            PARTITION BY category
            ORDER BY quarter_start
        ) AS prior_year_units
    FROM category_quarters
)
SELECT
    category,
    CAST(EXTRACT(year FROM quarter_start) AS VARCHAR)
        || '-Q'
        || CAST(EXTRACT(quarter FROM quarter_start) AS VARCHAR) AS year_quarter,
    total_units,
    ROUND(
        100.0 * (total_units - prior_quarter_units)
        / NULLIF(prior_quarter_units, 0),
        2
    ) AS qoq_pct_change,
    ROUND(
        100.0 * (total_units - prior_year_units)
        / NULLIF(prior_year_units, 0),
        2
    ) AS yoy_pct_change
FROM with_lags
ORDER BY category, quarter_start;
