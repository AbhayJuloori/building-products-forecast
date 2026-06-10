-- sql/local/01_top_branches_by_forecast.sql
-- Purpose: Rank branches by best-model forecasted units over the first 13 test weeks and compare to actual sales.
-- Expected runtime: < 5 seconds on the project parquet data.

WITH forecast_evaluation AS (
    SELECT *
    FROM read_parquet('data/processed/forecast_evaluation.parquet')
),
forecasts_test AS (
    SELECT *
    FROM read_parquet('data/processed/forecasts_test.parquet')
),
branches AS (
    SELECT *
    FROM read_parquet('data/raw/branches.parquet')
),
sales_history AS (
    SELECT *
    FROM read_parquet('data/raw/sales_history.parquet')
),
best_model_scores AS (
    SELECT
        sku_id,
        model,
        AVG(rmse) AS avg_rmse,
        ROW_NUMBER() OVER (
            PARTITION BY sku_id
            ORDER BY AVG(rmse), model
        ) AS model_rank
    FROM forecast_evaluation
    GROUP BY sku_id, model
),
best_model_per_sku AS (
    SELECT sku_id, model
    FROM best_model_scores
    WHERE model_rank = 1
),
test_horizon AS (
    SELECT
        MIN(week_start_date) AS first_test_week,
        MIN(week_start_date) + INTERVAL '13 weeks' AS horizon_end_week
    FROM forecasts_test
),
best_forecasts AS (
    SELECT
        f.branch_id,
        f.sku_id,
        f.week_start_date,
        f.y_true,
        f.y_pred
    FROM forecasts_test AS f
    INNER JOIN best_model_per_sku AS bm
        ON f.sku_id = bm.sku_id
       AND f.model = bm.model
    CROSS JOIN test_horizon AS h
    WHERE f.week_start_date >= h.first_test_week
      AND f.week_start_date < h.horizon_end_week
),
branch_rollup AS (
    SELECT
        bf.branch_id,
        SUM(bf.y_pred) AS total_forecast_units,
        SUM(COALESCE(sh.units_sold, bf.y_true, 0)) AS total_actual_units
    FROM best_forecasts AS bf
    LEFT JOIN sales_history AS sh
        ON bf.branch_id = sh.branch_id
       AND bf.sku_id = sh.sku_id
       AND bf.week_start_date = sh.week_start_date
    GROUP BY bf.branch_id
)
SELECT
    br.branch_id,
    b.name AS branch_name,
    ROUND(br.total_forecast_units, 2) AS total_forecast_units,
    br.total_actual_units,
    ROUND(
        100.0 * (
            1.0 - ABS(br.total_forecast_units - br.total_actual_units)
                  / NULLIF(br.total_actual_units, 0)
        ),
        2
    ) AS forecast_accuracy_pct
FROM branch_rollup AS br
INNER JOIN branches AS b
    ON br.branch_id = b.branch_id
ORDER BY total_forecast_units DESC
LIMIT 10;
