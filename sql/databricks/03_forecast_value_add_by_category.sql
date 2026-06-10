-- sql/databricks/03_forecast_value_add_by_category.sql
-- Purpose: Compare LightGBM against Seasonal Naive by category to quantify forecast value add.
-- Expected runtime: < 10 seconds on the project Delta tables.

WITH model_rmse_by_sku AS (
    SELECT
        p.category,
        fe.sku_id,
        AVG(CASE WHEN fe.model = 'seasonal_naive' THEN fe.rmse END) AS naive_rmse,
        AVG(CASE WHEN fe.model = 'lightgbm_global' THEN fe.rmse END) AS lgb_rmse
    FROM building_products.forecast_evaluation AS fe
    INNER JOIN building_products.products AS p
        ON fe.sku_id = p.sku_id
    WHERE fe.model IN ('seasonal_naive', 'lightgbm_global')
    GROUP BY p.category, fe.sku_id
),
category_rollup AS (
    SELECT
        category,
        COUNT(DISTINCT sku_id) AS n_skus,
        AVG(naive_rmse) AS avg_naive_rmse,
        AVG(lgb_rmse) AS avg_lgb_rmse
    FROM model_rmse_by_sku
    WHERE naive_rmse IS NOT NULL
      AND lgb_rmse IS NOT NULL
    GROUP BY category
)
SELECT
    category,
    n_skus,
    ROUND(avg_naive_rmse, 2) AS avg_naive_rmse,
    ROUND(avg_lgb_rmse, 2) AS avg_lgb_rmse,
    ROUND(
        100.0 * (avg_naive_rmse - avg_lgb_rmse)
        / NULLIF(avg_naive_rmse, 0),
        2
    ) AS avg_fva_pct
FROM category_rollup
ORDER BY avg_fva_pct DESC;
