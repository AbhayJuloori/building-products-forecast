-- sql/local/04_slow_moving_inventory_risk.sql
-- Purpose: Identify branch-SKU inventory positions carrying more than 26 weeks of demand coverage.
-- Expected runtime: < 5 seconds on the project parquet data.

WITH sales_history AS (
    SELECT *
    FROM read_parquet('data/raw/sales_history.parquet')
),
inventory_snapshots AS (
    SELECT *
    FROM read_parquet('data/raw/inventory_snapshots.parquet')
),
products AS (
    SELECT *
    FROM read_parquet('data/raw/products.parquet')
),
sales_window AS (
    SELECT
        MAX(week_start_date) AS latest_sales_week,
        MAX(week_start_date) - INTERVAL '25 weeks' AS trailing_26_week_start
    FROM sales_history
),
trailing_demand AS (
    SELECT
        s.branch_id,
        s.sku_id,
        AVG(CAST(s.units_sold AS DOUBLE)) AS avg_weekly_demand
    FROM sales_history AS s
    CROSS JOIN sales_window AS w
    WHERE s.week_start_date >= w.trailing_26_week_start
      AND s.week_start_date <= w.latest_sales_week
    GROUP BY s.branch_id, s.sku_id
),
latest_inventory_ranked AS (
    SELECT
        branch_id,
        sku_id,
        snapshot_date,
        on_hand_units,
        ROW_NUMBER() OVER (
            PARTITION BY branch_id, sku_id
            ORDER BY snapshot_date DESC
        ) AS snapshot_rank
    FROM inventory_snapshots
),
latest_inventory AS (
    SELECT
        branch_id,
        sku_id,
        on_hand_units
    FROM latest_inventory_ranked
    WHERE snapshot_rank = 1
),
risk_positions AS (
    SELECT
        li.branch_id,
        li.sku_id,
        p.name AS sku_name,
        p.category,
        p.unit_cost,
        li.on_hand_units,
        td.avg_weekly_demand,
        li.on_hand_units - (26.0 * td.avg_weekly_demand) AS excess_units
    FROM latest_inventory AS li
    INNER JOIN trailing_demand AS td
        ON li.branch_id = td.branch_id
       AND li.sku_id = td.sku_id
    INNER JOIN products AS p
        ON li.sku_id = p.sku_id
    WHERE td.avg_weekly_demand > 0
      AND li.on_hand_units > 26.0 * td.avg_weekly_demand
)
SELECT
    sku_id,
    sku_name,
    category,
    branch_id,
    on_hand_units,
    ROUND(on_hand_units / NULLIF(avg_weekly_demand, 0), 1) AS est_weeks_coverage,
    ROUND(excess_units, 2) AS excess_units,
    ROUND(excess_units * unit_cost, 2) AS excess_cost_dollars
FROM risk_positions
ORDER BY excess_cost_dollars DESC;
