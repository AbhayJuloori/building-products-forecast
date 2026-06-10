-- sql/local/03_inventory_scenario_comparison.sql
-- Purpose: Compare inventory policy scenarios by safety stock investment, reorder exposure, and service level.
-- Expected runtime: < 5 seconds on the project parquet data.

WITH inventory_scenarios AS (
    SELECT *
    FROM read_parquet('data/processed/inventory_scenarios.parquet')
),
inventory_snapshots AS (
    SELECT *
    FROM read_parquet('data/raw/inventory_snapshots.parquet')
),
latest_inventory_ranked AS (
    SELECT
        branch_id,
        sku_id,
        snapshot_date,
        on_hand_units,
        reorder_point,
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
        snapshot_date,
        on_hand_units,
        reorder_point
    FROM latest_inventory_ranked
    WHERE snapshot_rank = 1
),
scenario_inventory AS (
    SELECT
        s.scenario,
        s.branch_id,
        s.sku_id,
        s.service_level,
        s.safety_stock_units,
        s.reorder_point_units,
        s.safety_stock_cost,
        li.on_hand_units
    FROM inventory_scenarios AS s
    INNER JOIN latest_inventory AS li
        ON s.branch_id = li.branch_id
       AND s.sku_id = li.sku_id
    WHERE s.scenario IN ('A', 'B', 'C', 'D')
)
SELECT
    scenario,
    ROUND(SUM(safety_stock_cost), 2) AS total_safety_stock_investment,
    COUNT(*) FILTER (
        WHERE on_hand_units < reorder_point_units
    ) AS skus_needing_reorder,
    ROUND(AVG(service_level), 4) AS avg_service_level
FROM scenario_inventory
GROUP BY scenario
ORDER BY scenario;
