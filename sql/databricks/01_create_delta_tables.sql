-- sql/databricks/01_create_delta_tables.sql
-- Purpose: Register the Building Products project Delta tables in the building_products schema.
-- Expected runtime: < 1 minute when Delta locations already exist.

CREATE SCHEMA IF NOT EXISTS building_products;

CREATE TABLE IF NOT EXISTS building_products.branches (
    branch_id STRING,
    name STRING,
    region STRING,
    climate_zone STRING,
    density STRING
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/branches';

CREATE TABLE IF NOT EXISTS building_products.products (
    sku_id STRING,
    name STRING,
    category STRING,
    subcategory STRING,
    unit_cost DOUBLE,
    lead_time_days BIGINT,
    weight_class STRING,
    seasonality_profile ARRAY<DOUBLE>,
    is_slow_mover BOOLEAN
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/products';

CREATE TABLE IF NOT EXISTS building_products.sales_history (
    week_start_date TIMESTAMP,
    branch_id STRING,
    sku_id STRING,
    units_sold BIGINT,
    revenue DOUBLE,
    stockout_flag BOOLEAN,
    sales_year INT GENERATED ALWAYS AS (YEAR(week_start_date))
)
USING DELTA
PARTITIONED BY (sales_year)
LOCATION 'dbfs:/FileStore/building_products/delta/sales_history';

CREATE TABLE IF NOT EXISTS building_products.inventory_snapshots (
    snapshot_date TIMESTAMP,
    branch_id STRING,
    sku_id STRING,
    on_hand_units BIGINT,
    reorder_point BIGINT,
    lead_time_days BIGINT,
    snapshot_year INT GENERATED ALWAYS AS (YEAR(snapshot_date))
)
USING DELTA
PARTITIONED BY (snapshot_year)
LOCATION 'dbfs:/FileStore/building_products/delta/inventory_snapshots';

CREATE TABLE IF NOT EXISTS building_products.contractors (
    contractor_id STRING,
    branch_id STRING,
    name STRING,
    trade_type STRING,
    annual_spend_tier STRING,
    account_age_years BIGINT
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/contractors';

CREATE TABLE IF NOT EXISTS building_products.forecasts_test (
    model STRING,
    branch_id STRING,
    sku_id STRING,
    week_start_date TIMESTAMP,
    y_true BIGINT,
    y_pred DOUBLE
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/forecasts_test';

CREATE TABLE IF NOT EXISTS building_products.forecast_evaluation (
    model STRING,
    branch_id STRING,
    sku_id STRING,
    rmse DOUBLE,
    mae DOUBLE,
    mape DOUBLE,
    bias DOUBLE,
    forecast_value_add_vs_naive DOUBLE,
    n_test_weeks BIGINT
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/forecast_evaluation';

CREATE TABLE IF NOT EXISTS building_products.inventory_scenarios (
    branch_id STRING,
    sku_id STRING,
    scenario STRING,
    service_level DOUBLE,
    lead_time_days DOUBLE,
    avg_weekly_demand DOUBLE,
    demand_std DOUBLE,
    safety_stock_units DOUBLE,
    reorder_point_units DOUBLE,
    eoq_units DOUBLE,
    safety_stock_cost DOUBLE,
    on_hand_units DOUBLE,
    current_excess_units DOUBLE,
    excess_inventory_cost DOUBLE,
    projected_stockout_risk_pct DOUBLE,
    scenario_total_inventory_cost DOUBLE
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/inventory_scenarios';

CREATE TABLE IF NOT EXISTS building_products.sku_abc_xyz (
    sku_id STRING,
    category STRING,
    total_revenue DOUBLE,
    revenue_share DOUBLE,
    abc_class STRING,
    demand_cv DOUBLE,
    xyz_class STRING
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/sku_abc_xyz';

CREATE TABLE IF NOT EXISTS building_products.contractor_segments (
    contractor_id STRING,
    branch_id STRING,
    total_spend DOUBLE,
    order_frequency DOUBLE,
    avg_order_size DOUBLE,
    product_breadth BIGINT,
    recency_days BIGINT,
    cluster_id BIGINT,
    segment_label STRING
)
USING DELTA
LOCATION 'dbfs:/FileStore/building_products/delta/contractor_segments';
