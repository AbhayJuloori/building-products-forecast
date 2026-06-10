# SQL Artifacts

This folder contains local DuckDB queries over parquet files and Databricks SQL queries over registered Delta tables for the Building Products Demand Forecasting project.

## Run Local Queries

Run from the project root:

```sh
duckdb -c ".read sql/local/01_top_branches_by_forecast.sql"
```

DuckDB usually autoloads parquet support. If your installation requires the extension explicitly, run:

```sh
duckdb -c "INSTALL parquet; LOAD parquet;"
duckdb -c ".read sql/local/01_top_branches_by_forecast.sql"
```

Replace the filename with any query under `sql/local/`.

## Run Databricks Queries

1. Upload or convert the parquet files into Delta tables at `dbfs:/FileStore/building_products/delta/<table_name>`.
2. Paste `sql/databricks/01_create_delta_tables.sql` into the Databricks SQL Editor and run it to register the tables under `building_products`.
3. Paste any other query under `sql/databricks/` into the SQL Editor and run it against the same workspace/catalog context.

## What Each Query Proves

- `local/01_top_branches_by_forecast.sql` and `databricks/02_branch_demand_summary.sql`: show which branches carry the largest near-term demand signal and whether best-model forecasts track actual sales at branch level.
- `local/02_seasonal_demand_shift.sql`: shows product categories with quarter-over-quarter and year-over-year demand shifts, validating that the data captures seasonality and category-level demand cycles.
- `local/03_inventory_scenario_comparison.sql`: compares inventory policy scenarios by safety stock dollars, reorder exposure, and service level, connecting forecasts to inventory decisions.
- `local/04_slow_moving_inventory_risk.sql`: identifies branch-SKU positions with more than 26 weeks of coverage, quantifying excess units and capital tied up in slow-moving stock.
- `databricks/03_forecast_value_add_by_category.sql`: compares LightGBM against Seasonal Naive by category, showing where the model adds measurable forecast value.
- `databricks/04_contractor_concentration_by_branch.sql`: estimates top-5 contractor concentration by branch using segmented contractor spend, highlighting account concentration risk.

`databricks/01_create_delta_tables.sql` also registers `contractor_segments` because the contractor concentration query depends on `total_spend` from that processed artifact.
