# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Register Delta tables from Unity Catalog Volume
# MAGIC
# MAGIC Project: **Building Products Demand Forecasting & Inventory Optimization**
# MAGIC
# MAGIC This notebook reads the parquet files uploaded to
# MAGIC `/Volumes/workspace/building_products/project_data/` and registers
# MAGIC them as managed Delta tables in `workspace.building_products`. After
# MAGIC this notebook runs successfully, all downstream notebooks can
# MAGIC reference `workspace.building_products.<table_name>` directly.
# MAGIC
# MAGIC **Why Delta over parquet:**
# MAGIC - ACID transactions for safe re-runs
# MAGIC - Time travel — backtest forecasts against historical Delta versions
# MAGIC - Schema enforcement — catches accidental column drift between weekly runs
# MAGIC - Native Z-ORDER for predicate pushdown on `branch_id, sku_id` queries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Verify the schema exists

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "building_products"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/project_data"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA} COMMENT 'Demand forecasting portfolio project'")
spark.sql(f"USE SCHEMA {SCHEMA}")

display(spark.sql(f"DESCRIBE SCHEMA {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Register raw layer (5 tables)

# COMMAND ----------

RAW_TABLES = {
    "branches": "raw/branches.parquet",
    "products": "raw/products.parquet",
    "sales_history": "raw/sales_history.parquet",
    "inventory_snapshots": "raw/inventory_snapshots.parquet",
    "contractors": "raw/contractors.parquet",
}

for table, rel_path in RAW_TABLES.items():
    src = f"{VOLUME_PATH}/{rel_path}"
    df = spark.read.parquet(src)
    (df.write.format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(f"{CATALOG}.{SCHEMA}.{table}"))
    print(f"Registered {CATALOG}.{SCHEMA}.{table} from {src} — {df.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Register processed + external layers

# COMMAND ----------

PROCESSED_TABLES = {
    "modeling_table": "processed/modeling_table.parquet",
    "forecasts_test": "processed/forecasts_test.parquet",
    "forecast_evaluation": "processed/forecast_evaluation.parquet",
    "sku_abc_xyz": "processed/sku_abc_xyz.parquet",
    "contractor_segments": "processed/contractor_segments.parquet",
    "branch_demand_clusters": "processed/branch_demand_clusters.parquet",
    "inventory_scenarios": "processed/inventory_scenarios.parquet",
    "slow_mover_flags": "processed/slow_mover_flags.parquet",
}

EXTERNAL_TABLES = {
    "external_weekly": "external/external_weekly.parquet",
    "housing_starts": "external/housing_starts.parquet",
    "building_permits": "external/building_permits.parquet",
    "weather": "external/weather.parquet",
}

for table, rel_path in {**PROCESSED_TABLES, **EXTERNAL_TABLES}.items():
    src = f"{VOLUME_PATH}/{rel_path}"
    try:
        df = spark.read.parquet(src)
        (df.write.format("delta")
           .mode("overwrite")
           .option("overwriteSchema", "true")
           .saveAsTable(f"{CATALOG}.{SCHEMA}.{table}"))
        print(f"Registered {CATALOG}.{SCHEMA}.{table} — {df.count():,} rows")
    except Exception as exc:
        print(f"SKIPPED {table}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Apply Z-ORDER on high-cardinality query columns
# MAGIC
# MAGIC Z-ORDER colocates files by `branch_id, sku_id` for the heavy sales
# MAGIC and forecast tables. This shrinks IO when the dashboard or downstream
# MAGIC analytics filter by branch/SKU.

# COMMAND ----------

ZORDER_TABLES = ["sales_history", "modeling_table", "forecasts_test", "inventory_snapshots"]
for t in ZORDER_TABLES:
    try:
        spark.sql(f"OPTIMIZE {CATALOG}.{SCHEMA}.{t} ZORDER BY (branch_id, sku_id)")
        print(f"Z-ordered {t}")
    except Exception as exc:
        print(f"Skipped Z-ORDER on {t}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Sanity check

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        category,
        COUNT(DISTINCT sku_id) AS skus,
        SUM(units_sold) AS total_units,
        ROUND(SUM(revenue), 0) AS total_revenue
    FROM {CATALOG}.{SCHEMA}.sales_history
    JOIN {CATALOG}.{SCHEMA}.products USING (sku_id)
    GROUP BY category
    ORDER BY total_revenue DESC
"""))
