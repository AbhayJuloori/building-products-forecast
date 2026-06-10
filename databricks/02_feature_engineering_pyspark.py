# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature engineering in PySpark
# MAGIC
# MAGIC Port of `src/features/build_features.py` to PySpark using
# MAGIC Window functions over `(branch_id, sku_id)` partitions ordered by
# MAGIC `week_start_date`. Same lag/rolling logic that runs locally on pandas,
# MAGIC but parallelized across cluster cores via partition shuffle.
# MAGIC
# MAGIC **Why this is the right tool:** at 250K rows the pandas version
# MAGIC runs in seconds, but at 100M+ rows (a real branch network) the
# MAGIC `groupby.shift` path serializes through a single CPU. The PySpark
# MAGIC Window over a partitioned key parallelizes the same logic across
# MAGIC the cluster — same code shape, fundamentally different scale ceiling.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
SCHEMA = "building_products"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load source tables

# COMMAND ----------

sales = spark.table(f"{CATALOG}.{SCHEMA}.sales_history")
products = spark.table(f"{CATALOG}.{SCHEMA}.products").select(
    "sku_id", "category", "subcategory", "unit_cost", "lead_time_days", "is_slow_mover"
)
branches = spark.table(f"{CATALOG}.{SCHEMA}.branches").select(
    "branch_id", "region", "climate_zone", "density"
)
external = spark.table(f"{CATALOG}.{SCHEMA}.external_weekly")

print(f"sales: {sales.count():,} rows")
print(f"products: {products.count()} rows")
print(f"branches: {branches.count()} rows")
print(f"external: {external.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Lag features via Window
# MAGIC
# MAGIC `lag(col, n)` reads the value from `n` rows back within the partition,
# MAGIC ordered by `week_start_date`. The partition key `(branch_id, sku_id)`
# MAGIC isolates lags per individual series. Shifting by `n=1` and beyond
# MAGIC ensures the target week never sees its own value (anti-leakage).

# COMMAND ----------

w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date")

features = (sales
    .withColumn("sales_lag_1w", F.lag("units_sold", 1).over(w))
    .withColumn("lag_2w", F.lag("units_sold", 2).over(w))
    .withColumn("lag_4w", F.lag("units_sold", 4).over(w))
    .withColumn("lag_8w", F.lag("units_sold", 8).over(w))
    .withColumn("lag_12w", F.lag("units_sold", 12).over(w))
    .withColumn("lag_52w", F.lag("units_sold", 52).over(w))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Rolling means/std (anti-leakage windows)
# MAGIC
# MAGIC `rowsBetween(-window_size, -1)` takes the trailing `window_size` rows
# MAGIC ending the row BEFORE the current — so the rolling stat never includes
# MAGIC the target week. Equivalent to pandas `.shift(1).rolling(window=N)`.

# COMMAND ----------

roll_4w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date").rowsBetween(-4, -1)
roll_13w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date").rowsBetween(-13, -1)
roll_26w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date").rowsBetween(-26, -1)

features = (features
    .withColumn("rolling_mean_4w", F.avg("units_sold").over(roll_4w))
    .withColumn("rolling_mean_13w", F.avg("units_sold").over(roll_13w))
    .withColumn("rolling_mean_26w", F.avg("units_sold").over(roll_26w))
    .withColumn("rolling_std_4w", F.stddev("units_sold").over(roll_4w))
    .withColumn("rolling_std_13w", F.stddev("units_sold").over(roll_13w))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — YoY growth + calendar

# COMMAND ----------

features = (features
    .withColumn("yoy_growth", F.col("units_sold") / F.col("lag_52w") - 1)
    .withColumn("week_of_year", F.weekofyear("week_start_date"))
    .withColumn("month", F.month("week_start_date"))
    .withColumn("quarter", F.quarter("week_start_date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Days since last stockout (cumulative)

# COMMAND ----------

# Pyspark: count weeks since last True in stockout_flag
stockout_w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date")
# Mark week index relative to stockouts: last week with stockout per row
features = (features
    .withColumn("_week_idx", F.row_number().over(stockout_w))
    .withColumn("_stockout_week_idx", F.when(F.col("stockout_flag"), F.col("_week_idx")))
    .withColumn(
        "_last_stockout_idx",
        F.last("_stockout_week_idx", ignorenulls=True).over(
            Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date").rowsBetween(Window.unboundedPreceding, 0)
        ),
    )
    .withColumn("days_since_last_stockout", (F.col("_week_idx") - F.col("_last_stockout_idx")) * 7)
    .drop("_week_idx", "_stockout_week_idx", "_last_stockout_idx")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Broadcast joins to dim tables + external signals
# MAGIC
# MAGIC `branches` (12 rows) and `products` (80 rows) are tiny — broadcast
# MAGIC joins avoid the shuffle and run in seconds. Spark's
# MAGIC `autoBroadcastJoinThreshold` (10MB default) auto-broadcasts these,
# MAGIC but explicit `F.broadcast(...)` makes the optimizer choice obvious
# MAGIC at read time.

# COMMAND ----------

features = (features
    .join(F.broadcast(products), "sku_id", "left")
    .join(F.broadcast(branches), "branch_id", "left")
    .join(external, ["week_start_date", "climate_zone"], "left")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Drop warmup rows (no lag_52w) and write Delta

# COMMAND ----------

features_clean = features.filter(F.col("lag_52w").isNotNull())

print(f"Final modeling table: {features_clean.count():,} rows × {len(features_clean.columns)} cols")

(features_clean.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.modeling_table_spark"))

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        category,
        COUNT(*) AS row_count,
        ROUND(AVG(rolling_mean_13w), 2) AS avg_13w_demand,
        ROUND(AVG(yoy_growth), 3) AS avg_yoy_growth
    FROM {CATALOG}.{SCHEMA}.modeling_table_spark
    WHERE category IS NOT NULL
    GROUP BY category
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interview talking points
# MAGIC
# MAGIC 1. **Window functions are the Spark idiom for lag/rolling features.**
# MAGIC    The `partitionBy(...).orderBy(...).rowsBetween(...)` triplet maps
# MAGIC    directly to pandas `groupby.shift().rolling()` — same semantics,
# MAGIC    different execution model. Each partition runs on a single
# MAGIC    executor core; partitions execute in parallel.
# MAGIC
# MAGIC 2. **Anti-leakage is enforced by the window spec.**
# MAGIC    `rowsBetween(-N, -1)` is exclusive of the current row — equivalent
# MAGIC    to `.shift(1).rolling(N)` in pandas. The target week never appears
# MAGIC    in its own features.
# MAGIC
# MAGIC 3. **Why this scales.** At 250K rows the pandas version runs in
# MAGIC    seconds; this notebook runs in similar time. But at 100M rows
# MAGIC    (a real distributor network), pandas serializes through one core
# MAGIC    while Spark distributes across the cluster. Same business logic,
# MAGIC    different scale ceiling.
# MAGIC
# MAGIC 4. **Delta gives me schema enforcement** between weekly runs and
# MAGIC    **time travel** so I can backtest forecasts against historical
# MAGIC    feature sets without rebuilding.
