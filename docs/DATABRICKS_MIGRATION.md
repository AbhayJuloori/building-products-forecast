# Databricks Community Edition — Migration Guide

You built this locally. For interview authenticity, port two notebooks to
Databricks Community Edition (free, no credit card) so you can talk about
Databricks first-hand and show screenshots.

## Step 1 — Sign up Databricks Community Edition (~5 min)

1. Go to **https://www.databricks.com/try-databricks**
2. Click **Get started for free** → choose **Community Edition** at the bottom
   (skip the cloud-provider partner pages; CE is a separate "lite" workspace).
3. Verify email, sign in.
4. Confirm you land on a workspace URL like
   `community.cloud.databricks.com`.

**Limits to know (interview-grade answers):**
- Single 15 GB cluster, auto-terminates after 2 hours of idle
- No job scheduler, no MLflow Model Registry (tracking still works)
- One workspace, no teams/permissions
- Use this for personal projects + interview demos, not production

## Step 2 — Upload synthetic data to DBFS

In a local terminal, with the project's `data/raw/` populated:

```bash
cd ~/projects/building-products-forecast
# Install databricks-cli once
pip install databricks-cli

# Configure (one-time): paste workspace URL + personal access token from
# Databricks UI -> User Settings -> Developer -> Access tokens
databricks configure --token

# Upload the raw + processed parquet to DBFS
databricks fs cp -r data/raw dbfs:/FileStore/building_products/raw
databricks fs cp -r data/processed dbfs:/FileStore/building_products/processed
databricks fs ls dbfs:/FileStore/building_products/raw
```

Alternative (no CLI): in the Databricks UI go to **Data → Add Data → DBFS →
Upload File**, point to each parquet file. Slower but works.

## Step 3 — Create the two interview notebooks

In the workspace, create:

- `01_feature_engineering_databricks.ipynb`
- `02_forecasting_lightgbm_databricks.ipynb`

For each, **Compute → Create cluster** (use the free 11.3 LTS ML runtime
which already has lightgbm/mlflow installed). Attach the notebook to that
cluster.

### Notebook 1 — Feature engineering on Spark

Port `src/features/build_features.py` to PySpark. Key changes:

```python
# Cell 1 - Spark session is auto-provided as `spark`
from pyspark.sql import functions as F, Window

raw = spark.read.parquet("/FileStore/building_products/raw/sales_history.parquet")
products = spark.read.parquet("/FileStore/building_products/raw/products.parquet")
branches = spark.read.parquet("/FileStore/building_products/raw/branches.parquet")
external = spark.read.parquet("/FileStore/building_products/raw/external_weekly.parquet")

# Cell 2 - lag features via Window
w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date")
df = (raw
      .withColumn("sales_lag_1w", F.lag("units_sold", 1).over(w))
      .withColumn("sales_lag_4w", F.lag("units_sold", 4).over(w))
      .withColumn("sales_lag_52w", F.lag("units_sold", 52).over(w)))

# Cell 3 - rolling means via rangeBetween (weeks)
roll_w = Window.partitionBy("branch_id", "sku_id").orderBy("week_start_date").rowsBetween(-13, -1)
df = df.withColumn("rolling_mean_13w", F.avg("units_sold").over(roll_w))

# Cell 4 - join external + product/branch attrs
df = (df
      .join(products.select("sku_id", "category", "unit_cost", "is_slow_mover"), "sku_id")
      .join(branches.select("branch_id", "region", "climate_zone"), "branch_id")
      .join(external, ["week_start_date", "climate_zone"], "left"))

# Cell 5 - write back to DBFS
df.write.mode("overwrite").parquet("/FileStore/building_products/processed/modeling_table_spark.parquet")
```

**Interview talking points this notebook earns you:**
- "Window functions over (branch, SKU) ordered by week handle lag and rolling
  features without leakage — equivalent to `groupby.shift` in pandas but
  parallelizes across partitions."
- "I broadcast the small dimension tables (12 branches, 80 SKUs) to avoid
  shuffle on the join — Spark autoBroadcastJoinThreshold catches this for me
  since they're under the 10 MB default."
- "On CE the data is small enough to fit in driver memory, but the same code
  scales to billion-row datasets by adding more workers — `groupby.shift` in
  pandas does not."

### Notebook 2 — LightGBM training with MLflow tracking

Port the LightGBM part of `src/models/forecasting.py`:

```python
# Cell 1
import mlflow, lightgbm as lgb, pandas as pd
mlflow.set_experiment("/Users/<your-email>/building_products_demand_forecasting")

# Cell 2 - pull the Spark table back into pandas for LightGBM (single-node training)
df = spark.read.parquet("/FileStore/building_products/processed/modeling_table_spark.parquet").toPandas()

# Cell 3 - split walk-forward
train = df[df.week_start_date < "2023-06-19"]
val = df[(df.week_start_date >= "2023-06-19") & (df.week_start_date < "2024-06-24")]
test = df[df.week_start_date >= "2024-06-24"]

# Cell 4 - log run
with mlflow.start_run(run_name="lightgbm_global_databricks"):
    params = dict(num_leaves=63, learning_rate=0.05, min_child_samples=50, n_estimators=500)
    mlflow.log_params(params)
    sample_w = train.stockout_flag.map({True: 0.3, False: 1.0})
    model = lgb.LGBMRegressor(**params, n_jobs=-1)
    model.fit(train[FEATURE_COLS], train.units_sold, sample_weight=sample_w,
              eval_set=[(val[FEATURE_COLS], val.units_sold)],
              callbacks=[lgb.early_stopping(30)], categorical_feature=CAT_COLS)
    rmse = ((test.units_sold - model.predict(test[FEATURE_COLS]))**2).mean()**0.5
    mlflow.log_metric("test_rmse", rmse)
    mlflow.lightgbm.log_model(model, "model")
```

**Interview talking points:**
- "Databricks-managed MLflow gives me an experiment registry I can compare runs
  across, without standing up a server. Same `mlflow` Python API as locally — I
  set the tracking URI implicitly by being on a Databricks cluster."
- "I keep the heavy table in Spark for shuffle-friendly transforms, then collect
  the final modeling table to pandas for LightGBM because LightGBM's
  scalability strategy is different — distributed training would need
  `SynapseML`'s `LightGBMRegressor` or `dask-lightgbm`. For this dataset
  (~200 K rows × 34 features) single-node is faster than the overhead."

## Step 4 — Capture screenshots for the README

Capture these so reviewers can see the work was on Databricks:

1. The workspace homepage with the notebooks listed
2. The cluster details panel (runtime, memory)
3. Notebook 1 cell output showing the Spark DataFrame schema
4. Notebook 2 MLflow run page with metrics + parameters
5. The MLflow experiments page listing both runs

Put them in `docs/screenshots/` and reference in README.

## Step 5 — Update the README

Add this block to README.md under **Tech Stack**:

```markdown
### Databricks portion

Two notebooks ported to Databricks Community Edition:
- `databricks/01_feature_engineering_databricks.ipynb` — Spark Window-function
  port of `build_features.py`, demonstrating partitioned lag/rolling computation
  on the 250K-row sales history.
- `databricks/02_forecasting_lightgbm_databricks.ipynb` — LightGBM training
  with experiment tracking on Databricks-managed MLflow.

The full pipeline still runs locally end-to-end (`run_all.py`); the Databricks
notebooks demonstrate the same feature and modeling work at scale on a Spark
cluster.
```

## Step 6 — Export notebooks and commit

In each notebook: **File → Export → IPython Notebook (.ipynb)**, save to
`databricks/` in your repo, commit.

## What you can say in the interview

**Q: "Did you actually use Databricks?"**

> "Yes — the local pipeline is the reference implementation, and I ported the
> feature engineering and LightGBM training to Databricks Community Edition.
> The Spark Window-function port shows how the same lag/rolling logic
> parallelizes across partitions, and the MLflow runs are logged on the
> Databricks-managed tracking server. The screenshots in the repo show the
> notebooks and experiment runs."

**Q: "Why CE rather than full Databricks?"**

> "Cost — Community Edition is free and gives me the same notebook + Spark +
> MLflow experience for personal projects. The limitations are the auto-
> terminating cluster and no job scheduler, neither of which matter for a
> demo. At a real distributor with real volumes I'd use the standard tier so
> we get scheduled jobs, multi-user permissions, and Unity Catalog."

**Q: "Would you redo this on Databricks if you started over?"**

> "For a real branch network of this size — yes, Databricks would be the right
> stack day one. The argument is less about scale (200K rows isn't big) and more
> about the integration: Auto Loader picking up new branch sales weekly,
> Delta tables giving me schema enforcement and time travel for backtesting,
> Workflows scheduling the pipeline, and Unity Catalog tying lineage from raw
> sales to forecast outputs to the inventory recommendations. The local version
> is good for portfolio storytelling; the production version needs that
> end-to-end governance."
