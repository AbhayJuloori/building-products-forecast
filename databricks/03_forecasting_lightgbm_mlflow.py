# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — LightGBM forecasting + Databricks-managed MLflow
# MAGIC
# MAGIC Trains the global LightGBM demand model on the Spark feature table
# MAGIC from notebook 02 and tracks the run with Databricks-managed MLflow.
# MAGIC The local pipeline uses file-backed MLflow at `mlruns/`; in
# MAGIC Databricks the same `mlflow` API logs to the workspace tracking
# MAGIC server — no code change required.
# MAGIC
# MAGIC **Why single-node LightGBM (not distributed):** at ~200K rows the
# MAGIC overhead of distributing training (SynapseML LightGBMRegressor) is
# MAGIC larger than the gain. The pattern is: Spark for the feature
# MAGIC transform, pandas + LightGBM for the model fit, MLflow Registry
# MAGIC for the model artifact. This is the standard "single-node training
# MAGIC on a feature table" pattern documented in the Databricks ML guide.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.6.0
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
import pandas as pd

CATALOG = "workspace"
SCHEMA = "building_products"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Databricks auto-configures the tracking URI; experiment defaults to notebook path
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/demand_forecasting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the modeling table from Delta + walk-forward split

# COMMAND ----------

df = (spark.table(f"{CATALOG}.{SCHEMA}.modeling_table_spark")
        .toPandas()
        .sort_values(["branch_id", "sku_id", "week_start_date"])
        .reset_index(drop=True))

print(f"Loaded modeling table: {len(df):,} rows × {len(df.columns)} cols")

weeks = np.sort(df.week_start_date.unique())
train_cut = weeks[104]
val_cut = weeks[130]
test_start = weeks[-26]

train = df[df.week_start_date < train_cut]
val = df[(df.week_start_date >= train_cut) & (df.week_start_date < val_cut)]
test = df[df.week_start_date >= test_start]

print(f"Train weeks: {df[df.week_start_date < train_cut].week_start_date.nunique()}")
print(f"Val weeks: {val.week_start_date.nunique()}")
print(f"Test weeks: {test.week_start_date.nunique()}")
print(f"Train rows: {len(train):,}, Val rows: {len(val):,}, Test rows: {len(test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection — drop identifiers + target

# COMMAND ----------

DROP_COLS = {"week_start_date", "units_sold", "revenue", "branch_id_y"}
TARGET = "units_sold"
CAT_COLS = ["category", "subcategory", "region", "climate_zone", "density", "branch_id", "sku_id"]

feature_cols = [c for c in df.columns if c not in DROP_COLS and c != TARGET]
# coerce categorical for LightGBM
for c in CAT_COLS:
    if c in train.columns:
        train[c] = train[c].astype("category")
        val[c] = val[c].astype("category")
        test[c] = test[c].astype("category")

print(f"Features: {len(feature_cols)} ({len([c for c in feature_cols if c in CAT_COLS])} categorical)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train + track with MLflow

# COMMAND ----------

params = dict(
    objective="regression",
    metric="rmse",
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=50,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    lambda_l2=0.1,
    n_estimators=500,
    n_jobs=-1,
    seed=42,
    verbose=-1,
)

# stockout down-weighting
sample_weight = np.where(train.stockout_flag, 0.3, 1.0)

with mlflow.start_run(run_name="lightgbm_global_databricks"):
    mlflow.log_params(params)
    mlflow.set_tag("split_strategy", "walk_forward")
    mlflow.set_tag("target", TARGET)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[feature_cols],
        train[TARGET],
        sample_weight=sample_weight,
        eval_set=[(val[feature_cols], val[TARGET])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=[c for c in CAT_COLS if c in feature_cols],
    )

    y_pred = model.predict(test[feature_cols])
    y_true = test[TARGET].values
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.abs(y_true - y_pred).mean()
    mape = (np.abs(y_true - y_pred) / np.maximum(y_true, 1)).mean()
    bias = (y_pred - y_true).mean()

    mlflow.log_metric("test_rmse", float(rmse))
    mlflow.log_metric("test_mae", float(mae))
    mlflow.log_metric("test_mape", float(mape))
    mlflow.log_metric("test_bias", float(bias))

    # MLflow signature + sample for the Model Registry
    sample = test[feature_cols].head(3)
    mlflow.lightgbm.log_model(model, "model", input_example=sample)
    run_id = mlflow.active_run().info.run_id

print(f"Run id: {run_id}")
print(f"Test RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.3f} | Bias: {bias:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare to the local model (sanity check)
# MAGIC
# MAGIC The local LightGBM averaged ~3.6 RMSE on a 26-week test horizon.
# MAGIC The Databricks run should land in the same ballpark — different
# MAGIC random seed paths through Optuna would shift it but not by orders
# MAGIC of magnitude. If they diverge significantly, the feature table on
# MAGIC Databricks doesn't match the local table.

# COMMAND ----------

# Compare against in-repo forecast_evaluation
local_eval = spark.table(f"{CATALOG}.{SCHEMA}.forecast_evaluation").toPandas()
local_lgb = local_eval[local_eval.model == "lightgbm_global"]
print(f"Local LightGBM avg test RMSE: {local_lgb.rmse.mean():.3f}")
print(f"Databricks LightGBM test RMSE: {rmse:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save SKU-level predictions back to Delta

# COMMAND ----------

predictions = pd.DataFrame({
    "model": "lightgbm_global_databricks",
    "branch_id": test["branch_id"].astype(str).values,
    "sku_id": test["sku_id"].astype(str).values,
    "week_start_date": test["week_start_date"].values,
    "y_true": y_true,
    "y_pred": y_pred,
})

(spark.createDataFrame(predictions)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.forecasts_databricks_lgb"))

print("Wrote forecasts_databricks_lgb")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interview talking points
# MAGIC
# MAGIC 1. **Databricks-managed MLflow gives me the registry for free.**
# MAGIC    Same `mlflow.lightgbm.log_model(...)` API as locally, but the
# MAGIC    model artifact is centralized and discoverable across users.
# MAGIC    Promoting a model to Production becomes a stage transition
# MAGIC    (None → Staging → Production), not a deployment script.
# MAGIC
# MAGIC 2. **Stockout weighting carries over.** The local insight (truncated
# MAGIC    weeks should not train as ground truth) is one line of code that
# MAGIC    is identical in pandas and PySpark-prepared feature tables.
# MAGIC
# MAGIC 3. **Why this single-node-on-cluster pattern.** Distributed LightGBM
# MAGIC    (SynapseML) is the right tool when training data exceeds driver
# MAGIC    memory. At 200K rows this is single-node territory. The Spark
# MAGIC    cluster's role is the feature transform; the actual fit runs
# MAGIC    on the driver. Documented as the standard pattern in the
# MAGIC    Databricks ML guide.
# MAGIC
# MAGIC 4. **The `forecasts_databricks_lgb` Delta table is now a stable
# MAGIC    contract** — Power BI / Dash can point at it for inference results,
# MAGIC    and the time-travel feature lets downstream consumers pin to a
# MAGIC    specific model version's outputs.
