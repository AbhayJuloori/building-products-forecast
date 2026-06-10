"""Train demand forecasting models and evaluate test-period forecasts."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "building_products_forecast_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR / "xdg"))

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from prophet import Prophet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "modeling_table.parquet"
FORECASTS_OUTPUT_PATH = PROCESSED_DATA_DIR / "forecasts_test.parquet"
EVALUATION_OUTPUT_PATH = PROCESSED_DATA_DIR / "forecast_evaluation.parquet"

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "demand_forecasting"

IDENTIFIER_COLUMNS = ["branch_id", "sku_id", "week_start_date"]
TARGET_COLUMN = "units_sold"
MODEL_COLUMN = "model"
Y_TRUE_COLUMN = "y_true"
Y_PRED_COLUMN = "y_pred"

CATEGORICAL_FEATURES = ["category", "subcategory", "region", "climate_zone", "density", "velocity_tier"]
PROPHET_REGRESSORS = ["housing_starts", "building_permits", "temp_deviation_from_annual_avg"]
LGBM_EXCLUDED_COLUMNS = {
    *IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    "target_log",
    "yoy_growth",
    "stockout_flag",
    "days_since_last_stockout",
    "feature_set_version",
}

SEASONAL_NAIVE_MODEL = "seasonal_naive"
LIGHTGBM_MODEL = "lightgbm_global"
PROPHET_MODEL = "prophet_branch_category"
CROSTON_MODEL = "croston_tsb_slowmovers"

TRAIN_WEEK_COUNT = 104
VALIDATION_WEEK_COUNT = 26
TEST_WEEK_CAP = 26
OPTUNA_TRIALS = 20
RANDOM_SEED = 42

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    """Date boundaries for the walk-forward model split."""

    train_weeks: pd.DatetimeIndex
    validation_weeks: pd.DatetimeIndex
    unused_weeks: pd.DatetimeIndex
    test_weeks: pd.DatetimeIndex


def _configure_logging() -> None:
    """Configure console logging for direct module execution."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _require(condition: bool, message: str) -> None:
    """Raise a clear validation error when an expected invariant is missing."""
    if not condition:
        raise ValueError(message)


def _date_range_label(weeks: pd.DatetimeIndex) -> str:
    """Return a compact date-range label for a non-empty DatetimeIndex."""
    if weeks.empty:
        return "<empty>"
    return f"{weeks.min().date()} to {weeks.max().date()}"


def _set_common_mlflow_tags() -> None:
    """Set tags common to all model runs."""
    mlflow.set_tag("split_strategy", "walk_forward")
    mlflow.set_tag("target", TARGET_COLUMN)


def _log_finite_metrics(metrics: dict[str, float]) -> None:
    """Log only finite MLflow metrics."""
    for name, value in metrics.items():
        if np.isfinite(value):
            mlflow.log_metric(name, float(value))


def load_modeling_table(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the modeling table used by all forecasters."""
    _require(path.exists(), f"Input modeling table not found: {path}")
    frame = pd.read_parquet(path)

    required_columns = {
        *IDENTIFIER_COLUMNS,
        TARGET_COLUMN,
        "stockout_flag",
        "category",
        *CATEGORICAL_FEATURES,
        *PROPHET_REGRESSORS,
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    _require(not missing_columns, f"Input table is missing required columns: {missing_columns}")

    frame = frame.copy()
    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    frame = frame.sort_values(IDENTIFIER_COLUMNS, kind="mergesort").reset_index(drop=True)

    duplicate_count = frame.duplicated(IDENTIFIER_COLUMNS).sum()
    _require(duplicate_count == 0, f"Input table has {duplicate_count:,} duplicate branch/SKU/week rows")
    LOGGER.info("Loaded modeling table from %s with shape %s", path, frame.shape)
    return frame


def compute_walk_forward_split(frame: pd.DataFrame) -> WalkForwardSplit:
    """Compute train, validation, unused, and final test weeks from ranked dates."""
    unique_weeks = pd.DatetimeIndex(pd.to_datetime(frame["week_start_date"].drop_duplicates()).sort_values())
    _require(len(unique_weeks) > TRAIN_WEEK_COUNT + VALIDATION_WEEK_COUNT, "Not enough weeks for train/validation/test split")

    train_weeks = unique_weeks[:TRAIN_WEEK_COUNT]
    validation_weeks = unique_weeks[TRAIN_WEEK_COUNT : TRAIN_WEEK_COUNT + VALIDATION_WEEK_COUNT]
    remaining_weeks = unique_weeks[TRAIN_WEEK_COUNT + VALIDATION_WEEK_COUNT :]
    test_weeks = remaining_weeks[-TEST_WEEK_CAP:]
    unused_weeks = remaining_weeks[:-TEST_WEEK_CAP]

    _require(not validation_weeks.empty, "Validation split is empty")
    _require(not test_weeks.empty, "Test split is empty")

    LOGGER.info("Walk-forward split uses %d total weeks", len(unique_weeks))
    LOGGER.info("Train weeks (%d): %s", len(train_weeks), _date_range_label(train_weeks))
    LOGGER.info("Validation weeks (%d): %s", len(validation_weeks), _date_range_label(validation_weeks))
    LOGGER.info("Unused gap weeks (%d): %s", len(unused_weeks), _date_range_label(unused_weeks))
    LOGGER.info("Test weeks (%d): %s", len(test_weeks), _date_range_label(test_weeks))
    return WalkForwardSplit(
        train_weeks=train_weeks,
        validation_weeks=validation_weeks,
        unused_weeks=unused_weeks,
        test_weeks=test_weeks,
    )


def _rows_for_weeks(frame: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.Series:
    """Return a boolean row mask for the provided week_start_date values."""
    return frame["week_start_date"].isin(weeks)


def _prediction_frame(model: str, source: pd.DataFrame, y_pred: np.ndarray | pd.Series) -> pd.DataFrame:
    """Build the standard long-format prediction frame for a model."""
    predictions = source[IDENTIFIER_COLUMNS + [TARGET_COLUMN]].copy()
    predictions.insert(0, MODEL_COLUMN, model)
    predictions = predictions.rename(columns={TARGET_COLUMN: Y_TRUE_COLUMN})
    predictions[Y_PRED_COLUMN] = np.asarray(y_pred, dtype="float64")
    return predictions[[MODEL_COLUMN, *IDENTIFIER_COLUMNS, Y_TRUE_COLUMN, Y_PRED_COLUMN]]


def seasonal_naive_forecast(frame: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    """Forecast test demand with same-week-prior-year sales."""
    lag_column = "sales_lag_52w" if "sales_lag_52w" in frame.columns else "lag_52w"
    _require(lag_column in frame.columns, "Seasonal naive requires sales_lag_52w or lag_52w")

    test_frame = frame.loc[_rows_for_weeks(frame, split.test_weeks)].copy()
    predictions = _prediction_frame(SEASONAL_NAIVE_MODEL, test_frame, test_frame[lag_column].to_numpy())

    with mlflow.start_run(run_name=SEASONAL_NAIVE_MODEL):
        _set_common_mlflow_tags()
        mlflow.log_param("lag_column", lag_column)
        _log_finite_metrics(summary_metrics_for_predictions(predictions))
    LOGGER.info("Created %s forecasts with %d rows", SEASONAL_NAIVE_MODEL, len(predictions))
    return predictions


def _lightgbm_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select LightGBM numeric and categorical feature columns without target leakage."""
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in frame.columns]
    numeric_features: list[str] = []

    for column in frame.columns:
        if column in LGBM_EXCLUDED_COLUMNS or column in categorical_features:
            continue
        dtype = frame[column].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric_features.append(column)

    feature_columns = [*numeric_features, *categorical_features]
    _require(feature_columns, "LightGBM feature set is empty")
    return feature_columns, categorical_features


def _category_levels_from_train(train_frame: pd.DataFrame, categorical_features: list[str]) -> dict[str, pd.Index]:
    """Capture categorical levels from train only so validation/test preprocessing does not peek ahead."""
    levels: dict[str, pd.Index] = {}
    for column in categorical_features:
        levels[column] = pd.Categorical(train_frame[column]).categories
    return levels


def _prepare_lightgbm_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    category_levels: dict[str, pd.Index],
) -> pd.DataFrame:
    """Return a LightGBM-ready feature matrix with native categorical columns."""
    matrix = frame[feature_columns].copy()
    for column in categorical_features:
        matrix[column] = pd.Categorical(matrix[column], categories=category_levels[column])
    for column in matrix.select_dtypes(include="bool").columns:
        matrix[column] = matrix[column].astype("int8")
    return matrix


def _lightgbm_params(trial: optuna.Trial | None = None) -> dict[str, Any]:
    """Return LightGBM parameters, sampling tunable values when a trial is provided."""
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_freq": 1,
        "feature_pre_filter": False,
        "n_jobs": -1,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
        "data_random_seed": RANDOM_SEED,
    }
    if trial is None:
        return params

    params.update(
        {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 10.0, log=True),
        }
    )
    return params


def train_lightgbm_with_optuna(frame: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    """Tune and train a global LightGBM regressor, then forecast the test period."""
    train_frame = frame.loc[_rows_for_weeks(frame, split.train_weeks)].copy()
    validation_frame = frame.loc[_rows_for_weeks(frame, split.validation_weeks)].copy()
    test_frame = frame.loc[_rows_for_weeks(frame, split.test_weeks)].copy()

    feature_columns, categorical_features = _lightgbm_feature_columns(train_frame)
    category_levels = _category_levels_from_train(train_frame, categorical_features)
    category_indices = [feature_columns.index(column) for column in categorical_features]

    x_train = _prepare_lightgbm_matrix(train_frame, feature_columns, categorical_features, category_levels)
    x_validation = _prepare_lightgbm_matrix(validation_frame, feature_columns, categorical_features, category_levels)
    x_test = _prepare_lightgbm_matrix(test_frame, feature_columns, categorical_features, category_levels)

    train_weights = np.where(train_frame["stockout_flag"].astype(bool).to_numpy(), 0.3, 1.0)
    train_data = lgb.Dataset(
        x_train,
        label=train_frame[TARGET_COLUMN].to_numpy(dtype="float64"),
        weight=train_weights,
        categorical_feature=category_indices,
        free_raw_data=False,
    )
    validation_data = lgb.Dataset(
        x_validation,
        label=validation_frame[TARGET_COLUMN].to_numpy(dtype="float64"),
        categorical_feature=category_indices,
        reference=train_data,
        free_raw_data=False,
    )

    LOGGER.info(
        "Training %s with %d features (%d categorical) and %d Optuna trials",
        LIGHTGBM_MODEL,
        len(feature_columns),
        len(categorical_features),
        OPTUNA_TRIALS,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _lightgbm_params(trial)
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params({**{key: value for key, value in params.items() if key != "metric"}, "n_estimators": 500})
            booster = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[validation_data],
                valid_names=["validation"],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)],
            )
            validation_rmse = float(booster.best_score["validation"]["rmse"])
            mlflow.log_metric("validation_rmse", validation_rmse)
            mlflow.log_metric("best_iteration", float(booster.best_iteration or booster.current_iteration()))
            return validation_rmse

    with mlflow.start_run(run_name=LIGHTGBM_MODEL):
        _set_common_mlflow_tags()
        mlflow.log_params(
            {
                "n_estimators": 500,
                "early_stopping_rounds": 30,
                "optuna_trials": OPTUNA_TRIALS,
                "stockout_weight": 0.3,
                "feature_count": len(feature_columns),
                "categorical_feature_count": len(categorical_features),
            }
        )
        mlflow.log_text("\n".join(feature_columns), "feature_columns.txt")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best_params = {**_lightgbm_params(), **study.best_params}
        best_booster = lgb.train(
            best_params,
            train_data,
            num_boost_round=500,
            valid_sets=[validation_data],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)],
        )

        mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
        mlflow.log_metric("best_validation_rmse", float(study.best_value))
        mlflow.log_metric("best_iteration", float(best_booster.best_iteration or best_booster.current_iteration()))
        mlflow.lightgbm.log_model(best_booster, artifact_path="model")

        predictions = _prediction_frame(
            LIGHTGBM_MODEL,
            test_frame,
            best_booster.predict(x_test, num_iteration=best_booster.best_iteration),
        )
        _log_finite_metrics(summary_metrics_for_predictions(predictions))

    LOGGER.info("Created %s forecasts with %d rows", LIGHTGBM_MODEL, len(predictions))
    return predictions


def _prophet_training_frame(group: pd.DataFrame) -> pd.DataFrame:
    """Return Prophet-formatted history for one branch/category series."""
    prophet_frame = group[["week_start_date", TARGET_COLUMN, *PROPHET_REGRESSORS]].copy()
    prophet_frame = prophet_frame.rename(columns={"week_start_date": "ds", TARGET_COLUMN: "y"})
    return prophet_frame.sort_values("ds", kind="mergesort").reset_index(drop=True)


def _safe_run_token(value: object) -> str:
    """Return a compact MLflow run-name token."""
    return str(value).replace("/", "_").replace(" ", "_")


def _fit_prophet(model: Prophet, prophet_frame: pd.DataFrame) -> None:
    """Fit Prophet while suppressing cmdstan progress output where supported."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model.fit(prophet_frame)


def _branch_category_metrics(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Return aggregate test metrics for one Prophet branch/category model."""
    metrics = _metrics_from_arrays(actual.to_numpy(dtype="float64"), forecast.to_numpy(dtype="float64"))
    return {
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_mape": metrics["mape"],
        "test_bias": metrics["bias"],
    }


def _trailing_sku_shares(frame: pd.DataFrame, split: WalkForwardSplit, trailing_weeks: int = 26) -> pd.DataFrame:
    """Compute SKU shares from the final trailing weeks before the test period."""
    test_start = split.test_weeks.min()
    all_weeks = pd.DatetimeIndex(pd.to_datetime(frame["week_start_date"].drop_duplicates()).sort_values())
    lookback_weeks = all_weeks[all_weeks < test_start][-trailing_weeks:]
    _require(not lookback_weeks.empty, "Cannot compute trailing SKU shares without pre-test history")

    test_skus = frame.loc[_rows_for_weeks(frame, split.test_weeks), ["branch_id", "category", "sku_id"]].drop_duplicates()
    trailing_units = (
        frame.loc[_rows_for_weeks(frame, lookback_weeks)]
        .groupby(["branch_id", "category", "sku_id"], as_index=False, observed=True)[TARGET_COLUMN]
        .sum()
        .rename(columns={TARGET_COLUMN: "trailing_units"})
    )

    shares = test_skus.merge(trailing_units, on=["branch_id", "category", "sku_id"], how="left", validate="one_to_one")
    shares["trailing_units"] = shares["trailing_units"].fillna(0.0)

    group_keys = ["branch_id", "category"]
    total_units = shares.groupby(group_keys, observed=True)["trailing_units"].transform("sum")
    sku_counts = shares.groupby(group_keys, observed=True)["sku_id"].transform("nunique")
    shares["sku_share"] = np.where(total_units.gt(0), shares["trailing_units"] / total_units, 1.0 / sku_counts)

    LOGGER.info(
        "Computed Prophet SKU allocation shares from %d trailing weeks: %s",
        len(lookback_weeks),
        _date_range_label(lookback_weeks),
    )
    return shares[["branch_id", "category", "sku_id", "sku_share"]]


def train_prophet_models(frame: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    """Fit Prophet per branch/category and allocate test forecasts back to SKUs."""
    train_mask = _rows_for_weeks(frame, split.train_weeks)
    test_mask = _rows_for_weeks(frame, split.test_weeks)
    aggregate_columns = ["branch_id", "category", "week_start_date"]

    aggregate = (
        frame.groupby(aggregate_columns, as_index=False, observed=True)
        .agg(
            units_sold=(TARGET_COLUMN, "sum"),
            housing_starts=("housing_starts", "first"),
            building_permits=("building_permits", "first"),
            temp_deviation_from_annual_avg=("temp_deviation_from_annual_avg", "first"),
        )
        .sort_values(aggregate_columns, kind="mergesort")
    )

    aggregate_predictions: list[pd.DataFrame] = []
    group_count = aggregate[["branch_id", "category"]].drop_duplicates().shape[0]
    LOGGER.info("Training %d Prophet branch/category models without parallelism", group_count)

    with mlflow.start_run(run_name=PROPHET_MODEL):
        _set_common_mlflow_tags()
        mlflow.log_param("branch_category_model_count", group_count)
        mlflow.log_param("weekly_seasonality", True)
        mlflow.log_param("yearly_seasonality", True)
        mlflow.log_param("regressors", ",".join(PROPHET_REGRESSORS))

        for (branch_id, category), group in aggregate.groupby(["branch_id", "category"], observed=True, sort=True):
            branch_id_str = str(branch_id)
            category_str = str(category)
            run_name = f"prophet_{_safe_run_token(branch_id_str)}_{_safe_run_token(category_str)}"
            train_group = group.loc[group["week_start_date"].isin(split.train_weeks)].copy()
            test_group = group.loc[group["week_start_date"].isin(split.test_weeks)].copy()
            _require(not train_group.empty, f"Prophet train data is empty for {branch_id_str}/{category_str}")
            _require(not test_group.empty, f"Prophet test data is empty for {branch_id_str}/{category_str}")

            with mlflow.start_run(run_name=run_name, nested=True):
                _set_common_mlflow_tags()
                mlflow.set_tag("branch_id", branch_id_str)
                mlflow.set_tag("category", category_str)
                mlflow.log_param("train_weeks", len(train_group))
                mlflow.log_param("test_weeks", len(test_group))

                prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
                for regressor in PROPHET_REGRESSORS:
                    prophet_model.add_regressor(regressor)

                prophet_history = _prophet_training_frame(train_group)
                _fit_prophet(prophet_model, prophet_history)

                future = (
                    test_group[["week_start_date", *PROPHET_REGRESSORS]]
                    .rename(columns={"week_start_date": "ds"})
                    .sort_values("ds", kind="mergesort")
                    .reset_index(drop=True)
                )
                forecast = prophet_model.predict(future)
                group_predictions = test_group[["branch_id", "category", "week_start_date", TARGET_COLUMN]].copy()
                group_predictions["branch_category_yhat"] = forecast["yhat"].to_numpy(dtype="float64")
                aggregate_predictions.append(group_predictions)

                _log_finite_metrics(
                    _branch_category_metrics(group_predictions[TARGET_COLUMN], group_predictions["branch_category_yhat"])
                )

        aggregate_forecasts = pd.concat(aggregate_predictions, ignore_index=True)
        shares = _trailing_sku_shares(frame, split)
        test_sku_frame = frame.loc[test_mask, ["branch_id", "category", "sku_id", "week_start_date", TARGET_COLUMN]].copy()
        sku_forecasts = test_sku_frame.merge(
            aggregate_forecasts[["branch_id", "category", "week_start_date", "branch_category_yhat"]],
            on=["branch_id", "category", "week_start_date"],
            how="left",
            validate="many_to_one",
        ).merge(shares, on=["branch_id", "category", "sku_id"], how="left", validate="many_to_one")
        sku_forecasts["sku_share"] = sku_forecasts["sku_share"].fillna(0.0)
        sku_forecasts[Y_PRED_COLUMN] = sku_forecasts["branch_category_yhat"] * sku_forecasts["sku_share"]

        predictions = _prediction_frame(PROPHET_MODEL, sku_forecasts, sku_forecasts[Y_PRED_COLUMN])
        _log_finite_metrics(summary_metrics_for_predictions(predictions))

    LOGGER.info("Created %s forecasts with %d rows", PROPHET_MODEL, len(predictions))
    return predictions


def identify_slow_moving_skus(train_frame: pd.DataFrame, zero_threshold: float = 0.60) -> pd.Index:
    """Identify SKUs with more than the requested share of zero-sales train rows."""
    zero_rate = train_frame.groupby("sku_id", observed=True)[TARGET_COLUMN].apply(lambda values: values.eq(0).mean())
    return pd.Index(zero_rate[zero_rate.gt(zero_threshold)].index)


def croston_tsb(demand: np.ndarray, alpha: float = 0.10, beta: float = 0.05) -> float:
    """Return a Croston-TSB forecast from historical demand."""
    demand_values = np.asarray(demand, dtype="float64")
    if demand_values.size == 0:
        return math.nan

    positive_demand = demand_values[demand_values > 0]
    if positive_demand.size == 0:
        return 0.0

    z = float(positive_demand[0])
    p = float(np.mean(demand_values > 0))
    for value in demand_values:
        occurrence = 1.0 if value > 0 else 0.0
        if occurrence:
            z = alpha * float(value) + (1.0 - alpha) * z
        p = beta * occurrence + (1.0 - beta) * p
    return z * p


def train_croston_tsb_slowmovers(frame: pd.DataFrame, split: WalkForwardSplit) -> pd.DataFrame:
    """Forecast train-identified slow-moving SKUs with Croston-TSB."""
    train_frame = frame.loc[_rows_for_weeks(frame, split.train_weeks)].copy()
    test_frame = frame.loc[_rows_for_weeks(frame, split.test_weeks)].copy()
    slow_skus = identify_slow_moving_skus(train_frame)
    scalar_forecasts: list[dict[str, Any]] = []

    for (branch_id, sku_id), group in train_frame.groupby(["branch_id", "sku_id"], observed=True, sort=True):
        if sku_id not in slow_skus:
            continue
        demand = group.sort_values("week_start_date", kind="mergesort")[TARGET_COLUMN].to_numpy(dtype="float64")
        scalar_forecasts.append({"branch_id": branch_id, "sku_id": sku_id, Y_PRED_COLUMN: croston_tsb(demand)})

    scalar_frame = pd.DataFrame(scalar_forecasts, columns=["branch_id", "sku_id", Y_PRED_COLUMN])
    scored = test_frame[IDENTIFIER_COLUMNS + [TARGET_COLUMN]].merge(
        scalar_frame,
        on=["branch_id", "sku_id"],
        how="left",
        validate="many_to_one",
    )

    predictions = _prediction_frame(CROSTON_MODEL, scored, scored[Y_PRED_COLUMN])
    with mlflow.start_run(run_name=CROSTON_MODEL):
        _set_common_mlflow_tags()
        mlflow.log_param("alpha", 0.10)
        mlflow.log_param("beta", 0.05)
        mlflow.log_param("zero_threshold", 0.60)
        mlflow.log_metric("slow_sku_count", float(len(slow_skus)))
        mlflow.log_metric("slow_branch_sku_count", float(len(scalar_frame)))
        _log_finite_metrics(summary_metrics_for_predictions(predictions))

    LOGGER.info(
        "Created %s forecasts with %d rows; %d slow SKUs across %d branch/SKU series",
        CROSTON_MODEL,
        len(predictions),
        len(slow_skus),
        len(scalar_frame),
    )
    return predictions


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute requested metrics for aligned arrays, ignoring missing predictions."""
    y_true_values = np.asarray(y_true, dtype="float64")
    y_pred_values = np.asarray(y_pred, dtype="float64")
    valid = np.isfinite(y_true_values) & np.isfinite(y_pred_values)
    if not valid.any():
        return {"rmse": math.nan, "mae": math.nan, "mape": math.nan, "bias": math.nan, "n_test_weeks": 0.0}

    actual = y_true_values[valid]
    forecast = y_pred_values[valid]
    error = forecast - actual
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / np.maximum(actual, 1.0))),
        "bias": float(np.mean(error)),
        "n_test_weeks": float(valid.sum()),
    }


def summary_metrics_for_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Return average test metrics across branch/SKU series for MLflow logging."""
    records: list[dict[str, float]] = []
    for _, group in predictions.groupby(["branch_id", "sku_id"], observed=True, sort=False):
        records.append(_metrics_from_arrays(group[Y_TRUE_COLUMN].to_numpy(), group[Y_PRED_COLUMN].to_numpy()))

    metric_frame = pd.DataFrame(records)
    return {
        "avg_test_rmse": float(metric_frame["rmse"].mean(skipna=True)),
        "avg_test_mape": float(metric_frame["mape"].mean(skipna=True)),
        "avg_test_bias": float(metric_frame["bias"].mean(skipna=True)),
    }


def evaluate_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-branch, per-SKU metrics and FVA against seasonal naive."""
    metric_records: list[dict[str, Any]] = []
    group_columns = [MODEL_COLUMN, "branch_id", "sku_id"]

    for (model, branch_id, sku_id), group in forecasts.groupby(group_columns, observed=True, sort=False):
        metrics = _metrics_from_arrays(group[Y_TRUE_COLUMN].to_numpy(), group[Y_PRED_COLUMN].to_numpy())
        metric_records.append(
            {
                MODEL_COLUMN: model,
                "branch_id": branch_id,
                "sku_id": sku_id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "bias": metrics["bias"],
                "n_test_weeks": int(metrics["n_test_weeks"]),
            }
        )

    evaluation = pd.DataFrame(metric_records)
    naive_rmse = (
        evaluation.loc[evaluation[MODEL_COLUMN].eq(SEASONAL_NAIVE_MODEL), ["branch_id", "sku_id", "rmse"]]
        .rename(columns={"rmse": "naive_rmse"})
        .copy()
    )
    evaluation = evaluation.merge(naive_rmse, on=["branch_id", "sku_id"], how="left", validate="many_to_one")
    evaluation["forecast_value_add_vs_naive"] = np.where(
        evaluation["naive_rmse"].gt(0),
        (evaluation["naive_rmse"] - evaluation["rmse"]) / evaluation["naive_rmse"],
        np.nan,
    )
    evaluation.loc[
        evaluation[MODEL_COLUMN].eq(SEASONAL_NAIVE_MODEL) & evaluation["naive_rmse"].eq(0),
        "forecast_value_add_vs_naive",
    ] = 0.0

    output_columns = [
        MODEL_COLUMN,
        "branch_id",
        "sku_id",
        "rmse",
        "mae",
        "mape",
        "bias",
        "forecast_value_add_vs_naive",
        "n_test_weeks",
    ]
    return evaluation[output_columns].sort_values(group_columns, kind="mergesort").reset_index(drop=True)


def log_final_summary(evaluation: pd.DataFrame) -> None:
    """Log the final per-model metric summary."""
    summary = (
        evaluation.groupby(MODEL_COLUMN, as_index=False, observed=True)
        .agg(
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
            avg_mape=("mape", "mean"),
            avg_bias=("bias", "mean"),
            avg_fva_vs_naive=("forecast_value_add_vs_naive", "mean"),
            evaluated_skus=("n_test_weeks", lambda values: int((values > 0).sum())),
        )
        .sort_values(MODEL_COLUMN, kind="mergesort")
    )
    LOGGER.info("Final test summary by model:\n%s", summary.to_string(index=False))


def main() -> None:
    """Train all requested models, persist forecasts, and write evaluation metrics."""
    _configure_logging()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    frame = load_modeling_table(INPUT_PATH)
    split = compute_walk_forward_split(frame)

    forecasts = pd.concat(
        [
            seasonal_naive_forecast(frame, split),
            train_lightgbm_with_optuna(frame, split),
            train_prophet_models(frame, split),
            train_croston_tsb_slowmovers(frame, split),
        ],
        ignore_index=True,
    )
    forecasts = forecasts.sort_values([MODEL_COLUMN, "branch_id", "sku_id", "week_start_date"], kind="mergesort")

    evaluation = evaluate_forecasts(forecasts)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECASTS_OUTPUT_PATH, index=False)
    evaluation.to_parquet(EVALUATION_OUTPUT_PATH, index=False)

    LOGGER.info("Wrote test forecasts to %s with shape %s", FORECASTS_OUTPUT_PATH, forecasts.shape)
    LOGGER.info("Wrote forecast evaluation to %s with shape %s", EVALUATION_OUTPUT_PATH, evaluation.shape)
    log_final_summary(evaluation)


if __name__ == "__main__":
    main()
