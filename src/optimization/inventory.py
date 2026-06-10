"""Build inventory optimization scenarios from forecast and inventory inputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PRODUCTS_INPUT_PATH = RAW_DATA_DIR / "products.parquet"
BRANCHES_INPUT_PATH = RAW_DATA_DIR / "branches.parquet"
INVENTORY_INPUT_PATH = RAW_DATA_DIR / "inventory_snapshots.parquet"
FORECASTS_INPUT_PATH = PROCESSED_DATA_DIR / "forecasts_test.parquet"
FORECAST_EVALUATION_INPUT_PATH = PROCESSED_DATA_DIR / "forecast_evaluation.parquet"
SKU_SEGMENTS_INPUT_PATH = PROCESSED_DATA_DIR / "sku_abc_xyz.parquet"

INVENTORY_SCENARIOS_OUTPUT_PATH = PROCESSED_DATA_DIR / "inventory_scenarios.parquet"
SLOW_MOVER_FLAGS_OUTPUT_PATH = PROCESSED_DATA_DIR / "slow_mover_flags.parquet"

LIGHTGBM_MODEL = "lightgbm_global"
GROUP_KEYS = ["branch_id", "sku_id"]
HOLDING_COST_RATE_ANNUAL = 0.25
ORDER_COST_PER_ORDER = 50.0
STOCKOUT_COST_MULTIPLIER = 2.0
LEAD_TIME_VARIABILITY_FACTOR = 0.2
FORECAST_COVERAGE_EXCESS_WEEKS = 26.0
MAX_FORECAST_COVERAGE_WEEKS = 999.0

SCENARIO_SERVICE_LEVELS = {
    "A": 0.95,
    "B": 0.98,
    "C": 0.90,
}
SEGMENTED_SERVICE_LEVELS = {
    "A": 0.98,
    "B": 0.95,
    "C": 0.90,
}

INVENTORY_SCENARIO_COLUMNS = [
    "branch_id",
    "sku_id",
    "scenario",
    "service_level",
    "lead_time_days",
    "avg_weekly_demand",
    "demand_std",
    "safety_stock_units",
    "reorder_point_units",
    "eoq_units",
    "safety_stock_cost",
    "on_hand_units",
    "current_excess_units",
    "excess_inventory_cost",
    "projected_stockout_risk_pct",
    "scenario_total_inventory_cost",
]
SLOW_MOVER_FLAG_COLUMNS = [
    "sku_id",
    "branch_id",
    "on_hand_units",
    "weeks_of_forecast_coverage",
    "excess_units",
    "excess_cost_dollars",
    "flag_reason",
]

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging when the module is executed directly."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _require(condition: bool, message: str) -> None:
    """Raise a clear validation error when an expected invariant is not met."""
    if not condition:
        raise ValueError(message)


def _require_columns(frame: pd.DataFrame, required_columns: set[str], frame_name: str) -> None:
    """Validate that a frame contains the required columns."""
    missing_columns = sorted(required_columns.difference(frame.columns))
    _require(not missing_columns, f"{frame_name} is missing required columns: {missing_columns}")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a parquet artifact, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    LOGGER.info("Wrote %s with shape %s", path, frame.shape)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate all inputs required for inventory optimization."""
    for path in [
        PRODUCTS_INPUT_PATH,
        BRANCHES_INPUT_PATH,
        INVENTORY_INPUT_PATH,
        FORECASTS_INPUT_PATH,
        FORECAST_EVALUATION_INPUT_PATH,
        SKU_SEGMENTS_INPUT_PATH,
    ]:
        _require(path.exists(), f"Input file not found: {path}")

    products = pd.read_parquet(
        PRODUCTS_INPUT_PATH,
        columns=["sku_id", "category", "unit_cost", "lead_time_days", "is_slow_mover"],
    )
    branches = pd.read_parquet(BRANCHES_INPUT_PATH, columns=["branch_id"])
    inventory = pd.read_parquet(
        INVENTORY_INPUT_PATH,
        columns=["snapshot_date", "branch_id", "sku_id", "on_hand_units", "reorder_point", "lead_time_days"],
    )
    forecasts = pd.read_parquet(
        FORECASTS_INPUT_PATH,
        columns=["model", "branch_id", "sku_id", "week_start_date", "y_pred"],
    )
    evaluation = pd.read_parquet(
        FORECAST_EVALUATION_INPUT_PATH,
        columns=["model", "branch_id", "sku_id", "rmse"],
    )
    sku_segments = pd.read_parquet(
        SKU_SEGMENTS_INPUT_PATH,
        columns=["sku_id", "abc_class", "xyz_class", "demand_cv"],
    )

    _require_columns(products, {"sku_id", "category", "unit_cost", "lead_time_days", "is_slow_mover"}, "products")
    _require_columns(branches, {"branch_id"}, "branches")
    _require_columns(
        inventory,
        {"snapshot_date", "branch_id", "sku_id", "on_hand_units", "reorder_point", "lead_time_days"},
        "inventory_snapshots",
    )
    _require_columns(forecasts, {"model", "branch_id", "sku_id", "week_start_date", "y_pred"}, "forecasts_test")
    _require_columns(evaluation, {"model", "branch_id", "sku_id", "rmse"}, "forecast_evaluation")
    _require_columns(sku_segments, {"sku_id", "abc_class", "xyz_class", "demand_cv"}, "sku_abc_xyz")

    inventory = inventory.copy()
    inventory["snapshot_date"] = pd.to_datetime(inventory["snapshot_date"])
    forecasts = forecasts.copy()
    forecasts["week_start_date"] = pd.to_datetime(forecasts["week_start_date"])

    LOGGER.info(
        "Loaded inputs: products=%s branches=%s inventory=%s forecasts=%s evaluation=%s sku_segments=%s",
        products.shape,
        branches.shape,
        inventory.shape,
        forecasts.shape,
        evaluation.shape,
        sku_segments.shape,
    )
    return products, branches, inventory, forecasts, evaluation, sku_segments


def select_best_models(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-RMSE model for each branch/SKU, preferring LightGBM on ties."""
    duplicate_count = evaluation.duplicated(["model", *GROUP_KEYS]).sum()
    _require(duplicate_count == 0, f"forecast_evaluation has {duplicate_count:,} duplicate model/branch/SKU rows")

    ranked = evaluation.copy()
    ranked["rmse_rank"] = ranked["rmse"].fillna(np.inf)
    ranked["lightgbm_tie_rank"] = np.where(ranked["model"].eq(LIGHTGBM_MODEL), 0, 1)
    ranked = ranked.sort_values(
        [*GROUP_KEYS, "rmse_rank", "lightgbm_tie_rank", "model"],
        kind="mergesort",
    )
    best_models = ranked.groupby(GROUP_KEYS, as_index=False, observed=True).first()

    missing_rmse_count = int((~np.isfinite(best_models["rmse_rank"])).sum())
    _require(missing_rmse_count == 0, f"{missing_rmse_count:,} branch/SKU pairs have no finite RMSE")

    output = best_models[[*GROUP_KEYS, "model"]].sort_values(GROUP_KEYS, kind="mergesort").reset_index(drop=True)
    LOGGER.info("Selected best forecast models:\n%s", output["model"].value_counts().to_string())
    return output


def compute_historical_demand_std(inventory: pd.DataFrame) -> pd.DataFrame:
    """Estimate demand variability from positive week-over-week drops in on-hand inventory."""
    snapshots = inventory[[*GROUP_KEYS, "snapshot_date", "on_hand_units"]].copy()
    snapshots = snapshots.sort_values([*GROUP_KEYS, "snapshot_date"], kind="mergesort")
    snapshots["previous_on_hand_units"] = snapshots.groupby(GROUP_KEYS, observed=True)["on_hand_units"].shift(1)
    snapshots["snapshot_demand_units"] = (snapshots["previous_on_hand_units"] - snapshots["on_hand_units"]).clip(lower=0.0)

    historical_std = (
        snapshots.dropna(subset=["previous_on_hand_units"])
        .groupby(GROUP_KEYS, as_index=False, observed=True)["snapshot_demand_units"]
        .std(ddof=0)
        .rename(columns={"snapshot_demand_units": "historical_demand_std"})
    )
    historical_std["historical_demand_std"] = historical_std["historical_demand_std"].fillna(0.0).clip(lower=0.0)
    return historical_std


def build_forecast_demand(
    forecasts: pd.DataFrame,
    best_models: pd.DataFrame,
    historical_demand_std: pd.DataFrame,
) -> pd.DataFrame:
    """Compute average weekly demand and demand variability from the selected model forecasts."""
    best_forecasts = forecasts.merge(best_models, on=[*GROUP_KEYS, "model"], how="inner", validate="many_to_one")
    _require(not best_forecasts.empty, "No forecasts matched the selected best models")

    forecast_stats = (
        best_forecasts.groupby(GROUP_KEYS, as_index=False, observed=True)["y_pred"]
        .agg(
            avg_weekly_demand="mean",
            demand_std=lambda values: float(values.std(ddof=0)),
            forecast_week_count="count",
        )
        .sort_values(GROUP_KEYS, kind="mergesort")
        .reset_index(drop=True)
    )
    _require(forecast_stats["forecast_week_count"].gt(0).all(), "Some branch/SKU pairs have no finite selected forecasts")

    negative_demand_count = int(forecast_stats["avg_weekly_demand"].lt(0.0).sum())
    if negative_demand_count:
        LOGGER.warning("Clipping %d negative average demand estimates to zero", negative_demand_count)
        forecast_stats["avg_weekly_demand"] = forecast_stats["avg_weekly_demand"].clip(lower=0.0)

    forecast_stats = forecast_stats.merge(historical_demand_std, on=GROUP_KEYS, how="left", validate="one_to_one")
    fallback_mask = forecast_stats["demand_std"].isna() | forecast_stats["demand_std"].le(0.0)
    forecast_stats.loc[fallback_mask, "demand_std"] = forecast_stats.loc[fallback_mask, "historical_demand_std"]
    forecast_stats["demand_std"] = forecast_stats["demand_std"].fillna(0.0).clip(lower=0.0)

    if forecast_stats["forecast_week_count"].ne(26).any():
        LOGGER.warning(
            "%d branch/SKU pairs used a selected forecast horizon other than 26 finite weeks",
            int(forecast_stats["forecast_week_count"].ne(26).sum()),
        )
    LOGGER.info("Used snapshot-derived demand std fallback for %d branch/SKU pairs", int(fallback_mask.sum()))
    return forecast_stats[[*GROUP_KEYS, "avg_weekly_demand", "demand_std", "forecast_week_count"]]


def latest_inventory_snapshot(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return the latest on-hand inventory row for each branch/SKU."""
    latest = (
        inventory.sort_values([*GROUP_KEYS, "snapshot_date"], kind="mergesort")
        .groupby(GROUP_KEYS, as_index=False, observed=True)
        .tail(1)
        .copy()
    )
    latest = latest.rename(columns={"lead_time_days": "inventory_lead_time_days"})
    return latest[[*GROUP_KEYS, "on_hand_units", "inventory_lead_time_days"]].reset_index(drop=True)


def build_inventory_base(
    products: pd.DataFrame,
    branches: pd.DataFrame,
    latest_inventory: pd.DataFrame,
    forecast_demand: pd.DataFrame,
    sku_segments: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecast demand, current inventory, product costs, and SKU segments."""
    product_attrs = products[["sku_id", "unit_cost", "lead_time_days", "is_slow_mover"]].rename(
        columns={"lead_time_days": "product_lead_time_days"}
    )
    base = forecast_demand.merge(latest_inventory, on=GROUP_KEYS, how="left", validate="one_to_one")
    base = base.merge(product_attrs, on="sku_id", how="left", validate="many_to_one")
    base = base.merge(sku_segments[["sku_id", "abc_class"]], on="sku_id", how="left", validate="many_to_one")
    base["lead_time_days"] = base["inventory_lead_time_days"].fillna(base["product_lead_time_days"])

    known_branches = set(branches["branch_id"])
    unknown_branch_count = int((~base["branch_id"].isin(known_branches)).sum())
    _require(unknown_branch_count == 0, f"{unknown_branch_count:,} forecast branch IDs are missing from branches")
    _require(base["on_hand_units"].notna().all(), "Some branch/SKU pairs are missing latest on-hand inventory")
    _require(base["unit_cost"].notna().all(), "Some SKUs are missing unit_cost")
    _require(base["lead_time_days"].notna().all(), "Some branch/SKU pairs are missing lead_time_days")
    _require(base["abc_class"].notna().all(), "Some SKUs are missing ABC class")
    _require(base["unit_cost"].gt(0.0).all(), "unit_cost must be positive for all inventory scenarios")
    _require(base["lead_time_days"].ge(0.0).all(), "lead_time_days must be non-negative for all inventory scenarios")

    base["on_hand_units"] = base["on_hand_units"].astype("float64")
    base["lead_time_days"] = base["lead_time_days"].astype("float64")
    base["avg_weekly_demand"] = base["avg_weekly_demand"].astype("float64")
    base["demand_std"] = base["demand_std"].astype("float64")
    base["unit_cost"] = base["unit_cost"].astype("float64")

    coverage = np.where(
        base["avg_weekly_demand"].gt(0.0),
        base["on_hand_units"] / base["avg_weekly_demand"],
        MAX_FORECAST_COVERAGE_WEEKS,
    )
    base["weeks_of_forecast_coverage"] = np.minimum(coverage, MAX_FORECAST_COVERAGE_WEEKS)
    base["current_excess_units"] = np.maximum(
        0.0,
        base["on_hand_units"] - FORECAST_COVERAGE_EXCESS_WEEKS * base["avg_weekly_demand"],
    )
    base["excess_inventory_cost"] = base["current_excess_units"] * base["unit_cost"]
    return base


def _segmented_service_levels(abc_class: pd.Series) -> pd.Series:
    """Map ABC class to service levels for scenario D."""
    service_levels = abc_class.map(SEGMENTED_SERVICE_LEVELS)
    _require(service_levels.notna().all(), "Scenario D service level requires abc_class values A, B, or C")
    return service_levels.astype("float64")


def _scenario_frame(base: pd.DataFrame, scenario: str, service_level: float | pd.Series) -> pd.DataFrame:
    """Compute inventory math for one optimization scenario."""
    frame = base.copy()
    frame["scenario"] = scenario
    frame["service_level"] = service_level
    frame["lead_time_weeks"] = frame["lead_time_days"] / 7.0
    frame["sigma_lead"] = LEAD_TIME_VARIABILITY_FACTOR * frame["lead_time_weeks"]

    z_score = norm.ppf(frame["service_level"])
    demand_variance_during_lead_time = (
        frame["lead_time_weeks"] * np.square(frame["demand_std"])
        + np.square(frame["avg_weekly_demand"]) * np.square(frame["sigma_lead"])
    )
    frame["safety_stock_units"] = z_score * np.sqrt(np.maximum(demand_variance_during_lead_time, 0.0))
    frame["reorder_point_units"] = frame["avg_weekly_demand"] * frame["lead_time_weeks"] + frame["safety_stock_units"]

    annual_demand = frame["avg_weekly_demand"] * 52.0
    frame["eoq_units"] = np.sqrt(
        np.maximum(
            2.0 * annual_demand * ORDER_COST_PER_ORDER / (HOLDING_COST_RATE_ANNUAL * frame["unit_cost"]),
            0.0,
        )
    )
    frame["safety_stock_cost"] = frame["safety_stock_units"] * frame["unit_cost"]
    frame["projected_stockout_risk_pct"] = (1.0 - frame["service_level"]) * 100.0
    frame["scenario_total_inventory_cost"] = (
        (frame["safety_stock_units"] + frame["eoq_units"] / 2.0)
        * frame["unit_cost"]
        * HOLDING_COST_RATE_ANNUAL
    )
    return frame[INVENTORY_SCENARIO_COLUMNS]


def build_inventory_scenarios(base: pd.DataFrame) -> pd.DataFrame:
    """Build all requested service-level inventory scenarios."""
    scenario_frames = [
        _scenario_frame(base, scenario, service_level)
        for scenario, service_level in SCENARIO_SERVICE_LEVELS.items()
    ]
    scenario_frames.append(_scenario_frame(base, "D", _segmented_service_levels(base["abc_class"])))
    scenarios = pd.concat(scenario_frames, ignore_index=True)
    return scenarios.sort_values([*GROUP_KEYS, "scenario"], kind="mergesort").reset_index(drop=True)


def build_slow_mover_flags(base: pd.DataFrame) -> pd.DataFrame:
    """Return branch/SKU rows whose on-hand inventory exceeds 26 forecast weeks."""
    flags = base.loc[base["weeks_of_forecast_coverage"].gt(FORECAST_COVERAGE_EXCESS_WEEKS)].copy()
    flags["excess_units"] = flags["current_excess_units"]
    flags["excess_cost_dollars"] = flags["excess_inventory_cost"]
    flags["flag_reason"] = np.where(
        flags["is_slow_mover"].astype(bool),
        "slow_mover_forecast_coverage_gt_26_weeks",
        "forecast_coverage_gt_26_weeks",
    )
    flags = flags[SLOW_MOVER_FLAG_COLUMNS]
    return flags.sort_values(["sku_id", "branch_id"], kind="mergesort").reset_index(drop=True)


def log_scenario_summary(scenarios: pd.DataFrame) -> None:
    """Log aggregate scenario reporting requested by the optimization run."""
    summary = (
        scenarios.groupby("scenario", as_index=False, observed=True)
        .agg(
            total_safety_stock_investment=("safety_stock_cost", "sum"),
            total_excess_inventory_at_risk=("excess_inventory_cost", "sum"),
            avg_projected_service_level=("service_level", "mean"),
        )
        .sort_values("scenario", kind="mergesort")
    )
    LOGGER.info("Inventory scenario summary:\n%s", summary.to_string(index=False))


def main() -> None:
    """Run inventory optimization scenarios and persist parquet outputs."""
    _configure_logging()
    products, branches, inventory, forecasts, evaluation, sku_segments = load_inputs()

    best_models = select_best_models(evaluation)
    historical_demand_std = compute_historical_demand_std(inventory)
    forecast_demand = build_forecast_demand(forecasts, best_models, historical_demand_std)
    latest_inventory = latest_inventory_snapshot(inventory)
    base = build_inventory_base(products, branches, latest_inventory, forecast_demand, sku_segments)

    scenarios = build_inventory_scenarios(base)
    slow_mover_flags = build_slow_mover_flags(base)

    _write_parquet(scenarios, INVENTORY_SCENARIOS_OUTPUT_PATH)
    _write_parquet(slow_mover_flags, SLOW_MOVER_FLAGS_OUTPUT_PATH)
    log_scenario_summary(scenarios)
    LOGGER.info("Stockout cost multiplier parameter retained for scenario costing context: %.2f", STOCKOUT_COST_MULTIPLIER)


if __name__ == "__main__":
    main()
