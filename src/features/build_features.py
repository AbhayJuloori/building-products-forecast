"""Build the branch-SKU-week modeling table from raw and external data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "modeling_table.parquet"

GROUP_KEYS = ["branch_id", "sku_id"]
LAG_COLUMNS = {
    1: "sales_lag_1w",
    2: "lag_2w",
    4: "lag_4w",
    8: "lag_8w",
    12: "lag_12w",
    52: "lag_52w",
}
ROLLING_MEAN_WINDOWS = (4, 13, 26)
ROLLING_STD_WINDOWS = (4, 13)
CATEGORICAL_COLUMNS = ["category", "subcategory", "region", "climate_zone", "density", "velocity_tier"]
VELOCITY_TIERS = ["A", "B", "C", "D"]
EXTERNAL_COLUMNS = [
    "housing_starts",
    "housing_starts_yoy_pct",
    "building_permits",
    "permits_yoy_pct",
    "avg_temp_f",
    "temp_deviation_from_annual_avg",
    "total_precip_in",
    "precip_above_normal_flag",
]
OUTPUT_COLUMNS = [
    "branch_id",
    "sku_id",
    "week_start_date",
    "units_sold",
    "target_log",
    "stockout_flag",
    "sales_lag_1w",
    "lag_2w",
    "lag_4w",
    "lag_8w",
    "lag_12w",
    "lag_52w",
    "rolling_mean_4w",
    "rolling_mean_13w",
    "rolling_mean_26w",
    "rolling_std_4w",
    "rolling_std_13w",
    "yoy_growth",
    "week_of_year",
    "month",
    "quarter",
    "is_holiday_week",
    "days_since_last_stockout",
    "category",
    "subcategory",
    "unit_cost",
    "lead_time_days",
    "is_slow_mover",
    "region",
    "climate_zone",
    "density",
    "housing_starts",
    "housing_starts_yoy_pct",
    "building_permits",
    "permits_yoy_pct",
    "avg_temp_f",
    "temp_deviation_from_annual_avg",
    "total_precip_in",
    "precip_above_normal_flag",
    "cat_region_seasonality_index",
    "pct_tier_a_contractors",
    "velocity_tier",
    "feature_set_version",
]

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging when the module is executed directly."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _require(condition: bool, message: str) -> None:
    """Raise a clear validation error when an expected invariant is not met."""
    if not condition:
        raise ValueError(message)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source parquet files required to build the modeling table."""
    sales = pd.read_parquet(RAW_DATA_DIR / "sales_history.parquet")
    products = pd.read_parquet(RAW_DATA_DIR / "products.parquet")
    branches = pd.read_parquet(RAW_DATA_DIR / "branches.parquet")
    contractors = pd.read_parquet(RAW_DATA_DIR / "contractors.parquet")
    external = pd.read_parquet(EXTERNAL_DATA_DIR / "external_weekly.parquet")

    LOGGER.info(
        "Loaded inputs: sales=%s, products=%s, branches=%s, contractors=%s, external=%s",
        sales.shape,
        products.shape,
        branches.shape,
        contractors.shape,
        external.shape,
    )
    return sales, products, branches, contractors, external


def _prepare_base_table(sales: pd.DataFrame, products: pd.DataFrame, branches: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Join static SKU, branch, and external signals onto the weekly sales panel."""
    sales = sales.copy()
    sales["week_start_date"] = pd.to_datetime(sales["week_start_date"])

    product_attrs = products[["sku_id", "category", "subcategory", "unit_cost", "lead_time_days", "is_slow_mover"]]
    branch_attrs = branches[["branch_id", "region", "climate_zone", "density"]]
    external_attrs = external[["week_start_date", "climate_zone", *EXTERNAL_COLUMNS]].copy()
    external_attrs["week_start_date"] = pd.to_datetime(external_attrs["week_start_date"])

    frame = sales.merge(product_attrs, on="sku_id", how="left", validate="many_to_one")
    frame = frame.merge(branch_attrs, on="branch_id", how="left", validate="many_to_one")
    frame = frame.merge(external_attrs, on=["week_start_date", "climate_zone"], how="left", validate="many_to_one")
    frame = frame.sort_values([*GROUP_KEYS, "week_start_date"], kind="mergesort").reset_index(drop=True)

    _require(
        not frame.duplicated([*GROUP_KEYS, "week_start_date"]).any(),
        "Merged table has duplicate branch_id/sku_id/week_start_date rows",
    )
    return frame


def _add_lag_and_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add shifted lag and rolling features per branch-SKU series."""
    frame = frame.copy()
    grouped_units = frame.groupby(GROUP_KEYS, sort=False)["units_sold"]

    for weeks, column in LAG_COLUMNS.items():
        frame[column] = grouped_units.shift(weeks)

    shifted_units = grouped_units.shift(1)
    shifted_grouped = shifted_units.groupby([frame["branch_id"], frame["sku_id"]], sort=False)

    for window in ROLLING_MEAN_WINDOWS:
        frame[f"rolling_mean_{window}w"] = (
            shifted_grouped.rolling(window=window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )

    for window in ROLLING_STD_WINDOWS:
        frame[f"rolling_std_{window}w"] = (
            shifted_grouped.rolling(window=window, min_periods=1).std().reset_index(level=[0, 1], drop=True)
        )

    frame["yoy_growth"] = np.where(frame["lag_52w"].gt(0), frame["units_sold"] / frame["lag_52w"] - 1.0, np.nan)
    return frame


def _holiday_week_starts(years: pd.Index) -> set[pd.Timestamp]:
    """Return Monday week starts for requested US holiday weeks."""
    holidays: list[pd.Timestamp] = []

    for year in years:
        year_int = int(year)
        july_fourth = pd.Timestamp(year_int, 7, 4)
        christmas = pd.Timestamp(year_int, 12, 25)
        thanksgiving = pd.date_range(f"{year_int}-11-01", f"{year_int}-11-30", freq="W-THU")[3]
        memorial_day = pd.date_range(f"{year_int}-05-01", f"{year_int}-05-31", freq="W-MON")[-1]
        labor_day = pd.date_range(f"{year_int}-09-01", f"{year_int}-09-30", freq="W-MON")[0]
        holidays.extend([july_fourth, christmas, thanksgiving, memorial_day, labor_day])

    return {holiday - pd.Timedelta(days=int(holiday.weekday())) for holiday in holidays}


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and holiday-week features from week_start_date."""
    frame = frame.copy()
    week_start = frame["week_start_date"]
    frame["week_of_year"] = week_start.dt.isocalendar().week.astype("int16")
    frame["month"] = week_start.dt.month.astype("int8")
    frame["quarter"] = week_start.dt.quarter.astype("int8")
    holiday_week_starts = _holiday_week_starts(pd.Index(range(int(week_start.dt.year.min()), int(week_start.dt.year.max()) + 1)))
    frame["is_holiday_week"] = week_start.isin(holiday_week_starts)
    return frame


def _add_days_since_stockout(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the count of weeks since the most recent stockout per branch-SKU."""
    frame = frame.copy()
    row_number = frame.groupby(GROUP_KEYS, sort=False).cumcount()
    last_stockout_row = row_number.where(frame["stockout_flag"].astype(bool))
    last_stockout_row = last_stockout_row.groupby([frame["branch_id"], frame["sku_id"]], sort=False).ffill()
    frame["days_since_last_stockout"] = (row_number - last_stockout_row).astype("float64")
    return frame


def _add_seasonality_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Add category-region seasonality from the first two years within the training window."""
    frame = frame.copy()
    unique_dates = pd.Index(frame["week_start_date"].drop_duplicates().sort_values())
    train_date_count = max(1, int(np.floor(len(unique_dates) * 0.8)))
    seasonality_date_count = min(104, train_date_count)
    seasonality_dates = unique_dates[:seasonality_date_count]

    source = frame[frame["week_start_date"].isin(seasonality_dates)]
    weekly_units = (
        source.groupby(["category", "region", "week_of_year"], as_index=False, observed=True)["units_sold"]
        .mean()
        .rename(columns={"units_sold": "weekly_units"})
    )
    annual_avg = (
        source.groupby(["category", "region"], as_index=False, observed=True)["units_sold"]
        .mean()
        .rename(columns={"units_sold": "annual_avg_units"})
    )
    seasonality = weekly_units.merge(annual_avg, on=["category", "region"], how="left", validate="many_to_one")
    seasonality["cat_region_seasonality_index"] = np.where(
        seasonality["annual_avg_units"].gt(0),
        seasonality["weekly_units"] / seasonality["annual_avg_units"],
        np.nan,
    )

    frame = frame.merge(
        seasonality[["category", "region", "week_of_year", "cat_region_seasonality_index"]],
        on=["category", "region", "week_of_year"],
        how="left",
        validate="many_to_one",
    )

    LOGGER.info(
        "Computed category-region seasonality from %d weekly dates: %s through %s",
        len(seasonality_dates),
        seasonality_dates.min().date(),
        seasonality_dates.max().date(),
    )
    return frame


def _add_contractor_concentration(frame: pd.DataFrame, contractors: pd.DataFrame) -> pd.DataFrame:
    """Add branch-level share of tier A contractors."""
    contractor_stats = (
        contractors.assign(is_tier_a=contractors["annual_spend_tier"].eq("A"))
        .groupby("branch_id", as_index=False)
        .agg(total_contractors=("contractor_id", "nunique"), tier_a_contractors=("is_tier_a", "sum"))
    )
    contractor_stats["pct_tier_a_contractors"] = np.where(
        contractor_stats["total_contractors"].gt(0),
        contractor_stats["tier_a_contractors"] / contractor_stats["total_contractors"],
        0.0,
    )

    frame = frame.merge(
        contractor_stats[["branch_id", "pct_tier_a_contractors"]],
        on="branch_id",
        how="left",
        validate="many_to_one",
    )
    frame["pct_tier_a_contractors"] = frame["pct_tier_a_contractors"].fillna(0.0)
    return frame


def _add_velocity_tier(frame: pd.DataFrame) -> pd.DataFrame:
    """Add SKU velocity tier by weekly category rank of prior 13-week sales."""
    frame = frame.copy()
    sku_week = (
        frame.groupby(["sku_id", "category", "week_start_date"], as_index=False, observed=True)["units_sold"]
        .sum()
        .sort_values(["sku_id", "week_start_date"], kind="mergesort")
    )
    sku_week["trailing_13w_avg_sales"] = (
        sku_week.groupby("sku_id", sort=False)["units_sold"]
        .shift(1)
        .groupby(sku_week["sku_id"], sort=False)
        .rolling(window=13, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    sku_week["velocity_rank_pct"] = sku_week.groupby(["category", "week_start_date"], observed=True)[
        "trailing_13w_avg_sales"
    ].rank(method="first", ascending=False, pct=True)

    tier = pd.Series(pd.NA, index=sku_week.index, dtype="object")
    tier.loc[sku_week["velocity_rank_pct"].le(0.20)] = "A"
    tier.loc[sku_week["velocity_rank_pct"].gt(0.20) & sku_week["velocity_rank_pct"].le(0.50)] = "B"
    tier.loc[sku_week["velocity_rank_pct"].gt(0.50) & sku_week["velocity_rank_pct"].le(0.80)] = "C"
    tier.loc[sku_week["velocity_rank_pct"].gt(0.80)] = "D"
    sku_week["velocity_tier"] = pd.Categorical(tier, categories=VELOCITY_TIERS, ordered=True)

    frame = frame.merge(
        sku_week[["sku_id", "week_start_date", "velocity_tier"]],
        on=["sku_id", "week_start_date"],
        how="left",
        validate="many_to_one",
    )
    return frame


def _finalize_output(frame: pd.DataFrame, original_row_count: int, series_count: int) -> pd.DataFrame:
    """Drop warm-up rows, validate shape, cast dtypes, and select final columns."""
    frame = frame.copy()
    frame["target_log"] = np.log1p(frame["units_sold"])
    frame["feature_set_version"] = "v1"

    before_drop = len(frame)
    frame = frame[frame["lag_52w"].notna()].copy()
    dropped_rows = before_drop - len(frame)
    LOGGER.info("Dropped %d warm-up rows with missing lag_52w", dropped_rows)

    expected_rows = original_row_count - (series_count * 52)
    lower_bound = int(expected_rows * 0.95)
    upper_bound = int(expected_rows * 1.05)
    _require(
        lower_bound <= len(frame) <= upper_bound,
        f"Unexpected output row count: got {len(frame):,}, expected about {expected_rows:,}",
    )
    _require(
        not frame.duplicated([*GROUP_KEYS, "week_start_date"]).any(),
        "Output table has duplicate branch_id/sku_id/week_start_date rows",
    )

    frame["week_start_date"] = pd.to_datetime(frame["week_start_date"])
    for column in CATEGORICAL_COLUMNS:
        if column == "velocity_tier":
            frame[column] = pd.Categorical(frame[column], categories=VELOCITY_TIERS, ordered=True)
        else:
            frame[column] = frame[column].astype("category")

    output = frame[OUTPUT_COLUMNS].sort_values([*GROUP_KEYS, "week_start_date"], kind="mergesort").reset_index(drop=True)
    LOGGER.info("Output shape: %s; expected about %d rows", output.shape, expected_rows)
    LOGGER.info("Null counts:\n%s", output.isna().sum().to_string())
    LOGGER.info("Dtype summary:\n%s", output.dtypes.astype(str).to_string())
    return output


def build_modeling_table() -> pd.DataFrame:
    """Build and return the complete modeling table."""
    sales, products, branches, contractors, external = _load_inputs()
    original_row_count = len(sales)
    series_count = sales[GROUP_KEYS].drop_duplicates().shape[0]

    frame = _prepare_base_table(sales, products, branches, external)
    frame = _add_lag_and_rolling_features(frame)
    frame = _add_calendar_features(frame)
    frame = _add_days_since_stockout(frame)
    frame = _add_seasonality_index(frame)
    frame = _add_contractor_concentration(frame, contractors)
    frame = _add_velocity_tier(frame)
    return _finalize_output(frame, original_row_count=original_row_count, series_count=series_count)


def main() -> None:
    """Build and persist the modeling table parquet file."""
    _configure_logging()
    output = build_modeling_table()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    LOGGER.info("Wrote modeling table to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
