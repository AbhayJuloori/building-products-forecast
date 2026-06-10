"""Fetch or synthesize external macro and weather signals for weekly demand modeling."""

from __future__ import annotations

import logging
import os
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests


SEED = 42
RAW_DATA_DIR = Path("data/raw")
EXTERNAL_DATA_DIR = Path("data/external")
MONTHS = pd.date_range("2019-01-01", "2024-12-01", freq="MS")
CLIMATE_ZONES = ("cold", "mixed", "hot")
NOAA_GSOM_URL_TEMPLATE = "https://www.ncei.noaa.gov/data/global-summary-of-the-month/access/{station_id}.csv"
NOAA_ZONE_TO_STATION_MAP = {
    "cold": [
        NOAA_GSOM_URL_TEMPLATE.format(station_id="USW00014739"),
        NOAA_GSOM_URL_TEMPLATE.format(station_id="USW00094846"),
    ],
    "mixed": [NOAA_GSOM_URL_TEMPLATE.format(station_id="USW00013960")],
    "hot": [NOAA_GSOM_URL_TEMPLATE.format(station_id="USW00013874")],
}

HOUSING_COLUMNS = ["month_start", "housing_starts_thousands_saar"]
PERMIT_COLUMNS = ["month_start", "building_permits_thousands_saar"]
WEATHER_COLUMNS = ["month_start", "climate_zone", "avg_temp_f", "total_precip_in", "source"]
EXTERNAL_WEEKLY_COLUMNS = [
    "week_start_date",
    "housing_starts",
    "housing_starts_yoy_pct",
    "building_permits",
    "permits_yoy_pct",
    "climate_zone",
    "avg_temp_f",
    "temp_deviation_from_annual_avg",
    "total_precip_in",
    "precip_above_normal_flag",
    "weather_source",
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


def _load_fred_api_key() -> Optional[str]:
    """Load FRED_API_KEY from .env or the process environment when available."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        LOGGER.warning("python-dotenv is unavailable; reading FRED_API_KEY from environment only: %s", exc)
    else:
        load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        LOGGER.warning("FRED_API_KEY is missing; using deterministic synthetic macro fallback")
    return api_key


def _coerce_fred_series(series: pd.Series, output_column: str) -> pd.DataFrame:
    """Convert a FRED time series to the complete monthly frame required downstream."""
    data = series.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    monthly = (
        data.rename(output_column)
        .to_frame()
        .assign(month_start=lambda frame: frame.index.to_period("M").to_timestamp())
        .groupby("month_start", as_index=False)[output_column]
        .mean()
        .set_index("month_start")
        .reindex(MONTHS)
        .ffill()
        .bfill()
        .rename_axis("month_start")
        .reset_index()
    )
    monthly[output_column] = monthly[output_column].astype(float).round(1)
    return monthly


def _fetch_fred_macro(api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch housing starts and permits from FRED for the configured monthly window."""
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    start = MONTHS.min().strftime("%Y-%m-%d")
    end = (MONTHS.max() + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")

    housing_series = fred.get_series("HOUST", observation_start=start, observation_end=end)
    permits_series = fred.get_series("PERMIT", observation_start=start, observation_end=end)
    _require(not housing_series.empty, "FRED HOUST returned no rows")
    _require(not permits_series.empty, "FRED PERMIT returned no rows")

    housing = _coerce_fred_series(housing_series, "housing_starts_thousands_saar")
    permits = _coerce_fred_series(permits_series, "building_permits_thousands_saar")
    return housing, permits


def _generate_synthetic_macro(rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic HOUST and PERMIT fallback series with realistic shocks."""
    seasonal_by_month = np.array([0.93, 0.94, 1.00, 1.05, 1.08, 1.09, 1.08, 1.05, 1.02, 1.00, 0.97, 0.92])
    values: list[float] = []

    for month_start in MONTHS:
        year = int(month_start.year)
        month = int(month_start.month)

        if year == 2019:
            baseline = 1450.0 + (month - 1) * 4.0
        elif year == 2020:
            baseline = 1500.0
        elif year == 2021:
            baseline = 1475.0
        elif year == 2022:
            baseline = 1500.0 - (month - 1) * 6.0
        elif year == 2023:
            baseline = 1400.0 + (month - 1) * 1.5
        else:
            baseline = 1410.0 + (month - 1) * 1.0

        value = baseline * seasonal_by_month[month - 1]

        if pd.Timestamp("2020-03-01") <= month_start <= pd.Timestamp("2020-08-01"):
            value *= 0.75
        elif pd.Timestamp("2020-10-01") <= month_start <= pd.Timestamp("2021-12-01"):
            months_since_boom_start = (year - 2020) * 12 + (month - 10)
            value *= min(1.10 + months_since_boom_start * 0.006, 1.18)
        elif year == 2022:
            value *= 1.05 - (month - 1) * 0.006

        value += rng.normal(0.0, 28.0)
        values.append(float(np.clip(value, 1050.0, 1810.0)))

    housing_values = np.array(values)
    permit_values = np.clip(housing_values * 0.95 + rng.normal(0.0, 22.0, size=len(housing_values)), 950.0, 1800.0)

    housing = pd.DataFrame(
        {
            "month_start": MONTHS,
            "housing_starts_thousands_saar": np.round(housing_values, 1),
        }
    )
    permits = pd.DataFrame(
        {
            "month_start": MONTHS,
            "building_permits_thousands_saar": np.round(permit_values, 1),
        }
    )
    return housing, permits


def _load_macro_tables(rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load FRED macro tables when possible, otherwise return deterministic fallback data."""
    api_key = _load_fred_api_key()
    if api_key:
        try:
            housing, permits = _fetch_fred_macro(api_key)
        except Exception as exc:  # pragma: no cover - exercised only with live FRED/network failures.
            LOGGER.warning("FRED fetch failed; using deterministic synthetic macro fallback: %s", exc)
        else:
            LOGGER.info("Fetched macro series from FRED: HOUST and PERMIT")
            return housing, permits, "fred"

    housing, permits = _generate_synthetic_macro(rng)
    return housing, permits, "synthetic"


def _generate_weather(rng: np.random.Generator) -> pd.DataFrame:
    """Generate deterministic synthetic monthly weather by climate zone."""
    temp_ranges = {
        "cold": (25.0, 75.0),
        "mixed": (40.0, 85.0),
        "hot": (55.0, 92.0),
    }
    precip_bounds = {
        "cold": (2.5, 3.5),
        "mixed": (3.0, 4.0),
        "hot": (3.0, 5.5),
    }
    records: list[dict[str, object]] = []

    for climate_zone in CLIMATE_ZONES:
        low_temp, high_temp = temp_ranges[climate_zone]
        temp_mid = (low_temp + high_temp) / 2.0
        temp_amp = (high_temp - low_temp) / 2.0
        precip_low, precip_high = precip_bounds[climate_zone]

        for month_start in MONTHS:
            month = int(month_start.month)
            avg_temp = temp_mid + temp_amp * np.cos(2.0 * np.pi * (month - 7) / 12.0)
            avg_temp += rng.normal(0.0, 0.7)
            avg_temp = float(np.clip(avg_temp, low_temp, high_temp))

            if climate_zone == "cold":
                base_precip = 2.85 + 0.25 * np.cos(2.0 * np.pi * (month - 4) / 12.0)
                precip_noise = rng.normal(0.0, 0.18)
            elif climate_zone == "mixed":
                base_precip = 3.35 + 0.30 * np.cos(2.0 * np.pi * (month - 5) / 12.0)
                precip_noise = rng.normal(0.0, 0.20)
            else:
                summer_spike = np.exp(-((month - 8) ** 2) / 5.0)
                base_precip = 3.15 + 1.80 * summer_spike
                precip_noise = rng.normal(0.0, 0.28)

            anomaly = 0.0
            if rng.random() < 0.08:
                anomaly = rng.uniform(0.45, 0.80)
            total_precip = float(np.clip(base_precip + precip_noise + anomaly, precip_low, precip_high))

            records.append(
                {
                    "month_start": month_start,
                    "climate_zone": climate_zone,
                    "avg_temp_f": round(avg_temp, 1),
                    "total_precip_in": round(total_precip, 2),
                    "source": "synthetic_fallback",
                }
            )

    return pd.DataFrame.from_records(records, columns=WEATHER_COLUMNS)


def _fetch_noaa_weather(zone_to_station_map: dict) -> pd.DataFrame:
    """Fetch NOAA GSOM monthly weather for the configured climate-zone stations."""
    station_records: list[pd.DataFrame] = []

    for climate_zone, station_urls in zone_to_station_map.items():
        if isinstance(station_urls, str):
            station_urls = [station_urls]

        for station_url in station_urls:
            url = station_url
            if not url.startswith("http"):
                url = NOAA_GSOM_URL_TEMPLATE.format(station_id=url)

            response = None
            last_error: Optional[Exception] = None
            for _ in range(2):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"NOAA GSOM station fetch failed for {url}: {last_error}") from last_error

            station = pd.read_csv(StringIO(response.text), dtype=str)
            _require("DATE" in station.columns, f"NOAA GSOM station data missing DATE column: {url}")
            _require("PRCP" in station.columns, f"NOAA GSOM station data missing PRCP column: {url}")

            station["month_start"] = pd.to_datetime(station["DATE"].astype(str).str[:7] + "-01", errors="coerce")
            station = station[station["month_start"].isin(MONTHS)].copy()
            _require(len(station) == len(MONTHS), f"NOAA GSOM station month coverage incomplete for {url}")

            if "TAVG" in station.columns:
                avg_temp_tenths_c = pd.to_numeric(station["TAVG"], errors="coerce")
            else:
                avg_temp_tenths_c = pd.Series(np.nan, index=station.index)

            if avg_temp_tenths_c.isna().any():
                _require(
                    {"TMAX", "TMIN"}.issubset(station.columns),
                    f"NOAA GSOM station data missing TAVG and TMAX/TMIN fallback columns: {url}",
                )
                tmax_tenths_c = pd.to_numeric(station["TMAX"], errors="coerce")
                tmin_tenths_c = pd.to_numeric(station["TMIN"], errors="coerce")
                avg_temp_tenths_c = avg_temp_tenths_c.fillna((tmax_tenths_c + tmin_tenths_c) / 2.0)

            precip_tenths_mm = pd.to_numeric(station["PRCP"], errors="coerce")
            _require(avg_temp_tenths_c.notna().all(), f"NOAA GSOM station temperature values incomplete for {url}")
            _require(precip_tenths_mm.notna().all(), f"NOAA GSOM station precipitation values incomplete for {url}")

            # GSOM stores temperature in tenths of C and precipitation in tenths of mm.
            avg_temp_f = (avg_temp_tenths_c / 10.0) * 9.0 / 5.0 + 32.0
            precip_in = (precip_tenths_mm / 10.0) / 25.4

            station_weather = pd.DataFrame(
                {
                    "month_start": station["month_start"],
                    "climate_zone": climate_zone,
                    "avg_temp_f": avg_temp_f,
                    "total_precip_in": precip_in,
                }
            )
            _require(
                station_weather["avg_temp_f"].between(-80.0, 140.0).all(),
                f"NOAA GSOM station temperature values outside expected Fahrenheit range for {url}",
            )
            _require(
                station_weather["total_precip_in"].between(0.0, 80.0).all(),
                f"NOAA GSOM station precipitation values outside expected monthly range for {url}",
            )
            station_records.append(station_weather)

    all_stations = pd.concat(station_records, ignore_index=True)
    weather = (
        all_stations.groupby(["month_start", "climate_zone"], as_index=False)[["avg_temp_f", "total_precip_in"]]
        .mean()
        .assign(
            avg_temp_f=lambda frame: frame["avg_temp_f"].round(1),
            total_precip_in=lambda frame: frame["total_precip_in"].round(2),
            source="noaa_gsom",
        )
    )
    weather["climate_zone"] = pd.Categorical(weather["climate_zone"], categories=CLIMATE_ZONES, ordered=True)
    weather = weather.sort_values(["month_start", "climate_zone"], ignore_index=True)
    weather["climate_zone"] = weather["climate_zone"].astype(str)
    return weather[WEATHER_COLUMNS]


def _add_macro_features(housing: pd.DataFrame, permits: pd.DataFrame) -> pd.DataFrame:
    """Join monthly macro tables and add same-month prior-year percentage changes."""
    macro = housing.merge(permits, on="month_start", how="inner").sort_values("month_start", ignore_index=True)
    macro["housing_starts_yoy_pct"] = macro["housing_starts_thousands_saar"].pct_change(periods=12) * 100.0
    macro["permits_yoy_pct"] = macro["building_permits_thousands_saar"].pct_change(periods=12) * 100.0
    return macro.rename(
        columns={
            "housing_starts_thousands_saar": "housing_starts",
            "building_permits_thousands_saar": "building_permits",
        }
    )


def _add_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Add climate-zone temperature and precipitation features used by weekly modeling."""
    enriched = weather.copy()
    enriched["calendar_month"] = enriched["month_start"].dt.month
    annual_mean = enriched.groupby("climate_zone")["avg_temp_f"].transform("mean")
    monthly_precip_avg = enriched.groupby(["climate_zone", "calendar_month"])["total_precip_in"].transform("mean")
    enriched["temp_deviation_from_annual_avg"] = (enriched["avg_temp_f"] - annual_mean).round(1)
    enriched["precip_above_normal_flag"] = enriched["total_precip_in"] > (monthly_precip_avg * 1.2)
    return enriched.drop(columns=["calendar_month"]).rename(columns={"source": "weather_source"})


def _load_sales_weeks() -> pd.Series:
    """Read the sales calendar week starts from raw sales history."""
    sales_path = RAW_DATA_DIR / "sales_history.parquet"
    _require(sales_path.exists(), f"Missing required input: {sales_path}")
    sales_history = pd.read_parquet(sales_path, columns=["week_start_date"])
    weeks = sales_history["week_start_date"].drop_duplicates().sort_values(ignore_index=True)
    _require(pd.api.types.is_datetime64_any_dtype(weeks), "week_start_date must be datetime64")
    _require(weeks.dt.weekday.eq(0).all(), "All sales week_start_date values must be Monday-anchored")
    return weeks


def _load_climate_zones() -> list[str]:
    """Read and validate climate zones from raw branches."""
    branches_path = RAW_DATA_DIR / "branches.parquet"
    _require(branches_path.exists(), f"Missing required input: {branches_path}")
    branches = pd.read_parquet(branches_path, columns=["climate_zone"])
    zones = sorted(branches["climate_zone"].dropna().unique().tolist())
    _require(set(zones) == set(CLIMATE_ZONES), f"Expected climate zones {CLIMATE_ZONES}, found {zones}")
    return [zone for zone in CLIMATE_ZONES if zone in zones]


def _build_external_weekly(
    housing: pd.DataFrame,
    permits: pd.DataFrame,
    weather: pd.DataFrame,
    weeks: pd.Series,
    climate_zones: list[str],
) -> pd.DataFrame:
    """Forward-fill monthly macro and weather data to the sales weekly climate-zone grid."""
    macro = _add_macro_features(housing, permits)
    weather_features = _add_weather_features(weather)
    weekly_grid = pd.MultiIndex.from_product(
        [weeks, climate_zones],
        names=["week_start_date", "climate_zone"],
    ).to_frame(index=False)
    weekly_grid["month_start"] = weekly_grid["week_start_date"].dt.to_period("M").dt.to_timestamp()

    external_weekly = weekly_grid.merge(macro, on="month_start", how="left").merge(
        weather_features,
        on=["month_start", "climate_zone"],
        how="left",
    )
    external_weekly = external_weekly.sort_values(["week_start_date", "climate_zone"], ignore_index=True)
    return external_weekly[EXTERNAL_WEEKLY_COLUMNS]


def _validate_monthly_table(
    table: pd.DataFrame,
    columns: list[str],
    value_column: str,
    table_name: str,
) -> None:
    """Validate one complete monthly external signal table."""
    _require(table.columns.tolist() == columns, f"{table_name} columns mismatch: {table.columns.tolist()}")
    _require(len(table) == len(MONTHS), f"{table_name} expected {len(MONTHS)} rows, found {len(table)}")
    _require(table["month_start"].min() == MONTHS.min(), f"{table_name} starts outside expected window")
    _require(table["month_start"].max() == MONTHS.max(), f"{table_name} ends outside expected window")
    _require(table["month_start"].is_unique, f"{table_name} has duplicate month_start values")
    _require(table[value_column].notna().all(), f"{table_name} has missing {value_column} values")
    _require((table[value_column] > 0).all(), f"{table_name} has non-positive {value_column} values")


def _validate_outputs(
    housing: pd.DataFrame,
    permits: pd.DataFrame,
    weather: pd.DataFrame,
    external_weekly: pd.DataFrame,
    weeks: pd.Series,
    climate_zones: list[str],
) -> None:
    """Validate schemas, row counts, date ranges, and required feature coverage before writing."""
    _validate_monthly_table(housing, HOUSING_COLUMNS, "housing_starts_thousands_saar", "housing_starts")
    _validate_monthly_table(permits, PERMIT_COLUMNS, "building_permits_thousands_saar", "building_permits")

    expected_weather_rows = len(MONTHS) * len(CLIMATE_ZONES)
    _require(weather.columns.tolist() == WEATHER_COLUMNS, f"weather columns mismatch: {weather.columns.tolist()}")
    _require(len(weather) == expected_weather_rows, f"weather expected {expected_weather_rows} rows, found {len(weather)}")
    _require(set(weather["climate_zone"]) == set(CLIMATE_ZONES), "weather climate zones mismatch")
    _require(weather.groupby("climate_zone")["month_start"].nunique().eq(len(MONTHS)).all(), "weather month coverage incomplete")
    _require(weather[["avg_temp_f", "total_precip_in"]].notna().all().all(), "weather contains missing numeric values")
    _require(weather["source"].notna().all(), "weather contains missing source values")

    expected_weekly_rows = len(weeks) * len(climate_zones)
    _require(
        external_weekly.columns.tolist() == EXTERNAL_WEEKLY_COLUMNS,
        f"external_weekly columns mismatch: {external_weekly.columns.tolist()}",
    )
    _require(len(external_weekly) == expected_weekly_rows, f"external_weekly expected {expected_weekly_rows} rows, found {len(external_weekly)}")
    _require(external_weekly["week_start_date"].min() == weeks.min(), "external_weekly starts outside sales calendar")
    _require(external_weekly["week_start_date"].max() == weeks.max(), "external_weekly ends outside sales calendar")
    _require(set(external_weekly["climate_zone"]) == set(climate_zones), "external_weekly climate zones mismatch")

    required_non_null = [
        "housing_starts",
        "building_permits",
        "avg_temp_f",
        "temp_deviation_from_annual_avg",
        "total_precip_in",
        "precip_above_normal_flag",
        "weather_source",
    ]
    _require(external_weekly[required_non_null].notna().all().all(), "external_weekly contains missing required values")
    weekly_2020_plus = external_weekly["week_start_date"] >= pd.Timestamp("2020-01-01")
    _require(
        external_weekly.loc[weekly_2020_plus, ["housing_starts_yoy_pct", "permits_yoy_pct"]].notna().all().all(),
        "external_weekly has missing YoY values after 2019",
    )


def _write_parquet_outputs(tables: dict[str, pd.DataFrame]) -> None:
    """Write generated external tables to data/external as pyarrow-backed parquet files."""
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for table_name, table in tables.items():
        output_path = EXTERNAL_DATA_DIR / f"{table_name}.parquet"
        table.to_parquet(output_path, engine="pyarrow", index=False)
        LOGGER.info("Wrote %s (%s rows)", output_path, f"{len(table):,}")


def _log_summary(tables: dict[str, pd.DataFrame], macro_source: str, weather_fallback_reason: Optional[str]) -> None:
    """Log row counts, date ranges, and compact signal summaries."""
    LOGGER.info("Macro source: %s", macro_source)
    weather_source = tables["weather"]["source"].iloc[0]
    if weather_source == "synthetic_fallback":
        LOGGER.info("Weather source: synthetic_fallback (reason: %s)", weather_fallback_reason or "NOAA GSOM unavailable")
    else:
        LOGGER.info("Weather source: %s", weather_source)
    for table_name, table in tables.items():
        LOGGER.info("%s rows: %s", table_name, f"{len(table):,}")
        date_column = "week_start_date" if "week_start_date" in table.columns else "month_start"
        LOGGER.info(
            "%s %s range: %s to %s",
            table_name,
            date_column,
            table[date_column].min().date(),
            table[date_column].max().date(),
        )

    weekly = tables["external_weekly"]
    LOGGER.info(
        "external_weekly macro ranges: housing_starts %.1f-%.1f, building_permits %.1f-%.1f",
        weekly["housing_starts"].min(),
        weekly["housing_starts"].max(),
        weekly["building_permits"].min(),
        weekly["building_permits"].max(),
    )
    LOGGER.info(
        "external_weekly weather ranges: avg_temp_f %.1f-%.1f, total_precip_in %.2f-%.2f",
        weekly["avg_temp_f"].min(),
        weekly["avg_temp_f"].max(),
        weekly["total_precip_in"].min(),
        weekly["total_precip_in"].max(),
    )
    LOGGER.info(
        "precip_above_normal_flag true rate: %.2f%%",
        weekly["precip_above_normal_flag"].mean() * 100.0,
    )


def main() -> None:
    """Fetch or synthesize, validate, and save all external signal parquet outputs."""
    _configure_logging()
    macro_rng = np.random.default_rng(SEED)
    weather_rng = np.random.default_rng(SEED)

    weeks = _load_sales_weeks()
    climate_zones = _load_climate_zones()
    housing, permits, macro_source = _load_macro_tables(macro_rng)
    weather_fallback_reason = None
    try:
        weather = _fetch_noaa_weather(NOAA_ZONE_TO_STATION_MAP)
    except Exception as exc:  # pragma: no cover - exercised only with live NOAA/network failures.
        weather_fallback_reason = str(exc)
        LOGGER.warning("NOAA GSOM weather fetch failed; using deterministic synthetic weather fallback: %s", exc)
        weather = _generate_weather(weather_rng)
    external_weekly = _build_external_weekly(housing, permits, weather, weeks, climate_zones)

    _validate_outputs(housing, permits, weather, external_weekly, weeks, climate_zones)
    tables = {
        "housing_starts": housing,
        "building_permits": permits,
        "weather": weather,
        "external_weekly": external_weekly,
    }
    _write_parquet_outputs(tables)
    _log_summary(tables, macro_source, weather_fallback_reason)


if __name__ == "__main__":
    main()
