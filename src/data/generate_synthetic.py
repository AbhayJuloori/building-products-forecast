"""Generate deterministic synthetic data for branch-level demand modeling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 42
RAW_DATA_DIR = Path("data/raw")
MARKUP = 1.35
BURSTY_BRANCH_IDS = ("B003", "B007", "B010")

LOGGER = logging.getLogger(__name__)

CATEGORY_SEASONALITY: dict[str, list[float]] = {
    "Roofing": [0.55, 0.65, 0.95, 1.25, 1.45, 1.55, 1.50, 1.35, 1.10, 0.85, 0.65, 0.55],
    "Siding": [0.70, 0.75, 1.15, 1.35, 1.25, 1.05, 0.95, 1.05, 1.25, 1.30, 0.85, 0.70],
    "Exterior": [0.90, 0.92, 1.02, 1.08, 1.10, 1.06, 1.02, 1.03, 1.08, 1.05, 0.92, 0.92],
}
BRANCH_COLUMNS = ["branch_id", "name", "region", "climate_zone", "density"]
PRODUCT_COLUMNS = ["sku_id", "name", "category", "subcategory", "unit_cost", "lead_time_days", "weight_class", "seasonality_profile", "is_slow_mover"]
SALES_COLUMNS = ["week_start_date", "branch_id", "sku_id", "units_sold", "revenue", "stockout_flag"]
INVENTORY_COLUMNS = ["snapshot_date", "branch_id", "sku_id", "on_hand_units", "reorder_point", "lead_time_days"]
CONTRACTOR_COLUMNS = ["contractor_id", "branch_id", "name", "trade_type", "annual_spend_tier", "account_age_years"]


def _configure_logging() -> None:
    """Configure console logging when the module is executed directly."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _generate_branches() -> pd.DataFrame:
    """Build the fixed 12-branch network with prescribed regional and density mix."""
    return pd.DataFrame(
        [
            ("B001", "Boston Metro", "Northeast", "cold", "urban"),
            ("B002", "Hartford Valley", "Northeast", "cold", "suburban"),
            ("B003", "Albany Northway", "Northeast", "cold", "rural"),
            ("B004", "Chicago West", "Midwest", "cold", "urban"),
            ("B005", "Detroit Lakes", "Midwest", "cold", "suburban"),
            ("B006", "Des Moines Plains", "Midwest", "cold", "rural"),
            ("B007", "Atlanta Southside", "Southeast", "hot", "urban"),
            ("B008", "Charlotte Piedmont", "Southeast", "hot", "suburban"),
            ("B009", "Tampa Bay", "Southeast", "hot", "suburban"),
            ("B010", "Dallas Fort Worth", "South", "mixed", "urban"),
            ("B011", "Nashville Central", "South", "mixed", "suburban"),
            ("B012", "Little Rock Delta", "South", "mixed", "rural"),
        ],
        columns=BRANCH_COLUMNS,
    )


def _generate_products(rng: np.random.Generator) -> pd.DataFrame:
    """Generate an 80-SKU catalog across roofing, siding, and exterior products."""
    records: list[dict[str, Any]] = []

    def add_product(
        name: str,
        category: str,
        subcategory: str,
        unit_cost: float,
        lead_time_days: int,
        weight_class: str,
    ) -> None:
        """Append one product record with a deterministic SKU identifier."""
        records.append(
            {
                "sku_id": f"SKU{len(records) + 1:04d}",
                "name": name,
                "category": category,
                "subcategory": subcategory,
                "unit_cost": round(float(np.clip(unit_cost, 8, 450)), 2),
                "lead_time_days": int(lead_time_days),
                "weight_class": weight_class,
                "seasonality_profile": list(CATEGORY_SEASONALITY[category]),
                "is_slow_mover": False,
            }
        )

    colors = ["Charcoal", "Weathered Wood", "Driftwood", "Onyx Black", "Autumn Brown", "Estate Gray"]
    grades = [("3-Tab", 28.0), ("Architectural", 38.0), ("Premium Architectural", 54.0)]
    color_adjustments = {"Charcoal": 0.0, "Weathered Wood": 1.5, "Driftwood": 1.0, "Onyx Black": 2.0, "Autumn Brown": 0.5, "Estate Gray": 1.25}
    for grade, base_cost in grades:
        for color in colors:
            cost = base_cost + color_adjustments[color] + rng.uniform(-1.5, 1.5)
            add_product(f"{grade} Asphalt Shingle - {color}", "Roofing", "Asphalt Shingles", cost, rng.integers(5, 15), "heavy")

    for color in colors[:5]:
        cost = 45.0 + color_adjustments[color] + rng.uniform(-2.0, 3.0)
        add_product(f"Hip and Ridge Cap - {color}", "Roofing", "Ridge Caps", cost, rng.integers(4, 12), "medium")

    for name, cost in [
        ("10 Square Synthetic Underlayment Roll", 82.0),
        ("High Grip Synthetic Underlayment Roll", 118.0),
        ("Breathable Synthetic Underlayment Roll", 146.0),
        ("Economy Felt Underlayment Roll", 34.0),
    ]:
        add_product(name, "Roofing", "Underlayment", cost + rng.uniform(-4.0, 4.0), rng.integers(3, 13), "medium")

    for name, cost in [
        ("Granular Ice/Water Shield 2SQ", 92.0),
        ("High Temp Ice/Water Shield 2SQ", 142.0),
        ("Self-Adhered Eave Protection 1SQ", 76.0),
        ("Premium Valley Membrane 2SQ", 158.0),
    ]:
        add_product(name, "Roofing", "Ice/Water Shield", cost + rng.uniform(-5.0, 5.0), rng.integers(5, 16), "medium")

    for name, cost, weight in [
        ("Aluminum Step Flashing 4x4", 18.0, "light"),
        ("Galvanized Valley Flashing 20in", 42.0, "medium"),
        ("Drip Edge White 10ft Bundle", 26.0, "medium"),
        ("Copper Step Flashing 4x4", 112.0, "light"),
    ]:
        add_product(name, "Roofing", "Flashing", cost + rng.uniform(-3.0, 3.0), rng.integers(3, 11), weight)

    siding_colors = ["White", "Almond", "Clay", "Graphite", "Wicker"]
    profiles = [("Dutch Lap", 106.0), ("Traditional Lap", 98.0)]
    for color in siding_colors:
        for profile, cost in profiles:
            add_product(f"Vinyl {profile} Panel - {color}", "Siding", "Vinyl Panels", cost + rng.uniform(-5.0, 8.0), rng.integers(6, 19), "heavy")

    for color in ["Arctic White", "Cobble Gray", "Khaki Brown"]:
        for width, cost in [("6.25in", 214.0), ("8.25in", 286.0)]:
            name = f"Fiber Cement Lap Siding {width} - {color}"
            add_product(name, "Siding", "Fiber Cement", cost + rng.uniform(-10.0, 12.0), rng.integers(10, 22), "heavy")

    for name, cost in [
        ("PVC Trim Board 1x4 White", 24.0),
        ("PVC Trim Board 1x6 White", 36.0),
        ("PVC Trim Board 1x8 White", 54.0),
        ("Reversible Trim Board 5/4x4", 68.0),
    ]:
        add_product(name, "Siding", "Trim", cost + rng.uniform(-2.0, 3.0), rng.integers(4, 15), "medium")

    for color in siding_colors:
        add_product(f"Vinyl Outside Corner Post - {color}", "Siding", "Corner Posts", 22.0 + rng.uniform(-2.0, 4.0), rng.integers(4, 13), "light")

    for name, cost, weight in [
        ("Standard Housewrap 9x150 Roll", 92.0, "medium"),
        ("Commercial Housewrap 10x150 Roll", 148.0, "medium"),
        ("Drainable Housewrap 9x100 Roll", 132.0, "medium"),
        ("High Perm Housewrap 5x200 Roll", 186.0, "medium"),
        ("Housewrap Job Pack Pallet", 438.0, "heavy"),
    ]:
        add_product(name, "Exterior", "Housewrap", cost + rng.uniform(-6.0, 6.0), rng.integers(3, 12), weight)

    for name, cost in [
        ("Paintable Exterior Caulk Tube - White", 8.5),
        ("Silicone Exterior Sealant Tube - Clear", 10.5),
        ("Polyurethane Window Sealant - Bronze", 12.5),
        ("Siding Joint Sealant - Almond", 9.5),
        ("Roof Flashing Sealant Tube - Black", 11.0),
    ]:
        add_product(name, "Exterior", "Caulk", cost + rng.uniform(-0.75, 0.75), rng.integers(3, 8), "light")

    for name, cost, weight in [
        ("Roofing Nail Box 1.25in", 38.0, "medium"),
        ("Coil Siding Nail Box 2in", 64.0, "medium"),
        ("Stainless Trim Screw Box", 42.0, "light"),
        ("Cap Nail Box for Housewrap", 28.0, "light"),
        ("Fiber Cement Screw Box", 78.0, "medium"),
    ]:
        add_product(name, "Exterior", "Fasteners", cost + rng.uniform(-3.0, 3.0), rng.integers(3, 10), weight)

    for name, cost in [
        ("Composite Corner Board 4in White", 42.0),
        ("Composite Corner Board 6in White", 58.0),
        ("PVC Corner Board 4in Smooth", 64.0),
        ("PVC Corner Board 6in Smooth", 86.0),
        ("Primed Wood Corner Board 5/4x6", 36.0),
    ]:
        add_product(name, "Exterior", "Corner Boards", cost + rng.uniform(-4.0, 4.0), rng.integers(5, 15), "medium")

    products = pd.DataFrame.from_records(records)
    specialty_mask = products["subcategory"].isin(["Flashing", "Fiber Cement", "Corner Posts", "Corner Boards"])
    slow_weights = np.where(specialty_mask, 2.0, 1.0)
    slow_indices = rng.choice(products.index.to_numpy(), size=12, replace=False, p=slow_weights / slow_weights.sum())
    products.loc[slow_indices, "is_slow_mover"] = True
    return products[PRODUCT_COLUMNS]


def _weekly_dates() -> pd.DatetimeIndex:
    """Return Monday week starts from ISO 2020-W01 through ISO 2024-W52."""
    return pd.date_range("2019-12-30", "2024-12-23", freq="W-MON")


def _generate_sales_history(
    branches: pd.DataFrame,
    products: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate weekly branch-SKU sales with seasonality, shocks, stockouts, and bursts."""
    weeks = _weekly_dates()
    panel = pd.MultiIndex.from_product(
        [weeks, branches["branch_id"], products["sku_id"]],
        names=["week_start_date", "branch_id", "sku_id"],
    ).to_frame(index=False)

    branch_features = branches.copy()
    branch_features["branch_scalar"] = branch_features["density"].map({"urban": 1.35, "suburban": 1.00, "rural": 0.68})
    branch_features["branch_scalar"] *= branch_features["branch_id"].map(
        {
            "B001": 1.08,
            "B002": 0.96,
            "B003": 0.90,
            "B004": 1.12,
            "B005": 1.00,
            "B006": 0.88,
            "B007": 1.18,
            "B008": 1.05,
            "B009": 1.02,
            "B010": 1.20,
            "B011": 1.04,
            "B012": 0.82,
        }
    )

    product_features = products[["sku_id", "category", "subcategory", "unit_cost", "is_slow_mover"]].copy()
    product_features["category_mean"] = product_features["category"].map({"Roofing": 10.5, "Siding": 7.5, "Exterior": 4.8})
    product_features["subcategory_scalar"] = product_features["subcategory"].map(
        {
            "Asphalt Shingles": 1.65,
            "Ridge Caps": 0.62,
            "Underlayment": 0.78,
            "Ice/Water Shield": 0.46,
            "Flashing": 0.38,
            "Vinyl Panels": 1.32,
            "Fiber Cement": 0.72,
            "Trim": 0.58,
            "Corner Posts": 0.42,
            "Housewrap": 0.95,
            "Caulk": 0.90,
            "Fasteners": 1.05,
            "Corner Boards": 0.45,
        }
    )
    product_features["sku_scalar"] = rng.lognormal(mean=0.0, sigma=0.22, size=len(product_features))
    product_features["slow_lambda"] = rng.uniform(0.3, 0.8, size=len(product_features))

    sales = panel.merge(branch_features, on="branch_id", how="left").merge(product_features, on="sku_id", how="left")
    row_count = len(sales)

    month = sales["week_start_date"].dt.month.to_numpy()
    seasonality = np.ones(row_count)
    for category, profile in CATEGORY_SEASONALITY.items():
        mask = sales["category"].eq(category).to_numpy()
        seasonality[mask] = np.take(profile, month[mask] - 1)

    climate_adjustment = np.ones(row_count)
    cold = sales["climate_zone"].eq("cold").to_numpy()
    hot = sales["climate_zone"].eq("hot").to_numpy()
    south = sales["region"].isin(["Southeast", "South"]).to_numpy()
    roofing = sales["category"].eq("Roofing").to_numpy()
    ice_water = sales["subcategory"].eq("Ice/Water Shield").to_numpy()
    housewrap = sales["subcategory"].eq("Housewrap").to_numpy()
    climate_adjustment[cold & roofing] *= 1.08
    climate_adjustment[cold & ice_water] *= 1.70
    climate_adjustment[hot & housewrap] *= 1.45
    climate_adjustment[south & roofing] *= 1.12

    dates = sales["week_start_date"]
    covid_adjustment = np.ones(row_count)
    covid_adjustment[dates.between("2020-03-02", "2020-06-29").to_numpy()] = 0.75
    covid_adjustment[dates.between("2020-07-06", "2021-12-27").to_numpy()] = 1.30

    burst_candidates = products.loc[
        (~products["is_slow_mover"])
        & products["subcategory"].isin(["Asphalt Shingles", "Vinyl Panels", "Fiber Cement", "Housewrap"]),
        "sku_id",
    ].to_numpy()
    burst_skus = set(rng.choice(burst_candidates, size=20, replace=False))
    burst_mask = (
        sales["branch_id"].isin(BURSTY_BRANCH_IDS).to_numpy()
        & sales["sku_id"].isin(burst_skus).to_numpy()
        & (rng.random(row_count) < 0.05)
    )
    burst_adjustment = np.ones(row_count)
    burst_adjustment[burst_mask] = rng.uniform(2.0, 4.0, size=burst_mask.sum())

    expected_units = (
        sales["category_mean"].to_numpy()
        * sales["subcategory_scalar"].to_numpy()
        * sales["sku_scalar"].to_numpy()
        * sales["branch_scalar"].to_numpy()
        * seasonality
        * climate_adjustment
        * covid_adjustment
        * burst_adjustment
    )
    noisy_expected = np.clip(expected_units * rng.lognormal(mean=0.0, sigma=0.15, size=row_count), 0.0, None)
    normal_units = rng.poisson(noisy_expected)
    slow_units = rng.poisson(sales["slow_lambda"].to_numpy())
    demand_units = np.where(sales["is_slow_mover"].to_numpy(), slow_units, normal_units)

    stockout_flag = rng.random(row_count) < 0.10
    stockout_ratio = rng.uniform(0.25, 0.35, size=row_count)
    units_sold = np.where(stockout_flag, np.floor(demand_units * stockout_ratio), demand_units).astype(int)
    revenue = np.round(units_sold * sales["unit_cost"].to_numpy() * MARKUP, 2)

    output = pd.DataFrame(
        {
            "week_start_date": sales["week_start_date"],
            "branch_id": sales["branch_id"],
            "sku_id": sales["sku_id"],
            "units_sold": units_sold,
            "revenue": revenue,
            "stockout_flag": stockout_flag,
        }
    )
    return output


def _generate_inventory_snapshots(
    sales_history: pd.DataFrame,
    products: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create weekly inventory snapshots using trailing demand and sawtooth replenishment."""
    inventory = sales_history[["week_start_date", "branch_id", "sku_id", "units_sold"]].copy()
    inventory = inventory.rename(columns={"week_start_date": "snapshot_date"})
    inventory = inventory.merge(products[["sku_id", "lead_time_days"]], on="sku_id", how="left")
    inventory = inventory.sort_values(["branch_id", "sku_id", "snapshot_date"]).reset_index(drop=True)

    grouped = inventory.groupby(["branch_id", "sku_id"], sort=False)
    trailing_avg = grouped["units_sold"].rolling(window=13, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    inventory["reorder_point"] = np.ceil(trailing_avg.to_numpy() * 1.5).astype(int)

    group_code = grouped.ngroup().to_numpy()
    week_index = grouped.cumcount().to_numpy()
    lead_time_by_group = grouped["lead_time_days"].first().to_numpy()
    cycle_by_group = np.ceil(lead_time_by_group / 7).astype(int) + rng.integers(2, 6, size=len(lead_time_by_group))
    offset_by_group = rng.integers(0, cycle_by_group)
    cycle = cycle_by_group[group_code]
    phase = (week_index + offset_by_group[group_code]) % cycle
    sawtooth = 1.9 - (phase / np.maximum(cycle - 1, 1)) * 1.35
    reorder_point = inventory["reorder_point"].to_numpy()
    noise = rng.normal(0.0, np.maximum(reorder_point * 0.12, 0.75), size=len(inventory))
    on_hand_units = np.ceil(reorder_point * sawtooth + inventory["units_sold"].to_numpy() * 0.25 + noise)
    inventory["on_hand_units"] = np.clip(on_hand_units, 0, None).astype(int)

    return inventory[INVENTORY_COLUMNS].sort_values(["snapshot_date", "branch_id", "sku_id"], ignore_index=True)


def _generate_contractors(branches: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate contractor accounts with spend tiers weighted toward bursty branches."""
    tiers = np.array(["A"] * 30 + ["B"] * 70 + ["C"] * 100)
    rng.shuffle(tiers)

    density_weight = branches["density"].map({"urban": 1.35, "suburban": 1.0, "rural": 0.65}).to_numpy(dtype=float)
    burst_weight = np.where(branches["branch_id"].isin(BURSTY_BRANCH_IDS), 2.4, 1.0)
    branch_ids = branches["branch_id"].to_numpy()

    assigned_branches = np.empty(len(tiers), dtype=object)
    tier_a_mask = tiers == "A"
    tier_a_weights = density_weight * burst_weight
    assigned_branches[tier_a_mask] = rng.choice(branch_ids, size=tier_a_mask.sum(), p=tier_a_weights / tier_a_weights.sum())
    assigned_branches[~tier_a_mask] = rng.choice(branch_ids, size=(~tier_a_mask).sum(), p=density_weight / density_weight.sum())

    trade_types = rng.choice(["roofer", "sider", "GC"], size=len(tiers), p=[0.45, 0.35, 0.20])
    surnames = ["Anderson", "Bennett", "Carter", "Diaz", "Edwards", "Foster", "Garcia", "Hughes", "Iverson", "Jenkins", "Kaplan", "Lewis", "Miller", "Nguyen", "Ortiz", "Patel", "Quinn", "Reed", "Santos", "Turner"]
    suffixes = {"roofer": ["Roofing", "Exteriors", "Storm Repair"], "sider": ["Siding", "Exterior Systems", "Cladding"], "GC": ["Builders", "Construction", "Renovations"]}
    names = [
        f"{surnames[i % len(surnames)]} {suffixes[trade][(i // len(surnames)) % len(suffixes[trade])]}"
        for i, trade in enumerate(trade_types)
    ]

    account_age_years = np.empty(len(tiers), dtype=int)
    for tier, low, high in [("A", 5, 26), ("B", 2, 19), ("C", 0, 13)]:
        tier_mask = tiers == tier
        account_age_years[tier_mask] = rng.integers(low, high, size=tier_mask.sum())

    return pd.DataFrame(
        {
            "contractor_id": [f"C{i:04d}" for i in range(1, len(tiers) + 1)],
            "branch_id": assigned_branches,
            "name": names,
            "trade_type": trade_types,
            "annual_spend_tier": tiers,
            "account_age_years": account_age_years,
        }
    )


def _validate_outputs(
    branches: pd.DataFrame,
    products: pd.DataFrame,
    sales_history: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    contractors: pd.DataFrame,
) -> None:
    """Assert output schemas, sizes, date ranges, and core distribution requirements."""
    weeks = _weekly_dates()
    expected_panel_rows = len(weeks) * len(branches) * len(products)

    assert branches.columns.tolist() == BRANCH_COLUMNS
    assert len(branches) == 12
    assert branches["region"].value_counts().to_dict() == {"Northeast": 3, "Midwest": 3, "Southeast": 3, "South": 3}
    assert branches["climate_zone"].value_counts().to_dict() == {"cold": 6, "hot": 3, "mixed": 3}
    assert branches["density"].value_counts().to_dict() == {"suburban": 5, "urban": 4, "rural": 3}

    assert products.columns.tolist() == PRODUCT_COLUMNS
    assert len(products) == 80
    assert products["category"].value_counts().to_dict() == {"Roofing": 35, "Siding": 25, "Exterior": 20}
    assert products["is_slow_mover"].sum() == 12
    assert products["unit_cost"].between(8, 450).all()
    assert products["lead_time_days"].between(3, 21).all()
    assert products["seasonality_profile"].map(len).eq(12).all()

    assert sales_history.columns.tolist() == SALES_COLUMNS
    assert len(sales_history) == expected_panel_rows
    assert pd.api.types.is_datetime64_any_dtype(sales_history["week_start_date"])
    assert sales_history["week_start_date"].min() == weeks.min()
    assert sales_history["week_start_date"].max() == weeks.max()
    assert sales_history["units_sold"].ge(0).all()
    assert sales_history["revenue"].ge(0).all()
    assert sales_history["stockout_flag"].mean() > 0.08
    assert sales_history["stockout_flag"].mean() < 0.12

    assert inventory_snapshots.columns.tolist() == INVENTORY_COLUMNS
    assert len(inventory_snapshots) == expected_panel_rows
    assert pd.api.types.is_datetime64_any_dtype(inventory_snapshots["snapshot_date"])
    assert inventory_snapshots["snapshot_date"].min() == weeks.min()
    assert inventory_snapshots["snapshot_date"].max() == weeks.max()
    assert inventory_snapshots["on_hand_units"].ge(0).all()
    assert inventory_snapshots["reorder_point"].ge(0).all()

    assert contractors.columns.tolist() == CONTRACTOR_COLUMNS
    assert len(contractors) == 200
    assert contractors["annual_spend_tier"].value_counts().to_dict() == {"C": 100, "B": 70, "A": 30}
    assert contractors["branch_id"].isin(branches["branch_id"]).all()
    assert set(contractors["trade_type"]) == {"roofer", "sider", "GC"}


def _write_parquet_outputs(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write all generated tables as pyarrow-backed parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, table in tables.items():
        output_path = output_dir / f"{table_name}.parquet"
        table.to_parquet(output_path, engine="pyarrow", index=False)
        LOGGER.info("Wrote %s (%s rows)", output_path, f"{len(table):,}")


def _log_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Log row counts, date ranges, and compact samples for generated outputs."""
    LOGGER.info("Summary stats")
    for table_name, table in tables.items():
        LOGGER.info("%s rows: %s", table_name, f"{len(table):,}")
        date_columns = [column for column in table.columns if column.endswith("_date")]
        if date_columns:
            date_column = date_columns[0]
            LOGGER.info(
                "%s %s range: %s to %s",
                table_name,
                date_column,
                table[date_column].min().date(),
                table[date_column].max().date(),
            )
        LOGGER.info("%s sample:\n%s", table_name, table.head(3).to_string(index=False))


def main() -> None:
    """Generate, validate, and save all synthetic raw parquet datasets."""
    _configure_logging()
    rng = np.random.default_rng(SEED)

    branches = _generate_branches()
    products = _generate_products(rng)
    sales_history = _generate_sales_history(branches, products, rng)
    inventory_snapshots = _generate_inventory_snapshots(sales_history, products, rng)
    contractors = _generate_contractors(branches, rng)

    _validate_outputs(branches, products, sales_history, inventory_snapshots, contractors)
    tables = {
        "branches": branches,
        "products": products,
        "sales_history": sales_history,
        "inventory_snapshots": inventory_snapshots,
        "contractors": contractors,
    }
    _write_parquet_outputs(tables, RAW_DATA_DIR)
    _log_summary(tables)


if __name__ == "__main__":
    main()
