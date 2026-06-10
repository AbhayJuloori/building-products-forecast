"""Plotly Dash app for the Building Products Demand Forecasting project."""

from __future__ import annotations

import itertools
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

ROOT = Path(__file__).resolve().parents[2]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
LOGGER = logging.getLogger(__name__)
TEMPLATE, HEIGHT, ALL_BRANCHES = "plotly_dark", 400, "__all__"
LIGHTGBM_MODEL = "lightgbm_global"
CATEGORY_COLORS = {"Roofing": "#F59E0B", "Siding": "#10B981", "Exterior": "#3B82F6"}
BIAS_COLORS = {"Over-forecast": "#F59E0B", "Under-forecast": "#3B82F6"}
BRANCH_COORDS = {
    "B001": (42.36, -71.06), "B002": (41.76, -72.67), "B003": (42.65, -73.76),
    "B004": (41.88, -87.63), "B005": (42.33, -83.05), "B006": (41.59, -93.62),
    "B007": (33.75, -84.39), "B008": (35.23, -80.84), "B009": (27.95, -82.46),
    "B010": (32.78, -96.80), "B011": (36.16, -86.78), "B012": (34.75, -92.29),
}
TABLE_STYLE = {
    "style_as_list_view": True,
    "style_table": {"overflowX": "auto"},
    "style_header": {"backgroundColor": "#111827", "color": "#F9FAFB", "fontWeight": "700"},
    "style_data": {"backgroundColor": "#0F172A", "color": "#E5E7EB", "border": "1px solid #334155"},
    "style_cell": {"fontFamily": "Arial", "fontSize": 13, "padding": "8px", "minWidth": "130px"},
}
TAB_STYLE = {"backgroundColor": "#111827", "borderColor": "#374151", "color": "#9CA3AF"}
ACTIVE_TAB_STYLE = {"backgroundColor": "#1F2937", "borderColor": "#F59E0B", "color": "#F9FAFB", "fontWeight": "700"}
DROPDOWN_STYLE = {"color": "#111827"}

DATASETS: dict[str, tuple[Path, list[str]]] = {
    "branches": (RAW / "branches.parquet", ["branch_id", "name", "region", "climate_zone", "density"]),
    "products": (RAW / "products.parquet", ["sku_id", "name", "category", "subcategory", "unit_cost", "lead_time_days", "is_slow_mover"]),
    "sales_history": (RAW / "sales_history.parquet", ["week_start_date", "branch_id", "sku_id", "units_sold", "revenue", "stockout_flag"]),
    "contractors": (RAW / "contractors.parquet", ["contractor_id", "branch_id", "name", "trade_type", "annual_spend_tier", "account_age_years"]),
    "modeling_table": (PROCESSED / "modeling_table.parquet", ["branch_id", "sku_id", "week_start_date", "units_sold", "category", "region"]),
    "forecasts_test": (PROCESSED / "forecasts_test.parquet", ["model", "branch_id", "sku_id", "week_start_date", "y_true", "y_pred"]),
    "forecast_evaluation": (PROCESSED / "forecast_evaluation.parquet", ["model", "branch_id", "sku_id", "rmse", "mae", "mape", "bias", "fva_vs_naive"]),
    "sku_abc_xyz": (PROCESSED / "sku_abc_xyz.parquet", ["sku_id", "category", "total_revenue", "revenue_share", "abc_class", "demand_cv", "xyz_class"]),
    "inventory_scenarios": (PROCESSED / "inventory_scenarios.parquet", ["branch_id", "sku_id", "scenario", "service_level", "lead_time_days", "avg_weekly_demand", "demand_std", "safety_stock_units", "reorder_point_units", "eoq_units", "safety_stock_cost", "on_hand_units", "current_excess_units", "excess_inventory_cost", "projected_stockout_risk_pct", "scenario_total_inventory_cost"]),
    "contractor_segments": (PROCESSED / "contractor_segments.parquet", ["contractor_id", "branch_id", "cluster_id", "segment_label", "total_spend", "order_frequency", "avg_order_size", "product_breadth", "recency_days"]),
    "branch_demand_clusters": (PROCESSED / "branch_demand_clusters.parquet", ["branch_id", "cluster_id", "cluster_label"]),
    "slow_mover_flags": (PROCESSED / "slow_mover_flags.parquet", ["sku_id", "branch_id", "on_hand_units", "weeks_of_forecast_coverage", "excess_units", "excess_cost_dollars", "flag_reason"]),
}


def configure_logging() -> None:
    """Configure console logging when the app starts."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def empty_df(name: str) -> pd.DataFrame:
    """Return an empty frame with expected columns."""
    return pd.DataFrame(columns=DATASETS[name][1])


@lru_cache(maxsize=None)
def load_df(name: str) -> pd.DataFrame:
    """Read a parquet file, or log and return an empty schema-compatible frame."""
    path, _ = DATASETS[name]
    if not path.exists():
        LOGGER.error("Missing parquet for %s: %s", name, path)
        return empty_df(name)
    try:
        frame = pd.read_parquet(path)
    except Exception:
        LOGGER.exception("Failed to load parquet for %s: %s", name, path)
        return empty_df(name)
    if name == "forecast_evaluation" and "fva_vs_naive" not in frame and "forecast_value_add_vs_naive" in frame:
        frame = frame.rename(columns={"forecast_value_add_vs_naive": "fva_vs_naive"})
    LOGGER.info("Loaded %s rows=%s columns=%s", name, len(frame), len(frame.columns))
    return frame


def dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Parse date columns when present."""
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_datetime(output[column], errors="coerce")
    return output


def opts(values: list[str]) -> list[dict[str, str]]:
    """Build Dash options from strings."""
    return [{"label": value, "value": value} for value in values]


def fmt_units(value: float | int | None) -> str:
    """Format unit counts."""
    return "0" if value is None or not np.isfinite(value) else f"{value:,.0f}"


def fmt_money(value: float | int | None) -> str:
    """Format dollar values."""
    if value is None or not np.isfinite(value):
        return "$0"
    return f"${value / 1_000_000:,.1f}M" if abs(value) >= 1_000_000 else f"${value / 1_000:,.0f}K" if abs(value) >= 1_000 else f"${value:,.0f}"


def fmt_pct(value: float | int | None) -> str:
    """Format percentage deltas."""
    return "0.0%" if value is None or not np.isfinite(value) else f"{value:+.1f}%"


def finish_fig(fig: go.Figure, title: str) -> go.Figure:
    """Apply shared dark chart layout."""
    fig.update_layout(template=TEMPLATE, height=HEIGHT, title=title, margin={"l": 32, "r": 24, "t": 58, "b": 36}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    return fig


def empty_fig(title: str, message: str = "No data available") -> go.Figure:
    """Create an empty dark chart."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 16, "color": "#9CA3AF"})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return finish_fig(fig, title)


def kpi(title: str, value: str, subtitle: str = "") -> dbc.Card:
    """Build a KPI card."""
    return dbc.Card(dbc.CardBody([html.Div(title, className="text-muted small text-uppercase"), html.Div(value, className="display-6 fw-bold"), html.Div(subtitle, className="text-muted small")]), className="h-100 border-secondary", color="dark", inverse=True)


def control(label: str, component: Any) -> list[Any]:
    """Build a labeled control."""
    return [html.Label(label, className="text-muted small"), component]


def table(columns: list[str], table_id: str | None = None, data: list[dict[str, Any]] | None = None) -> dash_table.DataTable:
    """Build a styled Dash data table."""
    kwargs: dict[str, Any] = {"columns": [{"name": col, "id": col} for col in columns], "page_size": 25, "sort_action": "native", **TABLE_STYLE}
    if table_id:
        kwargs["id"] = table_id
    if data is not None:
        kwargs["data"] = data
    return dash_table.DataTable(**kwargs)


def tab(label: str, children: list[Any]) -> dbc.Tab:
    """Build a consistently styled tab."""
    return dbc.Tab(label=label, tab_style=TAB_STYLE, active_tab_style=ACTIVE_TAB_STYLE, children=children)


def best_model_rows(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest-RMSE model for each branch/SKU."""
    cols = ["branch_id", "sku_id", "model", "rmse", "mae", "mape", "bias"]
    if evaluation.empty or not set(cols).issubset(evaluation):
        return pd.DataFrame(columns=cols)
    ranked = evaluation[cols].copy()
    ranked["rmse_sort"] = pd.to_numeric(ranked["rmse"], errors="coerce").fillna(np.inf)
    ranked["tie_sort"] = np.where(ranked["model"].eq(LIGHTGBM_MODEL), 0, 1)
    ranked = ranked.sort_values(["branch_id", "sku_id", "rmse_sort", "tie_sort", "model"], kind="mergesort")
    return ranked.groupby(["branch_id", "sku_id"], as_index=False, observed=True).first()[cols]


def selected_forecast_rows(forecasts: pd.DataFrame, best: pd.DataFrame, products: pd.DataFrame, branches: pd.DataFrame) -> pd.DataFrame:
    """Filter forecasts to the chosen model and attach metadata."""
    cols = ["branch_id", "sku_id", "model", "week_start_date", "y_true", "y_pred", "category", "sku_name", "branch_name", "region"]
    if forecasts.empty or best.empty or not {"model", "branch_id", "sku_id", "week_start_date", "y_true", "y_pred"}.issubset(forecasts):
        return pd.DataFrame(columns=cols)
    out = forecasts.merge(best[["branch_id", "sku_id", "model"]], on=["branch_id", "sku_id", "model"])
    out = out.merge(products[["sku_id", "name", "category"]].rename(columns={"name": "sku_name"}), on="sku_id", how="left")
    out = out.merge(branches[["branch_id", "name", "region"]].rename(columns={"name": "branch_name"}), on="branch_id", how="left")
    out["week_start_date"] = pd.to_datetime(out["week_start_date"], errors="coerce")
    return out[cols]


def line_rows(selected: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selected forecasts to branch/category/week."""
    cols = ["branch_id", "category", "week_start_date", "y_true", "y_pred"]
    if selected.empty:
        return pd.DataFrame(columns=cols)
    return selected.groupby(["branch_id", "category", "week_start_date"], as_index=False, observed=True)[["y_true", "y_pred"]].sum().sort_values(["branch_id", "category", "week_start_date"], kind="mergesort")


def overview_kpis(selected: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame, best: pd.DataFrame) -> list[dbc.Card]:
    """Compute static overview KPI cards."""
    if selected.empty:
        return [kpi("Total Forecasted Units Next 13W", "0"), kpi("Forecast vs Prior Year", "0.0%"), kpi("Top SKU Next 4W", "N/A"), kpi("Avg MAPE", "0.0%")]
    weeks = sorted(selected["week_start_date"].dropna().unique())
    next13, next4 = selected[selected["week_start_date"].isin(weeks[:13])], selected[selected["week_start_date"].isin(weeks[:4])]
    total_forecast = float(next13["y_pred"].sum())
    prior = 0.0
    if not sales.empty and {"week_start_date", "units_sold"}.issubset(sales):
        prior = float(sales[sales["week_start_date"].isin(pd.Series(weeks[:13]).sub(pd.Timedelta(days=364)))]["units_sold"].sum())
    top_sku = "N/A"
    if not next4.empty:
        top = next4.groupby("sku_id", as_index=False, observed=True)["y_pred"].sum().nlargest(1, "y_pred")
        if not top.empty:
            sku_id = str(top.iloc[0]["sku_id"])
            sku_name = products.loc[products["sku_id"].eq(sku_id), "name"].head(1)
            top_sku = f"{sku_id} - {sku_name.iloc[0]}" if not sku_name.empty else sku_id
    avg_mape = pd.to_numeric(best.get("mape", pd.Series(dtype=float)), errors="coerce").mean()
    return [
        kpi("Total Forecasted Units Next 13W", fmt_units(total_forecast)),
        kpi("Forecast vs Prior Year", fmt_pct((total_forecast - prior) / prior * 100 if prior else 0.0)),
        kpi("Top SKU Next 4W", top_sku),
        kpi("Avg MAPE", f"{avg_mape:.1f}%" if np.isfinite(avg_mape) else "0.0%"),
    ]


def map_rows(selected: pd.DataFrame, branches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 13-week forecast units by branch and attach coordinates."""
    cols = ["branch_id", "branch_name", "region", "forecast_units_13w", "lat", "lon"]
    if selected.empty or branches.empty:
        return pd.DataFrame(columns=cols)
    weeks = sorted(selected["week_start_date"].dropna().unique())[:13]
    out = selected[selected["week_start_date"].isin(weeks)].groupby("branch_id", as_index=False, observed=True)["y_pred"].sum().rename(columns={"y_pred": "forecast_units_13w"})
    out = out.merge(branches[["branch_id", "name", "region"]], on="branch_id", how="left").rename(columns={"name": "branch_name"})
    out["lat"] = out["branch_id"].map(lambda bid: BRANCH_COORDS.get(bid, (np.nan, np.nan))[0])
    out["lon"] = out["branch_id"].map(lambda bid: BRANCH_COORDS.get(bid, (np.nan, np.nan))[1])
    return out[cols]


def map_fig(rows: pd.DataFrame) -> go.Figure:
    """Build the branch scatter_geo map."""
    if rows.empty:
        return empty_fig("Forecasted Units Next 13 Weeks by Branch")
    fig = px.scatter_geo(rows, lat="lat", lon="lon", scope="usa", size="forecast_units_13w", color="region", hover_name="branch_name", hover_data={"forecast_units_13w": ":,.0f", "lat": False, "lon": False}, size_max=34, template=TEMPLATE)
    fig.update_geos(bgcolor="rgba(0,0,0,0)", lakecolor="#111827", landcolor="#1F2937", subunitcolor="#4B5563")
    return finish_fig(fig, "Forecasted Units Next 13 Weeks by Branch")


def error_rows(best: pd.DataFrame, products: pd.DataFrame, branches: pd.DataFrame) -> pd.DataFrame:
    """Attach metadata to selected-model evaluation rows."""
    cols = ["branch_id", "sku_id", "model", "rmse", "mae", "mape", "bias", "category", "sku_name", "region"]
    if best.empty:
        return pd.DataFrame(columns=cols)
    out = best.merge(products[["sku_id", "name", "category"]].rename(columns={"name": "sku_name"}), on="sku_id", how="left")
    out = out.merge(branches[["branch_id", "region"]], on="branch_id", how="left")
    return out[cols]


def subsets(values: list[str]) -> list[tuple[str, ...]]:
    """Return empty all-values sentinel plus all non-empty subsets."""
    return [tuple(), *(combo for size in range(1, len(values) + 1) for combo in itertools.combinations(values, size))]


def key(selected: list[str] | str | None, all_values: list[str]) -> tuple[str, ...]:
    """Normalize a multi-select value to a cache key."""
    if selected is None or selected == []:
        return tuple()
    values = [selected] if isinstance(selected, str) else list(selected)
    return tuple(value for value in all_values if value in values)


def product_cache(errors: pd.DataFrame, selected: pd.DataFrame, abc_xyz: pd.DataFrame, regions: list[str], categories: list[str]) -> dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, pd.DataFrame]]:
    """Precompute product-mix aggregates for every filter combination."""
    cache: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, pd.DataFrame]] = {}
    actual_base = selected[["branch_id", "sku_id", "y_true", "category", "sku_name", "region"]].copy()
    for region_key in subsets(regions):
        for category_key in subsets(categories):
            err, actual, abc = errors.copy(), actual_base.copy(), abc_xyz.copy()
            if region_key:
                err, actual = err[err["region"].isin(region_key)], actual[actual["region"].isin(region_key)]
            if category_key:
                err, actual, abc = err[err["category"].isin(category_key)], actual[actual["category"].isin(category_key)], abc[abc["category"].isin(category_key)]
            if err.empty:
                bias = pd.DataFrame(columns=["sku_id", "sku_label", "bias", "abs_bias", "bias_sign"])
                mape = pd.DataFrame(columns=["sku_id", "sku_name", "category", "avg_actual_demand", "mape"])
            else:
                bias = err.groupby(["sku_id", "sku_name"], as_index=False, observed=True)["bias"].mean()
                bias = bias.assign(abs_bias=bias["bias"].abs()).sort_values("abs_bias", ascending=False, kind="mergesort").head(20)
                bias["bias_sign"] = np.where(bias["bias"].ge(0), "Over-forecast", "Under-forecast")
                bias["sku_label"] = bias["sku_id"] + " - " + bias["sku_name"].fillna("")
                mape = err.groupby(["sku_id", "sku_name", "category"], as_index=False, observed=True)["mape"].mean()
                demand = actual.groupby("sku_id", as_index=False, observed=True)["y_true"].mean()
                mape = mape.merge(demand, on="sku_id", how="left").rename(columns={"y_true": "avg_actual_demand"})
                mape["avg_actual_demand"] = pd.to_numeric(mape["avg_actual_demand"], errors="coerce").clip(lower=0.1)
            heat = abc.groupby(["abc_class", "xyz_class"], as_index=False, observed=True)["sku_id"].count() if not abc.empty else pd.DataFrame(columns=["abc_class", "xyz_class", "sku_id"])
            heat = heat.pivot(index="abc_class", columns="xyz_class", values="sku_id").reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0).fillna(0).astype(int)
            cache[(region_key, category_key)] = {"bias": bias, "mape": mape, "abc": heat}
    return cache


def line_fig(rows: pd.DataFrame, branch_id: str | None, category: str | None) -> go.Figure:
    """Build actual versus forecast line chart."""
    if not branch_id or not category:
        return empty_fig("Actual vs Forecast")
    data = rows[rows["branch_id"].eq(branch_id) & rows["category"].eq(category)]
    if data.empty:
        return empty_fig("Actual vs Forecast", "No branch/category forecast history available")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["week_start_date"], y=data["y_true"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=data["week_start_date"], y=data["y_pred"], mode="lines+markers", name="Forecast"))
    fig.update_xaxes(title_text="Week")
    fig.update_yaxes(title_text="Units")
    return finish_fig(fig, "Actual vs Forecast by Branch and Category")


def bias_fig(rows: pd.DataFrame) -> go.Figure:
    """Build top-bias SKU bar chart."""
    if rows.empty:
        return empty_fig("Top 20 SKUs by Forecast Bias")
    data = rows.sort_values("abs_bias", ascending=True, kind="mergesort")
    fig = px.bar(data, x="bias", y="sku_label", color="bias_sign", orientation="h", color_discrete_map=BIAS_COLORS, template=TEMPLATE, hover_data={"bias": ":,.2f", "abs_bias": ":,.2f", "sku_label": False})
    fig.update_xaxes(title_text="Average Bias")
    fig.update_yaxes(title_text="")
    return finish_fig(fig, "Top 20 SKUs by Forecast Bias")


def mape_fig(rows: pd.DataFrame) -> go.Figure:
    """Build MAPE versus average demand scatter."""
    if rows.empty:
        return empty_fig("MAPE vs Average Actual Demand")
    fig = px.scatter(rows, x="avg_actual_demand", y="mape", color="category", color_discrete_map=CATEGORY_COLORS, log_x=True, hover_name="sku_name", hover_data={"sku_id": True, "avg_actual_demand": ":,.1f", "mape": ":,.1f"}, template=TEMPLATE)
    fig.update_xaxes(title_text="Average Actual Weekly Demand")
    fig.update_yaxes(title_text="Average MAPE")
    return finish_fig(fig, "MAPE vs Average Actual Demand")


def abc_fig(rows: pd.DataFrame) -> go.Figure:
    """Build ABC-XYZ heatmap."""
    fig = px.imshow(rows, text_auto=True, color_continuous_scale="Viridis", aspect="auto", template=TEMPLATE)
    fig.update_xaxes(title_text="XYZ Class")
    fig.update_yaxes(title_text="ABC Class")
    return finish_fig(fig, "ABC-XYZ SKU Count")


def inventory_calcs(inventory: pd.DataFrame) -> pd.DataFrame:
    """Add shortage and suggested order quantities."""
    out = inventory.copy()
    if out.empty:
        out["shortage_units"], out["suggested_order_qty"] = pd.Series(dtype=float), pd.Series(dtype=float)
        return out
    shortage = (pd.to_numeric(out["reorder_point_units"], errors="coerce") - pd.to_numeric(out["on_hand_units"], errors="coerce")).clip(lower=0).fillna(0)
    eoq = pd.to_numeric(out["eoq_units"], errors="coerce").clip(lower=0).fillna(0)
    out["shortage_units"] = shortage
    out["suggested_order_qty"] = np.ceil(np.where(eoq.gt(0), np.ceil(shortage / eoq.replace(0, np.nan)) * eoq, shortage)).astype(int)
    return out


def inventory_cache(inventory: pd.DataFrame, branches: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """Precompute inventory KPI and reorder table data."""
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    cols = ["branch_id", "sku_id", "on_hand_units", "reorder_point_units", "eoq_units", "suggested_order_qty", "scenario_total_inventory_cost"]
    for scenario in ["A", "B", "C", "D"]:
        scenario_rows = inventory[inventory["scenario"].eq(scenario)] if "scenario" in inventory else inventory.iloc[0:0]
        for branch_id in [ALL_BRANCHES, *branches]:
            rows = scenario_rows if branch_id == ALL_BRANCHES else scenario_rows[scenario_rows["branch_id"].eq(branch_id)]
            kpis = {"safety_stock": float(rows["safety_stock_cost"].sum()) if "safety_stock_cost" in rows else 0.0, "service_level": float(rows["service_level"].mean()) if "service_level" in rows and not rows.empty else 0.0, "above_rop": int(rows["on_hand_units"].ge(rows["reorder_point_units"]).sum()) if not rows.empty else 0, "excess_cost": float(rows["excess_inventory_cost"].sum()) if "excess_inventory_cost" in rows else 0.0}
            actions = rows[rows["on_hand_units"].lt(rows["reorder_point_units"])].sort_values("suggested_order_qty", ascending=False, kind="mergesort").head(100) if not rows.empty else pd.DataFrame(columns=cols)
            cache[(scenario, branch_id)] = {"kpis": kpis, "actions": actions[cols] if not actions.empty else pd.DataFrame(columns=cols)}
    return cache


def inventory_cost_cache(inventory: pd.DataFrame, branches: list[str]) -> dict[str, pd.DataFrame]:
    """Precompute total inventory cost by scenario."""
    cache: dict[str, pd.DataFrame] = {}
    for branch_id in [ALL_BRANCHES, *branches]:
        rows = inventory if branch_id == ALL_BRANCHES else inventory[inventory["branch_id"].eq(branch_id)]
        costs = rows.groupby("scenario", as_index=False, observed=True)["scenario_total_inventory_cost"].sum() if not rows.empty else pd.DataFrame(columns=["scenario", "scenario_total_inventory_cost"])
        cache[branch_id] = costs.set_index("scenario").reindex(["A", "B", "C", "D"], fill_value=0).reset_index()
    return cache


def inventory_kpis(values: dict[str, float | int]) -> list[dbc.Col]:
    """Build inventory KPI columns."""
    cards = [kpi("Total Safety Stock Investment", fmt_money(float(values["safety_stock"]))), kpi("Avg Service Level", f"{float(values['service_level']) * 100:.1f}%"), kpi("# SKUs Above Reorder Point", fmt_units(int(values["above_rop"]))), kpi("Total Excess Inventory", fmt_money(float(values["excess_cost"])))]
    return [dbc.Col(card, md=3) for card in cards]


def inventory_cost_fig(costs: pd.DataFrame, selected: str) -> go.Figure:
    """Build inventory cost bar chart."""
    if costs.empty:
        return empty_fig("Total Inventory Cost by Scenario")
    colors = {scenario: ("#F59E0B" if scenario == selected else "#64748B") for scenario in ["A", "B", "C", "D"]}
    fig = px.bar(costs, x="scenario", y="scenario_total_inventory_cost", color="scenario", color_discrete_map=colors, template=TEMPLATE)
    fig.update_xaxes(title_text="Scenario")
    fig.update_yaxes(title_text="Total Inventory Cost")
    return finish_fig(fig, "Total Inventory Cost by Scenario")


def contractor_scatter(contractors: pd.DataFrame, branches: pd.DataFrame) -> go.Figure:
    """Build contractor spend scatter."""
    if contractors.empty:
        return empty_fig("Contractor Spend vs Order Frequency")
    rows = contractors.merge(branches[["branch_id", "name"]].rename(columns={"name": "branch_name"}), on="branch_id", how="left")
    rows["total_spend"] = pd.to_numeric(rows["total_spend"], errors="coerce").clip(lower=1)
    fig = px.scatter(rows, x="total_spend", y="order_frequency", color="segment_label", log_x=True, hover_data={"contractor_id": True, "branch_name": True, "total_spend": ":$,.0f", "order_frequency": ":,.1f"}, template=TEMPLATE)
    fig.update_xaxes(title_text="Total Spend")
    fig.update_yaxes(title_text="Order Frequency")
    return finish_fig(fig, "Contractor Spend vs Order Frequency")


def segment_counts(contractors: pd.DataFrame) -> go.Figure:
    """Build contractor segment counts."""
    if contractors.empty:
        return empty_fig("Contractor Segment Counts")
    counts = contractors["segment_label"].value_counts().rename_axis("segment_label").reset_index(name="contractors")
    fig = px.bar(counts, x="segment_label", y="contractors", color="segment_label", template=TEMPLATE)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Contractors")
    return finish_fig(fig, "Contractor Segment Counts")


def top5_share(contractors: pd.DataFrame, branches: pd.DataFrame, categories: list[str]) -> go.Figure:
    """Build top-five contractor share heatmap."""
    if contractors.empty or branches.empty:
        return empty_fig("Top-5 Contractor Revenue Share")
    names = branches.set_index("branch_id")["name"].to_dict()
    records = []
    for branch_id, rows in contractors.groupby("branch_id", observed=True):
        total = float(rows["total_spend"].sum())
        share = float(rows.nlargest(5, "total_spend")["total_spend"].sum() / total * 100) if total else 0.0
        records.extend({"branch": names.get(branch_id, branch_id), "category": category, "top5_share": share} for category in categories)
    matrix = pd.DataFrame(records).pivot(index="branch", columns="category", values="top5_share")
    fig = px.imshow(matrix, text_auto=".1f", color_continuous_scale="Blues", aspect="auto", template=TEMPLATE)
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Branch")
    return finish_fig(fig, "Top-5 Contractor Revenue Share (%)")


def seasonal_pattern(contractors: pd.DataFrame) -> go.Figure:
    """Build synthesized seasonal buying pattern heatmap."""
    if contractors.empty:
        return empty_fig("Seasonal Buying Pattern")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for segment, annual_frequency in contractors.groupby("segment_label", observed=True)["order_frequency"].mean().sort_index().items():
        mult = np.ones(12)
        if segment == "Seasonal Bulk":
            mult = np.array([0.6, 0.8, 1.8, 2.2, 1.9, 1.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.6])
        elif segment == "At-Risk":
            mult = np.linspace(1.35, 0.55, 12)
        elif segment == "High-Value Regular":
            mult = np.array([0.95, 0.98, 1.05, 1.05, 1.04, 1.02, 1.00, 0.99, 0.98, 0.98, 0.98, 0.98])
        rows.append(pd.Series((annual_frequency / 12.0) * mult / mult.mean(), index=months, name=segment))
    fig = px.imshow(pd.DataFrame(rows), text_auto=".1f", color_continuous_scale="Viridis", aspect="auto", template=TEMPLATE)
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Segment")
    return finish_fig(fig, "Seasonal Buying Pattern: Avg Orders per Contractor")


def excess_component(flags: pd.DataFrame) -> html.Div | dash_table.DataTable:
    """Build excess inventory table or empty state."""
    if flags.empty:
        return dbc.Alert("No excess inventory flagged at this snapshot", color="secondary", className="mb-0")
    return table(list(flags.columns), data=flags.to_dict("records"))


configure_logging()
BRANCHES = load_df("branches")
PRODUCTS = load_df("products")
SALES_HISTORY = dates(load_df("sales_history"), "week_start_date")
CONTRACTORS = load_df("contractors")
MODELING_TABLE = load_df("modeling_table")
FORECASTS_TEST = dates(load_df("forecasts_test"), "week_start_date")
FORECAST_EVALUATION = load_df("forecast_evaluation")
SKU_ABC_XYZ = load_df("sku_abc_xyz")
INVENTORY_SCENARIOS = inventory_calcs(load_df("inventory_scenarios"))
CONTRACTOR_SEGMENTS = load_df("contractor_segments")
BRANCH_DEMAND_CLUSTERS = load_df("branch_demand_clusters")
SLOW_MOVER_FLAGS = load_df("slow_mover_flags")

BRANCH_OPTIONS = BRANCHES.sort_values("branch_id")["branch_id"].tolist() if "branch_id" in BRANCHES else []
CATEGORY_OPTIONS = PRODUCTS.sort_values("category")["category"].dropna().unique().tolist() if "category" in PRODUCTS else []
REGION_OPTIONS = BRANCHES.sort_values("region")["region"].dropna().unique().tolist() if "region" in BRANCHES else []
DEFAULT_BRANCH = BRANCH_OPTIONS[0] if BRANCH_OPTIONS else None
DEFAULT_CATEGORY = CATEGORY_OPTIONS[0] if CATEGORY_OPTIONS else None

BEST_MODELS = best_model_rows(FORECAST_EVALUATION)
SELECTED_FORECASTS = selected_forecast_rows(FORECASTS_TEST, BEST_MODELS, PRODUCTS, BRANCHES)
LINE_ROWS = line_rows(SELECTED_FORECASTS)
GLOBAL_KPIS = overview_kpis(SELECTED_FORECASTS, SALES_HISTORY, PRODUCTS, BEST_MODELS)
MAP_FIG = map_fig(map_rows(SELECTED_FORECASTS, BRANCHES))
ERROR_ROWS = error_rows(BEST_MODELS, PRODUCTS, BRANCHES)
PRODUCT_CACHE = product_cache(ERROR_ROWS, SELECTED_FORECASTS, SKU_ABC_XYZ, REGION_OPTIONS, CATEGORY_OPTIONS)
INVENTORY_CACHE = inventory_cache(INVENTORY_SCENARIOS, BRANCH_OPTIONS)
INVENTORY_COST_CACHE = inventory_cost_cache(INVENTORY_SCENARIOS, BRANCH_OPTIONS)
CONTRACTOR_SCATTER = contractor_scatter(CONTRACTOR_SEGMENTS, BRANCHES)
SEGMENT_COUNTS = segment_counts(CONTRACTOR_SEGMENTS)
TOP5_SHARE = top5_share(CONTRACTOR_SEGMENTS, BRANCHES, CATEGORY_OPTIONS)
SEASONAL_PATTERN = seasonal_pattern(CONTRACTOR_SEGMENTS)

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Building Products Demand Forecasting"


def layout() -> html.Div:
    """Build the dashboard layout."""
    branch_dd = dcc.Dropdown(id="branch-dropdown", options=opts(BRANCH_OPTIONS), value=DEFAULT_BRANCH, clearable=False, style=DROPDOWN_STYLE)
    category_dd = dcc.Dropdown(id="category-dropdown", options=opts(CATEGORY_OPTIONS), value=DEFAULT_CATEGORY, clearable=False, style=DROPDOWN_STYLE)
    region_filter = dcc.Dropdown(id="region-filter", options=opts(REGION_OPTIONS), value=REGION_OPTIONS, multi=True, style=DROPDOWN_STYLE)
    category_filter = dcc.Dropdown(id="product-category-filter", options=opts(CATEGORY_OPTIONS), value=CATEGORY_OPTIONS, multi=True, style=DROPDOWN_STYLE)
    scenario_radio = dcc.RadioItems(id="scenario-radio", options=opts(["A", "B", "C", "D"]), value="A", inline=True, inputStyle={"marginRight": "6px", "marginLeft": "14px"})
    inventory_branch = dcc.Dropdown(id="inventory-branch-filter", options=[{"label": "All Branches", "value": ALL_BRANCHES}, *opts(BRANCH_OPTIONS)], value=ALL_BRANCHES, clearable=False, style=DROPDOWN_STYLE)
    reorder_cols = ["branch_id", "sku_id", "on_hand_units", "reorder_point_units", "eoq_units", "suggested_order_qty", "scenario_total_inventory_cost"]
    tabs = dbc.Tabs([
        tab("Branch Demand Overview", [
            dbc.Row([dbc.Col(card, md=3) for card in GLOBAL_KPIS], className="g-3 my-3"),
            dbc.Row([dbc.Col(dcc.Graph(figure=MAP_FIG), lg=12)], className="mb-3"),
            dbc.Row([dbc.Col(control("Branch", branch_dd), md=6), dbc.Col(control("Category", category_dd), md=6)], className="g-3 mb-2"),
            dcc.Graph(id="branch-category-line"),
        ]),
        tab("Product Mix & Forecast Error", [
            dbc.Row([dbc.Col(control("Region", region_filter), md=6), dbc.Col(control("Category", category_filter), md=6)], className="g-3 my-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="bias-bar"), lg=6), dbc.Col(dcc.Graph(id="mape-scatter"), lg=6)], className="g-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="abc-xyz-heatmap"), lg=12)], className="g-3"),
        ]),
        tab("Inventory & Replenishment", [
            dbc.Row([dbc.Col(control("Scenario", scenario_radio), md=6), dbc.Col(control("Branch", inventory_branch), md=6)], className="g-3 my-3"),
            dbc.Row(id="inventory-kpi-row", className="g-3 mb-3"),
            dbc.Row([dbc.Col([html.H5("Recommended Reorder Actions", className="mb-3"), table(reorder_cols, table_id="reorder-table")], lg=7), dbc.Col(dcc.Graph(id="inventory-cost-bar"), lg=5)], className="g-3"),
            dbc.Row([dbc.Col([html.H5("Excess Inventory Flags", className="my-3"), excess_component(SLOW_MOVER_FLAGS)])], className="mb-3"),
        ]),
        tab("Contractor Insights", [
            dbc.Row([dbc.Col(dcc.Graph(figure=CONTRACTOR_SCATTER), lg=6), dbc.Col(dcc.Graph(figure=SEGMENT_COUNTS), lg=6)], className="g-3 my-3"),
            dbc.Row([dbc.Col(dcc.Graph(figure=TOP5_SHARE), lg=6), dbc.Col(dcc.Graph(figure=SEASONAL_PATTERN), lg=6)], className="g-3 mb-3"),
        ]),
    ])
    return html.Div([
        dbc.Navbar(dbc.Container([dbc.NavbarBrand("Building Products Demand Forecasting", className="fw-bold"), html.Span("Portfolio Dashboard", className="text-muted")], fluid=True), color="dark", dark=True, className="border-bottom border-secondary"),
        dbc.Container([tabs], fluid=True, className="py-3"),
    ], style={"minHeight": "100vh", "backgroundColor": "#0B1020"})


app.layout = layout


@app.callback(Output("branch-category-line", "figure"), Input("branch-dropdown", "value"), Input("category-dropdown", "value"))
def update_line_chart(branch_id: str | None, category: str | None) -> go.Figure:
    """Update branch/category line chart from precomputed rows."""
    return line_fig(LINE_ROWS, branch_id or DEFAULT_BRANCH, category or DEFAULT_CATEGORY)


@app.callback(Output("bias-bar", "figure"), Output("mape-scatter", "figure"), Output("abc-xyz-heatmap", "figure"), Input("region-filter", "value"), Input("product-category-filter", "value"))
def update_product_mix(region_values: list[str] | None, category_values: list[str] | None) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Update product mix charts from precomputed filter caches."""
    frames = PRODUCT_CACHE.get((key(region_values, REGION_OPTIONS), key(category_values, CATEGORY_OPTIONS)), PRODUCT_CACHE.get((tuple(), tuple()), {}))
    blank_heat = pd.DataFrame(0, index=["A", "B", "C"], columns=["X", "Y", "Z"])
    return bias_fig(frames.get("bias", pd.DataFrame())), mape_fig(frames.get("mape", pd.DataFrame())), abc_fig(frames.get("abc", blank_heat))


@app.callback(Output("inventory-kpi-row", "children"), Output("reorder-table", "data"), Output("inventory-cost-bar", "figure"), Input("scenario-radio", "value"), Input("inventory-branch-filter", "value"))
def update_inventory(scenario: str | None, branch_id: str | None) -> tuple[list[dbc.Col], list[dict[str, Any]], go.Figure]:
    """Update inventory KPIs, reorder actions, and cost chart."""
    selected_scenario, selected_branch = scenario or "A", branch_id or ALL_BRANCHES
    fallback = {"kpis": {"safety_stock": 0.0, "service_level": 0.0, "above_rop": 0, "excess_cost": 0.0}, "actions": pd.DataFrame()}
    data = INVENTORY_CACHE.get((selected_scenario, selected_branch), fallback)
    costs = INVENTORY_COST_CACHE.get(selected_branch, INVENTORY_COST_CACHE.get(ALL_BRANCHES, pd.DataFrame()))
    return inventory_kpis(data["kpis"]), data["actions"].to_dict("records"), inventory_cost_fig(costs, selected_scenario)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
