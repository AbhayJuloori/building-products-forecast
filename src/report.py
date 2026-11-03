"""Write README charts and a results summary from the pipeline outputs.

Reads only the parquet files written by run_all.py and never refits anything,
so every number in docs/results.json traces to the latest pipeline run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "results.json"

LOGGER = logging.getLogger(__name__)
LABELS = {
    "seasonal_naive": "Seasonal naive",
    "lightgbm_global": "LightGBM global",
    "prophet_branch_category": "Prophet branch x category",
    "croston_tsb_slowmovers": "Croston-TSB (slow movers)",
}
COLORS = {
    "seasonal_naive": "#9aa0a6",
    "lightgbm_global": "#2f6fb0",
    "prophet_branch_category": "#d0802f",
    "croston_tsb_slowmovers": "#4f9a6a",
}
SCENARIO_NAMES = {"A": "A: 95% flat", "B": "B: 98% flat", "C": "C: 90% flat", "D": "D: ABC segmented"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


def model_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Per model averages over the branch x SKU series each model scored."""
    scored = evaluation[evaluation["n_test_weeks"] > 0]
    out = scored.groupby("model").agg(
        series=("sku_id", "size"),
        avg_rmse=("rmse", "mean"),
        avg_mae=("mae", "mean"),
        avg_mape=("mape", "mean"),
        avg_fva_vs_naive=("forecast_value_add_vs_naive", "mean"),
        median_fva_vs_naive=("forecast_value_add_vs_naive", "median"),
        share_beating_naive=("forecast_value_add_vs_naive", lambda v: float((v > 0).mean())),
    )
    return out.reindex([m for m in LABELS if m in out.index])


def scenario_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    return scenarios.groupby("scenario").agg(
        safety_stock_investment=("safety_stock_cost", "sum"),
        avg_service_level=("service_level", "mean"),
        avg_stockout_risk_pct=("projected_stockout_risk_pct", "mean"),
        total_inventory_cost=("scenario_total_inventory_cost", "sum"),
    )


def fva_chart(summary: pd.DataFrame) -> None:
    rows = summary.drop(index="seasonal_naive", errors="ignore")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    values = rows["avg_fva_vs_naive"] * 100
    ax.barh([LABELS[m] for m in rows.index], values, color=[COLORS[m] for m in rows.index])
    for y, v in enumerate(values):
        ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#555", lw=1)
    ax.set(xlabel="mean RMSE reduction vs seasonal naive per series (test weeks)",
           title="Forecast value add over the seasonal naive baseline")
    ax.set_xlim(min(values.min() - 6, -10), values.max() + 8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "forecast_value_add.png", dpi=150)
    plt.close(fig)


def actual_vs_forecast_chart(forecasts: pd.DataFrame) -> None:
    """Network total weekly units in the test window, summed over every branch x SKU."""
    weekly = forecasts.dropna(subset=["y_pred"])
    weekly = weekly[weekly["model"].isin(["seasonal_naive", "lightgbm_global", "prophet_branch_category"])]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    actual = (forecasts[forecasts["model"] == "seasonal_naive"]
              .groupby("week_start_date")["y_true"].sum())
    ax.plot(actual.index, actual.values, color="#111", lw=2, label="Actual")
    for model, g in weekly.groupby("model"):
        s = g.groupby("week_start_date")["y_pred"].sum()
        ax.plot(s.index, s.values, color=COLORS[model], lw=1.4, label=LABELS[model])
    ax.set(ylabel="units, all branches x SKUs", title="Test window: actual vs forecast, network total")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_forecast.png", dpi=150)
    plt.close(fig)


def scenario_chart(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for scenario, row in summary.iterrows():
        ax.scatter(row["avg_service_level"] * 100, row["safety_stock_investment"] / 1000, s=70,
                   color="#2f6fb0" if scenario != "D" else "#d0802f", zorder=3)
        ax.annotate(SCENARIO_NAMES.get(scenario, scenario),
                    (row["avg_service_level"] * 100, row["safety_stock_investment"] / 1000),
                    textcoords="offset points", xytext=(6, -12 if scenario == "D" else 4), fontsize=9)
    ax.set(xlabel="average service level target (%)", ylabel="safety stock investment ($k)",
           title="Working capital vs service level by scenario")
    ax.set_xlim(89, 99.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "inventory_scenarios.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    FIGURES.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_parquet(PROCESSED / "forecast_evaluation.parquet")
    forecasts = pd.read_parquet(PROCESSED / "forecasts_test.parquet")
    scenarios = pd.read_parquet(PROCESSED / "inventory_scenarios.parquet")

    models = model_summary(evaluation)
    scen = scenario_summary(scenarios)
    fva_chart(models)
    actual_vs_forecast_chart(forecasts)
    scenario_chart(scen)

    test_weeks = sorted(pd.to_datetime(forecasts["week_start_date"]).unique())
    results = {
        "test_window": {"start": str(test_weeks[0].date()), "end": str(test_weeks[-1].date()),
                        "weeks": len(test_weeks)},
        "models": models.round(6).reset_index().to_dict("records"),
        "scenarios": scen.round(4).reset_index().to_dict("records"),
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    LOGGER.info("Model summary:\n%s", models.round(4).to_string())
    LOGGER.info("Scenario summary:\n%s", scen.round(2).to_string())


if __name__ == "__main__":
    main()
