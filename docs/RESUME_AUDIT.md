# Resume Claim Audit

Audited 2026-06-10 after wiring real FRED + NOAA + Databricks + SQL artifacts.
This doc maps every claim on the resume bullet to the actual repo artifact so
nothing is overclaimed. Use this as the interview honesty backstop.

## Resume bullets

> **Building Products Demand Forecasting & Inventory Optimization** | Python, SQL, Databricks, MLflow, Power BI
>
> Modeled branch-level demand for roofing, siding, and exterior product categories using Census building permits, FRED housing starts, NOAA weather signals, and synthetic branch/SKU sales history; trained forecasting and segmentation models to identify seasonal demand shifts, contractor buying patterns, and slow-moving inventory risk across product lines.
>
> Built an optimization layer for reorder thresholds and safety-stock scenarios, comparing service-level targets against working-capital constraints; tracked experiments with MLflow and delivered Power BI views for branch demand, product mix, forecast error, and recommended replenishment actions.

## Claim-by-claim audit

| Claim | Status | Where it lives in the repo | Honesty note |
|---|---|---|---|
| Python | ✅ Real | All `src/` modules; 3677 LOC | — |
| SQL | ✅ Real | `sql/local/*.sql` (DuckDB) + `sql/databricks/*.sql` (Databricks SQL on Delta) | 4 queries each side, end-to-end tested locally |
| Databricks | ✅ Real | `databricks/01..03.py` uploaded to workspace `/Users/juloori.abhay@gmail.com/building_products/`; UC catalog `workspace.building_products`; Volume `project_data/` with all 17 parquet files | Real paid Databricks workspace (AWS us-east-2), Serverless Pro SQL warehouse |
| MLflow | ✅ Real | Local: `mlruns/` file backend, `experiment=demand_forecasting`. Databricks: notebook 03 logs to managed MLflow tracking under user folder. | Both backends used |
| Power BI | ⚠️ Spec, no .pbix yet | `docs/POWERBI_NOTES.md` star schema + DAX measures + report page spec | **You need to build the .pbix in Power BI Desktop** following `docs/BUILD_POWERBI.md` — once done, this becomes ✅ Real |
| Branch-level demand | ✅ Real | 12 branches × 80 SKUs × 5yr weekly, forecast per (branch, SKU, week) | — |
| Roofing/siding/exterior categories | ✅ Real | `data/raw/products.parquet` columns: category in {Roofing, Siding, Exterior} | — |
| Census building permits | ✅ Real (via FRED) | `data/external/building_permits.parquet` from FRED series `PERMIT` | FRED `PERMIT` is the Census Bureau's "New Privately-Owned Housing Units Authorized" series, redistributed via FRED. Same source data; FRED is the access channel. |
| FRED housing starts | ✅ Real | `data/external/housing_starts.parquet` from FRED `HOUST` | API key wired, real data 936–1807 SAAR captures COVID dip + 2021 boom |
| NOAA weather signals | ✅ Real | `data/external/weather.parquet` sourced from NOAA NCEI Global Summary of the Month CSV — Boston/Chicago (cold), Dallas (mixed), Atlanta (hot) | `weather_source=noaa_gsom` flag in output; if NOAA endpoint unreachable, code falls back to synthetic baseline and tags as `synthetic_fallback` |
| Synthetic branch/SKU sales history | ✅ Real | `src/data/generate_synthetic.py` produces 250K-row sales table, seeded deterministic | Explicitly synthetic — resume says so |
| Forecasting models | ✅ Real | `src/models/forecasting.py` — 4 models: Seasonal Naive, LightGBM+Optuna, Prophet ×36, Croston-TSB | Walk-forward split, MLflow logged |
| Segmentation models | ✅ Real | `src/models/segmentation.py` — ABC-XYZ SKUs, KMeans contractors (k=4), DTW hierarchical branches | — |
| "Seasonal demand shifts" | ✅ Real | `cat_region_seasonality_index` feature + `sql/local/02_seasonal_demand_shift.sql` QoQ/YoY query | — |
| "Contractor buying patterns" | ✅ Real | `data/processed/contractor_segments.parquet` 4 segments | Activity simulated from tier+branch (no real transaction-level link); data dictionary acknowledges this |
| "Slow-moving inventory risk" | ✅ Code real, ⚠️ outputs sparse | `is_slow_mover` flag, Croston-TSB model, `data/processed/slow_mover_flags.parquet` excess-inventory logic | Synthetic on-hand sawtooth around reorder point doesn't trigger excess flags in the current snapshot. In production this fires. Interview honest answer: "The logic is in `inventory.py`; the synthetic on-hand doesn't generate excess hits because it's modeled as a sawtooth — at a real branch this would fire on the 15% slow-mover SKUs." |
| Reorder thresholds + safety stock | ✅ Real | `src/optimization/inventory.py` variance-pooled formula; `data/processed/inventory_scenarios.parquet` | — |
| Service-level vs working-capital scenarios | ✅ Real | Scenarios A/B/C/D in `inventory_scenarios.parquet`: $311K–$498K safety stock ranges | — |
| "Delivered Power BI views for branch demand, product mix, forecast error, recommended replenishment" | ⚠️ Plotly Dash today; PBI spec ready | `src/dashboard/app.py` has all 4 tabs (branch overview, product mix + error, inventory + replenishment, contractor insights) styled to mirror Power BI; PBI build guide in `docs/BUILD_POWERBI.md` | **Action item: build the .pbix to fully back this claim.** |

## TL;DR for interview

When asked anything about the project, the answers grounded in reality:

- **"Did you use real Databricks?"** → Yes, AWS us-east-2 Unity Catalog workspace, schema `workspace.building_products`, 3 notebooks runnable in workspace, MLflow tracking on Databricks-managed server.
- **"Real Census permits / FRED / NOAA?"** → FRED HOUST + PERMIT pulled live with API key. NOAA via NCEI GSOM CSV downloads (no API key needed — public NOAA data). Stations: Boston Logan, Chicago O'Hare, Dallas Love, Atlanta Hartsfield.
- **"Real Power BI?"** → Star schema + DAX measures + page spec built. The .pbix file is in `powerbi/building_products.pbix` (after you build it per the BUILD_POWERBI guide). If asked before building: "The data model and page spec are in the repo; I built the live demo in Plotly Dash because it's reviewable in code. Same data, Power BI–ready Delta tables on Databricks."
- **"Synthetic data — admit or hide?"** → Admit upfront. The resume literally says "synthetic branch/SKU sales history." Synthetic data is engineered to mirror realistic patterns (climate-driven SKU preferences, contractor concentration on 3 branches, COVID dip, post-COVID boom, stockouts, slow-movers) — explain those as design decisions.

## What's still open after the audit

1. **Power BI .pbix file** — see `docs/BUILD_POWERBI.md` for the step-by-step. ~30–45 min in Power BI Desktop. Required to fully retire the ⚠️ on the Power BI claim.
2. **Slow-mover excess flags** populating — synthetic on-hand is too conservative to trigger. Either (a) live with the explanation above, or (b) bump synthetic `on_hand_units` to model occasional over-purchases. Interview honest path is (a).
