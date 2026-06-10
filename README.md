# Building Products Demand Forecasting & Inventory Optimization

End-to-end demand forecasting and inventory scenario engine for a synthetic
building materials distributor that sells roofing, siding, and exterior product
lines through 12 regional branches to a mix of contractor and retail customers.

The project models branch × SKU × week demand using engineered sales history,
FRED macro housing indicators (Housing Starts, Building Permits), and
climate-zone weather signals; trains four forecasting models in parallel
(Seasonal Naive baseline, LightGBM with Optuna tuning, Prophet per branch ×
category, Croston-TSB for intermittent slow-movers); segments SKUs (ABC-XYZ),
contractors (KMeans), and branches (DTW hierarchical); and runs a service-level
scenario analysis on reorder points and safety stock to compare working capital
against stockout risk. All experiments are tracked in MLflow, and a Plotly Dash
app surfaces the results across branch, product mix, replenishment, and
contractor lenses.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Data Layer                                                          │
│  ├─ src/data/generate_synthetic.py  → branches, SKUs, sales (5yr),  │
│  │                                     inventory snapshots, contractors│
│  └─ src/data/fetch_external.py      → FRED HOUST, PERMIT + NOAA-style │
│                                       weather, aligned to weekly grid │
├──────────────────────────────────────────────────────────────────────┤
│  Feature Layer                                                       │
│  └─ src/features/build_features.py  → lags, rolling stats, calendar, │
│                                       macro, contractor mix, velocity │
├──────────────────────────────────────────────────────────────────────┤
│  Modeling Layer                                                      │
│  ├─ src/models/forecasting.py       → 4 models, walk-forward eval,   │
│  │                                     MLflow tracking                │
│  └─ src/models/segmentation.py      → ABC-XYZ, KMeans, DTW clusters  │
├──────────────────────────────────────────────────────────────────────┤
│  Optimization Layer                                                  │
│  └─ src/optimization/inventory.py   → reorder pt, safety stock,      │
│                                       EOQ, 4 service-level scenarios │
├──────────────────────────────────────────────────────────────────────┤
│  Presentation Layer                                                  │
│  └─ src/dashboard/app.py            → Plotly Dash, 4 tabs            │
└──────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Tools |
|-------|-------|
| Synthetic + external data | `pandas`, `numpy`, `pyarrow`, `fredapi` |
| Features | `pandas` (vectorized groupby/rolling) |
| Forecasting | `lightgbm`, `prophet`, `optuna`, `statsmodels` |
| Segmentation | `scikit-learn`, `dtaidistance` |
| Optimization | `scipy.stats` |
| Tracking | `mlflow` |
| Dashboard | `plotly`, `dash`, `dash-bootstrap-components` |
| Packaging | `requirements.txt`, `python-dotenv` |

## Quickstart

```bash
# 1. Clone & install
git clone <repo-url> building-products-forecast
cd building-products-forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Optional — FRED API key for real macro signals
cp .env.example .env
# edit .env, paste FRED_API_KEY (free at fred.stlouisfed.org/docs/api/api_key.html)
# If missing, code falls back to a deterministic synthetic series.

# 3. Run the full pipeline end-to-end
python run_all.py
# Skip steps: python run_all.py --skip "synthetic data" "external data"

# 4. Launch the dashboard
python -m src.dashboard.app
# open http://127.0.0.1:8050
```

### What gets produced

| Output | Path |
|--------|------|
| Branches, products, sales, inventory, contractors | `data/raw/*.parquet` |
| FRED + weather aligned to weekly grid | `data/external/*.parquet` |
| Modeling table (one row per branch × SKU × week) | `data/processed/modeling_table.parquet` |
| Test-period forecasts (long format, per model) | `data/processed/forecasts_test.parquet` |
| Per-SKU model metrics + FVA vs naive | `data/processed/forecast_evaluation.parquet` |
| SKU ABC-XYZ, contractor segments, branch DTW clusters | `data/processed/{sku_abc_xyz,contractor_segments,branch_demand_clusters}.parquet` |
| Reorder point / safety stock by scenario | `data/processed/inventory_scenarios.parquet` |
| Excess inventory flags | `data/processed/slow_mover_flags.parquet` |
| MLflow experiment runs | `mlruns/` |

## Methodology notes

### Stockout censoring
Weeks with `stockout_flag=True` in the synthetic data are truncated to ~30% of
true demand (the SKU was on the shelf for part of the week then ran out). For
LightGBM these weeks are down-weighted (sample weight = 0.3) so the model
doesn't learn truncated demand as the new normal; for the Croston-TSB intermittent
forecaster they are excluded from the demand-size update. This is a common
real-world correction — naive treatment would bias forecasts downward and
push reorder points too low, propagating the stockout pattern.

### Walk-forward validation
Time-series leakage is the most common modeling mistake on demand data.
The split uses:
- **Train**: first 104 weeks (~2 years)
- **Validation**: next 26 weeks (Optuna objective + early stopping)
- **Gap**: remaining weeks until the test window (avoids contamination)
- **Test**: final 26 weeks (held out for FVA reporting)

All lag/rolling features are computed via `groupby(...).shift(1)` so the
target week never sees its own value. The category × region seasonality
index is computed only on the first 80% of dates and held fixed.

### Why LightGBM at SKU level, Prophet at category level
- **LightGBM** handles 960 branch × SKU series in a single global model. The
  tree splits on `branch_id`/`sku_id`/`category` learn per-entity intercepts
  while sharing signal across SKUs — important for the long-tail SKUs that
  don't have enough history for a per-SKU model. It also natively absorbs
  weather and macro covariates, calendar features, and stockout weights.
- **Prophet** is fit at branch × category granularity (36 models) because
  Prophet's additive seasonality + holiday + trend decomposition is much more
  expressive at an aggregated level where the signal-to-noise is high. SKU-level
  Prophet would over-fit weekly noise. SKU forecasts are obtained by proportional
  allocation of the category forecast by trailing 26-week SKU share within
  the branch × category.

### Croston-TSB for intermittent demand
SKUs with demand in fewer than 40% of training weeks are classified as
intermittent. For these the Croston-TSB method tracks two state variables —
demand size `z` (smoothed at α=0.1) and demand probability `p` (smoothed at
β=0.05) — and forecasts `z × p`. This is the standard spare-parts/slow-mover
approach; LightGBM and Prophet systematically over-forecast intermittent items
because they fit zeros as part of the signal.

### Safety stock formula
The optimizer uses the standard variance-pooled formula that accounts for
both demand and lead-time variability:

```
safety_stock = z(SL) × √(L × σ_d² + d̄² × σ_L²)
```

where `L` is mean lead time (weeks), `σ_d` is demand standard deviation,
`d̄` is mean weekly demand, and `σ_L` is lead-time standard deviation
(modeled as 20% of mean lead time). `z(SL)` is the inverse normal CDF
at the service-level target.

EOQ is the textbook Wilson formula:

```
EOQ = √(2 × D × S / (h × c))
```

with annual demand `D`, order cost `S = $50`, holding rate `h = 25%`,
and unit cost `c` per SKU.

### Scenarios
- **A — 95% service level**: baseline, matches a typical distributor target.
- **B — 98% aggressive**: prioritizes availability for project-driven contractors.
- **C — 90% lean**: minimizes working capital tied up in safety stock.
- **D — Demand-segmented**: 98% on ABC class A items (revenue critical),
  95% on B, 90% on C. Usually dominates flat targets — same average service
  level with materially lower investment.

## Headline outputs (from latest pipeline run)

> Update with your own numbers after running. Placeholder shown below.

| Scenario | Total Safety Stock | Avg Service Level |
|----------|--------------------|--------------------|
| A — 95% baseline | ~$399k | 95.0% |
| B — 98% aggressive | ~$498k | 98.0% |
| C — 90% lean | ~$311k | 90.0% |
| D — segmented | ~$466k | 93.6% (weighted) |

**Forecast value add vs naive (test set, avg per SKU):**
- LightGBM global: ~+26%
- Croston-TSB (slow-movers only): ~+25%
- Prophet branch × category: ~+4%

## Dashboard tabs

1. **Branch Demand Overview** — US map of branches colored by region, sized
   by next-13-week forecast. Branch × category actual-vs-forecast lines.
2. **Product Mix & Forecast Error** — top bias SKUs, MAPE vs demand volume
   scatter, ABC-XYZ heatmap.
3. **Inventory & Replenishment** — scenario toggle (A/B/C/D), recommended
   reorder action table, working capital comparison, excess inventory flags.
4. **Contractor Insights** — segment scatter (spend vs frequency), branch
   concentration heatmap, seasonal buying pattern by segment.

## Repo layout

```
building-products-forecast/
├── data/
│   ├── raw/                  # synthetic source tables
│   ├── external/             # FRED + weather aligned
│   └── processed/            # modeling table + outputs
├── notebooks/                # exploratory work (optional)
├── src/
│   ├── data/                 # generate_synthetic.py, fetch_external.py
│   ├── features/             # build_features.py
│   ├── models/               # forecasting.py, segmentation.py
│   ├── optimization/         # inventory.py
│   └── dashboard/            # app.py
├── mlruns/                   # MLflow file backend
├── requirements.txt
├── run_all.py
└── README.md
```

## Reproducibility

- All RNG paths seeded with `42` (data generation, contractor activity sim,
  KMeans `random_state`, Optuna sampler, LightGBM `seed`).
- Synthetic data is deterministic — same seed produces byte-identical parquet.
- MLflow runs are file-backed at `./mlruns`; no server required.
- The dashboard never recomputes; it only reads pre-built parquet outputs.

## Why this project shape

Building products distribution sits at the intersection of three real
forecasting problems: highly seasonal SKUs (roofing tracks summer install
season), intermittent slow-movers (specialty trim colors), and concentrated
contractor demand (a single roofing crew can swing a branch's weekly numbers).
A flat ARIMA-per-SKU approach handles none of these well. The project shows
how to pick the right tool per problem — global gradient boosting for the
fat-tail, Prophet for the aggregated seasonal decomposition, Croston for the
zero-heavy tail — and connect the resulting forecasts into a working-capital
scenario engine that a branch operations manager can actually use.
