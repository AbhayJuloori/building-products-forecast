# Interview Prep — Building Products Demand Forecasting

This is a deep-dive guide for the interview. Two sections:
1. **The story** — how to frame the project in 90 seconds and 5 minutes.
2. **Drill-down questions** — anticipated technical questions with answers
   grounded in actual code paths.

---

## 1. The story

### 90-second pitch

> "I built an end-to-end demand forecasting and inventory optimization system
> for a building products distributor — roofing, siding, exterior, sold through
> 12 regional branches to contractor and retail customers. The hard part isn't
> any single forecast, it's that the SKU portfolio has three different demand
> regimes mixed together: heavily seasonal SKUs that track install weather,
> intermittent slow-movers that are zero most weeks, and bursty contractor-
> driven SKUs that move on project schedules. So I trained four models — a
> seasonal-naive baseline, a global LightGBM, a Prophet-per-(branch × category)
> with weather and housing-start regressors, and a Croston-TSB intermittent
> forecaster for slow-movers — picked the best per SKU, and fed those forecasts
> into a service-level scenario engine that compares four working-capital
> strategies. Everything is tracked in MLflow and surfaced through a Plotly
> Dash app that mirrors what a Power BI deployment would look like."

### Why the project is shaped this way

| Choice | Why |
|---|---|
| 12 branches across 4 climate zones | Lets me show regional demand pattern differences (ice/water shield demand only in cold zones, housewrap heavier in hot/humid). Forces the model to learn climate effects. |
| 80 SKUs over 3 categories | Enough to need a global model (LightGBM) but small enough that per-SKU diagnostics are tractable. ABC-XYZ matrix shows the dispersion. |
| 5-year history (2020–2024) | Captures the COVID dip and post-COVID housing boom — gives the model exogenous regime shifts to handle rather than a flat baseline. |
| Walk-forward split | Required for time series; train 104w / val 26w / gap / test 26w. Random CV would leak. |
| Stockout flag | Truncated weeks are flagged and down-weighted (sample weight 0.3) — otherwise the model learns the truncation as the new demand level and reorder points drift down. |
| Contractor concentration feature | Bursty contractor branches have very different demand shape vs retail-heavy branches; the model needs that as a feature. |

---

## 2. Drill-down questions

### Q1. Why did you use four models instead of one?

The SKU portfolio has three demand regimes:

- **Fat-tailed seasonal SKUs** (asphalt shingles, fiber cement panels) —
  high volume, strong seasonality. LightGBM with lag/rolling/weather features
  is best — it can absorb everything and pool signal across the 960 series.
- **Aggregated category-level patterns** (roofing-as-a-whole peaks in
  spring/summer) — Prophet handles these well at branch × category
  granularity with explicit yearly/weekly seasonality components and
  housing-starts as an additive regressor. Then I disaggregate to SKU by
  trailing 26-week SKU share within the (branch, category).
- **Intermittent slow-movers** (specialty trim colors) — sparse with many
  zero weeks. Tree models and Prophet both over-forecast these because they
  treat zeros as part of the signal. Croston-TSB is the textbook fix:
  separately smooth demand size and demand probability, multiply.

The selection logic at inference time picks the lowest test-RMSE model per
SKU (with a tie-break to LightGBM). In the latest run that's ~85% LightGBM,
~8% Prophet, ~5% Croston, ~1% naive (a few series the naive happens to be
unbeatable on).

### Q2. How did you handle stockout censoring?

Three mechanisms:

1. **Synthetic data flags stockout weeks** at ~10% rate with units truncated to
   ~30% of true demand — this matches the pattern where a branch runs out
   mid-week, sells what it had, and the rest is unmet demand.
2. **LightGBM training** uses `sample_weight=0.3` on stockout weeks. The
   weights pass through to the gradient calculation, so the loss surface
   weights non-truncated weeks much more heavily — the model learns the
   correct demand distribution.
3. **Croston-TSB** excludes stockout weeks from the demand-size smoothing
   entirely (treats them as missing rather than zero). This avoids dragging
   the smoothed demand size down.

The alternative is to impute the truncated demand and train on imputed values,
but imputation introduces its own bias. Weighted loss is simpler and works.

### Q3. Walk-forward — why those windows specifically?

The data spans ~209 usable weeks (lag-52 warmup drops year 1). I split:

- **Train: first 104 weeks** — two full seasonal cycles, enough for the model
  to learn both yearly seasonality and the lag-52 feature.
- **Validation: next 26 weeks** — 6 months. Used as Optuna objective and
  LightGBM early-stopping. Long enough to span at least one seasonal transition.
- **Gap: 53 unused weeks** — separates val from test so model selection on val
  doesn't peek at test-adjacent dynamics. Costly on a small dataset but
  important for honest FVA reporting.
- **Test: final 26 weeks** — 6 months reserved for forecast value-add metrics.

If you don't have a gap, the Optuna tuning subtly contaminates the test
window through selection bias. With a gap, validation MAPE and test MAPE
diverge naturally as expected.

### Q4. Forecast value add — why that metric over MAPE?

MAPE alone doesn't tell you whether the model is earning its complexity. FVA
measures `(naive_rmse - model_rmse) / naive_rmse` — the percent improvement
over the simplest possible forecast (last year same week). If a model has
MAPE 30% that sounds bad, but if naive is 60%, the model is doing real work.

In this project LightGBM averages ~26% FVA vs naive on the test set. Prophet
is ~4% — disappointing at first glance, but Prophet is the right model for
the 8% of SKUs it wins on (cleaner aggregated seasonality), and the SKU-share
allocation back to individual SKUs adds noise that drags average FVA down.

### Q5. Safety stock formula — defend the assumption.

The formula is:

```
safety_stock = z(SL) × √(L × σ_d² + d̄² × σ_L²)
```

Where:
- `z(SL)` — inverse normal CDF at service level. Assumes demand and lead-time
  variability are normally distributed.
- `L` — mean lead time in weeks.
- `σ_d` — standard deviation of weekly demand.
- `d̄` — mean weekly demand.
- `σ_L` — standard deviation of lead time (modeled as 0.2 × `L`).

The variance-pool form accounts for both sources of variability — demand can
spike during the lead window, *and* the lead window itself can stretch. A
naive `z × σ_d × √L` formula assumes deterministic lead time, which is wrong
for building materials (supplier capacity, freight, weather all affect it).

**Where the formula breaks:**
- **Intermittent demand** — the normality assumption fails for Poisson-like
  series. For slow-movers I'd switch to a target fill rate calculation or
  service-level-equivalent buffer derived from the Croston-TSB demand
  probability. The current code applies the standard formula uniformly,
  which is a known limitation.
- **Highly seasonal SKUs** — the σ_d should be computed within season, not
  across the full year. The current code uses test-period forecast std as a
  proxy, which partially handles this since the test horizon is recent.

### Q6. Scenario D (segmented) — why does it dominate?

The math: aggressive service level (98%) on the top revenue SKUs costs
proportionally less than running 98% on the entire portfolio, because the long
tail of C-class SKUs is most of the SKU count but a small share of revenue.
Running them at 90% saves real working capital with minimal customer impact —
the marginal stockout on a C SKU is a small dollar miss.

In the latest run:
- Scenario A (95% flat): $399K safety stock
- Scenario B (98% flat): $498K
- Scenario D (segmented): $466K weighted avg service level 93.6%

D doesn't dominate B on total dollars, but it dominates A *in availability for
the SKUs that matter most* while costing only ~17% more. The "right" answer
depends on whether the distributor's pain is contractor satisfaction (B looks
better) or working capital (C/D).

### Q7. Why LightGBM over XGBoost or CatBoost?

- **vs XGBoost** — LightGBM trains faster on the same data (leaf-wise vs
  level-wise growth), and the categorical_feature parameter handles the
  branch/SKU/region categoricals natively without one-hot encoding. With 960
  series and 6 categorical features that's a real win.
- **vs CatBoost** — CatBoost is excellent on tabular but LightGBM has the
  Optuna integration and the ecosystem (mlflow.lightgbm.log_model, feature
  importance interpretability via SHAP) more mature in my workflow.

Honest answer: any of the three would work. The choice matters less than
the walk-forward split, the stockout weighting, and the feature engineering.

### Q8. Prophet — why per (branch × category) and not per SKU?

Prophet's seasonality decomposition is most valuable when the signal-to-noise
is high. At SKU level for a low-volume product (50 units/week), weekly noise
dominates the Fourier yearly seasonality. At branch × category (3000+ units/
week summed across SKUs in roofing for Boston), the seasonal component is
clearly recoverable.

Then I disaggregate back to SKU via trailing 26-week SKU share within the
(branch, category) — assumes SKU mix is stable, which is mostly true except
during category-level repositioning (new product launches).

### Q9. Why DTW for branch clustering?

Branches differ in **shape** of demand, not just **level**. Boston peaks in
summer, Atlanta has a flatter year-round pattern, Detroit has a sharper
spring rebound. Euclidean distance on the raw weekly series would conflate
shape and level. DTW is shape-aware — it allows time warping so a Boston
peak in week 25 matches an Atlanta peak in week 22 as similar shapes.

For 12 branches the DTW distance matrix is 12×12 — trivial compute. At 1000s
of stores you'd subsample or use lower-bound approximations
(LB_Keogh, FastDTW).

### Q10. MLflow — what does it actually buy you here?

Three things:

1. **Reproducibility** — every Optuna trial logs params + RMSE/MAPE/bias.
   If a future run regresses I can `mlflow ui` and compare runs side-by-side
   to find the parameter that changed.
2. **Model artifacts** — the LightGBM booster is saved as part of the run.
   Loading via `mlflow.lightgbm.load_model(run_id)` is the deployment path —
   in production this is what an inference service would do.
3. **Experiment hygiene** — separate experiments for `demand_forecasting`,
   `segmentation`. Each run is tagged with the split strategy and target so
   future-me knows what they're looking at.

For Community Edition Databricks the same API works against the
Databricks-managed tracking server with zero code change.

### Q11. How would you productionize this?

Three layers:

1. **Data ingestion** — Databricks Auto Loader monitoring an S3 bucket where
   the ERP drops weekly branch × SKU sales extracts. Write to a Delta table
   with schema enforcement and time travel for backtesting. External APIs
   (FRED, NOAA) refreshed monthly via scheduled Workflow.
2. **Training** — Workflow that runs `build_features → forecasting` weekly.
   Optuna tuning runs monthly or only when MAPE degrades > 10% from baseline.
   New best model promoted via MLflow Model Registry stages
   (None → Staging → Production).
3. **Serving** — A REST endpoint (Databricks Model Serving or a Lambda) that
   takes a branch + SKU and returns the next-13-week forecast + recommended
   reorder action under the chosen scenario. The Power BI / Dash dashboard
   reads this from the inventory_scenarios Delta table.

Monitoring: track forecast bias rolling 4w by SKU + branch — alert if bias
exceeds ±2σ for two consecutive weeks (early indicator of demand regime
shift).

### Q12. What was hardest?

Three honest answers:

1. **Stockout handling** — I went back and forth. First version trained on
   raw units. The model learned the truncated demand as ground truth and
   safety stock recommendations were systematically low. The fix (sample
   weighting) is one line but the diagnosis took looking at residuals by
   stockout flag.
2. **Prophet at SKU level** — first attempt was Prophet per SKU. The fits
   were noisy and computationally expensive (960 models). Moving to
   branch × category (36 models) and disaggregating cleaned up the FVA
   numbers and runtime.
3. **Scenario interpretation** — the math is straightforward but presenting
   the tradeoff to a non-technical operations manager is the actual product.
   Scenario D (segmented) is the right answer most of the time, but only if
   you've walked them through *why* A items deserve 98% and C items don't.

---

## Common follow-ups

**"How would you forecast at SKU level if you had 10x the SKUs?"**
> Hierarchical reconciliation — train at SKU level, then enforce that SKU-level
> forecasts sum to category-level forecasts. The `MinT` (minimum trace) method
> in `scikit-hts` is the standard approach. Avoids both bottom-up bias and
> top-down naivete.

**"What if a new branch opens with no sales history?"**
> Cold-start via lookalike. Pick the existing branch with the closest match on
> climate zone, urban/suburban/rural, and population, use its forecast for the
> new branch, dampen by 30% for the first quarter as the new branch ramps. A
> hierarchical Bayesian model with branch-level random effects shrunk toward
> the region mean would be the production-grade answer.

**"How do you know the model isn't overfitting?"**
> The gap between val MAPE and test MAPE is the cleanest signal — both come
> from honest holdouts. In the latest run val MAPE and test MAPE differ by
> less than 5 percentage points, which suggests the model generalizes. Also,
> seasonal-naive is the floor — if I were overfitting in a meaningful way I'd
> see test FVA collapse to zero or go negative on some SKUs. The current
> distribution of per-SKU FVA centered around +20% with a long positive tail
> looks like genuine signal.

**"Why dash and not Power BI for the demo?"**
> Power BI is a UI tool with no programmatic interface from this project. Dash
> uses the same chart primitives (Plotly), runs against the same parquet
> outputs, and lets me check it into version control. At an interview I can
> walk through `app.py` and show how each chart maps to a callback. For a
> production rollout at the distributor I'd ship the same data tables to a
> Power BI Direct Query / Import refresh against the Delta outputs.
