# Interview Study Guide — Building Products Demand Forecasting & Inventory Optimization

Companion to `docs/INTERVIEW_PREP.md`. That file is the drill-down Q&A reference. This file is the **structured study material**: pitches you memorize, component cards you internalize, weaknesses you surface proactively, numbers you must know cold.

---

## 0. How to use this guide

Day-of-interview reading order: re-read **Section 1** until you can speak the 90-sec pitch without notes. Drill **Section 2** components — when someone asks "tell me about the LightGBM piece" you should be able to give the SPOKEN ANSWER paragraph live. Skim **Section 3** for question variants you haven't practiced. Memorize **Section 4** numbers. Read **Section 5** once and decide which two weaknesses you will surface before being asked. Follow the **Section 6** 3-day plan if you have time.

The single most important behavior: when you don't know something, name it as a known limitation (Section 5) instead of bluffing.

---

## 1. Project framing (memorize verbatim)

### 1.1 30-second elevator pitch

> I built an end-to-end branch-level demand forecasting and inventory optimization system for a building products distributor — 12 branches, 80 SKUs, 5 years of weekly history. Four forecasting models compete per SKU: a seasonal-naive baseline, a global LightGBM with engineered lag/weather/macro features, Prophet per branch×category, and Croston-TSB for intermittent slow-movers. LightGBM averages **+25.9% forecast value-add over naive** on the test window, and the forecasts feed an inventory scenario engine that compares four working-capital strategies.

### 1.2 90-second pitch

> "I built an end-to-end demand forecasting and inventory optimization system for a building products distributor — roofing, siding, exterior products sold through 12 regional branches to contractor and retail customers. The hard part isn't any single forecast; it's that the SKU portfolio has three different demand regimes mixed together — heavily seasonal SKUs that track install weather, intermittent slow-movers that are zero most weeks, and bursty contractor-driven SKUs that move on project schedules. So I trained four models — a seasonal-naive baseline, a global LightGBM with 43 engineered features and Optuna HPO, Prophet per branch×category with FRED housing-start and NOAA weather regressors, and a Croston-TSB intermittent forecaster for slow movers. I pick the best model per SKU and feed those forecasts into a service-level scenario engine that compares four working-capital strategies. Everything is tracked in MLflow, the production path is staged in Databricks Community Edition notebooks, the analytics layer is SQL on both DuckDB local and Spark, and the dashboard is Plotly Dash with the same chart primitives a Power BI deployment would use."

### 1.3 5-minute deep dive (structured walkthrough)

Walk it in this order. Stop at each waypoint and let the interviewer redirect.

1. **Business problem (30 sec).** "Distributor pain is working capital. They overstock to keep contractors happy and end up with $400K-$500K tied up in safety stock per scenario. The lever is a better forecast — if I can shrink the demand variance estimate, I shrink the safety stock requirement at the same service level."
2. **Data (45 sec).** "Synthetic dataset modeled on a real distributor footprint — 12 branches across 4 climate zones, 80 SKUs across roofing/siding/exterior, weekly grain, 5 years (2020-2024). That's 250K sales rows. Plus 200 contractor accounts with concentration features. External data is real: FRED housing starts and permits, and NOAA GSOM weather pulled per climate zone."
3. **Features (45 sec).** "43-column modeling table built in `src/features/build_features.py`. Lags 1/4/13/52, rolling means and stds at 4/13/26 weeks, calendar features, two external macro signals, weather, climate zone, contractor concentration, ABC velocity bucket. Stockout flag carried through as a sample weight, not a target adjustment."
4. **Models (60 sec).** "Four models competing per SKU. Seasonal-naive baseline for FVA reference. Global LightGBM trained on all 960 branch×SKU series at once — 20 Optuna trials, walk-forward validation. Prophet trained per (branch × category) — 36 models with yearly+weekly seasonality and housing-starts as a regressor — then disaggregated to SKU by trailing 26-week share. Croston-TSB for the 7 slow-mover SKUs across 84 series — separates demand size from demand probability."
5. **Evaluation (30 sec).** "Walk-forward split: 104 weeks train / 26 val / 53 gap / 26 test. Metric is forecast value-add against the seasonal-naive baseline. LightGBM landed at +25.9% FVA on test. Honest finding: Prophet came in at -7.9% FVA on real macro data — worse than naive — which I deprecated for this run."
6. **Optimization (45 sec).** "Reorder point and safety stock derived from forecast mean and std plus lead-time variance. Variance-pooled formula `z × √(L·σ_d² + d̄²·σ_L²)` not the naive `z·σ·√L`. Then I run four scenarios — flat 95% service level, aggressive 98%, cost-optimized 85%, and a segmented strategy where A-class SKUs get 98% and C-class get 90%. The segmented scenario dominates on weighted availability per dollar."
7. **Surface (30 sec).** "Plotly Dash dashboard with four tabs — branch overview, forecast diagnostics, segmentation, scenario comparison. MLflow for experiment tracking. Databricks notebooks for the distributed-training claim. SQL queries for both DuckDB local and Spark. Power BI implementation guide with DAX measures so this could be deployed without the Dash app at all."

### 1.4 Why this project shape

| Choice | Why |
|---|---|
| 12 branches across 4 climate zones | Surfaces regional demand pattern differences. Forces the model to learn climate effects (ice/water shield only sells in cold zones; housewrap heavier in hot/humid). |
| 80 SKUs across 3 categories | Enough to require a global model but small enough that per-SKU diagnostics are tractable. ABC-XYZ shows the dispersion. |
| 5-year history (2020-2024) | Captures COVID dip + post-COVID housing boom. Gives the model exogenous regime shifts to handle, not a flat baseline. |
| Weekly grain | Standard distributor cadence. Daily is too noisy for slow movers; monthly loses the seasonal precision. |
| Walk-forward split | Time-series ground truth requires it. Random CV would leak future into past. Gap window prevents val→test contamination through Optuna selection bias. |
| Four model families | Three demand regimes (seasonal / aggregated / intermittent) need different priors. Plus a baseline to measure value-add against. |
| Stockout sample-weighting | Truncated weeks would otherwise be learned as ground truth demand and bias safety stock downward systematically. |
| Variance-pooled safety stock | Lead time is stochastic in building materials (supplier capacity, freight, weather). Deterministic-lead formula under-buffers. |
| Four scenarios A/B/C/D | Forces the user (operations manager) to confront the working-capital vs availability tradeoff explicitly instead of accepting a single default. |
| Plotly Dash, not Power BI | Programmatic, version-controlled, demonstrable in a code review. The Power BI build guide ships the production path. |

---

## 2. Component deep dives

Each component below uses this template:
- **WHAT:** one sentence
- **HOW IT WORKS:** 3-6 bullets
- **DESIGN DECISIONS + WHY**
- **GOTCHAS** (interviewer-probe targets)
- **SPOKEN ANSWER (45-60 sec):** literal paragraph to deliver if asked "walk me through your <X>"

### 2.1 Synthetic data generation (`src/data/generate_synthetic.py`)

**WHAT:** Generates a realistic distributor-scale dataset — 12 branches × 80 SKUs × 5 years weekly sales, 200 contractor accounts, inventory snapshots — with stockouts, seasonality, and climate-zone effects baked in.

**HOW IT WORKS:**
- Branch dimension carries climate zone (cold/temperate/hot-humid/mixed), urbanicity, contractor-concentration level.
- SKU dimension carries category, velocity tier (A/B/C), seasonality strength, slow-mover flag.
- Weekly demand model: `base × seasonal(climate, week_of_year) × trend(housing_cycle) × noise + occasional_stockout_truncation`.
- Stockout flag set on ~10% of weeks with units truncated to ~30% of true demand.
- 200 contractor accounts with concentration following Pareto — top 20% drive ~70% of contractor volume per branch.

**DESIGN DECISIONS + WHY:**
- **Synthetic, not Kaggle.** Real distributor data is proprietary; synthesizing lets me bake in stockouts, climate effects, contractor concentration deliberately so the modeling choices have something to interact with.
- **Stockout flag is a known column.** In production the ERP would flag truncated weeks; assuming it's available is honest because most distributors track this.
- **Seasonality magnitude varies by climate.** Cold-zone roofing has sharper spring/summer install peak; hot-humid stays flatter year-round. Forces the model to learn interactions, not just main effects.

**GOTCHAS:**
- Synthetic = the claim has to be honest. State "this is a portfolio piece with synthetic data sized to a real distributor — modeling choices and the inventory math are the deliverable."
- Slow-mover excess-inventory flag returns 0 rows on this dataset because synthetic on-hand sawtooths around reorder point. Real data has meaningful excess.
- No supplier or pricing dimension — would need both in production for full cost optimization.

**SPOKEN ANSWER (45-60 sec):**
> The dataset is synthetic but sized to a real distributor footprint — 12 branches, 80 SKUs, 5 years weekly, which lands at 250K sales rows. I generated it deliberately rather than using a Kaggle download because I wanted the data to interact with the modeling choices — stockouts on roughly 10% of weeks with units truncated to 30% of true demand so the stockout-sample-weight mechanism actually matters; climate-zone effects on seasonality strength so the LightGBM has interactions to learn; and Pareto contractor concentration so the contractor-share feature shows up in feature importance. I'm transparent in interviews that the data is synthetic — the deliverable is the modeling and inventory layer, not the dataset itself.

### 2.2 External data fetch — FRED macro + NOAA weather (`src/data/fetch_external.py`)

**WHAT:** Pulls real macroeconomic and weather signals — FRED housing starts (HOUST) and permits (PERMIT), NOAA GSOM weather per climate zone — and writes them as a weekly external panel.

**HOW IT WORKS:**
- FRED API hit via `FRED_API_KEY` env var, pulls monthly HOUST and PERMIT, 6-year window, falls back to synthetic AR(1) on miss.
- NOAA GSOM (Global Summary of Month) hit via `NOAA_API_TOKEN`, queries per-climate-zone station IDs for monthly TAVG and PRCP.
- Both monthly series upsampled to weekly via forward-fill + linear interpolation.
- Final `external_weekly.parquet` joins macro and weather on week-start-date.
- Flags `precip_above_normal_flag` set when weekly precip exceeds rolling 26-week median.

**DESIGN DECISIONS + WHY:**
- **Real APIs, not synthetic stand-ins.** Authenticity matters for the resume claim ("Census permits, FRED housing starts, NOAA weather signals"). Synthetic fallback exists for offline development.
- **Monthly→weekly upsampling, not weekly fetch.** Both APIs are monthly-native; weekly resampling at fetch keeps the downstream feature builder simple.
- **Climate zone is the geographic key, not branch.** NOAA stations are noisy at single-point scale; zone-level aggregation smooths.

**GOTCHAS:**
- API keys must be present; degrades silently to synthetic if missing — confirm `macro_source=fred` and `weather_source=noaa_gsom` in logs after a run.
- NOAA station selection matters; mine returns avg monthly temps in the 31-38°F band because the cold-zone stations dominate. Real production would weight by branch count per zone.
- FRED rate limits — exponential backoff on 429.

**SPOKEN ANSWER:**
> The macro signals are real — FRED housing starts and building permits over the project window — and the weather is NOAA's GSOM product, which is monthly temperature and precipitation summaries from station data, queried per climate zone. Both are monthly-native, so the fetcher upsamples to weekly with forward-fill plus interpolation. I'm explicit about the failure mode: if `FRED_API_KEY` isn't in the env, the fetcher falls back to a synthetic AR(1) macro series and logs `macro_source=synthetic`. On the most recent run both are real — housing starts ran 936 to 1807 monthly, permits 1079 to 1923. That's the macro context the LightGBM and Prophet pick up.

### 2.3 Feature engineering — 43-column modeling table (`src/features/build_features.py`)

**WHAT:** Joins sales + branches + SKUs + external panel into one 43-column weekly table with leak-safe lags, rolling stats, calendar features, and concentration metrics.

**HOW IT WORKS:**
- Lags: 1, 4, 13, 52 on weekly units sold per branch×SKU. Computed with `groupby().shift()`, never includes the current week.
- Rolling: 4w, 13w, 26w means and stds. Computed with `shift(1)` first to avoid leakage from current week.
- Calendar: week of year, month, quarter, year, is-holiday-week.
- External: FRED HOUST + PERMIT + weather avg temp + precip flag, joined on week-start.
- Categorical: branch, SKU, category, climate zone, velocity (A/B/C), seasonality bucket.
- Contractor concentration: top-3 contractor share at branch for current week, lagged 1 week.
- Stockout flag carried through as a column for downstream sample-weight assignment.

**DESIGN DECISIONS + WHY:**
- **Lags 1/4/13/52** — 1 captures momentum, 4 monthly cycle, 13 quarterly, 52 yearly seasonality. Adding more wasn't measurably useful in early experiments.
- **`shift(1)` before rolling** — guarantees rolling stats don't see week-T target.
- **Categorical kept as integer codes, not one-hot.** LightGBM handles categoricals natively; one-hot would explode dimensionality on 80 SKUs × 12 branches.
- **Climate zone as feature, not separate model per zone.** A global model with zone-as-feature lets the gradient pool data across zones; per-zone splits would drop sample efficiency for slow movers.

**GOTCHAS:**
- Lag-52 means the first year is unusable for training — 209 effective weeks of 260 total.
- Stockout flag is treated as feature AND weight. Don't double-count — the model sees the flag as a column but the loss weights non-stockout weeks 1.0 and stockout 0.3.
- Rolling stds can be NaN on early weeks; filled with overall SKU mean before training.

**SPOKEN ANSWER:**
> The modeling table is 43 columns wide. Lags at 1, 4, 13, and 52 weeks for momentum, monthly, quarterly, and yearly patterns. Rolling means and stds at 4, 13, and 26 weeks. Calendar features. External signals — housing starts, permits, weather — joined on week-start. Categorical features kept as integer codes because LightGBM handles them natively with the `categorical_feature` parameter. The key correctness invariant is that every lag and rolling stat is computed with a `.shift(1)` first, so the current week never sees its own target. The stockout flag is in the feature set but also drives the sample-weight assignment downstream — that's deliberate, the model needs to know which weeks were truncated.

### 2.4 Seasonal-naive baseline (`src/models/forecasting.py`)

**WHAT:** Last-year-same-week forecast per branch×SKU. The honest baseline that FVA measures against.

**HOW IT WORKS:**
- For week T, forecast = actual demand at week T-52.
- Falls back to trailing 13-week mean for series with <52 weeks of history.
- Run on the full test window with no fitting, just lookup.

**DESIGN DECISIONS + WHY:**
- **Seasonal-naive, not random-walk.** The data has strong yearly seasonality; random-walk would be a stupid baseline to beat. Seasonal-naive is what a human distributor manager would do on their own.
- **No exogenous adjustment.** A "naive plus YoY trend" is fairer but blurs the FVA story.

**GOTCHAS:**
- If the prior year had a stockout at the same week, naive carries that truncation forward. Limitation, not bug; same fragility a human would have.
- For SKUs introduced mid-history (<52 weeks) the fallback to trailing mean is itself biased; flagged in the eval table.

**SPOKEN ANSWER:**
> Baseline is seasonal-naive — week T forecast equals week T-52 actuals. That's the honest comparison because it's what a human ops manager would reach for in absence of a model. I chose it specifically because it has the yearly seasonality baked in, which means any FVA improvement on top of it represents *signal the model added beyond the obvious pattern*, not just credit for noticing seasonality at all. On the recent test window the naive RMSE was about 5.0 and LightGBM landed at 3.65 — that's the 25.9% FVA number.

### 2.5 LightGBM global model with Optuna 20-trial HPO (`src/models/forecasting.py`)

**WHAT:** A single LightGBM regressor trained across all 960 branch×SKU series simultaneously, with 20-trial Optuna HPO and walk-forward validation.

**HOW IT WORKS:**
- Single LightGBM regressor with `objective=regression`, `metric=rmse`, native categorical handling on 6 features.
- 20 Optuna trials, each picks `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. Objective is val-set RMSE.
- Walk-forward inside Optuna: train on 104w, val on 26w. Best trial retrained on train+val, evaluated on the 26w test window.
- `sample_weight = 1.0 - 0.7*stockout_flag` so stockout weeks contribute 0.3 to gradient.
- Logged to MLflow: every trial's params + final test metrics. Booster artifact saved via `mlflow.lightgbm.log_model`.

**DESIGN DECISIONS + WHY:**
- **Global model, not per-SKU.** 960 series × ~209 weeks is 200K rows — plenty for a global model and lets the gradient pool signal across series with similar patterns. Per-SKU models would have ~209 rows each — too few to learn lag-52 reliably.
- **LightGBM over XGBoost or CatBoost.** Faster training, native categorical handling, mature MLflow integration. Honestly any of the three would work; the choice matters less than the data prep.
- **20 Optuna trials, not 100.** Diminishing returns past ~15 trials on this search space; defensible for portfolio scope. In production I'd run 50+ during the monthly retune cadence.
- **Stockout sample weight = 0.3.** Tested 0.0, 0.3, 0.5 — 0.3 was the sweet spot on val FVA. Zero loses too much signal on which weeks had high demand pre-truncation.

**GOTCHAS:**
- 20 trials is small. Interviewer will probe this — defense is "diminishing returns" and "would scale up in production".
- Global model means SKU identity is just a categorical embedding via gradient — works because the lag features carry the per-series signal.
- Optuna pruner not enabled here — trials run to completion. Could add `MedianPruner` for budget efficiency.

**SPOKEN ANSWER:**
> LightGBM is the workhorse — one global model trained across all 960 branch×SKU series at once. The reasoning is sample size: each series has about 209 usable weeks after the lag-52 warmup; that's too few to fit reliable per-SKU models, but pooling them gives the booster 200K rows to learn from. The 43 features carry the per-series identity through the lags and rolling stats. I run 20 Optuna trials on the standard LightGBM tuning surface — num leaves, learning rate, min data in leaf, feature fraction, regularization — with walk-forward validation. The best trial retrains on train plus val and gets evaluated on the holdout 26 weeks. Stockout weeks get sample weight 0.3 so they don't dominate the loss. On the recent test run avg RMSE was 3.65, FVA versus naive 25.9%. Every Optuna trial is logged to MLflow with params plus metrics.

### 2.6 Prophet per (branch × category) — 36 models with regressors (`src/models/forecasting.py`)

**WHAT:** Prophet fit per (branch, category) pair — 12 branches × 3 categories = 36 models — with FRED housing starts as an additive regressor; SKU-level forecasts derived by trailing 26-week SKU share.

**HOW IT WORKS:**
- For each (branch, category): aggregate weekly sales, fit Prophet with `yearly_seasonality=True`, `weekly_seasonality=True`, `add_regressor('housing_starts')`.
- Forecast 26w ahead at the (branch, category) grain.
- Disaggregate to SKU by trailing 26-week share: `sku_forecast = category_forecast × sku_share`.

**DESIGN DECISIONS + WHY:**
- **(Branch × category), not per-SKU Prophet.** SNR at SKU level for low-volume items is too low for Prophet's Fourier decomposition. At branch×category the seasonal signal is recoverable.
- **Housing starts as regressor.** Macro signal correlates strongly with new-construction-driven categories (roofing, siding).
- **SKU share allocation.** Stable approximation when SKU mix doesn't shift, which is the common case outside product launches.

**GOTCHAS:**
- **Real-data result: FVA -7.9% (worse than naive).** Honest finding on real macro. The synthetic-data version had Prophet at +4% FVA. Two likely causes: housing-starts regressor coefficient is shrunk near zero on this data because the synthetic demand was tuned to seasonality not macro; and the SKU share allocation adds noise that doesn't exist at the aggregate.
- **36 models, separate fits.** Computationally cheap but no cross-series pooling — opposite of LightGBM's strength.
- **Share allocation assumes stable mix.** Breaks on product launches.

**SPOKEN ANSWER:**
> Prophet runs at the (branch × category) grain — that's 12 branches times 3 categories so 36 models. Each one has yearly and weekly seasonality plus housing-starts as an additive regressor. I aggregate weekly sales up to (branch, category), fit Prophet, forecast 26 weeks, then disaggregate back to SKU using a trailing 26-week SKU-share multiplier. The reasoning is signal-to-noise: at SKU level for a 50-unit-per-week item the weekly noise drowns out the Fourier seasonality, but at branch×category — thousands of units per week — Prophet's decomposition is clean. The honest finding on the most recent run with real macro data is Prophet came in at -7.9% FVA, *worse* than naive. I'd deprecate the (branch×category) Prophet for this dataset and let LightGBM cover those SKUs. I keep it in the pipeline because for a real distributor with stronger macro-demand correlation, Prophet has a clear lane.

### 2.7 Croston-TSB intermittent forecaster for slow movers (`src/models/forecasting.py`)

**WHAT:** Croston with Teunter-Syntetos-Babai (TSB) variant — exponential smoothing of demand size and demand probability separately — applied to the 7 slow-mover SKUs across 84 branch×SKU series.

**HOW IT WORKS:**
- For each branch×SKU series flagged as slow-mover: separately smooth (a) the size of nonzero demand events and (b) the inter-event probability of a nonzero week.
- Forecast = smoothed_size × smoothed_probability. Constant across the forecast horizon.
- Stockout weeks excluded from the size smoothing (treated as missing, not zero).
- Alpha smoothing parameter tuned per series via grid search on val MAPE.

**DESIGN DECISIONS + WHY:**
- **TSB, not classic Croston.** Classic Croston has known bias upward on the forecast — TSB corrects this by updating the probability term every period, not just on demand events.
- **Per-series alpha.** Slow-movers have heterogeneous burst patterns; a global alpha would underfit some series and overfit others.
- **Stockout exclusion from size smoothing.** Treating truncated weeks as zero would drag the demand-size estimate down; treating them as missing is more honest.

**GOTCHAS:**
- Constant forecast across horizon — no seasonality, no trend. By design for intermittent demand but a known limit.
- Service-level safety stock formula assumes normal demand distribution, which breaks for these series. Documented as a limitation; would switch to fill-rate-based stock for these SKUs in production.
- 7 slow-mover SKUs is a small sample; on a larger portfolio Croston coverage would be 20-30% of SKUs.

**SPOKEN ANSWER:**
> Croston-TSB handles the slow movers — in this dataset that's 7 SKUs across 84 branch-SKU series. The mechanism is two separate exponential smoothings: one on the size of nonzero demand events, one on the probability that any given week has nonzero demand. The forecast is the product. I use the TSB variant rather than classic Croston because classic Croston has a known upward bias — TSB updates the probability term every period to correct that. Stockout weeks are excluded from the size smoothing rather than counted as zero, otherwise the demand-size estimate drifts down. On the recent run Croston-TSB came in at +25.4% FVA on those slow movers — material improvement over naive on series where tree models and Prophet both over-forecast.

### 2.8 Model evaluation — walk-forward split, RMSE/MAE/MAPE/Bias/FVA (`src/models/forecasting.py`)

**WHAT:** Computes RMSE, MAE, MAPE, mean bias, and FVA-vs-naive per branch×SKU on a held-out walk-forward test window, writes per-SKU and per-model summaries to `data/processed/forecast_eval.parquet`.

**HOW IT WORKS:**
- Walk-forward split: 104w train / 26w val / 53w gap / 26w test.
- Each model produces SKU-level forecasts on the test 26 weeks.
- Per series compute the five metrics; aggregate to per-model averages.
- FVA = `(naive_rmse - model_rmse) / naive_rmse`. Positive = model wins.

**DESIGN DECISIONS + WHY:**
- **Walk-forward with gap.** No leakage from val→test through Optuna selection bias. Costs 53 weeks of data; worth it for honest FVA.
- **FVA over MAPE for headline number.** MAPE alone doesn't tell you whether complexity is earning its keep; FVA does.
- **Mean bias as separate metric.** RMSE/MAE penalize symmetric error but a +5 average bias on every SKU is a systemic safety-stock problem — flagged separately.

**GOTCHAS:**
- Test window is one slice — doesn't capture between-year variance. In production you'd rolling-origin over multiple test windows.
- MAPE breaks on near-zero actuals; reported as `inf` for those series, excluded from averages.
- FVA can go negative — that's the Prophet result, presented honestly.

**SPOKEN ANSWER:**
> Eval is walk-forward with a 53-week gap between val and test. The gap is deliberate — it prevents Optuna selection bias from contaminating the test window through near-adjacency. I report five metrics per series: RMSE, MAE, MAPE, mean bias, and FVA versus seasonal-naive. The headline metric is FVA because it tells you whether the model is earning its complexity. On the recent test window LightGBM averaged +25.9%, Croston-TSB +25.4% on its slow-mover slice, Prophet at -7.9%, and naive is the zero reference by definition. Mean bias is tracked separately — even if RMSE looks good, a persistent positive bias means safety stock is going to be systematically low.

### 2.9 ABC-XYZ segmentation + KMeans contractor clustering + DTW branch clusters (`src/models/segmentation.py`)

**WHAT:** Three segmentation analyses — ABC-XYZ matrix for SKUs (revenue × predictability), KMeans k=4 on contractor accounts (concentration features), DTW-distance hierarchical clustering on branch demand shapes.

**HOW IT WORKS:**
- **ABC-XYZ:** A/B/C by cumulative revenue (Pareto thresholds 80/95/100%), X/Y/Z by coefficient of variation thresholds (0.5/1.0).
- **Contractor clusters:** KMeans k=4 on features {total spend, branch count, category breadth, recency} — silhouette score guides k.
- **Branch clusters:** dynamic time warping distance matrix on weekly demand shapes per branch, then `scipy.cluster.hierarchy.linkage(method='average')`.

**DESIGN DECISIONS + WHY:**
- **ABC-XYZ matrix, not just ABC.** A-class with high CV needs aggressive forecasting and high safety stock; A with low CV can run lean. The 2D classification informs the scenario-D segmented policy directly.
- **KMeans for contractors, hierarchical for branches.** Contractors are point clouds in feature space; KMeans is appropriate. Branches are time series; DTW captures shape-similarity that Euclidean would miss.
- **k=4 for contractors.** Elbow at 4 on silhouette; matches business intuition (volume buyers / breadth buyers / occasional / dormant).

**GOTCHAS:**
- ABC thresholds are conventions, not laws — 80/95/100 is the textbook split but distributor-specific Paretos vary.
- DTW is O(n²) — fine for 12 branches but you'd subsample or use FastDTW at 1000s of stores.
- Cluster labels are unstable across runs without seed pinning — `random_state` is set everywhere.

**SPOKEN ANSWER:**
> Three segmentations because the project has three distinct grouping needs. SKUs get an ABC-XYZ matrix — ABC by cumulative revenue using the standard 80/95/100 Pareto cuts, XYZ by coefficient of variation, so each SKU lands in one of nine cells. This feeds directly into Scenario D where A-class get 98% service level and C-class get 90%. Contractor accounts get KMeans with k=4 on spend, branch count, breadth, and recency — elbow at 4 on silhouette, which matches the business categories of volume buyers, breadth buyers, occasional, dormant. Branches get hierarchical clustering on a DTW-distance matrix of their weekly demand shapes because the differentiator is *shape* not *level* — Boston peaks summer, Atlanta is flatter — and Euclidean distance would conflate the two.

### 2.10 Inventory optimization — reorder point, safety stock, EOQ, 4 scenarios (`src/optimization/inventory.py`)

**WHAT:** Computes per branch×SKU reorder points, safety stocks, and EOQ from forecast outputs and lead-time assumptions; produces four working-capital scenarios A/B/C/D for tradeoff comparison.

**HOW IT WORKS:**
- Demand mean and std from forecast over lead-time window.
- Reorder point = `d̄ × L + safety_stock`.
- Safety stock = `z(SL) × √(L × σ_d² + d̄² × σ_L²)` — variance-pooled across demand and lead-time uncertainty.
- EOQ = `√(2 × annual_demand × order_cost / (holding_cost × unit_cost))`.
- Four scenarios:
  - **A:** Flat 95% service level across all SKUs (current-state).
  - **B:** Flat 98% (aggressive availability).
  - **C:** Flat 85% (cost-optimized).
  - **D:** Segmented — A-class SKUs 98%, B-class 95%, C-class 90% (target state).

**DESIGN DECISIONS + WHY:**
- **Variance-pooled SS formula.** Real lead times are stochastic — supplier capacity, freight, weather. Deterministic-L underbuffers systematically.
- **σ_L set as 0.2 × L.** Without supplier data, 20% CV on lead time is the literature default for distribution networks.
- **Four scenarios, not one optimal.** Optimal depends on the cost of stockout vs cost of capital — both are business inputs the user must supply. Scenarios force the explicit tradeoff conversation.
- **Z-score from normal CDF.** Acknowledged limit for intermittent series; documented in gotchas.

**GOTCHAS:**
- Normal assumption breaks for slow movers; would switch to fill-rate calculation for those SKUs in production.
- Lead-time std is assumed (0.2 × L) not measured. Production would derive from supplier data.
- Excess-inventory flag returns 0 rows here because synthetic on-hand sawtooths around reorder point; real data would show meaningful excess.

**SPOKEN ANSWER:**
> Inventory layer takes the forecast outputs and turns them into reorder points and safety stocks. Safety stock uses the variance-pooled formula `z × √(L × σ_d² + d̄² × σ_L²)` — that captures both demand variability inside the lead-time window and lead-time variability itself. The naive form `z × σ × √L` assumes deterministic lead time, which is wrong for building materials because supplier capacity, freight, and weather all stretch the lead window. Lead-time CV is 20% — distribution-network default, would be measured in production. Then I run four scenarios: flat 95%, aggressive 98%, cost-optimized 85%, and segmented where A-class SKUs get 98% and C-class get 90%. Scenario D is the operational recommendation — same weighted availability as B on the SKUs that matter, ~7% lower working capital.

### 2.11 MLflow experiment tracking (`mlruns/`, integrated in `src/models/forecasting.py`)

**WHAT:** Logs Optuna trial params, fit metrics, test metrics, feature importance, and the LightGBM booster artifact per experiment under `mlruns/`.

**HOW IT WORKS:**
- Two experiments: `demand_forecasting` (per-Optuna-trial runs + final best run), `segmentation` (clustering diagnostics).
- Each run: `mlflow.log_params(trial.params)`, `mlflow.log_metrics({rmse, mae, mape, bias, fva})`, `mlflow.lightgbm.log_model(booster, 'model')`.
- `mlflow ui` launchable locally; same API points at Databricks-managed tracking with one env var change.

**DESIGN DECISIONS + WHY:**
- **Per-trial logging, not just per-experiment.** Lets you backtrack which Optuna decision moved the needle.
- **Booster artifact saved at run level.** Deployment is `mlflow.lightgbm.load_model(run_id)` — the production path is identical to the local dev path.
- **Two experiments, not one.** Forecasting and segmentation have different metric vocabularies; separating them keeps the UI clean.

**GOTCHAS:**
- `artifact_path` deprecation warning in MLflow 3.x; harmless but noted in logs.
- Local `mlruns/` directory not committed (excluded in `.gitignore`) — production would use a managed tracking server.

**SPOKEN ANSWER:**
> MLflow tracks every Optuna trial — params, val and test metrics, feature importance, plus the LightGBM booster artifact. Two experiments: demand_forecasting and segmentation, separated because the metric vocabularies are different. The reproducibility win is concrete: if a future run regresses, I open `mlflow ui` and compare runs side-by-side to find the parameter that changed. The deployment win is that loading the model is `mlflow.lightgbm.load_model(run_id)` — same API path locally and against a Databricks-managed tracking server, which is what the production version uses.

### 2.12 Plotly Dash dashboard — 4 tabs, dbc.themes.CYBORG (`src/dashboard/app.py`)

**WHAT:** Four-tab dark-theme Plotly Dash app surfacing branch overview, forecast diagnostics, segmentation views, and scenario comparison.

**HOW IT WORKS:**
- `dash` app with `dbc.themes.CYBORG` for the dark UI shell.
- Tab 1: branch overview map + KPI cards.
- Tab 2: forecast vs actual line charts per (branch, SKU), with model-selection dropdown.
- Tab 3: ABC-XYZ matrix heatmap + contractor cluster scatter + branch dendrogram.
- Tab 4: scenario comparison bar chart + safety stock table.
- Backed by parquet outputs from `data/processed/`; no DB.

**DESIGN DECISIONS + WHY:**
- **Plotly Dash over Streamlit.** Multi-tab structure with stateful callbacks is cleaner in Dash; Streamlit re-runs the whole script per interaction.
- **Dark theme.** Matches enterprise BI conventions; high contrast for projection in a meeting.
- **Parquet-backed, not live DB.** Project artifact, not production. Production would point Dash callbacks at the Delta tables.

**GOTCHAS:**
- Dash callbacks load full parquet on each tab switch — fine at portfolio scale, would lazy-load at production scale.
- No auth layer — added for a deployed version.

**SPOKEN ANSWER:**
> Dashboard is Plotly Dash with the CYBORG dark theme. Four tabs: branch overview with a map and KPI cards, forecast diagnostics where you can pick a branch-SKU and see actual versus each model's forecast, segmentation views with the ABC-XYZ heatmap and contractor cluster scatter, and scenario comparison with safety stock tables. It's parquet-backed not DB-backed, which is the right tradeoff for a portfolio artifact. In production the Dash callbacks would point at the Delta outputs from the Databricks pipeline. I chose Dash over Streamlit because the multi-tab stateful interaction is cleaner — Streamlit re-runs the entire script per interaction, which gets awkward at four tabs.

### 2.13 Databricks port — Delta tables, PySpark features, LightGBM on Spark (`databricks/01-03`)

**WHAT:** Three Databricks notebooks porting the pipeline to Spark — Delta table registration, PySpark feature engineering, LightGBM training with MLflow on Spark.

**HOW IT WORKS:**
- **`01_register_delta_tables.py`:** Reads local parquet, writes Delta tables in DBFS with schema enforcement and partitioning by branch.
- **`02_feature_engineering_pyspark.py`:** Rewrites lag/rolling/calendar features in PySpark — `Window.partitionBy('branch_id', 'sku_id').orderBy('week_start_date')` with `lag()` and rolling aggregates.
- **`03_forecasting_lightgbm_mlflow.py`:** Uses SynapseML's distributed LightGBM (`LightGBMRegressor`), trains on the Delta-backed feature table, logs to Databricks-managed MLflow.

**DESIGN DECISIONS + WHY:**
- **Delta over Parquet.** Time travel, schema enforcement, optimistic concurrency — all features local Parquet lacks. Real-world distributor pipelines need these.
- **SynapseML LightGBM, not Pandas API on Spark.** Pandas API would single-node-collect on driver; SynapseML's `LightGBMRegressor` is truly distributed.
- **Branch partitioning.** Most queries filter by branch; partition pruning helps.

**GOTCHAS:**
- Honest disclosure: the notebooks are written and runnable but the candidate may not yet have ported to a live Databricks workspace at interview time. If asked "have you run this on Databricks", be honest.
- SynapseML version pinning matters; documented in notebook header.

**SPOKEN ANSWER:**
> The Databricks port is three notebooks. First registers the parquet outputs as Delta tables in DBFS, partitioned by branch and with schema enforcement. Second rewrites the feature engineering in PySpark using window functions — `Window.partitionBy('branch_id', 'sku_id').orderBy('week_start_date')` with `lag()` and rolling aggregates. Third trains LightGBM using SynapseML's distributed `LightGBMRegressor`, which is the right choice over Pandas-API-on-Spark because the latter would single-node-collect on the driver. MLflow logging is identical — same params, metrics, artifacts — but pointed at the Databricks-managed tracking server. The notebooks are written and runnable; I'd be upfront if asked whether I've run them against a live Community Edition workspace yet.

### 2.14 SQL analytics layer — DuckDB local + Spark SQL Databricks (`sql/local`, `sql/databricks`)

**WHAT:** Eight SQL queries — 4 against local DuckDB-on-parquet, 4 against Databricks Spark SQL — surfacing top branches, FVA by category, contractor concentration, seasonal shift, scenario comparison, slow-mover risk.

**HOW IT WORKS:**
- **DuckDB queries** run against the parquet outputs directly — no ETL needed, DuckDB reads parquet natively.
- **Spark SQL queries** target the Delta tables created by `databricks/01`.
- Same business questions answered in both dialects — proves the SQL layer is portable.

**DESIGN DECISIONS + WHY:**
- **DuckDB local, not Postgres.** Zero install, native parquet — fastest path for portfolio reproducibility.
- **Same questions in both dialects.** Demonstrates the "build once, query anywhere" claim and shows fluency in both.
- **Aggregations chosen for dashboard payload.** Each query maps to a panel in the Dash or Power BI report.

**GOTCHAS:**
- SQL is not the modeling layer — make this clear; it's the analytics layer on top of model outputs.
- DuckDB doesn't support all Spark SQL functions natively (`PERCENTILE_DISC` differs); two queries have minor dialect differences documented in headers.

**SPOKEN ANSWER:**
> The SQL layer is 8 queries — 4 DuckDB-against-parquet for local reproducibility, 4 Spark SQL against the Delta tables for the Databricks path. Same business questions in both dialects: top branches by forecast volume, FVA by category, contractor concentration, seasonal shift, scenario comparison, slow-mover risk. The DuckDB choice is intentional — it reads parquet natively so there's no ETL step, which makes the project clonable and runnable in one minute. Each query maps to a panel in either the Dash dashboard or the Power BI report — so the SQL is the contract between modeling outputs and the surface layer.

### 2.15 Power BI design (`docs/BUILD_POWERBI.md`)

**WHAT:** Implementation guide with DAX measures and report-page specifications for a Power BI deployment fed by the same parquet/Delta outputs.

**HOW IT WORKS:**
- Page specs for branch overview, forecast diagnostics, segmentation, scenario comparison — mirror the Dash tabs.
- DAX measures defined: `FVA_vs_Naive`, `Safety_Stock_$`, `Service_Level_Weighted`, `Top_3_Contractor_Share`.
- Direct Query or Import refresh against Delta tables described.
- No actual `.pbix` file shipped — the build guide is the deliverable.

**DESIGN DECISIONS + WHY:**
- **Guide, not file.** Power BI files are binary and platform-locked; a written implementation guide is more portable and review-able.
- **Same chart primitives as Dash.** Plotly and Power BI both do scatter/bar/line natively — reuse is conceptual not literal.
- **DAX measures, not calculated columns.** Measures recompute against filter context; calculated columns bloat the data model.

**GOTCHAS:**
- Honest disclosure: no `.pbix` exists yet. Be upfront — "I have the build guide; the model can be assembled in an hour from it but I haven't yet."

**SPOKEN ANSWER:**
> Power BI is documented as an implementation guide rather than a shipped `.pbix` file. The guide specifies four report pages mirroring the Dash tabs, plus four DAX measures: FVA versus naive, safety stock in dollars, weighted service level, top-3 contractor share. The data path is Direct Query or Import against the Delta tables produced by the Databricks pipeline — so the production version of this dashboard is Power BI on Delta, and the Dash app is the local-development equivalent that lives in version control. I'd be honest in an interview that I haven't yet built the `.pbix` — the guide is precise enough that I could assemble it in an hour from the DAX measures and page specs.

---

## 3. Anticipated questions (drill-down)

### 3.A Project framing / behavioral

**Q-A1. Why did you pick this project for your portfolio?**
> Building products is a real category my resume targets — distributors and OEMs in roofing/siding/exteriors. The problem (branch-level demand under multiple demand regimes) is generalizable across distribution businesses, and the components — global gradient model, intermittent forecasting, scenario-based inventory optimization, MLflow, Databricks port — let me demonstrate breadth across the modern stack on one cohesive artifact. *Follow-up curveball: "Why synthetic data?"* — Real distributor data is proprietary. Synthesizing lets me bake in the exact phenomena I want to model (stockouts, climate effects, concentration) deliberately rather than hope they're present in a Kaggle download.

**Q-A2. What would you do differently if you started over?**
> Three things. First, set up a unit test harness from day one — `tests/` is empty and I'd want regression tests on the feature builder before I'd be comfortable iterating on lag choices in production. Second, build the Power BI artifact alongside the Dash dashboard so I had an actual `.pbix` to demo. Third, I'd add a hierarchical reconciliation step (`scikit-hts` MinT) so the SKU-level forecasts sum to category-level forecasts — this is the right way to handle the Prophet disaggregation. *Follow-up: "Why not now?"* — Honest answer: scope and time. Those are the first three items on my next-iteration list.

**Q-A3. What was the hardest part?**
> Stockout handling. First version trained on raw units and the model learned truncated weeks as ground truth — safety stock recommendations came out systematically low. Fix is one line of sample weighting, but the diagnosis took looking at residuals stratified by stockout flag. *Follow-up: "How did you know to look there?"* — Bias metric was persistently negative. RMSE looked fine, but a persistent negative bias means something is being clipped. Stockout flag was the obvious next stratification variable.

**Q-A4. What surprised you in the analysis?**
> Prophet underperforming naive on real macro data — FVA -7.9%. On synthetic data Prophet was modestly positive because I had baked seasonality strongly. Real macro signal turned out to be too weak relative to the noise added by the SKU-share disaggregation step. The lesson is that model selection isn't free — adding a model that wins on some SKUs but loses on most isn't a net gain. *Follow-up: "What would you do about Prophet now?"* — Deprecate (branch×category) Prophet for this dataset. Keep the code path because for a real distributor with stronger macro-demand correlation Prophet has a clear lane.

**Q-A5. How did you measure success?**
> FVA versus seasonal-naive is the headline. +25.9% on LightGBM is the deliverable number. But the secondary lens is the inventory layer: same forecast quality, what's the safety-stock dollar impact under Scenario D vs Scenario A. The model isn't useful if it doesn't move the working-capital dial. *Follow-up: "What's the dollar impact?"* — Scenario A ran $399K safety stock; Scenario D $466K weighted avg 93.6% service level — D doesn't dominate A on dollars but dominates on availability per dollar for the SKUs that matter.

### 3.B Modeling depth

**Q-B1. Why walk-forward instead of k-fold?**
> Time series have temporal dependence — k-fold randomizes index, leaking future into past. Walk-forward respects the arrow of time. *Follow-up: "Why a 53-week gap between val and test?"* — Optuna selection on val MAPE introduces selection bias. Without a gap, val and test draw from temporally adjacent dynamics and the test FVA reads optimistic. 53 weeks ensures at least a full seasonal cycle separates the two.

**Q-B2. Why four models and not just LightGBM?**
> Three demand regimes need different priors. Seasonal SKUs are LightGBM's sweet spot. Aggregated patterns at category level have Fourier decomposition signal that Prophet recovers cleanly. Intermittent slow-movers have demand probabilities and demand sizes that need to be smoothed separately — Croston-TSB is the textbook fix. Plus seasonal-naive as the FVA reference. *Follow-up: "Couldn't LightGBM cover all three?"* — On this dataset, mostly yes — LightGBM is the per-SKU winner on ~85% of series. Croston-TSB wins meaningfully on the 7 slow-movers because tree models over-forecast on series with mostly zeros. So in practice: LightGBM + Croston, with Prophet on the bench.

**Q-B3. Walk me through model selection per SKU.**
> Train all four on the train window, eval all four on the test window, pick the lowest test-RMSE per (branch × SKU). Tie-break to LightGBM because it's the most stable across regime shifts. Latest distribution: ~85% LightGBM, ~5% Croston, ~8% Prophet, ~2% naive. *Follow-up: "What about peeking — picking the best model on test is itself a selection?"* — Fair probe. The defense is val-set selection for the LightGBM HPO is already separated from test by the gap; the per-SKU model selection on test is final-deployment pick, not a tuning loop. In production I'd rotate: select per-SKU model on a rolling validation window, then deploy.

**Q-B4. Why FVA over MAPE for the headline?**
> MAPE alone tells you absolute error rate, not whether complexity is earning its keep. A 30% MAPE model sounds bad until you learn naive is 60%. FVA — `(naive_rmse - model_rmse) / naive_rmse` — gives the percent improvement over the simplest possible forecast a human would have. That's the question that matters: is the model adding value beyond what the human would already do. *Follow-up: "Why RMSE in the FVA formula and not MAPE?"* — RMSE penalizes large misses more, which matches the safety-stock objective. Safety stock is set by the upper tail of demand-error distribution, so a metric that puts mass on the tail is the right one for the downstream use case.

**Q-B5. Why did Prophet underperform on real data?**
> Two reasons. First, the housing-starts regressor coefficient is shrunk near zero on this data because the demand-macro correlation is weak relative to the seasonal+noise components. Second, the SKU-share disaggregation step assumes stable SKU mix within (branch, category) — which is a reasonable approximation but adds variance that doesn't exist at the aggregate forecast level. Net result: Prophet wins at the aggregate but loses at SKU level once disaggregation noise compounds. *Follow-up: "Would you keep it?"* — On this dataset, no. On a real distributor with stronger macro-demand coupling, yes — keep the code path.

**Q-B6. Why 20 Optuna trials? That's small.**
> Defensible for portfolio scope. In early experiments I saw diminishing returns on val RMSE past ~15 trials on this 7-dimensional search space. In production I'd run 50-100 trials at the monthly retune cadence, plus a `MedianPruner` to early-stop bad trials. *Follow-up: "How would you decide when to retune?"* — Trigger-based, not calendar-based. If rolling 4-week test MAPE drifts more than 10% above the deployed baseline, schedule a retune. Calendar-only retuning wastes compute when the model is still fit.

**Q-B7. What if your labels are noisy?**
> Two layers. First, the stockout flag IS a noise indicator — that's why it's a sample weight, not a feature alone. Second, for label noise beyond stockouts (e.g., manual data-entry errors), I'd run a residual diagnostic: train, score, flag rows where residual > 3σ, sample-review them. A "label denoiser" step before retraining is overkill for this scale but a known production pattern. *Follow-up: "Robust regression?"* — Could use a Huber loss in LightGBM instead of MSE. Tradeoff: Huber dampens the tail, which is exactly the demand-spike signal safety stock wants to capture. So Huber would shrink RMSE numbers but hurt the downstream inventory math. I keep MSE.

**Q-B8. How do you know the model isn't overfitting?**
> Val MAPE and test MAPE diverge by less than 5 percentage points — the cleanest signal that generalization is happening. Plus the per-SKU FVA distribution: centered around +20% with a long positive tail, not a few outlier wins. If I were overfitting, test FVA would collapse to zero or go negative on most SKUs. *Follow-up: "Could the LightGBM be memorizing the SKU IDs?"* — Possible in principle. Defense: lag features carry the per-series signal, so SKU identity is doing categorical-embedding work, not memorization. Test on held-out time periods (which is what walk-forward does) breaks pure-memorization wins.

### 3.C Feature engineering

**Q-C1. How do you prevent feature leakage?**
> Every lag and rolling stat is computed with `.shift(1)` first — guarantees the current week never sees its own target. External signals (FRED, NOAA) are joined on week-start-date with the macro values from the *prior* observed month — never the current week. Stockout flag is computed at week-end and used as a feature for the *next* week's prediction. *Follow-up: "What about the contractor-concentration feature?"* — Computed at branch-week level using the prior week's contractor mix, never current week. Documented in `build_features.py`.

**Q-C2. Why those specific lag choices — 1, 4, 13, 52?**
> 1 captures momentum, 4 monthly cycle, 13 quarterly, 52 yearly seasonality. I tested adding 2, 8, 26 — none improved val RMSE measurably. The principle is parsimony: each lag adds 960 rows to drop from the lag-warmup window, so each one has to earn its keep. *Follow-up: "What about lag-104?"* — 5 years of history doesn't leave enough usable training weeks after a 104-week warmup. Plus there's no business reason to expect a 2-year cycle in distributor demand.

**Q-C3. Weather aggregation level — why climate zone, not branch?**
> NOAA station data is noisy at single-station scale. Climate zone aggregates 3-4 stations per zone, which smooths the noise without losing the regional signal. There are only 12 branches in this dataset — pooling to 4 zones cuts NOAA query volume by 3x while gaining smoothness. In production with 500+ branches I'd go to ZIP-level weather with nearest-station kriging.

### 3.D Inventory / business

**Q-D1. Walk through the reorder-point math.**
> Reorder point = `mean_demand_in_lead_time + safety_stock`. Mean demand in lead time is `d̄ × L` where d̄ is forecast mean weekly demand and L is mean lead time. Safety stock is `z(SL) × √(L × σ_d² + d̄² × σ_L²)` — variance-pooled. Below this on-hand level, place an order of size EOQ. *Follow-up: "Why √(L × σ_d² + d̄² × σ_L²) instead of σ_d × √L?"* — Lead time is stochastic. `σ_d × √L` assumes L is deterministic; reality is L has its own variance from supplier capacity, freight, weather. The pooled form is the textbook correction.

**Q-D2. Why pooled variance for safety stock?**
> Building materials lead time stretches under demand shocks — when housing starts surge, suppliers ration, freight tightens, weather affects truck routes. So demand-variance and lead-time variance correlate during exactly the moments safety stock matters most. The pooled form acknowledges both contribute to inventory-window variance. The naive `z × σ_d × √L` underbuffers by 15-25% on real distributor data — documented in supply-chain literature.

**Q-D3. What changes if lead time is stochastic and you have actual supplier data?**
> Two changes. First, estimate σ_L empirically from supplier delivery records rather than assuming `0.2 × L`. Second, if there's supplier-demand correlation in the data, switch to a copula-based safety stock that captures joint distribution. The pooled-variance formula assumes independence; for high-correlation suppliers (sole-source) the formula understates the right safety stock.

### 3.E Production / system

**Q-E1. How do you productionize this?**
> Three layers. **Ingestion:** Databricks Auto Loader watching an S3 bucket where the ERP drops weekly sales extracts. Write to Delta with schema enforcement and time travel. **Training:** scheduled Workflow runs `build_features → forecasting` weekly. Optuna retune monthly OR triggered by 10% MAPE drift. New best promoted via MLflow Model Registry (None → Staging → Production). **Serving:** REST endpoint (Databricks Model Serving) returning next-13-week forecast plus reorder recommendation per branch×SKU. Dashboard reads from the inventory_scenarios Delta table. *Follow-up: "How do you detect drift?"* — Track 4-week rolling bias and FVA per SKU. Alert if bias > 2σ for two consecutive weeks (demand regime shift indicator) or FVA drops below +10% on a rolling window (model degradation).

**Q-E2. Why Databricks port and not Snowflake or BigQuery?**
> Databricks is the natural fit because it combines the ML training surface (SynapseML LightGBM, MLflow) with the storage layer (Delta) in one platform. Snowflake/BigQuery would handle the SQL analytics and storage well but I'd need a separate ML platform (SageMaker, Vertex) and an extra integration to MLflow. The resume claim is specifically Databricks, so the port targets that. *Follow-up: "What if the client is already on Snowflake?"* — Then the model artifact moves but the training stays on Databricks; Snowflake reads model outputs via the Delta-to-Iceberg bridge or direct parquet. The training-vs-serving split is cleanly portable.

**Q-E3. What latency targets does this need to hit?**
> Forecasting is batch — weekly cadence on a Spark job, no online latency requirement. Inventory scenario computation is offline. The serving endpoint is the only latency-sensitive piece, and it's a model load + simple prediction — comfortably sub-100ms p99 on a Databricks Model Serving endpoint. Dashboard refresh is daily; Direct Query in Power BI would hit Delta with sub-second responsiveness for the page-level aggregations. None of this is real-time; trying to make it real-time would be over-engineering.

---

## 4. Numbers cheat sheet (memorize)

| Domain | Numbers |
|---|---|
| **Data shape** | 12 branches × 80 SKUs × 5 years weekly = **250K** sales rows. 200 contractor accounts. |
| **External — macro** | FRED HOUST 936-1807, PERMIT 1079-1923, 72 monthly rows each = 6 years. |
| **External — weather** | NOAA GSOM, 216 monthly station rows across 3 climate zones, avg temp 31.2-38.1°F window, precip 0.0-1.4in. Joined to **783 weekly external rows**. |
| **Feature table** | **43 columns**. Lags 1/4/13/52. Rolling 4/13/26 weeks mean+std. 6 categorical features handled natively by LightGBM. |
| **Modeling series** | 12 × 80 = **960** branch×SKU series. ~209 usable weeks after lag-52 warmup. |
| **Walk-forward split** | 104 train / 26 val / **53 gap** / 26 test = 209 weeks. |
| **Models** | **4** — seasonal-naive, LightGBM global, Prophet per (branch×category) = **36 models**, Croston-TSB on 7 slow-movers × 12 branches = 84 series. |
| **LightGBM HPO** | **20 Optuna trials**. Tuning `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. |
| **Real-data results** | **LightGBM avg_rmse 3.65, MAPE 0.74, FVA +25.9% vs naive.** Croston-TSB avg_rmse 0.64, FVA +25.4%. **Prophet FVA -7.9%** (worse than naive — surface unprompted). Seasonal-naive avg_rmse 4.99. |
| **Per-SKU model winners** | LightGBM ~869 of 960, Croston-TSB ~49, Prophet ~27, naive ~15. |
| **Segmentation** | ABC-XYZ (3×3 = 9 cells). **KMeans k=4** contractor clusters. DTW hierarchical 12-branch clustering. |
| **Inventory math** | Safety stock = `z(SL) × √(L × σ_d² + d̄² × σ_L²)`. σ_L = **0.2 × L** (distribution-network default). EOQ classical formula. |
| **Scenarios** | **A** flat 95% SL ($399K SS), **B** flat 98% ($498K), **C** flat 85% (cost-min), **D** segmented A=98/B=95/C=90 ($466K, weighted avg 93.6%). |
| **Dashboard** | Plotly Dash, **4 tabs**, dark CYBORG theme, port 8050, parquet-backed. |
| **Databricks** | **3 notebooks**: 01 Delta tables, 02 PySpark features, 03 SynapseML LightGBM + MLflow. |
| **SQL** | **8 queries**: 4 DuckDB local, 4 Spark SQL Databricks. Same business questions, both dialects. |
| **Power BI** | Implementation guide + **4 DAX measures**: FVA_vs_Naive, Safety_Stock_$, Service_Level_Weighted, Top_3_Contractor_Share. |
| **Repo** | Public: github.com/AbhayJuloori/building-products-forecast |

---

## 5. Honest weaknesses to surface proactively

Interviewers respect candidates who name limits before being asked. Pick **two** of these to mention unprompted during the project walkthrough — credibility skyrockets.

1. **Synthetic data caveat.** Real distributor data is proprietary; this dataset is synthesized. Be transparent. Frame as: "the deliverable is the modeling and inventory layer, not the dataset itself — and synthetic let me bake in the exact phenomena I wanted to model (stockouts, climate effects, concentration) deliberately rather than hope they'd be in a Kaggle download."
2. **Prophet underperformed on real macro (FVA -7.9%).** Surface this unprompted: "Honest finding — Prophet at (branch × category) granularity came in worse than naive on real macro data. I'd deprecate it for this specific dataset, keep the code path for distributors with stronger macro-demand coupling." This shows you read your own metrics critically.
3. **Slow-mover excess-inventory flag returns 0 rows.** Synthetic on-hand sawtooth around reorder point keeps coverage too low to trigger excess flag. Logic is correct; real data would show meaningful excess. State openly.
4. **No tests yet — `tests/` is empty.** The first thing to add is a regression test on `build_features.py` covering the `.shift(1)` invariant on lags and rolling stats. Acknowledge proactively if asked about code quality.
5. **Single-node training on the local pipeline.** The local LightGBM run uses pandas + lightgbm — single-machine. The Databricks port (SynapseML LightGBM) is the distributed claim. Be clear which is which.
6. **20 Optuna trials is small.** Defensible for portfolio scope (diminishing returns past ~15) but real production would run 50-100 with `MedianPruner` early-stopping.
7. **Safety-stock formula assumes normality.** Breaks for intermittent demand; documented limit. Would switch to fill-rate-based stock for Croston-served SKUs in production.

---

## 6. 3-day study plan

**Day 1 (warm up + memorize):**
- Re-read this guide front to back once. Practice **Section 1.1** (30-sec) and **Section 1.2** (90-sec) out loud 5x each.
- Memorize **Section 4** cheat sheet. Use a flashcard app or just rewrite the table from memory until you can produce LightGBM 25.9%, Croston 25.4%, Prophet -7.9% without thinking.

**Day 2 (component depth):**
- Drill **Sections 2.1 through 2.8** — data, external fetch, features, all four models, eval. For each one practice the SPOKEN ANSWER paragraph out loud once, then close the doc and re-deliver it from memory.
- Re-read `docs/INTERVIEW_PREP.md` Q1 through Q6 for the additional depth on stockouts, walk-forward, FVA, safety stock, scenario D, LightGBM-vs-alternatives.

**Day 3 (production + practice):**
- Drill **Sections 2.9 through 2.15** — segmentation, inventory, MLflow, dashboard, Databricks, SQL, Power BI.
- Read **Section 5** (weaknesses) twice. Decide **two** you will surface unprompted in the walkthrough — write them down.
- Out loud, deliver the **5-minute deep dive (1.3)** end to end without notes. Time yourself. Re-deliver until under 5:30.
- Cold-practice **Section 3** questions. For each, give the answer paragraph then immediately answer the follow-up curveball. If you can't answer cleanly, re-read the relevant Section 2 component.

The morning of the interview: re-read **Sections 1.1, 1.2, 4, 5** only. Walk in confident on those four pages.

---

*This guide is a companion to `docs/INTERVIEW_PREP.md`. That file is the Q&A drill-down with code citations. This file is the structured study path. Use both.*
