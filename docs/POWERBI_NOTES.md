# Power BI Companion Notes

The Dash app mirrors what would be deployed in Power BI for a real distributor.
This doc maps the four tabs to Power BI report pages and notes the model
relationships.

## Data model (star schema)

```
              ┌──────────────────┐
              │   DimBranch      │
              │ branch_id (PK)   │
              │ region, climate  │
              └────────┬─────────┘
                       │
                       │
   ┌──────────────┐    │    ┌──────────────────┐
   │  DimProduct  │    │    │   DimDate        │
   │ sku_id (PK)  │    │    │ week_start (PK)  │
   │ category     │    │    │ week, month, qtr │
   └──────┬───────┘    │    └────────┬─────────┘
          │            │             │
          │     ┌──────▼─────────────▼──────┐
          └────►│   FactSales (weekly)      │
                │  branch_id, sku_id, week  │
                │  units_sold, revenue      │
                └────────────┬──────────────┘
                             │
                             │
                ┌────────────▼──────────────┐
                │   FactForecast (weekly)   │
                │  branch_id, sku_id, week  │
                │  y_pred, model            │
                └───────────────────────────┘

         ┌──────────────────────────────────┐
         │   FactInventoryScenario          │
         │  branch_id, sku_id, scenario     │
         │  safety_stock, reorder_point     │
         └──────────────────────────────────┘
```

All fact tables joined to the dimensions on the natural keys. Use
**week_start_date** as the relationship key for the DimDate join.

## Report pages

### Page 1 — Branch Demand Overview
- **Visuals**: filled US map (branch locations), KPI cards, line chart
- **Measures**:
  - `[Forecasted Units Next 13W] = SUMX(FILTER(FactForecast, FactForecast[week_start] BETWEEN TODAY() AND TODAY()+91), FactForecast[y_pred])`
  - `[Forecast vs PY %] = DIVIDE([Forecasted Units 13W], CALCULATE([Forecasted Units 13W], SAMEPERIODLASTYEAR(DimDate[week_start]))) - 1`

### Page 2 — Product Mix & Forecast Error
- **Visuals**: bar chart top-bias SKUs, MAPE vs volume scatter, ABC-XYZ matrix
- **Measure**: `[Bias] = AVERAGEX(FactForecast, FactForecast[y_pred] - FactForecast[y_true])`

### Page 3 — Inventory & Replenishment
- **Visuals**: scenario slicer, recommended-action table, scenario comparison bar
- **Parameter slicer** drives the scenario filter on FactInventoryScenario.
- **Measure**: `[Suggested Order Qty] = IF([On Hand] < [Reorder Point], MAX(0, [Reorder Point] - [On Hand]), 0)`

### Page 4 — Contractor Insights
- **Visuals**: contractor scatter (spend × frequency, colored by cluster),
  branch concentration heatmap, segment seasonal pattern

## Power BI refresh strategy (production)

- **Mode**: Import (not DirectQuery) — the parquet outputs are small enough
  (~200 MB total) and import gives faster slicer response than DirectQuery
  against Delta.
- **Schedule**: weekly Monday morning, matching the weekly pipeline run.
- **Gateway**: enterprise gateway to the Delta Lake on the company's cloud.

## What's intentionally NOT in this project

- An actual Power BI .pbix file. The interview talking point is: I built the
  data model and the report definitions, and the same data flows into either
  Dash (for portfolio demo) or Power BI (for production rollout). The
  visuals + measures + relationships above are the production spec.
