# Build the Power BI Report

This is the step-by-step to materialize the Power BI deliverable. Aim for
~30–45 minutes start to .pbix-in-repo. Power BI Desktop is **free on Windows**.

> **No Windows machine?** Two options:
> 1. Power BI Desktop runs on Windows VM (Parallels / UTM on Mac, free tier OK)
> 2. Open Source alternative — build the same report in **Power BI service**
>    (browser, requires Microsoft 365 work/school account; personal email
>    not always accepted). The browser version lacks parts of the Desktop
>    feature set but the report is shareable as a URL.
>
> If neither is workable, the honest fallback is to update the resume to:
> "delivered Plotly Dash dashboard mirroring Power BI report spec (data model + DAX measures in repo)."

## Step 1 — Install Power BI Desktop

- https://www.microsoft.com/en-us/download/details.aspx?id=58494
- Sign in with personal Microsoft account on first launch.

## Step 2 — Connect to the data

Two acceptable sources (pick one):

### Option A — Connect to Databricks (production-realistic)

1. **Home → Get Data → Azure → Azure Databricks** (yes, even for AWS Databricks).
2. Server hostname: `dbc-c715dbcf-567f.cloud.databricks.com`
3. HTTP Path: `/sql/1.0/warehouses/4c4a2523505197b3` (your Serverless Starter Warehouse)
4. Authentication: **Personal Access Token** → paste the token from `.env`
5. Catalog: `workspace`, Schema: `building_products`
6. Load these tables (Import mode, NOT DirectQuery):
   - `branches`
   - `products`
   - `sales_history`
   - `forecasts_test`
   - `forecast_evaluation`
   - `sku_abc_xyz`
   - `inventory_scenarios`
   - `contractor_segments`

### Option B — Connect to local parquet (simpler, no warehouse)

1. **Home → Get Data → File → Parquet**
2. Point to `~/projects/building-products-forecast/data/raw/branches.parquet`
3. Repeat for each parquet under `data/raw/`, `data/processed/`, `data/external/external_weekly.parquet`
4. Or load all at once: **Get Data → Folder** pointing at `data/` and filter by `.parquet` extension.

Option A is the better interview story. Option B is faster to set up.

## Step 3 — Set up relationships

**Model view → Manage relationships.** Create these:

| From | To | Cardinality |
|---|---|---|
| `sales_history.branch_id` | `branches.branch_id` | many-to-one |
| `sales_history.sku_id` | `products.sku_id` | many-to-one |
| `forecasts_test.branch_id` | `branches.branch_id` | many-to-one |
| `forecasts_test.sku_id` | `products.sku_id` | many-to-one |
| `forecast_evaluation.sku_id` | `products.sku_id` | many-to-one |
| `forecast_evaluation.branch_id` | `branches.branch_id` | many-to-one |
| `inventory_scenarios.branch_id` | `branches.branch_id` | many-to-one |
| `inventory_scenarios.sku_id` | `products.sku_id` | many-to-one |
| `contractor_segments.branch_id` | `branches.branch_id` | many-to-one |
| `sku_abc_xyz.sku_id` | `products.sku_id` | one-to-one |

Add a **Date table**: New table → DAX:
```dax
DimDate = ADDCOLUMNS(
    CALENDAR(DATE(2020,1,1), DATE(2024,12,31)),
    "Year", YEAR([Date]),
    "Quarter", "Q" & QUARTER([Date]),
    "YearQuarter", YEAR([Date]) & "-Q" & QUARTER([Date]),
    "Month", FORMAT([Date], "MMM"),
    "WeekStart", [Date] - WEEKDAY([Date], 2) + 1
)
```
Relate `DimDate[WeekStart] → sales_history[week_start_date]` and same for
`forecasts_test[week_start_date]`. Mark DimDate as Date Table.

## Step 4 — Add measures

In Modeling tab → New measure:

```dax
[Total Units Sold] = SUM(sales_history[units_sold])

[Forecasted Units 13W] =
    CALCULATE(
        SUM(forecasts_test[y_pred]),
        forecasts_test[week_start_date] >= TODAY(),
        forecasts_test[week_start_date] < TODAY() + 91
    )

[Forecast Bias] =
    AVERAGEX(forecasts_test, forecasts_test[y_pred] - forecasts_test[y_true])

[MAPE] =
    AVERAGEX(
        FILTER(forecasts_test, forecasts_test[y_true] > 0),
        ABS(forecasts_test[y_pred] - forecasts_test[y_true]) / forecasts_test[y_true]
    )

[Forecast vs PY %] =
    DIVIDE([Forecasted Units 13W],
           CALCULATE([Forecasted Units 13W], SAMEPERIODLASTYEAR(DimDate[Date])))
    - 1

[Total Safety Stock $] = SUM(inventory_scenarios[safety_stock_cost])

[SKUs Below Reorder Pt] =
    CALCULATE(
        DISTINCTCOUNT(inventory_scenarios[sku_id]),
        inventory_scenarios[on_hand_units] < inventory_scenarios[reorder_point_units]
    )

[Scenario] =
    SELECTEDVALUE(inventory_scenarios[scenario], "A")
```

## Step 5 — Build the 4 report pages

### Page 1 — Branch Demand Overview

**Visuals:**
- **Filled Map** — Location: `branches.name`, Size: `[Forecasted Units 13W]`, Color: `branches.region`
- **4 KPI cards** in a row: Total Units Sold, Forecasted Units 13W, Forecast vs PY %, Avg MAPE
- **Line chart**: X = `DimDate[WeekStart]`, Y values = `[Total Units Sold]` and `SUM(forecasts_test[y_pred])`. Filter to one branch + one category via slicers.
- **Slicers**: `branches[name]`, `products[category]`

### Page 2 — Product Mix & Forecast Error

- **Bar chart**: top 20 SKUs by `ABS([Forecast Bias])`, color by sign(bias).
- **Scatter**: X = avg `[Total Units Sold]` log scale, Y = `[MAPE]`, points = SKUs.
- **Matrix**: rows = `sku_abc_xyz[abc_class]`, cols = `sku_abc_xyz[xyz_class]`, values = COUNT(sku_id). Conditional formatting (heatmap).
- **Slicers**: region, category.

### Page 3 — Inventory & Replenishment

- **Scenario slicer**: `inventory_scenarios[scenario]` as buttons (A/B/C/D).
- **4 KPI cards**: `[Total Safety Stock $]`, Avg Service Level, `[SKUs Below Reorder Pt]`, Excess Inventory $.
- **Table**: SKUs where on_hand < reorder_point. Columns: branch, sku, category, on_hand, reorder_point, eoq_units, suggested_order_qty (DAX: `MAX(0, reorder_point - on_hand)` rounded to EOQ multiple), scenario_total_inventory_cost.
- **Clustered bar**: total inventory cost by scenario (all 4 scenarios).

### Page 4 — Contractor Insights

- **Scatter**: X = `total_spend` (log), Y = `order_frequency`, color = `segment_label`, hover = contractor_id + branch.
- **Pie / donut**: count of contractors per segment.
- **Matrix heatmap**: rows = `branches[name]`, cols = `products[category]`, values = % branch revenue from top-5 contractors (DAX in segment table).

## Step 6 — Style

- Theme: **View → Themes → Dark theme** (closest to the Dash app)
- Color palette: pick a 3-color theme matching the Dash app:
  - Roofing — orange `#F59E0B`
  - Siding — green `#10B981`
  - Exterior — blue `#3B82F6`
- Apply via View → Themes → Customize → Theme colors

## Step 7 — Save + commit

1. **File → Save As** → `~/projects/building-products-forecast/powerbi/building_products.pbix`
2. **Take screenshots** of each of the 4 pages → `docs/screenshots/powerbi_page_{1..4}.png`
3. Update `docs/RESUME_AUDIT.md` to flip Power BI from ⚠️ to ✅
4. `git add powerbi/building_products.pbix docs/screenshots/*.png` → commit → push

## Step 8 — Publish (optional)

If you want a shareable URL:
1. **Home → Publish** in Power BI Desktop
2. Choose your workspace in Power BI service (free tier OK)
3. Add the published URL to README.md screenshots section

This is the canonical interview demo path — "here's the URL, here's the
.pbix in the repo, here's the DAX measures next to the same logic in the
Plotly Dash code."
