-- sql/databricks/04_contractor_concentration_by_branch.sql
-- Purpose: Estimate branch contractor concentration using segmented contractor spend and top-5 branch share.
-- Expected runtime: < 10 seconds on the project Delta tables.

WITH branch_sales AS (
    SELECT
        branch_id,
        SUM(revenue) AS branch_sales_revenue
    FROM building_products.sales_history
    GROUP BY branch_id
),
contractor_spend AS (
    SELECT
        c.branch_id,
        c.contractor_id,
        cs.total_spend
    FROM building_products.contractors AS c
    INNER JOIN building_products.contractor_segments AS cs
        ON c.contractor_id = cs.contractor_id
       AND c.branch_id = cs.branch_id
),
ranked_contractors AS (
    SELECT
        branch_id,
        contractor_id,
        total_spend,
        ROW_NUMBER() OVER (
            PARTITION BY branch_id
            ORDER BY total_spend DESC, contractor_id
        ) AS spend_rank
    FROM contractor_spend
),
branch_concentration AS (
    SELECT
        branch_id,
        SUM(CASE WHEN spend_rank <= 5 THEN total_spend ELSE 0 END) AS top_5_spend,
        SUM(total_spend) AS total_contractor_spend,
        COUNT(DISTINCT contractor_id) AS contractor_count
    FROM ranked_contractors
    GROUP BY branch_id
)
SELECT
    bc.branch_id,
    b.name AS branch_name,
    ROUND(
        100.0 * bc.top_5_spend / NULLIF(bc.total_contractor_spend, 0),
        2
    ) AS top_5_contractor_share_pct,
    bc.contractor_count
FROM branch_concentration AS bc
INNER JOIN building_products.branches AS b
    ON bc.branch_id = b.branch_id
INNER JOIN branch_sales AS bs
    ON bc.branch_id = bs.branch_id
WHERE bs.branch_sales_revenue > 0
ORDER BY top_5_contractor_share_pct DESC, bc.branch_id;
