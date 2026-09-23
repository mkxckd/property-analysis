-- Monthly rent index: average annual rent per sqft for newly posted
-- units, with the month-over-month change computed by a window function.
WITH unique_units AS (
    SELECT f.*,
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE(f.dup_cluster_id, 'L' || f.listing_id)
               ORDER BY f.date_id, f.listing_id
           ) AS nth_posting
    FROM fact_listings f
    WHERE f.is_active = 1 AND f.is_flagged_fraud = 0
),
monthly AS (
    SELECT printf('%04d-%02d', d.year, d.month)        AS month,
           COUNT(*)                                    AS new_units,
           AVG(u.price_aed_month * 12.0 / u.size_sqft) AS psf
    FROM unique_units u
    JOIN dim_date d USING (date_id)
    WHERE u.nth_posting = 1
    GROUP BY d.year, d.month
)
SELECT month,
       new_units,
       ROUND(psf, 1) AS avg_price_per_sqft_annual,
       ROUND(100.0 * (psf - LAG(psf) OVER (ORDER BY month)) / LAG(psf) OVER (ORDER BY month), 1) AS mom_change_pct
FROM monthly
ORDER BY month;
