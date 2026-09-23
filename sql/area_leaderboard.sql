-- Area leaderboard: average annual rent per sqft on the cleaned market.
-- "Cleaned" happens here in SQL, not upstream: each duplicate cluster
-- counts once (its earliest listing), and model-flagged fraud is excluded.
WITH ranked AS (
    SELECT f.*,
           d.full_date,
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE(f.dup_cluster_id, 'L' || f.listing_id)
               ORDER BY d.full_date, f.listing_id
           ) AS nth_posting
    FROM fact_listings f
    JOIN dim_date d USING (date_id)
    WHERE f.is_active = 1
)
SELECT a.area_name AS area,
       ROUND(AVG(CASE WHEN r.is_flagged_fraud = 0
                      THEN r.price_aed_month * 12.0 / r.size_sqft END), 1) AS avg_price_per_sqft_annual,
       SUM(r.is_flagged_fraud = 0)                                      AS clean_units,
       SUM(r.is_flagged_fraud)                                          AS flagged_fraud
FROM ranked r
JOIN dim_area a USING (area_id)
WHERE r.nth_posting = 1
GROUP BY a.area_name
ORDER BY avg_price_per_sqft_annual DESC;
