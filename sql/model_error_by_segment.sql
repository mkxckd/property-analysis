-- Where is the fair-value model weakest? Out-of-fold gap between listed
-- and predicted price (as a % of predicted), per property
-- type, on listings not flagged as fraud. Positive bias means
-- listings in that segment tend to be priced above the model's estimate.
SELECT p.property_type,
       COUNT(*)                                            AS listings,
       ROUND(AVG(ABS(f.residual_pct)) * 100, 1)            AS avg_abs_gap_pct,
       ROUND(AVG(f.residual_pct) * 100, 1)                 AS bias_pct,
       ROUND(AVG(ABS(f.price_aed_month - f.predicted_price_aed_month))) AS mae_aed
FROM fact_listings f
JOIN dim_property_type p USING (property_type_id)
WHERE f.is_active = 1 AND f.is_flagged_fraud = 0
GROUP BY p.property_type
ORDER BY avg_abs_gap_pct DESC;
