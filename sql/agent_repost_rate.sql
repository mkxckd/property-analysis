-- Which agents re-post units that are already on the market? A repost is
-- any listing that is not the first posting of its duplicate cluster.
WITH postings AS (
    SELECT f.agent_id,
           f.is_flagged_fraud,
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE(f.dup_cluster_id, 'L' || f.listing_id)
               ORDER BY f.date_id, f.listing_id
           ) AS nth_posting
    FROM fact_listings f
    WHERE f.is_active = 1
)
SELECT g.agent_name                                   AS agent,
       COUNT(*)                                       AS listings,
       SUM(p.nth_posting > 1)                         AS reposts,
       ROUND(100.0 * SUM(p.nth_posting > 1) / COUNT(*), 1) AS repost_rate_pct,
       SUM(p.is_flagged_fraud)                        AS flagged_fraud
FROM postings p
JOIN dim_agent g USING (agent_id)
GROUP BY g.agent_name
ORDER BY repost_rate_pct DESC, listings DESC;
