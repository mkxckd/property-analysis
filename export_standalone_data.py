"""
Builds a trimmed JSON payload embedded directly into
frontend/standalone_demo.html, so that one file is fully self-contained
(no fetch() calls, works as a published preview) while the real project
(frontend/index.html) fetches the full, untrimmed data from ./data/.

Trimming keeps the demo file a reasonable size without changing any of
the underlying numbers -- every metric embedded here is copied verbatim
from the real pipeline output, just with fewer example rows shown.
"""
import json
import os

FRONTEND_DATA = os.path.join(os.path.dirname(__file__), "frontend", "data")

LISTING_FIELDS = [
    "listing_id", "title", "area", "property_type", "bedrooms", "size_sqft",
    "price_aed_month", "agent_name", "listed_date",
]


def trim_listing(row):
    return {k: row[k] for k in LISTING_FIELDS if k in row}


def main():
    with open(os.path.join(FRONTEND_DATA, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "area_summary.json"), encoding="utf-8") as f:
        area_summary = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "duplicate_clusters.json"), encoding="utf-8") as f:
        clusters = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "flagged_listings.json"), encoding="utf-8") as f:
        flagged = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "fair_value_grid.json"), encoding="utf-8") as f:
        grid = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "test_predictions.json"), encoding="utf-8") as f:
        test_predictions = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "live_feed.json"), encoding="utf-8") as f:
        live_feed = json.load(f)
    with open(os.path.join(FRONTEND_DATA, "warehouse_queries.json"), encoding="utf-8") as f:
        warehouse_queries = json.load(f)

    triples = [c for c in clusters if len(c["members"]) == 3]
    pairs = [c for c in clusters if len(c["members"]) == 2]
    sample_clusters = triples[:12] + pairs[:38]
    trimmed_clusters = [{
        "cluster_id": c["cluster_id"],
        "members": [trim_listing(m) for m in c["members"]],
        "pair_scores": c["pair_scores"],
    } for c in sample_clusters]

    payload = {
        "metrics": metrics,
        "area_summary": area_summary,
        "duplicate_clusters": trimmed_clusters,
        "duplicate_clusters_total": len(clusters),
        "flagged_listings": flagged,  # already small (<=~65 rows)
        "fair_value_grid": grid,      # needed in full for the estimator to work correctly
        "test_predictions": test_predictions,
        "live_feed": live_feed,
        "warehouse_queries": warehouse_queries,
    }

    out_path = os.path.join(os.path.dirname(__file__), "frontend", "embedded_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
