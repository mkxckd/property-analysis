"""
Real-time scoring simulation.

An earlier version of this file maintained a running mean as a stand-in
for "real-time" -- that's real streaming-aggregation logic, but it never
touched the actual model, so the "real-time" claim and the "fair-value
model" claim were disconnected from each other. This version is what the
pipeline would actually do in production: take the model that's already
been trained (training happens once, offline, on the batch/clean data --
nobody retrains a GBM per incoming request), and use it to score each
new listing the instant it's posted.

What's measured is genuine per-row inference latency through the fitted
sklearn pipeline (one-hot encoding + gradient boosting), not a synthetic
stand-in -- this is the number that would actually matter if this sat
behind an API endpoint scoring listings as they're submitted.
"""
import time

import pandas as pd

from pipeline.valuation_model import add_derived_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def run_live_scoring_feed(pipe, df: pd.DataFrame, feed_length: int = 60):
    """Replays listings in posted-date order through the trained model
    one at a time, as if they were arriving live. Returns:
      - metrics: per-event inference latency stats
      - feed: an ordered sample of scored events, for the dashboard's
        live-feed animation
    """
    working = add_derived_features(df).sort_values("listed_date").reset_index(drop=True)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # warm-up calls so first-call overhead (sklearn's internal setup, not
    # representative of steady-state latency) doesn't skew the stats
    for _ in range(5):
        pipe.predict(working[feature_cols].iloc[[0]])

    latencies_ms = []
    feed = []

    for i, row in working.iterrows():
        single = working[feature_cols].iloc[[i]]
        t0 = time.perf_counter()
        pred = pipe.predict(single)[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        if i % max(1, len(working) // feed_length) == 0:
            residual_pct = (row["price_aed_month"] - pred) / pred
            feed.append({
                "listing_id": int(row["listing_id"]),
                "title": row["title"],
                "area": row["area"],
                "property_type": row["property_type"],
                "listed_date": row["listed_date"],
                "price_aed_month": int(row["price_aed_month"]),
                "predicted_price_aed_month": round(float(pred)),
                "residual_pct": round(float(residual_pct), 4),
                "flag": "underpriced" if residual_pct <= -0.30 else ("overpriced" if residual_pct >= 0.30 else "fair"),
            })

    latencies_ms.sort()
    n = len(latencies_ms)
    metrics = {
        "events_scored": n,
        "avg_latency_ms": round(sum(latencies_ms) / n, 4),
        "p50_latency_ms": round(latencies_ms[n // 2], 4),
        "p99_latency_ms": round(latencies_ms[int(n * 0.99)], 4),
        "max_latency_ms": round(latencies_ms[-1], 4),
    }
    return metrics, feed
