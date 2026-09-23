"""
Runs the full SouqPulse pipeline end to end. Order matters here -- this
isn't three independent scripts, it's a dependency chain:

  1. generate synthetic listings (skip if already present)
  2. dedup -- must run before training, so one physical unit doesn't
     get counted (and trained on) multiple times
  3. naive suspicious-price flag -- a cheap first-pass filter that
     bootstraps a clean-enough training set (its own precision is
     mediocre; that's fine here, it only has to be a reasonable filter,
     not the final fraud signal)
  4. build the clean training set from (2) + (3)
  5. train the fair-value model on the clean set, evaluate on a held-out
     test split
  6. score every listing (including the ones excluded from training)
     with the trained model
  7. evaluate the model-residual fraud signal against ground truth, and
     against the naive rule from (3) -- this is the "did the ML
     actually help" comparison
  8. run the real-time scoring simulation (per-event latency + a live
     feed sample)
  9. load the SQLite warehouse
  10. export JSON for the dashboard

Usage: python run_pipeline.py [--regenerate]
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate as generate_data, AREAS, PROPERTY_TYPES
from pipeline.dedup import find_duplicates
from pipeline.etl import load_warehouse
from pipeline.evaluate import (
    evaluate_data_quality,
    evaluate_dedup,
    evaluate_fraud_detection,
    naive_suspicious_flags,
)
from pipeline.streaming_sim import run_live_scoring_feed
from pipeline.warehouse_queries import rows_as_dicts, run_all as run_warehouse_queries
from pipeline.valuation_model import (
    add_derived_features,
    feature_importances,
    get_clean_training_set,
    score_all_listings_out_of_fold,
    train_and_evaluate,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(ROOT, "data", "raw", "listings.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
FRONTEND_DATA_DIR = os.path.join(ROOT, "frontend", "data")
MODEL_PATH = os.path.join(ROOT, "models", "fair_value.joblib")

FRAUD_THRESHOLD = -0.30


def build_fair_value_grid(pipe):
    """Precomputes model predictions across a grid of (area, property
    type, size, furnished) combinations, so the dashboard's interactive
    estimator can look up a prediction instantly client-side instead of
    needing a live Python process behind it. Every number in the grid
    comes from the same trained model used everywhere else -- this is
    the model's real output, just precomputed for a fixed set of inputs."""
    rows = []
    for area, lat, lon, _ in AREAS:
        for ptype, (beds_min, beds_max, size_range) in PROPERTY_TYPES.items():
            bedrooms = beds_min if ptype == "Studio" else round((beds_min + beds_max) / 2)
            sizes = np.linspace(size_range[0], size_range[1], 6).round().astype(int)
            is_tower = ptype in ("Studio", "1BR Apartment", "2BR Apartment", "3BR Apartment")
            for size in sizes:
                for furnished in (0, 1):
                    rows.append({
                        "area": area, "lat": lat, "lon": lon, "property_type": ptype,
                        "bedrooms": bedrooms, "size_sqft": int(size), "furnished": furnished,
                        "building_age_years": 8, "floor": 15 if is_tower else 0,
                    })
    grid_df = pd.DataFrame(rows)
    grid_df = add_derived_features(grid_df)
    feature_cols = ["bedrooms", "size_sqft", "building_age_years", "floor", "furnished",
                     "distance_to_downtown_km", "area", "property_type"]
    grid_df["predicted_price_aed_month"] = pipe.predict(grid_df[feature_cols]).round(0)
    return grid_df[["area", "property_type", "size_sqft", "furnished", "predicted_price_aed_month"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)

    if args.regenerate or not os.path.exists(RAW_PATH):
        print("[1/10] Generating synthetic listings...")
        generate_data()
    else:
        print("[1/10] Using existing data/raw/listings.csv (pass --regenerate to rebuild)")

    df = pd.read_csv(RAW_PATH)
    print(f"       loaded {len(df)} listings")

    print("[2/10] Running duplicate detection...")
    import time
    t0 = time.time()
    dedup_result = find_duplicates(df)
    dedup_runtime = time.time() - t0
    dedup_metrics = evaluate_dedup(df, dedup_result, dedup_runtime)
    print(f"       precision={dedup_metrics['precision']} recall={dedup_metrics['recall']} f1={dedup_metrics['f1']}")

    print("[3/10] Building clean training set (dedup + naive fraud filter)...")
    naive_flags = naive_suspicious_flags(df)
    clean_df = get_clean_training_set(df, dedup_result, naive_flags)
    print(f"       raw={len(df)} -> clean={len(clean_df)}")

    print("[4/10] Training fair-value model...")
    pipe, model_metrics, test_predictions = train_and_evaluate(clean_df)
    print(f"       held-out test: MAE={model_metrics['mae_aed']} AED, MAPE={model_metrics['mape']*100:.1f}%, R2={model_metrics['r2']}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"       saved fitted model to {os.path.relpath(MODEL_PATH, ROOT)} (served by api/main.py)")

    print("[5/10] Scoring all listings (out-of-fold, grouped by physical unit)...")
    scored_df = score_all_listings_out_of_fold(df, clean_df, dedup_result)

    print("[6/10] Evaluating fraud detection: naive rule vs model residual...")
    fraud_metrics = evaluate_fraud_detection(df, scored_df, FRAUD_THRESHOLD)
    print(f"       naive:  precision={fraud_metrics['naive_baseline']['precision']} recall={fraud_metrics['naive_baseline']['recall']} f1={fraud_metrics['naive_baseline']['f1']}")
    print(f"       model:  precision={fraud_metrics['model_based']['precision']} recall={fraud_metrics['model_based']['recall']} f1={fraud_metrics['model_based']['f1']}")

    print("[7/10] Data quality checks...")
    dq_metrics = evaluate_data_quality(df)

    print("[8/10] Real-time scoring simulation...")
    stream_metrics, live_feed = run_live_scoring_feed(pipe, df)
    print(f"       avg latency={stream_metrics['avg_latency_ms']}ms  p99={stream_metrics['p99_latency_ms']}ms")

    print("[9/10] Loading SQLite warehouse (incremental upsert)...")
    warehouse_counts, etl_run = load_warehouse(scored_df, dedup_result, FRAUD_THRESHOLD)
    print(f"       inserted={etl_run['rows_inserted']} updated={etl_run['rows_updated']} "
          f"unchanged={etl_run['rows_unchanged']} deactivated={etl_run['rows_deactivated']}")
    warehouse_results = run_warehouse_queries()

    print("[10/10] Exporting frontend JSON...")

    feat_imp = feature_importances(pipe)

    # area leaderboard comes from the warehouse (sql/area_leaderboard.sql),
    # which does its own dedup-collapse and fraud exclusion in SQL
    area_summary = rows_as_dicts(next(q for q in warehouse_results if q["name"] == "area_leaderboard"))

    fair_value_grid = build_fair_value_grid(pipe)

    all_results = {
        "dedup": dedup_metrics,
        "model": model_metrics,
        "fraud_detection": fraud_metrics,
        "data_quality": dq_metrics,
        "streaming": stream_metrics,
        "warehouse": warehouse_counts,
        "etl_last_run": etl_run,
        "feature_importances": feat_imp,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(os.path.join(RESULTS_DIR, "evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(FRONTEND_DATA_DIR, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(FRONTEND_DATA_DIR, "area_summary.json"), "w") as f:
        json.dump(area_summary, f, indent=2)
    with open(os.path.join(FRONTEND_DATA_DIR, "warehouse_queries.json"), "w") as f:
        json.dump(warehouse_results, f, indent=2)
    fair_value_grid.to_json(os.path.join(FRONTEND_DATA_DIR, "fair_value_grid.json"), orient="records", indent=2)
    test_predictions.to_json(os.path.join(FRONTEND_DATA_DIR, "test_predictions.json"), orient="records", indent=2)

    with open(os.path.join(FRONTEND_DATA_DIR, "live_feed.json"), "w") as f:
        json.dump(live_feed, f, indent=2, default=str)

    # duplicate cluster review payload
    clusters_payload = []
    for cid, members in dedup_result.clusters.items():
        rows = df[df["listing_id"].isin(members)].to_dict(orient="records")
        pair_scores = [{"a": a, "b": b, "score": s} for (a, b, s) in dedup_result.pairs if a in members and b in members]
        clusters_payload.append({"cluster_id": cid, "members": rows, "pair_scores": pair_scores})
    clusters_payload.sort(key=lambda c: len(c["members"]), reverse=True)
    with open(os.path.join(FRONTEND_DATA_DIR, "duplicate_clusters.json"), "w") as f:
        json.dump(clusters_payload, f, indent=2, default=str)

    # flagged listings payload, sorted by how far below predicted value
    flagged = scored_df[scored_df["residual_pct"] <= FRAUD_THRESHOLD].sort_values("residual_pct")
    with open(os.path.join(FRONTEND_DATA_DIR, "flagged_listings.json"), "w") as f:
        json.dump(flagged.to_dict(orient="records"), f, indent=2, default=str)

    print(f"\nDone. Warehouse: db/souqpulse.db | Frontend data: {FRONTEND_DATA_DIR}")
    return all_results, pipe


if __name__ == "__main__":
    main()
