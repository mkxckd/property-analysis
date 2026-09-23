"""
Evaluation for SouqPulse. Three things get measured honestly against
ground truth baked into the synthetic data, not eyeballed:

  1. Dedup: pairwise precision/recall/F1 (standard for entity
     resolution -- see dedup.py's docstring for why pairwise, not
     cluster-to-cluster).
  2. Fraud/suspicious pricing: TWO methods evaluated against the same
     ground truth, so the model-based approach's improvement over the
     naive rule is a measured comparison, not an assertion.
  3. Data quality: field completeness and naive exact-duplicate counts
     on the raw table.
"""
import itertools

import pandas as pd


def true_pairs_from_groups(df, group_col="dup_group_truth"):
    pairs = set()
    for _, group in df.groupby(group_col):
        ids = sorted(group["listing_id"].tolist())
        if len(ids) > 1:
            for a, b in itertools.combinations(ids, 2):
                pairs.add((a, b))
    return pairs


def predicted_pairs_from_clusters(clusters):
    pairs = set()
    for members in clusters.values():
        members = sorted(members)
        for a, b in itertools.combinations(members, 2):
            pairs.add((a, b))
    return pairs


def evaluate_dedup(df, dedup_result, runtime_seconds):
    truth = true_pairs_from_groups(df)
    predicted = predicted_pairs_from_clusters(dedup_result.clusters)

    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_duplicate_pairs": len(truth),
        "predicted_duplicate_pairs": len(predicted),
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "clusters_found": len(dedup_result.clusters),
        "runtime_seconds": round(runtime_seconds, 3),
        "listings_scored": len(df),
    }


def naive_suspicious_flags(df: pd.DataFrame) -> pd.Series:
    """The baseline being improved on: flag anything priced under 65% of
    its area+type median. No model, no per-unit context."""
    df = df.copy()
    df["area_type_key"] = list(zip(df["area"], df["property_type"]))
    medians = df.groupby("area_type_key")["price_aed_month"].median()
    df["_median"] = df["area_type_key"].map(medians)
    flagged = (df["price_aed_month"] < df["_median"] * 0.65).astype(int)
    return pd.Series(flagged.values, index=df["listing_id"].values)


def _prf(flagged: pd.Series, truth: pd.Series):
    tp = int(((flagged == 1) & (truth == 1)).sum())
    fp = int(((flagged == 1) & (truth == 0)).sum())
    fn = int(((flagged == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "flagged": int(flagged.sum()), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def evaluate_fraud_detection(df: pd.DataFrame, scored_df: pd.DataFrame, model_threshold: float = -0.30):
    truth = pd.Series(df["is_suspicious_truth"].values, index=df["listing_id"].values)

    naive_flags = naive_suspicious_flags(df)
    naive_metrics = _prf(naive_flags, truth)
    naive_metrics["method"] = "naive: price < 65% of area+type median"

    model_flags = (scored_df.set_index("listing_id")["residual_pct"] <= model_threshold).astype(int)
    model_flags = model_flags.reindex(truth.index).fillna(0).astype(int)
    model_metrics = _prf(model_flags, truth)
    model_metrics["method"] = f"model: priced >{abs(model_threshold)*100:.0f}% below predicted fair value"
    model_metrics["threshold"] = model_threshold

    return {
        "true_suspicious": int(truth.sum()),
        "naive_baseline": naive_metrics,
        "model_based": model_metrics,
        "precision_improvement": round(model_metrics["precision"] - naive_metrics["precision"], 4),
        "f1_improvement": round(model_metrics["f1"] - naive_metrics["f1"], 4),
    }


def evaluate_data_quality(df):
    n = len(df)
    required_cols = ["title", "area", "property_type", "price_aed_month", "size_sqft", "listed_date"]
    completeness = {c: round(1 - df[c].isna().mean(), 4) for c in required_cols}
    dupes_exact = int(df.duplicated(subset=["title", "area", "price_aed_month"]).sum())
    price_outliers = int((df.price_aed_month <= 0).sum())
    return {
        "total_listings": n,
        "field_completeness": completeness,
        "exact_duplicate_rows": dupes_exact,
        "invalid_price_rows": price_outliers,
    }
