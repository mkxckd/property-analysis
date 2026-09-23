"""
The fair-value model (AVM) -- the actual center of this project.

Rather than reporting "the area average" (a single number that ignores
what's actually different about one listing), this trains a regression
model to predict what a *specific* listing should rent for, given its
area, type, size, bedrooms, furnishing, building age, floor, and
distance to the city center. Two things fall out of having that
prediction for free:

  - a fair-value estimate for any listing, including ones that haven't
    been posted yet (the "try it" estimator in the dashboard)
  - a much sharper fraud/anomaly signal: how far below THIS unit's
    predicted price a listing is, rather than how far below the area's
    blended average -- which is what the naive rule in evaluate.py
    uses, and is a weaker signal because it can't tell a 500 sqft
    unfurnished 40th-floor studio from a 550 sqft furnished ground-floor
    one in the same area.

Why the training set is NOT the raw listings table: duplicate listings
would let one physical unit's price count multiple times, and the
injected fraud listings are, by construction, mispriced -- training on
them would teach the model that scam prices are normal. Both are
filtered out first, using dedup.py's clusters and the suspicious-price
flag. This is the one dependency that makes the pipeline a pipeline
rather than three unrelated scripts: the trust layer runs BEFORE the
model, because the model's honesty depends on it.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOWNTOWN_LAT, DOWNTOWN_LON = 25.1972, 55.2744

NUMERIC_FEATURES = ["bedrooms", "size_sqft", "building_age_years", "floor", "furnished", "distance_to_downtown_km"]
CATEGORICAL_FEATURES = ["area", "property_type"]
TARGET = "price_aed_month"


def _distance_km(lat1, lon1, lat2, lon2):
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dlat, dlon)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_to_downtown_km"] = _distance_km(df["lat"], df["lon"], DOWNTOWN_LAT, DOWNTOWN_LON).round(2)
    return df


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.06,
        subsample=0.85, random_state=42,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def get_clean_training_set(df: pd.DataFrame, dedup_result, suspicious_flags: pd.Series) -> pd.DataFrame:
    """One row per physical unit (first-seen listing of each duplicate
    cluster kept, the rest dropped), with fraud-flagged listings removed
    entirely. This is what the model is allowed to learn from."""
    df = df.copy()
    df["dup_cluster_id"] = df["listing_id"].map(dedup_result.listing_to_cluster)
    df["is_suspicious"] = df["listing_id"].map(suspicious_flags).fillna(0).astype(int)

    keep_mask = df["is_suspicious"] == 0
    clean = df[keep_mask].copy()

    # within each duplicate cluster, keep only the earliest-listed row
    clean = clean.sort_values("listed_date")
    clean["_dedup_key"] = clean["dup_cluster_id"].fillna(clean["listing_id"].astype(str))
    clean = clean.drop_duplicates(subset="_dedup_key", keep="first")

    return clean.drop(columns=["_dedup_key"])


def train_and_evaluate(clean_df: pd.DataFrame):
    df = add_derived_features(clean_df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds_test = pipe.predict(X_test)
    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae_aed": round(float(mean_absolute_error(y_test, preds_test)), 1),
        "mape": round(float(mean_absolute_percentage_error(y_test, preds_test)), 4),
        "r2": round(float(r2_score(y_test, preds_test)), 4),
    }

    # held-out actual-vs-predicted pairs, for an honest (out-of-sample)
    # scatter plot -- NOT the refit-on-everything model's in-sample fit,
    # which would look artificially tighter than the model really is
    test_predictions = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds_test.round(0),
        "area": df.loc[y_test.index, "area"].values,
    })

    # refit on ALL clean data for the model that actually scores every
    # listing / powers the estimator -- standard practice once the
    # held-out test metrics above are already banked as the honest
    # generalization estimate.
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    return final_pipe, metrics, test_predictions


def feature_importances(pipe) -> list:
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names

    # collapse one-hot columns back to their parent categorical feature
    # for a readable importance chart (nobody wants "area_Al Furjan" as
    # its own bar next to "size_sqft").
    raw_importances = dict(zip(all_names, model.feature_importances_))
    collapsed = {f: 0.0 for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    for name, imp in raw_importances.items():
        if name in collapsed:
            collapsed[name] += imp
        else:
            for cat in CATEGORICAL_FEATURES:
                if name.startswith(cat + "_"):
                    collapsed[cat] += imp
                    break

    total = sum(collapsed.values()) or 1.0
    result = [{"feature": k, "importance": round(v / total, 4)} for k, v in collapsed.items()]
    result.sort(key=lambda r: -r["importance"])
    return result


def _add_residuals(df: pd.DataFrame, predictions) -> pd.DataFrame:
    df["predicted_price_aed_month"] = np.asarray(predictions).round(0)
    df["residual_aed"] = df["price_aed_month"] - df["predicted_price_aed_month"]
    df["residual_pct"] = (df["residual_aed"] / df["predicted_price_aed_month"]).round(4)
    return df


def score_all_listings(pipe, df: pd.DataFrame) -> pd.DataFrame:
    """Scores every listing with an already-fitted model and returns
    predicted price + residual columns. Fine for listings the model has
    never seen; for evaluating the pipeline's own dataset use
    score_all_listings_out_of_fold instead."""
    df = add_derived_features(df)
    return _add_residuals(df, pipe.predict(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]))


def score_all_listings_out_of_fold(df: pd.DataFrame, clean_df: pd.DataFrame, dedup_result,
                                   n_splits: int = 5) -> pd.DataFrame:
    """Scores every listing with a model that never trained on it.

    Scoring the dataset with the final model (fit on every clean listing)
    would give the genuine listings in-sample predictions: their residuals
    come out artificially small, the fraud flag gets fewer false
    positives, and its precision looks better than it would on new data.
    Instead, listings are split into folds by physical unit (a duplicate
    cluster stays in one fold, so a re-post can't leak its twin's price),
    and each fold is scored by a model trained only on the clean listings
    of the other folds."""
    df = add_derived_features(df).reset_index(drop=True)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    groups = df["listing_id"].map(dedup_result.listing_to_cluster).fillna(df["listing_id"].astype(str))
    in_clean = df["listing_id"].isin(clean_df["listing_id"]).to_numpy()

    predictions = np.empty(len(df))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        train_idx = train_idx[in_clean[train_idx]]
        fold_pipe = build_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = fold_pipe.predict(X.iloc[test_idx])
    return _add_residuals(df, predictions)


if __name__ == "__main__":
    from pipeline.dedup import find_duplicates
    from pipeline.evaluate import naive_suspicious_flags

    raw_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "listings.csv")
    df = pd.read_csv(raw_path)

    print("Running dedup (needed to build a clean training set)...")
    dedup_result = find_duplicates(df)

    print("Running naive suspicious-pricing flag (needed for the same reason)...")
    suspicious_flags = naive_suspicious_flags(df)

    clean_df = get_clean_training_set(df, dedup_result, suspicious_flags)
    print(f"Raw listings: {len(df)} -> clean training set: {len(clean_df)} "
          f"(dropped {len(df) - len(clean_df)}: duplicates collapsed + fraud excluded)")

    pipe, metrics, _ = train_and_evaluate(clean_df)
    print("Held-out test performance:", metrics)

    print("\nFeature importances:")
    for row in feature_importances(pipe):
        print(f"  {row['feature']:<28} {row['importance']:.3f}")

    scored = score_all_listings(pipe, df)
    print("\nSample scored listings:")
    print(scored[["listing_id", "area", "price_aed_month", "predicted_price_aed_month", "residual_pct"]].head(8))
