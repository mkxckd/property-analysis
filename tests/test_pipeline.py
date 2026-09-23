"""Unit tests for the dedup, fraud-flag, and valuation stages, on small
hand-built frames so they run in well under a second and don't depend on
data/raw/listings.csv."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dedup import find_duplicates  # noqa: E402
from pipeline.evaluate import _prf, naive_suspicious_flags, predicted_pairs_from_clusters, true_pairs_from_groups  # noqa: E402
from pipeline.valuation_model import add_derived_features, get_clean_training_set  # noqa: E402


def listing(listing_id, **overrides):
    row = {
        "listing_id": listing_id,
        "title": "Spacious 1BR Apartment in Dubai Marina with Sea View",
        "description": "Bright unit, balcony, gym and pool access, close to tram.",
        "property_type": "1BR Apartment", "area": "Dubai Marina",
        "lat": 25.0805, "lon": 55.1403, "bedrooms": 1, "size_sqft": 800,
        "furnished": 0, "building_age_years": 8, "floor": 12,
        "price_aed_month": 9000, "agent_name": "Layla Haddad",
        "listed_date": "2026-05-01", "status": "active",
        "is_suspicious_truth": 0, "dup_group_truth": listing_id,
    }
    row.update(overrides)
    return row


# ---------------- dedup ----------------

def test_repost_with_small_drift_is_clustered():
    df = pd.DataFrame([
        listing(1),
        listing(2, price_aed_month=9200, size_sqft=805, lat=25.0806, agent_name="Omar Farouk",
                title="1BR Apartment in Dubai Marina, Sea View, Spacious", listed_date="2026-05-09"),
    ])
    result = find_duplicates(df)
    assert predicted_pairs_from_clusters(result.clusters) == {(1, 2)}
    assert result.listing_to_cluster[1] == result.listing_to_cluster[2]


def test_distant_listing_is_not_a_duplicate():
    # identical text/price/size, but ~5 km away -> fails the geo gate
    df = pd.DataFrame([listing(1), listing(2, lat=25.1255)])
    assert find_duplicates(df).clusters == {}


def test_different_block_is_never_compared():
    df = pd.DataFrame([listing(1), listing(2, bedrooms=2)])
    assert find_duplicates(df).clusters == {}


def test_chained_pairs_collapse_into_one_cluster():
    df = pd.DataFrame([
        listing(1),
        listing(2, price_aed_month=9100, size_sqft=802),
        listing(3, price_aed_month=9200, size_sqft=805),
    ])
    clusters = find_duplicates(df).clusters
    assert len(clusters) == 1
    assert sorted(next(iter(clusters.values()))) == [1, 2, 3]


def test_true_pairs_from_groups():
    df = pd.DataFrame([listing(1, dup_group_truth=1), listing(2, dup_group_truth=1), listing(3)])
    assert true_pairs_from_groups(df) == {(1, 2)}


# ---------------- fraud flags ----------------

def test_naive_flag_uses_area_type_median():
    prices = [10000, 10000, 10000, 10000, 5000]  # median 10000; 5000 < 65% of it
    df = pd.DataFrame([listing(i, price_aed_month=p) for i, p in enumerate(prices, start=1)])
    flags = naive_suspicious_flags(df)
    assert flags.to_dict() == {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}


def test_prf_counts():
    flagged = pd.Series([1, 1, 0, 0])
    truth = pd.Series([1, 0, 1, 0])
    m = _prf(flagged, truth)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_prf_handles_no_flags():
    m = _prf(pd.Series([0, 0]), pd.Series([1, 0]))
    assert m["precision"] == 0.0 and m["f1"] == 0.0


# ---------------- valuation model inputs ----------------

def test_distance_to_downtown_is_zero_at_downtown():
    df = add_derived_features(pd.DataFrame([{"lat": 25.1972, "lon": 55.2744}]))
    assert df["distance_to_downtown_km"].iloc[0] == 0


def test_distance_to_downtown_marina_is_plausible():
    df = add_derived_features(pd.DataFrame([{"lat": 25.0805, "lon": 55.1403}]))
    assert 15 < df["distance_to_downtown_km"].iloc[0] < 22


class _FakeDedup:
    def __init__(self, mapping):
        self.listing_to_cluster = mapping


def test_clean_training_set_drops_fraud_and_keeps_earliest_duplicate():
    df = pd.DataFrame([
        listing(1, listed_date="2026-05-09"),
        listing(2, listed_date="2026-05-01"),  # earlier repost of the same unit -> kept
        listing(3),
        listing(4, price_aed_month=3000),      # flagged -> dropped
    ])
    dedup = _FakeDedup({1: "c1", 2: "c1"})
    flags = pd.Series({1: 0, 2: 0, 3: 0, 4: 1})
    clean = get_clean_training_set(df, dedup, flags)
    assert sorted(clean["listing_id"]) == [2, 3]


@pytest.mark.parametrize("flag_value", [0, 1])
def test_clean_training_set_does_not_mutate_input(flag_value):
    df = pd.DataFrame([listing(1)])
    before = df.copy()
    get_clean_training_set(df, _FakeDedup({}), pd.Series({1: flag_value}))
    pd.testing.assert_frame_equal(df, before)
