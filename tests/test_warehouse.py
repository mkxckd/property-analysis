"""Incremental-load and SQL tests against a throwaway SQLite warehouse."""
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.etl import load_warehouse  # noqa: E402
from pipeline.warehouse_queries import QUERIES, rows_as_dicts, run_all  # noqa: E402


class _FakeDedup:
    def __init__(self, mapping):
        self.listing_to_cluster = mapping


def scored(listing_id, **overrides):
    row = {
        "listing_id": listing_id, "area": "Dubai Marina", "property_type": "1BR Apartment",
        "agent_name": "Layla Haddad", "listed_date": "2026-05-01", "title": f"Listing {listing_id}",
        "bedrooms": 1, "size_sqft": 800, "furnished": 0, "building_age_years": 8, "floor": 12,
        "price_aed_month": 9000, "predicted_price_aed_month": 9000.0, "residual_pct": 0.0,
        "lat": 25.08, "lon": 55.14,
    }
    row.update(overrides)
    return row


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "wh.db")


def batch(*rows):
    return pd.DataFrame(list(rows))


def test_first_load_inserts_everything(db):
    counts, run = load_warehouse(batch(scored(1), scored(2)), _FakeDedup({}), db_path=db)
    assert run["rows_inserted"] == 2 and run["rows_updated"] == 0
    assert counts["fact_listings"] == 2 and counts["etl_runs"] == 1


def test_reloading_same_batch_is_a_no_op(db):
    b = batch(scored(1), scored(2))
    load_warehouse(b, _FakeDedup({}), db_path=db)
    counts, run = load_warehouse(b, _FakeDedup({}), db_path=db)
    assert (run["rows_inserted"], run["rows_updated"], run["rows_unchanged"]) == (0, 0, 2)
    assert counts["fact_listings"] == 2 and counts["etl_runs"] == 2


def test_changed_row_is_updated_and_new_dimension_added(db):
    load_warehouse(batch(scored(1), scored(2)), _FakeDedup({}), db_path=db)
    _, run = load_warehouse(batch(scored(1, price_aed_month=8500), scored(2), scored(3, area="Mirdif")),
                            _FakeDedup({}), db_path=db)
    assert (run["rows_inserted"], run["rows_updated"], run["rows_unchanged"]) == (1, 1, 1)

    conn = sqlite3.connect(db)
    price, first, last = conn.execute(
        "SELECT price_aed_month, first_loaded_at, last_updated_at FROM fact_listings WHERE listing_id = 1").fetchone()
    areas = [r[0] for r in conn.execute("SELECT area_name FROM dim_area ORDER BY area_name")]
    conn.close()
    assert price == 8500
    assert first <= last
    assert areas == ["Dubai Marina", "Mirdif"]


def test_missing_listing_is_soft_deleted_then_reactivated(db):
    load_warehouse(batch(scored(1), scored(2)), _FakeDedup({}), db_path=db)
    _, run = load_warehouse(batch(scored(1)), _FakeDedup({}), db_path=db)
    assert run["rows_deactivated"] == 1

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT is_active FROM fact_listings WHERE listing_id = 2").fetchone()[0] == 0
    conn.close()

    _, run = load_warehouse(batch(scored(1), scored(2)), _FakeDedup({}), db_path=db)
    assert run["rows_updated"] == 1  # reactivated


def test_fraud_and_duplicate_flags_are_derived(db):
    load_warehouse(batch(scored(1), scored(2, residual_pct=-0.45)), _FakeDedup({1: "cluster_0"}), db_path=db)
    conn = sqlite3.connect(db)
    rows = dict(((r[0]), r[1:]) for r in conn.execute(
        "SELECT listing_id, is_duplicate, is_flagged_fraud, dup_cluster_id FROM fact_listings"))
    conn.close()
    assert rows[1] == (1, 0, "cluster_0")
    assert rows[2] == (0, 1, None)


def test_leaderboard_counts_each_cluster_once_and_excludes_fraud(db):
    load_warehouse(batch(
        scored(1, price_aed_month=10000, listed_date="2026-05-01"),
        scored(2, price_aed_month=20000, listed_date="2026-05-09"),  # later repost of 1 -> not counted
        scored(3, price_aed_month=10000),
        scored(4, price_aed_month=1000, residual_pct=-0.8),          # fraud -> excluded from the average
    ), _FakeDedup({1: "cluster_0", 2: "cluster_0"}), db_path=db)
    board = rows_as_dicts(run_all(db)[0])
    assert board == [{"area": "Dubai Marina", "avg_price_per_sqft_annual": 150.0, "clean_units": 2, "flagged_fraud": 1}]


def test_every_query_runs(db):
    load_warehouse(batch(scored(1), scored(2, listed_date="2026-06-02")), _FakeDedup({}), db_path=db)
    results = run_all(db)
    assert [r["name"] for r in results] == [q[0] for q in QUERIES]
    assert all(r["rows"] for r in results)
