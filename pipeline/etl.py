"""
ETL: loads raw listings plus everything the pipeline computed (dedup
clusters, valuation model predictions/residuals) into a small SQLite
warehouse laid out as a star schema:

    dim_area, dim_property_type, dim_agent, dim_date   (dimensions)
    fact_listings                                      (fact table, one
                                                          row per listing,
                                                          FKs into the
                                                          dimensions above,
                                                          carrying the
                                                          model's prediction
                                                          and residual)
    etl_runs                                           (audit log, one row
                                                          per load)

Loads are incremental and idempotent, ELT-style:

  1. the batch is written to a staging table as-is
  2. new dimension members are inserted from staging (INSERT OR IGNORE)
  3. facts are upserted from staging joined to the dimensions; a row
     hash means an unchanged listing is not rewritten, so re-running the
     same batch changes nothing
  4. listings missing from the batch are soft-deleted (is_active = 0),
     not dropped -- history stays queryable
  5. the run's inserted/updated/unchanged/deactivated counts go to etl_runs

This mirrors the semantic-cube-layer pattern from the Business Gateways
internship: don't make every dashboard query re-run the model or
re-aggregate the raw fact table -- precompute what the dashboard needs
once here, and let it read that directly. The analytical queries over
this schema live in /sql (see pipeline/warehouse_queries.py).
"""
import hashlib
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "souqpulse.db")

# Bump when SCHEMA changes; an older warehouse is rebuilt rather than
# half-migrated.
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_area (
    area_id INTEGER PRIMARY KEY AUTOINCREMENT,
    area_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_property_type (
    property_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_type TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_agent (
    agent_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_date TEXT UNIQUE NOT NULL,
    year INTEGER,
    month INTEGER,
    day INTEGER
);

CREATE TABLE IF NOT EXISTS fact_listings (
    listing_id INTEGER PRIMARY KEY,
    area_id INTEGER NOT NULL REFERENCES dim_area(area_id),
    property_type_id INTEGER NOT NULL REFERENCES dim_property_type(property_type_id),
    agent_id INTEGER NOT NULL REFERENCES dim_agent(agent_id),
    date_id INTEGER NOT NULL REFERENCES dim_date(date_id),
    title TEXT,
    bedrooms INTEGER,
    size_sqft INTEGER,
    furnished INTEGER,
    building_age_years INTEGER,
    floor INTEGER,
    price_aed_month INTEGER,
    predicted_price_aed_month REAL,
    residual_pct REAL,
    lat REAL,
    lon REAL,
    dup_cluster_id TEXT,
    is_duplicate INTEGER DEFAULT 0,
    is_flagged_fraud INTEGER DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    row_hash TEXT NOT NULL,
    first_loaded_at TEXT NOT NULL,
    last_updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_area ON fact_listings(area_id);
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_listings(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_dupcluster ON fact_listings(dup_cluster_id);
CREATE INDEX IF NOT EXISTS idx_fact_flag ON fact_listings(is_flagged_fraud);

CREATE TABLE IF NOT EXISTS etl_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    rows_in_batch INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    rows_updated INTEGER NOT NULL,
    rows_unchanged INTEGER NOT NULL,
    rows_deactivated INTEGER NOT NULL
);
"""

STAGING_COLUMNS = [
    "listing_id", "area", "property_type", "agent_name", "listed_date", "title",
    "bedrooms", "size_sqft", "furnished", "building_age_years", "floor",
    "price_aed_month", "predicted_price_aed_month", "residual_pct", "lat", "lon",
    "dup_cluster_id", "is_duplicate", "is_flagged_fraud",
]

UPSERT_DIMENSIONS = [
    "INSERT OR IGNORE INTO dim_area (area_name) SELECT DISTINCT area FROM stg_listings",
    "INSERT OR IGNORE INTO dim_property_type (property_type) SELECT DISTINCT property_type FROM stg_listings",
    "INSERT OR IGNORE INTO dim_agent (agent_name) SELECT DISTINCT agent_name FROM stg_listings",
    """INSERT OR IGNORE INTO dim_date (full_date, year, month, day)
       SELECT DISTINCT listed_date,
              CAST(substr(listed_date, 1, 4) AS INTEGER),
              CAST(substr(listed_date, 6, 2) AS INTEGER),
              CAST(substr(listed_date, 9, 2) AS INTEGER)
       FROM stg_listings""",
]

# `WHERE true` is required: without it SQLite parses ON CONFLICT as part
# of the SELECT's join clause.
UPSERT_FACTS = """
INSERT INTO fact_listings (
    listing_id, area_id, property_type_id, agent_id, date_id, title,
    bedrooms, size_sqft, furnished, building_age_years, floor,
    price_aed_month, predicted_price_aed_month, residual_pct, lat, lon,
    dup_cluster_id, is_duplicate, is_flagged_fraud, is_active,
    row_hash, first_loaded_at, last_updated_at
)
SELECT s.listing_id, a.area_id, p.property_type_id, g.agent_id, d.date_id, s.title,
       s.bedrooms, s.size_sqft, s.furnished, s.building_age_years, s.floor,
       s.price_aed_month, s.predicted_price_aed_month, s.residual_pct, s.lat, s.lon,
       s.dup_cluster_id, s.is_duplicate, s.is_flagged_fraud, 1,
       s.row_hash, :now, :now
FROM stg_listings s
JOIN dim_area a          ON a.area_name = s.area
JOIN dim_property_type p ON p.property_type = s.property_type
JOIN dim_agent g         ON g.agent_name = s.agent_name
JOIN dim_date d          ON d.full_date = s.listed_date
WHERE true
ON CONFLICT (listing_id) DO UPDATE SET
    area_id = excluded.area_id,
    property_type_id = excluded.property_type_id,
    agent_id = excluded.agent_id,
    date_id = excluded.date_id,
    title = excluded.title,
    bedrooms = excluded.bedrooms,
    size_sqft = excluded.size_sqft,
    furnished = excluded.furnished,
    building_age_years = excluded.building_age_years,
    floor = excluded.floor,
    price_aed_month = excluded.price_aed_month,
    predicted_price_aed_month = excluded.predicted_price_aed_month,
    residual_pct = excluded.residual_pct,
    lat = excluded.lat,
    lon = excluded.lon,
    dup_cluster_id = excluded.dup_cluster_id,
    is_duplicate = excluded.is_duplicate,
    is_flagged_fraud = excluded.is_flagged_fraud,
    is_active = 1,
    row_hash = excluded.row_hash,
    last_updated_at = excluded.last_updated_at
WHERE fact_listings.row_hash != excluded.row_hash OR fact_listings.is_active = 0;
"""


def _row_hash(row) -> str:
    return hashlib.sha1("|".join(str(v) for v in row).encode("utf-8")).hexdigest()


def build_staging_frame(scored_df: pd.DataFrame, dedup_result, fraud_threshold: float) -> pd.DataFrame:
    stg = scored_df.copy()
    stg["dup_cluster_id"] = stg["listing_id"].map(dedup_result.listing_to_cluster)
    stg["is_duplicate"] = stg["dup_cluster_id"].notna().astype(int)
    stg["is_flagged_fraud"] = (stg["residual_pct"] <= fraud_threshold).astype(int)
    stg = stg[STAGING_COLUMNS]
    stg["row_hash"] = [_row_hash(r) for r in stg.itertuples(index=False)]
    return stg


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        existing = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        for table in existing:
            conn.execute(f"DROP TABLE {table}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.executescript(SCHEMA)
    return conn


def load_warehouse(scored_df: pd.DataFrame, dedup_result, fraud_threshold: float = -0.30,
                   db_path: str = DB_PATH):
    """Incrementally loads one batch. Returns (table row counts, this run's stats)."""
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stg = build_staging_frame(scored_df, dedup_result, fraud_threshold)

    conn = _connect(db_path)
    try:
        with conn:  # one transaction: a failed load leaves the warehouse untouched
            # Staged row by row rather than with DataFrame.to_sql, which
            # would commit mid-transaction.
            cols = list(stg.columns)
            conn.execute(f"CREATE TEMP TABLE stg_listings ({', '.join(cols)})")
            conn.executemany(
                f"INSERT INTO stg_listings VALUES ({', '.join('?' * len(cols))})",
                [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row)
                 for row in stg.itertuples(index=False)],
            )

            inserted, updated, unchanged, deactivated = conn.execute("""
                SELECT
                    SUM(f.listing_id IS NULL),
                    SUM(f.listing_id IS NOT NULL AND (f.row_hash != s.row_hash OR f.is_active = 0)),
                    SUM(f.listing_id IS NOT NULL AND f.row_hash = s.row_hash AND f.is_active = 1),
                    (SELECT COUNT(*) FROM fact_listings
                     WHERE is_active = 1 AND listing_id NOT IN (SELECT listing_id FROM stg_listings))
                FROM stg_listings s
                LEFT JOIN fact_listings f ON f.listing_id = s.listing_id
            """).fetchone()

            for statement in UPSERT_DIMENSIONS:
                conn.execute(statement)
            conn.execute(UPSERT_FACTS, {"now": started_at})
            conn.execute("""
                UPDATE fact_listings SET is_active = 0, last_updated_at = :now
                WHERE is_active = 1 AND listing_id NOT IN (SELECT listing_id FROM stg_listings)
            """, {"now": started_at})
            conn.execute("DROP TABLE stg_listings")

            run = {
                "rows_in_batch": len(stg),
                "rows_inserted": inserted or 0,
                "rows_updated": updated or 0,
                "rows_unchanged": unchanged or 0,
                "rows_deactivated": deactivated or 0,
            }
            conn.execute("""
                INSERT INTO etl_runs (started_at, finished_at, rows_in_batch, rows_inserted,
                                      rows_updated, rows_unchanged, rows_deactivated)
                VALUES (:started_at, :finished_at, :rows_in_batch, :rows_inserted,
                        :rows_updated, :rows_unchanged, :rows_deactivated)
            """, {**run, "started_at": started_at,
                  "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ["fact_listings", "dim_area", "dim_property_type", "dim_agent", "dim_date", "etl_runs"]}
    finally:
        conn.close()
    return counts, run
