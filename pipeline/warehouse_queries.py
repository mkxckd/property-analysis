"""
Runs the analytical queries in /sql against the warehouse and returns
their results for the dashboard. The .sql files are the source of truth:
the dashboard shows each query's text next to its result.
"""
import os
import sqlite3

from pipeline.etl import DB_PATH

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")

QUERIES = [
    ("area_leaderboard", "Area leaderboard",
     "Annual rent per sqft by area, with duplicates collapsed and fraud excluded in SQL."),
    ("monthly_rent_index", "Monthly rent index",
     "Rent per sqft of newly posted units by month, with month-over-month change."),
    ("agent_repost_rate", "Agent repost rate",
     "How often each agent re-posts a unit that is already listed."),
    ("model_error_by_segment", "Model error by segment",
     "Out-of-fold pricing gap per property type: where the model is weakest."),
]


def run_query(conn: sqlite3.Connection, name: str) -> dict:
    with open(os.path.join(SQL_DIR, f"{name}.sql"), encoding="utf-8") as fh:
        sql = fh.read()
    cur = conn.execute(sql)
    columns = [c[0] for c in cur.description]
    return {"sql": sql.strip(), "columns": columns, "rows": [list(r) for r in cur.fetchall()]}


def run_all(db_path: str = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    try:
        return [{"name": name, "title": title, "description": desc, **run_query(conn, name)}
                for name, title, desc in QUERIES]
    finally:
        conn.close()


def rows_as_dicts(result: dict) -> list:
    return [dict(zip(result["columns"], row)) for row in result["rows"]]
