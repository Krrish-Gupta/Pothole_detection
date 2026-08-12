import sqlite3
import json
from pathlib import Path
import pandas as pd
from config import DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detection_results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name    TEXT    NOT NULL,
                image_path    TEXT,
                defect_count  INTEGER DEFAULT 0,
                low_count     INTEGER DEFAULT 0,
                medium_count  INTEGER DEFAULT 0,
                high_count    INTEGER DEFAULT 0,
                severity      TEXT    CHECK(severity IN ('Low','Medium','High','None','Ignore')),
                confidence    REAL,
                bbox_data     TEXT,
                timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_severity  ON detection_results(severity);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_results(timestamp);
        """)
        # Migrate older databases that predate the low/medium/high columns
        cols = [row[1] for row in conn.execute("PRAGMA table_info(detection_results)")]
        for col in ["low_count", "medium_count", "high_count"]:
            if col not in cols:
                conn.execute(f"ALTER TABLE detection_results ADD COLUMN {col} INTEGER DEFAULT 0")


def insert_result(
    image_name: str, image_path: str, defect_count: int,
    low_count: int, medium_count: int, high_count: int,
    severity: str, confidence: float, bboxes: list
) -> int:
    """Insert one detection result, including the per-severity breakdown."""
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO detection_results
               (image_name, image_path, defect_count, low_count, medium_count,
                high_count, severity, confidence, bbox_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (image_name, image_path, defect_count, low_count, medium_count,
             high_count, severity, round(confidence, 4), json.dumps(bboxes))
        )
        return cur.lastrowid


def get_all_results() -> pd.DataFrame:
    """Return every row as a DataFrame, newest first."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM detection_results ORDER BY timestamp DESC",
            conn, parse_dates=["timestamp"]
        )
    return df


def get_severity_summary() -> dict:
    """Return {severity: count} of IMAGES by their dominant/worst severity."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT severity, COUNT(*) as n FROM detection_results GROUP BY severity"
        ).fetchall()
    return {row["severity"]: row["n"] for row in rows}


def get_defect_summary() -> dict:
    """Return total counts of individual DEFECTS (not images) by severity, across all rows."""
    with _connect() as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM(low_count),0)    as low,
                      COALESCE(SUM(medium_count),0) as medium,
                      COALESCE(SUM(high_count),0)   as high,
                      COUNT(*)                       as n_images
               FROM detection_results"""
        ).fetchone()
    total = row["low"] + row["medium"] + row["high"]
    return {
        "n_images": row["n_images"],
        "total": total,
        "low": row["low"],
        "medium": row["medium"],
        "high": row["high"],
    }


def get_daily_counts() -> pd.DataFrame:
    """Return detections aggregated by date for the trend line chart."""
    with _connect() as conn:
        df = pd.read_sql_query(
            """SELECT DATE(timestamp) as date,
                      COUNT(*) as total_images,
                      SUM(defect_count) as total_defects,
                      SUM(low_count) as low_count,
                      SUM(medium_count) as medium_count,
                      SUM(high_count) as high_count
               FROM detection_results
               GROUP BY DATE(timestamp)
               ORDER BY date""",
            conn
        )
    return df


def clear_results() -> None:
    """Delete all records. Used by the dashboard reset button."""
    with _connect() as conn:
        conn.execute("DELETE FROM detection_results")