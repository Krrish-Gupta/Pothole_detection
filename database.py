import sqlite3
import json
from datetime import datetime
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
                severity      TEXT    CHECK(severity IN ('Low','Medium','High','None','Ignore')),
                confidence    REAL,
                bbox_data     TEXT,
                timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_severity
                ON detection_results(severity);

            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON detection_results(timestamp);
        """)


def insert_result(
    image_name:   str,
    image_path:   str,
    defect_count: int,
    severity:     str,
    confidence:   float,
    bboxes:       list
) -> int:
    """Insert one detection result. Returns the new row ID."""
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO detection_results
               (image_name, image_path, defect_count, severity, confidence, bbox_data)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (image_name, image_path, defect_count,
             severity, round(confidence, 4), json.dumps(bboxes))
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
    """Return {severity: count} for the dashboard pie chart."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT severity, COUNT(*) as n FROM detection_results GROUP BY severity"
        ).fetchall()
    return {row["severity"]: row["n"] for row in rows}


def get_daily_counts() -> pd.DataFrame:
    """Return detections aggregated by date for the trend line chart."""
    with _connect() as conn:
        df = pd.read_sql_query(
            """SELECT DATE(timestamp) as date,
                      COUNT(*) as total_images,
                      SUM(defect_count) as total_defects
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