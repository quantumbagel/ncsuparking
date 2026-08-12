"""Prediction-vs-actual accuracy tracking.

Joins stored predictions to the observed occupancy (the most recent snapshot
at or before each forecast time — matching the forward-fill semantics used in
feature engineering) and computes per-lot MAE at representative horizons.
"""

import logging
from datetime import datetime, timezone

import pandas as pd
import psycopg2

from config import Config
from database import save_accuracy_snapshots

logger = logging.getLogger(__name__)

# Representative horizons to evaluate (minutes)
EVAL_HORIZONS = [15, 30, 60, 120, 180, 360, 720, 1440]


def compute_accuracy(conn_string: str | None = None,
                     hours_back: int = 24) -> pd.DataFrame:
    """Return per-lot MAE at representative horizons over the last `hours_back`.

    Returns a DataFrame with columns: lot, horizon_minutes, mae, n.
    """
    if conn_string is None:
        conn_string = Config.db_conn_string()

    query = """
        SELECT p.lot,
               p.horizon_minutes,
               p.predicted_occupancy,
               s.occupancy AS actual_occupancy
        FROM predictions p
        CROSS JOIN LATERAL (
            SELECT occupancy
            FROM parking_snapshots
            WHERE location_name = p.lot
              AND recorded_at <= p.predicted_at
                                  + make_interval(mins => p.horizon_minutes)
            ORDER BY recorded_at DESC
            LIMIT 1
        ) s
        WHERE p.horizon_minutes = ANY(%(horizons)s)
          AND p.predicted_at >= NOW() - make_interval(hours => %(hours_back)s)
          AND p.predicted_at + make_interval(mins => p.horizon_minutes) <= NOW();
    """
    with psycopg2.connect(conn_string) as conn:
        df = pd.read_sql_query(
            query, conn,
            params={"horizons": EVAL_HORIZONS, "hours_back": hours_back},
        )

    if df.empty:
        logger.info("No matured predictions to evaluate yet.")
        return pd.DataFrame(columns=["lot", "horizon_minutes", "mae", "n"])

    df["error"] = (df["predicted_occupancy"] - df["actual_occupancy"]).abs()

    mae = (
        df.groupby(["lot", "horizon_minutes"])["error"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mae", "count": "n"})
        .reset_index()
    )
    return mae


def compute_overall_mae(conn_string: str | None = None,
                        hours_back: int = 24) -> float | None:
    """Return the sample-weighted mean absolute error across all lots/horizons."""
    mae = compute_accuracy(conn_string, hours_back)
    if mae.empty:
        return None
    return float((mae["mae"] * mae["n"]).sum() / mae["n"].sum())


def record_accuracy(conn_string: str | None = None, hours_back: int = 24) -> int:
    """Snapshot current per-horizon MAE into `accuracy_snapshots`.

    Returns the number of rows recorded (0 if nothing matured to evaluate).
    """
    mae = compute_accuracy(conn_string, hours_back)
    if mae.empty:
        logger.info("Accuracy snapshot skipped — no matured predictions.")
        return 0

    ts = datetime.now(timezone.utc)
    rows = [
        {
            "recorded_at": ts,
            "horizon_minutes": int(r.horizon_minutes),
            "mae": float(r.mae),
            "n": int(r.n),
        }
        for r in mae.itertuples(index=False)
    ]
    saved = save_accuracy_snapshots(rows)
    logger.info("Recorded %d accuracy snapshots.", saved)
    return saved


def query_accuracy_history(conn_string: str | None = None) -> pd.DataFrame:
    """Return overall (avg) MAE over time, one row per recorded snapshot batch."""
    if conn_string is None:
        conn_string = Config.db_conn_string()
    query = """
        SELECT recorded_at, AVG(mae) AS overall_mae
        FROM accuracy_snapshots
        GROUP BY recorded_at
        ORDER BY recorded_at;
    """
    with psycopg2.connect(conn_string) as conn:
        df = pd.read_sql_query(query, conn)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df
