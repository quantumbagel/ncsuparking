"""Prediction-vs-actual accuracy tracking."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import psycopg2

from config import Config
from database import save_accuracy_snapshots

logger = logging.getLogger(__name__)

EVAL_HORIZONS = list(Config.FORECAST_HORIZONS)


def compute_accuracy(conn_string: str | None = None,
                     hours_back: int = 24) -> pd.DataFrame:
    """Return per-lot MAE at eval horizons over the last ``hours_back``.

    Columns: lot, horizon_minutes, mae, baseline_mae, n, model_name.
    """
    if conn_string is None:
        conn_string = Config.db_conn_string()

    query = """
        SELECT p.lot,
               p.horizon_minutes,
               p.predicted_occupancy,
               p.baseline_occupancy,
               COALESCE(p.model_name, 'unknown') AS model_name,
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

    empty_cols = ["lot", "horizon_minutes", "mae", "baseline_mae", "n", "model_name"]
    if df.empty:
        logger.info("No matured predictions to evaluate yet.")
        return pd.DataFrame(columns=empty_cols)

    df["error"] = (df["predicted_occupancy"] - df["actual_occupancy"]).abs()
    df["baseline_error"] = (df["baseline_occupancy"] - df["actual_occupancy"]).abs()

    mae = (
        df.groupby(["lot", "horizon_minutes", "model_name"], dropna=False)
        .agg(
            mae=("error", "mean"),
            baseline_mae=("baseline_error", "mean"),
            n=("error", "count"),
        )
        .reset_index()
    )
    return mae


def compute_overall_mae(conn_string: str | None = None,
                        hours_back: int = 24) -> float | None:
    """Sample-weighted mean absolute error across all lots/horizons."""
    mae = compute_accuracy(conn_string, hours_back)
    if mae.empty:
        return None
    return float((mae["mae"] * mae["n"]).sum() / mae["n"].sum())


def record_accuracy(conn_string: str | None = None, hours_back: int = 24) -> int:
    """Snapshot current per-horizon MAE into ``accuracy_snapshots``."""
    mae = compute_accuracy(conn_string, hours_back)
    if mae.empty:
        logger.info("Accuracy snapshot skipped — no matured predictions.")
        return 0

    ts = datetime.now(timezone.utc)
    rows = []
    # Weighted overall per horizon (not an unweighted mean of lot MAEs).
    for horizon, group in mae.groupby("horizon_minutes"):
        n = int(group["n"].sum())
        if n == 0:
            continue
        rows.append({
            "recorded_at": ts,
            "horizon_minutes": int(horizon),
            "mae": float((group["mae"] * group["n"]).sum() / n),
            "n": n,
            "model_name": "active",
        })
        if group["baseline_mae"].notna().any():
            rows.append({
                "recorded_at": ts,
                "horizon_minutes": int(horizon),
                "mae": float((group["baseline_mae"] * group["n"]).sum() / n),
                "n": n,
                "model_name": "baseline",
            })
    saved = save_accuracy_snapshots(rows)
    logger.info("Recorded %d accuracy snapshots.", saved)
    return saved


def query_accuracy_history(conn_string: str | None = None) -> pd.DataFrame:
    """Weighted overall MAE over time, one row per (batch, model_name)."""
    if conn_string is None:
        conn_string = Config.db_conn_string()
    query = """
        SELECT recorded_at,
               COALESCE(model_name, 'active') AS model_name,
               CASE WHEN SUM(n) > 0
                    THEN SUM(mae * n) / SUM(n)
                    ELSE AVG(mae) END AS overall_mae
        FROM accuracy_snapshots
        GROUP BY recorded_at, COALESCE(model_name, 'active')
        ORDER BY recorded_at;
    """
    with psycopg2.connect(conn_string) as conn:
        df = pd.read_sql_query(query, conn)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df
