"""Load trained models and predict 24h-ahead occupancy for every lot."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2

from features import (
    FEATURE_COLS,
    FORECAST_MINUTES,
    LAG_MINUTES,
    ROLLING_WINDOWS,
    _add_event_features,
    _load_events_active,
    _load_lot_coords,
    _resample_minutely,
)
from database import save_predictions

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")

# Fetch snapshots from the last N hours.  Must exceed the longest lag
# (24h = 1440 min) plus margin so that minute-resampling yields enough rows.
HISTORY_HOURS = 48


def _load_latest_snapshots(conn_string: str) -> pd.DataFrame:
    """Return all snapshots from the last HISTORY_HOURS (for minute resampling)."""
    query = """
        SELECT recorded_at, location_name, occupancy
        FROM parking_snapshots
        WHERE recorded_at >= NOW() - make_interval(hours => %(hours)s)
        ORDER BY location_name, recorded_at;
    """
    with psycopg2.connect(conn_string) as conn:
        df = pd.read_sql_query(query, conn, params={"hours": HISTORY_HOURS})
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df


def _build_inference_features(resampled: pd.DataFrame, lot_name: str,
                              lot_coords: dict,
                              events: pd.DataFrame) -> pd.Series:
    """Build a single feature row from the *last resampled row* for a lot.

    The resampled DataFrame is already on a 1-minute grid.  The last row
    represents the most recent minute; we use its features to predict the
    next FORECAST_MINUTES minutes.
    """
    lot_df = resampled[resampled["location_name"] == lot_name].sort_values("recorded_at")
    max_lag = max(LAG_MINUTES)
    if len(lot_df) <= max_lag:
        raise ValueError(
            f"Not enough history for {lot_name} ({len(lot_df)} rows, need > {max_lag})"
        )

    latest = lot_df.iloc[-1]
    ts = latest["recorded_at"]

    # Time features from the latest minute grid point
    row = pd.DataFrame([{
        "recorded_at": ts,
        "location_name": lot_name,
        "hour": ts.hour,
        "minute_of_hour": ts.minute,
        "day_of_week": ts.dayofweek,
        "is_weekend": 1 if ts.dayofweek in (5, 6) else 0,
        "month": ts.month,
    }])

    # Lag features: shift relative to the minute grid
    occ_series = lot_df["occupancy"]
    for lag_minutes in LAG_MINUTES:
        row[f"lag_{lag_minutes}min"] = (
            occ_series.iloc[-1 - lag_minutes]
            if len(occ_series) > lag_minutes else occ_series.iloc[-1]
        )

    # Rolling means from the minute grid
    for window in ROLLING_WINDOWS:
        row[f"rolling_mean_{window}min"] = occ_series.tail(window).mean()

    # Event features (uses recorded_at for time overlap)
    row = _add_event_features(row, lot_coords, events)

    return row[FEATURE_COLS].iloc[0]


def predict_all(conn_string: str | None = None) -> pd.DataFrame:
    """Predict 24h-ahead occupancy for every lot with a trained model.

    Returns a DataFrame with columns:
        lot, predicted_at, horizon_hours, predicted_occupancy
    """
    if conn_string is None:
        from config import Config
        conn_string = Config.db_conn_string()

    summary_path = MODEL_DIR / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No {summary_path} found. Run train.py first."
        )
    with open(summary_path) as f:
        summary = json.load(f)

    logger.info("Loading latest snapshots…")
    raw = _load_latest_snapshots(conn_string)
    df = _resample_minutely(raw)
    lot_coords = _load_lot_coords(conn_string)

    # Load events from 24h in the past through +FORECAST_MINUTES + margin,
    # so past-proximity features (minutes_since_last_event) stay consistent
    # with training and future features cover the whole forecast horizon.
    now = datetime.now(timezone.utc)
    events = _load_events_active(
        conn_string,
        min_date=now - timedelta(hours=24),
        max_date=now + timedelta(minutes=FORECAST_MINUTES + 60),
    )

    predictions = []
    now_iso = now.isoformat()

    for lot_name, meta in summary["lots"].items():
        model_path = Path(meta["model_path"])
        if not model_path.exists():
            logger.warning("Model file missing: %s — skipping %s", model_path, lot_name)
            continue

        # Validate feature columns match training (exact set equality required)
        train_cols = meta.get("feature_columns", FEATURE_COLS)
        if set(train_cols) != set(FEATURE_COLS):
            raise ValueError(
                f"Feature column mismatch for {lot_name}: "
                f"trained with {sorted(train_cols)}, "
                f"current FEATURE_COLS has {sorted(FEATURE_COLS)}"
            )

        model = joblib.load(model_path)

        try:
            features = _build_inference_features(df, lot_name, lot_coords, events)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", lot_name, exc)
            continue

        # Reindex to training order if needed
        features = features.reindex(train_cols)

        X_input = features.values.reshape(1, -1)
        preds = model.predict(X_input)[0]  # shape (FORECAST_MINUTES,)

        # Clip to valid percentage range
        preds = np.clip(preds, 0, 100)

        for m, occ in enumerate(preds, start=1):
            predictions.append({
                "lot": lot_name,
                "predicted_at": now_iso,
                "horizon_minutes": m,
                "predicted_occupancy": round(float(occ), 1),
            })

    result = pd.DataFrame(predictions)
    logger.info("Predicted %d lots × %dmin = %d rows.",
                result["lot"].nunique(), FORECAST_MINUTES, len(result))
    return result


# ── Scheduled prediction ─────────────────────────────────────────────────────


def poll_and_predict(conn_string: str | None = None) -> int:
    """Scheduled prediction wrapper — saves predictions to the database.

    Gracefully skips if no trained models exist yet.

    Returns:
        Number of prediction rows saved, or 0 if skipped.
    """
    if conn_string is None:
        from config import Config
        conn_string = Config.db_conn_string()

    summary_path = MODEL_DIR / "summary.json"
    if not summary_path.exists():
        logger.info("No trained models yet (%s not found) — skipping prediction.",
                     summary_path)
        return 0

    try:
        df = predict_all(conn_string)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return 0

    rows = df.to_dict("records")
    saved = save_predictions(rows)
    logger.info("Saved %d predictions to database.", saved)
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    preds = predict_all()
    print(preds.to_string(index=False))
