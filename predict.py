"""Predict occupancy at the 8 eval horizons using baseline and/or XGBoost."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2

from baseline import load_baseline, predict_now_plus
from database import save_predictions
from features import (
    FEATURE_COLS,
    FORECAST_HORIZONS,
    FORECAST_MINUTES,
    add_base_features,
    build_inference_rows,
    event_word_cols,
    _load_events_active,
    _load_lot_coords,
    _load_parking,
    _resample_grid,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
EVENT_VOCAB_PATH = MODEL_DIR / "event_vocab.json"
HISTORY_HOURS = 8 * 24 + 6  # real week-lag when we have it; else median-filled


def _load_event_vocab() -> list[str]:
    if not EVENT_VOCAB_PATH.exists():
        return []
    with open(EVENT_VOCAB_PATH) as f:
        return json.load(f)


def _load_summary() -> dict:
    path = MODEL_DIR / "summary.json"
    if not path.exists():
        return {"trained_at": None, "lots": {}}
    with open(path) as f:
        return json.load(f)


def _known_lots(conn_string: str) -> list[str]:
    query = "SELECT DISTINCT location_name FROM parking_snapshots ORDER BY location_name;"
    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]


def predict_all(conn_string: str | None = None) -> pd.DataFrame:
    """Predict every known lot at each horizon.

    Always emits a baseline. Uses XGBoost only when that lot's active_model is
    ``xgb`` and the feature columns still match.
    """
    if conn_string is None:
        from config import Config
        conn_string = Config.db_conn_string()

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    baseline_table = load_baseline()
    summary = _load_summary()
    vocab = _load_event_vocab()
    feature_cols = FEATURE_COLS + event_word_cols(vocab)

    raw = _load_parking(conn_string, min_date=pd.Timestamp(now - timedelta(hours=HISTORY_HOURS), tz="UTC"))
    lots = _known_lots(conn_string)
    if raw.empty or not lots:
        logger.warning("No parking snapshots — nothing to predict.")
        return pd.DataFrame()

    resampled = _resample_grid(raw)
    lot_coords = _load_lot_coords(conn_string)
    events = _load_events_active(
        conn_string,
        min_date=now - timedelta(hours=24),
        max_date=now + timedelta(minutes=FORECAST_MINUTES + 60),
    )

    try:
        featured = add_base_features(resampled, lot_coords, events, vocab)
    except Exception as exc:
        logger.error("Feature build failed (%s) — baseline only.", exc)
        featured = None

    predictions: list[dict] = []
    for lot_name in lots:
        typical = predict_now_plus(baseline_table, lot_name, now, FORECAST_HORIZONS)
        model_name = "baseline"
        preds: dict[int, float] = dict(typical)

        meta = (summary.get("lots") or {}).get(lot_name) or {}
        model_path = meta.get("model_path")
        active = meta.get("active_model", "baseline")
        if (
            featured is not None
            and active == "xgb"
            and model_path
            and Path(model_path).exists()
        ):
            train_cols = meta.get("feature_columns", feature_cols)
            lot_feat = featured[featured["location_name"] == lot_name]
            try:
                if set(train_cols) != set(feature_cols):
                    raise ValueError(
                        f"feature mismatch trained={sorted(train_cols)} "
                        f"current={sorted(feature_cols)}"
                    )
                rows = build_inference_rows(lot_feat)
                X = rows.reindex(columns=train_cols)
                if X.isna().any().any():
                    raise ValueError("NaN features at inference")
                model = joblib.load(model_path)
                yhat = np.clip(model.predict(X), 0, 100)
                for horizon, value in zip(rows["horizon_minutes"], yhat):
                    preds[int(horizon)] = float(value)
                model_name = "xgb"
            except Exception as exc:
                logger.warning("Skipping XGB for %s (%s) — using baseline.", lot_name, exc)

        for horizon in FORECAST_HORIZONS:
            predictions.append({
                "lot": lot_name,
                "predicted_at": now_iso,
                "horizon_minutes": int(horizon),
                "predicted_occupancy": round(preds[horizon], 1),
                "baseline_occupancy": round(typical[horizon], 1),
                "model_name": model_name,
            })

    result = pd.DataFrame(predictions)
    logger.info("Predicted %d lots × %d horizons = %d rows (%s).",
                result["lot"].nunique() if not result.empty else 0,
                len(FORECAST_HORIZONS), len(result),
                "xgb+baseline" if summary.get("lots") else "baseline")
    return result


def poll_and_predict(conn_string: str | None = None) -> int:
    """Scheduled prediction wrapper — always tries to write a forecast."""
    if conn_string is None:
        from config import Config
        conn_string = Config.db_conn_string()

    try:
        df = predict_all(conn_string)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return 0

    if df.empty:
        return 0
    saved = save_predictions(df.to_dict("records"))
    logger.info("Saved %d predictions to database.", saved)
    return saved


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    preds = predict_all()
    print(preds.to_string(index=False) if not preds.empty else "(no predictions)")
