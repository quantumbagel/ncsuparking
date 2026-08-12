"""Train per-lot XGBoost models to predict 24h-ahead parking occupancy."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from features import (
    FEATURE_COLS,
    TARGET_COLS,
    FORECAST_MINUTES,
    build_training_data,
    event_word_cols,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
EVENT_VOCAB_PATH = MODEL_DIR / "event_vocab.json"


def _save_event_vocab(vocab: list[str]) -> None:
    """Atomically persist the event word vocabulary used for features."""
    tmp = EVENT_VOCAB_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(vocab, f)
    os.replace(tmp, EVENT_VOCAB_PATH)

# ── Async training state (shared with the dashboard) ─────────────────────────
#
# Kept module-level (not in the Streamlit script) so it survives Streamlit's
# re-execution of the script on every rerun.  `train.py` is imported once.

_TRAIN_LOCK = threading.Lock()
_TRAIN_STATE = {"running": False, "done": False, "error": None, "started_at": None}


def get_train_state() -> dict:
    """Return a snapshot of the async-training state."""
    with _TRAIN_LOCK:
        return dict(_TRAIN_STATE)


def start_training_async() -> bool:
    """Start `train_all` in a background thread.  Returns False if already running."""
    with _TRAIN_LOCK:
        if _TRAIN_STATE["running"]:
            return False
        _TRAIN_STATE.update(
            running=True, done=False, error=None,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
    threading.Thread(target=_training_worker, daemon=True).start()
    return True


def _training_worker() -> None:
    """Runs train_all and records the outcome in _TRAIN_STATE."""
    try:
        train_all()
    except Exception as exc:  # surface to the dashboard
        with _TRAIN_LOCK:
            _TRAIN_STATE["error"] = str(exc)
    finally:
        with _TRAIN_LOCK:
            _TRAIN_STATE["running"] = False
            _TRAIN_STATE["done"] = True

# ── Training ──────────────────────────────────────────────────────────────────


def train_lot_model(X_lot: pd.DataFrame, y_lot: pd.DataFrame,
                    lot_name: str) -> tuple[xgb.XGBRegressor, dict]:
    """Train a multi-output XGBoost model for a single parking lot.

    Args:
        X_lot: Feature DataFrame (rows × FEATURE_COLS) for this lot.
        y_lot: Target DataFrame (rows × FORECAST_MINUTES) for this lot.

    Returns:
        (trained_model, metrics_dict).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_lot, y_lot, test_size=0.2, shuffle=False,
    )

    # NOTE: 1440 outputs with multi_output_tree is memory/time heavy.
    # Shallower trees + fewer estimators keep training tractable.
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        multi_strategy="multi_output_tree",
        num_target=FORECAST_MINUTES,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Per-horizon MAE on validation set
    preds = model.predict(X_val)
    mae_by_horizon = {}
    for i, col in enumerate(TARGET_COLS):
        mae_by_horizon[col] = float(
            np.mean(np.abs(preds[:, i] - y_val[col].values))
        )

    metrics = {
        "lot": lot_name,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "val_mae_mean": float(np.mean(list(mae_by_horizon.values()))),
        "val_mae_by_horizon": mae_by_horizon,
    }

    def _m(mins: int) -> float:
        return mae_by_horizon.get(f"target_{mins}min", float("nan"))

    logger.info("  %-35s  MAE avg=%.1f%%  (1min=%.1f, 15min=%.1f, 1h=%.1f, 6h=%.1f, 24h=%.1f)",
                lot_name, metrics["val_mae_mean"],
                _m(1), _m(15), _m(60), _m(360), _m(1440))
    return model, metrics


def train_all(conn_string: str | None = None) -> dict:
    """Train one XGBoost model per parking lot and persist to disk.

    Returns a summary dict with per-lot metrics and artifact paths.
    """
    if conn_string is None:
        from config import Config
        conn_string = Config.db_conn_string()

    logger.info("=" * 60)
    logger.info("Building training dataset…")
    X, y, vocab = build_training_data(conn_string)
    word_cols = event_word_cols(vocab)
    feature_names = FEATURE_COLS + word_cols  # consistent across all lots

    lots = X["location_name"].unique()
    logger.info("Training models for %d lots…", len(lots))

    MODEL_DIR.mkdir(exist_ok=True)
    # Persist the vocabulary BEFORE training so predictions can rebuild the
    # exact same feature columns (written atomically).
    _save_event_vocab(vocab)
    logger.info("Event word features: %d (%s).",
                len(word_cols), ", ".join(vocab) or "none")

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "lots": {},
    }

    for lot_name in sorted(lots):
        mask = X["location_name"] == lot_name
        X_lot = X.loc[mask, feature_names]
        y_lot = y.loc[mask]

        if len(X_lot) < FORECAST_MINUTES * 2:
            logger.warning("  %s: only %d rows (< 2×%d min) — skipping.",
                           lot_name, len(X_lot), FORECAST_MINUTES)
            continue

        model, metrics = train_lot_model(X_lot, y_lot, lot_name)

        # Persist atomically (write temp + os.replace) so concurrent
        # readers in the collector's predict loop never see a half-written file.
        fname = f"{lot_name.lower().replace(' ', '_').replace('-','_')}.pkl"
        path = MODEL_DIR / fname
        tmp = path.with_suffix(".pkl.tmp")
        joblib.dump(model, tmp)
        os.replace(tmp, path)

        summary["lots"][lot_name] = {
            **metrics,
            "model_path": str(path),
            "feature_columns": feature_names,
        }

    # Don't overwrite a good summary with an empty one (e.g. insufficient data)
    if not summary["lots"]:
        logger.warning("No lots had enough data to train — keeping existing models.")
        return summary

    # Write summary metadata atomically
    summary_path = MODEL_DIR / "summary.json"
    tmp_summary = summary_path.with_suffix(".json.tmp")
    with open(tmp_summary, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_summary, summary_path)

    logger.info("Saved %d models + summary to %s/", len(summary["lots"]), MODEL_DIR)
    logger.info("Summary → %s", summary_path)
    return summary


def retrain_if_ready(conn_string: str | None = None) -> dict:
    """Scheduled retraining wrapper — catches errors so the scheduler keeps going.

    Returns the summary dict, or {"lots": {}} on failure/skip.
    """
    try:
        return train_all(conn_string)
    except Exception as exc:  # keep the scheduler alive
        logger.error("Auto-retrain failed: %s", exc)
        return {"lots": {}}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train_all()
