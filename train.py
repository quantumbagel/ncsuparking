"""Train per-lot XGBoost models that beat an hour-of-week baseline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from baseline import BASELINE_PATH, fit_baseline, predict_times, save_baseline
from config import Config
from features import (
    FEATURE_COLS,
    FORECAST_HORIZONS,
    build_training_data,
    event_word_cols,
    _load_parking,
    _resample_grid,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
EVENT_VOCAB_PATH = MODEL_DIR / "event_vocab.json"
TRAIN_STATE_PATH = MODEL_DIR / "train_state.json"
TRAIN_REQUEST_PATH = MODEL_DIR / "train.request"
TRAIN_LOCK_PATH = MODEL_DIR / "train.lock"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _write_state(**kwargs) -> None:
    current = get_train_state()
    current.update(kwargs)
    _atomic_json(TRAIN_STATE_PATH, current)


def get_train_state() -> dict:
    """Return collector-owned training state from the shared models volume."""
    if not TRAIN_STATE_PATH.exists():
        return {
            "running": False,
            "done": False,
            "error": None,
            "started_at": None,
            "step": None,
            "lot": None,
        }
    try:
        with open(TRAIN_STATE_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {
            "running": False,
            "done": False,
            "error": None,
            "started_at": None,
            "step": None,
            "lot": None,
        }


def start_training_async() -> bool:
    """Ask the collector to train. Dashboard must not run XGBoost itself."""
    if get_train_state().get("running"):
        return False
    MODEL_DIR.mkdir(exist_ok=True)
    TRAIN_REQUEST_PATH.write_text(datetime.now(timezone.utc).isoformat())
    _write_state(
        running=False,
        done=False,
        error=None,
        started_at=None,
        step="queued",
        lot=None,
    )
    logger.info("Wrote training request → %s", TRAIN_REQUEST_PATH)
    return True


def _acquire_lock() -> int | None:
    """Exclusive file lock. Returns fd, or None if another train is running."""
    MODEL_DIR.mkdir(exist_ok=True)
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _save_event_vocab(vocab: list[str]) -> None:
    tmp = EVENT_VOCAB_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(vocab, f)
    os.replace(tmp, EVENT_VOCAB_PATH)


def train_lot_model(X_lot: pd.DataFrame, y_lot: pd.Series,
                    lot_name: str, baseline_table: dict,
                    recorded_at: pd.Series) -> tuple[xgb.XGBRegressor, dict]:
    """Train a single-output XGBoost model and compare it to the baseline."""
    order = np.argsort(recorded_at.to_numpy())
    X_lot = X_lot.iloc[order]
    y_lot = y_lot.iloc[order]
    recorded_at = recorded_at.iloc[order]

    cut = recorded_at.quantile(0.8)
    train_mask = recorded_at <= cut
    # Keep at least a few val rows even on tiny lots.
    if train_mask.sum() < 10 or (~train_mask).sum() < 5:
        split = max(1, int(len(X_lot) * 0.8))
        train_mask = pd.Series([True] * split + [False] * (len(X_lot) - split),
                               index=X_lot.index)

    X_train, X_val = X_lot.loc[train_mask], X_lot.loc[~train_mask]
    y_train, y_val = y_lot.loc[train_mask], y_lot.loc[~train_mask]
    t_val = recorded_at.loc[X_val.index]
    target_times = t_val + pd.to_timedelta(X_val["horizon_minutes"], unit="m")
    base_val = predict_times(baseline_table, lot_name, target_times)
    baseline_mae = float(np.mean(np.abs(base_val - y_val.to_numpy())))

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=2,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = np.clip(model.predict(X_val), 0, 100)
    val_mae = float(np.mean(np.abs(preds - y_val.to_numpy())))

    mae_by_horizon: dict[str, float] = {}
    for horizon in FORECAST_HORIZONS:
        mask = X_val["horizon_minutes"] == horizon
        if not mask.any():
            continue
        mae_by_horizon[f"target_{horizon}min"] = float(
            np.mean(np.abs(preds[mask.to_numpy()] - y_val.loc[mask].to_numpy()))
        )

    active = "xgb" if val_mae < baseline_mae else "baseline"
    metrics = {
        "lot": lot_name,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "val_mae_mean": val_mae,
        "baseline_mae": baseline_mae,
        "active_model": active,
        "val_mae_by_horizon": mae_by_horizon,
    }
    logger.info(
        "  %-35s  xgb=%.1f%%  baseline=%.1f%%  → %s",
        lot_name, val_mae, baseline_mae, active,
    )
    return model, metrics


def _history_start() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)


def _fit_baseline_from_db(conn_string: str) -> dict:
    raw = _load_parking(conn_string, min_date=_history_start())
    if raw.empty:
        return fit_baseline(raw)
    grid = _resample_grid(raw)
    return fit_baseline(grid)


def train_all(conn_string: str | None = None) -> dict:
    """Fit baseline + one XGBoost model per lot. Always writes a baseline."""
    if conn_string is None:
        conn_string = Config.db_conn_string()

    lock_fd = _acquire_lock()
    if lock_fd is None:
        logger.warning("Training already running — skipping.")
        return {"lots": {}}

    started = datetime.now(timezone.utc).isoformat()
    _write_state(
        running=True, done=False, error=None,
        started_at=started, step="loading", lot=None,
    )
    try:
        return _train_all_inner(conn_string)
    except Exception as exc:
        logger.exception("Training failed")
        _write_state(running=False, done=True, error=str(exc), step="failed")
        raise
    finally:
        _release_lock(lock_fd)


def _train_all_inner(conn_string: str) -> dict:
    logger.info("=" * 60)
    logger.info("Fitting hour-of-week baseline…")
    _write_state(step="baseline")
    baseline_table = _fit_baseline_from_db(conn_string)
    save_baseline(baseline_table)
    logger.info("Saved baseline → %s", BASELINE_PATH)

    logger.info("Building training dataset…")
    _write_state(step="features")
    try:
        X, y, vocab = build_training_data(conn_string, min_date=_history_start())
    except ValueError as exc:
        logger.warning("Not enough data for ML (%s) — baseline only.", exc)
        summary = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "lots": {},
            "baseline_only": True,
        }
        _atomic_json(MODEL_DIR / "summary.json", summary)
        _save_event_vocab([])
        _write_state(running=False, done=True, error=None, step="baseline_only")
        return summary

    word_cols = event_word_cols(vocab)
    feature_names = FEATURE_COLS + word_cols
    MODEL_DIR.mkdir(exist_ok=True)
    _save_event_vocab(vocab)

    lots = X["location_name"].unique()
    logger.info("Training models for %d lots…", len(lots))

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "lots": {},
        "baseline_only": False,
    }

    for lot_name in sorted(lots):
        _write_state(step="training", lot=lot_name)
        mask = X["location_name"] == lot_name
        X_lot = X.loc[mask, feature_names]
        y_lot = y.loc[mask]
        recorded_at = X.loc[mask, "recorded_at"]

        if len(X_lot) < Config.MIN_TRAIN_ROWS:
            logger.warning(
                "  %s: only %d rows (< %d) — baseline only.",
                lot_name, len(X_lot), Config.MIN_TRAIN_ROWS,
            )
            summary["lots"][lot_name] = {
                "train_rows": int(len(X_lot)),
                "val_rows": 0,
                "val_mae_mean": None,
                "baseline_mae": None,
                "active_model": "baseline",
                "val_mae_by_horizon": {},
                "model_path": None,
                "feature_columns": feature_names,
            }
            continue

        model, metrics = train_lot_model(
            X_lot, y_lot, lot_name, baseline_table, recorded_at,
        )

        fname = f"{lot_name.lower().replace(' ', '_').replace('-', '_')}.pkl"
        path = MODEL_DIR / fname
        if metrics["active_model"] == "xgb":
            tmp = path.with_suffix(".pkl.tmp")
            joblib.dump(model, tmp)
            os.replace(tmp, path)
            metrics["model_path"] = str(path)
        else:
            metrics["model_path"] = None
            if path.exists():
                path.unlink()

        metrics["feature_columns"] = feature_names
        summary["lots"][lot_name] = metrics

    _atomic_json(MODEL_DIR / "summary.json", summary)
    logger.info("Saved summary for %d lots → %s", len(summary["lots"]), MODEL_DIR)
    _write_state(running=False, done=True, error=None, step="done", lot=None)
    return summary


def consume_train_request(conn_string: str | None = None) -> dict | None:
    """Run train_all if the dashboard (or a human) dropped a request file."""
    if not TRAIN_REQUEST_PATH.exists():
        return None
    try:
        TRAIN_REQUEST_PATH.unlink(missing_ok=True)
        return train_all(conn_string)
    except Exception as exc:
        logger.error("Requested retrain failed: %s", exc)
        return {"lots": {}}


def retrain_if_ready(conn_string: str | None = None) -> dict:
    """Hourly check: always keep the baseline fresh; retrain ML when due."""
    try:
        requested = consume_train_request(conn_string)
        if requested is not None:
            return requested

        summary_path = MODEL_DIR / "summary.json"
        baseline_missing = not BASELINE_PATH.exists()
        summary_missing = not summary_path.exists()
        stale = False
        if summary_path.exists():
            age = datetime.now(timezone.utc).timestamp() - summary_path.stat().st_mtime
            stale = age >= Config.RETRAIN_INTERVAL
        if baseline_missing or summary_missing or stale:
            return train_all(conn_string)
        logger.info("Models are fresh — skipping scheduled retrain.")
        return {"lots": {}}
    except Exception as exc:
        logger.error("Auto-retrain failed: %s", exc)
        return {"lots": {}}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train_all()
