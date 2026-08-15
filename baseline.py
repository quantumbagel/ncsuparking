"""Hour-of-week median baseline for parking occupancy.

Fits from a resampled occupancy grid and looks up the typical occupancy
for a lot at a given Eastern weekday + time-of-day bucket.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from features import GRID_MINUTES, hour_of_week_key

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
BASELINE_PATH = MODEL_DIR / "baseline.json"


def fit_baseline(df: pd.DataFrame) -> dict:
    """Build {lot: {median, by_how}} from a grid with occupancy + recorded_at."""
    if df.empty:
        return {"grid_minutes": GRID_MINUTES, "timezone": Config.TIMEZONE, "lots": {}}

    work = df[["location_name", "recorded_at", "occupancy"]].copy()
    work["recorded_at"] = pd.to_datetime(work["recorded_at"], utc=True)
    work["how"] = work["recorded_at"].map(hour_of_week_key)

    lots: dict[str, dict] = {}
    for lot, group in work.groupby("location_name"):
        by_how = (
            group.groupby("how")["occupancy"]
            .median()
            .to_dict()
        )
        lots[str(lot)] = {
            "median": float(group["occupancy"].median()),
            "by_how": {str(int(k)): float(v) for k, v in by_how.items()},
        }
    logger.info("Fitted hour-of-week baseline for %d lots.", len(lots))
    return {
        "grid_minutes": GRID_MINUTES,
        "timezone": Config.TIMEZONE,
        "lots": lots,
    }


def save_baseline(table: dict, path: Path = BASELINE_PATH) -> Path:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(table, f)
    os.replace(tmp, path)
    return path


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    if not path.exists():
        return {"grid_minutes": GRID_MINUTES, "timezone": Config.TIMEZONE, "lots": {}}
    with open(path) as f:
        return json.load(f)


def _lookup_one(lot_table: dict, how: int) -> float:
    by_how = lot_table.get("by_how") or {}
    if str(how) in by_how:
        return float(by_how[str(how)])
    return float(lot_table.get("median", 50.0))


def predict_times(table: dict, lot: str, target_times: pd.Series) -> np.ndarray:
    """Typical occupancy for ``lot`` at each target timestamp."""
    lots = table.get("lots") or {}
    lot_table = lots.get(lot)
    times = pd.to_datetime(target_times, utc=True)
    if not lot_table:
        # Fall back to campus-wide median if this lot is unseen.
        campus = [info.get("median", 50.0) for info in lots.values()]
        fill = float(np.median(campus)) if campus else 50.0
        return np.full(len(times), fill, dtype=float)

    keys = times.map(hour_of_week_key)
    return np.array([_lookup_one(lot_table, int(k)) for k in keys], dtype=float)


def predict_now_plus(table: dict, lot: str, now: pd.Timestamp,
                     horizons: tuple[int, ...]) -> dict[int, float]:
    """Map horizon_minutes → typical occupancy at now+horizon."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    times = pd.Series([now + pd.Timedelta(minutes=h) for h in horizons])
    values = predict_times(table, lot, times)
    return {int(h): float(v) for h, v in zip(horizons, values)}
