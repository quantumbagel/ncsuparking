"""Feature engineering for parking occupancy forecasts.

Resamples each lot to a 5-minute grid, builds Eastern-time / lag / event /
calendar features, then expands each timestamp across the 8 forecast horizons
so the model is a single-target regressor with horizon as a feature.
"""

from __future__ import annotations

import logging
import math
import re
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import psycopg2

from academic_calendar import calendar_flags
from config import Config

logger = logging.getLogger(__name__)

GRID_MINUTES = Config.GRID_MINUTES
FORECAST_HORIZONS = Config.FORECAST_HORIZONS
FORECAST_MINUTES = max(FORECAST_HORIZONS)
LAG_MINUTES = (5, 15, 60, 180)
WEEK_LAG_MINUTES = 7 * 24 * 60
ROLLING_WINDOWS = (15, 60)

EVENT_ACTIVE_RADII = ((300, "events_active_300m"), (500, "events_active_500m"))
EVENT_START_WINDOWS = (
    (1, "events_starting_1h"),
    (3, "events_starting_3h"),
    (6, "events_starting_6h"),
    (24, "events_starting_24h"),
)
EVENT_CAP_MINUTES = 1440.0
ONE_MINUTE_NS = np.timedelta64(1, "m")
EVENT_MAJOR_MIN_NS = np.timedelta64(3, "h")
EVENT_MAJOR_MAX_NS = np.timedelta64(48, "h")

EVENT_VOCAB_SIZE = 10
EVENT_VOCAB_MIN_DOCS = 3
EVENT_STOPWORDS = frozenset({
    "the", "and", "for", "with", "vs", "at", "of", "in", "on", "to", "a",
    "nc", "state", "university", "ncsu", "event", "free", "tickets", "ticket",
    "series", "day", "days", "annual", "presented", "presents", "pm", "am",
    "time", "times", "join", "us", "come", "see", "info", "information",
    "click", "register", "registration", "more", "details", "learn", "like",
})

_FOOTBALL_RE = re.compile(r"\bfootball\b", re.I)
_EVENT_TOKEN_RE = re.compile(r"[a-z0-9]+")

BUCKETS_PER_DAY = 1440 // GRID_MINUTES


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in metres between two (lat, lng) points."""
    r = 6_371_000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(a))


def hour_of_week_key(ts: pd.Timestamp, grid: int = GRID_MINUTES) -> int:
    """Stable key: dow * buckets_per_day + minutes_since_midnight // grid."""
    local = ts.tz_convert(Config.TIMEZONE) if ts.tzinfo else ts.tz_localize(Config.TIMEZONE)
    minutes = local.hour * 60 + local.minute
    return int(local.dayofweek) * (1440 // grid) + minutes // grid


def _as_text(value) -> str:
    return value if isinstance(value, str) else ""


def _event_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {
        tok for tok in _EVENT_TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and not tok.isdigit() and tok not in EVENT_STOPWORDS
    }


def _derive_event_vocab(events: pd.DataFrame,
                        top_k: int = EVENT_VOCAB_SIZE,
                        min_docs: int = EVENT_VOCAB_MIN_DOCS) -> list[str]:
    if events.empty:
        return []
    counts: dict[str, int] = {}
    for _, ev in events.iterrows():
        text = f"{_as_text(ev.get('title', ''))} {_as_text(ev.get('description', ''))}"
        for tok in _event_tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(
        ((c, w) for w, c in counts.items() if c >= min_docs),
        key=lambda p: (-p[0], p[1]),
    )
    return [w for _, w in ranked[:top_k]]


def event_word_cols(vocab: list[str]) -> list[str]:
    return [f"evword_{w}" for w in vocab]


def _load_parking(conn_string: str, min_date: pd.Timestamp | None = None) -> pd.DataFrame:
    query = """
        SELECT recorded_at, location_name, latitude, longitude,
               total_spaces, free_spaces, occupancy
        FROM parking_snapshots
    """
    params = {}
    if min_date is not None:
        query += " WHERE recorded_at >= %(min_date)s"
        params["min_date"] = min_date
    query += " ORDER BY location_name, recorded_at;"

    with psycopg2.connect(conn_string) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    logger.info("Loaded %d parking snapshots across %d lots.",
                len(df), df["location_name"].nunique())
    return df


def _load_lot_coords(conn_string: str) -> dict[str, tuple[float, float]]:
    query = """
        SELECT DISTINCT ON (location_name) location_name, latitude, longitude
        FROM parking_snapshots
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY location_name, recorded_at DESC;
    """
    with psycopg2.connect(conn_string) as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _load_events_active(conn_string: str, min_date: pd.Timestamp,
                        max_date: pd.Timestamp) -> pd.DataFrame:
    query = """
        SELECT ei.id, ei.event_id, ei.start_time, ei.end_time,
               e.title, e.description, e.location_name, e.latitude, e.longitude
        FROM event_instances ei
        JOIN events e ON ei.event_id = e.id
        WHERE ei.start_time <= %(max_date)s
          AND (ei.end_time IS NULL OR ei.end_time >= %(min_date)s)
          AND e.latitude IS NOT NULL
          AND e.longitude IS NOT NULL;
    """
    with psycopg2.connect(conn_string) as conn:
        df = pd.read_sql_query(
            query, conn,
            params={"min_date": min_date, "max_date": max_date},
        )
    if df.empty:
        return df
    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["end_time"] = df["end_time"].fillna(pd.Timestamp.max.tz_localize("UTC"))
    return df


def _resample_grid(df: pd.DataFrame, minutes: int = GRID_MINUTES) -> pd.DataFrame:
    """Resample each lot to a regular grid and forward-fill."""
    n_before = len(df)
    freq = f"{minutes}min"
    df = df.sort_values(["location_name", "recorded_at"])
    df["recorded_at"] = df["recorded_at"].dt.floor(freq)
    df = df.drop_duplicates(subset=["location_name", "recorded_at"], keep="last")

    parts = []
    for lot, group in df.groupby("location_name"):
        group = group.set_index("recorded_at")
        lo = group.index.min().floor(freq)
        hi = group.index.max().ceil(freq)
        full_range = pd.date_range(lo, hi, freq=freq, tz=group.index.tz)
        group = group.reindex(full_range, method="ffill")
        group["location_name"] = lot
        parts.append(group)

    if not parts:
        return df
    df = pd.concat(parts).reset_index(names="recorded_at")
    logger.info("Resampled %d → %d rows (%d-minute grid).", n_before, len(df), minutes)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Eastern-local clock features plus cyclic encodings."""
    df = df.copy()
    local = df["recorded_at"].dt.tz_convert(Config.TIMEZONE)
    df["hour"] = local.dt.hour
    df["minute_of_hour"] = local.dt.minute
    df["day_of_week"] = local.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = local.dt.month
    hour_frac = df["hour"] + df["minute_of_hour"] / 60.0
    df["hour_sin"] = np.sin(2 * math.pi * hour_frac / 24.0)
    df["hour_cos"] = np.cos(2 * math.pi * hour_frac / 24.0)
    df["dow_sin"] = np.sin(2 * math.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * math.pi * df["day_of_week"] / 7.0)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flags = df["recorded_at"].map(calendar_flags)
    for col in ("is_instructional", "is_break", "is_exam_week", "is_holiday"):
        df[col] = flags.map(lambda d, c=col: d[c]).astype(int)
    return df


def _steps(minutes: int) -> int:
    return max(1, minutes // GRID_MINUTES)


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location_name", "recorded_at"])
    for lag_minutes in LAG_MINUTES:
        df[f"lag_{lag_minutes}min"] = (
            df.groupby("location_name")["occupancy"].shift(_steps(lag_minutes))
        )
    df["lag_week"] = (
        df.groupby("location_name")["occupancy"].shift(_steps(WEEK_LAG_MINUTES))
    )
    return df


def _fill_week_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing 7-day lag with this lot's hour-of-week median."""
    df = df.copy()
    df["_how"] = (
        df["day_of_week"] * BUCKETS_PER_DAY
        + (df["hour"] * 60 + df["minute_of_hour"]) // GRID_MINUTES
    )
    med = df.groupby(["location_name", "_how"])["occupancy"].transform("median")
    df["lag_week"] = df["lag_week"].fillna(med)
    df["lag_week"] = df["lag_week"].fillna(df["occupancy"])
    return df.drop(columns=["_how"])


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location_name", "recorded_at"])
    for window in ROLLING_WINDOWS:
        col = f"rolling_mean_{window}min"
        df[col] = (
            df.groupby("location_name")["occupancy"]
            .transform(lambda x, w=_steps(window): x.rolling(w, min_periods=1).mean())
        )
    return df


def _to_naive_ns(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).dt.tz_convert(None).to_numpy()


def _add_event_features(df: pd.DataFrame, lot_coords: dict,
                        events: pd.DataFrame,
                        vocab: list[str] | None = None) -> pd.DataFrame:
    """Add event-proximity, home-football, and optional word features."""
    df = df.copy()
    for _, col in EVENT_ACTIVE_RADII:
        df[col] = 0
    df["minutes_until_next_event"] = EVENT_CAP_MINUTES
    df["minutes_since_last_event"] = EVENT_CAP_MINUTES
    for _, col in EVENT_START_WINDOWS:
        df[col] = 0
    df["is_major_event_day"] = 0
    df["is_home_football"] = 0

    word_cols = event_word_cols(vocab or [])
    for col in word_cols:
        df[col] = 0

    if events.empty:
        return df

    ev_starts = _to_naive_ns(events["start_time"])
    ev_ends = _to_naive_ns(events["end_time"])
    ev_lat = events["latitude"].to_numpy(dtype=float)
    ev_lng = events["longitude"].to_numpy(dtype=float)
    titles = events["title"].map(_as_text).to_numpy()
    football_mask = np.array([bool(_FOOTBALL_RE.search(t)) for t in titles])

    ev_word_mask = None
    if word_cols:
        ev_word_sets = [
            _event_tokens(f"{_as_text(r.title)} {_as_text(r.description)}")
            for r in events.itertuples(index=False)
        ]
        ev_word_mask = np.array(
            [[w in ws for w in vocab] for ws in ev_word_sets], dtype=bool,
        )

    # Campus-wide football: any matching event starting in the next 24h.
    if football_mask.any():
        fb_starts = np.sort(ev_starts[football_mask])
        t_all = _to_naive_ns(df["recorded_at"])
        lo = np.searchsorted(fb_starts, t_all, side="left")
        hi = np.searchsorted(fb_starts, t_all + np.timedelta64(24, "h"), side="right")
        df["is_home_football"] = ((hi - lo) > 0).astype(int)

    for lot_name, (lot_lat, lot_lng) in lot_coords.items():
        mask_lot = df["location_name"] == lot_name
        if not mask_lot.any():
            continue

        t = _to_naive_ns(df.loc[mask_lot, "recorded_at"])
        dists = np.array([
            haversine_m(lot_lat, lot_lng, la, lo)
            for la, lo in zip(ev_lat, ev_lng)
        ])

        for radius, col in EVENT_ACTIVE_RADII:
            m = dists <= radius
            if not m.any():
                continue
            counts = np.zeros(len(t), dtype=int)
            for s, e in zip(ev_starts[m], ev_ends[m]):
                counts += ((t >= s) & (t <= e)).astype(int)
            df.loc[mask_lot, col] = counts

        m500 = dists <= 500
        if not m500.any():
            continue
        starts_raw = ev_starts[m500]
        ends_raw = ev_ends[m500]
        starts = np.sort(starts_raw)
        ends = np.sort(ends_raw)

        idx = np.searchsorted(starts, t, side="left")
        until = np.full(len(t), EVENT_CAP_MINUTES)
        valid = idx < len(starts)
        until[valid] = np.minimum(
            (starts[idx[valid]] - t[valid]) / ONE_MINUTE_NS, EVENT_CAP_MINUTES,
        )

        idx_end = np.searchsorted(ends, t, side="right") - 1
        since = np.full(len(t), EVENT_CAP_MINUTES)
        valid2 = idx_end >= 0
        since[valid2] = np.minimum(
            (t[valid2] - ends[idx_end[valid2]]) / ONE_MINUTE_NS, EVENT_CAP_MINUTES,
        )

        df.loc[mask_lot, "minutes_until_next_event"] = until
        df.loc[mask_lot, "minutes_since_last_event"] = since

        for hours, col in EVENT_START_WINDOWS:
            lo = np.searchsorted(starts, t, side="left")
            hi = np.searchsorted(starts, t + np.timedelta64(hours, "h"), side="right")
            df.loc[mask_lot, col] = (hi - lo).astype(float)

        durations = ends_raw - starts_raw
        major = (durations >= EVENT_MAJOR_MIN_NS) & (durations <= EVENT_MAJOR_MAX_NS)
        if major.any():
            major_starts = np.sort(starts_raw[major])
            lo = np.searchsorted(major_starts, t, side="left")
            hi = np.searchsorted(major_starts, t + np.timedelta64(24, "h"), side="right")
            df.loc[mask_lot, "is_major_event_day"] = ((hi - lo) > 0).astype(float)

        if ev_word_mask is not None:
            starts5 = ev_starts[m500]
            ends5 = ev_ends[m500]
            active5 = (t[:, None] >= starts5[None, :]) & (t[:, None] <= ends5[None, :])
            soon5 = (starts5[None, :] >= t[:, None]) & (
                starts5[None, :] <= t[:, None] + np.timedelta64(24, "h")
            )
            word_counts = (
                (active5 | soon5).astype(np.int64) @ ev_word_mask[m500].astype(np.int64)
            )
            for j, col in enumerate(word_cols):
                if word_counts[:, j].any():
                    df.loc[mask_lot, col] = word_counts[:, j]

    return df


def _add_horizon_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["horizon_hours"] = df["horizon_minutes"] / 60.0
    return df


def expand_horizons(df: pd.DataFrame,
                    horizons: tuple[int, ...] = FORECAST_HORIZONS,
                    with_target: bool = True) -> pd.DataFrame:
    """Copy each timestamp once per horizon; optionally attach occupancy(t+h)."""
    parts = []
    grouped_occ = None
    if with_target:
        grouped_occ = df.groupby("location_name")["occupancy"]
    for horizon in horizons:
        chunk = df.copy()
        chunk["horizon_minutes"] = horizon
        if with_target:
            chunk["target"] = grouped_occ.shift(-_steps(horizon))
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True)
    return _add_horizon_columns(out)


BASE_FEATURE_COLS = (
    ["hour", "minute_of_hour", "day_of_week", "is_weekend", "month",
     "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    + [f"lag_{m}min" for m in LAG_MINUTES]
    + ["lag_week"]
    + [f"rolling_mean_{m}min" for m in ROLLING_WINDOWS]
    + [col for _, col in EVENT_ACTIVE_RADII]
    + ["minutes_until_next_event", "minutes_since_last_event"]
    + [col for _, col in EVENT_START_WINDOWS]
    + ["is_major_event_day", "is_home_football"]
    + ["is_instructional", "is_break", "is_exam_week", "is_holiday"]
)

HORIZON_COLS = ["horizon_minutes", "horizon_hours"]
FEATURE_COLS = list(BASE_FEATURE_COLS) + HORIZON_COLS


def add_base_features(df: pd.DataFrame, lot_coords: dict,
                      events: pd.DataFrame, vocab: list[str]) -> pd.DataFrame:
    """Add all non-horizon features to a resampled occupancy frame."""
    df = _add_time_features(df)
    df = _add_calendar_features(df)
    df = _add_lag_features(df)
    df = _fill_week_lag(df)
    df = _add_rolling_features(df)
    df = _add_event_features(df, lot_coords, events, vocab=vocab)
    return df


def build_training_data(conn_string: str | None = None,
                        min_date: pd.Timestamp | None = None) -> tuple[
                            pd.DataFrame, pd.Series, list[str]]:
    """Engineer features and a single-target series (one row per t × horizon).

    Returns (X, y, vocab). X includes location_name + recorded_at + features.
    Rows missing required lags or the horizon target are dropped.
    """
    if conn_string is None:
        conn_string = Config.db_conn_string()

    logger.info("Loading parking data…")
    df = _load_parking(conn_string, min_date=min_date)
    if df.empty:
        raise ValueError("No parking data in database.")

    lot_coords = _load_lot_coords(conn_string)
    events = _load_events_active(
        conn_string,
        min_date=df["recorded_at"].min(),
        max_date=df["recorded_at"].max() + pd.Timedelta(minutes=FORECAST_MINUTES),
    )
    logger.info("Loaded %d event instances with geo.", len(events))

    vocab = _derive_event_vocab(events)
    word_cols = event_word_cols(vocab)
    if vocab:
        logger.info("Event word features: %d (%s).", len(word_cols), ", ".join(vocab))
    else:
        logger.info("No event word features — vocabulary is empty.")

    df = _resample_grid(df)
    df = add_base_features(df, lot_coords, events, vocab)
    df = expand_horizons(df, with_target=True)

    required = list(BASE_FEATURE_COLS) + word_cols + HORIZON_COLS + ["target"]
    before = len(df)
    df = df.dropna(subset=required)
    logger.info("Dropped %d rows with NaN features/targets (%d remaining).",
                before - len(df), len(df))

    X = df[["location_name", "recorded_at"] + FEATURE_COLS + word_cols].reset_index(drop=True)
    y = df["target"].reset_index(drop=True)

    logger.info("Feature matrix: %d rows × %d features.", X.shape[0], X.shape[1] - 2)
    return X, y, vocab


def build_inference_rows(featured_lot: pd.DataFrame) -> pd.DataFrame:
    """Expand the latest already-featured grid point across all horizons.

    ``featured_lot`` must already have base features (run ``add_base_features``
    on the full lot history first so lags/rolling are valid).
    """
    lot_df = featured_lot.sort_values("recorded_at")
    need = _steps(max(LAG_MINUTES))
    if len(lot_df) <= need:
        lot = lot_df["location_name"].iloc[0] if len(lot_df) else "?"
        raise ValueError(
            f"Not enough history for {lot} ({len(lot_df)} rows, need > {need})"
        )
    latest = lot_df.iloc[[-1]].copy()
    return expand_horizons(latest, with_target=False)
