"""Feature engineering for the parking occupancy prediction model.

Pulls parking snapshots + events from PostgreSQL, resamples each lot to a
regular 1-minute grid (so lag / rolling / target shifts are time-accurate),
engineers time, lag, rolling, and event-proximity features, then builds a
feature matrix X and multi-horizon target matrix y (1440 columns = 24h).
"""

import logging
import re
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import psycopg2

from config import Config

logger = logging.getLogger(__name__)

# ── Forecast granularity ──────────────────────────────────────────────────────

FORECAST_MINUTES = Config.FORECAST_MINUTES  # 1440 = 24h at 1-min steps

# Minute lags used as features (1min, 5min, 15min, 1h, 24h)
LAG_MINUTES = (1, 5, 15, 60, 1440)

# Rolling-mean windows in minutes
ROLLING_WINDOWS = (15, 60)

# Event-proximity features
EVENT_ACTIVE_RADII = ((300, "events_active_300m"), (500, "events_active_500m"))
# "Events starting in the next N hours" (counts instances within 500m)
EVENT_START_WINDOWS = (
    (1, "events_starting_1h"),
    (3, "events_starting_3h"),
    (6, "events_starting_6h"),
    (24, "events_starting_24h"),
)
# Cap for minutes_until/since (24h ≈ "no nearby event")
EVENT_CAP_MINUTES = 1440.0
ONE_MINUTE_NS = np.timedelta64(1, "m")

# "Major" event day: any nearby event lasting 3–48h starting in next 24h.
# (Upper bound excludes the far-future sentinel used for NULL end times.)
EVENT_MAJOR_MIN_NS = np.timedelta64(3, "h")
EVENT_MAJOR_MAX_NS = np.timedelta64(48, "h")

# ── Event word features ──────────────────────────────────────────────────────
# Word-level features let the model learn that certain event types (from the
# title/description, e.g. "soccer") have a bigger impact on parking.  The
# vocabulary is derived from the training events and persisted by train.py so
# predictions use the exact same columns.
EVENT_VOCAB_SIZE = 25          # max words kept as features
EVENT_VOCAB_MIN_DOCS = 1       # a word must appear in ≥ this many events
EVENT_STOPWORDS = frozenset({
    "the", "and", "for", "with", "vs", "at", "of", "in", "on", "to", "a",
    "nc", "state", "university", "ncsu", "event", "free", "tickets", "ticket",
    "series", "day", "days", "annual", "presented", "presents", "pm", "am",
    "time", "times", "join", "us", "come", "see", "info", "information",
    "click", "register", "registration", "more", "details", "learn", "like",
})


# ── Haversine ─────────────────────────────────────────────────────────────────


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in metres between two (lat, lng) points."""
    r = 6_371_000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(a))


# ── Event word features ───────────────────────────────────────────────────────

_EVENT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _as_text(value) -> str:
    """Return the string form of a DB value, or '' for NULL/NaN."""
    return value if isinstance(value, str) else ""


def _event_tokens(text: str) -> set[str]:
    """Lowercased word tokens from event text, minus stopwords/numbers."""
    if not text:
        return set()
    return {
        tok for tok in _EVENT_TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and not tok.isdigit() and tok not in EVENT_STOPWORDS
    }


def _derive_event_vocab(events: pd.DataFrame,
                        top_k: int = EVENT_VOCAB_SIZE,
                        min_docs: int = EVENT_VOCAB_MIN_DOCS) -> list[str]:
    """Top event words across titles+descriptions, used as word features.

    Each event contributes a word at most once (membership, not multiplicity).
    Words are ranked by frequency, then alphabetically for determinism.
    """
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
    """Feature column names for the given event word vocabulary."""
    return [f"evword_{w}" for w in vocab]


# ── Database queries ─────────────────────────────────────────────────────────


def _load_parking(conn_string: str, min_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load parking snapshots, optionally filtered by min_date."""
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
    """Return {lot_name: (lat, lng)} for every distinct lot."""
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
    """Load event instances that fall within [min_date, max_date]."""
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
    # Treat NULL end_time as "still ongoing" (far-future end)
    df["end_time"] = df["end_time"].fillna(pd.Timestamp.max.tz_localize("UTC"))
    return df


# ── Resampling ────────────────────────────────────────────────────────────────


def _resample_minutely(df: pd.DataFrame) -> pd.DataFrame:
    """Resample each lot to regular 1-minute intervals with forward-fill.

    Core fix for delta-only storage: after this, shift(1) genuinely means
    "1 minute ago", not "the previous row" (which could be minutes or hours).

    Steps per lot:
      1. Keep only the *last* observation in each minute bucket.
      2. Reindex to a full minute range (min→max) and forward-fill gaps.
    """
    n_before = len(df)

    # 1. Deduplicate to one row per minute per lot (keep last, floor to minute)
    df = df.sort_values(["location_name", "recorded_at"])
    df["recorded_at"] = df["recorded_at"].dt.floor("1min")
    df = df.drop_duplicates(subset=["location_name", "recorded_at"], keep="last")

    # 2. Reindex each lot to a continuous minute grid and forward-fill
    parts = []
    for lot, group in df.groupby("location_name"):
        group = group.set_index("recorded_at")
        lo = group.index.min().floor("1min")
        hi = group.index.max().ceil("1min")
        full_range = pd.date_range(lo, hi, freq="1min", tz=group.index.tz)
        group = group.reindex(full_range, method="ffill")
        group["location_name"] = lot
        parts.append(group)

    df = pd.concat(parts).reset_index(names="recorded_at")
    logger.info("Resampled %d → %d rows (1-minute grid).", n_before, len(df))
    return df


# ── Feature builders ─────────────────────────────────────────────────────────


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, minute_of_hour, day_of_week, is_weekend, month columns."""
    df = df.copy()
    df["hour"] = df["recorded_at"].dt.hour
    df["minute_of_hour"] = df["recorded_at"].dt.minute
    df["day_of_week"] = df["recorded_at"].dt.dayofweek  # 0=Mon
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["recorded_at"].dt.month
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add *time-accurate* occupancy lag features per lot (in minutes).

    Assumes data is already resampled to 1-minute intervals.
    """
    df = df.sort_values(["location_name", "recorded_at"])
    for lag_minutes in LAG_MINUTES:
        col = f"lag_{lag_minutes}min"
        df[col] = df.groupby("location_name")["occupancy"].shift(lag_minutes)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean occupancy over minute windows (now time-accurate)."""
    df = df.sort_values(["location_name", "recorded_at"])
    for window in ROLLING_WINDOWS:
        col = f"rolling_mean_{window}min"
        df[col] = (
            df.groupby("location_name")["occupancy"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build FORECAST_MINUTES target columns (occupancy shifted -1…-1440 min).

    Uses numpy per-lot for speed and to avoid pandas frame fragmentation
    that results from assigning 1440 columns one-by-one.

    The DataFrame index is reset to a positional RangeIndex so that numpy
    slicing by position maps 1:1 onto rows (robust to any prior reordering).
    """
    df = df.sort_values(["location_name", "recorded_at"]).reset_index(drop=True)
    n_rows = len(df)
    target_arrays = np.full((n_rows, FORECAST_MINUTES), np.nan, dtype=float)

    for _, group in df.groupby("location_name"):
        start = group.index[0]          # positional after reset_index
        occ = group["occupancy"].to_numpy()
        n = len(occ)
        for k in range(1, FORECAST_MINUTES + 1):
            # target_{k}min[i] = occupancy k minutes ahead
            if k < n:
                target_arrays[start: start + n - k, k - 1] = occ[k:]

    target_df = pd.DataFrame(target_arrays, index=df.index, columns=TARGET_COLS)
    return pd.concat([df, target_df], axis=1)


def _to_naive_ns(s: pd.Series) -> np.ndarray:
    """Convert tz-aware timestamps to tz-naive datetime64[ns] (UTC wall clock)."""
    return pd.to_datetime(s, utc=True).dt.tz_convert(None).to_numpy()


def _add_event_features(df: pd.DataFrame, lot_coords: dict,
                        events: pd.DataFrame,
                        vocab: list[str] | None = None) -> pd.DataFrame:
    """Add event-proximity features for each snapshot row.

    Features (all computed from event instances with geo):
      - events_active_300m / 500m  : count overlapping the row timestamp
      - minutes_until_next_event   : min start-time ahead (500m), capped 24h
      - minutes_since_last_event   : min end-time behind (500m), capped 24h
      - events_starting_1h/3h/6h/24h : count of instances starting in next N hours (500m)
      - is_major_event_day         : any 3-48h event starts in next 24h (500m)
      - evword_<word>              : if `vocab` is given, count of events within
                                     500m whose title/description contains the
                                     word and that are active now or start in
                                     the next 24h — lets the model learn that
                                     e.g. "soccer" events hit parking harder.

    Note: the *_starting_* features count *instances* (a recurring event adds
    one per occurrence) — a deliberate proxy for event intensity.

    The future/past features are what let the model *anticipate* event-driven
    spikes rather than only react to events already underway.
    """
    df = df.copy()
    for col in EVENT_ACTIVE_RADII:
        df[col[1]] = 0
    df["minutes_until_next_event"] = EVENT_CAP_MINUTES
    df["minutes_since_last_event"] = EVENT_CAP_MINUTES
    for _, col in EVENT_START_WINDOWS:
        df[col] = 0
    df["is_major_event_day"] = 0

    # Word-level features: one count column per vocabulary word (all 0 by
    # default; filled below only for lots within 500m of matching events).
    word_cols = event_word_cols(vocab or [])
    if word_cols:
        for col in word_cols:
            df[col] = 0

    if events.empty:
        return df

    # Pre-convert event times to naive datetime64[ns] once
    ev_starts = _to_naive_ns(events["start_time"])
    ev_ends = _to_naive_ns(events["end_time"])
    ev_lat = events["latitude"].to_numpy(dtype=float)
    ev_lng = events["longitude"].to_numpy(dtype=float)

    # Pre-compute which events contain each vocabulary word (for word features)
    ev_word_mask = None
    if word_cols:
        ev_word_sets = [
            _event_tokens(f"{_as_text(r.title)} {_as_text(r.description)}")
            for r in events.itertuples(index=False)
        ]
        ev_word_mask = np.array(
            [[w in ws for w in vocab] for ws in ev_word_sets], dtype=bool,
        )

    for lot_name, (lot_lat, lot_lng) in lot_coords.items():
        mask_lot = df["location_name"] == lot_name
        if not mask_lot.any():
            continue

        t = _to_naive_ns(df.loc[mask_lot, "recorded_at"])  # sorted ascending

        # Distance from this lot to every event
        dists = np.array([
            haversine_m(lot_lat, lot_lng, la, lo)
            for la, lo in zip(ev_lat, ev_lng)
        ])

        # 1. Active-event counts within each radius
        for radius, col in EVENT_ACTIVE_RADII:
            m = dists <= radius
            if not m.any():
                continue
            counts = np.zeros(len(t), dtype=int)
            for s, e in zip(ev_starts[m], ev_ends[m]):
                counts += ((t >= s) & (t <= e)).astype(int)
            df.loc[mask_lot, col] = counts

        # 2. Future/past proximity + start-count windows at 500m
        m500 = dists <= 500
        if not m500.any():
            continue
        starts_raw = ev_starts[m500]
        ends_raw = ev_ends[m500]
        starts = np.sort(starts_raw)
        ends = np.sort(ends_raw)

        # minutes until next event start (>= t)
        idx = np.searchsorted(starts, t, side="left")
        until = np.full(len(t), EVENT_CAP_MINUTES)
        valid = idx < len(starts)
        until[valid] = np.minimum(
            (starts[idx[valid]] - t[valid]) / ONE_MINUTE_NS, EVENT_CAP_MINUTES,
        )

        # minutes since last event end (<= t)
        idx_end = np.searchsorted(ends, t, side="right") - 1
        since = np.full(len(t), EVENT_CAP_MINUTES)
        valid2 = idx_end >= 0
        since[valid2] = np.minimum(
            (t[valid2] - ends[idx_end[valid2]]) / ONE_MINUTE_NS, EVENT_CAP_MINUTES,
        )

        df.loc[mask_lot, "minutes_until_next_event"] = until
        df.loc[mask_lot, "minutes_since_last_event"] = since

        # events starting in the next N hours (count of instances in [t, t+Nh])
        for hours, col in EVENT_START_WINDOWS:
            lo = np.searchsorted(starts, t, side="left")
            hi = np.searchsorted(starts, t + np.timedelta64(hours, "h"), side="right")
            df.loc[mask_lot, col] = (hi - lo).astype(float)

        # is_major_event_day: any 3-48h event starting in next 24h
        durations = ends_raw - starts_raw
        major = (durations >= EVENT_MAJOR_MIN_NS) & (durations <= EVENT_MAJOR_MAX_NS)
        if major.any():
            major_starts = np.sort(starts_raw[major])
            lo = np.searchsorted(major_starts, t, side="left")
            hi = np.searchsorted(major_starts, t + np.timedelta64(24, "h"), side="right")
            df.loc[mask_lot, "is_major_event_day"] = ((hi - lo) > 0).astype(float)

        # Word-level counts: events within 500m that are active now or start
        # within the next 24h, bucketed by vocabulary word.
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


# ── Public API ────────────────────────────────────────────────────────────────

FEATURE_COLS = (
    ["hour", "minute_of_hour", "day_of_week", "is_weekend", "month"]
    + [f"lag_{m}min" for m in LAG_MINUTES]
    + [f"rolling_mean_{m}min" for m in ROLLING_WINDOWS]
    + [col for _, col in EVENT_ACTIVE_RADII]
    + ["minutes_until_next_event", "minutes_since_last_event"]
    + [col for _, col in EVENT_START_WINDOWS]
    + ["is_major_event_day"]
)

TARGET_COLS = [f"target_{m}min" for m in range(1, FORECAST_MINUTES + 1)]


def build_training_data(conn_string: str | None = None,
                        min_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame,
                                                                        pd.DataFrame]:
    """Engineer features and multi-horizon targets from the database.

    Args:
        conn_string: PostgreSQL connection string.  Defaults to Config.
        min_date: Optional lower bound on parking data (e.g. last 90 days).

    Returns:
        (X, y, vocab) tuple.  X has FEATURE_COLS + word columns + 'location_name'.
        y has FORECAST_MINUTES target columns (target_1min … target_1440min).
        vocab is the event word vocabulary used for the word columns.
        Rows where any lag or target is NaN are dropped.
    """
    if conn_string is None:
        conn_string = Config.db_conn_string()

    logger.info("Loading parking data…")
    df = _load_parking(conn_string, min_date=min_date)
    if df.empty:
        raise ValueError("No parking data in database.")

    logger.info("Loading lot coordinates…")
    lot_coords = _load_lot_coords(conn_string)

    logger.info("Loading events…")
    events = _load_events_active(
        conn_string,
        min_date=df["recorded_at"].min(),
        max_date=df["recorded_at"].max() + pd.Timedelta(minutes=FORECAST_MINUTES),
    )
    logger.info("Loaded %d event instances with geo.", len(events))

    # Derive the word vocabulary from these events; it is returned so train.py
    # can persist it for inference to reuse (identical feature columns).
    vocab = _derive_event_vocab(events)
    word_cols = event_word_cols(vocab)
    if vocab:
        logger.info("Event word features: %d (%s).", len(word_cols), ", ".join(vocab))
    else:
        logger.warning("No event word features — vocabulary is empty "
                       "(few or no events in the training window).")

    # ── Resample to 1-minute grid (critical: makes shifts time-accurate) ──
    df = _resample_minutely(df)

    # ── Feature engineering (order matters) ──────────────────────────────
    df = _add_time_features(df)
    df = _add_lag_features(df)        # depends on 1-min spacing
    df = _add_rolling_features(df)    # depends on 1-min spacing
    df = _add_event_features(df, lot_coords, events, vocab=vocab)

    # Multi-horizon targets (time-accurate: shift(-1440) = 24h ahead)
    df = _add_targets(df)

    # Drop rows missing any feature or target
    required = FEATURE_COLS + word_cols + TARGET_COLS
    before = len(df)
    df = df.dropna(subset=required)
    logger.info("Dropped %d rows with NaN features/targets (%d remaining).",
                before - len(df), len(df))

    X = df[["location_name"] + FEATURE_COLS + word_cols].reset_index(drop=True)
    y = df[TARGET_COLS].reset_index(drop=True)

    logger.info("Feature matrix: %d rows × %d features.", X.shape[0], X.shape[1] - 1)
    logger.info("Target matrix:   %d rows × %d minute-horizons.", y.shape[0], y.shape[1])
    return X, y, vocab
