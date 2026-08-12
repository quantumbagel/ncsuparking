"""PostgreSQL database connection and schema management."""

import datetime
import logging
from contextlib import contextmanager

import psycopg2

from config import Config

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS parking_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    location_name   TEXT NOT NULL,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    total_spaces    INTEGER NOT NULL,
    free_spaces     INTEGER NOT NULL,
    used_spaces     INTEGER NOT NULL,
    occupancy       INTEGER NOT NULL
);

-- Migration: add used_spaces and derive occupancy from raw counts.
-- The occupancy percentage returned by the API is deliberately NOT stored;
-- instead occupancy is recomputed as used / total * 100 so it always agrees
-- with the stored counts.  Idempotent: rewrites only rows that differ.
ALTER TABLE parking_snapshots ADD COLUMN IF NOT EXISTS used_spaces INTEGER;

UPDATE parking_snapshots
SET used_spaces = GREATEST(0, total_spaces - free_spaces),
    occupancy   = CASE WHEN total_spaces > 0
                       THEN ROUND(100.0 * GREATEST(0, total_spaces - free_spaces) / total_spaces)
                       ELSE 0 END
WHERE used_spaces IS NULL
   OR occupancy IS DISTINCT FROM CASE WHEN total_spaces > 0
                                      THEN ROUND(100.0 * GREATEST(0, total_spaces - free_spaces) / total_spaces)
                                      ELSE 0 END;

ALTER TABLE parking_snapshots ALTER COLUMN used_spaces SET NOT NULL;

-- Speed up queries by location + time
CREATE INDEX IF NOT EXISTS idx_snapshots_location_time
    ON parking_snapshots (location_name, recorded_at DESC);

-- Speed up most-recent lookups
CREATE INDEX IF NOT EXISTS idx_snapshots_recorded_at
    ON parking_snapshots (recorded_at DESC);

-- ── Events ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id              BIGINT PRIMARY KEY,
    title           TEXT NOT NULL,
    url             TEXT,
    location_name   TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    description     TEXT,
    free            BOOLEAN,
    experience      TEXT,
    recurring       BOOLEAN,
    first_date      DATE,
    last_date       DATE,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS event_instances (
    id              BIGINT PRIMARY KEY,
    event_id        BIGINT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ,
    all_day         BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_instances_start
    ON event_instances (start_time);
CREATE INDEX IF NOT EXISTS idx_instances_event
    ON event_instances (event_id);

CREATE INDEX IF NOT EXISTS idx_events_dates
    ON events (first_date, last_date);
CREATE INDEX IF NOT EXISTS idx_events_location
    ON events (location_name);

-- ── Predictions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL PRIMARY KEY,
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lot                 TEXT NOT NULL,
    horizon_minutes     INTEGER NOT NULL,
    predicted_occupancy DOUBLE PRECISION NOT NULL
);

-- Migration: earlier versions of this table used horizon_hours.
-- Add the new column if a pre-existing table lacks it.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS horizon_minutes INTEGER;

-- ── Accuracy history ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS accuracy_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    recorded_at     TIMESTAMPTZ NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    mae             DOUBLE PRECISION NOT NULL,
    n               INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_accuracy_time
    ON accuracy_snapshots (recorded_at DESC);

-- Speed up queries for latest predictions
CREATE INDEX IF NOT EXISTS idx_predictions_lot_time
    ON predictions (lot, predicted_at DESC);
"""

INSERT_SQL = """
INSERT INTO parking_snapshots
    (recorded_at, location_name, latitude, longitude,
     total_spaces, free_spaces, used_spaces, occupancy)
VALUES
    (%(recorded_at)s, %(location_name)s, %(latitude)s, %(longitude)s,
     %(total_spaces)s, %(free_spaces)s, %(used_spaces)s, %(occupancy)s);
"""


def init_db() -> None:
    """Create tables and indexes if they don't already exist."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    logger.info("Database schema initialized.")


LAST_STATE_SQL = """
SELECT DISTINCT ON (location_name)
    location_name, free_spaces, occupancy, total_spaces, used_spaces
FROM parking_snapshots
ORDER BY location_name, recorded_at DESC;
"""


def get_last_snapshots() -> dict[str, dict[str, int | float]]:
    """Return the most recent snapshot for every known lot.

    Returns:
        Dict keyed by location_name, e.g.:
        {"MRC Deck": {"free_spaces": 218, "occupancy": 33,
                       "total_spaces": 324, "used_spaces": 106}}
        Empty dict if the table has no rows yet.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(LAST_STATE_SQL)
            rows = cur.fetchall()
    return {
        row[0]: {
            "free_spaces": row[1],
            "occupancy": row[2],
            "total_spaces": row[3],
            "used_spaces": row[4],
        }
        for row in rows
    }


def insert_snapshot(lot: dict) -> None:
    """Insert a single parking lot snapshot row."""
    # Parse geocode "(lat, lng)" -> floats
    geocode = lot.get("geocode", "")
    lat, lng = None, None
    if geocode:
        try:
            parts = geocode.strip("()").split(",")
            lat, lng = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            logger.warning("Could not parse geocode: %s", geocode)

    total = int(lot["total_spaces"])
    free = int(lot["free_spaces"])
    # Store raw counts only.  occupancy is derived from the counts (matching
    # the DB migration) rather than trusting the API's returned percentage.
    # Integer math mirrors SQL ROUND() exactly (round half away from zero).
    used = max(0, total - free)
    occupancy = (200 * used + total) // (2 * total) if total > 0 else 0

    params = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc),
        "location_name": lot["location_name"],
        "latitude": lat,
        "longitude": lng,
        "total_spaces": total,
        "free_spaces": free,
        "used_spaces": used,
        "occupancy": occupancy,
    }

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, params)
        conn.commit()


def insert_snapshots(lots: list[dict]) -> int:
    """Insert a batch of parking lot snapshots. Returns count inserted."""
    for lot in lots:
        insert_snapshot(lot)
    return len(lots)


# ── Events ───────────────────────────────────────────────────────────────────

UPSERT_EVENT_SQL = """
INSERT INTO events (id, title, url, location_name, latitude, longitude,
                     description, free, experience, recurring,
                     first_date, last_date, fetched_at)
VALUES (%(id)s, %(title)s, %(url)s, %(location_name)s, %(latitude)s, %(longitude)s,
        %(description)s, %(free)s, %(experience)s, %(recurring)s,
        %(first_date)s, %(last_date)s, NOW())
ON CONFLICT (id) DO UPDATE SET
    title        = EXCLUDED.title,
    url          = EXCLUDED.url,
    location_name = EXCLUDED.location_name,
    latitude     = EXCLUDED.latitude,
    longitude    = EXCLUDED.longitude,
    description  = EXCLUDED.description,
    free         = EXCLUDED.free,
    experience   = EXCLUDED.experience,
    recurring    = EXCLUDED.recurring,
    first_date   = EXCLUDED.first_date,
    last_date    = EXCLUDED.last_date,
    fetched_at   = NOW();
"""

UPSERT_INSTANCE_SQL = """
INSERT INTO event_instances (id, event_id, start_time, end_time, all_day)
VALUES (%(id)s, %(event_id)s, %(start_time)s, %(end_time)s, %(all_day)s)
ON CONFLICT (id) DO UPDATE SET
    start_time = EXCLUDED.start_time,
    end_time   = EXCLUDED.end_time,
    all_day    = EXCLUDED.all_day;
"""


def upsert_events(events: list[dict]) -> int:
    """Insert or update a batch of events from the Localist API.

    Returns count of events upserted.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            for ev in events:
                geo = ev.get("geo") or {}
                desc = ev.get("description_text") or ev.get("description", "")
                cur.execute(UPSERT_EVENT_SQL, {
                    "id": ev["id"],
                    "title": ev["title"],
                    "url": ev.get("localist_url", ""),
                    "location_name": ev.get("location_name"),
                    "latitude": float(geo["latitude"]) if geo.get("latitude") else None,
                    "longitude": float(geo["longitude"]) if geo.get("longitude") else None,
                    "description": desc,
                    "free": ev.get("free"),
                    "experience": ev.get("experience"),
                    "recurring": ev.get("recurring", False),
                    "first_date": ev.get("first_date"),
                    "last_date": ev.get("last_date"),
                })
            conn.commit()
    return len(events)


def upsert_event_instances(instances: list[dict]) -> int:
    """Insert or update a batch of event instances.

    Each dict should have: id, event_id, start_time, end_time, all_day.
    Returns count of instances upserted.
    """
    if not instances:
        return 0
    with _get_conn() as conn:
        with conn.cursor() as cur:
            for inst in instances:
                cur.execute(UPSERT_INSTANCE_SQL, {
                    "id": inst["id"],
                    "event_id": inst["event_id"],
                    "start_time": inst["start_time"],
                    "end_time": inst["end_time"],
                    "all_day": inst.get("all_day", False),
                })
            conn.commit()
    return len(instances)


# ── Predictions ──────────────────────────────────────────────────────────────

SAVE_PREDICTIONS_SQL = """
INSERT INTO predictions (predicted_at, lot, horizon_minutes, predicted_occupancy)
VALUES (%(predicted_at)s, %(lot)s, %(horizon_minutes)s, %(predicted_occupancy)s);
"""


def save_predictions(rows: list[dict]) -> int:
    """Insert a batch of prediction rows. Returns count inserted."""
    if not rows:
        return 0
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(SAVE_PREDICTIONS_SQL, rows)
        conn.commit()
    return len(rows)


SAVE_ACCURACY_SQL = """
INSERT INTO accuracy_snapshots (recorded_at, horizon_minutes, mae, n)
VALUES (%(recorded_at)s, %(horizon_minutes)s, %(mae)s, %(n)s);
"""


def save_accuracy_snapshots(rows: list[dict]) -> int:
    """Insert a batch of accuracy-snapshot rows. Returns count inserted."""
    if not rows:
        return 0
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(SAVE_ACCURACY_SQL, rows)
        conn.commit()
    return len(rows)


@contextmanager
def _get_conn():
    conn = psycopg2.connect(Config.db_conn_string())
    try:
        yield conn
    finally:
        conn.close()
