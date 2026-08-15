"""PostgreSQL database connection and schema management."""

import datetime
import logging
from contextlib import contextmanager

import psycopg2

from config import Config
from occupancy import derive_counts

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

ALTER TABLE parking_snapshots ADD COLUMN IF NOT EXISTS used_spaces INTEGER;

-- Clip dirty API counts (negative free = overfull) and keep occupancy in 0–100.
UPDATE parking_snapshots
SET used_spaces = GREATEST(0, LEAST(total_spaces,
                    total_spaces - GREATEST(0, LEAST(free_spaces, total_spaces)))),
    occupancy   = CASE WHEN total_spaces > 0
                       THEN LEAST(100, GREATEST(0, ROUND(
                           100.0 * GREATEST(0, LEAST(total_spaces,
                               total_spaces - GREATEST(0, LEAST(free_spaces, total_spaces))))
                           / total_spaces)))
                       ELSE 0 END
WHERE used_spaces IS NULL
   OR occupancy < 0
   OR occupancy > 100
   OR occupancy IS DISTINCT FROM CASE WHEN total_spaces > 0
                                      THEN LEAST(100, GREATEST(0, ROUND(
                                          100.0 * GREATEST(0, LEAST(total_spaces,
                                              total_spaces - GREATEST(0, LEAST(free_spaces, total_spaces))))
                                          / total_spaces)))
                                      ELSE 0 END;

ALTER TABLE parking_snapshots ALTER COLUMN used_spaces SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_snapshots_location_time
    ON parking_snapshots (location_name, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_snapshots_recorded_at
    ON parking_snapshots (recorded_at DESC);

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

CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL PRIMARY KEY,
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lot                 TEXT NOT NULL,
    horizon_minutes     INTEGER NOT NULL,
    predicted_occupancy DOUBLE PRECISION NOT NULL,
    baseline_occupancy  DOUBLE PRECISION,
    model_name          TEXT
);

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS horizon_minutes INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS baseline_occupancy DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_name TEXT;

-- Drop leftover 1-minute-horizon rows from the old 1440-output models.
DELETE FROM predictions
WHERE horizon_minutes IS NULL
   OR horizon_minutes NOT IN (15, 30, 60, 120, 180, 360, 720, 1440);

DELETE FROM predictions a
    USING predictions b
WHERE a.id < b.id
  AND a.lot = b.lot
  AND a.predicted_at = b.predicted_at
  AND a.horizon_minutes = b.horizon_minutes;

CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique
    ON predictions (lot, predicted_at, horizon_minutes);

CREATE INDEX IF NOT EXISTS idx_predictions_lot_time
    ON predictions (lot, predicted_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_horizon_time
    ON predictions (horizon_minutes, predicted_at);

CREATE TABLE IF NOT EXISTS accuracy_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    recorded_at     TIMESTAMPTZ NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    mae             DOUBLE PRECISION NOT NULL,
    n               INTEGER NOT NULL,
    model_name      TEXT
);

ALTER TABLE accuracy_snapshots ADD COLUMN IF NOT EXISTS model_name TEXT;

CREATE INDEX IF NOT EXISTS idx_accuracy_time
    ON accuracy_snapshots (recorded_at DESC);

CREATE TABLE IF NOT EXISTS collector_heartbeat (
    name            TEXT PRIMARY KEY,
    last_poll_at    TIMESTAMPTZ NOT NULL,
    lots_seen       INTEGER,
    lots_changed    INTEGER,
    detail          TEXT
);
"""

INSERT_SQL = """
INSERT INTO parking_snapshots
    (recorded_at, location_name, latitude, longitude,
     total_spaces, free_spaces, used_spaces, occupancy)
VALUES
    (%(recorded_at)s, %(location_name)s, %(latitude)s, %(longitude)s,
     %(total_spaces)s, %(free_spaces)s, %(used_spaces)s, %(occupancy)s);
"""

HEARTBEAT_SQL = """
INSERT INTO collector_heartbeat (name, last_poll_at, lots_seen, lots_changed, detail)
VALUES (%(name)s, %(last_poll_at)s, %(lots_seen)s, %(lots_changed)s, %(detail)s)
ON CONFLICT (name) DO UPDATE SET
    last_poll_at = EXCLUDED.last_poll_at,
    lots_seen    = EXCLUDED.lots_seen,
    lots_changed = EXCLUDED.lots_changed,
    detail       = EXCLUDED.detail;
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
    """Return the most recent snapshot for every known lot."""
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


def _parse_geocode(lot: dict) -> tuple[float | None, float | None]:
    geocode = lot.get("geocode", "")
    if not geocode:
        return None, None
    try:
        parts = geocode.strip("()").split(",")
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        logger.warning("Could not parse geocode: %s", geocode)
        return None, None


def _snapshot_params(lot: dict, recorded_at: datetime.datetime) -> dict:
    lat, lng = _parse_geocode(lot)
    total, free, used, occupancy = derive_counts(
        int(lot["total_spaces"]), int(lot["free_spaces"]),
    )
    return {
        "recorded_at": recorded_at,
        "location_name": lot["location_name"],
        "latitude": lat,
        "longitude": lng,
        "total_spaces": total,
        "free_spaces": free,
        "used_spaces": used,
        "occupancy": occupancy,
    }


def insert_snapshot(lot: dict) -> None:
    """Insert a single parking lot snapshot row."""
    params = _snapshot_params(lot, datetime.datetime.now(datetime.timezone.utc))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, params)
        conn.commit()


def insert_snapshots(lots: list[dict]) -> int:
    """Insert a batch of parking lot snapshots. Returns count inserted."""
    if not lots:
        return 0
    now = datetime.datetime.now(datetime.timezone.utc)
    params = [_snapshot_params(lot, now) for lot in lots]
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, params)
        conn.commit()
    return len(lots)


def write_heartbeat(name: str, lots_seen: int = 0, lots_changed: int = 0,
                    detail: str | None = None) -> None:
    """Record that a collector job just ran (even if nothing changed)."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(HEARTBEAT_SQL, {
                "name": name,
                "last_poll_at": datetime.datetime.now(datetime.timezone.utc),
                "lots_seen": lots_seen,
                "lots_changed": lots_changed,
                "detail": detail,
            })
        conn.commit()


def get_heartbeat(name: str = "parking") -> dict | None:
    """Return the latest heartbeat row, or None."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_poll_at, lots_seen, lots_changed, detail "
                "FROM collector_heartbeat WHERE name = %s;",
                (name,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "last_poll_at": row[0],
        "lots_seen": row[1],
        "lots_changed": row[2],
        "detail": row[3],
    }


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
    """Insert or update a batch of events from the Localist API."""
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
    """Insert or update a batch of event instances."""
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
INSERT INTO predictions (predicted_at, lot, horizon_minutes,
                         predicted_occupancy, baseline_occupancy, model_name)
VALUES (%(predicted_at)s, %(lot)s, %(horizon_minutes)s,
        %(predicted_occupancy)s, %(baseline_occupancy)s, %(model_name)s)
ON CONFLICT (lot, predicted_at, horizon_minutes) DO UPDATE SET
    predicted_occupancy = EXCLUDED.predicted_occupancy,
    baseline_occupancy  = EXCLUDED.baseline_occupancy,
    model_name          = EXCLUDED.model_name;
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


def prune_predictions(retention_days: int | None = None) -> int:
    """Delete prediction rows older than the retention window."""
    days = retention_days if retention_days is not None else Config.PREDICTION_RETENTION_DAYS
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM predictions "
                "WHERE predicted_at < NOW() - make_interval(days => %s);",
                (days,),
            )
            deleted = cur.rowcount
        conn.commit()
    if deleted:
        logger.info("Pruned %d prediction rows older than %d days.", deleted, days)
    return deleted


SAVE_ACCURACY_SQL = """
INSERT INTO accuracy_snapshots (recorded_at, horizon_minutes, mae, n, model_name)
VALUES (%(recorded_at)s, %(horizon_minutes)s, %(mae)s, %(n)s, %(model_name)s);
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
