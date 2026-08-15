"""Poll the NCSU parking API and persist snapshots to PostgreSQL."""

import logging
from datetime import datetime, timezone

import requests

from config import Config
from database import insert_snapshots, get_last_snapshots, write_heartbeat

logger = logging.getLogger(__name__)


def fetch_parking_data() -> list[dict]:
    """Fetch current parking lot occupancy from the NCSU API.

    Returns:
        List of parking lot dicts, or empty list on failure.
    """
    try:
        resp = requests.get(Config.PARKING_API_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # The API wraps the array in an extra list: [[{...}, {...}]]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            data = data[0]
        return data
    except requests.RequestException as exc:
        logger.error("Failed to fetch parking data: %s", exc)
        return []


def poll_and_store() -> int:
    """Fetch parking data and store *only changed* lots to the database.

    Compares each lot's current free_spaces, used_spaces (= total − free),
    and total_spaces against the most recent snapshot.  The API's returned
    `occupancy` percentage is not trusted and is never stored.  Dynamically
    handles lots that appear or disappear from the API between polls.

    Returns:
        Number of lots whose state actually changed (and were stored).
    """
    lots = fetch_parking_data()
    if not lots:
        logger.warning("No parking data returned; skipping this poll.")
        write_heartbeat("parking", lots_seen=0, lots_changed=0, detail="empty_api")
        return 0

    last_state = get_last_snapshots()

    changed: list[dict] = []
    for lot in lots:
        name = lot["location_name"]
        free = int(lot["free_spaces"])
        total = int(lot["total_spaces"])
        used = max(0, total - free)  # used count — occupancy % is derived, never trusted

        prev = last_state.get(name)
        if prev is None:
            # Lot we've never seen before — always store
            changed.append(lot)
        elif (prev["free_spaces"] != free
              or prev["used_spaces"] != used
              or prev["total_spaces"] != total):
            changed.append(lot)
        # else: identical to last snapshot — skip

    if not changed:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info("[%s] No lots changed since last poll (%d total tracked).",
                     now, len(lots))
        write_heartbeat("parking", lots_seen=len(lots), lots_changed=0)
        return 0

    count = insert_snapshots(changed)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.info("[%s] %d/%d lots changed — stored %d snapshots.",
                 now, count, len(lots), count)
    write_heartbeat("parking", lots_seen=len(lots), lots_changed=count)
    return count
