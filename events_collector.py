"""Poll the NCSU Localist events API and persist to PostgreSQL."""

import logging
from datetime import datetime, timezone

import psycopg2
import requests

from config import Config
from database import upsert_events, upsert_event_instances

logger = logging.getLogger(__name__)


def _fetch_page(page: int, days: int) -> dict:
    """Fetch a single page of events from the Localist API.

    Returns the decoded JSON response dict.
    """
    resp = requests.get(
        Config.EVENTS_API_URL,
        params={"days": days, "per_page": 100, "page": page},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _flatten_event(event_wrapper: dict) -> dict:
    """Extract the inner 'event' dict from a Localist event wrapper."""
    return event_wrapper["event"]


def _extract_instances(event: dict) -> list[dict]:
    """Extract event_instances from a Localist event into flat dicts for the DB."""
    instances = []
    for wrapper in event.get("event_instances", []):
        inst = wrapper["event_instance"]
        instances.append({
            "id": inst["id"],
            "event_id": inst["event_id"],
            "start_time": inst["start"],
            "end_time": inst.get("end"),
            "all_day": inst.get("all_day", False),
        })
    return instances


def fetch_all_events(days: int | None = None) -> tuple[list[dict], list[dict]]:
    """Paginate through all Localist events and return (events, instances).

    Args:
        days: Number of days ahead to fetch. Defaults to Config.EVENT_DAYS.

    Returns:
        Tuple of (flat event dicts, flat instance dicts).
    """
    if days is None:
        days = Config.EVENT_DAYS

    all_events: list[dict] = []
    all_instances: list[dict] = []

    page = 1
    while True:
        data = _fetch_page(page, days)
        page_info = data.get("page", {})
        event_wrappers = data.get("events", [])

        for wrapper in event_wrappers:
            event = _flatten_event(wrapper)
            all_events.append(event)
            all_instances.extend(_extract_instances(event))

        total_pages = page_info.get("total", 1)
        logger.debug("Fetched page %d/%d — %d events so far.",
                      page, total_pages, len(all_events))

        if not event_wrappers or page >= total_pages:
            break
        page += 1

    return all_events, all_instances


def poll_and_store_events() -> int:
    """Fetch all upcoming events and upsert into the database.

    Returns:
        Number of events upserted, or 0 on failure.
    """
    try:
        events, instances = fetch_all_events()
    except requests.RequestException as exc:
        logger.error("Failed to fetch events: %s", exc)
        return 0

    if not events:
        logger.warning("No events returned from Localist API.")
        return 0

    try:
        ev_count = upsert_events(events)
        inst_count = upsert_event_instances(instances)
    except psycopg2.Error as exc:
        logger.error("Failed to upsert events into database: %s", exc)
        return 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.info("[%s] Upserted %d events + %d instances.", now, ev_count, inst_count)
    return ev_count
