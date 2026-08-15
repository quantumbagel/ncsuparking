"""Derive stored space counts from the dirty NCSU parking API."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def derive_counts(total_spaces: int, free_spaces: int) -> tuple[int, int, int, int]:
    """Return (total, free, used, occupancy) clipped to a valid lot.

    The live API sometimes reports negative ``free_spaces`` (overfull visitor
    decks).  Treat that as full.  Occupancy is always an integer 0–100.
    """
    total = max(0, int(total_spaces))
    free = int(free_spaces)
    if total == 0:
        return 0, 0, 0, 0
    if free < 0:
        free = 0
    if free > total:
        free = total
    used = total - free
    # Integer math matching SQL ROUND() (half away from zero).
    occupancy = (200 * used + total) // (2 * total)
    occupancy = min(100, max(0, occupancy))
    return total, free, used, occupancy


def range_freq(start: datetime, end: datetime) -> str:
    """Choose a chart bucket so long ranges stay readable."""
    span_hours = max(1.0, (end - start).total_seconds() / 3600.0)
    if span_hours <= 48:
        return "5min"
    if span_hours <= 14 * 24:
        return "15min"
    if span_hours <= 60 * 24:
        return "1h"
    return "6h"


def grid_fill_occupancy(df: pd.DataFrame, start: datetime, end: datetime,
                        freq: str) -> pd.DataFrame:
    """Forward-fill each lot onto a regular grid (delta-only storage)."""
    if df.empty:
        return df
    work = df.copy()
    work["recorded_at"] = pd.to_datetime(work["recorded_at"], utc=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    rng = pd.date_range(start_ts, end_ts, freq=freq, tz="UTC")
    parts = []
    for lot, group in work.groupby("location_name"):
        group = (
            group.sort_values("recorded_at")
            .drop_duplicates("recorded_at", keep="last")
            .set_index("recorded_at")
        )
        group = group.reindex(group.index.union(rng)).sort_index().ffill()
        group = group.reindex(rng)
        group["location_name"] = lot
        parts.append(group.reset_index(names="recorded_at"))
    return pd.concat(parts, ignore_index=True).dropna(subset=["occupancy"])
