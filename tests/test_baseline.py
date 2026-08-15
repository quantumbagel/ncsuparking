from datetime import timezone

import pandas as pd

from baseline import fit_baseline, predict_times
from features import hour_of_week_key


def _ts(year, month, day, hour, minute=0):
    return pd.Timestamp(year, month, day, hour, minute, tz="UTC")


def test_hour_of_week_uses_eastern_not_utc():
    # Tuesday 14:00 UTC = Tuesday 10:00 Eastern (EDT)
    utc = pd.Timestamp("2026-09-15 14:00", tz="UTC")
    eastern = utc.tz_convert("America/New_York")
    key = hour_of_week_key(utc)
    assert eastern.hour == 10
    assert key // (1440 // 5) == 1  # Tuesday
    assert key % (1440 // 5) == (10 * 60) // 5


def test_baseline_lookup_matches_median_of_that_slot():
    rows = []
    # Several Tuesdays at 14:00 UTC (10:00 ET) for one lot
    for day in (1, 8, 15, 22):
        rows.append({
            "location_name": "MRC Deck",
            "recorded_at": _ts(2026, 9, day, 14, 0),
            "occupancy": 40 + day,
        })
    # A different hour should not affect the 10am ET bucket
    rows.append({
        "location_name": "MRC Deck",
        "recorded_at": _ts(2026, 9, 15, 18, 0),
        "occupancy": 90,
    })
    table = fit_baseline(pd.DataFrame(rows))
    target = pd.Series([_ts(2026, 9, 29, 14, 0)])
    pred = predict_times(table, "MRC Deck", target)
    expected = pd.Series([40 + 1, 40 + 8, 40 + 15, 40 + 22]).median()
    assert pred[0] == expected


def test_unknown_lot_falls_back_to_campus_median():
    df = pd.DataFrame({
        "location_name": ["A", "A"],
        "recorded_at": [_ts(2026, 9, 15, 14), _ts(2026, 9, 15, 15)],
        "occupancy": [20, 40],
    })
    table = fit_baseline(df)
    pred = predict_times(table, "Missing Lot", pd.Series([_ts(2026, 9, 16, 14)]))
    assert pred[0] == 30.0


def test_empty_baseline_returns_fifty():
    table = fit_baseline(pd.DataFrame(columns=["location_name", "recorded_at", "occupancy"]))
    pred = predict_times(table, "X", pd.Series([pd.Timestamp.now(tz=timezone.utc)]))
    assert pred[0] == 50.0
