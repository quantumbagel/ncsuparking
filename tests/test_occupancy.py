from datetime import datetime, timezone

import pandas as pd

from occupancy import derive_counts, grid_fill_occupancy, range_freq


def test_normal_lot():
    total, free, used, occ = derive_counts(100, 25)
    assert (total, free, used) == (100, 25, 75)
    assert occ == 75


def test_negative_free_is_full():
    total, free, used, occ = derive_counts(232, -486)
    assert total == 232
    assert free == 0
    assert used == 232
    assert occ == 100


def test_free_over_capacity():
    total, free, used, occ = derive_counts(50, 80)
    assert (total, free, used, occ) == (50, 50, 0, 0)


def test_zero_capacity():
    assert derive_counts(0, 10) == (0, 0, 0, 0)


def test_rounds_half_away_from_zero():
    # 1 used of 3 → 33.333… → 33
    assert derive_counts(3, 2)[3] == 33
    # 2 used of 3 → 66.666… → 67
    assert derive_counts(3, 1)[3] == 67


def test_range_freq_widens_for_long_spans():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert range_freq(start, datetime(2026, 1, 2, tzinfo=timezone.utc)) == "5min"
    assert range_freq(start, datetime(2026, 1, 10, tzinfo=timezone.utc)) == "15min"
    assert range_freq(start, datetime(2026, 2, 15, tzinfo=timezone.utc)) == "1h"
    assert range_freq(start, datetime(2026, 8, 1, tzinfo=timezone.utc)) == "6h"


def test_grid_fill_keeps_unchanged_lot_in_campus_average():
    start = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 15, 13, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "recorded_at": [
            datetime(2026, 8, 15, 11, 50, tzinfo=timezone.utc),  # before window
            datetime(2026, 8, 15, 12, 10, tzinfo=timezone.utc),
        ],
        "location_name": ["Quiet Deck", "Busy Deck"],
        "occupancy": [10, 80],
        "used_spaces": [10, 80],
        "total_spaces": [100, 100],
    })
    out = grid_fill_occupancy(df, start, end, "15min")
    quiet = out[out["location_name"] == "Quiet Deck"]
    assert not quiet.empty
    assert (quiet["occupancy"] == 10).all()
    # Campus share at a later bucket still includes the quiet lot.
    later = out[out["recorded_at"] == pd.Timestamp("2026-08-15 12:45", tz="UTC")]
    assert set(later["location_name"]) == {"Quiet Deck", "Busy Deck"}
