import pandas as pd

from features import (
    FORECAST_HORIZONS,
    GRID_MINUTES,
    _add_time_features,
    _resample_grid,
    expand_horizons,
)


def test_resample_is_five_minute_grid():
    df = pd.DataFrame({
        "location_name": ["MRC Deck", "MRC Deck"],
        "recorded_at": [
            pd.Timestamp("2026-08-15 12:01:10", tz="UTC"),
            pd.Timestamp("2026-08-15 12:16:40", tz="UTC"),
        ],
        "occupancy": [10, 20],
    })
    out = _resample_grid(df, minutes=5)
    assert len(out) == 4  # 12:00, 12:05, 12:10, 12:15
    assert list(out["occupancy"]) == [10, 10, 10, 20]


def test_time_features_are_eastern():
    df = pd.DataFrame({
        "recorded_at": [pd.Timestamp("2026-09-15 14:00", tz="UTC")],  # 10:00 ET
    })
    out = _add_time_features(df)
    assert int(out["hour"].iloc[0]) == 10
    assert int(out["day_of_week"].iloc[0]) == 1  # Tuesday


def test_expand_horizons_shape():
    n = 20
    idx = pd.date_range("2026-08-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "location_name": ["MRC Deck"] * n,
        "recorded_at": idx,
        "occupancy": range(n),
    })
    out = expand_horizons(df, with_target=True)
    assert len(out) == n * len(FORECAST_HORIZONS)
    assert set(out["horizon_minutes"]) == set(FORECAST_HORIZONS)
    # 15-min horizon on a 5-min grid is a shift of 3
    first = out[out["horizon_minutes"] == 15].iloc[0]
    assert first["target"] == 3


def test_grid_minutes_default():
    assert GRID_MINUTES == 5
    assert FORECAST_HORIZONS[-1] == 1440
