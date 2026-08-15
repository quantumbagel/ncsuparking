import pandas as pd

from baseline import fit_baseline, predict_times
from features import FEATURE_COLS, FORECAST_HORIZONS
from train import train_lot_model


def test_train_lot_model_runs_and_reports_mae():
    n = 80
    times = pd.date_range("2026-08-01", periods=n, freq="1h", tz="UTC")
    rows = []
    targets = []
    for t in times:
        local = t.tz_convert("America/New_York")
        occ = 20 + local.hour * 2
        for horizon in FORECAST_HORIZONS:
            future = t + pd.Timedelta(minutes=horizon)
            future_local = future.tz_convert("America/New_York")
            y = 20 + future_local.hour * 2
            row = {col: 0.0 for col in FEATURE_COLS}
            row.update({
                "hour": float(local.hour),
                "minute_of_hour": float(local.minute),
                "day_of_week": float(local.dayofweek),
                "is_weekend": float(local.dayofweek >= 5),
                "month": float(local.month),
                "horizon_minutes": float(horizon),
                "horizon_hours": horizon / 60.0,
                "lag_5min": float(occ),
                "recorded_at": t,
            })
            rows.append(row)
            targets.append(y)

    X = pd.DataFrame(rows)
    y = pd.Series(targets)
    table = fit_baseline(pd.DataFrame({
        "location_name": ["MRC Deck"] * n,
        "recorded_at": times,
        "occupancy": [20 + t.tz_convert("America/New_York").hour * 2 for t in times],
    }))
    model, metrics = train_lot_model(
        X[FEATURE_COLS], y, "MRC Deck", table, X["recorded_at"],
    )
    assert metrics["val_rows"] > 0
    assert metrics["active_model"] in {"xgb", "baseline"}
    assert metrics["val_mae_mean"] < 25
    pred = model.predict(X[FEATURE_COLS].iloc[[-1]])
    assert pred.shape[0] == 1
