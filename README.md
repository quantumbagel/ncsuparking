# NCSU parking forecasts

Personal tool that watches NC State’s public parking API and answers: **if I go to this lot at this time, how full will it be?**

It stores occupancy, overlays campus events, and forecasts 15 minutes to 24 hours ahead. A typical-day baseline is available after the first poll. An XGBoost model is kept only when it beats that baseline.

## Run it

```bash
cp .env.example .env          # optional
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).

- **Find a spot** — pick a time, see every lot ranked emptiest → fullest
- **Forecast** — 24-hour curve vs a typical day, plus nearby events
- **Overview / History / Patterns** — any range, including **All** stored data
- **Training** — queues a retrain; the collector process actually trains

Times are **America/New_York**. Snapshots stay UTC in Postgres.

## When forecasts appear

| After | What you get |
| --- | --- |
| First successful poll | Current occupancy + a coarse typical-day baseline (may be flat on day 1) |
| A few days of data | Hour-of-week baseline that looks like campus rhythms |
| ~8 days | Last-week lag becomes real; baseline improves |
| Enough rows + a retrain | Per-lot XGBoost, used only if it beats baseline |

Retrain from the Training page, or wait for the hourly check. Models live in the `models` Docker volume (`baseline.json`, `summary.json`, `*.pkl`).

## Layout

| File | Role |
| --- | --- |
| `main.py` | Collector loop: poll, predict, retrain, retention |
| `collector.py` | Parking API → snapshots (delta-only) |
| `events_collector.py` | Localist events |
| `features.py` | 5-minute grid, Eastern time, events, calendar |
| `baseline.py` | Hour-of-week median |
| `train.py` | Baseline + XGB; file-based job for the dashboard |
| `predict.py` | 8 horizons per lot |
| `dashboard.py` | Streamlit UI |
| `academic_calendar.py` | Hardcoded NCSU terms |

## Tests

```bash
python -m pytest
```

## Notes

- Visitor decks sometimes report negative `free_spaces`. Those are stored as 100% full.
- Predictions keep 8 horizons (not one per minute) and are deleted after 14 days.
- The collector has a 4 GB memory limit so a train job cannot take the box down.
