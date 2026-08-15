"""Streamlit dashboard for the NCSU parking occupancy forecasting system."""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from accuracy import EVAL_HORIZONS, compute_accuracy, query_accuracy_history
from config import Config
from database import get_heartbeat
from features import FORECAST_HORIZONS, haversine_m
from occupancy import grid_fill_occupancy, range_freq
from train import MODEL_DIR, get_train_state, start_training_async

st.set_page_config(
    page_title="NCSU Parking Forecast",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NCSU_RED = "#CC0000"
NCSU_RED_DARK = "#990000"
GRID = "#E5E7EB"
TZ = ZoneInfo(Config.TIMEZONE)
LIVE_VIEWS = {"Find a spot", "Overview", "Map"}
NAV_PAGES = [
    "Find a spot", "Forecast", "Map", "Overview",
    "History", "Patterns", "Accuracy", "Training",
]
RANGE_PRESETS = {
    "24 hours": timedelta(hours=24),
    "7 days": timedelta(days=7),
    "30 days": timedelta(days=30),
    "All": None,
    "Custom": "custom",
}

st.markdown(
    f"""
    <style>
    .main .block-container {{ padding-top: 1.5rem; }}
    h1, h2, h3 {{ color: #1f2937; }}
    [data-testid="stMetricValue"] {{ color: {NCSU_RED_DARK}; }}
    [data-testid="stMetricLabel"] {{ color: #6b7280; }}
    div[data-testid="stMetric"] {{
        background: #f9fafb;
        border: 1px solid {GRID};
        border-radius: 10px;
        padding: 12px 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(Config.db_conn_string())


def to_et(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(TZ)


def now_et() -> datetime:
    return datetime.now(TZ)


def fmt_local(ts) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return "—"
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ).strftime("%Y-%m-%d %H:%M")


@st.cache_data(ttl=30)
def query_lots() -> list[str]:
    with _conn() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT location_name FROM parking_snapshots ORDER BY location_name;",
            conn,
        )
    return df["location_name"].tolist()


@st.cache_data(ttl=20)
def query_data_bounds() -> tuple[datetime | None, datetime | None]:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MIN(recorded_at), MAX(recorded_at) FROM parking_snapshots;")
        lo, hi = cur.fetchone()
    return lo, hi


@st.cache_data(ttl=20)
def query_overview() -> dict:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM parking_snapshots;")
        snapshots = cur.fetchone()[0]
        cur.execute("SELECT MAX(recorded_at) FROM parking_snapshots;")
        latest = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions;")
        preds = cur.fetchone()[0]
        cur.execute("SELECT MAX(predicted_at) FROM predictions;")
        latest_pred = cur.fetchone()[0]
    hb = get_heartbeat("parking")
    return {
        "snapshots": snapshots,
        "latest": latest,
        "predictions": preds,
        "latest_pred": latest_pred,
        "heartbeat": hb["last_poll_at"] if hb else None,
        "lots_seen": hb["lots_seen"] if hb else None,
    }


@st.cache_data(ttl=30)
def query_occupancy_range(start: datetime, end: datetime) -> pd.DataFrame:
    """Occupancy for every lot, ffilled onto a grid that fits the span.

    Snapshots are stored only when a lot changes, so a naive GROUP BY
    would drop lots between updates and warp the campus average.
    """
    query = """
        (
            SELECT recorded_at, location_name, occupancy, used_spaces, total_spaces
            FROM parking_snapshots
            WHERE recorded_at >= %(start)s AND recorded_at <= %(end)s
        )
        UNION ALL
        (
            SELECT s.recorded_at, s.location_name, s.occupancy, s.used_spaces, s.total_spaces
            FROM parking_snapshots s
            JOIN (
                SELECT DISTINCT ON (location_name) id
                FROM parking_snapshots
                WHERE recorded_at < %(start)s
                ORDER BY location_name, recorded_at DESC
            ) prev ON prev.id = s.id
        )
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn, params={"start": start, "end": end})
    if df.empty:
        return df
    return grid_fill_occupancy(df, start, end, range_freq(start, end))


@st.cache_data(ttl=20)
def query_current_occupancy_map() -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (location_name)
            location_name, latitude, longitude, occupancy, used_spaces, total_spaces
        FROM parking_snapshots
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY location_name, recorded_at DESC;
    """
    with _conn() as conn:
        return pd.read_sql_query(query, conn)


@st.cache_data(ttl=20)
def query_latest_per_lot() -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (location_name)
            location_name, occupancy, used_spaces, total_spaces, free_spaces, recorded_at
        FROM parking_snapshots
        ORDER BY location_name, recorded_at DESC;
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn)
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df


@st.cache_data(ttl=30)
def query_forecast_map(horizon_minutes: int) -> pd.DataFrame:
    query = """
        SELECT p.lot AS location_name, p.predicted_occupancy AS occupancy,
               p.baseline_occupancy, c.latitude, c.longitude
        FROM predictions p
        JOIN (
            SELECT DISTINCT ON (location_name) location_name, latitude, longitude
            FROM parking_snapshots
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY location_name, recorded_at DESC
        ) c ON c.location_name = p.lot
        WHERE p.horizon_minutes = %(horizon)s
          AND p.predicted_at = (SELECT MAX(predicted_at) FROM predictions)
        ORDER BY p.lot;
    """
    with _conn() as conn:
        return pd.read_sql_query(query, conn, params={"horizon": horizon_minutes})


@st.cache_data(ttl=20)
def query_latest_forecast(lot: str) -> pd.DataFrame:
    query = """
        SELECT predicted_at, horizon_minutes, predicted_occupancy,
               baseline_occupancy, model_name
        FROM predictions
        WHERE lot = %(lot)s
          AND predicted_at = (SELECT MAX(predicted_at) FROM predictions WHERE lot = %(lot)s)
        ORDER BY horizon_minutes;
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn, params={"lot": lot})
    if df.empty:
        return df
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True)
    df["forecast_time"] = df["predicted_at"] + pd.to_timedelta(df["horizon_minutes"], unit="m")
    return df


@st.cache_data(ttl=20)
def query_all_latest_forecasts() -> pd.DataFrame:
    query = """
        SELECT p.lot, p.predicted_at, p.horizon_minutes,
               p.predicted_occupancy, p.baseline_occupancy, p.model_name
        FROM predictions p
        WHERE p.predicted_at = (SELECT MAX(predicted_at) FROM predictions);
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn)
    if not df.empty:
        df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True)
    return df


@st.cache_data(ttl=60)
def query_history(lot: str, start: datetime, end: datetime) -> pd.DataFrame:
    query = """
        SELECT recorded_at, occupancy, free_spaces, total_spaces
        FROM parking_snapshots
        WHERE location_name = %(lot)s
          AND recorded_at >= %(start)s AND recorded_at <= %(end)s
        ORDER BY recorded_at;
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn, params={"lot": lot, "start": start, "end": end})
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df


@st.cache_data(ttl=60)
def query_lot_coords() -> dict[str, tuple[float, float]]:
    query = """
        SELECT DISTINCT ON (location_name) location_name, latitude, longitude
        FROM parking_snapshots
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY location_name, recorded_at DESC;
    """
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


@st.cache_data(ttl=60)
def query_events_in_range(start: datetime, end: datetime) -> pd.DataFrame:
    query = """
        SELECT e.title, e.latitude, e.longitude, ei.start_time, ei.end_time
        FROM event_instances ei
        JOIN events e ON ei.event_id = e.id
        WHERE ei.start_time <= %(end)s
          AND (ei.end_time IS NULL OR ei.end_time >= %(start)s)
          AND e.latitude IS NOT NULL AND e.longitude IS NOT NULL;
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn, params={"start": start, "end": end})
    if df.empty:
        return df
    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["end_time"] = df["end_time"].fillna(df["start_time"] + pd.Timedelta(hours=2))
    return df


def load_model_summary() -> dict:
    path = MODEL_DIR / "summary.json"
    if not path.exists():
        return {"trained_at": None, "lots": {}}
    import json
    with open(path) as f:
        return json.load(f)


def nearby_events(lot: str, events: pd.DataFrame, radius_m: int = 500) -> pd.DataFrame:
    coords = query_lot_coords()
    if lot not in coords or events.empty:
        return events.iloc[0:0]
    lat, lng = coords[lot]
    dists = np.array([
        haversine_m(lat, lng, float(r.latitude), float(r.longitude))
        for _, r in events.iterrows()
    ])
    return events.loc[dists <= radius_m]


def interpolate_at(fc: pd.DataFrame, minutes: float, column: str) -> float | None:
    if fc.empty or column not in fc.columns:
        return None
    series = fc[column].dropna()
    if series.empty:
        return None
    xs = fc.loc[series.index, "horizon_minutes"].to_numpy(dtype=float)
    ys = series.to_numpy(dtype=float)
    minutes = float(np.clip(minutes, xs.min(), xs.max()))
    return float(np.interp(minutes, xs, ys))


def smooth_forecast(fc: pd.DataFrame) -> pd.DataFrame:
    if fc.empty:
        return fc
    t0 = fc["predicted_at"].iloc[0]
    grid = np.arange(0, max(FORECAST_HORIZONS) + 1, 5)
    pred = np.interp(grid, fc["horizon_minutes"], fc["predicted_occupancy"])
    base_col = fc["baseline_occupancy"] if "baseline_occupancy" in fc.columns else fc["predicted_occupancy"]
    base = np.interp(grid, fc["horizon_minutes"], base_col.fillna(fc["predicted_occupancy"]))
    out = pd.DataFrame({
        "horizon_minutes": grid,
        "predicted_occupancy": pred,
        "baseline_occupancy": base,
        "forecast_time": t0 + pd.to_timedelta(grid, unit="m"),
    })
    return out


def resolve_range(preset: str, custom_start, custom_end) -> tuple[datetime, datetime, str]:
    now = datetime.now(timezone.utc)
    lo, hi = query_data_bounds()
    if preset == "All":
        start = lo or (now - timedelta(days=1))
        end = hi or now
        label = "all stored data"
    elif preset == "Custom":
        start = datetime.combine(custom_start, time.min, tzinfo=TZ).astimezone(timezone.utc)
        end = datetime.combine(custom_end, time.max, tzinfo=TZ).astimezone(timezone.utc)
        label = f"{custom_start} → {custom_end}"
    else:
        delta = RANGE_PRESETS[preset]
        start = now - delta
        end = now
        label = preset.lower()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return start, end, label


def range_picker(key: str, default: str = "7 days") -> tuple[datetime, datetime, str]:
    options = list(RANGE_PRESETS)
    idx = options.index(default) if default in options else 0
    preset = st.radio("Time range", options, index=idx, horizontal=True, key=f"{key}_preset")
    c1, c2 = st.columns(2)
    lo, hi = query_data_bounds()
    default_start = (now_et() - timedelta(days=7)).date()
    default_end = now_et().date()
    if lo is not None:
        default_start = pd.Timestamp(lo).tz_convert(TZ).date()
    start_date = c1.date_input(
        "Start", default_start, key=f"{key}_start",
        disabled=preset != "Custom",
    )
    end_date = c2.date_input(
        "End", default_end, key=f"{key}_end",
        disabled=preset != "Custom",
    )
    return resolve_range(preset, start_date, end_date)


def campus_series(recent: pd.DataFrame) -> pd.DataFrame:
    if recent.empty:
        return recent
    agg = recent.groupby("recorded_at").agg(
        used=("used_spaces", "sum"),
        total=("total_spaces", "sum"),
    )
    campus = (agg["used"] / agg["total"].replace(0, np.nan) * 100)
    return campus.rename("occupancy").reset_index()


def occupancy_chart(x, y, title: str, name: str = "Occupancy", extra=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=dict(color=NCSU_RED, width=2),
        fill="tozeroy", fillcolor="rgba(204,0,0,0.08)",
    ))
    if extra:
        for trace in extra:
            fig.add_trace(trace)
    fig.update_layout(
        title=title,
        xaxis_title=None, yaxis_title="Occupancy (%)",
        height=400, template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h"),
    )
    return fig


def lot_map(mdf: pd.DataFrame, title: str) -> None:
    if mdf.empty:
        st.info("No map data yet — still warming up.")
        return
    fig = px.scatter_mapbox(
        mdf, lat="latitude", lon="longitude",
        color="occupancy", hover_name="location_name",
        hover_data={"occupancy": ":.0f", "latitude": False, "longitude": False},
        color_continuous_scale=["#16a34a", "#facc15", "#cc0000"],
        range_color=[0, 100],
        zoom=14, height=520,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 35.7877, "lon": -78.6704},
        margin=dict(l=0, r=0, t=30, b=0),
        title=title,
        coloraxis_colorbar=dict(title="%"),
    )
    fig.update_traces(marker=dict(size=18))
    st.plotly_chart(fig, use_container_width=True)


def go_to_forecast(lot: str) -> None:
    st.session_state.selected_lot = lot
    # Cannot assign key="nav" after the radio exists; apply this next run.
    st.session_state.pending_nav = "Forecast"


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🅿️ NCSU Parking")
st.sidebar.caption(f"Times in {Config.TIMEZONE.replace('_', ' ')}")

if "pending_nav" in st.session_state:
    st.session_state.nav = st.session_state.pop("pending_nav")
elif "nav" not in st.session_state:
    st.session_state.nav = "Find a spot"

view = st.sidebar.radio("Navigate", NAV_PAGES, key="nav")

info = query_overview()
hb_label = fmt_local(info["heartbeat"]) if info["heartbeat"] else "—"
st.sidebar.markdown("---")
st.sidebar.caption(f"Collector last poll: {hb_label}")
if view in LIVE_VIEWS:
    st.sidebar.caption("This page auto-refreshes every 30 seconds.")
    st_autorefresh(interval=30_000, key="dashboard_auto_refresh")


# ── Find a spot ──────────────────────────────────────────────────────────────

if view == "Find a spot":
    st.title("Find a spot")
    st.caption("Pick a time. Lots are ranked from emptiest to fullest.")

    when = st.radio(
        "When do you want to park?",
        ["Now", "In 30 min", "In 1 hour", "In 3 hours", "Custom"],
        horizontal=True,
    )
    target = now_et()
    if when == "In 30 min":
        target = now_et() + timedelta(minutes=30)
    elif when == "In 1 hour":
        target = now_et() + timedelta(hours=1)
    elif when == "In 3 hours":
        target = now_et() + timedelta(hours=3)
    elif when == "Custom":
        c1, c2 = st.columns(2)
        d = c1.date_input("Date", now_et().date())
        t = c2.time_input("Time", (now_et() + timedelta(hours=1)).time().replace(second=0, microsecond=0))
        target = datetime.combine(d, t, tzinfo=TZ)

    minutes_ahead = (target.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 60.0
    st.write(f"**Looking at:** {target.strftime('%a %b %d, %I:%M %p')}")

    current = query_latest_per_lot()
    forecasts = query_all_latest_forecasts()
    if current.empty:
        st.info("No parking data yet — the collector is still warming up.")
    else:
        rows = []
        for _, lot_row in current.iterrows():
            lot = lot_row["location_name"]
            now_occ = float(lot_row["occupancy"])
            total = float(lot_row["total_spaces"] or 0)
            pred = now_occ
            typical = None
            model = "now"
            fc = forecasts[forecasts["lot"] == lot] if not forecasts.empty else forecasts
            if minutes_ahead > 0 and not fc.empty:
                pred = interpolate_at(fc, minutes_ahead, "predicted_occupancy")
                typical = interpolate_at(fc, minutes_ahead, "baseline_occupancy")
                if pred is None:
                    pred = now_occ
                model = str(fc["model_name"].iloc[0]) if "model_name" in fc.columns else "baseline"
            free_est = max(0, int(round((1 - pred / 100.0) * total))) if total else None
            status = "Likely full" if pred >= 90 else ("Filling up" if pred >= 75 else "Likely open")
            rows.append({
                "Lot": lot,
                "Now %": round(now_occ, 0),
                "At time %": round(pred, 0),
                "Typical %": None if typical is None else round(typical, 0),
                "Est. free": free_est,
                "Status": status,
                "Model": model,
            })
        table = pd.DataFrame(rows).sort_values("At time %")
        st.dataframe(table, use_container_width=True, hide_index=True, height=420)

        emptiest = table.iloc[0]
        st.success(
            f"Emptiest at that time: **{emptiest['Lot']}** "
            f"({emptiest['At time %']:.0f}% full, ~{emptiest['Est. free']} free)."
        )
        pick = st.selectbox("Open a lot forecast", table["Lot"].tolist())
        st.button("Open forecast", on_click=go_to_forecast, args=(pick,))


# ── Forecast ─────────────────────────────────────────────────────────────────

elif view == "Forecast":
    st.title("Forecast")
    st.caption("Next 24 hours vs a typical day at this hour, Eastern time.")

    lots = query_lots()
    if not lots:
        st.info("No parking data yet.")
    else:
        default_lot = st.session_state.get("selected_lot")
        index = lots.index(default_lot) if default_lot in lots else 0
        lot = st.selectbox("Select lot", lots, index=index)
        st.session_state.selected_lot = lot
        fc = query_latest_forecast(lot)

        if fc.empty:
            st.warning(
                "No forecast yet. Baseline appears after the first collector "
                "cycle; ML needs a few days of data and a successful retrain."
            )
        else:
            smooth = smooth_forecast(fc)
            hist = query_history(
                lot,
                datetime.now(timezone.utc) - timedelta(hours=6),
                datetime.now(timezone.utc),
            )
            current_occ = float(hist["occupancy"].iloc[-1]) if not hist.empty else None
            peak_6h = float(smooth.loc[smooth["horizon_minutes"] <= 360, "predicted_occupancy"].max())
            in_24h = float(smooth["predicted_occupancy"].iloc[-1])
            typical_now = interpolate_at(fc, 60, "baseline_occupancy")

            c1, c2, c3, c4 = st.columns(4)
            if current_occ is not None:
                c1.metric("Right now", f"{current_occ:.0f}%")
            c2.metric("Peak next 6h", f"{peak_6h:.0f}%")
            c3.metric("In 24h", f"{in_24h:.0f}%")
            if typical_now is not None:
                c4.metric("Typical in 1h", f"{typical_now:.0f}%")

            extra = [go.Scatter(
                x=to_et(smooth["forecast_time"]),
                y=smooth["baseline_occupancy"],
                mode="lines", name="Typical day",
                line=dict(color="#6b7280", width=1.5, dash="dash"),
            )]
            if not hist.empty:
                extra.append(go.Scatter(
                    x=to_et(hist["recorded_at"]),
                    y=hist["occupancy"],
                    mode="lines", name="Actual (last 6h)",
                    line=dict(color="#111827", width=2),
                ))

            events = query_events_in_range(
                datetime.now(timezone.utc) - timedelta(hours=6),
                datetime.now(timezone.utc) + timedelta(minutes=max(FORECAST_HORIZONS)),
            )
            nearby = nearby_events(lot, events)
            fig = occupancy_chart(
                to_et(smooth["forecast_time"]),
                smooth["predicted_occupancy"],
                f"{lot} — next 24 hours",
                name="Predicted",
                extra=extra,
            )
            for _, ev in nearby.iterrows():
                fig.add_vrect(
                    x0=ev["start_time"].tz_convert(TZ),
                    x1=ev["end_time"].tz_convert(TZ),
                    fillcolor="rgba(16,185,129,0.12)", line_width=0,
                )
            if current_occ is not None:
                fig.add_hline(
                    y=current_occ, line_dash="dot", line_color="#6b7280",
                    annotation_text="now", annotation_position="top left",
                )
            st.plotly_chart(fig, use_container_width=True)

            model = fc["model_name"].iloc[0] if "model_name" in fc.columns else "—"
            st.caption(f"Active model for this lot: `{model}`. Shaded green bands are nearby events.")

            if not nearby.empty:
                with st.expander(f"Nearby events ({len(nearby)})"):
                    ev_show = nearby[["title", "start_time", "end_time"]].copy()
                    ev_show["start_time"] = to_et(ev_show["start_time"])
                    ev_show["end_time"] = to_et(ev_show["end_time"])
                    ev_show.columns = ["Event", "Start", "End"]
                    st.dataframe(ev_show, use_container_width=True, hide_index=True)

            with st.expander("Horizon table"):
                show = fc[["forecast_time", "horizon_minutes", "predicted_occupancy"]].copy()
                if "baseline_occupancy" in fc.columns:
                    show["baseline_occupancy"] = fc["baseline_occupancy"]
                show["forecast_time"] = to_et(show["forecast_time"])
                show.columns = ["Time"] + list(show.columns[1:])
                st.dataframe(show, use_container_width=True, hide_index=True)


# ── Map ──────────────────────────────────────────────────────────────────────

elif view == "Map":
    st.title("Campus Map")
    st.caption("Hover a lot for its name and occupancy.")

    mode = st.radio("Mode", ["Current occupancy", "Predicted occupancy"], horizontal=True)
    if mode == "Current occupancy":
        mdf = query_current_occupancy_map()
        lot_map(mdf, "Occupancy right now")
        horizon_label = "now"
    else:
        horizon = st.select_slider(
            "Forecast horizon",
            options=list(EVAL_HORIZONS),
            value=60,
            format_func=lambda m: f"{m} min" if m < 60 else f"{m // 60} h",
        )
        mdf = query_forecast_map(horizon)
        lot_map(mdf, f"Predicted occupancy in {horizon} min")
        horizon_label = f"+{horizon} min"

    if not mdf.empty:
        st.caption(f"Showing {horizon_label}. Green = empty, red = full.")
        show_cols = [c for c in ("location_name", "occupancy", "used_spaces", "total_spaces") if c in mdf.columns]
        rename = {
            "location_name": "Lot", "occupancy": "Occupancy (%)",
            "used_spaces": "Used", "total_spaces": "Total",
        }
        st.dataframe(
            mdf[show_cols].sort_values("occupancy", ascending=False).rename(columns=rename),
            use_container_width=True, hide_index=True,
        )


# ── Overview ─────────────────────────────────────────────────────────────────

elif view == "Overview":
    st.title("Overview")
    st.caption("Campus occupancy over any range — including the full history.")

    cols = st.columns(4)
    cols[0].metric("Snapshots stored", f"{info['snapshots']:,}")
    cols[1].metric("Lots tracked", len(query_lots()))
    cols[2].metric("Predictions stored", f"{info['predictions']:,}")
    cols[3].metric("Last collector poll", fmt_local(info["heartbeat"] or info["latest"]))

    start, end, label = range_picker("overview", default="24 hours")
    recent = query_occupancy_range(start, end)
    if recent.empty:
        st.info("No data in this range.")
    else:
        campus = campus_series(recent)
        st.plotly_chart(
            occupancy_chart(
                to_et(campus["recorded_at"]), campus["occupancy"],
                f"How full is campus? ({label})",
            ),
            use_container_width=True,
        )
        lo, hi = query_data_bounds()
        st.caption(
            f"{len(recent):,} downsampled points · stored data "
            f"{fmt_local(lo)} → {fmt_local(hi)}"
        )

        latest_per_lot = (
            recent.sort_values("recorded_at")
            .groupby("location_name")["occupancy"].last()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig2 = go.Figure(go.Bar(
            x=latest_per_lot["occupancy"],
            y=latest_per_lot["location_name"],
            orientation="h",
            marker_color=NCSU_RED,
        ))
        fig2.update_layout(
            title="Latest occupancy in this range",
            xaxis_title="Occupancy (%)", yaxis_title=None,
            height=max(300, 28 * len(latest_per_lot)),
            template="plotly_white", margin=dict(l=10, r=20, t=50, b=40),
            xaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig2, use_container_width=True)

        forecasts = query_all_latest_forecasts()
        current = query_latest_per_lot()
        if not current.empty:
            now_best = current.sort_values("occupancy").head(3)
            st.subheader("Emptiest lots right now")
            st.dataframe(
                now_best[["location_name", "occupancy", "free_spaces", "total_spaces"]]
                .rename(columns={
                    "location_name": "Lot", "occupancy": "Now %",
                    "free_spaces": "Free", "total_spaces": "Total",
                }),
                use_container_width=True, hide_index=True,
            )
        if not forecasts.empty:
            at_1h = (
                forecasts[forecasts["horizon_minutes"] == 60]
                .sort_values("predicted_occupancy")
                .head(3)
            )
            if not at_1h.empty:
                st.subheader("Emptiest in 1 hour")
                st.dataframe(
                    at_1h[["lot", "predicted_occupancy", "baseline_occupancy"]]
                    .rename(columns={
                        "lot": "Lot",
                        "predicted_occupancy": "Predicted %",
                        "baseline_occupancy": "Typical %",
                    }),
                    use_container_width=True, hide_index=True,
                )


# ── History ──────────────────────────────────────────────────────────────────

elif view == "History":
    st.title("History")
    st.caption("Any lot, any range — including everything stored.")

    lots = query_lots()
    if not lots:
        st.info("No parking data yet.")
    else:
        default_lot = st.session_state.get("selected_lot")
        index = lots.index(default_lot) if default_lot in lots else 0
        lot = st.selectbox("Select lot", lots, index=index, key="hist_lot")
        start, end, label = range_picker("history", default="All")
        hist = query_history(lot, start, end)

        if hist.empty:
            st.info("No data in the selected range.")
        else:
            raw_n = len(hist)
            if raw_n > 15000:
                hist = (
                    hist.set_index("recorded_at")
                    .resample("5min")
                    .last()
                    .dropna(subset=["occupancy"])
                    .reset_index()
                )
            events = query_events_in_range(start, end)
            nearby = nearby_events(lot, events)
            fig = occupancy_chart(
                to_et(hist["recorded_at"]), hist["occupancy"],
                f"{lot} — {label} ({len(nearby)} nearby events)",
            )
            for _, ev in nearby.iterrows():
                fig.add_vrect(
                    x0=ev["start_time"].tz_convert(TZ),
                    x1=ev["end_time"].tz_convert(TZ),
                    fillcolor="rgba(16,185,129,0.15)", line_width=0,
                )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{raw_n:,} raw snapshots in this range.")

            if not nearby.empty:
                with st.expander(f"Nearby events ({len(nearby)})"):
                    ev_show = nearby[["title", "start_time", "end_time"]].copy()
                    ev_show["start_time"] = to_et(ev_show["start_time"])
                    ev_show["end_time"] = to_et(ev_show["end_time"])
                    ev_show.columns = ["Event", "Start", "End"]
                    st.dataframe(ev_show, use_container_width=True, hide_index=True)


# ── Patterns ─────────────────────────────────────────────────────────────────

elif view == "Patterns":
    st.title("Patterns")
    st.caption("Average occupancy by Eastern weekday and hour.")

    lots = query_lots()
    if not lots:
        st.info("No parking data yet.")
    else:
        default_lot = st.session_state.get("selected_lot")
        index = lots.index(default_lot) if default_lot in lots else 0
        lot = st.selectbox("Select lot", lots, index=index, key="pattern_lot")
        start, end, label = range_picker("patterns", default="All")
        hist = query_history(lot, start, end)

        if hist.empty:
            st.info("No data available for this lot.")
        else:
            h = hist.copy()
            local = to_et(h["recorded_at"])
            h["day"] = local.dt.dayofweek
            h["hour"] = local.dt.hour
            pivot = h.pivot_table(index="day", columns="hour", values="occupancy", aggfunc="mean")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=[days[i] for i in pivot.index],
                colorscale=[[0, "#ffffff"], [0.5, "#fca5a5"], [1, NCSU_RED_DARK]],
                colorbar=dict(title="Avg %"),
            ))
            fig.update_layout(
                title=f"{lot} — average occupancy ({label})",
                xaxis_title="Hour of day (Eastern)", yaxis_title=None,
                height=380, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Accuracy ─────────────────────────────────────────────────────────────────

elif view == "Accuracy":
    st.title("Forecast Accuracy")
    st.caption("Active model vs the typical-day baseline, once forecasts have matured.")

    history = query_accuracy_history()
    if not history.empty:
        fig_hist = go.Figure()
        for name, group in history.groupby("model_name"):
            fig_hist.add_trace(go.Scatter(
                x=to_et(group["recorded_at"]), y=group["overall_mae"],
                mode="lines+markers", name=str(name),
            ))
        fig_hist.update_layout(
            title="Overall error over time",
            xaxis_title=None, yaxis_title="Error (%)",
            height=240, template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    hours_back = st.slider("Show forecasts from the last", 6, 72, 24, step=6)
    mae = compute_accuracy(hours_back=hours_back)
    if mae.empty:
        st.info(
            "Nothing to evaluate yet. Forecasts can only be checked once "
            "the predicted time has passed."
        )
    else:
        overall = float((mae["mae"] * mae["n"]).sum() / mae["n"].sum())
        if mae["baseline_mae"].notna().any():
            base = float((mae["baseline_mae"] * mae["n"]).sum() / mae["n"].sum())
            c1, c2 = st.columns(2)
            c1.metric(f"Active model (last {hours_back}h)", f"{overall:.2f}%")
            c2.metric("Typical-day baseline", f"{base:.2f}%")
        else:
            st.metric(f"Overall error (last {hours_back}h)", f"{overall:.2f}%")

        avg = mae.groupby("horizon_minutes").apply(
            lambda g: (g["mae"] * g["n"]).sum() / g["n"].sum(),
            include_groups=False,
        ).reset_index(name="mae")
        fig = go.Figure(go.Scatter(
            x=avg["horizon_minutes"], y=avg["mae"],
            mode="lines+markers",
            line=dict(color=NCSU_RED, width=2),
            marker=dict(size=7),
            name="Active",
        ))
        if mae["baseline_mae"].notna().any():
            bavg = mae.groupby("horizon_minutes").apply(
                lambda g: (g["baseline_mae"] * g["n"]).sum() / g["n"].sum(),
                include_groups=False,
            ).reset_index(name="mae")
            fig.add_trace(go.Scatter(
                x=bavg["horizon_minutes"], y=bavg["mae"],
                mode="lines+markers", name="Baseline",
                line=dict(color="#6b7280", dash="dash"),
            ))
        fig.update_layout(
            title="Error by forecast horizon (all lots)",
            xaxis_title="Minutes ahead", yaxis_title="Error (%)",
            height=320, template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        pivot = mae.pivot_table(index="lot", columns="horizon_minutes", values="mae")
        pivot = pivot.round(2)
        pivot.columns = [f"{int(c)} min" for c in pivot.columns]
        st.subheader("Error by lot and horizon")
        st.dataframe(pivot, use_container_width=True)


# ── Training ─────────────────────────────────────────────────────────────────

elif view == "Training":
    st.title("Model Training")
    st.caption("Retrain runs in the collector. This page only requests it and shows status.")

    state = get_train_state()
    if state.get("running") or state.get("step") == "queued":
        st.warning(
            f"Training is `{state.get('step')}` "
            f"(started `{state.get('started_at')}`, lot `{state.get('lot')}`).\n\n"
            "Leave this page — the collector keeps working."
        )
    else:
        if state.get("done"):
            if state.get("error"):
                st.error(f"Training failed: {state['error']}")
            else:
                st.success("Training finished successfully.")
        if st.button("🔄 Retrain models", type="primary",
                     help="Writes a request file. The collector trains in the background."):
            start_training_async()
            st.rerun()

    summary = load_model_summary()
    if summary.get("trained_at"):
        st.markdown(f"**Last trained:** `{summary['trained_at']}`")
    else:
        st.info("No models trained yet. A typical-day baseline still forecasts after the first poll.")

    st.markdown("---")
    st.subheader("Per-lot performance")
    if not summary.get("lots"):
        st.info("No trained models to show.")
    else:
        rows = []
        for lot, meta in summary["lots"].items():
            mae = meta.get("val_mae_by_horizon", {}) or {}
            rows.append({
                "Lot": lot,
                "Active": meta.get("active_model"),
                "Training rows": meta.get("train_rows"),
                "Validation rows": meta.get("val_rows"),
                "XGB error": None if meta.get("val_mae_mean") is None else round(meta["val_mae_mean"], 2),
                "Baseline error": None if meta.get("baseline_mae") is None else round(meta["baseline_mae"], 2),
                "1 h": round(mae.get("target_60min", float("nan")), 2),
                "3 h": round(mae.get("target_180min", float("nan")), 2),
                "24 h": round(mae.get("target_1440min", float("nan")), 2),
            })
        mdf = pd.DataFrame(rows)
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        lot_names = list(summary["lots"])
        default_lot = st.session_state.get("selected_lot")
        index = lot_names.index(default_lot) if default_lot in lot_names else 0
        pick = st.selectbox("Error curve for", lot_names, index=index)
        mae = summary["lots"][pick].get("val_mae_by_horizon") or {}
        if mae:
            mins = [int(k.replace("target_", "").replace("min", "")) for k in mae]
            vals = list(mae.values())
            fig = go.Figure(go.Scatter(
                x=mins, y=vals, mode="lines+markers",
                line=dict(color=NCSU_RED, width=2),
            ))
            fig.update_layout(
                title=f"Validation error by horizon — {pick}",
                xaxis_title="Minutes ahead", yaxis_title="Error (%)",
                height=320, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
