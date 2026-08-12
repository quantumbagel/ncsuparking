"""Streamlit dashboard for the NCSU parking occupancy forecasting system.

Features:
  - Overview: live KPIs + campus-wide occupancy over the last 24h
  - Forecast: 24h (1440-min) predicted occupancy per lot
  - History:  raw occupancy over time, with nearby events overlaid
  - Patterns: day-of-week × hour-of-day occupancy heatmap
  - Training: force-retrain button + per-lot model metrics
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from accuracy import EVAL_HORIZONS, compute_accuracy, query_accuracy_history
from config import Config
from features import haversine_m
from train import MODEL_DIR, get_train_state, start_training_async

# ── Page config & theme ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="NCSU Parking Forecast",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NCSU_RED = "#CC0000"
NCSU_RED_DARK = "#990000"
GRID = "#E5E7EB"

st.markdown(
    f"""
    <style>
    .main .block-container {{ padding-top: 1.5rem; }}
    h1, h2, h3 {{ color: #1f2937; }}
    [data-testid="stMetricValue"] {{ color: {NCSU_RED_DARK}; }}
    [data-testid="stMetricLabel"] {{ color: #6b7280; }}
    .status-ok {{ color: #15803d; font-weight: 600; }}
    .status-warn {{ color: #b45309; font-weight: 600; }}
    .status-err {{ color: {NCSU_RED}; font-weight: 600; }}
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

# ── Data access helpers ───────────────────────────────────────────────────────


def _conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(Config.db_conn_string())


@st.cache_data(ttl=20)
def query_lots() -> list[str]:
    with _conn() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT location_name FROM parking_snapshots ORDER BY location_name;",
            conn,
        )
    return df["location_name"].tolist()


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
    return {
        "snapshots": snapshots,
        "latest": latest,
        "predictions": preds,
        "latest_pred": latest_pred,
    }


@st.cache_data(ttl=30)
def query_recent_occupancy(hours: int = 24) -> pd.DataFrame:
    query = """
        SELECT recorded_at, location_name, occupancy, used_spaces, total_spaces
        FROM parking_snapshots
        WHERE recorded_at >= NOW() - make_interval(hours => %(hours)s)
        ORDER BY recorded_at;
    """
    with _conn() as conn:
        df = pd.read_sql_query(query, conn, params={"hours": hours})
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
    return df


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


@st.cache_data(ttl=30)
def query_forecast_map(horizon_minutes: int) -> pd.DataFrame:
    query = """
        SELECT p.lot AS location_name, p.predicted_occupancy AS occupancy,
               c.latitude, c.longitude
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


@st.cache_data(ttl=30)
def query_latest_forecast(lot: str) -> pd.DataFrame:
    query = """
        SELECT predicted_at, horizon_minutes, predicted_occupancy
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
    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["end_time"] = df["end_time"].fillna(df["start_time"] + pd.Timedelta(hours=2))
    return df


def load_model_summary() -> dict:
    path = MODEL_DIR / "summary.json"
    if not path.exists():
        return {"trained_at": None, "lots": {}}
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


def occupancy_to_hex(occupancy: float) -> str:
    """Map an occupancy percentage (0–100) to a hex color on a green→yellow→red scale.

    st.map's `color` argument only accepts hex colors (not numeric columns),
    so we convert the occupancy value to a color ourselves.
    """
    pct = min(max(float(occupancy), 0.0), 100.0)
    if pct <= 50:
        t = pct / 50.0
        r = round(22 + (250 - 22) * t)
        g = round(163 + (204 - 163) * t)
        b = round(74 + (21 - 74) * t)
    else:
        t = (pct - 50.0) / 50.0
        r = round(250 + (204 - 250) * t)
        g = round(204 + (0 - 204) * t)
        b = round(21 + (0 - 21) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🅿️ NCSU Parking")
st.sidebar.caption("Parking occupancy and forecasts")

view = st.sidebar.radio(
    "Navigate",
    ["Overview", "Forecast", "Map", "History", "Patterns", "Accuracy", "Training"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Auto-refreshes every 30 seconds.")
st_autorefresh(interval=30_000, key="dashboard_auto_refresh")


# ── Overview ──────────────────────────────────────────────────────────────────

if view == "Overview":
    st.title("Overview")
    st.caption("How full campus is, plus a quick system check.")

    info = query_overview()

    cols = st.columns(4)
    cols[0].metric("Snapshots stored", f"{info['snapshots']:,}")
    cols[1].metric("Lots tracked", len(query_lots()))
    cols[2].metric("Predictions stored", f"{info['predictions']:,}")
    cols[3].metric(
        "Last update",
        info["latest"].strftime("%H:%M:%S") if info["latest"] else "—",
    )

    st.markdown("---")

    recent = query_recent_occupancy(24)
    if recent.empty:
        st.info("No data yet — the collector is still warming up.")
    else:
        # Campus-wide occupancy over time.  Snapshots are stored delta-only
        # with per-lot microsecond timestamps, so grouping raw rows by
        # recorded_at would average an arbitrary 1-lot subset per instant.
        # Instead: resample each lot to 5-minute buckets (last observed
        # value), then take the capacity-weighted share of all spaces in use.
        per_lot = (
            recent.set_index("recorded_at")
            .groupby("location_name")[["used_spaces", "total_spaces"]]
            .resample("5min")
            .last()
            .ffill()   # carry each lot's last-known state across empty buckets
            .dropna()  # only drops pre-first-observation buckets
            .reset_index()
        )
        agg = per_lot.groupby("recorded_at").agg(
            used=("used_spaces", "sum"),
            total=("total_spaces", "sum"),
        )
        campus = (agg["used"] / agg["total"].replace(0, np.nan) * 100)
        campus = campus.rename("occupancy").reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=campus["recorded_at"], y=campus["occupancy"],
            mode="lines",            name="In use",
            line=dict(color=NCSU_RED, width=2),
            fill="tozeroy", fillcolor="rgba(204,0,0,0.08)",
        ))
        fig.update_layout(
            title="How full is campus? (last 24h)",
            xaxis_title=None, yaxis_title="Occupancy (%)",
            height=380, template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-lot latest occupancy bars
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
            title="How full each lot is right now",
            xaxis_title="Occupancy (%)", yaxis_title=None,
            height=max(300, 28 * len(latest_per_lot)),
            template="plotly_white", margin=dict(l=10, r=20, t=50, b=40),
            xaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Forecast ──────────────────────────────────────────────────────────────────

elif view == "Forecast":
    st.title("24-Hour Forecast")
    st.caption("Predicted occupancy for the next 24 hours.")

    lots = query_lots()
    if not lots:
        st.info("No parking data yet.")
    else:
        lot = st.selectbox("Select lot", lots)
        fc = query_latest_forecast(lot)

        if fc.empty:
            st.warning("No forecast yet. Models need about 48 hours of data before they can train.")
        else:
            now = datetime.now(timezone.utc)
            # Current actual occupancy
            hist = query_history(lot, now - timedelta(minutes=30), now)
            current_occ = hist["occupancy"].iloc[-1] if not hist.empty else None

            c1, c2, c3, c4 = st.columns(4)
            if current_occ is not None:
                c1.metric("Right now", f"{current_occ:.0f}%")
            c2.metric("Peak (24h)", f"{fc['predicted_occupancy'].max():.0f}%")
            c3.metric("Low (24h)", f"{fc['predicted_occupancy'].min():.0f}%")
            c4.metric("In 24h", f"{fc['predicted_occupancy'].iloc[-1]:.0f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fc["forecast_time"], y=fc["predicted_occupancy"],
                mode="lines", name="Predicted",
                line=dict(color=NCSU_RED, width=2),
                fill="tozeroy", fillcolor="rgba(204,0,0,0.08)",
            ))
            if current_occ is not None:
                fig.add_hline(
                    y=current_occ, line_dash="dot", line_color="#6b7280",
                    annotation_text="current", annotation_position="top left",
                )
            fig.update_layout(
                title=f"{lot} — next 24 hours",
                xaxis_title=None, yaxis_title="Predicted occupancy (%)",
                height=420, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=40),
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("See the full forecast"):
                show = fc[["forecast_time", "horizon_minutes", "predicted_occupancy"]].copy()
                show.columns = ["Time", "Minutes ahead", "Predicted %"]
                st.dataframe(show, use_container_width=True, height=360)


# ── Map ──────────────────────────────────────────────────────────────────────

elif view == "Map":
    st.title("Campus Map")
    st.caption("Each lot is colored by how full it is. Red = full.")

    mode = st.radio("Mode", ["Current occupancy", "Predicted occupancy"], horizontal=True)

    if mode == "Current occupancy":
        mdf = query_current_occupancy_map()
        horizon_label = "now"
    else:
        horizon = st.select_slider(
            "Forecast horizon",
            options=list(EVAL_HORIZONS),
            value=60,
            format_func=lambda m: f"{m} min" if m < 60 else f"{m//60} h",
        )
        mdf = query_forecast_map(horizon)
        horizon_label = f"+{horizon} min"

    if mdf.empty:
        st.info("No map data yet — still warming up.")
    else:
        st.caption(f"Colors show occupancy at {horizon_label}. Red = full.")
        # st.map's `color` accepts only hex colors, so map the numeric
        # occupancy % to a green→yellow→red gradient first.
        mdf = mdf.copy()
        mdf["color"] = mdf["occupancy"].apply(occupancy_to_hex)
        st.map(
            mdf[['latitude', 'longitude', 'color']],
            latitude='latitude', longitude='longitude',
            color='color', zoom=14, use_container_width=True,
        )
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#6b7280;">'
            '<span>Empty</span>'
            '<span style="width:140px;height:10px;border-radius:5px;'
            'background:linear-gradient(to right,#16a34a,#facc15,#cc0000);"></span>'
            '<span>Full</span></div>',
            unsafe_allow_html=True,
        )
        if "used_spaces" in mdf.columns:
            show_cols = ["location_name", "occupancy", "used_spaces", "total_spaces"]
            rename = {"location_name": "Lot", "occupancy": "Occupancy (%)",
                      "used_spaces": "Used", "total_spaces": "Total"}
        else:
            show_cols = ["location_name", "occupancy"]
            rename = {"location_name": "Lot", "occupancy": "Occupancy (%)"}
        show = (
            mdf[show_cols]
            .sort_values("occupancy", ascending=False)
            .rename(columns=rename)
        )
        st.dataframe(show, use_container_width=True, hide_index=True)


# ── History ───────────────────────────────────────────────────────────────────

elif view == "History":
    st.title("History")
    st.caption("Occupancy over time, with nearby events highlighted.")

    lots = query_lots()
    if not lots:
        st.info("No parking data yet.")
    else:
        lot = st.selectbox("Select lot", lots, key="hist_lot")

        c1, c2 = st.columns(2)
        start = c1.date_input("Start date", datetime.now(timezone.utc).date() - timedelta(days=3))
        end = c2.date_input("End date", datetime.now(timezone.utc).date() + timedelta(days=1))

        start_ts = pd.Timestamp(start).tz_localize("UTC")
        end_ts = pd.Timestamp(end).tz_localize("UTC")

        hist = query_history(lot, start_ts.to_pydatetime(), end_ts.to_pydatetime())

        if hist.empty:
            st.info("No data in the selected range.")
        else:
            events = query_events_in_range(start_ts.to_pydatetime(), end_ts.to_pydatetime())
            nearby = nearby_events(lot, events)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["recorded_at"], y=hist["occupancy"],
                mode="lines", name="Occupancy",
                line=dict(color=NCSU_RED, width=1.5),
            ))

            # Shade nearby events
            for _, ev in nearby.iterrows():
                fig.add_vrect(
                    x0=ev["start_time"], x1=ev["end_time"],
                    fillcolor="rgba(16,185,129,0.15)", line_width=0,
                    annotation_text="", 
                )

            fig.update_layout(
                title=f"{lot} — occupancy ({len(nearby)} nearby events)",
                xaxis_title=None, yaxis_title="Occupancy (%)",
                height=440, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=40),
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig, use_container_width=True)

            if not nearby.empty:
                with st.expander(f"Nearby events ({len(nearby)})"):
                    ev_show = nearby[["title", "start_time", "end_time"]].copy()
                    ev_show.columns = ["Event", "Start", "End"]
                    st.dataframe(ev_show, use_container_width=True)


# ── Patterns ──────────────────────────────────────────────────────────────────

elif view == "Patterns":
    st.title("Patterns")
    st.caption("Average occupancy by day and hour.")

    lots = query_lots()
    if not lots:
        st.info("No parking data yet.")
    else:
        lot = st.selectbox("Select lot", lots, key="pattern_lot")

        start_ts = datetime.now(timezone.utc) - timedelta(days=21)
        hist = query_history(lot, start_ts, datetime.now(timezone.utc))

        if hist.empty:
            st.info("No data available for this lot.")
        else:
            h = hist.copy()
            h["day"] = h["recorded_at"].dt.dayofweek  # 0=Mon
            h["hour"] = h["recorded_at"].dt.hour
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
                title=f"{lot} — average occupancy (last 21 days)",
                xaxis_title="Hour of day", yaxis_title=None,
                height=380, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Accuracy ─────────────────────────────────────────────────────────────────

elif view == "Accuracy":
    st.title("Forecast Accuracy")
    st.caption("How close recent forecasts were to what actually happened.")

    # Accuracy trend over time (from periodic snapshots recorded by the collector)
    history = query_accuracy_history()
    if not history.empty:
        fig_hist = go.Figure(go.Scatter(
            x=history["recorded_at"], y=history["overall_mae"],
            mode="lines+markers",
            line=dict(color=NCSU_RED, width=2),
            marker=dict(size=6),
        ))
        fig_hist.update_layout(
            title="Overall error over time",
            xaxis_title=None, yaxis_title="Error (%)",
            height=240, template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    hours_back = st.slider("Show forecasts from the last", 6, 72, 24, step=6)

    mae = compute_accuracy(hours_back=hours_back)
    if mae.empty:
        st.info("Nothing to evaluate yet. Forecasts can only be checked once "
                "the predicted time has passed.")
    else:
        overall = float((mae["mae"] * mae["n"]).sum() / mae["n"].sum())
        st.metric(f"Overall error (last {hours_back}h)", f"{overall:.2f}%")

        # MAE vs horizon, averaged across lots
        avg = mae.groupby("horizon_minutes")["mae"].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=avg["horizon_minutes"], y=avg["mae"],
            mode="lines+markers",
            line=dict(color=NCSU_RED, width=2),
            marker=dict(size=7),
        ))
        fig.update_layout(
            title="Error by forecast horizon (all lots)",
            xaxis_title="Minutes ahead", yaxis_title="Error (%)",
            height=320, template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-lot MAE table
        pivot = mae.pivot_table(index="lot", columns="horizon_minutes", values="mae")
        pivot = pivot.round(2)
        pivot.columns = [f"{int(c)} min" for c in pivot.columns]
        st.subheader("Error by lot and horizon")
        st.dataframe(pivot, use_container_width=True)


# ── Training ──────────────────────────────────────────────────────────────────

elif view == "Training":
    st.title("Model Training")
    st.caption("Retrain the models and see how each lot performs.")

    state = get_train_state()

    if state["running"]:
        st.warning(
            f"Training is running (started `{state['started_at']}`).\n\n"
            "You can keep browsing — training continues in the background."
        )
    else:
        if state["done"]:
            if state["error"]:
                st.error(f"Training failed: {state['error']}")
            else:
                st.success("Training finished successfully.")

        if st.button(
            "🔄 Retrain models", type="primary",
            help="Re-trains every lot's model. Can take a while.",
        ):
            start_training_async()
            st.rerun()

    summary = load_model_summary()

    if summary["trained_at"]:
        st.markdown(f"**Last trained:** `{summary['trained_at']}`")
    else:
        st.info("No models trained yet.")

    st.markdown("---")
    st.subheader("Per-lot performance")

    if not summary["lots"]:
        st.info("No trained models to show.")
    else:
        rows = []
        for lot, meta in summary["lots"].items():
            mae = meta.get("val_mae_by_horizon", {})
            rows.append({
                "Lot": lot,
                "Training rows": meta.get("train_rows"),
                "Validation rows": meta.get("val_rows"),
                "Avg error": round(meta.get("val_mae_mean", float("nan")), 2),
                "1 min": round(mae.get("target_1min", float("nan")), 2),
                "15 min": round(mae.get("target_15min", float("nan")), 2),
                "1 h": round(mae.get("target_60min", float("nan")), 2),
                "6 h": round(mae.get("target_360min", float("nan")), 2),
                "24 h": round(mae.get("target_1440min", float("nan")), 2),
            })
        mdf = pd.DataFrame(rows)
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        # MAE by horizon chart (first lot only, to show the curve shape)
        first_lot = next(iter(summary["lots"]))
        mae = summary["lots"][first_lot]["val_mae_by_horizon"]
        mins = [int(k.replace("target_", "").replace("min", "")) for k in mae]
        vals = list(mae.values())
        fig = go.Figure(go.Scatter(
            x=mins, y=vals, mode="lines",
            line=dict(color=NCSU_RED, width=2),
        ))
        fig.update_layout(
            title=f"Error by forecast horizon — {first_lot}",
            xaxis_title="Minutes ahead", yaxis_title="Error (%)",
            height=320, template="plotly_white",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
