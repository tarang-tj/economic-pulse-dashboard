# app.py — Economic Pulse Dashboard
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from pipeline import load_all, get_latest
from analysis import get_summary_stats, compute_rolling, correlation_matrix, get_recession_bands
from config import FRED_SERIES, LOOKBACK_YEARS, CATEGORY_ORDER

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Economic Pulse Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .metric-card {
    background: #0f1923;
    border: 1px solid #1e3448;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 12px;
  }
  .metric-label  { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #7a9ab5; margin-bottom: 4px; }
  .metric-value  { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; color: #e8f0fe; }
  .metric-delta  { font-family: 'IBM Plex Mono', monospace; font-size: 13px; margin-top: 4px; }
  .metric-date   { font-size: 11px; color: #4a6a85; margin-top: 2px; }

  .trend-up   { color: #34d399; }
  .trend-down { color: #f87171; }
  .trend-flat { color: #94a3b8; }

  .section-header {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a6a85;
    border-bottom: 1px solid #1e3448;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
  }

  [data-testid="stSidebar"] {
    background: #080f16;
    border-right: 1px solid #1e3448;
  }

  div[data-testid="metric-container"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

PALETTE = {
    "primary":   "#3b82f6",
    "secondary": "#34d399",
    "accent":    "#f59e0b",
    "danger":    "#f87171",
    "muted":     "#334155",
    "recession": "rgba(248, 113, 113, 0.08)",
    "band":      "rgba(59, 130, 246, 0.08)",
    "bg":        "#0b1420",
    "grid":      "#1e3448",
    "text":      "#94a3b8",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["bg"],
    font=dict(family="IBM Plex Sans", color=PALETTE["text"], size=12),
    xaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["grid"], tickfont=dict(size=11)),
    yaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["grid"], tickfont=dict(size=11)),
    margin=dict(l=16, r=16, t=36, b=16),
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.15, x=0, font=dict(size=11)),
)


def _delta_html(val, unit="%", invert=False):
    if val is None:
        return ""
    positive_is_good = not invert if invert is not None else True
    good = (val > 0 and positive_is_good) or (val < 0 and not positive_is_good)
    cls = "trend-up" if good else "trend-down" if val != 0 else "trend-flat"
    arrow = "▲" if val > 0 else "▼" if val < 0 else "—"
    display = f"{arrow} {abs(val):.2f}{unit}"
    return f'<span class="{cls}">{display}</span>'


def _trend_badge(trend: dict, invert: bool = False) -> str:
    d = trend["direction"]
    if invert:
        icon = {"up": "▲↑ Rising", "down": "▼↓ Falling", "flat": "— Stable"}
    else:
        icon = {"up": "▲ Rising", "down": "▼ Falling", "flat": "— Stable"}
    cls = {
        ("up", False): "trend-up", ("down", False): "trend-down",
        ("up", True):  "trend-down", ("down", True): "trend-up",
        ("flat", False): "trend-flat", ("flat", True): "trend-flat",
    }.get((d, bool(invert)), "trend-flat")
    return f'<span class="{cls}" style="font-size:12px">{icon.get(d, d)}</span>'


def add_recession_shading(fig, start_date, end_date):
    for band in get_recession_bands(str(start_date), str(end_date)):
        fig.add_vrect(
            x0=band["start"], x1=band["end"],
            fillcolor=PALETTE["recession"],
            layer="below", line_width=0,
            annotation_text="Recession" if band["start"] == band["end"] else "",
        )


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600 * 12, show_spinner="Fetching economic data from FRED…")
def fetch_data():
    return load_all()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Economic Pulse")
    st.markdown('<p style="color:#4a6a85;font-size:12px;margin-top:-8px">US & WA State Indicators</p>', unsafe_allow_html=True)
    st.divider()

    lookback = st.slider("Years of history", min_value=1, max_value=20, value=LOOKBACK_YEARS, step=1)
    show_recession = st.checkbox("Show recession bands", value=True)
    show_rolling = st.checkbox("Show 3-month rolling mean", value=False)

    st.divider()
    st.markdown('<p class="section-header">Series</p>', unsafe_allow_html=True)
    selected_series = []
    for sid, meta in FRED_SERIES.items():
        if st.checkbox(meta["short"], value=True, key=f"cb_{sid}"):
            selected_series.append(sid)

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        '<p style="color:#2a4a65;font-size:11px;margin-top:8px">Data: Federal Reserve Bank of St. Louis (FRED)</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

try:
    data = fetch_data()
except EnvironmentError as e:
    st.error(f"**API key missing.** {e}")
    st.stop()

if not data:
    st.error("No data loaded. Check your FRED API key and network connection.")
    st.stop()

# Filter to selected series
data = {k: v for k, v in data.items() if k in selected_series}

# Apply lookback window
cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback)
windowed = {sid: df[df.index >= cutoff] for sid, df in data.items()}


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# Economic Pulse Dashboard")
    st.markdown(
        f'<p style="color:#4a6a85;font-size:14px">US & Washington State · FRED data · '
        f'Last {lookback} years · Updated {datetime.now().strftime("%b %d, %Y")}</p>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown("")


# ─────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────

st.markdown('<p class="section-header">Current Readings</p>', unsafe_allow_html=True)

cols = st.columns(len(data))
for i, (sid, df) in enumerate(data.items()):
    meta = FRED_SERIES[sid]
    stats = get_summary_stats(df, lookback_years=lookback)
    unit_sym = "%" if meta["unit"] == "%" else ""
    invert = meta.get("invert_signal", False)

    mom_html = _delta_html(stats["mom_change"], unit=unit_sym or "pp", invert=invert)
    trend_html = _trend_badge(stats["trend"], invert=bool(invert))

    val_display = f"{stats['current']:,.1f}{unit_sym}"
    if meta["unit"] == "Thousands":
        val_display = f"{stats['current'] / 1000:,.1f}M"

    with cols[i]:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{meta['short']}</div>
          <div class="metric-value">{val_display}</div>
          <div class="metric-delta">{mom_html} MoM</div>
          <div class="metric-delta">{trend_html}</div>
          <div class="metric-date">{stats['current_date']}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Charts by category
# ─────────────────────────────────────────────

# Group series by category
from collections import defaultdict
by_category = defaultdict(list)
for sid, df in windowed.items():
    cat = FRED_SERIES[sid]["category"]
    by_category[cat].append(sid)

colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["danger"], "#a78bfa", "#38bdf8"]

for cat in CATEGORY_ORDER:
    sids = by_category.get(cat)
    if not sids:
        continue

    st.markdown(f'<p class="section-header">{cat}</p>', unsafe_allow_html=True)

    if len(sids) == 1:
        chart_cols = [st.container()]
    else:
        chart_cols = st.columns(len(sids))

    for j, sid in enumerate(sids):
        df = windowed[sid]
        meta = FRED_SERIES[sid]
        stats = get_summary_stats(data[sid], lookback_years=lookback)

        fig = go.Figure()

        # Recession bands
        if show_recession and len(df) > 0:
            add_recession_shading(fig, df.index.min(), df.index.max())

        # Rolling mean band fill
        if show_rolling:
            df_r = compute_rolling(df)
            fig.add_trace(go.Scatter(
                x=df_r.index, y=df_r["upper_band"],
                fill=None, mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=df_r.index, y=df_r["lower_band"],
                fill="tonexty", mode="lines",
                fillcolor=PALETTE["band"],
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=df_r.index, y=df_r["rolling_mean"],
                mode="lines", name="3-mo avg",
                line=dict(color=colors[j % len(colors)], width=1.5, dash="dot"),
            ))

        # Main series line
        fig.add_trace(go.Scatter(
            x=df.index, y=df["value"],
            mode="lines", name=meta["short"],
            line=dict(color=colors[j % len(colors)], width=2),
            hovertemplate=f"<b>%{{y:.2f}}</b> {meta['unit']}<extra>{meta['short']}</extra>",
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=meta["name"], font=dict(size=13, color="#c8d8e8"), x=0),
            yaxis_title=meta["unit"],
            height=320,
        )

        if len(sids) == 1:
            chart_cols[0].plotly_chart(fig, use_container_width=True)
        else:
            chart_cols[j].plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# YoY Change Chart
# ─────────────────────────────────────────────

st.markdown('<p class="section-header">Year-over-Year Change</p>', unsafe_allow_html=True)

yoy_frames = []
for sid, df in windowed.items():
    tmp = df[["yoy_pct"]].copy()
    tmp.columns = [FRED_SERIES[sid]["short"]]
    yoy_frames.append(tmp.resample("MS").last())

if yoy_frames:
    yoy_df = pd.concat(yoy_frames, axis=1).dropna(how="all")

    fig_yoy = go.Figure()
    for k, col in enumerate(yoy_df.columns):
        fig_yoy.add_trace(go.Scatter(
            x=yoy_df.index, y=yoy_df[col],
            mode="lines", name=col,
            line=dict(color=colors[k % len(colors)], width=1.5),
        ))
    fig_yoy.add_hline(y=0, line_dash="dash", line_color=PALETTE["muted"], line_width=1)
    fig_yoy.update_layout(**PLOTLY_LAYOUT, title="YoY Change (pp for rates, % for levels)", height=300)
    st.plotly_chart(fig_yoy, use_container_width=True)


# ─────────────────────────────────────────────
# Correlation Heatmap
# ─────────────────────────────────────────────

if len(data) >= 3:
    st.markdown('<p class="section-header">Indicator Correlation (Pearson)</p>', unsafe_allow_html=True)

    corr = correlation_matrix(data, lookback_years=lookback)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale=[[0, "#f87171"], [0.5, "#1e3448"], [1, "#34d399"]],
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig_corr.update_layout(
        **PLOTLY_LAYOUT,
        coloraxis_colorbar=dict(title="r", tickfont=dict(size=10)),
        height=360,
    )
    fig_corr.update_traces(textfont=dict(size=11))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Positive values (green) = indicators move together. Negative (red) = inverse relationship.")


# ─────────────────────────────────────────────
# Data Table
# ─────────────────────────────────────────────

with st.expander("📋 Raw data table"):
    frames = []
    for sid, df in windowed.items():
        tmp = df[["value"]].copy()
        tmp.columns = [FRED_SERIES[sid]["short"]]
        frames.append(tmp.resample("MS").last())

    if frames:
        table = pd.concat(frames, axis=1)
        table.index = table.index.strftime("%Y-%m")
        st.dataframe(table.tail(36).iloc[::-1], use_container_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    '<p style="color:#2a4a65;font-size:11px;text-align:center">'
    'Data sourced from the Federal Reserve Bank of St. Louis (FRED) · '
    'Built with Python, Pandas, Plotly & Streamlit'
    '</p>',
    unsafe_allow_html=True,
)
