# analysis.py — Derived metrics, trend signals, and summary statistics
#
# Usage:
#   from analysis import get_summary_stats, detect_trend, compute_rolling

import pandas as pd
import numpy as np
from scipy import stats

from config import FRED_SERIES


# ─────────────────────────────────────────────
# Rolling statistics
# ─────────────────────────────────────────────

def compute_rolling(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add rolling mean and std to a series DataFrame.

    Args:
        df:     Series DataFrame from pipeline.load_series()
        window: Rolling window in months (default 3)

    Returns:
        DataFrame with added columns: rolling_mean, rolling_std, upper_band, lower_band
    """
    out = df.copy()
    out["rolling_mean"] = out["value"].rolling(window=window, min_periods=1).mean()
    out["rolling_std"] = out["value"].rolling(window=window, min_periods=1).std()
    out["upper_band"] = out["rolling_mean"] + 1.5 * out["rolling_std"]
    out["lower_band"] = out["rolling_mean"] - 1.5 * out["rolling_std"]
    return out


# ─────────────────────────────────────────────
# Trend detection
# ─────────────────────────────────────────────

def detect_trend(df: pd.DataFrame, months: int = 6) -> dict:
    """
    Detect short-term trend direction using linear regression over recent N months.

    Args:
        df:     Series DataFrame
        months: Lookback window for trend detection

    Returns:
        Dict with keys:
          direction  — 'up', 'down', or 'flat'
          slope      — monthly change per unit
          r_squared  — regression fit quality (0–1)
          pct_change — total % change over window
    """
    recent = df["value"].dropna().iloc[-months:]
    if len(recent) < 3:
        return {"direction": "flat", "slope": 0, "r_squared": 0, "pct_change": 0}

    x = np.arange(len(recent))
    slope, _, r_value, p_value, _ = stats.linregress(x, recent.values)

    r_sq = r_value ** 2
    pct_change = ((recent.iloc[-1] - recent.iloc[0]) / abs(recent.iloc[0])) * 100

    # Only call it a trend if regression is significant (p < 0.10) and fits well
    if p_value > 0.10 or r_sq < 0.3:
        direction = "flat"
    elif slope > 0:
        direction = "up"
    else:
        direction = "down"

    return {
        "direction": direction,
        "slope": round(slope, 4),
        "r_squared": round(r_sq, 3),
        "pct_change": round(pct_change, 2),
    }


# ─────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────

def get_summary_stats(df: pd.DataFrame, lookback_years: int = 5) -> dict:
    """
    Compute summary statistics for a series over a given lookback window.

    Returns:
        Dict with: current, prev_month, prev_year, mom_change, yoy_change,
                   period_min, period_max, period_mean, trend
    """
    cutoff = df.index.max() - pd.DateOffset(years=lookback_years)
    window = df[df.index >= cutoff]["value"].dropna()

    current = float(window.iloc[-1])
    prev_month = float(window.iloc[-2]) if len(window) >= 2 else None
    prev_year = float(window.iloc[-13]) if len(window) >= 13 else None

    mom = round(current - prev_month, 3) if prev_month is not None else None
    yoy = round(current - prev_year, 3) if prev_year is not None else None

    trend = detect_trend(df, months=6)

    return {
        "current": current,
        "current_date": window.index[-1].strftime("%b %Y"),
        "prev_month": prev_month,
        "prev_year": prev_year,
        "mom_change": mom,
        "yoy_change": yoy,
        "period_min": round(float(window.min()), 3),
        "period_max": round(float(window.max()), 3),
        "period_mean": round(float(window.mean()), 3),
        "trend": trend,
    }


# ─────────────────────────────────────────────
# Cross-series correlation
# ─────────────────────────────────────────────

def correlation_matrix(data: dict[str, pd.DataFrame], lookback_years: int = 5) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation of all series values over a shared window.

    Args:
        data: Dict of series_id → DataFrame (from pipeline.load_all())

    Returns:
        Correlation matrix DataFrame
    """
    frames = {}
    for sid, df in data.items():
        cutoff = df.index.max() - pd.DateOffset(years=lookback_years)
        series = df[df.index >= cutoff]["value"].dropna()
        # Resample to monthly to align frequency
        frames[FRED_SERIES[sid]["short"]] = series.resample("MS").last()

    combined = pd.DataFrame(frames).dropna()
    return combined.corr(method="pearson").round(3)


# ─────────────────────────────────────────────
# Recession shading helper
# ─────────────────────────────────────────────

# NBER recession periods (start, end) for shading charts
RECESSION_BANDS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def get_recession_bands(start: str, end: str) -> list[dict]:
    """
    Filter recession bands that overlap with the chart's date range.

    Returns list of dicts with 'start' and 'end' datetime keys.
    """
    chart_start = pd.Timestamp(start)
    chart_end = pd.Timestamp(end)

    visible = []
    for rs, re in RECESSION_BANDS:
        rs, re = pd.Timestamp(rs), pd.Timestamp(re)
        if re >= chart_start and rs <= chart_end:
            visible.append({
                "start": max(rs, chart_start),
                "end": min(re, chart_end),
            })
    return visible
