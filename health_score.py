# health_score.py — Economic Health Score: 0-100 composite gauge
#
# Usage:
#   from health_score import health_score
#   result = health_score(data)   # data = dict[series_id -> DataFrame] from pipeline.load_all()

import pandas as pd

from analysis import infer_periods_per_year
from config import (
    FRED_SERIES,
    HEALTH_SCORE_WEIGHTS,
    HEALTH_SCORE_DIRECTION,
    HEALTH_SCORE_LOOKBACK_YEARS,
    HEALTH_SCORE_MIN_OBSERVATIONS,
    HEALTH_SCORE_BANDS,
    HEALTH_SCORE_RATE_SERIES,
    HEALTH_SCORE_CPI_TARGET_PCT,
    HEALTH_SCORE_STALE_MONTHS_MONTHLY,
    HEALTH_SCORE_STALE_MONTHS_QUARTERLY,
)


def _band_for(score: float) -> str:
    for low, high, label in HEALTH_SCORE_BANDS:
        if low <= score < high:
            return label
    return HEALTH_SCORE_BANDS[-1][2]


def _stale_threshold_months(periods_per_year: int) -> int:
    return HEALTH_SCORE_STALE_MONTHS_MONTHLY if periods_per_year == 12 else HEALTH_SCORE_STALE_MONTHS_QUARTERLY


def _months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _series_for_scoring(sid: str, values: pd.Series, periods_per_year: int) -> pd.Series:
    """
    Return the series that should actually be normalised for this indicator.

    Trending LEVELS (CPI, payrolls, GDP) are converted to YoY % change first
    — CPI as distance from the 2% inflation target (lower distance is
    healthier), payrolls/GDP as raw YoY growth (higher is healthier).
    Already-a-rate indicators (UNRATE, WAUR, FEDFUNDS) are scored on their
    raw level, unchanged.
    """
    if sid not in HEALTH_SCORE_RATE_SERIES:
        return values

    yoy = values.pct_change(periods=periods_per_year) * 100
    if sid == "CPIAUCSL":
        return (yoy - HEALTH_SCORE_CPI_TARGET_PCT).abs()
    return yoy


def health_score(data: dict[str, pd.DataFrame]) -> dict:
    """
    Compute the Economic Health Score: a 0-100 composite synthesising every
    weighted indicator in HEALTH_SCORE_WEIGHTS.

    Method: CPI, payrolls, and GDP are trending levels, so their latest
    value is converted to a YoY rate first (CPI to distance from the 2%
    target, payrolls/GDP to YoY growth) — see `_series_for_scoring`.
    Every indicator's resulting series is then min-max normalised against
    its own trailing HEALTH_SCORE_LOOKBACK_YEARS range, direction-adjusted
    (so 1.0 always means "healthiest observed in the window"), then
    combined using the fixed weights in config.py.

    Args:
        data: dict mapping FRED series_id -> cleaned DataFrame (pipeline.load_all())

    Returns:
        dict with:
          score       — 0-100 float
          band        — health band label (Contraction/Weak/Moderate/Strong)
          components  — dict[series_id] -> {"normalized": float, "weight": float,
                                             "value": float, "as_of": str}
                        `value` is the raw level for level-scored indicators,
                        or the transformed YoY figure (target-distance for
                        CPI, YoY growth % for payrolls/GDP) for rate-scored
                        ones.

    Raises:
        ValueError: a required indicator is missing from `data`; has fewer
                    than HEALTH_SCORE_MIN_OBSERVATIONS observations (after
                    any YoY transform) in the trailing lookback window; has
                    a flat window (zero variance — cannot be normalised);
                    or its latest observation is stale (older than the
                    per-frequency threshold in config.py). Never silently
                    fills, skips, or neutral-scores an indicator — an
                    incomplete or stale score is not reported.
    """
    missing = [sid for sid in HEALTH_SCORE_WEIGHTS if sid not in data]
    if missing:
        raise ValueError(f"Health score requires all indicators; missing: {missing}")

    components = {}
    weighted_sum = 0.0
    now = pd.Timestamp.now()

    for sid, weight in HEALTH_SCORE_WEIGHTS.items():
        df = data[sid]
        values = df["value"].dropna()
        if values.empty:
            raise ValueError(f"{sid}: no observations available.")

        periods_per_year = infer_periods_per_year(values.index)

        latest_date = values.index.max()
        stale_months = _stale_threshold_months(periods_per_year)
        age_months = _months_between(latest_date, now)
        if age_months > stale_months:
            raise ValueError(
                f"{sid}: latest observation is {latest_date.strftime('%b %Y')}, "
                f"more than {stale_months} months old; refusing to score a stale series."
            )

        scored_series = _series_for_scoring(sid, values, periods_per_year)

        cutoff = latest_date - pd.DateOffset(years=HEALTH_SCORE_LOOKBACK_YEARS)
        window = scored_series[scored_series.index >= cutoff].dropna()

        if len(window) < HEALTH_SCORE_MIN_OBSERVATIONS:
            raise ValueError(
                f"{sid}: only {len(window)} observations in the trailing "
                f"{HEALTH_SCORE_LOOKBACK_YEARS}-year window; need at least "
                f"{HEALTH_SCORE_MIN_OBSERVATIONS} to normalise reliably."
            )

        # L3 (noted, not fixed): the 10-year min-max window includes the
        # Apr-2020 pandemic spike (UNRATE ~14.8) until it rolls off around
        # 2030. Until then it compresses UNRATE/WAUR toward the healthy end
        # of the scale relative to a "normal recession" range. A percentile
        # rank or winsorised window would reduce this; out of scope here.
        lo, hi = float(window.min()), float(window.max())
        current = float(window.iloc[-1])

        if hi == lo:
            raise ValueError(
                f"{sid}: flat window (min == max == {lo}) in the trailing "
                f"{HEALTH_SCORE_LOOKBACK_YEARS}-year range; cannot normalise "
                "a series with zero variance."
            )

        normalized = (current - lo) / (hi - lo)
        if HEALTH_SCORE_DIRECTION.get(sid, False):
            normalized = 1.0 - normalized

        weighted_sum += normalized * weight
        components[sid] = {
            "normalized": round(normalized, 4),
            "weight": weight,
            "value": current,
            "as_of": window.index[-1].strftime("%b %Y"),
            "label": FRED_SERIES[sid]["short"],
        }

    score = round(weighted_sum * 100, 1)
    return {
        "score": score,
        "band": _band_for(score),
        "components": components,
    }
