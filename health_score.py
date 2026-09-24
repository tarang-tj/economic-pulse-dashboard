# health_score.py — Economic Health Score: 0-100 composite gauge
#
# Usage:
#   from health_score import health_score
#   result = health_score(data)   # data = dict[series_id -> DataFrame] from pipeline.load_all()

import pandas as pd

from config import (
    FRED_SERIES,
    HEALTH_SCORE_WEIGHTS,
    HEALTH_SCORE_DIRECTION,
    HEALTH_SCORE_LOOKBACK_YEARS,
    HEALTH_SCORE_MIN_OBSERVATIONS,
    HEALTH_SCORE_BANDS,
)


def _band_for(score: float) -> str:
    for low, high, label in HEALTH_SCORE_BANDS:
        if low <= score < high:
            return label
    return HEALTH_SCORE_BANDS[-1][2]


def health_score(data: dict[str, pd.DataFrame]) -> dict:
    """
    Compute the Economic Health Score: a 0-100 composite synthesising every
    weighted indicator in HEALTH_SCORE_WEIGHTS.

    Method: each indicator's latest value is min-max normalised against its
    own trailing HEALTH_SCORE_LOOKBACK_YEARS range, direction-adjusted (so
    1.0 always means "healthiest observed in the window"), then combined
    using the fixed weights in config.py.

    Args:
        data: dict mapping FRED series_id -> cleaned DataFrame (pipeline.load_all())

    Returns:
        dict with:
          score       — 0-100 float
          band        — health band label (Contraction/Weak/Moderate/Strong)
          components  — dict[series_id] -> {"normalized": float, "weight": float,
                                             "value": float, "as_of": str}

    Raises:
        ValueError: a required indicator is missing from `data`, or has fewer
                    than HEALTH_SCORE_MIN_OBSERVATIONS observations in the
                    trailing lookback window. Never silently fills or skips
                    an indicator — an incomplete score is not reported.
    """
    missing = [sid for sid in HEALTH_SCORE_WEIGHTS if sid not in data]
    if missing:
        raise ValueError(f"Health score requires all indicators; missing: {missing}")

    components = {}
    weighted_sum = 0.0

    for sid, weight in HEALTH_SCORE_WEIGHTS.items():
        df = data[sid]
        cutoff = df.index.max() - pd.DateOffset(years=HEALTH_SCORE_LOOKBACK_YEARS)
        window = df[df.index >= cutoff]["value"].dropna()

        if len(window) < HEALTH_SCORE_MIN_OBSERVATIONS:
            raise ValueError(
                f"{sid}: only {len(window)} observations in the trailing "
                f"{HEALTH_SCORE_LOOKBACK_YEARS}-year window; need at least "
                f"{HEALTH_SCORE_MIN_OBSERVATIONS} to normalise reliably."
            )

        lo, hi = float(window.min()), float(window.max())
        current = float(window.iloc[-1])

        if hi == lo:
            normalized = 0.5  # flat history — no signal either way
        else:
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
