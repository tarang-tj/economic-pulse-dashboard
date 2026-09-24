# tests/test_health_score.py — unit tests for health_score.py
#
# Run: python3 -m pytest -q
# All series are synthetic; no network access or FRED_API_KEY required.
# Dates are anchored to "now" (not a hardcoded year) so the staleness check
# introduced for H3/L2 never makes these fixtures stale as time passes.

import numpy as np
import pandas as pd
import pytest

from health_score import health_score
from config import HEALTH_SCORE_WEIGHTS, HEALTH_SCORE_MIN_OBSERVATIONS


def _month_start_now():
    return pd.Timestamp.now().normalize().replace(day=1)


def _series_df(values, freq="MS"):
    n = len(values)
    end = _month_start_now()
    dates = pd.date_range(end=end, periods=n, freq=freq)
    return pd.DataFrame({"value": values}, index=dates)


def _u_shaped(high, low, end_val, n):
    """Fall from `high` to `low`, then rise back to `end_val` — so the final
    value is NOT automatically the window's min or max (unlike a monotonic
    ramp), letting `end_val` meaningfully move the normalised score."""
    half = n // 2
    down = np.linspace(high, low, half)
    up = np.linspace(low, end_val, n - half)
    return np.concatenate([down, up])


def _full_dataset(unrate_val=4.0):
    """Build a synthetic dataset covering all 7 HEALTH_SCORE_WEIGHTS series,
    with recent (non-stale) dates ending at the current month."""
    n = 130  # > 10 years monthly
    data = {}
    data["UNRATE"] = _series_df(_u_shaped(8.0, 2.0, unrate_val, n))
    data["WAUR"] = _series_df(_u_shaped(8.5, 2.5, unrate_val, n))
    data["JTSJOL"] = _series_df(np.linspace(5000, 9000, n))
    data["PAYEMS"] = _series_df(np.linspace(140000, 158000, n))
    data["CPIAUCSL"] = _series_df(np.linspace(230, 310, n))
    data["FEDFUNDS"] = _series_df(np.linspace(0.25, 5.25, n))
    data["GDPC1"] = _series_df(np.linspace(18000, 23000, n))
    return data


def _level_series_from_yoy(yoy_path, base_level=100.0, freq="MS"):
    """Construct a level series whose YoY % change follows yoy_path exactly.

    yoy_path[i] is the YoY % change of the observation at index i+lag, where
    lag is 12 for monthly ("MS") or 4 for quarterly ("QS"/"3MS") data.
    """
    lag = 12 if freq == "MS" else 4
    n = lag + len(yoy_path)
    values = np.empty(n)
    values[:lag] = base_level
    for i in range(lag, n):
        y = yoy_path[i - lag]
        values[i] = values[i - lag] * (1 + y / 100)
    end = _month_start_now()
    dates = pd.date_range(end=end, periods=n, freq=freq)
    return pd.DataFrame({"value": values}, index=dates)


def test_score_in_valid_range_and_has_band():
    data = _full_dataset()
    result = health_score(data)
    assert 0 <= result["score"] <= 100
    assert result["band"] in {"Contraction", "Weak", "Moderate", "Strong"}
    assert set(result["components"]) == set(HEALTH_SCORE_WEIGHTS)


def test_low_unemployment_scores_higher_than_high_unemployment():
    good = health_score(_full_dataset(unrate_val=3.0))
    bad = health_score(_full_dataset(unrate_val=6.0))
    assert good["score"] > bad["score"]


def test_weights_sum_to_one():
    assert abs(sum(HEALTH_SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def test_missing_indicator_raises_value_error():
    data = _full_dataset()
    del data["GDPC1"]
    with pytest.raises(ValueError, match="missing"):
        health_score(data)


def test_insufficient_history_raises_value_error():
    data = _full_dataset()
    short = data["UNRATE"].iloc[-(HEALTH_SCORE_MIN_OBSERVATIONS - 1):]
    data["UNRATE"] = short
    with pytest.raises(ValueError, match="observations"):
        health_score(data)


def test_flat_series_raises_value_error_not_neutral_score():
    """H3/L1: a flat window (min == max) must fail loud, per the docstring,
    not silently score 0.5."""
    data = _full_dataset()
    n = len(data["FEDFUNDS"])
    end = _month_start_now()
    dates = pd.date_range(end=end, periods=n, freq="MS")
    data["FEDFUNDS"] = pd.DataFrame({"value": np.full(n, 2.0)}, index=dates)
    with pytest.raises(ValueError, match="flat"):
        health_score(data)


def test_cpi_scores_on_disinflation_not_price_level():
    """H3 repro: inflation falling from 9% to 2% (approaching the 2% target)
    must raise the CPI component meaningfully, not stay pinned near 0.0
    just because the price level itself is still near its historical max."""
    data = _full_dataset()
    # 118 months of YoY inflation declining linearly from 9% to 2% (the
    # target) — current distance-from-target is ~0, the minimum (healthiest)
    # in the window, so the CPI component should be high, not ~0.
    yoy_path = np.linspace(9.0, 2.0, 118)
    data["CPIAUCSL"] = _level_series_from_yoy(yoy_path, base_level=230.0)

    result = health_score(data)
    cpi_component = result["components"]["CPIAUCSL"]["normalized"]
    assert cpi_component > 0.8, f"expected disinflation to score high, got {cpi_component}"


def test_series_not_ending_at_max_does_not_score_one():
    """A YoY growth series that peaks mid-window and pulls back before the
    latest observation must not score a perfect 1.0 — H3's bug pinned every
    trending-level series near 1.0 regardless of the actual latest reading."""
    data = _full_dataset()
    # YoY payroll growth rises then falls back before the final observation.
    rising = np.linspace(0.5, 4.0, 60)
    falling = np.linspace(4.0, 1.5, 58)
    yoy_path = np.concatenate([rising, falling])
    data["PAYEMS"] = _level_series_from_yoy(yoy_path, base_level=140000.0)

    result = health_score(data)
    payems_component = result["components"]["PAYEMS"]["normalized"]
    assert payems_component < 1.0, f"expected < 1.0 since latest isn't the window max, got {payems_component}"


def test_gdp_uses_quarterly_lag_not_monthly():
    """H3: GDPC1 is quarterly; YoY growth must use a 4-period lag, not 12.

    yoy_path is constructed so pct_change(periods=4) reconstructs it exactly.
    A wrong lag (e.g. 12, comparing 3 years apart on quarterly-spaced data)
    would reconstruct a materially different number, since the underlying
    level series does not grow at a constant rate.
    """
    data = _full_dataset()
    yoy_path = np.linspace(1.0, 3.0, 40)  # rising YoY growth, ends at its max
    data["GDPC1"] = _level_series_from_yoy(yoy_path, base_level=18000.0, freq="QS")

    result = health_score(data)
    gdp_value = result["components"]["GDPC1"]["value"]
    assert abs(gdp_value - 3.0) < 1e-6, f"expected the lag-4 YoY growth (~3.0), got {gdp_value}"
    # Latest reading ties the window max, so it should normalise near 1.0.
    assert result["components"]["GDPC1"]["normalized"] > 0.95


def test_stale_series_raises_value_error():
    """L2: a series whose latest observation is far behind 'now' must raise,
    not silently score using dead data."""
    data = _full_dataset()
    stale = data["UNRATE"].iloc[:-6].copy()  # drop the most recent 6 months
    data["UNRATE"] = stale
    with pytest.raises(ValueError, match="stale|months old"):
        health_score(data)
