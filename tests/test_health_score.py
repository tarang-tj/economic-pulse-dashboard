# tests/test_health_score.py — unit tests for health_score.py
#
# Run: python3 -m pytest -q
# All series are synthetic; no network access or FRED_API_KEY required.

import numpy as np
import pandas as pd
import pytest

from health_score import health_score
from config import HEALTH_SCORE_WEIGHTS, HEALTH_SCORE_MIN_OBSERVATIONS


def _series_df(values, months=None):
    n = len(values)
    months = months or n
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    return pd.DataFrame({"value": values}, index=dates)


def _full_dataset(unrate_val=4.0, cpi_trend="flat"):
    """Build a synthetic dataset covering all 7 HEALTH_SCORE_WEIGHTS series."""
    n = 130  # > 10 years monthly
    rng = np.random.default_rng(0)
    data = {}
    data["UNRATE"] = _series_df(np.linspace(6.0, unrate_val, n))
    data["WAUR"] = _series_df(np.linspace(6.5, unrate_val, n))
    data["JTSJOL"] = _series_df(np.linspace(5000, 9000, n))
    data["PAYEMS"] = _series_df(np.linspace(140000, 158000, n))
    cpi = np.linspace(230, 310, n) if cpi_trend == "flat" else np.linspace(230, 400, n)
    data["CPIAUCSL"] = _series_df(cpi)
    data["FEDFUNDS"] = _series_df(np.linspace(0.25, 5.25, n))
    data["GDPC1"] = _series_df(np.linspace(18000, 23000, n))
    return data


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


def test_flat_series_does_not_crash_and_scores_neutral_component():
    data = _full_dataset()
    n = len(data["UNRATE"])
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    data["FEDFUNDS"] = pd.DataFrame({"value": np.full(n, 2.0)}, index=dates)
    result = health_score(data)
    assert result["components"]["FEDFUNDS"]["normalized"] == 0.5
