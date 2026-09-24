# tests/test_forecast.py — unit tests for forecast.py
#
# Run: python3 -m pytest -q
# No network access or FRED_API_KEY required; all series are synthetic.

import numpy as np
import pandas as pd
import pytest

from forecast import forecast_series, MIN_OBSERVATIONS


def _random_walk_with_drift(n=60, drift=0.3, noise_std=1.0, seed=42):
    """Synthetic monthly series with known structure: random walk + drift."""
    rng = np.random.default_rng(seed)
    steps = drift + rng.normal(0, noise_std, size=n)
    values = 50 + np.cumsum(steps)
    dates = pd.date_range("2019-01-01", periods=n, freq="MS")
    return pd.Series(values, index=dates)


def test_forecast_shape_matches_horizon():
    series = _random_walk_with_drift()
    horizon = 6
    result = forecast_series(series, horizon=horizon)

    assert len(result.dates) == horizon
    assert len(result.point) == horizon
    assert len(result.lower_80) == horizon
    assert len(result.upper_80) == horizon
    assert len(result.lower_95) == horizon
    assert len(result.upper_95) == horizon


def test_forecast_dates_extend_monthly_from_last_observation():
    series = _random_walk_with_drift()
    result = forecast_series(series, horizon=6)

    last_obs_date = series.index[-1]
    assert result.dates[0] == last_obs_date + pd.DateOffset(months=1)
    # Dates are strictly increasing and monthly-spaced.
    diffs = result.dates.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(days=diffs.iloc[0].days)).all() or all(
        d.days in (28, 29, 30, 31) for d in diffs
    )


def test_confidence_intervals_bracket_point_forecast():
    series = _random_walk_with_drift()
    result = forecast_series(series, horizon=6)

    assert (result.lower_80 <= result.point).all()
    assert (result.point <= result.upper_80).all()
    assert (result.lower_95 <= result.point).all()
    assert (result.point <= result.upper_95).all()


def test_intervals_widen_monotonically_with_horizon():
    series = _random_walk_with_drift()
    result = forecast_series(series, horizon=6)

    width_80 = result.upper_80 - result.lower_80
    width_95 = result.upper_95 - result.lower_95

    # ARIMA forecast uncertainty grows (non-decreasing) with horizon.
    assert all(width_80[i] <= width_80[i + 1] + 1e-6 for i in range(len(width_80) - 1))
    assert all(width_95[i] <= width_95[i + 1] + 1e-6 for i in range(len(width_95) - 1))

    # 95% band is always wider than 80% band at each step.
    assert (width_95 >= width_80).all()


def test_aic_and_order_reported():
    series = _random_walk_with_drift()
    result = forecast_series(series, horizon=6)

    assert result.order == (1, 1, 1)
    assert np.isfinite(result.aic)


def test_too_short_series_raises_value_error():
    series = _random_walk_with_drift(n=MIN_OBSERVATIONS - 1)
    with pytest.raises(ValueError, match="observations"):
        forecast_series(series, horizon=6)


def test_empty_series_raises_value_error():
    series = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        forecast_series(series, horizon=6)


def test_invalid_horizon_raises_value_error():
    series = _random_walk_with_drift()
    with pytest.raises(ValueError, match="horizon"):
        forecast_series(series, horizon=0)


def test_series_with_nans_is_dropped_before_length_check():
    series = _random_walk_with_drift(n=MIN_OBSERVATIONS + 5)
    series.iloc[3:8] = np.nan  # still leaves enough clean points
    result = forecast_series(series, horizon=6)
    assert len(result.point) == 6


def test_to_frame_returns_dataframe_with_expected_columns():
    series = _random_walk_with_drift()
    result = forecast_series(series, horizon=6)
    frame = result.to_frame()

    assert list(frame.columns) == ["point", "lower_80", "upper_80", "lower_95", "upper_95"]
    assert len(frame) == 6
