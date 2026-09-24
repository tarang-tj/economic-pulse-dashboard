# forecast.py — ARIMA(1,1,1) forecasting with confidence intervals
#
# Usage:
#   from forecast import forecast_series
#   result = forecast_series(df["value"], horizon=6)

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

MIN_OBSERVATIONS = 24  # need enough history for ARIMA(1,1,1) to fit reliably
DEFAULT_ORDER = (1, 1, 1)


@dataclass
class ForecastResult:
    """ARIMA forecast output: point forecast plus 80%/95% confidence bands."""

    dates: pd.DatetimeIndex
    point: np.ndarray
    lower_80: np.ndarray
    upper_80: np.ndarray
    lower_95: np.ndarray
    upper_95: np.ndarray
    order: tuple = DEFAULT_ORDER
    aic: float = field(default=float("nan"))

    def to_frame(self) -> pd.DataFrame:
        """Return the forecast as a tidy DataFrame indexed by date."""
        return pd.DataFrame(
            {
                "point": self.point,
                "lower_80": self.lower_80,
                "upper_80": self.upper_80,
                "lower_95": self.lower_95,
                "upper_95": self.upper_95,
            },
            index=self.dates,
        )


def forecast_series(
    series: pd.Series,
    horizon: int = 6,
    order: tuple = DEFAULT_ORDER,
) -> ForecastResult:
    """
    Fit ARIMA(order) on a monthly time series and forecast `horizon` steps ahead.

    Args:
        series:  A pandas Series with a monotonic DatetimeIndex (or a
                 date-like index resamplable to monthly) and no NaNs.
        horizon: Number of future periods to forecast (default 6 months).
        order:   ARIMA (p, d, q) order. Default (1, 1, 1).

    Returns:
        ForecastResult with point forecast and 80%/95% confidence intervals,
        one row per forecast step, plus the fitted model's AIC.

    Raises:
        ValueError: if the series is too short, empty, non-numeric, or has
                    no usable frequency information; or if horizon < 1.
        RuntimeError: if the ARIMA model fails to converge/fit. Never falls
                      back to fabricated numbers — callers must handle this.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    clean = series.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Series has {len(clean)} usable observations; ARIMA{order} "
            f"needs at least {MIN_OBSERVATIONS} to fit reliably."
        )

    if not np.issubdtype(clean.dtype, np.number):
        raise ValueError(f"Series must be numeric, got dtype {clean.dtype}")

    # Ensure a monthly frequency so statsmodels can extrapolate dates.
    monthly = clean.copy()
    if isinstance(monthly.index, pd.DatetimeIndex):
        monthly = monthly.resample("MS").last().dropna()
        if len(monthly) < MIN_OBSERVATIONS:
            raise ValueError(
                f"Series has only {len(monthly)} monthly observations after "
                f"resampling; need at least {MIN_OBSERVATIONS}."
            )
        freq = "MS"
        last_date = monthly.index[-1]
        forecast_dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=freq
        )[1:]
    else:
        # Non-date index (e.g. synthetic test series): forecast on positional index.
        forecast_dates = pd.RangeIndex(
            start=len(monthly), stop=len(monthly) + horizon
        )

    try:
        model = ARIMA(monthly.values, order=order)
        fitted = model.fit()
    except Exception as exc:  # statsmodels raises varied exception types
        raise RuntimeError(
            f"ARIMA{order} failed to fit on {len(monthly)} observations: {exc}"
        ) from exc

    try:
        forecast_obj = fitted.get_forecast(steps=horizon)
        point = np.asarray(forecast_obj.predicted_mean)
        ci_80 = forecast_obj.conf_int(alpha=0.20)
        ci_95 = forecast_obj.conf_int(alpha=0.05)
    except Exception as exc:
        raise RuntimeError(
            f"ARIMA{order} fit succeeded but forecasting failed: {exc}"
        ) from exc

    ci_80 = np.asarray(ci_80)
    ci_95 = np.asarray(ci_95)

    return ForecastResult(
        dates=forecast_dates,
        point=point,
        lower_80=ci_80[:, 0],
        upper_80=ci_80[:, 1],
        lower_95=ci_95[:, 0],
        upper_95=ci_95[:, 1],
        order=order,
        aic=float(fitted.aic),
    )
