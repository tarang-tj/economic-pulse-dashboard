# sahm.py — Sahm Rule recession early-warning indicator
#
# Usage:
#   from sahm import sahm_rule
#   result = sahm_rule(data["UNRATE"]["value"])

import pandas as pd

from config import SAHM_TRIGGER_THRESHOLD


def sahm_rule(unrate: pd.Series) -> pd.DataFrame:
    """
    Compute the Sahm Rule recession indicator from a monthly unemployment
    rate series, using the standard real-time definition: the three-month
    moving average of the unemployment rate minus the minimum of that
    average over the previous 12 months (excluding the current month, as in
    FRED's SAHMCURRENT), triggered when
    that gap is >= SAHM_TRIGGER_THRESHOLD (0.50pp).

    Args:
        unrate: pandas Series of the national unemployment rate (%), indexed
                by a monthly DatetimeIndex (e.g. data["UNRATE"]["value"]).

    Returns:
        DataFrame indexed like the input, with columns:
          unrate        — raw input value
          three_mo_avg  — 3-month rolling mean of unrate
          twelve_mo_min — min of three_mo_avg over the previous 12 months
          gap           — three_mo_avg - twelve_mo_min, in percentage points
          triggered     — bool, True when gap >= SAHM_TRIGGER_THRESHOLD

        Rows before enough history has accumulated carry NaN (not filled),
        matching the real Sahm Rule's inability to signal until 12+ months
        of data exist.

    Raises:
        ValueError: input is empty or not a monotonic-ish datetime-indexed series.
    """
    if unrate.dropna().empty:
        raise ValueError("unrate series is empty; cannot compute the Sahm Rule.")
    if not isinstance(unrate.index, pd.DatetimeIndex):
        raise ValueError("unrate must have a DatetimeIndex.")

    # Keep calendar alignment: a skipped month (e.g. Oct 2025 is blank in FRED
    # UNRATE) must stay a gap, not shift every later
    # window by one row. With min_periods=2 the 3-month average uses the months
    # that exist, which reproduces FRED's SAHMCURRENT to within 0.004pp.
    clean = unrate.sort_index()
    # Snap any in-month date (e.g. month-end) to month start so asfreq keeps values.
    clean.index = clean.index.to_period("M").to_timestamp()
    clean = clean[~clean.index.duplicated(keep="last")].asfreq("MS")

    three_mo_avg = clean.rolling(window=3, min_periods=2).mean()
    # Previous 12 months only: shift(1) excludes the current month. Including it
    # clamps the gap at >= 0 and misses FRED's official series by up to 2.4pp.
    twelve_mo_min = three_mo_avg.shift(1).rolling(window=12, min_periods=12).min()
    gap = three_mo_avg - twelve_mo_min

    out = pd.DataFrame({
        "unrate": clean,
        "three_mo_avg": three_mo_avg,
        "twelve_mo_min": twelve_mo_min,
        "gap": gap,
    })
    # FRED publishes the gap to 2 decimals and triggers on that; compare the
    # rounded gap so float noise (0.4999999) cannot miss a 0.50 trigger month.
    out["triggered"] = out["gap"].round(2) >= SAHM_TRIGGER_THRESHOLD
    out.loc[out["gap"].isna(), "triggered"] = False
    return out


def current_status(unrate: pd.Series) -> dict:
    """
    Convenience summary of the latest Sahm Rule reading.

    Returns dict with: gap (float pp, may be nan), triggered (bool),
    as_of (str "Mon YYYY"), threshold (float).
    """
    df = sahm_rule(unrate)
    valid = df.dropna(subset=["gap"])
    if valid.empty:
        return {"gap": float("nan"), "triggered": False, "as_of": None, "threshold": SAHM_TRIGGER_THRESHOLD}

    latest = valid.iloc[-1]
    return {
        "gap": round(float(latest["gap"]), 3),
        "triggered": bool(latest["triggered"]),
        "as_of": valid.index[-1].strftime("%b %Y"),
        "threshold": SAHM_TRIGGER_THRESHOLD,
    }
