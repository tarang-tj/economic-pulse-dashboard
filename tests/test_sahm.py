# tests/test_sahm.py — unit tests for sahm.py
#
# Run: python3 -m pytest -q
# Offline: validates against a trimmed, downloaded-once FRED fixture
# (tests/fixtures/unrate_fred.csv, sahmcurrent_fred.csv), no live network call.

from pathlib import Path

import pandas as pd
import pytest

from sahm import sahm_rule, current_status
from config import SAHM_TRIGGER_THRESHOLD

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture_unrate() -> pd.Series:
    df = pd.read_csv(FIXTURES / "unrate_fred.csv", parse_dates=["observation_date"])
    return df.set_index("observation_date")["UNRATE"]


def _load_fixture_sahm_official() -> pd.Series:
    df = pd.read_csv(FIXTURES / "sahmcurrent_fred.csv", parse_dates=["observation_date"])
    return df.set_index("observation_date")["SAHMCURRENT"]


def test_matches_official_sahmcurrent_within_tolerance():
    """
    Compare our computed gap against FRED's own SAHMCURRENT series on the
    fixture window. SAHMCURRENT uses real-time (as-originally-published)
    unemployment vintages, while our fixture uses the latest revised
    UNRATE, so small drift is expected outside of major data-revision
    periods (e.g. the 2020-2021 pandemic revisions). Assert close agreement
    excluding that known outlier window.
    """
    unrate = _load_fixture_unrate()
    official = _load_fixture_sahm_official()

    result = sahm_rule(unrate)
    combined = pd.concat([result["gap"], official], axis=1, join="inner").dropna()
    combined.columns = ["computed", "official"]

    # Known outlier: 2020-2021 pandemic unemployment revisions.
    stable = combined[~combined.index.to_series().between("2020-01-01", "2021-12-31")]
    diff = (stable["computed"] - stable["official"]).abs()

    assert len(stable) > 100
    assert diff.max() < 0.75
    assert diff.mean() < 0.15


def test_triggers_on_known_2008_recession_window():
    unrate = _load_fixture_unrate()
    result = sahm_rule(unrate)
    window = result.loc["2008-01-01":"2008-12-31"]
    assert window["triggered"].any()


def test_gap_and_triggered_consistent_with_threshold():
    unrate = _load_fixture_unrate()
    result = sahm_rule(unrate)
    valid = result.dropna(subset=["gap"])
    assert (valid.loc[valid["gap"] >= SAHM_TRIGGER_THRESHOLD, "triggered"]).all()
    assert not (valid.loc[valid["gap"] < SAHM_TRIGGER_THRESHOLD, "triggered"]).any()


def test_current_status_returns_latest_reading():
    unrate = _load_fixture_unrate()
    status = current_status(unrate)
    assert status["threshold"] == SAHM_TRIGGER_THRESHOLD
    assert status["as_of"] is not None
    assert isinstance(status["triggered"], bool)


def test_empty_series_raises_value_error():
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError):
        sahm_rule(empty)


def test_non_datetime_index_raises_value_error():
    series = pd.Series([4.0, 4.1, 4.2])
    with pytest.raises(ValueError):
        sahm_rule(series)
