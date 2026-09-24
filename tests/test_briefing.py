# tests/test_briefing.py — unit tests for briefing.py
#
# Run: python3 -m pytest -q
# No network access: the Anthropic client is faked via sys.modules injection.

import sys
import types

import pytest

from briefing import (
    generate_briefing,
    build_briefing_inputs,
    _numbers_are_grounded,
    _sentence_count,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
)


SAMPLE_INPUTS = {
    "health_score": 62.5,
    "health_band": "Moderate",
    "sahm_gap_pp": 0.10,
    "UNRATE_current": 4.1,
    "UNRATE_mom_change_pp": 0.1,
    "CPIAUCSL_yoy_pct": 2.9,
}


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self._payload = payload
        self._stop_reason = stop_reason

    def create(self, **kwargs):
        if isinstance(self._payload, Exception):
            raise self._payload
        return _FakeResponse(self._payload, self._stop_reason)


class _FakeAnthropic:
    """Records the kwargs it was constructed with, so H2 (timeout/retries)
    can be asserted on."""

    last_kwargs = None

    def __init__(self, payload, stop_reason="end_turn"):
        self._payload = payload
        self._stop_reason = stop_reason

    def __call__(self, api_key=None, timeout=None, max_retries=None):
        _FakeAnthropic.last_kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        client = types.SimpleNamespace()
        client.messages = _FakeMessages(self._payload, self._stop_reason)
        return client


def _install_fake_anthropic(monkeypatch, payload, stop_reason="end_turn"):
    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = _FakeAnthropic(payload, stop_reason)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


def test_disabled_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = generate_briefing(SAMPLE_INPUTS, api_key=None)
    assert result["source"] == "disabled"
    assert "ANTHROPIC_API_KEY" in result["text"]


def test_grounded_output_is_accepted(monkeypatch):
    good_text = (
        "The Economic Health Score sits at 62.5, in Moderate territory. "
        "Unemployment is 4.1%, up 0.1 percentage points month over month. "
        "Inflation (CPI) is running at 2.9% year over year, with the Sahm gap at 0.10pp."
    )
    _install_fake_anthropic(monkeypatch, good_text)
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert result["source"] == "ai"
    assert result["text"] == good_text


def test_output_with_invented_number_is_rejected(monkeypatch):
    bad_text = (
        "The Economic Health Score sits at 62.5, in Moderate territory. "
        "Unemployment is 4.1%, but GDP grew a surprising 9.9% this quarter. "
        "The outlook remains mixed."
    )
    _install_fake_anthropic(monkeypatch, bad_text)
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert result["source"] == "fallback"
    assert "62.5" in result["text"]
    assert result["reason"] is not None


def test_api_failure_falls_back_without_crashing(monkeypatch):
    _install_fake_anthropic(monkeypatch, RuntimeError("network unreachable: request id req_abc123"))
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert result["source"] == "fallback"
    assert "62.5" in result["text"]


def test_api_failure_reason_never_leaks_raw_exception_text(monkeypatch):
    """H2/L5: the raw exception (which can contain SDK internals like
    request ids) must never reach the user-facing reason string."""
    _install_fake_anthropic(monkeypatch, RuntimeError("network unreachable: request id req_abc123"))
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert "req_abc123" not in result["reason"]
    assert "req_abc123" not in result["text"]
    assert "AI briefing unavailable" in result["reason"]


def test_client_uses_short_timeout_and_single_retry(monkeypatch):
    """H2: a hung Claude call must not block the page for the SDK's default
    ~600s read timeout / 2 retries."""
    _install_fake_anthropic(monkeypatch, "irrelevant, not exercised for kwargs check")
    generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert _FakeAnthropic.last_kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS == 15.0
    assert _FakeAnthropic.last_kwargs["max_retries"] == MAX_RETRIES == 1


def test_max_tokens_truncation_falls_back(monkeypatch):
    """M2: output truncated before finishing must not be accepted as-is."""
    truncated = "The Economic Health Score sits at 62.5, in Moderate territory. Unemployment is 4.1"
    _install_fake_anthropic(monkeypatch, truncated, stop_reason="max_tokens")
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert result["source"] == "fallback"
    assert "62.5" in result["text"]


def test_wrong_sentence_count_falls_back(monkeypatch):
    """M2: exactly 3 sentences is enforced, not just claimed."""
    two_sentences = (
        "The Economic Health Score sits at 62.5, in Moderate territory. "
        "Unemployment is 4.1%."
    )
    _install_fake_anthropic(monkeypatch, two_sentences)
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert result["source"] == "fallback"


def test_sentence_count_ignores_decimal_points():
    text = "The rate is 4.3%. It rose from 4.1% last month. Inflation held at 2.9%."
    assert _sentence_count(text) == 3


# ─────────────────────────────────────────────
# Grounding guardrail probes (H4/H5)
# ─────────────────────────────────────────────

def test_numbers_are_grounded_allows_rounding_tolerance():
    inputs = {"a": 4.10}
    assert _numbers_are_grounded("The rate is 4.1%.", inputs)
    assert not _numbers_are_grounded("The rate is 9.9%.", inputs)


def test_thousands_separator_is_not_split_into_two_numbers():
    inputs = {"PAYEMS_current": 159540.0}
    assert _numbers_are_grounded("Payrolls stand at 159,540 thousand.", inputs)


def test_negative_change_phrased_as_fell_is_grounded():
    inputs = {"UNRATE_mom_change_pp": -0.2}
    assert _numbers_are_grounded("Unemployment fell 0.2pp month over month.", inputs)


def test_year_is_allowlisted_not_required_as_a_data_point():
    inputs = {"health_score": 62.5}
    assert _numbers_are_grounded("As of September 2026, the score is 62.5.", inputs)


def test_range_is_parsed_as_two_positive_numbers():
    inputs = {"a": 4.3, "b": 4.4}
    assert _numbers_are_grounded("Forecasts range 4.3-4.4% next quarter.", inputs)
    # A genuinely fabricated range must still fail.
    assert not _numbers_are_grounded("Forecasts range 7.1-7.2% next quarter.", inputs)


def test_leading_dot_decimal_is_parsed_correctly():
    inputs = {"sahm_gap_pp": 0.10}
    assert _numbers_are_grounded("The Sahm gap is .10pp.", inputs)
    assert not _numbers_are_grounded("The Sahm gap is .10pp.", {"sahm_gap_pp": 10.0})


def test_structural_month_and_quarter_counts_are_allowlisted():
    inputs = {"health_score": 62.5}
    assert _numbers_are_grounded("Over the past 12-month period across 3 quarters, the score is 62.5.", inputs)


def test_spelled_out_number_words_are_rejected():
    inputs = {"health_score": 62.5, "UNRATE_yoy_change_pp": 5.0}
    assert not _numbers_are_grounded("Inflation is running at roughly five percent.", inputs)
    assert not _numbers_are_grounded("The score sits at sixty two point five.", inputs)


def test_number_tied_to_wrong_indicator_is_rejected():
    """H5 repro: 'GDP fell 0.1%' must not pass just because 0.1 happens to
    equal an unrelated indicator's value (the Sahm gap)."""
    inputs = {"sahm_gap_pp": 0.10, "GDPC1_qoq_pct": 2.4}
    assert not _numbers_are_grounded("GDP fell 0.1% this quarter.", inputs)
    assert _numbers_are_grounded("GDP grew 2.4% this quarter.", inputs)


# ─────────────────────────────────────────────
# Input labelling (M1)
# ─────────────────────────────────────────────

def test_build_briefing_inputs_flattens_summary_stats():
    summaries = {
        "UNRATE": {"current": 4.1, "mom_change": 0.1, "period_change_pct": 2.5, "yoy_pct": None},
    }
    score = {"score": 62.5, "band": "Moderate"}
    sahm = {"gap": 0.10, "as_of": "Aug 2026"}
    inputs = build_briefing_inputs(summaries, score, sahm)
    assert inputs["UNRATE_current"] == 4.1
    assert inputs["UNRATE_mom_change_pp"] == 0.1
    assert "UNRATE_yoy_pct" not in inputs
    assert "UNRATE_yoy_change_pp" not in inputs
    assert inputs["health_score"] == 62.5
    assert inputs["sahm_gap_pp"] == 0.10


def test_cpi_yoy_is_labelled_as_percent_inflation_not_index_points():
    """M1 repro: CPI's YoY figure must be percent inflation, not a raw
    index-point difference mislabelled as a percent."""
    summaries = {
        "CPIAUCSL": {"current": 310.0, "mom_change": 0.5, "period_change_pct": 0.16, "yoy_pct": 2.9},
    }
    score = {"score": 62.5, "band": "Moderate"}
    sahm = {"gap": None, "as_of": None}
    inputs = build_briefing_inputs(summaries, score, sahm)
    assert inputs["CPIAUCSL_yoy_pct"] == 2.9
    assert "CPIAUCSL_yoy_change" not in inputs


def test_gdp_period_change_is_labelled_qoq_not_mom():
    """M1 repro: GDPC1 is quarterly, so its latest-period change is
    quarter-over-quarter, not month-over-month."""
    summaries = {
        "GDPC1": {"current": 23000.0, "mom_change": 190.0, "period_change_pct": 0.83, "yoy_pct": 2.4},
    }
    score = {"score": 62.5, "band": "Moderate"}
    sahm = {"gap": None, "as_of": None}
    inputs = build_briefing_inputs(summaries, score, sahm)
    assert inputs["GDPC1_qoq_pct"] == 0.83
    assert inputs["GDPC1_yoy_pct"] == 2.4
    assert "GDPC1_mom_change" not in inputs
