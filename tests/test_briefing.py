# tests/test_briefing.py — unit tests for briefing.py
#
# Run: python3 -m pytest -q
# No network access: the Anthropic client is faked via sys.modules injection.

import sys
import types

import pytest

from briefing import generate_briefing, build_briefing_inputs, _numbers_are_grounded


SAMPLE_INPUTS = {
    "health_score": 62.5,
    "health_band": "Moderate",
    "sahm_gap_pp": 0.10,
    "UNRATE_current": 4.1,
    "UNRATE_mom_change": 0.1,
    "CPIAUCSL_yoy_change": 2.9,
}


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text_or_exc):
        self._payload = text_or_exc

    def create(self, **kwargs):
        if isinstance(self._payload, Exception):
            raise self._payload
        return _FakeResponse(self._payload)


class _FakeAnthropic:
    def __init__(self, text_or_exc):
        self._payload = text_or_exc

    def __call__(self, api_key=None):
        client = types.SimpleNamespace()
        client.messages = _FakeMessages(self._payload)
        return client


def _install_fake_anthropic(monkeypatch, text_or_exc):
    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = _FakeAnthropic(text_or_exc)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


def test_disabled_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = generate_briefing(SAMPLE_INPUTS, api_key=None)
    assert result["source"] == "disabled"
    assert "ANTHROPIC_API_KEY" in result["text"]


def test_grounded_output_is_accepted(monkeypatch):
    good_text = (
        "The Economic Health Score sits at 62.5, in Moderate territory. "
        "Unemployment is 4.1%, up 0.1pp month over month. "
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
    _install_fake_anthropic(monkeypatch, RuntimeError("network unreachable"))
    result = generate_briefing(SAMPLE_INPUTS, api_key="fake-key")
    assert result["source"] == "fallback"
    assert "62.5" in result["text"]


def test_numbers_are_grounded_allows_rounding_tolerance():
    inputs = {"a": 4.10}
    assert _numbers_are_grounded("The rate is 4.1%.", inputs)
    assert not _numbers_are_grounded("The rate is 9.9%.", inputs)


def test_build_briefing_inputs_flattens_summary_stats():
    summaries = {
        "UNRATE": {"current": 4.1, "mom_change": 0.1, "yoy_change": None},
    }
    score = {"score": 62.5, "band": "Moderate"}
    sahm = {"gap": 0.10, "as_of": "Aug 2026"}
    inputs = build_briefing_inputs(summaries, score, sahm)
    assert inputs["UNRATE_current"] == 4.1
    assert inputs["UNRATE_mom_change"] == 0.1
    assert "UNRATE_yoy_change" not in inputs
    assert inputs["health_score"] == 62.5
    assert inputs["sahm_gap_pp"] == 0.10
