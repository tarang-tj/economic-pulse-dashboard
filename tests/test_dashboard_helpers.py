# tests/test_dashboard_helpers.py — unit tests for dashboard_helpers.py
#
# Run: python3 -m pytest -q
# Pure-function helpers extracted out of app.py so they can be unit tested
# without executing the full Streamlit script (which runs top-level on
# import and requires a live ScriptRunContext for most st.* calls).

from dashboard_helpers import briefing_cache_key, safe_briefing_html


def test_briefing_cache_key_is_hashable_and_order_independent():
    a = {"health_score": 62.5, "UNRATE_current": 4.1}
    b = {"UNRATE_current": 4.1, "health_score": 62.5}
    key_a = briefing_cache_key(a)
    key_b = briefing_cache_key(b)
    hash(key_a)  # must not raise
    assert key_a == key_b


def test_briefing_cache_key_changes_when_inputs_change():
    a = {"health_score": 62.5}
    b = {"health_score": 60.0}
    assert briefing_cache_key(a) != briefing_cache_key(b)


def test_safe_briefing_html_escapes_angle_brackets():
    out = safe_briefing_html("The rate is <b>4.1%</b>")
    assert "<b>" not in out
    assert "&lt;b&gt;" in out


def test_safe_briefing_html_escapes_dollar_signs_to_avoid_latex():
    """M4: Streamlit interprets $...$ as LaTeX; a bare '$' in model output
    must not trigger that."""
    out = safe_briefing_html("Job openings cost employers $500 per posting")
    assert "$" not in out
    assert "&#36;" in out


def test_safe_briefing_html_preserves_plain_text():
    out = safe_briefing_html("Unemployment is 4.1%, up 0.1pp month over month.")
    assert "4.1%" in out
