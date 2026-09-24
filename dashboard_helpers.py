# dashboard_helpers.py — pure-function helpers for app.py
#
# Kept separate from app.py (which executes top-level as a Streamlit script
# on import) so they can be unit tested directly.

import html


def briefing_cache_key(inputs: dict) -> tuple:
    """
    A hashable, order-independent cache key for a briefing `inputs` dict.

    Used with @st.cache_data (H1): the dict itself isn't hashable, so it is
    passed to the cached function as an underscore-prefixed arg (excluded
    from Streamlit's hash), while this tuple is the actual cache key —
    changes whenever the underlying data changes, so the AI briefing is
    regenerated per data refresh instead of on every Streamlit rerun.
    """
    return tuple(sorted(inputs.items()))


def safe_briefing_html(text: str) -> str:
    """
    Escape model-generated text before rendering it with
    unsafe_allow_html=True (M4). Guards against:
      - HTML/markdown injection (`html.escape`)
      - Streamlit treating a bare '$' as LaTeX delimiters
    """
    escaped = html.escape(text)
    return escaped.replace("$", "&#36;")
