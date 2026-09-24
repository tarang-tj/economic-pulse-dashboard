# briefing.py — Optional AI-generated macro briefing via the Claude API
#
# Usage:
#   from briefing import generate_briefing
#   result = generate_briefing(inputs)   # see build_briefing_inputs()
#
# The model never does the math: every number it writes must already appear
# in `inputs` (rounding tolerance applies). If it invents a number, the
# briefing is rejected and a deterministic fallback summary is shown instead.

import os
import re

MODEL_ID = "claude-haiku-4-5"
MAX_TOKENS = 300
NUMBER_TOLERANCE = 0.05  # allow rounding drift, e.g. 4.05 vs 4.1

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def build_briefing_inputs(summaries: dict, score: dict, sahm: dict) -> dict:
    """
    Assemble the deterministic numbers the briefing is allowed to cite.

    Args:
        summaries: dict[series_id] -> analysis.get_summary_stats() output
        score:     health_score.health_score() output
        sahm:      sahm.current_status() output

    Returns:
        Flat dict of label -> numeric value, used both to build the prompt
        and to validate the model's output.
    """
    inputs = {
        "health_score": score["score"],
        "health_band": score["band"],
    }
    if sahm.get("as_of") is not None:
        inputs["sahm_gap_pp"] = sahm["gap"]
    for sid, s in summaries.items():
        inputs[f"{sid}_current"] = s["current"]
        if s.get("mom_change") is not None:
            inputs[f"{sid}_mom_change"] = s["mom_change"]
        if s.get("yoy_change") is not None:
            inputs[f"{sid}_yoy_change"] = s["yoy_change"]
    return inputs


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def _numbers_are_grounded(text: str, inputs: dict) -> bool:
    """True only if every number in `text` matches some value in `inputs`."""
    allowed = [v for v in inputs.values() if isinstance(v, (int, float))]
    for n in _extract_numbers(text):
        if not any(abs(n - a) <= NUMBER_TOLERANCE for a in allowed):
            return False
    return True


def _fallback_summary(inputs: dict) -> str:
    band = inputs.get("health_band", "Unknown")
    score = inputs.get("health_score", float("nan"))
    sahm_gap = inputs.get("sahm_gap_pp")
    sahm_line = (
        f"The Sahm Rule gap is {sahm_gap:.2f}pp against a 0.50pp trigger."
        if sahm_gap is not None else "Sahm Rule data is unavailable."
    )
    return (
        f"Economic Health Score: {score:.1f}/100 ({band}). {sahm_line} "
        f"(Deterministic summary: AI briefing unavailable or failed validation.)"
    )


def generate_briefing(inputs: dict, api_key: str | None = None) -> dict:
    """
    Generate a 3-sentence plain-English macro briefing citing only the
    numbers in `inputs`.

    Args:
        inputs:  output of build_briefing_inputs()
        api_key: Anthropic API key; falls back to ANTHROPIC_API_KEY env var
                 (callers may also pass st.secrets values here).

    Returns:
        dict with:
          text      — the briefing (AI-generated or deterministic fallback)
          source    — "ai" | "fallback" | "disabled"
          reason    — human-readable note (why fallback/disabled)
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return {
            "text": "AI briefing disabled: set ANTHROPIC_API_KEY",
            "source": "disabled",
            "reason": "No Anthropic API key configured.",
        }

    try:
        from anthropic import Anthropic
    except ImportError:
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": "anthropic package not installed.",
        }

    numbers_line = ", ".join(f"{k}={v}" for k, v in inputs.items())
    prompt = (
        "You are writing a macro briefing for an economic dashboard. "
        "Using ONLY the numbers below (do not calculate or invent any other "
        "number), write exactly 3 sentences in plain English summarising the "
        "US economy's current state. Cite the actual figures.\n\n"
        f"Data: {numbers_line}"
    )

    try:
        client = Anthropic(api_key=key)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    except Exception as exc:  # network/auth/rate-limit errors — never crash the app
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": f"Claude API call failed: {exc}",
        }

    if not text or not _numbers_are_grounded(text, inputs):
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": "Model output contained a number not present in the source data.",
        }

    return {"text": text, "source": "ai", "reason": None}
