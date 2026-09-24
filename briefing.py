# briefing.py — Optional AI-generated macro briefing via the Claude API
#
# Usage:
#   from briefing import generate_briefing
#   result = generate_briefing(inputs)   # see build_briefing_inputs()
#
# The model never does the math: every number it writes must already appear
# in `inputs` (rounding tolerance applies). If it invents a number, the
# briefing is rejected and a deterministic fallback summary is shown instead.
#
# Grounding limitation (documented, not silently hidden): numbers are tied
# to a nearby indicator name via a simple keyword-proximity heuristic (looks
# back up to _INDICATOR_TIE_WINDOW characters for the closest indicator
# mention). When no indicator keyword is found near a number — or that
# indicator has no matching input — the number is validated against the
# full set of input values instead of just that indicator's own figures.
# A number with no nearby indicator context can therefore still pass by
# coincidentally matching an unrelated indicator's value. Full semantic
# number-to-claim tying (e.g. via a second grounding pass) is out of scope.

import logging
import os
import re

MODEL_ID = "claude-haiku-4-5"
MAX_TOKENS = 300
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 1
REQUIRED_SENTENCE_COUNT = 3

# Relative tolerance dominates for larger values (e.g. payrolls in the
# thousands); the absolute floor covers small values like a 0.1pp Sahm gap.
NUMBER_TOLERANCE_ABS = 0.005
NUMBER_TOLERANCE_REL = 0.01

# Structural constants that appear in prose without being a "claim" about
# the data (e.g. "62.5 of 100").
_ALWAYS_ALLOWED = (0.0, 100.0)

_INDICATOR_TIE_WINDOW = 120  # chars to look back for a nearby indicator name

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Number extraction & grounding
# ─────────────────────────────────────────────

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
}
# Deliberately excludes "percent"/"percentage"/"thousand"/"million"/"billion":
# these are legitimate unit words in this domain (job counts in thousands,
# GDP in billions, rates in percent) that routinely follow a digit-based
# figure ("4.1 percent", "159,540 thousand jobs"), not a spelled-out number
# on their own.

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")
_QUARTER_LABEL_RE = re.compile(r"\bQ[1-4]\b", re.IGNORECASE)
# Small integer counts (<=12) attached to a unit word are structural
# ("3 sentences", "12-month", "one quarter"), not a cited data point.
_STRUCTURAL_COUNT_RE = re.compile(
    r"\b(\d{1,2})[\s-](?:month|months|quarter|quarters|year|years"
    r"|sentence|sentences|week|weeks|day|days)\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(r"\b(\d+\.\d+|\d+)\s*(?:-|to|–|—)\s*(\d+\.\d+|\d+)\b")
_NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\.\d+|-?\d+")

_INDICATOR_KEYWORDS = [
    ("unemployment", "UNRATE"),
    ("jobless", "UNRATE"),
    ("washington", "WAUR"),
    ("job opening", "JTSJOL"),
    ("payroll", "PAYEMS"),
    ("nonfarm", "PAYEMS"),
    ("inflation", "CPIAUCSL"),
    ("consumer price", "CPIAUCSL"),
    ("cpi", "CPIAUCSL"),
    ("fed funds", "FEDFUNDS"),
    ("federal funds", "FEDFUNDS"),
    ("interest rate", "FEDFUNDS"),
    ("gross domestic product", "GDPC1"),
    ("gdp", "GDPC1"),
    ("sahm", "sahm_"),  # gap and threshold are both citable
    ("health score", "health_score"),
]


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Spans of digits that are years, quarter labels, or structural counts
    — allowlisted, so they are never required to be grounded."""
    spans = [m.span() for m in _YEAR_RE.finditer(text)]
    spans += [m.span() for m in _QUARTER_LABEL_RE.finditer(text)]
    spans += [m.span(1) for m in _STRUCTURAL_COUNT_RE.finditer(text)]
    return spans


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(not (e <= a or s >= b) for a, b in spans)


def _extract_number_matches(text: str) -> list[tuple[float, int, int]]:
    """
    Extract (value, start, end) for every numeric claim in `text` that must
    be grounded. Strips thousands separators, parses ranges ("4.3-4.4") as
    two positive numbers rather than one negative number, handles
    leading-dot decimals (".10"), and skips years/quarter labels/small
    structural counts.
    """
    cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    protected = _protected_spans(cleaned)

    matches: list[tuple[float, int, int]] = []
    consumed: list[tuple[int, int]] = []

    for m in _RANGE_RE.finditer(cleaned):
        if _overlaps(m.span(), protected):
            continue
        g1, g2 = m.span(1), m.span(2)
        matches.append((float(m.group(1)), g1[0], g1[1]))
        matches.append((float(m.group(2)), g2[0], g2[1]))
        consumed.append(m.span())

    masked = list(cleaned)
    for start, end in consumed + protected:
        for i in range(start, end):
            masked[i] = " "
    masked_text = "".join(masked)

    for m in _NUMBER_RE.finditer(masked_text):
        matches.append((float(m.group()), m.start(), m.end()))

    return matches


def _contains_number_word(text: str) -> bool:
    """True if any spelled-out number word appears (e.g. 'roughly five
    percent') — these bypass digit extraction entirely, so they are
    rejected outright rather than silently ignored."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return any(w in _NUMBER_WORDS for w in words)


def _nearest_indicator_prefix(text: str, pos: int) -> str | None:
    preceding = text[:pos].lower()
    best_idx = -1
    best_prefix = None
    for keyword, prefix in _INDICATOR_KEYWORDS:
        idx = preceding.rfind(keyword)
        if idx > best_idx:
            best_idx = idx
            best_prefix = prefix
    if best_idx == -1 or pos - best_idx > _INDICATOR_TIE_WINDOW:
        return None
    return best_prefix


def _allowed_for_prefix(inputs: dict, prefix: str) -> list[float]:
    if prefix in inputs and isinstance(inputs[prefix], (int, float)):
        return [inputs[prefix]]
    return [v for k, v in inputs.items() if k.startswith(prefix) and isinstance(v, (int, float))]


def _numbers_are_grounded(text: str, inputs: dict) -> bool:
    """True only if every number in `text` matches some value in `inputs`
    (within tolerance) and no number is spelled out in words."""
    if _contains_number_word(text):
        return False

    full_allowed = [v for v in inputs.values() if isinstance(v, (int, float))]
    full_allowed = full_allowed + list(_ALWAYS_ALLOWED)

    for value, start, _end in _extract_number_matches(text):
        prefix = _nearest_indicator_prefix(text, start)
        candidates = full_allowed
        if prefix is not None:
            tied = _allowed_for_prefix(inputs, prefix)
            if tied:
                candidates = tied + list(_ALWAYS_ALLOWED)

        if not any(
            abs(abs(value) - abs(a)) <= max(NUMBER_TOLERANCE_ABS, NUMBER_TOLERANCE_REL * abs(a))
            for a in candidates
        ):
            return False
    return True


def _sentence_count(text: str) -> int:
    """Rough sentence count that does not split on decimal points (e.g.
    '4.3%') because there is no whitespace after the '.' in a decimal."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    return len(parts)


# ─────────────────────────────────────────────
# Inputs & fallback
# ─────────────────────────────────────────────

def build_briefing_inputs(summaries: dict, score: dict, sahm: dict) -> dict:
    """
    Assemble the deterministic numbers the briefing is allowed to cite.

    M1 fix: labels reflect what each figure actually is. CPI's YoY figure
    is percent inflation (`_yoy_pct`), GDP's period change is quarter-over-
    quarter (`_qoq_pct`, not `_mom_change`) since GDPC1 is quarterly data,
    and plain rate series (unemployment, Fed funds) keep their
    percentage-point framing (`_mom_change_pp` / `_yoy_change_pp`).

    Args:
        summaries: dict[series_id] -> analysis.get_summary_stats() output
        score:     health_score.health_score() output
        sahm:      sahm.current_status() output

    Returns:
        Flat dict of label -> numeric value, used both to build the prompt
        and to validate the model's output.
    """
    from config import FRED_SERIES  # local import: avoid a hard dependency for callers that stub summaries

    inputs = {
        "health_score": score["score"],
        "health_band": score["band"],
    }
    if sahm.get("as_of") is not None:
        inputs["sahm_gap_pp"] = sahm["gap"]
        # The trigger threshold is a fact the briefing may cite ("under the 0.5pp line").
        inputs["sahm_threshold_pp"] = sahm.get("threshold", 0.5)

    for sid, s in summaries.items():
        inputs[f"{sid}_current"] = s["current"]
        unit = FRED_SERIES.get(sid, {}).get("unit", "")

        period_pct = s.get("period_change_pct")
        if period_pct is not None:
            if sid == "GDPC1":
                inputs[f"{sid}_qoq_pct"] = period_pct
            elif unit == "%":
                inputs[f"{sid}_mom_change_pp"] = s["mom_change"]
            else:
                inputs[f"{sid}_mom_pct"] = period_pct

        yoy_pct = s.get("yoy_pct")
        if yoy_pct is not None:
            if unit == "%":
                inputs[f"{sid}_yoy_change_pp"] = yoy_pct
            else:
                inputs[f"{sid}_yoy_pct"] = yoy_pct

    return inputs


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
        f"(Deterministic summary: AI briefing unavailable, showing computed summary.)"
    )


def _format_for_prompt(value) -> str:
    """Pre-format a value with a fixed decimal count so the model can copy
    it verbatim, instead of reformatting it (which is how commas, unit
    conversions, and rounding drift get introduced)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

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
          reason    — human-readable note (why fallback/disabled); never
                      contains raw exception text (that is logged server-side
                      only, per H2/L5 — the review flagged leaking SDK
                      internals like request ids to end users).
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

    numbers_line = ", ".join(f"{k}={_format_for_prompt(v)}" for k, v in inputs.items())
    prompt = (
        "You are writing a macro briefing for an economic dashboard. "
        "Using ONLY the numbers below (do not calculate, convert units, or "
        "invent any other number), write exactly 3 sentences in plain "
        "English summarising the US economy's current state. Copy each "
        "figure exactly as given — same decimal places, no thousands "
        "separators, no unit conversions, and never spell a number out in "
        "words. Cite the actual figures.\n\n"
        f"Data: {numbers_line}"
    )

    try:
        client = Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        stop_reason = getattr(response, "stop_reason", None)
    except Exception as exc:  # network/auth/rate-limit errors — never crash the app
        logger.error("Claude API call failed: %s", exc, exc_info=True)
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": "AI briefing unavailable, showing computed summary.",
        }

    if stop_reason == "max_tokens":
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": "Model output was truncated before finishing (stop_reason=max_tokens).",
        }

    if not text or _sentence_count(text) != REQUIRED_SENTENCE_COUNT:
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": f"Model output was not exactly {REQUIRED_SENTENCE_COUNT} sentences.",
        }

    if not _numbers_are_grounded(text, inputs):
        return {
            "text": _fallback_summary(inputs),
            "source": "fallback",
            "reason": "Model output contained a number not present in the source data.",
        }

    return {"text": text, "source": "ai", "reason": None}
