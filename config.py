# config.py — Series definitions, labels, and dashboard constants

FRED_SERIES = {
    "UNRATE": {
        "name": "US Unemployment Rate",
        "short": "US Unemployment",
        "unit": "%",
        "category": "Labor Market",
        "description": "Civilian unemployment rate (seasonally adjusted)",
        "invert_signal": True,   # lower is better
    },
    "WAUR": {
        "name": "Washington State Unemployment Rate",
        "short": "WA Unemployment",
        "unit": "%",
        "category": "Labor Market",
        "description": "Washington state unemployment rate",
        "invert_signal": True,
    },
    "JTSJOL": {
        "name": "US Job Openings",
        "short": "Job Openings",
        "unit": "Thousands",
        "category": "Labor Market",
        "description": "Total nonfarm job openings (JOLTS)",
        "invert_signal": False,
    },
    "PAYEMS": {
        "name": "Total Nonfarm Payrolls",
        "short": "Nonfarm Payrolls",
        "unit": "Thousands",
        "category": "Labor Market",
        "description": "Total employed persons in nonfarm sector",
        "invert_signal": False,
    },
    "CPIAUCSL": {
        "name": "Consumer Price Index (CPI)",
        "short": "CPI (Inflation)",
        "unit": "Index (1982–84=100)",
        "category": "Inflation",
        "description": "All items CPI, seasonally adjusted",
        "invert_signal": True,
    },
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "short": "Fed Funds Rate",
        "unit": "%",
        "category": "Monetary Policy",
        "description": "Effective federal funds rate",
        "invert_signal": None,  # neutral — depends on context
    },
    "GDPC1": {
        "name": "Real GDP",
        "short": "Real GDP",
        "unit": "Billions USD (Chained 2017)",
        "category": "Economic Growth",
        "description": "Real gross domestic product, seasonally adjusted",
        "invert_signal": False,
    },
}

LOOKBACK_YEARS = 5          # default chart window
CACHE_TTL_HOURS = 12        # how long to cache fetched data
DATE_FORMAT = "%Y-%m-%d"

CATEGORY_ORDER = [
    "Labor Market",
    "Inflation",
    "Monetary Policy",
    "Economic Growth",
]

# ─────────────────────────────────────────────
# Economic Health Score
# ─────────────────────────────────────────────
# Weights per indicator (must sum to 1.0). Labor market and growth carry the
# most signal for near-term recession risk; WA-only unemployment and the Fed
# funds rate are supporting context, weighted lighter.
HEALTH_SCORE_WEIGHTS = {
    "UNRATE": 0.20,
    "WAUR": 0.10,
    "JTSJOL": 0.15,
    "PAYEMS": 0.15,
    "CPIAUCSL": 0.15,
    "FEDFUNDS": 0.10,
    "GDPC1": 0.15,
}
assert abs(sum(HEALTH_SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "HEALTH_SCORE_WEIGHTS must sum to 1.0"

# Direction for scoring purposes: True = higher raw value is WORSE (invert
# before scoring), False = higher is better. FRED_SERIES["invert_signal"] is
# None for FEDFUNDS because its effect is context-dependent for a human
# reader; for the composite score we treat a rising fed funds rate as
# tightening conditions (worse for near-term growth) — documented assumption.
HEALTH_SCORE_DIRECTION = {
    "UNRATE": True,
    "WAUR": True,
    "JTSJOL": False,
    "PAYEMS": False,
    "CPIAUCSL": True,
    "FEDFUNDS": True,
    "GDPC1": False,
}

HEALTH_SCORE_LOOKBACK_YEARS = 10
HEALTH_SCORE_MIN_OBSERVATIONS = 24  # min history required per indicator

# (low, high, label) — score bands, low inclusive, high exclusive (100 inclusive in top band)
HEALTH_SCORE_BANDS = [
    (0, 25, "Contraction"),
    (25, 50, "Weak"),
    (50, 75, "Moderate"),
    (75, 100.0001, "Strong"),
]

# ─────────────────────────────────────────────
# Sahm Rule
# ─────────────────────────────────────────────
SAHM_TRIGGER_THRESHOLD = 0.50  # percentage points
