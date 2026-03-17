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
