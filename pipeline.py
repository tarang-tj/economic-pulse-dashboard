# pipeline.py — ETL: Fetch, validate, clean, and cache FRED data
#
# Usage:
#   from pipeline import load_series, load_all
#   df = load_series("UNRATE")
#   all_data = load_all()

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from config import FRED_SERIES, CACHE_TTL_HOURS, DATE_FORMAT

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────

def _cache_path(series_id: str) -> Path:
    return CACHE_DIR / f"{series_id}.json"


def _is_cache_valid(series_id: str) -> bool:
    path = _cache_path(series_id)
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - modified < timedelta(hours=CACHE_TTL_HOURS)


def _write_cache(series_id: str, records: list[dict]) -> None:
    with open(_cache_path(series_id), "w") as f:
        json.dump(records, f)


def _read_cache(series_id: str) -> list[dict]:
    with open(_cache_path(series_id)) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# FRED API fetch
# ─────────────────────────────────────────────

def _fetch_from_fred(series_id: str) -> list[dict]:
    """Hit the FRED observations endpoint and return raw records."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and add it to your .env file."
        )

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }

    logger.info(f"Fetching {series_id} from FRED API…")
    response = requests.get(FRED_BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    if "observations" not in data:
        raise ValueError(f"Unexpected FRED response for {series_id}: {data}")

    return data["observations"]


# ─────────────────────────────────────────────
# Clean & transform
# ─────────────────────────────────────────────

def _clean(series_id: str, records: list[dict]) -> pd.DataFrame:
    """
    Convert raw FRED observations into a tidy DataFrame.

    Steps:
      1. Parse dates
      2. Drop missing/placeholder values ('.')
      3. Cast to float
      4. Set datetime index
      5. Drop duplicates, sort
      6. Compute YoY % change
    """
    df = pd.DataFrame(records)[["date", "value"]].copy()

    # 1. Parse dates
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)

    # 2. Remove FRED's missing-value placeholder
    df = df[df["value"] != "."].copy()

    # 3. Numeric cast
    df["value"] = df["value"].astype(float)

    # 4. Set index
    df = df.set_index("date").sort_index()

    # 5. Drop duplicates
    df = df[~df.index.duplicated(keep="last")]

    # 6. Year-over-year % change (meaningful for levels, not rates)
    meta = FRED_SERIES.get(series_id, {})
    unit = meta.get("unit", "")
    if unit not in ("%",):
        df["yoy_pct"] = df["value"].pct_change(periods=12) * 100
    else:
        df["yoy_pct"] = df["value"].diff(periods=12)   # pp change for rates

    df["series_id"] = series_id
    df["series_name"] = meta.get("short", series_id)

    logger.info(f"{series_id}: {len(df)} clean observations ({df.index.min().date()} → {df.index.max().date()})")
    return df


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_series(series_id: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load a single FRED series. Uses local cache if fresh, otherwise fetches.

    Args:
        series_id:     FRED series identifier (e.g. 'UNRATE')
        force_refresh: bypass cache and always fetch from API

    Returns:
        Cleaned DataFrame indexed by date with columns:
          value, yoy_pct, series_id, series_name
    """
    if not force_refresh and _is_cache_valid(series_id):
        logger.info(f"{series_id}: loaded from cache")
        records = _read_cache(series_id)
    else:
        records = _fetch_from_fred(series_id)
        _write_cache(series_id, records)

    return _clean(series_id, records)


def load_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load all series defined in config.FRED_SERIES.

    Returns:
        Dict mapping series_id → cleaned DataFrame
    """
    results = {}
    errors = []

    for series_id in FRED_SERIES:
        try:
            results[series_id] = load_series(series_id, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Failed to load {series_id}: {e}")
            errors.append(series_id)

    if errors:
        logger.warning(f"Could not load: {errors}")

    return results


def get_latest(df: pd.DataFrame) -> tuple[float, datetime]:
    """Return the most recent (value, date) from a series DataFrame."""
    latest_row = df["value"].dropna().iloc[-1]
    latest_date = df["value"].dropna().index[-1]
    return float(latest_row), latest_date


if __name__ == "__main__":
    # Quick smoke test — run: python pipeline.py
    data = load_all()
    for sid, df in data.items():
        val, dt = get_latest(df)
        print(f"{sid:12s}  latest={val:.2f}  date={dt.date()}")
