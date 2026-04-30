"""Shared loaders for Day-Ahead prices and Ned.nl generation CSVs.

All loaders return a single concatenated DataFrame with a tz-aware `time`
column converted to `Europe/Amsterdam`. Source CSVs are discovered
recursively under `data_dir/` so loaders survive folder reorganization.
"""
from __future__ import annotations

import glob
import pandas as pd
from datetime import datetime

DEFAULT_TZ = 'Europe/Amsterdam'

PRICE_GLOB = 'data/**/DA_prices_*.csv'
NED_PV_GLOB = 'data/**/data_NED_PV_*.csv'
NED_WIND_GLOB = 'data/**/data_NED_Wind_*.csv'


def _read_and_concat(pattern: str, label: str, tz: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if 'combined' not in f]
    assert files, f"No {label} files matched {pattern}"

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(tz)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_da_prices(tz: str = DEFAULT_TZ, clip_future: bool = True) -> pd.DataFrame:
    """Load all Day-Ahead price CSVs, concatenated and tz-aware.

    `clip_future` drops rows beyond today's local midnight — set False
    for forward-looking analyses that include forecast/forward prices.
    """
    df = _read_and_concat(PRICE_GLOB, 'DA_prices', tz)
    if clip_future:
        midnight_today = pd.Timestamp(
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            tz=tz,
        )
        df = df[df['time'] <= midnight_today]
    return df


def load_ned_pv(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load all Ned.nl Solar PV generation CSVs, concatenated and tz-aware."""
    return _read_and_concat(NED_PV_GLOB, 'NED PV', tz)


def load_ned_wind(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load all Ned.nl Wind generation CSVs, concatenated and tz-aware."""
    return _read_and_concat(NED_WIND_GLOB, 'NED Wind', tz)
