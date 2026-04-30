"""Shared loaders for Day-Ahead prices and Ned.nl generation CSVs.

All loaders return a single concatenated DataFrame with a tz-aware `time`
column converted to `Europe/Amsterdam`. Source CSVs are discovered
recursively under `data_dir/` so loaders survive folder reorganization.
"""
from __future__ import annotations

import glob
import re
import pandas as pd
from datetime import datetime

DEFAULT_TZ = 'Europe/Amsterdam'

PRICE_GLOB = 'data/**/DA_prices_*.csv'
NED_PV_GLOB = 'data/**/data_NED_PV_*.csv'
NED_WIND_GLOB = 'data/**/data_NED_Wind_*.csv'

# Matches monthly suffix "_YYYY_MM.csv" (DA_prices) or "_YYYYMM.csv" (NED) at end of path.
_MONTH_RE = re.compile(r'_(\d{4})_?(\d{2})\.csv$')


def _check_file_completeness(df: pd.DataFrame, filepath: str, tz: str) -> str | None:
    """Return a warning string if the monthly CSV is materially incomplete, else None.

    Calendar bounds come from the YYYY-MM in the filename. Frequency is inferred
    from the dominant time-delta in the file itself (so a 15-min file checks against
    a 15-min expectation, an hourly file against hourly). For the current month, the
    expected end is clipped to today's local midnight. A 2-row tolerance absorbs
    DST edge effects.
    """
    m = _MONTH_RE.search(filepath)
    if not m or len(df) < 2:
        return None
    year, month = int(m.group(1)), int(m.group(2))

    start = pd.Timestamp(f'{year}-{month:02d}-01', tz=tz)
    end = start + pd.offsets.MonthBegin(1)
    today_midnight = pd.Timestamp(datetime.now(), tz=tz).normalize()
    cap = min(end, today_midnight)
    if cap <= start:
        return None  # future month, nothing to expect yet

    freq = df['time'].sort_values().diff().dropna().mode().iloc[0]
    expected_n = len(pd.date_range(start, cap, freq=freq, inclusive='left'))
    actual = len(df)
    if actual >= expected_n - 2:
        return None
    pct = (actual / expected_n * 100) if expected_n else 0.0
    return f"{filepath}: {actual}/{expected_n} rows ({pct:.1f}%, freq={freq})"


def _read_and_concat(pattern: str, label: str, tz: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if 'combined' not in f]
    assert files, f"No {label} files matched {pattern}"

    frames = []
    completeness_warnings: list[str] = []
    for f in files:
        df = pd.read_csv(f)
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(tz)
        msg = _check_file_completeness(df, f, tz)
        if msg:
            completeness_warnings.append(msg)
        frames.append(df)

    if completeness_warnings:
        print(f"[data_loader] {label}: {len(completeness_warnings)} incomplete file(s):")
        for w in completeness_warnings:
            print(f"  - {w}")

    return pd.concat(frames, ignore_index=True)


def load_da_prices(
    tz: str = DEFAULT_TZ,
    clip_future: bool = True,
    aggregate_to_hourly: bool = False,
) -> pd.DataFrame:
    """Load all Day-Ahead price CSVs, concatenated and tz-aware.

    `clip_future` drops rows beyond today's local midnight — set False
    for forward-looking analyses that include forecast/forward prices.

    `aggregate_to_hourly` averages 15-min prices into hourly bins, needed
    when merging with hourly NED generation data. EPEX day-ahead switched
    from 1h to 15-min resolution on 2025-10-01; without this flag the
    post-cutoff rows duplicate or misalign on merge. Floor is computed in
    UTC to avoid DST-ambiguity on autumn fall-back.
    """
    df = _read_and_concat(PRICE_GLOB, 'DA_prices', tz)
    if aggregate_to_hourly:
        hour_floor = df['time'].dt.tz_convert('UTC').dt.floor('h').dt.tz_convert(tz)
        df = (
            df.assign(time=hour_floor)
              .groupby('time', as_index=False)['DA_price'].mean()
        )
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
