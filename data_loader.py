"""Shared loaders for Day-Ahead prices and NED.nl generation series.

All series are sourced from the birdcurve_nl DuckDB warehouse
(`birdcurve.duckdb`), which stores `timestamp_utc` as a tz-naive
TIMESTAMP in UTC by convention. Loaders localize to UTC at the read
boundary and convert to `Europe/Amsterdam` for downstream consumption.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_TZ = 'Europe/Amsterdam'
BIRDCURVE_DB = Path(os.environ.get(
    'BIRDCURVE_DB',
    '/Users/mayk/birdcurve_nl/data/birdcurve.duckdb',
))


def _connect() -> duckdb.DuckDBPyConnection:
    if not BIRDCURVE_DB.exists():
        raise FileNotFoundError(
            f"birdcurve DuckDB not found at {BIRDCURVE_DB}. "
            "Set BIRDCURVE_DB env var to override."
        )
    return duckdb.connect(str(BIRDCURVE_DB), read_only=True)


def _utc_to_local(s: pd.Series, tz: str) -> pd.Series:
    return s.dt.tz_localize('UTC').dt.tz_convert(tz)


def load_ned_pv(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL Solar PV generation (15-min) from birdcurve.ts_15min.

    Night-time NULLs are coalesced to 0 to match the legacy CSV semantics
    (NED reports zero PV at night). The upper bound is clipped to the last
    timestamp where PV was actually ingested, so we don't fabricate zeros
    for un-pulled future periods.
    """
    with _connect() as con:
        df = con.execute("""
            WITH bounds AS (
                SELECT MIN(timestamp_utc) AS lo, MAX(timestamp_utc) AS hi
                  FROM ts_15min WHERE NED_PV__PV IS NOT NULL
            )
            SELECT t.timestamp_utc       AS time,
                   COALESCE(t.NED_PV__PV, 0) AS Solar_production_MW
              FROM ts_15min t, bounds b
             WHERE t.timestamp_utc BETWEEN b.lo AND b.hi
             ORDER BY t.timestamp_utc
        """).df()
    df['time'] = _utc_to_local(df['time'], tz)
    return df


def load_ned_wind(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL total Wind (Onshore + Offshore, 15-min) from birdcurve.ts_15min.

    Time range bounded to where at least one of the two wind columns has
    been ingested; missing component values within that range are treated
    as 0 (matches legacy CSV semantics).
    """
    with _connect() as con:
        df = con.execute("""
            WITH bounds AS (
                SELECT MIN(timestamp_utc) AS lo, MAX(timestamp_utc) AS hi
                  FROM ts_15min
                 WHERE NED_Wind_Onshore__Wind_Onshore  IS NOT NULL
                    OR NED_Wind_Offshore__Wind_Offshore IS NOT NULL
            )
            SELECT t.timestamp_utc AS time,
                   COALESCE(t.NED_Wind_Onshore__Wind_Onshore,  0)
                 + COALESCE(t.NED_Wind_Offshore__Wind_Offshore, 0)
                   AS Wind_production_MW
              FROM ts_15min t, bounds b
             WHERE t.timestamp_utc BETWEEN b.lo AND b.hi
             ORDER BY t.timestamp_utc
        """).df()
    df['time'] = _utc_to_local(df['time'], tz)
    return df


def load_da_prices(
    tz: str = DEFAULT_TZ,
    clip_future: bool = True,
    aggregate_to_hourly: bool = False,
) -> pd.DataFrame:
    """Load NL Day-Ahead prices from birdcurve.ts_hourly.

    Despite the table name, post-2025-10-01 rows are at 15-min resolution
    (ENTSO-E NL switched MTU). `aggregate_to_hourly` averages 15-min bins
    into hourly bins (UTC-floor to dodge DST ambiguity), needed when
    merging against hourly series.
    """
    with _connect() as con:
        df = con.execute("""
            SELECT timestamp_utc      AS time,
                   DA_price__DA_price AS DA_price
              FROM ts_hourly
             WHERE DA_price__DA_price IS NOT NULL
             ORDER BY timestamp_utc
        """).df()
    df['time'] = _utc_to_local(df['time'], tz)
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


if __name__ == '__main__':
    for label, fn in [
        ('NED PV',    load_ned_pv),
        ('NED Wind',  load_ned_wind),
        ('DA prices', load_da_prices),
    ]:
        df = fn()
        print(f"{label}: {len(df):>8,} rows  "
              f"{df['time'].min()} → {df['time'].max()}")
