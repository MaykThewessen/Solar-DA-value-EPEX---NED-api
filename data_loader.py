"""Shared loaders for Day-Ahead prices and NED.nl generation series.

All series are sourced from the birdcurve_nl DuckDB warehouse
(`birdcurve.duckdb`), which stores `timestamp_utc` as a tz-naive
TIMESTAMP in UTC by convention. Loaders localize to UTC at the read
boundary and convert to `Europe/Amsterdam` for downstream consumption.

Resolution invariants:
  - All loaders return a uniform 15-min grid. NED tables are natively
    15-min; pre-2025-10-01 hourly DA prices are forward-filled into the
    four 15-min slots they cover (matches how the EPEX hourly product
    actually clears).
  - Production is returned in MWh per 15-min slot (= MW × 0.25), so
    `production * price` is always EUR for that slot, regardless of
    when the row sits relative to the EPEX 1h→15min cutover.
  - To plot instantaneous power (MW) from MWh, multiply by 4.
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


SLOT_HOURS = 0.25  # 15-min slot length, hours


def load_ned_pv(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL Solar PV energy per 15-min slot from birdcurve.ts_15min.

    Returns `Solar_production_MWh` (= MW × 0.25). Night-time NULLs are
    coalesced to 0 (NED reports zero PV at night). Time range clipped to
    where PV was actually ingested.
    """
    with _connect() as con:
        df = con.execute(f"""
            WITH bounds AS (
                SELECT MIN(timestamp_utc) AS lo, MAX(timestamp_utc) AS hi
                  FROM ts_15min WHERE NED_PV__PV IS NOT NULL
            )
            SELECT t.timestamp_utc                          AS time,
                   COALESCE(t.NED_PV__PV, 0) * {SLOT_HOURS} AS Solar_production_MWh
              FROM ts_15min t, bounds b
             WHERE t.timestamp_utc BETWEEN b.lo AND b.hi
             ORDER BY t.timestamp_utc
        """).df()
    df['time'] = _utc_to_local(df['time'], tz)
    return df


def load_ned_wind(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL total Wind energy per 15-min slot from birdcurve.ts_15min.

    Returns `Wind_production_MWh` (= MW × 0.25, summed Onshore + Offshore).
    Time range bounded to where at least one wind column has been ingested;
    missing components inside that range treated as 0.
    """
    with _connect() as con:
        df = con.execute(f"""
            WITH bounds AS (
                SELECT MIN(timestamp_utc) AS lo, MAX(timestamp_utc) AS hi
                  FROM ts_15min
                 WHERE NED_Wind_Onshore__Wind_Onshore  IS NOT NULL
                    OR NED_Wind_Offshore__Wind_Offshore IS NOT NULL
            )
            SELECT t.timestamp_utc AS time,
                   ( COALESCE(t.NED_Wind_Onshore__Wind_Onshore,  0)
                   + COALESCE(t.NED_Wind_Offshore__Wind_Offshore, 0)
                   ) * {SLOT_HOURS} AS Wind_production_MWh
              FROM ts_15min t, bounds b
             WHERE t.timestamp_utc BETWEEN b.lo AND b.hi
             ORDER BY t.timestamp_utc
        """).df()
    df['time'] = _utc_to_local(df['time'], tz)
    return df


def load_da_prices(tz: str = DEFAULT_TZ, clip_future: bool = True) -> pd.DataFrame:
    """Load NL Day-Ahead prices on a uniform 15-min grid.

    EPEX NL switched from hourly to 15-min on 2025-10-01. Pre-cutover
    hourly rows are forward-filled into the four 15-min slots they cover
    (matches actual settlement: one price applies to all four quarters
    of the hour). Reindex is done in UTC to avoid DST ambiguity at the
    autumn fall-back.
    """
    with _connect() as con:
        df = con.execute("""
            SELECT timestamp_utc      AS time,
                   DA_price__DA_price AS DA_price
              FROM ts_hourly
             WHERE DA_price__DA_price IS NOT NULL
             ORDER BY timestamp_utc
        """).df()
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')

    grid = pd.date_range(df['time'].min(), df['time'].max(), freq='15min', tz='UTC')
    df = df.set_index('time').reindex(grid)
    df['DA_price'] = df['DA_price'].ffill(limit=3)
    df.index.name = 'time'
    df = df.reset_index()
    df['time'] = df['time'].dt.tz_convert(tz)

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
