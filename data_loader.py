"""Shared loaders for Day-Ahead prices and NED.nl generation series.

All series are sourced from the birdcurve_nl DuckDB warehouse
(`birdcurve.duckdb`). DuckDB stores `timestamp_utc` as a tz-aware
TIMESTAMPTZ once `SET TimeZone='UTC'` is applied at connect time, so
loaders no longer need to round-trip through `tz_localize`.

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
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_TZ = 'Europe/Amsterdam'
ROOT = Path(__file__).resolve().parent
BIRDCURVE_DB = Path(os.environ.get(
    'BIRDCURVE_DB',
    '/Users/mayk/birdcurve_nl/data/birdcurve.duckdb',
))

SLOT_HOURS = 0.25  # 15-min slot length, hours

# Installed-capacity anchor points, one CSV per technology. This is the single
# source of truth: every dashboard resolves capacity through `load_capacity_points`
# against these files. Never inline an anchor list in a script — two writable
# copies of the same series drift (the offshore file once held onshore numbers,
# understating offshore €/MW by a third before anyone noticed).
CAPACITY_CSV: dict[str, Path] = {
    'solar_pv': ROOT / 'capacity_points_solar_PV_NL_v1.csv',
    'wind_onshore': ROOT / 'capacity_points_wind_onshore_NL_v1.csv',
    'wind_offshore': ROOT / 'capacity_points_wind_offshore_NL_v1.csv',
}


# Module-level DuckDB connection cache (read-only). Re-opening the database for
# every loader call adds ~50-100ms × 5 calls per dashboard run; cache it.
#
# Call `close()` as soon as the last loader call of a phase returns. DuckDB's
# file lock is per-process, not per-mode: `read_only=True` still blocks any
# writer in another process, so an open handle here stalls the birdcurve_nl
# CSV->DuckDB backfill for as long as it lives. A dashboard run spends seconds
# querying and minutes rendering; holding the connection across the render
# phase is what fails `backfill_duckdb_from_csv.py` (its retry budget is 31s).
_CON: duckdb.DuckDBPyConnection | None = None


def _connect() -> duckdb.DuckDBPyConnection:
    """Return a cached read-only DuckDB connection with timezone fixed to UTC."""
    global _CON
    if _CON is None:
        if not BIRDCURVE_DB.exists():
            raise FileNotFoundError(
                f"birdcurve DuckDB not found at {BIRDCURVE_DB}. "
                "Set BIRDCURVE_DB env var to override."
            )
        _CON = duckdb.connect(str(BIRDCURVE_DB), read_only=True)
        _CON.execute("SET TimeZone='UTC'")
    return _CON


def close() -> None:
    """Release the cached connection, and with it the DuckDB file lock.

    Idempotent, and safe to call between load phases: the next loader call
    simply reopens. Cheap to be liberal with it, expensive to forget it.
    """
    global _CON
    if _CON is not None:
        _CON.close()
        _CON = None


def _to_local(s: pd.Series, tz: str) -> pd.Series:
    """Convert a UTC timestamp Series to the display tz.

    DuckDB may return either tz-naive UTC or tz-aware UTC depending on the
    column type; handle both. `tz_localize('UTC')` raises on already-aware
    Series, so we check first.
    """
    if s.dt.tz is None:
        s = s.dt.tz_localize('UTC')
    return s.dt.tz_convert(tz)


def load_ned_pv(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL Solar PV energy per 15-min slot from birdcurve.ts_15min.

    Returns `Solar_production_MWh` (= MW × 0.25). Night-time NULLs are
    coalesced to 0 (NED reports zero PV at night). Time range clipped to
    where PV was actually ingested.
    """
    df = _connect().execute(f"""
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
    df['time'] = _to_local(df['time'], tz)
    return df


def _load_ned_wind_component(column: str, out_col: str, tz: str) -> pd.DataFrame:
    df = _connect().execute(f"""
        WITH bounds AS (
            SELECT MIN(timestamp_utc) AS lo, MAX(timestamp_utc) AS hi
              FROM ts_15min WHERE {column} IS NOT NULL
        )
        SELECT t.timestamp_utc                  AS time,
               COALESCE(t.{column}, 0) * {SLOT_HOURS} AS {out_col}
          FROM ts_15min t, bounds b
         WHERE t.timestamp_utc BETWEEN b.lo AND b.hi
         ORDER BY t.timestamp_utc
    """).df()
    df['time'] = _to_local(df['time'], tz)
    return df


def load_ned_wind_onshore(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL Onshore Wind energy per 15-min slot from birdcurve.ts_15min.

    Returns `Wind_production_MWh` (= MW × 0.25, Onshore only). Time range
    clipped to where Onshore Wind was actually ingested.
    """
    return _load_ned_wind_component(
        'NED_Wind_Onshore__Wind_Onshore', 'Wind_production_MWh', tz,
    )


def load_ned_wind_offshore(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL Offshore Wind energy per 15-min slot from birdcurve.ts_15min.

    Returns `Wind_production_MWh` (= MW × 0.25, Offshore only). Time range
    clipped to where Offshore Wind was actually ingested.
    """
    return _load_ned_wind_component(
        'NED_Wind_Offshore__Wind_Offshore', 'Wind_production_MWh', tz,
    )


def load_ned_wind(tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Load NL total Wind energy per 15-min slot (Onshore + Offshore).

    Returns `Wind_production_MWh` (= MW × 0.25, summed Onshore + Offshore).
    Time range bounded to where at least one wind column has been ingested;
    missing components inside that range treated as 0.
    """
    df = _connect().execute(f"""
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
    df['time'] = _to_local(df['time'], tz)
    return df


def load_da_prices(tz: str = DEFAULT_TZ, clip_future: bool = True) -> pd.DataFrame:
    """Load NL Day-Ahead prices on a uniform 15-min grid.

    EPEX NL switched from hourly to 15-min on 2025-10-01. Pre-cutover
    hourly rows are forward-filled into the four 15-min slots they cover
    (matches actual settlement: one price applies to all four quarters
    of the hour). Reindex is done in UTC to avoid DST ambiguity at the
    autumn fall-back.
    """
    df = _connect().execute("""
        SELECT timestamp_utc      AS time,
               DA_price__DA_price AS DA_price
          FROM ts_hourly
         WHERE DA_price__DA_price IS NOT NULL
         ORDER BY timestamp_utc
    """).df()
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')

    grid = pd.date_range(df['time'].min(), df['time'].max(), freq='15min', tz='UTC')
    df = df.set_index('time').reindex(grid)
    df['DA_price'] = df['DA_price'].ffill(limit=3)
    df.index.name = 'time'
    df = df.reset_index()
    df['time'] = df['time'].dt.tz_convert(tz)

    if clip_future:
        # Tz-aware "today midnight in the display tz" — direct replacement for
        # the older `datetime.now().replace(hour=0,…).astimezone(tz)` round-trip
        # which depended on the host machine's local clock.
        midnight_today = pd.Timestamp.now(tz=tz).normalize()
        df = df[df['time'] <= midnight_today]
    return df


def load_capacity_points(csv_path: str | Path,
                         tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Read a capacity-points CSV into a tz-aware (date, MW) DataFrame.

    Expects columns `date` (parseable timestamp with offset, e.g.
    `2024-12-31 00:00:00+01:00`) and `MW`.
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(tz)
    df['MW'] = df['MW'].astype(float)
    return df.sort_values('date').reset_index(drop=True)


def interp_capacity(times: pd.Series, anchors: pd.DataFrame) -> np.ndarray:
    """Piece-wise linear interpolation of installed capacity at `times`.

    Vectorised replacement for the per-row `.apply(...)` interpolator in the
    legacy dashboards. Outside the anchor range capacities are flat-extrapolated
    (numpy's default for `np.interp`), matching the legacy Solar behaviour.

    Parameters
    ----------
    times : pd.Series of tz-aware Timestamps.
    anchors : DataFrame with columns 'date' (tz-aware Timestamps, sorted asc)
        and 'MW' (float).

    Returns
    -------
    ndarray of float capacities aligned with `times`.
    """
    # Convert both sides to int64 nanoseconds-since-epoch so np.interp can run
    # on plain ints. Both inputs are tz-aware, so the underlying ns-grid is
    # absolute (no DST drift).
    x_anchor = anchors['date'].astype('int64').to_numpy()
    y_anchor = anchors['MW'].to_numpy(dtype=float)
    x = pd.to_datetime(times).astype('int64').to_numpy()
    return np.interp(x, x_anchor, y_anchor)


if __name__ == '__main__':
    for label, fn in [
        ('NED PV',            load_ned_pv),
        ('NED Wind Onshore',  load_ned_wind_onshore),
        ('NED Wind Offshore', load_ned_wind_offshore),
        ('NED Wind (total)',  load_ned_wind),
        ('DA prices',         load_da_prices),
    ]:
        df = fn()
        print(f"{label}: {len(df):>8,} rows  "
              f"{df['time'].min()} → {df['time'].max()}")
