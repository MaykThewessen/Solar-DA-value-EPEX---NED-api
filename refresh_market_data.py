#!/usr/bin/env python
"""Refresh the market-data chain that feeds this repo's dashboards.

The fetchers live in the birdcurve_nl repo; this repo only reads the DuckDB
warehouse they write. One command runs the whole chain:

  1. EPEX day-ahead prices   DA_prices/retrieve_EPEX_monthly_concat.py    (ENTSO-E)
  2. NED.nl PV + wind        NED_quarterly/a_retrieve_NED_quarterly_parallel.py
  3. CSV -> DuckDB self-heal backfill_duckdb_from_csv.py
  4. freshness report        read back through data_loader
  5. dashboards              Dashboard_market_prices_NL.py + …_profile_factor_…py

Two interpreters are involved. Steps 1-3 run on the birdcurve_nl interpreter
(that repo ships no pixi manifest, so it runs on the global env); step 5 runs on
this repo's pixi env, i.e. whatever interpreter started this script.

Usage
-----
    pixi run python refresh_market_data.py                    # whole chain
    pixi run python refresh_market_data.py --fetch-only        # steps 1-4
    pixi run python refresh_market_data.py --regenerate-only   # steps 4-5
    pixi run python refresh_market_data.py --full-backfill     # replay all history

Exit code is 1 when a series is stale after the run (usable from cron), and the
script aborts immediately if any step itself fails.

Overridable via the environment: BIRDCURVE_REPO, BIRDCURVE_PYTHON, BIRDCURVE_DB.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import data_loader

ROOT = Path(__file__).resolve().parent
BIRDCURVE_REPO = Path(os.environ.get('BIRDCURVE_REPO', '/Users/mayk/birdcurve_nl'))
BIRDCURVE_PYTHON = Path(os.environ.get(
    'BIRDCURVE_PYTHON',
    Path.home() / '.pixi' / 'envs' / 'main' / 'bin' / 'python',
))

# Warn when a series trails the wall clock by more than this. NED.nl publishes
# with a few hours' lag and the day-ahead auction clears a day out, so under a
# day is normal; a multi-day gap means a fetcher failed quietly upstream.
STALE_AFTER_HOURS = 36.0

# Both fetchers are incremental: they resume from the last stored timestamp, so
# "recent months" needs no date window. DuckDB is single-writer, so these run
# strictly sequentially, never in parallel.
FETCH_STEPS: list[tuple[str, list[str]]] = [
    ('Step 1/5  EPEX day-ahead prices (ENTSO-E)',
     ['DA_prices/retrieve_EPEX_monthly_concat.py']),
    ('Step 2/5  NED.nl PV + wind onshore/offshore',
     ['NED_quarterly/a_retrieve_NED_quarterly_parallel.py',
      '--sources', 'solar', 'wind_onshore', 'wind_offshore']),
]

# The per-source scripts write DuckDB themselves, but a lock conflict or a
# missing writer path fails silently there and leaves the warehouse behind the
# CSVs. This replay is what actually closes that gap.
BACKFILL_STEP: tuple[str, list[str]] = (
    'Step 3/5  CSV -> DuckDB self-heal', ['backfill_duckdb_from_csv.py'],
)

DASHBOARD_STEPS: list[tuple[str, list[str]]] = [
    ('Step 5/5a  market-price dashboards',
     ['Dashboard_market_prices_NL.py', 'all']),
    ('Step 5/5b  profile-factor dashboards',
     ['Dashboard_profile_factor_vs_capacity.py', 'all']),
]

# clip_future=False on day-ahead: the default drops tomorrow's cleared auction,
# which is exactly the freshness we want to see here.
SERIES: list[tuple[str, object, dict]] = [
    ('NED PV', data_loader.load_ned_pv, {}),
    ('NED Wind Onshore', data_loader.load_ned_wind_onshore, {}),
    ('NED Wind Offshore', data_loader.load_ned_wind_offshore, {}),
    ('DA prices', data_loader.load_da_prices, {'clip_future': False}),
]


def _run(label: str, python: Path, cwd: Path, args: list[str]) -> None:
    """Run one chain step with its output streaming through; abort on failure."""
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run([str(python), *args], cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {completed.returncode})")
    print(f"  ok ({time.perf_counter() - started:.1f}s)", flush=True)


def fetch(full_backfill: bool) -> None:
    """Run the three birdcurve_nl retrieval steps in order."""
    if not BIRDCURVE_REPO.is_dir():
        raise SystemExit(
            f"birdcurve_nl repo not found at {BIRDCURVE_REPO}. "
            "Set BIRDCURVE_REPO to override."
        )
    if not BIRDCURVE_PYTHON.exists():
        raise SystemExit(
            f"Interpreter for birdcurve_nl not found at {BIRDCURVE_PYTHON}. "
            "Set BIRDCURVE_PYTHON to override."
        )

    backfill_label, backfill_args = BACKFILL_STEP
    if full_backfill:
        backfill_args = [*backfill_args, '--full']

    for label, args in [*FETCH_STEPS, (backfill_label, backfill_args)]:
        _run(label, BIRDCURVE_PYTHON, BIRDCURVE_REPO, args)


def report() -> bool:
    """Print bounds and lag per series; return True when all are fresh."""
    print(f"\n{'=' * 78}\nStep 4/5  freshness of {data_loader.BIRDCURVE_DB}\n{'=' * 78}")
    now = pd.Timestamp.now(tz=data_loader.DEFAULT_TZ)
    all_fresh = True
    for label, load, kwargs in SERIES:
        times = load(**kwargs)['time']
        last = times.max()
        lag_hours = (now - last).total_seconds() / 3600
        stale = lag_hours > STALE_AFTER_HOURS
        all_fresh &= not stale
        print(f"  {label:<18}{len(times):>9,} rows  "
              f"{times.min():%Y-%m-%d} -> {last:%Y-%m-%d %H:%M}  "
              f"lag {lag_hours:6.1f}h{'   <-- STALE' if stale else ''}")
    if not all_fresh:
        print(f"\n  A series trails the clock by over {STALE_AFTER_HOURS:.0f}h. "
              "Check the fetcher output above for an upstream API error.")
    return all_fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--fetch-only', action='store_true',
                       help='Refresh data and report, skip the dashboards.')
    scope.add_argument('--regenerate-only', action='store_true',
                       help='Rebuild dashboards from the warehouse as it stands.')
    parser.add_argument('--full-backfill', action='store_true',
                        help='Replay the entire CSV history into DuckDB instead of '
                             'only rows past the watermark. Needed when upstream '
                             'revises timestamps that are already stored.')
    args = parser.parse_args(argv)

    if not args.regenerate_only:
        fetch(full_backfill=args.full_backfill)

    fresh = report()

    if not args.fetch_only:
        for label, cmd in DASHBOARD_STEPS:
            _run(label, Path(sys.executable), ROOT, cmd)

    print(f"\n{'=' * 78}\nRefresh complete"
          f"{'' if fresh else ' (with stale series, see above)'}\n{'=' * 78}")
    return 0 if fresh else 1


if __name__ == '__main__':
    raise SystemExit(main())
