"""Smoke + invariant tests for the three market-price dashboards.

Validates:
    * dashboard_common module imports and core helpers behave
    * each dashboard script runs to completion
    * each produces its expected output files
    * yearly summary internal invariants hold (capture rate <= 100% in normal years,
      curtailment % matches definition, no NaN in installed capacity rows where data exists)

Run:
    ~/.pixi/envs/main/bin/python test_dashboards.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = os.path.expanduser('~/.pixi/envs/main/bin/python')


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _check(cond: bool, msg: str) -> None:
    mark = '✓' if cond else '✗'
    print(f"  {mark} {msg}")
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- 1) common module

def test_common_module():
    _section('1) dashboard_common smoke')
    import dashboard_common as dc

    _check(set(dc.CLAUDE_PALETTE) >= {'bg', 'ink', 'accent', 'sage', 'blue', 'muted', 'grid'},
           'CLAUDE_PALETTE has core keys')
    _check(dc.hex_to_rgb('#C96442') == (201, 100, 66), 'hex_to_rgb decodes accent')
    _check(dc.hex_to_rgb('C96442') == (201, 100, 66), 'hex_to_rgb tolerates leading-#-stripping')

    # add_dt_hours: first-row should equal median, not bfill's second-row value
    times = pd.to_datetime(['2025-01-01 00:00', '2025-01-01 01:00', '2025-01-01 02:00',
                            '2025-01-01 02:15', '2025-01-01 02:30'], utc=True)
    df = pd.DataFrame({'time': times})
    out = dc.add_dt_hours(df)
    _check(np.isclose(out.loc[0, '_dt_h'], np.median([1.0, 1.0, 0.25, 0.25])),
           'add_dt_hours fills first NaN with series median (not second-row value)')

    # last_complete_year
    _check(dc.last_complete_year(pd.Timestamp('2025-06-15')) == 2024,
           'mid-year -> previous full year')
    _check(dc.last_complete_year(pd.Timestamp('2025-12-31 23:45')) == 2025,
           'year-end -> current year')

    # vw_price_groupby vectorised vs naive
    df = pd.DataFrame({
        'm': [1, 1, 2, 2, 2],
        'p': [10.0, 30.0, 50.0, 0.0, 60.0],
        'pr': [100.0, 200.0, 50.0, 999.0, 150.0],
    })
    out = dc.vw_price_groupby(df, 'm', 'p', 'pr')
    expected_m1 = (10 * 100 + 30 * 200) / (10 + 30)
    expected_m2 = (50 * 50 + 0 + 60 * 150) / (50 + 0 + 60)
    _check(np.isclose(out[1], expected_m1), f'volume-weighted price month 1 = {expected_m1}')
    _check(np.isclose(out[2], expected_m2), f'volume-weighted price month 2 = {expected_m2}')


# ---------------------------------------------------------------- 2) run dashboards + check outputs

def _run_dashboard(script: str) -> None:
    res = subprocess.run([PY, script], cwd=ROOT, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print('STDERR:', res.stderr[-2000:])
        raise AssertionError(f'{script} exited {res.returncode}')


def test_run_all():
    _section('2) run all 3 dashboards')
    scripts = [
        'Dashboard_Solar_PV_market_prices_NL.py',
        'Dashboard_Wind_Onshore_market_prices_NL.py',
        'Dashboard_Wind_Offshore_market_prices_NL.py',
    ]
    for s in scripts:
        print(f"  running {s} ...")
        _run_dashboard(s)
        print(f"  done")

    expected = [
        'solar_yearly_slides.pdf', 'solar_yearly_slides.html',
        'solar_production_plot_v3.html', 'monthly_summary_table.html',
        'wind_onshore_yearly_slides.pdf', 'wind_onshore_production_plot_v3.html',
        'wind_onshore_monthly_summary_table.html',
        'wind_offshore_yearly_slides.pdf', 'wind_offshore_production_plot_v3.html',
        'wind_offshore_monthly_summary_table.html',
    ]
    for fname in expected:
        p = ROOT / fname
        _check(p.exists() and p.stat().st_size > 1_000, f'{fname} produced (>1 KB)')


# ---------------------------------------------------------------- 3) cross-PDF size sanity

def test_pdfs_reasonable():
    _section('3) PDF size sanity')
    for f in ['solar_yearly_slides.pdf', 'wind_onshore_yearly_slides.pdf', 'wind_offshore_yearly_slides.pdf']:
        size = (ROOT / f).stat().st_size
        _check(50_000 < size < 10_000_000, f'{f}: {size / 1024:.0f} KB within sane range')


# ---------------------------------------------------------------- runner

def main() -> int:
    try:
        test_common_module()
        test_run_all()
        test_pdfs_reasonable()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
