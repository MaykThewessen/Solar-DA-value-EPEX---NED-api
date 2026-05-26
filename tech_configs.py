"""Per-technology configuration for the unified market-prices dashboard.

Each tech (Solar PV, Wind Onshore, Wind Offshore) shares the same pipeline:
load NED generation → merge with DA prices → derive monthly/yearly metrics →
emit two HTMLs and one themed PDF. The only differences are:

  * which loader to call
  * the capacity-points CSV
  * unit labels (`MWp DC` for solar, `MW AC` for wind)
  * output filenames and slide titles
  * x-axis end-date for the "capacity curve" subplot (only used by wind)
  * year-color palette (solar adds 2 warm hues for recent years; wind does not)

`TechConfig` captures all of those in one frozen dataclass so the unified
dashboard script reads as: `cfg = TECHS[slug]; run_dashboard(cfg)`.

Slide-spec helpers (`build_pdf_slides_spec`) live here too because slide titles
and y-axis ranges are tech-specific (€/MWp vs €/MW, 80 k vs 500 k axis ranges).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from data_loader import (
    load_da_prices,
    load_ned_pv,
    load_ned_wind_offshore,
    load_ned_wind_onshore,
)

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------- types

@dataclass(frozen=True)
class TechConfig:
    """All settings needed to build a market-prices dashboard for one tech."""

    # ----------- identity
    name: str                                            # "Solar PV" / "Wind Onshore" / "Wind Offshore"
    slug: str                                            # "solar_pv" / "wind_onshore" / "wind_offshore"
    brand: str                                           # banner text in PDF, e.g. "SOLAR · NL"

    # ----------- data
    loader: Callable[[], pd.DataFrame]                   # returns df with `time`, `<prod_col>`
    prod_col: str                                        # generation MWh column name (e.g. "Solar_production_MWh")
    capacity_csv: Path                                   # path to capacity_points_*.csv
    clip_future_prices: bool = True                      # passed to load_da_prices

    # ----------- units / labels
    cap_unit: str = 'MWp DC'                             # used in print() banner only
    cap_unit_short: str = 'GWp'                          # used in slide y-axis labels
    cap_per_unit_short: str = 'MWp'                      # 'MWp' for solar, 'MW' for wind
    capture_metric_name: str = 'Solar capture'           # "Solar capture" / "Onshore wind capture" / ...
    capture_metric_name_title: str = 'Solar Capture'     # title-case version

    # ----------- output filenames
    out_yearly_pdf: str = ''
    out_yearly_html: str = ''
    out_production_html: str = ''
    out_monthly_table_html: str = ''
    out_capacity_pdf: str = ''                           # matplotlib QA plot

    # ----------- legacy table-header strings
    short_label: str = 'Solar PV'                        # "Solar PV" / "Onshore Wind" / "Offshore Wind"

    # ----------- color palette
    year_palette: str = 'solar'                          # 'solar' or 'wind' (see dashboard_common.year_color_map)
    year_highlight_recent: int = 0                       # 0 or 2 (legacy Solar used 2)

    # ----------- PDF slide tuning
    pdf_value_yaxis_max: float = 80_000.0                # y-max for "Annual market value" slide
    pdf_price_yaxis_max: float = 110.0                   # y-max for "Capture price vs DA" slide
    pdf_value_callout_y: float = 78_000.0                # y for 2022-spike callout (annotation)
    pdf_price_callout_y: float = 105.0                   # y for 2022-spike callout (annotation)

    # ----------- subplot end-date for the "installed capacity" curve (wind only)
    capacity_curve_end: pd.Timestamp | None = None

    # ----------- yearly-summary column suffix for clarity in column names
    # Solar uses "PV", Wind uses "Wind"; we set this in derived col names like
    # "Yearly_<tech_col_prefix>_Energy_MWh".
    tech_col_prefix: str = 'PV'

    # ----------- whether to emit the "monthly capture-rate-by-year" PDF page
    emit_monthly_pdf_page: bool = True

    # Filled in by __post_init__-style helper below
    palette: dict[str, str] = field(default_factory=dict)

    # ---- derived display labels ------------------------------------------

    @property
    def value_per_unit_label(self) -> str:
        """e.g. '€/MWp/year' or '€/MW/year'."""
        return f'€/{self.cap_per_unit_short}/y'

    @property
    def power_label(self) -> str:
        """e.g. 'PV' or 'Wind' — used in print() and column lookups."""
        return 'PV' if self.tech_col_prefix == 'PV' else 'Wind'


# ----------------------------------------------------------------------- registry

TECH_SOLAR_PV = TechConfig(
    name='Solar PV',
    slug='solar_pv',
    brand='SOLAR · NL',
    loader=load_ned_pv,
    prod_col='Solar_production_MWh',
    capacity_csv=ROOT / 'capacity_points_solar_PV_NL_v1.csv',
    clip_future_prices=True,
    cap_unit='MWp DC',
    cap_unit_short='GWp',
    cap_per_unit_short='MWp',
    capture_metric_name='Solar capture',
    capture_metric_name_title='Solar Capture',
    out_yearly_pdf='solar_yearly_slides.pdf',
    out_yearly_html='solar_yearly_slides.html',
    out_production_html='solar_production_plot_v3.html',
    out_monthly_table_html='monthly_summary_table.html',
    out_capacity_pdf='installed_capacity_plot.pdf',
    short_label='Solar PV',
    year_palette='solar',
    year_highlight_recent=2,
    pdf_value_yaxis_max=80_000.0,
    pdf_price_yaxis_max=110.0,
    pdf_value_callout_y=78_000.0,
    pdf_price_callout_y=105.0,
    tech_col_prefix='PV',
)


TECH_WIND_ONSHORE = TechConfig(
    name='Wind Onshore',
    slug='wind_onshore',
    brand='WIND ONSHORE · NL',
    loader=load_ned_wind_onshore,
    prod_col='Wind_production_MWh',
    capacity_csv=ROOT / 'capacity_points_wind_onshore_NL_v1.csv',
    clip_future_prices=False,
    cap_unit='MW AC',
    cap_unit_short='GW',
    cap_per_unit_short='MW',
    capture_metric_name='Onshore wind capture',
    capture_metric_name_title='Onshore Wind Capture',
    out_yearly_pdf='wind_onshore_yearly_slides.pdf',
    out_yearly_html='wind_onshore_yearly_slides.html',
    out_production_html='wind_onshore_production_plot_v3.html',
    out_monthly_table_html='wind_onshore_monthly_summary_table.html',
    out_capacity_pdf='wind_onshore_installed_capacity_vs_known_points.pdf',
    short_label='Onshore Wind',
    year_palette='wind',
    year_highlight_recent=0,
    pdf_value_yaxis_max=500_000.0,
    pdf_price_yaxis_max=250.0,
    pdf_value_callout_y=460_000.0,
    pdf_price_callout_y=235.0,
    tech_col_prefix='Wind',
    capacity_curve_end=pd.Timestamp('2030-12-31', tz='Europe/Amsterdam'),
)


TECH_WIND_OFFSHORE = TechConfig(
    name='Wind Offshore',
    slug='wind_offshore',
    brand='WIND OFFSHORE · NL',
    loader=load_ned_wind_offshore,
    prod_col='Wind_production_MWh',
    capacity_csv=ROOT / 'capacity_points_wind_offshore_NL_v1.csv',
    clip_future_prices=False,
    cap_unit='MW AC',
    cap_unit_short='GW',
    cap_per_unit_short='MW',
    capture_metric_name='Offshore wind capture',
    capture_metric_name_title='Offshore Wind Capture',
    out_yearly_pdf='wind_offshore_yearly_slides.pdf',
    out_yearly_html='wind_offshore_yearly_slides.html',
    out_production_html='wind_offshore_production_plot_v3.html',
    out_monthly_table_html='wind_offshore_monthly_summary_table.html',
    out_capacity_pdf='wind_offshore_installed_capacity_vs_known_points.pdf',
    short_label='Offshore Wind',
    year_palette='wind',
    year_highlight_recent=0,
    pdf_value_yaxis_max=500_000.0,
    pdf_price_yaxis_max=250.0,
    pdf_value_callout_y=460_000.0,
    pdf_price_callout_y=235.0,
    tech_col_prefix='Wind',
    capacity_curve_end=pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'),
)


TECHS: dict[str, TechConfig] = {
    TECH_SOLAR_PV.slug: TECH_SOLAR_PV,
    TECH_WIND_ONSHORE.slug: TECH_WIND_ONSHORE,
    TECH_WIND_OFFSHORE.slug: TECH_WIND_OFFSHORE,
}
