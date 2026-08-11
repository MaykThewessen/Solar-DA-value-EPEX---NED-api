"""Unified market-prices dashboard for Solar PV / Wind Onshore / Wind Offshore (NL).

Replaces the three near-identical scripts:

    Dashboard_Solar_PV_market_prices_NL.py     (1004 LOC)
    Dashboard_Wind_Onshore_market_prices_NL.py ( 876 LOC)
    Dashboard_Wind_Offshore_market_prices_NL.py( 868 LOC)

with a single parametric implementation driven by `tech_configs.TechConfig`.

Usage
-----
    pixi run python Dashboard_market_prices_NL.py solar_pv
    pixi run python Dashboard_market_prices_NL.py wind_onshore wind_offshore
    pixi run python Dashboard_market_prices_NL.py --all
"""
from __future__ import annotations

import argparse
import html
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from data_loader import (
    interp_capacity,
    load_capacity_points,
    load_da_prices,
)
from dashboard_common import (
    CLAUDE_PALETTE,
    add_dt_hours,
    compact_figure_arrays,
    fmt_table_value,
    build_monthly_metric_by_year_fig,
    build_themed_slide_fig,
    build_yearly_summary_table_fig,
    last_complete_month_end,
    last_complete_year,
    render_slides_to_pdf,
    utc_today_str,
    vw_price_groupby,
    year_color_map,
)
from tech_configs import TECHS, TechConfig


# ----------------------------------------------------------------------- pipeline

def _load_combined(cfg: TechConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load + merge DA prices and tech generation, then attach installed_capacity_MW.

    Returns (df_combined, capacity_anchors).
    """
    df_prices = load_da_prices(clip_future=cfg.clip_future_prices)
    df_tech = cfg.loader()

    df = pd.merge(df_prices, df_tech, on='time', how='left')
    # data_loader returns Europe/Amsterdam already; no second tz_convert needed.

    # Compute production-weighted value (`Solar_value` or `Wind_value`).
    value_col = f'{cfg.power_label}_value'
    df[value_col] = df[cfg.prod_col] * df['DA_price']

    # Sort + per-row interval (vectorised). `_dt_h` handles hourly vs quarterly.
    df = add_dt_hours(df)

    # Installed capacity via vectorised np.interp (replaces .apply lambda).
    anchors = load_capacity_points(cfg.capacity_csv)
    df['installed_capacity_MW'] = interp_capacity(df['time'], anchors)
    return df, anchors


def _print_capacity_banner(cfg: TechConfig, anchors: pd.DataFrame) -> None:
    print(f"Installed {cfg.name} capacity NL ({cfg.cap_unit}, year-end):")
    now = pd.Timestamp.now(tz='Europe/Amsterdam')
    for row in anchors.itertuples(index=False):
        status = 'actual' if row.date <= now else 'outlook'
        print(f"  {row.date.year}   {int(row.MW):>5,} MW  ({status})")


def _save_capacity_qa_plot(cfg: TechConfig, df: pd.DataFrame,
                           anchors: pd.DataFrame) -> None:
    """One-off matplotlib PDF: fitted capacity curve vs known anchor points."""
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df['time'].dt.tz_localize(None), df['installed_capacity_MW'],
             label='Fitted Installed Capacity (MW)', color='tab:blue')
    plt.scatter(
        [d.tz_localize(None) for d in anchors['date']],
        anchors['MW'].tolist(),
        color='tab:red', label='Known Data Points', zorder=5,
    )
    plt.title(f'Installed {cfg.name} Capacity NL: Fitted vs Known Data Points')
    plt.xlabel('Date')
    plt.ylabel('Installed Capacity (MW)')
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.out_capacity_pdf, bbox_inches='tight')
    plt.close(fig)


def _filter_complete_months(df: pd.DataFrame) -> pd.DataFrame:
    last_complete = last_complete_month_end()
    return df[df['time'] <= last_complete]


def _attach_month_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month'] = df['time'].dt.tz_localize(None).dt.to_period('M')
    df['month_date'] = df['month'].dt.to_timestamp()
    df['year'] = df['time'].dt.year
    return df


def _build_monthly_summary(df: pd.DataFrame, cfg: TechConfig) -> pd.DataFrame:
    """Monthly summary frame (one row per month) used for the monthly tables/HTML."""
    prod = cfg.prod_col
    val = f'{cfg.power_label}_value'
    weighted = f'{cfg.power_label}_Weighted_Price'

    base = df.groupby('month').agg({
        prod: 'sum',
        val: 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean',
    }).reset_index()

    base[f'Total_{cfg.power_label}_Energy_GWh'] = (base[prod] / 1000).round(1)
    base[f'Value_per_{cfg.cap_per_unit_short}_AC_EUR' if cfg.power_label == 'Wind'
         else 'Value_per_MWp_DC_EUR'] = (
        base[val] / base['installed_capacity_MW']
    ).round(1)
    base['Avg_DA_Price'] = base['DA_price'].round(1)
    base[weighted] = base['month'].map(
        vw_price_groupby(df, 'month', prod, 'DA_price')
    ).round(1)
    base['profile_factor'] = (
        (base[weighted] / base['Avg_DA_Price']) * 100
    ).round(1)
    # One capacity column for both technologies; the GWp-DC / GW-AC distinction
    # lives in the header text, not in a second column name.
    base['Installed_Capacity_GW'] = base['installed_capacity_MW'] / 1000

    # Excluding-negative-price metrics
    pos = df[df['DA_price'] >= 0].copy()
    pos['_pv'] = pos[prod] * pos['DA_price']
    pos_monthly = pos.groupby('month').agg(
        _prod_pos=(prod, 'sum'),
        _val_pos=(val, 'sum'),
        Avg_DA_Price_pos=('DA_price', 'mean'),
        _pv_sum=('_pv', 'sum'),
    ).reset_index()
    pos_monthly[f'{cfg.power_label}_Weighted_Price_excl_neg'] = (
        pos_monthly['_pv_sum']
        / pos_monthly['_prod_pos'].replace(0, np.nan)
    )
    base = base.merge(pos_monthly, on='month', how='left')

    base[f'MWh_per_{cfg.cap_per_unit_short}_excl_neg'] = (
        base['_prod_pos'] / base['installed_capacity_MW']
    )
    base[f'Value_per_{cfg.cap_per_unit_short}_AC_EUR_excl_neg' if cfg.power_label == 'Wind'
         else 'Value_per_MWp_DC_EUR_excl_neg'] = (
        base['_val_pos'] / base['installed_capacity_MW']
    )
    base['profile_factor_excl_neg'] = (
        base[f'{cfg.power_label}_Weighted_Price_excl_neg']
        / base['Avg_DA_Price_pos']
    ) * 100
    base['curtailment_pct'] = (
        (base[prod] - base['_prod_pos'].fillna(0)) / base[prod].replace(0, np.nan)
    ) * 100

    neg_monthly = (
        df[df['DA_price'] < 0]
        .groupby('month')['_dt_h'].sum().round(0)
        .reset_index().rename(columns={'_dt_h': 'neg_hours'})
    )
    base = base.merge(neg_monthly, on='month', how='left')
    base['neg_hours'] = base['neg_hours'].fillna(0)

    # Rename internal `_prod_pos` / `_val_pos` to legacy names for downstream code.
    base = base.rename(columns={
        '_prod_pos': f'{cfg.prod_col}_pos',
        '_val_pos': f'{cfg.power_label}_value_pos',
    })

    return base


def _build_yearly_totals(df: pd.DataFrame, cfg: TechConfig) -> pd.DataFrame:
    """Yearly summary frame indexed by year. Mirrors legacy `yearly_totals` layout."""
    prod = cfg.prod_col
    val = f'{cfg.power_label}_value'
    weighted = f'{cfg.power_label}_Weighted_Price'

    # Single multi-column agg instead of 4 separate groupby calls (S4).
    by_year = df.groupby('year').agg(
        Yearly_Energy=(prod, 'sum'),
        Yearly_Installed_Capacity_MW=('installed_capacity_MW', 'mean'),
        Yearly_Total_Value=(val, 'sum'),
        Yearly_Avg_DA_Price=('DA_price', 'mean'),
    ).reset_index()

    by_year[f'Yearly_{cfg.power_label}_Energy_MWh'] = by_year['Yearly_Energy']
    by_year[f'Yearly_Value_per_{cfg.cap_per_unit_short}_AC_EUR' if cfg.power_label == 'Wind'
            else 'Yearly_Value_per_MWp_DC_EUR'] = (
        by_year['Yearly_Total_Value'] / by_year['Yearly_Installed_Capacity_MW']
    )
    by_year[f'Yearly_{cfg.power_label}_Weighted_Price'] = by_year['year'].map(
        vw_price_groupby(df, 'year', prod, 'DA_price')
    )
    by_year['Yearly_Profile_Factor'] = (
        (by_year[f'Yearly_{cfg.power_label}_Weighted_Price']
         / by_year['Yearly_Avg_DA_Price']) * 100
    )

    # ------- excl-negative-price branch
    df_pos = df[df['DA_price'] >= 0]
    df_neg = df[df['DA_price'] < 0]
    yearly_pos = df_pos.groupby('year').agg(
        Solar_or_Wind_production_MWh_pos=(prod, 'sum'),
        Solar_or_Wind_value_pos=(val, 'sum'),
        Avg_DA_Price_pos=('DA_price', 'mean'),
    ).reset_index()
    # Numerator: sum(prod*price) by year, restricted to positive-price rows.
    numerator = (
        df_pos.assign(_pv=df_pos[prod] * df_pos['DA_price'])
              .groupby('year')['_pv'].sum()
              .reindex(yearly_pos['year']).to_numpy()
    )
    denom = yearly_pos['Solar_or_Wind_production_MWh_pos'].replace(0, np.nan).to_numpy()
    yearly_pos[f'Yearly_{cfg.power_label}_Weighted_Price_excl_neg'] = numerator / denom

    # Rename generic alias to tech-specific cols used downstream.
    yearly_pos = yearly_pos.rename(columns={
        'Solar_or_Wind_production_MWh_pos': f'{cfg.prod_col}_pos',
        'Solar_or_Wind_value_pos': f'{cfg.power_label}_value_pos',
    })
    by_year = by_year.merge(yearly_pos, on='year', how='left')

    by_year[f'Yearly_MWh_per_{cfg.cap_per_unit_short}_excl_neg'] = (
        by_year[f'{cfg.prod_col}_pos'] / by_year['Yearly_Installed_Capacity_MW']
    )
    by_year[f'Yearly_Value_per_{cfg.cap_per_unit_short}_AC_EUR_excl_neg' if cfg.power_label == 'Wind'
            else 'Yearly_Value_per_MWp_DC_EUR_excl_neg'] = (
        by_year[f'{cfg.power_label}_value_pos']
        / by_year['Yearly_Installed_Capacity_MW']
    )
    by_year['Yearly_Profile_Factor_excl_neg'] = (
        by_year[f'Yearly_{cfg.power_label}_Weighted_Price_excl_neg']
        / by_year['Avg_DA_Price_pos']
    ) * 100
    by_year['Yearly_Curtailment_Pct'] = (
        (by_year[f'Yearly_{cfg.power_label}_Energy_MWh'] - by_year[f'{cfg.prod_col}_pos'])
        / by_year[f'Yearly_{cfg.power_label}_Energy_MWh']
    ) * 100

    yearly_neg_hours = (
        df_neg.groupby('year')['_dt_h'].sum().round(0)
              .reset_index().rename(columns={'_dt_h': 'Yearly_Neg_Hours'})
    )
    by_year = by_year.merge(yearly_neg_hours, on='year', how='left')
    by_year['Yearly_Neg_Hours'] = by_year['Yearly_Neg_Hours'].fillna(0)

    return by_year


def _make_year_summary_for_table(by_year: pd.DataFrame, df: pd.DataFrame,
                                 cfg: TechConfig) -> tuple[pd.DataFrame, int]:
    """Add display columns (year_label, MWh-per-unit, GWh/TWh, etc.).

    Returns (summary frame, last fully-observed calendar year).
    """
    yst = by_year.copy()
    yst[f'Yearly_{cfg.power_label}_Energy_GWh'] = yst[f'Yearly_{cfg.power_label}_Energy_MWh'] / 1000
    yst[f'Yearly_{cfg.power_label}_Energy_TWh'] = yst[f'Yearly_{cfg.power_label}_Energy_MWh'] / 1_000_000

    # Every per-unit metric divides by the same denominator: the time-average
    # installed capacity over the year. Solar used to divide the incl-neg yield
    # by the July-1 interpolated capacity while its excl-neg yield, market value
    # and capture columns used the yearly average, so the two yield columns sat
    # side by side without being comparable. The displayed capacity column is now
    # that denominator, for both technologies.
    if cfg.power_label == 'PV':
        yst['Yearly_Installed_Capacity_GWp_DC'] = yst['Yearly_Installed_Capacity_MW'] / 1000
        yst['Yearly_MWh_per_MWp'] = (
            yst['Yearly_PV_Energy_MWh'] / yst['Yearly_Installed_Capacity_MW']
        )
    else:
        yst['Yearly_Installed_Capacity_MW_AC'] = yst['Yearly_Installed_Capacity_MW']
        yst['Yearly_Installed_Capacity_GW_AC'] = yst['Yearly_Installed_Capacity_MW'] / 1000
        yst['Yearly_MWh_per_MW'] = yst['Yearly_Wind_Energy_MWh'] / yst['Yearly_Installed_Capacity_MW_AC']

    yst = yst.round(1)

    last_complete_y = last_complete_year(df['time'].max())
    yst['year_label'] = yst['year'].apply(
        lambda y: f"{int(y)} *" if y > last_complete_y else str(int(y))
    )
    return yst, last_complete_y


# ----------------------------------------------------------------------- subplots HTML

def _write_subplot_html(df: pd.DataFrame, monthly: pd.DataFrame, yst: pd.DataFrame,
                        anchors: pd.DataFrame, cfg: TechConfig) -> None:
    """Build the seven-row subplot HTML matching the legacy file layout per tech."""
    is_solar = (cfg.power_label == 'PV')

    if is_solar:
        subplot_titles = (
            'Hourly PV Power Output in NL',
            'Total PV Yield in NL',
            'Yield normalized per installed capacity',
            'Market Value per installed capacity',
            f'{cfg.capture_metric_name} rate (%)',
            f'{cfg.capture_metric_name} price (€/MWh)',
            ' ',
        )
        row_heights = [0.12] * 6 + [0.28]
        vertical_spacing = 0.06
        specs = [
            [{'secondary_y': False}], [{'secondary_y': False}],
            [{'secondary_y': False}], [{'secondary_y': False}],
            [{'secondary_y': False}], [{'secondary_y': False}],
            [{'type': 'table'}],
        ]
    else:
        subplot_titles = (
            f'Total {cfg.short_label} Yield in NL',
            f'Installed {cfg.short_label} Capacity NL',
            'Yield normalized per installed capacity',
            'Market Value per installed capacity',
            f'{cfg.capture_metric_name} rate (%)',
            f'{cfg.capture_metric_name} price (€/MWh)',
            ' ',
        )
        row_heights = [0.20] * 6 + [0.30]
        vertical_spacing = 0.08
        specs = [
            [{'secondary_y': True}], [{'secondary_y': False}],
            [{'secondary_y': False}], [{'secondary_y': False}],
            [{'secondary_y': False}], [{'secondary_y': False}],
            [{'type': 'table'}],
        ]

    fig = make_subplots(
        rows=7, cols=1, shared_xaxes=False,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
        specs=specs,
        row_heights=row_heights,
    )

    years_sorted = sorted(monthly['year'].unique())
    color_map = year_color_map(
        years_sorted,
        palette=cfg.year_palette,
        highlight_recent=cfg.year_highlight_recent,
    )

    _add_yearly_table(fig, yst, cfg)
    if is_solar:
        _add_solar_subplot_row1(fig, df, anchors)
    else:
        _add_wind_subplot_row1_energy(fig, monthly, years_sorted, color_map)
        _add_wind_subplot_row2_capacity(fig, df, anchors, cfg)

    _add_remaining_monthly_lines(fig, monthly, years_sorted, color_map, cfg, is_solar)

    fig.update_layout(
        height=1800,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
    )

    compact_figure_arrays(fig)
    fig.write_html(cfg.out_production_html, auto_open=False, include_plotlyjs='cdn')


def _add_solar_subplot_row1(fig: go.Figure, df: pd.DataFrame,
                            anchors: pd.DataFrame) -> None:
    hourly = df[['time', 'Solar_production_MWh']].copy()
    hourly['Hourly_PV_Power_GW'] = hourly['Solar_production_MWh'] * 4 / 1000
    fig.add_trace(
        go.Scatter(
            x=hourly['time'], y=hourly['Hourly_PV_Power_GW'],
            mode='lines', name='Hourly PV Power',
            line=dict(color='blue', width=0.5),
            fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.4)',
            opacity=0.9, showlegend=False,
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=anchors['date'], y=anchors['MW'] / 1000,
            mode='lines+markers', name='GWp PV in NL',
            line=dict(color='red', width=2, dash='dot'),
            marker=dict(color='red', size=6),
            showlegend=True,
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.update_yaxes(title_text='Power (GW)', row=1, col=1)
    fig.update_xaxes(title_text='', row=1, col=1)


def _yearly_metric_columns(cfg: TechConfig) -> list[dict[str, Any]]:
    """The 14 yearly metrics in display order — the one spec both tables render.

    The subplot-HTML table and the themed PDF table used to carry separate
    hardcoded header and format lists. They drifted: the HTML truncated where the
    PDF rounded, used different thousands separators, and labelled the capacity
    column "Average" for solar (which showed a mid-year value) and "mid-year" for
    wind (which showed an average) — exactly backwards.

    Each entry: `col` (column in `yst`), `fmt` (format spec, see
    `fmt_table_value`), `html` / `pdf` header markup, `width` (PDF only).
    """
    is_solar = (cfg.power_label == 'PV')
    tech = cfg.power_label                          # 'PV' | 'Wind'
    unit = cfg.cap_per_unit_short                   # 'MWp' | 'MW'
    short = cfg.short_label
    capture = cfg.capture_metric_name
    cap_col = ('Yearly_Installed_Capacity_GWp_DC' if is_solar
               else 'Yearly_Installed_Capacity_GW_AC')
    cap_unit = cfg.cap_unit_short                   # 'GWp' | 'GW'
    # Wind capacity is small enough (1-8 GW) that a second decimal carries real
    # information; solar spans 3-31 GWp where it would be noise.
    cap_fmt = '{:,.1f}' if is_solar else '{:,.2f}'
    value_col = ('Yearly_Value_per_MWp_DC_EUR' if is_solar
                 else 'Yearly_Value_per_MW_AC_EUR')
    grey = '<span style="font-size:10px;color:#8C8377">'

    def spec(col, fmt, html_header, pdf_header, width):
        return dict(col=col, fmt=fmt, html=html_header, pdf=pdf_header, width=width)

    return [
        spec('year_label', '{}',
             'Year (* = preliminary)',
             f'Year<br>{grey}(* preliminary)</span>', 60),
        spec(cap_col, cap_fmt,
             f'Installed {short} capacity in NL ({cap_unit}) yearly average',
             f'Installed {short}<br>capacity ({cap_unit})<br>{grey}yearly avg</span>', 80),
        spec(f'Yearly_{tech}_Energy_TWh', '{:,.1f}',
             f'{short} Energy produced (TWh/y) (NED.nl)',
             f'{short} energy<br>(TWh/y)<br>{grey}NED.nl</span>', 70),
        spec(f'Yearly_MWh_per_{unit}', '{:,.0f}',
             f'MWh yield / {unit} installed',
             f'MWh / {unit}<br>installed', 70),
        spec(f'Yearly_MWh_per_{unit}_excl_neg', '{:,.0f}',
             f'MWh yield / {unit} (excl. neg)',
             f'MWh / {unit}<br>{grey}excl. neg</span>', 75),
        spec('Yearly_Curtailment_Pct', '{:.0f}%',
             'Curtailment (%)',
             'Curtailment<br>(%)', 65),
        spec('Yearly_Neg_Hours', '{:,.0f}',
             'Negative-price hours (h/y)',
             'Neg-price<br>hours (h/y)', 70),
        spec(value_col, '{:,.0f}',
             f'Annual Market value (EUR/{unit}/y)',
             f'Market value<br>(€/{unit}/y)', 80),
        spec(f'{value_col}_excl_neg', '{:,.0f}',
             f'Market value EUR/{unit}/y (excl. neg)',
             f'Market value<br>(€/{unit}/y)<br>{grey}excl. neg</span>', 85),
        spec('Yearly_Avg_DA_Price', '{:,.0f}',
             'Day-Ahead linear avg price (EUR/MWh)',
             'DA avg price<br>(€/MWh)', 75),
        spec(f'Yearly_{tech}_Weighted_Price', '{:,.0f}',
             f'{capture} price (€/MWh)',
             'Capture price<br>(€/MWh)', 75),
        spec(f'Yearly_{tech}_Weighted_Price_excl_neg', '{:,.0f}',
             f'{capture} price (€/MWh) excl. neg',
             f'Capture price<br>(€/MWh)<br>{grey}excl. neg</span>', 80),
        spec('Yearly_Profile_Factor', '{:.0f}%',
             f'{capture} rate (%)',
             'Capture rate<br>(%)', 70),
        spec('Yearly_Profile_Factor_excl_neg', '{:.0f}%',
             f'{capture} rate (%) excl. neg',
             f'Capture rate<br>(%)<br>{grey}excl. neg</span>', 80),
    ]


def _add_yearly_table(fig: go.Figure, yst: pd.DataFrame, cfg: TechConfig) -> None:
    """Row-7 summary table of the subplot HTML, newest year first."""
    columns = _yearly_metric_columns(cfg)
    cells_values = [
        [fmt_table_value(v, c['fmt']) for v in yst[c['col']][::-1]]
        for c in columns
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=[c['html'] for c in columns],
                        font=dict(size=10), align='left'),
            cells=dict(values=cells_values, font=dict(size=9), align='left', height=20),
        ),
        row=7, col=1,
    )


def _add_wind_subplot_row1_energy(fig: go.Figure, monthly: pd.DataFrame,
                                  years_sorted: list[int],
                                  color_map: dict[int, str]) -> None:
    for year in years_sorted:
        year_data = monthly[monthly['year'] == year].copy()
        year_data['month_num'] = year_data['month_date'].dt.month
        year_data = year_data.sort_values('month_num')
        fig.add_trace(
            go.Scatter(
                x=year_data['month_num'],
                y=year_data['Monthly_Wind_Energy_MWh'] / 1000,
                name=f'{year}', mode='lines+markers',
                line=dict(width=2), marker=dict(size=6),
                hovertemplate=(
                    '<b>Year:</b> %{fullData.name}<br>'
                    '<b>Month:</b> %{customdata}<br>'
                    '<b>Energy:</b> %{y:.1f} GWh<extra></extra>'
                ),
                customdata=year_data['month_date'].dt.strftime('%B'),
                legendgroup=f'year_{year}',
                showlegend=True,
                line_color=color_map[year],
            ),
            row=1, col=1, secondary_y=False,
        )
    fig.update_yaxes(title_text='GWh produced', row=1, col=1, secondary_y=False)
    fig.update_xaxes(
        title_text='', row=1, col=1, tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        range=[0.5, 12.5],
    )


def _add_wind_subplot_row2_capacity(fig: go.Figure, df: pd.DataFrame,
                                    anchors: pd.DataFrame, cfg: TechConfig) -> None:
    # Fitted curve on a daily grid spanning the anchor series itself, so the
    # subplot always covers exactly the range the capacity CSV describes.
    date_range = pd.date_range(start=anchors['date'].min(),
                               end=anchors['date'].max(), freq='D')
    fitted_capacity = interp_capacity(pd.Series(date_range), anchors)

    fig.add_trace(
        go.Scatter(
            x=date_range, y=fitted_capacity,
            mode='lines', name='Fitted Capacity',
            line=dict(color='red', width=2, dash='dot'),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=anchors['date'], y=anchors['MW'],
            mode='markers', name='Actual Capacity Points',
            marker=dict(color='red', size=8, symbol='circle'),
        ),
        row=2, col=1,
    )
    # Hourly Wind Production as raw MW (MWh per 15-min × 4)
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['Wind_production_MWh'] * 4,
            mode='lines', name=f'{cfg.short_label} Production',
            line=dict(color='blue', width=2),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text='Power (MW AC)', row=2, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=1)


def _add_remaining_monthly_lines(fig: go.Figure, monthly: pd.DataFrame,
                                 years_sorted: list[int],
                                 color_map: dict[int, str],
                                 cfg: TechConfig, is_solar: bool) -> None:
    """Add the four monthly metric subplots (rows 2..6 for solar, rows 3..6 for wind).

    Row 2 (solar energy) keeps per-year lines so seasonal shape is visible.
    Rows 3-6 (yield, market value, capture rate, capture price) show all months
    concatenated on a continuous time axis.
    """
    energy_col = f'Monthly_{cfg.power_label}_Energy_MWh'
    cap_col = 'Monthly_Installed_Capacity_MW'
    value_col = f'Monthly_Value_per_{cfg.cap_per_unit_short}_AC_EUR' if cfg.power_label == 'Wind' \
                else 'Monthly_Value_per_MWp_DC_EUR'
    weighted_col = f'Monthly_{cfg.power_label}_Power_Weighted_DA_Price'

    row_yield = 3
    row_value = 4
    row_pf = 5
    row_cp = 6

    month_ticks = list(range(1, 13))
    month_text = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Row 2 (Solar only): PV Energy GWh — per-year lines for seasonal comparison.
    if is_solar:
        for year in years_sorted:
            yd = monthly[monthly['year'] == year].copy()
            yd['month_num'] = yd['month_date'].dt.month
            yd = yd.sort_values('month_num')
            fig.add_trace(
                go.Scatter(
                    x=yd['month_num'], y=yd[energy_col] / 1000,
                    name=f'{year}', mode='lines+markers',
                    line=dict(width=2), marker=dict(size=6),
                    hovertemplate=(
                        '<b>Year:</b> %{fullData.name}<br>'
                        '<b>Month:</b> %{customdata}<br>'
                        '<b>Energy:</b> %{y:.1f} GWh<extra></extra>'
                    ),
                    customdata=yd['month_date'].dt.strftime('%B'),
                    legendgroup=f'year_{year}', showlegend=True,
                    line_color=color_map[year],
                ),
                row=2, col=1, secondary_y=False,
            )
        fig.update_yaxes(title_text='GWh produced', row=2, col=1, secondary_y=False)
        fig.update_xaxes(title_text='', row=2, col=1, tickmode='array',
                         tickvals=month_ticks, ticktext=month_text, range=[0.5, 12.5])

    # Rows 3-6: continuous time series across all months.
    ts = monthly.sort_values('month_date').copy()
    x_dates = ts['month_date'].tolist()
    hover_labels = ts['month_date'].dt.strftime('%b %Y').tolist()
    blue = CLAUDE_PALETTE['blue']

    # Row 3: yield (MWh per MWp/MW)
    unit = 'MWh/MWp' if is_solar else 'MWh/MW'
    fig.add_trace(
        go.Scatter(
            x=x_dates, y=ts[energy_col] / ts[cap_col],
            name=unit, mode='lines+markers',
            line=dict(width=2, color=blue), marker=dict(size=4),
            hovertemplate=(
                '<b>Month:</b> %{customdata}<br>'
                f'<b>Yield:</b> %{{y:.1f}} {unit}<extra></extra>'
            ),
            customdata=hover_labels,
            showlegend=True,
        ),
        row=row_yield, col=1,
    )
    fig.update_yaxes(title_text='MWh per MWp' if is_solar else 'MWh per MW',
                     row=row_yield, col=1)
    fig.update_xaxes(title_text='', row=row_yield, col=1, type='date')

    # Row 4: market value
    unit_val = '€/MWp' if is_solar else '€/MW'
    fig.add_trace(
        go.Scatter(
            x=x_dates, y=ts[value_col],
            name=unit_val, mode='lines+markers',
            line=dict(width=2, color=blue), marker=dict(size=4),
            hovertemplate=(
                '<b>Month:</b> %{customdata}<br>'
                f'<b>Market Value:</b> €%{{y:.1f}} {unit_val}<extra></extra>'
            ),
            customdata=hover_labels,
            showlegend=True,
        ),
        row=row_value, col=1,
    )
    fig.update_yaxes(title_text='€ per MWp' if is_solar else '€ per MW',
                     row=row_value, col=1)
    fig.update_xaxes(title_text='', row=row_value, col=1, type='date')

    # Row 5: capture rate
    fig.add_trace(
        go.Scatter(
            x=x_dates, y=ts['Monthly_Profile_Factor'],
            name=f'{cfg.capture_metric_name} rate (%)', mode='lines+markers',
            line=dict(width=2, color=blue), marker=dict(size=4),
            hovertemplate=(
                '<b>Month:</b> %{customdata}<br>'
                '<b>Capture rate:</b> %{y:.1f}%<extra></extra>'
            ),
            customdata=hover_labels,
            showlegend=True,
        ),
        row=row_pf, col=1,
    )
    fig.update_yaxes(title_text=f'{cfg.capture_metric_name} rate (%)',
                     row=row_pf, col=1, range=[0, 150])
    fig.update_xaxes(title_text='', row=row_pf, col=1, type='date')

    # Row 6: capture price
    fig.add_trace(
        go.Scatter(
            x=x_dates, y=ts[weighted_col],
            name=f'{cfg.capture_metric_name} price (€/MWh)', mode='lines+markers',
            line=dict(width=2, color=blue), marker=dict(size=4),
            hovertemplate=(
                '<b>Month:</b> %{customdata}<br>'
                '<b>Capture Price:</b> €%{y:.1f}/MWh<extra></extra>'
            ),
            customdata=hover_labels,
            showlegend=True,
        ),
        row=row_cp, col=1,
    )
    fig.update_yaxes(title_text=f'{cfg.capture_metric_name} price (€/MWh)',
                     row=row_cp, col=1)
    fig.update_xaxes(title_text='', row=row_cp, col=1, type='date')


# ----------------------------------------------------------------------- monthly table HTML

# --------------------------------------------------------------- HTML table UI

# Standalone, dependency-free styling for the monthly summary table page.
# Warm "Claude" brand palette (matches the PDF slides); theme-aware via
# prefers-color-scheme with a manual override toggle (data-theme on <html>).
_TABLE_CSS = """
*{box-sizing:border-box}
:root{
  --bg:#FAF9F5; --panel:#FFFFFF; --panel2:#F3F0E8; --edge:#E5DFD0;
  --ink:#1F1E1D; --ink-soft:#3D3929; --muted:#8C8377; --accent:#C96442;
  --head-bg:#1F1E1D; --head-ink:#FAF9F5; --row-alt:#FBFAF7; --row-hover:#F1ECE1;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 10px 34px rgba(31,30,29,.07);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#1A1917; --panel:#242220; --panel2:#2A2825; --edge:#3A362F;
  --ink:#F0EEE8; --ink-soft:#C9C4B8; --muted:#94897B; --accent:#E7A87C;
  --head-bg:#0F0E0D; --head-ink:#F0EEE8; --row-alt:#201E1C; --row-hover:#2E2B27;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 34px rgba(0,0,0,.5);
}}
:root[data-theme="dark"]{
  --bg:#1A1917; --panel:#242220; --panel2:#2A2825; --edge:#3A362F;
  --ink:#F0EEE8; --ink-soft:#C9C4B8; --muted:#94897B; --accent:#E7A87C;
  --head-bg:#0F0E0D; --head-ink:#F0EEE8; --row-alt:#201E1C; --row-hover:#2E2B27;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 34px rgba(0,0,0,.5);
}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;font-size:15px;line-height:1.45}
.wrap{max-width:1500px;margin:0 auto;padding:32px 24px 56px}
header.page{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:22px}
h1{font-size:1.55rem;font-weight:650;letter-spacing:-.015em;margin:0 0 6px;color:var(--ink)}
.subtitle{color:var(--ink-soft);font-size:.95rem;max-width:70ch;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{display:inline-flex;align-items:center;gap:6px;background:var(--panel2);
  border:1px solid var(--edge);border-radius:999px;padding:5px 12px;font-size:.8rem;color:var(--ink-soft)}
.chip b{color:var(--accent);font-weight:650;font-variant-numeric:tabular-nums}
#themeToggle{flex:none;background:var(--panel2);border:1px solid var(--edge);color:var(--ink-soft);
  border-radius:10px;width:40px;height:40px;font-size:1.1rem;cursor:pointer;line-height:1;
  transition:background .15s,color .15s}
#themeToggle:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:16px;
  box-shadow:var(--shadow);overflow:hidden}
.table-wrap{overflow:auto;max-height:82vh}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums;
  font-feature-settings:"tnum" 1}
thead th{position:sticky;top:0;z-index:3;background:var(--head-bg);color:var(--head-ink);
  font-weight:600;font-size:.72rem;letter-spacing:.02em;text-transform:uppercase;
  padding:12px 12px;text-align:right;vertical-align:bottom;white-space:normal;min-width:74px;
  border-bottom:2px solid var(--head-bg)}
thead th.month{text-align:left;left:0;z-index:4}
tbody th.month{position:sticky;left:0;z-index:2;background:var(--panel);text-align:left;
  font-weight:600;color:var(--ink);white-space:nowrap}
th,td{padding:8px 12px;border-bottom:1px solid var(--edge)}
td.num{text-align:right;color:var(--ink-soft);white-space:nowrap}
tbody tr:nth-child(even) th.month,tbody tr:nth-child(even) td{background:var(--row-alt)}
tbody tr:nth-child(even) th.month{background:var(--row-alt)}
tbody tr:hover th.month,tbody tr:hover td{background:var(--row-hover)}
tr.year-sep th,tr.year-sep td{border-top:2px solid var(--accent)}
tfoot td{padding:14px 16px;color:var(--muted);font-size:.8rem;background:var(--panel2)}
.legend{display:flex;flex-wrap:wrap;gap:18px;align-items:center}
.legend .sw{display:inline-flex;align-items:center;gap:7px}
.legend i{width:26px;height:12px;border-radius:3px;display:inline-block}
.grad-cap{background:linear-gradient(90deg,rgba(214,69,58,.55),rgba(214,69,58,.05),rgba(46,160,67,.05),rgba(46,160,67,.5))}
.grad-bad{background:linear-gradient(90deg,rgba(214,69,58,.04),rgba(214,69,58,.55))}
@media (max-width:640px){.wrap{padding:18px 10px 40px}h1{font-size:1.2rem}}
"""

_HEAT_GREEN = (46, 160, 67)
_HEAT_RED = (214, 69, 58)


def _heat_style(kind: str, val: float | None, hi: float) -> str:
    """Inline background for a heat-mapped cell (rgba overlay, theme-safe).

    kind='diverge100' → green above the 100% baseload, red below (capture rate).
    kind='bad'        → sequential red scaled to the column max (curtailment etc).
    """
    if val is None or pd.isna(val):
        return ''
    if kind == 'diverge100':
        t = max(-1.0, min(1.0, (val - 100.0) / 40.0))
        r, g, b = _HEAT_GREEN if t >= 0 else _HEAT_RED
        alpha = abs(t) * 0.5
    else:  # 'bad' — 0 is neutral, column max is fully saturated
        span = hi if hi and hi > 0 else 1.0
        r, g, b = _HEAT_RED
        alpha = min(1.0, val / span) * 0.55
    if alpha < 0.02:
        return ''
    return f'background-color:rgba({r},{g},{b},{alpha:.3f});'


def _render_summary_table_html(
    title: str,
    subtitle: str,
    header_values: list[str],
    cells_values: list[list[str]],
    heat: dict[int, tuple[list[float], str]],
) -> str:
    """Render a styled, self-contained HTML page for the monthly summary table."""
    # Coerce every column to a plain list so positional/negative indexing is
    # safe (some columns arrive as pandas Series with a label index).
    cells = [list(col) for col in cells_values]
    n_cols = len(header_values)
    n_rows = len(cells[0]) if cells else 0

    heat_hi = {ci: max((v for v in vals if v is not None and not pd.isna(v)),
                       default=1.0)
               for ci, (vals, _) in heat.items()}

    thead = '<tr>' + ''.join(
        f'<th class="{"month" if ci == 0 else "num"}" scope="col">'
        f'{html.escape(h)}</th>'
        for ci, h in enumerate(header_values)
    ) + '</tr>'

    body_rows: list[str] = []
    prev_year: str | None = None
    for i in range(n_rows):
        month = str(cells[0][i])
        year = month[:4]
        tr_cls = ' class="year-sep"' if prev_year and year != prev_year else ''
        prev_year = year
        row = [f'<th scope="row" class="month">{html.escape(month)}</th>']
        for ci in range(1, n_cols):
            val = str(cells[ci][i])
            style = ''
            if ci in heat:
                style = _heat_style(heat[ci][1], heat[ci][0][i], heat_hi[ci])
            attr = f' style="{style}"' if style else ''
            row.append(f'<td class="num"{attr}>{html.escape(val)}</td>')
        body_rows.append(f'<tr{tr_cls}>' + ''.join(row) + '</tr>')

    latest = str(cells[0][0]) if n_rows else '—'
    earliest = str(cells[0][-1]) if n_rows else '—'
    chips = (
        f'<span class="chip">Coverage <b>{html.escape(earliest)}</b> → '
        f'<b>{html.escape(latest)}</b></span>'
        f'<span class="chip"><b>{n_rows}</b> months</span>'
        f'<span class="chip"><b>{n_cols}</b> metrics</span>'
    )

    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8"/>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        f'<title>{html.escape(title)}</title>\n'
        f'<style>{_TABLE_CSS}</style>\n</head>\n<body>\n'
        '<div class="wrap">\n'
        '<header class="page"><div>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p class="subtitle">{html.escape(subtitle)}</p>'
        f'<div class="chips">{chips}</div>'
        '</div>'
        '<button id="themeToggle" title="Toggle light / dark" aria-label="Toggle theme">◐</button>'
        '</header>\n'
        '<div class="card"><div class="table-wrap"><table>\n'
        f'<thead>{thead}</thead>\n'
        f'<tbody>\n{"".join(body_rows)}\n</tbody>\n'
        '<tfoot><tr><td colspan="' + str(n_cols) + '"><div class="legend">'
        '<span class="sw"><i class="grad-cap"></i>Capture rate: red below / green above the 100% baseload</span>'
        '<span class="sw"><i class="grad-bad"></i>Curtailment &amp; negative-price hours: deeper red = higher</span>'
        '<span>Source: EPEX day-ahead spot prices &times; NED.nl generation, NL.</span>'
        '</div></td></tr></tfoot>\n'
        '</table></div></div>\n</div>\n'
        '<script>'
        'var r=document.documentElement,b=document.getElementById("themeToggle");'
        'b.addEventListener("click",function(){'
        'var d=matchMedia("(prefers-color-scheme: dark)").matches,'
        'cur=r.getAttribute("data-theme")||(d?"dark":"light");'
        'r.setAttribute("data-theme",cur==="dark"?"light":"dark");});'
        '</script>\n'
        '</body>\n</html>\n'
    )


def _monthly_metric_columns(cfg: TechConfig) -> list[dict[str, Any]]:
    """The 13 monthly metrics in display order, for one technology.

    Each entry: `col` (column in the monthly summary), `fmt`, `header`, and an
    optional `heat` ('bad' | 'diverge100') marking it for background shading.
    Carrying `heat` on the column itself retires the old `{4: …, 5: …, 11: …}`
    index map, which silently mis-shaded any column that moved.
    """
    is_solar = (cfg.power_label == 'PV')
    tech = cfg.power_label                          # 'PV' | 'Wind'
    unit = cfg.cap_per_unit_short                   # 'MWp' | 'MW'
    short = cfg.short_label
    capture = cfg.capture_metric_name
    cap_unit = 'GWp DC' if is_solar else 'GW AC'
    cap_fmt = '{:,.1f}' if is_solar else '{:,.2f}'
    value_col = 'Value_per_MWp_DC_EUR' if is_solar else 'Value_per_MW_AC_EUR'

    def spec(col, fmt, header, heat=None):
        return dict(col=col, fmt=fmt, header=header, heat=heat)

    return [
        spec('month', '{}', 'Month'),
        spec('Installed_Capacity_GW', cap_fmt, f'{short} capacity NL ({cap_unit})'),
        spec(f'Total_{tech}_Energy_GWh', '{:,.0f}',
             f'{short} Energy produced (GWh/month) (NED.nl)'),
        spec(f'MWh_per_{unit}_excl_neg', '{:,.0f}', f'MWh yield / {unit} (excl. neg)'),
        spec('curtailment_pct', '{:.0f}%', 'Curtailment (%)', heat='bad'),
        spec('neg_hours', '{:,.0f}', 'Negative-price hours (h)', heat='bad'),
        # Per *month*, not per year: these are the month's revenue divided by
        # installed capacity. The header said "/year" while showing a twelfth of it.
        spec(value_col, '{:,.0f}', f'Market value {short} (EUR/{unit}/month)'),
        spec(f'{value_col}_excl_neg', '{:,.0f}',
             f'Market value EUR/{unit}/month (excl. neg)'),
        spec('Avg_DA_Price', '{:,.0f}', 'Day-Ahead average price (EUR/MWh)'),
        spec(f'{tech}_Weighted_Price', '{:,.0f}', f'{capture} price (€/MWh)'),
        spec(f'{tech}_Weighted_Price_excl_neg', '{:,.0f}',
             f'{capture} price (€/MWh) excl. neg'),
        spec('profile_factor', '{:.0f}%', f'{capture} rate (%)', heat='diverge100'),
        spec('profile_factor_excl_neg', '{:.0f}%', f'{capture} rate (%) excl. neg',
             heat='diverge100'),
    ]


def _write_monthly_table_html(monthly_summary: pd.DataFrame, cfg: TechConfig) -> None:
    """Standalone HTML with the monthly summary table only."""
    is_solar = (cfg.power_label == 'PV')
    rev = monthly_summary.sort_values('month', ascending=False).reset_index(drop=True)
    columns = _monthly_metric_columns(cfg)

    cells_values = [
        rev[c['col']].astype(str).tolist() if c['col'] == 'month'
        else [fmt_table_value(v, c['fmt']) for v in rev[c['col']]]
        for c in columns
    ]
    header_values = [c['header'] for c in columns]
    heat: dict[int, tuple[list[float], str]] = {
        i: (rev[c['col']].tolist(), c['heat'])
        for i, c in enumerate(columns) if c['heat']
    }
    title = (f'Monthly Summary Table — {cfg.short_label} NL '
             f'(EPEX day-ahead prices x NED.nl generation)')
    unit = 'MWp DC' if is_solar else 'MW AC'
    subtitle = (
        f'Monthly market value of Dutch {cfg.short_label} on the EPEX day-ahead '
        f'market, per {unit} installed. Newest month first; "excl. neg" variants '
        f'drop negative-price settlement periods. Capture rate is the '
        f'production-weighted price relative to the time-weighted baseload (100%).'
    )

    html_doc = _render_summary_table_html(
        title=title,
        subtitle=subtitle,
        header_values=header_values,
        cells_values=cells_values,
        heat=heat,
    )
    with open(cfg.out_monthly_table_html, 'w', encoding='utf-8') as fh:
        fh.write(html_doc)


# ----------------------------------------------------------------------- yearly HTML

def _write_yearly_slides_html(yst: pd.DataFrame, last_complete_y: int,
                              cfg: TechConfig) -> None:
    """Interactive HTML with a dropdown to switch between yearly slides."""
    yst_s = yst.sort_values('year').reset_index(drop=True)
    years = yst_s['year_label'].tolist()

    def _scatter(y, name, dash=None, color=None, fill=None, fillcolor=None):
        return go.Scatter(
            x=years, y=y, name=name, mode='lines+markers',
            line=dict(width=2, dash=dash) if dash else dict(width=2),
            marker=dict(size=8),
            line_color=color,
            fill=fill, fillcolor=fillcolor,
        )

    # Choose per-tech column names.
    if cfg.power_label == 'PV':
        cap_col = 'Yearly_Installed_Capacity_GWp_DC'
        cap_unit = 'GWp DC'
        energy_col = 'Yearly_PV_Energy_TWh'
        yield_incl = 'Yearly_MWh_per_MWp'
        yield_excl = 'Yearly_MWh_per_MWp_excl_neg'
        value_incl = 'Yearly_Value_per_MWp_DC_EUR'
        value_excl = 'Yearly_Value_per_MWp_DC_EUR_excl_neg'
        price_incl = 'Yearly_PV_Weighted_Price'
        price_excl = 'Yearly_PV_Weighted_Price_excl_neg'
        per_unit = 'MWp'
    else:
        cap_col = 'Yearly_Installed_Capacity_GW_AC'
        cap_unit = 'GW AC'
        energy_col = 'Yearly_Wind_Energy_TWh'
        yield_incl = 'Yearly_MWh_per_MW'
        yield_excl = 'Yearly_MWh_per_MW_excl_neg'
        value_incl = 'Yearly_Value_per_MW_AC_EUR'
        value_excl = 'Yearly_Value_per_MW_AC_EUR_excl_neg'
        price_incl = 'Yearly_Wind_Weighted_Price'
        price_excl = 'Yearly_Wind_Weighted_Price_excl_neg'
        per_unit = 'MW'

    twh_complete = yst_s[energy_col].where(yst_s['year'] <= last_complete_y)
    yield_incl_complete = yst_s[yield_incl].where(yst_s['year'] <= last_complete_y)
    yield_excl_complete = yst_s[yield_excl].where(yst_s['year'] <= last_complete_y)

    capture_lbl = cfg.capture_metric_name

    slides = [
        (f'Installed {cfg.short_label} capacity ({cap_unit})',
         [_scatter(yst_s[cap_col], 'Installed capacity', color='#d62728')],
         cap_unit.split()[0]),
        (f'{cfg.short_label} Energy produced (TWh/y) — NED.nl, complete years only',
         [_scatter(twh_complete, f'{cfg.power_label} energy', color='#1f77b4')],
         'TWh/y'),
        (f'MWh yield per {per_unit} installed (complete years only)',
         [_scatter(yield_incl_complete, f'MWh/{per_unit} (incl. neg)', color='#1f77b4'),
          _scatter(yield_excl_complete, f'MWh/{per_unit} (excl. neg)', color='#2ca02c', dash='dash')],
         f'MWh / {per_unit}'),
        ('Curtailment — share of MWh during DA<0 hours',
         [_scatter(yst_s['Yearly_Curtailment_Pct'], 'Curtailment %', color='#d62728')],
         '%'),
        ('Negative-price hours per year',
         [_scatter(yst_s['Yearly_Neg_Hours'], 'Hours with DA < 0', color='#9467bd')],
         'hours'),
        (f'Annual market value (€/{per_unit}/y)',
         [_scatter(yst_s[value_incl], 'Market value (incl. neg)', color='#1f77b4'),
          _scatter(yst_s[value_excl], 'Market value (excl. neg)', color='#2ca02c', dash='dash')],
         f'€ / {per_unit} / y'),
        (f'{cfg.capture_metric_name_title} price vs Day-Ahead average (€/MWh)',
         [_scatter(yst_s[price_incl], 'Capture price (incl. neg)', color='#1f77b4'),
          _scatter(yst_s[price_excl], 'Capture price (excl. neg)', color='#2ca02c', dash='dash'),
          _scatter(yst_s['Yearly_Avg_DA_Price'], 'Day-Ahead avg', color='#7f7f7f', dash='dot')],
         '€ / MWh'),
        (f'{capture_lbl} rate (%)',
         [_scatter(yst_s['Yearly_Profile_Factor_excl_neg'], 'Capture rate (excl. neg)',
                   color='#2ca02c', fill='tozeroy', fillcolor='rgba(44,160,44,0.25)'),
          _scatter(yst_s['Yearly_Profile_Factor'], 'Capture rate (incl. neg)',
                   color='#1f77b4', fill='tozeroy', fillcolor='rgba(31,119,180,0.35)')],
         '%'),
    ]

    slides_fig = go.Figure()
    trace_slide_idx: list[int] = []
    for s_idx, (_title, traces, _yaxis) in enumerate(slides):
        for t in traces:
            slides_fig.add_trace(t)
            trace_slide_idx.append(s_idx)
    for i, t in enumerate(slides_fig.data):
        t.visible = (trace_slide_idx[i] == 0)

    buttons = []
    for s_idx, (title, _traces, yaxis_title) in enumerate(slides):
        vis = [idx == s_idx for idx in trace_slide_idx]
        buttons.append(dict(
            label=title, method='update',
            args=[{'visible': vis},
                  {'title.text': title, 'yaxis.title.text': yaxis_title, 'yaxis.rangemode': 'tozero'}],
        ))

    slides_fig.update_layout(
        title=dict(text=slides[0][0]),
        yaxis=dict(title=dict(text=slides[0][2]), rangemode='tozero'),
        xaxis=dict(title='Year', type='category'),
        height=600,
        updatemenus=[dict(
            type='dropdown', direction='down', x=0.0, y=1.15,
            xanchor='left', yanchor='top', buttons=buttons, showactive=True,
        )],
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
    )
    compact_figure_arrays(slides_fig)
    slides_fig.write_html(cfg.out_yearly_html, auto_open=False, include_plotlyjs='cdn')


# ----------------------------------------------------------------------- PDF builder

def _build_pdf(yst: pd.DataFrame, monthly_summary: pd.DataFrame,
               last_complete_y: int, cfg: TechConfig) -> None:
    """Render the multi-page themed PDF."""
    yst_s = yst.sort_values('year').reset_index(drop=True)
    years_full = yst_s['year_label'].tolist()
    years_for_data = yst_s['year'].tolist()
    brand = cfg.brand
    source_date = utc_today_str()
    is_solar = (cfg.power_label == 'PV')

    cap_col = 'Yearly_Installed_Capacity_GWp_DC' if is_solar else 'Yearly_Installed_Capacity_GW_AC'
    energy_col = 'Yearly_PV_Energy_TWh' if is_solar else 'Yearly_Wind_Energy_TWh'
    per_unit = 'MWp' if is_solar else 'MW'
    cap_unit = 'GWp (DC)' if is_solar else 'GW (AC)'
    yield_incl = 'Yearly_MWh_per_MWp' if is_solar else 'Yearly_MWh_per_MW'
    yield_excl = f'{yield_incl}_excl_neg'
    value_incl = ('Yearly_Value_per_MWp_DC_EUR' if is_solar
                  else 'Yearly_Value_per_MW_AC_EUR')
    value_excl = f'{value_incl}_excl_neg'
    price_incl = f'Yearly_{cfg.power_label}_Weighted_Price'
    price_excl = f'{price_incl}_excl_neg'

    # 2022 spike callout (omitted if 2022 missing).
    yrs_set = set(yst_s['year'].tolist())
    callout_value: list[dict[str, Any]] = []
    callout_price: list[dict[str, Any]] = []
    if 2022 in yrs_set:
        v22 = float(yst_s.loc[yst_s['year'] == 2022, value_incl].iloc[0])
        v22x = float(yst_s.loc[yst_s['year'] == 2022, value_excl].iloc[0])
        callout_value = [dict(
            x='2022', y=cfg.pdf_value_callout_y,
            body=(f"<b>2022 spike</b><br>"
                  f"incl. neg: €{v22:,.0f}/{per_unit}<br>"
                  f"excl. neg: €{v22x:,.0f}/{per_unit}").replace(',', '.'),
        )]
        p22 = float(yst_s.loc[yst_s['year'] == 2022, price_incl].iloc[0])
        p22x = float(yst_s.loc[yst_s['year'] == 2022, price_excl].iloc[0])
        da22 = float(yst_s.loc[yst_s['year'] == 2022, 'Yearly_Avg_DA_Price'].iloc[0])
        callout_price = [dict(
            x='2022', y=cfg.pdf_price_callout_y,
            body=(f"<b>2022 spike</b><br>"
                  f"Capture incl. neg: €{p22:.0f}/MWh<br>"
                  f"Capture excl. neg: €{p22x:.0f}/MWh<br>"
                  f"DA avg: €{da22:.0f}/MWh"),
        )]

    # Slide definitions.
    pdf_slides: list[dict[str, Any]] = [
        dict(
            title=f'Installed {cfg.short_label} Capacity',
            subtitle=('Netherlands · NED.nl source · year-end GWp DC' if is_solar
                      else 'Netherlands · year-end GW AC (Birdview Central scenario)'),
            ytitle=cap_unit, kind='area_gradient',
            traces=[(yst_s[cap_col], 'Installed capacity', CLAUDE_PALETTE['accent'])],
            mask_partial=False, line_shape='spline',
        ),
        dict(
            title=f'{cfg.short_label} Energy Produced',
            subtitle=('Annual generation in TWh · complete years only' if is_solar
                      else 'Annual generation in TWh · complete years only · NED.nl'),
            ytitle='TWh / year', kind='bar_gradient',
            traces=[(yst_s[energy_col], f'{cfg.power_label} energy', CLAUDE_PALETTE['accent'])],
            mask_partial=True,
        ),
        dict(
            title='Specific Yield',
            subtitle=f'MWh produced per {per_unit} installed · with & without negative-price hours',
            ytitle=f'MWh / {per_unit}', kind='dual_bar_gradient',
            traces=[
                (yst_s[yield_incl], 'incl. neg-price hours', CLAUDE_PALETTE['blue']),
                (yst_s[yield_excl], 'excl. neg-price hours', CLAUDE_PALETTE['sage']),
            ],
            mask_partial=True,
        ),
        dict(
            title='Curtailment Share',
            subtitle=(f'Share of {cfg.power_label.lower()} MWh produced during DA < 0 €/MWh hours'),
            ytitle='% of yearly MWh', kind='bar_gradient',
            traces=[(yst_s['Yearly_Curtailment_Pct'], 'Curtailment', CLAUDE_PALETTE['accent'])],
            mask_partial=True,
        ),
        dict(
            title='Negative-Price Hours',
            subtitle='Hours per year with Day-Ahead price < 0 €/MWh',
            ytitle='Hours / year', kind='area_gradient',
            traces=[(yst_s['Yearly_Neg_Hours'], 'Negative-price hours', CLAUDE_PALETTE['sage'])],
            mask_partial=True,
        ),
        dict(
            title=f'Annual Market Value per {per_unit}',
            subtitle=f'Revenue per installed {per_unit} {"DC" if is_solar else "AC"} · with & without neg-price hours',
            ytitle=f'€ / {per_unit} / year', kind='dual_area',
            traces=[
                (yst_s[value_incl], 'incl. neg-price hours', CLAUDE_PALETTE['blue']),
                (yst_s[value_excl], 'excl. neg-price hours', CLAUDE_PALETTE['sage']),
            ],
            mask_partial=False,
            yaxis_range=(0, cfg.pdf_value_yaxis_max), yaxis_tickformat=',.0f',
            use_eu_thousands=True,
            callouts=callout_value,
        ),
        dict(
            title='Capture Price vs Day-Ahead Average',
            subtitle=(f'Volume-weighted {cfg.short_label.lower()} price compared with flat Day-Ahead average'
                      if not is_solar
                      else 'Volume-weighted solar price compared with flat Day-Ahead average'),
            ytitle='€ / MWh', kind='triple_line',
            traces=[
                (yst_s[price_incl], 'Capture price (incl. neg)', CLAUDE_PALETTE['blue']),
                (yst_s[price_excl], 'Capture price (excl. neg)', CLAUDE_PALETTE['sage']),
                (yst_s['Yearly_Avg_DA_Price'], 'Day-Ahead average', CLAUDE_PALETTE['muted']),
            ],
            mask_partial=False, yaxis_range=(0, cfg.pdf_price_yaxis_max),
            callouts=callout_price,
        ),
        dict(
            title=f'{cfg.capture_metric_name_title} Rate',
            subtitle='Capture price as % of Day-Ahead average · with & without neg-price hours',
            ytitle='%', kind='dual_area',
            traces=[
                (yst_s['Yearly_Profile_Factor'], 'incl. neg-price hours', CLAUDE_PALETTE['blue']),
                (yst_s['Yearly_Profile_Factor_excl_neg'], 'excl. neg-price hours', CLAUDE_PALETTE['sage']),
            ],
            mask_partial=False, legend_position='right',
            **({'label_trace_indices': [0], 'label_fmt': '{:.0f}%'} if is_solar else {}),
        ),
    ]

    slide_pages = [
        dict(shape='slide', fig=build_themed_slide_fig(
            s, years_full, years_for_data, int(last_complete_y), brand, source_date,
        ))
        for s in pdf_slides
    ]

    # Monthly capture-rate-by-year page: the three most recent years present in
    # the data. Hardcoding (2023, 2024, 2025) silently dropped each new year.
    if cfg.emit_monthly_pdf_page:
        months = monthly_summary['month'].astype(str)
        recent_years = sorted(months.str.slice(0, 4).astype(int).unique())[-3:]
        monthly_fig = build_monthly_metric_by_year_fig(
            monthly_summary, value_col='profile_factor',
            years_to_plot=recent_years,
            title=f'Monthly {cfg.capture_metric_name_title} Rate',
            subtitle='Netherlands · capture price ÷ Day-Ahead average · by year',
            ytitle='%', brand=brand, source_date=source_date,
            tick_suffix='%',
        )
        slide_pages.append(dict(shape='slide', fig=monthly_fig))

    # Final yearly-summary table page.
    table_columns = _build_table_columns(cfg)
    table_fig = build_yearly_summary_table_fig(
        yst_s, table_columns, brand=brand,
        title=(f'Yearly {cfg.short_label} Market Summary'
               if not is_solar
               else 'Yearly Solar PV Market Summary'),
        subtitle='Netherlands · Day-Ahead market · all metrics, with & without negative-price hours',
        source_date=source_date,
    )
    slide_pages.append(dict(shape='table', fig=table_fig))

    render_slides_to_pdf(slide_pages, cfg.out_yearly_pdf)
    print(f'PDF written: {cfg.out_yearly_pdf}')


def _build_table_columns(cfg: TechConfig) -> list[dict[str, Any]]:
    """Project the shared yearly spec onto `build_yearly_summary_table_fig`'s schema."""
    return [
        dict(col=c['col'], fmt=c['fmt'], header=c['pdf'], width=c['width'])
        for c in _yearly_metric_columns(cfg)
    ]


# ----------------------------------------------------------------------- monthly per-tech

def _build_monthly_per_tech_table(monthly: pd.DataFrame, cfg: TechConfig) -> pd.DataFrame:
    """Build the `monthly` DataFrame used by the line-per-year subplots.

    Adds: Monthly_<tech>_Energy_MWh, Monthly_Value_per_<unit>_AC_EUR (or _DC),
          Monthly_Installed_Capacity_MW, Monthly_Avg_DA_Price,
          Monthly_<tech>_Power_Weighted_DA_Price, Monthly_Profile_Factor.
    """
    out = monthly.copy()
    prod_col = cfg.prod_col
    val_col = f'{cfg.power_label}_value'
    weighted_out = f'Monthly_{cfg.power_label}_Power_Weighted_DA_Price'
    energy_out = f'Monthly_{cfg.power_label}_Energy_MWh'
    if cfg.power_label == 'PV':
        value_out = 'Monthly_Value_per_MWp_DC_EUR'
    else:
        value_out = f'Monthly_Value_per_{cfg.cap_per_unit_short}_AC_EUR'

    out[energy_out] = out[prod_col].round(1)
    out[value_out] = (out[val_col] / out['installed_capacity_MW']).round(1)
    out['Monthly_Installed_Capacity_MW'] = out['installed_capacity_MW']
    out['Monthly_Avg_DA_Price'] = out['DA_price'].round(1) if cfg.power_label == 'Wind' else out['DA_price']
    return out


# ----------------------------------------------------------------------- run one tech

def run_for(cfg: TechConfig) -> None:
    """End-to-end: load → derive → write three HTMLs + one matplotlib QA PDF + slide PDF."""
    print(f"\n=== {cfg.name} ({cfg.slug}) ===")

    df, anchors = _load_combined(cfg)

    # TODO(you): release the DuckDB read lock here — everything below this line
    # is pandas/plotly on in-memory frames and touches no database.
    # See data_loader.close().

    _print_capacity_banner(cfg, anchors)
    _save_capacity_qa_plot(cfg, df, anchors)

    # Filter to complete months. Solar legacy filters *after* building the
    # `monthly` table (and clips df_combined too); Wind filters first. Result
    # is identical: same set of complete months, same df_combined coverage.
    df_complete = _filter_complete_months(df)
    print(f"Date range: {df_complete['time'].min()} to {df_complete['time'].max()}")

    df_keys = _attach_month_keys(df_complete)
    monthly_summary = _build_monthly_summary(df_keys, cfg)

    print("\nMonthly Summary:" if cfg.power_label == 'Wind' else "\nMonthly Summary (Complete months only):")
    print(monthly_summary)

    # Year aggregations (single multi-column agg).
    yearly_totals = _build_yearly_totals(df_keys, cfg)

    yst, last_complete_y = _make_year_summary_for_table(yearly_totals, df_complete, cfg)

    # Rebuild a `monthly`-like frame (one row per month) for the subplot HTML.
    base_monthly = df_keys.groupby('month_date').agg({
        cfg.prod_col: 'sum',
        f'{cfg.power_label}_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean',
    }).reset_index()
    base_monthly['year'] = base_monthly['month_date'].dt.year
    base_monthly = _build_monthly_per_tech_table(base_monthly, cfg)
    # Add weighted price + profile factor.
    weighted_series = vw_price_groupby(df_keys, 'month_date', cfg.prod_col, 'DA_price')
    base_monthly[f'Monthly_{cfg.power_label}_Power_Weighted_DA_Price'] = (
        base_monthly['month_date'].map(weighted_series)
    )
    base_monthly['Monthly_Profile_Factor'] = (
        base_monthly[f'Monthly_{cfg.power_label}_Power_Weighted_DA_Price']
        / base_monthly['Monthly_Avg_DA_Price']
    ) * 100

    # HTMLs.
    _write_subplot_html(df_complete, base_monthly, yst, anchors, cfg)
    _write_monthly_table_html(monthly_summary, cfg)
    _write_yearly_slides_html(yst, last_complete_y, cfg)

    # Themed PDF.
    _build_pdf(yst, monthly_summary, last_complete_y, cfg)


# ----------------------------------------------------------------------- cli

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build NED.nl + EPEX market-prices dashboards.',
    )
    parser.add_argument(
        'techs', nargs='*', choices=list(TECHS) + ['all'],
        help='Tech slug(s) to run; or "all" / no args for everything.',
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all techs (same as passing every slug).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.all or not args.techs or 'all' in args.techs:
        selected = list(TECHS.values())
    else:
        selected = [TECHS[t] for t in args.techs]

    for cfg in selected:
        run_for(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
