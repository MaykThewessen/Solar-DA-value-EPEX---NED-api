"""Parametric dashboard: yearly profile factor vs installed capacity (NL).

Replaces three near-identical scripts:
  - Dashboard_PV_Profile_Factor_vs_Capacity.py
  - Dashboard_Wind_Onshore_Profile_Factor_vs_Capacity.py
  - Dashboard_Wind_Offshore_Profile_Factor_vs_Capacity.py

The profile factor for a generation technology is
    PF_year = volume_weighted_DA_price_year / arithmetic_mean_DA_price_year
expressed as a percentage. It captures how much of the average market
price the technology actually captures when generating.

Usage:
    pixi run python Dashboard_profile_factor_vs_capacity.py pv
    pixi run python Dashboard_profile_factor_vs_capacity.py wind_onshore wind_offshore
    pixi run python Dashboard_profile_factor_vs_capacity.py all
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from dashboard_common import compact_figure_arrays, last_complete_year, vw_price_groupby
from data_loader import (
    CAPACITY_CSV,
    load_capacity_points,
    load_da_prices,
    load_ned_pv,
    load_ned_wind_offshore,
    load_ned_wind_onshore,
)

TZ = 'Europe/Amsterdam'

# Distinct, distinguishable colors per year.
DISTINCT_COLORS: list[str] = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#ff9896', '#98df8a', '#ffbb78', '#aec7e8', '#c5b0d5',
]

CapacityFitMode = Literal['polyfit', 'piecewise']


@dataclass(frozen=True)
class ProfileFactorTechConfig:
    """Per-technology config for the profile-factor dashboard.

    Installed capacity is *not* configured here beyond a path: the anchors come
    from `data_loader.CAPACITY_CSV`, the same files the market-prices dashboard
    reads. The trend window's upper bound is derived from the data, not stored.
    """
    key: str                                    # 'pv' | 'wind_onshore' | 'wind_offshore'
    loader: Callable[[], pd.DataFrame]          # returns DataFrame with 'time' + production col
    production_col: str                         # column name in loader output ('Solar_production_MWh' / 'Wind_production_MWh')
    label_short: str                            # 'PV' | 'Onshore Wind' | 'Offshore Wind'
    label_long: str                             # 'Solar PV' | 'Onshore Wind' | 'Offshore Wind'
    capacity_unit: str                          # 'GWp DC' | 'GW AC'
    capacity_axis_label: str                    # axis title for capacity
    slope_unit: str                             # '%/GWp' | '%/GW'
    output_stem: str                            # 'pv_profile_factor_vs_capacity_dashboard' | ...
    capacity_csv: Path                          # (date, MW) anchors — see data_loader.CAPACITY_CSV
    capacity_fit_mode: CapacityFitMode          # 'polyfit' (global linear) | 'piecewise' (linear between anchors)
    trend_year_min: int                         # first year used for trend fit
    trend_year_excluded: int = 2022             # outlier year excluded from trend fit
    x_axis_range: tuple[float, float] = (0.0, 100.0)
    extrapolation_x_max: float = 200.0          # where to stop the exponential extrapolation
    callout_capacity_gw: float = 55.0           # capacity at which to annotate the exponential
    show_outlier_annotation: bool = False       # annotate 2022 'gas-crisis' point
    show_threshold_annotations: bool = False    # annotate per-threshold (5/2/1 %) crossings
    show_linear_zero_annotation: bool = True    # annotate where linear trend crosses 0%
    weighted_price_col: str = field(init=False)

    def __post_init__(self) -> None:
        # Derive a column label aligned with the original scripts.
        weighted = 'Yearly_PV_Weighted_Price' if self.key == 'pv' else 'Yearly_Wind_Weighted_Price'
        object.__setattr__(self, 'weighted_price_col', weighted)


# ---------------------------------------------------------------- tech configs

TECH_CONFIGS: dict[str, ProfileFactorTechConfig] = {
    'pv': ProfileFactorTechConfig(
        key='pv',
        loader=load_ned_pv,
        production_col='Solar_production_MWh',
        label_short='PV',
        label_long='Solar PV',
        capacity_unit='GWp DC',
        capacity_axis_label='Installed PV Capacity NL (GWp DC) yearly avg',
        slope_unit='%/GWp',
        output_stem='pv_profile_factor_vs_capacity_dashboard',
        capacity_csv=CAPACITY_CSV['solar_pv'],
        capacity_fit_mode='polyfit',
        trend_year_min=2018,
        x_axis_range=(0.0, 100.0),
        extrapolation_x_max=200.0,
        callout_capacity_gw=55.0,
        show_outlier_annotation=True,
        show_threshold_annotations=True,
    ),
    'wind_onshore': ProfileFactorTechConfig(
        key='wind_onshore',
        loader=load_ned_wind_onshore,
        production_col='Wind_production_MWh',
        label_short='Onshore Wind',
        label_long='Onshore Wind',
        capacity_unit='GW AC',
        capacity_axis_label='Installed Onshore Wind Capacity NL (GW AC) yearly avg',
        slope_unit='%/GW',
        output_stem='wind_onshore_profile_factor_vs_capacity_dashboard',
        capacity_csv=CAPACITY_CSV['wind_onshore'],
        capacity_fit_mode='piecewise',
        trend_year_min=2019,
        x_axis_range=(0.0, 50.0),
        extrapolation_x_max=50.0,
    ),
    'wind_offshore': ProfileFactorTechConfig(
        key='wind_offshore',
        loader=load_ned_wind_offshore,
        production_col='Wind_production_MWh',
        label_short='Offshore Wind',
        label_long='Offshore Wind',
        capacity_unit='GW AC',
        capacity_axis_label='Installed Offshore Wind Capacity NL (GW AC) yearly avg',
        slope_unit='%/GW',
        output_stem='wind_offshore_profile_factor_vs_capacity_dashboard',
        capacity_csv=CAPACITY_CSV['wind_offshore'],
        capacity_fit_mode='piecewise',
        trend_year_min=2019,
        x_axis_range=(0.0, 50.0),
        extrapolation_x_max=50.0,
    ),
}


# -------------------------------------------------------------- capacity model

@dataclass(frozen=True)
class CapacityModel:
    """Vectorised capacity-vs-time interpolator.

    `evaluate(times)` returns MW capacity at each input timestamp using either
    a single linear polyfit through all anchors (`polyfit`) or piecewise
    linear interpolation between anchors with linear extrapolation on the
    last segment (`piecewise`).
    """
    mode: CapacityFitMode
    x_anchors_i8: np.ndarray            # tz-aware int64 ns timestamps
    y_anchors: np.ndarray               # MW
    polyfit_coeffs: np.ndarray | None   # for mode='polyfit', length-2 [slope, intercept] on int64 ns

    @classmethod
    def build(cls, mode: CapacityFitMode, anchors: pd.DataFrame) -> 'CapacityModel':
        """Build from a `load_capacity_points` frame (columns `date`, `MW`, sorted asc)."""
        x = pd.DatetimeIndex(anchors['date']).view('i8').astype(np.int64)
        y = anchors['MW'].to_numpy(dtype=float)
        coeffs = np.polyfit(x.astype(np.float64), y, 1) if mode == 'polyfit' else None
        return cls(mode=mode, x_anchors_i8=x, y_anchors=y, polyfit_coeffs=coeffs)

    def evaluate(self, times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        idx = times if isinstance(times, pd.DatetimeIndex) else pd.DatetimeIndex(times)
        if idx.tz is None:
            idx = idx.tz_localize(TZ)
        xi = idx.view('i8').astype(np.int64)
        if self.mode == 'polyfit':
            assert self.polyfit_coeffs is not None
            return np.polyval(self.polyfit_coeffs, xi.astype(np.float64))
        # piecewise: np.interp clamps below first / above last, so handle
        # extrapolation on the trailing segment explicitly.
        result = np.interp(xi.astype(np.float64),
                           self.x_anchors_i8.astype(np.float64), self.y_anchors)
        last_x = float(self.x_anchors_i8[-1])
        prev_x = float(self.x_anchors_i8[-2])
        last_y = float(self.y_anchors[-1])
        prev_y = float(self.y_anchors[-2])
        if last_x != prev_x:
            slope = (last_y - prev_y) / (last_x - prev_x)
            after = xi > self.x_anchors_i8[-1]
            result = np.where(after, last_y + slope * (xi.astype(np.float64) - last_x), result)
        return result

    def evaluate_one(self, ts: pd.Timestamp) -> float:
        return float(self.evaluate(pd.DatetimeIndex([ts]))[0])


# -------------------------------------------------------------- core pipeline

def compute_yearly_plot_data(
    cfg: ProfileFactorTechConfig,
    model: CapacityModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_combined, plot_data).

    plot_data: one row per year with capacity (at July 1st) and profile factor.
    """
    df_prices = load_da_prices()
    df_gen = cfg.loader()

    df = pd.merge(df_prices, df_gen, on='time', how='left')

    # Ensure tz-aware Europe/Amsterdam.
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize(TZ)

    # V1 fix: vectorised capacity evaluation in place of df['time'].apply(...).
    df['installed_capacity_MW'] = model.evaluate(df['time'])

    # value = production * DA_price (revenue per slot).
    df['production_value_EUR'] = df[cfg.production_col] * df['DA_price']

    year_col = df['time'].dt.year
    df = df.assign(_year=year_col)

    yearly_totals = (
        df.groupby('_year', as_index=False)
          .agg(**{
              'Yearly_Production_MWh':        (cfg.production_col, 'sum'),
              'Yearly_Total_Production_Value': ('production_value_EUR', 'sum'),
              'Yearly_Installed_Capacity_MW': ('installed_capacity_MW', 'mean'),
              'Yearly_Avg_DA_Price':          ('DA_price', 'mean'),
          })
          .rename(columns={'_year': 'year'})
    )

    # V2 fix: vectorised volume-weighted price via shared helper, drops nested
    # df_combined lookups inside per-row .apply(lambda ...).
    weighted = vw_price_groupby(df, group_col='_year',
                                prod_col=cfg.production_col, price_col='DA_price')
    weighted = weighted.rename_axis('year').reset_index(name=cfg.weighted_price_col)

    yearly_totals = yearly_totals.merge(weighted, on='year')
    yearly_totals['Yearly_Profile_Factor'] = (
        yearly_totals[cfg.weighted_price_col] / yearly_totals['Yearly_Avg_DA_Price']
    ) * 100

    # July 1st capacity per year (vectorised via the same model).
    years = yearly_totals['year'].to_numpy()
    july_1st = pd.DatetimeIndex([pd.Timestamp(f'{y}-07-01', tz=TZ) for y in years])
    yearly_totals['Capacity_GW_July1'] = model.evaluate(july_1st) / 1000.0

    plot_data = (
        yearly_totals[['year', 'Capacity_GW_July1', 'Yearly_Profile_Factor']]
        .dropna()
        .copy()
    )
    return df, plot_data


# -------------------------------------------------------------- plotting

def _year_color_map(years: list[int]) -> dict[int, str]:
    return {y: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, y in enumerate(sorted(years))}


def _select_trend_data(plot_data: pd.DataFrame, cfg: ProfileFactorTechConfig,
                       trend_year_max: int) -> pd.DataFrame:
    """Years eligible for the trend fit: complete, in range, and not the outlier year."""
    mask = (
        (plot_data['year'] >= cfg.trend_year_min)
        & (plot_data['year'] <= trend_year_max)
        & (plot_data['year'] != cfg.trend_year_excluded)
    )
    return plot_data.loc[mask].copy()


def build_figure(
    plot_data: pd.DataFrame,
    cfg: ProfileFactorTechConfig,
    model: CapacityModel,
    trend_year_max: int,
) -> tuple[go.Figure, dict]:
    """Build the plotly figure. Returns (fig, summary_dict)."""
    summary: dict = {
        'correlation': None,
        'slope': None,
        'coeffs_linear': None,
        'x_trend_start': None,
        'y_trend_start': None,
        'jan_1_capacity_gw': None,
        'x_at_zero_linear': None,
        'a': None,
        'b': None,
        'trend_year_range_label': None,
        'profile_factor_at_callout': None,
    }

    fig = go.Figure()

    fig.update_layout(
        title=dict(
            text=(f'{cfg.label_short} Profile Factor vs Installed {cfg.label_long} '
                  f'Capacity in Netherlands<br>'
                  f'<sub>Each point represents year avg installed capacity vs yearly '
                  f'{cfg.label_short} profile factor</sub>'),
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text=cfg.capacity_axis_label, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            zeroline=False,
            range=list(cfg.x_axis_range),
        ),
        yaxis=dict(
            title=dict(text='Profile Factor (%)', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            zeroline=False,
            range=[0, 105],
        ),
        legend=dict(title='Year', font=dict(size=12)),
        plot_bgcolor='rgba(248, 248, 248, 1)',
        paper_bgcolor='rgba(248, 248, 248, 1)',
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80),
    )

    if len(plot_data) <= 2:
        _add_scatter_points(fig, plot_data, cfg)
        return fig, summary

    trend_data = _select_trend_data(plot_data, cfg, trend_year_max)
    if len(trend_data) > 2:
        x_vals = trend_data['Capacity_GW_July1'].to_numpy()
        y_vals = trend_data['Yearly_Profile_Factor'].to_numpy()
        summary['trend_year_range_label'] = (
            f'{cfg.trend_year_min}-{trend_year_max} (excluding {cfg.trend_year_excluded})'
        )
    else:
        x_vals = plot_data['Capacity_GW_July1'].to_numpy()
        y_vals = plot_data['Yearly_Profile_Factor'].to_numpy()
        summary['trend_year_range_label'] = 'all available data'

    coeffs_linear = np.polyfit(x_vals, y_vals, 1)
    summary['coeffs_linear'] = coeffs_linear
    summary['slope'] = float(coeffs_linear[0])

    # Capacity at Jan 1 of the year after the last complete year.
    jan_1_after = pd.Timestamp(f'{trend_year_max + 1}-01-01', tz=TZ)
    jan_1_capacity_gw = model.evaluate_one(jan_1_after) / 1000.0
    summary['jan_1_capacity_gw'] = jan_1_capacity_gw

    # Anchor first trend point.
    x_first = float(x_vals[0])
    y_first = float(y_vals[0])
    summary['x_trend_start'] = x_first
    summary['y_trend_start'] = y_first

    # Solid linear trend line from first trend point to Jan 1 (post-trend year).
    x_lin = np.linspace(x_first, jan_1_capacity_gw, 50)
    y_lin = np.polyval(coeffs_linear, x_lin)
    fig.add_trace(go.Scatter(
        x=x_lin, y=y_lin,
        mode='lines',
        name='Linear Trend',
        line=dict(color='red', width=2),
        hovertemplate=(
            f'<b>Linear Trend (from {cfg.trend_year_min} data to Jan 1, {trend_year_max + 1})</b><br>'
            f'<b>Equation:</b> y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}<br>'
            f'<b>Slope:</b> {coeffs_linear[0]:.2f} {cfg.slope_unit}<br>'
            f'<b>Starts at:</b> {x_first:.1f} {cfg.capacity_unit.split()[0]} ({y_first:.1f}%)<br>'
            f'<b>Ends at:</b> {jan_1_capacity_gw:.1f} {cfg.capacity_unit.split()[0]} '
            f'(Jan 1, {trend_year_max + 1})<br>'
            f'<extra></extra>'
        ),
        showlegend=True,
    ))

    # Exponential extrapolation on data with positive PF.
    valid = y_vals > 0
    if valid.sum() > 1:
        log_y = np.log(y_vals[valid])
        coeffs_log = np.polyfit(x_vals[valid], log_y, 1)
        a = float(np.exp(coeffs_log[1]))
        b = float(coeffs_log[0])
        summary['a'] = a
        summary['b'] = b

        if b < 0:
            # Dotted linear extrapolation to 0% (from Jan 1 endpoint).
            x_at_zero_linear = -coeffs_linear[1] / coeffs_linear[0]
            summary['x_at_zero_linear'] = x_at_zero_linear
            x_l2z = np.linspace(jan_1_capacity_gw, x_at_zero_linear, 50)
            y_l2z = np.polyval(coeffs_linear, x_l2z)
            fig.add_trace(go.Scatter(
                x=x_l2z, y=y_l2z,
                mode='lines',
                name='Linear Extrapolation',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate=(
                    '<b>Linear Extrapolation to 0%</b><br>'
                    f'<b>Capacity:</b> %{{x:.1f}} {cfg.capacity_unit.split()[0]}<br>'
                    '<b>Profile Factor:</b> %{y:.1f}%<br>'
                    f'<b>Equation:</b> y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}<br>'
                    f'<b>Reaches 0% at:</b> {x_at_zero_linear:.1f} {cfg.capacity_unit.split()[0]}<br>'
                    '<extra></extra>'
                ),
                showlegend=True,
            ))

            # Exponential extrapolation, re-anchored to linear endpoint at Jan 1.
            y_at_jan_1 = float(np.polyval(coeffs_linear, jan_1_capacity_gw))
            a_adj = y_at_jan_1
            x_exp = np.linspace(jan_1_capacity_gw, cfg.extrapolation_x_max, 100)
            y_exp = a_adj * np.exp(b * (x_exp - jan_1_capacity_gw))
            fig.add_trace(go.Scatter(
                x=x_exp, y=y_exp,
                mode='lines',
                name='Exponential Extrapolation',
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate=(
                    f'<b>Exponential Extrapolation (from Jan 1, {trend_year_max + 1})</b><br>'
                    f'<b>Capacity:</b> %{{x:.1f}} {cfg.capacity_unit.split()[0]}<br>'
                    '<b>Profile Factor:</b> %{y:.1f}%<br>'
                    f'<b>Equation:</b> y = {a_adj:.2f} * e^({b:.3f}*(x-{jan_1_capacity_gw:.1f}))<br>'
                    f'<b>Starts from:</b> {jan_1_capacity_gw:.1f} {cfg.capacity_unit.split()[0]} '
                    f'({y_at_jan_1:.1f}%)<br>'
                    '<extra></extra>'
                ),
                showlegend=True,
            ))

            if cfg.show_threshold_annotations:
                _add_threshold_annotations(fig, a_adj, b, jan_1_capacity_gw, cfg)

            # Annotation at callout capacity (using exponential).
            cap = cfg.callout_capacity_gw
            if cap <= cfg.extrapolation_x_max:
                pf_at_cap = a_adj * np.exp(b * (cap - jan_1_capacity_gw))
                summary['profile_factor_at_callout'] = pf_at_cap

            # General asymptotic note (PV plot only).
            if cfg.show_threshold_annotations:
                fig.add_annotation(
                    x=cfg.extrapolation_x_max * 0.75, y=5,
                    text=f'Asymptotic decay:<br>{cfg.label_short} always retains<br>some market value',
                    showarrow=False,
                    font=dict(size=10, color='orange'),
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='orange',
                    borderwidth=1,
                    xref='x', yref='y',
                )
    else:
        # Fallback: linear only.
        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_extended = np.linspace(cfg.x_axis_range[0], cfg.x_axis_range[1], 100)
        fig.add_trace(go.Scatter(
            x=x_extended,
            y=np.polyval(coeffs, x_extended),
            mode='lines',
            name='Linear Trend (fallback)',
            line=dict(color='red', width=2),
            hovertemplate=(
                '<b>Linear Trend (fallback)</b><br>'
                f'<b>Slope:</b> {coeffs[0]:.2f} {cfg.slope_unit}<br>'
                f'<b>Intercept:</b> {coeffs[1]:.2f}%<br>'
                '<extra></extra>'
            ),
            showlegend=True,
        ))

    _add_scatter_points(fig, plot_data, cfg)

    if cfg.show_outlier_annotation and cfg.trend_year_excluded in plot_data['year'].values:
        outlier = plot_data[plot_data['year'] == cfg.trend_year_excluded].iloc[0]
        fig.add_annotation(
            x=outlier['Capacity_GW_July1'] + 1,
            y=outlier['Yearly_Profile_Factor'] + 1,
            text='Outlier: gas-crisis',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='red', ax=50, ay=-30,
            font=dict(size=10, color='red'),
            bgcolor='rgba(255,255,255,0.9)', bordercolor='red', borderwidth=1,
        )

    # Slope annotation.
    if summary['slope'] is not None:
        _add_slope_annotation(fig, cfg, summary)

    # Linear-trend-reaches-zero annotation.
    if cfg.show_linear_zero_annotation and summary['x_at_zero_linear'] is not None:
        if cfg.show_threshold_annotations:
            fig.add_annotation(
                x=summary['x_at_zero_linear'], y=0,
                text=f'Linear trend at {summary["x_at_zero_linear"]:.0f} {cfg.capacity_unit.split()[0]}: 0%',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='red', ax=80, ay=-20,
                font=dict(size=10, color='red'),
                bgcolor='rgba(255,255,255,0.8)', bordercolor='red', borderwidth=1,
            )
        else:
            fig.add_annotation(
                text=f'Linear trend reaches 0% at {summary["x_at_zero_linear"]:.0f} {cfg.capacity_unit.split()[0]}',
                x=summary['x_at_zero_linear'], y=0,
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor='red', ax=0, ay=20,
                font=dict(size=12, color='red'),
                bgcolor='rgba(255,255,255,0.8)', bordercolor='red', borderwidth=1,
            )

    # Exponential-at-callout annotation (PV only).
    if cfg.show_threshold_annotations and summary['profile_factor_at_callout'] is not None:
        pf = summary['profile_factor_at_callout']
        cap = cfg.callout_capacity_gw
        fig.add_annotation(
            x=cap, y=pf,
            text=f'Exponential trend at {cap:.0f} {cfg.capacity_unit.split()[0]}: {pf:.0f}%',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='orange', ax=80, ay=-30,
            font=dict(size=11, color='black'),
            bgcolor='rgba(255,165,0,0.9)', bordercolor='orange', borderwidth=2,
        )

    return fig, summary


def _add_scatter_points(fig: go.Figure, plot_data: pd.DataFrame, cfg: ProfileFactorTechConfig) -> None:
    years_list = sorted(plot_data['year'].unique())
    year_to_color = _year_color_map(years_list)
    for row in plot_data.itertuples(index=False):
        year = int(row.year)
        capacity = float(row.Capacity_GW_July1)
        pf = float(row.Yearly_Profile_Factor)
        fig.add_trace(go.Scatter(
            x=[capacity], y=[pf],
            mode='markers+text',
            marker=dict(size=20, color=year_to_color[year], line=dict(width=2, color='white')),
            text=[f'{year % 100:02d}'],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            name=f'{year}',
            hovertemplate=(
                f'<b>Year:</b> {year}<br>'
                f'<b>July 1st Installed Capacity:</b> {capacity:.2f} {cfg.capacity_unit}<br>'
                f'<b>Profile Factor:</b> {pf:.1f}%<br>'
                '<extra></extra>'
            ),
            showlegend=True,
        ))


def _add_threshold_annotations(
    fig: go.Figure, a_adj: float, b: float, jan_1_capacity_gw: float,
    cfg: ProfileFactorTechConfig,
) -> None:
    for threshold in (5.0, 2.0, 1.0):
        if threshold >= a_adj:
            continue
        x_at_threshold = jan_1_capacity_gw + (np.log(threshold / a_adj)) / b
        if x_at_threshold > cfg.extrapolation_x_max:
            continue
        fig.add_annotation(
            x=x_at_threshold, y=threshold,
            text=f'{threshold}%',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='orange', ax=20, ay=-20,
            font=dict(size=9, color='orange'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='orange', borderwidth=1,
        )


def _add_slope_annotation(fig: go.Figure, cfg: ProfileFactorTechConfig, summary: dict) -> None:
    slope = summary['slope']
    coeffs_linear = summary['coeffs_linear']
    if cfg.show_threshold_annotations:
        # PV: arrow points at midpoint of the trend line.
        mid_cap = (summary['x_trend_start'] + summary['jan_1_capacity_gw']) / 2
        mid_pf = float(np.polyval(coeffs_linear, mid_cap))
        text_x = mid_cap - 8
        text_y = mid_pf - 5
        fig.add_annotation(
            text=f'Slope: {round(slope, 1)}{cfg.slope_unit}',
            x=text_x, y=text_y,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='red',
            ax=mid_cap - text_x, ay=mid_pf - text_y,
            font=dict(size=12, color='red'),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='red', borderwidth=1,
        )
    else:
        # Wind: fixed position at (5, 50), horizontal arrow.
        fig.add_annotation(
            text=f'Slope: {round(slope, 1)}{cfg.slope_unit}',
            x=5, y=50,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor='red', ax=5, ay=0,
            font=dict(size=12, color='red'),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='red', borderwidth=1,
        )


# -------------------------------------------------------------- entry points

def run_one(tech: str, *, open_html: bool = False, write_html: bool = True) -> None:
    cfg = TECH_CONFIGS[tech]
    anchors = load_capacity_points(cfg.capacity_csv, tz=TZ)
    model = CapacityModel.build(cfg.capacity_fit_mode, anchors)
    df_combined, plot_data = compute_yearly_plot_data(cfg, model)

    # The trend fit must not include a part-year profile factor, so it stops at
    # the last fully-observed calendar year. Derived, never hardcoded: a fixed
    # `trend_year_max` silently freezes the fit the moment the year rolls over.
    trend_year_max = last_complete_year(df_combined['time'].max())

    print(f"\n=== {cfg.label_short} ===")
    print('Combined data shape:', df_combined.shape)
    print('Date range:', df_combined['time'].min(), 'to', df_combined['time'].max())
    print(f'Capacity anchors: {cfg.capacity_csv.name} '
          f'({len(anchors)} points, {anchors["date"].min():%Y} - {anchors["date"].max():%Y})')
    print(f'Trend window: {cfg.trend_year_min} - {trend_year_max} '
          f'(excluding {cfg.trend_year_excluded})')
    print('\nYearly data for plotting:')
    print(plot_data.round(1))

    fig, summary = build_figure(plot_data, cfg, model, trend_year_max)

    pdf_path = f'{cfg.output_stem}.pdf'
    svg_path = f'{cfg.output_stem}.svg'
    fig.write_image(pdf_path, format='pdf')
    fig.write_image(svg_path, format='svg')
    if write_html:
        html_path = f'{cfg.output_stem}.html'
        # B1 fix: use the CDN to avoid embedding plotly.js (~3 MB) per HTML.
        # Compact only after the images are written: kaleido renders this figure
        # too, and only the HTML pays for the wire format.
        compact_figure_arrays(fig)
        fig.write_html(html_path, auto_open=open_html, include_plotlyjs='cdn')
        print(f'\nDashboard created:')
        print(f'  - HTML: {html_path}')
    else:
        print(f'\nDashboard created:')
    print(f'  - PDF:  {pdf_path}')
    print(f'  - SVG:  {svg_path}')

    _print_summary(plot_data, summary, cfg, trend_year_max)


def _print_summary(plot_data: pd.DataFrame, summary: dict, cfg: ProfileFactorTechConfig,
                   trend_year_max: int) -> None:
    print(f'Data points: {len(plot_data)} years')
    print(f"Years covered: {int(plot_data['year'].min())} - {int(plot_data['year'].max())}")
    print(f"Capacity range: {plot_data['Capacity_GW_July1'].min():.2f} - "
          f"{plot_data['Capacity_GW_July1'].max():.2f} {cfg.capacity_unit}")
    print(f"Profile factor range: {plot_data['Yearly_Profile_Factor'].min():.1f} - "
          f"{plot_data['Yearly_Profile_Factor'].max():.1f}%")

    if summary['slope'] is None:
        return

    # Correlation across the trend-fit window.
    trend = _select_trend_data(plot_data, cfg, trend_year_max)
    if len(trend) > 2:
        corr = float(np.corrcoef(trend['Capacity_GW_July1'].to_numpy(),
                                 trend['Yearly_Profile_Factor'].to_numpy())[0, 1])
    else:
        corr = float(np.corrcoef(plot_data['Capacity_GW_July1'].to_numpy(),
                                 plot_data['Yearly_Profile_Factor'].to_numpy())[0, 1])
    print(f'Correlation coefficient: {corr:.3f}')
    print(f"Slope: {round(summary['slope'], 1)}% per {cfg.capacity_unit.split()[0]}")

    cl = summary['coeffs_linear']
    rng = summary['trend_year_range_label']
    print(f'Linear trend equation (based on {rng}): y = {cl[0]:.2f}x + {cl[1]:.2f}')
    print(f"Linear trend starts at: {summary['x_trend_start']:.1f} {cfg.capacity_unit.split()[0]} "
          f"({summary['y_trend_start']:.1f}%)")
    print(f"Linear trend ends at: {summary['jan_1_capacity_gw']:.1f} {cfg.capacity_unit.split()[0]} "
          f"(January 1, {trend_year_max + 1})")

    if summary['x_at_zero_linear'] is not None:
        print(f"Linear trend reaches 0% at: {round(summary['x_at_zero_linear'], 0):.0f} "
              f"{cfg.capacity_unit.split()[0]}")

    if summary['a'] is not None and summary['b'] is not None:
        a, b = summary['a'], summary['b']
        print(f'Exponential extrapolation equation (based on {rng}): y = {a:.2f} * e^({b:.3f}*x)')
        print(f'Exponential decay rate: {b:.3f} per {cfg.capacity_unit.split()[0]}')
        print(f'Asymptotic behavior: {cfg.label_short} profile factor approaches 0% but never reaches it')
        print(f'{cfg.label_short} will always retain some market value on the day-ahead market')

        a_adj = float(np.polyval(summary['coeffs_linear'], summary['jan_1_capacity_gw']))
        jan_1 = summary['jan_1_capacity_gw']
        print('\nExponential extrapolation threshold milestones:')
        for threshold in (5.0, 2.0, 1.0):
            if threshold < a_adj:
                x_at_threshold = jan_1 + (np.log(threshold / a_adj)) / b
                print(f'  Profile factor reaches {threshold}% at: {x_at_threshold:.1f} '
                      f"{cfg.capacity_unit.split()[0]}")
        cap = cfg.callout_capacity_gw
        _cached = summary.get('profile_factor_at_callout')
        pf_at_cap = _cached if _cached is not None else a_adj * np.exp(b * (cap - jan_1))
        print(f'  Profile factor at {cap:.0f} {cfg.capacity_unit.split()[0]}: {pf_at_cap:.1f}%')
    else:
        print('Exponential extrapolation not available (fallback to linear only)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'techs',
        nargs='+',
        choices=['pv', 'wind_onshore', 'wind_offshore', 'all'],
        help="Which dashboards to build. 'all' = pv + wind_onshore + wind_offshore.",
    )
    parser.add_argument(
        '--no-html',
        action='store_true',
        help='Skip HTML export (PDF + SVG only).',
    )
    parser.add_argument(
        '--open',
        action='store_true',
        help='Open HTML in browser after export.',
    )
    args = parser.parse_args()

    techs: list[str] = []
    for t in args.techs:
        if t == 'all':
            techs.extend(['pv', 'wind_onshore', 'wind_offshore'])
        else:
            techs.append(t)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    techs = [t for t in techs if not (t in seen or seen.add(t))]

    for tech in techs:
        run_one(tech, open_html=args.open, write_html=not args.no_html)


if __name__ == '__main__':
    main()
