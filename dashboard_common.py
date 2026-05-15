"""Shared theme, plotting, and PDF-export helpers for the Solar/Wind market-prices dashboards.

Public surface:
    CLAUDE_PALETTE, FONT_FAMILY            - theme constants
    hex_to_rgb(hex_color) -> (r, g, b)
    gradient_fills(x, y, hex_color, n_layers=6) -> list[go.Scatter]
    bar_gradient_shapes(x_idx, y, hex_color, xref, yref) -> list[dict]
    add_dt_hours(df, time_col='time') -> df with `_dt_h` column (median-fallback, not bfill)
    last_complete_year(last_time: pd.Timestamp) -> int
    vw_price_groupby(df, group_col, prod_col, price_col) -> pd.Series       # vectorised capture price
    build_themed_slide_fig(slide, years, last_complete_year, brand, source_date) -> go.Figure
    build_monthly_metric_by_year_fig(monthly_summary, value_col, years_to_plot, ...) -> go.Figure
    build_yearly_summary_table_fig(yst, columns_spec, brand, title, subtitle, source_date) -> go.Figure
    render_slides_to_pdf(slides_pages, out_path) -> None

`slide` spec dict keys:
    title         (str)
    subtitle      (str)
    ytitle        (str)
    kind          ('area_gradient' | 'dual_area' | 'triple_line' | 'bar_gradient')
    traces        list[(y_series, name, hex_color)]   # for triple_line the 3rd is reference dotted
    mask_partial  (bool)                              # drop year > last_complete_year from x-axis
    yaxis_range   (tuple|None)                        # explicit (ymin, ymax) override
    yaxis_tickformat (str|None)                       # d3 tickformat, e.g. ',.0f'
    use_eu_thousands (bool)                           # separators=',.' for "150.000" style
    legend_position ('bottom' | 'right' | 'top-right-inside' | 'none')
    callouts      list[dict]                          # extra fig.add_annotation kwargs (auto themed)

Curtailment fix:
    Monthly+yearly curtailment % now computed from raw MWh sums (caller must pass raw, not rounded GWh).

_dt_h fix:
    First-row diff is NaN; we fill with the *median* dt of the series rather than bfill or 1.0 — robust
    across hourly/15-min cutovers without leaking the second row's value into the first.
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime, timezone
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread as _imread


# ------------------------------------------------------------------ theme

CLAUDE_PALETTE: dict[str, str] = dict(
    bg='#FAF9F5',
    panel='#F3F0E8',
    panel_edge='#E5DFD0',
    ink='#1F1E1D',
    ink_soft='#3D3929',
    muted='#8C8377',
    grid='#E5DFD0',
    accent='#C96442',
    accent_soft='#E7A87C',
    sage='#7A8471',
    blue='#4C6B8A',
)
FONT_FAMILY = 'Inter, Helvetica Neue, Arial, sans-serif'


# ------------------------------------------------------------------ utility

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def gradient_fills(x, y_vals, hex_color: str, n_layers: int = 6) -> list[go.Scatter]:
    """Stack n semi-transparent fill-to-zero scatter traces to fake a vertical gradient."""
    r, g, b = hex_to_rgb(hex_color)
    out: list[go.Scatter] = []
    for i in range(n_layers):
        frac = (i + 1) / n_layers
        y_layer = [None if (v is None or pd.isna(v)) else v * frac for v in y_vals]
        alpha = 0.10 + 0.05 * (n_layers - i) / n_layers
        out.append(go.Scatter(
            x=list(x), y=y_layer, mode='lines',
            line=dict(width=0),
            fill='tozeroy',
            fillcolor=f'rgba({r},{g},{b},{alpha:.3f})',
            hoverinfo='skip', showlegend=False,
        ))
    return out


def bar_gradient_shapes(x_idx, y_vals, hex_color: str,
                        xref: str = 'x', yref: str = 'y') -> list[dict]:
    """Per-bar vertical gradient as stacked thin rect shapes."""
    r, g, b = hex_to_rgb(hex_color)
    shapes: list[dict] = []
    half_w, n = 0.32, 14
    for xi, v in zip(x_idx, y_vals):
        if v is None or pd.isna(v) or v == 0:
            continue
        for k in range(n):
            y0 = v * (k / n)
            y1 = v * ((k + 1) / n)
            alpha = 0.30 + 0.65 * (k / max(n - 1, 1))
            shapes.append(dict(
                type='rect', xref=xref, yref=yref,
                x0=xi - half_w, x1=xi + half_w, y0=y0, y1=y1,
                line=dict(width=0),
                fillcolor=f'rgba({r},{g},{b},{alpha:.3f})',
                layer='below',
            ))
    return shapes


def add_dt_hours(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """Add per-row interval `_dt_h` (hours). First-row NaN filled with series median, not bfill.

    Robust to hourly/quarter-hourly transitions: median reflects the dominant cadence.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    dt_h = df[time_col].diff().dt.total_seconds().div(3600)
    median = dt_h.median()
    if pd.isna(median):
        median = 1.0
    df['_dt_h'] = dt_h.fillna(median)
    return df


def last_complete_month_end(now: pd.Timestamp | None = None,
                             tz: str = 'Europe/Amsterdam') -> pd.Timestamp:
    """End-of-month timestamp for the most recently completed calendar month.

    Always returns a tz-aware Timestamp in `tz`. Robust replacement for the divergent
    `Period.end_time` (Solar) vs `replace(day=1) - 1s` (Wind) constructs.
    """
    if now is None:
        now = pd.Timestamp.now(tz=tz)
    elif now.tz is None:
        now = now.tz_localize(tz)
    period_prev = pd.Period(now, freq='M') - 1
    return pd.Timestamp(period_prev.end_time, tz=tz)


def last_complete_year(last_time: pd.Timestamp) -> int:
    """Year of the most recent fully-observed calendar year, given the last data timestamp."""
    if pd.isna(last_time):
        return datetime.now(tz=timezone.utc).year - 1
    if last_time.month == 12 and last_time.day >= 31:
        return int(last_time.year)
    return int(last_time.year) - 1


def vw_price_groupby(df: pd.DataFrame, group_col: str,
                     prod_col: str, price_col: str) -> pd.Series:
    """Vectorised volume-weighted price = sum(prod*price) / sum(prod). Replaces per-row .apply(lambda)."""
    num = (df[prod_col] * df[price_col]).groupby(df[group_col]).sum()
    den = df[prod_col].groupby(df[group_col]).sum()
    out = num / den.where(den > 0)  # non-positive production -> NaN (matches old `sum > 0` guard)
    return out


# ------------------------------------------------------------------ slide figure

def _brand_strip(fig: go.Figure, brand: str, source_date: str,
                 brand_x: float = 0.06, footer_x: float = 0.06) -> None:
    p = CLAUDE_PALETTE
    fig.add_annotation(
        text=f"<b style='color:{p['accent']}'>{brand}</b>  ·  Day-Ahead market analysis",
        xref='paper', yref='paper', x=brand_x, y=1.06,
        showarrow=False, font=dict(size=11, color=p['muted'], family=FONT_FAMILY),
    )
    fig.add_annotation(
        text=f"Source: NED.nl (generation) · EPEX/ENTSO-E (Day-Ahead prices)   |   {source_date}",
        xref='paper', yref='paper', x=footer_x, y=-0.18,
        showarrow=False, font=dict(size=11, color=p['muted'], family=FONT_FAMILY),
        xanchor='left',
    )
    fig.add_shape(type='line', xref='paper', yref='paper',
                  x0=brand_x, x1=brand_x + 0.10, y0=1.01, y1=1.01,
                  line=dict(color=p['accent'], width=3))


def _legend_dict(position: str):
    p = CLAUDE_PALETTE
    if position == 'right':
        return dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02,
                    bgcolor='rgba(250,249,245,0.90)', bordercolor=p['panel_edge'], borderwidth=1,
                    font=dict(size=12, color=p['ink_soft']))
    if position == 'top-right-inside':
        return dict(orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.98,
                    bgcolor='rgba(250,249,245,0.85)', bordercolor=p['panel_edge'], borderwidth=1,
                    font=dict(size=12, color=p['ink_soft']))
    return dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='left', x=0.0,
                bgcolor='rgba(0,0,0,0)', font=dict(size=12, color=p['ink_soft']))


def _fmt_callout(p: dict, body: str, x, y, ax: int = 0, ay: int = -35) -> dict:
    return dict(
        x=x, y=y, xref='x', yref='y', text=body,
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
        arrowcolor=p['accent'], ax=ax, ay=ay,
        bgcolor='rgba(250,249,245,0.95)', bordercolor=p['accent'], borderwidth=1, borderpad=6,
        font=dict(size=12, color=p['ink'], family=FONT_FAMILY),
        align='left',
    )


def build_themed_slide_fig(slide: dict, years_full: list[str],
                           years_for_data,  # list[int] aligned with years_full
                           last_complete_year_int: int,
                           brand: str, source_date: str) -> go.Figure:
    p = CLAUDE_PALETTE
    fig = go.Figure()

    if slide.get('mask_partial'):
        keep = [yv <= last_complete_year_int for yv in years_for_data]
        years = [yl for yl, k in zip(years_full, keep) if k]
        new_traces = []
        for tup in slide['traces']:
            y = tup[0]
            y_filt = pd.Series(list(y))[pd.Series(keep)].reset_index(drop=True)
            new_traces.append((y_filt, *tup[1:]))
        slide = {**slide, 'traces': new_traces}
    else:
        years = years_full

    kind = slide['kind']

    if kind == 'area_gradient':
        y, name, color = slide['traces'][0]
        y_list = list(y)
        for t in gradient_fills(years, y_list, color):
            fig.add_trace(t)
        fig.add_trace(go.Scatter(
            x=years, y=y_list, mode='lines+markers', name=name,
            line=dict(color=color, width=3.5, shape='spline', smoothing=0.8),
            marker=dict(size=10, color=color, line=dict(color='white', width=2)),
        ))

    elif kind == 'dual_area':
        for y, name, color in slide['traces']:
            y_list = list(y)
            for t in gradient_fills(years, y_list, color, n_layers=5):
                fig.add_trace(t)
            fig.add_trace(go.Scatter(
                x=years, y=y_list, mode='lines+markers', name=name,
                line=dict(color=color, width=3.2, shape='spline', smoothing=0.7),
                marker=dict(size=9, color=color, line=dict(color='white', width=2)),
            ))

    elif kind == 'triple_line':
        for i, (y, name, color) in enumerate(slide['traces']):
            y_list = list(y)
            is_ref = (i == 2)  # third trace is reference (dotted, thinner)
            fig.add_trace(go.Scatter(
                x=years, y=y_list, mode='lines+markers', name=name,
                line=dict(color=color, width=2.0 if is_ref else 2.8,
                          dash='dot' if is_ref else 'solid',
                          shape='spline', smoothing=0.6),
                marker=dict(size=7 if is_ref else 9, color=color,
                            line=dict(color='white', width=2)),
            ))

    elif kind == 'bar_gradient':
        y, name, color = slide['traces'][0]
        y_list = list(y)
        if slide.get('mask_partial'):
            pairs = [(yr, v) for yr, v in zip(years, y_list) if not (v is None or pd.isna(v))]
            x_used = [pp[0] for pp in pairs]
            y_used = [pp[1] for pp in pairs]
        else:
            x_used, y_used = years, y_list
        fig.add_trace(go.Bar(
            x=x_used, y=y_used, name=name,
            marker=dict(color=color, opacity=0.0),
            hovertemplate='%{x}: %{y}<extra></extra>',
            showlegend=True,
        ))
        for shp in bar_gradient_shapes(list(range(len(x_used))), y_used, color):
            fig.add_shape(**shp)

    else:
        raise ValueError(f"unknown slide kind: {kind}")

    yaxis = dict(
        title=dict(text=slide['ytitle'], font=dict(size=13, color=p['muted'])),
        gridcolor=p['grid'], gridwidth=1,
        zerolinecolor=p['panel_edge'], zerolinewidth=1,
        tickfont=dict(size=12, color=p['ink_soft']),
    )
    if slide.get('yaxis_range') is not None:
        ylo, yhi = slide['yaxis_range']
        yaxis['range'] = [ylo, yhi]
    else:
        yaxis['rangemode'] = 'tozero'
    if slide.get('yaxis_tickformat'):
        yaxis['tickformat'] = slide['yaxis_tickformat']
    if slide.get('ytick_suffix'):
        yaxis['ticksuffix'] = slide['ytick_suffix']

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:30px;color:{p['ink']};font-weight:700'>{slide['title']}</span><br>"
                 f"<span style='font-size:15px;color:{p['muted']};font-weight:400'>{slide['subtitle']}</span>",
            x=0.06, y=0.94, xanchor='left',
        ),
        paper_bgcolor=p['bg'], plot_bgcolor=p['bg'],
        font=dict(family=FONT_FAMILY, color=p['ink_soft'], size=14),
        margin=dict(l=80, r=60, t=140, b=110),
        xaxis=dict(
            title=dict(text='Year', font=dict(size=13, color=p['muted'])),
            type='category', showgrid=False,
            linecolor=p['panel_edge'], linewidth=1,
            tickfont=dict(size=12, color=p['ink_soft']),
            ticks='outside', tickcolor=p['panel_edge'],
            categoryorder='array', categoryarray=years,
        ),
        yaxis=yaxis,
        legend=_legend_dict(slide.get('legend_position', 'bottom')),
        showlegend=(slide.get('legend_position') != 'none' and len(slide['traces']) > 1),
        width=1600, height=1000,
    )
    if slide.get('use_eu_thousands'):
        fig.update_layout(separators=',.')

    _brand_strip(fig, brand, source_date)

    for callout in slide.get('callouts', []) or []:
        fig.add_annotation(**_fmt_callout(p, **callout))

    return fig


# ------------------------------------------------------------------ monthly capture rate slide

def build_monthly_metric_by_year_fig(monthly_summary: pd.DataFrame,
                                     value_col: str,
                                     years_to_plot: Sequence[int],
                                     title: str, subtitle: str, ytitle: str,
                                     brand: str, source_date: str,
                                     tick_suffix: str | None = None,
                                     emphasize_last: bool = True) -> go.Figure:
    p = CLAUDE_PALETTE
    ms = monthly_summary.copy()
    ms['year'] = ms['month'].astype(str).str.slice(0, 4).astype(int)
    ms['month_num'] = ms['month'].astype(str).str.slice(5, 7).astype(int)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    palette = [p['accent'], p['sage'], p['blue'], p['muted']]
    fig = go.Figure()
    for i, yr in enumerate(years_to_plot):
        sub = ms[ms['year'] == yr].sort_values('month_num')
        if sub.empty:
            continue
        color = palette[i % len(palette)]
        y_vals = sub[value_col].tolist()
        x_vals = [month_labels[m - 1] for m in sub['month_num']]
        r, g, b = hex_to_rgb(color)
        if emphasize_last and yr == years_to_plot[-1]:
            for k in range(5):
                frac = (k + 1) / 5
                y_layer = [v * frac if not pd.isna(v) else None for v in y_vals]
                alpha = 0.06 + 0.04 * (5 - k) / 5
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_layer, mode='lines', line=dict(width=0),
                    fill='tozeroy', fillcolor=f'rgba({r},{g},{b},{alpha:.3f})',
                    hoverinfo='skip', showlegend=False,
                ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers',
            name=str(yr),
            line=dict(color=color, width=3.2, shape='spline', smoothing=0.7),
            marker=dict(size=10, color=color, line=dict(color='white', width=2)),
        ))

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:30px;color:{p['ink']};font-weight:700'>{title}</span><br>"
                 f"<span style='font-size:15px;color:{p['muted']};font-weight:400'>{subtitle}</span>",
            x=0.06, y=0.94, xanchor='left',
        ),
        paper_bgcolor=p['bg'], plot_bgcolor=p['bg'],
        font=dict(family=FONT_FAMILY, color=p['ink_soft'], size=14),
        margin=dict(l=80, r=60, t=140, b=110),
        xaxis=dict(
            title=dict(text='Month', font=dict(size=13, color=p['muted'])),
            type='category', categoryorder='array', categoryarray=month_labels,
            showgrid=False, linecolor=p['panel_edge'], linewidth=1,
            tickfont=dict(size=12, color=p['ink_soft']),
            ticks='outside', tickcolor=p['panel_edge'],
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(size=13, color=p['muted'])),
            rangemode='tozero',
            gridcolor=p['grid'], gridwidth=1,
            zerolinecolor=p['panel_edge'], zerolinewidth=1,
            tickfont=dict(size=12, color=p['ink_soft']),
            ticksuffix=(tick_suffix or ''),
        ),
        legend=_legend_dict('top-right-inside'),
        showlegend=True, width=1600, height=1000,
    )
    _brand_strip(fig, brand, source_date)
    return fig


# ------------------------------------------------------------------ table page

def _fmt_num(v, fmt: str) -> str:
    if v is None or pd.isna(v):
        return ''
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def build_yearly_summary_table_fig(yst: pd.DataFrame, columns_spec: list[dict],
                                   brand: str, title: str, subtitle: str,
                                   source_date: str) -> go.Figure:
    """columns_spec: list of dicts with keys:
        col       (column name in yst — None means use 'year_label')
        header    (HTML header markup)
        fmt       (format string applied per value)
        width     (relative column width)
    """
    p = CLAUDE_PALETTE
    yst_desc = yst.sort_values('year', ascending=False).reset_index(drop=True)
    headers = [c['header'] for c in columns_spec]
    widths = [c.get('width', 70) for c in columns_spec]
    cells_values: list[list[str]] = []
    for c in columns_spec:
        col_name = c.get('col')
        fmt = c.get('fmt', '{}')
        if col_name is None or col_name == 'year_label':
            cells_values.append(yst_desc['year_label'].tolist())
        else:
            cells_values.append([_fmt_num(v, fmt) for v in yst_desc[col_name]])
    n_rows = len(yst_desc)
    row_fill = [[p['panel'] if i % 2 == 0 else p['bg'] for i in range(n_rows)]] * len(headers)

    tfig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header=dict(
            values=headers,
            fill_color=p['accent'],
            font=dict(color='white', size=12, family=FONT_FAMILY),
            align='center', height=70,
            line=dict(color=p['accent'], width=0),
        ),
        cells=dict(
            values=cells_values,
            fill_color=row_fill,
            font=dict(color=p['ink'], size=12, family=FONT_FAMILY),
            align=['left'] + ['right'] * (len(headers) - 1),
            height=30,
            line=dict(color=p['panel_edge'], width=1),
        ),
    )])
    tfig.update_layout(
        title=dict(
            text=f"<span style='font-size:30px;color:{p['ink']};font-weight:700'>{title}</span><br>"
                 f"<span style='font-size:15px;color:{p['muted']};font-weight:400'>{subtitle}</span>",
            x=0.03, y=0.96, xanchor='left',
        ),
        paper_bgcolor=p['bg'], plot_bgcolor=p['bg'],
        font=dict(family=FONT_FAMILY, color=p['ink_soft']),
        margin=dict(l=40, r=40, t=140, b=80),
        width=2400, height=1000,
    )
    _brand_strip(tfig, brand, source_date, brand_x=0.03, footer_x=0.03)
    return tfig


# ------------------------------------------------------------------ pdf assembler

def _fig_to_pdf_page(pdf: PdfPages, fig: go.Figure, *, width: int, height: int, page_inches: tuple[float, float]):
    png = fig.to_image(format='png', width=width, height=height, scale=2)
    img = _imread(io.BytesIO(png), format='png')
    f, ax = plt.subplots(figsize=page_inches, dpi=200)
    f.patch.set_facecolor(CLAUDE_PALETTE['bg'])
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(f, bbox_inches='tight', facecolor=CLAUDE_PALETTE['bg'])
    plt.close(f)


def render_slides_to_pdf(pages: list[dict], out_path: str) -> None:
    """pages: ordered list of {'fig': go.Figure, 'shape': 'slide'|'table'} dicts."""
    with PdfPages(out_path) as pdf:
        for page in pages:
            shape = page.get('shape', 'slide')
            if shape == 'slide':
                _fig_to_pdf_page(pdf, page['fig'], width=1600, height=1000, page_inches=(11.69, 8.27))
            elif shape == 'table':
                _fig_to_pdf_page(pdf, page['fig'], width=2400, height=1000, page_inches=(16.54, 8.27))
            else:
                raise ValueError(f"unknown page shape: {shape}")


def utc_today_str() -> str:
    """`pd.Timestamp.utcnow()` is deprecated; this returns YYYY-MM-DD in UTC."""
    return datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')
