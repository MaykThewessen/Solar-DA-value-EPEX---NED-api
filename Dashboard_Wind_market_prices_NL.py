import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import glob
import warnings
os.system('clear')


# --- Load all monthly DA_prices and Wind data files ---
# Find all DA_prices and Wind files
price_files = sorted(glob.glob('data/DA_prices_20*.csv'))
wind_files = sorted(glob.glob('data/data_export_NED_Wind_20*.csv'))

# Exclude combined file from price_files
price_files = [f for f in price_files if 'combined' not in f]

# Load and concatenate all price files
price_dfs = []
for f in price_files:
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    price_dfs.append(df)
df_prices = pd.concat(price_dfs, ignore_index=True)

# Load and concatenate all Wind files
wind_dfs = []
for f in wind_files:
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    wind_dfs.append(df)
df_wind = pd.concat(wind_dfs, ignore_index=True)

#print(df_prices)
#print(df_wind)


# Merge the two dataframes on the 'time' column
df_combined = pd.merge(df_prices, df_wind, on='time', how='left')
#df_combined = df_combined.fillna(0)
#df_combined = df_combined.set_index('time').interpolate(method='time').reset_index()


df_combined['Wind_value'] = df_combined['Wind_production_MW'] * df_combined['DA_price']


# Create installed capacity column in MW using a linear fit (extrapolation allowed)
from datetime import datetime

# Known data points for installed capacity (AC) at year-end
capacity_points = [
    (pd.Timestamp('2019-01-01', tz='Europe/Amsterdam'), 3100), # MW AC
    (pd.Timestamp('2019-12-31', tz='Europe/Amsterdam'), 3190), # MW AC
    (pd.Timestamp('2020-12-31', tz='Europe/Amsterdam'), 3800),
    (pd.Timestamp('2021-12-31', tz='Europe/Amsterdam'), 4800),
    (pd.Timestamp('2022-12-31', tz='Europe/Amsterdam'), 5600),
    (pd.Timestamp('2023-12-31', tz='Europe/Amsterdam'), 6200),  # MW AC
    (pd.Timestamp('2024-12-31', tz='Europe/Amsterdam'), 6580),  # MW AC
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 6700),  # MW AC
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 7200),  # MW AC
]

def fit_installed_capacity_piecewise(date):
    # Ensure date is a pandas Timestamp with tz
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    if date.tz is None:
        date = date.tz_localize('Europe/Amsterdam')
    # If before first point, return first capacity
    if date <= capacity_points[0][0]:
        return capacity_points[0][1]
    # If after last point, extrapolate using last segment
    if date >= capacity_points[-1][0]:
        dt1, cap1 = capacity_points[-2]
        dt2, cap2 = capacity_points[-1]
        days_total = (dt2 - dt1).days
        if days_total == 0:
            return cap2
        days_since = (date - dt2).days
        slope = (cap2 - cap1) / days_total
        return round(cap2 + slope * days_since, 0)
    # Find the segment the date falls into
    for i in range(1, len(capacity_points)):
        dt1, cap1 = capacity_points[i-1]
        dt2, cap2 = capacity_points[i]
        if dt1 <= date <= dt2:
            days_total = (dt2 - dt1).days
            if days_total == 0:
                return cap1
            days_since = (date - dt1).days
            slope = (cap2 - cap1) / days_total
            return round(cap1 + slope * days_since, 0)
    # Fallback (should not reach here)
    return capacity_points[-1][1]

# Add the new column to df_combined
# Ensure 'time' is timezone-aware and in Europe/Amsterdam
if df_combined['time'].dt.tz is None:
    df_combined['time'] = df_combined['time'].dt.tz_localize('Europe/Amsterdam')
df_combined['installed_capacity_MW'] = df_combined['time'].apply(fit_installed_capacity_piecewise)


print(df_combined)


# summarize per month
# Remove timezone before converting to Period to avoid warning
df_combined['month'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M')

monthly_summary = (
    df_combined.groupby('month').apply(
        lambda x: (
            lambda avg_price, weighted_price: pd.Series({
                'Total_Wind_Energy_GWh': round(x['Wind_production_MW'].sum()/1000, 1),
                'Value_per_MW_AC_EUR': round(x['Wind_value'].sum() / x['installed_capacity_MW'].mean(), 1),
                'Avg_DA_Price': round(avg_price, 1),
                'Wind_Weighted_Price': round(weighted_price, 1),
                'profile_factor': round((weighted_price / avg_price)*100, 1) if avg_price != 0 else float('nan'),
                'Installed_Capacity_MW_AC': round(x['installed_capacity_MW'].mean(), 0)
            })
        )(
            x['DA_price'].mean(),
            (x['Wind_production_MW'] * x['DA_price']).sum() / x['Wind_production_MW'].sum() if x['Wind_production_MW'].sum() > 0 else float('nan')
        )
    )
    .reset_index()
)

print("\nMonthly Summary:")
# Round first 3 columns to 0 digits
monthly_summary_rounded = monthly_summary.copy()
monthly_summary_rounded.iloc[:, 1:4] = monthly_summary_rounded.iloc[:, 1:4].round(0)
#print(monthly_summary_rounded.to_string(index=False, float_format='%.0f').replace(',', '.'))
monthly_summary_df = pd.DataFrame(monthly_summary_rounded)
print(monthly_summary_df)


# Plot the combined dataframe using plotly
import plotly.graph_objs as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

# --- Prepare data for plotting ---
# 1. PV power and Day-ahead price (hourly)

# 2. Monthly PV yield (sum), PV value (sum), and PV_power_weighted_DA_price (monthly avg)
# Remove timezone before converting to Period to avoid warning
df_combined['month_date'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()

monthly = (
    df_combined.groupby('month_date').apply(
        lambda x: pd.Series({
            'Monthly_Wind_Energy_MWh': round(x['Wind_production_MW'].sum(), 1),
            'Monthly_Value_per_MW_AC_EUR': round(x['Wind_value'].sum() / x['installed_capacity_MW'].mean(), 1),
            # This is the correct formula for a weighted average:
            # weighted_avg = sum(value * weight) / sum(weight)
            # Here, DA_price is weighted by Wind_production_MW.
            'Monthly_Wind_Power_Weighted_DA_Price': (x['Wind_production_MW'] * x['DA_price']).sum() / x['Wind_production_MW'].sum() if x['Wind_production_MW'].sum() > 0 else float('nan'),
            'Monthly_Installed_Capacity_MW': x['installed_capacity_MW'].mean(),  # or .last() for end-of-month
            'Monthly_Avg_DA_Price': x['DA_price'].mean(),
        })
    )
    .reset_index()
)

# Calculate profile factor
monthly['Monthly_Profile_Factor'] = (monthly['Monthly_Wind_Power_Weighted_DA_Price'] / monthly['Monthly_Avg_DA_Price']) * 100

# Normalize by installed capacity
monthly['Monthly_Wind_Yield_per_MW'] = monthly['Monthly_Wind_Energy_MWh'] / monthly['Monthly_Installed_Capacity_MW']

# Calculate yearly totals
monthly['year'] = monthly['month_date'].dt.year
yearly_totals = monthly.groupby('year').agg({
    'Monthly_Wind_Energy_MWh': 'sum',
    'Monthly_Installed_Capacity_MW': 'mean'  # Average installed capacity for the year
}).reset_index()
yearly_totals.columns = ['year', 'Yearly_Wind_Energy_MWh', 'Yearly_Installed_Capacity_MW']

# Calculate total yearly wind value and divide by average installed capacity
yearly_wind_values = df_combined.groupby(df_combined['time'].dt.year).agg({
    'Wind_value': 'sum',  # Total yearly wind value
    'installed_capacity_MW': 'mean'  # Average installed capacity for the year
}).reset_index()
yearly_wind_values.columns = ['year', 'Yearly_Total_Wind_Value', 'Yearly_Installed_Capacity_MW']

# Calculate yearly value per MW
yearly_wind_values['Yearly_Value_per_MW_AC_EUR'] = yearly_wind_values['Yearly_Total_Wind_Value'] / yearly_wind_values['Yearly_Installed_Capacity_MW']

# Merge with yearly_totals
yearly_totals = yearly_totals.merge(yearly_wind_values[['year', 'Yearly_Value_per_MW_AC_EUR']], on='year')

# Calculate yearly weighted average price
yearly_weighted_prices = df_combined.groupby(df_combined['time'].dt.year).apply(
    lambda x: (x['Wind_production_MW'] * x['DA_price']).sum() / x['Wind_production_MW'].sum() if x['Wind_production_MW'].sum() > 0 else float('nan')
).reset_index()
yearly_weighted_prices.columns = ['year', 'Yearly_Wind_Weighted_Price']

# Merge yearly data
yearly_totals = yearly_totals.merge(yearly_weighted_prices, on='year')

# Calculate yearly profile factor
yearly_avg_prices = df_combined.groupby(df_combined['time'].dt.year)['DA_price'].mean().reset_index()
yearly_avg_prices.columns = ['year', 'Yearly_Avg_DA_Price']
yearly_totals = yearly_totals.merge(yearly_avg_prices, on='year')
yearly_totals['Yearly_Profile_Factor'] = (yearly_totals['Yearly_Wind_Weighted_Price'] / yearly_totals['Yearly_Avg_DA_Price']) * 100

# Add yearly data to monthly dataframe
monthly = monthly.merge(yearly_totals[['year', 'Yearly_Wind_Energy_MWh', 'Yearly_Value_per_MW_AC_EUR', 'Yearly_Wind_Weighted_Price', 'Yearly_Profile_Factor']], on='year')

# Prepare yearly summary data for table
yearly_summary_for_table = yearly_totals.copy()
yearly_summary_for_table['Yearly_Wind_Energy_GWh'] = yearly_summary_for_table['Yearly_Wind_Energy_MWh'] / 1000
yearly_summary_for_table['Yearly_Installed_Capacity_MW_AC'] = yearly_summary_for_table['Yearly_Installed_Capacity_MW']
# Calculate MWh/MW installed produced
yearly_summary_for_table['Yearly_MWh_per_MW'] = yearly_summary_for_table['Yearly_Wind_Energy_MWh'] / (yearly_summary_for_table['Yearly_Installed_Capacity_MW_AC'] )
yearly_summary_for_table = yearly_summary_for_table.round(0)

# Helper functions for formatting
def format_number(x):
    if pd.isna(x):
        return ''
    return f"{int(x):,}".replace(',', '.')

def format_mwp(x):
    if pd.isna(x):
        return ''
    return f"{x:,.0f}".replace(',', '.')

def format_percentage(x):
    if pd.isna(x):
        return ''
    return f"{x:.0f}%"

# --- Create subplots ---
fig = make_subplots(
    rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.08,
    subplot_titles=(
        'Yearly Summary',
        'Installed Wind Capacity',
        'Monthly Wind Yield',
        'Market Value per MW installed',
        'Wind Weighted DA Price & Profile Factor'
    ),
    specs=[
        [{"type": "table"}],
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": True}]
    ],
    row_heights=[0.3, 0.2, 0.2, 0.2, 0.2]  # Adjusted heights for 5 subplots
)



# First subplot: Yearly summary table (rows reversed)
fig.add_trace(
    go.Table(
        header=dict(
            values=['Year', 'Wind Energy produced (GWh/y)', 'Installed Wind Capacity in NL (MW) mid-year', 'MWh yield / MW installed', 'Annual Market value (EUR/MW/y)', 'Day-Ahead linear avg price (EUR/MWh)', 'Wind-profile weighted price (EUR/MWh)', 'Profile Factor of Wind (%)'],
            font=dict(size=10),
            align='left'
        ),
        cells=dict(
            values=[
                yearly_summary_for_table['year'].astype(str)[::-1],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Wind_Energy_GWh'][::-1]],
                [format_mwp(x) for x in yearly_summary_for_table['Yearly_Installed_Capacity_MW_AC'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_MWh_per_MW'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Value_per_MW_AC_EUR'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Avg_DA_Price'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Wind_Weighted_Price'][::-1]],
                [format_percentage(x) for x in yearly_summary_for_table['Yearly_Profile_Factor'][::-1]]
            ],
            font=dict(size=9),
            align='left',
            height=20
        )
    ),
    row=1, col=1
)

# Second subplot: Installed Wind Capacity
# Create date range for the fitted curve
start_date = pd.Timestamp('2019-01-01', tz='Europe/Amsterdam')
end_date = pd.Timestamp('2025-12-31', tz='Europe/Amsterdam')
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Calculate fitted capacity values
fitted_capacity = [fit_installed_capacity_piecewise(date) for date in date_range]

# Add the fitted curve
fig.add_trace(
    go.Scatter(
        x=date_range,
        y=fitted_capacity,
        mode='lines',
        name='Fitted Capacity',
        line=dict(color='red', width=2, dash='dot')
    ),
    row=2, col=1
)

# Add the capacity points as dots
capacity_dates = [point[0] for point in capacity_points]
capacity_values = [point[1] for point in capacity_points]

fig.add_trace(
    go.Scatter(x=capacity_dates, y=capacity_values, mode='markers', name='Actual Capacity Points', 
               marker=dict(color='red', size=8, symbol='circle')),
    row=2, col=1
)

# Add the hourly Wind production power as line
fig.add_trace(
    go.Scatter(x=df_combined['time'], y=df_combined['Wind_production_MW'], mode='lines', name='Hourly Wind Production', line=dict(color='blue', width=2)),
    row=2, col=1
)


fig.update_yaxes(title_text='Power (MW AC)', row=2, col=1)
fig.update_xaxes(title_text='Year', row=2, col=1)

# Third subplot: Monthly Wind energy production (bars)
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Wind_Energy_MWh']/1000, name='Monthly Wind Energy Production (GWh)', marker_color='green'),
    row=3, col=1, secondary_y=False
)
fig.update_yaxes(title_text='Energy (GWh)', row=3, col=1, secondary_y=False)


fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Value_per_MW_AC_EUR'], name='Wind Market Value (EUR/MW/Month)', marker_color='darkgreen'),
    row=4, col=1, secondary_y=False
)
fig.update_yaxes(title_text='EUR per month', row=4, col=1)
#fig.update_xaxes(title_text='Month', row=3, col=1, tickangle=45, tickformat='%b %Y')


# Fifth subplot: Monthly Wind Power Weighted DA Price (bar)
# Add average DA price as a line
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Avg_DA_Price'], name='Avg DA Price (EUR/MWh)', marker_color='royalblue'),
    row=5, col=1, secondary_y=False
)
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Wind_Power_Weighted_DA_Price'], name='Wind Weighted Market Price (EUR/MWh)', marker_color='teal'),
    row=5, col=1, secondary_y=False
)

# Add profile factor as a secondary y-axis line
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Profile_Factor'], name='Profile Factor (%)'),
    row=5, col=1, secondary_y=False
)
fig.update_yaxes(title_text='Profile factor (%)', row=5, col=1, secondary_y=False)
#fig.update_yaxes(title_text='Profile Factor (%)', row=5, col=1, secondary_y=True)
fig.update_xaxes(title_text='per month', row=5, col=1, tickangle=45, tickformat='%b %Y')

# Move legend below the plot
fig.update_layout(
    title_text='Analysis on Wind value (NL), EPEX spot prices + Wind production of NED.nl',
    legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
    margin=dict(b=120)
)

# Create a separate table figure
# Format numbers with thousands separators and percentage for profile factor
# Sort monthly_summary_rounded in reverse chronological order
monthly_summary_rounded_reversed = monthly_summary_rounded.sort_values('month', ascending=False).reset_index(drop=True)

table_fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Month', 'Wind Energy produced (GWh)', 'Installed Capacity (MW) month-avg', 'Market value (EUR/MW/year)', 'Day-Ahead linear average price (EUR/MWh)', 'Wind-profile Weighted price (EUR/MWh)', 'Profile Factor of Wind (%)'],
        font=dict(size=10),
        align='left'
    ),
            cells=dict(
            values=[
                monthly_summary_rounded_reversed['month'].astype(str),
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 1].round(0)],
                [format_mwp(x) for x in monthly_summary_rounded_reversed.iloc[:, 6]],  # Installed Capacity column
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 2].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 3].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 4].round(0)],
                [format_percentage(x) for x in monthly_summary_rounded_reversed.iloc[:, 5]]
            ],
        font=dict(size=9),
        align='left',
        height=20
    )
)])

table_fig.update_layout(
    title_text='Monthly Summary Table (Analysis on Wind value (NL), EPEX spot prices + Wind production of NED.nl)',
    margin=dict(l=0, r=0, t=50, b=0)
)

# Write both figures to separate files
fig.write_html('wind_production_plot_v3.html', auto_open=True)
table_fig.write_html('wind_monthly_summary_table.html', auto_open=True)



