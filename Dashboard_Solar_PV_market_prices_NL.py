import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import warnings
from data_loader import load_da_prices, load_ned_pv
os.system('clear')


# TODO: add a function to only export PV power when DA prices are non-negative
# Graph shows monthly values per year with each year as a different line on the month x-axis graph

df_prices = load_da_prices()
df_pv = load_ned_pv()

#print(df_prices)
#print(df_pv)


# Merge the two dataframes on the 'time' column
df_combined = pd.merge(df_prices, df_pv, on='time', how='left')

# Ensure time column is datetime and in Amsterdam timezone after merge
df_combined['time'] = pd.to_datetime(df_combined['time']).dt.tz_convert('Europe/Amsterdam')

#df_combined = df_combined.fillna(0)
#df_combined = df_combined.set_index('time').interpolate(method='time').reset_index()


df_combined['Solar_value'] = df_combined['Solar_production_MWh'] * df_combined['DA_price']


# Create installed capacity column in MW using a linear fit (extrapolation allowed)
from datetime import datetime

# Known data points for installed capacity (DC) at year-end
capacity_points = [
    (pd.Timestamp('2017-12-31', tz='Europe/Amsterdam'), 2911),
    (pd.Timestamp('2018-12-31', tz='Europe/Amsterdam'), 4610), # MWp DC https://www.cbs.nl/nl-nl/longread/rapportages/2024/hernieuwbare-energie-in-nederland-2023/5-zonne-energie
    (pd.Timestamp('2019-12-31', tz='Europe/Amsterdam'), 7225), # MWp DC https://opendata.cbs.nl/#/CBS/nl/dataset/85005NED/table
    (pd.Timestamp('2020-12-31', tz='Europe/Amsterdam'), 11108),
    (pd.Timestamp('2021-12-31', tz='Europe/Amsterdam'), 14822),
    (pd.Timestamp('2022-12-31', tz='Europe/Amsterdam'), 19536),
    (pd.Timestamp('2023-12-31', tz='Europe/Amsterdam'), 24302),  # MWp DC
    (pd.Timestamp('2024-12-31', tz='Europe/Amsterdam'), 28621),  # MWp DC
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 28621 + 1300),  # MWp DC # lower installed PV estimate update by https://x.com/BM_Visser/status/1954798688049697116
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 28621 + 1300 + 1240),  # MWp DC
]


print("Installed PV capacity NL (PV GWp DC, year-end):")
_now = pd.Timestamp.now(tz='Europe/Amsterdam')
for dt, cap in capacity_points:
    status = "actual" if dt <= _now else "outlook"
    print(f"  {dt.year}   {cap:>5,} MW  ({status})")



# Prepare arrays for fitting
dates = np.array([(dt - capacity_points[0][0]).days for dt, _ in capacity_points])
capacities = np.array([cap for _, cap in capacity_points])

df_capacity = pd.DataFrame({
    'date': [dt for dt, _ in capacity_points],
    'capacity_MW': [cap for _, cap in capacity_points]
})



# Piece-wise linear interpolation for installed capacity

# Prepare X (date as ordinal days) and Y (capacity)
capacity_dates_ord = np.array([(dt - capacity_points[0][0]).days for dt, _ in capacity_points])
capacity_values = np.array([cap for _, cap in capacity_points])

def interpolate_installed_capacity(date):
    # Ensure date is a pandas Timestamp
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    # Convert date to ordinal days relative to first anchor
    days_since = (date - capacity_points[0][0]).days

    # If before first anchor, return first capacity (flat extrapolation)
    if days_since <= capacity_dates_ord[0]:
        return capacity_values[0]

    # If after last anchor, return last capacity (flat extrapolation)
    if days_since >= capacity_dates_ord[-1]:
        return capacity_values[-1]

    # Find the segment for interpolation
    idx = np.searchsorted(capacity_dates_ord, days_since) - 1
    # Ensure idx is within valid bounds
    idx = max(0, min(idx, len(capacity_dates_ord) - 2))

    x0, x1 = capacity_dates_ord[idx], capacity_dates_ord[idx+1]
    y0, y1 = capacity_values[idx], capacity_values[idx+1]

    if x1 == x0:
        return y0  # avoid division by zero, degenerate case

    # Linear interpolation
    interp_value = y0 + (y1 - y0) * (days_since - x0) / (x1 - x0)
    return round(interp_value, 0)

# Add the new column to df_combined
df_combined['installed_capacity_MW'] = df_combined['time'].apply(interpolate_installed_capacity)
print(df_combined)



import matplotlib.pyplot as plt
# Plot installed capacity in df_combined (time vs installed_capacity_MW)
plt.figure(figsize=(10, 5))
plt.plot(df_combined['time'].dt.tz_localize(None), df_combined['installed_capacity_MW'], label='Fitted Installed Capacity (MW)', color='tab:blue')

# Plot the original capacity_points as scatter
capacity_dates = [dt for dt, cap in capacity_points]
capacity_values = [cap for dt, cap in capacity_points]
plt.scatter([d.tz_localize(None) for d in capacity_dates], capacity_values, color='tab:red', label='Known Data Points', zorder=5)
plt.title('Installed Capacity: Fitted vs Known Data Points')
plt.xlabel('Date')
plt.ylabel('Installed Capacity (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('installed_capacity_plot.pdf', bbox_inches='tight')
plt.close()






# summarize per month
# Convert to Period for monthly grouping (first remove timezone to avoid warning)
df_combined['month'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M')


# Plot the combined dataframe using plotly
import plotly.graph_objs as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import plotly.colors  # type: ignore

# --- Prepare data for plotting ---
# 1. PV power and Day-ahead price (hourly)

# 2. Monthly PV yield (sum), PV value (sum), and PV_power_weighted_DA_price (monthly avg)
# Convert to Period and then to timestamp for plotting (first remove timezone to avoid warning)
df_combined['month_date'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()

monthly = (
    df_combined.groupby('month_date').agg({
        'Solar_production_MWh': 'sum',
        'Solar_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean'
    }).reset_index()
)

# Calculate derived columns
monthly['Monthly_PV_Energy_MWh'] = round(monthly['Solar_production_MWh'], 1)
monthly['Monthly_Value_per_MWp_DC_EUR'] = round(monthly['Solar_value'] / monthly['installed_capacity_MW'], 1)
monthly['Monthly_Installed_Capacity_MW'] = monthly['installed_capacity_MW']
monthly['Monthly_Avg_DA_Price'] = monthly['DA_price']

# Calculate PV weighted price for each month
monthly['Monthly_PV_Power_Weighted_DA_Price'] = monthly.apply(
    lambda row: (df_combined[df_combined['month_date'] == row['month_date']]['Solar_production_MWh'] * 
                 df_combined[df_combined['month_date'] == row['month_date']]['DA_price']).sum() / 
                df_combined[df_combined['month_date'] == row['month_date']]['Solar_production_MWh'].sum() 
                if df_combined[df_combined['month_date'] == row['month_date']]['Solar_production_MWh'].sum() > 0 else float('nan'), axis=1
)

# Select and reorder columns
monthly = monthly[['month_date', 'Monthly_PV_Energy_MWh', 'Monthly_Value_per_MWp_DC_EUR', 'Monthly_PV_Power_Weighted_DA_Price', 'Monthly_Installed_Capacity_MW', 'Monthly_Avg_DA_Price']]

# Calculate profile factor
monthly['Monthly_Profile_Factor'] = (monthly['Monthly_PV_Power_Weighted_DA_Price'] / monthly['Monthly_Avg_DA_Price']) * 100

# Normalize by installed capacity
monthly['Monthly_PV_Yield_per_MW'] = monthly['Monthly_PV_Energy_MWh'] / monthly['Monthly_Installed_Capacity_MW']

# Filter to only include complete months (exclude current incomplete month)
from datetime import datetime
current_date = datetime.now()
last_complete_month = current_date.replace(day=1) - pd.Timedelta(days=1)  # Last day of previous month
last_complete_month_period = pd.Period(last_complete_month, freq='M')

# Convert to timezone-aware timestamp for comparison
last_complete_timestamp = pd.Timestamp(last_complete_month_period.end_time, tz='Europe/Amsterdam')

# Filter monthly data to exclude incomplete months
monthly = monthly[monthly['month_date'] <= last_complete_month_period.to_timestamp()]

# Also filter the original df_combined for consistency in calculations
df_combined = df_combined[df_combined['time'] <= last_complete_timestamp]

# Per-row interval in hours (handles hourly vs quarterly data automatically)
df_combined = df_combined.sort_values('time').reset_index(drop=True)
df_combined['_dt_h'] = df_combined['time'].diff().dt.total_seconds().div(3600)
df_combined['_dt_h'] = df_combined['_dt_h'].bfill().fillna(1.0)

# Recalculate monthly_summary with filtered data
monthly_summary = (
    df_combined.groupby('month').agg({
        'Solar_production_MWh': 'sum',
        'Solar_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean'
    }).reset_index()
)

# Recalculate derived columns for monthly_summary
monthly_summary['Total_PV_Energy_GWh'] = round(monthly_summary['Solar_production_MWh']/1000, 1)
monthly_summary['Value_per_MWp_DC_EUR'] = round(monthly_summary['Solar_value'] / monthly_summary['installed_capacity_MW'], 1)
monthly_summary['Avg_DA_Price'] = round(monthly_summary['DA_price'], 1)

# Recalculate PV weighted price for each month
monthly_summary['PV_Weighted_Price'] = monthly_summary.apply(
    lambda row: (df_combined[df_combined['month'] == row['month']]['Solar_production_MWh'] * 
                 df_combined[df_combined['month'] == row['month']]['DA_price']).sum() / 
                df_combined[df_combined['month'] == row['month']]['Solar_production_MWh'].sum() 
                if df_combined[df_combined['month'] == row['month']]['Solar_production_MWh'].sum() > 0 else float('nan'), axis=1
)

monthly_summary['profile_factor'] = round((monthly_summary['PV_Weighted_Price'] / monthly_summary['Avg_DA_Price'])*100, 1)
monthly_summary['Installed_Capacity_GWp_DC'] = round(monthly_summary['installed_capacity_MW'] / 1000, 2)

# Excluding-negative-price metrics per month
_pos = df_combined[df_combined['DA_price'] >= 0].copy()
_pos['_pv'] = _pos['Solar_production_MWh'] * _pos['DA_price']
_pos_monthly = _pos.groupby('month').agg(
    Solar_production_MWh_pos=('Solar_production_MWh', 'sum'),
    Solar_value_pos=('Solar_value', 'sum'),
    Avg_DA_Price_pos=('DA_price', 'mean'),
    _pv_sum=('_pv', 'sum'),
).reset_index()
_pos_monthly['PV_Weighted_Price_excl_neg'] = (
    _pos_monthly['_pv_sum'] / _pos_monthly['Solar_production_MWh_pos'].replace(0, np.nan)
)
monthly_summary = monthly_summary.merge(_pos_monthly, on='month', how='left')
monthly_summary['MWh_per_MWp_excl_neg'] = (
    monthly_summary['Solar_production_MWh_pos'] / monthly_summary['installed_capacity_MW']
)
monthly_summary['Value_per_MWp_DC_EUR_excl_neg'] = (
    monthly_summary['Solar_value_pos'] / monthly_summary['installed_capacity_MW']
)
monthly_summary['profile_factor_excl_neg'] = (
    monthly_summary['PV_Weighted_Price_excl_neg'] / monthly_summary['Avg_DA_Price_pos']
) * 100
monthly_summary['curtailment_pct'] = (
    (monthly_summary['Total_PV_Energy_GWh'] * 1000 - monthly_summary['Solar_production_MWh_pos'])
    / (monthly_summary['Total_PV_Energy_GWh'] * 1000)
) * 100
_neg_monthly = df_combined[df_combined['DA_price'] < 0].groupby('month')['_dt_h'].sum().round(0).reset_index().rename(columns={'_dt_h': 'neg_hours'})
monthly_summary = monthly_summary.merge(_neg_monthly, on='month', how='left')
monthly_summary['neg_hours'] = monthly_summary['neg_hours'].fillna(0)

monthly_summary = monthly_summary[['month', 'Total_PV_Energy_GWh', 'Value_per_MWp_DC_EUR', 'Avg_DA_Price', 'PV_Weighted_Price', 'profile_factor', 'Installed_Capacity_GWp_DC', 'MWh_per_MWp_excl_neg', 'Value_per_MWp_DC_EUR_excl_neg', 'PV_Weighted_Price_excl_neg', 'profile_factor_excl_neg', 'curtailment_pct', 'neg_hours']]

print("\nMonthly Summary (Complete months only):")
# Round first 3 columns to 1 digits
monthly_summary_rounded = monthly_summary.copy()
monthly_summary_rounded.loc[:, monthly_summary_rounded.columns[1:4]] = monthly_summary_rounded.iloc[:, 1:4].round(1)
monthly_summary_df = pd.DataFrame(monthly_summary_rounded)
print(monthly_summary_df)

# Calculate yearly totals
monthly['year'] = monthly['month_date'].dt.year
yearly_totals = monthly.groupby('year').agg({
    'Monthly_PV_Energy_MWh': 'sum',
    'Monthly_Installed_Capacity_MW': 'mean'  # Average installed capacity for the year
}).reset_index()
yearly_totals = yearly_totals.rename(columns={
    'Monthly_PV_Energy_MWh': 'Yearly_PV_Energy_MWh',
    'Monthly_Installed_Capacity_MW': 'Yearly_Installed_Capacity_MW'
})
yearly_totals = yearly_totals[['year', 'Yearly_PV_Energy_MWh', 'Yearly_Installed_Capacity_MW']]

# Calculate total yearly solar value and divide by average installed capacity
yearly_solar_values = df_combined.groupby(df_combined['time'].dt.year).agg({
    'Solar_value': 'sum',  # Total yearly solar value
    'installed_capacity_MW': 'mean'  # Average installed capacity for the year
}).reset_index()
yearly_solar_values = yearly_solar_values.rename(columns={
    'time': 'year',
    'Solar_value': 'Yearly_Total_Solar_Value',
    'installed_capacity_MW': 'Yearly_Installed_Capacity_MW'
})
yearly_solar_values = yearly_solar_values[['year', 'Yearly_Total_Solar_Value', 'Yearly_Installed_Capacity_MW']]

# Calculate yearly value per MWp
yearly_solar_values['Yearly_Value_per_MWp_DC_EUR'] = yearly_solar_values['Yearly_Total_Solar_Value'] / yearly_solar_values['Yearly_Installed_Capacity_MW']

# Merge with yearly_totals
yearly_totals = yearly_totals.merge(yearly_solar_values[['year', 'Yearly_Value_per_MWp_DC_EUR']], on='year')

# Calculate yearly weighted average price
yearly_weighted_prices = df_combined.groupby(df_combined['time'].dt.year).agg({
    'Solar_production_MWh': 'sum',
    'DA_price': 'mean'
}).reset_index()

# Calculate weighted price manually
yearly_weighted_prices['Yearly_PV_Weighted_Price'] = yearly_weighted_prices.apply(
    lambda row: (df_combined[df_combined['time'].dt.year == row['time']]['Solar_production_MWh'] * 
                 df_combined[df_combined['time'].dt.year == row['time']]['DA_price']).sum() / 
                df_combined[df_combined['time'].dt.year == row['time']]['Solar_production_MWh'].sum() 
                if df_combined[df_combined['time'].dt.year == row['time']]['Solar_production_MWh'].sum() > 0 else float('nan'), axis=1
)

yearly_weighted_prices = yearly_weighted_prices.rename(columns={'time': 'year'})
yearly_weighted_prices = yearly_weighted_prices[['year', 'Yearly_PV_Weighted_Price']]

# Merge yearly data
yearly_totals = yearly_totals.merge(yearly_weighted_prices, on='year')

# Calculate yearly profile factor
yearly_avg_prices = df_combined.groupby(df_combined['time'].dt.year)['DA_price'].mean().reset_index()
yearly_avg_prices = yearly_avg_prices.rename(columns={
    'time': 'year',
    'DA_price': 'Yearly_Avg_DA_Price'
})
yearly_avg_prices = yearly_avg_prices[['year', 'Yearly_Avg_DA_Price']]
yearly_totals = yearly_totals.merge(yearly_avg_prices, on='year')
yearly_totals['Yearly_Profile_Factor'] = (yearly_totals['Yearly_PV_Weighted_Price'] / yearly_totals['Yearly_Avg_DA_Price']) * 100

# --- Metrics excluding negative-price hours (subsidieloze-style) ---
df_pos = df_combined[df_combined['DA_price'] >= 0].copy()
df_neg = df_combined[df_combined['DA_price'] < 0].copy()
yearly_neg_hours = df_neg.groupby(df_neg['time'].dt.year)['_dt_h'].sum().round(0).reset_index().rename(columns={'time': 'year', '_dt_h': 'Yearly_Neg_Hours'})

yearly_pos = df_pos.groupby(df_pos['time'].dt.year).agg(
    Solar_production_MWh_pos=('Solar_production_MWh', 'sum'),
    Solar_value_pos=('Solar_value', 'sum'),
    Avg_DA_Price_pos=('DA_price', 'mean'),
).reset_index().rename(columns={'time': 'year'})

# Volume-weighted capture price on positive-price hours only
yearly_pos['Yearly_PV_Weighted_Price_excl_neg'] = (
    df_pos.assign(_pv=df_pos['Solar_production_MWh'] * df_pos['DA_price'])
          .groupby(df_pos['time'].dt.year)['_pv'].sum()
          .values
    / yearly_pos['Solar_production_MWh_pos'].replace(0, np.nan).values
)

yearly_totals = yearly_totals.merge(yearly_pos, on='year', how='left')
yearly_totals['Yearly_MWh_per_MWp_excl_neg'] = (
    yearly_totals['Solar_production_MWh_pos'] / yearly_totals['Yearly_Installed_Capacity_MW']
)
yearly_totals['Yearly_Value_per_MWp_DC_EUR_excl_neg'] = (
    yearly_totals['Solar_value_pos'] / yearly_totals['Yearly_Installed_Capacity_MW']
)
yearly_totals['Yearly_Profile_Factor_excl_neg'] = (
    yearly_totals['Yearly_PV_Weighted_Price_excl_neg'] / yearly_totals['Avg_DA_Price_pos']
) * 100
yearly_totals['Yearly_Curtailment_Pct'] = (
    (yearly_totals['Yearly_PV_Energy_MWh'] - yearly_totals['Solar_production_MWh_pos'])
    / yearly_totals['Yearly_PV_Energy_MWh']
) * 100
yearly_totals = yearly_totals.merge(yearly_neg_hours, on='year', how='left')
yearly_totals['Yearly_Neg_Hours'] = yearly_totals['Yearly_Neg_Hours'].fillna(0)

# Add yearly data to monthly dataframe
monthly = monthly.merge(yearly_totals[['year', 'Yearly_PV_Energy_MWh', 'Yearly_Value_per_MWp_DC_EUR', 'Yearly_PV_Weighted_Price', 'Yearly_Profile_Factor']], on='year')

# Prepare hourly data for the new subplot
hourly = (
    df_combined.groupby(df_combined['time']).agg({
        'Solar_production_MWh': 'first',
        'installed_capacity_MW': 'first'
    }).reset_index()
)
hourly['Hourly_PV_Power_GW'] = hourly['Solar_production_MWh'] * 4 / 1000  # MWh-per-15min × 4 → MW, /1000 → GW
hourly['Hourly_Installed_Capacity_GW'] = hourly['installed_capacity_MW'] / 1000  # Convert to GW
hourly = hourly[['time', 'Hourly_PV_Power_GW', 'Hourly_Installed_Capacity_GW']]

# Create custom color scheme with distinct, distinguishable colors for each year
years_list = sorted(monthly['year'].unique())
num_years = len(years_list)

# Define a palette of distinct colors that are easy to differentiate (for older years)
distinct_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Grey
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#ff9896',  # Light red
    '#98df8a',  # Light green
    '#ffbb78',  # Light orange
    '#aec7e8',  # Light blue
    '#c5b0d5',  # Light purple
]

# Define distinct but muted colors for the most recent year(s)
bright_colors_for_recent_years = [
    '#DC143C',  # Crimson red - for most recent year (opvallend maar niet te fel)
    '#FF8C00',  # Dark orange - for second most recent (warm en opvallend)
]

color_scheme = []
for i, year in enumerate(years_list):
    # Check if this is one of the last 2 years
    if i >= num_years - 2:
        # Use bright colors for the last 2 years
        recent_year_index = i - (num_years - 2)
        color_scheme.append(bright_colors_for_recent_years[recent_year_index])
    else:
        # Use regular colors for older years
        if i < len(distinct_colors):
            color_scheme.append(distinct_colors[i])
        else:
            # If we have more years than colors, cycle through the palette
            color_scheme.append(distinct_colors[i % len(distinct_colors)])

# Create year to color mapping
year_to_color = dict(zip(years_list, color_scheme))

# Prepare yearly summary data for table
yearly_summary_for_table = yearly_totals.copy()
yearly_summary_for_table['Yearly_PV_Energy_GWh'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / 1000
yearly_summary_for_table['Yearly_PV_Energy_TWh'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / 1_000_000

# Calculate July 1st installed capacity for each year
def get_july_1st_capacity(year):
    july_1st = pd.Timestamp(f'{year}-07-01', tz='Europe/Amsterdam')
    return interpolate_installed_capacity(july_1st)

yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC'] = yearly_summary_for_table['year'].apply(get_july_1st_capacity) / 1000
# Calculate MWh/MWp installed produced
yearly_summary_for_table['Yearly_MWh_per_MWp'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / (yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC']*1000 )
yearly_summary_for_table = yearly_summary_for_table.round(1)

# Mark incomplete years with " *" (preliminary)
_last_time = df_combined['time'].max()
_last_complete_year = _last_time.year if (_last_time.month == 12 and _last_time.day >= 31) else _last_time.year - 1
yearly_summary_for_table['year_label'] = yearly_summary_for_table['year'].apply(
    lambda y: f"{int(y)} *" if y > _last_complete_year else str(int(y))
)

# Helper functions for formatting
def format_number(x):
    if pd.isna(x):
        return ''
    return f"{int(x):,}".replace(',', '.')

def format_gwp(x):
    if pd.isna(x):
        return ''
    return f"{x:.1f}"

def format_percentage(x):
    if pd.isna(x):
        return ''
    return f"{x:.0f}%"

# --- Create subplots ---
fig = make_subplots(
    rows=7, cols=1, shared_xaxes=False, vertical_spacing=0.06,
    subplot_titles=(
        'Hourly PV Power Output in NL',
        'Total PV Yield in NL',
        'Yield normalized per installed capacity',
        'Market Value per installed capacity',
        'Solar capture rate (%)',
        'Solar capture price (€/MWh)',
        ' '
    ),
    specs=[
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
         [{"type": "table"}]
    ],
    row_heights=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.28]
)



# First subplot: Hourly PV power output with installed capacity line
fig.add_trace(
    go.Scatter(
        x=hourly['time'],
        y=hourly['Hourly_PV_Power_GW'],
        mode='lines',
        name='Hourly PV Power',
        line=dict(color='blue', width=0.5),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.4)',
        opacity=0.9,
        showlegend=False
    ),
    row=1, col=1, secondary_y=False
)

# Add red dotted line for installed capacity (yearly values only)
yearly_capacity_dates = []
yearly_capacity_values = []

for date, capacity in capacity_points:
    yearly_capacity_dates.append(date)
    yearly_capacity_values.append(capacity/1000)  # Convert to GWp

fig.add_trace(
    go.Scatter(
        x=yearly_capacity_dates,
        y=yearly_capacity_values,
        mode='lines+markers',
        name='GWp PV in NL',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(color='red', size=6),
        showlegend=True
    ),
    row=1, col=1, secondary_y=False
)

fig.update_yaxes(title_text='Power (GW)', row=1, col=1)
fig.update_xaxes(title_text='', row=1, col=1)

# Sixth subplot: Yearly summary table (rows reversed)
fig.add_trace(
    go.Table(
        header=dict(
            values=['Year (* = preliminary)', 'Installed PV Capacity in NL (GWp) Average', 'PV Energy produced (TWh/y) (NED.nl)', 'MWh yield / MWp installed', 'MWh yield / MWp (excl. neg)', 'Curtailment (%)', 'Negative-price hours (h/y)', 'Annual Market value (EUR/MWp/y)', 'Market value EUR/MWp/y (excl. neg)', 'Day-Ahead linear avg price (EUR/MWh)', 'Solar capture price (€/MWh)', 'Solar capture price (€/MWh) excl. neg', 'Solar capture rate (%)', 'Solar capture rate (%) excl. neg'],
            font=dict(size=10),
            align='left'
        ),
        cells=dict(
            values=[
                yearly_summary_for_table['year_label'][::-1],
                [format_gwp(x) for x in yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC'][::-1]],
                [format_gwp(x) for x in yearly_summary_for_table['Yearly_PV_Energy_TWh'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_MWh_per_MWp'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_MWh_per_MWp_excl_neg'][::-1]],
                [format_percentage(x) for x in yearly_summary_for_table['Yearly_Curtailment_Pct'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Neg_Hours'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Value_per_MWp_DC_EUR'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Value_per_MWp_DC_EUR_excl_neg'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Avg_DA_Price'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_PV_Weighted_Price'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_PV_Weighted_Price_excl_neg'][::-1]],
                [format_percentage(x) for x in yearly_summary_for_table['Yearly_Profile_Factor'][::-1]],
                [format_percentage(x) for x in yearly_summary_for_table['Yearly_Profile_Factor_excl_neg'][::-1]]
            ],
            font=dict(size=9),
            align='left',
            height=20
        )
    ),
    row=7, col=1
)



# Second subplot: Monthly PV energy production (lines per year)
# Create separate traces for each year
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    # Extract month number (1-12) for x-axis and ensure proper January to December order
    year_data['month_num'] = year_data['month_date'].dt.month
    # Sort by month to ensure January to December order
    year_data = year_data.sort_values('month_num')
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_PV_Energy_MWh']/1000, 
            name=f'{year}', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>Energy:</b> %{y:.1f} GWh<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=2, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='GWh produced', row=2, col=1, secondary_y=False)
fig.update_xaxes(title_text='', row=2, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])


# Third subplot: Monthly MWh yield per MWp installed (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    # Sort by month to ensure January to December order
    year_data = year_data.sort_values('month_num')
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_PV_Energy_MWh'] / (year_data['Monthly_Installed_Capacity_MW']), 
            name=f'{year}', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>MWh/MWp:</b> %{y:.1f}<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=3, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='MWh per MWp', row=3, col=1)
fig.update_xaxes(title_text='', row=3, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])


# Fourth subplot: Monthly PV Market Value (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    # Sort by month to ensure January to December order
    year_data = year_data.sort_values('month_num')
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_Value_per_MWp_DC_EUR'], 
            name=f'{year}', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>Market Value:</b> %{y:.1f} EUR/MWp<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=4, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='€ per MWp', row=4, col=1)
fig.update_xaxes(title_text='', row=4, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])


# Fifth subplot: Monthly Profile Factor only (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    # Sort by month to ensure January to December order
    year_data = year_data.sort_values('month_num')
    
    # Profile Factor line only
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_Profile_Factor'], 
            name=f'{year}', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>Capture rate:</b> %{y:.1f}%<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=5, col=1, secondary_y=False
    )

fig.update_yaxes(title_text='Solar capture rate (%)', row=5, col=1, range=[0, 100])
fig.update_xaxes(title_text='', row=5, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])

# Sixth subplot: Solar Capture Price (volume-weighted DA price for PV, €/MWh)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    year_data = year_data.sort_values('month_num')

    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'],
            y=year_data['Monthly_PV_Power_Weighted_DA_Price'],
            name=f'{year}',
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>Capture Price:</b> €%{y:.1f}/MWh<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=False,
            line_color=year_to_color[year]
        ),
        row=6, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='Solar capture price (€/MWh)', row=6, col=1)
fig.update_xaxes(title_text='', row=6, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])

# Update layout with individual legends for each subplot
#fig.update_layout(
    #title_text='Analysis on PV value (NL), EPEX spot prices + PV production of NED.nl',
    #margin=dict(b=50)
#)

# Add individual legends for each subplot
fig.update_layout(
    height=1800,
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top'
    )
)

# Update legend for each subplot to show only relevant traces
# Skip the first two traces (daily PV yield and installed capacity) and only update scatter plots
num_years = len(sorted(monthly['year'].unique()))
for i, year in enumerate(sorted(monthly['year'].unique())):
    # Update legend for second subplot (PV Energy) - starts at index 3 (after daily traces)
    fig.data[i + 3].update(showlegend=True)
    # Update legend for third subplot (MWh yield per MWp)
    fig.data[i + 3 + num_years].update(showlegend=True)
    # Update legend for fourth subplot (Market Value)
    fig.data[i + 3 + 2*num_years].update(showlegend=True)
    # Update legend for fifth subplot (Profile Factor)
    fig.data[i + 3 + 3*num_years].update(showlegend=True)

# Create a separate table figure
# Format numbers with thousands separators and percentage for profile factor
# Sort monthly_summary_rounded in reverse chronological order
monthly_summary_rounded_reversed = monthly_summary_rounded.sort_values('month', ascending=False).reset_index(drop=True)

table_fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Month', 'Solar PV capacity NL (GWp)', 'PV Energy produced (GWh/month) (NED.nl)', 'MWh yield / MWp (excl. neg)', 'Curtailment (%)', 'Negative-price hours (h)', 'Market value PV (EUR/MWp/year)', 'Market value EUR/MWp/y (excl. neg)', 'Day-Ahead average price (EUR/MWh)', 'Solar capture price (€/MWh)', 'Solar capture price (€/MWh) excl. neg', 'Solar capture rate (%)', 'Solar capture rate (%) excl. neg'],
        font=dict(size=10),
        align='left'
    ),
            cells=dict(
            values=[
                monthly_summary_rounded_reversed['month'].astype(str),
                [format_gwp(x) for x in monthly_summary_rounded_reversed['Installed_Capacity_GWp_DC']],
                [format_number(x) for x in monthly_summary_rounded_reversed['Total_PV_Energy_GWh'].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed['MWh_per_MWp_excl_neg'].round(0)],
                [format_percentage(x) for x in monthly_summary_rounded_reversed['curtailment_pct']],
                [format_number(x) for x in monthly_summary_rounded_reversed['neg_hours']],
                [format_number(x) for x in monthly_summary_rounded_reversed['Value_per_MWp_DC_EUR'].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed['Value_per_MWp_DC_EUR_excl_neg'].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed['Avg_DA_Price'].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed['PV_Weighted_Price'].round(0)],
                [format_number(x) for x in monthly_summary_rounded_reversed['PV_Weighted_Price_excl_neg'].round(0)],
                [format_percentage(x) for x in monthly_summary_rounded_reversed['profile_factor']],
                [format_percentage(x) for x in monthly_summary_rounded_reversed['profile_factor_excl_neg']]
            ],
        font=dict(size=9),
        align='left',
        height=20
    )
)])

table_fig.update_layout(
    title_text='Monthly Summary Table (Analysis on PV value (NL), EPEX spot prices + PV production of NED.nl)',
    margin=dict(l=0, r=0, t=50, b=0)
)

# Write both figures to separate files
table_fig.write_html('monthly_summary_table.html', auto_open=True)

fig.write_html('solar_production_plot_v3.html', auto_open=True)


# --- Slide-style yearly figure: one slide per metric, paired excl/incl-neg as separate lines ---
yst = yearly_summary_for_table.sort_values('year').reset_index(drop=True)
_years = yst['year_label'].tolist()

slides = []  # list of (title, list-of-traces, y_axis_title)

def _scatter(y, name, dash=None, color=None, fill=None, fillcolor=None):
    return go.Scatter(
        x=_years, y=y, name=name, mode='lines+markers',
        line=dict(width=2, dash=dash) if dash else dict(width=2),
        marker=dict(size=8),
        line_color=color,
        fill=fill,
        fillcolor=fillcolor,
    )

slides.append(('Installed PV capacity (GWp DC)',
               [_scatter(yst['Yearly_Installed_Capacity_GWp_DC'], 'Installed capacity', color='#d62728')],
               'GWp'))
_twh_complete = yst['Yearly_PV_Energy_TWh'].where(yst['year'] <= _last_complete_year)
slides.append(('PV Energy produced (TWh/y) — NED.nl, complete years only',
               [_scatter(_twh_complete, 'PV energy', color='#1f77b4')],
               'TWh/y'))
_yield_incl = yst['Yearly_MWh_per_MWp'].where(yst['year'] <= _last_complete_year)
_yield_excl = yst['Yearly_MWh_per_MWp_excl_neg'].where(yst['year'] <= _last_complete_year)
slides.append(('MWh yield per MWp installed (complete years only)',
               [_scatter(_yield_incl, 'MWh/MWp (incl. neg)', color='#1f77b4'),
                _scatter(_yield_excl, 'MWh/MWp (excl. neg)', color='#2ca02c', dash='dash')],
               'MWh / MWp'))
slides.append(('Curtailment — share of MWh during DA<0 hours',
               [_scatter(yst['Yearly_Curtailment_Pct'], 'Curtailment %', color='#d62728')],
               '%'))
slides.append(('Negative-price hours per year',
               [_scatter(yst['Yearly_Neg_Hours'], 'Hours with DA < 0', color='#9467bd')],
               'hours'))
slides.append(('Annual market value (€/MWp/y)',
               [_scatter(yst['Yearly_Value_per_MWp_DC_EUR'], 'Market value (incl. neg)', color='#1f77b4'),
                _scatter(yst['Yearly_Value_per_MWp_DC_EUR_excl_neg'], 'Market value (excl. neg)', color='#2ca02c', dash='dash')],
               '€ / MWp / y'))
slides.append(('Solar capture price vs Day-Ahead average (€/MWh)',
               [_scatter(yst['Yearly_PV_Weighted_Price'], 'Capture price (incl. neg)', color='#1f77b4'),
                _scatter(yst['Yearly_PV_Weighted_Price_excl_neg'], 'Capture price (excl. neg)', color='#2ca02c', dash='dash'),
                _scatter(yst['Yearly_Avg_DA_Price'], 'Day-Ahead avg', color='#7f7f7f', dash='dot')],
               '€ / MWh'))
slides.append(('Solar capture rate (%)',
               [_scatter(yst['Yearly_Profile_Factor_excl_neg'], 'Capture rate (excl. neg)', color='#2ca02c', fill='tozeroy', fillcolor='rgba(44,160,44,0.25)'),
                _scatter(yst['Yearly_Profile_Factor'], 'Capture rate (incl. neg)', color='#1f77b4', fill='tozeroy', fillcolor='rgba(31,119,180,0.35)')],
               '%'))

slides_fig = go.Figure()
trace_slide_idx = []
for s_idx, (_title, traces, _yaxis) in enumerate(slides):
    for t in traces:
        slides_fig.add_trace(t)
        trace_slide_idx.append(s_idx)

# Visible: only first slide on load
for i, t in enumerate(slides_fig.data):
    t.visible = (trace_slide_idx[i] == 0)

buttons = []
for s_idx, (title, _traces, yaxis_title) in enumerate(slides):
    vis = [idx == s_idx for idx in trace_slide_idx]
    buttons.append(dict(
        label=title,
        method='update',
        args=[
            {'visible': vis},
            {'title.text': title, 'yaxis.title.text': yaxis_title, 'yaxis.rangemode': 'tozero'}
        ],
    ))

slides_fig.update_layout(
    title=dict(text=slides[0][0]),
    yaxis=dict(title=dict(text=slides[0][2]), rangemode='tozero'),
    xaxis=dict(title='Year', type='category'),
    height=600,
    updatemenus=[dict(
        type='dropdown', direction='down', x=0.0, y=1.15, xanchor='left', yanchor='top',
        buttons=buttons, showactive=True,
    )],
    legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
)

slides_fig.write_html('solar_yearly_slides.html', auto_open=True)


# --- Claude-themed multi-page PDF: one slide per page, dashboard style, gradients ---
import io as _io
import matplotlib.pyplot as _plt
from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
from matplotlib.image import imread as _imread

# Claude palette
_CLAUDE = dict(
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

# Trace styling per slide: list of (col, name, kind, color)
# kind: 'area_gradient' | 'line' | 'bar_gradient'
_pdf_slides = [
    dict(
        title='Installed Solar PV Capacity',
        subtitle='Netherlands · NED.nl source · year-end GWp DC',
        ytitle='GWp (DC)',
        kind='area_gradient',
        traces=[(yst['Yearly_Installed_Capacity_GWp_DC'], 'Installed capacity', _CLAUDE['accent'])],
        mask_partial=False,
    ),
    dict(
        title='Solar PV Energy Produced',
        subtitle='Annual generation in TWh · complete years only',
        ytitle='TWh / year',
        kind='bar_gradient',
        traces=[(yst['Yearly_PV_Energy_TWh'].where(yst['year'] <= _last_complete_year), 'PV energy', _CLAUDE['accent'])],
        mask_partial=True,
    ),
    dict(
        title='Specific Yield',
        subtitle='MWh produced per MWp installed · with & without negative-price hours',
        ytitle='MWh / MWp',
        kind='dual_area',
        traces=[
            (yst['Yearly_MWh_per_MWp'].where(yst['year'] <= _last_complete_year), 'incl. neg-price hours', _CLAUDE['blue']),
            (yst['Yearly_MWh_per_MWp_excl_neg'].where(yst['year'] <= _last_complete_year), 'excl. neg-price hours', _CLAUDE['sage']),
        ],
        mask_partial=True,
    ),
    dict(
        title='Curtailment Share',
        subtitle='Share of solar MWh produced during DA < 0 €/MWh hours',
        ytitle='% of yearly MWh',
        kind='bar_gradient',
        traces=[(yst['Yearly_Curtailment_Pct'].where(yst['year'] <= _last_complete_year), 'Curtailment', _CLAUDE['accent'])],
        mask_partial=True,
    ),
    dict(
        title='Negative-Price Hours',
        subtitle='Hours per year with Day-Ahead price < 0 €/MWh',
        ytitle='Hours / year',
        kind='bar_gradient',
        traces=[(yst['Yearly_Neg_Hours'].where(yst['year'] <= _last_complete_year), 'Negative-price hours', _CLAUDE['sage'])],
        mask_partial=True,
    ),
    dict(
        title='Annual Market Value per MWp',
        subtitle='Revenue per installed MWp DC · with & without neg-price hours',
        ytitle='€ / MWp / year',
        kind='dual_area',
        traces=[
            (yst['Yearly_Value_per_MWp_DC_EUR'], 'incl. neg-price hours', _CLAUDE['blue']),
            (yst['Yearly_Value_per_MWp_DC_EUR_excl_neg'], 'excl. neg-price hours', _CLAUDE['sage']),
        ],
        mask_partial=False,
    ),
    dict(
        title='Capture Price vs Day-Ahead Average',
        subtitle='Volume-weighted solar price compared with flat Day-Ahead average',
        ytitle='€ / MWh',
        kind='triple_line',
        traces=[
            (yst['Yearly_PV_Weighted_Price'], 'Capture price (incl. neg)', _CLAUDE['blue']),
            (yst['Yearly_PV_Weighted_Price_excl_neg'], 'Capture price (excl. neg)', _CLAUDE['sage']),
            (yst['Yearly_Avg_DA_Price'], 'Day-Ahead average', _CLAUDE['muted']),
        ],
        mask_partial=False,
    ),
    dict(
        title='Solar Capture Rate',
        subtitle='Capture price as % of Day-Ahead average · with & without neg-price hours',
        ytitle='%',
        kind='dual_area',
        traces=[
            (yst['Yearly_Profile_Factor'], 'incl. neg-price hours', _CLAUDE['blue']),
            (yst['Yearly_Profile_Factor_excl_neg'], 'excl. neg-price hours', _CLAUDE['sage']),
        ],
        mask_partial=False,
    ),
]


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _gradient_fills(x_years, y_vals, hex_color, n_layers=6):
    """Stack semi-transparent fills to fake a vertical gradient under a line."""
    traces = []
    r, g, b = _hex_to_rgb(hex_color)
    for i in range(n_layers):
        frac = (i + 1) / n_layers
        y_layer = [None if (v is None or pd.isna(v)) else v * frac for v in y_vals]
        alpha = 0.10 + 0.05 * (n_layers - i) / n_layers
        traces.append(go.Scatter(
            x=x_years, y=y_layer, mode='lines',
            line=dict(width=0),
            fill='tozeroy',
            fillcolor=f'rgba({r},{g},{b},{alpha:.3f})',
            hoverinfo='skip', showlegend=False,
        ))
    return traces


def _bar_gradient_shapes(x_years, y_vals, hex_color, x_axis='x', y_axis='y'):
    """Per-bar vertical gradient via stacked thin rects."""
    shapes = []
    r, g, b = _hex_to_rgb(hex_color)
    half_w = 0.32
    n = 14
    for xi, v in enumerate(y_vals):
        if v is None or pd.isna(v) or v == 0:
            continue
        for k in range(n):
            y0 = v * (k / n)
            y1 = v * ((k + 1) / n)
            alpha = 0.30 + 0.65 * (k / max(n - 1, 1))
            shapes.append(dict(
                type='rect', xref=x_axis, yref=y_axis,
                x0=xi - half_w, x1=xi + half_w, y0=y0, y1=y1,
                line=dict(width=0),
                fillcolor=f'rgba({r},{g},{b},{alpha:.3f})',
                layer='below',
            ))
    return shapes


_font_family = 'Inter, Helvetica Neue, Arial, sans-serif'


def _build_themed_fig(slide):
    fig = go.Figure()
    if slide.get('mask_partial'):
        years = [yl for yl, yv in zip(_years, yst['year'].tolist()) if yv <= _last_complete_year]
        slide_traces_new = []
        for tup in slide['traces']:
            y = tup[0]
            y_filt = y[yst['year'] <= _last_complete_year].reset_index(drop=True)
            slide_traces_new.append((y_filt, *tup[1:]))
        slide = {**slide, 'traces': slide_traces_new}
    else:
        years = _years
    kind = slide['kind']

    if kind == 'area_gradient':
        y, name, color = slide['traces'][0]
        y_list = list(y)
        for t in _gradient_fills(years, y_list, color):
            fig.add_trace(t)
        fig.add_trace(go.Scatter(
            x=years, y=y_list, mode='lines+markers', name=name,
            line=dict(color=color, width=3.5, shape='spline', smoothing=0.8),
            marker=dict(size=10, color=color, line=dict(color='white', width=2)),
        ))

    elif kind == 'dual_area':
        for y, name, color in slide['traces']:
            y_list = list(y)
            for t in _gradient_fills(years, y_list, color, n_layers=5):
                fig.add_trace(t)
            fig.add_trace(go.Scatter(
                x=years, y=y_list, mode='lines+markers', name=name,
                line=dict(color=color, width=3.2, shape='spline', smoothing=0.7),
                marker=dict(size=9, color=color, line=dict(color='white', width=2)),
            ))

    elif kind == 'triple_line':
        for y, name, color in slide['traces']:
            y_list = list(y)
            is_ref = 'Day-Ahead' in name
            fig.add_trace(go.Scatter(
                x=years, y=y_list, mode='lines+markers', name=name,
                line=dict(color=color, width=2.8 if not is_ref else 2.0,
                          dash='dot' if is_ref else 'solid',
                          shape='spline', smoothing=0.6),
                marker=dict(size=9 if not is_ref else 7, color=color,
                            line=dict(color='white', width=2)),
            ))

    elif kind == 'bar_gradient':
        y, name, color = slide['traces'][0]
        y_list = list(y)
        if slide.get('mask_partial'):
            pairs = [(yr, v) for yr, v in zip(years, y_list) if not (v is None or pd.isna(v))]
            x_used = [p[0] for p in pairs]
            y_used = [p[1] for p in pairs]
        else:
            x_used, y_used = years, y_list
        fig.add_trace(go.Bar(
            x=x_used, y=y_used, name=name,
            marker=dict(color=color, opacity=0.0),
            hovertemplate='%{x}: %{y}<extra></extra>',
            showlegend=True,
        ))
        for shp in _bar_gradient_shapes(list(range(len(x_used))), y_used, color):
            fig.add_shape(**shp)

    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:30px;color:{_CLAUDE['ink']};font-weight:700'>{slide['title']}</span><br>"
                 f"<span style='font-size:15px;color:{_CLAUDE['muted']};font-weight:400'>{slide['subtitle']}</span>",
            x=0.06, y=0.94, xanchor='left',
        ),
        paper_bgcolor=_CLAUDE['bg'],
        plot_bgcolor=_CLAUDE['bg'],
        font=dict(family=_font_family, color=_CLAUDE['ink_soft'], size=14),
        margin=dict(l=80, r=60, t=140, b=110),
        xaxis=dict(
            title=dict(text='Year', font=dict(size=13, color=_CLAUDE['muted'])),
            type='category', showgrid=False,
            linecolor=_CLAUDE['panel_edge'], linewidth=1,
            tickfont=dict(size=12, color=_CLAUDE['ink_soft']),
            ticks='outside', tickcolor=_CLAUDE['panel_edge'],
            categoryorder='array', categoryarray=years,
        ),
        yaxis=dict(
            title=dict(text=slide['ytitle'], font=dict(size=13, color=_CLAUDE['muted'])),
            rangemode='tozero',
            gridcolor=_CLAUDE['grid'], gridwidth=1,
            zerolinecolor=_CLAUDE['panel_edge'], zerolinewidth=1,
            tickfont=dict(size=12, color=_CLAUDE['ink_soft']),
        ),
        legend=(
            dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02,
                 bgcolor='rgba(250,249,245,0.90)', bordercolor=_CLAUDE['panel_edge'], borderwidth=1,
                 font=dict(size=12, color=_CLAUDE['ink_soft']))
            if slide['title'] == 'Solar Capture Rate'
            else dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='left', x=0.0,
                     bgcolor='rgba(0,0,0,0)', font=dict(size=12, color=_CLAUDE['ink_soft']))
        ),
        showlegend=(len(slide['traces']) > 1),
        width=1600, height=1000,
    )

    # Brand strip + footer annotations
    fig.add_annotation(
        text=f"<b style='color:{_CLAUDE['accent']}'>SOLAR · NL</b>  ·  Day-Ahead market analysis",
        xref='paper', yref='paper', x=0.06, y=1.06,
        showarrow=False, font=dict(size=11, color=_CLAUDE['muted'], family=_font_family),
        align='left',
    )
    fig.add_annotation(
        text=f"Source: NED.nl (generation) · EPEX/ENTSO-E (Day-Ahead prices)   |   {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}",
        xref='paper', yref='paper', x=0.06, y=-0.18,
        showarrow=False, font=dict(size=11, color=_CLAUDE['muted'], family=_font_family),
        align='left', xanchor='left',
    )
    # Accent rule under title
    fig.add_shape(
        type='line', xref='paper', yref='paper',
        x0=0.06, x1=0.16, y0=1.01, y1=1.01,
        line=dict(color=_CLAUDE['accent'], width=3),
    )
    if slide['title'] == 'Annual Market Value per MWp':
        v22 = float(yst.loc[yst['year'] == 2022, 'Yearly_Value_per_MWp_DC_EUR'].iloc[0])
        v22x = float(yst.loc[yst['year'] == 2022, 'Yearly_Value_per_MWp_DC_EUR_excl_neg'].iloc[0])
        fig.update_layout(separators=',.', yaxis=dict(
            title=dict(text=slide['ytitle'], font=dict(size=13, color=_CLAUDE['muted'])),
            range=[0, 80000], gridcolor=_CLAUDE['grid'], gridwidth=1,
            zerolinecolor=_CLAUDE['panel_edge'], zerolinewidth=1,
            tickfont=dict(size=12, color=_CLAUDE['ink_soft']),
            tickformat=',.0f',
        ))
        fig.add_annotation(
            x='2022', y=78000, xref='x', yref='y',
            text=f"<b>2022 spike</b><br>incl. neg: €{v22:,.0f}/MWp<br>excl. neg: €{v22x:,.0f}/MWp".replace(',', '.'),
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor=_CLAUDE['accent'], ax=0, ay=-35,
            bgcolor='rgba(250,249,245,0.95)', bordercolor=_CLAUDE['accent'], borderwidth=1, borderpad=6,
            font=dict(size=12, color=_CLAUDE['ink'], family=_font_family),
            align='left',
        )
    if slide['title'] == 'Capture Price vs Day-Ahead Average':
        p22 = float(yst.loc[yst['year'] == 2022, 'Yearly_PV_Weighted_Price'].iloc[0])
        p22x = float(yst.loc[yst['year'] == 2022, 'Yearly_PV_Weighted_Price_excl_neg'].iloc[0])
        da22 = float(yst.loc[yst['year'] == 2022, 'Yearly_Avg_DA_Price'].iloc[0])
        fig.update_layout(yaxis=dict(
            title=dict(text=slide['ytitle'], font=dict(size=13, color=_CLAUDE['muted'])),
            range=[0, 110], gridcolor=_CLAUDE['grid'], gridwidth=1,
            zerolinecolor=_CLAUDE['panel_edge'], zerolinewidth=1,
            tickfont=dict(size=12, color=_CLAUDE['ink_soft']),
        ))
        fig.add_annotation(
            x='2022', y=105, xref='x', yref='y',
            text=f"<b>2022 spike</b><br>Capture incl. neg: €{p22:.0f}/MWh<br>Capture excl. neg: €{p22x:.0f}/MWh<br>DA avg: €{da22:.0f}/MWh",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor=_CLAUDE['accent'], ax=0, ay=-35,
            bgcolor='rgba(250,249,245,0.95)', bordercolor=_CLAUDE['accent'], borderwidth=1, borderpad=6,
            font=dict(size=12, color=_CLAUDE['ink'], family=_font_family),
            align='left',
        )
    return fig


def _fmt_num(v, fmt):
    if v is None or pd.isna(v):
        return ''
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def _build_table_fig():
    yst_desc = yst.sort_values('year', ascending=False).reset_index(drop=True)
    headers = [
        'Year<br><span style="font-size:10px;color:#8C8377">(* preliminary)</span>',
        'Installed PV<br>capacity (GWp)<br><span style="font-size:10px;color:#8C8377">avg</span>',
        'PV energy<br>(TWh/y)<br><span style="font-size:10px;color:#8C8377">NED.nl</span>',
        'MWh / MWp<br>installed',
        'MWh / MWp<br><span style="font-size:10px;color:#8C8377">excl. neg</span>',
        'Curtailment<br>(%)',
        'Neg-price<br>hours (h/y)',
        'Market value<br>(€/MWp/y)',
        'Market value<br>(€/MWp/y)<br><span style="font-size:10px;color:#8C8377">excl. neg</span>',
        'DA avg price<br>(€/MWh)',
        'Capture price<br>(€/MWh)',
        'Capture price<br>(€/MWh)<br><span style="font-size:10px;color:#8C8377">excl. neg</span>',
        'Capture rate<br>(%)',
        'Capture rate<br>(%)<br><span style="font-size:10px;color:#8C8377">excl. neg</span>',
    ]
    cells = [
        yst_desc['year_label'].tolist(),
        [_fmt_num(v, '{:,.1f}') for v in yst_desc['Yearly_Installed_Capacity_GWp_DC']],
        [_fmt_num(v, '{:,.1f}') for v in yst_desc['Yearly_PV_Energy_TWh']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_MWh_per_MWp']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_MWh_per_MWp_excl_neg']],
        [_fmt_num(v, '{:.0f}%') for v in yst_desc['Yearly_Curtailment_Pct']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_Neg_Hours']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_Value_per_MWp_DC_EUR']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_Value_per_MWp_DC_EUR_excl_neg']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_Avg_DA_Price']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_PV_Weighted_Price']],
        [_fmt_num(v, '{:,.0f}') for v in yst_desc['Yearly_PV_Weighted_Price_excl_neg']],
        [_fmt_num(v, '{:.0f}%') for v in yst_desc['Yearly_Profile_Factor']],
        [_fmt_num(v, '{:.0f}%') for v in yst_desc['Yearly_Profile_Factor_excl_neg']],
    ]
    n_rows = len(yst_desc)
    row_fill = [
        [_CLAUDE['panel'] if i % 2 == 0 else _CLAUDE['bg'] for i in range(n_rows)]
    ] * len(headers)

    tfig = go.Figure(data=[go.Table(
        columnwidth=[60, 80, 70, 70, 75, 65, 70, 80, 85, 75, 75, 80, 70, 80],
        header=dict(
            values=headers,
            fill_color=_CLAUDE['accent'],
            font=dict(color='white', size=12, family=_font_family),
            align='center',
            height=70,
            line=dict(color=_CLAUDE['accent'], width=0),
        ),
        cells=dict(
            values=cells,
            fill_color=row_fill,
            font=dict(color=_CLAUDE['ink'], size=12, family=_font_family),
            align=['left'] + ['right'] * (len(headers) - 1),
            height=30,
            line=dict(color=_CLAUDE['panel_edge'], width=1),
        ),
    )])
    tfig.update_layout(
        title=dict(
            text=f"<span style='font-size:30px;color:{_CLAUDE['ink']};font-weight:700'>Yearly Solar PV Market Summary</span><br>"
                 f"<span style='font-size:15px;color:{_CLAUDE['muted']};font-weight:400'>Netherlands · Day-Ahead market · all metrics, with & without negative-price hours</span>",
            x=0.03, y=0.96, xanchor='left',
        ),
        paper_bgcolor=_CLAUDE['bg'],
        plot_bgcolor=_CLAUDE['bg'],
        font=dict(family=_font_family, color=_CLAUDE['ink_soft']),
        margin=dict(l=40, r=40, t=140, b=80),
        width=2400, height=1000,
    )
    tfig.add_annotation(
        text=f"<b style='color:{_CLAUDE['accent']}'>SOLAR · NL</b>  ·  Day-Ahead market analysis",
        xref='paper', yref='paper', x=0.03, y=1.06,
        showarrow=False, font=dict(size=11, color=_CLAUDE['muted'], family=_font_family),
    )
    tfig.add_annotation(
        text=f"Source: NED.nl (generation) · EPEX/ENTSO-E (Day-Ahead prices)   |   {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}",
        xref='paper', yref='paper', x=0.03, y=-0.06,
        showarrow=False, font=dict(size=11, color=_CLAUDE['muted'], family=_font_family),
        xanchor='left',
    )
    tfig.add_shape(
        type='line', xref='paper', yref='paper',
        x0=0.03, x1=0.10, y0=1.01, y1=1.01,
        line=dict(color=_CLAUDE['accent'], width=3),
    )
    return tfig


def _build_monthly_capture_rate_fig(years_to_plot=(2023, 2024, 2025)):
    ms = monthly_summary.copy()
    ms['year'] = ms['month'].astype(str).str.slice(0, 4).astype(int)
    ms['month_num'] = ms['month'].astype(str).str.slice(5, 7).astype(int)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    palette = [_CLAUDE['accent'], _CLAUDE['sage'], _CLAUDE['blue'], _CLAUDE['muted']]

    fig = go.Figure()
    for i, yr in enumerate(years_to_plot):
        sub = ms[ms['year'] == yr].sort_values('month_num')
        if sub.empty:
            continue
        color = palette[i % len(palette)]
        y_vals = sub['profile_factor'].tolist()
        x_vals = [month_labels[m - 1] for m in sub['month_num']]
        r, g, b = _hex_to_rgb(color)
        # Faint gradient fill only for most recent year (top line emphasis)
        if yr == years_to_plot[-1]:
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
            text=f"<span style='font-size:30px;color:{_CLAUDE['ink']};font-weight:700'>Monthly Solar Capture Rate</span><br>"
                 f"<span style='font-size:15px;color:{_CLAUDE['muted']};font-weight:400'>Netherlands · capture price ÷ Day-Ahead average · by year</span>",
            x=0.06, y=0.94, xanchor='left',
        ),
        paper_bgcolor=_CLAUDE['bg'], plot_bgcolor=_CLAUDE['bg'],
        font=dict(family=_font_family, color=_CLAUDE['ink_soft'], size=14),
        margin=dict(l=80, r=60, t=140, b=110),
        xaxis=dict(
            title=dict(text='Month', font=dict(size=13, color=_CLAUDE['muted'])),
            type='category', categoryorder='array', categoryarray=month_labels,
            showgrid=False, linecolor=_CLAUDE['panel_edge'], linewidth=1,
            tickfont=dict(size=12, color=_CLAUDE['ink_soft']),
            ticks='outside', tickcolor=_CLAUDE['panel_edge'],
        ),
        yaxis=dict(
            title=dict(text='%', font=dict(size=13, color=_CLAUDE['muted'])),
            rangemode='tozero',
            gridcolor=_CLAUDE['grid'], gridwidth=1,
            zerolinecolor=_CLAUDE['panel_edge'], zerolinewidth=1,
            tickfont=dict(size=12, color=_CLAUDE['ink_soft']),
            ticksuffix='%',
        ),
        legend=dict(orientation='v', yanchor='top', y=0.98, xanchor='right', x=0.98,
                    bgcolor='rgba(250,249,245,0.85)', bordercolor=_CLAUDE['panel_edge'], borderwidth=1,
                    font=dict(size=12, color=_CLAUDE['ink_soft'])),
        showlegend=True, width=1600, height=1000,
    )
    fig.add_annotation(
        text=f"<b style='color:{_CLAUDE['accent']}'>SOLAR · NL</b>  ·  Day-Ahead market analysis",
        xref='paper', yref='paper', x=0.06, y=1.06,
        showarrow=False, font=dict(size=11, color=_CLAUDE['muted'], family=_font_family),
    )
    fig.add_annotation(
        text=f"Source: NED.nl (generation) · EPEX/ENTSO-E (Day-Ahead prices)   |   {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}",
        xref='paper', yref='paper', x=0.06, y=-0.18,
        showarrow=False, font=dict(size=11, color=_CLAUDE['muted'], family=_font_family),
        xanchor='left',
    )
    fig.add_shape(type='line', xref='paper', yref='paper',
                  x0=0.06, x1=0.16, y0=1.01, y1=1.01,
                  line=dict(color=_CLAUDE['accent'], width=3))
    return fig


_pdf_path = 'solar_yearly_slides.pdf'
with _PdfPages(_pdf_path) as _pdf:
    for _slide in _pdf_slides:
        _fig = _build_themed_fig(_slide)
        _png_bytes = _fig.to_image(format='png', width=1600, height=1000, scale=2)
        _img = _imread(_io.BytesIO(_png_bytes), format='png')
        _f, _ax = _plt.subplots(figsize=(11.69, 8.27), dpi=200)  # A4 landscape
        _f.patch.set_facecolor(_CLAUDE['bg'])
        _ax.imshow(_img)
        _ax.axis('off')
        _pdf.savefig(_f, bbox_inches='tight', facecolor=_CLAUDE['bg'])
        _plt.close(_f)
    _mfig = _build_monthly_capture_rate_fig()
    _png_bytes = _mfig.to_image(format='png', width=1600, height=1000, scale=2)
    _img = _imread(_io.BytesIO(_png_bytes), format='png')
    _f, _ax = _plt.subplots(figsize=(11.69, 8.27), dpi=200)
    _f.patch.set_facecolor(_CLAUDE['bg'])
    _ax.imshow(_img); _ax.axis('off')
    _pdf.savefig(_f, bbox_inches='tight', facecolor=_CLAUDE['bg'])
    _plt.close(_f)

    _tfig = _build_table_fig()
    _png_bytes = _tfig.to_image(format='png', width=2400, height=1000, scale=2)
    _img = _imread(_io.BytesIO(_png_bytes), format='png')
    _f, _ax = _plt.subplots(figsize=(16.54, 8.27), dpi=200)  # A3 landscape-ish, wider for table
    _f.patch.set_facecolor(_CLAUDE['bg'])
    _ax.imshow(_img)
    _ax.axis('off')
    _pdf.savefig(_f, bbox_inches='tight', facecolor=_CLAUDE['bg'])
    _plt.close(_f)
print(f'PDF written: {_pdf_path}')


# --- Multi-page PDF, one page per slide, Claude-style design ---
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as _plt

# Claude palette
_BG = '#F5F0E8'        # warm cream
_INK = '#2C2826'       # dark espresso
_MUTED = '#6B5D52'     # warm taupe
_CORAL = '#D97757'     # Claude coral (primary)
_OLIVE = '#7A8450'     # secondary accent
_GREY = '#94928D'      # reference / DA line
_GRID = '#DDD4C5'      # subtle grid

_plt.rcParams.update({
    'figure.facecolor': _BG,
    'axes.facecolor': _BG,
    'savefig.facecolor': _BG,
    'axes.edgecolor': _MUTED,
    'axes.labelcolor': _INK,
    'axes.titlecolor': _INK,
    'xtick.color': _INK,
    'ytick.color': _INK,
    'text.color': _INK,
    'font.family': ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': True,
    'grid.color': _GRID,
    'grid.linewidth': 0.7,
    'grid.alpha': 0.8,
    'axes.axisbelow': True,
})

# Per-slide PDF rendering spec: (title, list-of-(y_series, label, color, style, fill?), y_unit, hide_preliminary)
_pdf_slides = [
    ('Installed PV capacity in NL',
     [(yst['Yearly_Installed_Capacity_GWp_DC'], 'Installed capacity', _CORAL, 'solid', False)],
     'GWp DC', False),
    ('PV energy produced — NED.nl',
     [(_twh_complete, 'PV energy', _CORAL, 'solid', False)],
     'TWh / year', True),
    ('MWh yield per MWp installed',
     [(_yield_incl, 'Including negative-price hours', _CORAL, 'solid', False),
      (_yield_excl, 'Excluding negative-price hours', _OLIVE, 'dashed', False)],
     'MWh / MWp', True),
    ('Curtailment — share of MWh produced during DA < 0',
     [(yst['Yearly_Curtailment_Pct'], 'Curtailment', _CORAL, 'solid', False)],
     '%', False),
    ('Hours with negative day-ahead price',
     [(yst['Yearly_Neg_Hours'], 'Negative-price hours', _CORAL, 'solid', False)],
     'hours / year', False),
    ('Annual market value per MWp installed',
     [(yst['Yearly_Value_per_MWp_DC_EUR'], 'Including negative-price hours', _CORAL, 'solid', False),
      (yst['Yearly_Value_per_MWp_DC_EUR_excl_neg'], 'Excluding negative-price hours', _OLIVE, 'dashed', False)],
     '€ / MWp / year', False),
    ('Solar capture price vs day-ahead average',
     [(yst['Yearly_PV_Weighted_Price'], 'Capture price (incl. neg)', _CORAL, 'solid', False),
      (yst['Yearly_PV_Weighted_Price_excl_neg'], 'Capture price (excl. neg)', _OLIVE, 'dashed', False),
      (yst['Yearly_Avg_DA_Price'], 'Day-ahead average', _GREY, 'dotted', False)],
     '€ / MWh', False),
    ('Solar capture rate',
     [(yst['Yearly_Profile_Factor_excl_neg'], 'Capture rate (excl. neg)', _OLIVE, 'solid', True),
      (yst['Yearly_Profile_Factor'], 'Capture rate (incl. neg)', _CORAL, 'solid', True)],
     '%', False),
]

_x_labels = yst['year_label'].tolist()
_x_pos = list(range(len(_x_labels)))

with PdfPages('solar_yearly_slides.pdf') as pdf:
    for page_idx, (title, series_list, yunit, _hide_prelim) in enumerate(_pdf_slides):
        fig_pdf, ax = _plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
        fig_pdf.subplots_adjust(left=0.10, right=0.92, top=0.82, bottom=0.16)

        for (y, label, color, style, do_fill) in series_list:
            y_vals = list(y)
            if do_fill:
                ax.fill_between(_x_pos, 0, y_vals, color=color, alpha=0.22, linewidth=0)
            ax.plot(_x_pos, y_vals, label=label, color=color, linestyle=style,
                    linewidth=2.4, marker='o', markersize=7, markerfacecolor=color,
                    markeredgecolor=_BG, markeredgewidth=1.5)

        ax.set_xticks(_x_pos)
        ax.set_xticklabels(_x_labels, rotation=0)
        ax.set_ylabel(yunit, fontsize=12, color=_MUTED, labelpad=10)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', length=0, pad=8)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color(_MUTED)
            ax.spines[spine].set_linewidth(0.8)

        # Title block (left-aligned, above plot)
        fig_pdf.text(0.10, 0.92, title, fontsize=22, fontweight='600', color=_INK, ha='left', va='top')
        fig_pdf.text(0.10, 0.875, 'Netherlands · NED.nl × EPEX day-ahead', fontsize=11, color=_MUTED, ha='left', va='top')

        # Legend (only if multi-line)
        if len(series_list) > 1:
            leg = ax.legend(loc='upper left', frameon=False, fontsize=10.5,
                            labelcolor=_INK, handlelength=2.5, borderaxespad=0)

        # Footer
        fig_pdf.text(0.10, 0.04, '* = preliminary (year incomplete)', fontsize=9, color=_MUTED, ha='left')
        fig_pdf.text(0.92, 0.04, f'{page_idx + 1} / {len(_pdf_slides)}', fontsize=9, color=_MUTED, ha='right')

        pdf.savefig(fig_pdf, bbox_inches=None)
        _plt.close(fig_pdf)

print("Wrote solar_yearly_slides.pdf")




