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
monthly_summary = monthly_summary[['month', 'Total_PV_Energy_GWh', 'Value_per_MWp_DC_EUR', 'Avg_DA_Price', 'PV_Weighted_Price', 'profile_factor', 'Installed_Capacity_GWp_DC']]

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

# Calculate July 1st installed capacity for each year
def get_july_1st_capacity(year):
    july_1st = pd.Timestamp(f'{year}-07-01', tz='Europe/Amsterdam')
    return interpolate_installed_capacity(july_1st)

yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC'] = yearly_summary_for_table['year'].apply(get_july_1st_capacity) / 1000
# Calculate MWh/MWp installed produced
yearly_summary_for_table['Yearly_MWh_per_MWp'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / (yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC']*1000 )
yearly_summary_for_table = yearly_summary_for_table.round(1)

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
        'Capture Rate PV (%)',
        'Solar Capture Price (€/MWh)',
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
    row_heights=[0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.34]
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
            values=['Year', 'PV Energy produced (GWh/y)', 'Installed PV Capacity in NL (GWp) Average', 'MWh yield / MWp installed', 'Annual Market value (EUR/MWp/y)', 'Day-Ahead linear avg price (EUR/MWh)', 'PV Capture price (EUR/MWh)', 'Capture rate of PV (%)'],
            font=dict(size=10),
            align='left'
        ),
        cells=dict(
            values=[
                yearly_summary_for_table['year'].astype(str)[::-1],
                [format_number(x) for x in yearly_summary_for_table['Yearly_PV_Energy_GWh'][::-1]],
                [format_gwp(x) for x in yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_MWh_per_MWp'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Value_per_MWp_DC_EUR'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Avg_DA_Price'][::-1]],
                [format_number(x) for x in yearly_summary_for_table['Yearly_PV_Weighted_Price'][::-1]],
                [format_percentage(x) for x in yearly_summary_for_table['Yearly_Profile_Factor'][::-1]]
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

fig.update_yaxes(title_text='Capture rate (%)', row=5, col=1, range=[0, 100])
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
fig.update_yaxes(title_text='Capture Price (€/MWh)', row=6, col=1)
fig.update_xaxes(title_text='', row=6, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])

# Update layout with individual legends for each subplot
#fig.update_layout(
    #title_text='Analysis on PV value (NL), EPEX spot prices + PV production of NED.nl',
    #margin=dict(b=50)
#)

# Add individual legends for each subplot
fig.update_layout(
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
        values=['Month', 'PV Energy produced (GWh/month)', 'Solar PV capacity NL (GWp)', 'Market value PV (EUR/MWp/year)', 'Day-Ahead average price (EUR/MWh)', 'PV Capture price (EUR/MWh)', 'PV Capture rate (%)'],
        font=dict(size=10),
        align='left'
    ),
            cells=dict(
            values=[
                monthly_summary_rounded_reversed['month'].astype(str),
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 1].round(0)],
                [format_gwp(x) for x in monthly_summary_rounded_reversed.iloc[:, 6]],  # Installed Capacity column
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
    title_text='Monthly Summary Table (Analysis on PV value (NL), EPEX spot prices + PV production of NED.nl)',
    margin=dict(l=0, r=0, t=50, b=0)
)

# Write both figures to separate files
table_fig.write_html('monthly_summary_table.html', auto_open=True)

fig.write_html('solar_production_plot_v3.html', auto_open=True)




