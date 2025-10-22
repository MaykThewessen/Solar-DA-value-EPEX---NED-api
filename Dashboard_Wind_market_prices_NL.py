import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import glob
import warnings
os.system('clear')


# --- Load all monthly DA_prices and Wind data files ---
# Find all DA_prices and Wind files
price_files = sorted(glob.glob('data/DA_prices/DA_prices_20*.csv'))
wind_files  = sorted(glob.glob('data/NED_Wind/data_NED_Wind_20*.csv'))

# Exclude combined file from price_files
price_files = [f for f in price_files if 'combined' not in f]

# Load and concatenate all price files
price_dfs = []
for f in price_files:
    df = pd.read_csv(f)
    # Remove white spaces at beginning and end of all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    price_dfs.append(df)
df_prices = pd.concat(price_dfs, ignore_index=True)

# Load and concatenate all Wind files
wind_dfs = []
for f in wind_files:
    df = pd.read_csv(f)
    # Remove white spaces at beginning and end of all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    wind_dfs.append(df)
df_wind = pd.concat(wind_dfs, ignore_index=True)

#print(df_prices)
#print(df_wind)


# Merge the two dataframes on the 'time' column
df_combined = pd.merge(df_prices, df_wind, on='time', how='left')

# Remove white spaces at beginning and end of all string columns in combined dataframe
for col in df_combined.select_dtypes(include=['object']).columns:
    df_combined[col] = df_combined[col].astype(str).str.strip()

# Filter data to only include complete months up to and including September 2025
# Since today is October 6, 2025, we only want complete months
last_complete_month = pd.Timestamp('2025-09-30 23:59:59', tz='Europe/Amsterdam')
df_combined = df_combined[df_combined['time'] <= last_complete_month]

print(f"Data filtered to include only complete months up to September 2025")
print(f"Date range: {df_combined['time'].min()} to {df_combined['time'].max()}")

#df_combined = df_combined.fillna(0)
#df_combined = df_combined.set_index('time').interpolate(method='time').reset_index()


df_combined['Wind_value'] = df_combined['Wind_production_MW'] * df_combined['DA_price']


# Create installed capacity column in MW using a linear fit (extrapolation allowed)
from datetime import datetime

# Known data points for installed capacity (AC) at year-end
capacity_points = [
    (pd.Timestamp('2017-12-31', tz='Europe/Amsterdam'), 3245), # MW AC Onshore Wind only
    (pd.Timestamp('2018-12-31', tz='Europe/Amsterdam'), 3436), # MW AC
    (pd.Timestamp('2019-12-31', tz='Europe/Amsterdam'), 3527), # MW AC
    (pd.Timestamp('2020-12-31', tz='Europe/Amsterdam'), 4188),
    (pd.Timestamp('2021-12-31', tz='Europe/Amsterdam'), 5186),
    (pd.Timestamp('2022-12-31', tz='Europe/Amsterdam'), 6131),
    (pd.Timestamp('2023-12-31', tz='Europe/Amsterdam'), 6757),  # MW AC
    (pd.Timestamp('2024-12-31', tz='Europe/Amsterdam'), 6965),  # MW AC
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 6965 + 100),  # MW AC # lower installed Wind estimate update
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 6965 + 100 + 50),  # MW AC
]

print(capacity_points)



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

import matplotlib.pyplot as plt

# Prepare data for plotting
plt.figure(figsize=(10, 5))

# Plot fitted installed capacity (MW) from df_combined over time
plt.plot(
    df_combined['time'].dt.tz_localize(None),
    df_combined['installed_capacity_MW'],
    label='Fitted Installed Capacity (MW)', color='tab:blue'
)

# Plot the original capacity_points as red scatter points
cap_dates = [dt.tz_localize(None) for dt, cap in capacity_points]
cap_values = [cap for dt, cap in capacity_points]

plt.scatter(
    cap_dates,
    cap_values,
    color='tab:red',
    label='Known Data Points',
    zorder=5
)

plt.title('Installed Wind Capacity: Fitted vs Known Data Points')
plt.xlabel('Date')
plt.ylabel('Installed Capacity (MW)')
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('wind_installed_capacity_vs_known_points.pdf', bbox_inches='tight')
plt.close()




# summarize per month
# Remove timezone before converting to Period to avoid warning
df_combined['month'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M')

monthly_summary = (
    df_combined.groupby('month').agg({
        'Wind_production_MW': 'sum',
        'Wind_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean'
    }).reset_index().round(1)
)

# Calculate derived columns in the correct order to match expected format
monthly_summary['Total_Wind_Energy_GWh'] = round(monthly_summary['Wind_production_MW']/1000, 0)
monthly_summary['Value_per_MW_AC_EUR'] = round(monthly_summary['Wind_value'] / monthly_summary['installed_capacity_MW'], 0)
monthly_summary['Avg_DA_Price'] = round(monthly_summary['DA_price'], 1)

# Calculate Wind weighted price for each month
monthly_summary['Wind_Weighted_Price'] = round(monthly_summary.apply(
    lambda row: (df_combined[df_combined['month'] == row['month']]['Wind_production_MW'] * 
                 df_combined[df_combined['month'] == row['month']]['DA_price']).sum() / 
                df_combined[df_combined['month'] == row['month']]['Wind_production_MW'].sum() 
                if df_combined[df_combined['month'] == row['month']]['Wind_production_MW'].sum() > 0 else float('nan'), axis=1
), 1)

monthly_summary['profile_factor'] = round((monthly_summary['Wind_Weighted_Price'] / monthly_summary['Avg_DA_Price'])*100, 1)
monthly_summary['Installed_Capacity_MW_AC'] = round(monthly_summary['installed_capacity_MW'], 0)

# Reorder columns to match expected format
monthly_summary = monthly_summary[['month', 'Wind_production_MW', 'Wind_value', 'installed_capacity_MW', 
                                   'DA_price', 'Total_Wind_Energy_GWh', 'Value_per_MW_AC_EUR', 
                                   'Avg_DA_Price', 'Wind_Weighted_Price', 'profile_factor', 'Installed_Capacity_MW_AC']]

print("\nMonthly Summary:")
# Round first 3 columns to 0 digits
monthly_summary_rounded = monthly_summary.copy()
monthly_summary_rounded.iloc[:, 1:4] = monthly_summary_rounded.iloc[:, 1:4].round(1)
#print(monthly_summary_rounded.to_string(index=False, float_format='%.0f').replace(',', '.'))
monthly_summary_df = pd.DataFrame(monthly_summary_rounded)
print(monthly_summary_df)


# Plot the combined dataframe using plotly
import plotly.graph_objs as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import plotly.colors  # type: ignore

# --- Prepare data for plotting ---
# 1. PV power and Day-ahead price (hourly)

# 2. Monthly PV yield (sum), PV value (sum), and PV_power_weighted_DA_price (monthly avg)
# Remove timezone before converting to Period to avoid warning
df_combined['month_date'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()

monthly = (
    df_combined.groupby('month_date').agg({
        'Wind_production_MW': 'sum',
        'Wind_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean'
    }).reset_index()
)

# Calculate derived columns
monthly['Monthly_Wind_Energy_MWh'] = round(monthly['Wind_production_MW'], 1)
monthly['Monthly_Value_per_MW_AC_EUR'] = round(monthly['Wind_value'] / monthly['installed_capacity_MW'], 1)
monthly['Monthly_Installed_Capacity_MW'] = monthly['installed_capacity_MW']
monthly['Monthly_Avg_DA_Price'] = round(monthly['DA_price'], 1)


# Calculate Wind weighted price for each month
monthly['Monthly_Wind_Power_Weighted_DA_Price'] = monthly.apply(
    lambda row: (df_combined[df_combined['month_date'] == row['month_date']]['Wind_production_MW'] * 
                 df_combined[df_combined['month_date'] == row['month_date']]['DA_price']).sum() / 
                df_combined[df_combined['month_date'] == row['month_date']]['Wind_production_MW'].sum() 
                if df_combined[df_combined['month_date'] == row['month_date']]['Wind_production_MW'].sum() > 0 else float('nan'), axis=1
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
yearly_weighted_prices = df_combined.groupby(df_combined['time'].dt.year).agg({
    'Wind_production_MW': 'sum',
    'DA_price': 'mean'
}).reset_index()

# Calculate weighted price manually
yearly_weighted_prices['Yearly_Wind_Weighted_Price'] = yearly_weighted_prices.apply(
    lambda row: (df_combined[df_combined['time'].dt.year == row['time']]['Wind_production_MW'] * 
                 df_combined[df_combined['time'].dt.year == row['time']]['DA_price']).sum() / 
                df_combined[df_combined['time'].dt.year == row['time']]['Wind_production_MW'].sum() 
                if df_combined[df_combined['time'].dt.year == row['time']]['Wind_production_MW'].sum() > 0 else float('nan'), axis=1
)

yearly_weighted_prices = yearly_weighted_prices.rename(columns={'time': 'year'})
yearly_weighted_prices = yearly_weighted_prices[['year', 'Yearly_Wind_Weighted_Price']]

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
yearly_summary_for_table = yearly_summary_for_table.round(1)

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
    rows=6, cols=1, shared_xaxes=False, vertical_spacing=0.08,
    subplot_titles=(
        'Total Wind Yield in NL',
        'Installed Wind Capacity',
        'Yield normalized per installed capacity',
        'Market Value per installed capacity',
        'Profile Factor Wind (%)',
        ' '
    ),
    specs=[
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"type": "table"}]
    ],
    row_heights=[0.22, 0.22, 0.22, 0.22, 0.22, 0.35]  # Increased heights for better readability
)



# Sixth subplot: Yearly summary table (rows reversed)
fig.add_trace(
    go.Table(
        header=dict(
            values=['Year', 'Wind Energy produced (GWh/y)', 'Installed Wind Capacity in NL (MW) mid-year', 'MWh yield / MW installed', 'Annual Market value (EUR/MW/y)', 'Day-Ahead linear avg price (EUR/MWh)', 'Wind-profile weighted price (EUR/MWh)', 'Profile Factor of Wind (%)'],
            font=dict(size=10),
            align='left'
        ),
        cells=dict(
            values=[
                [str(year) + (' (Jan-Sep)' if year == 2025 else '') for year in yearly_summary_for_table['year'][::-1]],
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
    row=6, col=1
)

# Create custom color scheme with distinct, distinguishable colors for each year
years_list = sorted(monthly['year'].unique())
num_years = len(years_list)

# Define a palette of distinct colors that are easy to differentiate
distinct_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#C71585',  # Medium Violet Red
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#ff9896',  # Light red
    '#98df8a',  # Light green
    '#ffbb78',  # Light orange
    '#aec7e8',  # Light blue
    '#c5b0d5',  # Light purple
]

color_scheme = []
for i, year in enumerate(years_list):
    if i < len(distinct_colors):
        color_scheme.append(distinct_colors[i])
    else:
        # If we have more years than colors, cycle through the palette
        color_scheme.append(distinct_colors[i % len(distinct_colors)])

# Create year to color mapping
year_to_color = dict(zip(years_list, color_scheme))

# First subplot: Monthly Wind energy production (lines per year)
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
            y=year_data['Monthly_Wind_Energy_MWh']/1000, 
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
        row=1, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='GWh produced', row=1, col=1, secondary_y=False)
fig.update_xaxes(title_text='', row=1, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])

# Second subplot: Installed Wind Capacity
# Create date range for the fitted curve
start_date = pd.Timestamp('2018-01-01', tz='Europe/Amsterdam')
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

# Third subplot: Monthly MWh yield per MW installed (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    # Sort by month to ensure January to December order
    year_data = year_data.sort_values('month_num')
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_Wind_Energy_MWh'] / (year_data['Monthly_Installed_Capacity_MW']), 
            name=f'{year}', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>MWh/MW:</b> %{y:.1f}<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=3, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='MWh per MW', row=3, col=1)
fig.update_xaxes(title_text='', row=3, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])

# Fourth subplot: Monthly Wind Market Value (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    # Sort by month to ensure January to December order
    year_data = year_data.sort_values('month_num')
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_Value_per_MW_AC_EUR'], 
            name=f'{year}', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>Market Value:</b> %{y:.1f} EUR/MW<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=4, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='€ per MW', row=4, col=1)
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
            hovertemplate='<b>Year:</b> %{fullData.name}<br><b>Month:</b> %{customdata}<br><b>Profile Factor:</b> %{y:.1f}%<extra></extra>',
            customdata=year_data['month_date'].dt.strftime('%B'),
            legendgroup=f'group_{year}',
            showlegend=True,
            line_color=year_to_color[year]
        ),
        row=5, col=1, secondary_y=False
    )

fig.update_yaxes(title_text='Profile Factor (%)', row=5, col=1)
fig.update_xaxes(title_text='', row=5, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], range=[0.5, 12.5])

# Update layout with individual legends for each subplot
#fig.update_layout(
    #title_text='Analysis on Wind value (NL), EPEX spot prices + Wind production of NED.nl',
    #margin=dict(b=50)
#)

# Add individual legends for each subplot
fig.update_layout(
    height=1200,  # Increase overall figure height
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top'
    )
)

# Update legend for each subplot to show only relevant traces
# Skip the first trace (table) and only update scatter plots
num_years = len(sorted(monthly['year'].unique()))

# First subplot: Wind Energy (lines per year)
for i, year in enumerate(sorted(monthly['year'].unique())):
    fig.data[i + 1].update(showlegend=True)

# Second subplot: Installed Capacity (3 traces: fitted curve, capacity points, hourly production)
# These are already set to show legend

# Third subplot: MWh yield per MW (lines per year)
for i, year in enumerate(sorted(monthly['year'].unique())):
    fig.data[i + 1 + num_years + 3].update(showlegend=True)  # +3 for the 3 traces in subplot 2

# Fourth subplot: Market Value (lines per year)
for i, year in enumerate(sorted(monthly['year'].unique())):
    fig.data[i + 1 + 2*num_years + 3].update(showlegend=True)

# Fifth subplot: Profile Factor (lines per year)
for i, year in enumerate(sorted(monthly['year'].unique())):
    fig.data[i + 1 + 3*num_years + 3].update(showlegend=True)

# Create a separate table figure
# Format numbers with thousands separators and percentage for profile factor
# Sort monthly_summary_rounded in reverse chronological order
monthly_summary_rounded_reversed = monthly_summary_rounded.sort_values('month', ascending=False).reset_index(drop=True)

table_fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Month', 'Wind Energy produced (GWh/month)', 'Wind generation capacity NL (MW AC)', 'Market value Wind (EUR/MW/year)', 'Day-Ahead average price (EUR/MWh)', 'Wind-profile weighted price (EUR/MWh)', 'Wind Capture Rate NL(profile factor %)'],
        font=dict(size=10),
        align='left'
    ),
            cells=dict(
            values=[
                monthly_summary_rounded_reversed['month'].astype(str),
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 5]],  # Total_Wind_Energy_GWh
                [format_mwp(x) for x in monthly_summary_rounded_reversed.iloc[:, 10]],   # Installed_Capacity_MW_AC
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 6]],  # Value_per_MW_AC_EUR
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 7]],  # Avg_DA_Price
                [format_number(x) for x in monthly_summary_rounded_reversed.iloc[:, 8]],  # Wind_Weighted_Price
                [format_percentage(x) for x in monthly_summary_rounded_reversed.iloc[:, 9]]  # profile_factor
            ],
        font=dict(size=9),
        align='left',
        height=20
    )
)])

table_fig.update_layout(
    title_text='Monthly Summary Table (Analysis on Wind value (NL), EPEX spot prices + Wind production of NED.nl) - Data up to September 2025',
    margin=dict(l=0, r=0, t=50, b=0)
)

# Write both figures to separate files
table_fig.write_html('wind_monthly_summary_table.html', auto_open=True)
fig.write_html('wind_production_plot_v3.html', auto_open=True)




