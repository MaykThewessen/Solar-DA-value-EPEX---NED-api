import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import glob
import warnings
os.system('clear')


# TODO: add a function to only export PV power when DA prices are non-negative
# Graph shows monthly values per year with each year as a different line on the month x-axis graph


# --- Load all monthly DA_prices and PV data files ---
# Find all DA_prices and PV files
price_files = sorted(glob.glob('data/DA_prices_20*.csv'))
pv_files = sorted(glob.glob('data/data_export_NED_PV_20*.csv'))

# Exclude combined file from price_files
price_files = [f for f in price_files if 'combined' not in f]

# Load and concatenate all price files
price_dfs = []
for f in price_files:
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    price_dfs.append(df)
df_prices = pd.concat(price_dfs, ignore_index=True)

# Load and concatenate all PV files
pv_dfs = []
for f in pv_files:
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    pv_dfs.append(df)
df_pv = pd.concat(pv_dfs, ignore_index=True)

#print(df_prices)
#print(df_pv)


# Merge the two dataframes on the 'time' column
df_combined = pd.merge(df_prices, df_pv, on='time', how='left')
#df_combined = df_combined.fillna(0)
#df_combined = df_combined.set_index('time').interpolate(method='time').reset_index()


df_combined['Solar_value'] = df_combined['Solar_production_MW'] * df_combined['DA_price']


# Create installed capacity column in MW using a linear fit (extrapolation allowed)
from datetime import datetime

# Known data points for installed capacity (DC) at year-end
capacity_points = [
    (pd.Timestamp('2019-12-31', tz='Europe/Amsterdam'), 7226), # MWp DC
    (pd.Timestamp('2020-12-31', tz='Europe/Amsterdam'), 11108),
    (pd.Timestamp('2021-12-31', tz='Europe/Amsterdam'), 14822),
    (pd.Timestamp('2022-12-31', tz='Europe/Amsterdam'), 19536),
    (pd.Timestamp('2023-12-31', tz='Europe/Amsterdam'), 24302),  # MWp DC
    (pd.Timestamp('2024-12-31', tz='Europe/Amsterdam'), 28620),  # MWp DC
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 28620 + 2100),  # MWp DC # lower installed PV estimate update by https://x.com/BM_Visser/status/1954798688049697116
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 28620 + 2100 + 1440),  # MWp DC
]


# Prepare arrays for fitting
dates = np.array([(dt - capacity_points[0][0]).days for dt, _ in capacity_points])
capacities = np.array([cap for _, cap in capacity_points])

# Fit a linear model (polyfit degree 1)
fit_coeffs = np.polyfit(dates, capacities, 1)

def fit_installed_capacity(date):
    # Ensure date is a pandas Timestamp with tz
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    if date.tz is None:
        date = date.tz_localize('Europe/Amsterdam')
    # Convert date to days since first anchor
    days_since = (date - capacity_points[0][0]).days
    # Linear fit: capacity = m * days + b
    capacity = fit_coeffs[0] * days_since + fit_coeffs[1]
    return round(capacity, 0)

# Add the new column to df_combined
# Ensure 'time' is timezone-aware and in Europe/Amsterdam
if df_combined['time'].dt.tz is None:
    df_combined['time'] = df_combined['time'].dt.tz_localize('Europe/Amsterdam')
df_combined['installed_capacity_MW'] = df_combined['time'].apply(fit_installed_capacity)


print(df_combined)


# summarize per month
# Remove timezone before converting to Period to avoid warning
df_combined['month'] = df_combined['time'].dt.tz_localize(None).dt.to_period('M')

monthly_summary = (
    df_combined.groupby('month').agg({
        'Solar_production_MW': 'sum',
        'Solar_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean'
    }).reset_index()
)

# Calculate derived columns
monthly_summary['Total_PV_Energy_GWh'] = round(monthly_summary['Solar_production_MW']/1000, 1)
monthly_summary['Value_per_MWp_DC_EUR'] = round(monthly_summary['Solar_value'] / monthly_summary['installed_capacity_MW'], 1)
monthly_summary['Avg_DA_Price'] = round(monthly_summary['DA_price'], 1)

# Calculate PV weighted price for each month
monthly_summary['PV_Weighted_Price'] = monthly_summary.apply(
    lambda row: (df_combined[df_combined['month'] == row['month']]['Solar_production_MW'] * 
                 df_combined[df_combined['month'] == row['month']]['DA_price']).sum() / 
                df_combined[df_combined['month'] == row['month']]['Solar_production_MW'].sum() 
                if df_combined[df_combined['month'] == row['month']]['Solar_production_MW'].sum() > 0 else float('nan'), axis=1
)

monthly_summary['profile_factor'] = round((monthly_summary['PV_Weighted_Price'] / monthly_summary['Avg_DA_Price'])*100, 1)
monthly_summary['Installed_Capacity_GWp_DC'] = round(monthly_summary['installed_capacity_MW'] / 1000, 2)

# Select and reorder columns
monthly_summary = monthly_summary[['month', 'Total_PV_Energy_GWh', 'Value_per_MWp_DC_EUR', 'Avg_DA_Price', 'PV_Weighted_Price', 'profile_factor', 'Installed_Capacity_GWp_DC']]

print("\nMonthly Summary:")
# Round first 3 columns to 1 digits
monthly_summary_rounded = monthly_summary.copy()
monthly_summary_rounded.loc[:, monthly_summary_rounded.columns[1:4]] = monthly_summary_rounded.iloc[:, 1:4].round(1)
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
    df_combined.groupby('month_date').agg({
        'Solar_production_MW': 'sum',
        'Solar_value': 'sum',
        'installed_capacity_MW': 'mean',
        'DA_price': 'mean'
    }).reset_index()
)

# Calculate derived columns
monthly['Monthly_PV_Energy_MWh'] = round(monthly['Solar_production_MW'], 1)
monthly['Monthly_Value_per_MWp_DC_EUR'] = round(monthly['Solar_value'] / monthly['installed_capacity_MW'], 1)
monthly['Monthly_Installed_Capacity_MW'] = monthly['installed_capacity_MW']
monthly['Monthly_Avg_DA_Price'] = monthly['DA_price']

# Calculate PV weighted price for each month
monthly['Monthly_PV_Power_Weighted_DA_Price'] = monthly.apply(
    lambda row: (df_combined[df_combined['month_date'] == row['month_date']]['Solar_production_MW'] * 
                 df_combined[df_combined['month_date'] == row['month_date']]['DA_price']).sum() / 
                df_combined[df_combined['month_date'] == row['month_date']]['Solar_production_MW'].sum() 
                if df_combined[df_combined['month_date'] == row['month_date']]['Solar_production_MW'].sum() > 0 else float('nan'), axis=1
)

# Select and reorder columns
monthly = monthly[['month_date', 'Monthly_PV_Energy_MWh', 'Monthly_Value_per_MWp_DC_EUR', 'Monthly_PV_Power_Weighted_DA_Price', 'Monthly_Installed_Capacity_MW', 'Monthly_Avg_DA_Price']]

# Calculate profile factor
monthly['Monthly_Profile_Factor'] = (monthly['Monthly_PV_Power_Weighted_DA_Price'] / monthly['Monthly_Avg_DA_Price']) * 100

# Normalize by installed capacity
monthly['Monthly_PV_Yield_per_MW'] = monthly['Monthly_PV_Energy_MWh'] / monthly['Monthly_Installed_Capacity_MW']

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
    'Solar_production_MW': 'sum',
    'DA_price': 'mean'
}).reset_index()

# Calculate weighted price manually
yearly_weighted_prices['Yearly_PV_Weighted_Price'] = yearly_weighted_prices.apply(
    lambda row: (df_combined[df_combined['time'].dt.year == row['time']]['Solar_production_MW'] * 
                 df_combined[df_combined['time'].dt.year == row['time']]['DA_price']).sum() / 
                df_combined[df_combined['time'].dt.year == row['time']]['Solar_production_MW'].sum() 
                if df_combined[df_combined['time'].dt.year == row['time']]['Solar_production_MW'].sum() > 0 else float('nan'), axis=1
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

# Prepare yearly summary data for table
yearly_summary_for_table = yearly_totals.copy()
yearly_summary_for_table['Yearly_PV_Energy_GWh'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / 1000
yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC'] = yearly_summary_for_table['Yearly_Installed_Capacity_MW'] / 1000
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
    rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.08,
    subplot_titles=(
        'Yearly Summary',
        'Monthly PV Yield',
        'Market Value per MWp installed',
        'PV Weighted DA Price & Profile Factor'
    ),
    specs=[
        [{"type": "table"}],
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": True}]
    ],
    row_heights=[0.3, 0.25, 0.25, 0.25]  # Adjusted heights for 4 subplots
)



# First subplot: Yearly summary table (rows reversed)
fig.add_trace(
    go.Table(
        header=dict(
            values=['Year', 'PV Energy produced (GWh/y)', 'Installed PV Capacity in NL (GWp) mid-year', 'MWh yield / MWp installed', 'Annual Market value (EUR/MWp/y)', 'Day-Ahead linear avg price (EUR/MWh)', 'PV-profile weighted price (EUR/MWh)', 'Profile Factor of PV (%)'],
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
    row=1, col=1
)



# Second subplot: Monthly PV energy production (lines per year)
# Create separate traces for each year
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    # Extract month number (1-12) for x-axis
    year_data['month_num'] = year_data['month_date'].dt.month
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_PV_Energy_MWh']/1000, 
            name=f'PV Energy {year} (GWh)', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ),
        row=2, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='Energy (GWh)', row=2, col=1, secondary_y=False)
fig.update_xaxes(title_text='Month', row=2, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# Third subplot: Monthly PV Market Value (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_Value_per_MWp_DC_EUR'], 
            name=f'Market Value {year} (EUR/MWp)', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ),
        row=3, col=1, secondary_y=False
    )
fig.update_yaxes(title_text='EUR per MWp', row=3, col=1)
fig.update_xaxes(title_text='Month', row=3, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])


# Fourth subplot: Monthly Profile Factor (lines per year)
for year in sorted(monthly['year'].unique()):
    year_data = monthly[monthly['year'] == year].copy()
    year_data['month_num'] = year_data['month_date'].dt.month
    
    # Profile Factor line
    fig.add_trace(
        go.Scatter(
            x=year_data['month_num'], 
            y=year_data['Monthly_Profile_Factor'], 
            name=f'Profile Factor {year} (%)', 
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ),
        row=4, col=1, secondary_y=False
    )

fig.update_yaxes(title_text='Profile Factor (%)', row=4, col=1, secondary_y=False)
fig.update_xaxes(title_text='Month', row=4, col=1, tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Move legend below the plot
fig.update_layout(
    title_text='Analysis on PV value (NL), EPEX spot prices + PV production of NED.nl',
    legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
    margin=dict(b=120)
)

# Create a separate table figure
# Format numbers with thousands separators and percentage for profile factor
# Sort monthly_summary_rounded in reverse chronological order
monthly_summary_rounded_reversed = monthly_summary_rounded.sort_values('month', ascending=False).reset_index(drop=True)

table_fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Month', 'PV Energy produced (GWh)', 'Installed Capacity (GWp) month-avg', 'Market value (EUR/MWp/year)', 'Day-Ahead linear average price (EUR/MWh)', 'PV-profile Weighted price (EUR/MWh)', 'Profile Factor of PV (%)'],
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
fig.write_html('solar_production_plot_v3.html', auto_open=True)
table_fig.write_html('monthly_summary_table.html', auto_open=True)



