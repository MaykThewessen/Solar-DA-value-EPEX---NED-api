import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import glob
import warnings

# Suppress deprecation warnings for groupby.apply operations


os.system('clear')


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
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 28620 + 3500),  # MWp DC
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 28620 + 3500 + 3200),  # MWp DC
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
    df_combined.groupby('month').apply(
        lambda x: (
            lambda avg_price, weighted_price: pd.Series({
                'Total_PV_Energy_GWh': round(x['Solar_production_MW'].sum()/1000, 1),
                'Value_per_MWp_DC_EUR': round(x['Solar_value'].sum() / x['installed_capacity_MW'].mean(), 1),
                'Avg_DA_Price': round(avg_price, 1),
                'PV_Weighted_Price': round(weighted_price, 1),
                'profile_factor': round((weighted_price / avg_price)*100, 1) if avg_price != 0 else float('nan'),
                'Installed_Capacity_GWp_DC': round(x['installed_capacity_MW'].mean() / 1000, 1)
            })
        )(
            x['DA_price'].mean(),
            (x['Solar_production_MW'] * x['DA_price']).sum() / x['Solar_production_MW'].sum() if x['Solar_production_MW'].sum() > 0 else float('nan')
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
            'Monthly_PV_Energy_MWh': round(x['Solar_production_MW'].sum(), 1),
            'Monthly_Value_per_MWp_DC_EUR': round(x['Solar_value'].sum() / x['installed_capacity_MW'].mean(), 1),
            'Monthly_PV_Power_Weighted_DA_Price': (x['Solar_production_MW'] * x['DA_price']).sum() / x['Solar_production_MW'].sum() if x['Solar_production_MW'].sum() > 0 else float('nan'),
            'Monthly_Installed_Capacity_MW': x['installed_capacity_MW'].mean(),  # or .last() for end-of-month
            'Monthly_Avg_DA_Price': x['DA_price'].mean(),
        })
    )
    .reset_index()
)

# Calculate profile factor
monthly['Monthly_Profile_Factor'] = (monthly['Monthly_PV_Power_Weighted_DA_Price'] / monthly['Monthly_Avg_DA_Price']) * 100

# Normalize by installed capacity
monthly['Monthly_PV_Yield_per_MW'] = monthly['Monthly_PV_Energy_MWh'] / monthly['Monthly_Installed_Capacity_MW']

# Calculate yearly totals
monthly['year'] = monthly['month_date'].dt.year
yearly_totals = monthly.groupby('year').agg({
    'Monthly_PV_Energy_MWh': 'sum',
    'Monthly_Value_per_MWp_DC_EUR': 'mean',  # We'll calculate the weighted average separately
    'Monthly_Installed_Capacity_MW': 'mean'  # Average installed capacity for the year
}).reset_index()
yearly_totals.columns = ['year', 'Yearly_PV_Energy_MWh', 'Yearly_Value_per_MWp_DC_EUR', 'Yearly_Installed_Capacity_MW']

# Calculate weighted average value per MWp for each year
def calculate_weighted_value(group):
    if group['Monthly_Installed_Capacity_MW'].sum() > 0:
        return (group['Monthly_Value_per_MWp_DC_EUR'] * group['Monthly_Installed_Capacity_MW']).sum() / group['Monthly_Installed_Capacity_MW'].sum()
    else:
        return 0

yearly_weighted_values = monthly.groupby('year').apply(calculate_weighted_value).reset_index()
yearly_weighted_values.columns = ['year', 'Yearly_Value_per_MWp_DC_EUR']
yearly_totals = yearly_totals.drop('Yearly_Value_per_MWp_DC_EUR', axis=1).merge(yearly_weighted_values, on='year')

# Calculate yearly weighted average price
yearly_weighted_prices = df_combined.groupby(df_combined['time'].dt.year).apply(
    lambda x: (x['Solar_production_MW'] * x['DA_price']).sum() / x['Solar_production_MW'].sum() if x['Solar_production_MW'].sum() > 0 else float('nan')
).reset_index()
yearly_weighted_prices.columns = ['year', 'Yearly_PV_Weighted_Price']

# Merge yearly data
yearly_totals = yearly_totals.merge(yearly_weighted_prices, on='year')

# Calculate yearly profile factor
yearly_avg_prices = df_combined.groupby(df_combined['time'].dt.year)['DA_price'].mean().reset_index()
yearly_avg_prices.columns = ['year', 'Yearly_Avg_DA_Price']
yearly_totals = yearly_totals.merge(yearly_avg_prices, on='year')
yearly_totals['Yearly_Profile_Factor'] = (yearly_totals['Yearly_PV_Weighted_Price'] / yearly_totals['Yearly_Avg_DA_Price']) * 100

# Add yearly data to monthly dataframe
monthly = monthly.merge(yearly_totals[['year', 'Yearly_PV_Energy_MWh', 'Yearly_Value_per_MWp_DC_EUR', 'Yearly_PV_Weighted_Price', 'Yearly_Profile_Factor']], on='year')

# Prepare yearly summary data for table
yearly_summary_for_table = yearly_totals.copy()
yearly_summary_for_table['Yearly_PV_Energy_GWh'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / 1000
yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC'] = yearly_summary_for_table['Yearly_Installed_Capacity_MW'] / 1000
# Calculate kWh/kWp produced
yearly_summary_for_table['Yearly_kWh_per_kWp'] = yearly_summary_for_table['Yearly_PV_Energy_MWh'] / (yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC']*1000 )
yearly_summary_for_table = yearly_summary_for_table.round(0)

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
    row_heights=[0.4, 0.2, 0.2, 0.2]  # Give more height to the first subplot (table)
)



# First subplot: Yearly summary table
fig.add_trace(
    go.Table(
        header=dict(
            values=['Year', 'Total PV Energy (GWh)', 'Installed Capacity (GWp DC) mid-year', 'kWh/kWp Produced', 'Value per MWp DC (EUR)', 'Avg DA Price (EUR/MWh)', 'PV Weighted Price (EUR/MWh)', 'Profile Factor (%)'],
            font=dict(size=10),
            align='left'
        ),
        cells=dict(
            values=[
                yearly_summary_for_table['year'].astype(str),
                [format_number(x) for x in yearly_summary_for_table['Yearly_PV_Energy_GWh']],
                [format_gwp(x) for x in yearly_summary_for_table['Yearly_Installed_Capacity_GWp_DC']],
                [format_number(x) for x in yearly_summary_for_table['Yearly_kWh_per_kWp']],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Value_per_MWp_DC_EUR']],
                [format_number(x) for x in yearly_summary_for_table['Yearly_Avg_DA_Price']],
                [format_number(x) for x in yearly_summary_for_table['Yearly_PV_Weighted_Price']],
                [format_percentage(x) for x in yearly_summary_for_table['Yearly_Profile_Factor']]
            ],
            font=dict(size=9),
            align='left',
            height=20
        )
    ),
    row=1, col=1
)

# Second subplot: Monthly PV energy production (bars)
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_PV_Energy_MWh'], name='Monthly PV Energy Production (MWh)', marker_color='orange'),
    row=2, col=1, secondary_y=False
)
fig.update_yaxes(title_text='Energy (MWh)', row=2, col=1, secondary_y=False)


fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Value_per_MWp_DC_EUR'], name='PV Market Value (EUR/MWp/Month)', marker_color='goldenrod'),
    row=3, col=1, secondary_y=False
)
fig.update_yaxes(title_text='€ per MWp', row=3, col=1)
#fig.update_xaxes(title_text='Month', row=3, col=1, tickangle=45, tickformat='%b %Y')


# Fourth subplot: Monthly PV Power Weighted DA Price (bar)
# Add average DA price as a line
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Avg_DA_Price'], name='Avg DA Price (EUR/MWh)', marker_color='royalblue'),
    row=4, col=1, secondary_y=False
)
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_PV_Power_Weighted_DA_Price'], name='PV Weighted Market Price (EUR/MWh)', marker_color='purple'),
    row=4, col=1, secondary_y=False
)

# Add profile factor as a secondary y-axis line
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Profile_Factor'], name='Profile Factor (%)'),
    row=4, col=1, secondary_y=False
)
fig.update_yaxes(title_text='Profile factor (%)', row=4, col=1, secondary_y=False)
#fig.update_yaxes(title_text='Profile Factor (%)', row=4, col=1, secondary_y=True)
fig.update_xaxes(title_text='per month', row=4, col=1, tickangle=45, tickformat='%b %Y')

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
        values=['Month', 'Total PV Energy (GWh)', 'Installed Capacity (GWp DC) month avg', 'Value per MWp DC (EUR)', 'Avg DA Price (EUR/MWh)', 'PV Weighted Price (EUR/MWh)', 'Profile Factor'],
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
    title_text='Monthly Summary Table',
    margin=dict(l=0, r=0, t=50, b=0)
)

# Write both figures to separate files
fig.write_html('solar_production_plot_v3.html', auto_open=True)
table_fig.write_html('monthly_summary_table.html', auto_open=True)



