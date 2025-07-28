import pandas as pd
import numpy as np
import os
import glob
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


# Create installed capacity column in MW
PV_2023_12_31_AC = 21395 # MWp AC
PV_2023_12_31_DC = 24302 # MWp DC
AC_DC_ratio = PV_2023_12_31_AC / PV_2023_12_31_DC
print('AC_DC_ratio 2023_12_31:')
print(round(AC_DC_ratio * 100, 2), '%')

PV_2024_12_31_AC = 25635 # MWp AC
PV_2024_12_31_DC = 28620 # MWp DC
PV_2024_added = PV_2024_12_31_DC - PV_2023_12_31_DC
AC_DC_ratio = PV_2024_12_31_AC / PV_2024_12_31_DC
print(round(AC_DC_ratio * 100, 2), '%')

PV_2025_added = 3500 # MWp DC
PV_2025_12_31_AC = PV_2024_12_31_AC + PV_2025_added*AC_DC_ratio  # MWp AC
PV_2025_12_31_DC = PV_2024_12_31_DC + PV_2025_added              # MWp DC

# --- Interpolate installed capacity for each row ---
from datetime import datetime

def interpolate_installed_capacity(date):
    # Anchor points
    d1 = pd.Timestamp('2023-12-31', tz='Europe/Amsterdam')
    d2 = pd.Timestamp('2024-12-31', tz='Europe/Amsterdam')
    d3 = pd.Timestamp('2025-12-31', tz='Europe/Amsterdam')
    c1 = PV_2023_12_31_DC
    c2 = PV_2024_12_31_DC
    c3 = PV_2025_12_31_DC
    
    if date <= d1:
        return round(c1, 0)
    elif date <= d2:
        # Interpolate between 2023-12-31 and 2024-12-31
        total_days = (d2 - d1).days
        days_since = (date - d1).days
        return round(c1 + (c2 - c1) * days_since / total_days, 0)
    elif date <= d3:
        # Interpolate between 2024-12-31 and 2025-12-31
        total_days = (d3 - d2).days
        days_since = (date - d2).days
        return round(c2 + (c3 - c2) * days_since / total_days, 0)
    else:
        return round(c3, 0)

# Add the new column to df_combined
# Ensure 'time' is timezone-aware and in Europe/Amsterdam
if df_combined['time'].dt.tz is None:
    df_combined['time'] = df_combined['time'].dt.tz_localize('Europe/Amsterdam')
df_combined['installed_capacity_MW'] = df_combined['time'].apply(interpolate_installed_capacity)




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
                'profile_factor': round((weighted_price / avg_price)*100, 1) if avg_price != 0 else float('nan')
            })
        )(
            x['DA_price'].mean(),
            (x['Solar_production_MW'] * x['DA_price']).sum() / x['Solar_production_MW'].sum() if x['Solar_production_MW'].sum() > 0 else float('nan')
        )
    )
    .reset_index()
)

print("\nMonthly Summary:")
print(monthly_summary)


# Plot the combined dataframe using plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --- Prepare data for plotting ---
# 1. PV power and Day-ahead price (hourly)

# 2. Monthly PV yield (sum), PV value (sum), and PV_power_weighted_DA_price (monthly avg)
df_combined['month_date'] = df_combined['time'].dt.to_period('M').dt.to_timestamp()

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


# --- Create subplots ---
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.10,
    subplot_titles=(
        'Monthly PV Yield',
        'Market Value per MWp installed',
        'PV  Weighted DA Price (EUR/MWh), Avg DA Price, and Profile Factor (%)'
    ),
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]]
)



# Second subplot: Monthly PV yield/value per MW (bars)
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_PV_Yield_per_MW'], name='Monthly PV Yield MWh/MWp', marker_color='orange'),
    row=1, col=1, secondary_y=False
)
fig.update_yaxes(title_text='Yield MWh/MWp', row=1, col=1, secondary_y=False)


fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Value_per_MWp_DC_EUR'], name='PV Market Value (EUR/MWp/Month)', marker_color='goldenrod'),
    row=2, col=1, secondary_y=False
)
fig.update_yaxes(title_text='Value per MWp', row=2, col=1)
fig.update_xaxes(title_text='Month', row=2, col=1, tickangle=45, tickformat='%b %Y')


# Third subplot: Monthly PV Power Weighted DA Price (bar)
# Add average DA price as a line
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Avg_DA_Price'], name='Avg DA Price (EUR/MWh)', marker_color='royalblue'),
    row=3, col=1, secondary_y=False
)
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_PV_Power_Weighted_DA_Price'], name='PV Weighted Market Price (EUR/MWh)', marker_color='purple'),
    row=3, col=1, secondary_y=False
)

# Add profile factor as a secondary y-axis line
fig.add_trace(
    go.Bar(x=monthly['month_date'], y=monthly['Monthly_Profile_Factor'], name='Profile Factor (%)'),
    row=3, col=1, secondary_y=False
)
fig.update_yaxes(title_text='EUR/MWh', row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text='Profile Factor (%)', row=3, col=1, secondary_y=True)
fig.update_xaxes(title_text='Month', row=3, col=1, tickangle=45, tickformat='%b %Y')

# Move legend below the plot
fig.update_layout(
    title_text='Monthly PV Yield',
    legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
    margin=dict(b=120)
)

fig.write_html('solar_production_plot_v2.html', auto_open=True)

