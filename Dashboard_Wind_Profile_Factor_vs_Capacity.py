import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import glob
import warnings
os.system('clear')

# TODO: add a function to only export Wind power when DA prices are non-negative
# Graph shows yearly relationship between installed Wind capacity and profile factor

# --- Load all monthly DA_prices and Wind data files ---
# Find all DA_prices and Wind files
price_pattern = 'data/**/DA_prices_*.csv'
wind_pattern = 'data/**/data_NED_Wind_*.csv'
price_files = sorted(glob.glob(price_pattern, recursive=True))
wind_files = sorted(glob.glob(wind_pattern, recursive=True))

# Exclude combined file from price_files
price_files = [f for f in price_files if 'combined' not in f]

assert price_files, f"No DA_prices files matched {price_pattern}"
assert wind_files, f"No Wind files matched {wind_pattern}"

# Load and concatenate all price files
price_dfs = []
for f in price_files:
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    price_dfs.append(df)
df_prices = pd.concat(price_dfs, ignore_index=True)

# Filter out future dates (data should not extend beyond current date)
from datetime import datetime
current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
current_date_tz = pd.Timestamp(current_date, tz='Europe/Amsterdam')
df_prices = df_prices[df_prices['time'] <= current_date_tz]

# Load and concatenate all Wind files
wind_dfs = []
for f in wind_files:
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
    wind_dfs.append(df)
df_wind = pd.concat(wind_dfs, ignore_index=True)

# Merge the two dataframes on the 'time' column
df_combined = pd.merge(df_prices, df_wind, on='time', how='left')

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
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 6580 + (80*12)/0.60),  # MW AC # lower installed Wind estimate update
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 6580 + 1600 + 1440),  # MW AC
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

print("Combined data shape:", df_combined.shape)
print("Date range:", df_combined['time'].min(), "to", df_combined['time'].max())

# Calculate yearly data
yearly_totals = df_combined.groupby(df_combined['time'].dt.year).agg({
    'Wind_production_MW': 'sum',
    'Wind_value': 'sum',
    'installed_capacity_MW': 'mean',
    'DA_price': 'mean'
}).reset_index()

yearly_totals = yearly_totals.rename(columns={
    'time': 'year',
    'Wind_production_MW': 'Yearly_Wind_Energy_MWh',
    'Wind_value': 'Yearly_Total_Wind_Value',
    'installed_capacity_MW': 'Yearly_Installed_Capacity_MW',
    'DA_price': 'Yearly_Avg_DA_Price'
})

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
yearly_totals['Yearly_Profile_Factor'] = (yearly_totals['Yearly_Wind_Weighted_Price'] / yearly_totals['Yearly_Avg_DA_Price']) * 100

# Calculate July 1st installed capacity for each year
def get_july_1st_capacity(year):
    july_1st = pd.Timestamp(f'{year}-07-01', tz='Europe/Amsterdam')
    return fit_installed_capacity_piecewise(july_1st)

yearly_totals['July_1st_Installed_Capacity_GW_AC'] = yearly_totals['year'].apply(get_july_1st_capacity) / 1000

# Select and clean data for plotting
plot_data = yearly_totals[['year', 'July_1st_Installed_Capacity_GW_AC', 'Yearly_Profile_Factor']].copy()
plot_data = plot_data.dropna()

print("\nYearly data for plotting:")
print(plot_data)

# Create the scatter plot
import plotly.graph_objs as go  # type: ignore
import plotly.colors  # type: ignore

# Create custom color scheme with distinct, distinguishable colors for each year
years_list = sorted(plot_data['year'].unique())
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
    '#7f7f7f',  # Grey
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

# Create the scatter plot
fig = go.Figure()

# Update layout
fig.update_layout(
    title=dict(
        text='Wind Profile Factor vs Installed Capacity Wind in Netherlands<br><sub>Each point represents year avg installed capacity vs yearly Wind profile factor</sub>',
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        title=dict(text='Installed Wind Capacity NL (GW AC) yearly avg', font=dict(size=14)),
        tickfont=dict(size=12),
        gridcolor='lightgray',
        zeroline=False,
        range=[0, 50]
    ),
    yaxis=dict(
        title=dict(text='Profile Factor (%)', font=dict(size=14)),
        tickfont=dict(size=12),
        gridcolor='lightgray',
        zeroline=False,
        range=[0, 105]
    ),
    legend=dict(
        title='Year',
        font=dict(size=12)
    ),
    plot_bgcolor='rgba(248, 248, 248, 1)',
    paper_bgcolor='rgba(248, 248, 248, 1)',
    width=1000,
    height=600,
    margin=dict(l=80, r=80, t=100, b=80)
)

# Add trend line
if len(plot_data) > 2:
    # Filter data for trend line: 2019-2025, excluding 2022
    trend_data = plot_data[(plot_data['year'] >= 2019) & (plot_data['year'] <= 2025) & (plot_data['year'] != 2022)].copy()
    
    if len(trend_data) > 2:
        # Calculate exponential regression using filtered data
        x_vals = trend_data['July_1st_Installed_Capacity_GW_AC'].values
        y_vals = trend_data['Yearly_Profile_Factor'].values
    else:
        # Fallback to all data if filtered data has insufficient points
        x_vals = plot_data['July_1st_Installed_Capacity_GW_AC'].values
        y_vals = plot_data['Yearly_Profile_Factor'].values
    
    # Fit exponential regression: y = a * exp(b * x)
    # Transform to linear: ln(y) = ln(a) + b*x
    # Only fit where y > 0 to avoid log of negative numbers
    valid_indices = y_vals > 0
    x_valid = x_vals[valid_indices]
    y_valid = y_vals[valid_indices]
    
    # Add linear trend line starting from 2019 data point
    coeffs_linear = np.polyfit(x_vals, y_vals, 1)
    
    # Calculate installed capacity on January 1, 2026
    jan_1_2026 = pd.Timestamp('2026-01-01', tz='Europe/Amsterdam')
    jan_1_capacity = fit_installed_capacity_piecewise(jan_1_2026) / 1000  # Convert to GW AC
    
    # Create trend line starting from 2019 data point to January 1, 2026
    # Get the 2019 data point (first point in trend_data)
    x_2019 = trend_data['July_1st_Installed_Capacity_GW_AC'].iloc[0]
    y_2019 = trend_data['Yearly_Profile_Factor'].iloc[0]
    x_extended_linear = np.linspace(x_2019, jan_1_capacity, 50)
    trend_line_extended_linear = np.polyval(coeffs_linear, x_extended_linear)
    
    # Add solid linear trend line starting from 2019 data point
    fig.add_trace(
        go.Scatter(
            x=x_extended_linear,
            y=trend_line_extended_linear,
            mode='lines',
            name='Linear Trend',
            line=dict(color='red', width=2),
            hovertemplate='<b>Linear Trend (from 2019 data to Jan 1, 2026)</b><br>' +
                         f'<b>Equation:</b> y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}<br>' +
                         f'<b>Slope:</b> {coeffs_linear[0]:.2f} %/GW<br>' +
                         f'<b>Starts at:</b> {x_2019:.1f} GW ({y_2019:.1f}%)<br>' +
                         f'<b>Ends at:</b> {jan_1_capacity:.1f} GW (Jan 1, 2026)<br>' +
                         f'<extra></extra>',
            showlegend=True
        )
    )
    
    if len(y_valid) > 1:
        # Linear fit on log-transformed data for exponential extrapolation
        log_y = np.log(y_valid)
        coeffs_log = np.polyfit(x_valid, log_y, 1)
        
        # Convert back to exponential parameters
        a = np.exp(coeffs_log[1])  # a = exp(intercept)
        b = coeffs_log[0]          # b = slope
        
        # Calculate when profile factor reaches low thresholds (asymptotically approaches 0)
        # For exponential decay, it never actually reaches 0, but we can find when it reaches small thresholds
        thresholds = [5.0, 2.0, 1.0]  # Different thresholds to show asymptotic behavior
        if b < 0:  # Only for decay (negative slope)

            # Add linear trend line from January 1, 2026 to 0% (plot this first)
            # Calculate when linear trend reaches 0%
            x_at_zero_linear = -coeffs_linear[1] / coeffs_linear[0]
            
            # Create linear extrapolation from Jan 1, 2026 to 0%
            x_linear_to_zero = np.linspace(jan_1_capacity, x_at_zero_linear, 50)
            y_linear_to_zero = np.polyval(coeffs_linear, x_linear_to_zero)
            
            fig.add_trace(
                go.Scatter(
                    x=x_linear_to_zero,
                    y=y_linear_to_zero,
                    mode='lines',
                    name='Linear Extrapolation',
                    line=dict(color='red', width=2, dash='dot'),
                    hovertemplate='<b>Linear Extrapolation to 0%</b><br>' +
                                 '<b>Capacity:</b> %{x:.1f} GW<br>' +
                                 '<b>Profile Factor:</b> %{y:.1f}%<br>' +
                                 '<b>Equation:</b> y = ' + f'{coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}<br>' +
                                 '<b>Reaches 0% at:</b> {x_at_zero_linear:.1f} GW<br>' +
                                 '<extra></extra>',
                    showlegend=True
                )
            )

            
            # Start exponential extrapolation from January 1, 2026 (where linear trend ends)
            # Calculate the profile factor at Jan 1, 2026 using the linear trend
            y_at_jan_1_2026 = np.polyval(coeffs_linear, jan_1_capacity)
            
            # Adjust exponential equation to start from the linear trend endpoint
            # y = a * exp(b * (x - x0)) where x0 is jan_1_capacity
            # At x = jan_1_capacity, y should equal y_at_jan_1_2026
            # So: y_at_jan_1_2026 = a * exp(b * (jan_1_capacity - jan_1_capacity)) = a
            # Therefore: a_adjusted = y_at_jan_1_2026
            a_adjusted = y_at_jan_1_2026
            
            x_extrapolation = np.linspace(jan_1_capacity, 50, 100)  # Extend from Jan 1, 2026 to 50 GW to show asymptotic behavior
            y_extrapolation = a_adjusted * np.exp(b * (x_extrapolation - jan_1_capacity))
            
            fig.add_trace(
                go.Scatter(
                    x=x_extrapolation,
                    y=y_extrapolation,
                    mode='lines',
                    name='Exponential Extrapolation',
                    line=dict(color='orange', width=2, dash='dash'),
                    hovertemplate='<b>Exponential Extrapolation (from Jan 1, 2026)</b><br>' +
                                 '<b>Capacity:</b> %{x:.1f} GW<br>' +
                                 '<b>Profile Factor:</b> %{y:.1f}%<br>' +
                                 '<b>Equation:</b> y = ' + f'{a_adjusted:.2f} * e^({b:.3f}*(x-{jan_1_capacity:.1f}))<br>' +
                                 '<b>Starts from:</b> {jan_1_capacity:.1f} GW ({y_at_jan_1_2026:.1f}%)<br>' +
                                 '<extra></extra>',
                    showlegend=True
                )
            )
            
    else:
        # Fallback to linear if exponential fitting fails
        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_extended = np.linspace(0, 50, 100)
        trend_line_extended = np.polyval(coeffs, x_extended)
        
        fig.add_trace(
            go.Scatter(
                x=x_extended,
                y=trend_line_extended,
                mode='lines',
                name='Linear Trend (fallback)',
                line=dict(color='red', width=2),
                hovertemplate='<b>Linear Trend (fallback)</b><br>' +
                             f'<b>Slope:</b> {coeffs[0]:.2f} %/GW<br>' +
                             f'<b>Intercept:</b> {coeffs[1]:.2f}%<br>' +
                             f'<extra></extra>',
                showlegend=True
            )
        )

# Add scatter plot points (after trend lines to appear on top)
for i, row in plot_data.iterrows():
    year = row['year']
    capacity = row['July_1st_Installed_Capacity_GW_AC']
    profile_factor = row['Yearly_Profile_Factor']
    
    fig.add_trace(
        go.Scatter(
            x=[capacity],
            y=[profile_factor],
            mode='markers+text',
            marker=dict(
                size=20,
                color=year_to_color[year],
                line=dict(width=2, color='white')
            ),
            text=[str(int(year))[-2:]],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            name=f'{int(year)}',
            hovertemplate=f'<b>Year:</b> {int(year)}<br>' +
                         f'<b>July 1st Installed Capacity:</b> {capacity:.2f} GW AC<br>' +
                         f'<b>Profile Factor:</b> {profile_factor:.1f}%<br>' +
                         f'<extra></extra>',
            showlegend=True
        )
    )


# Add correlation coefficient and slope
if len(plot_data) > 2:
    # Use the same filtered data for correlation as used for trend line
    if 'trend_data' in locals() and len(trend_data) > 2:
        correlation = np.corrcoef(trend_data['July_1st_Installed_Capacity_GW_AC'].values, trend_data['Yearly_Profile_Factor'].values)[0, 1]
        slope = coeffs_linear[0]  # slope from linear trend
    else:
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        slope = coeffs_linear[0]  # slope from linear trend
    
    # Position slope text box at x=0.02, y=50
    fig.add_annotation(
        text=f'Slope: {round(slope, 1)}%/GW',
        x=5,
        y=50,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=5,  # Arrow points to the right towards the trend line
        ay=0,  # No vertical arrow offset
        font=dict(size=12, color='red'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='red',
        borderwidth=1
    )
    
    # Calculate where linear trend reaches 0%
    x_at_zero_linear = -coeffs_linear[1] / coeffs_linear[0]
    
    # Add text box at where linear trend line touches 0
    fig.add_annotation(
        text=f'Linear trend reaches 0% at {x_at_zero_linear:.0f} GW',
        x=x_at_zero_linear,
        y=0,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=0,  # No horizontal arrow offset
        ay=20,  # Arrow points up to the trend line
        font=dict(size=12, color='red'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='red',
        borderwidth=1
    )

# Write to HTML and PDF files
fig.write_html('wind_profile_factor_vs_capacity_dashboard.html', auto_open=True)
fig.write_image('wind_profile_factor_vs_capacity_dashboard.pdf', format='pdf')

print(f"\nDashboard created:")
print(f"  - HTML: wind_profile_factor_vs_capacity_dashboard.html")
print(f"  - PDF: wind_profile_factor_vs_capacity_dashboard.pdf")
print(f"Data points: {len(plot_data)} years")
print(f"Years covered: {plot_data['year'].min()} - {plot_data['year'].max()}")
print(f"Capacity range: {plot_data['July_1st_Installed_Capacity_GW_AC'].min():.2f} - {plot_data['July_1st_Installed_Capacity_GW_AC'].max():.2f} GW AC")
print(f"Profile factor range: {plot_data['Yearly_Profile_Factor'].min():.1f} - {plot_data['Yearly_Profile_Factor'].max():.1f}%")

if len(plot_data) > 2:
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Slope: {round(slope, 1)}% per GW")
    
    # Print linear trend (from 2019 data point to January 1, 2026)
    trend_years = "2019-2025 (excluding 2022)" if 'trend_data' in locals() and len(trend_data) > 2 else "all available data"
    print(f"Linear trend equation (based on {trend_years}): y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}")
    print(f"Linear trend starts at: {x_2019:.1f} GW ({y_2019:.1f}%)")
    print(f"Linear trend ends at: {jan_1_capacity:.1f} GW (January 1, 2026)")
    
    # Calculate when linear trend reaches 0%
    x_at_zero_linear = -coeffs_linear[1] / coeffs_linear[0]
    print(f"Linear trend reaches 0% at: {round(x_at_zero_linear, 0):.0f} GW")
    
    # Print exponential equation if available
    if 'a' in locals() and 'b' in locals():
        print(f"Exponential extrapolation equation (based on {trend_years}): y = {a:.2f} * e^({b:.3f}*x)")
        print(f"Exponential decay rate: {b:.3f} per GW")
        print("Asymptotic behavior: Wind profile factor approaches 0% but never reaches it")
        print("Wind will always retain some market value on the day-ahead market")
        
        # Calculate and print threshold milestones
        thresholds = [5.0, 2.0, 1.0]
        print("\nExponential extrapolation threshold milestones:")
        for threshold in thresholds:
            if threshold < a:
                x_at_threshold = (np.log(threshold / a)) / b
                print(f"  Profile factor reaches {threshold}% at: {x_at_threshold:.1f} GW")
        
        # Calculate profile factor at 25 GW
        profile_factor_at_25 = a * np.exp(b * 25)
        print(f"  Profile factor at 25 GW: {profile_factor_at_25:.1f}%")
    else:
        print("Exponential extrapolation not available (fallback to linear only)")
