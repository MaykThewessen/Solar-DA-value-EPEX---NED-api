import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import warnings
from data_loader import load_da_prices, load_ned_pv
os.system('clear')

# TODO: add a function to only export PV power when DA prices are non-negative
# Graph shows yearly relationship between installed PV capacity and profile factor

df_prices = load_da_prices(aggregate_to_hourly=True)
df_pv = load_ned_pv()

# Merge the two dataframes on the 'time' column
df_combined = pd.merge(df_prices, df_pv, on='time', how='left')

df_combined['Solar_value'] = df_combined['Solar_production_MW'] * df_combined['DA_price']

# Create installed capacity column in MW using a linear fit (extrapolation allowed)
from datetime import datetime

# Known data points for installed capacity (DC) at year-end
capacity_points = [
    (pd.Timestamp('2018-01-01', tz='Europe/Amsterdam'), 2911), # MWp DC https://www.cbs.nl/nl-nl/longread/rapportages/2024/hernieuwbare-energie-in-nederland-2023/5-zonne-energie
    (pd.Timestamp('2018-12-31', tz='Europe/Amsterdam'), 4609), # MWp DC https://www.cbs.nl/nl-nl/longread/rapportages/2024/hernieuwbare-energie-in-nederland-2023/5-zonne-energie
    (pd.Timestamp('2019-12-31', tz='Europe/Amsterdam'), 7226), # MWp DC https://opendata.cbs.nl/#/CBS/nl/dataset/85005NED/table
    (pd.Timestamp('2020-12-31', tz='Europe/Amsterdam'), 11108),
    (pd.Timestamp('2021-12-31', tz='Europe/Amsterdam'), 14822),
    (pd.Timestamp('2022-12-31', tz='Europe/Amsterdam'), 19536),
    (pd.Timestamp('2023-12-31', tz='Europe/Amsterdam'), 24302),  # MWp DC
    (pd.Timestamp('2024-12-31', tz='Europe/Amsterdam'), 28620),  # MWp DC
    (pd.Timestamp('2025-12-31', tz='Europe/Amsterdam'), 28620 + (80*12)/0.60),  # MWp DC # lower installed PV estimate update by https://x.com/BM_Visser/status/1954798688049697116
    (pd.Timestamp('2026-12-31', tz='Europe/Amsterdam'), 28620 + 1600 + 1440),  # MWp DC
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

print("Combined data shape:", df_combined.shape)
print("Date range:", df_combined['time'].min(), "to", df_combined['time'].max())

# Calculate yearly data
yearly_totals = df_combined.groupby(df_combined['time'].dt.year).agg({
    'Solar_production_MW': 'sum',
    'Solar_value': 'sum',
    'installed_capacity_MW': 'mean',
    'DA_price': 'mean'
}).reset_index()

yearly_totals = yearly_totals.rename(columns={
    'time': 'year',
    'Solar_production_MW': 'Yearly_PV_Energy_MWh',
    'Solar_value': 'Yearly_Total_Solar_Value',
    'installed_capacity_MW': 'Yearly_Installed_Capacity_MW',
    'DA_price': 'Yearly_Avg_DA_Price'
})

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
yearly_totals['Yearly_Profile_Factor'] = (yearly_totals['Yearly_PV_Weighted_Price'] / yearly_totals['Yearly_Avg_DA_Price']) * 100

# Calculate July 1st installed capacity for each year
def get_july_1st_capacity(year):
    july_1st = pd.Timestamp(f'{year}-07-01', tz='Europe/Amsterdam')
    return fit_installed_capacity(july_1st)

yearly_totals['Installed_Capacity_GWp_DC'] = yearly_totals['year'].apply(get_july_1st_capacity) / 1000

# Select and clean data for plotting
plot_data = yearly_totals[['year', 'Installed_Capacity_GWp_DC', 'Yearly_Profile_Factor']].copy()
plot_data = plot_data.dropna()

print("\nYearly data for plotting:")
print(plot_data.round(1))

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
        text='PV Profile Factor vs Installed Capacity Solar PV in Netherlands<br><sub>Each point represents year avg installed capacity vs yearly PV profile factor</sub>',
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        title=dict(text='Installed PV Capacity NL (GWp DC) yearly avg', font=dict(size=14)),
        tickfont=dict(size=12),
        gridcolor='lightgray',
        zeroline=False,
        range=[0, 100]
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
    # Filter data for trend line: 2018-2025, excluding 2022
    trend_data = plot_data[(plot_data['year'] >= 2018) & (plot_data['year'] <= 2025) & (plot_data['year'] != 2022)].copy()
    
    if len(trend_data) > 2:
        # Calculate exponential regression using filtered data
        x_vals = trend_data['Installed_Capacity_GWp_DC'].values
        y_vals = trend_data['Yearly_Profile_Factor'].values
    else:
        # Fallback to all data if filtered data has insufficient points
        x_vals = plot_data['Installed_Capacity_GWp_DC'].values
        y_vals = plot_data['Yearly_Profile_Factor'].values
    
    # Fit exponential regression: y = a * exp(b * x)
    # Transform to linear: ln(y) = ln(a) + b*x
    # Only fit where y > 0 to avoid log of negative numbers
    valid_indices = y_vals > 0
    x_valid = x_vals[valid_indices]
    y_valid = y_vals[valid_indices]
    
    # Add linear trend line starting from 2018 data point
    coeffs_linear = np.polyfit(x_vals, y_vals, 1)
    
    # Calculate installed capacity on January 1, 2026
    jan_1_2026 = pd.Timestamp('2026-01-01', tz='Europe/Amsterdam')
    jan_1_capacity = fit_installed_capacity(jan_1_2026) / 1000  # Convert to GWp
    
    # Create trend line starting from 2018 data point to January 1, 2026
    # Get the 2018 data point (first point in trend_data)
    x_2018 = trend_data['Installed_Capacity_GWp_DC'].iloc[0]
    y_2018 = trend_data['Yearly_Profile_Factor'].iloc[0]
    x_extended_linear = np.linspace(x_2018, jan_1_capacity, 50)
    trend_line_extended_linear = np.polyval(coeffs_linear, x_extended_linear)
    
    # Add solid linear trend line starting from 2018 data point
    fig.add_trace(
        go.Scatter(
            x=x_extended_linear,
            y=trend_line_extended_linear,
            mode='lines',
            name='Linear Trend',
            line=dict(color='red', width=2),
            hovertemplate='<b>Linear Trend (from 2018 data to Jan 1, 2026)</b><br>' +
                         f'<b>Equation:</b> y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}<br>' +
                         f'<b>Slope:</b> {coeffs_linear[0]:.2f} %/GWp<br>' +
                         f'<b>Starts at:</b> {x_2018:.1f} GWp ({y_2018:.1f}%)<br>' +
                         f'<b>Ends at:</b> {jan_1_capacity:.1f} GWp (Jan 1, 2026)<br>' +
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
                                 '<b>Capacity:</b> %{x:.1f} GWp<br>' +
                                 '<b>Profile Factor:</b> %{y:.1f}%<br>' +
                                 '<b>Equation:</b> y = ' + f'{coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}<br>' +
                                 '<b>Reaches 0% at:</b> {x_at_zero_linear:.1f} GWp<br>' +
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
            
            x_extrapolation = np.linspace(jan_1_capacity, 200, 100)  # Extend from Jan 1, 2026 to 200 GWp to show asymptotic behavior
            y_extrapolation = a_adjusted * np.exp(b * (x_extrapolation - jan_1_capacity))
            
            fig.add_trace(
                go.Scatter(
                    x=x_extrapolation,
                    y=y_extrapolation,
                    mode='lines',
                    name='Exponential Extrapolation',
                    line=dict(color='orange', width=2, dash='dash'),
                    hovertemplate='<b>Exponential Extrapolation (from Jan 1, 2026)</b><br>' +
                                 '<b>Capacity:</b> %{x:.1f} GWp<br>' +
                                 '<b>Profile Factor:</b> %{y:.1f}%<br>' +
                                 '<b>Equation:</b> y = ' + f'{a_adjusted:.2f} * e^({b:.3f}*(x-{jan_1_capacity:.1f}))<br>' +
                                 '<b>Starts from:</b> {jan_1_capacity:.1f} GWp ({y_at_jan_1_2026:.1f}%)<br>' +
                                 '<extra></extra>',
                    showlegend=True
                )
            )
            
            # Add annotations for different threshold levels
            for i, threshold in enumerate(thresholds):
                if threshold < a_adjusted:  # Only show if threshold is below the starting value
                    x_at_threshold = jan_1_capacity + (np.log(threshold / a_adjusted)) / b
                    if x_at_threshold <= 200:  # Only show if within our plot range
                        fig.add_annotation(
                            x=x_at_threshold,
                            y=threshold,
                            text=f'{threshold}%',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='orange',
                            ax=20,
                            ay=-20,
                            font=dict(size=9, color='orange'),
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='orange',
                            borderwidth=1
                        )
            
            
            
            # Add annotation for when linear trend reaches 0%
            fig.add_annotation(
                x=x_at_zero_linear,
                y=0,
                text=f'Linear trend at {x_at_zero_linear:.0f} GWp: 0%',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                ax=80,
                ay=-20,
                font=dict(size=10, color='red'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='red',
                borderwidth=1
            )
            
            # Add yellow arrow with text box for asymptotic extrapolation at 55 GWp
            # Calculate profile factor at 55 GWp using exponential equation
            profile_factor_at_55 = a_adjusted * np.exp(b * (55 - jan_1_capacity))
            
            fig.add_annotation(
                x=55,
                y=profile_factor_at_55,
                text=f'Exponential trend at 55 GWp: {profile_factor_at_55:.0f}%',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='orange',
                ax=80,
                ay=-30,
                font=dict(size=11, color='black'),
                bgcolor='rgba(255,165,0,0.9)',
                bordercolor='orange',
                borderwidth=2
            )
            
            # Add general annotation about asymptotic behavior
            fig.add_annotation(
                x=150,
                y=5,
                text='Asymptotic decay:<br>PV always retains<br>some market value',
                showarrow=False,
                font=dict(size=10, color='orange'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='orange',
                borderwidth=1,
                xref='x',
                yref='y'
            )
    else:
        # Fallback to linear if exponential fitting fails
        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_extended = np.linspace(0, 100, 100)
        trend_line_extended = np.polyval(coeffs, x_extended)
        
        fig.add_trace(
            go.Scatter(
                x=x_extended,
                y=trend_line_extended,
                mode='lines',
                name='Linear Trend (fallback)',
                line=dict(color='red', width=2),
                hovertemplate='<b>Linear Trend (fallback)</b><br>' +
                             f'<b>Slope:</b> {coeffs[0]:.2f} %/GWp<br>' +
                             f'<b>Intercept:</b> {coeffs[1]:.2f}%<br>' +
                             f'<extra></extra>',
                showlegend=True
            )
        )

# Add scatter plot points (after trend lines to appear on top)
for i, row in plot_data.iterrows():
    year = row['year']
    capacity = row['Installed_Capacity_GWp_DC']
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
                         f'<b>July 1st Installed Capacity:</b> {capacity:.2f} GWp DC<br>' +
                         f'<b>Profile Factor:</b> {profile_factor:.1f}%<br>' +
                         f'<extra></extra>',
            showlegend=True
        )
    )

# Add annotation for 2022 data point (Gascrisis outlier)
if 2022 in plot_data['year'].values:
    data_2022 = plot_data[plot_data['year'] == 2022].iloc[0]
    fig.add_annotation(
        x=data_2022['Installed_Capacity_GWp_DC']+1,
        y=data_2022['Yearly_Profile_Factor']+1,
        text='Outlier: gas-crisis',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=50,  # Adjusted to position arrow at circle edge (60 - 10 for radius)
        ay=-30,  # Adjusted to position arrow at circle edge (-40 + 10 for radius)
        font=dict(size=10, color='red'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='red',
        borderwidth=1
    )

# Add correlation coefficient and slope
if len(plot_data) > 2:
    # Use the same filtered data for correlation as used for trend line
    if 'trend_data' in locals() and len(trend_data) > 2:
        correlation = np.corrcoef(trend_data['Installed_Capacity_GWp_DC'].values, trend_data['Yearly_Profile_Factor'].values)[0, 1]
        slope = coeffs_linear[0]  # slope from linear trend
    else:
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        slope = coeffs_linear[0]  # slope from linear trend
    
    # Calculate a point on the trend line for the arrow to point to
    # Use a point around the middle of the trend line
    mid_capacity = (x_2018 + jan_1_capacity) / 2
    mid_profile_factor = np.polyval(coeffs_linear, mid_capacity)
    
    # Position text box to the left of the trend line
    text_x = mid_capacity - 8  # Position text to the left
    text_y = mid_profile_factor - 5  # Position text below the trend line
    
    fig.add_annotation(
        text=f'Slope: {round(slope, 1)}%/GWp',
        x=text_x,
        y=text_y,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=mid_capacity - text_x,  # Arrow points to trend line
        ay=mid_profile_factor - text_y,  # Arrow points to trend line
        font=dict(size=12, color='red'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='red',
        borderwidth=1
    )

# Write to HTML and PDF files
fig.write_html('pv_profile_factor_vs_capacity_dashboard.html', auto_open=True)
fig.write_image('pv_profile_factor_vs_capacity_dashboard.pdf', format='pdf')

print(f"\nDashboard created:")
print(f"  - HTML: pv_profile_factor_vs_capacity_dashboard.html")
print(f"  - PDF: pv_profile_factor_vs_capacity_dashboard.pdf")
print(f"Data points: {len(plot_data)} years")
print(f"Years covered: {plot_data['year'].min()} - {plot_data['year'].max()}")
print(f"Capacity range: {plot_data['Installed_Capacity_GWp_DC'].min():.2f} - {plot_data['Installed_Capacity_GWp_DC'].max():.2f} GWp DC")
print(f"Profile factor range: {plot_data['Yearly_Profile_Factor'].min():.1f} - {plot_data['Yearly_Profile_Factor'].max():.1f}%")

if len(plot_data) > 2:
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Slope: {round(slope, 1)}% per GWp")
    
    # Print linear trend (from 2018 data point to January 1, 2026)
    trend_years = "2018-2025 (excluding 2022)" if 'trend_data' in locals() and len(trend_data) > 2 else "all available data"
    print(f"Linear trend equation (based on {trend_years}): y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}")
    print(f"Linear trend starts at: {x_2018:.1f} GWp ({y_2018:.1f}%)")
    print(f"Linear trend ends at: {jan_1_capacity:.1f} GWp (January 1, 2026)")
    
    # Calculate when linear trend reaches 0%
    x_at_zero_linear = -coeffs_linear[1] / coeffs_linear[0]
    print(f"Linear trend reaches 0% at: {round(x_at_zero_linear, 0):.0f} GWp")
    
    # Print exponential equation if available
    if 'a' in locals() and 'b' in locals():
        print(f"Exponential extrapolation equation (based on {trend_years}): y = {a:.2f} * e^({b:.3f}*x)")
        print(f"Exponential decay rate: {b:.3f} per GWp")
        print("Asymptotic behavior: PV profile factor approaches 0% but never reaches it")
        print("PV will always retain some market value on the day-ahead market")
        
        # Calculate and print threshold milestones
        thresholds = [5.0, 2.0, 1.0]
        print("\nExponential extrapolation threshold milestones:")
        for threshold in thresholds:
            if threshold < a:
                x_at_threshold = (np.log(threshold / a)) / b
                print(f"  Profile factor reaches {threshold}% at: {x_at_threshold:.1f} GWp")
        
        # Calculate profile factor at 55 GWp
        profile_factor_at_55 = a * np.exp(b * 55)
        print(f"  Profile factor at 55 GWp: {profile_factor_at_55:.1f}%")
    else:
        print("Exponential extrapolation not available (fallback to linear only)")
