import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import glob
import warnings
os.system('clear')

# Compare Day-Ahead Prices of July 2025 vs June 2025
# Plot the prices and the statistics table
# Save the plot as a HTML file
# Hypothesis: is the root cause of higher PV weighted prices due to worse weather conditions in July 2025?
# Or is it due to more curtailment/gateway software installed compared to last year?



# --- Load all monthly DA_prices and PV data files ---
# Find all DA_prices and PV files
price_files = sorted(glob.glob('data/DA_prices_20*.csv'))


# Load July 2025 and June 2025 price files
june_file = None
july_file = None
for f in price_files:
    if '2025_06' in f or '202506' in f:
        june_file = f
    if '2025_07' in f or '202507' in f:
        july_file = f

if june_file is None or july_file is None:
    raise FileNotFoundError("Could not find both June 2025 and July 2025 price files.")

# Load data
df_june = pd.read_csv(june_file)
df_july = pd.read_csv(july_file)

# Parse time columns
df_june['time'] = pd.to_datetime(df_june['time'], utc=True).dt.tz_convert('Europe/Amsterdam')
df_july['time'] = pd.to_datetime(df_july['time'], utc=True).dt.tz_convert('Europe/Amsterdam')

# For plotting, align on hour of day if needed
df_june['hour'] = df_june['time'].dt.hour
df_june['date'] = df_june['time'].dt.date
df_july['hour'] = df_july['time'].dt.hour
df_july['date'] = df_july['time'].dt.date

# Plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create statistics tables
stats_june = df_june['DA_price'].describe().to_frame(name='June 2025')
stats_july = df_july['DA_price'].describe().to_frame(name='July 2025')
stats_table = pd.concat([stats_june, stats_july], axis=1)
stats_table = stats_table.round(2)

# Create subplots: 2 rows, 1 col (top: line plot, bottom: table)
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.15,
    row_heights=[0.7, 0.3],
    subplot_titles=("Day-Ahead Prices: June vs July 2025", "Statistics Table"),
    specs=[[{"type": "xy"}], [{"type": "table"}]]
)

# Plot June 2025
fig.add_trace(
    go.Scatter(
        x=df_june['time'],
        y=df_june['DA_price'],
        mode='lines',
        name='June 2025',
        line=dict(color='blue')
    ),
    row=1, col=1
)

# Plot July 2025
fig.add_trace(
    go.Scatter(
        x=df_july['time'],
        y=df_july['DA_price'],
        mode='lines',
        name='July 2025',
        line=dict(color='orange')
    ),
    row=1, col=1
)

# Add statistics table as a plotly Table
import plotly.figure_factory as ff

table_trace = go.Table(
    header=dict(
        values=["Statistic"] + list(stats_table.columns),
        fill_color='paleturquoise',
        align='left'
    ),
    cells=dict(
        values=[stats_table.index] + [stats_table[col].values for col in stats_table.columns],
        fill_color='lavender',
        align='left'
    )
)
fig.add_trace(table_trace, row=2, col=1)

# Update layout
fig.update_layout(
    title_text="Comparison of Day-Ahead Prices: July 2025 vs June 2025",
    showlegend=True
)
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text="DA Price (EUR/MWh)", row=1, col=1)

# Save as HTML and auto open
output_html = "compare_prices_july2025_vs_june2025.html"
fig.write_html(output_html, auto_open=True)
