import json, requests, time
from datetime import date, timedelta
import pandas as pd

# Load environment variables from .env file
import os
os.system('clear')
from dotenv import load_dotenv
load_dotenv()
NED_API_KEY = os.getenv("NED_API_KEY")


start_date  = date(2024, 5, 1)
end_date    = date(2024, 5, 31)
daysstep    = 5                 # De API kan om een of andere reden maar 144 datapunten per keer exporteren, max 6 dagen in uurwaardes, of 1 dag in kwartier of 10 minuten waardes.

exportname = 'data_export_NED_CO2_' + start_date.strftime("%Y%m%d") + '_to_' + end_date.strftime("%Y%m%d") + '_15min.xlsx'

url = "https://api.ned.nl/v1/utilizations"
df1 = pd.DataFrame(columns=['capacity', 'percentage','validfrom']) # initialise dataframe



current = start_date
while current <= end_date:
    next_date = current + timedelta(daysstep)
    if next_date > end_date + timedelta(1):
        next_date = end_date + timedelta(1)  # ensure we include the last day
    print(current.strftime("%Y-%m-%d"))
    headers = {
        'X-AUTH-TOKEN': NED_API_KEY,
        'accept': 'application/ld+json'
    }
    params = {
        'point': 0,                   # 0 = NL, https://ned.nl/nl/handleiding-api
        'type': 2,                    # 2 = Solar, 27 = CO2 emissions
        'granularity': 5,             # 3 = 10min, 4 = 15min, 5 = 1 hour, 6 = 1 day, 7 = 1 month, 8 = 1 year
        'granularitytimezone': 0,     # 0 = UTC, 1 = CET
        'classification': 2,          # 1 = future prediction, 2 = current, 3 = backcast
        'activity': 1,                # 1 = providing
        'validfrom[after]': current.strftime("%Y-%m-%d"),
        'validfrom[strictly_before]': next_date.strftime("%Y-%m-%d")
    }
    response = requests.get(url, headers=headers, params=params, allow_redirects=False).json()
    df = pd.json_normalize(response, "hydra:member")
    df = df.drop(columns=['@id','emissionfactor','emission', 'volume','@type','id','point','type','granularity','granularitytimezone','activity','classification','validto','lastupdate'])
    if df1.empty:
        df1 = df
    else:
        df1 = pd.concat([df1,df], ignore_index=True)
    time.sleep(0.01) # pauze om niet de API te overbevragen
    current = next_date

# print(json.dumps(response, separators=(",",":"), indent=4))
print(df1)



df1 = df1.rename(columns={'capacity': 'Solar_production_kW'})
df1['validfrom'] = pd.to_datetime(df1['validfrom'], utc=True) # .dt.tz_convert('Europe/Amsterdam'
df1 = df1.set_index('validfrom')




print(df1.columns)
print('Total dataframe to save:')
print(df1)
print(df1.describe())

# --- Plotly visualization ---
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 1. Prepare data for first subplot: Solar production in GW
solar_gw = df1['Solar_production_kW'] / 1e6

# 2. Prepare data for second subplot: Daily energy sum in GWh

# Detect timestep in hours from index

timestep_hours = (df1.index[1] - df1.index[0]).total_seconds() / 3600


energy_kwh = df1['Solar_production_kW'] * timestep_hours
energy_gwh = energy_kwh.resample('D').sum() / 1e6

# Month sum for subplot title (in GWh)
month_sum_gwh = energy_gwh.sum()

# Create subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                    subplot_titles=(
                        'Solar Production (GW)',
                        f'Daily Solar Energy (GWh) - Month sum: {month_sum_gwh:.2f} GWh'
                    ))

# First subplot: GW production
fig.add_trace(go.Scatter(x=solar_gw.index, y=solar_gw, name='Solar Production (GW)', line=dict(color='orange')), row=1, col=1)

# Second subplot: Daily GWh
fig.add_trace(go.Bar(x=energy_gwh.index, y=energy_gwh, name='Daily Energy (GWh)', marker_color='green'), row=2, col=1)

fig.update_layout(height=700, width=1000, title_text='Solar Production and Daily Energy (May 2024)', showlegend=False)
fig.update_xaxes(title_text='Date', row=2, col=1)
fig.update_yaxes(title_text='GW', row=1, col=1)
fig.update_yaxes(title_text='GWh', row=2, col=1)


fig.write_html("solar_production_plot.html", auto_open=True)



df1.to_csv(exportname, index=False)
print(f"Data exported to {exportname}")
