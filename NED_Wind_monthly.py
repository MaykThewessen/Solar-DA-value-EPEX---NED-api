import json, requests, time
from datetime import date, timedelta
import pandas as pd
import calendar

# Load environment variables from .env file
import os
os.system('clear')
from dotenv import load_dotenv
load_dotenv()
NED_API_KEY = os.getenv("NED_API_KEY")

daysstep    = 6                 # De API kan maar 144 datapunten per keer exporteren, max 6 dagen in uurwaardes, of 1 dag in kwartier of 10 minuten waardes.


# Ensure 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

start_date = date(2019, 1, 1)
end_date = date.today()

url = "https://api.ned.nl/v1/utilizations"
df1 = pd.DataFrame(columns=['capacity', 'percentage','validfrom']) # initialise dataframe



current = start_date
while current <= end_date:
    # Determine the last day of the current month or today if in the current month
    if current.year == end_date.year and current.month == end_date.month:
        month_end = end_date
        is_current_month = True
    else:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        is_current_month = False

    exportname = f"data/data_export_NED_Wind_{current.strftime('%Y%m')}.csv"
    if os.path.exists(exportname) and not is_current_month:
        print(f"{exportname} already exists, skipping...")
        # Move to next month
        current = (month_end + timedelta(1)).replace(day=1) if month_end.month != 12 else date(month_end.year + 1, 1, 1)
        continue

    df1 = pd.DataFrame(columns=['capacity', 'percentage','validfrom'])
    period_start = current
    while period_start <= month_end:
        next_date = period_start + timedelta(daysstep)
        if next_date > month_end + timedelta(1):
            next_date = month_end + timedelta(1)
        print(period_start.strftime("%Y-%m-%d"))
        headers = {
            'X-AUTH-TOKEN': NED_API_KEY,
            'accept': 'application/ld+json'
        }
        params = {
            'point': 0,                   # 0 = NL, https://ned.nl/nl/handleiding-api
            'type': 1,                    # 1 = Wind, 2 = Solar, 27 = CO2 emissions
            'granularity': 5,             # 3 = 10min, 4 = 15min, 5 = 1 hour, 6 = 1 day, 7 = 1 month, 8 = 1 year
            'granularitytimezone': 0,     # 0 = UTC, 1 = CET
            'classification': 2,          # 1 = future prediction, 2 = current, 3 = backcast
            'activity': 1,                # 1 = providing
            'validfrom[after]': period_start.strftime("%Y-%m-%d"),
            'validfrom[strictly_before]': next_date.strftime("%Y-%m-%d")
        }
        response = requests.get(url, headers=headers, params=params, allow_redirects=False).json()
        
        # Debug: Print response structure to understand what we're getting
        if not response or 'hydra:member' not in response:
            print(f"Warning: No 'hydra:member' found in response for {period_start.strftime('%Y-%m-%d')}")
            print(f"Response keys: {list(response.keys()) if response else 'Empty response'}")
            if response and 'hydra:description' in response:
                print(f"API Error: {response['hydra:description']}")
            time.sleep(0.1)
            period_start = next_date
            continue
            
        df = pd.json_normalize(response, "hydra:member")
        if not df.empty:
            df = df.drop(columns=['@id','emissionfactor','emission', 'volume','@type','id','point','type','granularity','granularitytimezone','activity','classification','validto','lastupdate'], errors='ignore')
            if df1.empty:
                df1 = df
            else:
                df1 = pd.concat([df1,df], ignore_index=True)
        time.sleep(0.1)
        period_start = next_date
    if not df1.empty:
        df1 = df1.rename(columns={'capacity': 'Wind_production_kW'})
        df1['Wind_production_kW'] = (df1['Wind_production_kW']/1000).round(0)
        df1 = df1.rename(columns={'Wind_production_kW': 'Wind_production_MW'})

        df1['percentage'] = df1['percentage'].round(4)
        df1 = df1.rename(columns={'validfrom': 'time'})
        df1['time'] = pd.to_datetime(df1['time'], utc=True)
        df1 = df1.set_index('time')
        

        timestep_hours = (df1.index[1] - df1.index[0]).total_seconds() / 3600
        
        #df1['energy_kwh'] = df1['Wind_production_kW'] * timestep_hours
        df1.to_csv(exportname)
        print(f"Data exported to {exportname}")
    # Always increment to the first day of the next month
    if current.month == 12:
        current = date(current.year + 1, 1, 1)
    else:
        current = date(current.year, current.month + 1, 1)



# print(json.dumps(response, separators=(",",":"), indent=4))
# Only print df1 if it exists and has data
if 'df1' in locals() and not df1.empty:
    print(df1)
