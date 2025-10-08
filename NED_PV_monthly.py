import json, requests, time
from datetime import date, timedelta, datetime
import pandas as pd
import calendar

# Load environment variables from .env file
import os
os.system('clear')
from dotenv import load_dotenv
load_dotenv()
NED_API_KEY = os.getenv("NED_API_KEY")
daysstep    = 6                 # De API kan om een of andere reden maar 144 datapunten per keer exporteren, max 6 dagen in uurwaardes, of 1 dag in kwartier of 10 minuten waardes.

# Ensure 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')


start_date = date(2018, 1, 1)
today = date.today()
end_date = date(today.year, today.month, 1) - timedelta(days=1)

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

    exportname = f"data/NED_PV/data_NED_PV_{current.strftime('%Y%m')}.csv"
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
            'type': 2,                    # 1 = Wind, 2 = Solar, 27 = CO2 emissions
            'granularity': 5,             # 3 = 10min, 4 = 15min, 5 = 1 hour, 6 = 1 day, 7 = 1 month, 8 = 1 year
            'granularitytimezone': 1,     # 0 = UTC, 1 = CET
            'classification': 2,          # 1 = future prediction (day-ahead), 2 = current, 3 = backcast
            'activity': 1,                # 1 = providing
            'validfrom[after]': period_start.strftime("%Y-%m-%d"),
            'validfrom[strictly_before]': next_date.strftime("%Y-%m-%d")
        }
        
        # Retry logic with exponential backoff
        max_retries = 5
        retry_count = 0
        retry_delay = 1.0
        response = None
        
        while retry_count < max_retries:
            response = requests.get(url, headers=headers, params=params, allow_redirects=False).json()
            
            # Check if response contains expected data
            if 'hydra:member' in response:
                break  # Success, exit retry loop
            
            # Rate limit or error hit
            retry_count += 1
            if retry_count < max_retries:
                print(f"Warning: Rate limit hit for {period_start.strftime('%Y-%m-%d')}, retrying ({retry_count}/{max_retries}) after {retry_delay:.1f}s")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Error: Failed to get data after {max_retries} retries for {period_start.strftime('%Y-%m-%d')}")
                if response:
                    print(f"Response: {json.dumps(response, indent=2)}")
                break
        
        # If we still don't have valid data after all retries, skip this period
        if not response or 'hydra:member' not in response:
            print(f"Skipping period {period_start.strftime('%Y-%m-%d')} due to persistent errors")
            period_start = next_date
            continue
            
        df = pd.json_normalize(response, "hydra:member")

        # Check if df is empty
        if df.empty:
            print(f"No data found for {period_start.strftime('%Y-%m-%d')}, skipping...")
            period_start = next_date
            continue
            
        if not df.empty:
            df = df.drop(columns=['@id','emissionfactor','emission', 'volume','@type','id','point','type','granularity','granularitytimezone','activity','classification','validto','lastupdate'], errors='ignore')
            if df1.empty:
                df1 = df
            else:
                df1 = pd.concat([df1,df], ignore_index=True)
        time.sleep(0.3)  # reduce rate limit hits
        period_start = next_date

    if not df1.empty:
        df1 = df1.rename(columns={'capacity': 'Solar_production_kW'})
        df1['Solar_production_kW'] = (df1['Solar_production_kW']/1000).astype(int)
        df1 = df1.rename(columns={'Solar_production_kW': 'Solar_production_MW'})

        df1['percentage'] = (df1['percentage']*100).round(2)
        df1 = df1.rename(columns={'validfrom': 'time'})
        # Convert to datetime and handle timezone
        df1['time'] = pd.to_datetime(df1['time'])
        if df1['time'].dt.tz is None:
            # If timezone-naive, localize to Europe/Brussels
            df1['time'] = df1['time'].dt.tz_localize('Europe/Brussels')
        else:
            # If already timezone-aware, convert to Europe/Brussels
            df1['time'] = df1['time'].dt.tz_convert('Europe/Brussels')
        df1 = df1.set_index('time')
        
        df1.to_csv(exportname, date_format='%Y-%m-%d %H:%M:%S%z')
        print(f"✓ Data exported to {exportname}")
    else:
        print(f"✗ No data available for {current.strftime('%Y-%m')}, skipping month...")
        

    # Always increment to the first day of the next month
    if current.month == 12:
        current = date(current.year + 1, 1, 1)
    else:
        current = date(current.year, current.month + 1, 1)



# print(json.dumps(response, separators=(",",":"), indent=4))
# Only print df1 if it exists and has data
if 'df1' in locals() and not df1.empty:
    print(df1)
