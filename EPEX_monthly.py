#%% import packages
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
from entsoe import EntsoeRawClient
from entsoe import EntsoePandasClient

# Clear console (optional)
import os
os.system('clear')  # for Mac/Linux

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

#%% set API key, date, location
api_key = os.getenv('ENTSOE_API_KEY', 'default_api_key')
client = EntsoePandasClient(api_key=api_key)
country_code = 'NL'  # Netherlands

#%% Retrieve and align price data

def get_da_prices_chunked(client, country_code, start, end):
    print(f"Retrieving Day-Ahead prices: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    try:
        prices = client.query_day_ahead_prices(
            country_code, 
            start=start,
            end=end
        )
        prices = prices.reset_index()  # Reset index to make datetime a column
        return prices
    except Exception as e:
        print(f"Error for period {start} to {end}: {e}")
        return pd.DataFrame()

#%% Retrieve and save/load data for each month from Jan 2024 to today

data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

start_year = 2019
start_month = 1

today = date.today()
end_year = today.year
end_month = today.month

all_data = []

for year in range(start_year, end_year + 1):
    # Determine which months to process for this year
    if year == start_year:
        month_start = start_month
    else:
        month_start = 1
    if year == end_year:
        month_end = end_month
    else:
        month_end = 12
    for month in range(month_start, month_end + 1):
        file_path = os.path.join(data_dir, f'DA_prices_{year}_{month:02d}.csv')
        # Determine if this is the last (most recent) month
        is_last_month = (year == end_year and month == end_month)
        if os.path.exists(file_path) and not is_last_month:
            print(f"File exists, skipping: {file_path}")
            continue
        # Set start and end timestamps for the month
        start = pd.Timestamp(f'{year}-{month:02d}-01 00:00:00', tz='Europe/Brussels')
        if month == 12:
            next_month = pd.Timestamp(f'{year+1}-01-01 00:00:00', tz='Europe/Brussels')
        else:
            next_month = pd.Timestamp(f'{year}-{month+1:02d}-01 00:00:00', tz='Europe/Brussels')
        # For current month, end at today (exclusive)
        if year == today.year and month == today.month:
            end = pd.Timestamp(today, tz='Europe/Brussels')
        else:
            end = next_month
        DA = get_da_prices_chunked(client, country_code, start, end)
        if not DA.empty:
            DA.rename(columns={DA.columns[0]: 'time', DA.columns[1]: 'DA_price'}, inplace=True)
            #DA['time'] = pd.to_datetime(DA['time'], utc=True)
            DA.to_csv(file_path, index=False)
            print(f"Saved data to {file_path}")
            all_data.append(DA)
        else:
            print(f"No data for {year}-{month:02d}")

# Optionally, combine all loaded data for further analysis
if all_data:
    df = pd.concat(all_data)
    print("Combined DataFrame:")
    print(df)
    df['time'] = pd.to_datetime(df['time'], utc=True) # .dt.tz_convert('Europe/Amsterdam')
    df.insert(1, 'hour', df['time'].dt.hour)
    print(df)
    df.to_csv('data/DA_prices_combined.csv', index=False)


