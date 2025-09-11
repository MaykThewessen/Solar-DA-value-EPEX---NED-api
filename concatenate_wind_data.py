#!/usr/bin/env python3
"""
Script to concatenate monthly wind data files by year.
Takes all data_export_NED_Wind_YYYYMM.csv files and combines them into yearly files
named wind_generation_YYYY.csv
"""

import os
import pandas as pd
import glob
from pathlib import Path
import re

def concatenate_wind_data_by_year():
    """
    Concatenate monthly wind data files by year and save as yearly CSV files.
    """
    # Define the data directory
    data_dir = Path("data")
    
    # Find all Wind data files matching the pattern
    wind_files = glob.glob(str(data_dir / "data_export_NED_Wind_*.csv"))
    
    if not wind_files:
        print("No Wind data files found in the data directory.")
        return
    
    # Group files by year
    yearly_data = {}
    
    for file_path in wind_files:
        # Extract year from filename using regex
        filename = os.path.basename(file_path)
        match = re.search(r'data_export_NED_Wind_(\d{4})\d{2}\.csv', filename)
        
        if match:
            year = match.group(1)
            
            if year not in yearly_data:
                yearly_data[year] = []
            
            yearly_data[year].append(file_path)
    
    # Process each year
    for year, file_list in yearly_data.items():
        print(f"Processing year {year} with {len(file_list)} files...")
        
        # Sort files by month to ensure proper chronological order
        file_list.sort()
        
        # Read and concatenate all files for this year
        dataframes = []
        
        for file_path in file_list:
            try:
                df = pd.read_csv(file_path)
                # Keep only time and Wind_production_MW columns, rename Wind_production_MW to wind_generation
                df = df[['time', 'Wind_production_MW']].copy()
                df = df.rename(columns={'Wind_production_MW': 'wind_generation'})
                dataframes.append(df)
                print(f"  - Loaded {os.path.basename(file_path)}: {len(df)} rows")
            except Exception as e:
                print(f"  - Error loading {file_path}: {e}")
                continue
        
        if dataframes:
            # Concatenate all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Sort by time to ensure chronological order
            combined_df['time'] = pd.to_datetime(combined_df['time'])
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            # Save the combined data to data/yearly folder
            output_filename = f"data/yearly/wind_generation_{year}.csv"
            combined_df.to_csv(output_filename, index=False)
            
            print(f"  - Saved {output_filename} with {len(combined_df)} total rows")
            print(f"  - Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
        else:
            print(f"  - No valid data found for year {year}")
        
        print()

def main():
    """
    Main function to run the concatenation process.
    """
    print("Starting wind data concatenation by year...")
    print("=" * 50)
    
    concatenate_wind_data_by_year()
    
    print("=" * 50)
    print("Concatenation complete!")

if __name__ == "__main__":
    main()
