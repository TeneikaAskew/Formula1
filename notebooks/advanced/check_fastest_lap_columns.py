#!/usr/bin/env python3
"""Check fastest lap data columns"""

from f1db_data_loader import load_f1db_data
import pandas as pd

# Load data
data = load_f1db_data()

# Check fastest laps data
fastest_laps = data.get('fastest_laps', pd.DataFrame())
print(f"Fastest laps columns: {list(fastest_laps.columns)}")

# Sample data
if not fastest_laps.empty:
    print("\nSample data:")
    print(fastest_laps[['raceId', 'year', 'driverId', 'lap', 'time', 'timeMillis']].head(10))
    
    # Check if we have time data
    print(f"\nTime values non-null: {fastest_laps['time'].notna().sum()}")
    print(f"TimeMillis non-null: {fastest_laps['timeMillis'].notna().sum()}")
    
    # Sample with times
    with_times = fastest_laps[fastest_laps['time'].notna()]
    if not with_times.empty:
        print("\nSample with times:")
        print(with_times[['year', 'driverId', 'lap', 'time']].head(10))