#!/usr/bin/env python3
"""Check what lap data is available in F1DB"""

from f1db_data_loader import load_f1db_data
import pandas as pd

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Check if we have lap times data
lap_times = data.get('lap_times', pd.DataFrame())
print(f"\nLap times data shape: {lap_times.shape}")
if not lap_times.empty:
    print(f"Lap times columns: {list(lap_times.columns)}")
    print(f"Sample data:\n{lap_times.head()}")
else:
    print("No lap times data available")

# Check fastest laps data
fastest_laps = data.get('fastest_laps', pd.DataFrame())
print(f"\nFastest laps data shape: {fastest_laps.shape}")
if not fastest_laps.empty:
    print(f"Fastest laps columns: {list(fastest_laps.columns)}")
    print(f"Sample data:\n{fastest_laps.head()}")

# Check results for any lap-related info
results = data.get('results', pd.DataFrame())
print(f"\nResults columns with 'lap':")
lap_cols = [col for col in results.columns if 'lap' in col.lower()]
print(lap_cols)

# Check for grand slam data (which implies leading every lap)
if 'grandSlam' in results.columns:
    grand_slams = results[results['grandSlam'] == True]
    print(f"\nGrand Slams (led every lap): {len(grand_slams)} occurrences")
else:
    print("\nNo grandSlam column in results")