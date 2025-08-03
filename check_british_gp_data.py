#!/usr/bin/env python3
"""Check British GP data for Oliver Bearman and Lewis Hamilton"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
import pandas as pd

# Load data
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Get relevant data
results = data.get('results', data.get('races_race_results', pd.DataFrame()))
grid = data.get('races_starting_grid_positions', pd.DataFrame())
races = data.get('races', pd.DataFrame())

# Find British GP 2025
british_gp = races[(races['year'] == 2025) & (races['officialName'].str.contains('British', na=False))]
if not british_gp.empty:
    race_id = british_gp.iloc[0]['id']
    print(f"British GP 2025 Race ID: {race_id}")
    
    # Get results for this race
    race_results = results[results['raceId'] == race_id]
    race_grid = grid[grid['raceId'] == race_id]
    
    # Merge to get full data
    full_data = race_results.merge(
        race_grid[['driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
        on='driverId',
        how='left'
    )
    
    # Add driver names
    drivers = data.get('drivers', pd.DataFrame())
    if not drivers.empty:
        driver_map = dict(zip(drivers['id'], drivers['name']))
        full_data['driver_name'] = full_data['driverId'].map(driver_map)
    
    # Check specific drivers
    print("\nBritish GP 2025 Data:")
    print("-" * 80)
    
    # First, let's check if Lewis Hamilton actually finished 7th with 6 points
    lewis_data = full_data[full_data['driverId'] == 'lewis-hamilton']
    if not lewis_data.empty:
        print(f"\nLewis Hamilton data check:")
        print(f"  Finish position from data: {lewis_data.iloc[0]['positionNumber']}")
        print(f"  Points from data: {lewis_data.iloc[0]['points']}")
    
    # Based on user data, let me check who finished 7th
    seventh_place = full_data[full_data['positionNumber'] == 7]
    if not seventh_place.empty:
        print(f"\nDriver who finished 7th: {seventh_place.iloc[0]['driver_name']}")
        print(f"Points for 7th: {seventh_place.iloc[0]['points']}")
    
    for driver_name in ['Oliver Bearman', 'Lewis Hamilton']:
        driver_data = full_data[full_data['driver_name'] == driver_name]
        if not driver_data.empty:
            row = driver_data.iloc[0]
            grid_pos = row['gridPosition'] if pd.notna(row['gridPosition']) else 20
            finish_pos = row['positionNumber']
            positions_gained = grid_pos - finish_pos
            
            print(f"\n{driver_name}:")
            print(f"  Grid Position: {int(grid_pos)}")
            print(f"  Finish Position: {int(finish_pos)}")
            print(f"  Positions Gained: {int(positions_gained)}")
            print(f"  Championship Points: {int(row['points']) if pd.notna(row['points']) else 0}")
            
            # Check for DNF
            if pd.notna(row.get('reasonRetired')):
                print(f"  Status: DNF - {row['reasonRetired']}")
    
    # Show all drivers for reference
    print("\n\nAll drivers (sorted by positions gained):")
    print("-" * 80)
    full_data['positions_gained'] = full_data['gridPosition'] - full_data['positionNumber']
    full_data_sorted = full_data.sort_values('positions_gained', ascending=False)
    
    for _, row in full_data_sorted.iterrows():
        if pd.notna(row['driver_name']):
            grid_pos = row['gridPosition'] if pd.notna(row['gridPosition']) else 20
            finish_pos = row['positionNumber']
            if pd.notna(finish_pos):
                print(f"{row['driver_name']:20s}: P{int(grid_pos):2d} → P{int(finish_pos):2d} "
                      f"(gained {int(row['positions_gained']):+3d})")
            else:
                print(f"{row['driver_name']:20s}: P{int(grid_pos):2d} → DNF")
else:
    print("British GP 2025 not found")