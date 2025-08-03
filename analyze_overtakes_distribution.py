#!/usr/bin/env python3
"""Analyze overtakes distribution to understand why median is mostly 0"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer
import pandas as pd

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Get results and grid data directly
results = data.get('results', data.get('races_race_results', pd.DataFrame())).copy()
grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
races = data.get('races', pd.DataFrame())

# Merge to calculate overtakes
if not results.empty and not grid.empty:
    # Merge results with starting grid
    overtake_data = results.merge(
        grid[['raceId', 'driverId', 'positionNumber']].rename(
            columns={'positionNumber': 'gridPosition'}
        ),
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Handle pit lane starts
    overtake_data['gridPosition'] = overtake_data['gridPosition'].fillna(20)
    
    # Calculate positions gained
    overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
    overtake_data['overtakes'] = overtake_data['positions_gained'].apply(lambda x: max(0, x))
    
    # Add year
    if not races.empty and 'year' not in overtake_data.columns:
        overtake_data = overtake_data.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
    
    # Filter recent years
    if 'year' in overtake_data.columns:
        recent_data = overtake_data[overtake_data['year'] >= 2022]
    else:
        recent_data = overtake_data
    
    # Analyze distribution for a few drivers
    print("Overtakes Distribution Analysis")
    print("=" * 80)
    
    drivers_to_check = ['max-verstappen', 'lewis-hamilton', 'lando-norris', 'charles-leclerc', 'fernando-alonso']
    
    for driver_id in drivers_to_check:
        driver_data = recent_data[recent_data['driverId'] == driver_id]['overtakes']
        if not driver_data.empty:
            drivers = data.get('drivers', pd.DataFrame())
            driver_name = drivers[drivers['id'] == driver_id]['name'].iloc[0] if not drivers.empty else driver_id
            
            print(f"\n{driver_name}:")
            print(f"  Total races: {len(driver_data)}")
            print(f"  Mean overtakes: {driver_data.mean():.2f}")
            print(f"  Median overtakes: {driver_data.median():.2f}")
            print(f"  Overtakes distribution:")
            
            # Count overtakes
            overtake_counts = driver_data.value_counts().sort_index()
            for overtakes, count in overtake_counts.items():
                if overtakes <= 5:  # Show first few values
                    print(f"    {int(overtakes)} overtakes: {count} races ({count/len(driver_data)*100:.1f}%)")
            
            # Show how many races with >5 overtakes
            more_than_5 = (driver_data > 5).sum()
            if more_than_5 > 0:
                print(f"    >5 overtakes: {more_than_5} races ({more_than_5/len(driver_data)*100:.1f}%)")
    
    # Overall statistics
    print("\n" + "="*80)
    print("Overall Statistics (2022-2025):")
    print(f"Total races analyzed: {len(recent_data['raceId'].unique())}")
    print(f"Percentage of race finishes with 0 overtakes: {(recent_data['overtakes'] == 0).sum() / len(recent_data) * 100:.1f}%")
    print(f"Percentage with 1-2 overtakes: {((recent_data['overtakes'] >= 1) & (recent_data['overtakes'] <= 2)).sum() / len(recent_data) * 100:.1f}%")
    print(f"Percentage with 3+ overtakes: {(recent_data['overtakes'] >= 3).sum() / len(recent_data) * 100:.1f}%")