#!/usr/bin/env python3
"""
Test analysis structure to understand why some drivers are missing
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1_performance_analysis import F1PerformanceAnalyzer
from f1db_data_loader import load_f1db_data

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get all active drivers
active_drivers = analyzer.get_active_drivers()
print(f"\nTotal active drivers for {analyzer.current_year}: {len(active_drivers)}")

# Check race results
results = data.get('results', None)
if results is not None:
    races = data.get('races', None)
    if races is not None:
        results = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
        results_2025 = results[results['year'] == 2025]
        
        drivers_with_results = results_2025['driverId'].unique()
        print(f"\nDrivers with race results in 2025: {len(drivers_with_results)}")
        
        # Find drivers without results
        active_ids = set(active_drivers['id'].values)
        drivers_with_results_set = set(drivers_with_results)
        missing_from_results = active_ids - drivers_with_results_set
        
        if missing_from_results:
            print(f"\nDrivers without race results ({len(missing_from_results)}):")
            for driver_id in sorted(missing_from_results):
                driver_info = active_drivers[active_drivers['id'] == driver_id].iloc[0]
                print(f"  - {driver_info['forename']} {driver_info['surname']} ({driver_id})")

# Check grid positions
grid = data.get('races_starting_grid_positions', None)
if grid is not None and races is not None:
    grid = grid.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
    grid_2025 = grid[grid['year'] == 2025]
    
    drivers_with_grid = grid_2025['driverId'].unique()
    print(f"\nDrivers with grid positions in 2025: {len(drivers_with_grid)}")
    
    missing_from_grid = active_ids - set(drivers_with_grid)
    if missing_from_grid:
        print(f"\nDrivers without grid positions ({len(missing_from_grid)}):")
        for driver_id in sorted(missing_from_grid):
            driver_info = active_drivers[active_drivers['id'] == driver_id].iloc[0]
            print(f"  - {driver_info['forename']} {driver_info['surname']} ({driver_id})")