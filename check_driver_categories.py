#!/usr/bin/env python3
"""
Check driver categories - who has raced vs who hasn't
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1db_data_loader import load_f1db_data
import pandas as pd

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

# Get 2025 season drivers
seasons_drivers = data.get('seasons_drivers', pd.DataFrame())
drivers_2025 = seasons_drivers[seasons_drivers['year'] == 2025]

print(f"\nTotal drivers listed for 2025 season: {len(drivers_2025)}")

# Get race results
results = data.get('results', pd.DataFrame())
races = data.get('races', pd.DataFrame())

# Get 2025 race results
races_2025 = races[races['year'] == 2025]
race_ids_2025 = races_2025['id'].values
results_2025 = results[results['raceId'].isin(race_ids_2025)]

# Drivers who have raced
drivers_who_raced = set(results_2025['driverId'].unique())
print(f"Drivers who have raced in 2025: {len(drivers_who_raced)}")

# Get driver standings
driver_standings = data.get('driver_standings', pd.DataFrame())
standings_2025 = driver_standings[driver_standings['raceId'].isin(race_ids_2025)]
drivers_in_standings = set(standings_2025['driverId'].unique())
print(f"Drivers in 2025 standings: {len(drivers_in_standings)}")

# All 2025 drivers
all_2025_drivers = set(drivers_2025['driverId'].unique())

# Find categories
only_in_seasons = all_2025_drivers - drivers_who_raced
print(f"\nDrivers listed but haven't raced: {len(only_in_seasons)}")

# Get driver details
drivers_table = data.get('drivers', pd.DataFrame())

if only_in_seasons:
    print("\nReserve/Test drivers (in seasons_drivers but no races):")
    for driver_id in sorted(only_in_seasons):
        driver_info = drivers_table[drivers_table['id'] == driver_id]
        if not driver_info.empty:
            d = driver_info.iloc[0]
            print(f"  - {d['forename']} {d['surname']} ({driver_id})")
            
# Check if any drivers in standings but not in race results
in_standings_not_results = drivers_in_standings - drivers_who_raced
if in_standings_not_results:
    print(f"\nDrivers in standings but not in results: {len(in_standings_not_results)}")
    for driver_id in sorted(in_standings_not_results):
        print(f"  - {driver_id}")