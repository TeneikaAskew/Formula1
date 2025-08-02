#!/usr/bin/env python3
"""
Find the correct driver IDs for new 2025 drivers
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1db_data_loader import load_f1db_data

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

drivers = data.get('drivers', None)
if drivers is not None:
    print(f"\nTotal drivers in database: {len(drivers)}")
    
    # Search for new drivers
    search_names = ['lawson', 'hadjar', 'bortoleto']
    
    print("\nSearching for new drivers:")
    print("-" * 60)
    
    for name in search_names:
        matches = drivers[
            drivers['surname'].str.lower().str.contains(name) | 
            drivers['forename'].str.lower().str.contains(name)
        ]
        
        if not matches.empty:
            print(f"\nMatches for '{name}':")
            for _, driver in matches.iterrows():
                print(f"  ID: {driver['id']}, Name: {driver['forename']} {driver['surname']}, Code: {driver.get('code', 'N/A')}")
        else:
            print(f"\nNo matches found for '{name}'")
    
    # Also show the most recent drivers (highest IDs)
    print("\n\nMost recent drivers in database (highest IDs):")
    print("-" * 60)
    recent = drivers.nlargest(10, 'id')
    for _, driver in recent.iterrows():
        print(f"ID: {driver['id']}, Name: {driver['forename']} {driver['surname']}, Code: {driver.get('code', 'N/A')}")
    
    # Check seasons_drivers for 2025
    season_drivers = data.get('seasons_drivers', None)
    if season_drivers is not None and 'year' in season_drivers.columns:
        print("\n\nDrivers listed for 2025 season:")
        print("-" * 60)
        drivers_2025 = season_drivers[season_drivers['year'] == 2025]
        if not drivers_2025.empty:
            # Get driver details for 2025 season
            driver_ids_2025 = drivers_2025['driverId'].unique()
            drivers_2025_details = drivers[drivers['id'].isin(driver_ids_2025)]
            for _, driver in drivers_2025_details.iterrows():
                print(f"ID: {driver['id']}, Name: {driver['forename']} {driver['surname']}, Code: {driver.get('code', 'N/A')}")