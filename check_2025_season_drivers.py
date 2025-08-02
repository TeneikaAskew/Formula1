#!/usr/bin/env python3
"""
Check all drivers for 2025 season
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1db_data_loader import load_f1db_data

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

# Check seasons_drivers for 2025
season_drivers = data.get('seasons_drivers', None)
drivers = data.get('drivers', None)

if season_drivers is not None and drivers is not None:
    if 'year' in season_drivers.columns:
        print("\nDrivers listed for 2025 season:")
        print("-" * 80)
        drivers_2025 = season_drivers[season_drivers['year'] == 2025]
        
        if not drivers_2025.empty:
            # Get driver details for 2025 season
            driver_ids_2025 = drivers_2025['driverId'].unique()
            print(f"Found {len(driver_ids_2025)} drivers for 2025 season")
            
            # Get full driver info
            drivers_2025_details = drivers[drivers['id'].isin(driver_ids_2025)].sort_values('surname')
            
            print("\n2025 F1 Grid:")
            print("-" * 80)
            for _, driver in drivers_2025_details.iterrows():
                # Find constructor info
                constructor_info = ""
                driver_season = drivers_2025[drivers_2025['driverId'] == driver['id']]
                if not driver_season.empty:
                    constructor_id = driver_season.iloc[0].get('constructorId', '')
                    if constructor_id and 'constructors' in data:
                        constructors = data['constructors']
                        constructor = constructors[constructors['id'] == constructor_id]
                        if not constructor.empty:
                            constructor_info = f" - {constructor.iloc[0]['name']}"
                
                print(f"  {driver['surname']:20s} {driver['forename']:15s} [{driver.get('code', 'N/A'):3s}] (ID: {driver['id']}){constructor_info}")
            
            # Show driver IDs for easy copying
            print("\n\nDriver IDs for 2025 season (for code update):")
            print("-" * 80)
            print("known_2025_drivers = {")
            for driver_id in sorted(driver_ids_2025):
                driver_info = drivers[drivers['id'] == driver_id].iloc[0]
                print(f"    '{driver_id}',  # {driver_info['forename']} {driver_info['surname']}")
            print("}")
        else:
            print("No drivers found for 2025 season")
    else:
        print("Year column not found in seasons_drivers table")
else:
    print("Could not load required tables")