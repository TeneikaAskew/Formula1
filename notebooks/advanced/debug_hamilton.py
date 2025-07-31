"""Debug Hamilton driver mapping issue"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from f1db_data_loader import load_f1db_data

# Load data
print("Loading F1 data...")
data = load_f1db_data()
results = data['results']
races = data['races']
drivers = data['drivers']

# Find all Hamiltons
print("\nAll Hamiltons in database:")
hamiltons = drivers[drivers['lastName'] == 'Hamilton']
print(hamiltons[['id', 'forename', 'lastName', 'dateOfBirth']])

# Get 2025 results
results_2025 = results[results['year'] == 2025]
drivers_2025 = results_2025['driverId'].unique()

print(f"\nTotal drivers in 2025: {len(drivers_2025)}")
print("\nHamiltons who raced in 2025:")
for _, hamilton in hamiltons.iterrows():
    if hamilton['id'] in drivers_2025:
        print(f"  - {hamilton['id']} ({hamilton['forename']} {hamilton['lastName']})")
        # Show some race results
        h_results = results_2025[results_2025['driverId'] == hamilton['id']]
        print(f"    Races: {len(h_results)}")
        if not h_results.empty:
            print(f"    Average position: {h_results['positionNumber'].mean():.1f}")

# Check which Hamilton raced in recent years
print("\n\nRace history for each Hamilton:")
for _, hamilton in hamiltons.iterrows():
    h_results = results[results['driverId'] == hamilton['id']]
    if not h_results.empty and 'year' in h_results.columns:
        years = sorted(h_results['year'].unique())
        print(f"{hamilton['id']} ({hamilton['forename']}): {min(years)} - {max(years)}")
        # Show recent years
        recent_years = [y for y in years if y >= 2020]
        if recent_years:
            print(f"  Recent years: {recent_years}")