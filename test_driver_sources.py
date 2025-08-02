#!/usr/bin/env python3
"""
Check which drivers come from which data source
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

print(f"\nCurrent year: {analyzer.current_year}")

# Check driver_standings
driver_standings = data.get('driver_standings', None)
if driver_standings is not None:
    races = data.get('races', None)
    if races is not None and 'raceId' in driver_standings.columns:
        driver_standings_with_year = driver_standings.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
        if 'year' in driver_standings_with_year.columns:
            standings_2025 = driver_standings_with_year[driver_standings_with_year['year'] == 2025]['driverId'].unique()
            print(f"\nDrivers from driver_standings (2025): {len(standings_2025)} drivers")
            print("Sample:", sorted(standings_2025)[:5])
        else:
            print("\nCould not add year to driver_standings")
    else:
        print("\nNo raceId column in driver_standings")

# Check seasons_drivers
seasons_drivers = data.get('seasons_drivers', None)
if seasons_drivers is not None:
    season_2025 = seasons_drivers[seasons_drivers['year'] == 2025]['driverId'].unique()
    print(f"\nDrivers from seasons_drivers (2025): {len(season_2025)} drivers")
    print("Sample:", sorted(season_2025)[:5])

# Check results
results = data.get('results', None)
if results is not None and races is not None:
    results_with_year = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
    if 'year' in results_with_year.columns:
        results_2025 = results_with_year[results_with_year['year'] == 2025]['driverId'].unique()
        print(f"\nDrivers from results (2025): {len(results_2025)} drivers")
        print("Sample:", sorted(results_2025)[:5])
    else:
        print("\nCould not add year to results")

# Get active drivers
active_drivers = analyzer.get_active_drivers()
print(f"\nTotal active drivers found: {len(active_drivers)}")

# Show which drivers would be missing without each source
print("\nChecking for drivers that appear in standings but not seasons_drivers:")
if 'standings_2025' in locals() and 'season_2025' in locals():
    standings_only = set(standings_2025) - set(season_2025)
    if standings_only:
        print(f"Found {len(standings_only)} drivers only in standings: {sorted(standings_only)}")
    else:
        print("All drivers in standings also appear in seasons_drivers")
        
# Show actual 2025 drivers from data
if 'season_2025' in locals():
    print(f"\nAll 2025 drivers from seasons_drivers ({len(season_2025)} total):")
    for driver_id in sorted(season_2025):
        print(f"  '{driver_id}',")