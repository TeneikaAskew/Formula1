"""Test script to check what data is available in F1DB"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import load_f1db_data

def check_data_availability():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    print("\n=== DATA AVAILABILITY CHECK ===")
    
    # Check races data
    races = data.get('races', pd.DataFrame())
    if not races.empty:
        print(f"\n✓ Races: {len(races)} races")
        print(f"  Years: {races['year'].min()} - {races['year'].max()}")
        print(f"  Latest 5 races:")
        latest_races = races.sort_values('date', ascending=False).head()
        print(latest_races[['id', 'year', 'date', 'officialName', 'circuitId']].to_string())
    else:
        print("\n✗ No races data found")
    
    # Check race results data
    results = data.get('results', pd.DataFrame())  # Note: mapped from races-race-results
    if not results.empty:
        print(f"\n✓ Race Results: {len(results)} results")
        # Check if year column exists
        if 'year' in results.columns:
            print(f"  Years: {results['year'].min()} - {results['year'].max()}")
        else:
            print("  'year' column not found in results - need to join with races")
        
        # Check sample of columns
        print(f"  Columns: {list(results.columns[:10])}")
        
        # Get unique drivers from recent data
        if 'year' in results.columns:
            recent_results = results[results['year'] >= 2022]
            unique_drivers = recent_results['driverId'].nunique()
            print(f"  Unique drivers (2022+): {unique_drivers}")
    else:
        print("\n✗ No race results data found")
    
    # Check drivers data
    drivers = data.get('drivers', pd.DataFrame())
    if not drivers.empty:
        print(f"\n✓ Drivers: {len(drivers)} drivers")
        print(f"  Sample columns: {list(drivers.columns[:10])}")
        # Show some recent drivers
        print("  Sample drivers:")
        print(drivers[['id', 'firstName', 'lastName', 'dateOfBirth']].tail(10).to_string())
    else:
        print("\n✗ No drivers data found")
    
    # Check pit stops data
    pit_stops = data.get('pit_stops', pd.DataFrame())  # Note: mapped from races-pit-stops
    if not pit_stops.empty:
        print(f"\n✓ Pit Stops: {len(pit_stops)} pit stops")
        # Check if time column exists and its format
        if 'time' in pit_stops.columns:
            sample_times = pit_stops['time'].dropna().head(10)
            print(f"  Sample times: {list(sample_times)}")
        if 'timeMillis' in pit_stops.columns:
            sample_millis = pit_stops['timeMillis'].dropna().head(10)
            print(f"  Sample timeMillis: {list(sample_millis)}")
    else:
        print("\n✗ No pit stops data found")
    
    # Check grid positions data
    grid = data.get('grid', pd.DataFrame())  # Check if this is mapped
    if grid.empty:
        grid = data.get('starting_grid', pd.DataFrame())
    if grid.empty:
        grid = data.get('races-starting-grid-positions', pd.DataFrame())
    
    if not grid.empty:
        print(f"\n✓ Starting Grid: {len(grid)} grid positions")
        print(f"  Columns: {list(grid.columns[:10])}")
    else:
        print("\n✗ No starting grid data found")
    
    # Check sprint results
    sprint = data.get('sprint_results', pd.DataFrame())
    if not sprint.empty:
        print(f"\n✓ Sprint Results: {len(sprint)} sprint results")
    else:
        print("\n✗ No sprint results data found")
    
    # Check data key mappings
    print("\n=== DATA KEYS AVAILABLE ===")
    for key in sorted(data.keys()):
        if isinstance(data[key], pd.DataFrame) and not data[key].empty:
            print(f"  '{key}': {len(data[key])} rows")

if __name__ == "__main__":
    check_data_availability()