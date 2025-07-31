"""Debug version of performance analysis to understand data flow"""

import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import load_f1db_data

def debug_data_flow():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    print("\n=== DEBUGGING DATA FLOW ===")
    
    # Check what year ranges we have
    results = data.get('results', pd.DataFrame())
    print(f"\nResults data shape: {results.shape}")
    print(f"Results columns: {list(results.columns[:15])}")
    
    # Check if year column exists
    if 'year' in results.columns:
        print(f"Year range in results: {results['year'].min()} - {results['year'].max()}")
        year_counts = results['year'].value_counts().sort_index().tail(10)
        print("\nResults per year (last 10 years):")
        print(year_counts)
    else:
        print("'year' column not found in results!")
    
    # Check races data
    races = data.get('races', pd.DataFrame())
    print(f"\nRaces data shape: {races.shape}")
    if not races.empty:
        print(f"Year range in races: {races['year'].min()} - {races['year'].max()}")
        recent_races = races[races['year'] >= 2022].sort_values('date')
        print(f"\nNumber of races 2022-2024: {len(recent_races)}")
        print("\nSample recent races:")
        print(recent_races[['id', 'year', 'date', 'officialName']].tail(10))
    
    # Test filtering for 2024 data
    print("\n=== TESTING 2024 DATA FILTERING ===")
    
    # Filter results for 2024
    if 'year' in results.columns:
        results_2024 = results[results['year'] == 2024]
        print(f"Results in 2024: {len(results_2024)}")
        if not results_2024.empty:
            print("Sample 2024 results:")
            print(results_2024[['raceId', 'driverId', 'positionNumber', 'points']].head())
    
    # Check pit stops
    pit_stops = data.get('pit_stops', pd.DataFrame())
    print(f"\nPit stops data shape: {pit_stops.shape}")
    print(f"Pit stops columns: {list(pit_stops.columns[:10])}")
    
    # Check if we need to merge with races to get year
    if 'year' not in pit_stops.columns and not races.empty:
        print("\nMerging pit stops with races to add year...")
        pit_stops_with_year = pit_stops.merge(
            races[['id', 'year']], 
            left_on='raceId', 
            right_on='id', 
            how='left'
        )
        recent_stops = pit_stops_with_year[pit_stops_with_year['year'] >= 2022]
        print(f"Pit stops 2022-2024: {len(recent_stops)}")
    
    # Check grid positions
    grid = data.get('races_starting_grid_positions', pd.DataFrame())
    print(f"\nStarting grid data shape: {grid.shape}")
    if not grid.empty:
        print(f"Grid columns: {list(grid.columns[:10])}")
    
    # Test active drivers for 2024
    print("\n=== TESTING ACTIVE DRIVERS ===")
    if 'year' in results.columns:
        results_2024 = results[results['year'] == 2024]
        if not results_2024.empty:
            unique_drivers_2024 = results_2024['driverId'].unique()
            print(f"Unique drivers in 2024: {len(unique_drivers_2024)}")
            print(f"Sample driver IDs: {list(unique_drivers_2024[:10])}")
            
            # Get driver names
            drivers = data.get('drivers', pd.DataFrame())
            if not drivers.empty:
                active_drivers = drivers[drivers['id'].isin(unique_drivers_2024)]
                print(f"\nActive drivers found: {len(active_drivers)}")
                print("Sample active drivers:")
                print(active_drivers[['id', 'firstName', 'lastName']].head(10))
    
    # Debug the year filtering issue
    print("\n=== DEBUGGING YEAR FILTERING ===")
    current_year = 2024
    print(f"Using current_year = {current_year}")
    print(f"Looking for data from {current_year-3} to {current_year}")
    
    if 'year' in results.columns:
        recent_results = results[(results['year'] >= current_year - 3) & (results['year'] <= current_year)]
        print(f"\nResults found for years {current_year-3}-{current_year}: {len(recent_results)}")
        
        # Group by year to see distribution
        year_distribution = recent_results.groupby('year').size()
        print("\nResults per year:")
        print(year_distribution)

if __name__ == "__main__":
    debug_data_flow()