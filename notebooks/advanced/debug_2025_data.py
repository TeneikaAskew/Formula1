"""Debug 2025 data availability"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import load_f1db_data

def debug_2025_data():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    print("\n=== 2025 DATA CHECK ===")
    
    # Check 2025 races
    races = data.get('races', pd.DataFrame())
    if not races.empty:
        races_2025 = races[races['year'] == 2025]
        print(f"\n2025 Races scheduled: {len(races_2025)}")
        print(races_2025[['id', 'date', 'officialName']].head(10))
    
    # Check 2025 results
    results = data.get('results', pd.DataFrame())
    if not results.empty:
        results_2025 = results[results['year'] == 2025]
        print(f"\n2025 Race results entries: {len(results_2025)}")
        
        if not results_2025.empty:
            # Count races with results
            races_with_results = results_2025['raceId'].nunique()
            print(f"Races with results: {races_with_results}")
            
            # Show which races have results
            race_summary = results_2025.groupby('raceId').agg({
                'driverId': 'count',
                'positionNumber': lambda x: x.notna().sum()
            }).rename(columns={'driverId': 'entries', 'positionNumber': 'finished'})
            
            print("\nRaces with results:")
            print(race_summary.head(10))
            
            # Show drivers in 2025
            drivers_2025 = results_2025['driverId'].unique()
            print(f"\nUnique drivers in 2025: {len(drivers_2025)}")
            print(f"Drivers: {', '.join(drivers_2025[:10])}...")
            
            # Check for valid finishing positions
            finished_results = results_2025[results_2025['positionNumber'].notna() & (results_2025['positionNumber'] > 0)]
            print(f"\nResults with valid finishing positions: {len(finished_results)}")
            
            if not finished_results.empty:
                print("\nSample 2025 results with positions:")
                print(finished_results[['raceId', 'driverId', 'positionNumber', 'points']].head(10))

if __name__ == "__main__":
    debug_2025_data()