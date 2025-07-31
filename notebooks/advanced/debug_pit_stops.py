"""Debug pit stops data"""

import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from f1db_data_loader import load_f1db_data

def debug_pit_stops():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    # Check pit stops data
    pit_stops = data.get('pit_stops', pd.DataFrame())
    print(f"\nPit stops shape: {pit_stops.shape}")
    print(f"Columns: {list(pit_stops.columns)}")
    
    if not pit_stops.empty:
        # Check if year column exists
        if 'year' in pit_stops.columns:
            print(f"\nYear range: {pit_stops['year'].min()} - {pit_stops['year'].max()}")
            print(f"2025 pit stops: {len(pit_stops[pit_stops['year'] == 2025])}")
        else:
            print("\nNo year column in pit_stops")
            
        # Check time columns
        print("\nTime-related columns:")
        for col in pit_stops.columns:
            if 'time' in col.lower():
                print(f"  {col}: {pit_stops[col].dtype}")
                print(f"    Sample values: {pit_stops[col].dropna().head(5).tolist()}")
        
        # Try to add year from races
        races = data.get('races', pd.DataFrame())
        if not races.empty and 'year' not in pit_stops.columns:
            print("\nMerging with races to add year...")
            pit_stops_with_year = pit_stops.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
            print(f"After merge shape: {pit_stops_with_year.shape}")
            
            if 'year' in pit_stops_with_year.columns:
                print(f"Year range after merge: {pit_stops_with_year['year'].min()} - {pit_stops_with_year['year'].max()}")
                print(f"2025 pit stops: {len(pit_stops_with_year[pit_stops_with_year['year'] == 2025])}")
                
                # Check 2025 specifically
                pit_2025 = pit_stops_with_year[pit_stops_with_year['year'] == 2025]
                if not pit_2025.empty:
                    print(f"\n2025 pit stops sample:")
                    print(pit_2025[['raceId', 'driverId', 'time', 'timeMillis']].head(10))

if __name__ == "__main__":
    debug_pit_stops()