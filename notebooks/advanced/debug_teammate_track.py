#!/usr/bin/env python3
"""Debug teammate overtakes track-specific data"""

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer
import pandas as pd

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)
next_race = analyzer.get_next_race()

if next_race is not None:
    print(f"\nNext race: {next_race.get('officialName', 'Unknown')}")
    print(f"Circuit ID: {next_race.get('circuitId', 'Missing')}")

# Get the data needed for teammate overtakes
results = data.get('results', pd.DataFrame()).copy()
grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
races = data.get('races', pd.DataFrame())

print(f"\nResults shape: {results.shape}")
print(f"Grid shape: {grid.shape}")
print(f"Grid columns: {list(grid.columns)}")

# Test the merge
if not results.empty and not grid.empty:
    print("\nTesting merge...")
    overtake_data = results.merge(
        grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
        on=['raceId', 'driverId'],
        how='left'
    )
    print(f"Merged shape: {overtake_data.shape}")
    print(f"Columns after merge: {list(overtake_data.columns)}")
    
    # Add year and circuit info
    if 'year' not in overtake_data.columns or 'circuitId' not in overtake_data.columns:
        merge_cols = ['id']
        if 'year' not in overtake_data.columns:
            merge_cols.append('year')
        if 'circuitId' not in overtake_data.columns:
            merge_cols.append('circuitId')
        print(f"\nMerging with races table for columns: {merge_cols}")
        overtake_data = overtake_data.merge(
            races[merge_cols], 
            left_on='raceId', 
            right_on='id', 
            how='left'
        )
        print(f"Shape after race merge: {overtake_data.shape}")
        print(f"Has year column: {'year' in overtake_data.columns}")
        print(f"Has circuitId column: {'circuitId' in overtake_data.columns}")
    
    # Check data for next circuit
    if next_race is not None and 'circuitId' in next_race:
        circuit_id = next_race['circuitId']
        circuit_data = overtake_data[overtake_data['circuitId'] == circuit_id].copy()
        print(f"\nData for circuit {circuit_id}: {len(circuit_data)} rows")
        
        if not circuit_data.empty:
            print(f"Years available: {sorted(circuit_data['year'].unique())}")
            print(f"Unique constructors: {circuit_data['constructorId'].nunique()}")
            
            # Check for teammate pairs
            print("\nChecking for teammate battles...")
            year = circuit_data['year'].max()
            year_data = circuit_data[circuit_data['year'] == year]
            
            for race_id in year_data['raceId'].unique()[:3]:  # Check first 3 races
                race_data = year_data[year_data['raceId'] == race_id]
                print(f"\n  Race {race_id}:")
                
                for constructor_id in race_data['constructorId'].unique():
                    team_data = race_data[race_data['constructorId'] == constructor_id]
                    if len(team_data) >= 2:
                        print(f"    {constructor_id}: {len(team_data)} drivers")
                        for _, driver in team_data.iterrows():
                            print(f"      - {driver['driverId']}: Grid {driver.get('gridPosition', 'N/A')}, Finish {driver.get('positionNumber', 'N/A')}")