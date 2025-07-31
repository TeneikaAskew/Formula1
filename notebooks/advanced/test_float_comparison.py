#!/usr/bin/env python3
"""Test float comparison in teammate overtakes"""

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer
import pandas as pd
import numpy as np

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Run the full analysis
analyzer = F1PerformanceAnalyzer(data)
result = analyzer.analyze_teammate_overtakes_by_track_year()

print(f"\nResult type: {type(result)}")
print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

if isinstance(result, dict) and 'year_by_year' in result:
    print(f"year_by_year shape: {result['year_by_year'].shape}")
    print(f"year_by_year data:\n{result['year_by_year'].head()}")
else:
    print("No year_by_year data")
    
    # Let's manually check a specific race
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    races = data.get('races', pd.DataFrame())
    
    # Get next race
    next_race = analyzer.get_next_race()
    if next_race is not None:
        circuit_id = next_race.get('circuitId', 'hungaroring')
        
        # Merge data
        overtake_data = results.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Add circuit info
        overtake_data = overtake_data.merge(
            races[['id', 'circuitId']], 
            left_on='raceId', 
            right_on='id', 
            how='left'
        )
        
        # Filter for circuit
        circuit_data = overtake_data[overtake_data['circuitId'] == circuit_id]
        
        print(f"\nCircuit data for {circuit_id}: {len(circuit_data)} rows")
        
        # Check 2024 data
        if 'year' in circuit_data.columns:
            year_2024 = circuit_data[circuit_data['year'] == 2024]
            print(f"2024 data: {len(year_2024)} rows")
            
            # Check for teams with 2 drivers
            for constructor_id in year_2024['constructorId'].unique()[:3]:
                team_data = year_2024[year_2024['constructorId'] == constructor_id]
                print(f"\n{constructor_id}: {len(team_data)} entries")
                if len(team_data) >= 2:
                    sample = team_data[['driverId', 'gridPosition', 'positionNumber']].head(2)
                    print(sample)
                    print(f"Grid dtypes: {sample['gridPosition'].dtype}")
                    print(f"Position dtypes: {sample['positionNumber'].dtype}")
                    
                    # Check for NaN
                    print(f"Grid NaN count: {sample['gridPosition'].isna().sum()}")
                    print(f"Position NaN count: {sample['positionNumber'].isna().sum()}")