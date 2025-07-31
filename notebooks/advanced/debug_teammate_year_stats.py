#!/usr/bin/env python3
"""Debug teammate overtakes year_stats collection"""

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
    circuit_id = next_race.get('circuitId', 'hungaroring')
    print(f"Circuit ID: {circuit_id}")

# Manually run part of the analysis
results = data.get('results', pd.DataFrame()).copy()
grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
races = data.get('races', pd.DataFrame())

# Merge all necessary data
overtake_data = results.merge(
    grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
    on=['raceId', 'driverId'],
    how='left'
)

# Add year and circuit info
if 'year' not in overtake_data.columns or 'circuitId' not in overtake_data.columns:
    merge_cols = ['id']
    if 'year' not in overtake_data.columns:
        merge_cols.append('year')
    if 'circuitId' not in overtake_data.columns:
        merge_cols.append('circuitId')
    overtake_data = overtake_data.merge(
        races[merge_cols], 
        left_on='raceId', 
        right_on='id', 
        how='left'
    )

# Filter for specific circuit
circuit_data = overtake_data[overtake_data['circuitId'] == circuit_id].copy()

print(f"\nCircuit data shape: {circuit_data.shape}")
print(f"Years available: {sorted(circuit_data['year'].unique())}")

# Drop rows with missing year
circuit_data = circuit_data.dropna(subset=['year'])

# Calculate teammate battles by year
year_stats = []
year_count = 0

for year in sorted(circuit_data['year'].unique())[-5:]:  # Last 5 years
    year_data = circuit_data[circuit_data['year'] == year]
    print(f"\nYear {year}: {len(year_data)} entries")
    
    # Process each race in that year
    for race_id in year_data['raceId'].unique():
        race_data = year_data[year_data['raceId'] == race_id]
        
        # Group by constructor
        for constructor_id in race_data['constructorId'].unique():
            team_data = race_data[race_data['constructorId'] == constructor_id]
            
            if len(team_data) >= 2:
                drivers = team_data[['driverId', 'gridPosition', 'positionNumber']].values
                
                for i in range(len(drivers)):
                    for j in range(i+1, len(drivers)):
                        driver1_id, driver1_grid, driver1_finish = drivers[i]
                        driver2_id, driver2_grid, driver2_finish = drivers[j]
                        
                        if pd.isna(driver1_grid) or pd.isna(driver1_finish) or pd.isna(driver2_grid) or pd.isna(driver2_finish):
                            continue
                        
                        # Determine winner and if it was an overtake
                        if driver1_finish < driver2_finish:
                            winner = driver1_id
                            loser = driver2_id
                            was_overtake = driver1_grid > driver2_grid
                        else:
                            winner = driver2_id
                            loser = driver1_id
                            was_overtake = driver2_grid > driver1_grid
                        
                        # Add stats for both drivers
                        year_stats.append({
                            'driverId': winner,
                            'year': year,
                            'teammate_win': 1,
                            'teammate_overtake': 1 if was_overtake else 0
                        })
                        
                        year_stats.append({
                            'driverId': loser,
                            'year': year,
                            'teammate_win': 0,
                            'teammate_overtake': -1 if was_overtake else 0
                        })
                        
                        year_count += 1

print(f"\n\nTotal year_stats entries: {len(year_stats)}")
print(f"Total teammate battles found: {year_count}")

if year_stats:
    print("\nSample entries:")
    for stat in year_stats[:10]:
        print(f"  {stat}")