#!/usr/bin/env python3
"""Test race info"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get race info
next_race = analyzer.get_next_race()
if next_race is not None:
    print("Next race:", next_race.get('name', 'Unknown'))
    print("Next race ID:", next_race.get('id', 'Unknown'))
else:
    print("Next race: None")
    print("Next race ID: None")

# Find the most recent race with results
races = data.get('races', {})
results = data.get('results', data.get('races_race_results', {}))

# Sort races by date to find the most recent one with results
recent_races = []
for _, race in races.iterrows():
    race_id = race['id']
    race_results = results[results['raceId'] == race_id] if 'raceId' in results.columns else results[results['race_id'] == race_id]
    if not race_results.empty:
        recent_races.append((race['date'], race['id'], race['name']))

recent_races.sort(reverse=True)
print(f"\\nMost recent races with results (top 5):")
for i, (date, race_id, name) in enumerate(recent_races[:5]):
    print(f"{i+1}. {date} - {name} (ID: {race_id})")
    
# Check if overtake data exists for the most recent race
if recent_races:
    last_race_id = recent_races[0][1]
    grid = data.get('races_starting_grid_positions', {})
    
    # Check for overtakes in results
    race_results = results[results['raceId'] == last_race_id] if 'raceId' in results.columns else results[results['race_id'] == last_race_id]
    if not race_results.empty:
        print(f"\\nResults found for race {last_race_id}: {len(race_results)} drivers")
        print("Sample results:")
        print(race_results[['driverId', 'positionNumber', 'points']].head() if 'driverId' in race_results.columns else race_results[['driver_id', 'position_number', 'points']].head())