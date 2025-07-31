#!/usr/bin/env python3
"""Debug track-specific data issues"""

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer
from datetime import datetime
import pandas as pd

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Check next race
print("\n1. Next Race Info:")
next_race = analyzer.get_next_race()
if next_race is not None:
    print(f"  Next race: {next_race.get('officialName', 'Unknown')}")
    print(f"  Circuit ID: {next_race.get('circuitId', 'Missing')}")
    print(f"  Date: {next_race.get('date', 'Missing')}")
    print(f"  All columns: {list(next_race.index)}")
else:
    print("  No next race found!")
    
# Check fastest laps data
print("\n2. Fastest Laps Data:")
fastest_laps = data.get('fastest_laps', pd.DataFrame())
print(f"  Shape: {fastest_laps.shape}")
print(f"  Columns: {list(fastest_laps.columns)}")
if not fastest_laps.empty:
    print(f"  Sample race IDs: {fastest_laps['raceId'].unique()[:5]}")

# Check races data
print("\n3. Races Data:")
races = data.get('races', pd.DataFrame())
print(f"  Shape: {races.shape}")
print(f"  Columns: {list(races.columns)}")
if not races.empty:
    print(f"  Years available: {sorted(races['year'].unique())[-10:]}")
    print(f"  Sample circuit IDs: {races['circuitId'].unique()[:10]}")
    
# Try merging
print("\n4. Testing merge:")
if not fastest_laps.empty and not races.empty:
    merged = fastest_laps.merge(
        races[['id', 'year', 'circuitId']], 
        left_on='raceId', 
        right_on='id', 
        how='left'
    )
    print(f"  Merged shape: {merged.shape}")
    print(f"  Circuit IDs in merged data: {merged['circuitId'].unique()[:10]}")
    
    # Check if next_race circuit exists in data
    if next_race is not None and 'circuitId' in next_race:
        circuit_id = next_race['circuitId']
        filtered = merged[merged['circuitId'] == circuit_id]
        print(f"\n  Data for next race circuit (ID={circuit_id}):")
        print(f"    Rows found: {len(filtered)}")
        if not filtered.empty:
            if 'year_x' in filtered.columns:
                print(f"    Years with data: {sorted(filtered['year_x'].unique())}")
            elif 'year' in filtered.columns:
                print(f"    Years with data: {sorted(filtered['year'].unique())}")
            else:
                print(f"    Available columns: {list(filtered.columns)}")

# Check upcoming races
print("\n5. Upcoming Races:")
if not races.empty:
    races_copy = races.copy()
    races_copy['date'] = pd.to_datetime(races_copy['date'])
    upcoming = races_copy[races_copy['date'] > datetime.now()].sort_values('date')
    if not upcoming.empty:
        print(f"  Found {len(upcoming)} upcoming races")
        for i, (idx, race) in enumerate(upcoming.head(5).iterrows()):
            print(f"  {i+1}. {race.get('officialName', race.get('name', 'Unknown'))} - {race['date'].strftime('%Y-%m-%d')} - Circuit ID: {race.get('circuitId', 'Missing')}")
    else:
        print("  No upcoming races found, checking recent races...")
        recent = races_copy.sort_values('date', ascending=False).head(5)
        for i, (idx, race) in enumerate(recent.iterrows()):
            print(f"  {i+1}. {race.get('officialName', race.get('name', 'Unknown'))} - {race['date'].strftime('%Y-%m-%d')} - Circuit ID: {race.get('circuitId', 'Missing')}")