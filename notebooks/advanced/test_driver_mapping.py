"""Test driver mapping to ensure correct Hamilton identification"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Test each analysis method
print("\n=== TESTING DRIVER MAPPING ===")

# 1. Overtakes
print("\n1. Checking Overtakes Analysis:")
overtakes = analyzer.analyze_overtakes()
if not overtakes.empty:
    # Check if lewis-hamilton is in the index
    if 'lewis-hamilton' in overtakes.index:
        print("✓ lewis-hamilton found in overtakes analysis")
        print(f"  Data: {overtakes.loc['lewis-hamilton'][['total_overtakes', 'avg_overtakes', 'races']]}")
    else:
        print("✗ lewis-hamilton NOT found in overtakes analysis")
    
    # Check for duncan-hamilton
    if 'duncan-hamilton' in overtakes.index:
        print("✗ ERROR: duncan-hamilton found in overtakes analysis (should not be there for 2025)")

# 2. Points
print("\n2. Checking Points Analysis:")
points = analyzer.analyze_points()
if not points.empty:
    if 'lewis-hamilton' in points.index:
        print("✓ lewis-hamilton found in points analysis")
        print(f"  Data: {points.loc['lewis-hamilton'][['total_points', 'avg_points', 'races']]}")
    else:
        print("✗ lewis-hamilton NOT found in points analysis")

# 3. Pit Stops
print("\n3. Checking Pit Stops Analysis:")
pit_stops = analyzer.analyze_pit_stops()
if not pit_stops.empty:
    if 'lewis-hamilton' in pit_stops.index:
        print("✓ lewis-hamilton found in pit stops analysis")
        print(f"  Data: {pit_stops.loc['lewis-hamilton'][['avg_stop_time', 'median_stop_time', 'total_stops']]}")
    else:
        print("✗ lewis-hamilton NOT found in pit stops analysis")

# 4. Starting Positions
print("\n4. Checking Starting Positions Analysis:")
grid = analyzer.analyze_starting_positions()
if not grid.empty:
    if 'lewis-hamilton' in grid.index:
        print("✓ lewis-hamilton found in starting positions analysis")
        print(f"  Data: {grid.loc['lewis-hamilton'][['avg_start_position', 'median_start_position', 'best_start_position']]}")
    else:
        print("✗ lewis-hamilton NOT found in starting positions analysis")

# 5. Sprint Points
print("\n5. Checking Sprint Points Analysis:")
sprint = analyzer.analyze_sprint_points()
if not sprint.empty:
    if 'lewis-hamilton' in sprint.index:
        print("✓ lewis-hamilton found in sprint points analysis")
        print(f"  Data: {sprint.loc['lewis-hamilton'][['total_sprint_points', 'avg_sprint_points', 'sprint_races']]}")
    else:
        print("✗ lewis-hamilton NOT found in sprint points analysis")

# Check display formatting
print("\n\n=== TESTING DISPLAY FORMATTING ===")
print("\nChecking format_for_display method:")
if not overtakes.empty:
    display_df = analyzer.format_for_display(overtakes)
    print(f"Original index type: {overtakes.index.name}")
    print(f"Display index type: {display_df.index.name}")
    
    # Check if Hamilton shows correctly
    hamilton_entries = display_df[display_df.index == 'Hamilton']
    if not hamilton_entries.empty:
        print(f"✓ Found {len(hamilton_entries)} Hamilton entries in display")
        print("Display sample:")
        print(hamilton_entries[['total_overtakes', 'avg_overtakes']].head())
    else:
        print("✗ No Hamilton found in display format")

print("\n=== TEST COMPLETE ===")