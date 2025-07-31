#!/usr/bin/env python3
"""Final debug for teammate overtakes"""

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer
import pandas as pd

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Get the data
results = data.get('results', pd.DataFrame()).copy()
grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
races = data.get('races', pd.DataFrame())

# Check what columns exist in results
print("\nResults columns:")
print(list(results.columns))

# Check grid columns
print("\nGrid columns:")
print(list(grid.columns))

# Check if positionNumber exists
print(f"\n'positionNumber' in results: {'positionNumber' in results.columns}")
print(f"'positionNumber' in grid: {'positionNumber' in grid.columns}")

# Check for alternative column names
print("\nColumns with 'position' in results:")
print([col for col in results.columns if 'position' in col.lower()])

print("\nColumns with 'position' in grid:")
print([col for col in grid.columns if 'position' in col.lower()])

# Sample data
print("\nSample results data:")
print(results[['raceId', 'driverId', 'positionNumber']].head())

print("\nSample grid data:")
print(grid[['raceId', 'driverId', 'positionNumber']].head())