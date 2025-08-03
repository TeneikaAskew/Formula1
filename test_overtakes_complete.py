#!/usr/bin/env python3
"""Test complete overtakes table with all columns"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get overtakes analysis
print("\\nComplete Overtakes Analysis:")
print("=" * 150)

overtakes = analyzer.analyze_overtakes()
print(f"\\nColumns: {overtakes.columns.tolist()}")

# Show a subset of columns for readability
print("\\nSample data (top 10 drivers by last race points):")
sorted_by_pts = overtakes.sort_values('last_race_pts', ascending=False)
display_cols = ['driver_name', 'last_race', 'last_race_pts', 'p_circuit_avg', 'p_circuit_pts', 'n_circuit_avg', 'n_circuit_pts']
print(sorted_by_pts[display_cols].head(10).to_string(index=False))

print("\\nColumn Explanations:")
print("- last_race: Overtakes in the last race (Belgian GP)")
print("- last_race_pts: Championship points scored in the last race")
print("- p_circuit_avg: Average overtakes at previous circuit (British GP/Silverstone)")
print("- p_circuit_pts: Overtake Points at previous circuit (grid - finish + teammate)")
print("- n_circuit_avg: Average overtakes at next circuit (Hungarian GP)")
print("- n_circuit_pts: Overtake Points at next circuit (grid - finish + teammate)")

# Show some interesting comparisons
print("\\nDrivers who score points but have negative overtake points at Hungary:")
hungary_contrast = sorted_by_pts[(sorted_by_pts['n_circuit_pts'] < 0) & (sorted_by_pts['avg_points'] > 5)]
if not hungary_contrast.empty:
    print(hungary_contrast[['driver_name', 'avg_points', 'n_circuit_avg', 'n_circuit_pts']].head(5).to_string(index=False))