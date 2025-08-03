#!/usr/bin/env python3
"""Test updated points analysis"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Generate points analysis
print("\\n================================================================================")
print("2. F1 POINTS BY DRIVER")
print("================================================================================")
points = analyzer.analyze_points()
if not points.empty:
    print(points.to_string(index=False))
    
    # Add explanations
    print("\\nColumn Definitions:")
    print("- avg_points: Average points per race in current season (2025)")
    print("- hist_avg_points: Historical average points per race (2022-2024)")
    print("- last_race: Points scored in the most recent race")
    print("- p_circuit_avg: Average points at the previous race's circuit (3-year history)")
    print("- c_circuit_avg: Average points at the current race's circuit (3-year history)")  
    print("- n_circuit_avg: Average points at the next race's circuit (3-year history)")
    print("\\nNote: Circuit averages are calculated from the last 3 years of racing at each specific track")
    
    # Debug info
    print(f"\\nDebug: Total drivers: {len(points)}")
    print(f"Columns: {points.columns.tolist()}")
else:
    print("No points data available")