#!/usr/bin/env python3
"""Test overtake points calculation"""

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
print("\\nOvertakes Analysis with Overtake Points:")
print("=" * 100)

overtakes = analyzer.analyze_overtakes()
print(f"\\nSample data (sorted by n_circuit_pts):")
# Sort by n_circuit_pts to see who has the best overtake points
sorted_overtakes = overtakes.sort_values('n_circuit_pts', ascending=False)
print(sorted_overtakes[['driver_name', 'avg_OT', 'n_circuit_avg', 'n_circuit_pts']].head(15).to_string(index=False))

print("\\nColumn Explanation:")
print("- n_circuit_avg: Average overtakes at Hungarian GP (positions gained)")
print("- n_circuit_pts: Overtake Points at Hungarian GP (grid - finish + teammate adjustments)")
print("  * Base: Grid Position - Finish Position")
print("  * +0.5 bonus for overtaking teammate")
print("  * -0.5 penalty for being overtaken by teammate")

# Show some examples of positive and negative overtake points
print("\\nDrivers with highest Overtake Points at Hungarian GP:")
for _, row in sorted_overtakes.head(5).iterrows():
    print(f"{row['driver_name']:20s}: {row['n_circuit_pts']:5.2f} overtake points")

print("\\nDrivers with lowest (most negative) Overtake Points at Hungarian GP:")
for _, row in sorted_overtakes.tail(5).iterrows():
    print(f"{row['driver_name']:20s}: {row['n_circuit_pts']:5.2f} overtake points")