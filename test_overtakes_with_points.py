#!/usr/bin/env python3
"""Test overtakes table with next circuit points"""

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
print("\\nOvertakes Analysis with Next Circuit Points:")
print("=" * 100)

overtakes = analyzer.analyze_overtakes()
print(f"\\nColumns: {overtakes.columns.tolist()}")
print(f"\\nSample data (first 10 rows):")
print(overtakes.head(10).to_string(index=False))

# Show some specific drivers to see the points
print(f"\\n\\nNext Circuit (Hungarian GP) Expected Points:")
print("-" * 50)
for _, row in overtakes.head(10).iterrows():
    if row['n_circuit_pts'] > 0:
        print(f"{row['driver_name']:20s}: {row['n_circuit_pts']:5.2f} points (avg OT: {row['n_circuit_avg']:4.2f})")