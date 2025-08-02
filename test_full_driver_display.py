#!/usr/bin/env python3
"""
Test that analysis shows all drivers, not just 20
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1_performance_analysis import F1PerformanceAnalyzer
from f1db_data_loader import load_f1db_data

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Test a specific analysis method
print(f"\nTesting overtakes analysis for {analyzer.current_year}...")
overtakes = analyzer.analyze_overtakes()

if not overtakes.empty:
    print(f"Total drivers in overtakes analysis: {len(overtakes)}")
    
    # Check if all expected drivers are present
    active_drivers = analyzer.get_active_drivers()
    print(f"Total active drivers: {len(active_drivers)}")
    
    # List all driver IDs in the results
    print("\nAll drivers in overtakes analysis:")
    for i, driver_id in enumerate(overtakes.index):
        print(f"{i+1:2d}. {driver_id}")
else:
    print("No data in overtakes analysis")

# Test points analysis
print(f"\n\nTesting points analysis for {analyzer.current_year}...")
points = analyzer.analyze_points()

if not points.empty:
    print(f"Total drivers in points analysis: {len(points)}")
    print("\nAll drivers in points analysis:")
    for i, driver_id in enumerate(points.index):
        print(f"{i+1:2d}. {driver_id}")