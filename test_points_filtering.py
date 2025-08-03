#!/usr/bin/env python3
"""Test points filtering"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get just the points analysis
print("\\nPoints Analysis:")
print("=" * 80)

points = analyzer.analyze_points()
print(f"\\nTotal drivers after filtering: {len(points)}")
print(f"\\nSample data:")
print(points.to_string(index=False))