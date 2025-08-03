#!/usr/bin/env python3
"""Test just the overtakes table"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get just the overtakes analysis
print("\\nOvertakes Analysis:")
print("=" * 80)

overtakes = analyzer.analyze_overtakes()
print(f"\\nColumns: {overtakes.columns.tolist()}")
print(f"\\nSample data (first 10 rows):")
print(overtakes.head(10).to_string(index=False))

# Check last race data
print(f"\\nLast race column values (first 10):")
print(overtakes['last_race'].head(10))