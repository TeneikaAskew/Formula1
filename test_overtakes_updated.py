#!/usr/bin/env python3
"""Test the updated overtakes table"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Generate just the overtakes table to see the changes
print("\nGenerating overtakes analysis with updates...")
print("=" * 80)

# Get the race info
races = data.get('races', {})
results = data.get('results', data.get('races_race_results', {}))

# Generate tables (this will show the race info and overtakes)
tables = analyzer.generate_all_tables()

# Also just show the overtakes dataframe structure
print("\n\nOvertakes DataFrame columns:")
overtakes = analyzer.analyze_overtakes()
print(overtakes.columns.tolist())
print(f"\nSample data (first 5 rows):")
print(overtakes.head().to_string(index=False))