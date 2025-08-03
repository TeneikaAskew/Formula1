#!/usr/bin/env python3
"""Test positions gained analysis with renamed columns"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Generate the positions gained analysis
print("\\n================================================================================")
print("1. POSITIONS GAINED BY DRIVER")
print("================================================================================")

positions = analyzer.analyze_overtakes()
if not positions.empty:
    print(positions.to_string(index=False))
    
    print("\\nColumn Explanations:")
    print("- avg_pos_gained: Average positions gained per race (grid - finish, positive = gained positions)")
    print("- median_pos_gained: Median positions gained per race")
    print("- last_race: Positions gained in the most recent race") 
    print("- p_circuit_avg: Average positions gained at previous race circuit")
    print("- n_circuit_avg: Average positions gained at next race circuit")
    print("- n_circuit_pts: Position-based points at next circuit (grid - finish + teammate adjustments)")
    print("\\nNote: This shows NET positions gained, not actual on-track overtakes")
    
    # Show sample data
    print(f"\\nColumns in data: {positions.columns.tolist()}")
else:
    print("No positions data available")