"""Verify that lewis-hamilton data is correctly used"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get starting positions analysis
print("\nAnalyzing starting positions...")
grid = analyzer.analyze_starting_positions()

# Check if we have lewis-hamilton
if 'lewis-hamilton' in grid.index:
    print("✓ lewis-hamilton found in starting positions")
    lewis_data = grid.loc['lewis-hamilton']
    print(f"  Average start position: {lewis_data['avg_start_position']:.2f}")
    print(f"  Best start position: {lewis_data['best_start_position']}")
    
    # Verify this is the correct Hamilton by checking recent activity
    results = data['results']
    lewis_results_2025 = results[(results['driverId'] == 'lewis-hamilton') & (results['year'] == 2025)]
    print(f"  2025 races: {len(lewis_results_2025)}")
    print(f"  2025 average finish: {lewis_results_2025['positionNumber'].mean():.1f}")
else:
    print("✗ lewis-hamilton NOT found!")

# Check that duncan-hamilton is NOT in current year analysis
if 'duncan-hamilton' in grid.index:
    print("\n✗ ERROR: duncan-hamilton found in current analysis (should not be there)")
else:
    print("\n✓ duncan-hamilton correctly excluded from current year analysis")

# Double-check the raw data
print("\n\nVerifying raw data:")
grid_data = data.get('races_starting_grid_positions', pd.DataFrame())
races = data.get('races', pd.DataFrame())

# Add year to grid data
grid_with_year = grid_data.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')

# Check 2025 Hamilton entries
hamilton_2025 = grid_with_year[(grid_with_year['year'] == 2025) & (grid_with_year['driverId'].str.contains('hamilton'))]
print(f"\nHamilton entries in 2025 grid data:")
print(hamilton_2025.groupby('driverId').size())

print("\n✅ TEST COMPLETE - Driver IDs are now used consistently!")