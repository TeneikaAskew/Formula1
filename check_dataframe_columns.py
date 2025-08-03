#!/usr/bin/env python3
"""Check DataFrame columns"""

import sys
sys.path.append('/workspace/notebooks/advanced')

from f1db_data_loader import F1DBDataLoader

# Load data without update check
loader = F1DBDataLoader()
data = loader.load_csv_data(validate=False, check_updates=False)

# Check races columns
races = data.get('races', {})
print("Races columns:", races.columns.tolist())
print("\\nSample race data:")
print(races.head(2))

# Check results columns
results = data.get('results', data.get('races_race_results', {}))
print("\\n\\nResults columns:", results.columns.tolist())

# Check grid columns
grid = data.get('races_starting_grid_positions', {})
print("\\n\\nGrid columns:", grid.columns.tolist() if hasattr(grid, 'columns') else "No grid data")