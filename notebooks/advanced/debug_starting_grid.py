"""Debug starting grid Hamilton issue"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from f1db_data_loader import load_f1db_data

# Load data
print("Loading F1 data...")
data = load_f1db_data()
grid = data.get('races_starting_grid_positions', pd.DataFrame())
results = data['results']
races = data['races']
drivers = data['drivers']

# Check the starting grid data
print("\nChecking starting grid data structure:")
print(f"Grid columns: {list(grid.columns)}")
print(f"Grid shape: {grid.shape}")

# Add year to grid data
if 'year' not in grid.columns:
    grid = grid.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')

# Check for Hamiltons in 2025 grid
grid_2025 = grid[grid['year'] == 2025]
print(f"\n2025 grid entries: {len(grid_2025)}")

# Find Hamilton entries
hamilton_grids = grid_2025[grid_2025['driverId'].isin(['lewis-hamilton', 'duncan-hamilton'])]
print(f"\nHamilton grid entries in 2025:")
print(hamilton_grids[['raceId', 'driverId', 'positionNumber']].head(10))

# Check all unique drivers in 2025 grid
drivers_in_grid_2025 = grid_2025['driverId'].unique()
print(f"\nTotal unique drivers in 2025 grid: {len(drivers_in_grid_2025)}")

# Find if duncan-hamilton is in 2025 grid
if 'duncan-hamilton' in drivers_in_grid_2025:
    print("\n⚠️ ERROR: duncan-hamilton found in 2025 grid data!")
    duncan_grids = grid_2025[grid_2025['driverId'] == 'duncan-hamilton']
    print(f"Number of duncan-hamilton entries: {len(duncan_grids)}")
    
# Check if there's a driver name collision issue
print("\nChecking for driver name issues in grid data...")
# Get driver info for all 2025 grid drivers
driver_info = drivers[drivers['id'].isin(drivers_in_grid_2025)][['id', 'forename', 'lastName']]
print(f"\nDrivers with lastName 'Hamilton' in 2025 grid:")
hamilton_drivers = driver_info[driver_info['lastName'] == 'Hamilton']
print(hamilton_drivers)