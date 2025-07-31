#!/usr/bin/env python3
"""Check driver ID format in different tables"""

from f1db_data_loader import load_f1db_data
import pandas as pd

# Load data
print("Loading F1 data...")
data = load_f1db_data()

# Check driver ID format in different tables
results = data.get('results', pd.DataFrame())
grid = data.get('races_starting_grid_positions', pd.DataFrame())
drivers = data.get('drivers', pd.DataFrame())

print("\nDriver ID formats:")
print(f"Results table - sample driver IDs: {results['driverId'].unique()[:5].tolist()}")
print(f"Grid table - sample driver IDs: {grid['driverId'].unique()[:5].tolist()}")
print(f"Drivers table - sample IDs: {drivers['id'].unique()[:5].tolist()}")

# Check if IDs match
results_ids = set(results['driverId'].unique())
grid_ids = set(grid['driverId'].unique())
driver_ids = set(drivers['id'].unique())

print(f"\nTotal unique driver IDs:")
print(f"Results: {len(results_ids)}")
print(f"Grid: {len(grid_ids)}")
print(f"Drivers: {len(driver_ids)}")

# Check overlap
print(f"\nOverlap between results and grid: {len(results_ids & grid_ids)}")
print(f"In results but not grid: {len(results_ids - grid_ids)}")
print(f"In grid but not results: {len(grid_ids - results_ids)}")

# Check data types
print(f"\nData types:")
print(f"Results driverId type: {results['driverId'].dtype}")
print(f"Grid driverId type: {grid['driverId'].dtype}")
print(f"Drivers id type: {drivers['id'].dtype}")