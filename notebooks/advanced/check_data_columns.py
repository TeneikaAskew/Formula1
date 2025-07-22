#!/usr/bin/env python3
"""Check what columns exist in the F1 data files"""

import pandas as pd
from pathlib import Path

data_dir = Path('../../data/f1db')

print("Checking F1 data columns...")
print("=" * 60)

# Check races columns
races_file = data_dir / 'races.csv'
if races_file.exists():
    races = pd.read_csv(races_file, nrows=5)
    print("\nRACES columns:")
    print(races.columns.tolist())
    print("\nSample data:")
    print(races.head(2))
else:
    print("races.csv not found!")

# Check results columns
results_file = data_dir / 'results.csv'
if results_file.exists():
    results = pd.read_csv(results_file, nrows=5)
    print("\n\nRESULTS columns:")
    print(results.columns.tolist())
else:
    print("results.csv not found!")

# Check drivers columns
drivers_file = data_dir / 'drivers.csv'
if drivers_file.exists():
    drivers = pd.read_csv(drivers_file, nrows=5)
    print("\n\nDRIVERS columns:")
    print(drivers.columns.tolist())
else:
    print("drivers.csv not found!")

# Check constructors columns
constructors_file = data_dir / 'constructors.csv'
if constructors_file.exists():
    constructors = pd.read_csv(constructors_file, nrows=5)
    print("\n\nCONSTRUCTORS columns:")
    print(constructors.columns.tolist())
else:
    print("constructors.csv not found!")

print("\n" + "=" * 60)