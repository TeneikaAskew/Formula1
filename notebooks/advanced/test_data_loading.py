#!/usr/bin/env python3
"""
Test data loading to understand actual data structure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from f1db_data_loader import load_f1db_data, F1DBDataLoader

# Try to load data
print("Testing F1DB data loading...")
print("=" * 60)

# Load data
data = load_f1db_data()

if data:
    print(f"\nLoaded {len(data)} datasets:")
    for name, df in data.items():
        if hasattr(df, 'shape'):
            print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            if hasattr(df, 'columns'):
                print(f"    Columns: {', '.join(df.columns[:5])}{', ...' if len(df.columns) > 5 else ''}")
        else:
            print(f"  - {name}: {type(df)}")
else:
    print("\nNo data loaded. Checking for existing data files...")
    
    # Check what files exist
    import os
    
    # Look for CSV files in various locations
    possible_paths = [
        Path("../../data/f1db"),
        Path("../data/f1db"),
        Path("data/f1db"),
        Path("./data/f1db"),
        Path("/workspace/data/f1db")
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"\nChecking {path}:")
            # List CSV files
            csv_files = list(path.glob("*.csv"))
            if csv_files:
                print(f"  Found {len(csv_files)} CSV files:")
                for f in csv_files[:10]:  # Show first 10
                    print(f"    - {f.name}")
            else:
                print("  No CSV files found")
                # Check subdirectories
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"  Subdirectories: {', '.join(d.name for d in subdirs)}")

# Try the data loader directly
print("\n" + "=" * 60)
print("Testing F1DBDataLoader directly...")

loader = F1DBDataLoader()
print(f"Data directory: {loader.data_dir}")

# Check if data needs to be downloaded
data_path = loader.data_dir / "f1db-csv"
if not data_path.exists():
    print("\nF1DB data not found locally. Would need to download.")
    print("Run: loader.download_latest_data() to fetch data")
else:
    print(f"\nData found at: {data_path}")
    csv_files = list(data_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    if csv_files:
        # Try loading a sample file
        import pandas as pd
        sample_file = csv_files[0]
        print(f"\nSample file: {sample_file.name}")
        df = pd.read_csv(sample_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns[:5])}{', ...' if len(df.columns) > 5 else ''}")