#!/usr/bin/env python3
"""Test script for Windows environment setup"""

import sys
from pathlib import Path
import os

print("Testing F1 Pipeline Setup on Windows...")
print("=" * 60)

# Show current directory
print(f"Current directory: {Path.cwd()}")
print(f"Script location: {Path(__file__).parent}")

# Test 1: Check if we can find the data
possible_data_paths = [
    Path('../../data/f1db'),
    Path('../data/f1db'),
    Path('data/f1db'),
    Path.cwd().parent.parent / 'data' / 'f1db',
    Path.cwd().parent / 'data' / 'f1db',
]

data_dir = None
for p in possible_data_paths:
    if p.exists():
        data_dir = p.resolve()
        print(f"\n✓ Found data directory at: {data_dir}")
        csv_files = list(data_dir.glob('*.csv'))
        print(f"  Contains {len(csv_files)} CSV files")
        break
else:
    print("\n✗ Could not find data/f1db directory")
    print("  Searched in:")
    for p in possible_data_paths:
        print(f"    - {p.resolve()}")

# Test 2: Check Python path
print(f"\nPython path includes:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Test 3: Try importing data loader
print("\nTesting imports:")
try:
    # Add current directory to path if not already there
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from f1db_data_loader import F1DBDataLoader, load_f1db_data
    print("✓ Successfully imported f1db_data_loader")
except ImportError as e:
    print(f"✗ Failed to import f1db_data_loader: {e}")
    print(f"  Looking in: {current_dir}")
    if (current_dir / 'f1db_data_loader.py').exists():
        print("  File exists but can't import - check for syntax errors")
    else:
        print("  File f1db_data_loader.py not found")

# Test 4: List notebooks in current directory
print("\nNotebooks in current directory:")
notebook_files = list(Path.cwd().glob('*.ipynb'))
if notebook_files:
    for nb in notebook_files[:10]:  # Show first 10
        print(f"  - {nb.name}")
else:
    print("  No notebooks found in current directory")

# Test 5: Check for required notebooks
required_notebooks = [
    'F1_Pipeline_Integration.ipynb',
    'F1_Core_Models.ipynb',
    'F1_Feature_Store.ipynb',
    'F1_Integrated_Driver_Evaluation.ipynb',
    'F1_Prize_Picks_Optimizer.ipynb'
]

print("\nRequired notebooks status:")
for nb in required_notebooks:
    if Path(nb).exists():
        print(f"  ✓ {nb}")
    else:
        print(f"  ✗ {nb}")

print("\n" + "=" * 60)
print("Setup diagnostics complete!")

# Provide guidance based on findings
if data_dir is None:
    print("\n⚠️  ACTION NEEDED:")
    print("  Create a data/f1db directory relative to your notebooks")
    print("  The structure should be:")
    print("    your_project/")
    print("      ├── notebooks/")
    print("      │   └── advanced/  (you are here)")
    print("      └── data/")
    print("          └── f1db/  (CSV files go here)")

print("\nTo run the pipeline:")
print("  1. Make sure you're in the notebooks/advanced directory")
print("  2. Open F1_Pipeline_Integration.ipynb in Jupyter")
print("  3. Run all cells")