#!/usr/bin/env python3
"""Test script to verify F1 performance analysis works on Windows"""

import os
import sys
from pathlib import Path

# Add notebooks/advanced to path
sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "advanced"))

# Change to notebooks/advanced directory
os.chdir(Path(__file__).parent / "notebooks" / "advanced")

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:2]}")

# Import and test
try:
    from f1_performance_analysis import test_analyzer
    
    print("\nRunning F1 Performance Analysis...")
    analyzer, tables = test_analyzer()
    
    print("\n✓ Analysis completed successfully!")
    print(f"✓ Loaded {len(analyzer.data)} data tables")
    print(f"✓ Generated {len(tables)} analysis tables")
    
    # Check if fantasy data was loaded
    if analyzer.fantasy_data:
        print(f"✓ Fantasy data loaded: {list(analyzer.fantasy_data.keys())}")
    else:
        print("! No fantasy data loaded")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()