#!/usr/bin/env python3
"""Test the real overtakes table display"""

import sys
import os
from pathlib import Path

# Add notebooks/advanced to path
sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "advanced"))

# Change to notebooks/advanced directory
os.chdir(Path(__file__).parent / "notebooks" / "advanced")

def test_overtakes_table():
    """Test just the overtakes table display"""
    print("Testing overtakes table display...")
    
    from f1db_data_loader import F1DBDataLoader
    from f1_performance_analysis import F1PerformanceAnalyzer
    
    # Load data
    loader = F1DBDataLoader(data_dir="../../data/f1db")
    data_dict = loader.load_csv_data(validate=False, check_updates=False)
    
    # Initialize analyzer
    analyzer = F1PerformanceAnalyzer(data_dict)
    
    # Generate just the overtakes table
    print("\n" + "="*80)
    if analyzer.lap_analyzer is not None:
        print("1. REAL OVERTAKES BY DRIVER (Lap-by-Lap Analysis)")
    else:
        print("1. POSITIONS GAINED BY DRIVER")
    print("="*80)
    
    overtakes = analyzer.analyze_overtakes()
    if not overtakes.empty:
        print(overtakes.to_string(index=False))
        
        print("\nColumn Explanations:")
        if analyzer.lap_analyzer is not None:
            print("- total_OTs: Total overtakes made across all races")
            print("- avg_OTs: Average real overtakes made per race (lap-by-lap analysis)")
            print("- median_OTs: Median overtakes made per race")
            print("- last_race_OTs: Overtakes made in the most recent race")
            print("- c_race_OTs: Historical average overtakes at current race circuit")
            print("- n_race_OTs: Historical average overtakes at next race circuit")
            print("- OTd_by: Total times overtaken by other drivers")
            print("- net_OTs: Net overtakes (overtakes made - times overtaken)")
            print("- max_OTs: Maximum overtakes in a single race")
            print("\nNote: This shows ACTUAL on-track overtakes from lap-by-lap data")

if __name__ == "__main__":
    test_overtakes_table()