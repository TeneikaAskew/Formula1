#!/usr/bin/env python
"""
Script to demonstrate the fixes needed for the F1PerformanceAnalyzer
"""

import pandas as pd
import numpy as np
from f1db_data_loader import load_f1db_data

def demonstrate_overtake_issue():
    """Show the overtake calculation issues and proposed fix"""
    print("\n" + "="*80)
    print("OVERTAKE CALCULATION ISSUE ANALYSIS")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    
    # Take a sample race with DNFs
    race_id = 1104  # Australian GP 2024 (has DNFs)
    race_results = results[results['raceId'] == race_id]
    race_grid = grid[grid['raceId'] == race_id]
    
    # Merge data
    merged = race_results.merge(
        race_grid[['driverId', 'positionNumber']].rename(
            columns={'positionNumber': 'gridPosition'}
        ),
        on='driverId',
        how='left'
    )
    
    print(f"\nRace ID {race_id} - Showing DNF/DNS issues:")
    print("-" * 80)
    print("Driver | Grid | Pos# | PosOrder | Status | Calc (Pos#) | Calc (Order)")
    print("-" * 80)
    
    # Show all drivers to see the issue
    for _, row in merged.sort_values('positionOrder').iterrows():
        driver = row['driverId'][:15]
        grid = row['gridPosition']
        pos_num = row['positionNumber']
        pos_order = row['positionOrder']
        status = row.get('statusId', 'Unknown')
        
        # Calculate overtakes both ways
        calc_num = grid - pos_num if pd.notna(grid) and pd.notna(pos_num) else np.nan
        calc_order = grid - pos_order if pd.notna(grid) and pos_order > 0 else np.nan
        
        print(f"{driver:15} | {grid:4.0f} | {pos_num if pd.notna(pos_num) else 'NaN':4} | {pos_order:8.0f} | {status:6} | "
              f"{calc_num if pd.notna(calc_num) else 'NaN':11} | {calc_order if pd.notna(calc_order) else 'NaN':12}")
    
    print("\n\nISSUE SUMMARY:")
    print("-" * 80)
    print("1. The analyzer uses 'positionNumber' which includes DNF/DNS drivers")
    print("2. DNF drivers have positionNumber but positionOrder = 0")
    print("3. This causes incorrect overtake calculations for DNF drivers")
    print("4. The fix: Use positionOrder and filter where positionOrder > 0")

def demonstrate_points_issue():
    """Show the points calculation issues"""
    print("\n\n" + "="*80)
    print("POINTS CALCULATION ISSUE ANALYSIS")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    
    # Check for NaN points
    nan_points = results[results['points'].isna()]
    print(f"\nRows with NaN points: {len(nan_points)}")
    
    if not nan_points.empty:
        print("\nSample of NaN points rows:")
        print(nan_points[['raceId', 'driverId', 'positionOrder', 'points']].head())
    
    # Check sprint races
    sprint_results = data.get('sprint_results', pd.DataFrame())
    print(f"\n\nSprint race results: {len(sprint_results)} rows")
    
    print("\n\nISSUE SUMMARY:")
    print("-" * 80)
    print("1. Some rows have NaN points values")
    print("2. Sprint race points are tracked separately")
    print("3. The analyzer doesn't handle NaN points properly in aggregations")

def proposed_fixes():
    """Show the proposed fixes"""
    print("\n\n" + "="*80)
    print("PROPOSED FIXES FOR F1PerformanceAnalyzer")
    print("="*80)
    
    print("""
1. OVERTAKE CALCULATION FIX:
   
   Current code (line 161):
   overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
   
   Fixed code:
   # Filter valid finishes only (positionOrder > 0)
   overtake_data = overtake_data[overtake_data['positionOrder'] > 0]
   # Use positionOrder instead of positionNumber
   overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionOrder']

2. POINTS CALCULATION FIX:
   
   In analyze_points() method:
   # Handle NaN values in points
   results['points'] = results['points'].fillna(0)
   
3. ADDITIONAL RECOMMENDED FIXES:
   
   a) Add validation for grid positions:
      # Some drivers may not have grid positions (pit lane starts, DNS)
      overtake_data = overtake_data[overtake_data['gridPosition'].notna()]
   
   b) Separate DNF analysis:
      # Create separate method for DNF analysis
      dnf_results = results[results['positionOrder'] == 0]
      
   c) Better status handling:
      # Map statusId to meaningful categories
      status_map = {1: 'Finished', 2: 'DNF', 3: 'DNS', ...}
""")

def main():
    """Run the analysis"""
    demonstrate_overtake_issue()
    demonstrate_points_issue()
    proposed_fixes()

if __name__ == "__main__":
    main()