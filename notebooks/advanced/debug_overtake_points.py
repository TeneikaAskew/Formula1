#!/usr/bin/env python
"""
Debug script to trace overtake and points calculation discrepancies
"""

import pandas as pd
import numpy as np
from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def debug_overtake_calculation():
    """Debug overtake calculations with specific race examples"""
    print("\n" + "="*80)
    print("OVERTAKE CALCULATION DEBUG")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    races = data.get('races', pd.DataFrame())
    
    # Get a recent race for detailed analysis
    recent_races = races[races['year'] == 2024].head(3)
    
    for _, race in recent_races.iterrows():
        race_id = race['id']
        print(f"\n\nRace: {race['name']} ({race['year']})")
        print("-" * 60)
        
        # Get results and grid for this race
        race_results = results[results['raceId'] == race_id].copy()
        race_grid = grid[grid['raceId'] == race_id].copy()
        
        # Merge data
        merged = race_results.merge(
            race_grid[['driverId', 'positionNumber']].rename(
                columns={'positionNumber': 'gridPosition'}
            ),
            on='driverId',
            how='left'
        )
        
        # Sort by finish position
        merged = merged.sort_values('positionOrder')
        
        print("\nDriver Performance (Grid -> Finish):")
        print("Driver ID | Grid | Finish (Order) | Finish (Number) | Status | Overtakes (Order) | Overtakes (Number)")
        print("-" * 100)
        
        for _, row in merged.head(10).iterrows():
            driver_id = row['driverId']
            grid_pos = row['gridPosition']
            finish_order = row['positionOrder']
            finish_number = row['positionNumber']
            status = row.get('statusId', 'Unknown')
            
            # Calculate overtakes both ways
            overtakes_order = grid_pos - finish_order if pd.notna(grid_pos) and finish_order > 0 else np.nan
            overtakes_number = grid_pos - finish_number if pd.notna(grid_pos) and pd.notna(finish_number) else np.nan
            
            print(f"{driver_id:15} | {grid_pos:4.0f} | {finish_order:15.0f} | {finish_number:16.0f} | {status:6} | {overtakes_order:18.0f} | {overtakes_number:19.0f}")
        
        # Show DNF/DNS entries
        dnf_dns = merged[merged['positionOrder'] == 0]
        if not dnf_dns.empty:
            print("\nDNF/DNS Drivers:")
            for _, row in dnf_dns.iterrows():
                print(f"{row['driverId']:15} | Grid: {row['gridPosition']:4.0f} | Status: {row.get('statusId', 'Unknown')}")

def debug_points_calculation():
    """Debug points calculations with specific examples"""
    print("\n" + "="*80)
    print("POINTS CALCULATION DEBUG")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    races = data.get('races', pd.DataFrame())
    
    # Standard F1 points system
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    # Get a specific driver's recent races
    driver_id = 'max-verstappen'
    driver_results = results[
        (results['driverId'] == driver_id) & 
        (results['year'] == 2024)
    ].copy()
    
    print(f"\nPoints Debug for {driver_id} (2024):")
    print("-" * 80)
    print("Race ID | Position Order | Position Number | Points (DB) | Points (Calc) | Fastest Lap | Match?")
    print("-" * 100)
    
    total_db_points = 0
    total_calc_points = 0
    
    for _, row in driver_results.iterrows():
        race_id = row['raceId']
        pos_order = row['positionOrder']
        pos_number = row['positionNumber'] 
        db_points = row['points']
        fastest_lap = row.get('fastestLap', False)
        
        # Calculate expected points
        calc_points = points_map.get(pos_order, 0)
        if fastest_lap and pos_order <= 10:
            calc_points += 1
        
        total_db_points += db_points
        total_calc_points += calc_points
        
        match = "✓" if db_points == calc_points else "✗"
        
        print(f"{race_id:8} | {pos_order:15} | {pos_number:16} | {db_points:11} | {calc_points:14} | {fastest_lap:12} | {match}")
    
    print("-" * 100)
    print(f"TOTALS:  |                 |                 | {total_db_points:11} | {total_calc_points:14} |              | {'✓' if total_db_points == total_calc_points else '✗'}")

def compare_analyzer_calculations():
    """Compare analyzer output with manual calculations"""
    print("\n" + "="*80)
    print("ANALYZER VS MANUAL CALCULATION COMPARISON")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    analyzer = F1PerformanceAnalyzer(data)
    
    # Get analyzer results
    overtakes = analyzer.analyze_overtakes()
    points = analyzer.analyze_points()
    
    # Manual calculations
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    
    # Filter for 2024-2025
    recent_results = results[results['year'].isin([2024, 2025])]
    
    # Calculate overtakes manually (with DNF filter)
    merged = recent_results.merge(
        grid[['raceId', 'driverId', 'positionNumber']].rename(
            columns={'positionNumber': 'gridPosition'}
        ),
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Filter valid finishes only
    valid_finishes = merged[merged['positionOrder'] > 0].copy()
    valid_finishes['manual_overtakes'] = valid_finishes['gridPosition'] - valid_finishes['positionOrder']
    
    # Group by driver
    manual_overtakes = valid_finishes.groupby('driverId').agg({
        'manual_overtakes': ['sum', 'mean', 'count']
    })
    manual_overtakes.columns = ['total_overtakes_manual', 'avg_overtakes_manual', 'races_manual']
    
    # Compare top drivers
    print("\nTop 10 Drivers - Overtake Comparison:")
    print("-" * 80)
    print("Driver ID       | Analyzer Total | Manual Total | Difference | Races (A) | Races (M)")
    print("-" * 80)
    
    for driver_id in overtakes.head(10).index:
        if driver_id in manual_overtakes.index:
            analyzer_total = overtakes.loc[driver_id, 'total_overtakes']
            manual_total = manual_overtakes.loc[driver_id, 'total_overtakes_manual']
            analyzer_races = overtakes.loc[driver_id, 'races']
            manual_races = manual_overtakes.loc[driver_id, 'races_manual']
            diff = analyzer_total - manual_total
            
            print(f"{driver_id:15} | {analyzer_total:14.0f} | {manual_total:12.0f} | {diff:10.0f} | {analyzer_races:9.0f} | {manual_races:9.0f}")

def main():
    """Run all debug checks"""
    print("F1 PERFORMANCE ANALYSIS DEBUG")
    print("="*80)
    
    try:
        debug_overtake_calculation()
    except Exception as e:
        print(f"\n✗ Overtake debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        debug_points_calculation()
    except Exception as e:
        print(f"\n✗ Points debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        compare_analyzer_calculations()
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()