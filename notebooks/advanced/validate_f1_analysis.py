#!/usr/bin/env python
"""
Clean validation script for f1_performance_analysis.py calculations
Focuses on key validations that can be performed with available data
"""

import pandas as pd
import numpy as np
from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def validate_overtakes():
    """Validate overtake calculations"""
    print("\n" + "="*80)
    print("1. VALIDATING OVERTAKE CALCULATIONS")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    
    # Merge for manual calculation
    merged = results.merge(
        grid[['raceId', 'driverId', 'positionNumber']].rename(
            columns={'positionNumber': 'gridPosition'}
        ),
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Filter valid finishes
    merged = merged[merged['positionOrder'] > 0]
    
    # Calculate overtakes manually (negative = gained positions)
    merged['manual_overtakes'] = merged['gridPosition'] - merged['positionOrder']
    
    # Test specific drivers
    test_drivers = {
        'max-verstappen': 'Max Verstappen',
        'lewis-hamilton': 'Lewis Hamilton', 
        'george-russell': 'George Russell'
    }
    
    # Create analyzer
    analyzer = F1PerformanceAnalyzer(data)
    overtake_analysis = analyzer.analyze_overtakes()
    
    print("\nManual vs Analyzer Comparison:")
    print("-" * 60)
    
    for driver_id, driver_name in test_drivers.items():
        # Manual calculation
        driver_data = merged[merged['driverId'] == driver_id]
        if not driver_data.empty:
            manual_total = driver_data['manual_overtakes'].sum()
            manual_avg = driver_data['manual_overtakes'].mean()
            races = len(driver_data)
            
            print(f"\n{driver_name} ({driver_id}):")
            print(f"  Manual: {manual_total} total overtakes in {races} races (avg: {manual_avg:.2f})")
            
            # Check analyzer data
            if driver_id in overtake_analysis.index:
                analyzer_data = overtake_analysis.loc[driver_id]
                print(f"  Analyzer: {analyzer_data.get('total_overtakes', 'N/A')} total overtakes")
                print(f"  Match: {'✓' if abs(manual_total - analyzer_data.get('total_overtakes', 0)) <= 1 else '✗'}")

def validate_points():
    """Validate points calculations"""
    print("\n" + "="*80)
    print("2. VALIDATING POINTS CALCULATIONS")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    
    # Points mapping
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    # Calculate points manually
    results['manual_points'] = results['positionOrder'].map(points_map).fillna(0)
    
    # Get recent results only (2024-2025)
    recent_results = results[results['year'].isin([2024, 2025])]
    
    # Test specific drivers
    test_drivers = {
        'max-verstappen': 'Max Verstappen',
        'lando-norris': 'Lando Norris'
    }
    
    # Create analyzer
    analyzer = F1PerformanceAnalyzer(data)
    points_analysis = analyzer.analyze_points()
    
    print("\nManual vs Analyzer Comparison (2024-2025 only):")
    print("-" * 60)
    
    for driver_id, driver_name in test_drivers.items():
        driver_results = recent_results[recent_results['driverId'] == driver_id]
        if not driver_results.empty:
            manual_total = driver_results['manual_points'].sum()
            races = driver_results['raceId'].nunique()
            
            print(f"\n{driver_name} ({driver_id}):")
            print(f"  Manual: {manual_total} points in {races} races")
            
            if driver_id in points_analysis.index:
                analyzer_data = points_analysis.loc[driver_id]
                print(f"  Analyzer: {analyzer_data.get('total_points', 'N/A')} points")
                # Allow some difference due to sprint races and fastest lap points
                diff = abs(manual_total - analyzer_data.get('total_points', 0))
                print(f"  Difference: {diff} points {'✓ (acceptable)' if diff <= 20 else '✗ (too large)'}")

def validate_year_filter():
    """Validate that only 2024-2025 drivers are included"""
    print("\n" + "="*80)
    print("3. VALIDATING YEAR FILTER (2024-2025)")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    results = data.get('results', pd.DataFrame()).copy()
    
    # Get drivers from 2024-2025
    recent_drivers = results[results['year'].isin([2024, 2025])]['driverId'].unique()
    all_drivers = results['driverId'].unique()
    
    print(f"\nTotal drivers in dataset: {len(all_drivers)}")
    print(f"Drivers who raced in 2024-2025: {len(recent_drivers)}")
    
    # Create analyzer and check its filter
    analyzer = F1PerformanceAnalyzer(data)
    overtakes = analyzer.analyze_overtakes()
    
    if not overtakes.empty:
        analyzer_drivers = overtakes.index.unique()
        print(f"Drivers in analyzer output: {len(analyzer_drivers)}")
        
        # Check overlap
        overlap = set(recent_drivers) & set(analyzer_drivers)
        only_in_analyzer = set(analyzer_drivers) - set(recent_drivers)
        
        print(f"\nValidation:")
        print(f"  Overlap: {len(overlap)} drivers")
        print(f"  In analyzer but not 2024-2025: {len(only_in_analyzer)} drivers")
        
        if len(only_in_analyzer) == 0:
            print("  ✓ Year filter is working correctly")
        else:
            print("  ✗ Year filter may have issues")
            print(f"  Extra drivers: {list(only_in_analyzer)[:5]}...")

def validate_teammate_overtakes():
    """Validate teammate overtake and PrizePicks calculations"""
    print("\n" + "="*80)
    print("4. VALIDATING TEAMMATE OVERTAKES & PRIZEPICKS")
    print("="*80)
    
    # Load data
    data = load_f1db_data()
    analyzer = F1PerformanceAnalyzer(data)
    
    # Get teammate analysis
    teammate_analysis = analyzer.analyze_teammate_overtakes()
    
    if not teammate_analysis.empty:
        print("\nTeammate Overtake Analysis Sample:")
        print("-" * 60)
        
        # Show first few rows
        sample = teammate_analysis.head(5)
        for idx, row in sample.iterrows():
            driver_id = idx if isinstance(idx, str) else row.get('driverId', idx)
            net_ot = row.get('net_OT', 0)
            pp_pts = row.get('prizepicks_pts', 0)
            
            # Validate PrizePicks calculation
            expected_pp = net_ot * 1.5
            
            print(f"\nDriver: {driver_id}")
            print(f"  Net overtakes: {net_ot}")
            print(f"  PrizePicks points: {pp_pts}")
            print(f"  Expected PP points: {expected_pp}")
            print(f"  Calculation: {'✓' if abs(pp_pts - expected_pp) < 0.01 else '✗'}")

def main():
    """Run all validations"""
    print("F1 PERFORMANCE ANALYSIS VALIDATION")
    print("="*80)
    
    try:
        validate_overtakes()
    except Exception as e:
        print(f"\n✗ Overtake validation failed: {e}")
    
    try:
        validate_points()
    except Exception as e:
        print(f"\n✗ Points validation failed: {e}")
    
    try:
        validate_year_filter()
    except Exception as e:
        print(f"\n✗ Year filter validation failed: {e}")
    
    try:
        validate_teammate_overtakes()
    except Exception as e:
        print(f"\n✗ Teammate overtake validation failed: {e}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()