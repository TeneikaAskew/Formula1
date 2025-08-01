#!/usr/bin/env python
"""
Validation script for f1_performance_analysis.py calculations
This script validates:
1. Overtake calculations
2. Points calculations  
3. Pit stop time calculations
4. Past year filter (2024-2025 drivers)
5. Teammate overtake calculations and PrizePicks scoring
"""

import pandas as pd
import numpy as np
from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer
import datetime

def validate_overtake_calculations(data, analyzer):
    """Manually calculate overtakes for 2-3 drivers and compare"""
    print("\n" + "="*80)
    print("VALIDATING OVERTAKE CALCULATIONS")
    print("="*80)
    
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    
    if results.empty or grid.empty:
        print("ERROR: Missing results or grid data")
        return
    
    # Merge grid positions with results
    merged = results.merge(
        grid[['raceId', 'driverId', 'positionNumber']].rename(
            columns={'positionNumber': 'gridPosition'}
        ),
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Filter for valid finishes (position > 0 means finished)
    merged = merged[merged['positionOrder'] > 0]
    
    # Calculate overtakes manually (negative means gained positions)
    merged['manual_overtakes'] = merged['gridPosition'] - merged['positionOrder']
    
    # Get analyzer's overtake data
    overtakes = analyzer.analyze_overtakes()
    
    # Test for specific drivers
    test_drivers = [830, 844, 815]  # Hamilton, Russell, Verstappen
    driver_names = {830: 'Hamilton', 844: 'Russell', 815: 'Verstappen'}
    
    for driver_id in test_drivers:
        if driver_id not in driver_names:
            continue
            
        driver_data = merged[merged['driverId'] == driver_id]
        
        if not driver_data.empty:
            # Manual calculations
            manual_total = driver_data['manual_overtakes'].sum()
            manual_avg = driver_data['manual_overtakes'].mean()
            manual_median = driver_data['manual_overtakes'].median()
            
            # Get analyzer's calculations
            analyzer_data = overtakes[overtakes['driverId'] == driver_id]
            
            print(f"\n{driver_names[driver_id]} (ID: {driver_id}):")
            print(f"  Manual Calculations:")
            print(f"    Total overtakes: {manual_total}")
            print(f"    Average overtakes: {manual_avg:.2f}")
            print(f"    Median overtakes: {manual_median:.2f}")
            print(f"    Races analyzed: {len(driver_data)}")
            
            if not analyzer_data.empty:
                print(f"  Analyzer Calculations:")
                print(f"    Total overtakes: {analyzer_data['total_overtakes'].iloc[0]}")
                print(f"    Average overtakes: {analyzer_data['avg_overtakes'].iloc[0]:.2f}")
                print(f"    Median overtakes: {analyzer_data['median_overtakes'].iloc[0]:.2f}")
                print(f"    Races: {analyzer_data['races'].iloc[0]}")
                
                # Check for discrepancies
                if abs(manual_total - analyzer_data['total_overtakes'].iloc[0]) > 1:
                    print("  ⚠️  DISCREPANCY in total overtakes!")
                else:
                    print("  ✓ Total overtakes match")
            else:
                print("  ⚠️  No analyzer data found for this driver")

def validate_points_calculations(data, analyzer):
    """Verify points calculations for top drivers"""
    print("\n" + "="*80)
    print("VALIDATING POINTS CALCULATIONS")
    print("="*80)
    
    results = data.get('results', pd.DataFrame()).copy()
    sprint_results = data.get('sprint_results', pd.DataFrame()).copy()
    
    # Points mapping for regular races
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    # Calculate points manually
    results['manual_points'] = results['positionOrder'].map(points_map).fillna(0)
    
    # Add fastest lap point - check if fastest_laps data exists
    fastest_laps = data.get('fastest_laps', pd.DataFrame())
    fastest_lap_points = pd.Series()
    
    if not fastest_laps.empty and 'positionNumber' in fastest_laps.columns:
        # Get drivers who had fastest lap and finished in top 10
        fl_with_results = fastest_laps[fastest_laps['positionNumber'] == 1].merge(
            results[['raceId', 'driverId', 'positionOrder']], 
            on=['raceId', 'driverId']
        )
        fastest_lap_points = fl_with_results[
            fl_with_results['positionOrder'] <= 10
        ].groupby('driverId').size()
    
    # Test for specific drivers
    test_drivers = [830, 844, 815]
    driver_names = {830: 'Hamilton', 844: 'Russell', 815: 'Verstappen'}
    
    # Get analyzer's points data
    points_analysis = analyzer.analyze_points()
    
    for driver_id in test_drivers:
        driver_results = results[results['driverId'] == driver_id]
        
        if not driver_results.empty:
            # Manual calculations
            manual_total = driver_results['manual_points'].sum()
            if driver_id in fastest_lap_points.index:
                manual_total += fastest_lap_points[driver_id]
            
            manual_avg = driver_results.groupby('raceId')['manual_points'].sum().mean()
            
            # Get analyzer's calculations
            analyzer_data = points_analysis[points_analysis['driverId'] == driver_id]
            
            print(f"\n{driver_names.get(driver_id, f'Driver {driver_id}')}:")
            print(f"  Manual Calculations:")
            print(f"    Total points: {manual_total}")
            print(f"    Average points per race: {manual_avg:.2f}")
            print(f"    Races: {driver_results['raceId'].nunique()}")
            
            if not analyzer_data.empty:
                print(f"  Analyzer Calculations:")
                print(f"    Total points: {analyzer_data['total_points'].iloc[0]}")
                print(f"    Average points: {analyzer_data['avg_points'].iloc[0]:.2f}")
                print(f"    Races: {analyzer_data['races'].iloc[0]}")
                
                # Allow small discrepancy due to sprint races
                if abs(manual_total - analyzer_data['total_points'].iloc[0]) > 10:
                    print("  ⚠️  DISCREPANCY in total points (may include sprint races)!")
                else:
                    print("  ✓ Points calculations appear correct")
            else:
                print("  ⚠️  No analyzer data found for this driver")

def validate_pit_stop_calculations(data, analyzer):
    """Check pit stop time calculations for sample drivers"""
    print("\n" + "="*80)
    print("VALIDATING PIT STOP CALCULATIONS")
    print("="*80)
    
    pit_stops = data.get('pit_stops', pd.DataFrame()).copy()
    
    if pit_stops.empty:
        print("ERROR: No pit stop data available")
        return
    
    # Check if we have time data
    if 'timeMillis' not in pit_stops.columns:
        print("ERROR: No pit stop time data available")
        return
    
    # Note: The pit stop times in the data are lap times, not duration
    # For proper validation, we'd need the actual pit stop duration data
    # Let's skip detailed pit stop validation for now
    print("Note: Pit stop data contains lap times, not actual pit stop durations")
    print("Skipping detailed pit stop duration validation")
    
    # Just check that analyzer has pit stop data
    pit_analysis = analyzer.analyze_pit_stops()
    if not pit_analysis.empty:
        print(f"\nAnalyzer found pit stop data for {len(pit_analysis)} drivers")
        print("Sample pit stop analysis:")
        # Show available columns first
        print(f"Available columns: {pit_analysis.columns.tolist()}")
        print(pit_analysis.head(5))
    return
    
    # This section is now handled above

def validate_past_year_filter(data, analyzer):
    """Validate that past year filter correctly shows 2024-2025 drivers"""
    print("\n" + "="*80)
    print("VALIDATING PAST YEAR FILTER (2024-2025)")
    print("="*80)
    
    results = data.get('results', pd.DataFrame()).copy()
    races = data.get('races', pd.DataFrame()).copy()
    
    # Current year from analyzer
    current_year = analyzer.current_year
    print(f"Analyzer current year: {current_year}")
    
    # Get drivers from past year (current and previous)
    past_year_results = results[
        (results['year'] == current_year) | 
        (results['year'] == current_year - 1)
    ]
    
    manual_drivers = past_year_results['driverId'].unique()
    print(f"\nManual calculation:")
    print(f"  Years included: {current_year-1}, {current_year}")
    print(f"  Unique drivers found: {len(manual_drivers)}")
    
    # Get analyzer's driver list (from any analysis that filters by year)
    overtakes = analyzer.analyze_overtakes()
    if not overtakes.empty:
        print(f"\nAnalyzer overtakes columns: {overtakes.columns.tolist()}")
        # The overtakes dataframe is indexed by driverId
        analyzer_drivers = overtakes.index.unique()
        print(f"\nAnalyzer results:")
        print(f"  Unique drivers in analysis: {len(analyzer_drivers)}")
        
        # Check overlap
        overlap = len(set(manual_drivers) & set(analyzer_drivers))
        print(f"  Overlap: {overlap} drivers")
        
        # Find discrepancies
        only_manual = set(manual_drivers) - set(analyzer_drivers)
        only_analyzer = set(analyzer_drivers) - set(manual_drivers)
        
        if only_manual:
            print(f"  ⚠️  Drivers in manual but not analyzer: {len(only_manual)}")
        if only_analyzer:
            print(f"  ⚠️  Drivers in analyzer but not manual: {len(only_analyzer)}")
        
        if not only_manual and not only_analyzer:
            print("  ✓ Past year filter is working correctly")

def validate_teammate_overtakes(data, analyzer):
    """Cross-check teammate overtake calculations and PrizePicks scoring"""
    print("\n" + "="*80)
    print("VALIDATING TEAMMATE OVERTAKES & PRIZEPICKS SCORING")
    print("="*80)
    
    results = data.get('results', pd.DataFrame()).copy()
    grid = data.get('races_starting_grid_positions', pd.DataFrame()).copy()
    
    # Merge grid positions with results
    overtake_data = results.merge(
        grid[['raceId', 'driverId', 'positionNumber']].rename(
            columns={'positionNumber': 'gridPosition'}
        ),
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Check if we have constructor data in results
    if 'constructorId' not in overtake_data.columns:
        print("Note: Constructor data not directly available in results")
        # Try to get from other sources or skip this validation
        return
    
    # Filter valid finishes
    overtake_data = overtake_data[overtake_data['positionOrder'] > 0]
    
    # Find teammate battles for a specific race
    test_race_id = overtake_data['raceId'].max()  # Most recent race
    race_data = overtake_data[overtake_data['raceId'] == test_race_id]
    
    print(f"\nAnalyzing Race ID: {test_race_id}")
    
    # Group by constructor to find teammates
    for constructor_id, team_data in race_data.groupby('constructorId'):
        if len(team_data) == 2:  # Both drivers finished
            drivers = team_data.sort_values('positionOrder')
            driver1 = drivers.iloc[0]
            driver2 = drivers.iloc[1]
            
            # Calculate who beat who
            d1_beat_d2 = driver1['positionOrder'] < driver2['positionOrder']
            
            # Check if it was an overtake
            d1_gained = driver1['gridPosition'] - driver1['positionOrder']
            d2_gained = driver2['gridPosition'] - driver2['positionOrder']
            
            was_overtake = False
            overtaker = None
            
            if d1_beat_d2 and driver1['gridPosition'] > driver2['gridPosition']:
                was_overtake = True
                overtaker = driver1['driverId']
            elif not d1_beat_d2 and driver2['gridPosition'] > driver1['gridPosition']:
                was_overtake = True
                overtaker = driver2['driverId']
            
            print(f"\nConstructor {constructor_id}:")
            print(f"  Driver 1 (ID {driver1['driverId']}): Grid {driver1['gridPosition']} -> Finish {driver1['positionOrder']}")
            print(f"  Driver 2 (ID {driver2['driverId']}): Grid {driver2['gridPosition']} -> Finish {driver2['positionOrder']}")
            print(f"  Winner: Driver {'1' if d1_beat_d2 else '2'}")
            print(f"  Was overtake: {was_overtake}")
            
            if was_overtake:
                print(f"  Overtaker: Driver ID {overtaker}")
                print(f"  PrizePicks: +1.5 for overtaker, -1.5 for overtaken")
    
    # Get analyzer's teammate overtake data
    teammate_analysis = analyzer.analyze_teammate_overtakes()
    
    if not teammate_analysis.empty:
        print("\n\nAnalyzer's Teammate Overtake Summary:")
        print(f"Columns: {teammate_analysis.columns.tolist()}")
        print(teammate_analysis.head(10))
        
        # Validate PrizePicks calculation
        for _, row in teammate_analysis.head(5).iterrows():
            expected_pp = row['net_OT'] * 1.5
            actual_pp = row['prizepicks_pts']
            
            if abs(expected_pp - actual_pp) > 0.01:
                print(f"  ⚠️  PrizePicks calculation error for driver {row['driverId']}")
            else:
                print(f"  ✓ PrizePicks calculation correct for driver {row['driverId']}")

def main():
    """Run all validations"""
    print("Loading F1 data...")
    data = load_f1db_data()
    
    if not data:
        print("ERROR: Failed to load F1 data")
        return
    
    print("Initializing analyzer...")
    analyzer = F1PerformanceAnalyzer(data)
    
    # Run all validations
    validate_overtake_calculations(data, analyzer)
    validate_points_calculations(data, analyzer)
    validate_pit_stop_calculations(data, analyzer)
    validate_past_year_filter(data, analyzer)
    validate_teammate_overtakes(data, analyzer)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()