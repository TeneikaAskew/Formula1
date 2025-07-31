"""Test the F1 Performance Analysis module"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def test_performance_analysis():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    print("\n=== TESTING PERFORMANCE ANALYSIS ===")
    
    # Create the analyzer
    analyzer = F1PerformanceAnalyzer(data)
    
    print(f"\nDetected current season: {analyzer.current_year}")
    print(f"Analysis will focus on drivers who raced in {analyzer.current_year}")
    
    # Test individual analysis methods
    print("\n--- Testing Overtakes Analysis ---")
    overtakes = analyzer.analyze_overtakes()
    print(f"Overtakes data shape: {overtakes.shape}")
    if not overtakes.empty:
        print("Sample overtakes data:")
        print(overtakes.head())
    else:
        print("No overtakes data returned!")
    
    print("\n--- Testing Points Analysis ---")
    points = analyzer.analyze_points()
    print(f"Points data shape: {points.shape}")
    if not points.empty:
        print("Sample points data:")
        print(points.head())
    else:
        print("No points data returned!")
    
    print("\n--- Testing Pit Stops Analysis ---")
    pit_stops = analyzer.analyze_pit_stops()
    print(f"Pit stops data shape: {pit_stops.shape}")
    if not pit_stops.empty:
        print("Sample pit stops data:")
        print(pit_stops.head())
    else:
        print("No pit stops data returned!")
    
    print("\n--- Testing Starting Positions Analysis ---")
    grid = analyzer.analyze_starting_positions()
    print(f"Starting positions data shape: {grid.shape}")
    if not grid.empty:
        print("Sample grid data:")
        print(grid.head())
    else:
        print("No grid data returned!")
    
    print("\n--- Testing Sprint Points Analysis ---")
    sprint = analyzer.analyze_sprint_points()
    print(f"Sprint points data shape: {sprint.shape}")
    if not sprint.empty:
        print("Sample sprint data:")
        print(sprint.head())
    else:
        print("No sprint data returned!")
    
    # Test the full table generation
    print("\n\n=== FULL PERFORMANCE TABLES ===")
    tables = analyzer.generate_all_tables()
    
    # Additional analysis for understanding the data
    print("\n\n=== ADDITIONAL DATA INSIGHTS ===")
    
    # Check year coverage
    results = data.get('results', pd.DataFrame())
    if not results.empty and 'year' in results.columns:
        print(f"\nData year range: {results['year'].min()} - {results['year'].max()}")
        print(f"Analysis focusing on: {analyzer.current_year-3} - {analyzer.current_year}")
        
        # Show how current season was determined
        results_with_data = results[results['positionNumber'].notna()]
        latest_result_year = results_with_data['year'].max() if not results_with_data.empty else 'No results'
        print(f"Latest year with race results: {latest_result_year}")
        
        # Count active drivers per year
        recent_years = list(range(max(2021, analyzer.current_year-3), analyzer.current_year+1))
        print("\nActive drivers per year:")
        for year in recent_years:
            year_results = results[results['year'] == year]
            unique_drivers = year_results['driverId'].nunique()
            races_count = year_results['raceId'].nunique()
            print(f"  {year}: {unique_drivers} drivers in {races_count} races")
    
    # Sprint race summary
    sprint_results = data.get('sprint_results', pd.DataFrame())
    if not sprint_results.empty:
        print("\n** Sprint Race Summary **")
        if 'year' in sprint_results.columns:
            sprint_by_year = sprint_results.groupby('year').agg({
                'raceId': 'nunique',
                'driverId': 'nunique'
            })
            sprint_by_year.columns = ['races', 'unique_drivers']
            print(sprint_by_year)
        
        # List all unique sprint participants
        sprint_drivers = sprint_results['driverId'].unique()
        print(f"\nTotal unique sprint participants: {len(sprint_drivers)}")
        
        # Sample of drivers who never did sprints despite racing in sprint era
        sprint_era_start = 2021
        all_recent_drivers = results[(results['year'] >= sprint_era_start) & (results['year'] <= analyzer.current_year)]['driverId'].unique()
        never_sprint = set(all_recent_drivers) - set(sprint_drivers)
        if never_sprint:
            drivers_df = data.get('drivers', pd.DataFrame())
            if not drivers_df.empty:
                # Focus on current season drivers who never did sprints
                current_season_drivers = results[results['year'] == analyzer.current_year]['driverId'].unique()
                current_never_sprint = set(current_season_drivers) & never_sprint
                if current_never_sprint:
                    current_never_names = drivers_df[drivers_df['id'].isin(list(current_never_sprint)[:5])]['lastName'].tolist()
                    print(f"\nCurrent season drivers who never did sprints: {', '.join(current_never_names)}")
                    if len(current_never_sprint) > 5:
                        print(f"(and {len(current_never_sprint) - 5} more)")
                
                print(f"\nTotal drivers in sprint era who never did sprints: {len(never_sprint)}")

if __name__ == "__main__":
    test_performance_analysis()
    print("\n" + "="*80)
    print("Test completed. Check the explanations above for understanding 0/NaN values.")
    print("="*80)