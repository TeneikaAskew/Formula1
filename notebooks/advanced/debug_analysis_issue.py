"""Debug why analysis methods return empty dataframes"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def debug_analysis():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    # Create analyzer
    analyzer = F1PerformanceAnalyzer(data)
    print(f"\nCurrent season detected: {analyzer.current_year}")
    
    # Check active drivers
    active_drivers = analyzer.get_active_drivers()
    print(f"\nActive drivers type: {type(active_drivers)}")
    print(f"Active drivers shape: {active_drivers.shape if isinstance(active_drivers, pd.DataFrame) else 'Not a DataFrame'}")
    
    if isinstance(active_drivers, pd.DataFrame) and not active_drivers.empty:
        print(f"Active drivers found: {len(active_drivers)}")
        print("Sample drivers:")
        print(active_drivers[['id', 'lastName']].head(10))
    
    # Debug overtakes analysis step by step
    print("\n=== DEBUGGING OVERTAKES ANALYSIS ===")
    
    results = data.get('results', pd.DataFrame())
    grid = data.get('races_starting_grid_positions', pd.DataFrame())
    
    print(f"Results shape: {results.shape}")
    print(f"Grid positions shape: {grid.shape}")
    
    if not results.empty and not grid.empty:
        # Check 2025 data specifically
        results_2025 = results[results['year'] == 2025]
        print(f"\n2025 results: {len(results_2025)}")
        
        # Try the merge
        print("\nAttempting merge...")
        overtake_data = results_2025.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        print(f"Merged data shape: {overtake_data.shape}")
        print(f"Rows with grid position: {overtake_data['gridPosition'].notna().sum()}")
        
        if overtake_data['gridPosition'].notna().any():
            # Calculate positions gained
            overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
            print(f"\nPositions gained calculated")
            print(overtake_data[['driverId', 'gridPosition', 'positionNumber', 'positions_gained']].head(10))
        
    # Debug points analysis
    print("\n=== DEBUGGING POINTS ANALYSIS ===")
    
    if not results.empty:
        # Check recent years data
        recent_results = results[(results['year'] >= 2022) & (results['year'] <= 2025)]
        print(f"Results 2022-2025: {len(recent_results)}")
        
        # Group by driver and year
        driver_year_points = recent_results.groupby(['driverId', 'year'])['points'].sum().reset_index()
        print(f"\nDriver-year combinations: {len(driver_year_points)}")
        print("\nSample points by driver and year:")
        print(driver_year_points.head(15))
        
        # Current season only
        current_season = recent_results[recent_results['year'] == 2025]
        current_stats = current_season.groupby('driverId').agg({
            'points': ['sum', 'mean', 'median', 'count']
        }).round(2)
        print(f"\n2025 driver stats shape: {current_stats.shape}")
        if not current_stats.empty:
            print("\nSample 2025 stats:")
            print(current_stats.head(10))

if __name__ == "__main__":
    debug_analysis()