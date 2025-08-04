#!/usr/bin/env python3
"""Test real overtake integration with debugging"""

import sys
import os
from pathlib import Path

# Add notebooks/advanced to path
sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "advanced"))

# Change to notebooks/advanced directory
os.chdir(Path(__file__).parent / "notebooks" / "advanced")

def test_real_overtakes():
    """Test the real overtake integration with debugging"""
    print("Testing real overtake integration...")
    
    from f1db_data_loader import F1DBDataLoader
    from f1_performance_analysis import F1PerformanceAnalyzer
    
    # Load data
    loader = F1DBDataLoader(data_dir="../../data/f1db")
    data_dict = loader.load_csv_data(validate=False, check_updates=False)
    
    # Initialize analyzer
    analyzer = F1PerformanceAnalyzer(data_dict)
    
    # Test lap analyzer directly
    if analyzer.lap_analyzer:
        print("\n=== Lap Analyzer Direct Test ===")
        
        # Check available years
        available_years = []
        for year_dir in analyzer.lap_analyzer.data_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                if year >= analyzer.current_year - 3:
                    available_years.append(year)
        print(f"Available years: {available_years}")
        
        # Test single season
        if available_years:
            test_year = available_years[-1]  # Most recent year
            print(f"\nTesting {test_year} season...")
            season_data = analyzer.lap_analyzer.analyze_season_overtakes(test_year)
            print(f"Season data shape: {season_data.shape}")
            if not season_data.empty:
                print("Sample drivers from lap data:")
                print(season_data['driverId'].unique()[:10])
                
                # Test driver summary
                driver_summary = analyzer.lap_analyzer.get_driver_overtake_summary(season_data)
                print(f"Driver summary shape: {driver_summary.shape}")
                if not driver_summary.empty:
                    print("Top 5 overtakers from lap data:")
                    print(driver_summary[['driverId', 'total_overtakes_made', 'races_participated']].head())
        
        # Test analyzer method
        print("\n=== Analyzer Method Test ===")
        overtakes = analyzer.analyze_overtakes()
        print(f"Overtakes result type: {type(overtakes)}")
        print(f"Overtakes shape/size: {overtakes.shape if hasattr(overtakes, 'shape') else len(overtakes) if hasattr(overtakes, '__len__') else 'unknown'}")
        if hasattr(overtakes, 'empty') and not overtakes.empty:
            print("Columns:", list(overtakes.columns))
            print("Sample data:")
            print(overtakes.head())
        elif hasattr(overtakes, '__len__') and len(overtakes) > 0:
            print("Keys:", list(overtakes.keys()) if hasattr(overtakes, 'keys') else str(overtakes)[:200])
    
    # Test current season drivers
    print("\n=== Current Season Drivers ===")
    current_drivers = analyzer.get_active_drivers()
    print(f"Current drivers count: {len(current_drivers)}")
    if not current_drivers.empty:
        print("Sample F1DB driver IDs:")
        print(current_drivers['id'].head(10).tolist())

if __name__ == "__main__":
    test_real_overtakes()