#!/usr/bin/env python3
"""Test circuit-specific averages in points analysis"""

import sys
import pandas as pd
from pathlib import Path

# Add the advanced directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

def test_points_circuit_averages():
    """Test that circuit-specific averages are calculated correctly for points"""
    print("Loading F1DB data...")
    loader = F1DBDataLoader()
    data = loader.load_csv_data()
    
    # Initialize analyzer
    analyzer = F1PerformanceAnalyzer(data)
    
    # Get points analysis
    print("\nAnalyzing points...")
    points_data = analyzer.analyze_points()
    
    # Handle both DataFrame and dict returns
    if isinstance(points_data, dict):
        print("analyze_points() returned a dict - unexpected!")
        print(f"Keys: {list(points_data.keys())}")
        return
    
    if points_data.empty:
        print("No points data found!")
        return
    
    # Print column names
    print("\nColumns in points analysis:")
    print(points_data.columns.tolist())
    
    # Check if circuit-specific columns exist
    circuit_cols = ['prev_circuit_avg', 'current_circuit_avg', 'next_circuit_avg']
    missing_cols = [col for col in circuit_cols if col not in points_data.columns]
    
    if missing_cols:
        print(f"\nMissing columns: {missing_cols}")
    else:
        print("\n✓ All circuit-specific columns present")
    
    # Sample a few drivers to check their values
    print("\nSample driver data:")
    sample_drivers = points_data.head(5)
    
    # Show relevant columns
    display_cols = ['driver_name', 'avg_points', 'prev_circuit_avg', 'current_circuit_avg', 'next_circuit_avg']
    display_cols = [col for col in display_cols if col in points_data.columns]
    
    print(sample_drivers[display_cols].to_string())
    
    # Check if values are different
    print("\nChecking if circuit averages differ from overall average:")
    for idx, row in sample_drivers.iterrows():
        if 'avg_points' in row and 'prev_circuit_avg' in row:
            if row['avg_points'] != row['prev_circuit_avg']:
                print(f"✓ {row.get('driver_name', 'Unknown')}: avg_points={row['avg_points']}, prev_circuit_avg={row['prev_circuit_avg']}")
            else:
                print(f"⚠ {row.get('driver_name', 'Unknown')}: values are the same ({row['avg_points']})")
    
    # Debug: Check if circuitId is in results data
    print("\n\nDebugging data structure:")
    results = data.get('results', pd.DataFrame())
    if not results.empty:
        print(f"Results columns: {results.columns.tolist()}")
        print(f"Results shape: {results.shape}")
        
        # Check if we have circuitId after merge
        if 'circuitId' not in results.columns:
            print("\n⚠ circuitId not in results! Checking races data...")
            races = data.get('races', pd.DataFrame())
            if not races.empty:
                print(f"Races columns: {races.columns.tolist()}")
                
                # Try manual merge to see what happens
                if 'id' in races.columns and 'raceId' in results.columns:
                    merged = results.merge(races[['id', 'circuitId']], 
                                         left_on='raceId', right_on='id', how='left')
                    print(f"\nAfter merge, circuitId present: {'circuitId' in merged.columns}")
                    if 'circuitId' in merged.columns:
                        print(f"Non-null circuitId values: {merged['circuitId'].notna().sum()}")
        else:
            print(f"✓ circuitId present in results")
            print(f"Non-null circuitId values: {results['circuitId'].notna().sum()}")
    
    # Check next race info
    next_race = analyzer.get_next_race()
    if next_race is not None:
        print(f"\nNext race: {next_race.get('name', 'Unknown')}")
        print(f"Next circuit ID: {next_race.get('circuitId', 'Unknown')}")

if __name__ == "__main__":
    test_points_circuit_averages()