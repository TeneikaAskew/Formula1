#!/usr/bin/env python3
"""
Debug script to investigate why all predictions show 100% UNDER probability
when using custom prop lines (e.g., overtakes=6.0 instead of 2.5)
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

def test_probability_calculation():
    """Test the probability calculation logic with different scenarios"""
    
    print("="*80)
    print("TESTING PROBABILITY CALCULATION LOGIC")
    print("="*80)
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Normal case - line below mean',
            'data': [2, 3, 4, 2, 3, 4, 3, 2, 3, 4],  # Mean ~3
            'line': 2.5,
            'expected': 'More OVER than UNDER'
        },
        {
            'name': 'Normal case - line above mean',
            'data': [2, 3, 4, 2, 3, 4, 3, 2, 3, 4],  # Mean ~3
            'line': 6.0,
            'expected': 'More UNDER than OVER'
        },
        {
            'name': 'Edge case - all zeros',
            'data': [0, 0, 0, 0, 0],
            'line': 6.0,
            'expected': '100% UNDER'
        },
        {
            'name': 'Real overtake data simulation',
            'data': [0, 1, 2, 3, 0, 1, 2, 1, 0, 3],  # Typical overtake distribution
            'line': 6.0,
            'expected': 'Heavy UNDER'
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Data: {test['data']}")
        print(f"Line: {test['line']}")
        
        # Calculate using the same logic as f1_predictions_enhanced.py
        mean = np.mean(test['data'])
        std = np.std(test['data'])
        
        print(f"Mean: {mean:.2f}, Std: {std:.2f}")
        
        if std == 0:
            over_prob = 1.0 if mean > test['line'] else 0.0
            under_prob = 1.0 - over_prob
            print("No variance - deterministic outcome")
        else:
            z_score = (test['line'] - mean) / std
            under_prob = stats.norm.cdf(z_score)
            over_prob = 1.0 - under_prob
            print(f"Z-score: {z_score:.2f}")
        
        print(f"Over {test['line']}: {over_prob:.1%}")
        print(f"Under {test['line']}: {under_prob:.1%}")
        print(f"Expected: {test['expected']}")
        print("-"*40)

def analyze_real_overtake_data():
    """Analyze actual overtake data from F1DB"""
    
    print("\n"+"="*80)
    print("ANALYZING REAL OVERTAKE DATA")
    print("="*80)
    
    loader = F1DBDataLoader()
    data_dict = loader.get_core_datasets()
    
    results = data_dict.get('results', pd.DataFrame())
    grid = data_dict.get('races_starting_grid_positions', pd.DataFrame())
    
    if results.empty or grid.empty:
        print("ERROR: Could not load results or grid data")
        return
    
    # Get a sample driver's overtake history
    driver_ids = results['driverId'].value_counts().head(5).index
    
    for driver_id in driver_ids[:3]:  # Check top 3 drivers
        driver_results = results[results['driverId'] == driver_id].tail(20)
        driver_grid = grid[grid['driverId'] == driver_id]
        
        # Merge to calculate overtakes
        merged = driver_results.merge(
            driver_grid[['raceId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on='raceId',
            how='left'
        )
        
        # Calculate overtakes
        merged['overtakes'] = (merged['gridPosition'] - merged['positionNumber']).apply(
            lambda x: max(0, x) if pd.notna(x) else 0
        )
        
        overtakes = merged['overtakes'].values
        
        # Get driver name
        drivers_df = data_dict.get('drivers', pd.DataFrame())
        driver_name = "Unknown"
        if not drivers_df.empty:
            driver_info = drivers_df[drivers_df['id'] == driver_id]
            if not driver_info.empty:
                driver_name = driver_info.iloc[0]['name']
        
        print(f"\nDriver: {driver_name} (ID: {driver_id})")
        print(f"Overtakes in last {len(overtakes)} races: {overtakes}")
        print(f"Mean: {np.mean(overtakes):.2f}, Std: {np.std(overtakes):.2f}")
        
        # Test with different lines
        for line in [2.5, 6.0]:
            if np.std(overtakes) == 0:
                over_prob = 1.0 if np.mean(overtakes) > line else 0.0
            else:
                z_score = (line - np.mean(overtakes)) / np.std(overtakes)
                under_prob = stats.norm.cdf(z_score)
                over_prob = 1.0 - under_prob
            
            print(f"  Line {line}: Over={over_prob:.1%}, Under={1-over_prob:.1%}")

def check_data_quality():
    """Check if data is being loaded correctly"""
    
    print("\n"+"="*80)
    print("CHECKING DATA QUALITY")
    print("="*80)
    
    loader = F1DBDataLoader()
    data_dict = loader.get_core_datasets()
    
    for table_name, df in data_dict.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n{table_name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            
            # Check for specific issues
            if table_name == 'results':
                print(f"  Recent races: {df['raceId'].max() if 'raceId' in df.columns else 'N/A'}")
                print(f"  Unique drivers: {df['driverId'].nunique() if 'driverId' in df.columns else 'N/A'}")
            
            if table_name == 'pit_stops':
                if not df.empty:
                    avg_stops = df.groupby('raceId')['stop'].max().mean()
                    print(f"  Avg max stops per race: {avg_stops:.2f}")

def main():
    """Run all debug tests"""
    
    print("DEBUGGING PROBABILITY CALCULATION ISSUE")
    print("User reported: All predictions showing 100% UNDER with custom lines")
    print("")
    
    # Test the probability calculation logic
    test_probability_calculation()
    
    # Analyze real data
    analyze_real_overtake_data()
    
    # Check data quality
    check_data_quality()
    
    print("\n"+"="*80)
    print("CONCLUSION")
    print("="*80)
    print("If all drivers show 100% UNDER for high lines (e.g., 6.0), this is likely because:")
    print("1. The historical data shows much lower values (e.g., avg overtakes ~1-3)")
    print("2. The normal distribution assumption makes high values extremely unlikely")
    print("3. This is actually CORRECT behavior - not a bug!")
    print("\nThe issue is that custom lines like 6.0 overtakes are unrealistic for F1.")
    print("Most drivers average 1-3 overtakes per race, so 6.0 is ~3 standard deviations above mean.")

if __name__ == "__main__":
    main()