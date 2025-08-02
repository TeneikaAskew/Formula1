#!/usr/bin/env python3
"""
Test script to demonstrate the difference between '-' and 'nan' in the 
Year-by-Year Starting Position Matrix.

The issue is:
1. When a driver didn't race in a specific year, the code displays '—' (em dash)
2. When a driver raced but has NaN values in avg_start or avg_points, it shows 'nan'
"""

import pandas as pd
import numpy as np

def demonstrate_matrix_display():
    """Demonstrate how the matrix handles missing data vs NaN values"""
    
    # Create sample data similar to what the analyze_starting_positions_by_track_year returns
    sample_data = pd.DataFrame([
        {'driverId': 1, 'driver_name': 'Driver A', 'year': 2022, 'avg_start': 5.5, 'avg_points': 12.3},
        {'driverId': 1, 'driver_name': 'Driver A', 'year': 2023, 'avg_start': 4.2, 'avg_points': 15.8},
        {'driverId': 1, 'driver_name': 'Driver A', 'year': 2024, 'avg_start': np.nan, 'avg_points': np.nan},  # NaN values
        
        {'driverId': 2, 'driver_name': 'Driver B', 'year': 2022, 'avg_start': 8.1, 'avg_points': 8.5},
        # Driver B missing 2023 data (didn't race)
        {'driverId': 2, 'driver_name': 'Driver B', 'year': 2024, 'avg_start': 7.3, 'avg_points': 10.2},
        
        {'driverId': 3, 'driver_name': 'Driver C', 'year': 2023, 'avg_start': 3.5, 'avg_points': 18.5},
        {'driverId': 3, 'driver_name': 'Driver C', 'year': 2024, 'avg_start': 2.8, 'avg_points': np.nan},  # Only points is NaN
    ])
    
    print("Sample Data:")
    print(sample_data)
    print("\n" + "="*80 + "\n")
    
    # Simulate the matrix display logic from f1_performance_analysis.py
    print("Year-by-Year Starting Position Matrix:")
    print("=" * 80)
    
    # Get unique years
    recent_years = sorted(sample_data['year'].unique())
    
    # Header with years
    header = f"{'Driver':<25}"
    for year in recent_years:
        header += f"{int(year):>22}"
    print(header)
    
    # Sub-header
    sub_header = f"{'':<25}"
    for year in recent_years:
        sub_header += f"{'Start  Pts/Race':>22}"
    print(sub_header)
    print("-" * (25 + len(recent_years) * 22))
    
    # Display each driver's data
    for driver_id in sample_data['driverId'].unique():
        driver_data = sample_data[sample_data['driverId'] == driver_id]
        driver_name = driver_data.iloc[0]['driver_name']
        
        row_str = f"{driver_name:<25}"
        
        for year in recent_years:
            year_row = driver_data[driver_data['year'] == year]
            if not year_row.empty:
                row = year_row.iloc[0]
                # This is where the issue occurs
                avg_start = f"{row['avg_start']:.1f}"  # Will show 'nan' if value is NaN
                
                if 'avg_points' in row:
                    pts_per_race = f"{row['avg_points']:.1f}"  # Will show 'nan' if value is NaN
                    row_str += f"{avg_start:>7}  {pts_per_race:>11}    "
                else:
                    row_str += f"{avg_start:>7}  {'—':>11}    "
            else:
                # Driver didn't race this year - shows '—'
                row_str += f"{'—':^22}"
        
        print(row_str)
    
    print("\n" + "="*80 + "\n")
    print("Key Observations:")
    print("1. Driver A 2024: Shows 'nan' because driver raced but has NaN values")
    print("2. Driver B 2023: Shows '—' because driver didn't race that year (no data)")
    print("3. Driver C 2024: Shows 'nan' for points but numeric value for start position")
    print("\nThe difference is:")
    print("- '—' (em dash) = No data for that year (driver didn't race)")
    print("- 'nan' = Driver raced but the value is NaN (missing or invalid data)")
    
    print("\n" + "="*80 + "\n")
    print("SOLUTION: The code should check for NaN values and display them as '—' or 'N/A'")
    print("Here's the fixed version:")
    print()
    
    # Fixed version
    print("Fixed Year-by-Year Starting Position Matrix:")
    print("=" * 80)
    print(header)
    print(sub_header)
    print("-" * (25 + len(recent_years) * 22))
    
    for driver_id in sample_data['driverId'].unique():
        driver_data = sample_data[sample_data['driverId'] == driver_id]
        driver_name = driver_data.iloc[0]['driver_name']
        
        row_str = f"{driver_name:<25}"
        
        for year in recent_years:
            year_row = driver_data[driver_data['year'] == year]
            if not year_row.empty:
                row = year_row.iloc[0]
                
                # Fixed: Check for NaN values
                if pd.notna(row['avg_start']) and pd.notna(row.get('avg_points', np.nan)):
                    avg_start = f"{row['avg_start']:.1f}"
                    pts_per_race = f"{row['avg_points']:.1f}"
                    row_str += f"{avg_start:>7}  {pts_per_race:>11}    "
                elif pd.notna(row['avg_start']):
                    avg_start = f"{row['avg_start']:.1f}"
                    row_str += f"{avg_start:>7}  {'—':>11}    "
                else:
                    row_str += f"{'—':^22}"
            else:
                row_str += f"{'—':^22}"
        
        print(row_str)

if __name__ == "__main__":
    demonstrate_matrix_display()