#!/usr/bin/env python3
"""
Fix for the Year-by-Year Starting Position Matrix display issue.
This script shows the specific code changes needed to handle NaN values properly.
"""

def show_fix():
    print("FIX FOR YEAR-BY-YEAR STARTING POSITION MATRIX NaN DISPLAY ISSUE")
    print("="*80)
    print()
    print("PROBLEM: The matrix shows 'nan' for NaN values instead of a consistent display")
    print()
    print("CURRENT CODE (around line 1760-1780 in f1_performance_analysis.py):")
    print("-"*80)
    print("""
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            avg_start = f"{row['avg_start']:.1f}"
                            # Check if we have points data
                            if 'avg_points' in row:
                                pts_per_race = f"{row['avg_points']:.1f}"
                                row_str += f"{avg_start:>7}  {pts_per_race:>11}    "
                            else:
                                row_str += f"{avg_start:>7}  {'—':>11}    "
                        else:
                            row_str += f"{'—':^22}"
    """)
    
    print("\nFIXED CODE:")
    print("-"*80)
    print("""
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            
                            # Check for NaN values in avg_start
                            if pd.notna(row['avg_start']):
                                avg_start = f"{row['avg_start']:.1f}"
                                
                                # Check if we have points data and it's not NaN
                                if 'avg_points' in row and pd.notna(row['avg_points']):
                                    pts_per_race = f"{row['avg_points']:.1f}"
                                    row_str += f"{avg_start:>7}  {pts_per_race:>11}    "
                                else:
                                    row_str += f"{avg_start:>7}  {'—':>11}    "
                            else:
                                # If avg_start is NaN, show em dash for entire cell
                                row_str += f"{'—':^22}"
                        else:
                            row_str += f"{'—':^22}"
    """)
    
    print("\nKEY CHANGES:")
    print("1. Added pd.notna(row['avg_start']) check before formatting")
    print("2. Added pd.notna(row['avg_points']) check for points data")
    print("3. If avg_start is NaN, display '—' for the entire cell")
    print("4. If only avg_points is NaN, show start position with '—' for points")
    print()
    print("This ensures consistent display:")
    print("- '—' for missing data (no race that year OR NaN values)")
    print("- Numeric values only when actual valid data exists")
    print()
    print("Note: Need to import pandas as pd at the top of the file if not already imported")

if __name__ == "__main__":
    show_fix()