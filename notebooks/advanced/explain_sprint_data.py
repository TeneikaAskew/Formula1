"""Explain why certain drivers have 0 or NaN values in sprint races"""

import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import load_f1db_data

def explain_sprint_zeros():
    # Load the F1 data
    print("Loading F1 data...")
    data_dir = Path("/workspace/data/f1db")
    data = load_f1db_data(data_dir)
    
    print("\n=== SPRINT RACE ANALYSIS ===")
    
    # Get sprint results
    sprint_results = data.get('sprint_results', pd.DataFrame())
    results = data.get('results', pd.DataFrame())
    drivers = data.get('drivers', pd.DataFrame())
    races = data.get('races', pd.DataFrame())
    
    if sprint_results.empty:
        print("No sprint race data available!")
        return
    
    # Add year to sprint results if not present
    if 'year' not in sprint_results.columns and not races.empty:
        sprint_results = sprint_results.merge(races[['id', 'year', 'date']], left_on='raceId', right_on='id', how='left')
    
    print(f"\nTotal sprint race results: {len(sprint_results)}")
    print(f"Years with sprint races: {sorted(sprint_results['year'].unique())}")
    
    # Count sprint races per year
    sprint_races_per_year = sprint_results.groupby('year')['raceId'].nunique()
    print("\nSprint races per year:")
    for year, count in sprint_races_per_year.items():
        print(f"  {year}: {count} sprint races")
    
    # Get all unique drivers from sprint races
    sprint_drivers = sprint_results['driverId'].unique()
    print(f"\nTotal drivers who participated in sprint races: {len(sprint_drivers)}")
    
    # Get drivers from 2021-2024 regular races
    recent_results = results[(results['year'] >= 2021) & (results['year'] <= 2024)]
    all_recent_drivers = recent_results['driverId'].unique()
    
    # Find drivers who raced in 2021-2024 but never in sprints
    never_sprint = set(all_recent_drivers) - set(sprint_drivers)
    print(f"\nDrivers who raced 2021-2024 but never in sprint races: {len(never_sprint)}")
    
    # Get driver names for those who never did sprints
    if not drivers.empty and len(never_sprint) > 0:
        never_sprint_details = drivers[drivers['id'].isin(never_sprint)][['id', 'lastName']]
        print("\nReasons for 0/NaN sprint values:")
        
        for _, driver in never_sprint_details.iterrows():
            driver_id = driver['id']
            driver_name = driver['lastName']
            
            # Check when they raced
            driver_races = recent_results[recent_results['driverId'] == driver_id]
            if not driver_races.empty:
                years_raced = sorted(driver_races['year'].unique())
                
                # Check if they raced in sprint years
                sprint_years = set(sprint_results['year'].unique())
                raced_in_sprint_years = set(years_raced) & sprint_years
                
                if not raced_in_sprint_years:
                    print(f"  • {driver_name}: Only raced in {years_raced} (no sprints those years)")
                else:
                    # They raced in sprint years but not in sprint races
                    # Check if they were reserve/substitute drivers
                    races_in_sprint_years = driver_races[driver_races['year'].isin(sprint_years)]
                    num_races = len(races_in_sprint_years)
                    
                    if num_races < 5:
                        print(f"  • {driver_name}: Limited appearances ({num_races} races in sprint years) - likely substitute/reserve driver")
                    else:
                        print(f"  • {driver_name}: Raced in sprint years {sorted(raced_in_sprint_years)} but missed all sprint weekends")
    
    # Analyze specific drivers mentioned
    print("\n=== SPECIFIC DRIVER ANALYSIS ===")
    specific_drivers = ['antonio-giovinazzi', 'franco-colapinto', 'gabriel-bortoleto', 
                       'guanyu-zhou', 'jack-doohan', 'kimi-raikkonen', 'liam-lawson']
    
    for driver_id in specific_drivers:
        driver_results = recent_results[recent_results['driverId'] == driver_id]
        driver_sprints = sprint_results[sprint_results['driverId'] == driver_id]
        
        if not driver_results.empty:
            years = sorted(driver_results['year'].unique())
            races_count = len(driver_results)
            sprint_count = len(driver_sprints)
            
            driver_info = drivers[drivers['id'] == driver_id].iloc[0] if not drivers[drivers['id'] == driver_id].empty else None
            name = driver_info['lastName'] if driver_info is not None else driver_id
            
            print(f"\n{name}:")
            print(f"  Regular races: {races_count} in years {years}")
            print(f"  Sprint races: {sprint_count}")
            
            if sprint_count == 0:
                if 2021 not in years and 2022 not in years and 2023 not in years and 2024 not in years:
                    print(f"  Reason: Did not race during sprint era (2021-2024)")
                elif races_count < 10:
                    print(f"  Reason: Limited F1 appearances - substitute/test driver")
                else:
                    print(f"  Reason: Raced in sprint era but missed all sprint weekends")

if __name__ == "__main__":
    explain_sprint_zeros()