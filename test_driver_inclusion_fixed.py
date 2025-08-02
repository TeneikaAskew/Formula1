#!/usr/bin/env python3
"""
Test if all 2025 drivers appear in performance analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1_performance_analysis import F1PerformanceAnalyzer
from f1db_data_loader import load_f1db_data

# Expected 2025 drivers
expected_2025_drivers = {
    'gabriel-bortoleto', 'nico-hulkenberg', 'franco-colapinto', 
    'isack-hadjar', 'liam-lawson'
}

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

# Get active drivers
print(f"\nCurrent year detected: {analyzer.current_year}")
active_drivers = analyzer.get_active_drivers()

if active_drivers.empty:
    print("ERROR: No active drivers found!")
else:
    print(f"\nFound {len(active_drivers)} active drivers")
    
    # Check if expected drivers are included
    active_driver_ids = set(active_drivers['id'].values)
    
    print("\nChecking for expected 2025 drivers:")
    print("-" * 50)
    
    all_found = True
    for driver_id in expected_2025_drivers:
        if driver_id in active_driver_ids:
            driver_info = active_drivers[active_drivers['id'] == driver_id].iloc[0]
            print(f"✓ Found: {driver_info['forename']} {driver_info['surname']} ({driver_id})")
        else:
            print(f"✗ MISSING: {driver_id}")
            all_found = False
    
    if all_found:
        print("\n✅ All expected 2025 drivers are included!")
    else:
        print("\n❌ Some 2025 drivers are missing!")
        
    # Test filtering
    print("\nTesting filter_current_season_drivers()...")
    
    # Create a dummy DataFrame with driver IDs
    import pandas as pd
    test_df = pd.DataFrame({
        'driverId': list(active_driver_ids),
        'test_value': range(len(active_driver_ids))
    })
    test_df.set_index('driverId', inplace=True)
    
    filtered_df = analyzer.filter_current_season_drivers(test_df)
    
    print(f"Original drivers: {len(test_df)}")
    print(f"Filtered drivers: {len(filtered_df)}")
    
    # Check if expected drivers survived filtering
    print("\nChecking filtered results:")
    print("-" * 50)
    
    all_in_filtered = True
    for driver_id in expected_2025_drivers:
        if driver_id in filtered_df.index:
            print(f"✓ {driver_id} is in filtered results")
        else:
            print(f"✗ {driver_id} is NOT in filtered results")
            all_in_filtered = False
    
    if all_in_filtered:
        print("\n✅ All expected drivers survived filtering!")
    else:
        print("\n❌ Some drivers were incorrectly filtered out!")