#!/usr/bin/env python3
"""
Test if performance tables include all 2025 drivers
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1_performance_analysis import F1PerformanceAnalyzer
from f1db_data_loader import load_f1db_data

# Expected 2025 drivers
expected_drivers = {
    'gabriel-bortoleto', 'nico-hulkenberg', 'franco-colapinto', 
    'isack-hadjar', 'liam-lawson'
}

# Load F1 database
print("Loading F1 database...")
data = load_f1db_data()

# Create analyzer
analyzer = F1PerformanceAnalyzer(data)

print(f"\nTesting performance tables for year: {analyzer.current_year}")
print("=" * 80)

# Test each analysis method
methods_to_test = [
    ('analyze_overtakes', 'Overtakes Analysis'),
    ('analyze_points', 'Points Analysis'),
    ('analyze_pit_stops', 'Pit Stops Analysis'),
    ('analyze_dhl_pit_stops', 'DHL Pit Stops Analysis'),
    ('analyze_starting_positions', 'Starting Positions Analysis'),
    ('analyze_sprint_points', 'Sprint Points Analysis'),
    ('analyze_teammate_overtakes', 'Teammate Overtakes Analysis'),
    ('analyze_fastest_laps', 'Fastest Laps Analysis')
]

all_consistent = True

for method_name, display_name in methods_to_test:
    print(f"\n{display_name}:")
    print("-" * 40)
    
    try:
        method = getattr(analyzer, method_name)
        result = method()
        
        # Handle direct DataFrame return
        if hasattr(result, 'index') and hasattr(result, 'columns'):
            df = result
            if not df.empty:
                # Get driver IDs from the result
                if df.index.name in ['id', 'driverId', None]:
                    driver_ids = set(df.index)
                    
                    # Check for expected drivers
                    missing = []
                    for driver_id in expected_drivers:
                        if driver_id not in driver_ids:
                            missing.append(driver_id)
                    
                    if missing:
                        print(f"❌ Missing drivers: {', '.join(missing)}")
                        all_consistent = False
                    else:
                        print(f"✅ All expected drivers present ({len(driver_ids)} total)")
                else:
                    print("⚠️  Could not identify driver column")
            else:
                print("⚠️  No data returned (empty DataFrame)")
        else:
            print("⚠️  Unexpected result format (not a DataFrame)")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        all_consistent = False

print("\n" + "=" * 80)
if all_consistent:
    print("✅ All performance tables consistently include all 2025 drivers!")
else:
    print("❌ Some tables are missing drivers - consistency issues remain")