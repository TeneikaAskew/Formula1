#!/usr/bin/env python3
"""
Test script to verify all 2025 drivers are included in performance analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'notebooks/advanced'))

from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

def test_driver_inclusion():
    """Test that all 2025 drivers are included"""
    print("="*60)
    print("TESTING 2025 DRIVER INCLUSION")
    print("="*60)
    
    # Load F1 database
    print("\nLoading F1 database...")
    data = load_f1db_data()
    
    # Create analyzer
    analyzer = F1PerformanceAnalyzer(data)
    
    # Get active drivers
    print(f"\nCurrent year: {analyzer.current_year}")
    active_drivers = analyzer.get_active_drivers()
    
    print(f"\nActive drivers found: {len(active_drivers)}")
    
    # Check for specific new drivers
    new_drivers = {
        1078: "Liam Lawson",
        4041: "Isack Hadjar", 
        4040: "Gabriel Bortoleto"
    }
    
    print("\nChecking for new 2025 drivers:")
    print("-" * 40)
    for driver_id, name in new_drivers.items():
        if driver_id in active_drivers['id'].values:
            driver_data = active_drivers[active_drivers['id'] == driver_id].iloc[0]
            print(f"✓ {name} (ID: {driver_id}) - Found as {driver_data['surname']}")
        else:
            print(f"✗ {name} (ID: {driver_id}) - NOT FOUND")
    
    # Show all active drivers
    print("\nAll active drivers:")
    print("-" * 40)
    for _, driver in active_drivers.iterrows():
        print(f"{driver['id']:4d}: {driver['surname']:20s} {driver.get('forename', '')}")
    
    # Test a specific analysis method to ensure drivers appear
    print("\nTesting lap time analysis for new drivers...")
    lap_times = analyzer.analyze_lap_times()
    
    if not lap_times.empty:
        print(f"\nDrivers in lap time analysis: {len(lap_times)}")
        for driver_id in new_drivers.keys():
            if driver_id in lap_times.index:
                print(f"✓ Driver {driver_id} appears in lap time analysis")
            else:
                print(f"✗ Driver {driver_id} missing from lap time analysis")
    
    return active_drivers

if __name__ == "__main__":
    active_drivers = test_driver_inclusion()