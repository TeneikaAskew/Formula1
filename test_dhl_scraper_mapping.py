#!/usr/bin/env python3
"""
Test script to verify DHL scraper mapping functionality
"""

import pandas as pd
from pathlib import Path
import sys

def test_scraper_output():
    """Test the output of the DHL scraper for proper mapping"""
    
    # Find the most recent DHL CSV file
    dhl_dir = Path("data/dhl")
    if not dhl_dir.exists():
        print("❌ DHL data directory not found")
        return False
    
    csv_files = list(dhl_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in DHL directory")
        return False
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"✓ Found DHL data file: {latest_file}")
    
    # Load the data
    df = pd.read_csv(latest_file)
    print(f"✓ Loaded {len(df)} pit stop records")
    
    # Check for new columns
    required_columns = ['driver_id', 'constructor_id', 'race_id', 'circuit_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        print("  Please run the updated scraper to generate data with ID mappings")
        return False
    
    print("✓ All required ID columns present")
    
    # Check mapping statistics
    print("\nMapping Statistics:")
    print(f"  Driver IDs mapped: {df['driver_id'].notna().sum()}/{len(df)} ({df['driver_id'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  Constructor IDs mapped: {df['constructor_id'].notna().sum()}/{len(df)} ({df['constructor_id'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  Race IDs mapped: {df['race_id'].notna().sum()}/{len(df)} ({df['race_id'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  Circuit IDs mapped: {df['circuit_id'].notna().sum()}/{len(df)} ({df['circuit_id'].notna().sum()/len(df)*100:.1f}%)")
    
    # Show sample mappings
    print("\nSample Mappings (first 5 records with driver_id):")
    sample = df[df['driver_id'].notna()][['driver', 'driver_id', 'team', 'constructor_id', 'race', 'race_id']].head()
    print(sample.to_string(index=False))
    
    # Show unmapped records
    unmapped_drivers = df[df['driver_id'].isna()]
    if not unmapped_drivers.empty:
        print(f"\nUnmapped Drivers ({len(unmapped_drivers)} records):")
        unique_unmapped = unmapped_drivers[['driver', 'team', 'year']].drop_duplicates()
        print(unique_unmapped.head(10).to_string(index=False))
        if len(unique_unmapped) > 10:
            print(f"... and {len(unique_unmapped) - 10} more unique driver/team combinations")
    
    # Test integration with f1_performance_analysis
    print("\nTesting integration with f1_performance_analysis.py...")
    
    # Check if driver_id values match F1DB format
    if df['driver_id'].notna().any():
        sample_id = df[df['driver_id'].notna()]['driver_id'].iloc[0]
        if isinstance(sample_id, str) and '-' in sample_id:
            print(f"✓ Driver IDs use F1DB format (e.g., '{sample_id}')")
        else:
            print(f"⚠️  Driver ID format may not match F1DB: '{sample_id}'")
    
    return True

def test_f1db_loading():
    """Test that F1DB data can be loaded"""
    try:
        # Try to load F1DB data
        drivers = pd.read_csv("data/f1db/drivers.csv")
        constructors = pd.read_csv("data/f1db/constructors.csv")
        races = pd.read_csv("data/f1db/races.csv")
        
        print("\nF1DB Data Summary:")
        print(f"  Drivers: {len(drivers)} records")
        print(f"  Constructors: {len(constructors)} records")
        print(f"  Races: {len(races)} records")
        
        # Show some current drivers
        current_drivers = ['max-verstappen', 'lewis-hamilton', 'charles-leclerc', 'sergio-perez', 'carlos-sainz-jr']
        print("\nChecking current drivers in F1DB:")
        for driver_id in current_drivers:
            driver = drivers[drivers['id'] == driver_id]
            if not driver.empty:
                print(f"  ✓ {driver.iloc[0]['fullName']} ({driver_id})")
            else:
                print(f"  ❌ {driver_id} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading F1DB data: {e}")
        return False

if __name__ == "__main__":
    print("DHL Scraper Mapping Test")
    print("=" * 50)
    
    # Test F1DB data availability
    if not test_f1db_loading():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Test scraper output
    if test_scraper_output():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Please run the updated scraper:")
        print("   python dhl_pitstop_scraper.py --summary")
        sys.exit(1)