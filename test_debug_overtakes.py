#!/usr/bin/env python3
"""Debug the overtakes data flow"""

import sys
import os
from pathlib import Path

# Add notebooks/advanced to path
sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "advanced"))

# Change to notebooks/advanced directory
os.chdir(Path(__file__).parent / "notebooks" / "advanced")

def debug_overtakes():
    """Debug the overtakes data flow"""
    print("Debugging overtakes data flow...")
    
    from f1db_data_loader import F1DBDataLoader
    from f1_performance_analysis import F1PerformanceAnalyzer
    from lap_by_lap_overtakes import LapByLapOvertakeAnalyzer
    
    # Load data
    loader = F1DBDataLoader(data_dir="../../data/f1db")
    data_dict = loader.load_csv_data(validate=False, check_updates=False)
    
    # Test lap analyzer directly
    lap_analyzer = LapByLapOvertakeAnalyzer()
    
    print("\n1. Testing lap analyzer...")
    # Analyze 2025 season
    overtake_data = lap_analyzer.analyze_season_overtakes(2025)
    print(f"Overtake data shape: {overtake_data.shape}")
    print(f"Overtake data columns: {list(overtake_data.columns)}")
    print(f"Sample overtake data:\n{overtake_data.head()}")
    
    print("\n2. Getting driver summary...")
    driver_summary = lap_analyzer.get_driver_overtake_summary(overtake_data)
    print(f"Driver summary shape: {driver_summary.shape}")
    print(f"Driver summary columns: {list(driver_summary.columns)}")
    print(f"Sample driver summary:\n{driver_summary.head()}")
    
    print("\n3. Testing ID mapping...")
    # Test the mapping manually
    driver_id_mapping = {
        'max_verstappen': 'max-verstappen',
        'norris': 'lando-norris',
        'russell': 'george-russell', 
        'hamilton': 'lewis-hamilton',
        'leclerc': 'charles-leclerc',
        'piastri': 'oscar-piastri',
        'alonso': 'fernando-alonso',
        'gasly': 'pierre-gasly',
        'ocon': 'esteban-ocon',
        'stroll': 'lance-stroll',
        'albon': 'alexander-albon',
        'tsunoda': 'yuki-tsunoda',
        'hulkenberg': 'nico-hulkenberg',
        'bearman': 'oliver-bearman',
        'antonelli': 'andrea-kimi-antonelli',
        'bortoleto': 'gabriel-bortoleto',
        'lawson': 'liam-lawson',
        'hadjar': 'isack-hadjar',
        'doohan': 'jack-doohan',
        'colapinto': 'franco-colapinto',
        'perez': 'sergio-perez',
        'sainz': 'carlos-sainz-jr',
        'bottas': 'valtteri-bottas',
        'magnussen': 'kevin-magnussen',
        'zhou': 'guanyu-zhou',
        'ricciardo': 'daniel-ricciardo',
        'sargeant': 'logan-sargeant'
    }
    
    # Map the driver IDs
    driver_summary['f1db_driver_id'] = driver_summary['driverId'].map(driver_id_mapping)
    mapped_drivers = driver_summary[driver_summary['f1db_driver_id'].notna()].copy()
    print(f"\nMapped drivers shape: {mapped_drivers.shape}")
    print(f"Sample mapped drivers:\n{mapped_drivers[['driverId', 'f1db_driver_id', 'total_overtakes_made']].head()}")

if __name__ == "__main__":
    debug_overtakes()