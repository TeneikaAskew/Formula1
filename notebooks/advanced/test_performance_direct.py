"""Direct test of F1 Performance Analysis without imports"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import just the analyzer class
from f1_performance_analysis import F1PerformanceAnalyzer

def load_data_directly():
    """Load F1 data directly without using f1db_data_loader"""
    data_dir = Path("/workspace/data/f1db")
    data = {}
    
    # Define the files we need
    files_to_load = {
        'races': 'races.csv',
        'results': 'races-race-results.csv',
        'drivers': 'drivers.csv',
        'circuits': 'circuits.csv',
        'pit_stops': 'races-pit-stops.csv',
        'races_starting_grid_positions': 'races-starting-grid-positions.csv',
        'sprint_results': 'races-sprint-race-results.csv'
    }
    
    print("Loading F1 data files directly...")
    for key, filename in files_to_load.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                data[key] = pd.read_csv(filepath)
                print(f"✓ Loaded {key}: {len(data[key])} rows")
            except Exception as e:
                print(f"✗ Error loading {key}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    # Map results data if loaded
    if 'results' in data:
        # Add year column if not present
        if 'year' not in data['results'].columns and 'races' in data:
            data['results'] = data['results'].merge(
                data['races'][['id', 'year', 'circuitId']], 
                left_on='raceId', 
                right_on='id', 
                how='left'
            )
    
    return data

def test_performance_analysis():
    # Load data directly
    data = load_data_directly()
    
    if not data:
        print("No data loaded!")
        return
    
    print("\n=== TESTING PERFORMANCE ANALYSIS ===")
    
    # Create the analyzer
    analyzer = F1PerformanceAnalyzer(data)
    
    # Generate all tables
    print("\n=== FULL PERFORMANCE TABLES ===")
    analyzer.generate_all_tables()

if __name__ == "__main__":
    test_performance_analysis()