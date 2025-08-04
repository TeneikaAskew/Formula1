#!/usr/bin/env python3
"""Test the F1 Fantasy workflow logic locally"""

import subprocess
import sys
import os
from pathlib import Path

def test_fantasy_fetcher():
    """Test the F1 Fantasy fetcher"""
    print("Testing F1 Fantasy fetcher...")
    
    # Change to notebooks/advanced
    os.chdir(Path(__file__).parent / 'notebooks' / 'advanced')
    
    # Run the fetcher
    result = subprocess.run([
        sys.executable, 
        'f1_fantasy_fetcher.py', 
        '--output-dir', '../../data/f1_fantasy'
    ], capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
    
    # Check if files were created
    fantasy_dir = Path('../../data/f1_fantasy')
    overview_file = fantasy_dir / 'driver_overview.csv'
    details_file = fantasy_dir / 'driver_details.csv'
    metadata_file = fantasy_dir / '.f1_fantasy_metadata.json'
    
    print(f"\nFile check:")
    print(f"Overview exists: {overview_file.exists()}")
    print(f"Details exists: {details_file.exists()}")
    print(f"Metadata exists: {metadata_file.exists()}")
    
    if overview_file.exists():
        import pandas as pd
        df = pd.read_csv(overview_file)
        print(f"\nOverview shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:5]}...")  # First 5 columns
        
        # Check for required columns
        required_cols = ['player_id', 'player_name', 'team_name', 'fantasy_points', 'current_price']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
        else:
            print("✅ All required columns present")
    
    return result.returncode == 0

if __name__ == "__main__":
    success = test_fantasy_fetcher()
    exit(0 if success else 1)