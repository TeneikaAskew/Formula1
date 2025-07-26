#!/usr/bin/env python3
"""
Test that notebooks can work with F1DB data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from f1db_data_loader import load_f1db_data
from f1_ml import fix_column_mappings, merge_race_data, create_temporal_features, create_prediction_targets

print("Testing F1 Notebook Data Compatibility")
print("=" * 80)

# Load data
print("\n1. Loading F1DB data...")
data = load_f1db_data(data_dir="/workspace/data/f1db")

# Fix column mappings
print("\n2. Fixing column mappings...")
data = fix_column_mappings(data)

# Check results columns
print("\n3. Checking results columns after mapping:")
if 'results' in data:
    results_df = data['results']
    required_cols = ['positionOrder', 'points', 'grid', 'statusId', 'driverId', 'constructorId', 'raceId']
    
    for col in required_cols:
        if col in results_df.columns:
            print(f"  ✓ {col} found")
        else:
            print(f"  ✗ {col} NOT FOUND")
            
    # Check added columns
    added_cols = ['dnf', 'win', 'podium', 'points_finish']
    print("\n  Added columns:")
    for col in added_cols:
        if col in results_df.columns:
            print(f"  ✓ {col} added")

# Test merge
print("\n4. Testing data merge...")
try:
    merged_df = merge_race_data(data)
    print(f"  ✓ Merged data shape: {merged_df.shape}")
    print(f"  ✓ Columns: {list(merged_df.columns)[:10]}...")
    
    # Check for driver age
    if 'driver_age' in merged_df.columns:
        print(f"  ✓ Driver age calculated")
        avg_age = merged_df['driver_age'].mean()
        print(f"    Average driver age: {avg_age:.1f} years")
        
except Exception as e:
    print(f"  ✗ Merge failed: {e}")

# Test temporal features
print("\n5. Testing temporal features...")
try:
    # Get recent data
    recent_df = merged_df[merged_df['year'] >= 2020].copy()
    
    # Create temporal features
    features_df = create_temporal_features(recent_df)
    
    # Check created features
    temporal_cols = ['avg_position_3', 'avg_points_5', 'career_wins', 'driver_track_avg']
    for col in temporal_cols:
        if col in features_df.columns:
            print(f"  ✓ {col} created")
            non_null = features_df[col].notna().sum()
            print(f"    Non-null values: {non_null}/{len(features_df)}")
            
except Exception as e:
    print(f"  ✗ Temporal features failed: {e}")
    import traceback
    traceback.print_exc()

# Test prediction targets
print("\n6. Testing prediction targets...")
try:
    targets_df = create_prediction_targets(features_df)
    
    target_cols = ['top_10', 'podium', 'winner', 'points_finish']
    for col in target_cols:
        if col in targets_df.columns:
            print(f"  ✓ {col} created")
            pct = targets_df[col].mean() * 100
            print(f"    Percentage: {pct:.1f}%")
            
except Exception as e:
    print(f"  ✗ Prediction targets failed: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY:")

if all(col in results_df.columns for col in required_cols):
    print("✅ Data is compatible with notebooks!")
else:
    print("❌ Data needs additional fixes for full compatibility")
    
print("\nRecommendations:")
print("- Use fix_column_mappings() when loading data in notebooks")
print("- Use merge_race_data() to prepare data for analysis")
print("- Update notebooks to handle F1DB column names where needed")