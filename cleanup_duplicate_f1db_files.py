#!/usr/bin/env python3
"""Clean up duplicate F1DB files from /workspace/data/ directory"""

import os
from pathlib import Path
import shutil

def cleanup_duplicate_f1db_files():
    """Remove duplicate F1DB CSV files from /workspace/data/"""
    data_dir = Path("/workspace/data")
    f1db_dir = Path("/workspace/data/f1db")
    
    # Get all CSV files in data directory
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in /workspace/data/")
        return
    
    print(f"Found {len(csv_files)} CSV files in /workspace/data/")
    
    removed_count = 0
    kept_count = 0
    
    for csv_file in csv_files:
        # Check if the same file exists in f1db directory
        f1db_file = f1db_dir / csv_file.name
        
        if f1db_file.exists():
            # File exists in correct location, safe to remove duplicate
            print(f"Removing duplicate: {csv_file.name}")
            csv_file.unlink()
            removed_count += 1
        else:
            # File doesn't exist in f1db directory, keep it for safety
            print(f"WARNING: {csv_file.name} not found in f1db/, keeping file")
            kept_count += 1
    
    print(f"\nCleanup complete:")
    print(f"  - Removed: {removed_count} duplicate files")
    print(f"  - Kept: {kept_count} files (not found in f1db/)")
    
    # Also check for any F1DB directories that might have been created
    for item in data_dir.iterdir():
        if item.is_dir() and item.name != "f1db" and item.name != "dhl":
            # Check if it looks like F1DB data
            if any((item / f).exists() for f in ["races.csv", "drivers.csv", "results.csv"]):
                print(f"\nWARNING: Found potential F1DB data directory: {item}")
                print("This should be manually reviewed and removed if appropriate")

if __name__ == "__main__":
    print("F1DB Duplicate File Cleanup")
    print("=" * 50)
    print("This script will remove duplicate F1DB CSV files from /workspace/data/")
    print("Files will only be removed if they exist in /workspace/data/f1db/")
    print()
    
    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() == "yes":
        cleanup_duplicate_f1db_files()
    else:
        print("Cleanup cancelled")