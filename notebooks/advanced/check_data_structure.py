#!/usr/bin/env python
"""Check the structure of the loaded F1 data"""

from f1db_data_loader import F1DBDataLoader
import pandas as pd

# Load data
print("Loading F1 data...")
loader = F1DBDataLoader()
data = loader.load_csv_data()

print("\n" + "="*80)
print("DATA STRUCTURE CHECK")
print("="*80)

# Check what tables we have
print("\nAvailable tables:")
for key, df in data.items():
    if isinstance(df, pd.DataFrame):
        print(f"  {key}: {len(df)} rows, columns: {list(df.columns)[:5]}...")

# Check specific important tables and look for proper names
important_tables = {
    'races': ['races', 'race'],
    'results': ['results', 'races_race_results', 'race_results'],
    'pit_stops': ['pit_stops', 'races_pit_stops', 'pitstops'],
    'qualifying': ['qualifying', 'races_qualifying_results', 'qualifying_results'],
    'lap_times': ['lap_times', 'races_laps', 'laps'],
    'drivers': ['drivers', 'driver']
}

for table_type, possible_names in important_tables.items():
    print(f"\n" + "-"*80)
    print(f"{table_type.upper()} TABLE:")
    
    found = False
    for name in possible_names:
        if name in data:
            df = data[name]
            print(f"  Found as: '{name}'")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()[:10]}...")  # First 10 columns
            
            # Check for ID columns
            id_cols = [col for col in df.columns if 'id' in col.lower()]
            if id_cols:
                print(f"  ID columns: {id_cols}")
            
            # Show first row
            if not df.empty:
                print(f"  First row sample:")
                for col in df.columns[:5]:
                    print(f"    {col}: {df.iloc[0][col]}")
            
            found = True
            break
    
    if not found:
        print(f"  NOT FOUND - tried: {possible_names}")

print("\n" + "-"*80)
print("CHECKING FOR YEAR INFORMATION:")
# Find which tables have year info
for key, df in data.items():
    if isinstance(df, pd.DataFrame) and 'year' in df.columns:
        print(f"  {key} has 'year' column")

# Check if results already has year merged
if 'results' in data and 'year' in data['results'].columns:
    print("\n✓ Results already has year column")
else:
    print("\n⚠️  Results doesn't have year column - need to merge with races")