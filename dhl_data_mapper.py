#!/usr/bin/env python3
"""
DHL Data Mapper - Maps DHL pit stop data to F1DB driver and constructor IDs
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DHLDataMapper:
    """Maps DHL pit stop data to F1DB IDs"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.f1db_dir = self.data_dir / "f1db"
        self.dhl_dir = self.data_dir / "dhl"
        
        # Load F1DB reference data
        self.drivers_df = pd.read_csv(self.f1db_dir / "drivers.csv")
        self.constructors_df = pd.read_csv(self.f1db_dir / "constructors.csv")
        self.season_drivers_df = pd.read_csv(self.f1db_dir / "seasons-entrants-drivers.csv")
        
        # Create team name mappings
        self.team_mappings = {
            # Current teams
            'Red Bull': 'red-bull',
            'Red Bull Racing': 'red-bull',
            'Mercedes': 'mercedes',
            'Mercedes AMG': 'mercedes',
            'Ferrari': 'ferrari',
            'McLaren': 'mclaren',
            'Aston Martin': 'aston-martin',
            'Alpine': 'alpine',
            'Williams': 'williams',
            'AlphaTauri': 'alphatauri',
            'Racing Bulls': 'rb',  # New name for RB/AlphaTauri
            'RB': 'rb',
            'Visa RB': 'rb',
            'Alfa Romeo': 'alfa-romeo',
            'Sauber': 'sauber',
            'Haas': 'haas',
            'Kick Sauber': 'sauber'  # Sauber's 2025 name
        }
    
    def extract_driver_id_from_lastname(self, driver_id):
        """Extract last name from driver ID (part after last dash)"""
        if pd.isna(driver_id):
            return None
        parts = str(driver_id).split('-')
        return parts[-1].lower()
    
    def get_driver_mapping_for_2025(self):
        """Create a mapping of (lastname, constructor) -> driver_id for 2025"""
        # Filter for 2025 season
        season_2025 = self.season_drivers_df[self.season_drivers_df['year'] == 2025].copy()
        
        # Merge with drivers data to get last names
        season_2025 = season_2025.merge(
            self.drivers_df[['id', 'lastName']], 
            left_on='driverId', 
            right_on='id', 
            suffixes=('', '_driver')
        )
        
        # Create mapping: (lastname, constructorId) -> driverId
        driver_mapping = {}
        for _, row in season_2025.iterrows():
            if not pd.isna(row['lastName']) and not row.get('testDriver', False):
                key = (row['lastName'].lower(), row['constructorId'])
                driver_mapping[key] = row['driverId']
        
        return driver_mapping
    
    def map_dhl_team_to_constructor(self, team_name):
        """Map DHL team name to F1DB constructor ID"""
        if pd.isna(team_name):
            return None
        
        # Try direct mapping
        for dhl_name, constructor_id in self.team_mappings.items():
            if dhl_name.lower() in str(team_name).lower():
                return constructor_id
        
        # If no match, try to find in team name
        team_lower = str(team_name).lower()
        if 'mercedes' in team_lower:
            return 'mercedes'
        elif 'ferrari' in team_lower:
            return 'ferrari'
        elif 'red bull' in team_lower:
            return 'red-bull'
        elif 'mclaren' in team_lower:
            return 'mclaren'
        elif 'alpine' in team_lower:
            return 'alpine'
        elif 'aston' in team_lower:
            return 'aston-martin'
        elif 'williams' in team_lower:
            return 'williams'
        elif 'haas' in team_lower:
            return 'haas'
        elif 'sauber' in team_lower or 'kick' in team_lower:
            return 'sauber'
        elif 'racing bulls' in team_lower or ' rb' in team_lower:
            return 'rb'
        
        logger.warning(f"No constructor mapping found for team: {team_name}")
        return None
    
    def enhance_driver_pitstops(self, input_file=None):
        """Enhance driver pit stops CSV with F1DB IDs"""
        # Find the latest driver pitstops file if not specified
        if input_file is None:
            csv_files = list(self.dhl_dir.glob("dhl_all_driver_pitstops_*.csv"))
            if not csv_files:
                logger.error("No driver pitstops CSV files found")
                return None
            input_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Processing: {input_file}")
        
        # Load DHL data
        dhl_df = pd.read_csv(input_file)
        
        # Get driver mapping for 2025
        driver_mapping = self.get_driver_mapping_for_2025()
        
        # Add constructor_id column
        dhl_df['constructor_id'] = dhl_df['team'].apply(self.map_dhl_team_to_constructor)
        
        # Add driver_id column
        def map_driver(row):
            if pd.isna(row['driver']) or pd.isna(row['constructor_id']):
                return None
            
            # DHL data has last name only
            driver_lastname = str(row['driver']).lower().strip()
            constructor_id = row['constructor_id']
            
            # Look up in mapping
            key = (driver_lastname, constructor_id)
            driver_id = driver_mapping.get(key)
            
            if not driver_id:
                # Try some special cases
                special_cases = {
                    ('antonelli', 'mercedes'): 'andrea-kimi-antonelli',
                    ('bearman', 'haas'): 'oliver-bearman',
                    ('bortoleto', 'sauber'): 'gabriel-bortoleto',
                    ('lawson', 'red-bull'): 'liam-lawson',
                    ('lawson', 'rb'): 'liam-lawson',  # Lawson might be at RB
                    ('hadjar', 'rb'): 'isack-hadjar',
                    ('tsunoda', 'rb'): 'yuki-tsunoda',  # Tsunoda might be at RB
                    ('tsunoda', 'red-bull'): 'yuki-tsunoda',  # Or at Red Bull
                    ('hulkenberg', 'sauber'): 'nico-hulkenberg',  # Hulkenberg to Sauber for 2025
                    ('sainz', 'williams'): 'carlos-sainz-jr',  # Sainz to Williams for 2025
                }
                driver_id = special_cases.get(key)
                
                if not driver_id:
                    logger.warning(f"No driver mapping found for: {driver_lastname} ({constructor_id})")
            
            return driver_id
        
        dhl_df['driver_id'] = dhl_df.apply(map_driver, axis=1)
        
        # Reorder columns
        columns = ['position', 'team', 'constructor_id', 'driver', 'driver_id', 
                   'time', 'lap', 'points', 'event_id', 'event_name', 'event_date', 'event_abbr']
        
        # Keep any additional columns that might exist
        for col in dhl_df.columns:
            if col not in columns:
                columns.append(col)
        
        dhl_df = dhl_df[columns]
        
        # Save enhanced file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.dhl_dir / f"dhl_driver_pitstops_mapped_{timestamp}.csv"
        dhl_df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Saved enhanced data to: {output_file}")
        
        # Show mapping statistics
        self.show_mapping_stats(dhl_df)
        
        return dhl_df
    
    def show_mapping_stats(self, df):
        """Show mapping statistics"""
        print("\n📊 MAPPING STATISTICS")
        print("=" * 50)
        print(f"Total pit stops: {len(df)}")
        print(f"Mapped drivers: {df['driver_id'].notna().sum()}/{len(df)} ({df['driver_id'].notna().sum()/len(df)*100:.1f}%)")
        print(f"Mapped constructors: {df['constructor_id'].notna().sum()}/{len(df)} ({df['constructor_id'].notna().sum()/len(df)*100:.1f}%)")
        
        # Show unmapped drivers
        unmapped = df[df['driver_id'].isna()][['driver', 'team', 'constructor_id']].drop_duplicates()
        if len(unmapped) > 0:
            print(f"\n⚠️  Unmapped drivers ({len(unmapped)}):")
            for _, row in unmapped.iterrows():
                print(f"  - {row['driver']} ({row['team']} -> {row['constructor_id']})")
        
        # Show unique teams and their mappings
        team_mappings = df[['team', 'constructor_id']].drop_duplicates().sort_values('team')
        print(f"\n🏎️  Team Mappings ({len(team_mappings)}):")
        for _, row in team_mappings.iterrows():
            print(f"  - {row['team']} -> {row['constructor_id']}")

def main():
    """Main function"""
    print("🏎️  DHL Data Mapper - Adding F1DB IDs")
    print("=" * 50)
    
    mapper = DHLDataMapper()
    enhanced_df = mapper.enhance_driver_pitstops()
    
    if enhanced_df is not None:
        print("\n✅ Mapping complete!")

if __name__ == "__main__":
    main()