#!/usr/bin/env python3
"""
DHL F1 Pit Stop Data Updater

Fetches DHL pit stop data from the official InMotion DHL API endpoints
using the same methodology as the existing 6365/6367 extractor:
- HTML table parsing from htmlList.table
- Full F1DB mappings for drivers, constructors, races
- Event-based extraction with complete context

API Endpoints:
- 2025: Drivers/times (6365), Races (6367) [existing]
- 2024: Drivers/times (6273), Races (6276) [new]
- 2023: Drivers/times (6282), Races (6284) [new]

"""

import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DHL_Data_Updater')


class DHLDataUpdater:
    """Updates DHL pit stop data using F1DB-integrated extraction methodology"""
    
    def __init__(self, output_dir: str = None):
        self.base_url = "https://inmotion.dhl"
        
        # API endpoint configurations
        self.endpoints = {
            2024: {'drivers': '6273', 'events': '6276'},
            2023: {'drivers': '6282', 'events': '6284'},
            2025: {'drivers': '6365', 'events': '6367'}  # Existing for reference
        }
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Output directory
        if output_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            self.output_dir = project_root / 'data' / 'dhl'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Load F1DB reference data
        self.f1db_dir = Path(__file__).parent.parent.parent / 'data' / 'f1db'
        self.drivers_df = pd.read_csv(self.f1db_dir / "drivers.csv")
        self.constructors_df = pd.read_csv(self.f1db_dir / "constructors.csv") 
        self.races_df = pd.read_csv(self.f1db_dir / "races.csv")
        self.season_drivers_df = pd.read_csv(self.f1db_dir / "seasons-entrants-drivers.csv")
        
        # Initialize mappings
        self._init_mappings()
        
        # Load existing data for comparison
        self.existing_data = self._load_existing_data()
    
    def _init_mappings(self):
        """Initialize all mapping dictionaries (from existing extractor)"""
        # Team name mappings
        self.team_mappings = {
            'Red Bull': 'red-bull',
            'Mercedes': 'mercedes', 
            'Ferrari': 'ferrari',
            'McLaren': 'mclaren',
            'Aston Martin': 'aston-martin',
            'Alpine': 'alpine',
            'Williams': 'williams',
            'Racing Bulls': 'rb',
            'RB': 'rb',  # Short form
            'Haas': 'haas',
            'Sauber': 'sauber',
            'AlphaTauri': 'alphatauri'  # 2023 team
        }
        
        # Circuit mappings for race matching
        self.circuit_mappings = {
            'Albert Park Grand Prix Circuit': 'melbourne',
            'Shanghai International Circuit': 'shanghai',
            'Suzuka Circuit': 'suzuka', 
            'Bahrain International Circuit': 'bahrain',
            'Jeddah Corniche Circuit': 'jeddah',
            'Miami International Autodrome': 'miami',
            'Autodromo Internazionale Enzo e Dino Ferrari': 'imola',
            'Circuit de Monaco': 'monaco',
            'Circuit de Barcelona-Catalunya': 'catalunya',
            'Circuit Gilles-Villeneuve': 'montreal',
            'Red Bull Ring': 'red-bull-ring',
            'Silverstone Circuit': 'silverstone',
            'Circuit de Spa-Francorchamps': 'spa-francorchamps',
            'Hungaroring': 'hungaroring',
            'Circuit Zandvoort': 'zandvoort',
            'Autodromo Nazionale Monza': 'monza',
            'Baku City Circuit': 'baku',
            'Marina Bay Street Circuit': 'marina-bay',
            'Circuit of The Americas': 'americas',
            'Autódromo Hermanos Rodríguez': 'rodriguez',
            'Autódromo José Carlos Pace': 'interlagos',
            'Las Vegas Strip Circuit': 'las-vegas',
            'Lusail International Circuit': 'losail',
            'Yas Marina Circuit': 'yas-marina'
        }
        
        # Build driver mappings for all years
        self.driver_mapping = self._build_driver_mapping()
    
    def _build_driver_mapping(self):
        """Build driver mapping for multiple seasons"""
        driver_mapping = {}
        
        # Build mappings for 2023 and 2024
        for year in [2023, 2024]:
            season_data = self.season_drivers_df[self.season_drivers_df['year'] == year].copy()
            season_data = season_data.merge(
                self.drivers_df[['id', 'lastName']], 
                left_on='driverId', 
                right_on='id', 
                suffixes=('', '_driver')
            )
            
            for _, row in season_data.iterrows():
                if not pd.isna(row['lastName']) and not row.get('testDriver', False):
                    key = (row['lastName'].lower(), self.map_team_to_constructor_basic(row['constructorId']), year)
                    driver_mapping[key] = row['driverId']
        
        # Add special cases for recent driver changes
        special_cases = {
            ('ricciardo', 'alphatauri', 2023): 'daniel-ricciardo',
            ('ricciardo', 'rb', 2024): 'daniel-ricciardo',
            ('tsunoda', 'alphatauri', 2023): 'yuki-tsunoda', 
            ('tsunoda', 'rb', 2024): 'yuki-tsunoda',
            ('sainz', 'ferrari', 2023): 'carlos-sainz-jr',
            ('sainz', 'ferrari', 2024): 'carlos-sainz-jr',
            ('piastri', 'mclaren', 2023): 'oscar-piastri',
            ('piastri', 'mclaren', 2024): 'oscar-piastri',
            ('doohan', 'alpine', 2024): 'jack-doohan'  # Late season entry
        }
        driver_mapping.update(special_cases)
        
        return driver_mapping
    
    def map_team_to_constructor_basic(self, constructor_id):
        """Basic constructor ID mapping"""
        mapping = {
            'red_bull': 'red-bull',
            'mercedes': 'mercedes',
            'ferrari': 'ferrari', 
            'mclaren': 'mclaren',
            'aston_martin': 'aston-martin',
            'alpine': 'alpine',
            'williams': 'williams',
            'alphatauri': 'alphatauri',
            'rb': 'rb',
            'haas': 'haas'
        }
        return mapping.get(constructor_id, constructor_id)
    
    def map_team_to_constructor(self, team_name):
        """Map DHL team name to F1DB constructor ID"""
        if pd.isna(team_name):
            return None
        
        team_lower = str(team_name).lower()
        for dhl_name, constructor_id in self.team_mappings.items():
            if dhl_name.lower() in team_lower:
                return constructor_id
        
        return None
    
    def map_driver(self, driver_name, constructor_id, year):
        """Map driver name to F1DB driver ID"""
        if pd.isna(driver_name) or pd.isna(constructor_id):
            return None
        
        driver_lastname = str(driver_name).lower().strip()
        key = (driver_lastname, constructor_id, year)
        return self.driver_mapping.get(key)
    
    def _load_existing_data(self) -> Optional[pd.DataFrame]:
        """Load existing DHL data if available"""
        existing_files = list(self.output_dir.glob("dhl_*.csv"))
        if existing_files:
            # Load the most recent file
            latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading existing data from: {latest_file}")
            return pd.read_csv(latest_file)
        return None
    
    def extract_events(self, year: int):
        """Extract events for a given year"""
        events_endpoint = f"{self.base_url}/api/f1-award-element-data/{self.endpoints[year]['events']}"
        
        try:
            logger.info(f"Extracting events for {year} from: {events_endpoint}")
            response = self.session.get(events_endpoint)
            response.raise_for_status()
            
            data = response.json()
            events = data.get('data', {}).get('chart', {}).get('events', [])
            
            # Convert to DataFrame
            events_df = pd.DataFrame(events)
            
            # Parse date field and add year
            if not events_df.empty and 'date' in events_df.columns:
                events_df['date_parsed'] = events_df['date'].apply(
                    lambda x: x.get('date', '').split(' ')[0] if isinstance(x, dict) else ''
                )
                events_df['year'] = year
            
            # Map to F1DB races
            events_df['f1db_race_id'] = events_df.apply(lambda row: self._map_event_to_race(row, year), axis=1)
            
            logger.info(f"✅ Extracted {len(events_df)} events for {year}")
            return events_df, data
            
        except Exception as e:
            logger.error(f"Failed to extract events for {year}: {e}")
            return pd.DataFrame(), None
    
    def _map_event_to_race(self, event_row, year):
        """Map a DHL event to F1DB race ID"""
        event_date = event_row.get('date_parsed', '')
        event_circuit = event_row.get('short_title', '')
        
        # Filter races by year
        year_races = self.races_df[self.races_df['year'] == year]
        
        # Try circuit mapping first
        circuit_id = self.circuit_mappings.get(event_circuit)
        if circuit_id:
            matches = year_races[year_races['circuitId'] == circuit_id]
            if len(matches) == 1:
                return matches.iloc[0]['id']
        
        # Try date matching (within 2 days)
        if event_date:
            try:
                event_date_obj = pd.to_datetime(event_date)
                for _, race in year_races.iterrows():
                    race_date_obj = pd.to_datetime(race['date'])
                    if abs((event_date_obj - race_date_obj).days) <= 2:
                        if not circuit_id or race['circuitId'] == circuit_id:
                            return race['id']
            except:
                pass
        
        return None
    
    def parse_pit_stop_table(self, html_content):
        """Parse HTML table to extract all pit stop data"""
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table', {'class': 'f1-award-table'})
        
        if not table:
            return []
        
        pit_stops = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 6:
                try:
                    pit_stop = {
                        'position': int(cells[0].get_text(strip=True)),
                        'team': cells[1].get_text(strip=True),
                        'driver': cells[2].get_text(strip=True),
                        'time': float(cells[3].get_text(strip=True)),
                        'lap': int(cells[4].get_text(strip=True)),
                        'points': int(cells[5].get_text(strip=True))
                    }
                    pit_stops.append(pit_stop)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue
        
        return pit_stops
    
    def extract_all_driver_data(self, events_df, year: int):
        """Extract driver pit stop data for all events with full F1DB mapping"""
        logger.info(f"Extracting driver pit stop data for {year} with full F1DB mapping...")
        
        all_driver_data = []
        drivers_endpoint = f"{self.base_url}/api/f1-award-element-data/{self.endpoints[year]['drivers']}"
        
        # Process each event
        for idx, event in tqdm(events_df.iterrows(), total=len(events_df), desc=f"Processing {year} events"):
            event_id = event['id']
            event_name = event.get('title', f'Event {event_id}')
            event_date = event.get('date_parsed', '')
            race_id = event.get('f1db_race_id')
            
            # Get race details if we have a race_id
            race_details = {}
            if race_id and not pd.isna(race_id):
                race = self.races_df[self.races_df['id'] == race_id]
                if not race.empty:
                    race_details = {
                        'race_id': int(race_id),
                        'grand_prix_id': race.iloc[0]['grandPrixId'],
                        'round': race.iloc[0]['round']
                    }
            
            try:
                # Fetch driver data for this event
                url = f"{drivers_endpoint}?event={event_id}"
                response = self.session.get(url)
                response.raise_for_status()
                
                data = response.json()
                
                # Get the HTML table which contains ALL pit stops
                html_table = data.get('htmlList', {}).get('table', '')
                
                if html_table:
                    # Parse HTML table
                    pit_stops = self.parse_pit_stop_table(html_table)
                    
                    # Add mappings to each pit stop
                    for stop in pit_stops:
                        # Map constructor
                        constructor_id = self.map_team_to_constructor(stop['team'])
                        
                        # Map driver
                        driver_id = self.map_driver(stop['driver'], constructor_id, year)
                        
                        # Build complete record
                        complete_stop = {
                            'position': stop['position'],
                            'team': stop['team'],
                            'constructor_id': constructor_id,
                            'driver': stop['driver'],
                            'driver_id': driver_id,
                            'time': stop['time'],
                            'lap': stop['lap'],
                            'points': stop['points'],
                            'event_id': event_id,
                            'event_name': event_name,
                            'event_date': event_date,
                            'event_abbr': event.get('abbr', ''),
                            **race_details  # Add race_id, grand_prix_id, round
                        }
                        
                        all_driver_data.append(complete_stop)
                    
                    logger.debug(f"Extracted {len(pit_stops)} pit stops for {event_name}")
                
            except Exception as e:
                logger.error(f"Failed to extract data for event {event_id} ({event_name}): {e}")
                continue
        
        # Convert to DataFrame
        driver_df = pd.DataFrame(all_driver_data)
        
        logger.info(f"✅ Extracted {len(driver_df)} fully integrated pit stops for {year}")
        return driver_df
    
    def update_dhl_data(self, years: List[int] = None) -> pd.DataFrame:
        """Update DHL data for specified years using proper F1DB integration"""
        if years is None:
            years = [2023, 2024]
        
        all_data = []
        
        for year in years:
            if year not in self.endpoints:
                logger.warning(f"No endpoint configuration for year {year}")
                continue
            
            logger.info(f"\n🏎️ Processing year {year}...")
            
            # 1. Extract events with race mapping
            events_df, events_data = self.extract_events(year)
            
            if events_df.empty:
                logger.warning(f"No events found for {year}")
                continue
            
            # 2. Extract all driver data with full F1DB mapping
            driver_df = self.extract_all_driver_data(events_df, year)
            
            if not driver_df.empty:
                all_data.append(driver_df)
                
                # Show summary for this year
                logger.info(f"📊 {year} Summary:")
                logger.info(f"  - Events: {len(events_df)}")
                logger.info(f"  - Events mapped to races: {events_df['f1db_race_id'].notna().sum()}/{len(events_df)}")
                logger.info(f"  - Pit stops: {len(driver_df)}")
                logger.info(f"  - Driver mapping success: {driver_df['driver_id'].notna().sum()}/{len(driver_df)} ({driver_df['driver_id'].notna().sum()/len(driver_df)*100:.1f}%)")
                logger.info(f"  - Constructor mapping success: {driver_df['constructor_id'].notna().sum()}/{len(driver_df)} ({driver_df['constructor_id'].notna().sum()/len(driver_df)*100:.1f}%)")
        
        # Combine all years
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Clean and standardize the data to match existing format
            combined_df = self._standardize_data(combined_df)
            
            # Save the updated data
            self._save_data(combined_df)
            
            return combined_df
        else:
            logger.error("No data was successfully fetched")
            return pd.DataFrame()
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format to match existing DHL backup structure"""
        if df.empty:
            return df
        
        # Create standardized DataFrame with the exact structure of the backup file
        standardized_data = []
        
        for _, row in df.iterrows():
            record = {
                'position': row.get('position'),
                'team': row.get('team'),
                'constructor_id': row.get('constructor_id'),
                'driver': row.get('driver'),
                'driver_id': row.get('driver_id'),
                'time': row.get('time'),
                'lap': row.get('lap'),
                'points': row.get('points'),
                'event_id': row.get('event_id'),
                'event_name': row.get('event_name'),
                'event_date': row.get('event_date'),
                'event_abbr': row.get('event_abbr'),
                'race_id': row.get('race_id'),
                'grand_prix_id': row.get('grand_prix_id'),
                'round': row.get('round')
            }
            standardized_data.append(record)
        
        # Convert to DataFrame
        standardized_df = pd.DataFrame(standardized_data)
        
        # Convert numeric columns
        numeric_columns = ['position', 'time', 'lap', 'points', 'event_id', 'race_id', 'round']
        for col in numeric_columns:
            if col in standardized_df.columns:
                standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        
        # Sort by event_date, event_id, and position
        standardized_df = standardized_df.sort_values(['event_date', 'event_id', 'position'], na_position='last')
        
        return standardized_df
    
    def _save_data(self, df: pd.DataFrame):
        """Save the updated DHL data"""
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"dhl_pitstops_integrated_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Saved updated DHL data to: {filepath}")
        
        # Also save a backup of existing data if it exists
        if self.existing_data is not None:
            backup_filename = f"dhl_pitstops_backup_{timestamp}.csv"
            backup_filepath = self.output_dir / backup_filename
            self.existing_data.to_csv(backup_filepath, index=False)
            logger.info(f"📦 Backed up existing data to: {backup_filepath}")
        
        # Log summary statistics
        logger.info(f"\n📊 Final Data Summary:")
        logger.info(f"  - Total records: {len(df)}")
        if 'event_date' in df.columns:
            years = pd.to_datetime(df['event_date'], errors='coerce').dt.year.dropna().unique()
            logger.info(f"  - Years covered: {sorted(years)}")
        logger.info(f"  - Number of events: {df['event_id'].nunique()}")
        logger.info(f"  - Number of drivers: {df['driver'].nunique()}")
        logger.info(f"  - Race mapping success: {df['race_id'].notna().sum()}/{len(df)} ({df['race_id'].notna().sum()/len(df)*100:.1f}%)")
        logger.info(f"  - Average pit stop time: {df['time'].mean():.3f}s")
        logger.info(f"  - Fastest pit stop: {df['time'].min():.3f}s")


def main():
    """Main function to update DHL data with proper F1DB integration"""
    parser = argparse.ArgumentParser(description='Update DHL F1 pit stop data using F1DB integration methodology')
    parser.add_argument('--years', nargs='+', type=int, 
                       help='Years to update (default: 2023 2024)')
    parser.add_argument('--output-dir', help='Output directory for data files')
    
    args = parser.parse_args()
    
    print("🏎️  DHL F1 Pit Stop Data Updater with F1DB Integration")
    print("=" * 70)
    
    # Initialize updater
    updater = DHLDataUpdater(output_dir=args.output_dir)
    
    # Update data
    years = args.years if args.years else [2023, 2024]
    logger.info(f"Updating DHL data for years: {years}")
    
    df = updater.update_dhl_data(years=years)
    
    if not df.empty:
        logger.info(f"\n✅ Update completed successfully!")
        
        # Show sample of the data
        logger.info(f"\n📋 Sample of updated data:")
        print(df.head())
        
        print(f"\n✅ Extraction complete! All data includes F1DB IDs for easy integration.")
    else:
        logger.error(f"\n❌ Update failed - no data retrieved")


if __name__ == "__main__":
    main()