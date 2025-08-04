#!/usr/bin/env python3
"""
DHL Integrated Data Extractor
Extracts DHL pit stop data with full F1DB mappings:
- Driver IDs
- Constructor IDs  
- Race IDs
All in one integrated process
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from difflib import SequenceMatcher
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DHLIntegratedExtractor:
    """Extract all DHL F1 pit stop data with F1DB mappings"""
    
    def __init__(self):
        self.base_url = "https://inmotion.dhl"
        
        # Multiple endpoint configurations for different years
        self.endpoints = {
            2025: {
                'events': f"{self.base_url}/api/f1-award-element-data/6367",
                'drivers': f"{self.base_url}/api/f1-award-element-data/6365"
            },
            2024: {
                'events': f"{self.base_url}/api/f1-award-element-data/6276",
                'drivers': f"{self.base_url}/api/f1-award-element-data/6273"
            },
            2023: {
                'events': f"{self.base_url}/api/f1-award-element-data/6284",
                'drivers': f"{self.base_url}/api/f1-award-element-data/6282"
            }
        }
        
        # Default to 2025 for backward compatibility
        self.events_endpoint = self.endpoints[2025]['events']
        self.driver_endpoint = self.endpoints[2025]['drivers']
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        self.output_dir = Path("data/dhl")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load F1DB reference data
        self.f1db_dir = Path("data/f1db")
        self.drivers_df = pd.read_csv(self.f1db_dir / "drivers.csv")
        self.constructors_df = pd.read_csv(self.f1db_dir / "constructors.csv")
        self.races_df = pd.read_csv(self.f1db_dir / "races.csv")
        self.season_drivers_df = pd.read_csv(self.f1db_dir / "seasons-entrants-drivers.csv")
        
        # Initialize mappings
        self._init_mappings()
        
    def check_data_freshness(self):
        """Check if we need to update data based on race dates"""
        logger.info("Checking data freshness...")
        
        # Check latest integrated data
        integrated_files = list(self.output_dir.glob('dhl_pitstops_integrated_*.csv'))
        if not integrated_files:
            logger.info("No existing integrated data found. Update needed.")
            return True
            
        # Get latest file
        latest_integrated = sorted(integrated_files)[-1]
        df = pd.read_csv(latest_integrated)
        
        if df.empty or 'event_date' not in df.columns:
            logger.info("Existing data is invalid. Update needed.")
            return True
            
        # Get latest event date from integrated data
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        latest_event_date = df['event_date'].max()
        
        logger.info(f"Latest event in integrated data: {latest_event_date.date()}")
        
        # Check all events to find next race
        all_events_df = self.merge_event_files()
        if all_events_df.empty:
            logger.info("No event files found. Update needed.")
            return True
            
        # Convert dates and sort
        if 'date_parsed' in all_events_df.columns:
            all_events_df['event_date'] = pd.to_datetime(all_events_df['date_parsed'], errors='coerce')
        elif 'event_date' not in all_events_df.columns:
            logger.error("No date column found in event files")
            return True
            
        all_events_df = all_events_df.sort_values('event_date')
        
        # Find future events
        today = datetime.now()
        future_events = all_events_df[all_events_df['event_date'] > latest_event_date]
        
        if future_events.empty:
            logger.info("No future events found after latest data.")
            # Check if we're missing current year data
            current_year = today.year
            if current_year not in df['event_date'].dt.year.unique():
                logger.info(f"Missing data for current year {current_year}. Update needed.")
                return True
            return False
            
        # Get next event
        next_event = future_events.iloc[0]
        next_event_date = next_event['event_date']
        
        # Get event name from title or short_title
        event_name = next_event.get('title', next_event.get('short_title', 'Unknown Event'))
        
        logger.info(f"Next event: {event_name} on {next_event_date.date()}")
        logger.info(f"Today's date: {today.date()}")
        
        # Check if next event has passed (with 1 day buffer for processing)
        if today > next_event_date + timedelta(days=1):
            logger.info("Next event has passed. Update needed.")
            return True
        else:
            days_until = (next_event_date - today).days
            logger.info(f"Next event is in {days_until} days. No update needed.")
            return False
    
    def merge_event_files(self):
        """Merge all yearly event files into one DataFrame"""
        event_files = list(self.output_dir.glob('dhl_events_mapped_*.csv'))
        
        if not event_files:
            return pd.DataFrame()
            
        all_events = []
        for file in sorted(event_files):
            df = pd.read_csv(file)
            all_events.append(df)
            
        if all_events:
            merged_df = pd.concat(all_events, ignore_index=True)
            
            # Save merged file
            merged_file = self.output_dir / 'dhl_events_all_years.csv'
            merged_df.to_csv(merged_file, index=False)
            logger.info(f"Saved merged events file: {merged_file}")
            
            return merged_df
        
        return pd.DataFrame()
        
    def _init_mappings(self):
        """Initialize all mapping dictionaries"""
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
            'RB': 'rb',  # Short form in 2024
            'Haas': 'haas',
            'Sauber': 'sauber',
            'AlphaTauri': 'alphatauri'  # 2023 team name
        }
        
        # Circuit mappings
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
        
        # GP name mappings
        self.gp_name_mappings = {
            'AUSTRALIAN': 'australia',
            'CHINESE': 'china',
            'JAPANESE': 'japan',
            'BAHRAIN': 'bahrain',
            'SAUDI ARABIAN': 'saudi-arabia',
            'MIAMI': 'miami',
            'EMILIA-ROMAGNA': 'emilia-romagna',
            'MONACO': 'monaco',
            'ESPAÑA': 'spain',
            'CANADA': 'canada',
            'AUSTRIAN': 'austria',
            'BRITISH': 'great-britain',
            'BELGIAN': 'belgium',
            'HUNGARIAN': 'hungary',
            'DUTCH': 'netherlands',
            'ITALIA': 'italy',
            'AZERBAIJAN': 'azerbaijan',
            'SINGAPORE': 'singapore',
            'UNITED STATES': 'united-states',
            'MÉXICO': 'mexico',
            'SÃO PAULO': 'brazil',
            'LAS VEGAS': 'las-vegas',
            'QATAR': 'qatar',
            'ABU DHABI': 'abu-dhabi'
        }
        
        # Build driver mapping for 2025
        self.driver_mapping = self._build_driver_mapping()
    
    def _build_driver_mapping(self):
        """Build driver mapping for multiple seasons"""
        driver_mapping = {}
        
        # Build mappings for 2023, 2024, and 2025
        for year in [2023, 2024, 2025]:
            season_data = self.season_drivers_df[self.season_drivers_df['year'] == year].copy()
            season_data = season_data.merge(
                self.drivers_df[['id', 'lastName']], 
                left_on='driverId', 
                right_on='id', 
                suffixes=('', '_driver')
            )
            
            for _, row in season_data.iterrows():
                if not pd.isna(row['lastName']) and not row.get('testDriver', False):
                    # Include year in the key for year-specific mappings
                    key = (row['lastName'].lower(), row['constructorId'], year)
                    driver_mapping[key] = row['driverId']
                    # Also add without year for backward compatibility
                    key_no_year = (row['lastName'].lower(), row['constructorId'])
                    driver_mapping[key_no_year] = row['driverId']
        
        # Add special cases for all years
        special_cases = {
            # 2025 special cases
            ('antonelli', 'mercedes'): 'andrea-kimi-antonelli',
            ('bearman', 'haas'): 'oliver-bearman',
            ('bortoleto', 'sauber'): 'gabriel-bortoleto',
            ('lawson', 'red-bull'): 'liam-lawson',
            ('lawson', 'rb'): 'liam-lawson',
            ('hadjar', 'rb'): 'isack-hadjar',
            ('tsunoda', 'rb'): 'yuki-tsunoda',
            ('tsunoda', 'red-bull'): 'yuki-tsunoda',
            ('hulkenberg', 'sauber'): 'nico-hulkenberg',
            ('sainz', 'williams'): 'carlos-sainz-jr',
            
            # 2023-2024 special cases
            ('ricciardo', 'alphatauri'): 'daniel-ricciardo',
            ('ricciardo', 'rb'): 'daniel-ricciardo',
            ('tsunoda', 'alphatauri'): 'yuki-tsunoda',
            ('piastri', 'mclaren'): 'oscar-piastri',
            ('doohan', 'alpine'): 'jack-doohan',
            ('de-vries', 'alphatauri'): 'nyck-de-vries',
            
            # Year-specific mappings
            ('ricciardo', 'alphatauri', 2023): 'daniel-ricciardo',
            ('ricciardo', 'rb', 2024): 'daniel-ricciardo',
            ('tsunoda', 'alphatauri', 2023): 'yuki-tsunoda',
            ('tsunoda', 'rb', 2024): 'yuki-tsunoda',
        }
        driver_mapping.update(special_cases)
        
        return driver_mapping
    
    def extract_events(self, year=None):
        """Extract all events and create race mapping for a specific year"""
        if year is None:
            year = 2025  # Default to 2025 for backward compatibility
            
        logger.info(f"Extracting events from DHL API for {year}...")
        
        try:
            events_endpoint = self.endpoints[year]['events']
            response = self.session.get(events_endpoint)
            response.raise_for_status()
            
            data = response.json()
            events = data.get('data', {}).get('chart', {}).get('events', [])
            
            # Convert to DataFrame
            events_df = pd.DataFrame(events)
            
            # Parse date field
            if not events_df.empty and 'date' in events_df.columns:
                events_df['date_parsed'] = events_df['date'].apply(
                    lambda x: x.get('date', '').split(' ')[0] if isinstance(x, dict) else ''
                )
                events_df['year'] = year
            
            # Map to F1DB races
            events_df['f1db_race_id'] = events_df.apply(self._map_event_to_race, axis=1)
            
            # Save events with mapping
            events_file = self.output_dir / f'dhl_events_mapped_{year}.csv'
            events_df.to_csv(events_file, index=False)
            logger.info(f"✅ Saved {len(events_df)} events with race mapping to {events_file}")
            
            return events_df, data
            
        except Exception as e:
            logger.error(f"Failed to extract events for {year}: {e}")
            return pd.DataFrame(), None
    
    def _map_event_to_race(self, event_row):
        """Map a DHL event to F1DB race ID"""
        event_date = event_row['date_parsed']
        event_circuit = event_row['short_title']
        event_year = event_row.get('year', 2025)
        
        # Filter races by year
        year_races = self.races_df[self.races_df['year'] == event_year]
        
        # Try circuit mapping first
        circuit_id = self.circuit_mappings.get(event_circuit)
        if circuit_id:
            matches = year_races[year_races['circuitId'] == circuit_id]
            if len(matches) == 1:
                return matches.iloc[0]['id']
        
        # Try date matching (within 2 days)
        event_date_obj = pd.to_datetime(event_date)
        for _, race in year_races.iterrows():
            race_date_obj = pd.to_datetime(race['date'])
            if abs((event_date_obj - race_date_obj).days) <= 2:
                if not circuit_id or race['circuitId'] == circuit_id:
                    return race['id']
        
        return None
    
    def map_team_to_constructor(self, team_name):
        """Map DHL team name to F1DB constructor ID"""
        if pd.isna(team_name):
            return None
        
        team_lower = str(team_name).lower()
        for dhl_name, constructor_id in self.team_mappings.items():
            if dhl_name.lower() in team_lower:
                return constructor_id
        
        return None
    
    def map_driver(self, driver_name, constructor_id, year=None):
        """Map driver name to F1DB driver ID"""
        if pd.isna(driver_name) or pd.isna(constructor_id):
            return None
        
        driver_lastname = str(driver_name).lower().strip()
        
        # Try year-specific mapping first
        if year:
            key_with_year = (driver_lastname, constructor_id, year)
            if key_with_year in self.driver_mapping:
                return self.driver_mapping[key_with_year]
        
        # Fall back to general mapping
        key = (driver_lastname, constructor_id)
        return self.driver_mapping.get(key)
    
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
    
    def extract_all_driver_data(self, events_df, year=None):
        """Extract driver pit stop data for all events with full mapping"""
        if year is None:
            year = 2025  # Default to 2025 for backward compatibility
            
        logger.info(f"Extracting driver pit stop data for {year} with full F1DB mapping...")
        
        all_driver_data = []
        drivers_endpoint = self.endpoints[year]['drivers']
        
        # Process each event
        for idx, event in tqdm(events_df.iterrows(), total=len(events_df), desc=f"Processing {year} events"):
            event_id = event['id']
            event_name = event['title']
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
                        
                        # Map driver with year
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
        
        # Don't save here - let the main function handle saving all years together
        logger.info(f"✅ Extracted {len(driver_df)} fully integrated pit stops for {year}")
        
        return driver_df
    
    def extract_constructor_averages(self, events_data):
        """Extract constructor average pit stop times"""
        logger.info("Extracting constructor averages...")
        
        try:
            values = events_data.get('data', {}).get('chart', {}).get('values', [])
            
            constructor_data = []
            
            for team_data in values:
                team_name = team_data.get('team_name', '')
                constructor_id = self.map_team_to_constructor(team_name)
                durations = team_data.get('duration', {})
                
                for event_id, avg_duration in durations.items():
                    if avg_duration is not None:
                        constructor_data.append({
                            'team_name': team_name,
                            'constructor_id': constructor_id,
                            'event_id': event_id,
                            'average_duration': avg_duration
                        })
            
            # Convert to DataFrame
            constructor_df = pd.DataFrame(constructor_data)
            
            # Save to CSV
            constructor_file = self.output_dir / 'dhl_constructor_averages_mapped_2025.csv'
            constructor_df.to_csv(constructor_file, index=False)
            logger.info(f"✅ Saved {len(constructor_df)} constructor averages to {constructor_file}")
            
            return constructor_df
            
        except Exception as e:
            logger.error(f"Failed to extract constructor averages: {e}")
            return pd.DataFrame()
    
    def show_summary(self, events_df, constructor_df, driver_df):
        """Show summary of extracted data"""
        print("\n📊 EXTRACTION SUMMARY")
        print("=" * 50)
        
        if isinstance(events_df, dict):
            # Multiple years
            total_events = sum(len(df) for df in events_df.values())
            mapped_events = sum(df['f1db_race_id'].notna().sum() for df in events_df.values())
            print(f"Events extracted: {total_events}")
            print(f"Events mapped to races: {mapped_events}/{total_events}")
            for year, df in events_df.items():
                print(f"  - {year}: {len(df)} events ({df['f1db_race_id'].notna().sum()} mapped)")
        else:
            print(f"Events extracted: {len(events_df)}")
            print(f"Events mapped to races: {events_df['f1db_race_id'].notna().sum()}/{len(events_df)}")
        
        if not constructor_df.empty:
            print(f"Constructor averages: {len(constructor_df)}")
        
        print(f"Driver pit stops: {len(driver_df)}")
        
        if not driver_df.empty:
            print(f"\nMapping success rates:")
            print(f"  - Drivers mapped: {driver_df['driver_id'].notna().sum()}/{len(driver_df)} ({driver_df['driver_id'].notna().sum()/len(driver_df)*100:.1f}%)")
            print(f"  - Constructors mapped: {driver_df['constructor_id'].notna().sum()}/{len(driver_df)} ({driver_df['constructor_id'].notna().sum()/len(driver_df)*100:.1f}%)")
            print(f"  - Races mapped: {driver_df['race_id'].notna().sum()}/{len(driver_df)} ({driver_df['race_id'].notna().sum()/len(driver_df)*100:.1f}%)")
            
            print(f"\nPit stop statistics:")
            print(f"  - Fastest pit stop: {driver_df['time'].min():.2f}s")
            print(f"  - Average pit stop: {driver_df['time'].mean():.2f}s")
            
            if 'event_date' in driver_df.columns:
                years = pd.to_datetime(driver_df['event_date'], errors='coerce').dt.year.dropna().unique()
                print(f"  - Years covered: {sorted(years)}")
    
    def extract_all_years(self, years=None):
        """Extract data for multiple years"""
        if years is None:
            years = [2023, 2024, 2025]
        
        all_events = {}
        all_driver_data = []
        
        for year in years:
            logger.info(f"\n🏎️ Processing year {year}...")
            
            # 1. Extract events
            events_df, events_data = self.extract_events(year)
            
            if events_df.empty:
                logger.warning(f"No events found for {year}")
                continue
            
            all_events[year] = events_df
            
            # 2. Extract driver data
            driver_df = self.extract_all_driver_data(events_df, year)
            
            if not driver_df.empty:
                all_driver_data.append(driver_df)
        
        # Combine all driver data
        if all_driver_data:
            combined_df = pd.concat(all_driver_data, ignore_index=True)
            
            # Save combined data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f'dhl_pitstops_integrated_{timestamp}.csv'
            combined_df.to_csv(output_file, index=False)
            logger.info(f"✅ Saved {len(combined_df)} total pit stops to {output_file}")
            
            # Show summary
            self.show_summary(all_events, pd.DataFrame(), combined_df)
            
            return combined_df
        else:
            logger.error("No data extracted")
            return pd.DataFrame()

def main():
    """Main extraction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DHL F1 Pit Stop Integrated Data Extractor')
    parser.add_argument('--years', nargs='+', type=int, 
                       help='Years to extract (default: 2023 2024 2025)')
    parser.add_argument('--year', type=int, 
                       help='Single year to extract (for backward compatibility)')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if data is fresh')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if update is needed, do not extract')
    
    args = parser.parse_args()
    
    print("🏎️  DHL F1 Pit Stop Integrated Data Extractor")
    print("=" * 70)
    
    extractor = DHLIntegratedExtractor()
    
    # Check data freshness first
    if not args.force:
        needs_update = extractor.check_data_freshness()
        
        if args.check_only:
            # Exit with appropriate code for scripts
            sys.exit(0 if needs_update else 1)
        
        if not needs_update:
            print("\n✅ Data is already up to date. Use --force to update anyway.")
            return
    
    # Determine which years to extract
    if args.year:
        # Single year mode (backward compatibility)
        year = args.year
        logger.info(f"Extracting data for single year: {year}")
        
        # 1. Extract events with race mapping
        events_df, events_data = extractor.extract_events(year)
        
        if events_df.empty:
            logger.error(f"No events found for {year}, aborting.")
            return
        
        # 2. Extract constructor averages
        constructor_df = extractor.extract_constructor_averages(events_data)
        
        # 3. Extract all driver data with full mapping
        driver_df = extractor.extract_all_driver_data(events_df, year)
        
        # Save the data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = extractor.output_dir / f'dhl_pitstops_integrated_{timestamp}.csv'
        driver_df.to_csv(output_file, index=False)
        logger.info(f"✅ Saved {len(driver_df)} pit stops to {output_file}")
        
        # Show summary
        extractor.show_summary(events_df, constructor_df, driver_df)
    else:
        # Multiple years mode
        years = args.years if args.years else [2023, 2024, 2025]
        logger.info(f"Extracting data for years: {years}")
        
        # Extract all years
        combined_df = extractor.extract_all_years(years)
    
    print("\n✅ Extraction complete! All data includes F1DB IDs for easy integration.")

if __name__ == "__main__":
    main()