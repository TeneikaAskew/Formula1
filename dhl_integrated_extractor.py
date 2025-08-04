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
from datetime import datetime
from pathlib import Path
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from difflib import SequenceMatcher

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
        self.events_endpoint = f"{self.base_url}/api/f1-award-element-data/6367"
        self.driver_endpoint = f"{self.base_url}/api/f1-award-element-data/6365"
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
            'Haas': 'haas',
            'Sauber': 'sauber'
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
        """Build driver mapping for 2025 season"""
        season_2025 = self.season_drivers_df[self.season_drivers_df['year'] == 2025].copy()
        season_2025 = season_2025.merge(
            self.drivers_df[['id', 'lastName']], 
            left_on='driverId', 
            right_on='id', 
            suffixes=('', '_driver')
        )
        
        driver_mapping = {}
        for _, row in season_2025.iterrows():
            if not pd.isna(row['lastName']) and not row.get('testDriver', False):
                key = (row['lastName'].lower(), row['constructorId'])
                driver_mapping[key] = row['driverId']
        
        # Add special cases
        special_cases = {
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
        }
        driver_mapping.update(special_cases)
        
        return driver_mapping
    
    def extract_events(self):
        """Extract all events and create race mapping"""
        logger.info("Extracting events from DHL API...")
        
        try:
            response = self.session.get(self.events_endpoint)
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
                events_df['year'] = pd.to_datetime(events_df['date_parsed']).dt.year
            
            # Map to F1DB races
            events_df['f1db_race_id'] = events_df.apply(self._map_event_to_race, axis=1)
            
            # Save events with mapping
            events_file = self.output_dir / 'dhl_events_mapped_2025.csv'
            events_df.to_csv(events_file, index=False)
            logger.info(f"✅ Saved {len(events_df)} events with race mapping to {events_file}")
            
            return events_df, data
            
        except Exception as e:
            logger.error(f"Failed to extract events: {e}")
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
    
    def map_driver(self, driver_name, constructor_id):
        """Map driver name to F1DB driver ID"""
        if pd.isna(driver_name) or pd.isna(constructor_id):
            return None
        
        driver_lastname = str(driver_name).lower().strip()
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
    
    def extract_all_driver_data(self, events_df):
        """Extract driver pit stop data for all events with full mapping"""
        logger.info("Extracting driver pit stop data with full F1DB mapping...")
        
        all_driver_data = []
        
        # Process each event
        for idx, event in tqdm(events_df.iterrows(), total=len(events_df), desc="Processing events"):
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
                url = f"{self.driver_endpoint}?event={event_id}"
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
                        driver_id = self.map_driver(stop['driver'], constructor_id)
                        
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
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        driver_file = self.output_dir / f'dhl_pitstops_integrated_{timestamp}.csv'
        driver_df.to_csv(driver_file, index=False)
        logger.info(f"✅ Saved {len(driver_df)} fully integrated pit stops to {driver_file}")
        
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
        print(f"Events extracted: {len(events_df)}")
        print(f"Events mapped to races: {events_df['f1db_race_id'].notna().sum()}/{len(events_df)}")
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
            print(f"  - Pit stops per event: ~{len(driver_df) / len(events_df):.1f}")

def main():
    """Main extraction function"""
    print("🏎️  DHL F1 Pit Stop Integrated Data Extractor")
    print("=" * 70)
    
    extractor = DHLIntegratedExtractor()
    
    # 1. Extract events with race mapping
    events_df, events_data = extractor.extract_events()
    
    if events_df.empty:
        logger.error("No events found, aborting.")
        return
    
    # 2. Extract constructor averages
    constructor_df = extractor.extract_constructor_averages(events_data)
    
    # 3. Extract all driver data with full mapping
    driver_df = extractor.extract_all_driver_data(events_df)
    
    # Show summary
    extractor.show_summary(events_df, constructor_df, driver_df)
    
    print("\n✅ Extraction complete! All data includes F1DB IDs for easy integration.")

if __name__ == "__main__":
    main()