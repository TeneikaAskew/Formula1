#!/usr/bin/env python3
"""
Enhanced DHL F1 Pit Stop Data Scraper using API endpoints
Extracts pit stop data directly from DHL API with proper F1DB ID mapping
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import os
import logging
from bs4 import BeautifulSoup
import re
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DHLAPIScraper:
    """API-based scraper for DHL F1 Pit Stop Award data"""
    
    def __init__(self):
        self.base_url = "https://inmotion.dhl"
        self.events_endpoint = f"{self.base_url}/api/f1-award-element-data/6367"
        self.data_endpoint = f"{self.base_url}/api/f1-award-element-data/6365"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.all_data = []
        self.f1db_data = self._load_f1db_data()
        
    def _load_f1db_data(self):
        """Load F1DB reference data for mapping"""
        f1db_data = {
            'drivers': pd.DataFrame(),
            'constructors': pd.DataFrame(),
            'races': pd.DataFrame(),
            'circuits': pd.DataFrame()
        }
        
        # Try different data paths
        data_paths = ["data/f1db", "../data/f1db", "../../data/f1db"]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    for file_type in ['drivers', 'constructors', 'races', 'circuits']:
                        file_path = os.path.join(path, f'{file_type}.csv')
                        if os.path.exists(file_path):
                            f1db_data[file_type] = pd.read_csv(file_path)
                            logger.debug(f"Loaded {file_type} from {file_path}")
                    
                    if not f1db_data['drivers'].empty:
                        logger.info(f"✓ Loaded F1DB data from {path}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Error loading F1DB data from {path}: {e}")
        
        return f1db_data
    
    def get_available_events(self):
        """Get all available race events from DHL API"""
        try:
            logger.info("Fetching available events from DHL API...")
            response = self.session.get(self.events_endpoint)
            response.raise_for_status()
            
            # Parse HTML to extract event options
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all radio input elements for events
            event_inputs = soup.find_all('input', {'type': 'radio', 'class': 'form-element'})
            
            events = []
            for input_elem in event_inputs:
                event_id = input_elem.get('value')
                if event_id:
                    # Find the corresponding label
                    label = soup.find('label', {'for': input_elem.get('id')})
                    if label:
                        # Extract race name from the span inside label
                        race_spans = label.find_all('span')
                        race_name = None
                        for span in race_spans:
                            # Skip flag spans (they have fi fi-xx classes)
                            if not span.get('class') or 'fi' not in ' '.join(span.get('class')):
                                race_name = span.get_text(strip=True)
                                break
                        
                        if race_name:
                            events.append({
                                'id': event_id,
                                'name': race_name
                            })
            
            logger.info(f"Found {len(events)} available events")
            for event in events:
                logger.debug(f"Event {event['id']}: {event['name']}")
                
            return events
            
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []
    
    def get_event_data(self, event_id, event_name):
        """Get pit stop data for a specific event"""
        try:
            url = f"{self.data_endpoint}?event={event_id}"
            logger.info(f"Fetching data for event {event_id}: {event_name}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse HTML to extract table data
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the pit stop table
            table = soup.find('table', {'class': 'f1-award-table'})
            if not table:
                logger.warning(f"No table found for event {event_id}")
                return []
            
            # Extract data from table rows
            tbody = table.find('tbody')
            if not tbody:
                logger.warning(f"No tbody found for event {event_id}")
                return []
            
            rows = tbody.find_all('tr')
            pit_stops = []
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 6:
                    try:
                        # Extract cell text
                        cell_texts = [cell.get_text(strip=True) for cell in cells]
                        
                        # Parse position
                        pos_text = cell_texts[0]
                        if not pos_text or not pos_text.isdigit():
                            continue
                        
                        # Parse other fields
                        position = int(pos_text)
                        team = cell_texts[1]
                        driver = cell_texts[2]
                        time_str = cell_texts[3]
                        lap_str = cell_texts[4]
                        points_str = cell_texts[5].replace('*', '').strip()
                        
                        # Validate essential fields
                        if not time_str:
                            continue
                            
                        pit_stop = {
                            'position': position,
                            'team': team,
                            'driver': driver,
                            'time': float(time_str),
                            'lap': int(lap_str) if lap_str and lap_str.isdigit() else None,
                            'points': int(points_str) if points_str and points_str.isdigit() else 0,
                            'event_id': event_id,
                            'race_name': event_name
                        }
                        
                        pit_stops.append(pit_stop)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing row: {e}")
                        continue
            
            logger.info(f"✓ Extracted {len(pit_stops)} pit stops from {event_name}")
            return pit_stops
            
        except Exception as e:
            logger.error(f"Failed to fetch data for event {event_id}: {e}")
            return []
    
    def _map_dhl_race_to_f1db(self, dhl_race_name, year=None):
        """Map DHL race name to F1DB race ID using circuit mapping"""
        if self.f1db_data['races'].empty or self.f1db_data['circuits'].empty:
            return None, None
        
        year = year or datetime.now().year
        
        # DHL race name to circuit ID mapping
        circuit_mappings = {
            'Albert Park Grand Prix Circuit': 'melbourne',
            'Shanghai International Circuit': 'shanghai',
            'Suzuka Circuit': 'suzuka',
            'Bahrain International Circuit': 'bahrain',
            'Jeddah Corniche Circuit': 'jeddah',
            'Miami International Autodrome': 'miami',
            'Autodromo Internazionale Enzo e Dino Ferrari': 'imola',
            'Circuit de Monaco': 'monaco',
            'Circuit de Barcelona-Catalunya': 'barcelona',
            'Circuit Gilles-Villeneuve': 'villeneuve',
            'Red Bull Ring': 'red-bull-ring',
            'Silverstone Circuit': 'silverstone',  
            'Circuit de Spa-Francorchamps': 'spa',
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
        
        # Get circuit ID
        circuit_id = circuit_mappings.get(dhl_race_name)
        
        if not circuit_id:
            logger.warning(f"No circuit mapping found for: {dhl_race_name}")
            return None, None
        
        # Find race in F1DB for this circuit and year
        races = self.f1db_data['races']
        race_match = races[(races['circuitId'] == circuit_id) & (races['year'] == year)]
        
        if not race_match.empty:
            race_id = race_match.iloc[0]['id']
            logger.debug(f"Mapped {dhl_race_name} -> circuit:{circuit_id} -> race:{race_id}")
            return race_id, circuit_id
        else:
            logger.warning(f"No race found for circuit {circuit_id} in year {year}")
            return None, circuit_id
    
    def _map_driver_to_id(self, driver_name, team_name=None):
        """Map driver name to F1DB driver ID"""
        if self.f1db_data['drivers'].empty or not driver_name:
            return None
        
        drivers = self.f1db_data['drivers']
        driver_name_lower = driver_name.lower().strip()
        
        # Special mappings for current drivers
        special_mappings = {
            'verstappen': 'max-verstappen',
            'perez': 'sergio-perez',
            'pérez': 'sergio-perez',
            'hamilton': 'lewis-hamilton',
            'russell': 'george-russell',
            'leclerc': 'charles-leclerc',
            'sainz': 'carlos-sainz-jr',
            'norris': 'lando-norris',
            'piastri': 'oscar-piastri',
            'alonso': 'fernando-alonso',
            'stroll': 'lance-stroll',
            'ocon': 'esteban-ocon',
            'gasly': 'pierre-gasly',
            'albon': 'alex-albon',
            'sargeant': 'logan-sargeant',
            'colapinto': 'franco-colapinto',
            'bottas': 'valtteri-bottas',
            'zhou': 'guanyu-zhou',
            'magnussen': 'kevin-magnussen',
            'hulkenberg': 'nico-hulkenberg',
            'hülkenberg': 'nico-hulkenberg',
            'ricciardo': 'daniel-ricciardo',
            'tsunoda': 'yuki-tsunoda',
            'lawson': 'liam-lawson',
            'bearman': 'oliver-bearman',
            'antonelli': 'andrea-kimi-antonelli',
            'bortoleto': 'gabriel-bortoleto',
            'hadjar': 'isack-hadjar',
            'doohan': 'jack-doohan'
        }
        
        # Try last name mapping first
        parts = driver_name.split()
        if parts:
            last_name = parts[-1].lower()
            if last_name in special_mappings:
                return special_mappings[last_name]
        
        # Try exact match
        exact_match = drivers[drivers['name'].str.lower() == driver_name_lower]
        if not exact_match.empty:
            return exact_match.iloc[0]['id']
        
        # Try last name match in database
        if parts:
            last_name_match = drivers[drivers['lastName'].str.lower() == parts[-1].lower()]
            if not last_name_match.empty:
                return last_name_match.iloc[0]['id']
        
        return None
    
    def _map_team_to_constructor_id(self, team_name):
        """Map team name to F1DB constructor ID"""
        if not team_name:
            return None
        
        team_name_lower = team_name.lower().strip()
        
        # Current team mappings
        team_mappings = {
            'red bull': 'red-bull',
            'red bull racing': 'red-bull',
            'mercedes': 'mercedes',
            'mercedes amg': 'mercedes',
            'ferrari': 'ferrari',
            'mclaren': 'mclaren',
            'aston martin': 'aston-martin',
            'alpine': 'alpine',
            'williams': 'williams',
            'alphatauri': 'alphatauri',
            'rb': 'rb',
            'visa rb': 'rb',
            'racing bulls': 'rb',  # New name for RB
            'alfa romeo': 'alfa-romeo',
            'sauber': 'sauber',
            'haas': 'haas'
        }
        
        for key, value in team_mappings.items():
            if key in team_name_lower:
                return value
        
        return None
    
    def scrape_all_events(self):
        """Scrape pit stop data for all available events"""
        logger.info("🏎️  DHL F1 Pit Stop API Scraper")
        logger.info("=" * 50)
        
        # Get all available events
        events = self.get_available_events()
        if not events:
            logger.error("No events found!")
            return
        
        # Check existing data to avoid re-scraping
        existing_data = self._get_existing_data()
        
        # Process each event
        processed = 0
        for event in events:
            event_id = event['id']
            event_name = event['name']
            
            # Check if we already have this data
            if self._already_have_event(existing_data, event_name):
                logger.info(f"Skipping {event_name} - already have data")
                continue
            
            # Get pit stop data for this event
            pit_stops = self.get_event_data(event_id, event_name)
            
            if pit_stops:
                # Add mappings to each pit stop
                year = datetime.now().year
                race_id, circuit_id = self._map_dhl_race_to_f1db(event_name, year)
                
                for stop in pit_stops:
                    # Add F1DB mappings
                    stop['driver_id'] = self._map_driver_to_id(stop['driver'], stop['team'])
                    stop['constructor_id'] = self._map_team_to_constructor_id(stop['team'])
                    stop['race_id'] = race_id
                    stop['circuit_id'] = circuit_id
                    stop['year'] = year
                    
                    self.all_data.append(stop)
                
                processed += 1
                
        logger.info(f"✅ Processed {processed} events, extracted {len(self.all_data)} pit stops")
    
    def _get_existing_data(self):
        """Get existing data to avoid re-scraping"""
        try:
            data_dir = Path("data/dhl")
            if not data_dir.exists():
                return pd.DataFrame()
            
            csv_files = list(data_dir.glob("*.csv"))
            if not csv_files:
                return pd.DataFrame()
            
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            return pd.read_csv(latest_file)
            
        except Exception as e:
            logger.debug(f"Could not load existing data: {e}")
            return pd.DataFrame()
    
    def _already_have_event(self, existing_data, event_name):
        """Check if we already have data for this event"""
        if existing_data.empty:
            return False
        
        # Check both 'race' and 'race_name' columns
        for col in ['race', 'race_name']:
            if col in existing_data.columns:
                if event_name in existing_data[col].values:
                    return True
        
        return False
    
    def save_data(self, filename=None, format='csv'):
        """Save scraped data to file"""
        if not self.all_data:
            logger.warning("No data to save!")
            return
        
        df = pd.DataFrame(self.all_data)
        
        # Create output directory
        output_dir = Path("data/dhl")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dhl_pitstops_api_{timestamp}"
        
        # Save data
        if format == 'csv':
            filepath = output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = output_dir / f"{filename}.json"
            df.to_json(filepath, orient='records', indent=2)
        else:
            filepath = output_dir / f"{filename}.xlsx"
            df.to_excel(filepath, index=False)
        
        logger.info(f"📁 Saved {len(df)} records to {filepath}")
        
        # Show mapping statistics
        self.show_mapping_stats(df)
        
        return filepath
    
    def show_mapping_stats(self, df):
        """Show mapping statistics"""
        print("\n🔗 ID MAPPING STATISTICS")
        print("-" * 40)
        print(f"Total records: {len(df)}")
        print(f"Mapped drivers: {df['driver_id'].notna().sum()}/{len(df)} ({df['driver_id'].notna().sum()/len(df)*100:.1f}%)")
        print(f"Mapped constructors: {df['constructor_id'].notna().sum()}/{len(df)} ({df['constructor_id'].notna().sum()/len(df)*100:.1f}%)")
        print(f"Mapped races: {df['race_id'].notna().sum()}/{len(df)} ({df['race_id'].notna().sum()/len(df)*100:.1f}%)")
        
        # Show unmapped items
        unmapped_drivers = df[df['driver_id'].isna()]['driver'].unique()
        if len(unmapped_drivers) > 0:
            print(f"\nUnmapped drivers: {', '.join(unmapped_drivers[:5])}")
            if len(unmapped_drivers) > 5:
                print(f"... and {len(unmapped_drivers) - 5} more")
        
        # Show race coverage
        races_covered = df['race_name'].nunique()
        print(f"\nRaces covered: {races_covered}")
        
    def show_summary(self):
        """Show summary of scraped data"""
        if not self.all_data:
            return
        
        df = pd.DataFrame(self.all_data)
        
        print("\n📊 SCRAPING SUMMARY")
        print("=" * 50)
        print(f"Total pit stops: {len(df)}")
        print(f"Races covered: {df['race_name'].nunique()}")
        print(f"Fastest pit stop: {df['time'].min():.2f}s")
        print(f"Average pit stop: {df['time'].mean():.2f}s")
        
        print("\n🏆 TOP 5 FASTEST STOPS")
        print("-" * 50)
        fastest = df.nsmallest(5, 'time')
        for _, stop in fastest.iterrows():
            print(f"{stop['driver']:<15} | {stop['team']:<12} | {stop['time']:.2f}s | {stop['race_name']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Scrape DHL F1 Pit Stop Data via API")
    parser.add_argument("--output", help="Output filename (without extension)")
    parser.add_argument("--format", choices=['csv', 'json', 'excel'], default='csv')
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run scraper
    scraper = DHLAPIScraper()
    scraper.scrape_all_events()
    
    if scraper.all_data:
        scraper.save_data(filename=args.output, format=args.format)
        
        if args.summary:
            scraper.show_summary()
    else:
        logger.error("No data was scraped!")

if __name__ == "__main__":
    main()