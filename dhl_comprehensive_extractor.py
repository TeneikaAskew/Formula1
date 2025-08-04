#!/usr/bin/env python3
"""
Comprehensive DHL F1 Pit Stop Data Extractor
Extracts:
1. Events and IDs
2. Constructor average pit stop times by event
3. All driver pit stop data (not just top 10)
"""

import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DHLDataExtractor:
    """Extract all DHL F1 pit stop data"""
    
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
    
    def extract_events(self):
        """Extract all events and save to CSV"""
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
                
                # Extract year
                events_df['year'] = pd.to_datetime(events_df['date_parsed']).dt.year
            
            # Save to CSV
            events_file = self.output_dir / 'dhl_events_2025.csv'
            events_df.to_csv(events_file, index=False)
            logger.info(f"✅ Saved {len(events_df)} events to {events_file}")
            
            return events_df, data
            
        except Exception as e:
            logger.error(f"Failed to extract events: {e}")
            return pd.DataFrame(), None
    
    def extract_constructor_averages(self, events_data):
        """Extract constructor average pit stop times"""
        logger.info("Extracting constructor averages...")
        
        try:
            # Get values section which contains constructor data
            values = events_data.get('data', {}).get('chart', {}).get('values', [])
            
            constructor_data = []
            
            for team_data in values:
                team_name = team_data.get('team_name', '')
                durations = team_data.get('duration', {})
                
                # Extract data for each event
                for event_id, avg_duration in durations.items():
                    if avg_duration is not None:
                        constructor_data.append({
                            'team_name': team_name,
                            'event_id': event_id,
                            'average_duration': avg_duration
                        })
            
            # Convert to DataFrame
            constructor_df = pd.DataFrame(constructor_data)
            
            # Save to CSV
            constructor_file = self.output_dir / 'dhl_constructor_averages_2025.csv'
            constructor_df.to_csv(constructor_file, index=False)
            logger.info(f"✅ Saved {len(constructor_df)} constructor averages to {constructor_file}")
            
            return constructor_df
            
        except Exception as e:
            logger.error(f"Failed to extract constructor averages: {e}")
            return pd.DataFrame()
    
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
        """Extract driver pit stop data for all events"""
        logger.info("Extracting driver pit stop data for all events...")
        
        all_driver_data = []
        
        # Process each event
        for idx, event in tqdm(events_df.iterrows(), total=len(events_df), desc="Processing events"):
            event_id = event['id']
            event_name = event['title']
            event_date = event.get('date_parsed', '')
            
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
                    
                    # Add event information to each pit stop
                    for stop in pit_stops:
                        stop['event_id'] = event_id
                        stop['event_name'] = event_name
                        stop['event_date'] = event_date
                        stop['event_abbr'] = event.get('abbr', '')
                        all_driver_data.append(stop)
                    
                    logger.debug(f"Extracted {len(pit_stops)} pit stops for {event_name}")
                else:
                    # Fallback to JSON data (top 10 only)
                    drivers = data.get('data', {}).get('chart', [])
                    for driver in drivers:
                        pit_stop = {
                            'position': None,  # Not in JSON data
                            'team': driver.get('team', ''),
                            'driver': f"{driver.get('firstName', '')} {driver.get('lastName', '')}",
                            'time': driver.get('duration', 0),
                            'lap': driver.get('lap', 0),
                            'points': driver.get('points', 0),
                            'event_id': event_id,
                            'event_name': event_name,
                            'event_date': event_date,
                            'event_abbr': event.get('abbr', ''),
                            'driver_number': driver.get('driverNr', None),
                            'tla': driver.get('tla', '')
                        }
                        all_driver_data.append(pit_stop)
                    
                    logger.warning(f"Using JSON data for {event_name} (top 10 only)")
                
            except Exception as e:
                logger.error(f"Failed to extract data for event {event_id} ({event_name}): {e}")
                continue
        
        # Convert to DataFrame
        driver_df = pd.DataFrame(all_driver_data)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        driver_file = self.output_dir / f'dhl_all_driver_pitstops_{timestamp}.csv'
        driver_df.to_csv(driver_file, index=False)
        logger.info(f"✅ Saved {len(driver_df)} driver pit stops to {driver_file}")
        
        return driver_df
    
    def show_summary(self, events_df, constructor_df, driver_df):
        """Show summary of extracted data"""
        print("\n📊 EXTRACTION SUMMARY")
        print("=" * 50)
        print(f"Events extracted: {len(events_df)}")
        print(f"Constructor averages: {len(constructor_df)} (teams × events)")
        print(f"Driver pit stops: {len(driver_df)}")
        
        if not driver_df.empty:
            print(f"\nPit stops per event: ~{len(driver_df) / len(events_df):.1f}")
            print(f"Fastest pit stop: {driver_df['time'].min():.2f}s")
            print(f"Average pit stop: {driver_df['time'].mean():.2f}s")
            
            print("\n🏆 TOP 5 FASTEST STOPS OF 2025")
            print("-" * 70)
            fastest = driver_df.nsmallest(5, 'time')
            for _, stop in fastest.iterrows():
                print(f"{stop['driver']:<20} | {stop['team']:<15} | {stop['time']:.2f}s | {stop['event_name']}")

def main():
    """Main extraction function"""
    print("🏎️  DHL F1 Pit Stop Data Comprehensive Extractor")
    print("=" * 70)
    
    extractor = DHLDataExtractor()
    
    # 1. Extract events
    events_df, events_data = extractor.extract_events()
    
    if events_df.empty:
        logger.error("No events found, aborting.")
        return
    
    # 2. Extract constructor averages
    constructor_df = extractor.extract_constructor_averages(events_data)
    
    # 3. Extract all driver data
    driver_df = extractor.extract_all_driver_data(events_df)
    
    # Show summary
    extractor.show_summary(events_df, constructor_df, driver_df)
    
    print("\n✅ Extraction complete! Check the data/dhl directory for CSV files.")

if __name__ == "__main__":
    main()