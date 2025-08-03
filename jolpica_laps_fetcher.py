#!/usr/bin/env python3
"""
Jolpica F1 API Laps Data Fetcher

This script fetches lap-by-lap timing data from the Jolpica F1 API
for the past 5 years (2020-2024) to enhance overtake analysis.

API Documentation: https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/laps.md
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JolpicaLapsFetcher:
    """Fetch lap timing data from Jolpica F1 API with rate limiting"""
    
    BASE_URL = "http://api.jolpi.ca/ergast/f1"
    
    # Rate limits from documentation
    BURST_LIMIT = 4  # requests per second
    SUSTAINED_LIMIT = 500  # requests per hour
    
    # Exponential backoff settings
    INITIAL_BACKOFF = 1.0  # Initial backoff in seconds
    MAX_BACKOFF = 300.0  # Maximum backoff (5 minutes)
    BACKOFF_MULTIPLIER = 2.0  # Exponential multiplier
    
    def __init__(self, data_dir: str = "data/jolpica"):
        """Initialize the fetcher with data directory"""
        self.data_dir = Path(data_dir)
        self.laps_dir = self.data_dir / "laps"
        self.metadata_file = self.data_dir / "fetch_metadata.json"
        
        # Create directories
        self.laps_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata if exists
        self.metadata = self._load_metadata()
        
        # Rate limiting tracking
        self.request_times = deque(maxlen=self.SUSTAINED_LIMIT)
        self.last_request_time = 0
        self.current_backoff = self.INITIAL_BACKOFF
        self.consecutive_429s = 0
        
    def _load_metadata(self) -> Dict:
        """Load metadata about previous fetches"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "last_fetch": None,
            "fetched_races": {},
            "errors": []
        }
    
    def _save_metadata(self):
        """Save metadata about fetches"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _wait_for_rate_limit(self):
        """Enforce rate limits before making a request"""
        current_time = time.time()
        
        # Check burst limit (4 per second)
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.BURST_LIMIT):
            sleep_time = (1.0 / self.BURST_LIMIT) - time_since_last
            time.sleep(sleep_time)
        
        # Check sustained limit (500 per hour)
        # Remove requests older than 1 hour
        hour_ago = current_time - 3600
        while self.request_times and self.request_times[0] < hour_ago:
            self.request_times.popleft()
        
        # If we're at the hourly limit, wait
        if len(self.request_times) >= self.SUSTAINED_LIMIT:
            oldest_request = self.request_times[0]
            wait_time = (oldest_request + 3600) - current_time + 1
            if wait_time > 0:
                logger.warning(f"Hourly rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    def _handle_rate_limit_error(self):
        """Handle 429 error with exponential backoff"""
        self.consecutive_429s += 1
        
        # Calculate backoff time
        backoff_time = min(
            self.current_backoff * (self.BACKOFF_MULTIPLIER ** (self.consecutive_429s - 1)),
            self.MAX_BACKOFF
        )
        
        logger.warning(f"Rate limited (429). Waiting {backoff_time:.1f}s (attempt {self.consecutive_429s})")
        time.sleep(backoff_time)
        
        # Update backoff for next time
        self.current_backoff = min(self.current_backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
    
    def _reset_backoff(self):
        """Reset backoff after successful request"""
        if self.consecutive_429s > 0:
            logger.info("Rate limit cleared, resetting backoff")
        self.consecutive_429s = 0
        self.current_backoff = self.INITIAL_BACKOFF
    
    def _make_request(self, url: str, timeout: int = 60) -> requests.Response:
        """Make a request with rate limiting and retry logic"""
        max_attempts = 5
        
        for attempt in range(max_attempts):
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            try:
                # Make request
                response = requests.get(url, timeout=timeout)
                
                # Track request time
                request_time = time.time()
                self.request_times.append(request_time)
                self.last_request_time = request_time
                
                # Handle rate limit error
                if response.status_code == 429:
                    error_msg = f"429 Rate Limited - {response.reason}"
                    if response.text:
                        error_msg += f" - Response: {response.text[:200]}"
                    logger.error(f"FULL 429 ERROR: {error_msg}")
                    self._handle_rate_limit_error()
                    continue
                
                # Success - reset backoff
                self._reset_backoff()
                
                # Raise for other HTTP errors
                response.raise_for_status()
                
                return response
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # Simple exponential backoff for timeouts
                
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {max_attempts} attempts")
    
    def get_season_info(self, year: int) -> Optional[Dict]:
        """Get information about races in a season"""
        url = f"{self.BASE_URL}/{year}.json"
        
        try:
            response = self._make_request(url, timeout=30)
            data = response.json()
            
            races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            logger.info(f"Found {len(races)} races in {year} season")
            
            return {
                'year': year,
                'total_races': len(races),
                'races': [
                    {
                        'round': int(race['round']),
                        'name': race['raceName'],
                        'date': race['date'],
                        'circuit': race['Circuit']['circuitName']
                    }
                    for race in races
                ]
            }
            
        except Exception as e:
            logger.error(f"Error fetching season {year}: {e}")
            return None
    
    def fetch_race_laps(self, year: int, round_num: int) -> Optional[Dict]:
        """Fetch all laps data for a specific race with pagination support"""
        all_laps = []
        offset = 0
        limit = 2000  # Maximum allowed by API
        
        race_info = None
        
        try:
            while True:
                url = f"{self.BASE_URL}/{year}/{round_num}/laps.json?limit={limit}&offset={offset}"
                
                response = self._make_request(url, timeout=60)
                data = response.json()
                
                mrdata = data.get('MRData', {})
                race_data = mrdata.get('RaceTable', {})
                races = race_data.get('Races', [])
                
                if not races:
                    if offset == 0:
                        logger.warning(f"No lap data for {year} Round {round_num}")
                    break
                
                race = races[0]
                
                # Store race info from first request
                if race_info is None:
                    race_info = {
                        'season': year,
                        'round': round_num,
                        'raceName': race.get('raceName'),
                        'circuitId': race.get('Circuit', {}).get('circuitId'),
                        'date': race.get('date'),
                        'time': race.get('time')
                    }
                
                laps = race.get('Laps', [])
                if not laps:
                    break
                
                all_laps.extend(laps)
                
                # Check if we've fetched all data
                total = int(mrdata.get('total', '0'))
                current_limit = int(mrdata.get('limit', '0'))
                current_offset = int(mrdata.get('offset', '0'))
                
                # Count timing records we've received so far
                timing_count = sum(len(lap.get('Timings', [])) for lap in all_laps)
                
                logger.debug(f"Fetched {len(all_laps)} laps, {timing_count}/{total} timing records (offset: {current_offset})")
                
                # Continue if there might be more data
                if current_offset + current_limit < total:
                    offset = current_offset + current_limit
                else:
                    break
            
            if not all_laps:
                logger.warning(f"No laps found for {year} Round {round_num}")
                return None
            
            # Deduplicate laps (in case of overlap)
            unique_laps = []
            seen_lap_numbers = set()
            
            for lap in all_laps:
                lap_num = lap['number']
                if lap_num not in seen_lap_numbers:
                    seen_lap_numbers.add(lap_num)
                    unique_laps.append(lap)
            
            logger.info(f"Fetched {len(unique_laps)} laps for {year} Round {round_num} - {race_info.get('raceName', 'Unknown')}")
            
            race_info['laps'] = sorted(unique_laps, key=lambda x: int(x['number']))
            return race_info
            
        except Exception as e:
            logger.error(f"Error fetching laps for {year} Round {round_num}: {e}")
            self.metadata['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'year': year,
                'round': round_num,
                'error': str(e)
            })
            return None
    
    def save_race_laps(self, race_data: Dict) -> str:
        """Save race laps data to JSON file"""
        year = race_data['season']
        round_num = race_data['round']
        
        # Create year directory
        year_dir = self.laps_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        filename = f"{year}_round_{round_num:02d}_laps.json"
        filepath = year_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(race_data, f, indent=2)
        
        logger.info(f"Saved laps data to {filepath}")
        return str(filepath)
    
    def convert_laps_to_dataframe(self, race_data: Dict) -> pd.DataFrame:
        """Convert race laps data to a pandas DataFrame for analysis"""
        rows = []
        
        for lap in race_data['laps']:
            lap_number = int(lap['number'])
            
            for timing in lap['Timings']:
                rows.append({
                    'season': race_data['season'],
                    'round': race_data['round'],
                    'raceName': race_data['raceName'],
                    'circuitId': race_data['circuitId'],
                    'date': race_data['date'],
                    'lap': lap_number,
                    'driverId': timing['driverId'],
                    'position': int(timing['position']),
                    'time': timing.get('time', None)
                })
        
        return pd.DataFrame(rows)
    
    def fetch_season_laps(self, year: int, force: bool = False) -> List[str]:
        """Fetch all laps for a complete season"""
        logger.info(f"\nFetching laps data for {year} season...")
        
        # Get season info
        season_info = self.get_season_info(year)
        if not season_info:
            return []
        
        saved_files = []
        
        # Create progress bar with rate limit info
        pbar = tqdm(season_info['races'], desc=f"{year} races")
        
        for race in pbar:
            round_num = race['round']
            
            # Check if already fetched
            race_key = f"{year}_{round_num}"
            if not force and race_key in self.metadata['fetched_races']:
                logger.debug(f"Skipping {year} Round {round_num} - already fetched")
                continue
            
            # Fetch race laps
            race_data = self.fetch_race_laps(year, round_num)
            
            if race_data:
                # Save data
                filepath = self.save_race_laps(race_data)
                saved_files.append(filepath)
                
                # Update metadata
                self.metadata['fetched_races'][race_key] = {
                    'fetched_at': datetime.now().isoformat(),
                    'filepath': filepath,
                    'race_name': race_data['raceName'],
                    'total_laps': len(race_data['laps'])
                }
            
            # Update progress bar with rate limit info
            status = self.get_rate_limit_status()
            pbar.set_postfix({
                'Reqs/hr': f"{status['requests_last_hour']}/{status['sustained_limit']}",
                'Backoff': f"{status['current_backoff']:.1f}s" if status['consecutive_429s'] > 0 else "0s"
            })
        
        return saved_files
    
    def fetch_multiple_seasons(self, start_year: int = 2020, end_year: int = 2024, force: bool = False):
        """Fetch laps data for multiple seasons"""
        logger.info(f"Fetching laps data from {start_year} to {end_year}")
        
        all_files = []
        
        for year in range(start_year, end_year + 1):
            files = self.fetch_season_laps(year, force=force)
            all_files.extend(files)
            
            # Save metadata after each season
            self.metadata['last_fetch'] = datetime.now().isoformat()
            self._save_metadata()
        
        logger.info(f"\nCompleted fetching {len(all_files)} race files")
        return all_files
    
    def create_consolidated_dataset(self) -> pd.DataFrame:
        """Create a consolidated DataFrame from all fetched laps data"""
        logger.info("Creating consolidated laps dataset...")
        
        all_laps = []
        
        # Read all JSON files
        for year_dir in sorted(self.laps_dir.iterdir()):
            if year_dir.is_dir():
                for json_file in sorted(year_dir.glob("*.json")):
                    with open(json_file, 'r') as f:
                        race_data = json.load(f)
                    
                    df = self.convert_laps_to_dataframe(race_data)
                    all_laps.append(df)
        
        if all_laps:
            consolidated_df = pd.concat(all_laps, ignore_index=True)
            
            # Save consolidated dataset
            output_file = self.data_dir / "all_laps_2020_2024.parquet"
            consolidated_df.to_parquet(output_file, index=False)
            
            csv_file = self.data_dir / "all_laps_2020_2024.csv"
            consolidated_df.to_csv(csv_file, index=False)
            
            logger.info(f"Saved consolidated dataset with {len(consolidated_df)} lap records")
            logger.info(f"Parquet: {output_file}")
            logger.info(f"CSV: {csv_file}")
            
            return consolidated_df
        else:
            logger.warning("No laps data found to consolidate")
            return pd.DataFrame()
    
    def generate_summary_report(self) -> Dict:
        """Generate a summary report of fetched data"""
        report = {
            'fetch_summary': {
                'last_fetch': self.metadata.get('last_fetch'),
                'total_races_fetched': len(self.metadata['fetched_races']),
                'errors_encountered': len(self.metadata['errors'])
            },
            'season_breakdown': {},
            'missing_races': []
        }
        
        # Analyze fetched races
        for race_key, race_info in self.metadata['fetched_races'].items():
            year, round_num = race_key.split('_')
            year = int(year)
            
            if year not in report['season_breakdown']:
                report['season_breakdown'][year] = {
                    'races_fetched': 0,
                    'total_laps': 0
                }
            
            report['season_breakdown'][year]['races_fetched'] += 1
            report['season_breakdown'][year]['total_laps'] += race_info.get('total_laps', 0)
        
        # Save report
        report_file = self.data_dir / "fetch_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Clean old requests
        while self.request_times and self.request_times[0] < hour_ago:
            self.request_times.popleft()
        
        requests_last_hour = len(self.request_times)
        remaining_hourly = self.SUSTAINED_LIMIT - requests_last_hour
        
        # Calculate requests in last second
        second_ago = current_time - 1
        requests_last_second = sum(1 for t in self.request_times if t > second_ago)
        
        return {
            'requests_last_hour': requests_last_hour,
            'remaining_hourly': remaining_hourly,
            'requests_last_second': requests_last_second,
            'burst_limit': self.BURST_LIMIT,
            'sustained_limit': self.SUSTAINED_LIMIT,
            'current_backoff': self.current_backoff,
            'consecutive_429s': self.consecutive_429s
        }


def main():
    """Main execution function"""
    # Initialize fetcher
    fetcher = JolpicaLapsFetcher()
    
    print("Jolpica F1 Laps Data Fetcher with Exponential Backoff")
    print("=" * 60)
    print(f"Data directory: {fetcher.data_dir}")
    print(f"Rate limits: {fetcher.BURST_LIMIT} req/sec, {fetcher.SUSTAINED_LIMIT} req/hour")
    print(f"Fetching seasons: 2021-2025 (current year first, then back 5 years)")
    print()
    
    # Check existing data
    if fetcher.metadata['fetched_races']:
        print(f"Found {len(fetcher.metadata['fetched_races'])} races already fetched")
        response = input("Continue from where left off? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Fetch data
    start_time = time.time()
    
    try:
        # Fetch all seasons - start with current year, then go back 5 years
        current_year = 2025
        start_year = current_year - 4  # 2021
        print(f"Fetching from {current_year} back to {start_year}")
        
        # Fetch in reverse order (current year first)
        for year in range(current_year, start_year - 1, -1):
            print(f"\nFetching {year} season...")
            
            # Check existing data for this season first
            season_info = fetcher.get_season_info(year)
            if season_info:
                total_races = len(season_info['races'])
                existing_races = [k for k in fetcher.metadata['fetched_races'].keys() if k.startswith(f"{year}_")]
                if len(existing_races) == total_races:
                    print(f"  All {total_races} races for {year} already fetched - skipping")
                    continue
                elif existing_races:
                    print(f"  Found {len(existing_races)}/{total_races} races already fetched for {year}")
            
            fetcher.fetch_season_laps(year, force=False)
        
        # Show final rate limit status
        status = fetcher.get_rate_limit_status()
        print(f"\nFinal rate limit status:")
        print(f"- Requests in last hour: {status['requests_last_hour']}")
        print(f"- Remaining hourly quota: {status['remaining_hourly']}")
        print(f"- Total 429 errors: {len(fetcher.metadata['errors'])}")
        
        # Create consolidated dataset
        print("\nCreating consolidated dataset...")
        df = fetcher.create_consolidated_dataset()
        
        if not df.empty:
            print(f"\nDataset Statistics:")
            print(f"- Total lap records: {len(df):,}")
            print(f"- Unique drivers: {df['driverId'].nunique()}")
            print(f"- Unique races: {len(df.groupby(['season', 'round']))}")
            print(f"- Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Generate report
        report = fetcher.generate_summary_report()
        print(f"\nFetch report saved to: {fetcher.data_dir / 'fetch_report.json'}")
        
        # Show summary
        print(f"\nSummary by season:")
        for year, info in sorted(report['season_breakdown'].items()):
            print(f"  {year}: {info['races_fetched']} races, {info['total_laps']:,} laps")
        
    except KeyboardInterrupt:
        print("\n\nFetch interrupted by user. Progress has been saved.")
        print("Run the script again to continue from where you left off.")
    except Exception as e:
        logger.error(f"Error during fetch: {e}")
        raise
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    main()