#!/usr/bin/env python3
"""
F1 Fantasy Data Fetcher

Fetches driver statistics and race-by-race details from the F1 Fantasy API
and saves them as two CSV files for integration with the F1 ML pipeline.

Usage:
    python f1_fantasy_fetcher.py [--output-dir /data/f1_fantasy]
"""

import requests
import pandas as pd
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import unicodedata
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('F1FantasyFetcher')


class F1FantasyFetcher:
    """Fetches and processes F1 Fantasy data into CSV format"""
    
    def __init__(self, output_dir: str = None):
        self.base_url = "https://fantasy.formula1.com/feeds"
        
        # Always use absolute path to /data/f1_fantasy
        if output_dir is None:
            # Get project root (2 levels up from notebooks/advanced)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            self.output_dir = project_root / 'data' / 'f1_fantasy'
            self.f1db_dir = project_root / 'data' / 'f1db'
        else:
            # If output_dir is provided, make it absolute
            self.output_dir = Path(output_dir).resolve()
            self.f1db_dir = self.output_dir.parent.parent / 'data' / 'f1db'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"F1DB directory: {self.f1db_dir}")
        
        # Configure session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://fantasy.formula1.com/',
            'Origin': 'https://fantasy.formula1.com'
        })
        
        # API delay to be respectful
        self.api_delay = 0.5  # seconds between requests
        
        # Load F1DB data for mapping
        self.drivers_df = None
        self.races_df = None
        self._load_f1db_data()
        
    def _load_f1db_data(self):
        """Load F1DB drivers and races data for mapping"""
        try:
            # Load drivers data
            drivers_path = self.f1db_dir / 'drivers.csv'
            if drivers_path.exists():
                self.drivers_df = pd.read_csv(drivers_path)
                # Filter to only recent drivers (those who have raced since 2020)
                # This helps with name matching accuracy
                logger.info(f"Loaded {len(self.drivers_df)} drivers from F1DB")
            else:
                logger.warning(f"F1DB drivers.csv not found at {drivers_path}")
            
            # Load races data
            races_path = self.f1db_dir / 'races.csv'
            if races_path.exists():
                self.races_df = pd.read_csv(races_path)
                # Filter to current season
                current_year = datetime.now().year
                self.races_df = self.races_df[self.races_df['year'] == current_year]
                logger.info(f"Loaded {len(self.races_df)} races for {current_year} from F1DB")
            else:
                logger.warning(f"F1DB races.csv not found at {races_path}")
                
        except Exception as e:
            logger.error(f"Error loading F1DB data: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for matching by removing accents and special characters"""
        # Remove accents
        normalized = unicodedata.normalize('NFD', name)
        normalized = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized.lower().strip())
        return normalized
    
    def _find_driver_id(self, fantasy_name: str) -> Optional[str]:
        """Find F1DB driver ID by matching fantasy name"""
        if self.drivers_df is None:
            return None
        
        # Normalize the fantasy name
        normalized_fantasy = self._normalize_name(fantasy_name)
        
        # First check special mappings for known current drivers
        special_mappings = {
            'max verstappen': 'max-verstappen',
            'lewis hamilton': 'lewis-hamilton',
            'george russell': 'george-russell',
            'lando norris': 'lando-norris',
            'oscar piastri': 'oscar-piastri',
            'charles leclerc': 'charles-leclerc',
            'carlos sainz': 'carlos-sainz',
            'sergio perez': 'sergio-perez',
            'fernando alonso': 'fernando-alonso',
            'lance stroll': 'lance-stroll',
            'pierre gasly': 'pierre-gasly',
            'esteban ocon': 'esteban-ocon',
            'valtteri bottas': 'valtteri-bottas',
            'guanyu zhou': 'zhou-guanyu',
            'yuki tsunoda': 'yuki-tsunoda',
            'daniel ricciardo': 'daniel-ricciardo',
            'alex albon': 'alex-albon',
            'alexander albon': 'alex-albon',
            'logan sargeant': 'logan-sargeant',
            'nico hulkenberg': 'nico-hulkenberg',
            'kevin magnussen': 'kevin-magnussen',
            'oliver bearman': 'oliver-bearman',
            'liam lawson': 'liam-lawson',
            'franco colapinto': 'franco-colapinto',
            'jack doohan': 'jack-doohan',
            'kimi antonelli': 'andrea-kimi-antonelli',
            'isack hadjar': 'isack-hadjar',
            'gabriel bortoleto': 'gabriel-bortoleto',
            'nyck de vries': 'nyck-de-vries'
        }
        
        if normalized_fantasy in special_mappings:
            return special_mappings[normalized_fantasy]
        
        # Try exact matching strategies only after special mappings
        for _, driver in self.drivers_df.iterrows():
            # Only exact full name match
            if self._normalize_name(driver['fullName']) == normalized_fantasy:
                return driver['id']
            
            # First name + last name exact match
            driver_full = self._normalize_name(f"{driver['firstName']} {driver['lastName']}")
            if driver_full == normalized_fantasy:
                return driver['id']
        
        logger.warning(f"Could not find F1DB driver ID for: {fantasy_name}")
        return None
    
    def _find_race_info(self, gameday_id: int) -> Dict:
        """Find race information by gameday ID (round number)"""
        if self.races_df is None:
            return {}
        
        # gameday_id corresponds to round number
        race = self.races_df[self.races_df['round'] == gameday_id]
        
        if not race.empty:
            race_row = race.iloc[0]
            return {
                'race_id': race_row['id'],
                'race_name': race_row['officialName'],
                'race_date': race_row['date'],
                'circuit_id': race_row['circuitId'],
                'grand_prix_id': race_row['grandPrixId']
            }
        
        return {}
    
    def get_buster_timestamp(self) -> str:
        """Generate cache buster timestamp"""
        return datetime.now().strftime("%Y%m%d%H%M%S")
    
    def fetch_driver_statistics(self) -> Optional[Dict]:
        """Fetch comprehensive driver statistics"""
        buster = self.get_buster_timestamp()
        url = f"{self.base_url}/statistics/drivers_3.json?buster={buster}"
        
        try:
            logger.info("Fetching driver statistics...")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching driver statistics: {e}")
            return None
    
    def fetch_player_details(self, player_id: str) -> Optional[Dict]:
        """Fetch detailed stats for a specific player"""
        buster = self.get_buster_timestamp()
        url = f"{self.base_url}/popup/playerstats_{player_id}.json?buster={buster}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching stats for player {player_id}: {e}")
            return None
    
    def extract_player_ids(self, stats_data: Dict) -> List[str]:
        """Extract all unique player IDs from statistics data"""
        player_ids = set()
        
        for stat_category in stats_data['Data']['statistics']:
            if stat_category['participants']:
                for participant in stat_category['participants']:
                    player_ids.add(participant['playerid'])
        
        return sorted(list(player_ids))
    
    def create_driver_overview(self, stats_data: Dict) -> pd.DataFrame:
        """Create driver overview DataFrame with all statistical categories"""
        
        # Start with fantasy points as base (includes driver info)
        base_data = {}
        for stat in stats_data['Data']['statistics']:
            if stat['config']['key'] == 'fPoints':
                for participant in stat['participants']:
                    player_id = participant['playerid']
                    base_data[player_id] = {
                        'player_id': player_id,
                        'player_name': participant['playername'],
                        'team_name': participant['teamname'],
                        'current_price': participant['curvalue'],
                        'fantasy_points': participant['statvalue'],
                        'fantasy_points_rank': participant['rnk']
                    }
                break
        
        # Add other statistics
        stat_mappings = {
            'fAvg': ('fantasy_avg', 'fantasy_avg_rank'),
            'priceChange': ('price_change', 'price_change_rank'),
            'mostPicked': ('most_picked_pct', 'most_picked_rank'),
            'pointsPermillion': ('points_per_million', 'points_per_million_rank'),
            'overTakepoints': ('overtake_points', 'overtake_points_rank'),
            'podiumsStats': ('podiums', 'podiums_rank'),
            'topFinshed': ('top_finishes', 'top_finishes_rank'),
            'mostDnf': ('dnfs', 'dnfs_rank'),
            'fastestLap': ('fastest_laps', 'fastest_laps_rank'),
            'driverOfday': ('driver_of_day', 'driver_of_day_rank')
        }
        
        for stat in stats_data['Data']['statistics']:
            key = stat['config']['key']
            if key in stat_mappings and stat['participants']:
                value_col, rank_col = stat_mappings[key]
                
                for participant in stat['participants']:
                    player_id = participant['playerid']
                    if player_id in base_data:
                        base_data[player_id][value_col] = participant['statvalue']
                        base_data[player_id][rank_col] = participant['rnk']
        
        # Convert to DataFrame
        df = pd.DataFrame(list(base_data.values()))
        
        # Add F1DB driver IDs
        df['f1db_driver_id'] = df['player_name'].apply(self._find_driver_id)
        
        # Add metadata columns
        df['last_updated'] = datetime.now().isoformat()
        df['season'] = stats_data['Data']['season']
        
        # Sort by fantasy points rank
        df = df.sort_values('fantasy_points_rank')
        
        # Reorder columns to put important ones first
        column_order = ['player_id', 'player_name', 'f1db_driver_id', 'team_name', 
                       'current_price', 'fantasy_points', 'fantasy_points_rank']
        other_columns = [col for col in df.columns if col not in column_order]
        df = df[column_order + other_columns]
        
        # Log mapping success rate
        mapped_count = df['f1db_driver_id'].notna().sum()
        logger.info(f"Successfully mapped {mapped_count}/{len(df)} drivers to F1DB IDs")
        
        return df
    
    def create_driver_details(self, player_details: Dict[str, Dict], 
                            driver_names: Dict[str, str]) -> pd.DataFrame:
        """Create driver details DataFrame with race-by-race breakdown"""
        
        all_details = []
        
        for player_id, data in player_details.items():
            if 'Value' not in data or 'GamedayWiseStats' not in data['Value']:
                continue
            
            player_name = driver_names.get(player_id, f"Unknown_{player_id}")
            
            for gameday in data['Value']['GamedayWiseStats']:
                # Get race info for this gameday
                race_info = self._find_race_info(gameday['GamedayId'])
                
                detail_row = {
                    'player_id': player_id,
                    'player_name': player_name,
                    'f1db_driver_id': self._find_driver_id(player_name),
                    'gameday_id': gameday['GamedayId'],
                    'player_value_at_race': gameday['PlayerValue'],
                    'player_value_before': gameday['OldPlayerValue'],
                    'is_played': gameday['IsPlayed'],
                    'is_active': gameday['IsActive'],
                    'total_points': 0,  # Will be updated from StatsWise
                    # Add race information
                    'race_id': int(race_info.get('race_id')) if race_info.get('race_id') is not None else None,
                    'race_name': race_info.get('race_name'),
                    'race_date': race_info.get('race_date'),
                    'circuit_id': race_info.get('circuit_id'),
                    'grand_prix_id': race_info.get('grand_prix_id')
                }
                
                # Process ALL stats dynamically
                for stat in gameday['StatsWise']:
                    event = stat['Event']
                    
                    # Create column names from event names
                    # Replace spaces with underscores and convert to lowercase
                    event_key = event.lower().replace(' ', '_')
                    
                    # Special handling for known important events
                    if event == 'Total':
                        detail_row['total_points'] = stat['Value']
                    elif event == 'Race Position':
                        detail_row['race_position'] = stat.get('Frequency', 'DNF')
                        detail_row['race_position_points'] = stat['Value']
                    elif event == 'Qualifying Position':
                        detail_row['quali_position'] = stat.get('Frequency', 'DNS')
                        detail_row['quali_position_points'] = stat['Value']
                    elif event == 'Sprint Position':
                        detail_row['sprint_position'] = stat.get('Frequency', 'DNS')
                        detail_row['sprint_position_points'] = stat['Value']
                    else:
                        # For all other events, store both frequency and value
                        if 'Frequency' in stat:
                            detail_row[f"{event_key}_freq"] = stat['Frequency']
                        detail_row[f"{event_key}_points"] = stat['Value']
                    
                    # Always store the raw value for completeness
                    detail_row[f"stat_{event_key}_value"] = stat['Value']
                    if 'Frequency' in stat:
                        detail_row[f"stat_{event_key}_frequency"] = stat['Frequency']
                
                all_details.append(detail_row)
        
        # Create DataFrame
        df = pd.DataFrame(all_details)
        
        # Log unique stat types found
        stat_columns = [col for col in df.columns if 'stat_' in col or col.endswith('_points')]
        unique_stats = set()
        for col in stat_columns:
            if col.endswith('_points') and not col.startswith('stat_'):
                stat_name = col.replace('_points', '').replace('_', ' ').title()
                unique_stats.add(stat_name)
        
        logger.info(f"Found {len(unique_stats)} unique stat types: {sorted(unique_stats)}")
        
        # Add metadata
        df['last_updated'] = datetime.now().isoformat()
        
        # Sort by player and gameday
        df = df.sort_values(['player_id', 'gameday_id'])
        
        # Reorder columns to put important ones first
        # Core identification columns
        core_columns = ['player_id', 'player_name', 'f1db_driver_id', 'gameday_id', 
                       'race_id', 'race_name', 'race_date', 'circuit_id']
        
        # Main result columns
        result_columns = ['total_points', 'race_position', 'quali_position', 'sprint_position',
                         'race_position_points', 'quali_position_points', 'sprint_position_points']
        
        # Find all other point columns
        point_columns = [col for col in df.columns if col.endswith('_points') and col not in result_columns]
        freq_columns = [col for col in df.columns if col.endswith('_freq')]
        stat_columns = [col for col in df.columns if col.startswith('stat_')]
        
        # Other metadata columns
        meta_columns = ['player_value_at_race', 'player_value_before', 'is_played', 
                       'is_active', 'last_updated']
        
        # Build final column order
        column_order = core_columns + result_columns + point_columns + freq_columns + stat_columns + meta_columns
        
        # Add any remaining columns
        other_columns = [col for col in df.columns if col not in column_order]
        
        # Only include columns that exist in the dataframe
        final_columns = [col for col in column_order + other_columns if col in df.columns]
        df = df[final_columns]
        
        # Log race mapping success
        races_mapped = df['race_id'].notna().sum()
        total_records = len(df)
        logger.info(f"Successfully mapped {races_mapped}/{total_records} race records")
        
        return df
    
    def save_metadata(self, num_drivers: int, num_races: int, 
                     drivers_mapped: int = 0, races_mapped: int = 0):
        """Save metadata about the data extraction"""
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'num_drivers': num_drivers,
            'num_races': num_races,
            'drivers_mapped_to_f1db': drivers_mapped,
            'races_mapped_to_f1db': races_mapped,
            'api_version': 'drivers_3',
            'data_source': 'F1 Fantasy API',
            'update_frequency': 'weekly',
            'f1db_integration': True
        }
        
        metadata_path = self.output_dir / '.f1_fantasy_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def fetch_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method to fetch all data and return as DataFrames"""
        
        # Step 1: Fetch driver statistics
        logger.info("Starting F1 Fantasy data extraction...")
        stats_data = self.fetch_driver_statistics()
        if not stats_data:
            raise Exception("Failed to fetch driver statistics")
        
        # Step 2: Extract player IDs and create driver name mapping
        player_ids = self.extract_player_ids(stats_data)
        logger.info(f"Found {len(player_ids)} drivers")
        
        # Create driver name mapping
        driver_names = {}
        for stat in stats_data['Data']['statistics']:
            if stat['config']['key'] == 'fPoints':
                for participant in stat['participants']:
                    driver_names[participant['playerid']] = participant['playername']
                break
        
        # Step 3: Create driver overview DataFrame
        driver_overview_df = self.create_driver_overview(stats_data)
        logger.info(f"Created driver overview with {len(driver_overview_df)} drivers")
        
        # Step 4: Fetch detailed stats for each player
        player_details = {}
        logger.info("Fetching detailed stats for each driver...")
        
        for i, player_id in enumerate(player_ids, 1):
            logger.info(f"Fetching player {player_id} ({i}/{len(player_ids)})")
            player_data = self.fetch_player_details(player_id)
            
            if player_data:
                player_details[player_id] = player_data
            
            # Respectful API usage
            time.sleep(self.api_delay)
        
        # Step 5: Create driver details DataFrame
        driver_details_df = self.create_driver_details(player_details, driver_names)
        logger.info(f"Created driver details with {len(driver_details_df)} records")
        
        # Step 6: Save metadata with mapping counts
        num_races = int(driver_details_df['gameday_id'].nunique()) if not driver_details_df.empty else 0
        drivers_mapped = int(driver_overview_df['f1db_driver_id'].notna().sum()) if not driver_overview_df.empty else 0
        races_mapped = int(driver_details_df['race_id'].notna().sum()) if not driver_details_df.empty else 0
        self.save_metadata(len(driver_overview_df), num_races, drivers_mapped, races_mapped)
        
        return driver_overview_df, driver_details_df
    
    def save_to_csv(self):
        """Fetch all data and save to CSV files"""
        try:
            # Fetch data
            overview_df, details_df = self.fetch_all_data()
            
            # Save driver overview
            overview_path = self.output_dir / 'driver_overview.csv'
            overview_df.to_csv(overview_path, index=False)
            logger.info(f"Saved driver overview to {overview_path}")
            
            # Save driver details
            details_path = self.output_dir / 'driver_details.csv'
            details_df.to_csv(details_path, index=False)
            logger.info(f"Saved driver details to {details_path}")
            
            # Print summary
            logger.info("\n" + "="*50)
            logger.info("F1 Fantasy Data Extraction Complete!")
            logger.info(f"✅ Driver Overview: {len(overview_df)} drivers")
            logger.info(f"✅ Driver Details: {len(details_df)} race records")
            logger.info(f"📁 Data saved to: {self.output_dir}")
            logger.info("="*50)
            
            # Show top 5 fantasy scorers
            logger.info("\n🏆 Top 5 Fantasy Point Scorers:")
            for idx, row in overview_df.head(5).iterrows():
                logger.info(f"{row['fantasy_points_rank']}. {row['player_name']} "
                          f"({row['team_name']}) - {row['fantasy_points']} pts")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during data extraction: {e}")
            return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Fetch F1 Fantasy data')
    parser.add_argument('--output-dir', default=None,
                      help='Output directory for CSV files (default: /data/f1_fantasy)')
    parser.add_argument('--api-delay', type=float, default=0.5,
                      help='Delay between API requests in seconds')
    
    args = parser.parse_args()
    
    # Create fetcher and run
    fetcher = F1FantasyFetcher(output_dir=args.output_dir)
    fetcher.api_delay = args.api_delay
    
    success = fetcher.save_to_csv()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()