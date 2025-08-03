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
        else:
            # If output_dir is provided, make it absolute
            self.output_dir = Path(output_dir).resolve()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
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
        
        # Add metadata columns
        df['last_updated'] = datetime.now().isoformat()
        df['season'] = stats_data['Data']['season']
        
        # Sort by fantasy points rank
        df = df.sort_values('fantasy_points_rank')
        
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
                detail_row = {
                    'player_id': player_id,
                    'player_name': player_name,
                    'gameday_id': gameday['GamedayId'],
                    'player_value_at_race': gameday['PlayerValue'],
                    'player_value_before': gameday['OldPlayerValue'],
                    'is_played': gameday['IsPlayed'],
                    'is_active': gameday['IsActive'],
                    'total_points': 0  # Will be updated from StatsWise
                }
                
                # Process individual stats
                for stat in gameday['StatsWise']:
                    event = stat['Event']
                    if event == 'Total':
                        detail_row['total_points'] = stat['Value']
                    elif event == 'Race Position':
                        detail_row['race_position'] = stat.get('Frequency', 'DNF')
                        detail_row['race_position_points'] = stat['Value']
                    elif event == 'Qualifying Position':
                        detail_row['quali_position'] = stat.get('Frequency', 'DNS')
                        detail_row['quali_position_points'] = stat['Value']
                    elif event == 'Finished Ahead of Teammate':
                        detail_row['beat_teammate'] = stat.get('Frequency', 'No')
                        detail_row['beat_teammate_points'] = stat['Value']
                    elif event == 'Overtaking':
                        detail_row['overtaking_count'] = stat.get('Frequency', '0')
                        detail_row['overtaking_points'] = stat['Value']
                    elif event == 'Fastest Lap':
                        detail_row['fastest_lap'] = stat.get('Frequency', 'No')
                        detail_row['fastest_lap_points'] = stat['Value']
                    elif event == 'Driver of the Day':
                        detail_row['driver_of_day'] = stat.get('Frequency', 'No')
                        detail_row['driver_of_day_points'] = stat['Value']
                
                all_details.append(detail_row)
        
        # Create DataFrame
        df = pd.DataFrame(all_details)
        
        # Add metadata
        df['last_updated'] = datetime.now().isoformat()
        
        # Sort by player and gameday
        df = df.sort_values(['player_id', 'gameday_id'])
        
        return df
    
    def save_metadata(self, num_drivers: int, num_races: int):
        """Save metadata about the data extraction"""
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'num_drivers': num_drivers,
            'num_races': num_races,
            'api_version': 'drivers_3',
            'data_source': 'F1 Fantasy API',
            'update_frequency': 'weekly'
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
        
        # Step 6: Save metadata
        num_races = driver_details_df['gameday_id'].nunique() if not driver_details_df.empty else 0
        self.save_metadata(len(driver_overview_df), num_races)
        
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