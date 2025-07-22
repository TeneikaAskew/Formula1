"""
Enhanced F1DB Data Loader with automated sync, caching, and error handling
Builds upon the existing f1db_data_loader.py with production-ready features
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import hashlib
from datetime import datetime, timedelta
import logging
import time
from functools import wraps
import pickle

# Import the original loader
from f1db_data_loader import F1DBDataLoader, load_f1db_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    wait_time = backoff_in_seconds * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


class EnhancedF1DBDataLoader(F1DBDataLoader):
    """Enhanced F1DB data loader with caching, validation, and automation features"""
    
    def __init__(self, data_dir: str = "data", format: str = "csv", cache_dir: Optional[str] = None):
        super().__init__(data_dir, format)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
        
    def get_data_hash(self, file_path: Path) -> str:
        """Calculate hash of a file for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about the current data state"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> Dict:
        """Load saved metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def download_latest_data(self, force: bool = False) -> bool:
        """Enhanced download with retry logic and better error handling"""
        try:
            result = super().download_latest_data(force)
            if result:
                # Update metadata after successful download
                metadata = {
                    'last_download': datetime.now().isoformat(),
                    'version': self.get_current_version(),
                    'file_hashes': {}
                }
                
                # Calculate hashes for all downloaded files
                for csv_file in self.data_dir.glob("*.csv"):
                    metadata['file_hashes'][csv_file.name] = self.get_data_hash(csv_file)
                
                self.save_metadata(metadata)
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading F1DB data: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading F1DB data: {e}")
            raise
    
    def get_current_version(self) -> Optional[str]:
        """Get the current data version"""
        marker_file = self.data_dir / ".f1db_version"
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                return f.read().strip()
        return None
    
    def validate_data_integrity(self) -> Tuple[bool, List[str]]:
        """Validate that all expected files exist and haven't been corrupted"""
        issues = []
        required_files = [
            'races.csv', 'drivers.csv', 'constructors.csv', 'circuits.csv',
            'race_results.csv', 'qualifying_results.csv'
        ]
        
        # Check for required files
        for filename in required_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                # Try alternative names
                alt_names = {
                    'race_results.csv': ['results.csv'],
                    'qualifying_results.csv': ['qualifying.csv']
                }
                found = False
                for alt in alt_names.get(filename, []):
                    if (self.data_dir / alt).exists():
                        found = True
                        break
                if not found:
                    issues.append(f"Missing required file: {filename}")
        
        # Check file integrity using stored hashes
        metadata = self.load_metadata()
        if 'file_hashes' in metadata:
            for filename, expected_hash in metadata['file_hashes'].items():
                file_path = self.data_dir / filename
                if file_path.exists():
                    actual_hash = self.get_data_hash(file_path)
                    if actual_hash != expected_hash:
                        issues.append(f"File integrity check failed: {filename}")
        
        return len(issues) == 0, issues
    
    def load_csv_data_cached(self, cache_expiry_hours: int = 24) -> Dict[str, pd.DataFrame]:
        """Load CSV data with caching support"""
        cache_file = self.cache_dir / "dataframes.pkl"
        cache_metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Check if cache exists and is valid
        if cache_file.exists() and cache_metadata_file.exists():
            with open(cache_metadata_file, 'r') as f:
                cache_metadata = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_metadata['cached_at'])
            if datetime.now() - cache_time < timedelta(hours=cache_expiry_hours):
                # Check if data hasn't changed
                current_version = self.get_current_version()
                if current_version == cache_metadata.get('version'):
                    logger.info("Loading data from cache...")
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
        
        # Load fresh data
        logger.info("Loading fresh data from CSV files...")
        dataframes = self.load_csv_data()
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump(dataframes, f)
        
        cache_metadata = {
            'cached_at': datetime.now().isoformat(),
            'version': self.get_current_version()
        }
        with open(cache_metadata_file, 'w') as f:
            json.dump(cache_metadata, f)
        
        return dataframes
    
    def automated_sync(self, check_interval_hours: int = 24) -> bool:
        """Automated sync that checks for updates periodically"""
        metadata = self.load_metadata()
        
        # Check if we need to sync
        if 'last_download' in metadata:
            last_download = datetime.fromisoformat(metadata['last_download'])
            if datetime.now() - last_download < timedelta(hours=check_interval_hours):
                logger.info(f"Data was synced {(datetime.now() - last_download).hours} hours ago. Skipping sync.")
                return False
        
        # Check for updates
        try:
            latest_version = self.get_latest_release_info()['tag_name']
            current_version = self.get_current_version()
            
            if current_version != latest_version:
                logger.info(f"New version available: {latest_version} (current: {current_version})")
                return self.download_latest_data(force=True)
            else:
                logger.info(f"Already on latest version: {current_version}")
                # Update last check time
                metadata['last_download'] = datetime.now().isoformat()
                self.save_metadata(metadata)
                return False
                
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False
    
    def get_race_calendar(self, year: Optional[int] = None) -> pd.DataFrame:
        """Get race calendar for a specific year or current year"""
        data = self.load_csv_data_cached()
        races_df = data.get('races', pd.DataFrame())
        
        if year:
            races_df = races_df[races_df['year'] == year]
        else:
            # Get current year races
            current_year = datetime.now().year
            races_df = races_df[races_df['year'] == current_year]
        
        if 'date' in races_df.columns:
            races_df['date'] = pd.to_datetime(races_df['date'])
            races_df = races_df.sort_values('date')
        
        return races_df
    
    def get_upcoming_race(self) -> Optional[pd.Series]:
        """Get the next upcoming race"""
        calendar = self.get_race_calendar()
        if calendar.empty:
            return None
        
        today = pd.Timestamp.now()
        future_races = calendar[calendar['date'] > today]
        
        if not future_races.empty:
            return future_races.iloc[0]
        return None
    
    def get_recent_results(self, n_races: int = 5) -> pd.DataFrame:
        """Get results from the most recent n races"""
        data = self.load_csv_data_cached()
        results_df = data.get('results', pd.DataFrame())
        races_df = data.get('races', pd.DataFrame())
        
        if results_df.empty or races_df.empty:
            return pd.DataFrame()
        
        # Merge to get race dates
        results_with_dates = results_df.merge(
            races_df[['raceId', 'date', 'name', 'year']], 
            on='raceId'
        )
        results_with_dates['date'] = pd.to_datetime(results_with_dates['date'])
        
        # Get unique races sorted by date
        recent_races = (results_with_dates[['raceId', 'date']]
                       .drop_duplicates()
                       .sort_values('date', ascending=False)
                       .head(n_races)['raceId'].tolist())
        
        return results_with_dates[results_with_dates['raceId'].isin(recent_races)]


# Enhanced convenience function
def load_f1db_data_enhanced(
    data_dir: Optional[str] = None, 
    format: str = "csv", 
    force_download: bool = False,
    use_cache: bool = True,
    auto_sync: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Enhanced F1DB data loading with caching and auto-sync
    
    Args:
        data_dir: Directory to store/load data
        format: Data format ('csv' recommended)
        force_download: Force re-download of data
        use_cache: Use cached DataFrames if available
        auto_sync: Automatically check for updates
        
    Returns:
        Dictionary of DataFrames with F1 data
    """
    if data_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_dir = str(project_root / "data" / "f1db")
    
    loader = EnhancedF1DBDataLoader(str(data_dir), format)
    
    # Auto-sync if enabled
    if auto_sync and not force_download:
        loader.automated_sync()
    elif force_download:
        loader.download_latest_data(force=True)
    
    # Validate data integrity
    is_valid, issues = loader.validate_data_integrity()
    if not is_valid:
        logger.warning(f"Data integrity issues found: {issues}")
        if force_download:
            logger.info("Attempting to fix by re-downloading...")
            loader.download_latest_data(force=True)
    
    # Load data with caching
    if use_cache:
        return loader.load_csv_data_cached()
    else:
        return loader.get_core_datasets()


# Add to existing notebooks:
# from enhanced_f1db_data_loader import load_f1db_data_enhanced
# data = load_f1db_data_enhanced(auto_sync=True)