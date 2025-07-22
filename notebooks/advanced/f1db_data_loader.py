"""
F1DB Data Loader
Fetches the latest F1 data from the official F1DB repository
https://github.com/f1db/f1db
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import json

class F1DBDataLoader:
    """Load F1 data from the official F1DB repository"""
    
    def __init__(self, data_dir: str = "data", format: str = "csv"):
        """
        Initialize the F1DB data loader
        
        Args:
            data_dir: Directory to store downloaded data
            format: Data format to use ('csv', 'json', 'sqlite')
        """
        self.data_dir = Path(data_dir)
        self.format = format
        self.base_url = "https://api.github.com/repos/f1db/f1db/releases/latest"
        self.data_dir.mkdir(exist_ok=True)
        
    def get_latest_release_info(self) -> Dict:
        """Get information about the latest F1DB release"""
        response = requests.get(self.base_url)
        response.raise_for_status()
        return response.json()
    
    def download_latest_data(self, force: bool = False) -> bool:
        """
        Download the latest F1DB data if not already present
        
        Args:
            force: Force download even if data exists
            
        Returns:
            True if data was downloaded, False if already exists
        """
        # Check if data already exists
        marker_file = self.data_dir / ".f1db_version"
        
        try:
            release_info = self.get_latest_release_info()
            latest_version = release_info['tag_name']
            
            # Check if we already have this version
            if marker_file.exists() and not force:
                with open(marker_file, 'r') as f:
                    current_version = f.read().strip()
                if current_version == latest_version:
                    print(f"Already have latest F1DB data (version {latest_version})")
                    return False
            
            # Find the appropriate asset
            asset_name = f"f1db-{self.format}.zip"
            asset = None
            for a in release_info['assets']:
                if a['name'] == asset_name:
                    asset = a
                    break
            
            if not asset:
                raise ValueError(f"No {self.format} format found in latest release")
            
            # Download the data
            print(f"Downloading F1DB {self.format} data (version {latest_version})...")
            download_url = asset['browser_download_url']
            zip_path = self.data_dir / asset_name
            
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')
            
            print("\nExtracting data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            # Save version marker
            with open(marker_file, 'w') as f:
                f.write(latest_version)
            
            print(f"Successfully downloaded F1DB data version {latest_version}")
            return True
            
        except Exception as e:
            print(f"Error downloading F1DB data: {e}")
            raise
    
    def load_csv_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files into pandas DataFrames"""
        dataframes = {}
        
        # Ensure data is downloaded
        self.download_latest_data()
        
        # Load all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found. Data may not be downloaded correctly.")
        
        print(f"Loading {len(csv_files)} CSV files...")
        for csv_file in csv_files:
            table_name = csv_file.stem
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                dataframes[table_name] = df
                print(f"  ✓ Loaded {table_name}: {len(df)} rows")
            except Exception as e:
                print(f"  ✗ Error loading {table_name}: {e}")
        
        return dataframes
    
    def get_core_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get the core datasets commonly used for analysis"""
        all_data = self.load_csv_data()
        
        # Define core datasets
        core_tables = [
            'races', 'drivers', 'constructors', 'circuits',
            'race_results', 'qualifying_results', 'sprint_race_results',
            'fastest_laps', 'pit_stops', 'race_driver_standings',
            'race_constructor_standings', 'seasons'
        ]
        
        core_data = {}
        for table in core_tables:
            if table in all_data:
                core_data[table] = all_data[table]
            else:
                print(f"Warning: Core table '{table}' not found in data")
        
        return core_data


# Convenience function for notebook usage
def load_f1db_data(data_dir: str = "data", format: str = "csv", force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load F1DB data with a simple function call
    
    Args:
        data_dir: Directory to store/load data
        format: Data format ('csv' recommended)
        force_download: Force re-download of data
        
    Returns:
        Dictionary of DataFrames with F1 data
    """
    loader = F1DBDataLoader(data_dir, format)
    loader.download_latest_data(force=force_download)
    return loader.get_core_datasets()


# Example usage in notebooks:
# from f1db_data_loader import load_f1db_data
# data = load_f1db_data()
# races_df = data['races']
# drivers_df = data['drivers']