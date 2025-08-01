"""
F1DB Data Loader
Fetches the latest F1 data from the official F1DB repository
https://github.com/f1db/f1db

Includes functionality for:
- Downloading and extracting F1DB data
- Validating data structure and schema
- Analyzing and fixing column mappings
- Checking data integrity
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import json
from datetime import datetime
import hashlib
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.schema_url = "https://raw.githubusercontent.com/f1db/f1db/main/src/schema/current/single/f1db.schema.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.schema_dir = self.data_dir / "schema"
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self._schema_cache = None
        self._schema_version = None
        
    def get_latest_release_info(self) -> Dict:
        """Get information about the latest F1DB release"""
        response = requests.get(self.base_url)
        response.raise_for_status()
        return response.json()
    
    def download_schema(self, force: bool = False) -> Dict[str, Any]:
        """
        Download and cache the F1DB schema
        
        Args:
            force: Force download even if schema exists
            
        Returns:
            The schema as a dictionary
        """
        schema_file = self.schema_dir / "f1db.schema.json"
        schema_version_file = self.schema_dir / ".schema_version"
        
        try:
            # Check if we need to download
            should_download = force or not schema_file.exists()
            
            if not should_download:
                # Check if schema is older than 24 hours
                if schema_file.exists():
                    mod_time = datetime.fromtimestamp(schema_file.stat().st_mtime)
                    if (datetime.now() - mod_time).total_seconds() > 86400:  # 24 hours
                        should_download = True
                        print("Schema is older than 24 hours, refreshing...")
            
            if should_download:
                print("Downloading F1DB schema...")
                response = requests.get(self.schema_url)
                response.raise_for_status()
                
                schema_data = response.json()
                
                # Save schema
                with open(schema_file, 'w', encoding='utf-8') as f:
                    json.dump(schema_data, f, indent=2)
                
                # Calculate and save schema hash as version
                schema_hash = hashlib.md5(response.text.encode()).hexdigest()
                with open(schema_version_file, 'w') as f:
                    f.write(schema_hash)
                
                print("Schema downloaded successfully")
                self._schema_cache = schema_data
                self._schema_version = schema_hash
            else:
                # Load from cache
                if self._schema_cache is None:
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        self._schema_cache = json.load(f)
                    
                    if schema_version_file.exists():
                        with open(schema_version_file, 'r') as f:
                            self._schema_version = f.read().strip()
            
            return self._schema_cache
            
        except Exception as e:
            print(f"Error downloading/loading schema: {e}")
            # Return empty schema as fallback
            return {}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the cached schema, downloading if necessary"""
        if self._schema_cache is None:
            self.download_schema()
        return self._schema_cache or {}
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema for the table or None if not found
        """
        schema = self.get_schema()
        
        # The schema structure has definitions for each entity
        if 'definitions' in schema:
            # Try different naming conventions
            possible_names = [
                table_name,
                table_name.replace('_', '-'),
                table_name.replace('-', '_')
            ]
            
            for name in possible_names:
                if name in schema['definitions']:
                    return schema['definitions'][name]
        
        return None
    
    def validate_dataframe_with_schema(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Validate a DataFrame against the schema
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table
            
        Returns:
            Validation results including warnings and errors
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'schema_found': False
        }
        
        table_schema = self.get_table_schema(table_name)
        if not table_schema:
            results['warnings'].append(f"No schema found for table '{table_name}'")
            return results
        
        results['schema_found'] = True
        
        # Check if schema has properties
        if 'properties' in table_schema:
            schema_columns = set(table_schema['properties'].keys())
            df_columns = set(df.columns)
            
            # Check for missing required columns
            if 'required' in table_schema:
                required_columns = set(table_schema['required'])
                missing_required = required_columns - df_columns
                if missing_required:
                    results['errors'].append(f"Missing required columns: {missing_required}")
                    results['valid'] = False
            
            # Check for extra columns not in schema
            extra_columns = df_columns - schema_columns
            if extra_columns:
                results['warnings'].append(f"Extra columns not in schema: {extra_columns}")
            
            # Check for missing optional columns
            missing_optional = schema_columns - df_columns
            if missing_optional:
                results['warnings'].append(f"Missing optional columns: {missing_optional}")
        
        return results
    
    def download_latest_data(self, force: bool = False, update_schema: bool = True) -> bool:
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
            
            # Check if we already have this version AND the CSV files
            if marker_file.exists() and not force:
                with open(marker_file, 'r') as f:
                    current_version = f.read().strip()
                if current_version == latest_version:
                    # Also check if CSV files actually exist
                    csv_files = list(self.data_dir.glob("*.csv"))
                    if csv_files:
                        print(f"Already have latest F1DB data (version {latest_version})")
                        return False
                    else:
                        print(f"Version file exists but CSV files are missing. Re-downloading...")
            
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
            
            # Rename files to remove f1db- prefix
            print("Renaming files to remove f1db- prefix...")
            for file_path in self.data_dir.glob("f1db-*.csv"):
                new_name = file_path.name[5:]  # Remove 'f1db-' prefix
                new_path = file_path.parent / new_name
                # Only rename if target doesn't exist
                if not new_path.exists():
                    file_path.rename(new_path)
                    print(f"  Renamed {file_path.name} to {new_name}")
                else:
                    # If target exists, remove the f1db- prefixed file
                    file_path.unlink()
                    print(f"  Removed {file_path.name} (target {new_name} already exists)")
            
            # Save version marker
            with open(marker_file, 'w') as f:
                f.write(latest_version)
            
            # Update schema when data is updated
            if update_schema:
                print("Updating schema...")
                self.download_schema(force=True)
            
            print(f"Successfully downloaded F1DB data version {latest_version}")
            return True
            
        except Exception as e:
            print(f"Error downloading F1DB data: {e}")
            raise
    
    def load_csv_data(self, validate: bool = True) -> Dict[str, pd.DataFrame]:
        """Load all CSV files into pandas DataFrames
        
        Args:
            validate: Whether to validate data against schema
            
        Returns:
            Dictionary of DataFrames
        """
        dataframes = {}
        
        # Ensure data is downloaded
        self.download_latest_data()
        
        # Ensure schema is loaded
        if validate:
            self.download_schema()
        
        # Load all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
            
        if not csv_files:
            raise FileNotFoundError("No CSV files found. Data may not be downloaded correctly.")
        
        print(f"Loading {len(csv_files)} CSV files...")
        validation_summary = {'valid': 0, 'warnings': 0, 'errors': 0}
        
        for csv_file in csv_files:
            # Get the table name
            table_name = csv_file.stem
            original_table_name = table_name
            
            # Replace dashes with underscores for consistency
            table_name = table_name.replace('-', '_')
            
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                dataframes[table_name] = df
                
                # Validate against schema if requested
                if validate:
                    validation = self.validate_dataframe_with_schema(df, original_table_name)
                    if validation['valid']:
                        validation_summary['valid'] += 1
                    if validation['warnings']:
                        validation_summary['warnings'] += len(validation['warnings'])
                    if validation['errors']:
                        validation_summary['errors'] += len(validation['errors'])
                        print(f"  ✗ Loaded {table_name}: {len(df)} rows (validation errors)")
                        for error in validation['errors']:
                            print(f"    Error: {error}")
                    else:
                        print(f"  ✓ Loaded {table_name}: {len(df)} rows")
                else:
                    print(f"  ✓ Loaded {table_name}: {len(df)} rows")
            except Exception as e:
                print(f"  ✗ Error loading {table_name}: {e}")
        
        if validate and validation_summary['warnings'] + validation_summary['errors'] > 0:
            print(f"\nValidation summary: {validation_summary['valid']} valid, "
                  f"{validation_summary['warnings']} warnings, {validation_summary['errors']} errors")
        
        return dataframes
    
    def get_core_datasets(self, fix_columns: bool = True) -> Dict[str, pd.DataFrame]:
        """Get the core datasets commonly used for analysis"""
        all_data = self.load_csv_data()
        
        # Define core datasets with fallback names for compatibility
        table_mappings = {
            # Standard name: [possible F1DB names]
            'races': ['races', 'race'],
            'drivers': ['drivers', 'driver'],
            'constructors': ['constructors', 'constructor', 'teams'],
            'circuits': ['circuits', 'circuit', 'tracks'],
            'results': ['races_race_results', 'race_results', 'results', 'race_result'],
            'qualifying': ['races_qualifying_results', 'qualifying_results', 'qualifying', 'qualifying_result'],
            'sprint_results': ['races_sprint_race_results', 'sprint_race_results', 'sprint_results', 'sprint_race_result'],
            'fastest_laps': ['races_fastest_laps', 'fastest_laps', 'fastest_lap'],
            'pit_stops': ['races_pit_stops', 'pit_stops', 'pitstops', 'pit_stop'],
            'driver_standings': ['races_driver_standings', 'race_driver_standings', 'driver_standings', 'driverStandings'],
            'constructor_standings': ['races_constructor_standings', 'race_constructor_standings', 'constructor_standings', 'constructorStandings'],
            'seasons': ['seasons', 'season'],
            'lap_times': ['races_laps', 'lap_times', 'lapTimes', 'laps'],
            'status': ['race_result_status', 'status'],
            'constructor_results': ['races_race_results', 'race_results', 'results'],  # Added for constructor results
            'races_starting_grid_positions': ['races_starting_grid_positions', 'races-starting-grid-positions', 'starting_grid_positions']
        }
        
        core_data = {}
        for standard_name, possible_names in table_mappings.items():
            found = False
            for name in possible_names:
                if name in all_data:
                    core_data[standard_name] = all_data[name]
                    if name != standard_name:
                        print(f"  → Mapped {name} to {standard_name}")
                    found = True
                    break
            
            if not found and standard_name in ['races', 'drivers', 'results']:
                print(f"Warning: Core table '{standard_name}' not found in data")
        
        # Also include any other tables that might be useful
        core_data_names = set(core_data.keys())
        for table_name, df in all_data.items():
            if table_name not in core_data_names:
                core_data[table_name] = df
        
        # Create a synthetic status table for compatibility with older code
        if 'results' in core_data and 'status' not in core_data:
            # Extract unique statuses from reasonRetired column
            results_df = core_data['results']
            if 'reasonRetired' in results_df.columns:
                unique_statuses = results_df['reasonRetired'].dropna().unique()
                # Create a status dataframe
                status_df = pd.DataFrame({
                    'statusId': range(1, len(unique_statuses) + 2),
                    'status': ['Finished'] + list(unique_statuses)
                })
                core_data['status'] = status_df
                print("  → Created synthetic status table from reasonRetired data")
        
        # Add positionOrder column to results for compatibility
        if 'results' in core_data:
            results_df = core_data['results']
            if 'positionOrder' not in results_df.columns and 'positionNumber' in results_df.columns:
                results_df['positionOrder'] = results_df['positionNumber']
                core_data['results'] = results_df
                print("  → Added positionOrder column to results for compatibility")
        
        # Apply column mappings if requested
        if fix_columns:
            core_data = self.fix_column_mappings(core_data)
            print("  → Applied column mappings for ML pipeline compatibility")
        
        return core_data
    
    def get_data_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a data dictionary from the schema
        
        Returns:
            Dictionary mapping table names to their field descriptions
        """
        schema = self.get_schema()
        data_dict = {}
        
        if 'definitions' in schema:
            for entity_name, entity_schema in schema['definitions'].items():
                if 'properties' in entity_schema:
                    fields = {}
                    for field_name, field_schema in entity_schema['properties'].items():
                        field_info = {
                            'type': field_schema.get('type', 'unknown'),
                            'description': field_schema.get('description', ''),
                            'format': field_schema.get('format', ''),
                            'required': field_name in entity_schema.get('required', [])
                        }
                        
                        # Add additional metadata if available
                        if 'enum' in field_schema:
                            field_info['enum'] = field_schema['enum']
                        if 'minimum' in field_schema:
                            field_info['minimum'] = field_schema['minimum']
                        if 'maximum' in field_schema:
                            field_info['maximum'] = field_schema['maximum']
                        
                        fields[field_name] = field_info
                    
                    data_dict[entity_name] = {
                        'description': entity_schema.get('description', ''),
                        'fields': fields
                    }
        
        return data_dict
    
    def print_table_info(self, table_name: str) -> None:
        """
        Print detailed information about a table from the schema
        
        Args:
            table_name: Name of the table
        """
        table_schema = self.get_table_schema(table_name)
        if not table_schema:
            print(f"No schema found for table '{table_name}'")
            return
        
        print(f"\nTable: {table_name}")
        if 'description' in table_schema:
            print(f"Description: {table_schema['description']}")
        
        if 'properties' in table_schema:
            print("\nFields:")
            required_fields = set(table_schema.get('required', []))
            
            for field_name, field_schema in table_schema['properties'].items():
                field_type = field_schema.get('type', 'unknown')
                required = '(required)' if field_name in required_fields else '(optional)'
                description = field_schema.get('description', '')
                
                print(f"  - {field_name}: {field_type} {required}")
                if description:
                    print(f"    {description}")
                if 'enum' in field_schema:
                    print(f"    Allowed values: {', '.join(map(str, field_schema['enum']))}")
    
    def fix_data_download(self) -> bool:
        """
        Fix data download issues by forcing re-download and handling extraction
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Fixing F1DB data download...")
            
            # Remove version file to force re-download
            version_file = self.data_dir / ".f1db_version"
            if version_file.exists():
                os.remove(version_file)
                logger.info("Removed version file to force re-download")
            
            # Download the data
            self.download_latest_data(force=True)
            
            # Check what was downloaded
            logger.info("Checking downloaded files...")
            for item in self.data_dir.iterdir():
                logger.info(f"  - {item.name} {'(dir)' if item.is_dir() else ''}")
            
            # Check for CSV files in subdirectories
            csv_path = self.data_dir / "csv"
            if csv_path.exists():
                csv_files = list(csv_path.glob("*.csv"))
                logger.info(f"Found {len(csv_files)} CSV files in {csv_path}")
                
                # Move CSV files to main data directory if needed
                for csv_file in csv_files:
                    # Remove f1db- prefix if present
                    dest_name = csv_file.name[5:] if csv_file.name.startswith('f1db-') else csv_file.name
                    dest = self.data_dir / dest_name
                    if not dest.exists():
                        csv_file.rename(dest)
                        logger.info(f"Moved {csv_file.name} to {dest_name}")
            
            # Check for any remaining zip files
            zip_files = list(self.data_dir.glob("*.zip"))
            if zip_files:
                logger.info(f"Found {len(zip_files)} zip files, extracting...")
                for z in zip_files:
                    logger.info(f"Extracting {z.name}...")
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(self.data_dir)
                    z.unlink()  # Remove zip after extraction
                    logger.info("Extracted and removed zip file")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing data download: {e}")
            traceback.print_exc()
            return False
    
    def check_data_structure(self) -> Dict[str, Any]:
        """
        Check F1DB data structure and analyze column names
        
        Returns:
            Dictionary with structure analysis results
        """
        logger.info("Checking F1DB data structure...")
        
        # Load data
        data = self.load_csv_data(validate=False)
        
        # Key datasets to check
        key_datasets = ['races', 'results', 'drivers', 'constructors', 'circuits']
        
        analysis = {
            'found_datasets': {},
            'missing_datasets': [],
            'column_analysis': {}
        }
        
        for dataset_name in key_datasets:
            # Try different naming conventions
            possible_names = [
                dataset_name,
                f'races-{dataset_name}',
                f'races-race-{dataset_name}',
                f'races_race_{dataset_name}',
                f'races_{dataset_name}'
            ]
            
            found = False
            for name in possible_names:
                if name in data:
                    df = data[name]
                    analysis['found_datasets'][dataset_name] = {
                        'actual_name': name,
                        'shape': df.shape,
                        'columns': list(df.columns)
                    }
                    found = True
                    logger.info(f"{dataset_name.upper()} found as '{name}': {df.shape}")
                    break
            
            if not found:
                analysis['missing_datasets'].append(dataset_name)
                logger.warning(f"{dataset_name.upper()}: NOT FOUND")
        
        # Check for results data specifically
        results_tables = [k for k in data.keys() if 'result' in k.lower()]
        analysis['results_tables'] = results_tables
        
        return analysis
    
    def analyze_column_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Analyze column mapping between expected and actual F1DB columns
        
        Returns:
            Dictionary mapping expected columns to actual columns
        """
        logger.info("Analyzing column mappings...")
        
        # Load data
        data = self.get_core_datasets()
        
        # Expected columns in notebooks (common usage)
        expected_columns = {
            'results': ['positionOrder', 'points', 'grid', 'statusId', 'driverId', 
                       'constructorId', 'raceId', 'position', 'status', 'time', 'laps'],
            'races': ['raceId', 'year', 'round', 'circuitId', 'name', 'date', 'time', 'url'],
            'drivers': ['driverId', 'driverRef', 'number', 'code', 'forename', 'surname', 
                       'dob', 'nationality', 'url'],
            'constructors': ['constructorId', 'constructorRef', 'name', 'nationality', 'url']
        }
        
        # Common column mappings
        common_mappings = {
            'raceId': ['id', 'race_id'],
            'driverId': ['id', 'driver_id'],
            'constructorId': ['id', 'constructor_id'],
            'driverRef': ['abbreviation', 'name'],
            'constructorRef': ['name'],
            'forename': ['firstName', 'first_name'],
            'surname': ['lastName', 'last_name'],
            'dob': ['dateOfBirth', 'date_of_birth'],
            'grid': ['gridPositionNumber', 'gridPosition', 'startingGridPosition'],
            'position': ['positionNumber', 'positionOrder'],
            'statusId': ['reasonRetired', 'status'],
            'code': ['abbreviation'],
            'url': ['link', 'website'],
            'circuitId': ['id', 'circuit_id']
        }
        
        mappings = {}
        
        for table_name, expected_cols in expected_columns.items():
            if table_name not in data:
                logger.warning(f"Table '{table_name}' not found in data")
                continue
            
            df = data[table_name]
            actual_cols = list(df.columns)
            
            table_mapping = {}
            
            for exp_col in expected_cols:
                if exp_col in actual_cols:
                    table_mapping[exp_col] = [exp_col]  # Exact match
                else:
                    # Try to find matches
                    matches = []
                    
                    # Check for exact substring
                    for act_col in actual_cols:
                        if exp_col.lower() in act_col.lower() or act_col.lower() in exp_col.lower():
                            matches.append(act_col)
                    
                    # Check common mappings
                    if exp_col in common_mappings:
                        for possible in common_mappings[exp_col]:
                            if possible in actual_cols and possible not in matches:
                                matches.append(possible)
                    
                    table_mapping[exp_col] = matches
            
            mappings[table_name] = table_mapping
        
        return mappings
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity and check for common issues
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data integrity...")
        
        data = self.get_core_datasets()
        validation_results = {
            'tables_checked': 0,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check key relationships
        if 'races' in data and 'results' in data:
            races_df = data['races']
            results_df = data['results']
            
            # Check if race IDs match
            race_ids = set(races_df['id']) if 'id' in races_df.columns else set()
            result_race_ids = set(results_df['race_id']) if 'race_id' in results_df.columns else set()
            
            if race_ids and result_race_ids:
                orphan_results = result_race_ids - race_ids
                if orphan_results:
                    validation_results['warnings'].append(
                        f"Found {len(orphan_results)} results with no matching race"
                    )
        
        # Check for required columns and data types
        for table_name, df in data.items():
            validation_results['tables_checked'] += 1
            
            # Check for null values in key columns
            null_counts = df.isnull().sum()
            high_null_cols = null_counts[null_counts > len(df) * 0.5]
            if len(high_null_cols) > 0:
                validation_results['warnings'].append(
                    f"Table '{table_name}' has columns with >50% null values: {list(high_null_cols.index)}"
                )
            
            # Collect statistics
            validation_results['statistics'][table_name] = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            }
        
        return validation_results
    
    def fix_column_mappings(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Fix column mappings to ensure compatibility with ML pipeline
        
        Args:
            data: Dictionary of DataFrames from F1DB
            
        Returns:
            Dictionary with fixed column mappings
        """
        # Fix results columns
        if 'results' in data:
            df = data['results']
            
            # Map grid column
            if 'grid' not in df.columns and 'gridPositionNumber' in df.columns:
                df['grid'] = df['gridPositionNumber']
                
            # Map position column
            if 'position' not in df.columns and 'positionNumber' in df.columns:
                df['position'] = df['positionNumber']
                
            # Create statusId from reasonRetired
            if 'statusId' not in df.columns and 'reasonRetired' in df.columns:
                # Map common retirement reasons to status IDs
                status_mapping = {
                    'Finished': 1,
                    'Accident': 2,
                    'Collision': 3,
                    'Engine': 4,
                    'Gearbox': 5,
                    'Transmission': 6,
                    'Clutch': 7,
                    'Hydraulics': 8,
                    'Electrical': 9,
                    '+1 Lap': 11,
                    '+2 Laps': 12,
                    '+3 Laps': 13,
                    '+4 Laps': 14,
                    '+5 Laps': 15,
                    '+6 Laps': 16,
                    '+7 Laps': 17,
                    '+8 Laps': 18,
                    '+9 Laps': 19,
                    'Spun off': 20,
                    'Radiator': 21,
                    'Suspension': 22,
                    'Brakes': 23,
                    'Differential': 24,
                    'Overheating': 25,
                    'Mechanical': 26,
                    'Tyre': 27,
                    'Driver Seat': 28,
                    'Puncture': 29,
                    'Disqualified': 30,
                    'Wheel': 31,
                    'Fuel system': 32,
                    'Throttle': 33,
                    'Steering': 34,
                    'Technical': 35,
                    'Electronics': 36,
                    'Broken wing': 37,
                    'Heat shield fire': 38,
                    'Exhaust': 39,
                    'Oil leak': 40
                }
                
                # Create statusId column
                df['statusId'] = df['reasonRetired'].fillna('Finished').map(
                    lambda x: status_mapping.get(x, 50)  # 50 for unknown
                )
                
            # Create status column if missing
            if 'status' not in df.columns:
                df['status'] = df['reasonRetired'].fillna('Finished')
                
            # Add DNF indicator
            df['dnf'] = (df['statusId'] > 1).astype(int)
            
            # Add win, podium, points columns
            df['win'] = (df['positionOrder'] == 1).astype(int)
            df['podium'] = (df['positionOrder'] <= 3).astype(int)
            df['points_finish'] = (df['points'] > 0).astype(int)
            
            # Map race_id to raceId if needed
            if 'raceId' not in df.columns and 'race_id' in df.columns:
                df['raceId'] = df['race_id']
                
            # Map driver_id to driverId if needed
            if 'driverId' not in df.columns and 'driver_id' in df.columns:
                df['driverId'] = df['driver_id']
                
            # Map constructor_id to constructorId if needed
            if 'constructorId' not in df.columns and 'constructor_id' in df.columns:
                df['constructorId'] = df['constructor_id']
                
            # Ensure we save the changes
            data['results'] = df
            
        # Fix races columns
        if 'races' in data:
            df = data['races']
            
            # Map id to raceId
            if 'raceId' not in df.columns and 'id' in df.columns:
                df['raceId'] = df['id']
                
            # Map officialName to name
            if 'name' not in df.columns and 'officialName' in df.columns:
                df['name'] = df['officialName']
                
            # Ensure we have the columns we need
            data['races'] = df
                
        # Fix drivers columns
        if 'drivers' in data:
            df = data['drivers']
            
            # Map id to driverId
            if 'driverId' not in df.columns and 'id' in df.columns:
                df['driverId'] = df['id']
                
            # Map names
            if 'forename' not in df.columns and 'firstName' in df.columns:
                df['forename'] = df['firstName']
                
            if 'surname' not in df.columns and 'lastName' in df.columns:
                df['surname'] = df['lastName']
                
            if 'driverRef' not in df.columns and 'abbreviation' in df.columns:
                df['driverRef'] = df['abbreviation']
                
            if 'code' not in df.columns and 'abbreviation' in df.columns:
                df['code'] = df['abbreviation']
                
            if 'dob' not in df.columns and 'dateOfBirth' in df.columns:
                df['dob'] = df['dateOfBirth']
                
            if 'nationality' not in df.columns and 'nationalityCountryId' in df.columns:
                df['nationality'] = df['nationalityCountryId']
                
            if 'number' not in df.columns and 'permanentNumber' in df.columns:
                df['number'] = df['permanentNumber']
                
            # Ensure we save the changes
            data['drivers'] = df
                
        # Fix constructors columns
        if 'constructors' in data:
            df = data['constructors']
            
            # Map id to constructorId
            if 'constructorId' not in df.columns and 'id' in df.columns:
                df['constructorId'] = df['id']
                
            if 'constructorRef' not in df.columns and 'name' in df.columns:
                df['constructorRef'] = df['name']
                
            if 'nationality' not in df.columns and 'countryId' in df.columns:
                df['nationality'] = df['countryId']
                
            # Ensure we save the changes
            data['constructors'] = df
                
        return data
    
    def merge_race_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge race-related data into a single DataFrame for analysis
        
        Args:
            data: Dictionary of DataFrames from F1DB
            
        Returns:
            Merged DataFrame with race results and related data
        """
        # Start with results
        if 'results' not in data:
            raise ValueError("No results data found")
            
        df = data['results'].copy()
        
        # Merge with races
        if 'races' in data:
            races_cols = ['raceId', 'year', 'round', 'circuitId', 'name', 'date']
            races_data = data['races'][races_cols].drop_duplicates(subset=['raceId'])
            df = df.merge(races_data, on='raceId', how='left', suffixes=('', '_race'))
            
            # Handle duplicate columns
            if 'year_race' in df.columns:
                df['year'] = df['year'].fillna(df['year_race'])
                df.drop('year_race', axis=1, inplace=True)
                
            if 'round_race' in df.columns:
                df['round'] = df['round'].fillna(df['round_race'])
                df.drop('round_race', axis=1, inplace=True)
        
        # Merge with drivers
        if 'drivers' in data:
            driver_cols = ['driverId', 'forename', 'surname', 'dob', 'nationality', 'driverRef']
            driver_data = data['drivers'][driver_cols].drop_duplicates(subset=['driverId'])
            df = df.merge(driver_data, on='driverId', how='left')
            
            # Calculate driver age at race time
            if 'date' in df.columns and 'dob' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['dob'] = pd.to_datetime(df['dob'])
                df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
        
        # Merge with constructors
        if 'constructors' in data:
            const_cols = ['constructorId', 'name', 'nationality']
            const_data = data['constructors'][const_cols].drop_duplicates(subset=['constructorId'])
            const_data.columns = ['constructorId', 'constructor_name', 'constructor_nationality']
            df = df.merge(const_data, on='constructorId', how='left')
            
        # Sort by race and finishing position
        df = df.sort_values(['raceId', 'positionOrder'])
        
        return df
    
    def get_recent_results(self, data: Dict[str, pd.DataFrame], n_years: int = 5) -> pd.DataFrame:
        """
        Get recent race results for analysis
        
        Args:
            data: Dictionary of DataFrames from F1DB
            n_years: Number of recent years to include
            
        Returns:
            DataFrame with recent results
        """
        df = self.merge_race_data(data)
        
        # Filter to recent years
        if 'year' in df.columns:
            max_year = df['year'].max()
            min_year = max_year - n_years + 1
            df = df[df['year'] >= min_year]
        
        return df
    
    def calculate_driver_stats(self, results_df: pd.DataFrame, driver_id: str) -> Dict:
        """
        Calculate driver statistics from results
        
        Args:
            results_df: DataFrame with race results
            driver_id: Driver ID to calculate stats for
            
        Returns:
            Dictionary of driver statistics
        """
        driver_results = results_df[results_df['driverId'] == driver_id]
        
        if len(driver_results) == 0:
            return {}
        
        stats = {
            'driverId': driver_id,
            'races': len(driver_results),
            'wins': (driver_results['positionOrder'] == 1).sum(),
            'podiums': (driver_results['positionOrder'] <= 3).sum(),
            'points': driver_results['points'].sum(),
            'dnf_rate': driver_results['dnf'].mean() if 'dnf' in driver_results else 0,
            'avg_position': driver_results['positionOrder'].mean(),
            'avg_grid': driver_results['grid'].mean() if 'grid' in driver_results else None,
            'best_result': driver_results['positionOrder'].min(),
            'points_finishes': (driver_results['points'] > 0).sum()
        }
        
        return stats
    
    def print_data_summary(self) -> None:
        """
        Print a comprehensive summary of the F1DB data
        """
        print("\n" + "=" * 80)
        print("F1DB DATA SUMMARY")
        print("=" * 80)
        
        # Check data structure
        structure = self.check_data_structure()
        
        print("\nFound Datasets:")
        for name, info in structure['found_datasets'].items():
            print(f"  - {name}: {info['shape'][0]} rows, {info['shape'][1]} columns")
            print(f"    (loaded as '{info['actual_name']}')")
        
        if structure['missing_datasets']:
            print(f"\nMissing Datasets: {', '.join(structure['missing_datasets'])}")
        
        # Analyze column mappings
        print("\n" + "-" * 40)
        print("COLUMN MAPPING ANALYSIS")
        print("-" * 40)
        
        mappings = self.analyze_column_mapping()
        for table_name, table_mappings in mappings.items():
            print(f"\n{table_name.upper()}:")
            for expected, actual_matches in table_mappings.items():
                if actual_matches and actual_matches[0] == expected:
                    print(f"  ✓ {expected}")
                elif actual_matches:
                    print(f"  → {expected} maps to: {actual_matches[0]}")
                else:
                    print(f"  ✗ {expected} NOT FOUND")
        
        # Validate data integrity
        print("\n" + "-" * 40)
        print("DATA INTEGRITY")
        print("-" * 40)
        
        validation = self.validate_data_integrity()
        print(f"\nTables checked: {validation['tables_checked']}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
        
        if validation['issues']:
            print("\nIssues:")
            for issue in validation['issues']:
                print(f"  ✗ {issue}")
        
        print("\nTable Statistics:")
        for table, stats in validation['statistics'].items():
            print(f"  - {table}: {stats['row_count']:,} rows, "
                  f"{stats['null_percentage']:.1f}% null values")
    
    def test_data_loading(self) -> None:
        """
        Test data loading to understand actual data structure
        """
        print("Testing F1DB data loading...")
        print("=" * 60)
        
        # Load data
        data = self.get_core_datasets()
        
        if data:
            print(f"\nLoaded {len(data)} datasets:")
            for name, df in data.items():
                if hasattr(df, 'shape'):
                    print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    if hasattr(df, 'columns'):
                        print(f"    Columns: {', '.join(df.columns[:5])}{', ...' if len(df.columns) > 5 else ''}")
                else:
                    print(f"  - {name}: {type(df)}")
        else:
            print("\nNo data loaded. Checking for existing data files...")
            
            # Check what files exist
            possible_paths = [
                Path("../../data/f1db"),
                Path("../data/f1db"),
                Path("data/f1db"),
                Path("./data/f1db"),
                Path("/workspace/data/f1db")
            ]
            
            for path in possible_paths:
                if path.exists():
                    print(f"\nChecking {path}:")
                    # List CSV files
                    csv_files = list(path.glob("*.csv"))
                    if csv_files:
                        print(f"  Found {len(csv_files)} CSV files:")
                        for f in csv_files[:10]:  # Show first 10
                            print(f"    - {f.name}")
                    else:
                        print("  No CSV files found")
                        # Check subdirectories
                        subdirs = [d for d in path.iterdir() if d.is_dir()]
                        if subdirs:
                            print(f"  Subdirectories: {', '.join(d.name for d in subdirs)}")
        
        # Test the data loader directly
        print("\n" + "=" * 60)
        print("Testing F1DBDataLoader directly...")
        
        print(f"Data directory: {self.data_dir}")
        
        # Check if data needs to be downloaded
        data_path = self.data_dir / "csv"
        if not data_path.exists():
            print("\nF1DB data not found locally. Would need to download.")
            print("Run: loader.download_latest_data() to fetch data")
        else:
            print(f"\nData found at: {data_path}")
            csv_files = list(data_path.glob("*.csv"))
            print(f"Found {len(csv_files)} CSV files")
            
            if csv_files:
                # Try loading a sample file
                sample_file = csv_files[0]
                print(f"\nSample file: {sample_file.name}")
                df = pd.read_csv(sample_file)
                print(f"Shape: {df.shape}")
                print(f"Columns: {', '.join(df.columns[:5])}{', ...' if len(df.columns) > 5 else ''}")


# Convenience function for notebook usage
def load_f1db_data(data_dir: Optional[str] = None, format: str = "csv", force_download: bool = False, validate: bool = True, fix_issues: bool = False, fix_columns: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load F1DB data with a simple function call
    
    Args:
        data_dir: Directory to store/load data (if None, uses ../../data relative to this file)
        format: Data format ('csv' recommended)
        force_download: Force re-download of data
        validate: Validate data against schema
        fix_issues: Attempt to fix any data download/extraction issues
        fix_columns: Apply column mappings for ML pipeline compatibility
        
    Returns:
        Dictionary of DataFrames with F1 data
    """
    if data_dir is None:
        # Determine the correct path relative to this file
        current_file = Path(__file__).resolve()
        # Go up to project root (2 levels from notebooks/advanced/)
        project_root = current_file.parent.parent.parent
        data_dir = str(project_root / "data" / "f1db")
    
    loader = F1DBDataLoader(str(data_dir), format)
    
    if fix_issues:
        loader.fix_data_download()
    else:
        loader.download_latest_data(force=force_download)
    
    return loader.get_core_datasets(fix_columns=fix_columns)


# Additional convenience functions from data_utils
def fix_column_mappings(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Convenience function that uses the loader's fix_column_mappings method"""
    # Use the same default data directory as load_f1db_data
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = str(project_root / "data" / "f1db")
    loader = F1DBDataLoader(data_dir)
    return loader.fix_column_mappings(data)

def merge_race_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience function that uses the loader's merge_race_data method"""
    # Use the same default data directory as load_f1db_data
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = str(project_root / "data" / "f1db")
    loader = F1DBDataLoader(data_dir)
    return loader.merge_race_data(data)

def get_recent_results(data: Dict[str, pd.DataFrame], n_years: int = 5) -> pd.DataFrame:
    """Convenience function that uses the loader's get_recent_results method"""
    # Use the same default data directory as load_f1db_data
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = str(project_root / "data" / "f1db")
    loader = F1DBDataLoader(data_dir)
    return loader.get_recent_results(data, n_years)

def calculate_driver_stats(results_df: pd.DataFrame, driver_id: str) -> Dict:
    """Convenience function that uses the loader's calculate_driver_stats method"""
    # Use the same default data directory as load_f1db_data
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = str(project_root / "data" / "f1db")
    loader = F1DBDataLoader(data_dir)
    return loader.calculate_driver_stats(results_df, driver_id)

# Export key functions and classes
__all__ = [
    'F1DBDataLoader',
    'load_f1db_data',
    'fix_column_mappings',
    'merge_race_data',
    'get_recent_results',
    'calculate_driver_stats'
]

# Example usage in notebooks:
# from f1db_data_loader import load_f1db_data
# data = load_f1db_data()
# races_df = data['races']
# drivers_df = data['drivers']

def main():
    """Command line interface for F1DB Data Loader"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "summary"
    
    # Use the same default data directory as load_f1db_data
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = str(project_root / "data" / "f1db")
    loader = F1DBDataLoader(data_dir)
    
    if command == "check":
        print("Checking data structure...")
        loader.check_data_structure()
    elif command == "validate":
        print("Validating data integrity...")
        loader.validate_data_integrity()
    elif command == "summary":
        print("Data summary:")
        loader.print_data_summary()
    elif command == "fix":
        print("Fixing data issues...")
        data = load_f1db_data(fix_issues=True)
        print(f"Data loaded and fixed: {len(data)} datasets")
    elif command == "load":
        print("Loading all data...")
        data = load_f1db_data()
        print(f"\nLoaded datasets:")
        for name, df in data.items():
            print(f"  - {name}: {len(df)} rows")
    elif command == "download":
        print("Downloading latest F1DB data...")
        loader.download_latest_data(force=True)
        print("Download complete!")
    elif command == "version":
        current = loader.get_current_version()
        latest = loader.get_latest_version()
        print(f"Current version: {current}")
        print(f"Latest version: {latest}")
        if current != latest:
            print("Update available! Run 'python f1db_data_loader.py download' to update")
    else:
        print("Usage: python f1db_data_loader.py [check|validate|summary|fix|load|download|version]")
        print()
        print("Commands:")
        print("  check     - Check data structure and columns")
        print("  validate  - Validate data integrity")
        print("  summary   - Print comprehensive data summary")
        print("  fix       - Load data and fix any issues")
        print("  load      - Load all datasets and show counts")
        print("  download  - Download latest F1DB data")
        print("  version   - Check current vs latest version")


# Test functionality if run directly
if __name__ == "__main__":
    main()