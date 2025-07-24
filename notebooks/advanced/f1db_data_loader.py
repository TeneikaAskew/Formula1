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
from typing import Dict, Optional, Any
import json
from datetime import datetime
import hashlib

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
        self.data_dir.mkdir(exist_ok=True)
        self.schema_dir = self.data_dir / "schema"
        self.schema_dir.mkdir(exist_ok=True)
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
                table_name.replace('-', '_'),
                f"f1db-{table_name}",
                f"f1db_{table_name}"
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
        
        # Load all CSV files (try with f1db- prefix first)
        csv_files = list(self.data_dir.glob("f1db-*.csv"))
        if not csv_files:
            # Fallback to files without prefix
            csv_files = list(self.data_dir.glob("*.csv"))
            
        if not csv_files:
            raise FileNotFoundError("No CSV files found. Data may not be downloaded correctly.")
        
        print(f"Loading {len(csv_files)} CSV files...")
        validation_summary = {'valid': 0, 'warnings': 0, 'errors': 0}
        
        for csv_file in csv_files:
            # Get the table name, removing f1db- prefix if present
            table_name = csv_file.stem
            original_table_name = table_name
            if table_name.startswith('f1db-'):
                table_name = table_name[5:]  # Remove 'f1db-' prefix
            
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
    
    def get_core_datasets(self) -> Dict[str, pd.DataFrame]:
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
            'constructor_results': ['races_race_results', 'race_results', 'results']  # Added for constructor results
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


# Convenience function for notebook usage
def load_f1db_data(data_dir: Optional[str] = None, format: str = "csv", force_download: bool = False, validate: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load F1DB data with a simple function call
    
    Args:
        data_dir: Directory to store/load data (if None, uses ../../data relative to this file)
        format: Data format ('csv' recommended)
        force_download: Force re-download of data
        validate: Validate data against schema
        
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
    loader.download_latest_data(force=force_download)
    return loader.get_core_datasets()


# Example usage in notebooks:
# from f1db_data_loader import load_f1db_data
# data = load_f1db_data()
# races_df = data['races']
# drivers_df = data['drivers']

# Test functionality if run directly
if __name__ == "__main__":
    print("\nTesting F1DB data loader...")
    
    # Show where data will be saved
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    default_data_dir = project_root / "data" / "f1db"
    print(f"Default data directory: {default_data_dir}")
    
    # Initialize loader
    loader = F1DBDataLoader(str(default_data_dir))
    
    # Test schema loading
    print("\nTesting schema functionality...")
    schema = loader.download_schema()
    print(f"Schema loaded: {'definitions' in schema}")
    
    # Load data with validation
    data = load_f1db_data(validate=True)
    print(f"\nSuccessfully loaded {len(data)} datasets!")
    print("\nAvailable datasets:")
    for name in sorted(data.keys()):
        print(f"  - {name}: {len(data[name])} rows")
    
    # Show sample table info
    print("\nSample table information:")
    loader.print_table_info('races')
    
    # Show sample of races data
    if 'races' in data:
        print(f"\nSample races data:")
        print(data['races'].head())