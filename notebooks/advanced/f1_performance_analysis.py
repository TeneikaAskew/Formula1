"""
F1 Performance Analysis Module

Generates comprehensive driver performance tables including:
- Overtakes by driver
- F1 points analysis
- Pit stop times
- Starting positions
- Sprint points
- Circuit-specific predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from lap_by_lap_overtakes import LapByLapOvertakeAnalyzer

class F1PerformanceAnalyzer:
    """Analyzes driver performance across multiple metrics"""
    
    def __init__(self, data_dict):
        self.data = data_dict
        # Dynamically determine current season from the data
        self.current_year = self._get_current_season()
        # Use today's actual date for finding the next race
        self.current_date = datetime.now()
        # Load DHL pit stop data
        self.dhl_data = self._load_dhl_data()
        # Load F1 Fantasy data
        self.fantasy_data = self._load_fantasy_data()
        # Initialize lap-by-lap overtake analyzer
        try:
            self.lap_analyzer = LapByLapOvertakeAnalyzer()
            print("✓ Lap-by-lap overtake analysis enabled")
        except Exception as e:
            print(f"⚠ Lap-by-lap data not available: {e}")
            self.lap_analyzer = None
    
    def _get_current_season(self):
        """Dynamically determine the current season from race results"""
        results = self.data.get('results', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        # First try results data
        if not results.empty and 'year' in results.columns:
            # Get the latest year that has actual race results (not just scheduled)
            results_with_data = results[results['positionNumber'].notna()]
            if not results_with_data.empty:
                return int(results_with_data['year'].max())
        
        # Fallback to races data
        if not races.empty and 'year' in races.columns:
            # Get races that have already happened (date in the past)
            races['date'] = pd.to_datetime(races['date'])
            past_races = races[races['date'] <= datetime.now()]
            if not past_races.empty:
                return int(past_races['year'].max())
        
        # Final fallback - use the latest year with any results
        if not results.empty and 'year' in results.columns:
            return int(results['year'].max())
        
        return datetime.now().year
    
    def _load_dhl_data(self):
        """Load DHL pit stop data from CSV file"""
        try:
            # Look for DHL data in data/dhl directory
            dhl_dir = Path("../../data/dhl")  # Relative to notebooks/advanced
            if not dhl_dir.exists():
                dhl_dir = Path("../data/dhl")  # Relative to notebooks
            if not dhl_dir.exists():
                dhl_dir = Path("data/dhl")  # From workspace root
            
            if dhl_dir.exists():
                # Look specifically for integrated pit stops files first
                integrated_files = list(dhl_dir.glob("dhl_pitstops_integrated_*.csv"))
                if integrated_files:
                    # Get the most recent integrated file
                    latest_file = max(integrated_files, key=lambda p: p.stat().st_mtime)
                    print(f"Loading DHL integrated pit stop data from: {latest_file}")
                    
                    df = pd.read_csv(latest_file)
                    # Debug: print column names
                    print(f"DHL data columns: {list(df.columns)}")
                    
                    # Check if circuit_id exists
                    if 'circuit_id' in df.columns:
                        print(f"Found circuit_id column with {df['circuit_id'].nunique()} unique circuits")
                    
                    # Ensure consistent column names
                    if 'time' in df.columns:
                        df['time_seconds'] = df['time']
                    
                    return df
                else:
                    # Fallback to any CSV file
                    csv_files = list(dhl_dir.glob("*.csv"))
                    if csv_files:
                        latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                        print(f"Loading DHL pit stop data from: {latest_file}")
                        
                        df = pd.read_csv(latest_file)
                        
                        # Ensure consistent column names
                        if 'time' in df.columns:
                            df['time_seconds'] = df['time']
                        
                        return df
            
            print("No DHL pit stop data found")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error loading DHL data: {e}")
            return pd.DataFrame()
    
    def _load_fantasy_data(self):
        """Load F1 Fantasy data from CSV files"""
        try:
            # Look for Fantasy data in data/f1_fantasy directory
            fantasy_dir = Path("../../data/f1_fantasy")  # Relative to notebooks/advanced
            if not fantasy_dir.exists():
                fantasy_dir = Path("../data/f1_fantasy")  # Relative to notebooks
            if not fantasy_dir.exists():
                fantasy_dir = Path("data/f1_fantasy")  # From workspace root
            
            if fantasy_dir.exists():
                fantasy_data = {}
                
                # Load driver overview
                overview_file = fantasy_dir / "driver_overview.csv"
                if overview_file.exists():
                    fantasy_data['driver_overview'] = pd.read_csv(overview_file)
                
                # Load driver details
                details_file = fantasy_dir / "driver_details.csv"
                if details_file.exists():
                    fantasy_data['driver_details'] = pd.read_csv(details_file)
                    
                if fantasy_data:
                    print(f"Loaded F1 Fantasy data from: {fantasy_dir}")
                    return fantasy_data
                    
            print("No F1 Fantasy data found")
            return {}
        except Exception as e:
            print(f"Error loading F1 Fantasy data: {e}")
            return {}
        
    def get_next_race(self):
        """Get the next upcoming race (first race without results)"""
        races = self.data.get('races', pd.DataFrame()).copy()
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame()))
        
        if races.empty or 'date' not in races.columns:
            return None
            
        races['date'] = pd.to_datetime(races['date'])
        
        # Get races that have results
        if not results.empty:
            races_with_results = results['raceId'].unique()
            # Find first race without results
            races_without_results = races[~races['id'].isin(races_with_results)]
            
            if not races_without_results.empty:
                # Return the earliest race without results
                return races_without_results.sort_values('date').iloc[0]
        
        # Fallback to date-based logic if no results data
        upcoming = races[races['date'] > datetime.now()].sort_values('date')
        
        if upcoming.empty:
            # If no future races, get the most recent
            return races.sort_values('date').iloc[-1]
        
        return upcoming.iloc[0]
    
    def get_previous_race(self):
        """Get the most recent completed race with results"""
        races = self.data.get('races', pd.DataFrame()).copy()
        results = self.data.get('results', pd.DataFrame())
        
        if races.empty or 'date' not in races.columns or results.empty:
            return None
            
        races['date'] = pd.to_datetime(races['date'])
        # Get races with results
        races_with_results = races[races['id'].isin(results['raceId'].unique())]
        
        # Get races before today
        past_races = races_with_results[races_with_results['date'] <= datetime.now()].sort_values('date', ascending=False)
        
        if past_races.empty:
            return None
        
        return past_races.iloc[0]
    
    def get_active_drivers(self, year=None):
        """Get list of active drivers for a given year or past 12 months"""
        # Start with empty set of driver IDs
        all_driver_ids = set()
        
        # Check driver_standings table first as it has the most complete list
        driver_standings = self.data.get('driver_standings', pd.DataFrame())
        if not driver_standings.empty:
            # Get standings with year info
            if 'year' not in driver_standings.columns:
                races = self.data.get('races', pd.DataFrame())
                if not races.empty and 'id' in races.columns and 'raceId' in driver_standings.columns:
                    driver_standings = driver_standings.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
            
            # Filter by year
            if 'year' in driver_standings.columns:
                if year is not None:
                    year_standings = driver_standings[driver_standings['year'] == year]
                else:
                    # Get drivers for current year
                    current_year = self.current_year
                    year_standings = driver_standings[driver_standings['year'] == current_year]
                
                if not year_standings.empty and 'driverId' in year_standings.columns:
                    all_driver_ids.update(year_standings['driverId'].unique())
        
        # First check seasons_drivers table for the most accurate list
        seasons_drivers = self.data.get('seasons_drivers', pd.DataFrame())
        if not seasons_drivers.empty and 'year' in seasons_drivers.columns:
            if year is not None:
                # Get drivers for specific year
                year_drivers = seasons_drivers[seasons_drivers['year'] == year]
                if not year_drivers.empty:
                    all_driver_ids.update(year_drivers['driverId'].unique())
            else:
                # Get drivers for current and previous year
                current_year = self.current_year
                recent_drivers = seasons_drivers[
                    (seasons_drivers['year'] == current_year) | 
                    (seasons_drivers['year'] == current_year - 1)
                ]
                if not recent_drivers.empty:
                    all_driver_ids.update(recent_drivers['driverId'].unique())
        
        # Also check results/qualifying for additional drivers (don't require empty all_driver_ids)
        # This ensures we get all drivers from all sources
        if True:  # Always check additional sources
            # Get drivers from race results
            results = self.data.get('results', pd.DataFrame())
            if not results.empty:
                # Get results with year info
                if 'year' not in results.columns:
                    races = self.data.get('races', pd.DataFrame())
                    if not races.empty and 'id' in races.columns and 'raceId' in results.columns:
                        results = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
                
                # Filter by year
                if 'year' in results.columns:
                    if year is not None:
                        year_results = results[results['year'] == year]
                    else:
                        # Get drivers who raced in the past year (current year and previous year)
                        current_year = self.current_year
                        year_results = results[
                            (results['year'] == current_year) | 
                            (results['year'] == current_year - 1)
                        ]
                    
                    if not year_results.empty:
                        all_driver_ids.update(year_results['driverId'].unique())
            
            # Also check qualifying data for additional drivers
            qualifying = self.data.get('qualifying', pd.DataFrame())
            if not qualifying.empty:
                # Get qualifying with year info
                if 'year' not in qualifying.columns:
                    races = self.data.get('races', pd.DataFrame())
                    if not races.empty and 'id' in races.columns and 'raceId' in qualifying.columns:
                        qualifying = qualifying.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
                
                # Filter by year
                if 'year' in qualifying.columns:
                    if year is not None:
                        year_qualifying = qualifying[qualifying['year'] == year]
                    else:
                        current_year = self.current_year
                        year_qualifying = qualifying[
                            (qualifying['year'] == current_year) | 
                            (qualifying['year'] == current_year - 1)
                        ]
                    
                    if not year_qualifying.empty and 'driverId' in year_qualifying.columns:
                        all_driver_ids.update(year_qualifying['driverId'].unique())
        
        # Get driver details
        drivers = self.data.get('drivers', pd.DataFrame())
        if drivers.empty or len(all_driver_ids) == 0:
            return pd.DataFrame()  # Return empty DataFrame
            
        active_drivers = drivers[drivers['id'].isin(all_driver_ids)]
        return active_drivers
    
    def calculate_overtake_points(self, race_data):
        """
        Calculate overtake points for a race with teammate considerations.
        Overtake Points = Grid Position - Finish Position
        +0.5 bonus for overtaking teammate
        -0.5 penalty for being overtaken by teammate
        """
        # Create a copy to avoid modifying original
        data = race_data.copy()
        
        # Basic overtake points (grid - finish)
        data['overtake_points'] = data['gridPosition'] - data['positionNumber']
        
        # Add constructor information for teammate detection
        if 'constructorId' in data.columns:
            # Group by race and constructor to find teammates
            for race_id in data['raceId'].unique():
                race_mask = data['raceId'] == race_id
                
                # Process each constructor's drivers
                for constructor_id in data[race_mask]['constructorId'].unique():
                    if pd.notna(constructor_id):
                        # Get drivers for this constructor in this race
                        constructor_mask = race_mask & (data['constructorId'] == constructor_id)
                        constructor_drivers = data[constructor_mask]
                        
                        if len(constructor_drivers) == 2:
                            # We have teammates
                            drivers = constructor_drivers.sort_values('positionNumber')
                            if len(drivers) == 2:
                                driver1_id = drivers.iloc[0]['driverId']
                                driver2_id = drivers.iloc[1]['driverId']
                                
                                driver1_grid = drivers.iloc[0]['gridPosition']
                                driver2_grid = drivers.iloc[1]['gridPosition']
                                
                                driver1_finish = drivers.iloc[0]['positionNumber']
                                driver2_finish = drivers.iloc[1]['positionNumber']
                                
                                # Check if positions changed between teammates
                                if driver1_grid > driver2_grid and driver1_finish < driver2_finish:
                                    # Driver 1 overtook teammate
                                    data.loc[(data['raceId'] == race_id) & (data['driverId'] == driver1_id), 'overtake_points'] += 0.5
                                    data.loc[(data['raceId'] == race_id) & (data['driverId'] == driver2_id), 'overtake_points'] -= 0.5
                                elif driver2_grid > driver1_grid and driver2_finish < driver1_finish:
                                    # Driver 2 overtook teammate
                                    data.loc[(data['raceId'] == race_id) & (data['driverId'] == driver2_id), 'overtake_points'] += 0.5
                                    data.loc[(data['raceId'] == race_id) & (data['driverId'] == driver1_id), 'overtake_points'] -= 0.5
        
        return data
    
    def filter_current_season_drivers(self, df):
        """Filter dataframe to only include drivers from current season"""
        if df.empty:
            return df
            
        # Get current season drivers
        current_drivers = self.get_active_drivers()
        
        # Check if we got a valid DataFrame
        if not isinstance(current_drivers, pd.DataFrame) or current_drivers.empty:
            print(f"Warning: No active drivers found for {self.current_year}, returning all drivers")
            return df
            
        # Filter by driver IDs (always use IDs for consistency)
        try:
            current_ids = current_drivers['id'].values
            # Check if index is already driver IDs
            if df.index.name == 'id' or df.index.name == 'driverId':
                return df[df.index.isin(current_ids)]
            # If we have an 'id' column, use it
            elif 'id' in df.columns:
                return df[df['id'].isin(current_ids)]
            # If we have a 'driverId' column, use it  
            elif 'driverId' in df.columns:
                return df[df['driverId'].isin(current_ids)]
            else:
                print(f"Warning: Cannot filter by driver ID, no id/driverId column found")
                return df
        except Exception as e:
            print(f"Warning: Error filtering drivers: {e}")
            return df
    
    def analyze_overtakes(self):
        """Analyze real overtakes using lap-by-lap data"""
        if self.lap_analyzer is None:
            # Fallback to position-based analysis if lap data not available
            return self._analyze_position_changes()
        
        try:
            # Get available years from lap data
            available_years = []
            for year_dir in self.lap_analyzer.data_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    year = int(year_dir.name)
                    if year >= self.current_year - 3:  # Recent years only
                        available_years.append(year)
            
            if not available_years:
                print("No recent lap data available, using position-based analysis")
                return self._analyze_position_changes()
            
            # Analyze overtakes for recent years
            overtake_data = self.lap_analyzer.analyze_multi_season_overtakes(available_years)
            
            if overtake_data.empty:
                print("No overtake data found, using position-based analysis")
                return self._analyze_position_changes()
            
            # Get driver summary
            driver_summary = self.lap_analyzer.get_driver_overtake_summary(overtake_data)
            
            # Map driver IDs to F1DB format
            driver_summary = self._map_lap_driver_ids_to_f1db(driver_summary)
            
            # Map overtake data IDs too
            overtake_data = overtake_data.copy()
            lap_to_f1db_map = {
                'max_verstappen': 'max-verstappen',
                'norris': 'lando-norris',
                'russell': 'george-russell', 
                'hamilton': 'lewis-hamilton',
                'leclerc': 'charles-leclerc',
                'piastri': 'oscar-piastri',
                'alonso': 'fernando-alonso',
                'gasly': 'pierre-gasly',
                'ocon': 'esteban-ocon',
                'stroll': 'lance-stroll',
                'albon': 'alexander-albon',
                'tsunoda': 'yuki-tsunoda',
                'hulkenberg': 'nico-hulkenberg',
                'bearman': 'oliver-bearman',
                'antonelli': 'andrea-kimi-antonelli',
                'bortoleto': 'gabriel-bortoleto',
                'lawson': 'liam-lawson',
                'hadjar': 'isack-hadjar',
                'doohan': 'jack-doohan',
                'colapinto': 'franco-colapinto',
                'perez': 'sergio-perez',
                'sainz': 'carlos-sainz-jr',
                'bottas': 'valtteri-bottas',
                'magnussen': 'kevin-magnussen',
                'zhou': 'guanyu-zhou',
                'ricciardo': 'daniel-ricciardo',
                'sargeant': 'logan-sargeant'
            }
            overtake_data['driverId'] = overtake_data['driverId'].map(lap_to_f1db_map)
            overtake_data = overtake_data[overtake_data['driverId'].notna()]
            
            # Convert to expected format
            return self._format_overtake_results(driver_summary, overtake_data)
            
        except Exception as e:
            print(f"Error in lap-by-lap analysis: {e}")
            return self._analyze_position_changes()
    
    def _analyze_position_changes(self):
        """Fallback method: Analyze positions gained by driver (grid vs finish)"""
        # Note: This calculates net positions gained (grid - finish), not actual on-track overtakes
        # Try both possible keys for results
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame())).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        
        if results.empty or grid.empty:
            return pd.DataFrame()
        
        # Merge results with starting grid
        overtake_data = results.merge(
            grid[['raceId', 'driverId', 'positionNumber', 'positionText']].rename(
                columns={'positionNumber': 'gridPosition', 'positionText': 'gridText'}
            ),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Handle pit lane starts - assign position 20 for calculation
        overtake_data['gridPosition'] = overtake_data.apply(
            lambda x: 20 if (pd.isna(x['gridPosition']) or x.get('gridText') == 'PL') else x['gridPosition'],
            axis=1
        )
        
        # Calculate position gained (negative means overtakes made)
        overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
        
        # Add year and circuit information
        races = self.data.get('races', pd.DataFrame())
        if not races.empty:
            # Always merge to ensure we have circuitId
            merge_cols = ['id']
            if 'year' not in overtake_data.columns:
                merge_cols.append('year')
            if 'circuitId' not in overtake_data.columns:
                merge_cols.append('circuitId')
            
            if len(merge_cols) > 1:  # Need to merge something besides just id
                overtake_data = overtake_data.merge(races[merge_cols], left_on='raceId', right_on='id', how='left')
        
        # Filter for recent years
        if 'year' in overtake_data.columns:
            recent_data = overtake_data[overtake_data['year'] >= self.current_year - 3]
        else:
            print("Warning: No year column found in overtake data")
            recent_data = overtake_data
        
        # Group by driver
        driver_overtakes = recent_data.groupby('driverId').agg({
            'positions_gained': ['sum', 'mean', 'median', 'count'],
            'points': ['sum', 'mean']
        }).round(2)
        
        driver_overtakes.columns = ['total_pos_gained_all', 'avg_pos_gained_all', 
                                   'median_pos_gained_all', 'races', 'total_points', 'avg_points']
        
        # Calculate positions gained (only positive position gains)
        recent_data['positions_gained_positive'] = recent_data['positions_gained'].apply(lambda x: max(0, x))
        positions_by_driver = recent_data.groupby('driverId')['positions_gained_positive'].agg(['sum', 'mean', 'median']).round(2)
        positions_by_driver.columns = ['total_pos_gained', 'avg_pos_gained', 'median_pos_gained']
        
        # Combine data - we'll use the positive gains only
        final_data = driver_overtakes[['races', 'total_points', 'avg_points']].join(positions_by_driver)
        
        # Keep driverId as index for proper identification
        
        # Filter to only current season drivers
        final_data = self.filter_current_season_drivers(final_data)
        
        # Ensure all active drivers are included (even with no data)
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            # Create empty rows for missing drivers
            all_driver_ids = set(active_drivers['id'].values)
            existing_ids = set(final_data.index)
            missing_ids = all_driver_ids - existing_ids
            
            if missing_ids:
                # Create empty DataFrame for missing drivers
                missing_data = pd.DataFrame(
                    index=list(missing_ids),
                    columns=final_data.columns
                )
                # Set default values
                for col in ['total_pos_gained', 'total_points', 'races']:
                    if col in missing_data.columns:
                        missing_data[col] = 0
                for col in ['avg_pos_gained', 'median_pos_gained', 'avg_points']:
                    if col in missing_data.columns:
                        missing_data[col] = 0.0
                
                # Combine with existing data
                final_data = pd.concat([final_data, missing_data])
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            final_data['driver_name'] = final_data.index.map(driver_map)
        
        # Add circuit-specific averages for previous, current, and next races
        races = self.data.get('races', pd.DataFrame())
        
        # Initialize with defaults
        final_data['p_circuit_avg'] = 0
        final_data['c_circuit_avg'] = 0
        final_data['n_circuit_avg'] = 0
        final_data['n_circuit_pts'] = 0
        final_data['p_circuit_pts'] = 0
        final_data['last_race_pts'] = 0
        
        if 'year' in recent_data.columns and 'date' in races.columns and 'circuitId' in recent_data.columns:
            # Ensure races have proper datetime
            races['date'] = pd.to_datetime(races['date'])
            
            # Get the most recent races with results
            results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame()))
            if not results.empty:
                races_with_results = results['raceId'].unique()
                
                # Get the most recent race with results (this is the "current" race)
                recent_races = races[races['id'].isin(races_with_results)].sort_values('date', ascending=False)
                
                if not recent_races.empty:
                    # Current race (most recent with results - should be 1138 Belgian GP)
                    current_race = recent_races.iloc[0]
                    current_circuit_id = current_race['circuitId']
                    
                    # Calculate circuit averages for current circuit
                    current_circuit_data = recent_data[recent_data['circuitId'] == current_circuit_id]
                    if not current_circuit_data.empty:
                        current_circuit_positions = current_circuit_data.groupby('driverId')['positions_gained_positive'].mean().round(2)
                        final_data['c_circuit_avg'] = final_data.index.map(
                            lambda x: current_circuit_positions.get(x, 0)
                        )
                    
                    # Previous race (race before current - should be 1137 British GP)
                    if len(recent_races) > 1:
                        prev_race = recent_races.iloc[1]
                        prev_circuit_id = prev_race['circuitId']
                        
                        # Calculate circuit averages for previous circuit
                        prev_circuit_data = recent_data[recent_data['circuitId'] == prev_circuit_id]
                        if not prev_circuit_data.empty:
                            prev_circuit_positions = prev_circuit_data.groupby('driverId')['positions_gained_positive'].mean().round(2)
                            final_data['p_circuit_avg'] = final_data.index.map(
                                lambda x: prev_circuit_positions.get(x, 0)
                            )
                            
                            # Calculate overtake points for previous circuit
                            if 'constructorId' not in prev_circuit_data.columns:
                                if 'constructorId' in results.columns:
                                    prev_circuit_data = prev_circuit_data.merge(
                                        results[['raceId', 'driverId', 'constructorId']].drop_duplicates(),
                                        on=['raceId', 'driverId'],
                                        how='left'
                                    )
                            
                            prev_circuit_with_ot_pts = self.calculate_overtake_points(prev_circuit_data)
                            prev_circuit_ot_points = prev_circuit_with_ot_pts.groupby('driverId')['overtake_points'].mean().round(2)
                            final_data['p_circuit_pts'] = final_data.index.map(
                                lambda x: prev_circuit_ot_points.get(x, 0)
                            )
        
        # Add circuit-specific prediction for next race
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race and 'circuitId' in recent_data.columns:
            # Next race circuit (should be 1139 Hungarian GP)
            next_circuit_id = next_race['circuitId']
            circuit_data = recent_data[recent_data['circuitId'] == next_circuit_id]
            
            if not circuit_data.empty:
                # Calculate average positions gained at this specific circuit
                circuit_positions = circuit_data.groupby('driverId')['positions_gained_positive'].mean().round(2)
                final_data['n_circuit_avg'] = final_data.index.map(
                    lambda x: circuit_positions.get(x, 0)
                )
                
                # Calculate overtake points for this circuit
                # First ensure we have constructor data
                if 'constructorId' not in circuit_data.columns:
                    # Try to merge constructor data if available
                    if 'constructorId' in results.columns:
                        circuit_data = circuit_data.merge(
                            results[['raceId', 'driverId', 'constructorId']].drop_duplicates(),
                            on=['raceId', 'driverId'],
                            how='left'
                        )
                
                # Calculate overtake points with teammate considerations
                circuit_data_with_ot_pts = self.calculate_overtake_points(circuit_data)
                
                # Average overtake points per driver at this circuit
                circuit_ot_points = circuit_data_with_ot_pts.groupby('driverId')['overtake_points'].mean().round(2)
                final_data['n_circuit_pts'] = final_data.index.map(
                    lambda x: circuit_ot_points.get(x, 0)
                )
        
        # Add last race overtakes
        final_data['last_race'] = 0
        if not results.empty and not recent_data.empty:
            # Get the most recent race with results
            races_with_results = results['raceId'].unique()
            recent_races = races[races['id'].isin(races_with_results)].sort_values('date', ascending=False)
            
            if not recent_races.empty:
                last_race_id = recent_races.iloc[0]['id']
                
                # Get overtakes from last race using recent_data which has the overtakes column
                last_race_data = recent_data[recent_data['raceId'] == last_race_id]
                if not last_race_data.empty and 'positions_gained_positive' in last_race_data.columns:
                    last_race_positions = last_race_data.set_index('driverId')['positions_gained_positive']
                    final_data['last_race'] = final_data.index.map(
                        lambda x: int(last_race_positions.get(x, 0))
                    )
                    
                    # Get championship points from last race
                    last_race_points = last_race_data.set_index('driverId')['points']
                    final_data['last_race_pts'] = final_data.index.map(
                        lambda x: int(last_race_points.get(x, 0)) if pd.notna(last_race_points.get(x)) else 0
                    )
        
        # Reorder columns for better presentation
        column_order = ['driver_name', 'avg_pos_gained', 'avg_points', 
                       'median_pos_gained', 'last_race', 'last_race_pts', 'p_circuit_avg', 'p_circuit_pts',
                       'c_circuit_avg', 'n_circuit_avg', 'n_circuit_pts', 'races']
        
        # Add any remaining columns not in the order list (except excluded columns)
        exclude_cols = ['total_pos_gained', 'total_points', 'total_pos_gained_positive']
        remaining_cols = [col for col in final_data.columns if col not in column_order and col not in exclude_cols]
        column_order.extend(remaining_cols)
        
        # Reorder columns, keeping only those that exist
        final_columns = [col for col in column_order if col in final_data.columns]
        final_data = final_data[final_columns]
        
        # Replace NaN values with 0 in numeric columns
        numeric_cols = ['p_circuit_avg', 'c_circuit_avg', 'n_circuit_avg', 'n_circuit_pts', 
                       'p_circuit_pts', 'last_race_pts']
        for col in numeric_cols:
            if col in final_data.columns:
                final_data[col] = final_data[col].fillna(0)
        
        # Drop driver_id index since we already have driver_name
        final_data = final_data.reset_index(drop=True)
        
        # Filter out drivers with all zeros (no racing data)
        # Check key columns for non-zero values
        data_columns = ['avg_OT', 'avg_points', 'races', 'total_pos_gained', 'avg_pos_gained']
        data_columns = [col for col in data_columns if col in final_data.columns]
        
        if data_columns:
            # Keep rows where at least one data column is non-zero
            has_data = final_data[data_columns].sum(axis=1) > 0
            final_data = final_data[has_data]
        
        return final_data
    
    def _format_overtake_results(self, driver_summary: pd.DataFrame, overtake_data: pd.DataFrame) -> pd.DataFrame:
        """Format lap-by-lap overtake results to match expected structure"""
        if driver_summary.empty:
            return pd.DataFrame()
        
        # Create base DataFrame with required columns
        formatted_data = pd.DataFrame()
        
        # Add driver IDs as index (driver_summary already has driverId as index after mapping)
        formatted_data.index = driver_summary.index
        
        # Add driver names FIRST
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            formatted_data['driver_name'] = formatted_data.index.map(driver_map)
        else:
            formatted_data['driver_name'] = formatted_data.index
        
        # Map from lap analyzer results to expected format - using OTs instead of overtakes
        formatted_data['total_OTs'] = driver_summary['total_overtakes_made']
        formatted_data['avg_OTs'] = driver_summary['avg_overtakes_per_race']
        formatted_data['median_OTs'] = (driver_summary['total_overtakes_made'] / driver_summary['races_participated']).round(2)
        formatted_data['OTd_by'] = driver_summary['total_overtaken_by']
        formatted_data['net_OTs'] = driver_summary['net_overtakes']
        formatted_data['races'] = driver_summary['races_participated']
        formatted_data['max_OTs'] = driver_summary['max_overtakes_single_race']
        
        # Get recent race performance
        if not overtake_data.empty:
            # Get most recent race data for each driver
            recent_race_data = overtake_data.loc[overtake_data.groupby('driverId')['round'].idxmax()]
            recent_race_overtakes = dict(zip(recent_race_data['driverId'], recent_race_data['overtakes_made']))
            formatted_data['last_race_OTs'] = formatted_data.index.map(lambda x: recent_race_overtakes.get(x, 0))
        else:
            formatted_data['last_race_OTs'] = 0
        
        # Get points data from F1DB for these drivers
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame()))
        if not results.empty and 'year' in results.columns:
            recent_results = results[results['year'] >= self.current_year - 3]
            points_by_driver = recent_results.groupby('driverId').agg({
                'points': ['sum', 'mean']
            }).round(2)
            points_by_driver.columns = ['total_points', 'avg_points']
            
            # Merge with overtake data
            formatted_data = formatted_data.join(points_by_driver, how='left')
        
        # Fill missing points data
        formatted_data['total_points'] = formatted_data.get('total_points', 0).fillna(0)
        formatted_data['avg_points'] = formatted_data.get('avg_points', 0.0).fillna(0.0)
        
        # Add circuit-specific overtake averages
        formatted_data['c_race_OTs'] = 0
        formatted_data['n_race_OTs'] = 0
        
        # Get race information
        races = self.data.get('races', pd.DataFrame())
        if not races.empty and not overtake_data.empty:
            # Ensure races have proper datetime
            races['date'] = pd.to_datetime(races['date'])
            
            # Get the most recent races with results
            results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame()))
            if not results.empty:
                races_with_results = results['raceId'].unique()
                
                # Get the most recent race with results (current race)
                recent_races = races[races['id'].isin(races_with_results)].sort_values('date', ascending=False)
                
                if not recent_races.empty:
                    # Current race circuit
                    current_race = recent_races.iloc[0]
                    current_circuit_id = current_race['circuitId']
                    
                    # Calculate average overtakes at current circuit
                    current_circuit_data = overtake_data[overtake_data['circuitId'] == current_circuit_id]
                    if not current_circuit_data.empty:
                        circuit_avg_overtakes = current_circuit_data.groupby('driverId')['overtakes_made'].mean().round(1)
                        formatted_data['c_race_OTs'] = formatted_data.index.map(
                            lambda x: circuit_avg_overtakes.get(x, 0)
                        )
            
            # Get next race circuit
            next_race = self.get_next_race()
            if next_race is not None and 'circuitId' in next_race:
                next_circuit_id = next_race['circuitId']
                
                # Calculate average overtakes at next circuit
                next_circuit_data = overtake_data[overtake_data['circuitId'] == next_circuit_id]
                if not next_circuit_data.empty:
                    next_circuit_avg = next_circuit_data.groupby('driverId')['overtakes_made'].mean().round(1)
                    formatted_data['n_race_OTs'] = formatted_data.index.map(
                        lambda x: next_circuit_avg.get(x, 0)
                    )
        
        # Get current season driver IDs
        current_drivers = self.get_active_drivers()
        if not current_drivers.empty:
            current_ids = current_drivers['id'].values
            # Filter formatted_data to only include current drivers
            formatted_data = formatted_data[formatted_data.index.isin(current_ids)]
        
        # Reset index and clean up
        formatted_data = formatted_data.reset_index(drop=True)
        
        # Define column order with driver_name first
        column_order = [
            'driver_name', 'total_OTs', 'avg_OTs', 'median_OTs', 
            'last_race_OTs', 'c_race_OTs', 'n_race_OTs', 
            'OTd_by', 'net_OTs', 'max_OTs',
            'races', 'total_points', 'avg_points'
        ]
        
        # Select only columns that exist in the order we want
        existing_columns = [col for col in column_order if col in formatted_data.columns]
        formatted_data = formatted_data[existing_columns]
        
        return formatted_data
    
    def _map_lap_driver_ids_to_f1db(self, driver_summary: pd.DataFrame) -> pd.DataFrame:
        """Map lap data driver IDs to F1DB driver IDs"""
        if driver_summary.empty:
            return driver_summary
        
        # Create mapping from lap data IDs to F1DB IDs
        driver_id_mapping = {
            'max_verstappen': 'max-verstappen',
            'norris': 'lando-norris',
            'russell': 'george-russell', 
            'hamilton': 'lewis-hamilton',
            'leclerc': 'charles-leclerc',
            'piastri': 'oscar-piastri',
            'alonso': 'fernando-alonso',
            'gasly': 'pierre-gasly',
            'ocon': 'esteban-ocon',
            'stroll': 'lance-stroll',
            'albon': 'alexander-albon',
            'tsunoda': 'yuki-tsunoda',
            'hulkenberg': 'nico-hulkenberg',
            'bearman': 'oliver-bearman',
            'antonelli': 'andrea-kimi-antonelli',
            'bortoleto': 'gabriel-bortoleto',
            'lawson': 'liam-lawson',
            'hadjar': 'isack-hadjar',
            'doohan': 'jack-doohan',
            'colapinto': 'franco-colapinto',
            'perez': 'sergio-perez',
            'sainz': 'carlos-sainz-jr',
            'bottas': 'valtteri-bottas',
            'magnussen': 'kevin-magnussen',
            'zhou': 'guanyu-zhou',
            'ricciardo': 'daniel-ricciardo',
            'sargeant': 'logan-sargeant'
        }
        
        # Map the driver IDs
        driver_summary['f1db_driver_id'] = driver_summary['driverId'].map(driver_id_mapping)
        
        # Keep only drivers that mapped successfully
        mapped_drivers = driver_summary[driver_summary['f1db_driver_id'].notna()].copy()
        mapped_drivers['driverId'] = mapped_drivers['f1db_driver_id']
        mapped_drivers = mapped_drivers.drop('f1db_driver_id', axis=1)
        
        # Set driverId as index to match expected format
        mapped_drivers = mapped_drivers.set_index('driverId')
        
        return mapped_drivers
    
    def analyze_fantasy_overtakes(self):
        """Analyze F1 Fantasy overtake data"""
        if not self.fantasy_data or 'driver_details' not in self.fantasy_data:
            return pd.DataFrame()
        
        fantasy_details = self.fantasy_data['driver_details'].copy()
        
        # Focus on overtake-related columns
        overtake_cols = [col for col in fantasy_details.columns if 'overtake' in col.lower()]
        
        if not overtake_cols:
            return pd.DataFrame()
        
        # Get next race info
        next_race = self.get_next_race()
        next_circuit_id = ''
        if next_race is not None:
            # next_race is a pandas Series when found
            if hasattr(next_race, 'get'):
                next_circuit_id = next_race.get('circuitId', '')
            elif hasattr(next_race, '__getitem__') and 'circuitId' in next_race:
                next_circuit_id = next_race['circuitId']
        
        # Group by driver to get overtake statistics
        fantasy_overtakes = []
        
        for driver_id in fantasy_details['f1db_driver_id'].unique():
            if pd.isna(driver_id):
                continue
                
            driver_data = fantasy_details[fantasy_details['f1db_driver_id'] == driver_id]
            driver_name = driver_data.iloc[0]['player_name']
            
            # Calculate overtake statistics
            overtake_stats = {
                'driverId': driver_id,
                'driver_name': driver_name,
                'total_races': len(driver_data),
            }
            
            # Process each overtake column
            for col in overtake_cols:
                if col.endswith('_points'):
                    col_base = col.replace('_points', '')
                    freq_col = f"{col_base}_freq"
                    
                    # Get points data
                    points_data = driver_data[col].fillna(0)
                    overtake_stats[f"{col_base}_total_pts"] = points_data.sum()
                    overtake_stats[f"{col_base}_avg_pts"] = points_data.mean()
                    overtake_stats[f"{col_base}_races_with"] = (points_data > 0).sum()
                    
                    # Get frequency data if available
                    if freq_col in driver_data.columns:
                        freq_data = pd.to_numeric(driver_data[freq_col], errors='coerce').fillna(0)
                        overtake_stats[f"{col_base}_total_count"] = freq_data.sum()
                        overtake_stats[f"{col_base}_avg_count"] = freq_data.mean()
            
            # Circuit-specific data if available
            if next_circuit_id and 'circuit_id' in driver_data.columns:
                circuit_data = driver_data[driver_data['circuit_id'] == next_circuit_id]
                if not circuit_data.empty:
                    for col in overtake_cols:
                        if col.endswith('_points'):
                            col_base = col.replace('_points', '')
                            circuit_pts = circuit_data[col].fillna(0).mean()
                            overtake_stats[f"{col_base}_circuit_avg_pts"] = circuit_pts
            
            fantasy_overtakes.append(overtake_stats)
        
        if not fantasy_overtakes:
            return pd.DataFrame()
        
        # Convert to DataFrame
        fantasy_df = pd.DataFrame(fantasy_overtakes).set_index('driverId')
        
        # Sort by total overtake points
        overtake_points_cols = [col for col in fantasy_df.columns if col.endswith('_total_pts')]
        if overtake_points_cols:
            fantasy_df['total_overtake_points'] = fantasy_df[overtake_points_cols].sum(axis=1)
            fantasy_df = fantasy_df.sort_values('total_overtake_points', ascending=False)
        
        # Filter to current season drivers
        fantasy_df = self.filter_current_season_drivers(fantasy_df)
        
        # Rename sprint_overtake_bonus columns to sprint_OTs
        rename_map = {}
        for col in fantasy_df.columns:
            if 'sprint_overtake_bonus' in col:
                new_col = col.replace('sprint_overtake_bonus', 'sprint_OTs')
                rename_map[col] = new_col
        
        if rename_map:
            fantasy_df = fantasy_df.rename(columns=rename_map)
        
        return fantasy_df
    
    def analyze_overtakes_by_track_year(self):
        """Analyze overtakes by driver for each track, broken down by year"""
        # Try both possible keys for results
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame())).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if results.empty or grid.empty or races.empty:
            return {}
        
        # First, ensure results has year and circuitId columns
        if ('year' not in results.columns or 'circuitId' not in results.columns) and not races.empty:
            merge_cols = ['id']
            if 'year' not in results.columns:
                merge_cols.append('year')
            if 'circuitId' not in results.columns:
                merge_cols.append('circuitId')
            results = results.merge(races[merge_cols], 
                                  left_on='raceId', right_on='id', how='left')
        
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None or 'circuitId' not in next_race:
            return {}
        
        next_circuit_id = next_race['circuitId']
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if not circuits.empty and 'id' in circuits.columns:
            circuit_info = circuits[circuits['id'] == next_circuit_id]
            if not circuit_info.empty:
                circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
        # Merge all necessary data - results already has year and circuitId from earlier merge
        overtake_data = results.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Filter for the next race's circuit only
        circuit_data = overtake_data[overtake_data['circuitId'] == next_circuit_id].copy()
        
        if circuit_data.empty:
            return {}
        
        # Ensure year column exists
        if 'year' not in circuit_data.columns:
            print("Warning: No year column found in circuit data")
            return {}
        
        # Calculate position changes
        circuit_data['positions_gained'] = circuit_data['gridPosition'] - circuit_data['positionNumber']
        circuit_data['overtakes'] = circuit_data['positions_gained'].apply(lambda x: max(0, x))
        
        # Group by driver and year
        track_year_analysis = circuit_data.groupby(['driverId', 'year']).agg({
            'overtakes': ['sum', 'mean', 'median'],
            'gridPosition': 'mean',
            'positionNumber': 'mean',
            'positions_gained': 'mean'
        }).round(2)
        
        # Flatten column names
        track_year_analysis.columns = [
            'total_OT', 'avg_OT', 'median_OT',
            'avg_start_pos', 'avg_finish_pos', 'avg_pos_change'
        ]
        
        # Reset index to make driverId and year regular columns
        track_year_analysis = track_year_analysis.reset_index()
        
        # Add overall statistics per driver across all years at this track
        overall_stats = circuit_data.groupby('driverId').agg({
            'overtakes': ['sum', 'mean', 'median'],
            'gridPosition': 'mean',
            'positionNumber': 'mean',
            'positions_gained': 'mean',
            'year': 'count'
        }).round(2)
        
        overall_stats.columns = [
            'career_OT', 'career_avg_OT', 'career_median_OT',
            'career_avg_start', 'career_avg_finish', 'career_avg_pos_change',
            'races_at_track'
        ]
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            track_year_analysis['driver_name'] = track_year_analysis['driverId'].map(driver_map)
            overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
        # Filter to only show current season drivers
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            current_driver_ids = active_drivers['id'].tolist()
            track_year_analysis = track_year_analysis[
                track_year_analysis['driverId'].isin(current_driver_ids)
            ]
            overall_stats = overall_stats[overall_stats.index.isin(current_driver_ids)]
        
        return {
            'circuit_name': circuit_name,
            'circuit_id': next_circuit_id,
            'year_by_year': track_year_analysis,
            'overall_stats': overall_stats
        }
    
    def analyze_points(self):
        """Analyze F1 points by driver"""
        # Try both possible keys for results
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame())).copy()
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty:
            return {}
        
        # Add year and circuit information
        if not races.empty:
            merge_cols = ['id']
            if 'year' not in results.columns:
                merge_cols.append('year')
            if 'circuitId' not in results.columns:
                merge_cols.append('circuitId')
            
            if len(merge_cols) > 1:
                results = results.merge(races[merge_cols], left_on='raceId', right_on='id', how='left')
        
        # Current season data
        current_season = results[results['year'] == self.current_year]
        
        # Historical data (last 3 years)
        historical = results[(results['year'] >= self.current_year - 3) & (results['year'] < self.current_year)]
        
        # Current season analysis
        current_stats = current_season.groupby('driverId').agg({
            'points': ['sum', 'mean', 'median', 'count']
        }).round(2)
        current_stats.columns = ['total_points', 'avg_points', 'median_points', 'races']
        
        # Historical analysis
        if not historical.empty:
            hist_stats = historical.groupby('driverId').agg({
                'points': ['mean', 'median']
            }).round(2)
            hist_stats.columns = ['hist_avg_points', 'hist_median_points']
            
            # Combine
            points_analysis = current_stats.join(hist_stats, how='left').fillna(0)
        else:
            points_analysis = current_stats
        
        # Keep driverId as index for proper identification
        
        # Filter to only current season drivers
        points_analysis = self.filter_current_season_drivers(points_analysis)
        
        # Ensure all active drivers are included (even with no data)
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            # Create empty rows for missing drivers
            all_driver_ids = set(active_drivers['id'].values)
            existing_ids = set(points_analysis.index)
            missing_ids = all_driver_ids - existing_ids
            
            if missing_ids:
                # Create empty DataFrame for missing drivers
                missing_data = pd.DataFrame(
                    index=list(missing_ids),
                    columns=points_analysis.columns
                )
                # Set default values
                for col in ['total_points', 'races']:
                    if col in missing_data.columns:
                        missing_data[col] = 0
                for col in ['avg_points', 'median_points', 'hist_avg_points', 'hist_median_points']:
                    if col in missing_data.columns:
                        missing_data[col] = 0.0
                
                # Combine with existing data
                points_analysis = pd.concat([points_analysis, missing_data])
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            points_analysis['driver_name'] = points_analysis.index.map(driver_map)
        
        # Add circuit-specific averages for previous, current, and next races
        # Initialize with defaults
        points_analysis['p_circuit_avg'] = 0
        points_analysis['c_circuit_avg'] = 0  
        points_analysis['n_circuit_avg'] = 0
        points_analysis['last_race'] = 0
        
        # Get all results data with circuitId for filtering (last 3 years)
        all_results = results[(results['year'] >= self.current_year - 3)]
        
        if 'circuitId' in all_results.columns and not races.empty:
            # Ensure races have proper datetime
            races['date'] = pd.to_datetime(races['date'])
            
            # Get the most recent races with results
            races_with_results = results['raceId'].unique()
            recent_races_sorted = races[races['id'].isin(races_with_results)].sort_values('date', ascending=False)
            
            if not recent_races_sorted.empty:
                # Current race (most recent with results - should be Belgian GP)
                current_race = recent_races_sorted.iloc[0]
                current_circuit_id = current_race['circuitId']
                last_race_id = current_race['id']
                
                # Calculate last race points
                last_race_data = results[results['raceId'] == last_race_id]
                if not last_race_data.empty:
                    last_race_points = last_race_data.set_index('driverId')['points']
                    points_analysis['last_race'] = points_analysis.index.map(
                        lambda x: int(last_race_points.get(x, 0)) if pd.notna(last_race_points.get(x)) else 0
                    )
                
                # Calculate circuit average for current circuit (historical average at this track)
                current_circuit_data = all_results[all_results['circuitId'] == current_circuit_id]
                if not current_circuit_data.empty:
                    current_circuit_points = current_circuit_data.groupby('driverId')['points'].mean().round(2)
                    points_analysis['c_circuit_avg'] = points_analysis.index.map(
                        lambda x: current_circuit_points.get(x, 0)
                    )
                
                # Previous race (race before current - should be British GP)
                if len(recent_races_sorted) > 1:
                    prev_race = recent_races_sorted.iloc[1]
                    prev_circuit_id = prev_race['circuitId']
                    
                    # Calculate circuit average for previous circuit
                    prev_circuit_data = all_results[all_results['circuitId'] == prev_circuit_id]
                    if not prev_circuit_data.empty:
                        prev_circuit_points = prev_circuit_data.groupby('driverId')['points'].mean().round(2)
                        points_analysis['p_circuit_avg'] = points_analysis.index.map(
                            lambda x: prev_circuit_points.get(x, 0)
                        )
        
        # Next race circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race and 'circuitId' in all_results.columns:
            # Next race circuit (should be Hungarian GP)
            next_circuit_id = next_race['circuitId']
            next_circuit_data = all_results[all_results['circuitId'] == next_circuit_id]
            
            if not next_circuit_data.empty:
                # Calculate average points at this specific circuit
                next_circuit_points = next_circuit_data.groupby('driverId')['points'].mean().round(2)
                points_analysis['n_circuit_avg'] = points_analysis.index.map(
                    lambda x: next_circuit_points.get(x, 0)
                )
        
        # Reorder columns for better presentation
        column_order = ['driver_name', 'total_points', 'avg_points', 'median_points', 
                       'last_race', 'races', 'hist_avg_points', 'hist_median_points',
                       'p_circuit_avg', 'c_circuit_avg', 'n_circuit_avg']
        
        # Keep only columns that exist
        final_columns = [col for col in column_order if col in points_analysis.columns]
        points_analysis = points_analysis[final_columns]
        
        # Replace NaN values with 0 in circuit average columns
        circuit_cols = ['p_circuit_avg', 'c_circuit_avg', 'n_circuit_avg']
        for col in circuit_cols:
            if col in points_analysis.columns:
                points_analysis[col] = points_analysis[col].fillna(0)
        
        # Drop driver_id index since we already have driver_name
        points_analysis = points_analysis.reset_index(drop=True)
        
        # Filter out drivers with no points in current season
        # A driver should have either scored points or participated in races
        if 'total_points' in points_analysis.columns and 'races' in points_analysis.columns:
            # Keep drivers who have either scored points OR participated in races in current season
            has_activity = (points_analysis['total_points'] > 0) | (points_analysis['races'] > 0)
            points_analysis = points_analysis[has_activity]
        
        return points_analysis
    
    def analyze_points_by_track_year(self):
        """Analyze points by driver for each track, broken down by year"""
        # Try both possible keys for results
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame())).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if results.empty or races.empty:
            return {}
        
        # Ensure results has year and circuitId columns
        if ('year' not in results.columns or 'circuitId' not in results.columns) and not races.empty:
            merge_cols = ['id']
            if 'year' not in results.columns:
                merge_cols.append('year')
            if 'circuitId' not in results.columns:
                merge_cols.append('circuitId')
            results = results.merge(races[merge_cols], 
                                  left_on='raceId', right_on='id', how='left')
        
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None or 'circuitId' not in next_race:
            return {}
        
        next_circuit_id = next_race['circuitId']
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if not circuits.empty and 'id' in circuits.columns:
            circuit_info = circuits[circuits['id'] == next_circuit_id]
            if not circuit_info.empty:
                circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
        # Filter for the next race's circuit only
        circuit_data = results[results['circuitId'] == next_circuit_id].copy()
        
        if circuit_data.empty:
            return {}
        
        # Ensure year column exists
        if 'year' not in circuit_data.columns:
            print("Warning: No year column found in circuit data")
            return {}
        
        # Group by driver and year
        track_year_points = circuit_data.groupby(['driverId', 'year']).agg({
            'points': ['sum', 'mean', 'median', 'count'],
            'positionNumber': ['mean', 'min']
        }).round(2)
        
        # Flatten column names
        track_year_points.columns = [
            'total_points', 'avg_points', 'median_points', 'races',
            'avg_finish', 'best_finish'
        ]
        
        # Reset index to make driverId and year regular columns
        track_year_points = track_year_points.reset_index()
        
        # Add overall statistics per driver across all years at this track
        overall_stats = circuit_data.groupby('driverId').agg({
            'points': ['sum', 'mean', 'median'],
            'positionNumber': ['mean', 'min'],
            'year': 'count'
        }).round(2)
        
        overall_stats.columns = [
            'career_points', 'career_avg_points', 'career_median_points',
            'career_avg_finish', 'career_best_finish',
            'races_at_track'
        ]
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            track_year_points['driver_name'] = track_year_points['driverId'].map(driver_map)
            overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
        # Filter to only show current season drivers
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            current_driver_ids = active_drivers['id'].tolist()
            track_year_points = track_year_points[
                track_year_points['driverId'].isin(current_driver_ids)
            ]
            overall_stats = overall_stats[overall_stats.index.isin(current_driver_ids)]
        
        return {
            'circuit_name': circuit_name,
            'circuit_id': next_circuit_id,
            'year_by_year': track_year_points,
            'overall_stats': overall_stats
        }
    
    def analyze_pit_stops(self):
        """Analyze pit stop times by driver"""
        # Try multiple possible keys for pit stops data
        pit_stops = pd.DataFrame()
        for key in ['races_pit_stops', 'pit_stops', 'pitstops', 'pit_stop']:
            if key in self.data and not self.data[key].empty:
                pit_stops = self.data[key].copy()
                print(f"Found pit stops data under key: {key}")
                break
        
        if pit_stops.empty:
            print("No pit stops data found in any expected keys")
            print(f"Available keys in data: {list(self.data.keys())}")
            return pd.DataFrame()
            
        races = self.data.get('races', pd.DataFrame())
        
        if pit_stops.empty:
            return pd.DataFrame()
        
        # Convert time to seconds
        if 'timeMillis' in pit_stops.columns:
            # Use timeMillis and convert to seconds
            pit_stops['time_seconds'] = pd.to_numeric(pit_stops['timeMillis'], errors='coerce') / 1000
        elif 'time' in pit_stops.columns:
            # Parse time string (format: MM:SS.mmm or SS.mmm)
            def parse_time(t):
                if pd.isna(t):
                    return np.nan
                t = str(t)
                if ':' in t:
                    parts = t.split(':')
                    return float(parts[0]) * 60 + float(parts[1])
                else:
                    return float(t)
            pit_stops['time_seconds'] = pit_stops['time'].apply(parse_time)
        else:
            return pd.DataFrame()
        
        # Add year information
        if not races.empty:
            pit_stops = pit_stops.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
        # Filter recent years
        recent_stops = pit_stops[pit_stops['year'] >= self.current_year - 3] if 'year' in pit_stops.columns else pit_stops
        
        # Analyze by driver
        pit_analysis = recent_stops.groupby('driverId').agg({
            'time_seconds': ['mean', 'median', 'min', 'std', 'count']
        }).round(3)
        pit_analysis.columns = ['avg_stop_time', 'median_stop_time', 'best_stop_time', 'std_dev', 'total_stops']
        
        # Keep driverId as index for proper identification
        
        # Filter to only current season drivers
        pit_analysis = self.filter_current_season_drivers(pit_analysis)
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            pit_analysis['driver_name'] = pit_analysis.index.map(driver_map)
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_stops = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_stops.empty:
                circuit_times = circuit_stops.groupby('driverId')['time_seconds'].mean().round(3)
                pit_analysis['next_circuit_avg'] = pit_analysis.index.map(
                    lambda x: circuit_times.get(x, pit_analysis.loc[x, 'avg_stop_time'] if x in pit_analysis.index else 0)
                )
        
        # Add DHL comparison if available
        if hasattr(self, 'dhl_data') and not self.dhl_data.empty:
            # Get DHL data with circuit mapping
            dhl_data = self.dhl_data.copy()
            
            # Map race_id to circuitId if needed
            if 'race_id' in dhl_data.columns and not races.empty and 'circuitId' not in dhl_data.columns:
                race_to_circuit = races.set_index('id')['circuitId'].to_dict()
                dhl_data['circuitId'] = dhl_data['race_id'].map(race_to_circuit)
            
            # Merge F1DB pit stops with races to get circuitId
            if 'circuitId' not in recent_stops.columns and not races.empty:
                recent_stops = recent_stops.merge(
                    races[['id', 'circuitId']], 
                    left_on='raceId', 
                    right_on='id', 
                    how='left',
                    suffixes=('', '_race')
                )
            
            # Calculate pit lane delta (total time - service time)
            pit_lane_deltas = []
            
            for driver_id in pit_analysis.index:
                # Get F1DB pit stops for this driver
                driver_f1db = recent_stops[recent_stops['driverId'] == driver_id]
                
                # Get DHL pit stops for this driver (using driver_id column)
                driver_dhl = dhl_data[dhl_data['driver_id'] == driver_id] if 'driver_id' in dhl_data.columns else pd.DataFrame()
                
                if not driver_f1db.empty and not driver_dhl.empty:
                    # Match by race and lap when possible
                    for _, f1db_stop in driver_f1db.iterrows():
                        race_id = f1db_stop.get('raceId')
                        lap = f1db_stop.get('lap')
                        
                        # Find matching DHL stop
                        dhl_match = driver_dhl[
                            (driver_dhl['race_id'] == race_id) & 
                            (driver_dhl['lap'] == lap)
                        ] if 'race_id' in driver_dhl.columns and 'lap' in driver_dhl.columns else pd.DataFrame()
                        
                        if not dhl_match.empty:
                            # Convert F1DB time from string to float
                            f1db_time_str = f1db_stop.get('time', f1db_stop.get('time_seconds'))
                            
                            # Parse time - could be seconds or MM:SS.mmm format
                            if pd.notna(f1db_time_str):
                                if ':' in str(f1db_time_str):
                                    # Format is MM:SS.mmm
                                    parts = str(f1db_time_str).split(':')
                                    minutes = float(parts[0])
                                    seconds = float(parts[1])
                                    f1db_time = minutes * 60 + seconds
                                else:
                                    # Already in seconds
                                    f1db_time = float(f1db_time_str)
                            else:
                                f1db_time = None
                                
                            dhl_time = float(dhl_match.iloc[0]['time']) if 'time' in dhl_match.columns else None
                            
                            if pd.notna(f1db_time) and pd.notna(dhl_time):
                                delta = f1db_time - dhl_time
                                circuit_id = f1db_stop.get('circuitId', 'unknown')
                                
                                pit_lane_deltas.append({
                                    'driverId': driver_id,
                                    'delta': delta,
                                    'circuitId': circuit_id,
                                    'f1db_time': f1db_time,
                                    'dhl_time': dhl_time
                                })
            
            # Aggregate pit lane deltas
            if pit_lane_deltas:
                delta_df = pd.DataFrame(pit_lane_deltas)
                
                # Calculate average delta by driver
                driver_deltas = delta_df.groupby('driverId')['delta'].agg(['mean', 'median', 'std', 'count']).round(3)
                driver_deltas.columns = ['avg_pit_lane_time', 'median_pit_lane_time', 'std_pit_lane_time', 'delta_samples']
                
                # Join to main analysis
                pit_analysis = pit_analysis.join(driver_deltas, how='left')
                
                # Calculate by circuit if next race circuit is known
                if next_race is not None and 'circuitId' in next_race:
                    next_circuit = next_race['circuitId']
                    circuit_deltas = delta_df[delta_df['circuitId'] == next_circuit]
                    
                    if not circuit_deltas.empty:
                        circuit_avg = circuit_deltas.groupby('driverId')['delta'].mean().round(3)
                        pit_analysis['next_circuit_pit_lane_time'] = pit_analysis.index.map(circuit_avg)
        
        # Reset index and ensure driver_name is present
        pit_analysis = pit_analysis.reset_index()
        pit_analysis = pit_analysis.rename(columns={'index': 'driverId'})
        
        # Re-add driver names if they were lost
        if 'driver_name' not in pit_analysis.columns and not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            pit_analysis['driver_name'] = pit_analysis['driverId'].map(driver_map)
        
        # Drop driverId column
        if 'driverId' in pit_analysis.columns:
            pit_analysis = pit_analysis.drop('driverId', axis=1)
        
        return pit_analysis
    
    def analyze_dhl_pit_stops(self):
        """Analyze DHL official pit stop data with avg_time, median_time, best_time, best_time_lap"""
        dhl_data = self.dhl_data.copy()
        
        if dhl_data.empty:
            return pd.DataFrame()
        
        # Map race names to circuit IDs for better integration
        races = self.data.get('races', pd.DataFrame())
        drivers = self.data.get('drivers', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        # Clean driver names and match to database
        if not drivers.empty:
            # First check if driver_id column already exists (new format)
            if 'driver_id' in dhl_data.columns and dhl_data['driver_id'].notna().any():
                # Use existing driver_id mapping
                print("Using driver_id column from DHL data")
                dhl_data['driverId'] = dhl_data['driver_id']
            else:
                # Fall back to name-based mapping (old format)
                # Find the driver name column - could be 'driver' or 'driver_name'
                driver_col = None
                for col in ['driver', 'driver_name', 'Driver', 'DRIVER']:
                    if col in dhl_data.columns:
                        driver_col = col
                        print(f"Using '{col}' column for driver names")
                        break
                
                if driver_col is None:
                    print(f"Error: Could not find driver column in DHL data. Available columns: {list(dhl_data.columns)}")
                    return pd.DataFrame()
                
                # Create mapping from various name formats to driverId
                driver_mapping = {}
                for _, driver in drivers.iterrows():
                    full_name = f"{driver.get('forename', '')} {driver.get('surname', '')}".strip()
                    last_name = driver.get('surname', '')
                    code = driver.get('code', '')
                    driver_id = driver.get('id')
                    
                    # Add various name formats to mapping
                    driver_mapping[full_name.lower()] = driver_id
                    driver_mapping[last_name.lower()] = driver_id
                    if code:
                        driver_mapping[code.lower()] = driver_id
            
                # Create special mappings for known mismatches
                special_mappings = {
                    'hulkenberg': 'nico-hulkenberg',
                    'hülkenberg': 'nico-hulkenberg',
                    'sainz': 'carlos-sainz-jr',      # DHL uses just "Sainz"
                    'perez': 'sergio-perez',          # DHL uses "Perez" without accent
                    'pérez': 'sergio-perez',
                    'magnussen': 'kevin-magnussen',   # Disambiguate from jan-magnussen
                    'leclerc': 'charles-leclerc',     # Disambiguate from arthur-leclerc
                    'zhou': 'guanyu-zhou',
                    'ricciardo': 'daniel-ricciardo',
                    'verstappen': 'max-verstappen',
                    'bottas': 'valtteri-bottas',
                    'bearman': 'oliver-bearman',
                    'colapinto': 'franco-colapinto',
                    'hamilton': 'lewis-hamilton',
                    'russell': 'george-russell',
                    'alonso': 'fernando-alonso',
                    'stroll': 'lance-stroll',
                    'norris': 'lando-norris',
                    'piastri': 'oscar-piastri',
                    'gasly': 'pierre-gasly',
                    'ocon': 'esteban-ocon',
                    'tsunoda': 'yuki-tsunoda',
                    'lawson': 'liam-lawson',
                    'albon': 'alexander-albon',
                    'sargeant': 'logan-sargeant',
                    'doohan': 'jack-doohan',
                    'hadjar': 'isack-hadjar',
                    'antonelli': 'kimi-antonelli',
                    'bortoleto': 'gabriel-bortoleto'
                }
                
                # Add driver IDs to DHL data
                def map_driver(driver_name):
                    driver_lower = driver_name.lower()
                    # First check special mappings
                    if driver_lower in special_mappings:
                        return special_mappings[driver_lower]
                    # Then check regular mapping
                    return driver_mapping.get(driver_lower)
                
                dhl_data['driverId'] = dhl_data[driver_col].apply(map_driver)
            
            # Log unmapped drivers for debugging (only if we had to map from names)
            if 'driver_id' not in dhl_data.columns or dhl_data['driver_id'].isna().all():
                unmapped = dhl_data[dhl_data['driverId'].isna()]
                if not unmapped.empty and 'driver_col' in locals():
                    unmapped_drivers = unmapped[driver_col].unique()
                    print(f"Warning: Could not map {len(unmapped_drivers)} DHL drivers: {', '.join(unmapped_drivers)}")
            
            dhl_data = dhl_data.dropna(subset=['driverId'])
        
        # Map race_id to circuit IDs using the races table
        if 'race_id' in dhl_data.columns and not races.empty:
            # Create mapping from race_id to circuitId
            race_to_circuit = races.set_index('id')['circuitId'].to_dict()
            
            # Map race_id to circuitId
            dhl_data['circuitId'] = dhl_data['race_id'].map(race_to_circuit)
            
            # Debug output
            mapped_circuits = dhl_data['circuitId'].notna().sum()
            total_rows = len(dhl_data)
            print(f"Mapped {mapped_circuits}/{total_rows} DHL pit stops to circuits using race_id")
            
            # Additional debug - check if mapping is working
            if mapped_circuits == 0:
                print("Warning: No circuits mapped. Checking data...")
                print(f"Sample race_ids from DHL: {dhl_data['race_id'].dropna().head().tolist()}")
                print(f"Sample race ids from races.csv: {list(race_to_circuit.keys())[:5]}")
                
                # Check data types
                print(f"DHL race_id dtype: {dhl_data['race_id'].dtype}")
                print(f"Races id dtype: {races['id'].dtype}")
        
        # Fallback: Map race names to circuit IDs if race_id mapping didn't work
        elif 'event_name' in dhl_data.columns or 'race' in dhl_data.columns:
            race_column = 'event_name' if 'event_name' in dhl_data.columns else 'race'
            if not races.empty and not circuits.empty:
                # Create mapping from race names to circuit IDs
                race_circuit_map = {}
                
                for _, race in races.iterrows():
                    race_name = race.get('name', '')
                    circuit_id = race.get('circuitId')
                    if circuit_id:
                        # Map various race name formats
                        race_circuit_map[race_name.lower()] = circuit_id
                        # Also try to match partial names
                        for word in race_name.split():
                            if len(word) > 4:  # Skip short words
                                race_circuit_map[word.lower()] = circuit_id
                
                # Add circuit IDs to DHL data
                dhl_data['circuitId'] = dhl_data[race_column].str.lower().map(
                    lambda x: next((cid for race_name, cid in race_circuit_map.items() 
                                  if race_name in x or x in race_name), None)
                )
        
        # Group by driver and calculate metrics
        dhl_analysis = dhl_data.groupby('driverId').agg({
            'time': ['mean', 'median', 'min', 'count'],
            'lap': lambda x: dhl_data.loc[x.index, 'lap'].iloc[dhl_data.loc[x.index, 'time'].argmin()] if len(x) > 0 else None
        }).round(3)
        
        # Flatten column names
        dhl_analysis.columns = ['avg_time', 'median_time', 'best_time', 'total_stops', 'best_time_lap']
        
        # Add first stop analysis
        # Use event_id instead of race for grouping
        sort_columns = ['driverId', 'lap']
        group_column = 'event_id' if 'event_id' in dhl_data.columns else 'race'
        
        if group_column in dhl_data.columns:
            first_stops = dhl_data.sort_values(sort_columns).groupby(['driverId', group_column]).first()
            first_stop_analysis = first_stops.groupby('driverId')['time'].agg(['mean', 'median', 'min', 'count']).round(3)
        else:
            # Fallback: just group by driver
            first_stop_analysis = dhl_data.groupby('driverId')['time'].agg(['mean', 'median', 'min', 'count']).round(3)
        first_stop_analysis.columns = ['avg_first_stop', 'median_first_stop', 'best_first_stop', 'first_stop_races']
        
        # Join first stop analysis to main analysis
        dhl_analysis = dhl_analysis.join(first_stop_analysis, how='left')
        
        # Add next circuit average if we can determine the next race circuit
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race and 'circuitId' in dhl_data.columns:
            next_circuit_id = next_race['circuitId']
            circuit_stops = dhl_data[dhl_data['circuitId'] == next_circuit_id]
            
            if not circuit_stops.empty:
                # Overall times at this circuit
                circuit_times = circuit_stops.groupby('driverId')['time'].mean().round(3)
                dhl_analysis['next_circuit_avg'] = dhl_analysis.index.map(
                    lambda x: circuit_times.get(x, dhl_analysis.loc[x, 'avg_time'] if x in dhl_analysis.index else None)
                )
                
                # First stop times at this circuit
                circuit_first_stops = circuit_stops.sort_values(['driverId', 'lap']).groupby('driverId').first()
                circuit_first_times = circuit_first_stops['time'].round(3)
                dhl_analysis['next_circuit_first_stop'] = dhl_analysis.index.map(
                    lambda x: circuit_first_times.get(x, None)
                )
        
        # Add year-by-year analysis
        if 'year' in dhl_data.columns:
            yearly_stats = []
            for driver_id in dhl_analysis.index:
                driver_data = dhl_data[dhl_data['driverId'] == driver_id]
                for year in driver_data['year'].unique():
                    year_data = driver_data[driver_data['year'] == year]
                    if not year_data.empty:
                        yearly_stats.append({
                            'driverId': driver_id,
                            'year': year,
                            'avg_time': year_data['time'].mean(),
                            'median_time': year_data['time'].median(),
                            'best_time': year_data['time'].min(),
                            'total_stops': len(year_data),
                            'best_time_lap': year_data.loc[year_data['time'].idxmin(), 'lap'] if len(year_data) > 0 else None
                        })
            
            if yearly_stats:
                dhl_analysis.yearly_data = pd.DataFrame(yearly_stats)
        
        # Filter to current season drivers
        dhl_analysis = self.filter_current_season_drivers(dhl_analysis)
        
        # Ensure all current season drivers are included, even with no data
        current_drivers = self.get_active_drivers()
        if not current_drivers.empty:
            # Get all driver IDs that should be included
            all_driver_ids = set(current_drivers['id'].values)
            # Get driver IDs that are already in the analysis
            existing_ids = set(dhl_analysis.index)
            # Find missing driver IDs
            missing_ids = all_driver_ids - existing_ids
            
            # Add missing drivers with NaN values
            if missing_ids:
                missing_data = pd.DataFrame(
                    index=list(missing_ids),
                    columns=dhl_analysis.columns
                )
                dhl_analysis = pd.concat([dhl_analysis, missing_data])
        
        return dhl_analysis
    
    def analyze_dhl_pit_stops_by_track_year(self):
        """Analyze DHL pit stop data for a specific track across years"""
        dhl_data = self.dhl_data.copy()
        
        if dhl_data.empty:
            return {}
            
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None:
            return {}
            
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        drivers = self.data.get('drivers', pd.DataFrame())
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if 'circuitId' in next_race and not circuits.empty:
            circuit = circuits[circuits['id'] == next_race['circuitId']]
            if not circuit.empty:
                circuit_name = circuit.iloc[0]['name']
        
        # Map driver IDs first (similar to main analyze_dhl_pit_stops method)
        if not drivers.empty:
            if 'driver_id' in dhl_data.columns and dhl_data['driver_id'].notna().any():
                dhl_data['driverId'] = dhl_data['driver_id']
            else:
                # Find driver column and map names
                driver_col = None
                for col in ['driver', 'driver_name', 'Driver', 'DRIVER']:
                    if col in dhl_data.columns:
                        driver_col = col
                        break
                
                if driver_col:
                    # Apply same mapping logic as main method
                    driver_mapping = {}
                    for _, driver in drivers.iterrows():
                        full_name = f"{driver.get('forename', '')} {driver.get('surname', '')}".strip()
                        last_name = driver.get('surname', '')
                        driver_id = driver.get('id')
                        driver_mapping[full_name.lower()] = driver_id
                        driver_mapping[last_name.lower()] = driver_id
                    
                    # Special mappings
                    special_mappings = {
                        'hulkenberg': 'nico-hulkenberg', 'hülkenberg': 'nico-hulkenberg',
                        'sainz': 'carlos-sainz-jr', 'perez': 'sergio-perez', 'pérez': 'sergio-perez',
                        'magnussen': 'kevin-magnussen', 'leclerc': 'charles-leclerc',
                        'zhou': 'guanyu-zhou', 'ricciardo': 'daniel-ricciardo',
                        'verstappen': 'max-verstappen', 'bottas': 'valtteri-bottas',
                        'bearman': 'oliver-bearman', 'colapinto': 'franco-colapinto'
                    }
                    
                    def map_driver(name):
                        name_lower = name.lower()
                        if name_lower in special_mappings:
                            return special_mappings[name_lower]
                        return driver_mapping.get(name_lower)
                    
                    dhl_data['driverId'] = dhl_data[driver_col].apply(map_driver)
                    dhl_data = dhl_data.dropna(subset=['driverId'])
        
        # Map race_id to circuit IDs using the races table (same as main method)
        if 'race_id' in dhl_data.columns and not races.empty:
            # Create mapping from race_id to circuitId
            race_to_circuit = races.set_index('id')['circuitId'].to_dict()
            
            # Map race_id to circuitId
            dhl_data['circuitId'] = dhl_data['race_id'].map(race_to_circuit)
            
            # Debug output
            mapped_circuits = dhl_data['circuitId'].notna().sum()
            total_rows = len(dhl_data)
            print(f"Mapped {mapped_circuits}/{total_rows} DHL pit stops to circuits using race_id")
            
            if 'circuitId' in next_race:
                target_circuit = next_race['circuitId']
                matches = (dhl_data['circuitId'] == target_circuit).sum()
                print(f"Found {matches} pit stops at {circuit_name} (circuit ID: {target_circuit})")
                
                # If no matches, debug further
                if matches == 0 and mapped_circuits > 0:
                    unique_circuits = dhl_data['circuitId'].dropna().unique()
                    print(f"DHL data contains circuits: {list(unique_circuits)[:10]}...")
                    
                    # Check if Hungary is in the data with different ID
                    hungary_races = races[races['circuitId'] == target_circuit]
                    if not hungary_races.empty:
                        hungary_race_ids = hungary_races['id'].tolist()
                        hungary_in_dhl = dhl_data['race_id'].isin(hungary_race_ids).sum()
                        print(f"Found {hungary_in_dhl} pit stops with Hungary race IDs: {hungary_race_ids[:5]}...")
        
        # Fallback: Use circuit_id column if it exists
        elif 'circuit_id' in dhl_data.columns and dhl_data['circuit_id'].notna().any():
            # Use existing circuit_id mapping from new format
            dhl_data['circuitId'] = dhl_data['circuit_id']
            print(f"Using circuit_id column from DHL data")
        elif not races.empty and 'race' in dhl_data.columns:
            race_circuit_map = {}
            for _, race in races.iterrows():
                race_name = race.get('name', '')
                circuit_id = race.get('circuitId')
                if circuit_id:
                    race_circuit_map[race_name.lower()] = circuit_id
                    for word in race_name.split():
                        if len(word) > 4:
                            race_circuit_map[word.lower()] = circuit_id
            
            dhl_data['circuitId'] = dhl_data['race'].str.lower().map(
                lambda x: next((cid for race_name, cid in race_circuit_map.items() 
                              if race_name in x or x in race_name), None)
            )
        
        # Filter for the specific circuit
        if 'circuitId' not in next_race or 'circuitId' not in dhl_data.columns:
            return {'circuit_name': circuit_name, 'year_by_year': pd.DataFrame(), 'overall_stats': pd.DataFrame()}
            
        circuit_id = next_race['circuitId']
        circuit_data = dhl_data[dhl_data['circuitId'] == circuit_id].copy()
        
        if circuit_data.empty:
            return {'circuit_name': circuit_name, 'year_by_year': pd.DataFrame(), 'overall_stats': pd.DataFrame()}
        
        # Add year information from races data if not present
        if 'year' not in circuit_data.columns and 'race_id' in circuit_data.columns and not races.empty:
            race_year_map = races.set_index('id')['year'].to_dict()
            circuit_data['year'] = circuit_data['race_id'].map(race_year_map)
        
        # Get first stops only for this circuit
        group_cols = ['driverId', 'race_id'] if 'race_id' in circuit_data.columns else ['driverId', 'event_id']
        sort_cols = ['driverId', 'lap']
        if 'year' in circuit_data.columns:
            sort_cols.insert(1, 'year')
            group_cols.insert(1, 'year')
        
        circuit_first_stops = circuit_data.sort_values(sort_cols).groupby(group_cols).first().reset_index()
        
        # Year-by-year analysis
        year_stats = []
        if 'year' in circuit_first_stops.columns and 'driverId' in circuit_first_stops.columns:
            for (driver_id, year), group in circuit_first_stops.groupby(['driverId', 'year']):
                year_stats.append({
                    'driverId': driver_id,
                    'year': year,
                    'avg_time': group['time'].mean(),
                    'median_time': group['time'].median(),
                    'best_time': group['time'].min(),
                    'stops': len(group),
                    'avg_lap': group['lap'].mean()
                })
        
        year_by_year_df = pd.DataFrame(year_stats) if year_stats else pd.DataFrame()
        
        # Overall career stats at this track (first stops only)
        overall_stats = circuit_first_stops.groupby('driverId').agg({
            'time': ['mean', 'min', 'count'],
            'lap': 'mean'
        }).round(3)
        overall_stats.columns = ['avg_first_stop', 'best_first_stop', 'total_races', 'avg_lap']
        
        # Add driver names
        if not drivers.empty and not year_by_year_df.empty:
            # Use 'name' column which contains full driver names
            driver_names = drivers.set_index('id')['name'].to_dict()
            year_by_year_df['driver_name'] = year_by_year_df['driverId'].map(driver_names)
            
        if not drivers.empty and not overall_stats.empty:
            driver_names = drivers.set_index('id')['name'].to_dict()
            overall_stats['driver_name'] = overall_stats.index.map(driver_names)
        
        # Filter to current season drivers
        current_drivers = self.get_active_drivers()
        if not current_drivers.empty:
            active_ids = set(current_drivers['id'].unique())
            if not year_by_year_df.empty:
                year_by_year_df = year_by_year_df[year_by_year_df['driverId'].isin(active_ids)]
            if not overall_stats.empty:
                overall_stats = overall_stats[overall_stats.index.isin(active_ids)]
        
        return {
            'circuit_name': circuit_name,
            'year_by_year': year_by_year_df,
            'overall_stats': overall_stats
        }
    
    def analyze_starting_positions(self):
        """Analyze starting positions by driver"""
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        results = self.data.get('races-race-results', pd.DataFrame())
        
        if grid.empty:
            return {}
        
        # Add year and race info
        if not races.empty:
            grid = grid.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
        # Filter recent years
        recent_grid = grid[grid['year'] >= self.current_year - 3] if 'year' in grid.columns else grid
        
        # Analyze by driver
        grid_analysis = recent_grid.groupby('driverId').agg({
            'positionNumber': ['mean', 'median', 'min', 'count']
        }).round(2)
        grid_analysis.columns = ['avg_start_position', 'median_start_position', 'best_start_position', 'races']
        
        # Add finish position correlation
        if not results.empty:
            # Merge grid with results
            grid_results = recent_grid.merge(
                results[['raceId', 'driverId', 'position', 'points']].rename(columns={'position': 'finish_position'}),
                on=['raceId', 'driverId'],
                how='left'
            )
            
            # Calculate average points from each starting position
            points_by_start = grid_results.groupby('driverId').agg({
                'points': ['sum', 'mean']
            }).round(2)
            points_by_start.columns = ['total_points', 'avg_points_per_race']
            
            grid_analysis = grid_analysis.join(points_by_start, how='left')
        
        # Keep driverId as index for proper identification
        
        # Filter to only current season drivers
        grid_analysis = self.filter_current_season_drivers(grid_analysis)
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            grid_analysis['driver_name'] = grid_analysis.index.map(driver_map)
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_grid = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_grid.empty:
                circuit_positions = circuit_grid.groupby('driverId')['position'].mean().round(2)
                grid_analysis['next_circuit_avg'] = grid_analysis.index.map(
                    lambda x: circuit_positions.get(x, grid_analysis.loc[x, 'avg_start_position'] if x in grid_analysis.index else 0)
                )
        
        # Drop driver_id index since we already have driver_name
        grid_analysis = grid_analysis.reset_index(drop=True)
        
        return grid_analysis
    
    def analyze_starting_positions_by_track_year(self):
        """Analyze starting positions by driver for each track, broken down by year"""
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if grid.empty or races.empty:
            return {}
        
        # The grid data already has year, but we need circuitId from races
        if 'circuitId' not in grid.columns:
            grid = grid.merge(races[['id', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None or 'circuitId' not in next_race:
            return {}
        
        next_circuit_id = next_race['circuitId']
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if not circuits.empty and 'id' in circuits.columns:
            circuit_info = circuits[circuits['id'] == next_circuit_id]
            if not circuit_info.empty:
                circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
        # Filter for the next race's circuit only
        circuit_grid = grid[grid['circuitId'] == next_circuit_id].copy()
        
        if circuit_grid.empty:
            return {}
        
        # Ensure year column exists after filtering
        if 'year' not in circuit_grid.columns:
            print("Warning: No year column found in circuit grid data")
            return {}
        
        # Get finish positions if available
        if not results.empty:
            # Merge with results to get finish positions
            circuit_grid = circuit_grid.merge(
                results[['raceId', 'driverId', 'positionNumber', 'points']].rename(
                    columns={'positionNumber': 'finishPosition'}
                ),
                on=['raceId', 'driverId'],
                how='left'
            )
        
        # Group by driver and year
        track_year_grid = circuit_grid.groupby(['driverId', 'year']).agg({
            'positionNumber': ['mean', 'median', 'min', 'count']
        }).round(2)
        
        # Add points data if available
        if 'points' in circuit_grid.columns:
            points_agg = circuit_grid.groupby(['driverId', 'year'])['points'].agg(['sum', 'mean']).round(2)
            track_year_grid = pd.concat([track_year_grid, points_agg], axis=1)
            
            # Flatten column names with points
            track_year_grid.columns = [
                'avg_start', 'median_start', 'best_start', 'races',
                'total_points', 'avg_points'
            ]
        else:
            # Flatten column names without points
            track_year_grid.columns = [
                'avg_start', 'median_start', 'best_start', 'races'
            ]
        
        # Reset index to make driverId and year regular columns
        track_year_grid = track_year_grid.reset_index()
        
        # Add overall statistics per driver across all years at this track
        overall_agg = {
            'positionNumber': ['mean', 'median', 'min'],
            'year': 'count'
        }
        if 'points' in circuit_grid.columns:
            overall_agg['points'] = ['sum', 'mean']
            
        overall_stats = circuit_grid.groupby('driverId').agg(overall_agg).round(2)
        
        # Set column names based on what was aggregated
        if 'points' in circuit_grid.columns:
            overall_stats.columns = [
                'career_avg_start', 'career_median_start', 'career_best_start',
                'races_at_track', 'career_points', 'career_avg_points'
            ]
        else:
            overall_stats.columns = [
                'career_avg_start', 'career_median_start', 'career_best_start',
                'races_at_track'
            ]
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            track_year_grid['driver_name'] = track_year_grid['driverId'].map(driver_map)
            overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
        # Filter to only show current season drivers
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            current_driver_ids = active_drivers['id'].tolist()
            track_year_grid = track_year_grid[
                track_year_grid['driverId'].isin(current_driver_ids)
            ]
            overall_stats = overall_stats[overall_stats.index.isin(current_driver_ids)]
        
        return {
            'circuit_name': circuit_name,
            'circuit_id': next_circuit_id,
            'year_by_year': track_year_grid,
            'overall_stats': overall_stats
        }
    
    def analyze_sprint_points(self):
        """Analyze sprint race points by driver"""
        sprint_results = self.data.get('sprint_results', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        
        if sprint_results.empty:
            return pd.DataFrame()
        
        # Add year information
        if not races.empty:
            sprint_results = sprint_results.merge(
                races[['id', 'year', 'circuitId']], 
                left_on='raceId', 
                right_on='id', 
                how='left'
            )
        
        # Filter recent years
        recent_sprints = sprint_results[sprint_results['year'] >= self.current_year - 3] if 'year' in sprint_results.columns else sprint_results
        
        # Check if we have data
        if recent_sprints.empty:
            print(f"No sprint data found for years {self.current_year - 3} to {self.current_year}")
            return pd.DataFrame()
        
        # Analyze by driver with correct column names
        agg_dict = {'points': ['sum', 'mean', 'median', 'count']}
        if 'positionNumber' in recent_sprints.columns:
            agg_dict['positionNumber'] = ['mean', 'median']
        
        sprint_analysis = recent_sprints.groupby('driverId').agg(agg_dict).round(2)
        
        # Set column names based on what was aggregated
        col_names = ['total_sprint_points', 'avg_sprint_points', 'median_sprint_points', 'sprint_races']
        if 'positionNumber' in recent_sprints.columns:
            col_names.extend(['avg_sprint_position', 'median_sprint_position'])
        sprint_analysis.columns = col_names
        
        # Keep driverId as index for proper identification
        
        # Filter to only current season drivers
        sprint_analysis = self.filter_current_season_drivers(sprint_analysis)
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            sprint_analysis['driver_name'] = sprint_analysis.index.map(driver_map)
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_sprints = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_sprints.empty:
                circuit_sprint_points = circuit_sprints.groupby('driverId')['points'].mean().round(2)
                sprint_analysis['next_circuit_avg'] = sprint_analysis.index.map(
                    lambda x: circuit_sprint_points.get(x, sprint_analysis.loc[x, 'avg_sprint_points'] if x in sprint_analysis.index else 0)
                )
        
        # Drop driver_id index since we already have driver_name
        sprint_analysis = sprint_analysis.reset_index(drop=True)
        
        return sprint_analysis
    
    def analyze_sprint_points_by_track_year(self):
        """Analyze sprint points by driver for each track, broken down by year"""
        sprint_results = self.data.get('sprint_results', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if sprint_results.empty or races.empty:
            return pd.DataFrame()
        
        # Add year and circuitId information
        if 'year' not in sprint_results.columns or 'circuitId' not in sprint_results.columns:
            sprint_results = sprint_results.merge(
                races[['id', 'year', 'circuitId']], 
                left_on='raceId', 
                right_on='id', 
                how='left'
            )
        
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None or 'circuitId' not in next_race:
            return {}
        
        next_circuit_id = next_race['circuitId']
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if not circuits.empty and 'id' in circuits.columns:
            circuit_info = circuits[circuits['id'] == next_circuit_id]
            if not circuit_info.empty:
                circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
        # Filter for the next race's circuit only
        circuit_sprints = sprint_results[sprint_results['circuitId'] == next_circuit_id].copy()
        
        if circuit_sprints.empty:
            return {}
        
        # Ensure year column exists
        if 'year' not in circuit_sprints.columns:
            print("Warning: No year column found in sprint data")
            return {}
        
        # Group by driver and year
        track_year_sprints = circuit_sprints.groupby(['driverId', 'year']).agg({
            'points': ['sum', 'mean', 'count'],
            'positionNumber': ['mean', 'min']
        }).round(2)
        
        # Flatten column names
        track_year_sprints.columns = [
            'total_points', 'avg_points', 'races',
            'avg_finish', 'best_finish'
        ]
        
        # Reset index
        track_year_sprints = track_year_sprints.reset_index()
        
        # Add overall statistics per driver across all years at this track
        overall_stats = circuit_sprints.groupby('driverId').agg({
            'points': ['sum', 'mean', 'count'],
            'positionNumber': ['mean', 'min'],
            'year': 'nunique'
        }).round(2)
        
        overall_stats.columns = [
            'career_points', 'career_avg_points', 'total_races',
            'career_avg_finish', 'career_best_finish',
            'years_raced'
        ]
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            track_year_sprints['driver_name'] = track_year_sprints['driverId'].map(driver_map)
            overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
        # Filter to only show current season drivers
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            current_driver_ids = active_drivers['id'].tolist()
            track_year_sprints = track_year_sprints[
                track_year_sprints['driverId'].isin(current_driver_ids)
            ]
            overall_stats = overall_stats[overall_stats.index.isin(current_driver_ids)]
        
        return {
            'circuit_name': circuit_name,
            'circuit_id': next_circuit_id,
            'year_by_year': track_year_sprints,
            'overall_stats': overall_stats
        }
    
    def analyze_teammate_overtakes(self):
        """Analyze teammate overtakes with PrizePicks scoring (+/- 0.5 for teammate battles)"""
        results = self.data.get('results', pd.DataFrame()).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty or grid.empty:
            return pd.DataFrame()
        
        # Merge results with starting grid
        overtake_data = results.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Add year information if not present
        if 'year' not in overtake_data.columns and not races.empty:
            overtake_data = overtake_data.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
        
        # Filter for recent years
        if 'year' in overtake_data.columns:
            recent_data = overtake_data[overtake_data['year'] >= self.current_year - 3]
        else:
            recent_data = overtake_data
        
        # Get teammate pairs for each race
        teammate_results = []
        
        for race_id in recent_data['raceId'].unique():
            race_data = recent_data[recent_data['raceId'] == race_id]
            
            # Group by constructor to find teammates
            for constructor_id in race_data['constructorId'].unique():
                team_data = race_data[race_data['constructorId'] == constructor_id]
                
                if len(team_data) >= 2:  # Need at least 2 drivers
                    drivers = team_data[['driverId', 'gridPosition', 'positionNumber']].values
                    
                    # Compare each pair of teammates
                    for i in range(len(drivers)):
                        for j in range(i+1, len(drivers)):
                            driver1_id, driver1_grid, driver1_finish = drivers[i]
                            driver2_id, driver2_grid, driver2_finish = drivers[j]
                            
                            # Skip if any position is NaN
                            if pd.isna(driver1_grid) or pd.isna(driver1_finish) or pd.isna(driver2_grid) or pd.isna(driver2_finish):
                                continue
                            
                            # Calculate who beat whom
                            if driver1_finish < driver2_finish:  # Driver 1 beat Driver 2
                                winner_id = driver1_id
                                loser_id = driver2_id
                                winner_started_ahead = driver1_grid < driver2_grid
                            else:  # Driver 2 beat Driver 1
                                winner_id = driver2_id
                                loser_id = driver1_id
                                winner_started_ahead = driver2_grid < driver1_grid
                            
                            # Check if it was an overtake (winner started behind)
                            was_overtake = not winner_started_ahead
                            
                            teammate_results.append({
                                'raceId': race_id,
                                'year': race_data.iloc[0]['year'] if 'year' in race_data.columns else None,
                                'winnerId': winner_id,
                                'loserId': loser_id,
                                'was_overtake': was_overtake,
                                'constructorId': constructor_id
                            })
        
        if not teammate_results:
            return pd.DataFrame()
        
        teammate_df = pd.DataFrame(teammate_results)
        
        # Aggregate by driver
        driver_stats = []
        all_drivers = pd.concat([teammate_df['winnerId'], teammate_df['loserId']]).unique()
        
        for driver_id in all_drivers:
            # Wins against teammates
            wins = teammate_df[teammate_df['winnerId'] == driver_id]
            losses = teammate_df[teammate_df['loserId'] == driver_id]
            
            # Overtakes of teammates
            overtakes_made = wins[wins['was_overtake'] == True]
            overtaken_by_teammate = losses[losses['was_overtake'] == True]
            
            driver_stats.append({
                'driverId': driver_id,
                'total_battles': len(wins) + len(losses),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0,
                'OT_made': len(overtakes_made),
                'OT_received': len(overtaken_by_teammate),
                'net_OT': len(overtakes_made) - len(overtaken_by_teammate),
                'prizepicks_pts': len(overtakes_made) * 1.5 - len(overtaken_by_teammate) * 1.5
            })
        
        teammate_analysis = pd.DataFrame(driver_stats).set_index('driverId')
        teammate_analysis = teammate_analysis.round(3)
        
        # Filter to only current season drivers
        teammate_analysis = self.filter_current_season_drivers(teammate_analysis)
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            teammate_analysis['driver_name'] = teammate_analysis.index.map(driver_map)
        
        return teammate_analysis
    
    def analyze_fastest_laps(self):
        """Analyze fastest lap achievements by driver"""
        fastest_laps = self.data.get('fastest_laps', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        
        if fastest_laps.empty:
            return pd.DataFrame()
        
        # Add year information if not present
        if 'year' not in fastest_laps.columns and not races.empty:
            fastest_laps = fastest_laps.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
        # Filter recent years
        if 'year' in fastest_laps.columns:
            recent_laps = fastest_laps[fastest_laps['year'] >= self.current_year - 3]
        else:
            recent_laps = fastest_laps
        
        # Get total races per driver to calculate rates
        results = self.data.get('results', pd.DataFrame())
        if not results.empty and 'year' in results.columns:
            recent_results = results[results['year'] >= self.current_year - 3]
            races_per_driver = recent_results.groupby('driverId').size()
        else:
            races_per_driver = pd.Series()
        
        # Analyze by driver
        lap_analysis = recent_laps.groupby('driverId').agg({
            'positionNumber': 'count',  # Number of fastest laps
            'lap': ['mean', 'std'],     # Which lap they typically get fastest lap
            'year': 'nunique'           # Number of seasons with fastest laps
        })
        
        lap_analysis.columns = ['total_fastest_laps', 'avg_lap_number', 'lap_number_std', 'seasons_with_fl']
        
        # Add race count and calculate rate
        lap_analysis['total_races'] = lap_analysis.index.map(lambda x: races_per_driver.get(x, 0))
        lap_analysis['fastest_lap_rate'] = (lap_analysis['total_fastest_laps'] / lap_analysis['total_races']).fillna(0)
        
        # Add points earned from fastest laps (1 point each)
        lap_analysis['fastest_lap_points'] = lap_analysis['total_fastest_laps']
        
        # Round values
        lap_analysis = lap_analysis.round(3)
        
        # Filter to only current season drivers
        lap_analysis = self.filter_current_season_drivers(lap_analysis)
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            lap_analysis['driver_name'] = lap_analysis.index.map(driver_map)
        
        return lap_analysis
    
    def analyze_teammate_overtakes_by_track_year(self):
        """Analyze teammate overtakes by track and year"""
        results = self.data.get('results', pd.DataFrame()).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if results.empty or grid.empty or races.empty:
            return {}
        
        # Merge all necessary data
        overtake_data = results.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Add circuit info from races table (year already exists in results)
        if 'circuitId' not in overtake_data.columns:
            overtake_data = overtake_data.merge(
                races[['id', 'circuitId']], 
                left_on='raceId', 
                right_on='id', 
                how='left'
            )
        
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None or 'circuitId' not in next_race:
            return {}
        
        next_circuit_id = next_race['circuitId']
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if not circuits.empty:
            circuit_info = circuits[circuits['id'] == next_circuit_id]
            if not circuit_info.empty:
                circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
        # Filter for the specific circuit
        circuit_data = overtake_data[overtake_data['circuitId'] == next_circuit_id].copy()
        
        if circuit_data.empty or 'year' not in circuit_data.columns:
            return {}
        
        # Drop any rows where year is missing
        circuit_data = circuit_data.dropna(subset=['year'])
        
        if circuit_data.empty:
            return {}
        
        # Calculate teammate battles by year
        year_stats = []
        
        for year in circuit_data['year'].unique():
            year_data = circuit_data[circuit_data['year'] == year]
            
            # Process each race in that year
            for race_id in year_data['raceId'].unique():
                race_data = year_data[year_data['raceId'] == race_id]
                
                # Group by constructor
                for constructor_id in race_data['constructorId'].unique():
                    team_data = race_data[race_data['constructorId'] == constructor_id]
                    
                    if len(team_data) >= 2:
                        drivers = team_data[['driverId', 'gridPosition', 'positionNumber']].values
                        
                        for i in range(len(drivers)):
                            for j in range(i+1, len(drivers)):
                                driver1_id, driver1_grid, driver1_finish = drivers[i]
                                driver2_id, driver2_grid, driver2_finish = drivers[j]
                                
                                if pd.isna(driver1_grid) or pd.isna(driver1_finish) or pd.isna(driver2_grid) or pd.isna(driver2_finish):
                                    continue
                                
                                # Determine winner and if it was an overtake
                                if driver1_finish < driver2_finish:
                                    winner_id = driver1_id
                                    loser_id = driver2_id
                                    was_overtake = driver1_grid > driver2_grid
                                else:
                                    winner_id = driver2_id
                                    loser_id = driver1_id
                                    was_overtake = driver2_grid > driver1_grid
                                
                                year_stats.append({
                                    'driverId': winner_id,
                                    'year': year,
                                    'teammate_win': 1,
                                    'teammate_overtake': 1 if was_overtake else 0
                                })
                                year_stats.append({
                                    'driverId': loser_id,
                                    'year': year,
                                    'teammate_win': 0,
                                    'teammate_overtake': -1 if was_overtake else 0
                                })
        
        if not year_stats:
            return {}
        
        year_df = pd.DataFrame(year_stats)
        
        # Aggregate by driver and year
        track_year_analysis = year_df.groupby(['driverId', 'year']).agg({
            'teammate_win': 'sum',
            'teammate_overtake': 'sum'
        }).reset_index()
        
        # Calculate additional stats
        track_year_analysis['teammate_battles'] = track_year_analysis.groupby(['driverId', 'year'])['teammate_win'].transform('count')
        track_year_analysis['teammate_win_rate'] = (track_year_analysis['teammate_win'] / track_year_analysis['teammate_battles']).round(3)
        track_year_analysis['prizepicks_points'] = track_year_analysis['teammate_overtake'] * 1.5
        
        # Overall stats
        overall_stats = year_df.groupby('driverId').agg({
            'teammate_win': ['sum', 'count'],
            'teammate_overtake': 'sum'
        })
        
        overall_stats.columns = ['total_wins', 'total_battles', 'net_overtakes']
        overall_stats['win_rate'] = (overall_stats['total_wins'] / overall_stats['total_battles']).round(3)
        overall_stats['career_prizepicks_points'] = overall_stats['net_overtakes'] * 1.5
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            track_year_analysis['driver_name'] = track_year_analysis['driverId'].map(driver_map)
            overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
        # For track-specific analysis, show all drivers who have raced at this track
        # Don't filter to only current season drivers - show historical data
        
        
        return {
            'circuit_name': circuit_name,
            'circuit_id': next_circuit_id,
            'year_by_year': track_year_analysis,
            'overall_stats': overall_stats
        }
    
    def analyze_fastest_laps_by_track_year(self):
        """Analyze fastest laps by track and year"""
        fastest_laps = self.data.get('fastest_laps', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if fastest_laps.empty or races.empty:
            return {}
        
        # Add circuit info from races table (year already exists in fastest_laps)
        if 'circuitId' not in fastest_laps.columns:
            fastest_laps = fastest_laps.merge(
                races[['id', 'circuitId']], 
                left_on='raceId', 
                right_on='id', 
                how='left'
            )
        
        # Get next race info
        next_race = self.get_next_race()
        if next_race is None or 'circuitId' not in next_race:
            return {}
        
        next_circuit_id = next_race['circuitId']
        
        # Get circuit name
        circuit_name = 'Unknown Circuit'
        if not circuits.empty:
            circuit_info = circuits[circuits['id'] == next_circuit_id]
            if not circuit_info.empty:
                circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
        # Filter for specific circuit
        circuit_laps = fastest_laps[fastest_laps['circuitId'] == next_circuit_id].copy()
        
        if circuit_laps.empty or 'year' not in circuit_laps.columns:
            return {}
        
        # Drop any rows where year is missing
        circuit_laps = circuit_laps.dropna(subset=['year'])
        
        if circuit_laps.empty:
            return {}
        
        # Year by year analysis
        track_year_analysis = circuit_laps.groupby(['driverId', 'year']).agg({
            'positionNumber': 'count',  # Count of fastest laps
            'lap': 'first',  # Which lap
            'time': 'first'  # Lap time
        }).reset_index()
        
        track_year_analysis.columns = ['driverId', 'year', 'fastest_laps', 'lap_number', 'lap_time']
        
        # Overall stats
        overall_stats = circuit_laps.groupby('driverId').agg({
            'positionNumber': 'count',
            'lap': ['mean', 'std'],
            'year': 'nunique'
        })
        
        overall_stats.columns = ['total_fastest_laps', 'avg_lap_number', 'lap_std', 'years_with_fl']
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            track_year_analysis['driver_name'] = track_year_analysis['driverId'].map(driver_map)
            overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
        # Filter to current drivers
        active_drivers = self.get_active_drivers()
        if not active_drivers.empty:
            current_ids = active_drivers['id'].tolist()
            track_year_analysis = track_year_analysis[track_year_analysis['driverId'].isin(current_ids)]
            overall_stats = overall_stats[overall_stats.index.isin(current_ids)]
        
        return {
            'circuit_name': circuit_name,
            'circuit_id': next_circuit_id,
            'year_by_year': track_year_analysis,
            'overall_stats': overall_stats
        }
    
    def explain_zero_or_nan_values(self, df, analysis_type):
        """Explain why certain drivers have 0 or NaN values"""
        explanations = []
        
        if analysis_type == 'sprint':
            # Check sprint race participation
            sprint_results = self.data.get('sprint_results', pd.DataFrame())
            if not sprint_results.empty:
                sprint_years = sprint_results['year'].unique() if 'year' in sprint_results.columns else []
                explanations.append(f"\nSprint races only started in 2021 (years with sprints: {sorted(sprint_years)})")
                
                # Count sprint races per year
                if 'year' in sprint_results.columns:
                    sprint_counts = sprint_results.groupby('year')['raceId'].nunique()
                    explanations.append("Sprint races per year:")
                    for year, count in sprint_counts.items():
                        explanations.append(f"  {year}: {count} sprint races")
                
                # Find drivers with 0 or NaN
                for driver in df.index:
                    if pd.isna(df.loc[driver, 'avg_sprint_points']) or df.loc[driver, 'sprint_races'] == 0:
                        driver_sprints = sprint_results[sprint_results['driverId'] == driver]
                        if driver_sprints.empty:
                            explanations.append(f"  • {driver}: Never participated in sprint races (may have raced before 2021 or after retirement)")
                        else:
                            explanations.append(f"  • {driver}: Data issue - found in sprint results but showing as 0")
        
        elif analysis_type == 'points':
            for driver in df.index:
                if df.loc[driver, 'total_points'] == 0:
                    explanations.append(f"  • {driver}: Scored 0 points in the analyzed period (finished outside points or DNF)")
        
        return "\n".join(explanations)
    
    def format_for_display(self, df):
        """Format dataframe for display - keep driverId as index and identifier"""
        if df.empty:
            return df
        
        # Return as-is, keeping driverId as the index
        return df
    
    def get_driver_names(self, driver_ids):
        """Return driver IDs as-is for consistent identification"""
        # Always use driver IDs to avoid ambiguity
        return driver_ids
    
    def calculate_first_pit_stop_time(self):
        """Calculate first pit stop duration (service time) using DHL data with detailed breakdown"""
        dhl_data = self.dhl_data.copy()
        
        if dhl_data.empty:
            return pd.DataFrame()
        
        # Map driver names to IDs (reuse logic from analyze_dhl_pit_stops)
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            # First check if driver_id column already exists
            if 'driver_id' in dhl_data.columns and dhl_data['driver_id'].notna().any():
                dhl_data['driverId'] = dhl_data['driver_id']
            else:
                # Find the driver name column - could be 'driver' or 'driver_name'
                driver_col = None
                for col in ['driver', 'driver_name', 'Driver', 'DRIVER']:
                    if col in dhl_data.columns:
                        driver_col = col
                        break
                
                if driver_col is None:
                    print(f"Error: Could not find driver column in DHL data. Available columns: {list(dhl_data.columns)}")
                    return pd.DataFrame()
                
                # Create driver mapping
                driver_mapping = {}
                for _, driver in drivers.iterrows():
                    full_name = f"{driver.get('forename', '')} {driver.get('surname', '')}".strip()
                    last_name = driver.get('surname', '')
                    driver_id = driver.get('id')
                    driver_mapping[full_name.lower()] = driver_id
                    driver_mapping[last_name.lower()] = driver_id
                
                # Special mappings - handle DHL name variations
                special_mappings = {
                    'hulkenberg': 'nico-hulkenberg', 
                    'hülkenberg': 'nico-hulkenberg',
                    'sainz': 'carlos-sainz-jr',  # DHL uses just "Sainz"
                    'perez': 'sergio-perez',      # DHL uses "Perez" without accent
                    'pérez': 'sergio-perez',
                    'magnussen': 'kevin-magnussen',
                    'leclerc': 'charles-leclerc',
                    'zhou': 'guanyu-zhou',
                    'ricciardo': 'daniel-ricciardo',
                    'verstappen': 'max-verstappen',
                    'bottas': 'valtteri-bottas',
                    'bearman': 'oliver-bearman',
                    'colapinto': 'franco-colapinto',
                    'hamilton': 'lewis-hamilton',
                    'russell': 'george-russell',
                    'alonso': 'fernando-alonso',
                    'stroll': 'lance-stroll',
                    'norris': 'lando-norris',
                    'piastri': 'oscar-piastri',
                    'gasly': 'pierre-gasly',
                    'ocon': 'esteban-ocon',
                    'tsunoda': 'yuki-tsunoda',
                    'lawson': 'liam-lawson',
                    'albon': 'alexander-albon',
                    'sargeant': 'logan-sargeant',
                    'doohan': 'jack-doohan',
                    'hadjar': 'isack-hadjar',
                    'antonelli': 'kimi-antonelli',
                    'bortoleto': 'gabriel-bortoleto'
                }
                
                def map_driver(driver_name):
                    driver_lower = driver_name.lower()
                    if driver_lower in special_mappings:
                        return special_mappings[driver_lower]
                    return driver_mapping.get(driver_lower)
                
                dhl_data['driverId'] = dhl_data[driver_col].apply(map_driver)
            
            dhl_data = dhl_data.dropna(subset=['driverId'])
        
        # Get event/race column
        group_column = 'event_id' if 'event_id' in dhl_data.columns else 'race'
        
        # Calculate first stop statistics
        first_stop_stats = pd.DataFrame()
        
        if group_column in dhl_data.columns:
            # Get first stop for each driver in each race (lowest lap number)
            first_stops = dhl_data.sort_values(['driverId', group_column, 'lap']).groupby(['driverId', group_column]).first()
            
            # Calculate statistics for first stops only
            first_stop_stats = first_stops.groupby('driverId')['time'].agg([
                ('avg_1st_stop', 'mean'),
                ('median_1st_stop', 'median'),
                ('best_1st_stop', 'min'),
                ('worst_1st_stop', 'max'),
                ('races_with_1st_stop', 'count')
            ]).round(3)
            
            # Also add the average lap number of first stop
            avg_first_lap = first_stops.groupby('driverId')['lap'].mean().round(1)
            first_stop_stats['avg_1st_stop_lap'] = avg_first_lap
        
        # Calculate overall statistics (all stops)
        overall_stats = dhl_data.groupby('driverId')['time'].agg([
            ('avg_all_stops', 'mean'),
            ('median_all_stops', 'median'),
            ('best_all_stops', 'min'),
            ('total_stops', 'count')
        ]).round(3)
        
        # Combine first stop and overall statistics
        combined_stats = first_stop_stats.join(overall_stats, how='outer')
        
        # Add driver names
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            combined_stats['driver_name'] = combined_stats.index.map(driver_map)
        
        # Calculate consistency metric for first stops
        if 'avg_1st_stop' in combined_stats.columns and 'median_1st_stop' in combined_stats.columns:
            combined_stats['1st_stop_consistency'] = (
                combined_stats['avg_1st_stop'] - combined_stats['median_1st_stop']
            ).abs().round(3)
        
        # Filter to current season drivers
        combined_stats = self.filter_current_season_drivers(combined_stats)
        
        # Sort by average first stop time
        if 'avg_1st_stop' in combined_stats.columns:
            combined_stats = combined_stats.sort_values('avg_1st_stop')
        
        return combined_stats
    
    def calculate_prizepicks_overtake_points(self):
        """Calculate PrizePicks overtake points with teammate adjustments"""
        results = self.data.get('results', pd.DataFrame()).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        drivers = self.data.get('drivers', pd.DataFrame())
        driver_standings = self.data.get('driver_standings', pd.DataFrame())
        
        if results.empty or grid.empty:
            return pd.DataFrame()
        
        # Add race year
        if not races.empty:
            results = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
        
        # Filter recent years
        recent_results = results[results['year'] >= self.current_year - 1] if 'year' in results.columns else results
        
        # Merge with grid positions
        overtake_data = recent_results.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Get constructor info for teammate detection
        if 'constructorId' in overtake_data.columns and not driver_standings.empty:
            # Get current constructor mappings
            current_constructors = driver_standings[driver_standings['year'] == self.current_year][['driverId', 'constructorId']].drop_duplicates()
            constructor_map = dict(zip(current_constructors['driverId'], current_constructors['constructorId']))
            overtake_data['current_constructorId'] = overtake_data['driverId'].map(constructor_map)
        
        # Calculate base overtake points
        overtake_data['base_overtake_points'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
        
        # Identify teammate overtakes
        if 'constructorId' in overtake_data.columns:
            # Group by race and constructor to find teammates
            for race_id in overtake_data['raceId'].unique():
                race_data = overtake_data[overtake_data['raceId'] == race_id]
                
                for constructor_id in race_data['constructorId'].unique():
                    if pd.notna(constructor_id):
                        constructor_drivers = race_data[race_data['constructorId'] == constructor_id]
                        
                        if len(constructor_drivers) == 2:
                            driver1, driver2 = constructor_drivers['driverId'].values[:2]
                            pos1_start = constructor_drivers[constructor_drivers['driverId'] == driver1]['gridPosition'].values[0]
                            pos1_end = constructor_drivers[constructor_drivers['driverId'] == driver1]['positionNumber'].values[0]
                            pos2_start = constructor_drivers[constructor_drivers['driverId'] == driver2]['gridPosition'].values[0]
                            pos2_end = constructor_drivers[constructor_drivers['driverId'] == driver2]['positionNumber'].values[0]
                            
                            # Check if they swapped positions
                            if pd.notna(pos1_start) and pd.notna(pos1_end) and pd.notna(pos2_start) and pd.notna(pos2_end):
                                if (pos1_start > pos2_start and pos1_end < pos2_end):
                                    # Driver 1 overtook teammate
                                    overtake_data.loc[(overtake_data['raceId'] == race_id) & 
                                                    (overtake_data['driverId'] == driver1), 'teammate_bonus'] = 0.5
                                    overtake_data.loc[(overtake_data['raceId'] == race_id) & 
                                                    (overtake_data['driverId'] == driver2), 'teammate_penalty'] = -0.5
                                elif (pos2_start > pos1_start and pos2_end < pos1_end):
                                    # Driver 2 overtook teammate
                                    overtake_data.loc[(overtake_data['raceId'] == race_id) & 
                                                    (overtake_data['driverId'] == driver2), 'teammate_bonus'] = 0.5
                                    overtake_data.loc[(overtake_data['raceId'] == race_id) & 
                                                    (overtake_data['driverId'] == driver1), 'teammate_penalty'] = -0.5
        
        # Calculate final PrizePicks overtake points
        overtake_data['teammate_bonus'] = overtake_data.get('teammate_bonus', 0).fillna(0)
        overtake_data['teammate_penalty'] = overtake_data.get('teammate_penalty', 0).fillna(0)
        overtake_data['prizepicks_overtake_points'] = (overtake_data['base_overtake_points'] + 
                                                       overtake_data['teammate_bonus'] + 
                                                       overtake_data['teammate_penalty'])
        
        # Aggregate by driver
        driver_overtake_points = overtake_data.groupby('driverId').agg({
            'prizepicks_overtake_points': ['mean', 'median', 'sum', 'std'],
            'base_overtake_points': ['mean', 'sum'],
            'teammate_bonus': 'sum',
            'teammate_penalty': 'sum',
            'raceId': 'count'
        }).round(2)
        
        driver_overtake_points.columns = ['avg_pp_overtake_pts', 'median_pp_overtake_pts', 
                                         'total_pp_overtake_pts', 'std_pp_overtake_pts',
                                         'avg_base_overtake_pts', 'total_base_overtake_pts',
                                         'total_teammate_bonuses', 'total_teammate_penalties',
                                         'races_analyzed']
        
        # Filter to current season drivers
        driver_overtake_points = self.filter_current_season_drivers(driver_overtake_points)
        
        # Add driver names
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            driver_overtake_points['driver_name'] = driver_overtake_points.index.map(driver_map)
        
        return driver_overtake_points
    
    def calculate_win_podium_probabilities(self):
        """Calculate win and podium probabilities based on historical performance"""
        results = self.data.get('results', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty:
            return pd.DataFrame()
        
        # Add year info
        if not races.empty:
            results = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
        
        # Filter recent years
        recent_results = results[results['year'] >= self.current_year - 2] if 'year' in results.columns else results
        
        # Calculate probabilities
        driver_probs = recent_results.groupby('driverId').agg({
            'positionNumber': [
                lambda x: (x == 1).sum(),  # wins
                lambda x: (x <= 3).sum(),  # podiums
                lambda x: (x <= 10).sum(), # points finishes
                'count'                    # total races
            ]
        })
        
        driver_probs.columns = ['wins', 'podiums', 'points_finishes', 'total_races']
        
        # Calculate probabilities
        driver_probs['win_probability'] = (driver_probs['wins'] / driver_probs['total_races'] * 100).round(1)
        driver_probs['podium_probability'] = (driver_probs['podiums'] / driver_probs['total_races'] * 100).round(1)
        driver_probs['points_probability'] = (driver_probs['points_finishes'] / driver_probs['total_races'] * 100).round(1)
        
        # Calculate expected points per race
        points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        
        def calc_expected_points(driver_results):
            total_points = 0
            for pos in driver_results:
                if pd.notna(pos) and pos <= 10:
                    total_points += points_map.get(int(pos), 0)
            return total_points / len(driver_results) if len(driver_results) > 0 else 0
        
        expected_points = recent_results.groupby('driverId')['positionNumber'].apply(calc_expected_points)
        driver_probs['expected_points_per_race'] = expected_points.round(2)
        
        # DNF probability
        driver_probs['dnf_count'] = recent_results.groupby('driverId')['statusId'].apply(
            lambda x: (x > 1).sum()  # statusId > 1 usually means DNF
        )
        driver_probs['dnf_probability'] = (driver_probs['dnf_count'] / driver_probs['total_races'] * 100).round(1)
        
        # Filter to current season drivers
        driver_probs = self.filter_current_season_drivers(driver_probs)
        
        # Add driver names
        drivers = self.data.get('drivers', pd.DataFrame())
        if not drivers.empty:
            driver_map = dict(zip(drivers['id'], drivers['name']))
            driver_probs['driver_name'] = driver_probs.index.map(driver_map)
        
        return driver_probs
    
    def generate_all_tables(self):
        """Generate all performance analysis tables"""
        print("\n" + "="*80)
        print("F1 DRIVER PERFORMANCE ANALYSIS")
        print("="*80)
        print(f"Analysis Date: {self.current_date.strftime('%Y-%m-%d')}")
        print(f"Current Season: {self.current_year}")
        print(f"Years Analyzed: {self.current_year-3} to {self.current_year}")
        print(f"Note: Showing only drivers who participated in {self.current_year}")
        
        # Show current season drivers
        current_drivers = self.get_active_drivers()
        if not current_drivers.empty:
            print(f"\nActive drivers in {self.current_year}: {len(current_drivers)}")
            driver_list = ', '.join(current_drivers['id'].tolist())
            print(f"Driver IDs: {driver_list}")
        print(f"Current Analysis Year: {self.current_year}")
        print(f"Years Analyzed: {self.current_year-3} to {self.current_year}")
        
        # Get race information
        races = self.data.get('races', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if not races.empty and not results.empty:
            races_with_results = results['raceId'].unique()
            recent_races = races[races['id'].isin(races_with_results)]
            
            if not recent_races.empty:
                # Ensure date is datetime
                recent_races = recent_races.copy()
                recent_races['date'] = pd.to_datetime(recent_races['date'])
                recent_races_sorted = recent_races.sort_values('date', ascending=False)
                
                # Current race (most recent with results - Hungarian GP)
                if len(recent_races_sorted) > 0:
                    current_race = recent_races_sorted.iloc[0]
                    current_circuit_name = 'Unknown'
                    if not circuits.empty and 'circuitId' in current_race:
                        circuit = circuits[circuits['id'] == current_race['circuitId']]
                        if not circuit.empty:
                            current_circuit_name = circuit.iloc[0]['name']
                    
                    print(f"\nCurrent Race: {current_race.get('officialName', 'Unknown')} at {current_circuit_name}")
                    print(f"Date: {current_race['date'].strftime('%Y-%m-%d') if pd.notna(current_race.get('date')) else 'Unknown'}")
                
                # Previous race (second most recent - Belgian GP)
                if len(recent_races_sorted) > 1:
                    prev_race = recent_races_sorted.iloc[1]
                    prev_circuit_name = 'Unknown'
                    if not circuits.empty and 'circuitId' in prev_race:
                        circuit = circuits[circuits['id'] == prev_race['circuitId']]
                        if not circuit.empty:
                            prev_circuit_name = circuit.iloc[0]['name']
                    
                    print(f"\nPrevious Race: {prev_race.get('officialName', 'Unknown')} at {prev_circuit_name}")
                    print(f"Date: {prev_race['date'].strftime('%Y-%m-%d') if pd.notna(prev_race.get('date')) else 'Unknown'}")
            
                    # Show example result - Hamilton's performance in previous race
                    drivers = self.data.get('drivers', pd.DataFrame())
                    grid = self.data.get('races_starting_grid_positions', pd.DataFrame())
                    
                    if not drivers.empty and not grid.empty and 'prev_race' in locals():
                        # Get Hamilton's ID
                        hamilton = drivers[drivers['id'] == 'lewis-hamilton']
                        if not hamilton.empty:
                            prev_results = results[results['raceId'] == prev_race['id']]
                            prev_grid = grid[grid['raceId'] == prev_race['id']]
                            
                            hamilton_result = prev_results[prev_results['driverId'] == 'lewis-hamilton']
                            hamilton_grid = prev_grid[prev_grid['driverId'] == 'lewis-hamilton']
                            
                            if not hamilton_result.empty and not hamilton_grid.empty:
                                start_pos = hamilton_grid.iloc[0]['positionNumber']
                                start_text = hamilton_grid.iloc[0].get('positionText', '')
                                end_pos = hamilton_result.iloc[0]['positionNumber']
                                
                                # Handle pit lane starts
                                if pd.isna(start_pos) or start_text == 'PL':
                                    # For pit lane starts, count from back of grid (20th)
                                    start_pos = 20
                                    start_display = "PL (20)"
                                else:
                                    start_display = f"{int(start_pos)}"
                                
                                # Calculate positions gained
                                if pd.notna(start_pos) and pd.notna(end_pos):
                                    positions_gained = start_pos - end_pos
                                    overtakes = max(0, positions_gained)  # Current calculation method
                                    print(f"Example: Hamilton started P{start_display}, finished P{int(end_pos)} (gained {int(positions_gained)} positions)")
                                    print(f"         Current overtake calculation: {overtakes} (only counts forward progress)")
                                else:
                                    print(f"Example: Hamilton started P{start_display}, finished P{end_pos} (unable to calculate positions gained)")
                                
                                # Show all drivers' performance for verification
                                print("\n         All drivers in previous race:")
                                prev_data = prev_results.merge(
                                    prev_grid[['raceId', 'driverId', 'positionNumber', 'positionText']].rename(
                                        columns={'positionNumber': 'gridPosition', 'positionText': 'gridText'}
                                    ),
                                    on=['raceId', 'driverId'],
                                    how='left'
                                )
                                
                                # Handle pit lane starts - assign position 20 for calculation
                                prev_data['grid_calc'] = prev_data.apply(
                                    lambda x: 20 if (pd.isna(x['gridPosition']) or x['gridText'] == 'PL') else x['gridPosition'],
                                    axis=1
                                )
                                prev_data['grid_display'] = prev_data.apply(
                                    lambda x: 'PL' if (pd.isna(x['gridPosition']) or x['gridText'] == 'PL') else f"{int(x['gridPosition'])}",
                                    axis=1
                                )
                                
                                prev_data['positions_gained'] = prev_data['grid_calc'] - prev_data['positionNumber']
                                prev_data['overtakes'] = prev_data['positions_gained'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
                                
                                # Add driver names
                                driver_map = dict(zip(drivers['id'], drivers['name']))
                                prev_data['driver_name'] = prev_data['driverId'].map(driver_map)
                                
                                # Sort by positions gained and filter valid results
                                prev_data_valid = prev_data[pd.notna(prev_data['positionNumber'])].copy()
                                prev_data_sorted = prev_data_valid.sort_values('positions_gained', ascending=False)
                                for _, row in prev_data_sorted.head(10).iterrows():
                                    print(f"         {row['driver_name']}: P{row['grid_display']} → P{int(row['positionNumber'])} "
                                          f"(gained {int(row['positions_gained'])} positions, {int(row['overtakes'])} overtakes)")
        
        next_race = self.get_next_race()
        if next_race is not None:
            circuits = self.data.get('circuits', pd.DataFrame())
            circuit_name = 'Unknown'
            if not circuits.empty and 'circuitId' in next_race:
                circuit = circuits[circuits['id'] == next_race['circuitId']]
                if not circuit.empty:
                    circuit_name = circuit.iloc[0]['name']
            
            print(f"\nNext Race: {next_race.get('officialName', 'Unknown')} at {circuit_name}")
            print(f"Date: {next_race['date'].strftime('%Y-%m-%d') if pd.notna(next_race.get('date')) else 'Unknown'}")
        
        # Add current and previous race info
        races = self.data.get('races', pd.DataFrame())
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame()))
        if not races.empty and not results.empty:
            races_with_results = results['raceId'].unique()
            recent_races_sorted = races[races['id'].isin(races_with_results)].sort_values('date', ascending=False)
            
            if len(recent_races_sorted) >= 1:
                # Current race (most recent with results)
                current_race = recent_races_sorted.iloc[0]
                print(f"\nCurrent Race (most recent with results): {current_race.get('officialName', 'Unknown')}")
                print(f"Date: {current_race['date'].strftime('%Y-%m-%d') if pd.notna(current_race.get('date')) else 'Unknown'}")
                
            if len(recent_races_sorted) >= 2:
                # Previous race
                prev_race = recent_races_sorted.iloc[1]
                print(f"\nPrevious Race: {prev_race.get('officialName', 'Unknown')}")
                print(f"Date: {prev_race['date'].strftime('%Y-%m-%d') if pd.notna(prev_race.get('date')) else 'Unknown'}")
        
        # 1. Overtakes Analysis
        print("\n" + "="*80)
        if self.lap_analyzer is not None:
            print("1. REAL OVERTAKES BY DRIVER (Lap-by-Lap Analysis)")
        else:
            print("1. POSITIONS GAINED BY DRIVER")
        print("="*80)
        overtakes = self.analyze_overtakes()
        if not overtakes.empty:
            # Note: Column order is already set in analyze_overtakes method
            print(overtakes.to_string(index=False))
            
            print("\nColumn Explanations:")
            if self.lap_analyzer is not None:
                print("- total_OTs: Total overtakes made across all races")
                print("- avg_OTs: Average real overtakes made per race (lap-by-lap analysis)")
                print("- median_OTs: Median overtakes made per race")
                print("- last_race_OTs: Overtakes made in the most recent race")
                print("- c_race_OTs: Historical average overtakes at current race circuit")
                print("- n_race_OTs: Historical average overtakes at next race circuit")
                print("- OTd_by: Total times overtaken by other drivers")
                print("- net_OTs: Net overtakes (overtakes made - times overtaken)")
                print("- max_OTs: Maximum overtakes in a single race")
                print("\nNote: This shows ACTUAL on-track overtakes from lap-by-lap data")
            else:
                print("- avg_pos_gained: Average positions gained per race (grid - finish, positive = gained positions)")
                print("- median_pos_gained: Median positions gained per race")
                print("- last_race: Positions gained in the most recent race")
                print("- p_circuit_avg: Average positions gained at previous race circuit")
                print("- n_circuit_avg: Average positions gained at next race circuit")
                print("- n_circuit_pts: Position-based points at next circuit (grid - finish + teammate adjustments)")
                print("\nNote: This shows NET positions gained, not actual on-track overtakes")
            
            # Show drivers with notable statistics
            print("\nNotable insights:")
            
            if self.lap_analyzer is not None:
                # Real overtakes analysis
                if 'avg_OTs' in overtakes.columns:
                    top_overtakers = overtakes.nlargest(5, 'avg_OTs')
                    if 'driver_name' in overtakes.columns:
                        top_names = [overtakes.loc[idx, 'driver_name'] for idx in top_overtakers.index if idx in overtakes.index]
                        print(f"Top 5 overtakers (by avg/race): {', '.join(top_names)}")
                    
                    # Show total overtakes too
                    if 'total_OTs' in overtakes.columns:
                        total_ots = overtakes['total_OTs'].sum()
                        print(f"Total overtakes recorded: {int(total_ots)}")
                    
                    # Show best net overtakers
                    if 'net_OTs' in overtakes.columns:
                        best_net = overtakes.nlargest(3, 'net_OTs')
                        if 'driver_name' in overtakes.columns:
                            net_names = [overtakes.loc[idx, 'driver_name'] for idx in best_net.index if idx in overtakes.index]
                            print(f"Best net overtakers: {', '.join(net_names)}")
            else:
                # Position-based analysis
                if 'avg_pos_gained' in overtakes.columns:
                    top_gainers = overtakes.nlargest(5, 'avg_pos_gained')
                    if 'driver_name' in overtakes.columns:
                        top_names = [overtakes.loc[idx, 'driver_name'] for idx in top_gainers.index if idx in overtakes.index]
                        print(f"Top 5 position gainers: {', '.join(top_names)}")
                    else:
                        print(f"Top 5 position gainers: {', '.join(top_gainers.index.tolist())}")
                    
                    # For position-based analysis, show negative performers
                    lost_positions = overtakes[overtakes['avg_pos_gained'] < 0]
                    if not lost_positions.empty:
                        if 'driver_name' in overtakes.columns:
                            lost_names = [overtakes.loc[idx, 'driver_name'] for idx in lost_positions.index if idx in overtakes.index]
                            print(f"Drivers who typically lost positions: {', '.join(lost_names)}")
                        else:
                            print(f"Drivers who typically lost positions: {', '.join(lost_positions.index.tolist())}")
                else:
                    # For real overtakes, show total overtakes
                    if 'total_pos_gained' in overtakes.columns:
                        total_overtakes = overtakes['total_pos_gained'].sum()
                        print(f"Total overtakes recorded: {int(total_overtakes)}")
                        if 'net_overtakes' in overtakes.columns:
                            top_net = overtakes.nlargest(3, 'net_overtakes')
                            if 'driver_name' in overtakes.columns:
                                net_names = [overtakes.loc[idx, 'driver_name'] for idx in top_net.index if idx in overtakes.index]
                                print(f"Best net overtakers: {', '.join(net_names)}")
        else:
            print("No overtake data available")
        
        # 1a. Fantasy Overtakes Analysis - RACE
        print("\n" + "-"*80)
        print("1a. F1 FANTASY RACE OVERTAKE STATISTICS")
        print("-"*80)
        fantasy_overtakes = self.analyze_fantasy_overtakes()
        if not fantasy_overtakes.empty:
            # Race overtakes section
            race_cols = ['driver_name', 'total_races', 'total_overtake_points']
            race_overtake_cols = [col for col in fantasy_overtakes.columns if 'race_overtake' in col and '_avg_pts' in col]
            race_count_cols = [col for col in fantasy_overtakes.columns if 'race_overtake' in col and '_avg_count' in col]
            
            # Add race columns
            race_cols.extend(race_overtake_cols)
            race_cols.extend(race_count_cols)
            
            # Filter to existing columns
            race_cols = [col for col in race_cols if col in fantasy_overtakes.columns]
            
            # Sort by total overtake points for better readability
            fantasy_overtakes_sorted = fantasy_overtakes.sort_values('total_overtake_points', ascending=False)
            
            print("\nRace Overtake Points Summary:")
            print(fantasy_overtakes_sorted[race_cols].head(20).to_string(index=False))
            
            print("\nColumn Explanations:")
            print("- total_overtake_points: Total fantasy points from all overtaking (race + sprint)")
            print("- race_overtake_bonus_avg_pts: Average fantasy points per race from overtaking")
            print("- race_overtake_bonus_circuit_avg_pts: Average at the next race circuit")
            print("- race_overtake_bonus_avg_count: Average number of overtakes per race")
            
        # 1b. Fantasy Overtakes Analysis - SPRINT
        print("\n" + "-"*80)
        print("1b. F1 FANTASY SPRINT OVERTAKE STATISTICS")
        print("-"*80)
        if not fantasy_overtakes.empty:
            # Sprint overtakes section
            sprint_cols = ['driver_name', 'total_races']
            sprint_overtake_cols = [col for col in fantasy_overtakes.columns if ('sprint_overtake' in col or 'sprint_OTs' in col) and '_avg_pts' in col]
            sprint_count_cols = [col for col in fantasy_overtakes.columns if ('sprint_overtake' in col or 'sprint_OTs' in col) and '_avg_count' in col]
            
            # Add sprint columns
            sprint_cols.extend(sprint_overtake_cols)
            sprint_cols.extend(sprint_count_cols)
            
            # Filter to existing columns
            sprint_cols = [col for col in sprint_cols if col in fantasy_overtakes.columns]
            
            # Filter to drivers with sprint overtake data
            # Check for both old and new column names
            sprint_pts_col = 'sprint_OTs_avg_pts' if 'sprint_OTs_avg_pts' in fantasy_overtakes.columns else 'sprint_overtake_bonus_avg_pts'
            has_sprint_data = fantasy_overtakes[fantasy_overtakes[sprint_pts_col] > 0] if sprint_pts_col in fantasy_overtakes.columns else fantasy_overtakes
            
            if not has_sprint_data.empty:
                print("\nSprint Overtake Points Summary:")
                print(has_sprint_data.sort_values(sprint_pts_col, ascending=False)[sprint_cols].head(15).to_string(index=False))
                
                print("\nColumn Explanations:")
                print("- sprint_OTs_avg_pts: Average fantasy points per race from sprint overtaking")
                print("- sprint_OTs_circuit_avg_pts: Average at the next race circuit")
                print("- sprint_OTs_avg_count: Average number of sprint overtakes per race")
            else:
                print("No sprint overtake data available")
            
            # Notable insights
            print("\nNotable Fantasy Overtaking Insights:")
            if 'total_overtake_points' in fantasy_overtakes.columns:
                top_fantasy_overtakers = fantasy_overtakes.nlargest(5, 'total_overtake_points')
                print(f"Top 5 fantasy overtakers by total points: {', '.join(top_fantasy_overtakers['driver_name'].tolist())}")
            
            # Compare with position-based overtakes if both available
            if not overtakes.empty and 'driver_name' in overtakes.columns:
                print("\nComparison: Position-based vs Fantasy overtakes:")
                # Find drivers who excel in fantasy but not position-based
                fantasy_leaders = set(fantasy_overtakes.nlargest(10, 'total_overtake_points')['driver_name'].tolist())
                # Use appropriate column name based on whether we have real overtakes or position-based
                overtake_col = 'avg_OTs' if self.lap_analyzer is not None else 'avg_pos_gained'
                if overtake_col in overtakes.columns:
                    position_leaders = set(overtakes.nlargest(10, overtake_col)['driver_name'].tolist())
                else:
                    position_leaders = set()
                
                fantasy_only = fantasy_leaders - position_leaders
                if fantasy_only:
                    print(f"Strong in fantasy overtakes but not position gains: {', '.join(fantasy_only)}")
                
                position_only = position_leaders - fantasy_leaders
                if position_only:
                    print(f"Strong in position gains but not fantasy overtakes: {', '.join(position_only)}")
        else:
            print("No F1 Fantasy overtake data available")
        
        # 1b. Track-specific positions gained by year
        print("\n" + "-"*80)
        print("1b. TRACK-SPECIFIC POSITIONS GAINED BY YEAR")
        print("-"*80)
        track_year_overtakes = self.analyze_overtakes_by_track_year()
        if isinstance(track_year_overtakes, dict) and 'year_by_year' in track_year_overtakes:
            circuit_name = track_year_overtakes.get('circuit_name', 'Unknown Circuit')
            print(f"\nAnalysis for: {circuit_name}")
            
            # Show year-by-year data in matrix format
            year_data = track_year_overtakes['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Performance Matrix:")
                print("=" * 150)
                
                # Create a more compact matrix showing only recent years and key metrics
                year_data_sorted = year_data.sort_values(['driver_name', 'year'])
                
                # Get unique years - limit to recent years for readability
                all_years = sorted(year_data_sorted['year'].unique())
                recent_years = all_years[-5:] if len(all_years) > 5 else all_years  # Last 5 years
                
                # Header with years
                header = f"{'Driver':<25}"
                for year in recent_years:
                    header += f"{int(year):>20}"
                print(header)
                
                # Sub-header with metrics
                sub_header = f"{'':<25}"
                for year in recent_years:
                    sub_header += f"{'Start→Fin    +/-':>20}"
                print(sub_header)
                print("-" * (25 + len(recent_years) * 20))
                
                # Sort drivers by their average overtakes in the most recent year they raced
                driver_sort_data = []
                for driver_id in year_data_sorted['driverId'].unique():
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    # Get their most recent year's average overtakes
                    recent_data = driver_data[driver_data['year'].isin(recent_years)]
                    if not recent_data.empty:
                        most_recent = recent_data.nlargest(1, 'year').iloc[0]
                        driver_sort_data.append((driver_id, most_recent['avg_OT']))
                    else:
                        driver_sort_data.append((driver_id, 0))
                
                # Sort by average overtakes descending
                driver_sort_data.sort(key=lambda x: x[1], reverse=True)
                
                # Display each driver's data
                for driver_id, _ in driver_sort_data:
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    
                    # Truncate long names
                    if len(driver_name) > 24:
                        driver_name = driver_name[:21] + "..."
                    
                    row_str = f"{driver_name:<25}"
                    
                    # For each recent year
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            # Format: Start→Finish  +/-
                            start_pos = f"{row['avg_start_pos']:.0f}"
                            finish_pos = f"{row['avg_finish_pos']:.0f}"
                            pos_change = f"{row['avg_pos_change']:+.1f}"
                            row_str += f"{start_pos:>6}→{finish_pos:<6} {pos_change:>6}  "
                        else:
                            row_str += f"{'—':^20}"
                    
                    print(row_str)
                
                # Add a note about full history
                if len(all_years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history available from {min(all_years)} to {max(all_years)}")
            
            # Show overall career stats at this track
            overall_stats = track_year_overtakes['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer Statistics at this Track:")
                print("-" * 100)
                print(f"{'Driver':<25} {'Races':<8} {'Total OT':<10} {'Avg OT':<8} {'Med OT':<8} "
                      f"{'Avg Start':<10} {'Avg Finish':<12} {'Avg Change':<10}")
                print("-" * 100)
                
                # Sort by average overtakes descending
                overall_sorted = overall_stats.sort_values('career_avg_OT', ascending=False)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['races_at_track']:<8.0f} "
                          f"{row['career_OT']:<10.0f} {row['career_avg_OT']:<8.1f} "
                          f"{row['career_median_OT']:<8.0f} {row['career_avg_start']:<10.1f} "
                          f"{row['career_avg_finish']:<12.1f} {row['career_avg_pos_change']:+10.1f}")
        else:
            print("No track-specific overtake data available for the next race")
        
        # 2. Points Analysis
        print("\n" + "="*80)
        print("2. F1 POINTS BY DRIVER")
        print("="*80)
        points = self.analyze_points()
        if not points.empty:
            # Note: Column order is already set in analyze_points method
            print(points.to_string(index=False))
            
            # Add explanations
            print("\nColumn Definitions:")
            print("- avg_points: Average points per race in current season (2025)")
            print("- hist_avg_points: Historical average points per race (2022-2024)")
            print("- last_race: Points scored in the most recent race")
            print("- p_circuit_avg: Average points at the previous race's circuit (3-year history)")
            print("- c_circuit_avg: Average points at the current race's circuit (3-year history)")
            print("- n_circuit_avg: Average points at the next race's circuit (3-year history)")
            print("\nNote: Circuit averages are calculated from the last 3 years of racing at each specific track")
        else:
            print("No points data available")
        
        # 2b. Track-specific points by year
        print("\n" + "-"*80)
        print("2b. TRACK-SPECIFIC POINTS BY YEAR")
        print("-"*80)
        track_year_points = self.analyze_points_by_track_year()
        if isinstance(track_year_points, dict) and 'year_by_year' in track_year_points:
            circuit_name = track_year_points.get('circuit_name', 'Unknown Circuit')
            print(f"\nAnalysis for: {circuit_name}")
            
            # Show year-by-year data in matrix format
            year_data = track_year_points['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Points Matrix:")
                print("=" * 120)
                
                # Create a compact matrix
                year_data_sorted = year_data.sort_values(['driver_name', 'year'])
                
                # Get unique years - limit to recent years
                all_years = sorted(year_data_sorted['year'].unique())
                recent_years = all_years[-5:] if len(all_years) > 5 else all_years
                
                # Header with years
                header = f"{'Driver':<25}"
                for year in recent_years:
                    header += f"{int(year):>20}"
                print(header)
                
                # Sub-header
                sub_header = f"{'':<25}"
                for year in recent_years:
                    sub_header += f"{'Pts  Avg  Best':>20}"
                print(sub_header)
                print("-" * (25 + len(recent_years) * 20))
                
                # Sort drivers by total points in most recent year
                driver_sort_data = []
                for driver_id in year_data_sorted['driverId'].unique():
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    recent_data = driver_data[driver_data['year'].isin(recent_years)]
                    if not recent_data.empty:
                        most_recent = recent_data.nlargest(1, 'year').iloc[0]
                        driver_sort_data.append((driver_id, most_recent.get('total_points', 0)))
                    else:
                        driver_sort_data.append((driver_id, 0))
                
                driver_sort_data.sort(key=lambda x: x[1], reverse=True)
                
                # Display each driver's data
                for driver_id, _ in driver_sort_data:
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    
                    if len(driver_name) > 24:
                        driver_name = driver_name[:21] + "..."
                    
                    row_str = f"{driver_name:<25}"
                    
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            total_pts = f"{row['total_points']:.0f}"
                            avg_pts = f"{row['avg_points']:.1f}"
                            best_fin = f"{row['best_finish']:.0f}"
                            row_str += f"{total_pts:>4} {avg_pts:>5} {best_fin:>5}     "
                        else:
                            row_str += f"{'—':^20}"
                    
                    print(row_str)
                
                if len(all_years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history from {min(all_years)} to {max(all_years)}")
            
            # Show career stats
            overall_stats = track_year_points['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer Statistics at this Track:")
                print("-" * 100)
                print(f"{'Driver':<25} {'Races':<8} {'Total Pts':<10} {'Avg Pts':<10} "
                      f"{'Best Fin':<10} {'Avg Fin':<10}")
                print("-" * 100)
                
                overall_sorted = overall_stats.sort_values('career_points', ascending=False)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['races_at_track']:<8.0f} "
                          f"{row['career_points']:<10.0f} {row['career_avg_points']:<10.1f} "
                          f"{row['career_best_finish']:<10.0f} {row['career_avg_finish']:<10.1f}")
        else:
            print("No track-specific points data available for the next race")
        
        # 3. Pit Stop Analysis
        print("\n" + "="*80)
        print("3. PIT STOP TIMES BY DRIVER (seconds)")
        print("="*80)
        pit_stops = self.analyze_pit_stops()
        if not pit_stops.empty:
            # Basic columns
            display_cols = ['driver_name', 'avg_stop_time', 'median_stop_time', 'best_stop_time', 'total_stops']
            
            # Add pit lane delta columns if available
            if 'avg_pit_lane_time' in pit_stops.columns:
                display_cols.extend(['avg_pit_lane_time', 'median_pit_lane_time', 'delta_samples'])
            
            # Add circuit-specific columns
            if 'next_circuit_avg' in pit_stops.columns:
                display_cols.append('next_circuit_avg')
            if 'next_circuit_pit_lane_time' in pit_stops.columns:
                display_cols.append('next_circuit_pit_lane_time')
                
            # Remove non-existent columns
            display_cols = [col for col in display_cols if col in pit_stops.columns]
            
            # Filter out rows where all numeric columns are NaN
            numeric_cols = ['avg_stop_time', 'median_stop_time', 'best_stop_time', 'total_stops']
            numeric_cols = [col for col in numeric_cols if col in pit_stops.columns]
            pit_stops_filtered = pit_stops.dropna(subset=numeric_cols, how='all')
            
            if not pit_stops_filtered.empty:
                print("F1DB Pit Stop Times (Total time from pit entry to exit):")
                print(pit_stops_filtered[display_cols].to_string(index=False))
                
                # Add explanations for new columns
                if 'avg_pit_lane_time' in display_cols:
                    print("\nPit Lane Delta Analysis (F1DB total time - DHL service time):")
                    print("- avg_pit_lane_time: Average time spent entering/exiting pit lane (seconds)")
                    print("- median_pit_lane_time: Median pit lane entry/exit time")
                    print("- delta_samples: Number of matched pit stops between F1DB and DHL data")
                    print("- next_circuit_pit_lane_time: Historical pit lane time at next race circuit")
                    
                    # Show insights
                    if 'avg_pit_lane_time' in pit_stops_filtered.columns:
                        avg_delta = pit_stops_filtered['avg_pit_lane_time'].mean()
                        if pd.notna(avg_delta):
                            print(f"\nAverage pit lane entry/exit time across all drivers: {avg_delta:.1f} seconds")
            else:
                print("No valid pit stop data available")
            
            # Add pit stop explanations
            if not pit_stops.empty:
                explanations = self.explain_zero_or_nan_values(pit_stops, 'pit_stops')
                print(explanations)
        else:
            print("No pit stop data available")
        
        # 3b. DHL Official Pit Stop Analysis
        print("\n" + "-"*80)
        print("3b. DHL OFFICIAL PIT STOP TIMES BY DRIVER (seconds)")
        print("-"*80)
        dhl_stops = self.analyze_dhl_pit_stops()
        if not dhl_stops.empty:
            # Add driver names
            drivers = self.data.get('drivers', pd.DataFrame())
            if not drivers.empty:
                driver_names = drivers.set_index('id')['name'].to_dict()
                dhl_stops['driver_name'] = dhl_stops.index.map(driver_names)
            
            # Show main analysis with first stop info
            display_cols = ['driver_name', 'avg_time', 'median_time', 'best_time', 'best_time_lap', 'total_stops']
            if 'avg_first_stop' in dhl_stops.columns:
                display_cols.extend(['avg_first_stop', 'best_first_stop'])
            if 'next_circuit_avg' in dhl_stops.columns:
                display_cols.append('next_circuit_avg')
            if 'next_circuit_first_stop' in dhl_stops.columns:
                display_cols.append('next_circuit_first_stop')
            display_cols = [col for col in display_cols if col in dhl_stops.columns]
            
            # Sort by average time (fastest first)
            dhl_stops_sorted = dhl_stops.sort_values('avg_time', ascending=True)
            
            # Filter out rows where all numeric columns are NaN
            numeric_cols = ['avg_time', 'median_time', 'best_time', 'total_stops']
            if 'avg_first_stop' in dhl_stops.columns:
                numeric_cols.extend(['avg_first_stop', 'best_first_stop'])
            numeric_cols = [col for col in numeric_cols if col in dhl_stops.columns]
            dhl_stops_filtered = dhl_stops_sorted.dropna(subset=numeric_cols, how='all')
            
            if not dhl_stops_filtered.empty:
                print(dhl_stops_filtered[display_cols].to_string(index=False))
            else:
                print("No valid DHL pit stop data available")
            
            # Show yearly analysis if available
            if hasattr(dhl_stops, 'yearly_data') and not dhl_stops.yearly_data.empty:
                print("\nYearly DHL Pit Stop Performance:")
                print("-" * 80)
                yearly_data = dhl_stops.yearly_data
                
                # Add driver names to yearly data
                if not drivers.empty:
                    yearly_data['driver_name'] = yearly_data['driverId'].map(driver_names)
                
                # Create pivot table for easy viewing
                years = sorted(yearly_data['year'].unique())
                
                if len(years) > 1:
                    print(f"{'Driver':<20}", end="")
                    for year in years:
                        print(f"{int(year):>25}", end="")
                    print()
                    
                    print(f"{'':<20}", end="")
                    for year in years:
                        print(f"{'Avg  Med  Best  Stops':>25}", end="")
                    print()
                    print("-" * (20 + len(years) * 25))
                    
                    # Sort drivers by most recent year average
                    driver_sort = []
                    for driver_id in yearly_data['driverId'].unique():
                        driver_years = yearly_data[yearly_data['driverId'] == driver_id]
                        most_recent = driver_years[driver_years['year'] == max(driver_years['year'])]
                        if not most_recent.empty:
                            avg_time = most_recent.iloc[0]['avg_time']
                            driver_sort.append((driver_id, avg_time))
                    
                    driver_sort.sort(key=lambda x: x[1])  # Sort by avg time
                    
                    for driver_id, _ in driver_sort[:15]:  # Top 15 drivers
                        driver_years = yearly_data[yearly_data['driverId'] == driver_id]
                        driver_name = driver_years.iloc[0].get('driver_name', str(driver_id))
                        
                        if len(driver_name) > 19:
                            driver_name = driver_name[:16] + "..."
                        
                        print(f"{driver_name:<20}", end="")
                        
                        for year in years:
                            year_data = driver_years[driver_years['year'] == year]
                            if not year_data.empty:
                                row = year_data.iloc[0]
                                avg = f"{row['avg_time']:.2f}"
                                med = f"{row['median_time']:.2f}"
                                best = f"{row['best_time']:.2f}"
                                stops = f"{int(row['total_stops'])}"
                                print(f"{avg:>5} {med:>5} {best:>5} {stops:>5}", end="")
                            else:
                                print(f"{'---':>5} {'---':>5} {'---':>5} {'---':>5}", end="")
                        print()
                else:
                    # Single year view
                    year = years[0]
                    print(f"Data available for {int(year)} only:")
                    year_summary = yearly_data.groupby('driverId').agg({
                        'avg_time': 'first',
                        'total_stops': 'sum'
                    }).sort_values('avg_time')
                    
                    if not drivers.empty:
                        year_summary['driver_name'] = year_summary.index.map(driver_names)
                    
                    print(year_summary[['driver_name', 'avg_time', 'total_stops']].to_string(index=False))
        else:
            print("No DHL pit stop data available")
        
        # 3c. Track-specific DHL First Stop Analysis
        print("\n" + "-"*80)
        print("3c. TRACK-SPECIFIC DHL FIRST STOP ANALYSIS")
        print("-"*80)
        track_dhl = self.analyze_dhl_pit_stops_by_track_year()
        if isinstance(track_dhl, dict) and 'year_by_year' in track_dhl:
            circuit_name = track_dhl.get('circuit_name', 'Unknown Circuit')
            print(f"\nFirst Pit Stop Analysis for: {circuit_name}")
            
            # Debug: check if we have data
            year_data = track_dhl.get('year_by_year', pd.DataFrame())
            overall_stats = track_dhl.get('overall_stats', pd.DataFrame())
            if year_data.empty and overall_stats.empty:
                print("No historical DHL pit stop data available for this circuit")
                print("This could be due to circuit name mapping issues or lack of data")
                
                # Additional debug info
                next_race = self.get_next_race()
                if next_race is not None:
                    print(f"Debug: Looking for circuit ID: {next_race.get('circuitId', 'Unknown')}")
                    
                # Check if DHL data has circuit mapping at all
                if hasattr(self, 'dhl_data') and not self.dhl_data.empty:
                    if 'circuitId' in self.dhl_data.columns:
                        unique_circuits = self.dhl_data['circuitId'].dropna().unique()
                        print(f"Debug: DHL data contains {len(unique_circuits)} unique circuits")
                    else:
                        print("Debug: DHL data does not have circuitId column mapped")
            
            # Show year-by-year first stop data
            year_data = track_dhl['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year First Stop Times:")
                print("=" * 100)
                
                # Get unique years
                years = sorted(year_data['year'].unique())
                recent_years = years[-5:] if len(years) > 5 else years
                
                # Header
                print(f"{'Driver':<20}", end="")
                for year in recent_years:
                    print(f"{int(year):>18}", end="")
                print()
                
                # Sub-header
                print(f"{'':<20}", end="")
                for year in recent_years:
                    print(f"{'Avg  Best  Lap':>18}", end="")
                print()
                print("-" * (20 + len(recent_years) * 18))
                
                # Sort drivers by most recent year average
                driver_sort = []
                for driver_id in year_data['driverId'].unique():
                    driver_years = year_data[year_data['driverId'] == driver_id]
                    most_recent = driver_years[driver_years['year'] == max(driver_years['year'])]
                    if not most_recent.empty:
                        avg_time = most_recent.iloc[0]['avg_time']
                        driver_sort.append((driver_id, avg_time))
                
                driver_sort.sort(key=lambda x: x[1])  # Sort by avg time
                
                for driver_id, _ in driver_sort[:15]:  # Top 15 drivers
                    driver_years = year_data[year_data['driverId'] == driver_id]
                    driver_name = driver_years.iloc[0].get('driver_name', str(driver_id))
                    
                    if len(driver_name) > 19:
                        driver_name = driver_name[:16] + "..."
                    
                    print(f"{driver_name:<20}", end="")
                    
                    for year in recent_years:
                        year_row = driver_years[driver_years['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            avg = f"{row['avg_time']:.2f}"
                            best = f"{row['best_time']:.2f}"
                            lap = f"{int(row['avg_lap'])}"
                            print(f"{avg:>5} {best:>5} {lap:>5}", end="")
                        else:
                            print(f"{'---':>5} {'---':>5} {'---':>5}", end="")
                    print()
                
                if len(years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history from {min(years)} to {max(years)}")
            
            # Career stats at this track
            overall_stats = track_dhl['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer First Stop Statistics at this Track (Active Drivers):")
                print("-" * 80)
                print(f"{'Driver':<25} {'Avg First Stop':<15} {'Best First Stop':<15} {'Races':<10} {'Avg Lap':<10}")
                print("-" * 80)
                
                overall_sorted = overall_stats.sort_values('avg_first_stop', ascending=True)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['avg_first_stop']:<15.3f} {row['best_first_stop']:<15.3f} "
                          f"{int(row['total_races']):<10} {row['avg_lap']:<10.1f}")
        else:
            print("No track-specific DHL data available for the next race")
        
        # 4. Starting Position Analysis
        print("\n" + "="*80)
        print("4. STARTING POSITIONS BY DRIVER")
        print("="*80)
        grid = self.analyze_starting_positions()
        if not grid.empty:
            display_cols = ['driver_name', 'avg_start_position', 'median_start_position', 'best_start_position']
            if 'avg_points_per_race' in grid.columns:
                display_cols.append('avg_points_per_race')
            if 'next_circuit_avg' in grid.columns:
                display_cols.append('next_circuit_avg')
            # Remove driver_name if it doesn't exist
            display_cols = [col for col in display_cols if col in grid.columns]
            print(grid[display_cols].to_string(index=False))
        else:
            print("No starting position data available")
        
        # 4b. Track-specific starting positions by year
        print("\n" + "-"*80)
        print("4b. TRACK-SPECIFIC STARTING POSITIONS BY YEAR")
        print("-"*80)
        track_year_grid = self.analyze_starting_positions_by_track_year()
        if isinstance(track_year_grid, dict) and 'year_by_year' in track_year_grid:
            circuit_name = track_year_grid.get('circuit_name', 'Unknown Circuit')
            print(f"\nAnalysis for: {circuit_name}")
            
            # Show year-by-year data in matrix format
            year_data = track_year_grid['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Starting Position Matrix:")
                print("=" * 120)
                
                # Create a compact matrix
                year_data_sorted = year_data.sort_values(['driver_name', 'year'])
                
                # Get unique years - limit to recent years
                all_years = sorted(year_data_sorted['year'].unique())
                recent_years = all_years[-5:] if len(all_years) > 5 else all_years
                
                # Header with years
                header = f"{'Driver':<25}"
                for year in recent_years:
                    header += f"{int(year):>22}"
                print(header)
                
                # Sub-header
                sub_header = f"{'':<25}"
                for year in recent_years:
                    sub_header += f"{'Start  Pts/Race':>22}"
                print(sub_header)
                print("-" * (25 + len(recent_years) * 22))
                
                # Sort drivers by best average starting position in most recent year
                driver_sort_data = []
                for driver_id in year_data_sorted['driverId'].unique():
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    recent_data = driver_data[driver_data['year'].isin(recent_years)]
                    if not recent_data.empty:
                        most_recent = recent_data.nlargest(1, 'year').iloc[0]
                        driver_sort_data.append((driver_id, most_recent.get('avg_start', 20)))
                    else:
                        driver_sort_data.append((driver_id, 20))
                
                driver_sort_data.sort(key=lambda x: x[1])  # Sort by avg start (lower is better)
                
                # Display each driver's data
                for driver_id, _ in driver_sort_data:
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    
                    if len(driver_name) > 24:
                        driver_name = driver_name[:21] + "..."
                    
                    row_str = f"{driver_name:<25}"
                    
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            # Check for NaN values
                            if pd.notna(row['avg_start']):
                                avg_start = f"{row['avg_start']:.1f}"
                                # Check if we have points data and it's not NaN
                                if 'avg_points' in row and pd.notna(row['avg_points']):
                                    pts_per_race = f"{row['avg_points']:.1f}"
                                    row_str += f"{avg_start:>7}  {pts_per_race:>11}    "
                                else:
                                    row_str += f"{avg_start:>7}  {'—':>11}    "
                            else:
                                # If avg_start is NaN, show em dash
                                row_str += f"{'—':^22}"
                        else:
                            row_str += f"{'—':^22}"
                    
                    print(row_str)
                
                if len(all_years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history from {min(all_years)} to {max(all_years)}")
            
            # Show career stats
            overall_stats = track_year_grid['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer Statistics at this Track:")
                print("-" * 100)
                # Check which columns we have
                if 'career_avg_points' in overall_stats.columns:
                    print(f"{'Driver':<25} {'Races':<8} {'Avg Start':<12} "
                          f"{'Total Pts':<10} {'Avg Pts':<10}")
                else:
                    print(f"{'Driver':<25} {'Races':<8} {'Avg Start':<12}")
                print("-" * 100)
                
                overall_sorted = overall_stats.sort_values('career_avg_start')
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    base_info = f"{driver_name:<25} {row['races_at_track']:<8.0f} " \
                               f"{row['career_avg_start']:<12.1f}"
                    
                    if 'career_avg_points' in overall_stats.columns:
                        print(f"{base_info} {row.get('career_points', 0):<10.0f} "
                              f"{row.get('career_avg_points', 0):<10.1f}")
                    else:
                        print(base_info)
        else:
            print("No track-specific starting position data available for the next race")
        
        # 5. Sprint Points Analysis
        print("\n" + "="*80)
        print("5. SPRINT POINTS BY DRIVER")
        print("="*80)
        sprint = self.analyze_sprint_points()
        if not sprint.empty:
            display_cols = ['driver_name', 'total_sprint_points', 'avg_sprint_points', 'median_sprint_points', 'sprint_races']
            if 'next_circuit_avg' in sprint.columns:
                display_cols.append('next_circuit_avg')
            # Remove driver_name if it doesn't exist
            display_cols = [col for col in display_cols if col in sprint.columns]
            print(sprint[display_cols].to_string(index=False))
        else:
            print("No sprint race data available")
        
        # Explain zero/NaN values if sprint data exists
        if not sprint.empty:
            explanations = self.explain_zero_or_nan_values(sprint, 'sprint')
            if explanations:
                print("\nExplanation for 0/NaN values:")
                print(explanations)
        
        # 5b. Track-specific sprint points by year
        print("\n" + "-"*80)
        print("5b. TRACK-SPECIFIC SPRINT POINTS BY YEAR")
        print("-"*80)
        track_year_sprints = self.analyze_sprint_points_by_track_year()
        if isinstance(track_year_sprints, dict) and 'year_by_year' in track_year_sprints:
            circuit_name = track_year_sprints.get('circuit_name', 'Unknown Circuit')
            print(f"\nAnalysis for: {circuit_name}")
            
            # Show year-by-year data in matrix format
            year_data = track_year_sprints['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Sprint Points Matrix:")
                print("=" * 100)
                
                # Create a compact matrix
                year_data_sorted = year_data.sort_values(['driver_name', 'year'])
                
                # Get unique years
                all_years = sorted(year_data_sorted['year'].unique())
                
                # Header with years
                header = f"{'Driver':<25}"
                for year in all_years:
                    header += f"{int(year):>20}"
                print(header)
                
                # Sub-header
                sub_header = f"{'':<25}"
                for year in all_years:
                    sub_header += f"{'Pts  Avg  Finish':>20}"
                print(sub_header)
                print("-" * (25 + len(all_years) * 20))
                
                # Sort drivers by total sprint points in most recent year
                driver_sort_data = []
                for driver_id in year_data_sorted['driverId'].unique():
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    most_recent = driver_data.nlargest(1, 'year').iloc[0]
                    driver_sort_data.append((driver_id, most_recent.get('total_points', 0)))
                
                driver_sort_data.sort(key=lambda x: x[1], reverse=True)
                
                # Display each driver's data
                for driver_id, _ in driver_sort_data:
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    
                    if len(driver_name) > 24:
                        driver_name = driver_name[:21] + "..."
                    
                    row_str = f"{driver_name:<25}"
                    
                    for year in all_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            total_pts = f"{row['total_points']:.0f}"
                            avg_pts = f"{row['avg_points']:.1f}"
                            avg_finish = f"{row['avg_finish']:.1f}"
                            row_str += f"{total_pts:>4} {avg_pts:>5} {avg_finish:>7}    "
                        else:
                            row_str += f"{'—':^20}"
                    
                    print(row_str)
                
                # Note about sprint races
                print(f"\n* Sprint races at this circuit: {len(all_years)} year(s)")
                print("* Sprint races only started in 2021")
            
            # Show career stats
            overall_stats = track_year_sprints['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer Sprint Statistics at this Track:")
                print("-" * 100)
                print(f"{'Driver':<25} {'Years':<8} {'Races':<8} {'Total Pts':<12} "
                      f"{'Avg Pts':<10} {'Avg Finish':<12}")
                print("-" * 100)
                
                overall_sorted = overall_stats.sort_values('career_points', ascending=False)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['years_raced']:<8.0f} {row['total_races']:<8.0f} "
                          f"{row['career_points']:<12.0f} {row['career_avg_points']:<10.1f} "
                          f"{row['career_avg_finish']:<12.1f}")
        else:
            print("No sprint races have been held at the next race circuit")
        
        
        # 6. Fastest Lap Analysis
        print("\n" + "="*80)
        print("6. FASTEST LAP ANALYSIS")
        print("="*80)
        fastest = self.analyze_fastest_laps()
        if not fastest.empty:
            display_cols = ['driver_name', 'total_fastest_laps', 'total_races', 'fastest_lap_rate', 
                          'avg_lap_number', 'seasons_with_fl', 'fastest_lap_points']
            # Remove driver_name if it doesn't exist
            display_cols = [col for col in display_cols if col in fastest.columns]
            print(fastest[display_cols].to_string(index=False))
            print("\nNote: Fastest lap awards 1 F1 point (if finishing in top 10)")
        else:
            print("No fastest lap data available")
        
        # 6b. Track-specific fastest laps
        print("\n" + "-"*80)
        print("6b. TRACK-SPECIFIC FASTEST LAPS BY YEAR")
        print("-"*80)
        track_fastest = self.analyze_fastest_laps_by_track_year()
        if isinstance(track_fastest, dict) and 'year_by_year' in track_fastest:
            circuit_name = track_fastest.get('circuit_name', 'Unknown Circuit')
            print(f"\nAnalysis for: {circuit_name}")
            
            year_data = track_fastest['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Fastest Laps:")
                print("=" * 120)
                
                # Get unique years and sort
                all_years = sorted(year_data['year'].unique())
                recent_years = all_years[-5:] if len(all_years) > 5 else all_years
                
                # Header with years
                header = f"{'Driver':<25}"
                for year in recent_years:
                    header += f"{int(year):>20}"
                print(header)
                
                # Sub-header
                sub_header = f"{'':<25}"
                for year in recent_years:
                    sub_header += f"{'Lap (Time)':>20}"
                print(sub_header)
                print("-" * (25 + len(recent_years) * 20))
                
                # Sort drivers by total fastest laps
                driver_totals = year_data.groupby('driverId')['fastest_laps'].sum().sort_values(ascending=False)
                
                for driver_id in driver_totals.index[:20]:  # Top 20 drivers
                    driver_data = year_data[year_data['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    
                    # Truncate long names
                    if len(driver_name) > 24:
                        driver_name = driver_name[:21] + "..."
                    
                    row_str = f"{driver_name:<25}"
                    
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty and year_row.iloc[0]['fastest_laps'] > 0:
                            row = year_row.iloc[0]
                            lap = row['lap_number']
                            time = row['lap_time']
                            if pd.notna(lap) and pd.notna(time):
                                row_str += f"{int(lap):>3} ({time})".rjust(20)
                            elif pd.notna(lap):
                                row_str += f"{int(lap):>3}".rjust(20)
                            else:
                                row_str += f"{'FL':>20}"
                        else:
                            row_str += f"{'—':^20}"
                    
                    print(row_str)
                
                if len(all_years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history from {min(all_years)} to {max(all_years)}")
            
            # Career stats
            overall_stats = track_fastest['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer Fastest Lap Statistics at this Track:")
                print("-" * 80)
                print(f"{'Driver':<25} {'Total FL':<10} {'Avg Lap':<10}")
                print("-" * 80)
                
                overall_sorted = overall_stats.sort_values('total_fastest_laps', ascending=False)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['total_fastest_laps']:<10.0f} "
                          f"{row['avg_lap_number']:<10.1f}")
        else:
            print("No fastest lap data available for this circuit")
        
        # 8. First Pit Stop Time Analysis (PrizePicks)
        print("\n" + "="*80)
        print("8. FIRST PIT STOP TIME ANALYSIS (seconds) - PrizePicks Metric")
        print("="*80)
        first_stops = self.calculate_first_pit_stop_time()
        if not first_stops.empty:
            # Display columns for first stop analysis
            display_cols = ['driver_name', 'avg_1st_stop', 'median_1st_stop', 
                          'best_1st_stop', 'worst_1st_stop', 'avg_1st_stop_lap', 
                          'races_with_1st_stop', '1st_stop_consistency']
            # Add overall stop columns
            overall_cols = ['avg_all_stops', 'median_all_stops', 'total_stops']
            display_cols.extend(overall_cols)
            
            # Filter to existing columns
            display_cols = [col for col in display_cols if col in first_stops.columns]
            
            # Filter out rows with NaN values in first stop data
            numeric_cols = ['avg_1st_stop', 'median_1st_stop', 'best_1st_stop']
            numeric_cols = [col for col in numeric_cols if col in first_stops.columns]
            first_stops_filtered = first_stops.dropna(subset=numeric_cols, how='all')
            
            if not first_stops_filtered.empty:
                print("First pit stop vs Overall pit stop analysis from DHL official data:")
                print("\nFIRST STOP ONLY | OVERALL STATS")
                print(first_stops_filtered[display_cols].to_string(index=False))
                print("\nColumn Explanations:")
                print("FIRST STOP COLUMNS:")
                print("- avg_1st_stop: Average duration of FIRST pit stop only (seconds)")
                print("- median_1st_stop: Median duration of FIRST pit stop (typical performance)")
                print("- best_1st_stop: Fastest FIRST pit stop recorded")
                print("- worst_1st_stop: Slowest FIRST pit stop recorded")
                print("- avg_1st_stop_lap: Average lap number when first stop occurs")
                print("- races_with_1st_stop: Number of races with first stop data")
                print("- 1st_stop_consistency: |avg - median| for first stops (lower = more consistent)")
                print("\nOVERALL COLUMNS:")
                print("- avg_all_stops: Average duration across ALL pit stops")
                print("- median_all_stops: Median duration across ALL pit stops")
                print("- total_stops: Total number of pit stops recorded")
                print("\nNote: For PrizePicks '1st Pit Stop Time' betting, use the FIRST STOP columns")
                
                # Show insights
                if 'avg_1st_stop' in first_stops_filtered.columns and 'avg_all_stops' in first_stops_filtered.columns:
                    # Find drivers whose first stops are notably different from their overall average
                    first_stops_filtered['diff_1st_vs_all'] = (
                        first_stops_filtered['avg_1st_stop'] - first_stops_filtered['avg_all_stops']
                    ).round(3)
                    
                    slower_first = first_stops_filtered[first_stops_filtered['diff_1st_vs_all'] > 0.1].nlargest(5, 'diff_1st_vs_all')
                    if not slower_first.empty:
                        print("\nDrivers with slower first stops than average:")
                        for _, driver in slower_first.iterrows():
                            print(f"- {driver['driver_name']}: First stop {driver['diff_1st_vs_all']:.3f}s slower than overall avg")
            else:
                print("No first pit stop time data available")
        else:
            print("No DHL pit stop data available for analysis")
        
        # 9. PrizePicks Overtake Points Analysis
        print("\n" + "="*80)
        print("9. PRIZEPICKS OVERTAKE POINTS ANALYSIS")
        print("="*80)
        pp_overtakes = self.calculate_prizepicks_overtake_points()
        if not pp_overtakes.empty:
            display_cols = ['driver_name', 'avg_pp_overtake_pts', 'median_pp_overtake_pts', 
                          'total_teammate_bonuses', 'total_teammate_penalties', 'races_analyzed']
            display_cols = [col for col in display_cols if col in pp_overtakes.columns]
            
            # Sort by average PrizePicks overtake points
            pp_overtakes_sorted = pp_overtakes.sort_values('avg_pp_overtake_pts', ascending=False)
            
            print("PrizePicks Overtake Points = (Start Position - Finish Position) + Teammate Adjustments")
            print("Teammate bonus: +0.5 for passing teammate, -0.5 for being passed by teammate")
            print(pp_overtakes_sorted[display_cols].head(20).to_string(index=False))
            
            # Notable insights
            if 'total_teammate_bonuses' in pp_overtakes.columns:
                most_teammate_passes = pp_overtakes.nlargest(5, 'total_teammate_bonuses')
                if not most_teammate_passes.empty:
                    print("\nDrivers with most teammate overtakes:")
                    for _, driver in most_teammate_passes.iterrows():
                        if 'driver_name' in driver:
                            print(f"- {driver['driver_name']}: {driver['total_teammate_bonuses']:.1f} bonuses")
        else:
            print("No data available for PrizePicks overtake points calculation")
        
        # 10. Win/Podium/DNF Probability Analysis
        print("\n" + "="*80)
        print("10. WIN/PODIUM/DNF PROBABILITY ANALYSIS")
        print("="*80)
        probabilities = self.calculate_win_podium_probabilities()
        if not probabilities.empty:
            display_cols = ['driver_name', 'win_probability', 'podium_probability', 
                          'points_probability', 'expected_points_per_race', 'dnf_probability', 'total_races']
            display_cols = [col for col in display_cols if col in probabilities.columns]
            
            # Sort by expected points per race
            if 'expected_points_per_race' in probabilities.columns:
                probabilities_sorted = probabilities.sort_values('expected_points_per_race', ascending=False)
            else:
                probabilities_sorted = probabilities.sort_values('win_probability', ascending=False)
            
            print("Probabilities based on last 2 years of race results:")
            print(probabilities_sorted[display_cols].head(20).to_string(index=False))
            
            # Highlight key insights
            print("\nKey Insights:")
            if 'win_probability' in probabilities.columns:
                top_winners = probabilities[probabilities['win_probability'] > 0].nlargest(5, 'win_probability')
                if not top_winners.empty:
                    print(f"Most likely to win: {', '.join([row['driver_name'] for _, row in top_winners.iterrows() if 'driver_name' in row])}")
            
            if 'dnf_probability' in probabilities.columns:
                high_dnf = probabilities[probabilities['dnf_probability'] > 20]
                if not high_dnf.empty:
                    print(f"High DNF risk (>20%): {', '.join([row['driver_name'] for _, row in high_dnf.iterrows() if 'driver_name' in row])}")
        else:
            print("No data available for probability calculations")
        
        
        print("\n" + "="*80)
        print("ANALYSIS NOTES:")
        print("="*80)
        print("- All statistics based on real F1 data (no synthetic data)")
        print("- 'next_circuit_avg' shows historical performance at the upcoming race circuit")
        print("- Median values provide insight into typical performance (less affected by outliers)")
        print("- Data includes races from the last 3 years for relevance")
        print("- DHL pit stop data: Official DHL fastest pit stop competition results")
        print("- First Pit Stop Time: Actual pit box service duration from DHL data (PrizePicks metric)")
        print("- PrizePicks Overtake Points: Grid - Finish + teammate adjustments (+0.5/-0.5)")
        print("- Probabilities: Based on last 2 years, useful for betting market analysis")
        
        # Initialize missing variables to empty DataFrames if not analyzed
        if 'pit_stops' not in locals():
            pit_stops = pd.DataFrame()
        if 'dhl_stops' not in locals():
            dhl_stops = pd.DataFrame()
        if 'grid' not in locals():
            grid = pd.DataFrame()
        
        return {
            'overtakes': overtakes,
            'points': points,
            'pit_stops': pit_stops,
            'dhl_pit_stops': dhl_stops,
            'starting_positions': grid,
            'sprint_points': sprint,
            'fastest_laps': fastest,
            'first_pit_stop_time': first_stops if 'first_stops' in locals() else pd.DataFrame(),
            'prizepicks_overtakes': pp_overtakes if 'pp_overtakes' in locals() else pd.DataFrame(),
            'probabilities': probabilities if 'probabilities' in locals() else pd.DataFrame()
        }


def test_analyzer():
    """Test the analyzer with sample data"""
    from f1db_data_loader import F1DBDataLoader
    
    # Load data using F1DBDataLoader with correct path for notebooks/advanced
    loader = F1DBDataLoader(data_dir="../../data/f1db")
    data = loader.load_csv_data(validate=False, check_updates=False)
    
    # Create analyzer
    analyzer = F1PerformanceAnalyzer(data)
    
    # Generate all tables
    tables = analyzer.generate_all_tables()
    
    return analyzer, tables


if __name__ == "__main__":
    analyzer, tables = test_analyzer()
# """
# F1 Performance Analysis Module

# Generates comprehensive driver performance tables including:
# - Overtakes by driver
# - F1 points analysis
# - Pit stop times
# - Starting positions
# - Sprint points
# - Circuit-specific predictions
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# class F1PerformanceAnalyzer:
#     """Analyzes driver performance across multiple metrics"""
    
#     def __init__(self, data_dict):
#         self.data = data_dict
#         # Dynamically determine current season from the data
#         self.current_year = self._get_current_season()
#         # Use today's actual date for finding the next race
#         self.current_date = datetime.now()
    
#     def _get_current_season(self):
#         """Dynamically determine the current season from race results"""
#         results = self.data.get('results', pd.DataFrame())
#         races = self.data.get('races', pd.DataFrame())
        
#         # First try results data
#         if not results.empty and 'year' in results.columns:
#             # Get the latest year that has actual race results (not just scheduled)
#             results_with_data = results[results['positionNumber'].notna()]
#             if not results_with_data.empty:
#                 return int(results_with_data['year'].max())
        
#         # Fallback to races data
#         if not races.empty and 'year' in races.columns:
#             # Get races that have already happened (date in the past)
#             races['date'] = pd.to_datetime(races['date'])
#             past_races = races[races['date'] <= datetime.now()]
#             if not past_races.empty:
#                 return int(past_races['year'].max())
        
#         # Final fallback - use the latest year with any results
#         if not results.empty and 'year' in results.columns:
#             return int(results['year'].max())
        
#         return datetime.now().year
        
#     def get_next_race(self):
#         """Get the next upcoming race"""
#         races = self.data.get('races', pd.DataFrame()).copy()
#         if races.empty or 'date' not in races.columns:
#             return None
            
#         races['date'] = pd.to_datetime(races['date'])
#         # Get races after today's actual date
#         upcoming = races[races['date'] > datetime.now()].sort_values('date')
        
#         if upcoming.empty:
#             # If no future races, get the most recent
#             return races.sort_values('date').iloc[-1]
        
#         return upcoming.iloc[0]
    
#     def get_active_drivers(self, year=None):
#         """Get list of active drivers for a given year"""
#         if year is None:
#             year = self.current_year
            
#         results = self.data.get('results', pd.DataFrame())  # Fixed key name
#         if results.empty:
#             return {}  # Return DataFrame not list
            
#         # Get drivers who raced in the specified year
#         if 'year' in results.columns:
#             year_results = results[results['year'] == year]
#         else:
#             # Try to join with races to get year
#             races = self.data.get('races', pd.DataFrame())
#             if not races.empty and 'id' in races.columns and 'raceId' in results.columns:
#                 results_with_year = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
#                 year_results = results_with_year[results_with_year['year'] == year]
#             else:
#                 # Fallback: use most recent results
#                 year_results = results.tail(1000)
        
#         if year_results.empty:
#             return {}  # Return DataFrame not list
            
#         # Get unique driver IDs
#         driver_ids = year_results['driverId'].unique()
        
#         # Get driver details
#         drivers = self.data.get('drivers', pd.DataFrame())
#         if drivers.empty:
#             return {}  # Return DataFrame not list
            
#         active_drivers = drivers[drivers['id'].isin(driver_ids)]
#         return active_drivers
    
#     def filter_current_season_drivers(self, df):
#         """Filter dataframe to only include drivers from current season"""
#         if df.empty:
#             return df
            
#         # Get current season drivers
#         current_drivers = self.get_active_drivers()
        
#         # Check if we got a valid DataFrame
#         if not isinstance(current_drivers, pd.DataFrame) or current_drivers.empty:
#             print(f"Warning: No active drivers found for {self.current_year}, returning all drivers")
#             return df
            
#         # Filter by driver IDs (always use IDs for consistency)
#         try:
#             current_ids = current_drivers['id'].values
#             # Check if index is already driver IDs
#             if df.index.name == 'id' or df.index.name == 'driverId':
#                 return df[df.index.isin(current_ids)]
#             # If we have an 'id' column, use it
#             elif 'id' in df.columns:
#                 return df[df['id'].isin(current_ids)]
#             # If we have a 'driverId' column, use it  
#             elif 'driverId' in df.columns:
#                 return df[df['driverId'].isin(current_ids)]
#             else:
#                 print(f"Warning: Cannot filter by driver ID, no id/driverId column found")
#                 return df
#         except Exception as e:
#             print(f"Warning: Error filtering drivers: {e}")
#             return df
    
#     def analyze_overtakes(self):
#         """Analyze overtakes by driver"""
#         # Note: F1 data doesn't directly track overtakes, so we'll calculate position changes
#         results = self.data.get('results', pd.DataFrame()).copy()
#         grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        
#         if results.empty or grid.empty:
#             return {}
        
#         # Merge results with starting grid
#         overtake_data = results.merge(
#             grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
#             on=['raceId', 'driverId'],
#             how='left'
#         )
        
#         # Calculate position gained (negative means overtakes made)
#         overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
        
#         # Add year information if not present
#         if 'year' not in overtake_data.columns:
#             races = self.data.get('races', pd.DataFrame())
#             if not races.empty:
#                 overtake_data = overtake_data.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
#         # Filter for recent years
#         if 'year' in overtake_data.columns:
#             recent_data = overtake_data[overtake_data['year'] >= self.current_year - 3]
#         else:
#             print("Warning: No year column found in overtake data")
#             recent_data = overtake_data
        
#         # Group by driver
#         driver_overtakes = recent_data.groupby('driverId').agg({
#             'positions_gained': ['sum', 'mean', 'median', 'count'],
#             'points': ['sum', 'mean']
#         }).round(2)
        
#         driver_overtakes.columns = ['total_positions_gained', 'avg_positions_gained', 
#                                    'median_positions_gained', 'races', 'total_points', 'avg_points']
        
#         # Calculate overtakes (only positive position gains)
#         recent_data['overtakes'] = recent_data['positions_gained'].apply(lambda x: max(0, x))
#         overtakes_by_driver = recent_data.groupby('driverId')['overtakes'].agg(['sum', 'mean', 'median']).round(2)
#         overtakes_by_driver.columns = ['total_OT', 'avg_OT', 'median_OT']
        
#         # Combine data
#         final_data = driver_overtakes.join(overtakes_by_driver)
        
#         # Keep driverId as index for proper identification
        
#         # Filter to only current season drivers
#         final_data = self.filter_current_season_drivers(final_data)
        
#         # Add circuit-specific prediction for next race
#         next_race = self.get_next_race()
#         if next_race is not None and 'circuitId' in next_race:
#             circuit_data = recent_data  # Skip circuit filtering for now
#             if not circuit_data.empty:
#                 circuit_overtakes = circuit_data.groupby('driverId')['overtakes'].mean().round(2)
#                 final_data['next_circuit_avg'] = final_data.index.map(
#                     lambda x: circuit_overtakes.get(x, final_data.loc[x, 'avg_OT'] if x in final_data.index else 0)
#                 )
        
#         return final_data
    
#     def analyze_overtakes_by_track_year(self):
#         """Analyze overtakes by driver for each track, broken down by year"""
#         results = self.data.get('results', pd.DataFrame()).copy()
#         grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
#         races = self.data.get('races', pd.DataFrame())
#         circuits = self.data.get('circuits', pd.DataFrame())
        
#         if results.empty or grid.empty or races.empty:
#             return {}
        
#         # First, ensure results has year and circuitId columns
#         if ('year' not in results.columns or 'circuitId' not in results.columns) and not races.empty:
#             merge_cols = ['id']
#             if 'year' not in results.columns:
#                 merge_cols.append('year')
#             if 'circuitId' not in results.columns:
#                 merge_cols.append('circuitId')
#             results = results.merge(races[merge_cols], 
#                                   left_on='raceId', right_on='id', how='left')
        
#         # Get next race info
#         next_race = self.get_next_race()
#         if next_race is None or 'circuitId' not in next_race:
#             return {}
        
#         next_circuit_id = next_race['circuitId']
        
#         # Get circuit name
#         circuit_name = 'Unknown Circuit'
#         if not circuits.empty and 'id' in circuits.columns:
#             circuit_info = circuits[circuits['id'] == next_circuit_id]
#             if not circuit_info.empty:
#                 circuit_name = circuit_info.iloc[0].get('name', 'Unknown Circuit')
        
#         # Merge all necessary data - results already has year and circuitId from earlier merge
#         overtake_data = results.merge(
#             grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
#             on=['raceId', 'driverId'],
#             how='left'
#         )
        
#         # Filter for the next race's circuit only
#         circuit_data = overtake_data[overtake_data['circuitId'] == next_circuit_id].copy()
        
#         if circuit_data.empty:
#             return {}
        
#         # Ensure year column exists
#         if 'year' not in circuit_data.columns:
#             print("Warning: No year column found in circuit data")
#             return {}
        
#         # Calculate position changes
#         circuit_data['positions_gained'] = circuit_data['gridPosition'] - circuit_data['positionNumber']
#         circuit_data['overtakes'] = circuit_data['positions_gained'].apply(lambda x: max(0, x))
        
#         # Group by driver and year
#         track_year_analysis = circuit_data.groupby(['driverId', 'year']).agg({
#             'overtakes': ['sum', 'mean', 'median'],
#             'gridPosition': 'mean',
#             'positionNumber': 'mean',
#             'positions_gained': 'mean'
#         }).round(2)
        
#         # Flatten column names
#         track_year_analysis.columns = [
#             'total_overtakes', 'avg_overtakes', 'median_overtakes',
#             'avg_start_pos', 'avg_finish_pos', 'avg_pos_change'
#         ]
        
#         # Reset index to make driverId and year regular columns
#         track_year_analysis = track_year_analysis.reset_index()
        
#         # Add overall statistics per driver across all years at this track
#         overall_stats = circuit_data.groupby('driverId').agg({
#             'overtakes': ['sum', 'mean', 'median'],
#             'gridPosition': 'mean',
#             'positionNumber': 'mean',
#             'positions_gained': 'mean',
#             'year': 'count'
#         }).round(2)
        
#         overall_stats.columns = [
#             'career_overtakes', 'career_avg_overtakes', 'career_median_overtakes',
#             'career_avg_start', 'career_avg_finish', 'career_avg_pos_change',
#             'races_at_track'
#         ]
        
#         # Add driver names
#         drivers = self.data.get('drivers', pd.DataFrame())
#         if not drivers.empty:
#             driver_map = dict(zip(drivers['id'], drivers['name']))
#             track_year_analysis['driver_name'] = track_year_analysis['driverId'].map(driver_map)
#             overall_stats['driver_name'] = overall_stats.index.map(driver_map)
        
#         # Filter to only show current season drivers
#         active_drivers = self.get_active_drivers()
#         if not active_drivers.empty:
#             current_driver_ids = active_drivers['id'].tolist()
#             track_year_analysis = track_year_analysis[
#                 track_year_analysis['driverId'].isin(current_driver_ids)
#             ]
#             overall_stats = overall_stats[overall_stats.index.isin(current_driver_ids)]
        
#         return {
#             'circuit_name': circuit_name,
#             'circuit_id': next_circuit_id,
#             'year_by_year': track_year_analysis,
#             'overall_stats': overall_stats
#         }
    
#     def analyze_points(self):
#         """Analyze F1 points by driver"""
#         results = self.data.get('results', pd.DataFrame()).copy()
#         races = self.data.get('races', pd.DataFrame())
        
#         if results.empty:
#             return {}
        
#         # Add year information
#         if not races.empty and 'year' not in results.columns:
#             results = results.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
#         # Current season data
#         current_season = results[results['year'] == self.current_year]
        
#         # Historical data (last 3 years)
#         historical = results[(results['year'] >= self.current_year - 3) & (results['year'] < self.current_year)]
        
#         # Current season analysis
#         current_stats = current_season.groupby('driverId').agg({
#             'points': ['sum', 'mean', 'median', 'count']
#         }).round(2)
#         current_stats.columns = ['total_points', 'avg_points', 'median_points', 'races']
        
#         # Historical analysis
#         if not historical.empty:
#             hist_stats = historical.groupby('driverId').agg({
#                 'points': ['mean', 'median']
#             }).round(2)
#             hist_stats.columns = ['hist_avg_points', 'hist_median_points']
            
#             # Combine
#             points_analysis = current_stats.join(hist_stats, how='left').fillna(0)
#         else:
#             points_analysis = current_stats
        
#         # Keep driverId as index for proper identification
        
#         # Filter to only current season drivers
#         points_analysis = self.filter_current_season_drivers(points_analysis)
        
#         # Circuit-specific prediction
#         next_race = self.get_next_race()
#         if next_race is not None and 'circuitId' in next_race:
#             circuit_results = pd.DataFrame()  # Skip circuit filtering for now
#             if not circuit_results.empty:
#                 circuit_points = circuit_results.groupby('driverId')['points'].mean().round(2)
#                 points_analysis['next_circuit_avg'] = points_analysis.index.map(
#                     lambda x: circuit_points.get(x, points_analysis.loc[x, 'avg_points'] if x in points_analysis.index else 0)
#                 )
        
#         return points_analysis
    
#     def analyze_pit_stops(self):
#         """Analyze pit stop times by driver"""
#         pit_stops = self.data.get('pit_stops', pd.DataFrame()).copy()
#         races = self.data.get('races', pd.DataFrame())
        
#         if pit_stops.empty:
#             return {}
        
#         # Convert time to seconds
#         if 'timeMillis' in pit_stops.columns:
#             # Use timeMillis and convert to seconds
#             pit_stops['time_seconds'] = pd.to_numeric(pit_stops['timeMillis'], errors='coerce') / 1000
#         elif 'time' in pit_stops.columns:
#             # Parse time string (format: MM:SS.mmm or SS.mmm)
#             def parse_time(t):
#                 if pd.isna(t):
#                     return np.nan
#                 t = str(t)
#                 if ':' in t:
#                     parts = t.split(':')
#                     return float(parts[0]) * 60 + float(parts[1])
#                 else:
#                     return float(t)
#             pit_stops['time_seconds'] = pit_stops['time'].apply(parse_time)
#         else:
#             return {}
        
#         # Add year information
#         if not races.empty:
#             pit_stops = pit_stops.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
#         # Filter recent years
#         recent_stops = pit_stops[pit_stops['year'] >= self.current_year - 3] if 'year' in pit_stops.columns else pit_stops
        
#         # Analyze by driver
#         pit_analysis = recent_stops.groupby('driverId').agg({
#             'time_seconds': ['mean', 'median', 'min', 'std', 'count']
#         }).round(3)
#         pit_analysis.columns = ['avg_stop_time', 'median_stop_time', 'best_stop_time', 'std_dev', 'total_stops']
        
#         # Keep driverId as index for proper identification
        
#         # Filter to only current season drivers
#         pit_analysis = self.filter_current_season_drivers(pit_analysis)
        
#         # Circuit-specific prediction
#         next_race = self.get_next_race()
#         if next_race is not None and 'circuitId' in next_race:
#             circuit_stops = pd.DataFrame()  # Skip circuit filtering for now
#             if not circuit_stops.empty:
#                 circuit_times = circuit_stops.groupby('driverId')['time_seconds'].mean().round(3)
#                 pit_analysis['next_circuit_avg'] = pit_analysis.index.map(
#                     lambda x: circuit_times.get(x, pit_analysis.loc[x, 'avg_stop_time'] if x in pit_analysis.index else 0)
#                 )
        
#         return pit_analysis
    
#     def analyze_starting_positions(self):
#         """Analyze starting positions by driver"""
#         grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
#         races = self.data.get('races', pd.DataFrame())
#         results = self.data.get('races-race-results', pd.DataFrame())
        
#         if grid.empty:
#             return {}
        
#         # Add year and race info
#         if not races.empty:
#             grid = grid.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
#         # Filter recent years
#         recent_grid = grid[grid['year'] >= self.current_year - 3] if 'year' in grid.columns else grid
        
#         # Analyze by driver
#         grid_analysis = recent_grid.groupby('driverId').agg({
#             'positionNumber': ['mean', 'median', 'min', 'count']
#         }).round(2)
#         grid_analysis.columns = ['avg_start_position', 'median_start_position', 'best_start_position', 'races']
        
#         # Add finish position correlation
#         if not results.empty:
#             # Merge grid with results
#             grid_results = recent_grid.merge(
#                 results[['raceId', 'driverId', 'position', 'points']].rename(columns={'position': 'finish_position'}),
#                 on=['raceId', 'driverId'],
#                 how='left'
#             )
            
#             # Calculate average points from each starting position
#             points_by_start = grid_results.groupby('driverId').agg({
#                 'points': ['sum', 'mean']
#             }).round(2)
#             points_by_start.columns = ['total_points', 'avg_points_per_race']
            
#             grid_analysis = grid_analysis.join(points_by_start, how='left')
        
#         # Keep driverId as index for proper identification
        
#         # Filter to only current season drivers
#         grid_analysis = self.filter_current_season_drivers(grid_analysis)
        
#         # Circuit-specific prediction
#         next_race = self.get_next_race()
#         if next_race is not None and 'circuitId' in next_race:
#             circuit_grid = pd.DataFrame()  # Skip circuit filtering for now
#             if not circuit_grid.empty:
#                 circuit_positions = circuit_grid.groupby('driverId')['position'].mean().round(2)
#                 grid_analysis['next_circuit_avg'] = grid_analysis.index.map(
#                     lambda x: circuit_positions.get(x, grid_analysis.loc[x, 'avg_start_position'] if x in grid_analysis.index else 0)
#                 )
        
#         return grid_analysis
    
#     def analyze_sprint_points(self):
#         """Analyze sprint race points by driver"""
#         sprint_results = self.data.get('sprint_results', pd.DataFrame()).copy()
#         races = self.data.get('races', pd.DataFrame())
        
#         if sprint_results.empty:
#             return {}
        
#         # Add year information
#         if not races.empty:
#             sprint_results = sprint_results.merge(
#                 races[['id', 'year', 'circuitId']], 
#                 left_on='raceId', 
#                 right_on='id', 
#                 how='left'
#             )
        
#         # Filter recent years
#         recent_sprints = sprint_results[sprint_results['year'] >= self.current_year - 3] if 'year' in sprint_results.columns else sprint_results
        
#         # Check if we have data
#         if recent_sprints.empty:
#             print(f"No sprint data found for years {self.current_year - 3} to {self.current_year}")
#             return {}
        
#         # Analyze by driver with correct column names
#         agg_dict = {'points': ['sum', 'mean', 'median', 'count']}
#         if 'positionNumber' in recent_sprints.columns:
#             agg_dict['positionNumber'] = ['mean', 'median']
        
#         sprint_analysis = recent_sprints.groupby('driverId').agg(agg_dict).round(2)
        
#         # Set column names based on what was aggregated
#         col_names = ['total_sprint_points', 'avg_sprint_points', 'median_sprint_points', 'sprint_races']
#         if 'positionNumber' in recent_sprints.columns:
#             col_names.extend(['avg_sprint_position', 'median_sprint_position'])
#         sprint_analysis.columns = col_names
        
#         # Keep driverId as index for proper identification
        
#         # Filter to only current season drivers
#         sprint_analysis = self.filter_current_season_drivers(sprint_analysis)
        
#         # Circuit-specific prediction
#         next_race = self.get_next_race()
#         if next_race is not None and 'circuitId' in next_race:
#             circuit_sprints = pd.DataFrame()  # Skip circuit filtering for now
#             if not circuit_sprints.empty:
#                 circuit_sprint_points = circuit_sprints.groupby('driverId')['points'].mean().round(2)
#                 sprint_analysis['next_circuit_avg'] = sprint_analysis.index.map(
#                     lambda x: circuit_sprint_points.get(x, sprint_analysis.loc[x, 'avg_sprint_points'] if x in sprint_analysis.index else 0)
#                 )
        
#         return sprint_analysis
    
#     def explain_zero_or_nan_values(self, df, analysis_type):
#         """Explain why certain drivers have 0 or NaN values"""
#         explanations = []
        
#         if analysis_type == 'sprint':
#             # Check sprint race participation
#             sprint_results = self.data.get('sprint_results', pd.DataFrame())
#             if not sprint_results.empty:
#                 sprint_years = sprint_results['year'].unique() if 'year' in sprint_results.columns else []
#                 explanations.append(f"\nSprint races only started in 2021 (years with sprints: {sorted(sprint_years)})")
                
#                 # Count sprint races per year
#                 if 'year' in sprint_results.columns:
#                     sprint_counts = sprint_results.groupby('year')['raceId'].nunique()
#                     explanations.append("Sprint races per year:")
#                     for year, count in sprint_counts.items():
#                         explanations.append(f"  {year}: {count} sprint races")
                
#                 # Find drivers with 0 or NaN
#                 for driver in df.index:
#                     if pd.isna(df.loc[driver, 'avg_sprint_points']) or df.loc[driver, 'sprint_races'] == 0:
#                         driver_sprints = sprint_results[sprint_results['driverId'] == driver]
#                         if driver_sprints.empty:
#                             explanations.append(f"  • {driver}: Never participated in sprint races (may have raced before 2021 or after retirement)")
#                         else:
#                             explanations.append(f"  • {driver}: Data issue - found in sprint results but showing as 0")
        
#         elif analysis_type == 'points':
#             for driver in df.index:
#                 if df.loc[driver, 'total_points'] == 0:
#                     explanations.append(f"  • {driver}: Scored 0 points in the analyzed period (finished outside points or DNF)")
        
#         return "\n".join(explanations)
    
#     def format_for_display(self, df):
#         """Format dataframe for display - keep driverId as index and identifier"""
#         if df.empty:
#             return df
        
#         # Return as-is, keeping driverId as the index
#         return df
    
#     def get_driver_names(self, driver_ids):
#         """Return driver IDs as-is for consistent identification"""
#         # Always use driver IDs to avoid ambiguity
#         return driver_ids
    
#     def generate_all_tables(self):
#         """Generate all performance analysis tables"""
#         print("\n" + "="*80)
#         print("F1 DRIVER PERFORMANCE ANALYSIS")
#         print("="*80)
#         print(f"Analysis Date: {self.current_date.strftime('%Y-%m-%d')}")
#         print(f"Current Season: {self.current_year}")
#         print(f"Years Analyzed: {self.current_year-3} to {self.current_year}")
#         print(f"Note: Showing only drivers who participated in {self.current_year}")
        
#         # Show current season drivers
#         current_drivers = self.get_active_drivers()
#         if not current_drivers.empty:
#             print(f"\nActive drivers in {self.current_year}: {len(current_drivers)}")
#             driver_list = ', '.join(current_drivers['id'].head(20).tolist())
#             if len(current_drivers) > 20:
#                 driver_list += f" and {len(current_drivers) - 20} more..."
#             print(f"Driver IDs: {driver_list}")
#         print(f"Current Analysis Year: {self.current_year}")
#         print(f"Years Analyzed: {self.current_year-3} to {self.current_year}")
        
#         next_race = self.get_next_race()
#         if next_race is not None:
#             circuits = self.data.get('circuits', pd.DataFrame())
#             circuit_name = 'Unknown'
#             if not circuits.empty and 'circuitId' in next_race:
#                 circuit = circuits[circuits['id'] == next_race['circuitId']]
#                 if not circuit.empty:
#                     circuit_name = circuit.iloc[0]['name']
            
#             print(f"Next Race: {next_race.get('officialName', 'Unknown')} at {circuit_name}")
#             print(f"Date: {next_race['date'].strftime('%Y-%m-%d') if pd.notna(next_race.get('date')) else 'Unknown'}")
        
#         # 1. Overtakes Analysis
#         print("\n" + "="*80)
#         print("1. OVERTAKES BY DRIVER (Position Changes)")
#         print("="*80)
#         overtakes = self.analyze_overtakes()
#         if not overtakes.empty:
#             display_cols = ['total_overtakes', 'avg_overtakes', 'median_overtakes', 'avg_points', 'races']
#             if 'next_circuit_avg' in overtakes.columns:
#                 display_cols.append('next_circuit_avg')
#             print(overtakes[display_cols].to_string())
            
#             # Show drivers with notable statistics
#             print("\nNotable insights:")
#             if 'total_overtakes' in overtakes.columns:
#                 top_overtakers = overtakes.nlargest(5, 'total_overtakes')
#                 print(f"Top 5 overtakers: {', '.join(top_overtakers.index.tolist())}")
                
#                 # Drivers with negative average (lost positions)
#                 lost_positions = overtakes[overtakes['avg_pos_gained'] < 0]
#                 if not lost_positions.empty:
#                     print(f"Drivers who typically lost positions: {', '.join(lost_positions.index.tolist())}")
#         else:
#             print("No overtake data available")
        
#         # 1b. Track-specific overtakes by year
#         print("\n" + "-"*80)
#         print("1b. TRACK-SPECIFIC OVERTAKES BY YEAR")
#         print("-"*80)
#         track_year_overtakes = self.analyze_overtakes_by_track_year()
#         if isinstance(track_year_overtakes, dict) and 'year_by_year' in track_year_overtakes:
#             circuit_name = track_year_overtakes.get('circuit_name', 'Unknown Circuit')
#             print(f"\nAnalysis for: {circuit_name}")
            
#             # Show year-by-year data
#             year_data = track_year_overtakes['year_by_year']
#             if not year_data.empty:
#                 print("\nYear-by-Year Performance:")
#                 print("-" * 120)
#                 # Sort by driver and year for better readability
#                 year_data_sorted = year_data.sort_values(['driver_name', 'year'])
                
#                 # Group by driver to show their progression
#                 for driver_id in year_data_sorted['driverId'].unique():
#                     driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
#                     driver_name = driver_data.iloc[0].get('driver_name', driver_id)
#                     print(f"\n{driver_name}:")
                    
#                     # Show year-by-year stats
#                     for _, row in driver_data.iterrows():
#                         print(f"  {int(row['year'])}: "
#                               f"Overtakes: {row['total_overtakes']:.0f} total, "
#                               f"{row['avg_overtakes']:.1f} avg, "
#                               f"{row['median_overtakes']:.0f} median | "
#                               f"Start: {row['avg_start_pos']:.1f} | "
#                               f"Finish: {row['avg_finish_pos']:.1f} | "
#                               f"Avg Change: {row['avg_pos_change']:+.1f}")
            
#             # Show overall career stats at this track
#             overall_stats = track_year_overtakes['overall_stats']
#             if not overall_stats.empty:
#                 print("\n\nCareer Statistics at this Track:")
#                 print("-" * 100)
#                 print(f"{'Driver':<25} {'Races':<8} {'Total OT':<10} {'Avg OT':<8} {'Med OT':<8} "
#                       f"{'Avg Start':<10} {'Avg Finish':<12} {'Avg Change':<10}")
#                 print("-" * 100)
                
#                 # Sort by average overtakes descending
#                 overall_sorted = overall_stats.sort_values('career_avg_OT', ascending=False)
                
#                 for driver_id, row in overall_sorted.iterrows():
#                     driver_name = row.get('driver_name', driver_id)
#                     print(f"{driver_name:<25} {row['races_at_track']:<8.0f} "
#                           f"{row['career_overtakes']:<10.0f} {row['career_avg_overtakes']:<8.1f} "
#                           f"{row['career_median_overtakes']:<8.0f} {row['career_avg_start']:<10.1f} "
#                           f"{row['career_avg_finish']:<12.1f} {row['career_avg_pos_change']:+10.1f}")
#         else:
#             print("No track-specific overtake data available for the next race")
        
#         # 2. Points Analysis
#         print("\n" + "="*80)
#         print("2. F1 POINTS BY DRIVER")
#         print("="*80)
#         points = self.analyze_points()
#         if not points.empty:
#             display_cols = ['total_points', 'avg_points', 'median_points', 'races']
#             if 'hist_avg_points' in points.columns:
#                 display_cols.append('hist_avg_points')
#             if 'next_circuit_avg' in points.columns:
#                 display_cols.append('next_circuit_avg')
#             print(points[display_cols].to_string())
            
#             # Explain any zeros
#             if not points.empty:
#                 zero_points = points[points['total_points'] == 0]
#                 if not zero_points.empty:
#                     print(f"\nDrivers with 0 points: {', '.join(zero_points.index.tolist())}")
#                     print("(These drivers either didn't finish in points positions or had limited races)")
#         else:
#             print("No points data available")
        
#         # 3. Pit Stop Analysis
#         print("\n" + "="*80)
#         print("3. PIT STOP TIMES BY DRIVER (seconds)")
#         print("="*80)
#         pit_stops = self.analyze_pit_stops()
#         if not pit_stops.empty:
#             display_cols = ['avg_stop_time', 'median_stop_time', 'best_stop_time', 'total_stops']
#             if 'next_circuit_avg' in pit_stops.columns:
#                 display_cols.append('next_circuit_avg')
#             print(pit_stops[display_cols].to_string(index=False))
            
#             # Add pit stop explanations
#             if not pit_stops.empty:
#                 explanations = self.explain_zero_or_nan_values(pit_stops, 'pit_stops')
#                 print(explanations)
#         else:
#             print("No pit stop data available")
        
#         # 4. Starting Position Analysis
#         print("\n" + "="*80)
#         print("4. STARTING POSITIONS BY DRIVER")
#         print("="*80)
#         grid = self.analyze_starting_positions()
#         if not grid.empty:
#             display_cols = ['avg_start_position', 'median_start_position', 'best_start_position']
#             if 'avg_points_per_race' in grid.columns:
#                 display_cols.append('avg_points_per_race')
#             if 'next_circuit_avg' in grid.columns:
#                 display_cols.append('next_circuit_avg')
#             print(grid[display_cols].to_string(index=False))
#         else:
#             print("No starting position data available")
        
#         # 5. Sprint Points Analysis
#         print("\n" + "="*80)
#         print("5. SPRINT POINTS BY DRIVER")
#         print("="*80)
#         sprint = self.analyze_sprint_points()
#         if not sprint.empty:
#             display_cols = ['total_sprint_points', 'avg_sprint_points', 'median_sprint_points', 'sprint_races']
#             if 'next_circuit_avg' in sprint.columns:
#                 display_cols.append('next_circuit_avg')
#             print(sprint[display_cols].to_string(index=False))
#         else:
#             print("No sprint race data available")
        
#         # Explain zero/NaN values if sprint data exists
#         if not sprint.empty:
#             explanations = self.explain_zero_or_nan_values(sprint, 'sprint')
#             if explanations:
#                 print("\nExplanation for 0/NaN values:")
#                 print(explanations)
        
#         print("\n" + "="*80)
#         print("ANALYSIS NOTES:")
#         print("="*80)
#         print("- All statistics based on real F1 data (no synthetic data)")
#         print("- 'next_circuit_avg' shows historical performance at the upcoming race circuit")
#         print("- Median values provide insight into typical performance (less affected by outliers)")
#         print("- Data includes races from the last 3 years for relevance")
        
#         return {
#             'overtakes': overtakes,
#             'points': points,
#             'pit_stops': pit_stops,
#             'starting_positions': grid,
#             'sprint_points': sprint
#         }


# def test_analyzer():
#     """Test the analyzer with sample data"""
#     from f1db_data_loader import load_f1db_data
    
#     # Load data
#     data = load_f1db_data()
    
#     # Create analyzer
#     analyzer = F1PerformanceAnalyzer(data)
    
#     # Generate all tables
#     tables = analyzer.generate_all_tables()
    
#     return analyzer, tables


# if __name__ == "__main__":
#     analyzer, tables = test_analyzer()