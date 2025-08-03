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
                # Find the most recent CSV file
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
        """Analyze overtakes by driver"""
        # Note: F1 data doesn't directly track overtakes, so we'll calculate position changes
        # Try both possible keys for results
        results = self.data.get('results', self.data.get('races_race_results', pd.DataFrame())).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        
        if results.empty or grid.empty:
            return {}
        
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
        
        driver_overtakes.columns = ['total_pos_gained', 'avg_pos_gained', 
                                   'median_pos_gained', 'races', 'total_points', 'avg_points']
        
        # Calculate overtakes (only positive position gains)
        recent_data['overtakes'] = recent_data['positions_gained'].apply(lambda x: max(0, x))
        overtakes_by_driver = recent_data.groupby('driverId')['overtakes'].agg(['sum', 'mean', 'median']).round(2)
        overtakes_by_driver.columns = ['total_OT', 'avg_OT', 'median_OT']
        
        # Combine data
        final_data = driver_overtakes.join(overtakes_by_driver)
        
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
                for col in ['total_pos_gained', 'total_OT', 'total_points', 'races']:
                    if col in missing_data.columns:
                        missing_data[col] = 0
                for col in ['avg_pos_gained', 'avg_OT', 'median_pos_gained', 
                           'median_OT', 'avg_points']:
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
                        current_circuit_overtakes = current_circuit_data.groupby('driverId')['overtakes'].mean().round(2)
                        final_data['c_circuit_avg'] = final_data.index.map(
                            lambda x: current_circuit_overtakes.get(x, 0)
                        )
                    
                    # Previous race (race before current - should be 1137 British GP)
                    if len(recent_races) > 1:
                        prev_race = recent_races.iloc[1]
                        prev_circuit_id = prev_race['circuitId']
                        
                        # Calculate circuit averages for previous circuit
                        prev_circuit_data = recent_data[recent_data['circuitId'] == prev_circuit_id]
                        if not prev_circuit_data.empty:
                            prev_circuit_overtakes = prev_circuit_data.groupby('driverId')['overtakes'].mean().round(2)
                            final_data['p_circuit_avg'] = final_data.index.map(
                                lambda x: prev_circuit_overtakes.get(x, 0)
                            )
        
        # Add circuit-specific prediction for next race
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race and 'circuitId' in recent_data.columns:
            # Next race circuit (should be 1139 Hungarian GP)
            next_circuit_id = next_race['circuitId']
            circuit_data = recent_data[recent_data['circuitId'] == next_circuit_id]
            
            if not circuit_data.empty:
                # Calculate average overtakes at this specific circuit
                circuit_overtakes = circuit_data.groupby('driverId')['overtakes'].mean().round(2)
                final_data['n_circuit_avg'] = final_data.index.map(
                    lambda x: circuit_overtakes.get(x, 0)
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
                if not last_race_data.empty and 'overtakes' in last_race_data.columns:
                    last_race_overtakes = last_race_data.set_index('driverId')['overtakes']
                    final_data['last_race'] = final_data.index.map(
                        lambda x: int(last_race_overtakes.get(x, 0))
                    )
        
        # Reorder columns for better presentation
        column_order = ['driver_name', 'avg_OT', 'avg_points', 
                       'median_OT', 'last_race', 'p_circuit_avg', 'c_circuit_avg', 
                       'n_circuit_avg', 'races']
        
        # Add any remaining columns not in the order list (except excluded columns)
        exclude_cols = ['total_OT', 'total_pos_gained', 'total_points', 'avg_pos_gained', 'median_pos_gained']
        remaining_cols = [col for col in final_data.columns if col not in column_order and col not in exclude_cols]
        column_order.extend(remaining_cols)
        
        # Reorder columns, keeping only those that exist
        final_columns = [col for col in column_order if col in final_data.columns]
        final_data = final_data[final_columns]
        
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
        
        # Add year information
        if not races.empty and 'year' not in results.columns:
            results = results.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
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
        # Need to get all results data with circuitId for filtering
        all_results = results[(results['year'] >= self.current_year - 3)]  # Last 3 years for circuit averages
        
        if 'circuitId' in all_results.columns:
            # Get the most recent races with results
            races_with_results = all_results['raceId'].unique()
            recent_races = races[races['id'].isin(races_with_results)]
            
            if not recent_races.empty:
                # Ensure date is datetime
                recent_races = recent_races.copy()
                recent_races['date'] = pd.to_datetime(recent_races['date'])
                recent_races_sorted = recent_races.sort_values('date', ascending=False)
                
                # Current race (most recent with results)
                if len(recent_races_sorted) > 0:
                    current_circuit_id = recent_races_sorted.iloc[0]['circuitId']
                    current_circuit_data = all_results[all_results['circuitId'] == current_circuit_id]
                    if not current_circuit_data.empty:
                        current_circuit_points = current_circuit_data.groupby('driverId')['points'].mean().round(2)
                        points_analysis['current_circuit_avg'] = points_analysis.index.map(
                            lambda x: current_circuit_points.get(x, points_analysis.loc[x, 'avg_points'] if x in points_analysis.index else 0)
                        )
                    else:
                        points_analysis['current_circuit_avg'] = points_analysis['avg_points']
                
                # Previous race (second most recent)
                if len(recent_races_sorted) > 1:
                    prev_circuit_id = recent_races_sorted.iloc[1]['circuitId']
                    prev_circuit_data = all_results[all_results['circuitId'] == prev_circuit_id]
                    if not prev_circuit_data.empty:
                        prev_circuit_points = prev_circuit_data.groupby('driverId')['points'].mean().round(2)
                        points_analysis['prev_circuit_avg'] = points_analysis.index.map(
                            lambda x: prev_circuit_points.get(x, points_analysis.loc[x, 'avg_points'] if x in points_analysis.index else 0)
                        )
                    else:
                        points_analysis['prev_circuit_avg'] = points_analysis['avg_points']
                else:
                    points_analysis['prev_circuit_avg'] = 0
            else:
                points_analysis['prev_circuit_avg'] = 0
                points_analysis['current_circuit_avg'] = points_analysis['avg_points']
        else:
            points_analysis['prev_circuit_avg'] = 0
            points_analysis['current_circuit_avg'] = points_analysis.get('avg_points', 0)
        
        # Next race circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race and 'circuitId' in all_results.columns:
            next_circuit_id = next_race['circuitId']
            next_circuit_data = all_results[all_results['circuitId'] == next_circuit_id]
            if not next_circuit_data.empty:
                next_circuit_points = next_circuit_data.groupby('driverId')['points'].mean().round(2)
                points_analysis['next_circuit_avg'] = points_analysis.index.map(
                    lambda x: next_circuit_points.get(x, points_analysis.loc[x, 'avg_points'] if x in points_analysis.index else 0)
                )
            else:
                points_analysis['next_circuit_avg'] = points_analysis['avg_points']
        else:
            points_analysis['next_circuit_avg'] = points_analysis.get('avg_points', 0)
        
        # Ensure columns exist for all drivers
        if 'prev_circuit_avg' not in points_analysis.columns:
            points_analysis['prev_circuit_avg'] = 0
        if 'current_circuit_avg' not in points_analysis.columns:
            points_analysis['current_circuit_avg'] = points_analysis.get('avg_points', 0)
        if 'next_circuit_avg' not in points_analysis.columns:
            points_analysis['next_circuit_avg'] = points_analysis.get('avg_points', 0)
        
        # Reorder columns for better presentation
        column_order = ['driver_name', 'total_points', 'avg_points', 'median_points', 
                       'races', 'hist_avg_points', 'hist_median_points',
                       'prev_circuit_avg', 'current_circuit_avg', 'next_circuit_avg']
        
        # Keep only columns that exist
        final_columns = [col for col in column_order if col in points_analysis.columns]
        points_analysis = points_analysis[final_columns]
        
        # Drop driver_id index since we already have driver_name
        points_analysis = points_analysis.reset_index(drop=True)
        
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
        pit_stops = self.data.get('pit_stops', pd.DataFrame()).copy()
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
        
        # Drop driver_id index since we already have driver_name
        pit_analysis = pit_analysis.reset_index(drop=True)
        
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
                'sainz': 'carlos-sainz-jr',
                'perez': 'sergio-perez',
                'pérez': 'sergio-perez',
                'magnussen': 'kevin-magnussen',  # Disambiguate from jan-magnussen
                'leclerc': 'charles-leclerc',  # Disambiguate from arthur-leclerc
                'zhou': 'guanyu-zhou',
                'ricciardo': 'daniel-ricciardo',
                'verstappen': 'max-verstappen',
                'bottas': 'valtteri-bottas'
            }
            
            # Add driver IDs to DHL data
            def map_driver(driver_name):
                driver_lower = driver_name.lower()
                # First check special mappings
                if driver_lower in special_mappings:
                    return special_mappings[driver_lower]
                # Then check regular mapping
                return driver_mapping.get(driver_lower)
            
            dhl_data['driverId'] = dhl_data['driver'].apply(map_driver)
            
            # Log unmapped drivers for debugging
            unmapped = dhl_data[dhl_data['driverId'].isna()]
            if not unmapped.empty:
                unmapped_drivers = unmapped['driver'].unique()
                print(f"Warning: Could not map {len(unmapped_drivers)} DHL drivers: {', '.join(unmapped_drivers)}")
            
            dhl_data = dhl_data.dropna(subset=['driverId'])
        
        # Map race names to circuit IDs
        if not races.empty and not circuits.empty and 'race' in dhl_data.columns:
            # Create mapping from race names to circuit IDs
            race_circuit_map = {}
            circuit_name_map = {}
            
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
            
            # Create circuit ID to name mapping
            for _, circuit in circuits.iterrows():
                circuit_name_map[circuit['id']] = circuit['name']
            
            # Add circuit IDs to DHL data
            dhl_data['circuitId'] = dhl_data['race'].str.lower().map(
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
        first_stops = dhl_data.sort_values(['driverId', 'year', 'race', 'lap']).groupby(['driverId', 'race']).first()
        first_stop_analysis = first_stops.groupby('driverId')['time'].agg(['mean', 'median', 'min', 'count']).round(3)
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
        
        # Map race names to circuit IDs (similar to main method)
        if not races.empty and 'race' in dhl_data.columns:
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
        
        # Get first stops only for this circuit
        circuit_first_stops = circuit_data.sort_values(['driverId', 'year', 'lap']).groupby(['driverId', 'year', 'race']).first().reset_index()
        
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
            driver_names = drivers.set_index('id')['surname'].to_dict()
            year_by_year_df['driver_name'] = year_by_year_df['driverId'].map(driver_names)
            
        if not drivers.empty and not overall_stats.empty:
            driver_names = drivers.set_index('id')['surname'].to_dict()
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
        print("1. OVERTAKES BY DRIVER (Position Changes)")
        print("="*80)
        overtakes = self.analyze_overtakes()
        if not overtakes.empty:
            # Note: Column order is already set in analyze_overtakes method
            print(overtakes.to_string(index=False))
            
            # Show drivers with notable statistics
            print("\nNotable insights:")
            if 'total_OT' in overtakes.columns:
                top_overtakers = overtakes.nlargest(5, 'total_OT')
                if 'driver_name' in overtakes.columns:
                    top_names = [overtakes.loc[idx, 'driver_name'] for idx in top_overtakers.index if idx in overtakes.index]
                    print(f"Top 5 overtakers: {', '.join(top_names)}")
                else:
                    print(f"Top 5 overtakers: {', '.join(top_overtakers.index.tolist())}")
                
                # Drivers with negative average (lost positions)
                lost_positions = overtakes[overtakes['avg_pos_gained'] < 0]
                if not lost_positions.empty:
                    if 'driver_name' in overtakes.columns:
                        lost_names = [overtakes.loc[idx, 'driver_name'] for idx in lost_positions.index if idx in overtakes.index]
                        print(f"Drivers who typically lost positions: {', '.join(lost_names)}")
                    else:
                        print(f"Drivers who typically lost positions: {', '.join(lost_positions.index.tolist())}")
        else:
            print("No overtake data available")
        
        # 1b. Track-specific overtakes by year
        print("\n" + "-"*80)
        print("1b. TRACK-SPECIFIC OVERTAKES BY YEAR")
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
            
            # Explain any zeros
            if not points.empty:
                zero_points = points[points['total_points'] == 0]
                if not zero_points.empty:
                    if 'driver_name' in points.columns:
                        zero_names = [points.loc[idx, 'driver_name'] for idx in zero_points.index if idx in points.index]
                        print(f"\nDrivers with 0 points: {', '.join(zero_names)}")
                    else:
                        print(f"\nDrivers with 0 points: {', '.join(zero_points.index.tolist())}")
                    print("(These drivers either didn't finish in points positions or had limited races)")
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
            display_cols = ['driver_name', 'avg_stop_time', 'median_stop_time', 'best_stop_time', 'total_stops']
            if 'next_circuit_avg' in pit_stops.columns:
                display_cols.append('next_circuit_avg')
            # Remove driver_name if it doesn't exist
            display_cols = [col for col in display_cols if col in pit_stops.columns]
            print(pit_stops[display_cols].to_string(index=False))
            
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
            print(dhl_stops_sorted[display_cols].to_string(index=False))
            
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
        
        # 6. Teammate Overtake Analysis
        print("\n" + "="*80)
        print("6. TEAMMATE OVERTAKE ANALYSIS (PrizePicks Scoring)")
        print("="*80)
        teammate = self.analyze_teammate_overtakes()
        if not teammate.empty:
            display_cols = ['driver_name', 'total_battles', 'wins', 'losses', 
                          'win_rate', 'OT_made', 'OT_received',
                          'net_OT', 'prizepicks_pts']
            # Remove driver_name if it doesn't exist
            display_cols = [col for col in display_cols if col in teammate.columns]
            print(teammate[display_cols].to_string(index=False))
            print("\nNote: PrizePicks awards +1.5 points for overtaking teammate, -1.5 for being overtaken")
        else:
            print("No teammate battle data available")
        
        # 6b. Track-specific teammate overtakes
        print("\n" + "-"*80)
        print("6b. TRACK-SPECIFIC TEAMMATE OVERTAKES BY YEAR")
        print("-"*80)
        track_teammate = self.analyze_teammate_overtakes_by_track_year()
        if isinstance(track_teammate, dict) and 'year_by_year' in track_teammate:
            circuit_name = track_teammate.get('circuit_name', 'Unknown Circuit')
            print(f"\nAnalysis for: {circuit_name}")
            
            year_data = track_teammate['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Teammate Battles:")
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
                    sub_header += f"{'W-L (OT) [PP]':>20}"
                print(sub_header)
                print("-" * (25 + len(recent_years) * 20))
                
                # Sort drivers by total wins
                driver_wins = year_data.groupby('driverId')['teammate_win'].sum().sort_values(ascending=False)
                
                for driver_id in driver_wins.index[:20]:  # Top 20 drivers
                    driver_data = year_data[year_data['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    
                    # Truncate long names
                    if len(driver_name) > 24:
                        driver_name = driver_name[:21] + "..."
                    
                    row_str = f"{driver_name:<25}"
                    
                    for year in recent_years:
                        year_row = driver_data[driver_data['year'] == year]
                        if not year_row.empty:
                            row = year_row.iloc[0]
                            wins = int(row['teammate_win'])
                            losses = int(row['teammate_battles']) - wins if 'teammate_battles' in row else 1 - wins
                            net_ot = int(row['teammate_overtake'])
                            pp_pts = row['prizepicks_points']
                            
                            # Check for NaN values
                            if pd.notna(wins) and pd.notna(losses) and pd.notna(net_ot) and pd.notna(pp_pts):
                                # Format: W-L (OT) [PP]
                                # Example: 1-0 (+1) [+1.5]
                                battle_str = f"{wins}-{losses}"
                                ot_str = f"({net_ot:+d})"
                                pp_str = f"[{pp_pts:+.1f}]"
                                
                                combined = f"{battle_str} {ot_str} {pp_str}"
                                row_str += f"{combined:>20}"
                            else:
                                row_str += f"{'—':^20}"
                        else:
                            row_str += f"{'—':^20}"
                    
                    print(row_str)
                
                if len(all_years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history from {min(all_years)} to {max(all_years)}")
                print("\n* Format: Wins-Losses (Net Overtakes) [PrizePicks Points]")
            
            # Career stats
            overall_stats = track_teammate['overall_stats']
            if not overall_stats.empty:
                # Get active drivers from past and current season
                active_drivers = self.get_active_drivers()
                active_driver_ids = set(active_drivers['id'].unique())
                
                # Filter overall stats to only include active drivers
                overall_stats_filtered = overall_stats[overall_stats.index.isin(active_driver_ids)]
                
                if not overall_stats_filtered.empty:
                    print("\n\nCareer Teammate Statistics at this Track (Active Drivers Only):")
                    print("-" * 100)
                    print(f"{'Driver':<25} {'Battles':<10} {'Wins':<8} {'Win Rate':<10} "
                          f"{'Net OT':<10} {'PP Points':<10}")
                    print("-" * 100)
                    
                    overall_sorted = overall_stats_filtered.sort_values('win_rate', ascending=False)
                    
                    for driver_id, row in overall_sorted.iterrows():
                        driver_name = row.get('driver_name', driver_id)
                        print(f"{driver_name:<25} {row['total_battles']:<10.0f} {row['total_wins']:<8.0f} "
                              f"{row['win_rate']:<10.1%} {row['net_overtakes']:+10.0f} "
                              f"{row['career_prizepicks_points']:+10.1f}")
                else:
                    print("\n\nNo career teammate statistics at this track for active drivers")
        else:
            print("No teammate battle data available for this circuit")
        
        # 7. Fastest Lap Analysis
        print("\n" + "="*80)
        print("7. FASTEST LAP ANALYSIS")
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
        
        # 7b. Track-specific fastest laps
        print("\n" + "-"*80)
        print("7b. TRACK-SPECIFIC FASTEST LAPS BY YEAR")
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
        
        
        print("\n" + "="*80)
        print("ANALYSIS NOTES:")
        print("="*80)
        print("- All statistics based on real F1 data (no synthetic data)")
        print("- 'next_circuit_avg' shows historical performance at the upcoming race circuit")
        print("- Median values provide insight into typical performance (less affected by outliers)")
        print("- Data includes races from the last 3 years for relevance")
        print("- Teammate overtake scoring: +1.5 for overtaking teammate, -1.5 for being overtaken")
        print("- DHL pit stop data: Official DHL fastest pit stop competition results")
        
        return {
            'overtakes': overtakes,
            'points': points,
            'pit_stops': pit_stops,
            'dhl_pit_stops': dhl_stops,
            'starting_positions': grid,
            'sprint_points': sprint,
            'teammate_overtakes': teammate,
            'fastest_laps': fastest
        }


def test_analyzer():
    """Test the analyzer with sample data"""
    from f1db_data_loader import F1DBDataLoader
    
    # Load data using F1DBDataLoader with no update check for tests
    loader = F1DBDataLoader()
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