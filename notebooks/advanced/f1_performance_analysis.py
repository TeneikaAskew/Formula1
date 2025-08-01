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
        
    def get_next_race(self):
        """Get the next upcoming race"""
        races = self.data.get('races', pd.DataFrame()).copy()
        if races.empty or 'date' not in races.columns:
            return None
            
        races['date'] = pd.to_datetime(races['date'])
        # Get races after today's actual date
        upcoming = races[races['date'] > datetime.now()].sort_values('date')
        
        if upcoming.empty:
            # If no future races, get the most recent
            return races.sort_values('date').iloc[-1]
        
        return upcoming.iloc[0]
    
    def get_active_drivers(self, year=None):
        """Get list of active drivers for a given year"""
        if year is None:
            year = self.current_year
            
        results = self.data.get('results', pd.DataFrame())  # Fixed key name
        if results.empty:
            return {}  # Return DataFrame not list
            
        # Get drivers who raced in the specified year
        if 'year' in results.columns:
            year_results = results[results['year'] == year]
        else:
            # Try to join with races to get year
            races = self.data.get('races', pd.DataFrame())
            if not races.empty and 'id' in races.columns and 'raceId' in results.columns:
                results_with_year = results.merge(races[['id', 'year']], left_on='raceId', right_on='id', how='left')
                year_results = results_with_year[results_with_year['year'] == year]
            else:
                # Fallback: use most recent results
                year_results = results.tail(1000)
        
        if year_results.empty:
            return {}  # Return DataFrame not list
            
        # Get unique driver IDs
        driver_ids = year_results['driverId'].unique()
        
        # Get driver details
        drivers = self.data.get('drivers', pd.DataFrame())
        if drivers.empty:
            return {}  # Return DataFrame not list
            
        active_drivers = drivers[drivers['id'].isin(driver_ids)]
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
        results = self.data.get('results', pd.DataFrame()).copy()
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame()).copy()
        
        if results.empty or grid.empty:
            return {}
        
        # Merge results with starting grid
        overtake_data = results.merge(
            grid[['raceId', 'driverId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Calculate position gained (negative means overtakes made)
        overtake_data['positions_gained'] = overtake_data['gridPosition'] - overtake_data['positionNumber']
        
        # Add year information if not present
        if 'year' not in overtake_data.columns:
            races = self.data.get('races', pd.DataFrame())
            if not races.empty:
                overtake_data = overtake_data.merge(races[['id', 'year', 'circuitId']], left_on='raceId', right_on='id', how='left')
        
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
        
        driver_overtakes.columns = ['total_positions_gained', 'avg_positions_gained', 
                                   'median_positions_gained', 'races', 'total_points', 'avg_points']
        
        # Calculate overtakes (only positive position gains)
        recent_data['overtakes'] = recent_data['positions_gained'].apply(lambda x: max(0, x))
        overtakes_by_driver = recent_data.groupby('driverId')['overtakes'].agg(['sum', 'mean', 'median']).round(2)
        overtakes_by_driver.columns = ['total_overtakes', 'avg_overtakes', 'median_overtakes']
        
        # Combine data
        final_data = driver_overtakes.join(overtakes_by_driver)
        
        # Keep driverId as index for proper identification
        
        # Filter to only current season drivers
        final_data = self.filter_current_season_drivers(final_data)
        
        # Add circuit-specific prediction for next race
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_data = recent_data  # Skip circuit filtering for now
            if not circuit_data.empty:
                circuit_overtakes = circuit_data.groupby('driverId')['overtakes'].mean().round(2)
                final_data['next_circuit_avg'] = final_data.index.map(
                    lambda x: circuit_overtakes.get(x, final_data.loc[x, 'avg_overtakes'] if x in final_data.index else 0)
                )
        
        return final_data
    
    def analyze_overtakes_by_track_year(self):
        """Analyze overtakes by driver for each track, broken down by year"""
        results = self.data.get('results', pd.DataFrame()).copy()
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
            'total_overtakes', 'avg_overtakes', 'median_overtakes',
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
            'career_overtakes', 'career_avg_overtakes', 'career_median_overtakes',
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
        results = self.data.get('results', pd.DataFrame()).copy()
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
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_results = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_results.empty:
                circuit_points = circuit_results.groupby('driverId')['points'].mean().round(2)
                points_analysis['next_circuit_avg'] = points_analysis.index.map(
                    lambda x: circuit_points.get(x, points_analysis.loc[x, 'avg_points'] if x in points_analysis.index else 0)
                )
        
        return points_analysis
    
    def analyze_points_by_track_year(self):
        """Analyze points by driver for each track, broken down by year"""
        results = self.data.get('results', pd.DataFrame()).copy()
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
            return {}
        
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
            return {}
        
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
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_stops = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_stops.empty:
                circuit_times = circuit_stops.groupby('driverId')['time_seconds'].mean().round(3)
                pit_analysis['next_circuit_avg'] = pit_analysis.index.map(
                    lambda x: circuit_times.get(x, pit_analysis.loc[x, 'avg_stop_time'] if x in pit_analysis.index else 0)
                )
        
        return pit_analysis
    
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
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_grid = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_grid.empty:
                circuit_positions = circuit_grid.groupby('driverId')['position'].mean().round(2)
                grid_analysis['next_circuit_avg'] = grid_analysis.index.map(
                    lambda x: circuit_positions.get(x, grid_analysis.loc[x, 'avg_start_position'] if x in grid_analysis.index else 0)
                )
        
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
            return {}
        
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
            return {}
        
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
        
        # Circuit-specific prediction
        next_race = self.get_next_race()
        if next_race is not None and 'circuitId' in next_race:
            circuit_sprints = pd.DataFrame()  # Skip circuit filtering for now
            if not circuit_sprints.empty:
                circuit_sprint_points = circuit_sprints.groupby('driverId')['points'].mean().round(2)
                sprint_analysis['next_circuit_avg'] = sprint_analysis.index.map(
                    lambda x: circuit_sprint_points.get(x, sprint_analysis.loc[x, 'avg_sprint_points'] if x in sprint_analysis.index else 0)
                )
        
        return sprint_analysis
    
    def analyze_sprint_points_by_track_year(self):
        """Analyze sprint points by driver for each track, broken down by year"""
        sprint_results = self.data.get('sprint_results', pd.DataFrame()).copy()
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if sprint_results.empty or races.empty:
            return {}
        
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
            driver_list = ', '.join(current_drivers['id'].head(20).tolist())
            if len(current_drivers) > 20:
                driver_list += f" and {len(current_drivers) - 20} more..."
            print(f"Driver IDs: {driver_list}")
        print(f"Current Analysis Year: {self.current_year}")
        print(f"Years Analyzed: {self.current_year-3} to {self.current_year}")
        
        next_race = self.get_next_race()
        if next_race is not None:
            circuits = self.data.get('circuits', pd.DataFrame())
            circuit_name = 'Unknown'
            if not circuits.empty and 'circuitId' in next_race:
                circuit = circuits[circuits['id'] == next_race['circuitId']]
                if not circuit.empty:
                    circuit_name = circuit.iloc[0]['name']
            
            print(f"Next Race: {next_race.get('officialName', 'Unknown')} at {circuit_name}")
            print(f"Date: {next_race['date'].strftime('%Y-%m-%d') if pd.notna(next_race.get('date')) else 'Unknown'}")
        
        # 1. Overtakes Analysis
        print("\n" + "="*80)
        print("1. OVERTAKES BY DRIVER (Position Changes)")
        print("="*80)
        overtakes = self.analyze_overtakes()
        if not overtakes.empty:
            display_cols = ['total_overtakes', 'avg_overtakes', 'median_overtakes', 'avg_points', 'races']
            if 'next_circuit_avg' in overtakes.columns:
                display_cols.append('next_circuit_avg')
            print(overtakes[display_cols].head(20).to_string())
            
            # Show drivers with notable statistics
            print("\nNotable insights:")
            if 'total_overtakes' in overtakes.columns:
                top_overtakers = overtakes.nlargest(5, 'total_overtakes')
                print(f"Top 5 overtakers: {', '.join(top_overtakers.index.tolist())}")
                
                # Drivers with negative average (lost positions)
                lost_positions = overtakes[overtakes['avg_positions_gained'] < 0]
                if not lost_positions.empty:
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
                        driver_sort_data.append((driver_id, most_recent['avg_overtakes']))
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
                overall_sorted = overall_stats.sort_values('career_avg_overtakes', ascending=False)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['races_at_track']:<8.0f} "
                          f"{row['career_overtakes']:<10.0f} {row['career_avg_overtakes']:<8.1f} "
                          f"{row['career_median_overtakes']:<8.0f} {row['career_avg_start']:<10.1f} "
                          f"{row['career_avg_finish']:<12.1f} {row['career_avg_pos_change']:+10.1f}")
        else:
            print("No track-specific overtake data available for the next race")
        
        # 2. Points Analysis
        print("\n" + "="*80)
        print("2. F1 POINTS BY DRIVER")
        print("="*80)
        points = self.analyze_points()
        if not points.empty:
            display_cols = ['total_points', 'avg_points', 'median_points', 'races']
            if 'hist_avg_points' in points.columns:
                display_cols.append('hist_avg_points')
            if 'next_circuit_avg' in points.columns:
                display_cols.append('next_circuit_avg')
            print(points[display_cols].head(20).to_string())
            
            # Explain any zeros
            if not points.empty:
                zero_points = points[points['total_points'] == 0]
                if not zero_points.empty:
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
            display_cols = ['avg_stop_time', 'median_stop_time', 'best_stop_time', 'total_stops']
            if 'next_circuit_avg' in pit_stops.columns:
                display_cols.append('next_circuit_avg')
            print(pit_stops[display_cols].head(20).to_string())
            
            # Add pit stop explanations
            if not pit_stops.empty:
                explanations = self.explain_zero_or_nan_values(pit_stops, 'pit_stops')
                print(explanations)
        else:
            print("No pit stop data available")
        
        # 4. Starting Position Analysis
        print("\n" + "="*80)
        print("4. STARTING POSITIONS BY DRIVER")
        print("="*80)
        grid = self.analyze_starting_positions()
        if not grid.empty:
            display_cols = ['avg_start_position', 'median_start_position', 'best_start_position']
            if 'avg_points_per_race' in grid.columns:
                display_cols.append('avg_points_per_race')
            if 'next_circuit_avg' in grid.columns:
                display_cols.append('next_circuit_avg')
            print(grid[display_cols].head(20).to_string())
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
                            avg_start = f"{row['avg_start']:.1f}"
                            # Check if we have points data
                            if 'avg_points' in row:
                                pts_per_race = f"{row['avg_points']:.1f}"
                                row_str += f"{avg_start:>7}  {pts_per_race:>11}    "
                            else:
                                row_str += f"{avg_start:>7}  {'—':>11}    "
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
            display_cols = ['total_sprint_points', 'avg_sprint_points', 'median_sprint_points', 'sprint_races']
            if 'next_circuit_avg' in sprint.columns:
                display_cols.append('next_circuit_avg')
            print(sprint[display_cols].head(20).to_string())
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
            display_cols = ['total_battles', 'wins', 'losses', 
                          'win_rate', 'OT_made', 'OT_received',
                          'net_OT', 'prizepicks_pts']
            print(teammate[display_cols].head(20).to_string())
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
                            
                            # Format: W-L (OT) [PP]
                            # Example: 1-0 (+1) [+1.5]
                            battle_str = f"{wins}-{losses}"
                            ot_str = f"({net_ot:+d})"
                            pp_str = f"[{pp_pts:+.1f}]"
                            
                            combined = f"{battle_str} {ot_str} {pp_str}"
                            row_str += f"{combined:>20}"
                        else:
                            row_str += f"{'—':^20}"
                    
                    print(row_str)
                
                if len(all_years) > len(recent_years):
                    print(f"\n* Showing last {len(recent_years)} years. Full history from {min(all_years)} to {max(all_years)}")
                print("\n* Format: Wins-Losses (Net Overtakes) [PrizePicks Points]")
            
            # Career stats
            overall_stats = track_teammate['overall_stats']
            if not overall_stats.empty:
                print("\n\nCareer Teammate Statistics at this Track:")
                print("-" * 100)
                print(f"{'Driver':<25} {'Battles':<10} {'Wins':<8} {'Win Rate':<10} "
                      f"{'Net OT':<10} {'PP Points':<10}")
                print("-" * 100)
                
                overall_sorted = overall_stats.sort_values('win_rate', ascending=False)
                
                for driver_id, row in overall_sorted.iterrows():
                    driver_name = row.get('driver_name', driver_id)
                    print(f"{driver_name:<25} {row['total_battles']:<10.0f} {row['total_wins']:<8.0f} "
                          f"{row['win_rate']:<10.1%} {row['net_overtakes']:+10.0f} "
                          f"{row['career_prizepicks_points']:+10.1f}")
        else:
            print("No teammate battle data available for this circuit")
        
        # 7. Fastest Lap Analysis
        print("\n" + "="*80)
        print("7. FASTEST LAP ANALYSIS")
        print("="*80)
        fastest = self.analyze_fastest_laps()
        if not fastest.empty:
            display_cols = ['total_fastest_laps', 'total_races', 'fastest_lap_rate', 
                          'avg_lap_number', 'seasons_with_fl', 'fastest_lap_points']
            print(fastest[display_cols].head(20).to_string())
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
                
                for driver_id, row in overall_sorted.head(15).iterrows():
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
        
        return {
            'overtakes': overtakes,
            'points': points,
            'pit_stops': pit_stops,
            'starting_positions': grid,
            'sprint_points': sprint,
            'teammate_overtakes': teammate,
            'fastest_laps': fastest
        }


def test_analyzer():
    """Test the analyzer with sample data"""
    from f1db_data_loader import load_f1db_data
    
    # Load data
    data = load_f1db_data()
    
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
#         overtakes_by_driver.columns = ['total_overtakes', 'avg_overtakes', 'median_overtakes']
        
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
#                     lambda x: circuit_overtakes.get(x, final_data.loc[x, 'avg_overtakes'] if x in final_data.index else 0)
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
#             print(overtakes[display_cols].head(20).to_string())
            
#             # Show drivers with notable statistics
#             print("\nNotable insights:")
#             if 'total_overtakes' in overtakes.columns:
#                 top_overtakers = overtakes.nlargest(5, 'total_overtakes')
#                 print(f"Top 5 overtakers: {', '.join(top_overtakers.index.tolist())}")
                
#                 # Drivers with negative average (lost positions)
#                 lost_positions = overtakes[overtakes['avg_positions_gained'] < 0]
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
#                 overall_sorted = overall_stats.sort_values('career_avg_overtakes', ascending=False)
                
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
#             print(points[display_cols].head(20).to_string())
            
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
#             print(pit_stops[display_cols].head(20).to_string())
            
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
#             print(grid[display_cols].head(20).to_string())
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
#             print(sprint[display_cols].head(20).to_string())
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