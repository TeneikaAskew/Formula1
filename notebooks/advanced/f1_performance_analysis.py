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
            return pd.DataFrame()  # Return DataFrame not list
            
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
            return pd.DataFrame()  # Return DataFrame not list
            
        # Get unique driver IDs
        driver_ids = year_results['driverId'].unique()
        
        # Get driver details
        drivers = self.data.get('drivers', pd.DataFrame())
        if drivers.empty:
            return pd.DataFrame()  # Return DataFrame not list
            
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
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
        # Ensure year column exists
        if 'year' not in circuit_data.columns:
            print("Warning: No year column found in circuit data")
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
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
            
            # Show year-by-year data
            year_data = track_year_overtakes['year_by_year']
            if not year_data.empty:
                print("\nYear-by-Year Performance:")
                print("-" * 120)
                # Sort by driver and year for better readability
                year_data_sorted = year_data.sort_values(['driver_name', 'year'])
                
                # Group by driver to show their progression
                for driver_id in year_data_sorted['driverId'].unique():
                    driver_data = year_data_sorted[year_data_sorted['driverId'] == driver_id]
                    driver_name = driver_data.iloc[0].get('driver_name', driver_id)
                    print(f"\n{driver_name}:")
                    
                    # Show year-by-year stats
                    for _, row in driver_data.iterrows():
                        print(f"  {int(row['year'])}: "
                              f"Overtakes: {row['total_overtakes']:.0f} total, "
                              f"{row['avg_overtakes']:.1f} avg, "
                              f"{row['median_overtakes']:.0f} median | "
                              f"Start: {row['avg_start_pos']:.1f} | "
                              f"Finish: {row['avg_finish_pos']:.1f} | "
                              f"Avg Change: {row['avg_pos_change']:+.1f}")
            
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
        
        print("\n" + "="*80)
        print("ANALYSIS NOTES:")
        print("="*80)
        print("- All statistics based on real F1 data (no synthetic data)")
        print("- 'next_circuit_avg' shows historical performance at the upcoming race circuit")
        print("- Median values provide insight into typical performance (less affected by outliers)")
        print("- Data includes races from the last 3 years for relevance")
        
        return {
            'overtakes': overtakes,
            'points': points,
            'pit_stops': pit_stops,
            'starting_positions': grid,
            'sprint_points': sprint
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