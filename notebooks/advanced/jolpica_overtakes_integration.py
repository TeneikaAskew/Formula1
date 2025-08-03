#!/usr/bin/env python3
"""
Jolpica Overtakes Integration

Integrates Jolpica lap-by-lap timing data with F1 performance analysis
to calculate true on-track overtakes instead of just net position changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class JolpicaOvertakesAnalyzer:
    """Analyze overtakes using Jolpica lap timing data"""
    
    def __init__(self, jolpica_data_path: str = "../../data/jolpica/all_laps_2020_2024.parquet"):
        """Initialize with Jolpica laps data"""
        self.jolpica_path = Path(jolpica_data_path)
        self.laps_data = None
        
        if self.jolpica_path.exists():
            self.load_jolpica_data()
        else:
            logger.warning(f"Jolpica data not found at {jolpica_data_path}")
    
    def load_jolpica_data(self):
        """Load Jolpica laps data"""
        try:
            self.laps_data = pd.read_parquet(self.jolpica_path)
            logger.info(f"Loaded {len(self.laps_data)} lap records from Jolpica")
        except Exception as e:
            logger.error(f"Error loading Jolpica data: {e}")
            self.laps_data = None
    
    def get_race_overtakes(self, season: int, round_num: int) -> pd.DataFrame:
        """Get all overtakes for a specific race"""
        if self.laps_data is None:
            return pd.DataFrame()
        
        # Filter to specific race
        race_laps = self.laps_data[
            (self.laps_data['season'] == season) & 
            (self.laps_data['round'] == round_num)
        ].copy()
        
        if race_laps.empty:
            logger.warning(f"No lap data found for {season} Round {round_num}")
            return pd.DataFrame()
        
        # Sort by lap and position
        race_laps = race_laps.sort_values(['lap', 'position'])
        
        overtakes = []
        
        # Analyze each lap transition
        for lap_num in range(2, race_laps['lap'].max() + 1):
            prev_lap = race_laps[race_laps['lap'] == lap_num - 1]
            curr_lap = race_laps[race_laps['lap'] == lap_num]
            
            # Create position dictionaries
            prev_positions = dict(zip(prev_lap['driverId'], prev_lap['position']))
            curr_positions = dict(zip(curr_lap['driverId'], curr_lap['position']))
            
            # Find overtakes
            for driver in curr_positions:
                if driver in prev_positions:
                    prev_pos = prev_positions[driver]
                    curr_pos = curr_positions[driver]
                    
                    if curr_pos < prev_pos:
                        # Driver improved position - who did they overtake?
                        for other_driver in prev_positions:
                            if other_driver != driver and other_driver in curr_positions:
                                other_prev = prev_positions[other_driver]
                                other_curr = curr_positions[other_driver]
                                
                                # Check if this driver overtook the other
                                if (prev_pos > other_prev and curr_pos < other_curr):
                                    overtakes.append({
                                        'season': season,
                                        'round': round_num,
                                        'lap': lap_num,
                                        'overtaking_driver': driver,
                                        'overtaken_driver': other_driver,
                                        'overtaking_from_pos': prev_pos,
                                        'overtaking_to_pos': curr_pos,
                                        'overtaken_from_pos': other_prev,
                                        'overtaken_to_pos': other_curr
                                    })
        
        return pd.DataFrame(overtakes)
    
    def get_driver_overtake_stats(self, driver_id: str, season: Optional[int] = None) -> Dict:
        """Get overtaking statistics for a specific driver"""
        if self.laps_data is None:
            return {}
        
        # Filter by season if specified
        if season:
            season_laps = self.laps_data[self.laps_data['season'] == season]
        else:
            season_laps = self.laps_data
        
        # Get unique races
        races = season_laps[['season', 'round']].drop_duplicates()
        
        total_overtakes_made = 0
        total_overtakes_received = 0
        overtakes_by_circuit = {}
        
        for _, race in races.iterrows():
            race_overtakes = self.get_race_overtakes(race['season'], race['round'])
            
            if not race_overtakes.empty:
                # Count overtakes made
                made = len(race_overtakes[race_overtakes['overtaking_driver'] == driver_id])
                total_overtakes_made += made
                
                # Count times overtaken
                received = len(race_overtakes[race_overtakes['overtaken_driver'] == driver_id])
                total_overtakes_received += received
                
                # Get circuit from race data
                race_info = season_laps[
                    (season_laps['season'] == race['season']) & 
                    (season_laps['round'] == race['round'])
                ].iloc[0]
                circuit = race_info['circuitId']
                
                if circuit not in overtakes_by_circuit:
                    overtakes_by_circuit[circuit] = {'made': 0, 'received': 0}
                
                overtakes_by_circuit[circuit]['made'] += made
                overtakes_by_circuit[circuit]['received'] += received
        
        return {
            'driver': driver_id,
            'season': season,
            'total_overtakes_made': total_overtakes_made,
            'total_overtakes_received': total_overtakes_received,
            'net_overtakes': total_overtakes_made - total_overtakes_received,
            'races_analyzed': len(races),
            'avg_overtakes_per_race': total_overtakes_made / len(races) if len(races) > 0 else 0,
            'overtakes_by_circuit': overtakes_by_circuit
        }
    
    def compare_with_position_changes(self, f1db_results: pd.DataFrame) -> pd.DataFrame:
        """Compare actual overtakes with net position changes from F1DB"""
        if self.laps_data is None:
            return pd.DataFrame()
        
        comparison = []
        
        # Get unique races from F1DB results
        races = f1db_results[['season', 'round']].drop_duplicates()
        
        for _, race in races.iterrows():
            # Get race results
            race_results = f1db_results[
                (f1db_results['season'] == race['season']) & 
                (f1db_results['round'] == race['round'])
            ]
            
            # Get overtakes from Jolpica
            race_overtakes = self.get_race_overtakes(race['season'], race['round'])
            
            if not race_overtakes.empty:
                for _, result in race_results.iterrows():
                    driver = result['driverId']
                    
                    # Net position change
                    net_change = result.get('gridPosition', 20) - result.get('positionNumber', 20)
                    
                    # Actual overtakes
                    overtakes_made = len(race_overtakes[
                        race_overtakes['overtaking_driver'] == driver
                    ])
                    overtakes_received = len(race_overtakes[
                        race_overtakes['overtaken_driver'] == driver
                    ])
                    
                    comparison.append({
                        'season': race['season'],
                        'round': race['round'],
                        'driver': driver,
                        'net_position_change': net_change,
                        'actual_overtakes_made': overtakes_made,
                        'actual_overtakes_received': overtakes_received,
                        'actual_net_overtakes': overtakes_made - overtakes_received
                    })
        
        return pd.DataFrame(comparison)
    
    def get_overtaking_hotspots(self, season: int, round_num: int) -> pd.DataFrame:
        """Identify laps/sections with most overtaking activity"""
        overtakes = self.get_race_overtakes(season, round_num)
        
        if overtakes.empty:
            return pd.DataFrame()
        
        # Count overtakes by lap
        lap_counts = overtakes.groupby('lap').size().reset_index(name='overtake_count')
        lap_counts = lap_counts.sort_values('overtake_count', ascending=False)
        
        # Add percentage
        total_overtakes = len(overtakes)
        lap_counts['percentage'] = (lap_counts['overtake_count'] / total_overtakes * 100).round(1)
        
        return lap_counts


def integrate_jolpica_with_f1_analysis(f1_performance_analyzer):
    """
    Integrate Jolpica overtakes data with F1PerformanceAnalyzer
    
    Args:
        f1_performance_analyzer: Instance of F1PerformanceAnalyzer
    
    Returns:
        Enhanced overtakes analysis DataFrame
    """
    # Initialize Jolpica analyzer
    jolpica = JolpicaOvertakesAnalyzer()
    
    if jolpica.laps_data is None:
        logger.warning("Jolpica data not available, using position-based analysis only")
        return f1_performance_analyzer.analyze_overtakes()
    
    # Get base overtakes analysis
    position_analysis = f1_performance_analyzer.analyze_overtakes()
    
    # Get F1DB results data for comparison
    results = f1_performance_analyzer.data.get('results', pd.DataFrame())
    
    if not results.empty:
        # Add actual overtakes data
        driver_stats = []
        
        for driver in position_analysis['driver_name'].unique():
            # Get driver ID (might need mapping)
            driver_id = driver.lower().replace(' ', '_')
            
            # Get current season stats
            current_year = pd.Timestamp.now().year
            stats = jolpica.get_driver_overtake_stats(driver_id, current_year)
            
            if stats['races_analyzed'] > 0:
                driver_stats.append({
                    'driver_name': driver,
                    'actual_overtakes_made': stats['total_overtakes_made'],
                    'actual_overtakes_received': stats['total_overtakes_received'],
                    'actual_net_overtakes': stats['net_overtakes'],
                    'jolpica_races': stats['races_analyzed']
                })
        
        if driver_stats:
            jolpica_df = pd.DataFrame(driver_stats)
            
            # Merge with position analysis
            enhanced_analysis = position_analysis.merge(
                jolpica_df, 
                on='driver_name', 
                how='left'
            )
            
            # Add comparison columns
            enhanced_analysis['overtake_efficiency'] = (
                enhanced_analysis['actual_overtakes_made'] / 
                enhanced_analysis['total_pos_gained'].clip(lower=1)
            ).fillna(0).round(2)
            
            logger.info("Successfully integrated Jolpica overtakes data")
            return enhanced_analysis
    
    return position_analysis


# Example usage
if __name__ == "__main__":
    # Test the integration
    analyzer = JolpicaOvertakesAnalyzer()
    
    # Example: Get overtakes for 2020 Austrian GP
    if analyzer.laps_data is not None:
        print(f"Loaded {len(analyzer.laps_data)} lap records")
        
        # Test with 2020 Austrian GP (Round 1)
        overtakes = analyzer.get_race_overtakes(2020, 1)
        print(f"\n2020 Austrian GP Overtakes: {len(overtakes)}")
        
        if not overtakes.empty:
            print("\nTop 5 overtaking drivers:")
            top_overtakers = overtakes['overtaking_driver'].value_counts().head()
            print(top_overtakers)
            
            print("\nMost overtaken drivers:")
            most_overtaken = overtakes['overtaken_driver'].value_counts().head()
            print(most_overtaken)
            
            print("\nOvertaking hotspots (by lap):")
            hotspots = analyzer.get_overtaking_hotspots(2020, 1)
            print(hotspots.head(10))
        
        # Test driver stats
        print("\nHamilton 2020 stats:")
        hamilton_stats = analyzer.get_driver_overtake_stats('hamilton', 2020)
        for key, value in hamilton_stats.items():
            if key != 'overtakes_by_circuit':
                print(f"  {key}: {value}")
    else:
        print("No Jolpica data available. Run jolpica_laps_fetcher.py first.")