#!/usr/bin/env python3
"""
Lap-by-lap overtake analyzer using Jolpica lap data.

This module analyzes real overtakes by comparing driver positions 
between consecutive laps, providing accurate overtake statistics.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LapByLapOvertakeAnalyzer:
    """Analyzes overtakes using lap-by-lap position data"""
    
    def __init__(self, jolpica_data_dir: str = None):
        """
        Initialize the overtake analyzer
        
        Args:
            jolpica_data_dir: Path to jolpica data directory
        """
        if jolpica_data_dir is None:
            # Auto-detect data directory
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            self.data_dir = project_root / 'data' / 'jolpica' / 'laps'
        else:
            self.data_dir = Path(jolpica_data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Jolpica lap data directory not found: {self.data_dir}")
        
        print(f"Initialized LapByLapOvertakeAnalyzer with data from: {self.data_dir}")
    
    def load_race_data(self, year: int, round_num: int) -> Optional[Dict]:
        """Load lap data for a specific race"""
        race_file = self.data_dir / str(year) / f"{year}_round_{round_num:02d}_laps.json"
        
        if not race_file.exists():
            return None
        
        try:
            with open(race_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {race_file}: {e}")
            return None
    
    def calculate_race_overtakes(self, race_data: Dict) -> pd.DataFrame:
        """
        Calculate overtakes for a single race
        
        Returns DataFrame with columns:
        - driverId: driver identifier
        - overtakes_made: number of overtakes made
        - overtaken_by: number of times overtaken
        - laps_completed: number of laps completed
        - race_details: race info
        """
        if not race_data or 'laps' not in race_data:
            return pd.DataFrame()
        
        # Initialize tracking
        overtakes_made = {}
        overtaken_by = {}
        laps_completed = {}
        
        # Process each lap
        previous_positions = {}
        
        for lap_data in race_data['laps']:
            lap_num = int(lap_data['number'])
            current_positions = {}
            
            # Get current lap positions
            for timing in lap_data['Timings']:
                driver_id = timing['driverId']
                position = int(timing['position'])
                current_positions[driver_id] = position
                
                # Track laps completed
                laps_completed[driver_id] = lap_num
            
            # Compare with previous lap to find overtakes
            if previous_positions:
                self._detect_overtakes_between_laps(
                    previous_positions, current_positions, 
                    overtakes_made, overtaken_by
                )
            
            previous_positions = current_positions.copy()
        
        # Create results DataFrame
        all_drivers = set(list(overtakes_made.keys()) + list(overtaken_by.keys()) + list(laps_completed.keys()))
        
        results = []
        for driver_id in all_drivers:
            results.append({
                'driverId': driver_id,
                'overtakes_made': overtakes_made.get(driver_id, 0),
                'overtaken_by': overtaken_by.get(driver_id, 0),
                'laps_completed': laps_completed.get(driver_id, 0),
                'season': race_data.get('season'),
                'round': race_data.get('round'),
                'raceName': race_data.get('raceName'),
                'circuitId': race_data.get('circuitId'),
                'date': race_data.get('date')
            })
        
        return pd.DataFrame(results)
    
    def _detect_overtakes_between_laps(self, prev_pos: Dict, curr_pos: Dict, 
                                     overtakes_made: Dict, overtaken_by: Dict):
        """Detect overtakes between two consecutive laps"""
        
        # Find drivers who improved position (lower number = better position)
        for driver_id in curr_pos:
            if driver_id not in prev_pos:
                continue  # Driver wasn't in previous lap (DNF/DNS)
            
            prev_position = prev_pos[driver_id]
            curr_position = curr_pos[driver_id]
            
            # Driver improved position (moved up)
            if curr_position < prev_position:
                positions_gained = prev_position - curr_position
                
                # Initialize counters
                if driver_id not in overtakes_made:
                    overtakes_made[driver_id] = 0
                
                # Count overtakes (each position gained = 1 overtake)
                overtakes_made[driver_id] += positions_gained
                
                # Find who was overtaken
                for other_driver in prev_pos:
                    if other_driver == driver_id or other_driver not in curr_pos:
                        continue
                    
                    other_prev = prev_pos[other_driver]
                    other_curr = curr_pos[other_driver]
                    
                    # If the other driver was ahead before and behind after
                    if (other_prev < prev_position and other_curr > curr_position):
                        if other_driver not in overtaken_by:
                            overtaken_by[other_driver] = 0
                        overtaken_by[other_driver] += 1
    
    def analyze_season_overtakes(self, year: int) -> pd.DataFrame:
        """Analyze overtakes for entire season"""
        season_data = []
        
        # Get all race files for the year
        year_dir = self.data_dir / str(year)
        if not year_dir.exists():
            print(f"No data found for year {year}")
            return pd.DataFrame()
        
        race_files = sorted(year_dir.glob(f"{year}_round_*_laps.json"))
        
        for race_file in race_files:
            # Extract round number from filename
            round_num = int(race_file.stem.split('_')[2])
            
            race_data = self.load_race_data(year, round_num)
            if race_data:
                race_overtakes = self.calculate_race_overtakes(race_data)
                if not race_overtakes.empty:
                    season_data.append(race_overtakes)
        
        if not season_data:
            return pd.DataFrame()
        
        # Combine all races
        return pd.concat(season_data, ignore_index=True)
    
    def analyze_multi_season_overtakes(self, years: List[int]) -> pd.DataFrame:
        """Analyze overtakes across multiple seasons"""
        all_data = []
        
        for year in years:
            print(f"Processing {year} season...")
            season_data = self.analyze_season_overtakes(year)
            if not season_data.empty:
                all_data.append(season_data)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_driver_overtake_summary(self, overtake_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create driver summary from overtake data
        
        Returns DataFrame with:
        - driverId
        - total_overtakes_made
        - total_overtaken_by  
        - net_overtakes
        - races_participated
        - avg_overtakes_per_race
        - best_overtake_race
        - seasons_active
        """
        if overtake_data.empty:
            return pd.DataFrame()
        
        summary = overtake_data.groupby('driverId').agg({
            'overtakes_made': ['sum', 'mean', 'max'],
            'overtaken_by': ['sum', 'mean'],
            'round': 'count',  # races participated
            'season': ['nunique', lambda x: list(x.unique())],
            'raceName': lambda x: x.iloc[x.values.argmax()] if len(x) > 0 else None  # best race
        }).round(2)
        
        # Flatten column names
        summary.columns = [
            'total_overtakes_made', 'avg_overtakes_per_race', 'max_overtakes_single_race',
            'total_overtaken_by', 'avg_overtaken_per_race',
            'races_participated', 'seasons_active', 'seasons_list',
            'best_overtake_race_name'
        ]
        
        # Calculate net overtakes
        summary['net_overtakes'] = summary['total_overtakes_made'] - summary['total_overtaken_by']
        
        # Reset index to make driverId a column
        summary = summary.reset_index()
        
        # Sort by total overtakes made
        summary = summary.sort_values('total_overtakes_made', ascending=False)
        
        return summary
    
    def get_circuit_overtake_analysis(self, overtake_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze overtakes by circuit"""
        if overtake_data.empty:
            return pd.DataFrame()
        
        circuit_analysis = overtake_data.groupby(['circuitId', 'raceName']).agg({
            'overtakes_made': ['sum', 'mean', 'max'],
            'driverId': 'nunique',  # number of drivers
            'season': lambda x: list(x.unique())
        }).round(2)
        
        circuit_analysis.columns = [
            'total_overtakes', 'avg_overtakes_per_driver', 'max_overtakes_single_driver',
            'num_drivers', 'seasons'
        ]
        
        # Calculate overtakes per race
        circuit_analysis['races_analyzed'] = circuit_analysis['seasons'].apply(len)
        circuit_analysis['avg_overtakes_per_race'] = (
            circuit_analysis['total_overtakes'] / circuit_analysis['races_analyzed']
        ).round(2)
        
        # Reset index
        circuit_analysis = circuit_analysis.reset_index()
        
        # Sort by average overtakes per race
        circuit_analysis = circuit_analysis.sort_values('avg_overtakes_per_race', ascending=False)
        
        return circuit_analysis


def test_overtake_analyzer():
    """Test the overtake analyzer with sample data"""
    analyzer = LapByLapOvertakeAnalyzer()
    
    # Test single race
    print("Testing single race analysis...")
    race_data = analyzer.load_race_data(2025, 1)
    if race_data:
        race_overtakes = analyzer.calculate_race_overtakes(race_data)
        print(f"✓ Race: {race_data['raceName']}")
        print(f"✓ Total overtakes in race: {race_overtakes['overtakes_made'].sum()}")
        print(f"✓ Top overtaker: {race_overtakes.loc[race_overtakes['overtakes_made'].idxmax(), 'driverId']} ({race_overtakes['overtakes_made'].max()} overtakes)")
    
    # Test season analysis
    print("\nTesting season analysis...")
    season_data = analyzer.analyze_season_overtakes(2025)
    if not season_data.empty:
        print(f"✓ Processed {season_data['round'].nunique()} races")
        
        # Driver summary
        driver_summary = analyzer.get_driver_overtake_summary(season_data)
        print(f"✓ Top 3 overtakers:")
        for i, (_, row) in enumerate(driver_summary.head(3).iterrows()):
            print(f"  {i+1}. {row['driverId']}: {row['total_overtakes_made']} overtakes in {row['races_participated']} races")
    
    return analyzer, season_data


if __name__ == "__main__":
    test_overtake_analyzer()