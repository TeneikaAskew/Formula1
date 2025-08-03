"""
F1 Fantasy Data Integration Module

This module provides functions to integrate F1 Fantasy data with the existing
F1 ML pipeline, adding fantasy-based features and validation metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger('F1FantasyIntegration')


class F1FantasyIntegration:
    """Integrates F1 Fantasy data with the main pipeline"""
    
    def __init__(self, fantasy_data_dir: str = None):
        # Always use absolute path to /data/f1_fantasy
        if fantasy_data_dir is None:
            # Get project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            self.fantasy_dir = project_root / 'data' / 'f1_fantasy'
        else:
            self.fantasy_dir = Path(fantasy_data_dir).resolve()
            
        self.driver_overview = None
        self.driver_details = None
        self._load_data()
    
    def _load_data(self):
        """Load F1 Fantasy data files"""
        overview_path = self.fantasy_dir / "driver_overview.csv"
        details_path = self.fantasy_dir / "driver_details.csv"
        
        if overview_path.exists():
            self.driver_overview = pd.read_csv(overview_path)
            logger.info(f"Loaded {len(self.driver_overview)} drivers from fantasy overview")
        else:
            logger.warning(f"Fantasy driver overview not found at {overview_path}")
        
        if details_path.exists():
            self.driver_details = pd.read_csv(details_path)
            logger.info(f"Loaded {len(self.driver_details)} race records from fantasy details")
        else:
            logger.warning(f"Fantasy driver details not found at {details_path}")
    
    def get_driver_fantasy_features(self, driver_name: str) -> Dict:
        """Get fantasy-based features for a specific driver"""
        if self.driver_overview is None:
            return {}
        
        # Find driver in overview
        driver_data = self.driver_overview[
            self.driver_overview['player_name'].str.contains(driver_name, case=False, na=False)
        ]
        
        if driver_data.empty:
            logger.warning(f"Driver {driver_name} not found in fantasy data")
            return {}
        
        driver = driver_data.iloc[0]
        
        # Extract key features
        features = {
            'fantasy_points_total': driver.get('fantasy_points', 0),
            'fantasy_points_avg': driver.get('fantasy_avg', 0),
            'fantasy_price': driver.get('current_price', 0),
            'fantasy_price_change': driver.get('price_change', 0),
            'fantasy_value_ratio': driver.get('points_per_million', 0),
            'fantasy_ownership_pct': driver.get('most_picked_pct', 0),
            'fantasy_podiums': driver.get('podiums', 0),
            'fantasy_dnfs': driver.get('dnfs', 0),
            'fantasy_overtake_points': driver.get('overtake_points', 0)
        }
        
        # Add recent form if available
        if self.driver_details is not None:
            recent_form = self._calculate_recent_form(driver['player_id'])
            features.update(recent_form)
        
        return features
    
    def _calculate_recent_form(self, player_id: str, last_n_races: int = 5) -> Dict:
        """Calculate recent form metrics for a driver"""
        driver_races = self.driver_details[
            self.driver_details['player_id'] == player_id
        ].sort_values('gameday_id', ascending=False).head(last_n_races)
        
        if driver_races.empty:
            return {}
        
        return {
            'fantasy_recent_avg': driver_races['total_points'].mean(),
            'fantasy_recent_std': driver_races['total_points'].std(),
            'fantasy_recent_trend': self._calculate_trend(driver_races['total_points'].values[::-1]),
            'fantasy_recent_consistency': 1 - (driver_races['total_points'].std() / 
                                             (driver_races['total_points'].mean() + 1e-6))
        }
    
    def _calculate_trend(self, points: np.ndarray) -> float:
        """Calculate trend coefficient (-1 to 1) for recent performance"""
        if len(points) < 2:
            return 0.0
        
        x = np.arange(len(points))
        # Simple linear regression
        coef = np.polyfit(x, points, 1)[0]
        # Normalize to -1 to 1 range
        return np.tanh(coef / 10)
    
    def get_all_drivers_features(self) -> pd.DataFrame:
        """Get fantasy features for all drivers"""
        if self.driver_overview is None:
            return pd.DataFrame()
        
        all_features = []
        
        for _, driver in self.driver_overview.iterrows():
            features = self.get_driver_fantasy_features(driver['player_name'])
            features['driver_name'] = driver['player_name']
            features['team_name'] = driver['team_name']
            all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def validate_predictions_with_fantasy(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Validate ML predictions against fantasy performance"""
        if self.driver_overview is None:
            logger.warning("No fantasy data available for validation")
            return predictions_df
        
        # Merge predictions with fantasy data
        validation_df = predictions_df.copy()
        
        # Add fantasy metrics for comparison
        for idx, row in validation_df.iterrows():
            driver_name = row.get('driver', row.get('driver_name', ''))
            fantasy_features = self.get_driver_fantasy_features(driver_name)
            
            if fantasy_features:
                # Add validation columns
                validation_df.loc[idx, 'fantasy_points_avg'] = fantasy_features.get('fantasy_points_avg', 0)
                validation_df.loc[idx, 'fantasy_value_ratio'] = fantasy_features.get('fantasy_value_ratio', 0)
                
                # Calculate alignment score (how well ML predictions align with fantasy performance)
                if 'predicted_points' in row:
                    fantasy_avg = fantasy_features.get('fantasy_points_avg', 0)
                    if fantasy_avg > 0:
                        alignment = 1 - abs(row['predicted_points'] - fantasy_avg) / fantasy_avg
                        validation_df.loc[idx, 'fantasy_alignment_score'] = max(0, alignment)
        
        return validation_df
    
    def get_value_drivers(self, max_price: float = 15.0, min_races: int = 3) -> pd.DataFrame:
        """Identify value drivers based on fantasy metrics"""
        if self.driver_overview is None or self.driver_details is None:
            return pd.DataFrame()
        
        # Filter by price
        value_candidates = self.driver_overview[
            self.driver_overview['current_price'] <= max_price
        ].copy()
        
        # Filter by minimum races
        race_counts = self.driver_details.groupby('player_id')['gameday_id'].count()
        valid_players = race_counts[race_counts >= min_races].index
        
        value_candidates = value_candidates[
            value_candidates['player_id'].isin(valid_players)
        ]
        
        # Calculate value score
        if 'points_per_million' in value_candidates.columns:
            value_candidates['value_score'] = (
                value_candidates['points_per_million'] * 0.4 +
                value_candidates['fantasy_avg'] / value_candidates['current_price'] * 0.3 +
                (100 - value_candidates.get('most_picked_pct', 50)) / 100 * 0.3  # Contrarian factor
            )
        else:
            value_candidates['value_score'] = (
                value_candidates['fantasy_avg'] / value_candidates['current_price']
            )
        
        # Sort by value score
        return value_candidates.nlargest(10, 'value_score')[
            ['player_name', 'team_name', 'current_price', 'fantasy_avg', 
             'points_per_million', 'value_score']
        ]
    
    def get_consistency_metrics(self) -> pd.DataFrame:
        """Calculate consistency metrics for all drivers"""
        if self.driver_details is None:
            return pd.DataFrame()
        
        consistency_data = []
        
        for player_id in self.driver_details['player_id'].unique():
            player_races = self.driver_details[
                self.driver_details['player_id'] == player_id
            ]
            
            if len(player_races) < 3:
                continue
            
            player_name = player_races.iloc[0]['player_name']
            points = player_races['total_points'].values
            
            consistency_data.append({
                'player_name': player_name,
                'races': len(player_races),
                'avg_points': np.mean(points),
                'std_points': np.std(points),
                'consistency_score': 1 - (np.std(points) / (np.mean(points) + 1e-6)),
                'min_points': np.min(points),
                'max_points': np.max(points),
                'q1_points': np.percentile(points, 25),
                'q3_points': np.percentile(points, 75)
            })
        
        return pd.DataFrame(consistency_data).sort_values('consistency_score', ascending=False)


# Helper functions for quick access
def load_fantasy_features(driver_names: list) -> pd.DataFrame:
    """Quick function to load fantasy features for a list of drivers"""
    integration = F1FantasyIntegration()
    
    features_list = []
    for driver in driver_names:
        features = integration.get_driver_fantasy_features(driver)
        features['driver_name'] = driver
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def get_fantasy_validation_report(predictions_df: pd.DataFrame) -> Dict:
    """Generate a validation report comparing predictions with fantasy data"""
    integration = F1FantasyIntegration()
    validated = integration.validate_predictions_with_fantasy(predictions_df)
    
    report = {
        'total_drivers': len(validated),
        'drivers_with_fantasy_data': len(validated.dropna(subset=['fantasy_points_avg'])),
        'avg_alignment_score': validated['fantasy_alignment_score'].mean() if 'fantasy_alignment_score' in validated else 0,
        'validation_dataframe': validated
    }
    
    return report


# Example usage
if __name__ == "__main__":
    # Initialize integration
    f1_fantasy = F1FantasyIntegration()
    
    # Get features for a specific driver
    norris_features = f1_fantasy.get_driver_fantasy_features("Lando Norris")
    print("Lando Norris Fantasy Features:")
    for key, value in norris_features.items():
        print(f"  {key}: {value}")
    
    # Get value drivers
    print("\nTop Value Drivers:")
    value_drivers = f1_fantasy.get_value_drivers(max_price=20.0)
    print(value_drivers)
    
    # Get consistency metrics
    print("\nMost Consistent Drivers:")
    consistency = f1_fantasy.get_consistency_metrics()
    print(consistency.head())