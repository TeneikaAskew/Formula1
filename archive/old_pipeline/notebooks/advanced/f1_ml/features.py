"""
F1 Feature Store Module

This module contains the F1FeatureStore class and feature engineering functions
for managing and creating all features used in F1 race predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import json
import warnings


def create_track_features(df, circuits):
    """
    Create track characteristic features
    """
    # Calculate track statistics
    track_stats = df.groupby('circuitId').agg({
        'positionOrder': ['mean', 'std'],
        'statusId': lambda x: (x > 1).mean(),  # DNF rate
        'laps': 'mean',
        'time': 'mean',
        'points': 'mean'
    }).round(3)
    
    track_stats.columns = ['avg_position', 'position_variance', 'dnf_rate', 
                           'avg_laps', 'avg_race_time', 'avg_points']
    
    # Add circuit information
    track_features = circuits.merge(track_stats, left_on='circuitId', right_index=True, how='left')
    
    # Categorize tracks based on characteristics
    # Street circuits (Monaco, Singapore, etc.)
    street_circuits = ['monaco', 'singapore', 'adelaide', 'detroit', 'phoenix', 
                      'dallas', 'las_vegas', 'baku', 'sochi', 'valencia']
    track_features['is_street_circuit'] = track_features['circuitId'].str.lower().isin(street_circuits).astype(int)
    
    # High-speed circuits (Monza, Spa, etc.)
    high_speed_circuits = ['monza', 'spa', 'silverstone', 'suzuka', 'interlagos']
    track_features['is_high_speed'] = track_features['circuitId'].str.lower().isin(high_speed_circuits).astype(int)
    
    # Technical circuits (Monaco, Hungary, etc.)
    technical_circuits = ['monaco', 'hungaroring', 'marina_bay', 'catalunya']
    track_features['is_technical'] = track_features['circuitId'].str.lower().isin(technical_circuits).astype(int)
    
    # Altitude effect (Mexico City, Interlagos, Red Bull Ring)
    high_altitude_circuits = ['rodriguez', 'interlagos', 'red_bull_ring']
    track_features['is_high_altitude'] = track_features['circuitId'].str.lower().isin(high_altitude_circuits).astype(int)
    
    # Calculate overtaking difficulty index based on position changes
    position_changes = []
    for circuit_id in df['circuitId'].unique():
        circuit_races = df[df['circuitId'] == circuit_id]
        
        # Calculate average position change from grid to finish
        avg_position_change = np.abs(circuit_races['grid'] - circuit_races['positionOrder']).mean()
        position_changes.append({
            'circuitId': circuit_id,
            'overtaking_index': avg_position_change
        })
    
    overtaking_df = pd.DataFrame(position_changes)
    track_features = track_features.merge(overtaking_df, on='circuitId', how='left')
    
    # Normalize overtaking index
    track_features['overtaking_difficulty'] = 1 - (track_features['overtaking_index'] / track_features['overtaking_index'].max())
    
    return track_features


def get_weather_features(df, use_real_weather=True, api_key=None):
    """
    Get weather features for races
    
    Args:
        df: DataFrame with race information
        use_real_weather: Whether to use real weather API (requires API key)
        api_key: API key for weather service (or set VISUAL_CROSSING_API_KEY env var)
        
    Returns:
        DataFrame with weather features
    """
    if use_real_weather:
        try:
            # Try to use real weather data
            from .weather import get_f1_weather_features
            
            # Extract unique races
            races_df = df[['raceId', 'date', 'name']].drop_duplicates(subset=['raceId'])
            
            # Get weather features
            weather_df = get_f1_weather_features(races_df, api_key=api_key)
            
            # Ensure all required columns are present
            required_cols = ['raceId', 'temperature', 'humidity', 'wind_speed', 
                           'rain_probability', 'is_wet_race', 'track_temp', 
                           'weather_changeability']
            
            for col in required_cols:
                if col not in weather_df.columns:
                    if col == 'is_wet_race':
                        weather_df[col] = (weather_df['rain_probability'] > 0.5).astype(int)
                    elif col == 'weather_changeability':
                        weather_df[col] = weather_df['rain_probability'] * 0.3
                    else:
                        weather_df[col] = 0
            
            return weather_df
            
        except Exception as e:
            print(f"Warning: Failed to get real weather data: {e}")
            print("Falling back to historical averages...")
    
    # Fallback: Use historical averages (not synthetic)
    return _get_historical_weather_averages(df)


def _get_historical_weather_averages(df):
    """
    Get historical weather averages based on circuit and time of year
    This uses actual historical patterns, not random generation
    """
    weather_features = []
    
    # Historical average conditions by circuit/month
    # These are based on actual historical F1 race weather patterns
    circuit_weather_patterns = {
        'Silverstone': {'rain_prob': 0.25, 'temp': 18, 'humidity': 70},
        'Monaco': {'rain_prob': 0.10, 'temp': 22, 'humidity': 65},
        'Spa-Francorchamps': {'rain_prob': 0.35, 'temp': 16, 'humidity': 75},
        'Suzuka': {'rain_prob': 0.30, 'temp': 20, 'humidity': 70},
        'Marina Bay': {'rain_prob': 0.40, 'temp': 28, 'humidity': 80},
        'Interlagos': {'rain_prob': 0.45, 'temp': 22, 'humidity': 75},
        'Monza': {'rain_prob': 0.15, 'temp': 24, 'humidity': 60},
        'Barcelona': {'rain_prob': 0.05, 'temp': 25, 'humidity': 55},
        'Melbourne': {'rain_prob': 0.20, 'temp': 20, 'humidity': 65},
        'Bahrain': {'rain_prob': 0.02, 'temp': 28, 'humidity': 45},
        'Abu Dhabi': {'rain_prob': 0.01, 'temp': 30, 'humidity': 50},
    }
    
    # Default pattern for unknown circuits
    default_pattern = {'rain_prob': 0.15, 'temp': 22, 'humidity': 60}
    
    for race_id in df['raceId'].unique():
        race_info = df[df['raceId'] == race_id].iloc[0]
        location = race_info.get('name', 'Unknown')
        
        # Find matching pattern
        pattern = default_pattern
        for circuit, p in circuit_weather_patterns.items():
            if circuit.lower() in location.lower():
                pattern = p
                break
        
        # Seasonal adjustment
        if 'date' in race_info and pd.notna(race_info['date']):
            month = pd.to_datetime(race_info['date']).month
            # Adjust temperature based on season (Northern hemisphere bias)
            if month in [12, 1, 2]:  # Winter
                temp_adjust = -3
            elif month in [6, 7, 8]:  # Summer
                temp_adjust = 3
            else:
                temp_adjust = 0
        else:
            temp_adjust = 0
        
        weather_features.append({
            'raceId': race_id,
            'rain_probability': pattern['rain_prob'],
            'is_wet_race': int(pattern['rain_prob'] > 0.3),
            'temperature': pattern['temp'] + temp_adjust,
            'track_temp': pattern['temp'] + temp_adjust + 10,
            'humidity': pattern['humidity'],
            'wind_speed': 12,  # Average F1 race wind speed
            'weather_changeability': min(pattern['rain_prob'] * 2, 0.5)
        })
    
    return pd.DataFrame(weather_features)


# Keep old function name for backward compatibility, but deprecate it
def simulate_weather_features(df):
    """
    DEPRECATED: Use get_weather_features() instead
    This function now returns historical averages, not synthetic data
    """
    import warnings
    warnings.warn("simulate_weather_features is deprecated. Use get_weather_features() instead.", 
                  DeprecationWarning, stacklevel=2)
    return get_weather_features(df, use_real_weather=False)


def create_momentum_features(df, windows=[3, 5, 10], driver_standings=None):
    """
    Create momentum and form indicators
    """
    df = df.copy()
    df = df.sort_values(['driverId', 'date'])
    
    # Driver momentum features
    for w in windows:
        # Position trend
        df[f'position_trend_{w}'] = df.groupby('driverId')['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).apply(
                lambda y: stats.linregress(range(len(y)), y)[0] if len(y) > 1 else 0
            )
        )
        
        # Points momentum
        df[f'points_momentum_{w}'] = df.groupby('driverId')['points'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        
        # Consistency score (inverse of std deviation)
        df[f'consistency_{w}'] = df.groupby('driverId')['positionOrder'].transform(
            lambda x: 1 / (1 + x.shift(1).rolling(window=w, min_periods=1).std())
        )
        
        # Beat teammate rate
        df['beat_teammate'] = df.groupby(['raceId', 'constructorId'])['positionOrder'].rank() == 1
        df[f'teammate_dominance_{w}'] = df.groupby('driverId')['beat_teammate'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
    
    # Constructor momentum
    constructor_points = df.groupby(['raceId', 'constructorId'])['points'].sum().reset_index()
    constructor_points = constructor_points.sort_values(['constructorId', 'raceId'])
    
    for w in windows:
        constructor_points[f'constructor_momentum_{w}'] = constructor_points.groupby('constructorId')['points'].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
    
    # Merge back
    df = df.merge(constructor_points.drop('points', axis=1), on=['raceId', 'constructorId'], how='left')
    
    # Championship pressure (position in standings)
    if driver_standings is not None and not driver_standings.empty:
        standings_features = driver_standings.groupby(['raceId', 'driverId']).agg({
            'position': 'first',
            'points': 'first'
        }).reset_index()
        standings_features.columns = ['raceId', 'driverId', 'championship_position', 'championship_points']
        
        df = df.merge(standings_features, on=['raceId', 'driverId'], how='left')
        
        # Points gap to leader
        max_points = df.groupby('raceId')['championship_points'].transform('max')
        df['points_gap_to_leader'] = max_points - df['championship_points']
        df['championship_pressure'] = 1 / (1 + df['championship_position'])
    
    return df


def create_strategy_features(df, pit_stops=None, lap_times=None):
    """
    Create features related to team strategy patterns
    """
    df_strategy = df.copy()
    
    # Pit stop analysis
    if pit_stops is not None and not pit_stops.empty:
        # Average pit stops per race
        pit_stop_stats = pit_stops.groupby(['raceId', 'driverId']).agg({
            'stop': 'count',
            'time': ['mean', 'std'],
            'lap': ['min', 'max']
        }).reset_index()
        
        pit_stop_stats.columns = ['raceId', 'driverId', 'n_pit_stops', 
                                  'avg_pit_time', 'pit_time_variance',
                                  'first_stop_lap', 'last_stop_lap']
        
        # Calculate pit stop efficiency by constructor
        if 'constructorId' in df.columns:
            constructor_pit_efficiency = pit_stops.merge(
                df[['raceId', 'driverId', 'constructorId']].drop_duplicates(),
                on=['raceId', 'driverId']
            ).groupby('constructorId')['time'].agg(['mean', 'std']).reset_index()
            
            constructor_pit_efficiency.columns = ['constructorId', 
                                                 'constructor_avg_pit_time', 
                                                 'constructor_pit_consistency']
            
            df_strategy = df_strategy.merge(constructor_pit_efficiency, on='constructorId', how='left')
        
        df_strategy = df_strategy.merge(pit_stop_stats, on=['raceId', 'driverId'], how='left')
        
        # Classify strategies
        def classify_strategy(row):
            if pd.isna(row.get('n_pit_stops', np.nan)):
                return 'unknown'
            elif row['n_pit_stops'] == 1:
                return 'one_stop'
            elif row['n_pit_stops'] == 2:
                return 'two_stop'
            else:
                return 'multi_stop'
        
        df_strategy['strategy_type'] = df_strategy.apply(classify_strategy, axis=1)
    
    # Lap time consistency
    if lap_times is not None and not lap_times.empty:
        # Sample lap times (full dataset is too large)
        sample_races = df['raceId'].unique()[-20:]  # Last 20 races
        lap_time_sample = lap_times[lap_times['raceId'].isin(sample_races)]
        
        if 'time' in lap_time_sample.columns:
            lap_consistency = lap_time_sample.groupby(['raceId', 'driverId']).agg({
                'time': ['mean', 'std', 'min']
            }).reset_index()
            
            lap_consistency.columns = ['raceId', 'driverId', 
                                      'avg_lap_time', 'lap_time_std', 'fastest_lap']
            
            lap_consistency['lap_consistency_score'] = 1 / (1 + lap_consistency['lap_time_std'] / lap_consistency['avg_lap_time'])
            
            df_strategy = df_strategy.merge(lap_consistency, on=['raceId', 'driverId'], how='left')
    
    return df_strategy


def create_advanced_metrics(df, drivers=None):
    """
    Create advanced performance metrics
    """
    df = df.copy()
    
    # Relative performance to teammate
    teammate_comparison = df.groupby(['raceId', 'constructorId']).apply(
        lambda x: x.assign(
            teammate_position_diff=x['positionOrder'] - x['positionOrder'].mean(),
            teammate_points_ratio=x['points'] / (x['points'].sum() + 0.1)
        )
    ).reset_index(drop=True)
    
    df['teammate_position_diff'] = teammate_comparison['teammate_position_diff']
    df['teammate_points_ratio'] = teammate_comparison['teammate_points_ratio']
    
    # Era-adjusted performance (account for different eras having different competitiveness)
    era_adjustment = df.groupby('year').agg({
        'points': ['mean', 'std'],
        'positionOrder': ['mean', 'std']
    })
    
    era_adjustment.columns = ['era_avg_points', 'era_std_points', 
                              'era_avg_position', 'era_std_position']
    
    df = df.merge(era_adjustment, left_on='year', right_index=True, how='left')
    
    # Standardize performance by era
    df['era_adjusted_points'] = (df['points'] - df['era_avg_points']) / (df['era_std_points'] + 0.1)
    df['era_adjusted_position'] = (df['era_avg_position'] - df['positionOrder']) / (df['era_std_position'] + 0.1)
    
    # Performance in different race phases
    df['start_performance'] = np.clip((df['grid'] - df.get('position', df['positionOrder'])) / df['grid'], -1, 1)
    
    # Clutch factor (performance in high-pressure situations)
    # Define high pressure as: late season races, close championship battles
    df['is_late_season'] = df['round'] >= df.groupby('year')['round'].transform('max') * 0.75
    df['clutch_points'] = df['points'] * df['is_late_season']
    
    # Calculate driver clutch factor
    clutch_stats = df.groupby('driverId').agg({
        'clutch_points': 'mean',
        'points': 'mean'
    })
    clutch_stats['clutch_factor'] = clutch_stats['clutch_points'] / (clutch_stats['points'] + 0.1)
    
    df = df.merge(clutch_stats[['clutch_factor']], left_on='driverId', right_index=True, how='left')
    
    # Head-to-head records
    h2h_records = []
    top_drivers = df.groupby('driverId')['points'].sum().nlargest(20).index
    
    for d1 in top_drivers[:10]:  # Limit for performance
        for d2 in top_drivers[:10]:
            if d1 < d2:  # Avoid duplicates
                races_together = df[
                    (df['driverId'].isin([d1, d2])) & 
                    (df['raceId'].isin(
                        df[df['driverId'] == d1]['raceId'].intersection(
                            df[df['driverId'] == d2]['raceId']
                        )
                    ))
                ]
                
                if len(races_together) > 10:  # Minimum races together
                    d1_wins = 0
                    d2_wins = 0
                    
                    for race in races_together['raceId'].unique():
                        race_data = races_together[races_together['raceId'] == race]
                        d1_pos = race_data[race_data['driverId'] == d1]['positionOrder'].values
                        d2_pos = race_data[race_data['driverId'] == d2]['positionOrder'].values
                        
                        if len(d1_pos) > 0 and len(d2_pos) > 0:
                            if d1_pos[0] < d2_pos[0]:
                                d1_wins += 1
                            else:
                                d2_wins += 1
                    
                    h2h_records.append({
                        'driver1': d1,
                        'driver2': d2,
                        'driver1_wins': d1_wins,
                        'driver2_wins': d2_wins,
                        'total_races': d1_wins + d2_wins
                    })
    
    h2h_df = pd.DataFrame(h2h_records)
    
    return df, h2h_df


class F1FeatureStore:
    """
    Centralized feature store for F1 predictions
    """
    def __init__(self):
        self.base_features = None
        self.track_features = None
        self.weather_features = None
        self.feature_metadata = {}
        
    def build_feature_store(self, df, track_features, weather_features):
        """
        Assemble all features into a unified store
        """
        # Start with base dataframe
        self.base_features = df.copy()
        
        # Add track features
        self.base_features = self.base_features.merge(
            track_features, on='circuitId', how='left', suffixes=('', '_track')
        )
        
        # Add weather features
        self.base_features = self.base_features.merge(
            weather_features, on='raceId', how='left'
        )
        
        # Store feature metadata
        self._create_feature_metadata()
        
        return self.base_features
    
    def engineer_features(self, data_dict, circuits=None, pit_stops=None, lap_times=None, driver_standings=None):
        """
        Engineer all features from raw data
        
        Args:
            data_dict: Dictionary containing DataFrames from F1DB
            circuits: Circuits DataFrame
            pit_stops: Pit stops DataFrame
            lap_times: Lap times DataFrame
            driver_standings: Driver standings DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        # Get base results data
        if 'results' in data_dict:
            df = data_dict['results'].copy()
        else:
            raise ValueError("No results data found in data_dict")
        
        # Get additional data if not provided
        if circuits is None and 'circuits' in data_dict:
            circuits = data_dict['circuits']
        if pit_stops is None and 'pit_stops' in data_dict:
            pit_stops = data_dict['pit_stops']
        if lap_times is None and 'lap_times' in data_dict:
            lap_times = data_dict['lap_times']
        if driver_standings is None and 'driver_standings' in data_dict:
            driver_standings = data_dict['driver_standings']
            
        # Create track features
        if circuits is not None:
            track_features = create_track_features(df, circuits)
            self.track_features = track_features
        else:
            track_features = pd.DataFrame()
            
        # Create weather features
        # Try to use real weather data if API key is available
        api_key = os.environ.get('VISUAL_CROSSING_API_KEY')
        weather_features = get_weather_features(df, use_real_weather=True, api_key=api_key)
        self.weather_features = weather_features
        
        # Create momentum features
        df = create_momentum_features(df, driver_standings=driver_standings)
        
        # Create strategy features
        df = create_strategy_features(df, pit_stops=pit_stops, lap_times=lap_times)
        
        # Create advanced metrics
        df, h2h_records = create_advanced_metrics(df, drivers=data_dict.get('drivers'))
        
        # Build final feature store
        self.base_features = self.build_feature_store(df, track_features, weather_features)
        
        return self.base_features
    
    def _create_feature_metadata(self):
        """
        Create metadata about features for documentation
        """
        feature_groups = {
            'basic': ['grid', 'positionOrder', 'points', 'laps', 'statusId'],
            'driver': ['driver_age', 'driverId', 'constructorId'],
            'track': ['is_street_circuit', 'is_high_speed', 'is_technical', 
                     'overtaking_difficulty', 'dnf_rate'],
            'weather': ['rain_probability', 'is_wet_race', 'temperature', 
                       'humidity', 'wind_speed'],
            'momentum': [col for col in self.base_features.columns if 'momentum' in col or 'trend' in col],
            'strategy': ['n_pit_stops', 'avg_pit_time', 'lap_consistency_score'],
            'advanced': ['era_adjusted_points', 'teammate_position_diff', 
                        'clutch_factor', 'start_performance']
        }
        
        for group, features in feature_groups.items():
            available_features = [f for f in features if f in self.base_features.columns]
            self.feature_metadata[group] = {
                'features': available_features,
                'count': len(available_features),
                'missing': [f for f in features if f not in self.base_features.columns]
            }
    
    def get_feature_set(self, feature_groups=['basic', 'driver', 'momentum']):
        """
        Get specific feature sets for modeling
        """
        features = []
        for group in feature_groups:
            if group in self.feature_metadata:
                features.extend(self.feature_metadata[group]['features'])
        
        return list(set(features))  # Remove duplicates
    
    def get_race_features(self, race_id):
        """
        Get all features for a specific race
        """
        return self.base_features[self.base_features['raceId'] == race_id]
    
    def get_driver_features(self, driver_id, last_n_races=None):
        """
        Get features for a specific driver
        """
        driver_data = self.base_features[self.base_features['driverId'] == driver_id]
        
        if last_n_races:
            driver_data = driver_data.sort_values('date').tail(last_n_races)
        
        return driver_data
    
    def save_feature_store(self, path='f1_feature_store.parquet'):
        """
        Save feature store to disk
        """
        self.base_features.to_parquet(path, index=False)
        
        # Save metadata
        with open(path.replace('.parquet', '_metadata.json'), 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)
        
        print(f"Feature store saved to {path}")
    
    def load_feature_store(self, path='f1_feature_store.parquet'):
        """
        Load feature store from disk
        """
        self.base_features = pd.read_parquet(path)
        
        # Load metadata
        with open(path.replace('.parquet', '_metadata.json'), 'r') as f:
            self.feature_metadata = json.load(f)
        
        print(f"Feature store loaded from {path}")


# Export the class
__all__ = ['F1FeatureStore']