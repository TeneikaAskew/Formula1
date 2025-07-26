"""
F1 Feature Store Module

This module contains the F1FeatureStore class for managing and organizing
all features used in F1 race predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings


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