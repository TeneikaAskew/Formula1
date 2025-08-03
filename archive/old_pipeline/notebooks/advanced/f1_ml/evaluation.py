"""
F1 Driver Evaluation Module

This module contains the IntegratedF1Predictor class and supporting functions
for evaluating drivers based on age, performance, and constructor compatibility.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')


def calculate_enhanced_driver_metrics(df):
    """
    Calculate comprehensive driver metrics including evaluation scores
    """
    metrics = []
    
    # Focus on recent seasons for relevance
    recent_df = df[df['year'] >= 2019].copy()
    
    for driver_id in recent_df['driverId'].unique():
        driver_data = recent_df[recent_df['driverId'] == driver_id]
        
        # Basic information
        driver_info = {
            'driverId': driver_id,
            'driverRef': driver_data['driverRef'].iloc[0],
            'surname': driver_data['surname'].iloc[0],
            'current_age': driver_data['driver_age'].iloc[-1] if len(driver_data) > 0 else 0,
            'races_completed': len(driver_data),
            'years_active': driver_data['year'].nunique()
        }
        
        # Performance metrics
        driver_info['avg_position'] = driver_data['positionOrder'].mean()
        driver_info['avg_points'] = driver_data['points'].mean()
        driver_info['total_points'] = driver_data['points'].sum()
        driver_info['podium_rate'] = (driver_data['positionOrder'] <= 3).mean()
        driver_info['top10_rate'] = (driver_data['positionOrder'] <= 10).mean()
        driver_info['dnf_rate'] = (driver_data['statusId'] > 1).mean()
        driver_info['wins'] = (driver_data['positionOrder'] == 1).sum()
        driver_info['win_rate'] = (driver_data['positionOrder'] == 1).mean()
        
        # Consistency metrics
        driver_info['position_std'] = driver_data['positionOrder'].std()
        driver_info['points_std'] = driver_data['points'].std()
        driver_info['consistency_score'] = 1 / (1 + driver_info['position_std']) if driver_info['position_std'] > 0 else 1
        
        # Recent form (last 10 races)
        recent_races = driver_data.sort_values('date').tail(10)
        if len(recent_races) > 0:
            driver_info['recent_avg_position'] = recent_races['positionOrder'].mean()
            driver_info['recent_avg_points'] = recent_races['points'].mean()
            driver_info['recent_form_trend'] = driver_info['avg_position'] - driver_info['recent_avg_position']
        else:
            driver_info['recent_avg_position'] = driver_info['avg_position']
            driver_info['recent_avg_points'] = driver_info['avg_points']
            driver_info['recent_form_trend'] = 0
        
        # Track diversity
        driver_info['track_diversity'] = driver_data['circuitId'].nunique()
        driver_info['avg_tracks_per_season'] = driver_info['track_diversity'] / max(1, driver_info['years_active'])
        
        # Constructor performance
        current_constructor = driver_data['constructorId'].iloc[-1] if len(driver_data) > 0 else None
        driver_info['current_constructor'] = current_constructor
        driver_info['constructor_changes'] = driver_data['constructorId'].nunique()
        
        metrics.append(driver_info)
    
    return pd.DataFrame(metrics)


def create_age_performance_model(df, driver_metrics):
    """
    Model the relationship between age and performance
    """
    # Create age bins for analysis
    age_bins = [18, 22, 25, 28, 32, 36, 45]
    age_labels = ['18-22', '23-25', '26-28', '29-32', '33-36', '37+']
    
    df['age_group'] = pd.cut(df['driver_age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # Calculate performance by age group
    age_performance = df.groupby('age_group').agg({
        'positionOrder': ['mean', 'std'],
        'points': ['mean', 'std'],
        'driverId': 'nunique'
    }).round(2)
    
    # Peak performance window
    peak_age_range = (26, 32)
    
    # Calculate development scores
    development_scores = []
    
    for _, driver in driver_metrics.iterrows():
        age = driver['current_age']
        
        # Development phase
        if age < peak_age_range[0]:
            development_phase = 'pre-peak'
            years_to_peak = peak_age_range[0] - age
            # Young drivers have high potential
            age_factor = 1.1 + (0.05 * min(years_to_peak, 4))
        elif age <= peak_age_range[1]:
            development_phase = 'peak'
            years_to_peak = 0
            age_factor = 1.2  # Peak performance
        else:
            development_phase = 'post-peak'
            years_past_peak = age - peak_age_range[1]
            # Gradual decline
            age_factor = 1.0 - (0.03 * min(years_past_peak, 10))
        
        # Experience bonus
        experience_factor = min(1.2, 1 + (driver['races_completed'] / 100))
        
        # Combined development score
        development_score = age_factor * experience_factor * driver['consistency_score']
        
        development_scores.append({
            'driverId': driver['driverId'],
            'surname': driver['surname'],
            'current_age': age,
            'development_phase': development_phase,
            'age_factor': age_factor,
            'experience_factor': experience_factor,
            'development_score': development_score
        })
    
    development_df = pd.DataFrame(development_scores)
    
    return development_df, age_performance


def calculate_constructor_compatibility(df, driver_metrics):
    """
    Calculate how well drivers fit with their current constructors
    """
    compatibility_scores = []
    
    # Calculate constructor profiles
    constructor_profiles = df.groupby('constructorId').agg({
        'points': ['mean', 'sum'],
        'positionOrder': 'mean',
        'statusId': lambda x: (x > 1).mean()  # DNF rate
    }).round(3)
    
    for _, driver in driver_metrics.iterrows():
        driver_id = driver['driverId']
        constructor_id = driver['current_constructor']
        
        if pd.isna(constructor_id) or constructor_id not in constructor_profiles.index:
            compatibility_scores.append({
                'driverId': driver_id,
                'surname': driver['surname'],
                'compatibility_score': 0.5,  # Neutral if no data
                'team_performance_match': 0.5,
                'reliability_match': 0.5
            })
            continue
        
        # Get constructor profile
        constructor_avg_points = constructor_profiles.loc[constructor_id, ('points', 'mean')]
        constructor_avg_position = constructor_profiles.loc[constructor_id, ('positionOrder', 'mean')]
        constructor_dnf_rate = constructor_profiles.loc[constructor_id, ('statusId', '<lambda>')]
        
        # Performance match (closer is better)
        points_diff = abs(driver['avg_points'] - constructor_avg_points / 2)  # Assume 2 drivers
        performance_match = 1 / (1 + points_diff / 10)
        
        # Reliability match
        reliability_match = 1 - abs(driver['dnf_rate'] - constructor_dnf_rate)
        
        # Experience with constructor
        driver_constructor_races = df[(df['driverId'] == driver_id) & 
                                     (df['constructorId'] == constructor_id)]
        races_together = len(driver_constructor_races)
        experience_bonus = min(0.2, races_together / 100)  # Max 20% bonus
        
        # Overall compatibility
        compatibility = (
            0.5 * performance_match +
            0.3 * reliability_match +
            0.2 * (1 + experience_bonus)
        )
        
        compatibility_scores.append({
            'driverId': driver_id,
            'surname': driver['surname'],
            'constructorId': constructor_id,
            'compatibility_score': min(1.0, compatibility),
            'team_performance_match': performance_match,
            'reliability_match': reliability_match,
            'races_together': races_together
        })
    
    return pd.DataFrame(compatibility_scores)


class IntegratedF1Predictor:
    """
    Combines fixed prediction models with driver evaluation metrics
    """
    def __init__(self, base_model_path=None):
        self.base_model = None
        self.scaler = None
        self.feature_columns = None
        
        if base_model_path:
            self.load_base_model(base_model_path)
    
    def load_base_model(self, model_path):
        """Load the fixed prediction model"""
        try:
            model_artifacts = joblib.load(model_path)
            self.base_model = model_artifacts['model']
            self.scaler = model_artifacts['scaler']
            self.feature_columns = model_artifacts['feature_columns']
            print(f"Loaded model: {model_artifacts['model_name']}")
        except FileNotFoundError:
            print("Base model not found. Training new model...")
            self._train_default_model()
    
    def _train_default_model(self):
        """Train a default model if no saved model exists"""
        # Simplified training for demonstration
        self.base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=50,
            random_state=42
        )
        self.scaler = StandardScaler()
        print("Initialized default model")
    
    def predict_with_evaluation(self, race_features, driver_evaluation, compatibility):
        """
        Make predictions incorporating driver evaluation metrics
        """
        predictions = []
        
        for idx, features in race_features.iterrows():
            driver_id = features.get('driverId')
            
            # Get base prediction (if model is loaded and trained)
            base_prob = 0.5  # Default probability
            
            if self.base_model is not None and self.feature_columns is not None and self.scaler is not None:
                try:
                    # Check if model is fitted
                    if hasattr(self.base_model, 'n_features_in_'):
                        X = features[self.feature_columns].values.reshape(1, -1)
                        X_scaled = self.scaler.transform(X)
                        base_prob = self.base_model.predict_proba(X_scaled)[0, 1]
                except:
                    base_prob = 0.5  # Default if prediction fails
            
            # Get driver evaluation metrics
            driver_eval = driver_evaluation[driver_evaluation['driverId'] == driver_id]
            if driver_eval.empty:
                age_factor = 1.0
                development_score = 1.0
            else:
                age_factor = driver_eval.iloc[0]['age_factor']
                development_score = driver_eval.iloc[0]['development_score']
            
            # Get compatibility score
            driver_compat = compatibility[compatibility['driverId'] == driver_id]
            if driver_compat.empty:
                compat_score = 0.5
            else:
                compat_score = driver_compat.iloc[0]['compatibility_score']
            
            # Adjust prediction based on evaluation metrics
            # Weight: 60% base model, 20% age factor, 20% compatibility
            adjusted_prob = (
                0.6 * base_prob +
                0.2 * (base_prob * age_factor) +
                0.2 * (base_prob * compat_score)
            )
            
            # Ensure probability is in valid range
            adjusted_prob = np.clip(adjusted_prob, 0.01, 0.99)
            
            predictions.append({
                'driverId': driver_id,
                'base_probability': base_prob,
                'age_factor': age_factor,
                'compatibility_score': compat_score,
                'development_score': development_score,
                'adjusted_probability': adjusted_prob,
                'confidence': self._calculate_confidence(base_prob, age_factor, compat_score)
            })
        
        return pd.DataFrame(predictions)
    
    def _calculate_confidence(self, base_prob, age_factor, compat_score):
        """Calculate prediction confidence"""
        # Higher confidence when all factors align
        factor_variance = np.std([base_prob, age_factor, compat_score])
        confidence = 1 - (2 * factor_variance)  # Lower variance = higher confidence
        return np.clip(confidence, 0.1, 0.9)


# Export key components
__all__ = [
    'calculate_enhanced_driver_metrics',
    'create_age_performance_model',
    'calculate_constructor_compatibility',
    'IntegratedF1Predictor'
]