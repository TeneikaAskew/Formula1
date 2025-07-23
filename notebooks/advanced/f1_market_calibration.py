"""
F1 Market Calibration Module

This module contains key functions extracted from F1_Betting_Market_Models.ipynb
for enhancing the prize picks pipeline with market-calibrated predictions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


class OrdinalRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal regression for predicting race finishing positions.
    Properly handles the ordered nature of F1 finishing positions.
    """
    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier or LogisticRegression(max_iter=1000)
        self.classifiers = {}
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        
        # Train binary classifiers for each threshold
        for i, threshold in enumerate(self.classes_[:-1]):
            # Create binary target: 1 if position <= threshold
            binary_y = (y <= threshold).astype(int)
            
            # Clone and train classifier
            clf = clone(self.base_classifier)
            clf.fit(X, binary_y)
            self.classifiers[threshold] = clf
            
        return self
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        
        # Get cumulative probabilities
        cum_probas = np.zeros((n_samples, n_classes))
        
        for i, threshold in enumerate(self.classes_[:-1]):
            cum_probas[:, i] = self.classifiers[threshold].predict_proba(X)[:, 1]
        
        # Ensure monotonicity
        for i in range(1, n_classes - 1):
            cum_probas[:, i] = np.maximum(cum_probas[:, i], cum_probas[:, i-1])
        
        # Convert cumulative to individual probabilities
        probas[:, 0] = cum_probas[:, 0]
        for i in range(1, n_classes - 1):
            probas[:, i] = cum_probas[:, i] - cum_probas[:, i-1]
        probas[:, -1] = 1 - cum_probas[:, -2]
        
        # Normalize to ensure sum to 1
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def create_betting_features(df):
    """
    Create features specifically designed for betting predictions.
    Includes recent form, constructor performance, track history, and championship pressure.
    """
    df = df.sort_values(['driverId', 'date']).copy()
    
    # Recent form features (last 5 races)
    for window in [3, 5, 10]:
        df[f'avg_position_last_{window}'] = df.groupby('driverId')['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'dnf_rate_last_{window}'] = df.groupby('driverId')['finished'].transform(
            lambda x: 1 - x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        if 'points_scored' in df.columns:
            df[f'points_rate_last_{window}'] = df.groupby('driverId')['points_scored'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        elif 'points' in df.columns:
            df[f'points_rate_last_{window}'] = df.groupby('driverId')['points'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
    
    # Constructor form
    df['constructor_avg_position'] = df.groupby(['constructorId', 'raceId'])['positionOrder'].transform('mean')
    df['constructor_reliability'] = df.groupby('constructorId')['finished'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Track-specific features
    df['driver_track_history'] = df.groupby(['driverId', 'circuitId']).cumcount()
    df['driver_track_avg_position'] = df.groupby(['driverId', 'circuitId'])['positionOrder'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['driver_track_dnf_rate'] = df.groupby(['driverId', 'circuitId'])['finished'].transform(
        lambda x: 1 - x.shift(1).expanding().mean()
    )
    
    # Qualifying performance
    if 'position_quali' in df.columns:
        df['quali_position'] = df['position_quali'].fillna(20)
    elif 'qualifying_position' in df.columns:
        df['quali_position'] = df['qualifying_position'].fillna(20)
    else:
        df['quali_position'] = df['grid']
        
    df['quali_to_grid_change'] = df['grid'] - df['quali_position']
    
    # Season progress
    df['season_progress'] = df.groupby('year')['round'].transform(lambda x: x / x.max())
    
    # Championship pressure
    if 'points' in df.columns:
        season_standings = df.groupby(['year', 'driverId'])['points'].sum().reset_index()
        season_standings['championship_position'] = season_standings.groupby('year')['points'].rank(ascending=False)
        df = df.merge(season_standings[['year', 'driverId', 'championship_position']], on=['year', 'driverId'], how='left')
    
    return df


def create_head_to_head_dataset(df, feature_cols):
    """
    Create pairwise comparisons for head-to-head predictions.
    Returns a dataset suitable for training H2H matchup models.
    """
    h2h_data = []
    
    # Group by race
    for race_id, race_data in df.groupby('raceId'):
        drivers = race_data['driverId'].unique()
        
        # Create all pairwise comparisons
        for i in range(len(drivers)):
            for j in range(i+1, len(drivers)):
                driver1_data = race_data[race_data['driverId'] == drivers[i]].iloc[0]
                driver2_data = race_data[race_data['driverId'] == drivers[j]].iloc[0]
                
                # Only include if both drivers finished
                if driver1_data['finished'] == 1 and driver2_data['finished'] == 1:
                    # Create feature differences
                    features_diff = {}
                    for col in feature_cols:
                        if col in driver1_data and col in driver2_data:
                            features_diff[f'{col}_diff'] = driver1_data[col] - driver2_data[col]
                    
                    # Target: 1 if driver1 finished ahead
                    features_diff['driver1_wins'] = int(driver1_data['positionOrder'] < driver2_data['positionOrder'])
                    features_diff['race_id'] = race_id
                    features_diff['driver1_id'] = drivers[i]
                    features_diff['driver2_id'] = drivers[j]
                    
                    if 'driverRef' in driver1_data:
                        features_diff['driver1_ref'] = driver1_data['driverRef']
                        features_diff['driver2_ref'] = driver2_data['driverRef']
                    
                    features_diff['date'] = driver1_data['date']
                    
                    h2h_data.append(features_diff)
    
    return pd.DataFrame(h2h_data)


def categorize_points(points):
    """
    Categorize points into brackets for multi-class prediction.
    """
    if points == 0:
        return 0  # No points
    elif points <= 4:
        return 1  # 1-4 points (P7-P10)
    elif points <= 10:
        return 2  # 6-10 points (P5-P6)
    elif points <= 18:
        return 3  # 12-18 points (P3-P4)
    else:
        return 4  # 25+ points (P1-P2)


def generate_betting_odds(race_data, models, scalers, feature_cols):
    """
    Generate comprehensive betting odds for a race.
    Combines DNF, position, and points predictions into calibrated betting odds.
    """
    # Prepare features
    X = race_data[feature_cols]
    X_scaled = scalers['main'].transform(X)
    
    odds_data = []
    
    for idx, (_, driver) in enumerate(race_data.iterrows()):
        driver_odds = {
            'driver': driver.get('driverRef', driver.get('driver_name', f"Driver_{driver['driverId']}")),
            'constructor': driver.get('constructorRef', driver.get('constructor_name', f"Constructor_{driver['constructorId']}")),
            'grid': driver['grid']
        }
        
        # DNF probability
        if 'dnf' in models:
            dnf_prob = models['dnf'].predict_proba(X_scaled[idx:idx+1])[:, 1][0]
            driver_odds['dnf_probability'] = dnf_prob
            driver_odds['finish_probability'] = 1 - dnf_prob
        else:
            driver_odds['dnf_probability'] = 0.15  # Default DNF rate
            driver_odds['finish_probability'] = 0.85
        
        # Position probabilities (if finishes)
        if 'position' in models:
            pos_probas = models['position'].predict_proba(X_scaled[idx:idx+1])[0]
            finish_prob = driver_odds['finish_probability']
            
            # Adjust probabilities for DNF risk
            driver_odds['win_probability'] = pos_probas[0] * finish_prob if len(pos_probas) > 0 else 0
            driver_odds['podium_probability'] = pos_probas[:3].sum() * finish_prob if len(pos_probas) >= 3 else 0
            driver_odds['top5_probability'] = pos_probas[:5].sum() * finish_prob if len(pos_probas) >= 5 else 0
            driver_odds['top10_probability'] = pos_probas[:10].sum() * finish_prob if len(pos_probas) >= 10 else 0
        
        # Points probability
        if 'points' in models:
            points_probas = models['points'].predict_proba(X_scaled[idx:idx+1])[0]
            driver_odds['points_probability'] = 1 - points_probas[0]  # Probability of scoring any points
            
            # Expected points
            points_mapping = {0: 0, 1: 2, 2: 8, 3: 15, 4: 25}
            driver_odds['expected_points'] = sum(points_probas[i] * points_mapping.get(i, 0) for i in range(len(points_probas)))
        else:
            driver_odds['points_probability'] = 0.5
            driver_odds['expected_points'] = 5.0
        
        # Convert probabilities to decimal odds
        for key in ['win_probability', 'podium_probability', 'top5_probability', 
                   'top10_probability', 'points_probability']:
            if key in driver_odds and driver_odds[key] > 0.01:  # Avoid division by very small numbers
                driver_odds[f'{key.replace("_probability", "")}_odds'] = 1 / driver_odds[key]
        
        odds_data.append(driver_odds)
    
    return pd.DataFrame(odds_data)


def calculate_prediction_confidence(models, X_scaled, n_iterations=100):
    """
    Calculate prediction confidence using bootstrapping.
    Returns confidence intervals and uncertainty estimates.
    """
    n_samples = X_scaled.shape[0]
    
    # Store predictions from each iteration
    predictions = {model_name: [] for model_name in models.keys()}
    
    # Bootstrap predictions
    for _ in range(n_iterations):
        # Sample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_scaled[idx]
        
        # Get predictions for each model
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_boot)
                if pred.shape[1] == 2:  # Binary classification
                    predictions[model_name].append(pred[:, 1])
                else:  # Multi-class
                    predictions[model_name].append(pred)
    
    # Calculate confidence intervals
    confidence_data = {}
    
    for model_name, preds in predictions.items():
        if len(preds) > 0:
            preds = np.array(preds)
            
            if preds.ndim == 2:  # Binary predictions
                mean_pred = np.mean(preds, axis=0)
                std_pred = np.std(preds, axis=0)
                lower_pred = np.percentile(preds, 5, axis=0)
                upper_pred = np.percentile(preds, 95, axis=0)
                
                confidence_data[model_name] = pd.DataFrame({
                    f'{model_name}_mean': mean_pred,
                    f'{model_name}_std': std_pred,
                    f'{model_name}_lower_90': lower_pred,
                    f'{model_name}_upper_90': upper_pred,
                    f'{model_name}_confidence': 1 - (std_pred / (mean_pred + 0.001))
                })
    
    return confidence_data


def calibrate_probabilities_isotonic(y_true, y_pred):
    """
    Apply isotonic regression for probability calibration.
    Ensures predicted probabilities match actual frequencies.
    """
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    calibrated = iso_reg.fit_transform(y_pred, y_true)
    return calibrated, iso_reg


def generate_head_to_head_odds(driver1_data, driver2_data, h2h_model, scaler, feature_cols):
    """
    Generate head-to-head matchup probabilities between two drivers.
    """
    # Create feature differences
    features_diff = {}
    for col in feature_cols:
        if col in driver1_data and col in driver2_data:
            features_diff[f'{col}_diff'] = [driver1_data[col] - driver2_data[col]]
    
    # Convert to DataFrame and scale
    X_h2h = pd.DataFrame(features_diff)
    X_h2h_scaled = scaler.transform(X_h2h)
    
    # Get probability
    prob_driver1_wins = h2h_model.predict_proba(X_h2h_scaled)[0, 1]
    
    return {
        'driver1': driver1_data.get('driverRef', 'Driver 1'),
        'driver2': driver2_data.get('driverRef', 'Driver 2'),
        'driver1_win_probability': prob_driver1_wins,
        'driver2_win_probability': 1 - prob_driver1_wins,
        'driver1_odds': 1 / prob_driver1_wins if prob_driver1_wins > 0.01 else 100,
        'driver2_odds': 1 / (1 - prob_driver1_wins) if (1 - prob_driver1_wins) > 0.01 else 100
    }


def kelly_criterion_bet_size(probability, odds, kelly_fraction=0.25):
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Args:
        probability: Estimated probability of winning
        odds: Decimal odds offered by bookmaker
        kelly_fraction: Fraction of Kelly to use (default 0.25 for conservative approach)
    
    Returns:
        Fraction of bankroll to bet
    """
    # Kelly formula: f = (p * o - 1) / (o - 1)
    # where p = probability, o = decimal odds
    
    if odds <= 1:
        return 0
    
    edge = probability * odds - 1
    if edge <= 0:
        return 0
    
    kelly_bet = edge / (odds - 1)
    
    # Apply fractional Kelly for risk management
    return min(kelly_bet * kelly_fraction, 0.1)  # Cap at 10% of bankroll


def calculate_expected_value(probability, odds):
    """
    Calculate expected value of a bet.
    
    Args:
        probability: Estimated probability of winning
        odds: Decimal odds offered
    
    Returns:
        Expected value as a percentage of stake
    """
    return (probability * odds - 1) * 100