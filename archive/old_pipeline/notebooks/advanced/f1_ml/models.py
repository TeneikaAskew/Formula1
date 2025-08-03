"""
F1 Model Training Module

This module contains model training functions and classes for F1 predictions.
Includes feature engineering, model configuration, training, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')


def create_temporal_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Create time-aware features with proper temporal shifting
    """
    df = df.sort_values(['driverId', 'date']).copy()
    
    # Rolling averages - SHIFTED to avoid leakage
    for window in windows:
        df[f'avg_position_{window}'] = df.groupby('driverId')['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'avg_points_{window}'] = df.groupby('driverId')['points'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'dnf_rate_{window}'] = df.groupby('driverId')['dnf'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # Career statistics - CUMULATIVE up to previous race
    df['career_wins'] = df.groupby('driverId')['win'].transform(lambda x: x.shift(1).cumsum())
    df['career_podiums'] = df.groupby('driverId')['podium'].transform(lambda x: x.shift(1).cumsum())
    df['career_points'] = df.groupby('driverId')['points'].transform(lambda x: x.shift(1).cumsum())
    df['career_races'] = df.groupby('driverId').cumcount()
    
    # Constructor features - SHIFTED
    df['constructor_avg_position'] = df.groupby('constructorId')['positionOrder'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    
    # Track-specific features - using PREVIOUS performances at this track
    df['driver_track_avg'] = df.groupby(['driverId', 'circuitId'])['positionOrder'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Fill NaN values with sensible defaults
    position_cols = [col for col in df.columns if 'position' in col and col != 'positionOrder']
    df[position_cols] = df[position_cols].fillna(10.5)  # Middle of the grid
    
    other_cols = [col for col in df.columns if 'points' in col or 'dnf' in col or 'wins' in col or 'podiums' in col]
    df[other_cols] = df[other_cols].fillna(0)
    
    return df


def create_prediction_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification targets for F1 predictions
    """
    df['top_10'] = (df['positionOrder'] <= 10).astype(int)
    df['podium'] = (df['positionOrder'] <= 3).astype(int)
    df['winner'] = (df['positionOrder'] == 1).astype(int)
    df['points_finish'] = (df['points'] > 0).astype(int)
    df['top_5'] = (df['positionOrder'] <= 5).astype(int)
    
    return df


def get_regularized_models() -> Dict:
    """
    Get pre-configured models with regularization to prevent overfitting
    """
    return {
        'rf': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,          # Shallow trees
            min_samples_split=20,  # Require more samples to split
            min_samples_leaf=10,   # Require more samples in leaves
            max_features='sqrt',   # Use subset of features
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,           # Even shallower for GB
            min_samples_split=20,
            min_samples_leaf=10,
            learning_rate=0.1,
            subsample=0.8,         # Use subset of data
            random_state=42
        ),
        'lr': LogisticRegression(
            C=0.1,                 # Strong regularization
            max_iter=1000,
            random_state=42
        )
    }


def create_ensemble_model(models: Dict) -> VotingClassifier:
    """
    Create an ensemble of the given models
    """
    return VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    model_name: str,
    mlflow=None
) -> Tuple[object, Dict]:
    """
    Train and evaluate a model with comprehensive metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Log to MLflow if available
    if mlflow:
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_name}_{metric_name}", value)
    
    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, metrics


def calibrate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
    """
    Calibrate model probabilities using isotonic regression
    """
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)
    return calibrated


def perform_temporal_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model,
    n_splits: int = 3
) -> Dict:
    """
    Perform time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    X = df[feature_cols]
    y = df[target_col]
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train and evaluate
        model.fit(X_train_scaled, y_train)
        score = model.score(X_val_scaled, y_val)
        cv_scores.append(score)
    
    return {
        'mean_cv_score': np.mean(cv_scores),
        'std_cv_score': np.std(cv_scores),
        'cv_scores': cv_scores
    }


def save_model(model, scaler, feature_cols: List[str], model_path: str, metadata: Dict = None):
    """
    Save model with metadata
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path: str) -> Dict:
    """
    Load model with metadata
    """
    return joblib.load(model_path)


def create_race_predictions(model_data: Dict, race_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictions for a specific race
    """
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['feature_columns']
    
    # Prepare features
    X = race_data[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_scaled)
    
    # Create results dataframe
    results = race_data[['driverId', 'driver_name', 'constructorId', 'constructor_name']].copy()
    results['prediction_score'] = predictions
    results['predicted_position'] = results['prediction_score'].rank(method='dense', ascending=False)
    
    return results.sort_values('predicted_position')


def evaluate_f1_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    """
    Evaluate predictions with F1-specific metrics
    """
    # Convert position predictions to binary outcomes
    true_winner = (y_true == 1).astype(int)
    pred_winner = (y_pred == 1).astype(int)
    
    true_podium = (y_true <= 3).astype(int)
    pred_podium = (y_pred <= 3).astype(int)
    
    true_points = (y_true <= 10).astype(int)
    pred_points = (y_pred <= 10).astype(int)
    
    metrics = {
        'winner_accuracy': accuracy_score(true_winner, pred_winner),
        'podium_accuracy': accuracy_score(true_podium, pred_podium),
        'points_accuracy': accuracy_score(true_points, pred_points),
        'position_mae': np.abs(y_true - y_pred).mean(),
        'position_rmse': np.sqrt(((y_true - y_pred) ** 2).mean())
    }
    
    return metrics


def get_position_group(position: int) -> str:
    """
    Group positions into categories for analysis
    """
    if position == 1:
        return 'Winner'
    elif position <= 3:
        return 'Podium'
    elif position <= 10:
        return 'Points'
    else:
        return 'Outside Points'


class F1ModelTrainer:
    """
    Comprehensive model trainer for F1 predictions
    """
    def __init__(self, mlflow_tracking: bool = False):
        self.mlflow_tracking = mlflow_tracking
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
        if mlflow_tracking:
            import mlflow
            self.mlflow = mlflow
        
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data with temporal split
        """
        # Add temporal features
        df = create_temporal_features(df)
        
        # Create targets
        df = create_prediction_targets(df)
        
        # Select features
        feature_cols = [
            'grid', 'driver_age', 'avg_position_3', 'avg_position_5', 'avg_position_10',
            'avg_points_3', 'avg_points_5', 'avg_points_10',
            'dnf_rate_3', 'dnf_rate_5', 'dnf_rate_10',
            'career_wins', 'career_podiums', 'career_points', 'career_races',
            'constructor_avg_position', 'driver_track_avg'
        ]
        
        # Remove rows with NaN in features
        df_clean = df.dropna(subset=feature_cols + [target_col])
        
        # Temporal split
        split_date = df_clean['date'].quantile(0.8)
        train_data = df_clean[df_clean['date'] <= split_date]
        test_data = df_clean[df_clean['date'] > split_date]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, df: pd.DataFrame, target_col: str, model_name: str) -> Dict:
        """
        Train a model for a specific target
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get models
        models = get_regularized_models()
        
        # Train ensemble
        ensemble = create_ensemble_model(models)
        trained_model, metrics = train_and_evaluate_model(
            pd.DataFrame(X_train_scaled, columns=X_train.columns),
            y_train,
            pd.DataFrame(X_test_scaled, columns=X_test.columns),
            y_test,
            ensemble,
            model_name,
            self.mlflow if self.mlflow_tracking else None
        )
        
        # Store model components
        self.models[target_col] = trained_model
        self.scalers[target_col] = scaler
        self.feature_columns[target_col] = list(X_train.columns)
        
        return metrics
    
    def save_all_models(self, base_path: str = '.'):
        """
        Save all trained models
        """
        for target_col, model in self.models.items():
            model_path = os.path.join(base_path, f'f1_model_{target_col}.pkl')
            save_model(
                model,
                self.scalers[target_col],
                self.feature_columns[target_col],
                model_path,
                {'target': target_col}
            )


# Export key components
__all__ = [
    'create_temporal_features',
    'create_prediction_targets',
    'get_regularized_models',
    'create_ensemble_model',
    'train_and_evaluate_model',
    'calibrate_model',
    'perform_temporal_cv',
    'save_model',
    'load_model',
    'create_race_predictions',
    'evaluate_f1_predictions',
    'get_position_group',
    'F1ModelTrainer'
]