"""
F1 Models Module - Core ML models and utilities for F1 predictions
Extracted from F1_Core_Models.ipynb

This module provides:
1. Temporal feature engineering functions that prevent data leakage
2. Regularized model definitions (Random Forest, Gradient Boosting, Logistic Regression)
3. Model training and evaluation utilities with MLflow support
4. Model calibration for better probability estimates
5. Time series cross-validation
6. F1-specific evaluation metrics
7. Comprehensive F1ModelTrainer class for end-to-end model development

Key Features:
- Strict temporal validation to prevent future data leakage
- Multiple prediction targets (top_10, podium, winner, DNF, etc.)
- Ensemble model support
- MLflow integration for experiment tracking
- Model persistence with joblib

Usage:
    from f1_models import F1ModelTrainer, create_temporal_features
    
    # Create trainer
    trainer = F1ModelTrainer()
    
    # Prepare data with temporal features
    data_splits = trainer.prepare_data(df, target='top_10')
    
    # Train all models with MLflow tracking
    models, metrics = trainer.train_all_models(
        data_splits, 
        calibrate=True,
        mlflow_experiment='F1_Models'
    )
    
    # Save best model
    trainer.save_best_model('f1_model_top10.pkl')
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
import warnings
from datetime import datetime
import joblib

# Optional MLflow support
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
warnings.filterwarnings('ignore')


def create_temporal_features(df, windows=[3, 5, 10]):
    """
    Create features that STRICTLY respect temporal constraints.
    All features use .shift(1) to ensure we only use past data.
    
    Args:
        df: DataFrame with race results
        windows: List of window sizes for rolling features
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Sort by driver and date to ensure proper ordering
    df = df.sort_values(['driverId', 'date'])
    
    # Basic position features (shifted to avoid leakage)
    df['prev_position'] = df.groupby('driverId')['positionOrder'].shift(1)
    df['prev_grid'] = df.groupby('driverId')['grid'].shift(1)
    df['prev_points'] = df.groupby('driverId')['points'].shift(1)
    
    # Rolling averages (all shifted)
    for w in windows:
        # Driver performance
        df[f'avg_position_{w}'] = df.groupby('driverId')['positionOrder'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f'avg_points_{w}'] = df.groupby('driverId')['points'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        
        # DNF rate
        df['dnf'] = (df['positionOrder'].isna() | (df['statusId'] > 1)).astype(int)
        df[f'dnf_rate_{w}'] = df.groupby('driverId')['dnf'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        
        # Constructor performance
        df[f'constructor_avg_points_{w}'] = df.groupby('constructorId')['points'].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
    
    # Career statistics (always based on past races)
    df['races_completed'] = df.groupby('driverId').cumcount()
    df['career_points'] = df.groupby('driverId')['points'].cumsum().shift(1)
    df['career_wins'] = df.groupby('driverId')['position'].transform(
        lambda x: (x == 1).shift(1).cumsum()
    )
    df['career_podiums'] = df.groupby('driverId')['position'].transform(
        lambda x: (x <= 3).shift(1).cumsum()
    )
    
    # Track-specific features (based on past performance)
    df['driver_track_avg'] = df.groupby(['driverId', 'circuitId'])['positionOrder'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df['driver_track_races'] = df.groupby(['driverId', 'circuitId']).cumcount()
    
    return df


def create_prediction_targets(df):
    """
    Create various prediction targets for F1 models.
    
    Args:
        df: DataFrame with race results
        
    Returns:
        DataFrame with target columns added
    """
    df = df.copy()
    
    # Target 1: Top 10 finish (points scoring)
    df['top_10'] = (df['positionOrder'] <= 10).astype(int)
    
    # Target 2: Top 3 finish (podium)
    df['top_3'] = (df['positionOrder'] <= 3).astype(int)
    
    # Target 3: Beat teammate
    teammate_results = df.groupby(['raceId', 'constructorId'])['positionOrder'].rank(method='min')
    df['beat_teammate'] = (teammate_results == 1).astype(int)
    
    # Target 4: Points finish
    df['scored_points'] = (df['points'] > 0).astype(int)
    
    # Target 5: Winner
    df['winner'] = (df['positionOrder'] == 1).astype(int)
    
    # Target 6: DNF
    df['dnf_target'] = (df['positionOrder'].isna() | (df['statusId'] > 1)).astype(int)
    
    return df


def get_regularized_models():
    """
    Get dictionary of regularized models optimized to prevent overfitting.
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=0.1,  # Strong regularization
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # Shallow trees
            min_samples_split=50,  # Require many samples to split
            min_samples_leaf=20,  # Require many samples in leaves
            max_features='sqrt',  # Use subset of features
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,  # Very shallow trees
            learning_rate=0.05,  # Small learning rate
            subsample=0.7,  # Use 70% of data for each tree
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
    }
    return models


def create_ensemble_model(base_models, voting='soft', weights=None):
    """
    Create a voting ensemble from base models.
    
    Args:
        base_models: Dictionary or list of (name, model) tuples
        voting: 'soft' for probability averaging, 'hard' for majority voting
        weights: Optional weights for each model
        
    Returns:
        VotingClassifier instance
    """
    if isinstance(base_models, dict):
        estimators = list(base_models.items())
    else:
        estimators = base_models
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1
    )
    
    return ensemble


def get_default_feature_columns(windows=[3, 5, 10]):
    """
    Get the default list of feature columns used in models.
    
    Args:
        windows: List of window sizes to include features for
        
    Returns:
        List of feature column names
    """
    feature_cols = ['grid', 'prev_position', 'prev_points']
    
    # Add window-based features
    for w in windows:
        feature_cols.extend([
            f'avg_position_{w}',
            f'avg_points_{w}',
            f'dnf_rate_{w}',
        ])
    
    # Add constructor features for smaller windows
    for w in [w for w in windows if w <= 5]:
        feature_cols.append(f'constructor_avg_points_{w}')
    
    # Add career and track features
    feature_cols.extend([
        'races_completed', 'career_points', 'career_wins', 'career_podiums',
        'driver_track_avg', 'driver_track_races'
    ])
    
    return feature_cols


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, 
                           mlflow_tracking=False, model_name=None):
    """
    Train a model and evaluate its performance.
    
    Args:
        model: Sklearn model instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Optional test data
        mlflow_tracking: Whether to log to MLflow
        model_name: Name for MLflow run
        
    Returns:
        Dictionary of metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Probabilities for AUC
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_acc': accuracy_score(y_train, train_pred),
        'val_acc': accuracy_score(y_val, val_pred),
        'train_auc': roc_auc_score(y_train, train_prob),
        'val_auc': roc_auc_score(y_val, val_prob),
        'val_precision': precision_score(y_val, val_pred),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'overfit_score': None  # Will be calculated below
    }
    
    # Add test metrics if provided
    if X_test is not None and y_test is not None:
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        metrics.update({
            'test_acc': accuracy_score(y_test, test_pred),
            'test_auc': roc_auc_score(y_test, test_prob),
            'test_precision': precision_score(y_test, test_pred),
            'test_recall': recall_score(y_test, test_pred),
            'test_f1': f1_score(y_test, test_pred)
        })
    
    # Calculate overfitting score
    metrics['overfit_score'] = metrics['train_acc'] - metrics['val_acc']
    
    # MLflow tracking if enabled
    if mlflow_tracking and MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=model_name or model.__class__.__name__):
            # Log model parameters
            mlflow.log_params(model.get_params())
            
            # Log metrics
            for metric_name, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(metric_name, value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
    
    return metrics


def calibrate_model(model, X_train, y_train, method='isotonic', cv=3):
    """
    Calibrate a model for better probability estimates.
    
    Args:
        model: Trained sklearn model
        X_train, y_train: Training data for calibration
        method: Calibration method ('isotonic' or 'sigmoid')
        cv: Number of CV folds
        
    Returns:
        Calibrated model
    """
    calibrated_model = CalibratedClassifierCV(
        model, 
        method=method,
        cv=cv
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def perform_temporal_cv(model, X, y, n_splits=5):
    """
    Perform time series cross-validation.
    
    Args:
        model: Sklearn model instance
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV splits
        
    Returns:
        List of CV scores
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone model to avoid fitting on previous data
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_cv, y_train_cv)
        
        val_pred = model_clone.predict(X_val_cv)
        score = accuracy_score(y_val_cv, val_pred)
        scores.append(score)
    
    return scores


def create_temporal_split(df, train_end_date, val_end_date):
    """
    Create temporal train/validation/test split.
    
    Args:
        df: DataFrame with 'date' column
        train_end_date: End date for training set (string or datetime)
        val_end_date: End date for validation set (string or datetime)
        
    Returns:
        Three boolean masks: train_mask, val_mask, test_mask
    """
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)
    
    train_mask = df['date'] <= train_end
    val_mask = (df['date'] > train_end) & (df['date'] <= val_end)
    test_mask = df['date'] > val_end
    
    return train_mask, val_mask, test_mask


def save_model_artifacts(model, scaler, feature_cols, metrics, output_path, 
                        model_name=None, additional_info=None):
    """
    Save model artifacts including model, scaler, and metadata.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        metrics: Dictionary of model metrics
        output_path: Path to save the artifacts
        model_name: Optional model name
        additional_info: Optional dictionary of additional information
    """
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'model_name': model_name or model.__class__.__name__,
        'metrics': metrics,
        'training_date': datetime.now().isoformat(),
        'data_version': 'f1db_latest'
    }
    
    if additional_info:
        artifacts.update(additional_info)
    
    joblib.dump(artifacts, output_path)
    return artifacts


def load_model_artifacts(model_path):
    """
    Load model artifacts from disk.
    
    Args:
        model_path: Path to the saved model artifacts
        
    Returns:
        Dictionary of model artifacts
    """
    return joblib.load(model_path)


def get_feature_importance(model, feature_cols, top_n=15):
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model
        feature_cols: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        return None


class F1ModelTrainer:
    """
    Comprehensive class for training F1 prediction models with proper temporal validation.
    """
    
    def __init__(self, feature_windows=[3, 5, 10]):
        self.feature_windows = feature_windows
        self.scaler = StandardScaler()
        self.models = {}
        self.metrics = {}
        self.feature_cols = None
        
    def prepare_data(self, df, target='top_10', train_end='2019-12-31', val_end='2021-12-31'):
        """
        Prepare data with features and temporal splits.
        
        Args:
            df: Raw DataFrame with race results
            target: Target column to predict
            train_end: End date for training set
            val_end: End date for validation set
            
        Returns:
            Dictionary with prepared data splits
        """
        # Create features
        df_features = create_temporal_features(df, self.feature_windows)
        
        # Create targets
        df_features = create_prediction_targets(df_features)
        
        # Get feature columns based on feature windows
        self.feature_cols = get_default_feature_columns(self.feature_windows)
        
        # Add qualifying features if available
        if 'qualifying_position' in df_features.columns:
            self.feature_cols.extend(['qualifying_position', 'quali_grid_diff'])
        
        # Create modeling dataset
        df_model = df_features.dropna(subset=self.feature_cols + [target])
        
        # Create temporal splits
        train_mask, val_mask, test_mask = create_temporal_split(df_model, train_end, val_end)
        
        # Prepare data splits
        data_splits = {
            'X_train': df_model[train_mask][self.feature_cols],
            'X_val': df_model[val_mask][self.feature_cols],
            'X_test': df_model[test_mask][self.feature_cols],
            'y_train': df_model[train_mask][target],
            'y_val': df_model[val_mask][target],
            'y_test': df_model[test_mask][target],
            'df_model': df_model,
            'masks': {'train': train_mask, 'val': val_mask, 'test': test_mask}
        }
        
        # Scale features
        data_splits['X_train_scaled'] = self.scaler.fit_transform(data_splits['X_train'])
        data_splits['X_val_scaled'] = self.scaler.transform(data_splits['X_val'])
        data_splits['X_test_scaled'] = self.scaler.transform(data_splits['X_test'])
        
        return data_splits
    
    def train_all_models(self, data_splits, calibrate=True, mlflow_experiment=None):
        """
        Train all regularized models and optionally calibrate them.
        
        Args:
            data_splits: Dictionary with prepared data splits
            calibrate: Whether to calibrate the best model
            mlflow_experiment: Optional MLflow experiment name
            
        Returns:
            Dictionary of trained models and metrics
        """
        # Set up MLflow experiment if specified
        if mlflow_experiment and MLFLOW_AVAILABLE:
            mlflow.set_experiment(mlflow_experiment)
        # Get regularized models
        model_dict = get_regularized_models()
        
        # Train each model
        for name, model in model_dict.items():
            print(f"Training {name}...")
            
            metrics = train_and_evaluate_model(
                model,
                data_splits['X_train_scaled'],
                data_splits['y_train'],
                data_splits['X_val_scaled'],
                data_splits['y_val'],
                data_splits['X_test_scaled'],
                data_splits['y_test'],
                mlflow_tracking=(mlflow_experiment is not None),
                model_name=name
            )
            
            self.models[name] = model
            self.metrics[name] = metrics
            
            print(f"Val Accuracy: {metrics['val_acc']:.3f}, Val AUC: {metrics['val_auc']:.3f}")
            print(f"Overfitting score: {metrics['overfit_score']:.3f}")
        
        # Calibrate best model
        if calibrate:
            best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['val_auc'])
            best_model = self.models[best_model_name]
            
            print(f"\nCalibrating {best_model_name}...")
            calibrated = calibrate_model(
                best_model,
                data_splits['X_train_scaled'],
                data_splits['y_train']
            )
            
            self.models[f"{best_model_name} (Calibrated)"] = calibrated
            
            # Evaluate calibrated model
            cal_metrics = train_and_evaluate_model(
                calibrated,
                data_splits['X_train_scaled'],
                data_splits['y_train'],
                data_splits['X_val_scaled'],
                data_splits['y_val'],
                data_splits['X_test_scaled'],
                data_splits['y_test']
            )
            self.metrics[f"{best_model_name} (Calibrated)"] = cal_metrics
        
        return self.models, self.metrics
    
    def get_best_model(self, metric='val_auc'):
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_name, model, metrics)
        """
        best_name = max(self.metrics, key=lambda x: self.metrics[x][metric])
        return best_name, self.models[best_name], self.metrics[best_name]
    
    def save_best_model(self, output_path, metric='val_auc'):
        """
        Save the best model with all artifacts.
        
        Args:
            output_path: Path to save the model
            metric: Metric to use for selection
            
        Returns:
            Saved artifacts dictionary
        """
        best_name, best_model, best_metrics = self.get_best_model(metric)
        
        artifacts = save_model_artifacts(
            best_model,
            self.scaler,
            self.feature_cols,
            best_metrics,
            output_path,
            model_name=best_name
        )
        
        print(f"Saved {best_name} to {output_path}")
        print(f"Validation {metric}: {best_metrics[metric]:.3f}")
        
        return artifacts


def create_position_groups(position, groups='default'):
    """
    Convert exact position to grouped categories for more realistic prediction.
    
    Args:
        position: Position values (1-20+)
        groups: Grouping strategy
            - 'default': Win(1), Podium(2-3), Points(4-10), Outside(11+)
            - 'detailed': More granular groups
            - 'binary': Points(1-10) vs No Points(11+)
            
    Returns:
        Categorical position groups
    """
    if groups == 'default':
        conditions = [
            position == 1,
            (position >= 2) & (position <= 3),
            (position >= 4) & (position <= 10),
            position > 10
        ]
        choices = ['Win', 'Podium', 'Points', 'Outside']
    elif groups == 'detailed':
        conditions = [
            position == 1,
            (position >= 2) & (position <= 3),
            (position >= 4) & (position <= 6),
            (position >= 7) & (position <= 10),
            (position >= 11) & (position <= 15),
            position > 15
        ]
        choices = ['Win', 'Podium', 'Top6', 'Points', 'Midfield', 'Back']
    elif groups == 'binary':
        return (position <= 10).astype(int)
    else:
        raise ValueError(f"Unknown grouping strategy: {groups}")
    
    return pd.Series(np.select(conditions, choices, default='DNF'))


def evaluate_position_predictions(y_true, y_pred, position_type='exact'):
    """
    Evaluate position predictions with F1-specific metrics.
    
    Args:
        y_true: True positions
        y_pred: Predicted positions
        position_type: 'exact' or 'grouped'
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    if position_type == 'exact':
        # Mean Absolute Error for position
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Position accuracy within N places
        for n in [0, 1, 2, 3]:
            within_n = np.mean(np.abs(y_true - y_pred) <= n)
            metrics[f'within_{n}_places'] = within_n
    
    # Key position accuracies
    metrics['winner_accuracy'] = np.mean((y_true == 1) & (y_pred == 1))
    metrics['podium_accuracy'] = np.mean((y_true <= 3) & (y_pred <= 3))
    metrics['points_accuracy'] = np.mean((y_true <= 10) & (y_pred <= 10))
    
    # Confusion between points/no-points
    points_true = y_true <= 10
    points_pred = y_pred <= 10
    metrics['points_precision'] = precision_score(points_true, points_pred)
    metrics['points_recall'] = recall_score(points_true, points_pred)
    
    return metrics


def predict_race_probabilities(model, X, feature_cols=None, scaler=None, 
                              calibrated=True):
    """
    Generate probability predictions for a race.
    
    Args:
        model: Trained model (binary classifier)
        X: Feature matrix for all drivers in the race
        feature_cols: List of feature columns (if X is DataFrame)
        scaler: Fitted StandardScaler
        calibrated: Whether model is already calibrated
        
    Returns:
        DataFrame with driver predictions and probabilities
    """
    # Prepare features
    if feature_cols and isinstance(X, pd.DataFrame):
        X_model = X[feature_cols]
    else:
        X_model = X
    
    # Scale if scaler provided
    if scaler:
        X_scaled = scaler.transform(X_model)
    else:
        X_scaled = X_model
    
    # Get predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    # Add driver info if X is DataFrame
    if isinstance(X, pd.DataFrame):
        if 'driverId' in X.columns:
            results['driverId'] = X['driverId'].values
        if 'driverRef' in X.columns:
            results['driverRef'] = X['driverRef'].values
    
    # Sort by probability
    results = results.sort_values('probability', ascending=False)
    
    return results