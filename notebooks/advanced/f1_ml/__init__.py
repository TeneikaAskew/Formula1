"""
F1 Machine Learning Package

This package contains the core components for F1 race predictions and analysis.
"""

from .features import (
    F1FeatureStore,
    create_track_features,
    get_weather_features,
    create_momentum_features,
    create_strategy_features,
    create_advanced_metrics
)
from .weather import F1WeatherProvider
from .models import F1ModelTrainer, create_temporal_features, create_prediction_targets
from .evaluation import IntegratedF1Predictor
from .optimization import PrizePicksOptimizer
from .explainability import PredictionExplainer, PrizePicksExplainer
# Import data utilities from f1db_data_loader instead
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from f1db_data_loader import fix_column_mappings, merge_race_data, get_recent_results

__all__ = [
    'F1FeatureStore',
    'create_track_features',
    'get_weather_features',
    'create_momentum_features',
    'create_strategy_features',
    'create_advanced_metrics',
    'F1ModelTrainer',
    'create_temporal_features',
    'create_prediction_targets',
    'IntegratedF1Predictor',
    'PrizePicksOptimizer',
    'PredictionExplainer',
    'PrizePicksExplainer',
    'F1WeatherProvider',
    'fix_column_mappings',
    'merge_race_data',
    'get_recent_results'
]

__version__ = '1.0.0'