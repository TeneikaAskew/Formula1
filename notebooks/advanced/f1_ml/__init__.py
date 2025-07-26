"""
F1 Machine Learning Package

This package contains the core components for F1 race predictions and analysis.
"""

from .features import F1FeatureStore
from .models import F1ModelTrainer, create_temporal_features, create_prediction_targets
from .evaluation import IntegratedF1Predictor
from .optimization import PrizePicksOptimizer
from .explainability import PredictionExplainer, PrizePicksExplainer

__all__ = [
    'F1FeatureStore',
    'F1ModelTrainer',
    'create_temporal_features',
    'create_prediction_targets',
    'IntegratedF1Predictor',
    'PrizePicksOptimizer',
    'PredictionExplainer',
    'PrizePicksExplainer'
]

__version__ = '1.0.0'