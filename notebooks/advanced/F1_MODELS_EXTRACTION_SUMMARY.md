# F1 Models Module Extraction Summary

## Overview
Successfully extracted model-related classes and functions from `F1_Core_Models.ipynb` into a reusable module: `f1_models.py`

## Extracted Components

### 1. **Feature Engineering Functions**
- `create_temporal_features()` - Creates time-aware features with proper shifting to prevent data leakage
- `create_prediction_targets()` - Creates various binary classification targets (top_10, podium, winner, etc.)

### 2. **Model Management Functions**
- `get_regularized_models()` - Returns dictionary of pre-configured models with regularization to prevent overfitting
- `create_ensemble_model()` - Creates voting ensemble from base models
- `get_default_feature_columns()` - Returns list of feature columns based on window sizes

### 3. **Training and Evaluation Functions**
- `train_and_evaluate_model()` - Trains model and calculates comprehensive metrics with optional MLflow tracking
- `calibrate_model()` - Calibrates model probabilities using isotonic regression
- `perform_temporal_cv()` - Performs time series cross-validation
- `get_feature_importance()` - Extracts feature importance from tree-based models

### 4. **Utility Functions**
- `create_temporal_split()` - Creates train/validation/test splits based on dates
- `save_model_artifacts()` - Saves model with metadata
- `load_model_artifacts()` - Loads saved model artifacts
- `create_position_groups()` - Groups exact positions into categories
- `evaluate_position_predictions()` - F1-specific evaluation metrics
- `predict_race_probabilities()` - Generates predictions for a race

### 5. **F1ModelTrainer Class**
A comprehensive class that encapsulates the entire model training workflow:
- `prepare_data()` - Prepares data with features and temporal splits
- `train_all_models()` - Trains all regularized models with optional calibration
- `get_best_model()` - Returns best model based on specified metric
- `save_best_model()` - Saves best model with all artifacts

## Key Features

### Temporal Integrity
- All features use `.shift(1)` to ensure only past data is used
- Strict temporal train/validation/test splits
- Time series cross-validation support

### Model Regularization
- **Random Forest**: max_depth=8, min_samples_split=50, min_samples_leaf=20
- **Gradient Boosting**: max_depth=4, learning_rate=0.05, subsample=0.7
- **Logistic Regression**: C=0.1 (strong L2 regularization)

### MLflow Integration
- Optional MLflow tracking for experiments
- Logs model parameters, metrics, and artifacts
- Supports experiment naming

### Multiple Targets
- top_10: Points scoring position
- top_3: Podium finish
- beat_teammate: Better position than teammate
- scored_points: Any points scored
- winner: First place
- dnf_target: Did not finish

## Usage Example

```python
from f1_models import F1ModelTrainer
from f1db_data_loader import load_f1db_data

# Load data
data = load_f1db_data()
df = prepare_race_data(data)  # Your data preparation

# Initialize trainer
trainer = F1ModelTrainer(feature_windows=[3, 5, 10])

# Prepare data
data_splits = trainer.prepare_data(
    df, 
    target='top_10',
    train_end='2019-12-31',
    val_end='2021-12-31'
)

# Train models with MLflow tracking
models, metrics = trainer.train_all_models(
    data_splits,
    calibrate=True,
    mlflow_experiment='F1_Model_Training'
)

# Save best model
trainer.save_best_model('models/f1_top10_model.pkl')
```

## Benefits of Extraction

1. **Reusability**: Functions can be imported and used across different notebooks
2. **Consistency**: Ensures same preprocessing and model configurations
3. **Maintainability**: Single source of truth for model definitions
4. **Testing**: Can be unit tested independently
5. **Version Control**: Easier to track changes to model logic

## Integration Points

This module is designed to work with:
- `f1db_data_loader.py` - For loading F1 data
- `f1_market_calibration.py` - For betting market calibration
- Pipeline scripts that need consistent model training

## Notes

- The module includes comprehensive docstrings for all functions
- Error handling is minimal - assumes valid inputs
- Designed for binary classification tasks (can be extended for multi-class)
- All models use random_state=42 for reproducibility