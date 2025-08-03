# F1 Project File Dependency Guide

## Production Files and Their Dependencies

### 1. **f1_predictions_v4_production.py** (MAIN PRODUCTION FILE)
**Dependencies:**
- `f1db_data_loader.py` ✅
- `f1_performance_analysis.py` ✅
- `f1_probability_calibration.py` ✅
- `f1_risk_dashboard.py` ✅

### 2. **f1_predictions_enhanced_v3.py**
**Dependencies:**
- `f1db_data_loader.py` ✅
- `f1_performance_analysis.py` ✅
- `f1_probability_calibration.py` ✅
- `f1_bayesian_priors.py` ✅
- `f1_contextual_features.py` ✅

### 3. **f1_performance_analysis.py**
**Dependencies:**
- Only pandas/numpy (no custom modules)

## Files to KEEP (Required Dependencies)

### Core Data & Analysis
- ✅ `f1db_data_loader.py` - Core data loader
- ✅ `f1_performance_analysis.py` - Performance analyzer
- ✅ `f1_market_calibration.py` - Market calibration

### Prediction Modules (Used by v3 and v4)
- ✅ `f1_probability_calibration.py` - Used by both v3 and v4
- ✅ `f1_bayesian_priors.py` - Used by v3
- ✅ `f1_contextual_features.py` - Used by v3
- ✅ `f1_risk_dashboard.py` - Used by v4

### Ensemble & Analysis (May be used)
- ✅ `f1_correlation_analysis.py` - Correlation analysis
- ✅ `f1_ensemble_integration.py` - Ensemble methods

### Main Pipeline
- ✅ `run_f1_pipeline.py` - Main pipeline runner
- ✅ `fetch_weather_data.py` - Weather data fetcher

### Production Versions
- ✅ `f1_predictions_v4_production.py` - Main production file
- ✅ `f1_predictions_enhanced_v3.py` - Still being used

## Files to REMOVE (Old/Test/Debug)

### Old Versions
- ❌ `f1_predictions_enhanced.py` - Original version
- ❌ `f1_predictions_enhanced_v2.py` - Old version
- ❌ `f1_predictions_enhanced_v4.py` - Problematic version
- ❌ `f1_predictions_enhanced_v4_fixed.py` - Incomplete fix
- ❌ `f1_predictions_enhanced_v4_full.py` - Has issues
- ❌ `f1_predictions_simple.py` - Simple version
- ❌ `f1_predictions_v4_minimal.py` - Minimal version

### Test Files
- ❌ `test_contextual_predictions.py`
- ❌ `test_dashboard.py`
- ❌ `test_v4_output.py`
- ❌ `view_dashboard_example.py`

### Temporary Files
- ❌ `run_v4_simple.py`
- ❌ `explain_sprint_data.py`
- ❌ `extracted_explainer_classes.py`
- ❌ `f1_models.py` (superseded by f1_ml/models.py)

### Duplicate Functionality
- ❌ `f1_backtesting_framework.py` (functionality in f1_ml/)

## Summary

Keep 12 core files that are actively used as dependencies. Remove only the old versions, test files, and temporary files that aren't referenced by the production code.

The key insight is that v3 and v4_production are both in use and have different dependency requirements, so we need to keep their dependencies.