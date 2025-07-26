#!/usr/bin/env python3
"""
Test script for the refactored F1 ML package

This script tests that all modules can be imported and classes can be instantiated
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing F1 ML Package Imports...")
    print("=" * 60)
    
    # Test package import
    try:
        import f1_ml
        print("✓ f1_ml package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import f1_ml package: {e}")
        return False
    
    # Test individual module imports
    modules = [
        ('f1_ml.features', 'F1FeatureStore'),
        ('f1_ml.models', 'F1ModelTrainer'),
        ('f1_ml.evaluation', 'IntegratedF1Predictor'),
        ('f1_ml.optimization', 'PrizePicksOptimizer'),
        ('f1_ml.explainability', 'PredictionExplainer')
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            return False
        except AttributeError as e:
            print(f"✗ {class_name} not found in {module_name}: {e}")
            return False
    
    return True

def test_instantiation():
    """Test that classes can be instantiated"""
    print("\nTesting Class Instantiation...")
    print("=" * 60)
    
    # Test F1FeatureStore
    try:
        from f1_ml.features import F1FeatureStore
        fs = F1FeatureStore()
        print("✓ F1FeatureStore instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate F1FeatureStore: {e}")
        return False
    
    # Test F1ModelTrainer
    try:
        from f1_ml.models import F1ModelTrainer
        trainer = F1ModelTrainer(mlflow_tracking=False)
        print("✓ F1ModelTrainer instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate F1ModelTrainer: {e}")
        return False
    
    # Test IntegratedF1Predictor
    try:
        from f1_ml.evaluation import IntegratedF1Predictor
        predictor = IntegratedF1Predictor()
        print("✓ IntegratedF1Predictor instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate IntegratedF1Predictor: {e}")
        return False
    
    # Test PrizePicksOptimizer
    try:
        from f1_ml.optimization import PrizePicksOptimizer
        optimizer = PrizePicksOptimizer()
        print("✓ PrizePicksOptimizer instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate PrizePicksOptimizer: {e}")
        return False
    
    # Test Explainers
    try:
        from f1_ml.explainability import PredictionExplainer, PrizePicksExplainer
        pred_exp = PredictionExplainer()
        pp_exp = PrizePicksExplainer()
        print("✓ PredictionExplainer instantiated successfully")
        print("✓ PrizePicksExplainer instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate explainers: {e}")
        return False
    
    return True

def test_integration():
    """Test integration between modules"""
    print("\nTesting Module Integration...")
    print("=" * 60)
    
    try:
        # Import all components
        from f1_ml import (
            F1FeatureStore, 
            F1ModelTrainer,
            IntegratedF1Predictor,
            PrizePicksOptimizer,
            PredictionExplainer,
            PrizePicksExplainer
        )
        
        # Test that components can work together
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dummy_predictions = pd.DataFrame({
            'driver': ['Driver A', 'Driver B', 'Driver C'],
            'driverId': [1, 2, 3],
            'top10_prob': [0.8, 0.6, 0.4],
            'top5_prob': [0.6, 0.4, 0.2],
            'top3_prob': [0.4, 0.2, 0.1],
            'points_prob': [0.85, 0.65, 0.45],
            'beat_teammate_prob': [0.6, 0.4, 0.5],
            'confidence': [0.8, 0.7, 0.6]
        })
        
        # Test optimizer with dummy data
        optimizer = PrizePicksOptimizer()
        picks = optimizer.generate_all_picks(dummy_predictions, min_edge=0.05)
        print(f"✓ Generated {len(picks)} picks from predictions")
        
        # Test explainer
        explainer = PrizePicksExplainer()
        if len(picks) > 0:
            # Create a dummy parlay
            parlay = {
                'picks': picks.head(2),
                'n_picks': 2,
                'correlation': 0.3,
                'adjusted_prob': 0.5,
                'expected_value': 0.1,
                'kelly_stake': 0.05,
                'payout': 3.0,
                'bet_size': 10.0
            }
            explanation = explainer.explain_parlay(parlay, detailed=False)
            print("✓ Generated parlay explanation successfully")
        
        print("\n✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("F1 ML PACKAGE TEST SUITE")
    print("=" * 80)
    
    # Run tests
    import_success = test_imports()
    if not import_success:
        print("\n❌ Import tests failed. Cannot continue.")
        return 1
    
    instantiation_success = test_instantiation()
    if not instantiation_success:
        print("\n❌ Instantiation tests failed.")
        return 1
    
    integration_success = test_integration()
    if not integration_success:
        print("\n❌ Integration tests failed.")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe F1 ML package is working correctly.")
    print("You can now use it in notebooks and scripts by importing:")
    print("  from f1_ml import F1FeatureStore, PrizePicksOptimizer, etc.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())