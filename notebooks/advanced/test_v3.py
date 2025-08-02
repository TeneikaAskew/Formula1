#!/usr/bin/env python3
"""Test F1 predictions enhanced v3 with default inputs"""

import sys
import io

# Redirect stdin to provide default inputs
sys.stdin = io.StringIO('1\n')

# Import and run the predictor
from f1_predictions_enhanced_v3 import F1PrizePicksPredictor

try:
    print("Creating predictor with default settings...")
    predictor = F1PrizePicksPredictor()
    print("Predictor created successfully!")
    print("\nGenerating predictions...")
    predictions = predictor.generate_all_predictions()
    print(f"\nGenerated predictions for {len(predictions)} drivers")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore stdin
    sys.stdin = sys.__stdin__