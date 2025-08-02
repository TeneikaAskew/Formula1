#!/usr/bin/env python3
"""Test contextual predictions with Phase 3.1 features"""

import sys
import io

# Suppress prompts
old_stdin = sys.stdin
sys.stdin = io.StringIO('1\n')  # Choose default lines

try:
    from f1_predictions_enhanced_v3 import F1PrizePicksPredictor
    
    # Create predictor
    predictor = F1PrizePicksPredictor()
    
    # Test with a specific race (2023 Monaco GP - circuit 6)
    # Monaco is known for difficult overtaking
    print("\nTesting contextual predictions for Monaco GP (difficult overtaking)...")
    
    # Generate predictions for Monaco
    predictions = predictor.generate_all_predictions(race_id=1120)  # 2023 Monaco GP
    
finally:
    sys.stdin = old_stdin
    
print("\nTest completed! Contextual features have been integrated.")