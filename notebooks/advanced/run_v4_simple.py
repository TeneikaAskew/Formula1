#!/usr/bin/env python3
"""Simple runner for v4 that handles initialization issues"""

import warnings
warnings.filterwarnings('ignore')

# Mock the problematic modules with simple versions
class SimpleBayesianPriors:
    def __init__(self, data_dict=None):
        self.team_priors = {}
        self.track_priors = {}
        self.driver_priors = {}
        
    def get_hierarchical_prior(self, driver_id, constructor_id, circuit_id, prop_type):
        # Return simple default priors
        base_rates = {
            'points': 0.45,
            'overtakes': 0.65,
            'dnf': 0.12,
            'starting_position': 0.50,
            'pit_stops': 0.70,
            'teammate_overtakes': 0.50
        }
        return {
            'probability': base_rates.get(prop_type, 0.5),
            'confidence': 0.3
        }

class SimpleContextualFeatures:
    def __init__(self, data_dict=None):
        pass
        
    def get_all_contextual_features(self, driver_id, constructor_id, circuit_id, race_id, weather_data=None):
        return {
            'track_overtaking_difficulty': 1.0,
            'momentum_score': 0,
            'risk_score': 0.5
        }

# Replace the imports
import sys
sys.modules['f1_bayesian_priors'] = sys.modules[__name__]
sys.modules['f1_contextual_features'] = sys.modules[__name__]

F1BayesianPriors = SimpleBayesianPriors
F1ContextualFeatures = SimpleContextualFeatures

# Now import and run v4
from f1_predictions_enhanced_v4_fixed import main

if __name__ == "__main__":
    main()