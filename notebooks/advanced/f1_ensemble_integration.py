#!/usr/bin/env python3
"""F1 Ensemble Integration Module - Phase 3.2 Implementation

This module integrates the existing ML ensemble models with our enhanced 
prediction system, adding XGBoost and LightGBM to the mix.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from f1_ml.models import get_regularized_models, create_ensemble_model
from f1_ml.optimization import KellyCriterion, PrizePicksOptimizer

# Additional ensemble methods
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class F1AdvancedEnsemble:
    """Advanced ensemble methods for F1 predictions"""
    
    def __init__(self):
        self.base_models = get_regularized_models()
        self.advanced_models = {}
        self._initialize_advanced_models()
        
    def _initialize_advanced_models(self):
        """Initialize XGBoost and LightGBM if available"""
        if HAS_XGBOOST:
            self.advanced_models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,  # Regularization
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                n_jobs=-1
            )
            
        if HAS_LIGHTGBM:
            self.advanced_models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
    
    def get_all_models(self) -> Dict:
        """Get all available models including advanced ones"""
        all_models = self.base_models.copy()
        all_models.update(self.advanced_models)
        return all_models
    
    def create_stacking_ensemble(self, X_train, y_train, cv=5):
        """Create a stacking ensemble with cross-validation"""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Base models
        base_estimators = [(name, model) for name, model in self.get_all_models().items()]
        
        # Meta-learner
        meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        
        # Stacking ensemble
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cv,  # Use cross-validation to generate meta-features
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking


class F1PredictionEnsemble:
    """Ensemble predictions for F1 prop bets"""
    
    def __init__(self, models: Optional[Dict] = None):
        self.models = models or {}
        self.ensemble = F1AdvancedEnsemble()
        self.predictions = {}
        
    def combine_predictions(self, predictions_list: List[Dict], method: str = 'weighted_average') -> Dict:
        """Combine predictions from multiple models
        
        Args:
            predictions_list: List of prediction dictionaries from different models
            method: Combination method ('average', 'weighted_average', 'voting', 'stacking')
            
        Returns:
            Combined predictions
        """
        if not predictions_list:
            return {}
            
        if method == 'average':
            return self._average_predictions(predictions_list)
        elif method == 'weighted_average':
            return self._weighted_average_predictions(predictions_list)
        elif method == 'voting':
            return self._voting_predictions(predictions_list)
        else:
            return predictions_list[0]  # Default to first model
    
    def _average_predictions(self, predictions_list: List[Dict]) -> Dict:
        """Simple average of predictions"""
        combined = {}
        
        # Get all drivers
        all_drivers = set()
        for preds in predictions_list:
            all_drivers.update(preds.keys())
        
        for driver_id in all_drivers:
            driver_preds = []
            
            # Collect predictions from all models
            for preds in predictions_list:
                if driver_id in preds and isinstance(preds[driver_id], dict):
                    driver_preds.append(preds[driver_id])
            
            if not driver_preds:
                continue
                
            # Average each prop type
            combined[driver_id] = {
                'driver_id': driver_id,
                'driver_name': driver_preds[0].get('driver_name', f'Driver {driver_id}'),
                'predictions': {}
            }
            
            # Get all prop types
            prop_types = set()
            for pred in driver_preds:
                if 'predictions' in pred:
                    prop_types.update(pred['predictions'].keys())
            
            for prop in prop_types:
                prop_values = []
                
                for pred in driver_preds:
                    if 'predictions' in pred and prop in pred['predictions']:
                        prop_data = pred['predictions'][prop]
                        
                        # Average probabilities
                        if 'over_prob' in prop_data:
                            prop_values.append({
                                'over_prob': prop_data['over_prob'],
                                'under_prob': prop_data['under_prob'],
                                'predicted': prop_data.get('predicted', 0),
                                'line': prop_data.get('line', 0)
                            })
                
                if prop_values:
                    avg_over = np.mean([p['over_prob'] for p in prop_values])
                    avg_under = np.mean([p['under_prob'] for p in prop_values])
                    avg_predicted = np.mean([p['predicted'] for p in prop_values])
                    
                    combined[driver_id]['predictions'][prop] = {
                        'over_prob': round(avg_over, 3),
                        'under_prob': round(avg_under, 3),
                        'predicted': round(avg_predicted, 2),
                        'line': prop_values[0]['line'],
                        'recommendation': 'OVER' if avg_over > 0.55 else ('UNDER' if avg_under > 0.55 else 'PASS'),
                        'ensemble_method': 'average'
                    }
        
        return combined
    
    def _weighted_average_predictions(self, predictions_list: List[Dict], 
                                    weights: Optional[List[float]] = None) -> Dict:
        """Weighted average based on model performance"""
        if weights is None:
            # Default weights favoring more sophisticated models
            weights = [0.2, 0.3, 0.3, 0.2]  # RF, GB, XGB, LGB
            weights = weights[:len(predictions_list)]
            weights = [w / sum(weights) for w in weights]  # Normalize
        
        combined = {}
        
        # Similar to average but with weights
        all_drivers = set()
        for preds in predictions_list:
            all_drivers.update(preds.keys())
        
        for driver_id in all_drivers:
            driver_preds = []
            driver_weights = []
            
            for i, preds in enumerate(predictions_list):
                if driver_id in preds and isinstance(preds[driver_id], dict):
                    driver_preds.append(preds[driver_id])
                    driver_weights.append(weights[i])
            
            if not driver_preds:
                continue
            
            # Normalize weights for this driver
            driver_weights = [w / sum(driver_weights) for w in driver_weights]
            
            combined[driver_id] = {
                'driver_id': driver_id,
                'driver_name': driver_preds[0].get('driver_name', f'Driver {driver_id}'),
                'predictions': {}
            }
            
            # Process each prop type
            prop_types = set()
            for pred in driver_preds:
                if 'predictions' in pred:
                    prop_types.update(pred['predictions'].keys())
            
            for prop in prop_types:
                prop_values = []
                prop_weights = []
                
                for i, pred in enumerate(driver_preds):
                    if 'predictions' in pred and prop in pred['predictions']:
                        prop_data = pred['predictions'][prop]
                        if 'over_prob' in prop_data:
                            prop_values.append(prop_data)
                            prop_weights.append(driver_weights[i])
                
                if prop_values:
                    # Normalize prop weights
                    prop_weights = [w / sum(prop_weights) for w in prop_weights]
                    
                    weighted_over = sum(p['over_prob'] * w for p, w in zip(prop_values, prop_weights))
                    weighted_under = sum(p['under_prob'] * w for p, w in zip(prop_values, prop_weights))
                    weighted_predicted = sum(p.get('predicted', 0) * w for p, w in zip(prop_values, prop_weights))
                    
                    combined[driver_id]['predictions'][prop] = {
                        'over_prob': round(weighted_over, 3),
                        'under_prob': round(weighted_under, 3),
                        'predicted': round(weighted_predicted, 2),
                        'line': prop_values[0].get('line', 0),
                        'recommendation': 'OVER' if weighted_over > 0.55 else ('UNDER' if weighted_under > 0.55 else 'PASS'),
                        'ensemble_method': 'weighted_average',
                        'confidence': max(weighted_over, weighted_under)
                    }
        
        return combined
    
    def _voting_predictions(self, predictions_list: List[Dict]) -> Dict:
        """Majority voting for recommendations"""
        combined = {}
        
        all_drivers = set()
        for preds in predictions_list:
            all_drivers.update(preds.keys())
        
        for driver_id in all_drivers:
            driver_preds = []
            
            for preds in predictions_list:
                if driver_id in preds and isinstance(preds[driver_id], dict):
                    driver_preds.append(preds[driver_id])
            
            if not driver_preds:
                continue
            
            combined[driver_id] = {
                'driver_id': driver_id,
                'driver_name': driver_preds[0].get('driver_name', f'Driver {driver_id}'),
                'predictions': {}
            }
            
            # Get all prop types
            prop_types = set()
            for pred in driver_preds:
                if 'predictions' in pred:
                    prop_types.update(pred['predictions'].keys())
            
            for prop in prop_types:
                votes = {'OVER': 0, 'UNDER': 0, 'PASS': 0}
                probs = []
                
                for pred in driver_preds:
                    if 'predictions' in pred and prop in pred['predictions']:
                        prop_data = pred['predictions'][prop]
                        rec = prop_data.get('recommendation', 'PASS')
                        votes[rec] += 1
                        
                        if 'over_prob' in prop_data:
                            probs.append({
                                'over': prop_data['over_prob'],
                                'under': prop_data['under_prob']
                            })
                
                if sum(votes.values()) > 0:
                    # Determine winner
                    winner = max(votes, key=votes.get)
                    
                    # Average probabilities for confidence
                    if probs:
                        avg_over = np.mean([p['over'] for p in probs])
                        avg_under = np.mean([p['under'] for p in probs])
                    else:
                        avg_over = avg_under = 0.5
                    
                    combined[driver_id]['predictions'][prop] = {
                        'over_prob': round(avg_over, 3),
                        'under_prob': round(avg_under, 3),
                        'recommendation': winner,
                        'votes': votes,
                        'ensemble_method': 'voting',
                        'confidence': votes[winner] / sum(votes.values())
                    }
        
        return combined


class F1OptimalBetting:
    """Optimal betting strategy using Kelly Criterion"""
    
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.kelly = KellyCriterion(kelly_fraction)
        self.optimizer = PrizePicksOptimizer(self.bankroll)
        
    def optimize_bets(self, predictions: Dict, max_exposure: float = 0.25) -> Dict:
        """Optimize bet sizing across all predictions
        
        Args:
            predictions: Dictionary of predictions
            max_exposure: Maximum fraction of bankroll to risk
            
        Returns:
            Optimized betting portfolio
        """
        # Convert predictions to betting opportunities
        betting_opps = []
        
        for driver_id, driver_data in predictions.items():
            if not isinstance(driver_data, dict) or 'predictions' not in driver_data:
                continue
                
            driver_name = driver_data.get('driver_name', f'Driver {driver_id}')
            
            for prop, pred in driver_data['predictions'].items():
                if pred.get('recommendation', 'PASS') != 'PASS':
                    # Get probability and direction
                    if pred['recommendation'] == 'OVER':
                        prob = pred.get('over_prob', 0.5)
                    else:
                        prob = pred.get('under_prob', 0.5)
                    
                    # Only bet if we have edge
                    if prob > 0.55:
                        betting_opps.append({
                            'driver': driver_name,
                            'prop': prop,
                            'direction': pred['recommendation'],
                            'probability': prob,
                            'line': pred.get('line', 0),
                            'confidence': pred.get('confidence', prob)
                        })
        
        # Sort by edge (probability)
        betting_opps.sort(key=lambda x: x['probability'], reverse=True)
        
        # Optimize portfolio
        portfolio = self._optimize_portfolio(betting_opps, max_exposure)
        
        return portfolio
    
    def _optimize_portfolio(self, opportunities: List[Dict], max_exposure: float) -> Dict:
        """Optimize betting portfolio with constraints"""
        portfolio = {
            'bets': [],
            'total_stake': 0,
            'expected_value': 0,
            'risk_metrics': {}
        }
        
        if not opportunities:
            return portfolio
        
        # Group by parlay size
        for n_picks in [2, 3, 4, 5, 6]:
            if len(opportunities) >= n_picks:
                # Get best opportunities for this parlay size
                best_picks = opportunities[:n_picks]
                
                # Calculate combined probability (assuming independence)
                combined_prob = np.prod([p['probability'] for p in best_picks])
                
                # Get payout odds
                from f1_ml.optimization import PrizePicksBetTypes
                payout = PrizePicksBetTypes.PAYOUTS.get(n_picks, 0)
                
                # Calculate Kelly stake
                kelly_stake = self.kelly.calculate_kelly_stake(combined_prob, payout)
                
                # Apply bankroll constraint
                stake = min(kelly_stake * self.bankroll, max_exposure * self.bankroll / n_picks)
                
                if stake > 0:
                    expected_return = stake * (combined_prob * payout - 1)
                    
                    portfolio['bets'].append({
                        'type': f'{n_picks}-pick',
                        'selections': best_picks,
                        'stake': round(stake, 2),
                        'probability': round(combined_prob, 4),
                        'payout': payout,
                        'expected_value': round(expected_return, 2)
                    })
                    
                    portfolio['total_stake'] += stake
                    portfolio['expected_value'] += expected_return
        
        # Calculate risk metrics
        portfolio['risk_metrics'] = {
            'total_exposure': round(portfolio['total_stake'], 2),
            'exposure_pct': round(portfolio['total_stake'] / self.bankroll * 100, 1),
            'expected_roi': round(portfolio['expected_value'] / portfolio['total_stake'] * 100, 1) if portfolio['total_stake'] > 0 else 0,
            'num_bets': len(portfolio['bets'])
        }
        
        return portfolio