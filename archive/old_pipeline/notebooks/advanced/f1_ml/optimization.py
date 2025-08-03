"""
F1 Prize Picks Optimization Module

This module contains the PrizePicksOptimizer class and supporting classes
for optimizing betting portfolios using Kelly Criterion and correlation management.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PrizePicksBetTypes:
    """
    Define Prize Picks bet types and payout structures
    """
    # Standard Prize Picks multipliers
    PAYOUTS = {
        2: 3.0,    # 2-pick entry pays 3x
        3: 6.0,    # 3-pick entry pays 6x
        4: 10.0,   # 4-pick entry pays 10x
        5: 20.0,   # 5-pick entry pays 20x
        6: 25.0    # 6-pick entry pays 25x
    }
    
    # Bet types available
    BET_TYPES = {
        'top_10': 'Will finish in top 10',
        'top_5': 'Will finish in top 5',
        'top_3': 'Will finish in top 3 (podium)',
        'points': 'Will score points',
        'h2h': 'Head-to-head matchup',
        'beat_teammate': 'Will beat teammate',
        'fastest_lap': 'Will set fastest lap',
        'grid_gain': 'Positions gained from start',
        'dnf': 'Will not finish (DNF)'
    }
    
    @staticmethod
    def calculate_payout(n_picks, stake=1.0):
        """Calculate potential payout for n picks"""
        if n_picks not in PrizePicksBetTypes.PAYOUTS:
            return 0
        return stake * PrizePicksBetTypes.PAYOUTS[n_picks]
    
    @staticmethod
    def required_win_rate(n_picks):
        """Calculate required win rate to break even"""
        if n_picks not in PrizePicksBetTypes.PAYOUTS:
            return 1.0
        return 1.0 / PrizePicksBetTypes.PAYOUTS[n_picks]


class KellyCriterion:
    """
    Kelly Criterion for optimal bet sizing
    """
    def __init__(self, kelly_fraction=0.25):
        """
        Initialize with fractional Kelly (more conservative)
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        """
        self.kelly_fraction = kelly_fraction
    
    def calculate_kelly_stake(self, probability, odds):
        """
        Calculate optimal stake using Kelly Criterion
        
        Args:
            probability: True probability of winning
            odds: Decimal odds (payout multiplier)
        
        Returns:
            Optimal fraction of bankroll to bet
        """
        if probability <= 0 or probability >= 1:
            return 0
        
        # Kelly formula: f = (p * o - 1) / (o - 1)
        # where p = probability, o = decimal odds
        q = 1 - probability  # Probability of losing
        
        kelly_full = (probability * odds - 1) / (odds - 1)
        
        # Apply fractional Kelly
        kelly_stake = kelly_full * self.kelly_fraction
        
        # Ensure non-negative and reasonable bounds
        return max(0, min(kelly_stake, 0.25))  # Max 25% of bankroll
    
    def calculate_multi_kelly(self, probabilities, n_picks):
        """
        Calculate Kelly stake for multi-pick parlays
        """
        # Combined probability (all must win)
        combined_prob = np.prod(probabilities)
        
        # Get Prize Picks payout
        payout = PrizePicksBetTypes.PAYOUTS.get(n_picks, 0)
        
        if payout == 0:
            return 0
        
        return self.calculate_kelly_stake(combined_prob, payout)
    
    def expected_value(self, probability, odds, stake=1.0):
        """
        Calculate expected value of a bet
        """
        win_amount = stake * (odds - 1)
        ev = probability * win_amount - (1 - probability) * stake
        return ev
    
    def calculate_growth_rate(self, probability, odds, kelly_stake):
        """
        Calculate expected growth rate using this stake size
        """
        if kelly_stake <= 0:
            return 0
        
        p = probability
        q = 1 - probability
        b = odds - 1
        f = kelly_stake
        
        growth_rate = p * np.log(1 + b * f) + q * np.log(1 - f)
        return growth_rate


class CorrelationManager:
    """
    Manage correlations between different bet types
    """
    def __init__(self):
        # Define correlation matrix between bet types
        self.correlation_matrix = self._build_correlation_matrix()
    
    def _build_correlation_matrix(self):
        """
        Build correlation matrix between different bet types
        """
        bet_types = ['top_10', 'top_5', 'top_3', 'points', 'h2h', 
                    'beat_teammate', 'grid_gain', 'dnf']
        
        n = len(bet_types)
        corr_matrix = np.eye(n)
        
        # Define correlations (based on domain knowledge)
        correlations = {
            ('top_10', 'top_5'): 0.8,
            ('top_10', 'top_3'): 0.6,
            ('top_10', 'points'): 0.95,
            ('top_5', 'top_3'): 0.8,
            ('top_5', 'points'): 0.85,
            ('top_3', 'points'): 0.7,
            ('top_10', 'dnf'): -0.9,
            ('points', 'dnf'): -0.95,
            ('h2h', 'beat_teammate'): 0.3,
            ('grid_gain', 'top_10'): 0.2,
            ('grid_gain', 'dnf'): -0.3
        }
        
        # Build symmetric matrix
        for (type1, type2), corr in correlations.items():
            idx1 = bet_types.index(type1)
            idx2 = bet_types.index(type2)
            corr_matrix[idx1, idx2] = corr
            corr_matrix[idx2, idx1] = corr
        
        return pd.DataFrame(corr_matrix, index=bet_types, columns=bet_types)
    
    def calculate_parlay_correlation(self, bet_types, drivers=None):
        """
        Calculate overall correlation for a parlay
        """
        if len(bet_types) < 2:
            return 0
        
        total_correlation = 0
        count = 0
        
        # Pairwise correlations
        for i in range(len(bet_types)):
            for j in range(i + 1, len(bet_types)):
                bet1, bet2 = bet_types[i], bet_types[j]
                
                # Base correlation from bet types
                if bet1 in self.correlation_matrix.index and bet2 in self.correlation_matrix.index:
                    base_corr = self.correlation_matrix.loc[bet1, bet2]
                else:
                    base_corr = 0
                
                # Additional correlation if same driver
                if drivers and drivers[i] == drivers[j]:
                    base_corr = min(base_corr + 0.2, 0.95)
                
                total_correlation += abs(base_corr)
                count += 1
        
        return total_correlation / count if count > 0 else 0
    
    def adjust_probability_for_correlation(self, probabilities, correlation):
        """
        Adjust combined probability based on correlation
        """
        # Independent probability
        independent_prob = np.prod(probabilities)
        
        # Adjust for correlation (simplified model)
        # High correlation means events are more dependent
        if correlation > 0:
            # Positive correlation: if one wins, others more likely
            min_prob = min(probabilities)
            adjustment = (min_prob - independent_prob) * correlation
            adjusted_prob = independent_prob + adjustment
        else:
            # Negative correlation: if one wins, others less likely
            adjusted_prob = independent_prob * (1 + correlation * 0.5)
        
        return np.clip(adjusted_prob, 0, 1)


class PrizePicksOptimizer:
    """
    Main optimization engine for Prize Picks selections
    """
    def __init__(self, kelly_fraction=0.25, max_correlation=0.5):
        self.kelly = KellyCriterion(kelly_fraction)
        self.corr_manager = CorrelationManager()
        self.max_correlation = max_correlation
    
    def generate_all_picks(self, predictions, min_edge=0.05):
        """
        Generate all possible picks with positive edge
        """
        picks = []
        
        for _, pred in predictions.iterrows():
            driver = pred['driver']
            
            # Check each bet type
            bet_opportunities = [
                ('top_10', pred.get('top10_prob', 0.5), 0.5),  # Implied prob
                ('top_5', pred.get('top5_prob', 0.3), 0.3),
                ('top_3', pred.get('top3_prob', 0.15), 0.15),
                ('points', pred.get('points_prob', 0.5), 0.5),
                ('beat_teammate', pred.get('beat_teammate_prob', 0.5), 0.5)
            ]
            
            for bet_type, true_prob, implied_prob in bet_opportunities:
                edge = true_prob - implied_prob
                
                if edge >= min_edge:
                    picks.append({
                        'driver': driver,
                        'bet_type': bet_type,
                        'true_prob': true_prob,
                        'implied_prob': implied_prob,
                        'edge': edge,
                        'confidence': pred.get('confidence', 0.7)
                    })
        
        return pd.DataFrame(picks)
    
    def optimize_parlay(self, available_picks, n_picks, constraints=None):
        """
        Optimize selection of n picks for a parlay
        """
        if len(available_picks) < n_picks:
            return None
        
        best_parlay = None
        best_ev = -float('inf')
        
        # Consider all combinations
        for combo in combinations(range(len(available_picks)), n_picks):
            parlay_picks = available_picks.iloc[list(combo)]
            
            # Check constraints
            if not self._check_constraints(parlay_picks, constraints):
                continue
            
            # Calculate correlation
            bet_types = parlay_picks['bet_type'].tolist()
            drivers = parlay_picks['driver'].tolist()
            correlation = self.corr_manager.calculate_parlay_correlation(bet_types, drivers)
            
            # Skip if correlation too high
            if correlation > self.max_correlation:
                continue
            
            # Calculate adjusted probability
            probs = parlay_picks['true_prob'].values
            adjusted_prob = self.corr_manager.adjust_probability_for_correlation(probs, correlation)
            
            # Calculate EV
            payout = PrizePicksBetTypes.PAYOUTS[n_picks]
            ev = self.kelly.expected_value(adjusted_prob, payout)
            
            if ev > best_ev:
                best_ev = ev
                best_parlay = {
                    'picks': parlay_picks,
                    'n_picks': n_picks,
                    'correlation': correlation,
                    'adjusted_prob': adjusted_prob,
                    'expected_value': ev,
                    'kelly_stake': self.kelly.calculate_kelly_stake(adjusted_prob, payout),
                    'payout': payout
                }
        
        return best_parlay
    
    def _check_constraints(self, picks, constraints):
        """
        Check if picks meet constraints
        """
        if not constraints:
            return True
        
        # Max picks per driver
        if 'max_per_driver' in constraints:
            driver_counts = picks['driver'].value_counts()
            if any(count > constraints['max_per_driver'] for count in driver_counts):
                return False
        
        # Max picks per bet type
        if 'max_per_type' in constraints:
            type_counts = picks['bet_type'].value_counts()
            if any(count > constraints['max_per_type'] for count in type_counts):
                return False
        
        # Minimum average edge
        if 'min_avg_edge' in constraints:
            if picks['edge'].mean() < constraints['min_avg_edge']:
                return False
        
        return True
    
    def optimize_portfolio(self, available_picks, bankroll=100, constraints=None):
        """
        Optimize entire betting portfolio across different parlay sizes
        """
        portfolio = []
        
        for n_picks in range(2, 7):  # 2-6 pick parlays
            best_parlay = self.optimize_parlay(available_picks, n_picks, constraints)
            
            if best_parlay and best_parlay['expected_value'] > 0:
                portfolio.append(best_parlay)
        
        # Allocate bankroll
        if portfolio:
            # Normalize Kelly stakes
            total_kelly = sum(p['kelly_stake'] for p in portfolio)
            
            for parlay in portfolio:
                if total_kelly > 0:
                    parlay['allocation'] = (parlay['kelly_stake'] / total_kelly) * 0.5  # Use 50% of bankroll max
                    parlay['bet_size'] = parlay['allocation'] * bankroll
                else:
                    parlay['allocation'] = 0
                    parlay['bet_size'] = 0
        
        return portfolio


# Export key components
__all__ = [
    'PrizePicksBetTypes',
    'KellyCriterion',
    'CorrelationManager',
    'PrizePicksOptimizer'
]