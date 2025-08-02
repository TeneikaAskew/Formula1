#!/usr/bin/env python3
"""F1 Correlation Analysis Module - Phase 4.2 Implementation

This module analyzes correlations between different prop bets to avoid
over-exposure to correlated outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict


class F1CorrelationAnalyzer:
    """Analyze correlations between F1 prop bets"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.data = data_dict
        self.correlations = {}
        self._calculate_base_correlations()
        
    def _calculate_base_correlations(self):
        """Calculate base correlations from historical data"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return
            
        # Group by race to analyze correlations
        race_groups = results.groupby('raceId')
        
        # Initialize correlation matrices
        self.correlations = {
            'driver_props': defaultdict(dict),
            'team_correlations': defaultdict(dict),
            'prop_correlations': defaultdict(dict)
        }
        
        # Calculate driver-level correlations
        self._calculate_driver_correlations(results)
        
        # Calculate team-level correlations
        self._calculate_team_correlations(results)
        
        # Calculate prop-type correlations
        self._calculate_prop_correlations(results)
    
    def _calculate_driver_correlations(self, results: pd.DataFrame):
        """Calculate correlations between drivers' performances"""
        # Get drivers with sufficient data
        driver_counts = results['driverId'].value_counts()
        active_drivers = driver_counts[driver_counts >= 20].index
        
        # Calculate position correlations between drivers
        for driver1 in active_drivers[:20]:  # Limit to top 20 for efficiency
            driver1_results = results[results['driverId'] == driver1][['raceId', 'positionNumber']]
            
            for driver2 in active_drivers[:20]:
                if driver1 >= driver2:  # Avoid duplicates
                    continue
                    
                driver2_results = results[results['driverId'] == driver2][['raceId', 'positionNumber']]
                
                # Merge on common races
                merged = driver1_results.merge(
                    driver2_results, 
                    on='raceId', 
                    suffixes=('_1', '_2')
                )
                
                if len(merged) >= 10:
                    # Calculate correlation
                    positions1 = merged['positionNumber_1'].fillna(20)
                    positions2 = merged['positionNumber_2'].fillna(20)
                    
                    corr, _ = spearmanr(positions1, positions2)
                    
                    self.correlations['driver_props'][driver1][driver2] = corr
                    self.correlations['driver_props'][driver2][driver1] = corr
    
    def _calculate_team_correlations(self, results: pd.DataFrame):
        """Calculate correlations between teammates"""
        # Group by constructor and race
        constructor_groups = results.groupby(['raceId', 'constructorId'])
        
        teammate_correlations = []
        
        for (race_id, constructor_id), group in constructor_groups:
            if len(group) == 2:  # Both teammates finished
                positions = group['positionNumber'].values
                if not pd.isna(positions).any():
                    # Perfect negative correlation means one always beats the other
                    teammate_correlations.append({
                        'constructorId': constructor_id,
                        'position_diff': abs(positions[0] - positions[1])
                    })
        
        if teammate_correlations:
            teammate_df = pd.DataFrame(teammate_correlations)
            avg_correlations = teammate_df.groupby('constructorId')['position_diff'].mean()
            
            for constructor_id, avg_diff in avg_correlations.items():
                # Convert position difference to correlation metric
                # Larger difference = lower correlation
                correlation = 1 - (avg_diff / 20)  # Normalize by max position
                self.correlations['team_correlations'][constructor_id] = correlation
    
    def _calculate_prop_correlations(self, results: pd.DataFrame):
        """Calculate correlations between different prop types"""
        # Create prop outcomes for correlation analysis
        prop_data = pd.DataFrame({
            'raceId': results['raceId'],
            'driverId': results['driverId'],
            'points_scored': (results['points'] > 0).astype(int),
            'top_10': (results['positionNumber'] <= 10).astype(int),
            'dnf': results['positionText'].isin(['DNF', 'DNS', 'DSQ']).astype(int),
            'podium': (results['positionNumber'] <= 3).astype(int)
        })
        
        # Calculate correlations between prop types
        prop_types = ['points_scored', 'top_10', 'dnf', 'podium']
        
        for prop1 in prop_types:
            for prop2 in prop_types:
                if prop1 != prop2:
                    corr = prop_data[prop1].corr(prop_data[prop2])
                    self.correlations['prop_correlations'][prop1][prop2] = corr
    
    def get_bet_correlation(self, bet1: Dict, bet2: Dict) -> float:
        """Get correlation between two specific bets
        
        Args:
            bet1: First bet details (driver, prop_type, direction)
            bet2: Second bet details (driver, prop_type, direction)
            
        Returns:
            Correlation coefficient between -1 and 1
        """
        # Same driver, different props
        if bet1['driver'] == bet2['driver']:
            prop_corr = self.correlations['prop_correlations'].get(
                bet1['prop_type'], {}
            ).get(bet2['prop_type'], 0.5)
            
            # Adjust for direction
            if bet1['direction'] != bet2['direction']:
                prop_corr *= -0.5
                
            return prop_corr
            
        # Teammates
        if bet1.get('constructor') == bet2.get('constructor'):
            team_corr = self.correlations['team_correlations'].get(
                bet1['constructor'], 0.3
            )
            
            # Head-to-head props are negatively correlated
            if bet1['prop_type'] == 'beat_teammate':
                return -0.8
                
            return team_corr
            
        # Different drivers
        driver_corr = self.correlations['driver_props'].get(
            bet1['driver'], {}
        ).get(bet2['driver'], 0)
        
        return driver_corr
    
    def calculate_parlay_correlation(self, bets: List[Dict]) -> float:
        """Calculate overall correlation for a parlay
        
        Args:
            bets: List of bet dictionaries
            
        Returns:
            Average pairwise correlation
        """
        if len(bets) < 2:
            return 0
            
        correlations = []
        
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                corr = self.get_bet_correlation(bets[i], bets[j])
                correlations.append(abs(corr))  # Use absolute correlation
                
        return np.mean(correlations) if correlations else 0
    
    def get_correlation_matrix(self, drivers: List[int]) -> pd.DataFrame:
        """Get correlation matrix for specific drivers
        
        Args:
            drivers: List of driver IDs
            
        Returns:
            Correlation matrix as DataFrame
        """
        n = len(drivers)
        matrix = np.zeros((n, n))
        
        for i, driver1 in enumerate(drivers):
            for j, driver2 in enumerate(drivers):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    corr = self.correlations['driver_props'].get(
                        driver1, {}
                    ).get(driver2, 0)
                    matrix[i, j] = corr
                    
        return pd.DataFrame(matrix, index=drivers, columns=drivers)
    
    def plot_correlation_heatmap(self, drivers: List[int], 
                                driver_names: Optional[Dict[int, str]] = None,
                                save_path: Optional[str] = None):
        """Plot correlation heatmap for drivers
        
        Args:
            drivers: List of driver IDs
            driver_names: Optional mapping of ID to name
            save_path: Optional path to save figure
        """
        corr_matrix = self.get_correlation_matrix(drivers)
        
        # Use names if provided
        if driver_names:
            labels = [driver_names.get(d, f"Driver {d}") for d in drivers]
            corr_matrix.index = labels
            corr_matrix.columns = labels
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation'}
        )
        
        plt.title('Driver Performance Correlations', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_low_correlation_pairs(self, bets: List[Dict], 
                                 max_correlation: float = 0.3) -> List[Tuple[Dict, Dict]]:
        """Find bet pairs with low correlation for diversification
        
        Args:
            bets: List of available bets
            max_correlation: Maximum acceptable correlation
            
        Returns:
            List of low-correlation bet pairs
        """
        low_corr_pairs = []
        
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                corr = abs(self.get_bet_correlation(bets[i], bets[j]))
                
                if corr <= max_correlation:
                    low_corr_pairs.append((bets[i], bets[j], corr))
        
        # Sort by correlation (lowest first)
        low_corr_pairs.sort(key=lambda x: x[2])
        
        return [(pair[0], pair[1]) for pair in low_corr_pairs]
    
    def diversification_score(self, portfolio: List[Dict]) -> float:
        """Calculate diversification score for a portfolio
        
        Args:
            portfolio: List of bets in portfolio
            
        Returns:
            Diversification score (0-1, higher is better)
        """
        if len(portfolio) < 2:
            return 1.0
            
        avg_correlation = self.calculate_parlay_correlation(portfolio)
        
        # Convert correlation to diversification score
        # Low correlation = high diversification
        return 1 - avg_correlation
    
    def recommend_diversified_portfolio(self, available_bets: List[Dict], 
                                      portfolio_size: int = 6) -> List[Dict]:
        """Recommend a diversified portfolio of bets
        
        Args:
            available_bets: List of all available bets
            portfolio_size: Target portfolio size
            
        Returns:
            Recommended diversified portfolio
        """
        if len(available_bets) <= portfolio_size:
            return available_bets
            
        # Start with highest EV bet
        portfolio = [available_bets[0]]
        remaining = available_bets[1:]
        
        # Greedily add bets with lowest correlation to existing portfolio
        while len(portfolio) < portfolio_size and remaining:
            best_bet = None
            best_score = -1
            
            for bet in remaining:
                # Calculate average correlation with existing portfolio
                correlations = [
                    abs(self.get_bet_correlation(bet, p_bet)) 
                    for p_bet in portfolio
                ]
                avg_corr = np.mean(correlations)
                
                # Score combines low correlation with bet quality
                score = (1 - avg_corr) * bet.get('probability', 0.5)
                
                if score > best_score:
                    best_score = score
                    best_bet = bet
            
            if best_bet:
                portfolio.append(best_bet)
                remaining.remove(best_bet)
            else:
                break
                
        return portfolio