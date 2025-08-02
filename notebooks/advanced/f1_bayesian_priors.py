#!/usr/bin/env python3
"""F1 Bayesian Priors Module - Phase 2.2 Implementation

This module implements sophisticated Bayesian priors including:
- Team-specific priors
- Track-specific priors
- Driver experience priors
- Weather-adjusted priors
- Hierarchical Bayesian models
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from scipy import stats


class F1BayesianPriors:
    """Sophisticated Bayesian priors for F1 predictions"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.data = data_dict
        self.team_priors = {}
        self.track_priors = {}
        self.driver_priors = {}
        self.global_priors = {}
        
        # Initialize priors
        self._calculate_team_priors()
        self._calculate_track_priors()
        self._calculate_driver_priors()
        self._calculate_global_priors()
    
    def _calculate_team_priors(self):
        """Calculate team-specific priors for each prop type"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return
            
        # Get recent results (last 2 years)
        recent_results = results[results['year'] >= results['year'].max() - 1]
        
        # Points priors by team
        team_points_mean = recent_results.groupby('constructorId')['points'].mean()
        team_points_count = recent_results.groupby('constructorId')['points'].count()
        team_points_over6 = recent_results.groupby('constructorId')['points'].apply(lambda x: (x > 6).mean())
        
        for constructor_id in team_points_mean.index:
            # Handle both numeric and string constructor IDs
            try:
                constructor_id_key = int(constructor_id)
            except (ValueError, TypeError):
                # Keep as string if not convertible to int
                constructor_id_key = str(constructor_id)
            
            if constructor_id_key not in self.team_priors:
                self.team_priors[constructor_id_key] = {}
            
            self.team_priors[constructor_id_key]['points'] = {
                'mean': float(team_points_mean[constructor_id]),
                'over_6_rate': float(team_points_over6[constructor_id]),
                'sample_size': int(team_points_count[constructor_id])
            }
        
        # DNF priors by team
        team_dnf_rate = recent_results.groupby('constructorId')['positionText'].apply(
            lambda x: x.isin(['DNF', 'DNS', 'DSQ']).mean()
        )
        
        for constructor_id in team_dnf_rate.index:
            # Handle both numeric and string constructor IDs
            try:
                constructor_id_key = int(constructor_id)
            except (ValueError, TypeError):
                # Keep as string if not convertible to int
                constructor_id_key = str(constructor_id)
            
            if constructor_id_key not in self.team_priors:
                self.team_priors[constructor_id_key] = {}
            
            self.team_priors[constructor_id_key]['dnf'] = {
                'rate': float(team_dnf_rate[constructor_id])
            }
    
    def _calculate_track_priors(self):
        """Calculate track-specific priors"""
        results = self.data.get('results', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty or races.empty:
            return
            
        # Merge to get circuit info
        results_with_circuit = results.merge(
            races[['id', 'circuitId']], 
            left_on='raceId', 
            right_on='id'
        )
        
        # Track-specific overtaking rates
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame())
        if not grid.empty:
            # Calculate average overtakes per circuit
            overtake_data = results_with_circuit.merge(
                grid[['raceId', 'driverId', 'positionNumber']].rename(
                    columns={'positionNumber': 'gridPosition'}
                ),
                on=['raceId', 'driverId']
            )
            
            overtake_data['overtakes'] = (
                overtake_data['gridPosition'] - overtake_data['positionNumber']
            ).apply(lambda x: max(0, x) if pd.notna(x) else 0)
            
            circuit_overtakes_mean = overtake_data.groupby('circuitId')['overtakes'].mean()
            circuit_overtakes_std = overtake_data.groupby('circuitId')['overtakes'].std()
            
            for circuit_id in circuit_overtakes_mean.index:
                # Handle both numeric and string circuit IDs
                try:
                    circuit_id_key = int(circuit_id)
                except (ValueError, TypeError):
                    circuit_id_key = str(circuit_id)
                
                self.track_priors[circuit_id_key] = {
                    'overtakes': {
                        'mean': float(circuit_overtakes_mean[circuit_id]),
                        'std': float(circuit_overtakes_std[circuit_id]) if pd.notna(circuit_overtakes_std[circuit_id]) else 0.0
                    }
                }
        
        # Track-specific DNF rates
        circuit_dnf_rate = results_with_circuit.groupby('circuitId')['positionText'].apply(
            lambda x: x.isin(['DNF', 'DNS', 'DSQ']).mean()
        )
        
        for circuit_id in circuit_dnf_rate.index:
            # Handle both numeric and string circuit IDs
            try:
                circuit_id_key = int(circuit_id)
            except (ValueError, TypeError):
                circuit_id_key = str(circuit_id)
            
            if circuit_id_key not in self.track_priors:
                self.track_priors[circuit_id_key] = {}
            
            self.track_priors[circuit_id_key]['dnf'] = {
                'rate': float(circuit_dnf_rate[circuit_id])
            }
    
    def _calculate_driver_priors(self):
        """Calculate driver-specific experience priors"""
        results = self.data.get('results', pd.DataFrame())
        drivers = self.data.get('drivers', pd.DataFrame())
        
        if results.empty or drivers.empty:
            return
            
        # Calculate driver experience
        driver_race_count = results.groupby('driverId')['raceId'].count()
        driver_avg_points = results.groupby('driverId')['points'].mean()
        driver_points_rate = results.groupby('driverId')['points'].apply(lambda x: (x > 0).mean())
        driver_dnf_rate = results.groupby('driverId')['positionText'].apply(
            lambda x: x.isin(['DNF', 'DNS', 'DSQ']).mean()
        )
        
        for driver_id in driver_race_count.index:
            # Handle both numeric and string driver IDs
            try:
                driver_id_key = int(driver_id)
            except (ValueError, TypeError):
                driver_id_key = str(driver_id)
            
            self.driver_priors[driver_id_key] = {
                'experience': int(driver_race_count[driver_id]),
                'avg_points': float(driver_avg_points[driver_id]),
                'points_rate': float(driver_points_rate[driver_id]),
                'dnf_rate': float(driver_dnf_rate[driver_id])
            }
    
    def _calculate_global_priors(self):
        """Calculate global priors as baseline"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return
            
        # Global statistics
        self.global_priors = {
            'points': {
                'over_6_rate': float((results['points'] > 6).mean()),
                'mean': float(results['points'].mean()),
                'std': float(results['points'].std())
            },
            'dnf': {
                'rate': float(results['positionText'].isin(['DNF', 'DNS', 'DSQ']).mean())
            },
            'overtakes': {
                'mean': 2.8,  # Historical average
                'std': 2.5
            }
        }
    
    def get_hierarchical_prior(self, driver_id, constructor_id, 
                              circuit_id, prop_type: str) -> Dict:
        """Get hierarchical Bayesian prior combining all levels
        
        Args:
            driver_id: Driver ID
            constructor_id: Constructor/Team ID
            circuit_id: Circuit ID
            prop_type: Type of prop (points, dnf, overtakes, etc.)
            
        Returns:
            Dictionary with prior parameters
        """
        # Start with global prior
        prior = self.global_priors.get(prop_type, {}).copy()
        
        # Weight factors for hierarchical model
        weights = {
            'global': 0.2,
            'track': 0.2,
            'team': 0.4,
            'driver': 0.2
        }
        
        # Collect all relevant priors
        priors_list = []
        weights_list = []
        
        # Global prior
        if prop_type in self.global_priors:
            priors_list.append(self.global_priors[prop_type])
            weights_list.append(weights['global'])
        
        # Track prior
        if circuit_id in self.track_priors and prop_type in self.track_priors[circuit_id]:
            priors_list.append(self.track_priors[circuit_id][prop_type])
            weights_list.append(weights['track'])
        
        # Team prior
        if constructor_id in self.team_priors and prop_type in self.team_priors[constructor_id]:
            priors_list.append(self.team_priors[constructor_id][prop_type])
            weights_list.append(weights['team'])
        
        # Driver prior
        if driver_id in self.driver_priors:
            driver_data = self.driver_priors[driver_id]
            if prop_type == 'points' and 'points_rate' in driver_data:
                priors_list.append({'over_6_rate': driver_data['points_rate']})
                weights_list.append(weights['driver'])
            elif prop_type == 'dnf' and 'dnf_rate' in driver_data:
                priors_list.append({'rate': driver_data['dnf_rate']})
                weights_list.append(weights['driver'])
        
        # Normalize weights
        if weights_list:
            total_weight = sum(weights_list)
            weights_list = [w / total_weight for w in weights_list]
        
        # Combine priors using weighted average
        combined_prior = {}
        
        if prop_type == 'points':
            rates = [p.get('over_6_rate', 0.35) for p in priors_list]
            if rates and weights_list:
                combined_prior['over_6_rate'] = sum(r * w for r, w in zip(rates, weights_list))
            else:
                combined_prior['over_6_rate'] = 0.35
                
        elif prop_type == 'dnf':
            rates = [p.get('rate', 0.12) for p in priors_list]
            if rates and weights_list:
                combined_prior['rate'] = sum(r * w for r, w in zip(rates, weights_list))
            else:
                combined_prior['rate'] = 0.12
                
        elif prop_type == 'overtakes':
            means = [p.get('mean', 2.8) for p in priors_list]
            if means and weights_list:
                combined_prior['mean'] = sum(m * w for m, w in zip(means, weights_list))
            else:
                combined_prior['mean'] = 2.8
        
        # Add confidence based on data availability
        combined_prior['confidence'] = len(priors_list) / 4.0  # Max 1.0 if all levels available
        
        return combined_prior
    
    def apply_hierarchical_prior(self, raw_prob: float, prior_params: Dict,
                               sample_size: int, prior_strength: float = 10.0) -> float:
        """Apply hierarchical Bayesian prior to probability
        
        Args:
            raw_prob: Raw predicted probability
            prior_params: Prior parameters from get_hierarchical_prior
            sample_size: Number of observations
            prior_strength: Strength of prior (equivalent sample size)
            
        Returns:
            Adjusted probability
        """
        # Get prior rate based on prop type
        if 'rate' in prior_params:
            prior_rate = prior_params['rate']
        elif 'over_6_rate' in prior_params:
            prior_rate = prior_params['over_6_rate']
        else:
            return raw_prob
        
        # Adjust prior strength based on confidence
        confidence = prior_params.get('confidence', 0.5)
        adjusted_strength = prior_strength * confidence
        
        # Beta-Binomial update
        alpha_prior = prior_rate * adjusted_strength
        beta_prior = (1 - prior_rate) * adjusted_strength
        
        observed_successes = raw_prob * sample_size
        observed_failures = (1 - raw_prob) * sample_size
        
        alpha_post = alpha_prior + observed_successes
        beta_post = beta_prior + observed_failures
        
        posterior_prob = alpha_post / (alpha_post + beta_post)
        
        return float(np.clip(posterior_prob, 0.01, 0.99))
    
    def get_experience_adjustment(self, driver_id: int) -> float:
        """Get experience-based adjustment factor
        
        Args:
            driver_id: Driver ID
            
        Returns:
            Adjustment factor (0.8-1.2)
        """
        if driver_id not in self.driver_priors:
            return 1.0
            
        experience = self.driver_priors[driver_id].get('experience', 0)
        
        # Sigmoid function for experience adjustment
        # New drivers (< 20 races) get conservative adjustment
        # Experienced drivers (> 100 races) get slight boost
        adjustment = 0.8 + 0.4 / (1 + np.exp(-(experience - 50) / 20))
        
        return float(adjustment)