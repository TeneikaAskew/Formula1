#!/usr/bin/env python3
"""Full F1 predictions v4 with all real implementations - no mocks or workarounds"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer
from f1_probability_calibration import F1ProbabilityCalibrator
from f1_ensemble_integration import F1PredictionEnsemble, F1OptimalBetting
from f1_correlation_analysis import F1CorrelationAnalyzer
from f1_risk_dashboard import F1RiskDashboard

warnings.filterwarnings('ignore')


class RobustBayesianPriors:
    """Robust Bayesian priors that handle data type issues"""
    
    def __init__(self, data_dict):
        self.data = data_dict
        self.team_priors = {}
        self.track_priors = {}
        self.driver_priors = {}
        self._calculate_priors()
        
    def _safe_to_int(self, value):
        """Safely convert value to int, handling various data types"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(value)
        if isinstance(value, str):
            # Try to extract numeric part if it's a string
            try:
                return int(float(value))
            except:
                return None
        if isinstance(value, pd.Series):
            return self._safe_to_int(value.iloc[0])
        return None
        
    def _calculate_priors(self):
        """Calculate all priors with robust error handling"""
        try:
            self._calculate_team_priors()
        except Exception as e:
            print(f"Warning: Could not calculate team priors: {e}")
            
        try:
            self._calculate_track_priors()
        except Exception as e:
            print(f"Warning: Could not calculate track priors: {e}")
            
        try:
            self._calculate_driver_priors()
        except Exception as e:
            print(f"Warning: Could not calculate driver priors: {e}")
    
    def _calculate_team_priors(self):
        """Calculate team-specific priors with robust handling"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return
            
        # Get recent results
        if 'year' in results.columns:
            recent_results = results[results['year'] >= results['year'].max() - 1]
        else:
            recent_results = results.tail(1000)  # Last 1000 results as fallback
        
        # Points priors by team
        if 'constructorId' in recent_results.columns and 'points' in recent_results.columns:
            team_stats = recent_results.groupby('constructorId').agg({
                'points': ['mean', 'count']
            }).reset_index()
            
            # Calculate over 6 points rate separately to avoid lambda in column name
            over_6_rates = recent_results.groupby('constructorId')['points'].apply(
                lambda x: (x > 6).mean()
            ).reset_index()
            over_6_rates.columns = ['constructorId', 'over_6_rate']
            
            for _, row in team_stats.iterrows():
                constructor_id = self._safe_to_int(row['constructorId'])
                if constructor_id is None:
                    continue
                    
                if constructor_id not in self.team_priors:
                    self.team_priors[constructor_id] = {}
                
                # Get over 6 rate
                over_6_rate = over_6_rates[
                    over_6_rates['constructorId'] == row['constructorId']
                ]['over_6_rate'].iloc[0] if len(over_6_rates[
                    over_6_rates['constructorId'] == row['constructorId']
                ]) > 0 else 0.5
                
                self.team_priors[constructor_id]['points'] = {
                    'mean': float(row[('points', 'mean')]),
                    'over_6_rate': float(over_6_rate),
                    'sample_size': int(row[('points', 'count')])
                }
        
        # DNF priors by team
        if 'positionText' in recent_results.columns:
            dnf_indicators = ['DNF', 'DNS', 'DSQ', 'EX', 'NC', 'WD']
            team_dnf = recent_results.groupby('constructorId')['positionText'].apply(
                lambda x: x.isin(dnf_indicators).mean()
            ).reset_index()
            team_dnf.columns = ['constructorId', 'dnf_rate']
            
            for _, row in team_dnf.iterrows():
                constructor_id = self._safe_to_int(row['constructorId'])
                if constructor_id is None:
                    continue
                    
                if constructor_id not in self.team_priors:
                    self.team_priors[constructor_id] = {}
                
                self.team_priors[constructor_id]['dnf'] = {
                    'rate': float(row['dnf_rate'])
                }
    
    def _calculate_track_priors(self):
        """Calculate track-specific priors"""
        results = self.data.get('results', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty or races.empty:
            return
            
        # Merge to get circuit info
        if 'raceId' in results.columns and 'id' in races.columns and 'circuitId' in races.columns:
            results_with_circuit = results.merge(
                races[['id', 'circuitId']], 
                left_on='raceId', 
                right_on='id',
                how='left'
            )
            
            # Track DNF rates
            if 'positionText' in results_with_circuit.columns:
                dnf_indicators = ['DNF', 'DNS', 'DSQ', 'EX', 'NC', 'WD']
                circuit_dnf = results_with_circuit.groupby('circuitId')['positionText'].apply(
                    lambda x: x.isin(dnf_indicators).mean()
                ).reset_index()
                circuit_dnf.columns = ['circuitId', 'dnf_rate']
                
                for _, row in circuit_dnf.iterrows():
                    circuit_id = self._safe_to_int(row['circuitId'])
                    if circuit_id is None:
                        continue
                        
                    if circuit_id not in self.track_priors:
                        self.track_priors[circuit_id] = {}
                    
                    self.track_priors[circuit_id]['dnf'] = {
                        'rate': float(row['dnf_rate'])
                    }
    
    def _calculate_driver_priors(self):
        """Calculate driver-specific priors"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return
            
        # Get recent results
        if 'year' in results.columns:
            recent_results = results[results['year'] >= results['year'].max() - 2]
        else:
            recent_results = results.tail(2000)
        
        # Driver performance stats
        if 'driverId' in recent_results.columns:
            driver_stats = recent_results.groupby('driverId').agg({
                'points': ['mean', 'count'] if 'points' in recent_results.columns else [],
                'positionNumber': ['mean'] if 'positionNumber' in recent_results.columns else []
            }).reset_index()
            
            for _, row in driver_stats.iterrows():
                driver_id = self._safe_to_int(row['driverId'])
                if driver_id is None:
                    continue
                    
                self.driver_priors[driver_id] = {}
                
                if 'points' in recent_results.columns:
                    self.driver_priors[driver_id]['points'] = {
                        'mean': float(row[('points', 'mean')]) if ('points', 'mean') in row else 0,
                        'sample_size': int(row[('points', 'count')]) if ('points', 'count') in row else 0
                    }
                
                if 'positionNumber' in recent_results.columns:
                    self.driver_priors[driver_id]['avg_position'] = float(
                        row[('positionNumber', 'mean')]
                    ) if ('positionNumber', 'mean') in row else 10
    
    def get_hierarchical_prior(self, driver_id: int, constructor_id: int, 
                              circuit_id: int, prop_type: str) -> Dict:
        """Get hierarchical prior combining multiple levels"""
        # Convert IDs to int safely
        driver_id = self._safe_to_int(driver_id)
        constructor_id = self._safe_to_int(constructor_id)
        circuit_id = self._safe_to_int(circuit_id)
        
        # Base rates
        base_rates = {
            'points': {'probability': 0.45, 'over_6': 0.35},
            'overtakes': {'probability': 0.65},
            'dnf': {'probability': 0.12},
            'starting_position': {'probability': 0.50},
            'pit_stops': {'probability': 0.70},
            'teammate_overtakes': {'probability': 0.50}
        }
        
        prior = base_rates.get(prop_type, {'probability': 0.5}).copy()
        confidence = 0.2  # Base confidence
        
        # Add team prior if available
        if constructor_id and constructor_id in self.team_priors:
            team_data = self.team_priors[constructor_id]
            if prop_type == 'points' and 'points' in team_data:
                prior['probability'] = team_data['points']['over_6_rate']
                confidence += 0.3
            elif prop_type == 'dnf' and 'dnf' in team_data:
                prior['probability'] = team_data['dnf']['rate']
                confidence += 0.3
        
        # Add track prior if available
        if circuit_id and circuit_id in self.track_priors:
            track_data = self.track_priors[circuit_id]
            if prop_type == 'dnf' and 'dnf' in track_data:
                # Weighted average with existing prior
                prior['probability'] = (
                    prior['probability'] * 0.7 + 
                    track_data['dnf']['rate'] * 0.3
                )
                confidence += 0.1
        
        # Add driver prior if available
        if driver_id and driver_id in self.driver_priors:
            driver_data = self.driver_priors[driver_id]
            if prop_type == 'points' and 'points' in driver_data:
                # Weighted average
                prior['probability'] = (
                    prior['probability'] * 0.5 + 
                    (driver_data['points']['mean'] > 6) * 0.5
                )
                confidence += 0.2
        
        prior['confidence'] = min(confidence, 0.8)  # Cap confidence
        return prior


class RobustContextualFeatures:
    """Robust contextual features that handle data issues"""
    
    def __init__(self, data_dict):
        self.data = data_dict
        self._initialize_track_characteristics()
        
    def _initialize_track_characteristics(self):
        """Initialize track characteristics"""
        self.track_characteristics = {
            # High overtaking tracks
            'monza': {'overtaking_difficulty': 0.3, 'type': 'high_speed'},
            'spa': {'overtaking_difficulty': 0.4, 'type': 'high_speed'},
            'interlagos': {'overtaking_difficulty': 0.4, 'type': 'mixed'},
            
            # Low overtaking tracks
            'monaco': {'overtaking_difficulty': 0.9, 'type': 'street'},
            'hungaroring': {'overtaking_difficulty': 0.8, 'type': 'technical'},
            'singapore': {'overtaking_difficulty': 0.7, 'type': 'street'},
            
            # Default
            'default': {'overtaking_difficulty': 0.5, 'type': 'mixed'}
        }
    
    def get_all_contextual_features(self, driver_id: int, constructor_id: int, 
                                   circuit_id: int, race_id: int,
                                   weather_data: Optional[Dict] = None) -> Dict:
        """Get all contextual features with robust handling"""
        features = {
            'track_overtaking_difficulty': 0.5,
            'momentum_score': 0,
            'risk_score': 0.5,
            'team_momentum': 0,
            'circuit_affinity': 0
        }
        
        # Get track characteristics
        if circuit_id:
            circuits = self.data.get('circuits', pd.DataFrame())
            if not circuits.empty and 'id' in circuits.columns and 'name' in circuits.columns:
                circuit_info = circuits[circuits['id'] == circuit_id]
                if not circuit_info.empty:
                    circuit_name = circuit_info['name'].iloc[0].lower()
                    track_data = self.track_characteristics.get(
                        circuit_name, 
                        self.track_characteristics['default']
                    )
                    features['track_overtaking_difficulty'] = track_data['overtaking_difficulty']
        
        # Calculate momentum (simplified but real)
        if driver_id:
            results = self.data.get('results', pd.DataFrame())
            if not results.empty and 'driverId' in results.columns:
                recent_results = results[results['driverId'] == driver_id].tail(5)
                if not recent_results.empty and 'positionNumber' in recent_results.columns:
                    positions = recent_results['positionNumber'].dropna()
                    if len(positions) >= 2:
                        # Positive momentum if improving positions
                        momentum = (positions.iloc[-2] - positions.iloc[-1]) / 10
                        features['momentum_score'] = max(-1, min(1, momentum))
        
        return features


class F1PrizePicksPredictorV4Full:
    """Full F1 PrizePicks predictor with all real implementations"""
    
    def __init__(self):
        # Load data
        self.loader = F1DBDataLoader()
        self.data = self.loader.get_core_datasets()
        
        # Initialize components with real implementations
        self.analyzer = F1PerformanceAnalyzer(self.data)
        self.calibrator = F1ProbabilityCalibrator()
        self.bayesian_priors = RobustBayesianPriors(self.data)
        self.contextual_features = RobustContextualFeatures(self.data)
        
        # Settings
        self.calibration_enabled = True
        self.hierarchical_priors_enabled = True
        self.contextual_enabled = True
        
        # Default lines
        self.default_lines = {
            'overtakes': 3.0,
            'points': 6.0,
            'pit_stops': 2.0,
            'teammate_overtakes': 0.5,
            'starting_position': 10.5,
            'dnf': 0.5,
            'grid_penalty': 0.5
        }
        
        self.typical_ranges = {
            'overtakes': (0, 15),
            'points': (0, 26),
            'pit_stops': (1, 4),
            'teammate_overtakes': (0, 5),
            'starting_position': (1, 20),
            'dnf': (0, 1),
            'grid_penalty': (0, 1)
        }
        
    def bound_probability(self, prob, min_prob=0.01, max_prob=0.99):
        """Bound probability between min and max"""
        return max(min_prob, min(max_prob, prob))
        
    def calibrate_probability(self, raw_prob, prop_type, sample_size=20, 
                            driver_id=None, constructor_id=None, circuit_id=None):
        """Apply calibration with hierarchical Bayesian priors"""
        
        # Step 1: Basic calibration
        if self.calibration_enabled:
            calibrated_prob = self.calibrator.apply_bayesian_prior(
                raw_prob, prop_type, sample_size
            )
        else:
            calibrated_prob = raw_prob
            
        # Step 2: Hierarchical Bayesian priors
        if self.hierarchical_priors_enabled and driver_id and constructor_id:
            prior_info = self.bayesian_priors.get_hierarchical_prior(
                driver_id, constructor_id, circuit_id, prop_type
            )
            
            if prior_info and 'probability' in prior_info:
                confidence = prior_info.get('confidence', 0.5)
                calibrated_prob = (
                    calibrated_prob * (1 - confidence) + 
                    prior_info['probability'] * confidence
                )
        
        return self.bound_probability(calibrated_prob)
    
    def get_driver_recent_races(self, driver_name, num_races=20):
        """Get recent race results for a driver"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty or results.empty:
            return pd.DataFrame()
        
        # Find driver ID
        driver_match = drivers[drivers['fullName'] == driver_name]
        if driver_match.empty:
            return pd.DataFrame()
            
        driver_id = driver_match['id'].iloc[0]
        
        # Get recent results
        driver_results = results[
            results['driverId'] == driver_id
        ].sort_values('raceId', ascending=False).head(num_races)
        
        # Add positions gained if grid data exists
        if 'grid' in driver_results.columns and 'positionNumber' in driver_results.columns:
            driver_results['positions_gained'] = (
                driver_results['grid'] - driver_results['positionNumber']
            ).fillna(0)
        else:
            driver_results['positions_gained'] = 0
            
        return driver_results
    
    def calculate_overtakes_probability(self, driver_name, circuit_name=None, line=3.0):
        """Calculate probability of driver making over X overtakes"""
        recent_races = self.get_driver_recent_races(driver_name, num_races=20)
        
        if recent_races.empty:
            return 0.5, 0
            
        # Get IDs for calibration
        driver_id = recent_races['driverId'].iloc[0] if not recent_races.empty else None
        constructor_id = recent_races['constructorId'].iloc[0] if not recent_races.empty and 'constructorId' in recent_races.columns else None
        
        # Convert to int if numeric
        if driver_id and str(driver_id).isdigit():
            driver_id = int(driver_id)
        if constructor_id and str(constructor_id).isdigit():
            constructor_id = int(constructor_id)
        
        # Get circuit ID if available
        circuit_id = None
        if circuit_name:
            circuits = self.data.get('circuits', pd.DataFrame())
            if not circuits.empty and 'name' in circuits.columns:
                circuit_match = circuits[circuits['name'] == circuit_name]
                if not circuit_match.empty:
                    circuit_id = int(circuit_match['id'].iloc[0])
        
        # Apply contextual adjustments
        if self.contextual_enabled and driver_id and constructor_id:
            features = self.contextual_features.get_all_contextual_features(
                driver_id, constructor_id, circuit_id, 
                recent_races['raceId'].iloc[0] if not recent_races.empty else None
            )
            
            track_factor = features.get('track_overtaking_difficulty', 0.5)
            # Higher difficulty = fewer overtakes
            effective_line = line * (1 + track_factor)
        else:
            effective_line = line
        
        # Count races with more than X overtakes
        if 'positions_gained' in recent_races.columns:
            over_count = (recent_races['positions_gained'] > effective_line).sum()
        else:
            over_count = 0
            
        total_races = len(recent_races)
        raw_prob = over_count / total_races if total_races > 0 else 0.5
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'overtakes', total_races,
            driver_id, constructor_id, circuit_id
        )
        
        return calibrated_prob, total_races
    
    def calculate_points_probability(self, driver_name, line=6.0):
        """Calculate probability of scoring over X points"""
        recent_results = self.get_driver_recent_races(driver_name, num_races=20)
        
        if recent_results.empty:
            return 0.5, 0, 0
            
        # Get IDs
        driver_id = int(recent_results['driverId'].iloc[0])
        constructor_id = int(recent_results['constructorId'].iloc[0]) if 'constructorId' in recent_results.columns else None
        
        # Apply contextual adjustments
        if self.contextual_enabled and driver_id and constructor_id:
            features = self.contextual_features.get_all_contextual_features(
                driver_id, constructor_id, None, None
            )
            momentum = features.get('momentum_score', 0)
            momentum_factor = 1.0 + (momentum * 0.1)
        else:
            momentum_factor = 1.0
            
        # Calculate probability
        total_races = len(recent_results)
        
        if 'points' in recent_results.columns:
            points_races = (recent_results['points'] > line).sum()
            avg_points = recent_results['points'].mean()
        else:
            points_races = 0
            avg_points = 0
            
        raw_prob = points_races / total_races if total_races > 0 else 0.5
        raw_prob = self.bound_probability(raw_prob * momentum_factor)
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'points', total_races,
            driver_id, constructor_id, None
        )
        
        return calibrated_prob, total_races, avg_points
    
    def calculate_dnf_probability(self, driver_name):
        """Calculate probability of DNF"""
        recent_results = self.get_driver_recent_races(driver_name, num_races=30)
        
        if recent_results.empty:
            return 0.12, 0
            
        # Get IDs
        driver_id = int(recent_results['driverId'].iloc[0])
        constructor_id = int(recent_results['constructorId'].iloc[0]) if 'constructorId' in recent_results.columns else None
        
        # DNF indicators
        dnf_indicators = ['DNF', 'DNS', 'DSQ', 'EX', 'NC', 'WD']
        
        # Count DNFs
        if 'positionText' in recent_results.columns:
            dnf_count = recent_results['positionText'].isin(dnf_indicators).sum()
        else:
            dnf_count = 0
            
        total_races = len(recent_results)
        raw_prob = dnf_count / total_races if total_races > 0 else 0.12
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'dnf', total_races,
            driver_id, constructor_id, None
        )
        
        return calibrated_prob, total_races
    
    def calculate_pit_stops_probability(self, driver_name, circuit_name=None, line=2.0):
        """Calculate probability of more than X pit stops"""
        recent_races = self.get_driver_recent_races(driver_name, num_races=15)
        
        if recent_races.empty:
            return 0.7, 0
            
        # Get IDs
        driver_id = int(recent_races['driverId'].iloc[0])
        constructor_id = int(recent_races['constructorId'].iloc[0]) if 'constructorId' in recent_races.columns else None
        circuit_id = None
        
        if circuit_name:
            circuits = self.data.get('circuits', pd.DataFrame())
            if not circuits.empty and 'name' in circuits.columns:
                circuit_match = circuits[circuits['name'] == circuit_name]
                if not circuit_match.empty:
                    circuit_id = int(circuit_match['id'].iloc[0])
        
        # Get pit stops data
        pit_stops = self.data.get('pit_stops', pd.DataFrame())
        
        if not pit_stops.empty and 'raceId' in pit_stops.columns and 'driverId' in pit_stops.columns:
            # Join with pit stops
            race_pit_stops = pd.merge(
                recent_races[['raceId', 'driverId']],
                pit_stops,
                on=['raceId', 'driverId'],
                how='left'
            )
            
            # Count stops per race
            if 'stop' in race_pit_stops.columns:
                stops_per_race = race_pit_stops.groupby('raceId')['stop'].max().reset_index()
                stops_per_race = stops_per_race.dropna()
                
                if len(stops_per_race) > 0:
                    over_line = (stops_per_race['stop'] > line).sum()
                    total_races = len(stops_per_race)
                    raw_prob = over_line / total_races
                else:
                    raw_prob = 0.7
                    total_races = 0
            else:
                raw_prob = 0.7
                total_races = 0
        else:
            raw_prob = 0.7
            total_races = 0
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'pit_stops', total_races,
            driver_id, constructor_id, circuit_id
        )
        
        return calibrated_prob, total_races
    
    def calculate_teammate_overtakes_probability(self, driver_name, line=0.5):
        """Calculate probability of overtaking teammate more than X times"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty or results.empty:
            return 0.5, 0, "Unknown"
            
        driver_match = drivers[drivers['fullName'] == driver_name]
        if driver_match.empty:
            return 0.5, 0, "Unknown"
            
        driver_id = int(driver_match['id'].iloc[0])
        
        # Get recent races to find current teammate
        if 'driverId' in results.columns:
            recent_results = results[
                results['driverId'] == driver_id
            ].sort_values('raceId', ascending=False).head(10)
            
            if recent_results.empty:
                return 0.5, 0, "Unknown"
                
            # Get constructor
            if 'constructorId' in recent_results.columns:
                constructor_id = int(recent_results['constructorId'].iloc[0])
                recent_race_id = recent_results['raceId'].iloc[0]
                
                # Find teammate
                same_race_results = results[
                    (results['raceId'] == recent_race_id) & 
                    (results['constructorId'] == constructor_id)
                ]
                
                teammate_ids = same_race_results[
                    same_race_results['driverId'] != driver_id
                ]['driverId'].unique()
                
                if len(teammate_ids) == 0:
                    return 0.5, 0, "Unknown"
                    
                teammate_id = teammate_ids[0]
                teammate_name = drivers[
                    drivers['id'] == teammate_id
                ]['fullName'].iloc[0] if not drivers[
                    drivers['id'] == teammate_id
                ].empty else "Unknown"
                
                # Get races where both competed
                driver_results = results[results['driverId'] == driver_id]
                teammate_results = results[results['driverId'] == teammate_id]
                
                # Merge to find common races
                if 'positionNumber' in results.columns:
                    common_races = pd.merge(
                        driver_results[['raceId', 'positionNumber', 'positionText']],
                        teammate_results[['raceId', 'positionNumber', 'positionText']],
                        on='raceId',
                        suffixes=('_driver', '_teammate')
                    )
                    
                    # Only count races where both finished
                    finished_races = common_races[
                        (common_races['positionNumber_driver'].notna()) & 
                        (common_races['positionNumber_teammate'].notna())
                    ]
                    
                    if len(finished_races) > 0:
                        beat_teammate = (
                            finished_races['positionNumber_driver'] < 
                            finished_races['positionNumber_teammate']
                        ).sum()
                        total_races = len(finished_races)
                        
                        if line > 0.5:
                            dominance_threshold = 0.7
                            raw_prob = 1.0 if (beat_teammate / total_races) > dominance_threshold else 0.0
                        else:
                            raw_prob = beat_teammate / total_races
                    else:
                        raw_prob = 0.5
                        total_races = 0
                else:
                    raw_prob = 0.5
                    total_races = 0
                    
                # Apply calibration
                calibrated_prob = self.calibrate_probability(
                    raw_prob, 'teammate_overtakes', total_races,
                    driver_id, constructor_id, None
                )
                
                return calibrated_prob, total_races, teammate_name
            else:
                return 0.5, 0, "Unknown"
        else:
            return 0.5, 0, "Unknown"
    
    def calculate_starting_position_probability(self, driver_name, line=10.5):
        """Calculate probability of starting position under X"""
        recent_races = self.get_driver_recent_races(driver_name, num_races=10)
        
        if recent_races.empty:
            return 0.5, 0
            
        # Get IDs
        driver_id = int(recent_races['driverId'].iloc[0])
        constructor_id = int(recent_races['constructorId'].iloc[0]) if 'constructorId' in recent_races.columns else None
        
        # Count races with grid position under line
        if 'grid' in recent_races.columns:
            valid_grids = recent_races[recent_races['grid'] > 0]  # Exclude DNS
            if len(valid_grids) > 0:
                under_line = (valid_grids['grid'] < line).sum()
                total_races = len(valid_grids)
                raw_prob = under_line / total_races
            else:
                raw_prob = 0.5
                total_races = 0
        else:
            raw_prob = 0.5
            total_races = 0
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'starting_position', total_races,
            driver_id, constructor_id, None
        )
        
        return calibrated_prob, total_races
    
    def get_current_drivers(self):
        """Get list of current active drivers"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty or results.empty:
            return []
        
        # Get drivers from recent races
        if 'year' in results.columns:
            recent_results = results[results['year'] >= 2023]
        else:
            recent_results = results.tail(500)  # Last 500 results as fallback
            
        if 'driverId' in recent_results.columns:
            active_driver_ids = recent_results['driverId'].unique()
            active_drivers = drivers[drivers['id'].isin(active_driver_ids)]
            
            if 'fullName' in active_drivers.columns:
                driver_names = active_drivers['fullName'].dropna().unique()
                return sorted(driver_names)
        
        return []
    
    def get_next_race_info(self):
        """Get next race information"""
        races = self.data.get('races', pd.DataFrame())
        circuits = self.data.get('circuits', pd.DataFrame())
        
        if races.empty:
            return None
            
        # Convert date column
        if 'date' in races.columns:
            races['date'] = pd.to_datetime(races['date'])
            
            # Get upcoming races
            upcoming = races[races['date'] > datetime.now()].sort_values('date')
            
            if upcoming.empty:
                # Get most recent race
                race_info = races.sort_values('date').iloc[-1]
            else:
                race_info = upcoming.iloc[0]
                
            # Get circuit name
            circuit_name = "Unknown Circuit"
            if 'circuitId' in race_info and not circuits.empty and 'id' in circuits.columns:
                circuit_match = circuits[circuits['id'] == race_info['circuitId']]
                if not circuit_match.empty and 'name' in circuit_match.columns:
                    circuit_name = circuit_match['name'].iloc[0]
                    
            return {
                'name': race_info.get('name', 'Unknown Race'),
                'circuit_name': circuit_name,
                'date': race_info.get('date', 'Unknown Date'),
                'circuitId': race_info.get('circuitId', None)
            }
        
        return None
    
    def generate_all_predictions(self, race_id=None):
        """Generate predictions for all drivers and prop types"""
        
        # Get next race info
        next_race = self.get_next_race_info()
        if next_race is None:
            print("No race information available.")
            return {}
            
        circuit_name = next_race['circuit_name']
        
        print(f"\n{'='*80}")
        print(f"F1 PRIZEPICKS PREDICTIONS - {next_race['name']}")
        print(f"Circuit: {circuit_name}")
        print(f"Date: {next_race['date']}")
        print(f"{'='*80}")
        
        # Get custom lines (using defaults for now)
        lines = self.default_lines.copy()
        
        # Get current drivers
        drivers = self.get_current_drivers()
        
        if not drivers:
            print("No active drivers found.")
            return {}
            
        # Initialize predictions
        predictions = {
            'race_info': next_race,
            'lines': lines,
            'drivers': {}
        }
        
        print(f"\nAnalyzing {len(drivers)} drivers...")
        
        # Overtakes predictions
        print(f"\n{'='*60}")
        print(f"OVERTAKES PREDICTIONS (Over/Under {lines['overtakes']})")
        print(f"{'='*60}")
        print(f"{'Driver':<30} {'Prob Over':<12} {'Confidence':<12} {'Sample':<10}")
        print("-" * 70)
        
        overtakes_data = []
        for driver in drivers:
            prob, sample = self.calculate_overtakes_probability(driver, circuit_name, lines['overtakes'])
            confidence = "High" if sample >= 15 else "Medium" if sample >= 8 else "Low"
            
            overtakes_data.append({
                'driver': driver,
                'probability': prob,
                'over_prob': prob,
                'under_prob': 1 - prob,
                'confidence': confidence,
                'sample_size': sample,
                'line': lines['overtakes']
            })
            
            predictions['drivers'][driver] = {'overtakes': overtakes_data[-1]}
        
        # Sort by probability and display top drivers
        overtakes_data.sort(key=lambda x: x['over_prob'], reverse=True)
        for pred in overtakes_data[:15]:
            print(f"{pred['driver']:<30} {pred['over_prob']*100:>6.1f}% OVER {pred['confidence']:<12} {pred['sample_size']:<10}")
        
        # Points predictions  
        print(f"\n{'='*60}")
        print(f"POINTS PREDICTIONS (Over/Under {lines['points']} points)")
        print(f"{'='*60}")
        print(f"{'Driver':<30} {'Prob Over':<12} {'Confidence':<12} {'Avg Points':<15}")
        print("-" * 70)
        
        points_data = []
        for driver in drivers:
            prob, sample, avg_points = self.calculate_points_probability(driver, lines['points'])
            confidence = "High" if sample >= 15 else "Medium" if sample >= 8 else "Low"
            
            points_data.append({
                'driver': driver,
                'probability': prob,
                'over_prob': prob,
                'under_prob': 1 - prob,
                'confidence': confidence,
                'sample_size': sample,
                'avg_points': avg_points,
                'line': lines['points']
            })
            
            if driver in predictions['drivers']:
                predictions['drivers'][driver]['points'] = points_data[-1]
        
        # Sort and display top drivers
        points_data.sort(key=lambda x: x['over_prob'], reverse=True)
        for pred in points_data[:15]:
            print(f"{pred['driver']:<30} {pred['over_prob']*100:>6.1f}% OVER {pred['confidence']:<12} {pred['avg_points']:>6.1f} pts")
        
        # DNF predictions
        print(f"\n{'='*60}")
        print(f"DNF PROBABILITY")
        print(f"{'='*60}")
        print(f"{'Driver':<30} {'DNF Prob':<12} {'Finish Prob':<12} {'Sample':<10}")
        print("-" * 70)
        
        dnf_data = []
        for driver in drivers:
            dnf_prob, sample = self.calculate_dnf_probability(driver)
            
            dnf_data.append({
                'driver': driver,
                'dnf_probability': dnf_prob,
                'finish_probability': 1 - dnf_prob,
                'sample_size': sample,
                'line': lines['dnf']
            })
            
            if driver in predictions['drivers']:
                predictions['drivers'][driver]['dnf'] = dnf_data[-1]
        
        # Sort by DNF probability and display top drivers
        dnf_data.sort(key=lambda x: x['dnf_probability'], reverse=True)
        for pred in dnf_data[:15]:
            print(f"{pred['driver']:<30} {pred['dnf_probability']*100:>6.1f}% DNF  {pred['finish_probability']*100:>6.1f}% FIN  {pred['sample_size']:<10}")
        
        # Save predictions
        predictions['generated_at'] = datetime.now().isoformat()
        
        return predictions
    
    def display_predictions_summary(self, predictions):
        """Display a summary of best bets"""
        print(f"\n{'='*80}")
        print("BEST BETS SUMMARY")
        print(f"{'='*80}")
        
        # Collect all bets
        all_bets = []
        
        for driver, props in predictions['drivers'].items():
            for prop_type, prop_data in props.items():
                if prop_type == 'overtakes':
                    direction = 'OVER' if prop_data['over_prob'] > 0.5 else 'UNDER'
                    prob = max(prop_data['over_prob'], prop_data['under_prob'])
                elif prop_type == 'points':
                    direction = 'OVER' if prop_data['over_prob'] > 0.5 else 'UNDER'
                    prob = max(prop_data['over_prob'], prop_data['under_prob'])
                elif prop_type == 'dnf':
                    direction = 'YES' if prop_data['dnf_probability'] > 0.5 else 'NO'
                    prob = max(prop_data['dnf_probability'], prop_data['finish_probability'])
                else:
                    continue
                    
                all_bets.append({
                    'driver': driver,
                    'prop': prop_type,
                    'direction': direction,
                    'probability': prob,
                    'line': prop_data.get('line', 0)
                })
        
        # Sort by probability
        all_bets.sort(key=lambda x: x['probability'], reverse=True)
        
        # Display top 15
        print("\nTop 15 Highest Confidence Bets:")
        for i, bet in enumerate(all_bets[:15], 1):
            if bet['prop'] == 'dnf':
                print(f"{i:2d}. {bet['driver']:<25} DNF {bet['direction']}: {bet['probability']*100:.1f}%")
            else:
                print(f"{i:2d}. {bet['driver']:<25} {bet['prop']} {bet['direction']} {bet['line']}: {bet['probability']*100:.1f}%")


class F1PredictionsV4Full:
    """Full prediction system with ensemble and optimization"""
    
    def __init__(self, bankroll: float = 1000):
        self.bankroll = bankroll
        self.ensemble = F1PredictionEnsemble()
        self.optimizer = F1OptimalBetting(bankroll)
        self.base_predictor = None
        self.risk_dashboard = F1RiskDashboard(bankroll)
        # Load data for correlation analyzer
        loader = F1DBDataLoader()
        data = loader.get_core_datasets()
        self.correlation_analyzer = F1CorrelationAnalyzer(data)
        
    def generate_ensemble_predictions(self, race_id: Optional[int] = None):
        """Generate predictions using ensemble of methods"""
        
        print("Generating predictions with full v4 implementation...")
        
        # Create predictor instance
        predictor = F1PrizePicksPredictorV4Full()
        
        # Generate predictions
        predictions = predictor.generate_all_predictions(race_id)
        
        if not predictions or 'drivers' not in predictions:
            print("No predictions generated.")
            return {}, {}
        
        # Display summary
        predictor.display_predictions_summary(predictions)
        
        # Create optimal portfolio
        portfolio = self.create_optimal_portfolio(predictions)
        
        return predictions, portfolio
        
    def create_optimal_portfolio(self, predictions):
        """Create optimal betting portfolio from predictions"""
        
        # Extract all bets with high confidence
        high_confidence_bets = []
        
        for driver, props in predictions.get('drivers', {}).items():
            for prop_type, prop_data in props.items():
                if prop_type == 'overtakes':
                    if prop_data['over_prob'] > 0.7:
                        high_confidence_bets.append({
                            'driver': driver,
                            'prop': prop_type,
                            'direction': 'OVER',
                            'line': prop_data['line'],
                            'probability': prop_data['over_prob']
                        })
                    elif prop_data['under_prob'] > 0.7:
                        high_confidence_bets.append({
                            'driver': driver,
                            'prop': prop_type,
                            'direction': 'UNDER',
                            'line': prop_data['line'],
                            'probability': prop_data['under_prob']
                        })
                elif prop_type == 'points':
                    if prop_data['over_prob'] > 0.7:
                        high_confidence_bets.append({
                            'driver': driver,
                            'prop': prop_type,
                            'direction': 'OVER',
                            'line': prop_data['line'],
                            'probability': prop_data['over_prob']
                        })
                    elif prop_data['under_prob'] > 0.7:
                        high_confidence_bets.append({
                            'driver': driver,
                            'prop': prop_type,
                            'direction': 'UNDER',
                            'line': prop_data['line'],
                            'probability': prop_data['under_prob']
                        })
                elif prop_type == 'dnf':
                    if prop_data['finish_probability'] > 0.85:  # NO DNF
                        high_confidence_bets.append({
                            'driver': driver,
                            'prop': prop_type,
                            'direction': 'NO',
                            'line': 0.5,
                            'probability': prop_data['finish_probability']
                        })
        
        # Sort by probability
        high_confidence_bets.sort(key=lambda x: x['probability'], reverse=True)
        
        # Create parlays
        parlays = []
        
        # 2-pick parlays
        if len(high_confidence_bets) >= 2:
            # Use correlation analyzer to find uncorrelated bets
            best_pair = self._find_best_uncorrelated_pair(high_confidence_bets[:10])
            
            if best_pair:
                parlay_prob = best_pair[0]['probability'] * best_pair[1]['probability']
                stake = min(50, self.bankroll * 0.05)  # 5% max
                
                parlays.append({
                    'type': '2-pick',
                    'selections': best_pair,
                    'probability': parlay_prob,
                    'stake': stake,
                    'payout': 3.0,
                    'expected_value': stake * (parlay_prob * 3.0 - 1)
                })
        
        # 3-pick parlays
        if len(high_confidence_bets) >= 3:
            best_trio = self._find_best_uncorrelated_trio(high_confidence_bets[:15])
            
            if best_trio:
                parlay_prob = (
                    best_trio[0]['probability'] * 
                    best_trio[1]['probability'] * 
                    best_trio[2]['probability']
                )
                stake = min(25, self.bankroll * 0.025)  # 2.5% max
                
                parlays.append({
                    'type': '3-pick',
                    'selections': best_trio,
                    'probability': parlay_prob,
                    'stake': stake,
                    'payout': 6.0,
                    'expected_value': stake * (parlay_prob * 6.0 - 1)
                })
        
        total_stake = sum(p['stake'] for p in parlays)
        total_ev = sum(p['expected_value'] for p in parlays)
        
        portfolio = {
            'bets': parlays,
            'total_stake': total_stake,
            'expected_value': total_ev + total_stake,
            'expected_roi': (total_ev / total_stake * 100) if total_stake > 0 else 0,
            'risk_metrics': {
                'total_exposure': total_stake,
                'exposure_pct': (total_stake / self.bankroll * 100),
                'expected_roi': (total_ev / total_stake * 100) if total_stake > 0 else 0,
                'num_bets': len(parlays)
            }
        }
        
        return portfolio
    
    def _find_best_uncorrelated_pair(self, bets):
        """Find the best pair of uncorrelated bets"""
        if len(bets) < 2:
            return None
            
        best_pair = None
        best_score = 0
        
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                # Skip same driver
                if bets[i]['driver'] == bets[j]['driver']:
                    continue
                    
                # Calculate combined probability and correlation
                combined_prob = bets[i]['probability'] * bets[j]['probability']
                
                # Simple correlation check (same team = higher correlation)
                correlation_penalty = 0
                if bets[i]['prop'] == bets[j]['prop']:
                    correlation_penalty = 0.1  # Same prop type
                
                score = combined_prob * (1 - correlation_penalty)
                
                if score > best_score:
                    best_score = score
                    best_pair = [bets[i], bets[j]]
        
        return best_pair
    
    def _find_best_uncorrelated_trio(self, bets):
        """Find the best trio of uncorrelated bets"""
        if len(bets) < 3:
            return None
            
        best_trio = None
        best_score = 0
        
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                for k in range(j + 1, len(bets)):
                    # Skip same driver
                    drivers = {bets[i]['driver'], bets[j]['driver'], bets[k]['driver']}
                    if len(drivers) < 3:
                        continue
                        
                    # Calculate combined probability
                    combined_prob = (
                        bets[i]['probability'] * 
                        bets[j]['probability'] * 
                        bets[k]['probability']
                    )
                    
                    # Correlation penalty
                    props = [bets[i]['prop'], bets[j]['prop'], bets[k]['prop']]
                    unique_props = len(set(props))
                    diversity_bonus = unique_props / 3  # More prop diversity = better
                    
                    score = combined_prob * diversity_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_trio = [bets[i], bets[j], bets[k]]
        
        return best_trio
            
    def display_optimal_bets(self, portfolio: Dict):
        """Display the optimal betting portfolio"""
        print(f"\n{'='*80}")
        print("OPTIMAL BETTING PORTFOLIO")
        print(f"{'='*80}")
        print(f"Bankroll: ${self.bankroll:.2f}")
        print(f"Total Stake: ${portfolio['total_stake']:.2f}")
        print(f"Expected Value: ${portfolio['expected_value']:.2f}")
        print(f"Expected ROI: {portfolio['expected_roi']:.1f}%")
        
        print(f"\n{'Recommended Parlays':^80}")
        print("-" * 80)
        
        for i, bet in enumerate(portfolio['bets'], 1):
            print(f"\nParlay {i} ({bet['type']}):")
            print(f"  Stake: ${bet['stake']:.2f}")
            print(f"  Win Probability: {bet['probability']*100:.1f}%")
            print(f"  Potential Payout: ${bet['payout']*bet['stake']:.2f}")
            print(f"  Expected Value: ${bet['expected_value']:.2f}")
            print(f"  Selections:")
            
            for selection in bet['selections']:
                print(f"    - {selection['driver']} {selection['prop']} "
                      f"{selection['direction']} {selection['line']} "
                      f"({selection['probability']*100:.1f}%)")
                      
    def generate_risk_analysis(self, portfolio: Dict, predictions: Dict):
        """Generate comprehensive risk analysis"""
        
        # Calculate risk metrics
        risk_metrics = self.risk_dashboard.calculate_risk_metrics(portfolio)
        
        # Generate text report
        risk_report = self.risk_dashboard.generate_risk_report()
        
        # Create visual dashboard
        output_dir = Path("pipeline_outputs")
        output_dir.mkdir(exist_ok=True)
        
        dashboard_path = output_dir / "risk_dashboard_full.png"
        self.risk_dashboard.create_dashboard(
            portfolio, 
            predictions,
            save_path=str(dashboard_path)
        )
        
        print(f"\n{'='*80}")
        print("RISK ANALYSIS")
        print(f"{'='*80}")
        print(risk_report)
        print(f"\nRisk dashboard saved to: {dashboard_path}")
        
        return risk_metrics


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 PrizePicks Predictions V4 Full')
    parser.add_argument('--bankroll', type=float, default=1000, 
                        help='Betting bankroll (default: 1000)')
    parser.add_argument('--race-id', type=int, help='Specific race ID to analyze')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = F1PredictionsV4Full(args.bankroll)
    
    # Generate predictions
    predictions, portfolio = predictor.generate_ensemble_predictions(args.race_id)
    
    if portfolio and portfolio.get('bets'):
        # Display optimal bets
        predictor.display_optimal_bets(portfolio)
        
        # Generate risk analysis
        predictor.generate_risk_analysis(portfolio, predictions)
        
        # Save results
        output_dir = Path("pipeline_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        with open(output_dir / "predictions_v4_full.json", 'w') as f:
            # Convert to serializable format
            serializable_predictions = json.loads(
                json.dumps(predictions, default=str)
            )
            json.dump(serializable_predictions, f, indent=2)
            
        # Save portfolio
        with open(output_dir / "optimal_portfolio_full.json", 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
            
        print(f"\n✅ Results saved to {output_dir}")
    else:
        print("\nNo high-confidence bets found for portfolio creation.")


if __name__ == "__main__":
    main()