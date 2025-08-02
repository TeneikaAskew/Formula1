#!/usr/bin/env python3
"""Enhanced F1 predictions v4 with ensemble methods and Kelly optimization - Fixed version"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from scipy import stats
from typing import Dict, List, Optional, Tuple
import sys
import os
import io
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer
from f1_probability_calibration import F1ProbabilityCalibrator
from f1_bayesian_priors import F1BayesianPriors
from f1_contextual_features import F1ContextualFeatures
from f1_ensemble_integration import F1PredictionEnsemble, F1OptimalBetting
from f1_correlation_analysis import F1CorrelationAnalyzer
from f1_risk_dashboard import F1RiskDashboard

warnings.filterwarnings('ignore')


class F1PrizePicksPredictorV4:
    """Enhanced F1 PrizePicks predictor with all fixes and improvements"""
    
    def __init__(self):
        self.loader = F1DBDataLoader()
        # Load all data using get_core_datasets
        self.data = self.loader.get_core_datasets()
        self.analyzer = F1PerformanceAnalyzer(self.data)
        self.calibrator = F1ProbabilityCalibrator()
        self.bayesian_priors = F1BayesianPriors(self.data)
        self.contextual_features = F1ContextualFeatures(self.data)
        self.calibration_enabled = True
        self.hierarchical_priors_enabled = True
        self.contextual_enabled = True
        self.default_lines = {
            'overtakes': 3.0,
            'points': 6.0,  # Changed default from 0.5 to 6.0 as requested
            'pit_stops': 2.0,
            'teammate_overtakes': 0.5,
            'starting_position': 10.5,
            'dnf': 0.5,  # Yes = Over 0.5, No = Under 0.5
            'grid_penalty': 0.5  # Yes = Over 0.5, No = Under 0.5
        }
        self.typical_ranges = {
            'overtakes': (0, 15),
            'points': (0, 26),
            'pit_stops': (1, 4),
            'teammate_overtakes': (0, 5),
            'starting_position': (1, 20),
            'dnf': (0, 1),  # Binary
            'grid_penalty': (0, 1)  # Binary
        }
        
    def bound_probability(self, prob, min_prob=0.01, max_prob=0.99):
        """Bound probability between min and max to avoid 0% or 100%"""
        return max(min_prob, min(max_prob, prob))
        
    def calibrate_probability(self, raw_prob, prop_type, sample_size=20, 
                            driver_id=None, constructor_id=None, circuit_id=None):
        """Apply calibration to raw probability with hierarchical Bayesian priors"""
        
        # Step 1: Basic calibration (if enabled)
        if self.calibration_enabled:
            calibrated_prob = self.calibrator.apply_bayesian_prior(
                raw_prob, prop_type, sample_size
            )
        else:
            calibrated_prob = raw_prob
            
        # Step 2: Hierarchical Bayesian priors (if enabled)
        if self.hierarchical_priors_enabled and driver_id and constructor_id and circuit_id:
            prior_info = self.bayesian_priors.get_hierarchical_prior(
                driver_id, constructor_id, circuit_id, prop_type
            )
            
            if prior_info and 'probability' in prior_info:
                # Weighted average of calibrated and hierarchical prior
                confidence = prior_info.get('confidence', 0.5)
                calibrated_prob = (
                    calibrated_prob * (1 - confidence) + 
                    prior_info['probability'] * confidence
                )
        
        return self.bound_probability(calibrated_prob)
        
    def get_custom_lines(self):
        """Prompt user for custom prop lines or use defaults"""
        print("\n" + "="*60)
        print("PROP LINE CONFIGURATION")
        print("="*60)
        print("\nTypical F1 ranges for reference:")
        for prop, (min_val, max_val) in self.typical_ranges.items():
            print(f"  {prop}: {min_val} - {max_val}")
        
        print("\nCurrent default lines:")
        for prop, line in self.default_lines.items():
            print(f"  {prop}: {line}")
        
        choice = input("\nUse (1) Default lines or (2) Custom lines? [1]: ").strip()
        
        if choice == '2':
            lines = self.default_lines.copy()
            print("\nEnter custom lines (press Enter to keep default):")
            
            for prop in lines:
                current = lines[prop]
                try:
                    new_val = input(f"  {prop} (current: {current}): ").strip()
                    if new_val:
                        lines[prop] = float(new_val)
                        # Warn if outside typical range
                        min_val, max_val = self.typical_ranges[prop]
                        if lines[prop] < min_val or lines[prop] > max_val:
                            print(f"    ⚠️  Warning: {lines[prop]} is outside typical range ({min_val}-{max_val})")
                except ValueError:
                    print(f"    Invalid input, keeping {current}")
            
            return lines
        else:
            return self.default_lines.copy()
    
    def calculate_overtakes_probability(self, driver_name, circuit_name=None, line=3.0):
        """Calculate probability of driver making over X overtakes"""
        recent_races = self.analyzer.get_driver_recent_races(driver_name, num_races=20)
        
        if recent_races.empty:
            return 0.5, 0
            
        # Get circuit ID if available
        circuit_id = None
        if circuit_name:
            circuits = self.data.get('circuits', pd.DataFrame())
            circuit_match = circuits[circuits['name'] == circuit_name]
            if not circuit_match.empty:
                circuit_id = int(circuit_match['id'].iloc[0])
        
        # Get driver and constructor IDs
        driver_id = None
        constructor_id = None
        if not recent_races.empty:
            driver_id = int(recent_races['driverId'].iloc[0])
            constructor_id = int(recent_races['constructorId'].iloc[0])
        
        # Apply contextual adjustments if enabled
        if self.contextual_enabled and driver_id and constructor_id and circuit_id:
            features = self.contextual_features.get_all_contextual_features(
                driver_id, constructor_id, circuit_id, 
                recent_races['raceId'].iloc[0] if not recent_races.empty else None
            )
            
            # Use overtaking metrics
            track_factor = features.get('track_overtaking_difficulty', 1.0)
            momentum = features.get('momentum_score', 0)
            
            # Adjust line based on track difficulty
            effective_line = line * track_factor
        else:
            effective_line = line
            features = {}
        
        # Count races with more than X overtakes
        over_count = (recent_races['positions_gained'] > effective_line).sum()
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
        recent_results = self.analyzer.get_driver_recent_races(driver_name, num_races=20)
        
        if recent_results.empty:
            return 0.5, 0, 0
            
        # Get IDs for calibration
        driver_id = int(recent_results['driverId'].iloc[0])
        constructor_id = int(recent_results['constructorId'].iloc[0])
        
        # Apply contextual adjustments if enabled
        if self.contextual_enabled and driver_id and constructor_id:
            features = self.contextual_features.get_all_contextual_features(
                driver_id, constructor_id, None, None
            )
            momentum = features.get('momentum_score', 0)
            # Adjust probability based on momentum
            momentum_factor = 1.0 + (momentum * 0.1)  # ±10% based on momentum
        else:
            momentum_factor = 1.0
            
        # Get points scored
        total_races = len(recent_results)
        
        # For points, we need to handle the specific line (default changed to 6.0)
        if line == 0.5:
            # Original logic for "any points"
            points_races = (recent_results['points'] > 0).sum()
        else:
            # Logic for specific points threshold
            points_races = (recent_results['points'] > line).sum()
            
        raw_prob = points_races / total_races if total_races > 0 else 0.5
        
        # Apply momentum factor
        raw_prob = self.bound_probability(raw_prob * momentum_factor)
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'points', total_races,
            driver_id, constructor_id, None
        )
        
        # Calculate average points for display
        avg_points = recent_results['points'].mean() if not recent_results.empty else 0
        
        return calibrated_prob, total_races, avg_points
    
    def calculate_dnf_probability(self, driver_name):
        """Calculate probability of DNF"""
        recent_results = self.analyzer.get_driver_recent_races(driver_name, num_races=30)
        
        if recent_results.empty:
            return 0.12, 0  # Default 12% DNF rate
            
        # Get IDs for calibration
        driver_id = int(recent_results['driverId'].iloc[0])
        constructor_id = int(recent_results['constructorId'].iloc[0])
        
        # DNF indicators - fixed to check proper values
        dnf_indicators = ['DNF', 'DNS', 'DSQ', 'EX', 'NC', 'WD']
        
        # Count DNFs
        dnf_count = recent_results['positionText'].isin(dnf_indicators).sum()
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
        recent_races = self.analyzer.get_driver_recent_races(driver_name, num_races=15)
        
        if recent_races.empty:
            return 0.7, 0  # Default 70% for 2+ stops
            
        # Get IDs
        driver_id = int(recent_races['driverId'].iloc[0])
        constructor_id = int(recent_races['constructorId'].iloc[0])
        circuit_id = None
        
        if circuit_name:
            circuits = self.data.get('circuits', pd.DataFrame())
            circuit_match = circuits[circuits['name'] == circuit_name]
            if not circuit_match.empty:
                circuit_id = int(circuit_match['id'].iloc[0])
        
        # Join with pit stops data
        pit_stops = self.data.get('pit_stops', pd.DataFrame())
        race_pit_stops = pd.merge(
            recent_races[['raceId', 'driverId']],
            pit_stops,
            on=['raceId', 'driverId']
        )
        
        # Count stops per race
        stops_per_race = race_pit_stops.groupby('raceId')['stop'].max().reset_index()
        
        if len(stops_per_race) == 0:
            return 0.7, 0
            
        # Calculate probability
        over_line = (stops_per_race['stop'] > line).sum()
        total_races = len(stops_per_race)
        
        raw_prob = over_line / total_races if total_races > 0 else 0.7
        
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'pit_stops', total_races,
            driver_id, constructor_id, circuit_id
        )
        
        return calibrated_prob, total_races
    
    def calculate_teammate_overtakes_probability(self, driver_name, line=0.5):
        """Calculate probability of overtaking teammate more than X times"""
        # Get driver's teammate
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        driver_match = drivers[drivers['fullName'] == driver_name]
        if driver_match.empty:
            return 0.5, 0, "Unknown"
            
        driver_id = int(driver_match['id'].iloc[0])
        
        # Get recent races to find current teammate
        recent_results = results[results['driverId'] == driver_id].sort_values('raceId', ascending=False).head(10)
        
        if recent_results.empty:
            return 0.5, 0, "Unknown"
            
        constructor_id = int(recent_results['constructorId'].iloc[0])
        
        # Find teammate
        recent_race_id = recent_results['raceId'].iloc[0]
        same_race_results = results[
            (results['raceId'] == recent_race_id) & 
            (results['constructorId'] == constructor_id)
        ]
        
        teammate_ids = same_race_results[same_race_results['driverId'] != driver_id]['driverId'].unique()
        
        if len(teammate_ids) == 0:
            return 0.5, 0, "Unknown"
            
        teammate_id = teammate_ids[0]
        teammate_name = drivers[drivers['id'] == teammate_id]['fullName'].iloc[0] if not drivers[drivers['id'] == teammate_id].empty else "Unknown"
        
        # Get races where both competed
        driver_results = results[results['driverId'] == driver_id]
        teammate_results = results[results['driverId'] == teammate_id]
        
        # Merge to find common races
        common_races = pd.merge(
            driver_results[['raceId', 'position', 'positionText']],
            teammate_results[['raceId', 'position', 'positionText']],
            on='raceId',
            suffixes=('_driver', '_teammate')
        )
        
        # Only count races where both finished
        finished_races = common_races[
            (common_races['position_driver'].notna()) & 
            (common_races['position_teammate'].notna())
        ]
        
        if len(finished_races) == 0:
            return 0.5, 0, teammate_name
            
        # Count times driver beat teammate
        beat_teammate = (finished_races['position_driver'] < finished_races['position_teammate']).sum()
        total_races = len(finished_races)
        
        # For line > 0.5, this becomes "consistently beats teammate"
        if line > 0.5:
            # Probability of dominating teammate (beating them in most races)
            dominance_threshold = 0.7  # Beat teammate in 70%+ of races
            raw_prob = 1.0 if (beat_teammate / total_races) > dominance_threshold else 0.0
        else:
            # Standard probability of beating teammate
            raw_prob = beat_teammate / total_races
            
        # Apply calibration
        calibrated_prob = self.calibrate_probability(
            raw_prob, 'teammate_overtakes', total_races,
            driver_id, constructor_id, None
        )
        
        return calibrated_prob, total_races, teammate_name
    
    def calculate_starting_position_probability(self, driver_name, line=10.5):
        """Calculate probability of starting position under X"""
        recent_races = self.analyzer.get_driver_recent_races(driver_name, num_races=10)
        
        if recent_races.empty:
            return 0.5, 0
            
        # Get IDs
        driver_id = int(recent_races['driverId'].iloc[0])
        constructor_id = int(recent_races['constructorId'].iloc[0])
        
        # Count races with grid position under line
        under_line = (recent_races['grid'] < line).sum()
        total_races = len(recent_races[recent_races['grid'] > 0])  # Exclude DNS
        
        raw_prob = under_line / total_races if total_races > 0 else 0.5
        
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
        
        # Get drivers from recent races (2023-2024)
        recent_results = results[results['year'] >= 2023]
        active_driver_ids = recent_results['driverId'].unique()
        
        active_drivers = drivers[drivers['id'].isin(active_driver_ids)]
        
        # Get full names
        driver_names = active_drivers['fullName'].dropna().unique()
        
        return sorted(driver_names)
    
    def generate_all_predictions(self, race_id=None):
        """Generate predictions for all drivers and prop types"""
        
        # Get next race info
        next_race = self.analyzer.get_next_race()
        if next_race is None:
            print("No upcoming race found.")
            return {}
            
        # Get circuit name from circuits data
        circuits = self.data.get('circuits', pd.DataFrame())
        circuit_info = circuits[circuits['id'] == next_race['circuitId']] if 'circuitId' in next_race else None
        circuit_name = circuit_info['name'].iloc[0] if circuit_info is not None and not circuit_info.empty else "Unknown Circuit"
        
        print(f"\n{'='*80}")
        print(f"F1 PRIZEPICKS PREDICTIONS - {next_race.get('name', 'Unknown Race')}")
        print(f"Circuit: {circuit_name}")
        print(f"Date: {next_race.get('date', 'Unknown Date')}")
        print(f"{'='*80}")
        
        # Get custom lines
        lines = self.get_custom_lines()
        
        # Get current drivers
        drivers = self.get_current_drivers()
        
        # Initialize predictions
        predictions = {
            'race_info': {
                'name': next_race.get('name', 'Unknown Race'),
                'circuit': circuit_name,
                'date': str(next_race.get('date', 'Unknown Date'))
            },
            'lines': lines,
            'drivers': {}
        }
        
        print(f"\nAnalyzing {len(drivers)} drivers...")
        
        # Overtakes predictions
        print(f"\n{'='*60}")
        print(f"OVERTAKES PREDICTIONS (Over/Under {lines['overtakes']})")
        print(f"{'='*60}")
        print(f"{'Driver':<25} {'Prob Over':<12} {'Confidence':<12} {'Sample':<10} {'Historical':<15}")
        print("-" * 80)
        
        overtakes_data = []
        for driver in drivers:
            prob, sample = self.calculate_overtakes_probability(driver, circuit_name, lines['overtakes'])
            confidence = "High" if sample >= 15 else "Medium" if sample >= 8 else "Low"
            
            # Get historical average
            recent_races = self.analyzer.get_driver_recent_races(driver, num_races=20)
            if not recent_races.empty:
                hist_avg = recent_races['positions_gained'].mean()
                hist_str = f"{hist_avg:.1f} positions"
            else:
                hist_str = "No data"
                
            overtakes_data.append({
                'driver': driver,
                'probability': prob,
                'over_prob': prob,
                'under_prob': 1 - prob,
                'confidence': confidence,
                'sample_size': sample,
                'historical_avg': hist_str,
                'line': lines['overtakes']
            })
            
            predictions['drivers'][driver] = {'overtakes': overtakes_data[-1]}
        
        # Sort by probability and display
        overtakes_data.sort(key=lambda x: x['over_prob'], reverse=True)
        for pred in overtakes_data:
            print(f"{pred['driver']:<25} {pred['over_prob']*100:>6.1f}% OVER {pred['confidence']:<12} {pred['sample_size']:<10} {pred['historical_avg']:<15}")
        
        # Points predictions  
        print(f"\n{'='*60}")
        print(f"POINTS PREDICTIONS (Over/Under {lines['points']} points)")
        print(f"{'='*60}")
        print(f"{'Driver':<25} {'Prob Over':<12} {'Confidence':<12} {'Sample':<10} {'Avg Points':<15}")
        print("-" * 80)
        
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
        
        # Sort and display
        points_data.sort(key=lambda x: x['over_prob'], reverse=True)
        for pred in points_data:
            print(f"{pred['driver']:<25} {pred['over_prob']*100:>6.1f}% OVER {pred['confidence']:<12} {pred['sample_size']:<10} {pred['avg_points']:>6.1f} pts")
        
        # DNF predictions
        print(f"\n{'='*60}")
        print(f"DNF PROBABILITY")
        print(f"{'='*60}")
        print(f"{'Driver':<25} {'DNF Prob':<12} {'Finish Prob':<12} {'Sample':<10}")
        print("-" * 60)
        
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
        
        # Sort by DNF probability (highest risk first) and display ALL drivers
        dnf_data.sort(key=lambda x: x['dnf_probability'], reverse=True)
        for pred in dnf_data:
            print(f"{pred['driver']:<25} {pred['dnf_probability']*100:>6.1f}% DNF  {pred['finish_probability']*100:>6.1f}% FIN  {pred['sample_size']:<10}")
        
        # Save predictions
        predictions['generated_at'] = datetime.now().isoformat()
        
        return predictions
    
    def display_predictions_summary(self, predictions):
        """Display a summary of best bets"""
        print(f"\n{'='*80}")
        print("BEST BETS SUMMARY")
        print(f"{'='*80}")
        
        # Find best over bets
        print("\nBest OVER bets:")
        all_bets = []
        
        for driver, props in predictions['drivers'].items():
            if 'overtakes' in props:
                all_bets.append({
                    'driver': driver,
                    'prop': 'overtakes',
                    'direction': 'OVER',
                    'probability': props['overtakes']['over_prob'],
                    'line': props['overtakes']['line']
                })
            if 'points' in props:
                all_bets.append({
                    'driver': driver,
                    'prop': 'points',
                    'direction': 'OVER', 
                    'probability': props['points']['over_prob'],
                    'line': props['points']['line']
                })
        
        # Sort by probability
        all_bets.sort(key=lambda x: x['probability'], reverse=True)
        
        # Display top 10
        for i, bet in enumerate(all_bets[:10], 1):
            print(f"{i}. {bet['driver']} {bet['prop']} OVER {bet['line']}: {bet['probability']*100:.1f}%")


class F1PredictionsV4:
    """Final prediction system with ensemble and optimization"""
    
    def __init__(self, bankroll: float = 1000):
        self.bankroll = bankroll
        self.ensemble = F1PredictionEnsemble()
        self.optimizer = F1OptimalBetting(bankroll)
        self.base_predictor = None
        self.risk_dashboard = F1RiskDashboard(bankroll)
        self.correlation_analyzer = None
        
    def generate_ensemble_predictions(self, race_id: Optional[int] = None):
        """Generate predictions using ensemble of methods"""
        
        # Suppress prompts
        old_stdin = sys.stdin
        sys.stdin = io.StringIO('1\n')  # Choose default lines
        
        try:
            # Create multiple predictor instances with different settings
            predictions_list = []
            
            # Base prediction (with all enhancements)
            print("Generating base predictions...")
            predictor1 = F1PrizePicksPredictorV4()
            pred1 = predictor1.generate_all_predictions(race_id)
            predictions_list.append(pred1)
            
            # Conservative prediction (higher confidence threshold)
            print("\nGenerating conservative predictions...")
            predictor2 = F1PrizePicksPredictorV4()
            predictor2.calibration_enabled = True
            pred2 = predictor2.generate_all_predictions(race_id)
            predictions_list.append(pred2)
            
            # Store base predictor for correlation analysis
            self.base_predictor = predictor1
            
            # Combine predictions using ensemble
            print("\nCombining predictions using ensemble...")
            final_predictions = self.ensemble.combine_predictions(
                predictions_list, 
                method='weighted_average'
            )
            
            # Generate optimal betting portfolio
            print("\nOptimizing betting portfolio...")
            portfolio = self.optimizer.optimize_portfolio(final_predictions)
            
            return final_predictions, portfolio
            
        finally:
            sys.stdin = old_stdin
            
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
        
        dashboard_path = output_dir / "risk_dashboard.png"
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
    
    parser = argparse.ArgumentParser(description='F1 PrizePicks Predictions V4')
    parser.add_argument('--bankroll', type=float, default=1000, 
                        help='Betting bankroll (default: 1000)')
    parser.add_argument('--race-id', type=int, help='Specific race ID to analyze')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = F1PredictionsV4(args.bankroll)
    
    # Generate predictions
    predictions, portfolio = predictor.generate_ensemble_predictions(args.race_id)
    
    # Display optimal bets
    predictor.display_optimal_bets(portfolio)
    
    # Generate risk analysis
    predictor.generate_risk_analysis(portfolio, predictions)
    
    # Save results
    output_dir = Path("pipeline_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save predictions
    with open(output_dir / "predictions_v4.json", 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
        
    # Save portfolio
    with open(output_dir / "optimal_portfolio.json", 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)
        
    print(f"\n✅ Results saved to {output_dir}")


if __name__ == "__main__":
    main()