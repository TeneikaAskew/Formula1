#!/usr/bin/env python3
"""F1 Backtesting Framework - Phase 2.3 Implementation

This module implements comprehensive backtesting for F1 predictions including:
- Historical prediction accuracy
- Prop bet hit rates
- ROI calculations
- Calibration testing
- Performance by driver/team/track
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats

# Add the current directory to Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import F1DBDataLoader
from f1_predictions_enhanced_v3 import F1PrizePicksPredictor


class F1BacktestingFramework:
    """Comprehensive backtesting for F1 predictions"""
    
    def __init__(self):
        self.loader = F1DBDataLoader()
        self.data = self.loader.get_core_datasets()
        self.results = []
        self.prop_results = defaultdict(list)
        
    def backtest_race(self, race_id: int, predictor: F1PrizePicksPredictor) -> Dict:
        """Backtest predictions for a single race
        
        Args:
            race_id: Race ID to test
            predictor: Predictor instance
            
        Returns:
            Dictionary with backtest results
        """
        # Get race info
        races = self.data.get('races', pd.DataFrame())
        race_info = races[races['id'] == race_id]
        
        if race_info.empty:
            return {}
            
        race_date = race_info.iloc[0]['date']
        circuit_id = race_info.iloc[0]['circuitId']
        race_name = race_info.iloc[0]['officialName']
        
        print(f"\nBacktesting: {race_name} ({race_date})")
        
        # Get actual results
        results = self.data.get('results', pd.DataFrame())
        race_results = results[results['raceId'] == race_id]
        
        if race_results.empty:
            return {}
            
        # Generate predictions (suppress output)
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            predictions = predictor.generate_all_predictions(race_id)
        
        # Evaluate each prediction
        backtest_results = {
            'race_id': race_id,
            'race_name': race_name,
            'race_date': race_date,
            'circuit_id': circuit_id,
            'prop_results': {},
            'overall_accuracy': 0,
            'roi': 0
        }
        
        prop_hits = defaultdict(lambda: {'correct': 0, 'total': 0})
        total_bets = 0
        total_wins = 0
        
        for driver_id, driver_preds in predictions.items():
            if not isinstance(driver_preds, dict) or 'predictions' not in driver_preds:
                continue
                
            # Get actual results for this driver
            driver_results = race_results[race_results['driverId'] == driver_id]
            
            if driver_results.empty:
                continue
                
            driver_name = driver_preds.get('driver_name', f'Driver {driver_id}')
            
            # Evaluate each prop
            for prop_type, pred in driver_preds['predictions'].items():
                if prop_type == 'overtakes':
                    actual = self._get_actual_overtakes(driver_id, race_id)
                elif prop_type == 'points':
                    actual = driver_results.iloc[0]['points']
                elif prop_type == 'starting_position':
                    actual = self._get_actual_grid_position(driver_id, race_id)
                elif prop_type == 'pit_stops':
                    actual = self._get_actual_pit_stops(driver_id, race_id)
                elif prop_type == 'dnf':
                    actual = 1 if driver_results.iloc[0]['positionText'] in ['DNF', 'DNS', 'DSQ'] else 0
                elif prop_type == 'teammate_overtakes':
                    actual = self._get_actual_teammate_overtakes(driver_id, race_id)
                else:
                    continue
                    
                if actual is None:
                    continue
                    
                # Check if prediction was correct
                line = pred.get('line', 0.5)
                over_prob = pred.get('over_prob', 0.5)
                under_prob = pred.get('under_prob', 0.5)
                recommendation = pred.get('recommendation', 'PASS')
                
                if recommendation != 'PASS':
                    total_bets += 1
                    
                    if prop_type == 'dnf':
                        # DNF is binary
                        predicted_outcome = 'YES' if recommendation == 'YES' else 'NO'
                        actual_outcome = 'YES' if actual > 0 else 'NO'
                        correct = predicted_outcome == actual_outcome
                    else:
                        # Over/Under props
                        predicted_outcome = recommendation  # OVER or UNDER
                        actual_outcome = 'OVER' if actual > line else 'UNDER'
                        correct = predicted_outcome == actual_outcome
                    
                    if correct:
                        total_wins += 1
                        
                    prop_hits[prop_type]['total'] += 1
                    prop_hits[prop_type]['correct'] += int(correct)
                    
                    # Store detailed result
                    self.prop_results[prop_type].append({
                        'race_id': race_id,
                        'driver_id': driver_id,
                        'driver_name': driver_name,
                        'line': line,
                        'predicted': recommendation,
                        'actual_value': actual,
                        'actual_outcome': actual_outcome,
                        'correct': correct,
                        'confidence': max(over_prob, under_prob)
                    })
        
        # Calculate overall metrics
        if total_bets > 0:
            backtest_results['overall_accuracy'] = total_wins / total_bets
            # Simple ROI calculation (assuming -110 odds)
            backtest_results['roi'] = (total_wins * 0.91 - (total_bets - total_wins)) / total_bets
        
        # Prop-specific accuracy
        backtest_results['prop_results'] = {}
        for prop, hits in prop_hits.items():
            if hits['total'] > 0:
                backtest_results['prop_results'][prop] = {
                    'accuracy': hits['correct'] / hits['total'],
                    'total_bets': hits['total'],
                    'correct': hits['correct']
                }
        
        return backtest_results
    
    def _get_actual_overtakes(self, driver_id: int, race_id: int) -> Optional[int]:
        """Get actual overtakes for a driver in a race"""
        results = self.data.get('results', pd.DataFrame())
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame())
        
        driver_result = results[(results['driverId'] == driver_id) & (results['raceId'] == race_id)]
        driver_grid = grid[(grid['driverId'] == driver_id) & (grid['raceId'] == race_id)]
        
        if driver_result.empty or driver_grid.empty:
            return None
            
        start_pos = driver_grid.iloc[0]['positionNumber']
        finish_pos = driver_result.iloc[0]['positionNumber']
        
        if pd.isna(start_pos) or pd.isna(finish_pos):
            return None
            
        return max(0, int(start_pos - finish_pos))
    
    def _get_actual_grid_position(self, driver_id: int, race_id: int) -> Optional[int]:
        """Get actual starting grid position"""
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame())
        driver_grid = grid[(grid['driverId'] == driver_id) & (grid['raceId'] == race_id)]
        
        if driver_grid.empty:
            return None
            
        return driver_grid.iloc[0]['positionNumber']
    
    def _get_actual_pit_stops(self, driver_id: int, race_id: int) -> Optional[int]:
        """Get actual number of pit stops"""
        pit_stops = self.data.get('pit_stops', pd.DataFrame())
        driver_stops = pit_stops[(pit_stops['driverId'] == driver_id) & (pit_stops['raceId'] == race_id)]
        
        return len(driver_stops) if not driver_stops.empty else None
    
    def _get_actual_teammate_overtakes(self, driver_id: int, race_id: int) -> Optional[float]:
        """Get actual teammate overtake score (PrizePicks scoring)"""
        results = self.data.get('results', pd.DataFrame())
        
        driver_result = results[(results['driverId'] == driver_id) & (results['raceId'] == race_id)]
        if driver_result.empty:
            return None
            
        constructor_id = driver_result.iloc[0]['constructorId']
        teammate_result = results[(results['raceId'] == race_id) & 
                                (results['constructorId'] == constructor_id) & 
                                (results['driverId'] != driver_id)]
        
        if teammate_result.empty:
            return None
            
        driver_pos = driver_result.iloc[0]['positionNumber']
        teammate_pos = teammate_result.iloc[0]['positionNumber']
        
        if pd.isna(driver_pos) or pd.isna(teammate_pos):
            return None
            
        # PrizePicks scoring: +1.5 if beat teammate, -1.5 if lost
        return 1.5 if driver_pos < teammate_pos else -1.5
    
    def backtest_season(self, year: int, sample_races: Optional[int] = None) -> Dict:
        """Backtest an entire season
        
        Args:
            year: Season year
            sample_races: Number of races to sample (None for all)
            
        Returns:
            Season backtest results
        """
        races = self.data.get('races', pd.DataFrame())
        season_races = races[races['year'] == year].sort_values('date')
        
        if sample_races:
            season_races = season_races.head(sample_races)
            
        print(f"\nBacktesting {year} season ({len(season_races)} races)")
        
        # Create predictor instance with disabled prompts
        import io
        import sys
        
        # Suppress prompts by simulating user input
        old_stdin = sys.stdin
        sys.stdin = io.StringIO('1\n')  # Choose default lines
        
        try:
            predictor = F1PrizePicksPredictor()
        finally:
            sys.stdin = old_stdin
        
        season_results = []
        
        for _, race in season_races.iterrows():
            race_id = race['id']
            
            # Skip if too early in season (need historical data)
            race_round = race.get('round', 0)
            if race_round < 3:
                continue
                
            result = self.backtest_race(race_id, predictor)
            if result:
                season_results.append(result)
        
        # Aggregate season metrics
        total_races = len(season_results)
        total_accuracy = np.mean([r['overall_accuracy'] for r in season_results if r['overall_accuracy'] > 0])
        total_roi = np.mean([r['roi'] for r in season_results if r['roi'] != 0])
        
        # Prop-specific metrics
        prop_accuracy = defaultdict(list)
        for result in season_results:
            for prop, metrics in result['prop_results'].items():
                prop_accuracy[prop].append(metrics['accuracy'])
        
        season_summary = {
            'year': year,
            'races_tested': total_races,
            'overall_accuracy': total_accuracy,
            'overall_roi': total_roi,
            'prop_accuracy': {prop: np.mean(accs) for prop, accs in prop_accuracy.items()},
            'race_results': season_results
        }
        
        return season_summary
    
    def evaluate_calibration(self) -> Dict:
        """Evaluate probability calibration quality"""
        calibration_results = {}
        
        for prop_type, results in self.prop_results.items():
            if not results:
                continue
                
            # Extract probabilities and outcomes
            confidences = [r['confidence'] for r in results]
            outcomes = [1 if r['correct'] else 0 for r in results]
            
            # Bin probabilities
            bins = np.linspace(0.5, 1.0, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            bin_accuracies = []
            bin_counts = []
            
            for i in range(len(bins) - 1):
                mask = (confidences >= bins[i]) & (confidences < bins[i+1])
                bin_outcomes = [o for c, o in zip(confidences, outcomes) if bins[i] <= c < bins[i+1]]
                
                if bin_outcomes:
                    bin_accuracies.append(np.mean(bin_outcomes))
                    bin_counts.append(len(bin_outcomes))
                else:
                    bin_accuracies.append(None)
                    bin_counts.append(0)
            
            # Calculate calibration error
            ece = 0
            for i, (acc, count) in enumerate(zip(bin_accuracies, bin_counts)):
                if acc is not None:
                    ece += (count / len(results)) * abs(acc - bin_centers[i])
            
            calibration_results[prop_type] = {
                'ece': ece,
                'bin_centers': bin_centers.tolist(),
                'bin_accuracies': bin_accuracies,
                'bin_counts': bin_counts,
                'total_predictions': len(results)
            }
        
        return calibration_results
    
    def plot_calibration_curves(self, save_path: Optional[str] = None):
        """Plot calibration curves for each prop type"""
        calibration = self.evaluate_calibration()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (prop_type, cal_data) in enumerate(calibration.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Filter out None values
            valid_points = [(c, a) for c, a in zip(cal_data['bin_centers'], cal_data['bin_accuracies']) 
                          if a is not None]
            
            if valid_points:
                centers, accuracies = zip(*valid_points)
                ax.plot(centers, accuracies, 'o-', label='Actual')
                ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', label='Perfect calibration')
                
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Actual Accuracy')
                ax.set_title(f'{prop_type.replace("_", " ").title()}\nECE: {cal_data["ece"]:.3f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0.5, 1.0)
                ax.set_ylim(0.5, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def generate_report(self, output_path: str = "backtest_report.json"):
        """Generate comprehensive backtest report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_predictions': sum(len(results) for results in self.prop_results.values()),
                'prop_types_tested': list(self.prop_results.keys()),
            },
            'prop_performance': {},
            'calibration': self.evaluate_calibration()
        }
        
        # Analyze each prop type
        for prop_type, results in self.prop_results.items():
            if not results:
                continue
                
            correct = sum(1 for r in results if r['correct'])
            total = len(results)
            
            # ROI calculation
            roi = (correct * 0.91 - (total - correct)) / total if total > 0 else 0
            
            report['prop_performance'][prop_type] = {
                'total_bets': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0,
                'roi': roi,
                'confidence_avg': np.mean([r['confidence'] for r in results]),
                'confidence_std': np.std([r['confidence'] for r in results])
            }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBacktest report saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        
        for prop_type, perf in report['prop_performance'].items():
            print(f"\n{prop_type.upper()}:")
            print(f"  Accuracy: {perf['accuracy']:.1%} ({perf['correct']}/{perf['total_bets']})")
            print(f"  ROI: {perf['roi']:.1%}")
            print(f"  Avg Confidence: {perf['confidence_avg']:.1%}")
        
        return report


def run_backtest_example():
    """Run example backtest on recent season"""
    framework = F1BacktestingFramework()
    
    # Backtest 2023 season (first 3 races for testing)
    season_results = framework.backtest_season(2023, sample_races=3)
    
    print(f"\n2023 Season Backtest Results:")
    print(f"Races tested: {season_results['races_tested']}")
    print(f"Overall accuracy: {season_results['overall_accuracy']:.1%}")
    print(f"Overall ROI: {season_results['overall_roi']:.1%}")
    
    print("\nProp-specific accuracy:")
    for prop, acc in season_results['prop_accuracy'].items():
        print(f"  {prop}: {acc:.1%}")
    
    # Generate report
    report = framework.generate_report("backtest_results_2023.json")
    
    # Plot calibration curves
    framework.plot_calibration_curves("calibration_curves_2023.png")
    
    return framework, season_results


if __name__ == "__main__":
    framework, results = run_backtest_example()