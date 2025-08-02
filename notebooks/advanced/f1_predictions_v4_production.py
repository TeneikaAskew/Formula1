#!/usr/bin/env python3
"""Production-ready F1 predictions v4 with string ID support"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer
from f1_probability_calibration import F1ProbabilityCalibrator
from f1_risk_dashboard import F1RiskDashboard

warnings.filterwarnings('ignore')


class F1PrizePicksPredictorV4:
    """Production-ready F1 PrizePicks predictor"""
    
    def __init__(self):
        # Load data
        self.loader = F1DBDataLoader()
        self.data = self.loader.get_core_datasets()
        
        # Initialize components
        self.analyzer = F1PerformanceAnalyzer(self.data)
        self.calibrator = F1ProbabilityCalibrator()
        
        # Default lines
        self.default_lines = {
            'overtakes': 3.0,
            'points': 6.0,
            'pit_stops': 2.0,
            'teammate_overtakes': 0.5,
            'starting_position': 10.5,
            'dnf': 0.5
        }
        
    def bound_probability(self, prob, min_prob=0.05, max_prob=0.95):
        """Bound probability between min and max"""
        return max(min_prob, min(max_prob, prob))
        
    def get_driver_recent_races(self, driver_name, num_races=20):
        """Get recent race results for a driver"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty or results.empty:
            return pd.DataFrame()
        
        # Find driver ID (string format)
        driver_match = drivers[drivers['fullName'] == driver_name]
        if driver_match.empty:
            return pd.DataFrame()
            
        driver_id = driver_match['driverId'].iloc[0]  # Use driverId column (string)
        
        # Get recent results
        driver_results = results[
            results['driverId'] == driver_id
        ].sort_values('raceId', ascending=False).head(num_races)
        
        # Add positions gained if grid data exists
        if 'grid' in driver_results.columns and 'positionNumber' in driver_results.columns:
            # Calculate positions gained (positive = gained positions)
            driver_results['positions_gained'] = (
                driver_results['grid'] - driver_results['positionNumber']
            )
            # Handle NaN values (DNS, DNF, etc)
            driver_results['positions_gained'] = driver_results['positions_gained'].fillna(0)
            # Only count positive values for overtakes
            driver_results['positions_gained'] = driver_results['positions_gained'].apply(lambda x: max(0, x))
        else:
            driver_results['positions_gained'] = 0
            
        return driver_results
    
    def calculate_overtakes_probability(self, driver_name, line=3.0):
        """Calculate probability of driver making over X overtakes"""
        recent_races = self.get_driver_recent_races(driver_name, num_races=20)
        
        if recent_races.empty:
            return 0.5, 0
            
        # Count races with more than X overtakes
        if 'positions_gained' in recent_races.columns:
            over_count = (recent_races['positions_gained'] > line).sum()
        else:
            over_count = 0
            
        total_races = len(recent_races)
        raw_prob = over_count / total_races if total_races > 0 else 0.5
        
        # Apply calibration with Bayesian prior
        calibrated_prob = self.calibrator.apply_bayesian_prior(
            raw_prob, 'overtakes', total_races
        )
        
        return self.bound_probability(calibrated_prob), total_races
    
    def calculate_points_probability(self, driver_name, line=6.0):
        """Calculate probability of scoring over X points"""
        recent_results = self.get_driver_recent_races(driver_name, num_races=20)
        
        if recent_results.empty:
            return 0.5, 0, 0
            
        # Calculate probability
        total_races = len(recent_results)
        
        if 'points' in recent_results.columns:
            points_races = (recent_results['points'] > line).sum()
            avg_points = recent_results['points'].mean()
        else:
            points_races = 0
            avg_points = 0
            
        raw_prob = points_races / total_races if total_races > 0 else 0.5
        
        # Apply calibration
        calibrated_prob = self.calibrator.apply_bayesian_prior(
            raw_prob, 'points', total_races
        )
        
        return self.bound_probability(calibrated_prob), total_races, avg_points
    
    def calculate_dnf_probability(self, driver_name):
        """Calculate probability of DNF"""
        recent_results = self.get_driver_recent_races(driver_name, num_races=30)
        
        if recent_results.empty:
            return 0.12, 0
            
        # DNF indicators
        dnf_indicators = ['DNF', 'DNS', 'DSQ', 'EX', 'NC', 'WD', 'R']
        
        # Count DNFs
        if 'positionText' in recent_results.columns:
            dnf_count = recent_results['positionText'].isin(dnf_indicators).sum()
        elif 'statusId' in recent_results.columns:
            # Some F1DB versions use statusId
            dnf_status_ids = [3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            dnf_count = recent_results['statusId'].isin(dnf_status_ids).sum()
        else:
            dnf_count = 0
            
        total_races = len(recent_results)
        raw_prob = dnf_count / total_races if total_races > 0 else 0.12
        
        # Apply calibration
        calibrated_prob = self.calibrator.apply_bayesian_prior(
            raw_prob, 'dnf', total_races
        )
        
        return self.bound_probability(calibrated_prob), total_races
    
    def calculate_pit_stops_probability(self, driver_name, line=2.0):
        """Calculate probability of more than X pit stops"""
        recent_races = self.get_driver_recent_races(driver_name, num_races=15)
        
        if recent_races.empty:
            return 0.7, 0
            
        # Get driver ID
        driver_id = recent_races['driverId'].iloc[0]
        
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
                stops_per_race = race_pit_stops.groupby('raceId')['stop'].max().fillna(0)
                
                if len(stops_per_race) > 0:
                    over_line = (stops_per_race > line).sum()
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
        calibrated_prob = self.calibrator.apply_bayesian_prior(
            raw_prob, 'pit_stops', total_races
        )
        
        return self.bound_probability(calibrated_prob), total_races
    
    def calculate_teammate_overtakes_probability(self, driver_name, line=0.5):
        """Calculate probability of overtaking teammate"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty or results.empty:
            return 0.5, 0, "Unknown"
            
        driver_match = drivers[drivers['fullName'] == driver_name]
        if driver_match.empty:
            return 0.5, 0, "Unknown"
            
        driver_id = driver_match['driverId'].iloc[0]
        
        # Get recent races to find current teammate
        if 'driverId' in results.columns:
            recent_results = results[
                results['driverId'] == driver_id
            ].sort_values('raceId', ascending=False).head(10)
            
            if recent_results.empty:
                return 0.5, 0, "Unknown"
                
            # Get constructor
            if 'constructorId' in recent_results.columns:
                constructor_id = recent_results['constructorId'].iloc[0]
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
                teammate_match = drivers[drivers['driverId'] == teammate_id]
                teammate_name = teammate_match['fullName'].iloc[0] if not teammate_match.empty else "Unknown"
                
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
                        
                        raw_prob = beat_teammate / total_races
                    else:
                        raw_prob = 0.5
                        total_races = 0
                else:
                    raw_prob = 0.5
                    total_races = 0
                    
                # Apply calibration
                calibrated_prob = self.calibrator.apply_bayesian_prior(
                    raw_prob, 'teammate_overtakes', total_races
                )
                
                return self.bound_probability(calibrated_prob), total_races, teammate_name
            else:
                return 0.5, 0, "Unknown"
        else:
            return 0.5, 0, "Unknown"
    
    def calculate_starting_position_probability(self, driver_name, line=10.5):
        """Calculate probability of starting position under X"""
        recent_races = self.get_driver_recent_races(driver_name, num_races=10)
        
        if recent_races.empty:
            return 0.5, 0
            
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
        calibrated_prob = self.calibrator.apply_bayesian_prior(
            raw_prob, 'starting_position', total_races
        )
        
        return self.bound_probability(calibrated_prob), total_races
    
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
            active_drivers = drivers[drivers['driverId'].isin(active_driver_ids)]
            
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
            if 'circuitId' in race_info and not circuits.empty:
                circuit_match = circuits[circuits['id'] == race_info['circuitId']]
                if not circuit_match.empty and 'name' in circuit_match.columns:
                    circuit_name = circuit_match['name'].iloc[0]
                    
            return {
                'name': race_info.get('name', 'Unknown Race'),
                'circuit_name': circuit_name,
                'date': race_info.get('date', 'Unknown Date')
            }
        
        return None
    
    def generate_all_predictions(self):
        """Generate predictions for all drivers and prop types"""
        
        # Get next race info
        next_race = self.get_next_race_info()
        if next_race is None:
            print("No race information available.")
            return {}
            
        print(f"\n{'='*80}")
        print(f"F1 PRIZEPICKS PREDICTIONS - {next_race['name']}")
        print(f"Circuit: {next_race['circuit_name']}")
        print(f"Date: {next_race['date']}")
        print(f"{'='*80}")
        
        # Get current drivers
        drivers = self.get_current_drivers()
        
        if not drivers:
            print("No active drivers found.")
            return {}
            
        # Initialize predictions
        predictions = {
            'race_info': next_race,
            'lines': self.default_lines,
            'drivers': {}
        }
        
        print(f"\nAnalyzing {len(drivers)} drivers...")
        
        # Collect all predictions
        all_predictions = []
        
        # Overtakes predictions
        print(f"\n{'='*60}")
        print(f"OVERTAKES PREDICTIONS (Over/Under {self.default_lines['overtakes']})")
        print(f"{'='*60}")
        
        for driver in drivers:
            prob, sample = self.calculate_overtakes_probability(driver, self.default_lines['overtakes'])
            
            pred = {
                'driver': driver,
                'prop': 'overtakes',
                'line': self.default_lines['overtakes'],
                'over_prob': prob,
                'under_prob': 1 - prob,
                'sample_size': sample,
                'confidence': 'High' if sample >= 15 else 'Medium' if sample >= 8 else 'Low'
            }
            
            all_predictions.append(pred)
            
            if driver not in predictions['drivers']:
                predictions['drivers'][driver] = {}
            predictions['drivers'][driver]['overtakes'] = pred
        
        # Sort and display top overtakes
        overtakes_sorted = sorted(all_predictions, key=lambda x: x['over_prob'], reverse=True)
        print(f"{'Driver':<30} {'Prob Over':<12} {'Confidence':<12} {'Sample':<10}")
        print("-" * 70)
        for pred in overtakes_sorted[:10]:
            print(f"{pred['driver']:<30} {pred['over_prob']*100:>6.1f}% OVER {pred['confidence']:<12} {pred['sample_size']:<10}")
        
        # Points predictions
        print(f"\n{'='*60}")
        print(f"POINTS PREDICTIONS (Over/Under {self.default_lines['points']} points)")
        print(f"{'='*60}")
        
        points_predictions = []
        for driver in drivers:
            prob, sample, avg_points = self.calculate_points_probability(driver, self.default_lines['points'])
            
            pred = {
                'driver': driver,
                'prop': 'points',
                'line': self.default_lines['points'],
                'over_prob': prob,
                'under_prob': 1 - prob,
                'sample_size': sample,
                'avg_points': avg_points,
                'confidence': 'High' if sample >= 15 else 'Medium' if sample >= 8 else 'Low'
            }
            
            points_predictions.append(pred)
            all_predictions.append(pred)
            
            if driver not in predictions['drivers']:
                predictions['drivers'][driver] = {}
            predictions['drivers'][driver]['points'] = pred
        
        # Sort and display top points
        points_sorted = sorted(points_predictions, key=lambda x: x['over_prob'], reverse=True)
        print(f"{'Driver':<30} {'Prob Over':<12} {'Avg Points':<12} {'Sample':<10}")
        print("-" * 70)
        for pred in points_sorted[:10]:
            print(f"{pred['driver']:<30} {pred['over_prob']*100:>6.1f}% OVER {pred['avg_points']:>8.1f} pts {pred['sample_size']:<10}")
        
        # DNF predictions
        print(f"\n{'='*60}")
        print(f"DNF PROBABILITY")
        print(f"{'='*60}")
        
        dnf_predictions = []
        for driver in drivers:
            dnf_prob, sample = self.calculate_dnf_probability(driver)
            
            pred = {
                'driver': driver,
                'prop': 'dnf',
                'line': self.default_lines['dnf'],
                'dnf_prob': dnf_prob,
                'finish_prob': 1 - dnf_prob,
                'sample_size': sample
            }
            
            dnf_predictions.append(pred)
            
            # Add as a bet (NO DNF if finish prob is high)
            if pred['finish_prob'] > 0.8:
                all_predictions.append({
                    'driver': driver,
                    'prop': 'dnf',
                    'line': 0.5,
                    'direction': 'NO',
                    'probability': pred['finish_prob'],
                    'sample_size': sample
                })
            
            if driver not in predictions['drivers']:
                predictions['drivers'][driver] = {}
            predictions['drivers'][driver]['dnf'] = pred
        
        # Sort and display DNF
        dnf_sorted = sorted(dnf_predictions, key=lambda x: x['dnf_prob'], reverse=True)
        print(f"{'Driver':<30} {'DNF Prob':<12} {'Finish Prob':<12} {'Sample':<10}")
        print("-" * 70)
        for pred in dnf_sorted[:10]:
            print(f"{pred['driver']:<30} {pred['dnf_prob']*100:>6.1f}% DNF  {pred['finish_prob']*100:>6.1f}% FIN  {pred['sample_size']:<10}")
        
        return predictions, all_predictions
    
    def create_optimal_portfolio(self, all_predictions, bankroll=1000):
        """Create optimal betting portfolio"""
        
        # Filter high confidence bets
        high_confidence = []
        
        for pred in all_predictions:
            if 'over_prob' in pred:
                if pred['over_prob'] > 0.75:
                    high_confidence.append({
                        'driver': pred['driver'],
                        'prop': pred['prop'],
                        'direction': 'OVER',
                        'line': pred['line'],
                        'probability': pred['over_prob']
                    })
                elif pred['under_prob'] > 0.75:
                    high_confidence.append({
                        'driver': pred['driver'],
                        'prop': pred['prop'],
                        'direction': 'UNDER',
                        'line': pred['line'],
                        'probability': pred['under_prob']
                    })
            elif 'probability' in pred:  # DNF bets
                if pred['probability'] > 0.85:
                    high_confidence.append(pred)
        
        # Sort by probability
        high_confidence.sort(key=lambda x: x['probability'], reverse=True)
        
        # Create parlays
        parlays = []
        
        # Best 2-pick
        if len(high_confidence) >= 2:
            # Find uncorrelated pair
            best_pair = self._find_best_pair(high_confidence[:10])
            
            if best_pair:
                parlay_prob = best_pair[0]['probability'] * best_pair[1]['probability']
                stake = min(50, bankroll * 0.05)
                
                parlays.append({
                    'type': '2-pick',
                    'selections': best_pair,
                    'probability': parlay_prob,
                    'stake': stake,
                    'payout': 3.0,
                    'expected_value': stake * (parlay_prob * 3.0 - 1)
                })
        
        # Best 3-pick
        if len(high_confidence) >= 3:
            best_trio = self._find_best_trio(high_confidence[:15])
            
            if best_trio:
                parlay_prob = (
                    best_trio[0]['probability'] * 
                    best_trio[1]['probability'] * 
                    best_trio[2]['probability']
                )
                stake = min(25, bankroll * 0.025)
                
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
        
        return {
            'bets': parlays,
            'total_stake': total_stake,
            'expected_value': total_ev + total_stake,
            'expected_roi': (total_ev / total_stake * 100) if total_stake > 0 else 0
        }
    
    def _find_best_pair(self, bets):
        """Find best uncorrelated pair"""
        if len(bets) < 2:
            return None
            
        best_pair = None
        best_score = 0
        
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                # Different drivers
                if bets[i]['driver'] != bets[j]['driver']:
                    # Prefer different prop types
                    diversity_bonus = 1.1 if bets[i]['prop'] != bets[j]['prop'] else 1.0
                    
                    score = (
                        bets[i]['probability'] * 
                        bets[j]['probability'] * 
                        diversity_bonus
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_pair = [bets[i], bets[j]]
        
        return best_pair
    
    def _find_best_trio(self, bets):
        """Find best uncorrelated trio"""
        if len(bets) < 3:
            return None
            
        best_trio = None
        best_score = 0
        
        for i in range(len(bets)):
            for j in range(i + 1, len(bets)):
                for k in range(j + 1, len(bets)):
                    # All different drivers
                    drivers = {bets[i]['driver'], bets[j]['driver'], bets[k]['driver']}
                    if len(drivers) == 3:
                        # Prefer different prop types
                        props = {bets[i]['prop'], bets[j]['prop'], bets[k]['prop']}
                        diversity_bonus = 1.0 + (0.1 * len(props))
                        
                        score = (
                            bets[i]['probability'] * 
                            bets[j]['probability'] * 
                            bets[k]['probability'] * 
                            diversity_bonus
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_trio = [bets[i], bets[j], bets[k]]
        
        return best_trio
    
    def display_portfolio(self, portfolio, bankroll=1000):
        """Display betting portfolio"""
        print(f"\n{'='*80}")
        print("OPTIMAL BETTING PORTFOLIO")
        print(f"{'='*80}")
        print(f"Bankroll: ${bankroll:.2f}")
        print(f"Total Stake: ${portfolio['total_stake']:.2f}")
        print(f"Expected Value: ${portfolio['expected_value']:.2f}")
        print(f"Expected ROI: {portfolio['expected_roi']:.1f}%")
        
        for i, bet in enumerate(portfolio['bets'], 1):
            print(f"\nParlay {i} ({bet['type']}):")
            print(f"  Stake: ${bet['stake']:.2f}")
            print(f"  Win Probability: {bet['probability']*100:.1f}%")
            print(f"  Potential Payout: ${bet['payout']*bet['stake']:.2f}")
            print(f"  Expected Value: ${bet['expected_value']:.2f}")
            print(f"  Selections:")
            
            for sel in bet['selections']:
                direction = sel.get('direction', 'UNKNOWN')
                line = sel.get('line', 0)
                if sel['prop'] == 'dnf' and direction == 'NO':
                    print(f"    - {sel['driver']} NO DNF ({sel['probability']*100:.1f}%)")
                else:
                    print(f"    - {sel['driver']} {sel['prop']} {direction} {line} "
                          f"({sel['probability']*100:.1f}%)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 PrizePicks Predictions V4 Production')
    parser.add_argument('--bankroll', type=float, default=1000, 
                        help='Betting bankroll (default: 1000)')
    
    args = parser.parse_args()
    
    print("F1 PrizePicks Predictions V4 - Production Version")
    print("=" * 80)
    
    # Create predictor
    predictor = F1PrizePicksPredictorV4()
    
    # Generate predictions
    predictions, all_predictions = predictor.generate_all_predictions()
    
    if predictions and all_predictions:
        # Create optimal portfolio
        portfolio = predictor.create_optimal_portfolio(all_predictions, args.bankroll)
        
        # Display portfolio
        predictor.display_portfolio(portfolio, args.bankroll)
        
        # Save results
        output_dir = Path("pipeline_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save portfolio
        with open(output_dir / "portfolio_v4_production.json", 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
            
        print(f"\n✅ Results saved to {output_dir}/portfolio_v4_production.json")
        
        # Create risk dashboard if we have a portfolio
        if portfolio['bets']:
            risk_dashboard = F1RiskDashboard(args.bankroll)
            
            # Calculate risk metrics
            risk_metrics = risk_dashboard.calculate_risk_metrics(portfolio)
            
            # Generate risk report
            risk_report = risk_dashboard.generate_risk_report()
            print(f"\n{'='*80}")
            print("RISK ANALYSIS")
            print(f"{'='*80}")
            print(risk_report)
            
            # Create visual dashboard
            dashboard_path = output_dir / "risk_dashboard_v4.png"
            risk_dashboard.create_dashboard(
                portfolio, 
                predictions,
                save_path=str(dashboard_path)
            )
            print(f"\nRisk dashboard saved to: {dashboard_path}")
    else:
        print("\nNo predictions generated.")


if __name__ == "__main__":
    main()