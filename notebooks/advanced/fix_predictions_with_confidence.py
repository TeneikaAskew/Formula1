#!/usr/bin/env python3
"""Fix predictions with proper confidence intervals and all prop types for PrizePicks"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

class F1PrizePicksPredictor:
    """Generate predictions with confidence intervals for all PrizePicks prop types"""
    
    def __init__(self):
        self.loader = F1DBDataLoader()
        # Load all data for the analyzer
        data_dict = self.loader.get_core_datasets()
        self.analyzer = F1PerformanceAnalyzer(data_dict)
        # Store data for reuse
        self.data = data_dict
        
        # Define prop types and their characteristics
        self.prop_types = {
            'overtakes': {'default_line': 2.5, 'type': 'over_under'},
            'points': {'default_line': 0.5, 'type': 'over_under'},
            'pit_stops': {'default_line': 2.5, 'type': 'over_under'},
            'starting_position': {'default_line': 10.5, 'type': 'over_under'},
            'sprint_points': {'default_line': 0.5, 'type': 'over_under'},
            'teammate_overtakes': {'default_line': 0.5, 'type': 'over_under'},
            'fastest_lap': {'default_line': 0.5, 'type': 'binary'},
            'first_pit_stop': {'default_line': 0.5, 'type': 'binary'},
            'laps_led': {'default_line': 0.5, 'type': 'over_under'},
            'dnf': {'default_line': 0.5, 'type': 'binary'}
        }
        
    def calculate_confidence_interval(self, historical_data, confidence=0.95):
        """Calculate confidence interval for predictions"""
        if len(historical_data) == 0:
            return 0, 0, 0, 0
            
        mean = np.mean(historical_data)
        std_err = stats.sem(historical_data)
        
        if std_err == 0 or np.isnan(std_err):
            return mean, mean, mean, 0
            
        ci = stats.t.interval(confidence, len(historical_data)-1, loc=mean, scale=std_err)
        
        return mean, ci[0], ci[1], std_err
    
    def calculate_probability_distribution(self, historical_data, line_value):
        """Calculate probability of going over/under a line"""
        if len(historical_data) == 0:
            return 0.5, 0.5, 0.5
            
        # Calculate mean and std
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        
        if std == 0:
            # No variance, deterministic outcome
            over_prob = 1.0 if mean > line_value else 0.0
            under_prob = 1.0 - over_prob
            confidence = 1.0
        else:
            # Use normal distribution assumption
            z_score = (line_value - mean) / std
            under_prob = stats.norm.cdf(z_score)
            over_prob = 1.0 - under_prob
            
            # Confidence based on sample size and consistency
            confidence = min(0.95, 0.5 + (len(historical_data) / 100) * 0.45)
            
        return over_prob, under_prob, confidence
    
    def get_first_pit_stop_probability(self, driver_id, race_id):
        """Calculate probability of driver making first pit stop"""
        pit_stops = self.data.get('pit_stops', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        if pit_stops.empty or races.empty:
            return 0.5, 0.5
            
        # Get recent races
        recent_races = races[races['id'] < race_id].tail(20)
        
        first_stops = []
        for rid in recent_races['id']:
            race_stops = pit_stops[pit_stops['raceId'] == rid]
            if not race_stops.empty:
                first_stop = race_stops.nsmallest(1, 'lap')
                if not first_stop.empty and first_stop.iloc[0]['driverId'] == driver_id:
                    first_stops.append(1)
                else:
                    first_stops.append(0)
        
        if first_stops:
            prob = np.mean(first_stops)
            confidence = min(0.9, 0.5 + len(first_stops) / 40)
            return prob, confidence
        
        return 0.5, 0.5
    
    def get_laps_led_prediction(self, driver_id, circuit_id):
        """Predict laps led based on historical data"""
        data = self.loader.load_csv_data()
        lap_times = data.get('lap_times', pd.DataFrame())
        races = data.get('races', pd.DataFrame())
        
        if lap_times.empty or races.empty:
            return 0, 0, 10, 0.5
            
        # Get races at this circuit
        circuit_races = races[races['circuitId'] == circuit_id]['id'].values
        
        laps_led_history = []
        for race_id in circuit_races[-5:]:  # Last 5 races at circuit
            race_laps = lap_times[lap_times['raceId'] == race_id]
            if not race_laps.empty:
                # Count laps where driver had position 1
                driver_laps = race_laps[race_laps['driverId'] == driver_id]
                laps_in_lead = len(driver_laps[driver_laps['position'] == 1])
                laps_led_history.append(laps_in_lead)
        
        if laps_led_history:
            mean, lower, upper, confidence = self.calculate_confidence_interval(laps_led_history)
            return mean, lower, upper, confidence
        
        return 0, 0, 10, 0.5
    
    def get_dnf_probability(self, driver_id, recent_races=20):
        """Calculate DNF probability"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return 0.15, 0.5  # Default 15% DNF rate
            
        # Get driver's recent results
        driver_results = results[results['driverId'] == driver_id].tail(recent_races)
        
        if driver_results.empty:
            return 0.15, 0.5
            
        # Count DNFs (positionText 'R' usually means retired/DNF)
        dnf_count = len(driver_results[driver_results['positionText'] == 'R'])
        dnf_rate = dnf_count / len(driver_results)
        
        # Confidence based on sample size
        confidence = min(0.9, 0.5 + len(driver_results) / 40)
        
        return dnf_rate, confidence
    
    def generate_all_predictions(self, race_id=None):
        """Generate predictions for all prop types with confidence intervals"""
        print("="*80)
        print("F1 PRIZEPICKS PREDICTIONS WITH CONFIDENCE INTERVALS")
        print("="*80)
        
        # Get active drivers
        drivers = self.analyzer.get_active_drivers()
        races = self.data.get('races', pd.DataFrame())
        
        # Get next race info
        if race_id is None:
            future_races = races[races['date'] > datetime.now().strftime('%Y-%m-%d')]
            if future_races.empty:
                print("No future races found")
                return {}
            race_id = future_races.iloc[0]['id']
            circuit_id = future_races.iloc[0]['circuitId']
            race_name = future_races.iloc[0]['officialName']
        else:
            race_info = races[races['id'] == race_id]
            if race_info.empty:
                print(f"Race {race_id} not found")
                return {}
            circuit_id = race_info.iloc[0]['circuitId']
            race_name = race_info.iloc[0]['officialName']
        
        print(f"\nGenerating predictions for: {race_name}")
        print(f"Race ID: {race_id}, Circuit ID: {circuit_id}")
        
        predictions = {}
        
        for driver_id in drivers['id'].unique():
            driver_name = drivers[drivers['id'] == driver_id].iloc[0]['name']
            print(f"\n{'='*60}")
            print(f"DRIVER: {driver_name} (ID: {driver_id})")
            print(f"{'='*60}")
            
            driver_predictions = {
                'driver_id': driver_id,
                'driver_name': driver_name,
                'predictions': {}
            }
            
            # 1. OVERTAKES
            overtake_history = self.analyzer.get_driver_overtakes(driver_id, limit=20)
            if not overtake_history.empty:
                overtakes_per_race = overtake_history.groupby('raceId').size().values
                mean, lower, upper, conf = self.calculate_confidence_interval(overtakes_per_race)
                line = 2.5
                over_prob, under_prob, _ = self.calculate_probability_distribution(overtakes_per_race, line)
                
                driver_predictions['predictions']['overtakes'] = {
                    'predicted': round(mean, 2),
                    'confidence_interval': [round(lower, 2), round(upper, 2)],
                    'confidence': round(conf, 3),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'under_prob': round(under_prob, 3),
                    'historical_avg': round(mean, 2),
                    'recommendation': 'OVER' if over_prob > 0.55 else ('UNDER' if under_prob > 0.55 else 'PASS')
                }
                
                print(f"\nOVERTAKES:")
                print(f"  Predicted: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
                print(f"  Line: {line}")
                print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                print(f"  Recommendation: {driver_predictions['predictions']['overtakes']['recommendation']}")
            
            # 2. POINTS
            points_history = self.analyzer.calculate_points_statistics(driver_id)
            if points_history and 'recent_points' in points_history:
                recent_points = points_history['recent_points']
                mean, lower, upper, conf = self.calculate_confidence_interval(recent_points)
                line = 0.5
                over_prob = sum(1 for p in recent_points if p > line) / len(recent_points) if recent_points else 0.5
                under_prob = 1 - over_prob
                
                driver_predictions['predictions']['points'] = {
                    'predicted': round(mean, 2),
                    'confidence_interval': [round(lower, 2), round(upper, 2)],
                    'confidence': round(conf, 3),
                    'line': line,
                    'over_prob': round(over_prob, 3),
                    'under_prob': round(under_prob, 3),
                    'points_finish_rate': points_history.get('points_finish_rate', 0),
                    'recommendation': 'OVER' if over_prob > 0.55 else ('UNDER' if under_prob > 0.55 else 'PASS')
                }
                
                print(f"\nPOINTS:")
                print(f"  Predicted: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
                print(f"  Points finish rate: {points_history.get('points_finish_rate', 0):.1%}")
                print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                print(f"  Recommendation: {driver_predictions['predictions']['points']['recommendation']}")
            
            # 3. STARTING POSITION
            quali_results = self.data.get('qualifying', pd.DataFrame())
            if not quali_results.empty:
                driver_quali = quali_results[quali_results['driverId'] == driver_id].tail(10)
                if not driver_quali.empty:
                    positions = driver_quali['position'].values
                    mean, lower, upper, conf = self.calculate_confidence_interval(positions)
                    line = 10.5
                    over_prob, under_prob, _ = self.calculate_probability_distribution(positions, line)
                    
                    driver_predictions['predictions']['starting_position'] = {
                        'predicted': round(mean, 1),
                        'confidence_interval': [round(lower, 1), round(upper, 1)],
                        'confidence': round(conf, 3),
                        'line': line,
                        'over_prob': round(over_prob, 3),
                        'under_prob': round(under_prob, 3),
                        'avg_grid': round(mean, 1),
                        'recommendation': 'UNDER' if under_prob > 0.55 else ('OVER' if over_prob > 0.55 else 'PASS')
                    }
                    
                    print(f"\nSTARTING POSITION:")
                    print(f"  Predicted: {mean:.1f} [{lower:.1f}, {upper:.1f}]")
                    print(f"  Line: {line}")
                    print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                    print(f"  Recommendation: {driver_predictions['predictions']['starting_position']['recommendation']}")
            
            # 4. FIRST PIT STOP
            first_pit_prob, confidence = self.get_first_pit_stop_probability(driver_id, race_id)
            driver_predictions['predictions']['first_pit_stop'] = {
                'probability': round(first_pit_prob, 3),
                'confidence': round(confidence, 3),
                'recommendation': 'YES' if first_pit_prob > 0.15 else 'NO'
            }
            
            print(f"\nFIRST PIT STOP:")
            print(f"  Probability: {first_pit_prob:.1%}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Recommendation: {driver_predictions['predictions']['first_pit_stop']['recommendation']}")
            
            # 5. LAPS LED
            mean_laps, lower_laps, upper_laps, laps_conf = self.get_laps_led_prediction(driver_id, circuit_id)
            line = 0.5
            over_prob = 1.0 if mean_laps > line else 0.0
            
            driver_predictions['predictions']['laps_led'] = {
                'predicted': round(mean_laps, 1),
                'confidence_interval': [round(lower_laps, 1), round(upper_laps, 1)],
                'confidence': round(laps_conf, 3),
                'line': line,
                'over_prob': round(over_prob, 3),
                'under_prob': round(1 - over_prob, 3),
                'recommendation': 'OVER' if mean_laps > 5 else 'UNDER'
            }
            
            print(f"\nLAPS LED:")
            print(f"  Predicted: {mean_laps:.1f} [{lower_laps:.1f}, {upper_laps:.1f}]")
            print(f"  Over {line}: {over_prob:.1%}")
            print(f"  Recommendation: {driver_predictions['predictions']['laps_led']['recommendation']}")
            
            # 6. DNF PROBABILITY
            dnf_prob, dnf_conf = self.get_dnf_probability(driver_id)
            driver_predictions['predictions']['dnf'] = {
                'probability': round(dnf_prob, 3),
                'confidence': round(dnf_conf, 3),
                'recommendation': 'YES' if dnf_prob > 0.25 else 'NO'
            }
            
            print(f"\nDNF PROBABILITY:")
            print(f"  Probability: {dnf_prob:.1%}")
            print(f"  Confidence: {dnf_conf:.1%}")
            print(f"  Recommendation: {driver_predictions['predictions']['dnf']['recommendation']}")
            
            # 7. TEAMMATE OVERTAKES
            teammate_stats = self.analyzer.calculate_teammate_overtake_statistics(driver_id)
            if teammate_stats and 'recent_scores' in teammate_stats:
                scores = teammate_stats['recent_scores']
                if scores:
                    mean, lower, upper, conf = self.calculate_confidence_interval(scores)
                    line = 0.5
                    over_prob = sum(1 for s in scores if s > line) / len(scores)
                    under_prob = 1 - over_prob
                    
                    driver_predictions['predictions']['teammate_overtakes'] = {
                        'predicted': round(mean, 2),
                        'confidence_interval': [round(lower, 2), round(upper, 2)],
                        'confidence': round(conf, 3),
                        'line': line,
                        'over_prob': round(over_prob, 3),
                        'under_prob': round(under_prob, 3),
                        'net_overtakes': teammate_stats.get('net_overtakes', 0),
                        'recommendation': 'OVER' if over_prob > 0.55 else ('UNDER' if under_prob > 0.55 else 'PASS')
                    }
                    
                    print(f"\nTEAMMATE OVERTAKES (PrizePicks Scoring):")
                    print(f"  Predicted score: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
                    print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                    print(f"  Net overtakes vs teammate: {teammate_stats.get('net_overtakes', 0)}")
                    print(f"  Recommendation: {driver_predictions['predictions']['teammate_overtakes']['recommendation']}")
            
            predictions[driver_id] = driver_predictions
        
        # Summary statistics
        print("\n" + "="*80)
        print("BETTING EDGE SUMMARY")
        print("="*80)
        
        strong_plays = []
        for driver_id, data in predictions.items():
            for prop, pred in data['predictions'].items():
                if 'over_prob' in pred:
                    edge_over = pred['over_prob'] - 0.5
                    edge_under = pred['under_prob'] - 0.5
                    
                    if edge_over > 0.1 or edge_under > 0.1:
                        strong_plays.append({
                            'driver': data['driver_name'],
                            'prop': prop,
                            'direction': 'OVER' if edge_over > edge_under else 'UNDER',
                            'probability': pred['over_prob'] if edge_over > edge_under else pred['under_prob'],
                            'edge': max(edge_over, edge_under),
                            'confidence': pred.get('confidence', 0.5)
                        })
        
        # Sort by edge
        strong_plays.sort(key=lambda x: x['edge'], reverse=True)
        
        print("\nSTRONG PLAYS (Edge > 10%):")
        for play in strong_plays[:10]:
            print(f"  {play['driver']:20} {play['prop']:20} {play['direction']:5} "
                  f"Prob: {play['probability']:.1%} Edge: {play['edge']*100:.1f}% "
                  f"Conf: {play['confidence']:.1%}")
        
        return predictions

def main():
    """Run the enhanced prediction system"""
    predictor = F1PrizePicksPredictor()
    predictions = predictor.generate_all_predictions()
    
    # Save predictions
    output_path = Path("pipeline_outputs/enhanced_predictions.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    print(f"\nPredictions saved to: {output_path}")
    
    # Create betting recommendations
    print("\n" + "="*80)
    print("RECOMMENDED PRIZEPICKS PARLAYS")
    print("="*80)
    
    # Find best 6-pick parlay
    all_picks = []
    for driver_id, data in predictions.items():
        for prop, pred in data['predictions'].items():
            if 'recommendation' in pred and pred['recommendation'] != 'PASS':
                if 'over_prob' in pred:
                    prob = pred['over_prob'] if pred['recommendation'] == 'OVER' else pred['under_prob']
                    edge = prob - 0.5
                elif 'probability' in pred:
                    prob = pred['probability']
                    edge = prob - 0.15  # Assume 15% baseline for binary props
                else:
                    continue
                    
                all_picks.append({
                    'driver': data['driver_name'],
                    'prop': prop,
                    'pick': pred['recommendation'],
                    'probability': prob,
                    'edge': edge,
                    'confidence': pred.get('confidence', 0.5)
                })
    
    # Sort by edge * confidence
    all_picks.sort(key=lambda x: x['edge'] * x['confidence'], reverse=True)
    
    # Show top 6-pick parlay
    print("\nBEST 6-PICK PARLAY:")
    parlay_prob = 1.0
    for i, pick in enumerate(all_picks[:6]):
        parlay_prob *= pick['probability']
        print(f"{i+1}. {pick['driver']:20} {pick['prop']:20} {pick['pick']:5} "
              f"({pick['probability']:.1%} win rate)")
    
    print(f"\nParlay win probability: {parlay_prob:.2%}")
    print(f"PrizePicks payout: 25x")
    print(f"Expected value: {25 * parlay_prob:.2f}x")
    
    # Alternative safer 4-pick
    print("\nSAFER 4-PICK PARLAY:")
    parlay_prob = 1.0
    for i, pick in enumerate(all_picks[:4]):
        parlay_prob *= pick['probability']
        print(f"{i+1}. {pick['driver']:20} {pick['prop']:20} {pick['pick']:5} "
              f"({pick['probability']:.1%} win rate)")
    
    print(f"\nParlay win probability: {parlay_prob:.2%}")
    print(f"PrizePicks payout: 10x")
    print(f"Expected value: {10 * parlay_prob:.2f}x")

if __name__ == "__main__":
    main()