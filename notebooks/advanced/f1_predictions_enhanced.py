#!/usr/bin/env python3
"""Enhanced F1 predictions with confidence intervals for PrizePicks"""

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
    
    def get_driver_overtake_stats(self, driver_id):
        """Get overtake statistics for a specific driver"""
        overtakes = self.analyzer.analyze_overtakes()
        if isinstance(overtakes, pd.DataFrame) and driver_id in overtakes.index:
            return overtakes.loc[driver_id]
        return None
    
    def get_driver_points_stats(self, driver_id):
        """Get points statistics for a specific driver"""
        # Get points analysis
        points_analysis = self.analyzer.analyze_points()
        if isinstance(points_analysis, pd.DataFrame) and driver_id in points_analysis.index:
            driver_data = points_analysis.loc[driver_id]
            
            # Get recent points from results
            results = self.data.get('results', pd.DataFrame())
            if not results.empty:
                driver_results = results[results['driverId'] == driver_id].tail(20)
                recent_points = driver_results['points'].values if 'points' in driver_results.columns else []
                points_finish_rate = len(driver_results[driver_results['points'] > 0]) / len(driver_results) if len(driver_results) > 0 else 0
                
                return {
                    'recent_points': recent_points,
                    'points_finish_rate': points_finish_rate,
                    'avg_points': driver_data.get('avg_points', 0),
                    'total_points': driver_data.get('total_points', 0)
                }
        return {}
    
    def get_driver_qualifying_stats(self, driver_id):
        """Get qualifying statistics for a specific driver"""
        quali_results = self.data.get('qualifying', pd.DataFrame())
        if quali_results.empty:
            return []
            
        driver_quali = quali_results[quali_results['driverId'] == driver_id].tail(10)
        if 'positionNumber' in driver_quali.columns:
            positions = driver_quali['positionNumber'].dropna().values
            return positions if len(positions) > 0 else []
        elif 'position' in driver_quali.columns:
            positions = driver_quali['position'].dropna().values
            return positions if len(positions) > 0 else []
        return []
    
    def get_driver_pit_stop_stats(self, driver_id):
        """Get pit stop statistics for a specific driver"""
        pit_stops = self.data.get('pit_stops', pd.DataFrame())
        if pit_stops.empty:
            return {}
            
        driver_stops = pit_stops[pit_stops['driverId'] == driver_id]
        if driver_stops.empty:
            return {}
            
        # Group by race to get stops per race
        stops_per_race = driver_stops.groupby('raceId').size()
        
        return {
            'avg_stops': stops_per_race.mean(),
            'median_stops': stops_per_race.median(),
            'recent_stops': stops_per_race.tail(10).values
        }
    
    def get_teammate_overtake_stats(self, driver_id):
        """Get teammate overtake statistics"""
        # Get teammate overtake analysis
        teammate_analysis = self.analyzer.analyze_teammate_overtakes()
        
        if isinstance(teammate_analysis, pd.DataFrame) and driver_id in teammate_analysis.index:
            driver_data = teammate_analysis.loc[driver_id]
            
            # Calculate recent scores based on PrizePicks scoring
            results = self.data.get('results', pd.DataFrame())
            if not results.empty:
                # Get recent races for this driver
                driver_results = results[results['driverId'] == driver_id].tail(10)
                
                scores = []
                for race_id in driver_results['raceId'].unique():
                    race_results = results[results['raceId'] == race_id]
                    
                    # Find teammate result
                    driver_result = race_results[race_results['driverId'] == driver_id]
                    if not driver_result.empty:
                        driver_constructor = driver_result.iloc[0]['constructorId']
                        teammate_result = race_results[(race_results['constructorId'] == driver_constructor) & 
                                                     (race_results['driverId'] != driver_id)]
                        
                        if not teammate_result.empty and not driver_result.empty:
                            driver_pos = driver_result.iloc[0]['positionNumber']
                            teammate_pos = teammate_result.iloc[0]['positionNumber']
                            
                            if pd.notna(driver_pos) and pd.notna(teammate_pos):
                                # PrizePicks scoring: +1.5 if beat teammate, -1.5 if lost
                                if driver_pos < teammate_pos:
                                    scores.append(1.5)
                                else:
                                    scores.append(-1.5)
                
                return {
                    'recent_scores': scores,
                    'net_overtakes': driver_data.get('net_overtakes', 0),
                    'win_rate': driver_data.get('teammate_win_rate', 0.5)
                }
        
        return {}
    
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
        
        # Get all analysis data upfront
        overtake_analysis = self.analyzer.analyze_overtakes()
        
        for driver_id in drivers['id'].unique():  # Process all drivers
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
            if isinstance(overtake_analysis, pd.DataFrame) and driver_id in overtake_analysis.index:
                driver_overtake_data = overtake_analysis.loc[driver_id]
                avg_overtakes = driver_overtake_data.get('avg_overtakes', 2.5)
                
                # Get recent race overtakes for distribution
                results = self.data.get('results', pd.DataFrame())
                grid = self.data.get('races_starting_grid_positions', pd.DataFrame())
                
                if not results.empty and not grid.empty:
                    # Get driver's recent races
                    driver_results = results[results['driverId'] == driver_id].tail(20)
                    driver_grid = grid[grid['driverId'] == driver_id]
                    
                    # Merge to calculate overtakes
                    merged = driver_results.merge(
                        driver_grid[['raceId', 'positionNumber']].rename(columns={'positionNumber': 'gridPosition'}),
                        on='raceId',
                        how='left'
                    )
                    
                    merged['overtakes'] = (merged['gridPosition'] - merged['positionNumber']).apply(lambda x: max(0, x) if pd.notna(x) else 0)
                    overtakes_per_race = merged['overtakes'].values
                    
                    if len(overtakes_per_race) > 0:
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
                            'historical_avg': round(avg_overtakes, 2),
                            'recommendation': 'OVER' if over_prob > 0.55 else ('UNDER' if under_prob > 0.55 else 'PASS')
                        }
                        
                        print(f"\nOVERTAKES:")
                        print(f"  Predicted: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
                        print(f"  Line: {line}")
                        print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                        print(f"  Recommendation: {driver_predictions['predictions']['overtakes']['recommendation']}")
            
            # 2. POINTS
            try:
                points_stats = self.get_driver_points_stats(driver_id)
                if points_stats and 'recent_points' in points_stats:
                    recent_points = points_stats['recent_points']
                    # Check if we have valid data
                    if isinstance(recent_points, (list, np.ndarray)) and len(recent_points) > 0:
                        mean, lower, upper, conf = self.calculate_confidence_interval(recent_points)
                        line = 0.5
                        over_prob = sum(1 for p in recent_points if p > line) / len(recent_points)
                        under_prob = 1 - over_prob
                        
                        driver_predictions['predictions']['points'] = {
                            'predicted': round(mean, 2),
                            'confidence_interval': [round(lower, 2), round(upper, 2)],
                            'confidence': round(conf, 3),
                            'line': line,
                            'over_prob': round(over_prob, 3),
                            'under_prob': round(under_prob, 3),
                            'points_finish_rate': points_stats.get('points_finish_rate', 0),
                            'recommendation': 'OVER' if over_prob > 0.55 else ('UNDER' if under_prob > 0.55 else 'PASS')
                        }
                        
                        print(f"\nPOINTS:")
                        print(f"  Predicted: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
                        print(f"  Points finish rate: {points_stats.get('points_finish_rate', 0):.1%}")
                        print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                        print(f"  Recommendation: {driver_predictions['predictions']['points']['recommendation']}")
            except Exception as e:
                print(f"\nPOINTS: Error calculating - {str(e)}")
            
            # 3. STARTING POSITION
            quali_positions = self.get_driver_qualifying_stats(driver_id)
            if len(quali_positions) > 0:
                mean, lower, upper, conf = self.calculate_confidence_interval(quali_positions)
                line = 10.5
                over_prob, under_prob, _ = self.calculate_probability_distribution(quali_positions, line)
                
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
            
            # 4. PIT STOPS
            pit_stats = self.get_driver_pit_stop_stats(driver_id)
            if pit_stats and 'recent_stops' in pit_stats:
                recent_stops = pit_stats['recent_stops']
                if len(recent_stops) > 0:
                    mean, lower, upper, conf = self.calculate_confidence_interval(recent_stops)
                    line = 2.5
                    over_prob, under_prob, _ = self.calculate_probability_distribution(recent_stops, line)
                    
                    driver_predictions['predictions']['pit_stops'] = {
                        'predicted': round(mean, 2),
                        'confidence_interval': [round(lower, 2), round(upper, 2)],
                        'confidence': round(conf, 3),
                        'line': line,
                        'over_prob': round(over_prob, 3),
                        'under_prob': round(under_prob, 3),
                        'avg_stops': round(pit_stats['avg_stops'], 2),
                        'recommendation': 'UNDER' if under_prob > 0.55 else ('OVER' if over_prob > 0.55 else 'PASS')
                    }
                    
                    print(f"\nPIT STOPS:")
                    print(f"  Predicted: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
                    print(f"  Line: {line}")
                    print(f"  Over {line}: {over_prob:.1%} | Under {line}: {under_prob:.1%}")
                    print(f"  Recommendation: {driver_predictions['predictions']['pit_stops']['recommendation']}")
            
            # 5. DNF PROBABILITY
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
            
            # 6. TEAMMATE OVERTAKES
            teammate_stats = self.get_teammate_overtake_stats(driver_id)
            if teammate_stats and 'recent_scores' in teammate_stats:
                scores = teammate_stats['recent_scores']
                if len(scores) > 0:
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
        
        # Generate formatted tables
        self.generate_formatted_tables(predictions)
        
        return predictions
    
    def generate_formatted_tables(self, predictions):
        """Generate formatted markdown-style tables for all predictions"""
        print("\n" + "="*80)
        print("F1 PROP BETTING ANALYSIS TABLES")
        print("="*80)
        
        # Organize data by prop type
        overtakes_data = []
        points_data = []
        starting_pos_data = []
        pit_stops_data = []
        teammate_data = []
        dnf_data = []
        
        for driver_id, data in predictions.items():
            driver_name = data['driver_name']
            preds = data['predictions']
            
            # Overtakes
            if 'overtakes' in preds:
                overtakes_data.append({
                    'driver': driver_name,
                    'predicted': preds['overtakes']['predicted'],
                    'ci': preds['overtakes']['confidence_interval'],
                    'over_prob': preds['overtakes']['over_prob'],
                    'under_prob': preds['overtakes']['under_prob'],
                    'recommendation': preds['overtakes']['recommendation']
                })
            
            # Points
            if 'points' in preds:
                points_data.append({
                    'driver': driver_name,
                    'predicted': preds['points']['predicted'],
                    'points_rate': preds['points']['points_finish_rate'],
                    'over_prob': preds['points']['over_prob'],
                    'under_prob': preds['points']['under_prob'],
                    'recommendation': preds['points']['recommendation']
                })
            
            # Starting Position
            if 'starting_position' in preds:
                starting_pos_data.append({
                    'driver': driver_name,
                    'predicted': preds['starting_position']['predicted'],
                    'ci': preds['starting_position']['confidence_interval'],
                    'over_prob': preds['starting_position']['over_prob'],
                    'under_prob': preds['starting_position']['under_prob'],
                    'recommendation': preds['starting_position']['recommendation']
                })
            
            # Pit Stops
            if 'pit_stops' in preds:
                pit_stops_data.append({
                    'driver': driver_name,
                    'predicted': preds['pit_stops']['predicted'],
                    'ci': preds['pit_stops']['confidence_interval'],
                    'over_prob': preds['pit_stops']['over_prob'],
                    'under_prob': preds['pit_stops']['under_prob'],
                    'recommendation': preds['pit_stops']['recommendation']
                })
            
            # Teammate Overtakes
            if 'teammate_overtakes' in preds:
                teammate_data.append({
                    'driver': driver_name,
                    'predicted': preds['teammate_overtakes']['predicted'],
                    'ci': preds['teammate_overtakes']['confidence_interval'],
                    'over_prob': preds['teammate_overtakes']['over_prob'],
                    'under_prob': preds['teammate_overtakes']['under_prob'],
                    'recommendation': preds['teammate_overtakes']['recommendation']
                })
            
            # DNF
            if 'dnf' in preds:
                dnf_data.append({
                    'driver': driver_name,
                    'probability': preds['dnf']['probability'],
                    'confidence': preds['dnf']['confidence'],
                    'recommendation': preds['dnf']['recommendation']
                })
        
        # Print Overtakes Table
        print("\n" + "="*120)
        print("OVERTAKES (Over/Under 2.5)")
        print("="*120)
        print(f"{'Driver':<25} {'Predicted':<10} {'Confidence Interval':<20} {'Over 2.5%':<12} {'Under 2.5%':<12} {'Recommendation':<15}")
        print("-"*120)
        
        # Sort by predicted overtakes descending
        overtakes_data.sort(key=lambda x: x['predicted'], reverse=True)
        
        for row in overtakes_data:
            rec = row['recommendation']
            if rec != 'PASS':
                rec = f">>> {rec} <<<"
            ci_str = f"[{row['ci'][0]:.2f}, {row['ci'][1]:.2f}]"
            print(f"{row['driver']:<25} {row['predicted']:<10.2f} {ci_str:<20} "
                  f"{row['over_prob']*100:<12.1f}% {row['under_prob']*100:<12.1f}% {rec:<15}")
        
        # Print Points Table
        print("\n" + "="*120)
        print("POINTS FINISH (Over/Under 0.5)")
        print("="*120)
        print(f"{'Driver':<25} {'Points Rate':<15} {'Predicted':<12} {'Over 0.5%':<12} {'Under 0.5%':<12} {'Recommendation':<15}")
        print("-"*120)
        
        # Sort by points finish rate descending
        points_data.sort(key=lambda x: x['points_rate'], reverse=True)
        
        for row in points_data:
            rec = row['recommendation']
            if rec != 'PASS':
                rec = f">>> {rec} <<<"
            print(f"{row['driver']:<25} {row['points_rate']*100:<15.1f}% {row['predicted']:<12.2f} "
                  f"{row['over_prob']*100:<12.1f}% {row['under_prob']*100:<12.1f}% {rec:<15}")
        
        # Print Starting Position Table
        print("\n" + "="*120)
        print("STARTING POSITION (Over/Under 10.5)")
        print("="*120)
        print(f"{'Driver':<25} {'Predicted':<10} {'Confidence Interval':<20} {'Over 10.5%':<12} {'Under 10.5%':<13} {'Recommendation':<15}")
        print("-"*120)
        
        # Sort by predicted position descending (higher position = worse starting spot)
        starting_pos_data.sort(key=lambda x: x['predicted'], reverse=True)
        
        for row in starting_pos_data:
            rec = row['recommendation']
            if rec != 'PASS':
                rec = f">>> {rec} <<<"
            ci_str = f"[{row['ci'][0]:.1f}, {row['ci'][1]:.1f}]"
            print(f"{row['driver']:<25} {row['predicted']:<10.1f} {ci_str:<20} "
                  f"{row['over_prob']*100:<12.1f}% {row['under_prob']*100:<13.1f}% {rec:<15}")
        
        # Print Pit Stops Table
        print("\n" + "="*120)
        print("PIT STOPS (Over/Under 2.5)")
        print("="*120)
        print(f"{'Driver':<25} {'Predicted':<10} {'Confidence Interval':<20} {'Over 2.5%':<12} {'Under 2.5%':<12} {'Recommendation':<15}")
        print("-"*120)
        
        # Sort by predicted pit stops descending
        pit_stops_data.sort(key=lambda x: x['predicted'], reverse=True)
        
        for row in pit_stops_data:
            rec = row['recommendation']
            if rec != 'PASS':
                rec = f">>> {rec} <<<"
            ci_str = f"[{row['ci'][0]:.2f}, {row['ci'][1]:.2f}]"
            print(f"{row['driver']:<25} {row['predicted']:<10.2f} {ci_str:<20} "
                  f"{row['over_prob']*100:<12.1f}% {row['under_prob']*100:<12.1f}% {rec:<15}")
        
        # Print Teammate Overtakes Table
        print("\n" + "="*120)
        print("TEAMMATE OVERTAKES - PrizePicks Scoring (Over/Under 0.5)")
        print("="*120)
        print(f"{'Driver':<25} {'Predicted':<12} {'Confidence Interval':<20} {'Over 0.5%':<12} {'Under 0.5%':<12} {'Recommendation':<15}")
        print("-"*120)
        
        # Sort by predicted score descending
        teammate_data.sort(key=lambda x: x['predicted'], reverse=True)
        
        for row in teammate_data:
            rec = row['recommendation']
            if rec != 'PASS':
                rec = f">>> {rec} <<<"
            ci_str = f"[{row['ci'][0]:.2f}, {row['ci'][1]:.2f}]"
            print(f"{row['driver']:<25} {row['predicted']:<12.2f} {ci_str:<20} "
                  f"{row['over_prob']*100:<12.1f}% {row['under_prob']*100:<12.1f}% {rec:<15}")
        
        # Print DNF Summary
        print("\n" + "="*100)
        print("DNF PROBABILITY SUMMARY")
        print("="*100)
        print(f"{'Driver':<25} {'DNF Probability':<20} {'Confidence':<15} {'Recommendation':<15}")
        print("-"*100)
        
        # Sort by DNF probability descending
        dnf_data.sort(key=lambda x: x['probability'], reverse=True)
        
        # If all DNF probabilities are similar, show summary
        unique_probs = set(d['probability'] for d in dnf_data)
        if len(unique_probs) == 1:
            avg_conf = sum(d['confidence'] for d in dnf_data) / len(dnf_data)
            rec = dnf_data[0]['recommendation']
            print(f"{'All Drivers':<25} {dnf_data[0]['probability']*100:<20.1f}% "
                  f"{min(d['confidence'] for d in dnf_data)*100:.1f}% - {max(d['confidence'] for d in dnf_data)*100:.1f}%{'':<5} "
                  f">>> {rec} <<<")
            print("\n* Note: All drivers have 0.0% predicted DNF probability with varying confidence levels.")
            print("* The model strongly recommends betting against DNF for all drivers.")
        else:
            for row in dnf_data[:10]:  # Show top 10 highest DNF risks
                rec = row['recommendation']
                if rec in ['HIGH_RISK', 'NO']:
                    rec = f">>> {rec} <<<"
                print(f"{row['driver']:<25} {row['probability']*100:<20.1f}% "
                      f"{row['confidence']*100:<15.1f}% {rec:<15}")
        
        # Print top betting edges
        print("\n" + "="*110)
        print("TOP BETTING EDGES (>15% Edge)")
        print("="*110)
        print(f"{'Driver':<25} {'Prop Type':<20} {'Direction':<10} {'Line':<8} {'Probability':<12} {'Edge':<10}")
        print("-"*110)
        
        edges = []
        for driver_id, data in predictions.items():
            for prop, pred in data['predictions'].items():
                if 'over_prob' in pred:
                    edge_over = pred['over_prob'] - 0.5
                    edge_under = pred['under_prob'] - 0.5
                    
                    if edge_over > 0.15 or edge_under > 0.15:
                        direction = 'OVER' if edge_over > edge_under else 'UNDER'
                        prob = pred['over_prob'] if direction == 'OVER' else pred['under_prob']
                        edge = max(edge_over, edge_under)
                        
                        edges.append({
                            'driver': data['driver_name'],
                            'prop': prop.replace('_', ' ').title(),
                            'direction': direction,
                            'line': pred.get('line', 'N/A'),
                            'probability': prob,
                            'edge': edge
                        })
        
        edges.sort(key=lambda x: x['edge'], reverse=True)
        
        for edge in edges[:20]:
            print(f"{edge['driver']:<25} {edge['prop']:<20} >>> {edge['direction']:<7} "
                  f"{str(edge['line']):<8} {edge['probability']*100:<12.1f}% +{edge['edge']*100:<9.1f}%")

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

if __name__ == "__main__":
    main()