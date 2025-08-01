#!/usr/bin/env python3
"""Simple F1 predictions with better error handling"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import F1DBDataLoader
from f1_performance_analysis import F1PerformanceAnalyzer

class F1SimplePredictions:
    """Simplified prediction generator with robust error handling"""
    
    def __init__(self):
        print("Initializing F1 Predictions...")
        self.loader = F1DBDataLoader()
        self.data = self.loader.get_core_datasets()
        self.analyzer = F1PerformanceAnalyzer(self.data)
        
    def safe_mean(self, values):
        """Calculate mean with safety checks"""
        if values is None or len(values) == 0:
            return 0
        return np.mean(values)
    
    def safe_probability(self, values, threshold):
        """Calculate probability with safety checks"""
        if values is None or len(values) == 0:
            return 0.5, 0.5
        
        over_count = sum(1 for v in values if v > threshold)
        over_prob = over_count / len(values)
        under_prob = 1 - over_prob
        
        return over_prob, under_prob
    
    def generate_predictions(self):
        """Generate predictions for all drivers"""
        print("\n" + "="*80)
        print("F1 PRIZEPICKS PREDICTIONS")
        print("="*80)
        
        # Get active drivers
        drivers = self.analyzer.get_active_drivers()
        races = self.data.get('races', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty:
            print("No active drivers found")
            return {}
        
        # Get next race
        future_races = races[races['date'] > datetime.now().strftime('%Y-%m-%d')]
        if future_races.empty:
            print("No future races found")
            return {}
            
        next_race = future_races.iloc[0]
        print(f"\nNext Race: {next_race.get('officialName', 'Unknown')}")
        print(f"Circuit: {next_race.get('circuitId', 'Unknown')}")
        
        predictions = {}
        
        # Process top 10 drivers only for testing
        for idx, (_, driver) in enumerate(drivers.head(10).iterrows()):
            driver_id = driver['id']
            driver_name = driver['name']
            
            print(f"\n{'='*60}")
            print(f"{idx+1}. {driver_name} (ID: {driver_id})")
            print(f"{'='*60}")
            
            driver_preds = {
                'driver_id': driver_id,
                'driver_name': driver_name,
                'predictions': {}
            }
            
            try:
                # 1. OVERTAKES - Based on position changes
                print("\nOVERTAKES:")
                driver_results = results[results['driverId'] == driver_id].tail(20)
                
                if not driver_results.empty:
                    # Simple calculation: count races where driver gained 3+ positions
                    grid_data = self.data.get('races_starting_grid_positions', pd.DataFrame())
                    
                    overtakes_list = []
                    for _, race in driver_results.iterrows():
                        race_id = race.get('raceId')
                        finish_pos = race.get('positionNumber')
                        
                        if pd.notna(finish_pos) and not grid_data.empty:
                            grid_pos = grid_data[(grid_data['raceId'] == race_id) & 
                                               (grid_data['driverId'] == driver_id)]
                            
                            if not grid_pos.empty:
                                start_pos = grid_pos.iloc[0].get('positionNumber', finish_pos)
                                if pd.notna(start_pos):
                                    overtakes = max(0, start_pos - finish_pos)
                                    overtakes_list.append(overtakes)
                    
                    if overtakes_list:
                        avg_overtakes = self.safe_mean(overtakes_list)
                        over_prob, under_prob = self.safe_probability(overtakes_list, 2.5)
                        
                        driver_preds['predictions']['overtakes'] = {
                            'avg': round(avg_overtakes, 2),
                            'line': 2.5,
                            'over_prob': round(over_prob, 3),
                            'under_prob': round(under_prob, 3),
                            'recommendation': 'OVER' if over_prob > 0.55 else ('UNDER' if under_prob > 0.55 else 'PASS')
                        }
                        
                        print(f"  Average: {avg_overtakes:.2f}")
                        print(f"  Over 2.5: {over_prob:.1%} | Under 2.5: {under_prob:.1%}")
                        print(f"  Recommendation: {driver_preds['predictions']['overtakes']['recommendation']}")
                    else:
                        print("  No data available")
                
                # 2. POINTS - Recent points scored
                print("\nPOINTS:")
                if 'points' in driver_results.columns:
                    recent_points = driver_results['points'].dropna().values
                    
                    if len(recent_points) > 0:
                        avg_points = self.safe_mean(recent_points)
                        over_prob, under_prob = self.safe_probability(recent_points, 0.5)
                        points_rate = sum(1 for p in recent_points if p > 0) / len(recent_points)
                        
                        driver_preds['predictions']['points'] = {
                            'avg': round(avg_points, 2),
                            'line': 0.5,
                            'over_prob': round(over_prob, 3),
                            'under_prob': round(under_prob, 3),
                            'points_finish_rate': round(points_rate, 3),
                            'recommendation': 'OVER' if over_prob > 0.6 else ('UNDER' if under_prob > 0.6 else 'PASS')
                        }
                        
                        print(f"  Average: {avg_points:.2f}")
                        print(f"  Points finish rate: {points_rate:.1%}")
                        print(f"  Over 0.5: {over_prob:.1%} | Under 0.5: {under_prob:.1%}")
                        print(f"  Recommendation: {driver_preds['predictions']['points']['recommendation']}")
                    else:
                        print("  No data available")
                
                # 3. QUALIFYING POSITION
                print("\nSTARTING POSITION:")
                quali_data = self.data.get('qualifying', pd.DataFrame())
                
                if not quali_data.empty:
                    driver_quali = quali_data[quali_data['driverId'] == driver_id].tail(10)
                    
                    if not driver_quali.empty:
                        # Try different column names
                        pos_col = None
                        for col in ['positionNumber', 'position', 'gridPosition']:
                            if col in driver_quali.columns:
                                pos_col = col
                                break
                        
                        if pos_col:
                            positions = driver_quali[pos_col].dropna().values
                            
                            if len(positions) > 0:
                                avg_pos = self.safe_mean(positions)
                                over_prob, under_prob = self.safe_probability(positions, 10.5)
                                
                                driver_preds['predictions']['starting_position'] = {
                                    'avg': round(avg_pos, 1),
                                    'line': 10.5,
                                    'over_prob': round(over_prob, 3),
                                    'under_prob': round(under_prob, 3),
                                    'recommendation': 'UNDER' if under_prob > 0.55 else ('OVER' if over_prob > 0.55 else 'PASS')
                                }
                                
                                print(f"  Average: {avg_pos:.1f}")
                                print(f"  Over 10.5: {over_prob:.1%} | Under 10.5: {under_prob:.1%}")
                                print(f"  Recommendation: {driver_preds['predictions']['starting_position']['recommendation']}")
                        else:
                            print("  No position data found")
                    else:
                        print("  No qualifying data for driver")
                else:
                    print("  No qualifying data available")
                
                # 4. DNF PROBABILITY
                print("\nDNF PROBABILITY:")
                if 'positionText' in driver_results.columns:
                    dnf_count = len(driver_results[driver_results['positionText'] == 'R'])
                    total_races = len(driver_results)
                    dnf_rate = dnf_count / total_races if total_races > 0 else 0.15
                    
                    driver_preds['predictions']['dnf'] = {
                        'probability': round(dnf_rate, 3),
                        'races_checked': total_races,
                        'dnf_count': dnf_count,
                        'recommendation': 'HIGH_RISK' if dnf_rate > 0.2 else 'LOW_RISK'
                    }
                    
                    print(f"  DNF Rate: {dnf_rate:.1%} ({dnf_count}/{total_races} races)")
                    print(f"  Risk Level: {driver_preds['predictions']['dnf']['recommendation']}")
                else:
                    print("  No DNF data available")
                
            except Exception as e:
                print(f"\nError processing driver {driver_name}: {str(e)}")
                continue
            
            predictions[driver_id] = driver_preds
        
        # Summary
        print("\n" + "="*80)
        print("BETTING RECOMMENDATIONS SUMMARY")
        print("="*80)
        
        strong_plays = []
        for driver_id, data in predictions.items():
            for prop_type, prop_data in data['predictions'].items():
                if 'over_prob' in prop_data:
                    # Calculate edge
                    edge_over = prop_data['over_prob'] - 0.5
                    edge_under = prop_data['under_prob'] - 0.5
                    
                    if edge_over > 0.1 or edge_under > 0.1:
                        direction = 'OVER' if edge_over > edge_under else 'UNDER'
                        prob = prop_data['over_prob'] if direction == 'OVER' else prop_data['under_prob']
                        edge = max(edge_over, edge_under)
                        
                        strong_plays.append({
                            'driver': data['driver_name'],
                            'prop': prop_type,
                            'direction': direction,
                            'probability': prob,
                            'edge': edge,
                            'line': prop_data.get('line', 'N/A')
                        })
        
        # Sort by edge
        strong_plays.sort(key=lambda x: x['edge'], reverse=True)
        
        print("\nTOP BETTING EDGES (>10%):")
        print(f"{'Driver':<25} {'Prop':<20} {'Direction':<10} {'Line':<8} {'Prob':<8} {'Edge':<8}")
        print("-" * 90)
        
        for play in strong_plays[:15]:
            print(f"{play['driver']:<25} {play['prop']:<20} {play['direction']:<10} "
                  f"{str(play['line']):<8} {play['probability']:.1%}    +{play['edge']*100:.1f}%")
        
        if not strong_plays:
            print("No strong plays found (need >10% edge)")
        
        return predictions

def main():
    """Run the prediction system"""
    try:
        predictor = F1SimplePredictions()
        predictions = predictor.generate_predictions()
        
        # Save results
        output_path = Path("pipeline_outputs/simple_predictions.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        print(f"\n\nPredictions saved to: {output_path}")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()