#!/usr/bin/env python3
"""Minimal working version of v4 predictions"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json
from pathlib import Path
from f1db_data_loader import F1DBDataLoader

warnings.filterwarnings('ignore')

class F1PredictionsV4Minimal:
    """Minimal F1 predictions without complex dependencies"""
    
    def __init__(self, bankroll=1000):
        self.bankroll = bankroll
        self.loader = F1DBDataLoader()
        self.data = self.loader.get_core_datasets()
        
    def get_driver_recent_races(self, driver_name, num_races=20):
        """Get recent race results for a driver"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        # Find driver ID
        driver_match = drivers[drivers['fullName'] == driver_name]
        if driver_match.empty:
            return pd.DataFrame()
            
        driver_id = driver_match['id'].iloc[0]
        
        # Get recent results
        driver_results = results[results['driverId'] == driver_id].sort_values('raceId', ascending=False).head(num_races)
        
        # Add positions gained calculation if grid data exists
        if 'grid' in driver_results.columns:
            driver_results['positions_gained'] = driver_results['grid'] - driver_results['positionNumber']
            driver_results['positions_gained'] = driver_results['positions_gained'].fillna(0)
        else:
            driver_results['positions_gained'] = 0
            
        return driver_results
    
    def calculate_simple_probability(self, driver_name, prop_type='overtakes', line=3.0):
        """Calculate simple probability for a prop"""
        recent_races = self.get_driver_recent_races(driver_name)
        
        if recent_races.empty:
            return 0.5, 0
            
        if prop_type == 'overtakes':
            over_count = (recent_races['positions_gained'] > line).sum()
        elif prop_type == 'points':
            over_count = (recent_races['points'] > line).sum()
        elif prop_type == 'dnf':
            dnf_indicators = ['DNF', 'DNS', 'DSQ', 'EX', 'NC', 'WD']
            over_count = recent_races['positionText'].isin(dnf_indicators).sum()
        else:
            return 0.5, 0
            
        total_races = len(recent_races)
        prob = over_count / total_races if total_races > 0 else 0.5
        
        # Simple bounds
        prob = max(0.05, min(0.95, prob))
        
        return prob, total_races
    
    def get_current_drivers(self):
        """Get active drivers"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        # Get drivers from recent races
        recent_results = results[results['year'] >= 2023]
        active_driver_ids = recent_results['driverId'].unique()
        
        active_drivers = drivers[drivers['id'].isin(active_driver_ids)]
        driver_names = active_drivers['fullName'].dropna().unique()
        
        return sorted(driver_names)
    
    def generate_predictions(self):
        """Generate simple predictions"""
        print("\n" + "="*80)
        print("F1 PRIZEPICKS PREDICTIONS - MINIMAL VERSION")
        print("="*80)
        
        drivers = self.get_current_drivers()
        print(f"\nAnalyzing {len(drivers)} drivers...")
        
        # Lines
        lines = {
            'overtakes': 3.0,
            'points': 6.0,
            'dnf': 0.5
        }
        
        all_predictions = []
        
        # Generate predictions for each prop
        for prop_type, line in lines.items():
            print(f"\n{prop_type.upper()} PREDICTIONS (Line: {line})")
            print("-" * 60)
            
            predictions = []
            for driver in drivers:
                prob, sample = self.calculate_simple_probability(driver, prop_type, line)
                predictions.append({
                    'driver': driver,
                    'prop': prop_type,
                    'line': line,
                    'over_prob': prob,
                    'under_prob': 1 - prob,
                    'sample_size': sample
                })
                all_predictions.append(predictions[-1])
            
            # Sort and display top 10
            predictions.sort(key=lambda x: x['over_prob'], reverse=True)
            for pred in predictions[:10]:
                print(f"{pred['driver']:<30} OVER: {pred['over_prob']*100:>5.1f}% (n={pred['sample_size']})")
        
        # Create simple portfolio
        portfolio = self.create_simple_portfolio(all_predictions)
        self.display_portfolio(portfolio)
        
        return all_predictions, portfolio
    
    def create_simple_portfolio(self, predictions):
        """Create a simple betting portfolio"""
        # Find best bets (confidence > 70%)
        high_confidence = [p for p in predictions if p['over_prob'] > 0.7 or p['under_prob'] > 0.7]
        
        # Sort by confidence
        high_confidence.sort(key=lambda x: max(x['over_prob'], x['under_prob']), reverse=True)
        
        # Create 2-pick and 3-pick parlays
        parlays = []
        
        if len(high_confidence) >= 2:
            # Best 2-pick
            selections = []
            for bet in high_confidence[:2]:
                direction = 'OVER' if bet['over_prob'] > bet['under_prob'] else 'UNDER'
                prob = max(bet['over_prob'], bet['under_prob'])
                selections.append({
                    'driver': bet['driver'],
                    'prop': bet['prop'],
                    'direction': direction,
                    'line': bet['line'],
                    'probability': prob
                })
            
            parlay_prob = selections[0]['probability'] * selections[1]['probability']
            stake = min(50, self.bankroll * 0.05)  # 5% of bankroll max
            
            parlays.append({
                'type': '2-pick',
                'selections': selections,
                'probability': parlay_prob,
                'stake': stake,
                'payout': 3.0,  # Standard 2-pick payout
                'expected_value': stake * (parlay_prob * 3.0 - 1)
            })
        
        if len(high_confidence) >= 3:
            # Best 3-pick
            selections = []
            for bet in high_confidence[:3]:
                direction = 'OVER' if bet['over_prob'] > bet['under_prob'] else 'UNDER'
                prob = max(bet['over_prob'], bet['under_prob'])
                selections.append({
                    'driver': bet['driver'],
                    'prop': bet['prop'],
                    'direction': direction,
                    'line': bet['line'],
                    'probability': prob
                })
            
            parlay_prob = selections[0]['probability'] * selections[1]['probability'] * selections[2]['probability']
            stake = min(25, self.bankroll * 0.025)  # 2.5% of bankroll max
            
            parlays.append({
                'type': '3-pick',
                'selections': selections,
                'probability': parlay_prob,
                'stake': stake,
                'payout': 6.0,  # Standard 3-pick payout
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
    
    def display_portfolio(self, portfolio):
        """Display the betting portfolio"""
        print(f"\n{'='*80}")
        print("OPTIMAL BETTING PORTFOLIO")
        print(f"{'='*80}")
        print(f"Bankroll: ${self.bankroll:.2f}")
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
            
            for selection in bet['selections']:
                print(f"    - {selection['driver']} {selection['prop']} "
                      f"{selection['direction']} {selection['line']} "
                      f"({selection['probability']*100:.1f}%)")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 PrizePicks Predictions V4 Minimal')
    parser.add_argument('--bankroll', type=float, default=500, 
                        help='Betting bankroll (default: 500)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = F1PredictionsV4Minimal(args.bankroll)
    
    # Generate predictions
    predictions, portfolio = predictor.generate_predictions()
    
    # Save results
    output_dir = Path("pipeline_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save portfolio
    with open(output_dir / "portfolio_minimal.json", 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)
        
    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    main()