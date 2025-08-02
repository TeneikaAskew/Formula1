#!/usr/bin/env python3
"""Enhanced F1 predictions v4 with ensemble methods and Kelly optimization"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1_predictions_enhanced_v3 import F1PrizePicksPredictor
from f1_ensemble_integration import F1PredictionEnsemble, F1OptimalBetting
from f1_correlation_analysis import F1CorrelationAnalyzer
from f1_risk_dashboard import F1RiskDashboard
import io
import json
from pathlib import Path
from typing import Dict, Optional


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
            predictor1 = F1PrizePicksPredictor()
            pred1 = predictor1.generate_all_predictions(race_id)
            predictions_list.append(pred1)
            
            # Conservative prediction (higher confidence threshold)
            print("\nGenerating conservative predictions...")
            predictor2 = F1PrizePicksPredictor()
            predictor2.calibration_enabled = True
            pred2 = predictor2.generate_all_predictions(race_id)
            predictions_list.append(pred2)
            
            # Aggressive prediction (lower threshold)
            print("\nGenerating aggressive predictions...")
            predictor3 = F1PrizePicksPredictor()
            predictor3.kelly_fraction = 0.5  # More aggressive
            pred3 = predictor3.generate_all_predictions(race_id)
            predictions_list.append(pred3)
            
        finally:
            sys.stdin = old_stdin
        
        # Combine predictions using ensemble
        print("\nCombining predictions with ensemble methods...")
        ensemble_predictions = self.ensemble.combine_predictions(
            predictions_list, 
            method='weighted_average'
        )
        
        # Initialize correlation analyzer if needed
        if self.correlation_analyzer is None and predictor1:
            self.correlation_analyzer = F1CorrelationAnalyzer(predictor1.data)
        
        # Optimize betting portfolio
        print("\nOptimizing betting portfolio...")
        optimal_portfolio = self.optimizer.optimize_bets(
            ensemble_predictions,
            max_exposure=0.25  # Max 25% of bankroll
        )
        
        # Apply correlation analysis to improve diversification
        if optimal_portfolio.get('bets'):
            print("\nAnalyzing correlations...")
            self._apply_correlation_analysis(optimal_portfolio)
        
        return ensemble_predictions, optimal_portfolio
    
    def _apply_correlation_analysis(self, portfolio: Dict):
        """Apply correlation analysis to portfolio"""
        if not self.correlation_analyzer or not portfolio.get('bets'):
            return
            
        # Calculate correlation scores for each parlay
        for bet in portfolio['bets']:
            if 'selections' in bet:
                # Convert selections to format for correlation analyzer
                bet_list = []
                for sel in bet['selections']:
                    bet_list.append({
                        'driver': sel.get('driver', 'Unknown'),
                        'prop_type': sel.get('prop', 'unknown'),
                        'direction': sel.get('direction', 'OVER')
                    })
                
                # Calculate correlation
                avg_correlation = self.correlation_analyzer.calculate_parlay_correlation(bet_list)
                bet['correlation_score'] = round(avg_correlation, 3)
                bet['diversification_score'] = round(1 - avg_correlation, 3)
    
    def display_optimal_bets(self, portfolio: Dict):
        """Display optimized betting recommendations"""
        print("\n" + "="*80)
        print("OPTIMAL BETTING PORTFOLIO (Kelly Criterion)")
        print("="*80)
        print(f"Bankroll: ${self.bankroll}")
        print(f"Total Exposure: ${portfolio['risk_metrics']['total_exposure']} ({portfolio['risk_metrics']['exposure_pct']}%)")
        print(f"Expected ROI: {portfolio['risk_metrics']['expected_roi']}%")
        print(f"Number of Bets: {portfolio['risk_metrics']['num_bets']}")
        
        print("\n" + "-"*80)
        print("RECOMMENDED BETS:")
        print("-"*80)
        
        for bet in portfolio['bets']:
            print(f"\n{bet['type'].upper()} PARLAY")
            print(f"Stake: ${bet['stake']}")
            print(f"Win Probability: {bet['probability']*100:.1f}%")
            print(f"Payout: {bet['payout']}x")
            print(f"Expected Value: ${bet['expected_value']}")
            print("\nSelections:")
            
            for i, selection in enumerate(bet['selections'], 1):
                print(f"  {i}. {selection['driver']} - {selection['prop'].upper()} {selection['direction']}")
                print(f"     Line: {selection['line']} | Probability: {selection['probability']*100:.1f}%")
            
            # Add correlation info if available
            if 'diversification_score' in bet:
                print(f"\nDiversification Score: {bet['diversification_score']:.2f} (Higher is better)")
    
    def save_predictions(self, predictions: Dict, portfolio: Dict, 
                        output_dir: str = "pipeline_outputs"):
        """Save predictions and portfolio to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save ensemble predictions
        with open(output_path / "ensemble_predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        # Save optimal portfolio
        with open(output_path / "optimal_portfolio.json", 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)
        
        print(f"\nPredictions saved to {output_path}")


def main():
    """Run the complete enhanced prediction system"""
    print("F1 ENHANCED PREDICTIONS V4 - With Ensemble Methods & Kelly Optimization")
    print("="*80)
    
    # Create predictor
    predictor = F1PredictionsV4(bankroll=1000)
    
    # Generate predictions
    predictions, portfolio = predictor.generate_ensemble_predictions()
    
    # Display optimal bets
    predictor.display_optimal_bets(portfolio)
    
    # Generate risk analysis
    print("\nGenerating risk analysis...")
    risk_metrics = predictor.risk_dashboard.calculate_risk_metrics(portfolio)
    
    # Display risk report
    print("\n" + "="*80)
    print("RISK ANALYSIS")
    print("="*80)
    risk_report = predictor.risk_dashboard.generate_risk_report()
    print(risk_report)
    
    # Create visual dashboard
    print("\nCreating risk dashboard visualization...")
    predictor.risk_dashboard.create_dashboard(
        portfolio, 
        predictions,
        save_path="pipeline_outputs/risk_dashboard.png"
    )
    
    # Save results
    predictor.save_predictions(predictions, portfolio)
    
    return predictions, portfolio


if __name__ == "__main__":
    predictions, portfolio = main()