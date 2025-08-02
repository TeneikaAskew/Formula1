#!/usr/bin/env python3
"""Test script to generate and view the F1 risk dashboard"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1_predictions_enhanced_v4 import F1PredictionsV4
import matplotlib.pyplot as plt

def test_dashboard():
    """Generate predictions and display risk dashboard"""
    
    print("="*80)
    print("F1 RISK DASHBOARD TEST")
    print("="*80)
    
    # Create predictor with $1000 bankroll
    predictor = F1PredictionsV4(bankroll=1000)
    
    print("\nGenerating predictions and optimal portfolio...")
    print("This will take a moment as it runs multiple prediction models...\n")
    
    # Generate predictions
    predictions, portfolio = predictor.generate_ensemble_predictions()
    
    # Display the optimal bets
    predictor.display_optimal_bets(portfolio)
    
    # Generate risk analysis
    print("\n" + "="*80)
    print("GENERATING RISK ANALYSIS")
    print("="*80)
    
    # Calculate risk metrics
    risk_metrics = predictor.risk_dashboard.calculate_risk_metrics(portfolio)
    
    # Display text risk report
    risk_report = predictor.risk_dashboard.generate_risk_report()
    print(risk_report)
    
    # Create and display visual dashboard
    print("\nCreating visual risk dashboard...")
    predictor.risk_dashboard.create_dashboard(
        portfolio, 
        predictions,
        save_path="pipeline_outputs/risk_dashboard.png"
    )
    
    print("\n✅ Risk dashboard saved to: pipeline_outputs/risk_dashboard.png")
    print("\nTo view the dashboard:")
    print("1. The dashboard image has been saved to pipeline_outputs/risk_dashboard.png")
    print("2. You can open it with any image viewer")
    print("3. On Windows: start pipeline_outputs/risk_dashboard.png")
    print("4. On Mac: open pipeline_outputs/risk_dashboard.png")
    print("5. On Linux: xdg-open pipeline_outputs/risk_dashboard.png")
    
    # Also display using matplotlib if in interactive environment
    try:
        from IPython.display import Image, display
        display(Image("pipeline_outputs/risk_dashboard.png"))
    except:
        print("\nNote: To view the dashboard interactively, run this in a Jupyter notebook")
    
    return predictions, portfolio, risk_metrics

if __name__ == "__main__":
    # Run the test
    predictions, portfolio, metrics = test_dashboard()
    
    # Print summary of risk metrics
    print("\n" + "="*80)
    print("RISK METRICS SUMMARY")
    print("="*80)
    print(f"Total Exposure: ${metrics['total_exposure']:.2f} ({metrics['exposure_pct']:.1f}% of bankroll)")
    print(f"Expected ROI: {metrics['expected_roi']:.1f}%")
    print(f"Risk Score: {metrics['risk_score']:.0f}/100")
    print(f"Value at Risk (95%): ${metrics['value_at_risk']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")