#!/usr/bin/env python3
"""Example of how to create and view the risk dashboard with sample data"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1_risk_dashboard import F1RiskDashboard
import matplotlib.pyplot as plt

def create_sample_portfolio():
    """Create a sample portfolio for demonstration"""
    portfolio = {
        'bets': [
            {
                'type': '3-pick',
                'stake': 50.0,
                'probability': 0.65,
                'payout': 6.0,
                'expected_value': 45.0,
                'selections': [
                    {'driver': 'Max Verstappen', 'prop': 'points', 'direction': 'OVER', 'line': 6.0, 'probability': 0.85},
                    {'driver': 'Lewis Hamilton', 'prop': 'overtakes', 'direction': 'OVER', 'line': 2.5, 'probability': 0.75},
                    {'driver': 'Charles Leclerc', 'prop': 'dnf', 'direction': 'NO', 'line': 0.5, 'probability': 0.88}
                ]
            },
            {
                'type': '2-pick',
                'stake': 75.0,
                'probability': 0.72,
                'payout': 3.0,
                'expected_value': 37.0,
                'selections': [
                    {'driver': 'Fernando Alonso', 'prop': 'points', 'direction': 'OVER', 'line': 6.0, 'probability': 0.80},
                    {'driver': 'Lando Norris', 'prop': 'starting_position', 'direction': 'UNDER', 'line': 10.5, 'probability': 0.90}
                ]
            },
            {
                'type': '4-pick',
                'stake': 25.0,
                'probability': 0.45,
                'payout': 10.0,
                'expected_value': 87.5,
                'selections': [
                    {'driver': 'George Russell', 'prop': 'points', 'direction': 'OVER', 'line': 6.0, 'probability': 0.85},
                    {'driver': 'Carlos Sainz', 'prop': 'pit_stops', 'direction': 'UNDER', 'line': 2.5, 'probability': 0.70},
                    {'driver': 'Sergio Perez', 'prop': 'overtakes', 'direction': 'OVER', 'line': 2.5, 'probability': 0.75},
                    {'driver': 'Oscar Piastri', 'prop': 'teammate_overtakes', 'direction': 'OVER', 'line': 0.5, 'probability': 0.80}
                ]
            }
        ],
        'total_stake': 150.0,
        'expected_value': 169.5,
        'risk_metrics': {
            'total_exposure': 150.0,
            'exposure_pct': 15.0,
            'expected_roi': 13.0,
            'num_bets': 3
        }
    }
    
    return portfolio

def main():
    """Create and display the risk dashboard"""
    
    print("="*80)
    print("F1 RISK DASHBOARD VIEWER")
    print("="*80)
    
    # Create risk dashboard with $1000 bankroll
    dashboard = F1RiskDashboard(bankroll=1000)
    
    # Create sample portfolio
    portfolio = create_sample_portfolio()
    
    # Calculate risk metrics
    print("\nCalculating risk metrics...")
    metrics = dashboard.calculate_risk_metrics(portfolio)
    
    # Display text risk report
    print("\n" + "="*80)
    print("RISK REPORT")
    print("="*80)
    report = dashboard.generate_risk_report()
    print(report)
    
    # Create visual dashboard
    print("\nGenerating visual dashboard...")
    
    # Create the dashboard (it will display automatically if not in notebook)
    dashboard.create_dashboard(
        portfolio, 
        {},  # Empty predictions dict for this example
        save_path="risk_dashboard_example.png"
    )
    
    print("\n✅ Dashboard saved to: risk_dashboard_example.png")
    
    # If you want to display it programmatically:
    # Option 1: Using matplotlib (already shown by create_dashboard)
    # Option 2: Using PIL/Pillow
    try:
        from PIL import Image
        img = Image.open("risk_dashboard_example.png")
        img.show()  # This will open your default image viewer
    except ImportError:
        print("\nInstall Pillow to automatically open the image: pip install Pillow")
    except Exception as e:
        print(f"\nCould not automatically open image: {e}")
    
    # Option 3: Display in Jupyter
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage("risk_dashboard_example.png"))
    except:
        pass
    
    return metrics

if __name__ == "__main__":
    metrics = main()
    
    # Print key metrics
    print("\n" + "="*80)
    print("KEY RISK METRICS")
    print("="*80)
    print(f"Risk Score: {metrics['risk_score']:.0f}/100")
    print(f"Value at Risk (95%): ${metrics['value_at_risk']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Expected ROI: {metrics['expected_roi']:.1f}%")