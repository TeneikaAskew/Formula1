#!/usr/bin/env python3
"""Test the optimizer directly with the predictions"""

import json
import pandas as pd
from f1_ml.optimization import PrizePicksOptimizer

# Load the latest report
with open('pipeline_outputs/latest_report.json', 'r') as f:
    report = json.load(f)

# Load the config
with open('pipeline_config.json', 'r') as f:
    config = json.load(f)

# Create predictions dataframe
predictions_df = pd.DataFrame(report['predictions'])

# Initialize optimizer
optimizer = PrizePicksOptimizer(
    kelly_fraction=config['kelly_fraction'],
    max_correlation=config['max_correlation']
)

print("=== TESTING OPTIMIZER ===")
print(f"Predictions shape: {predictions_df.shape}")
print(f"Min edge: {config['min_edge']}")
print(f"Constraints: {config['constraints']}")

# Generate all picks
print("\n=== GENERATING PICKS ===")
all_picks = optimizer.generate_all_picks(predictions_df, min_edge=config['min_edge'])
print(f"Generated picks: {len(all_picks)}")

if not all_picks.empty:
    print("\nFirst 10 picks:")
    print(all_picks.head(10))
    
    # Try to optimize portfolio
    print("\n=== OPTIMIZING PORTFOLIO ===")
    portfolio = optimizer.optimize_portfolio(
        all_picks,
        bankroll=config['bankroll'],
        constraints=config['constraints']
    )
    
    print(f"\nPortfolio size: {len(portfolio)}")
    
    if portfolio:
        for i, parlay in enumerate(portfolio, 1):
            print(f"\nParlay {i}:")
            print(f"  N picks: {parlay['n_picks']}")
            print(f"  Expected value: {parlay['expected_value']:.3f}")
            print(f"  Bet size: ${parlay['bet_size']:.2f}")
            print(f"  Adjusted prob: {parlay['adjusted_prob']:.3f}")
            print(f"  Picks:")
            for _, pick in parlay['picks'].iterrows():
                print(f"    - {pick['driver']} {pick['bet_type']} (edge: {pick['edge']:.3f})")
else:
    print("No picks generated!")
    
# Debug the issue by checking step by step
print("\n=== DEBUGGING PICK GENERATION ===")
for i, pred in predictions_df.head(5).iterrows():
    print(f"\nDriver: {pred['driver']}")
    print(f"  top10_prob: {pred.get('top10_prob', 'N/A')}")
    print(f"  top5_prob: {pred.get('top5_prob', 'N/A')}")
    print(f"  top3_prob: {pred.get('top3_prob', 'N/A')}")
    print(f"  points_prob: {pred.get('points_prob', 'N/A')}")
    print(f"  beat_teammate_prob: {pred.get('beat_teammate_prob', 'N/A')}")

# Check what's in the optimizer's generate_all_picks method
print("\n=== CHECKING PRIZEPICKS LINES ===")
print("The issue is that the optimizer needs actual PrizePicks lines to compare against.")
print("Currently, it's looking for 'implied_prob' fields that don't exist in predictions.")

# Simulate some prize picks lines
print("\n=== SIMULATING PRIZEPICKS LINES ===")
sample_lines = {
    'points_finish': {'Verstappen': 0.7, 'Hamilton': 0.65, 'Norris': 0.6},
    'top5_finish': {'Verstappen': 0.6, 'Hamilton': 0.5, 'Norris': 0.45},
    'overtakes_ou': {'Verstappen': 2.5, 'Hamilton': 3.5, 'Norris': 4.5}
}

print("\nTo fix this, we need to:")
print("1. Add prop-specific predictions (overtakes, pit stops, fastest lap, etc.)")
print("2. Load or generate PrizePicks implied probabilities")
print("3. Calculate edges = our_probability - implied_probability")
print("4. Only generate picks where edge > min_edge")