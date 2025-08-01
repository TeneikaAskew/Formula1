#!/usr/bin/env python3
"""Debug why betting recommendations aren't being generated"""

import json
import pandas as pd
from pathlib import Path

# Load the latest report
with open('pipeline_outputs/latest_report.json', 'r') as f:
    report = json.load(f)

# Load the config
with open('pipeline_config.json', 'r') as f:
    config = json.load(f)

print("=== CONFIGURATION ===")
print(f"Min Edge Required: {config['min_edge']} ({config['min_edge']*100}%)")
print(f"Min Avg Edge (constraint): {config['constraints']['min_avg_edge']} ({config['constraints']['min_avg_edge']*100}%)")
print(f"Kelly Fraction: {config['kelly_fraction']}")

print("\n=== PREDICTIONS ANALYSIS ===")
predictions = pd.DataFrame(report['predictions'])
print(f"Total predictions: {len(predictions)}")

# Analyze each prediction to see what edges would be calculated
print("\n=== EDGE ANALYSIS ===")
print("Checking edges for each bet type (true_prob - implied_prob):")

# The implied probabilities used in generate_all_picks
implied_probs = {
    'top_10': 0.5,
    'top_5': 0.3,
    'top_3': 0.15,
    'points': 0.5,
    'beat_teammate': 0.5
}

edges_found = []
for _, pred in predictions.iterrows():
    driver = pred['driver']
    
    # Check each bet type
    bet_opportunities = [
        ('top_10', pred.get('top10_prob', 0.5), implied_probs['top_10']),
        ('top_5', pred.get('top5_prob', 0.3), implied_probs['top_5']),
        ('top_3', pred.get('top3_prob', 0.15), implied_probs['top_3']),
        ('points', pred.get('points_prob', 0.5), implied_probs['points']),
        ('beat_teammate', pred.get('beat_teammate_prob', 0.5), implied_probs['beat_teammate'])
    ]
    
    for bet_type, true_prob, implied_prob in bet_opportunities:
        edge = true_prob - implied_prob
        
        if edge > 0:
            edges_found.append({
                'driver': driver,
                'bet_type': bet_type,
                'true_prob': true_prob,
                'implied_prob': implied_prob,
                'edge': edge,
                'meets_min_edge': edge >= config['min_edge']
            })

edges_df = pd.DataFrame(edges_found)

print(f"\nTotal positive edges found: {len(edges_df)}")
print(f"Edges meeting min_edge ({config['min_edge']}): {edges_df['meets_min_edge'].sum()}")

# Show top edges
print("\n=== TOP 10 EDGES ===")
if not edges_df.empty:
    top_edges = edges_df.nlargest(10, 'edge')
    for _, edge in top_edges.iterrows():
        status = "✓" if edge['meets_min_edge'] else "✗"
        print(f"{status} {edge['driver']:15} {edge['bet_type']:15} Edge: {edge['edge']:.3f} ({edge['edge']*100:.1f}%) True: {edge['true_prob']:.3f} Implied: {edge['implied_prob']:.3f}")

# Analyze why beat_teammate is always 0.5
print("\n=== BEAT TEAMMATE ANALYSIS ===")
print("All predictions have beat_teammate_prob = 0.5 (no edge possible)")
print("This needs to be calculated based on historical teammate performance")

# Show what would happen with lower thresholds
print("\n=== THRESHOLD ANALYSIS ===")
for threshold in [0.05, 0.04, 0.03, 0.02, 0.01]:
    qualifying = edges_df[edges_df['edge'] >= threshold]
    print(f"Min edge {threshold:.2f} ({threshold*100}%): {len(qualifying)} picks qualify")

# Check if any combinations would meet the min_avg_edge constraint
print("\n=== PARLAY CONSTRAINT ANALYSIS ===")
print(f"Constraint: min_avg_edge = {config['constraints']['min_avg_edge']}")
if not edges_df.empty:
    qualifying_edges = edges_df[edges_df['meets_min_edge']]
    if not qualifying_edges.empty:
        # Group by average edge
        for n_picks in [2, 3, 4, 5, 6]:
            print(f"\n{n_picks}-pick parlays:")
            # Get top n picks by edge
            top_n = qualifying_edges.nlargest(n_picks, 'edge')
            if len(top_n) >= n_picks:
                avg_edge = top_n['edge'].mean()
                meets_constraint = avg_edge >= config['constraints']['min_avg_edge']
                status = "✓" if meets_constraint else "✗"
                print(f"  {status} Best possible avg edge: {avg_edge:.3f} ({avg_edge*100:.1f}%)")
                if meets_constraint:
                    print(f"  Drivers: {', '.join(top_n['driver'].tolist())}")