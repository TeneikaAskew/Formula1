#!/usr/bin/env python3
"""
Generate Prize Picks Report with Pattern Analysis Insights

This script generates comprehensive Prize Picks reports including:
- Historical pattern analysis
- Portfolio optimization recommendations
- Risk assessment based on patterns
- Strategy recommendations
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import the pattern analyzer
from f1_prize_picks_insights import PrizePicksPatternAnalyzer, add_insights_to_prize_picks_report


def load_portfolio(portfolio_path: str = None) -> Dict:
    """Load portfolio from JSON file"""
    if portfolio_path is None:
        # Default to latest portfolio
        portfolio_path = Path(__file__).parent / 'pipeline_outputs' / 'portfolio_v4_production.json'
    
    with open(portfolio_path, 'r') as f:
        return json.load(f)


def generate_comprehensive_report(portfolio: Dict, output_path: str = None) -> str:
    """
    Generate comprehensive Prize Picks report with insights
    
    Args:
        portfolio: Portfolio dictionary with bets
        output_path: Optional path to save report
        
    Returns:
        Report string
    """
    analyzer = PrizePicksPatternAnalyzer()
    
    # Start with header
    report = []
    report.append("="*100)
    report.append("F1 PRIZE PICKS COMPREHENSIVE REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*100)
    
    # Add pattern analysis insights
    report.append(analyzer.format_insights_for_report())
    
    # Analyze current portfolio
    report.append("\n\n" + "="*80)
    report.append("CURRENT PORTFOLIO ANALYSIS")
    report.append("="*80)
    
    if 'bets' in portfolio:
        bets = portfolio['bets']
        
        # Portfolio summary
        total_stake = sum(bet.get('stake', 0) for bet in bets)
        total_expected_value = sum(bet.get('expected_value', 0) for bet in bets)
        expected_roi = ((total_expected_value - total_stake) / total_stake * 100) if total_stake > 0 else 0
        
        report.append(f"\n📊 Portfolio Summary:")
        report.append(f"  • Total Bets: {len(bets)}")
        report.append(f"  • Total Stake: ${total_stake:.2f}")
        report.append(f"  • Expected Value: ${total_expected_value:.2f}")
        report.append(f"  • Expected ROI: {expected_roi:.1f}%")
        
        # Analyze bet composition
        prop_types = {}
        drivers = {}
        lineup_sizes = {}
        
        for bet in bets:
            # Get lineup size
            bet_type = bet.get('type', '')
            if '-pick' in bet_type:
                size = int(bet_type.split('-')[0])
                lineup_sizes[size] = lineup_sizes.get(size, 0) + 1
            
            # Analyze selections
            for selection in bet.get('selections', []):
                prop = selection.get('prop', 'Unknown')
                driver = selection.get('driver', 'Unknown')
                
                # Map prop types to pattern analysis categories
                prop_mapping = {
                    'overtakes': 'Overtake Points',
                    'points': 'F1 Points',
                    'starting_position': 'Starting Position',
                    'pit_stop': '1st Pit Stop Time',
                    'sprint_points': 'F1 Sprint Points'
                }
                
                prop_category = prop_mapping.get(prop, prop)
                prop_types[prop_category] = prop_types.get(prop_category, 0) + 1
                drivers[driver] = drivers.get(driver, 0) + 1
        
        # Prop type analysis
        report.append("\n\n📈 Prop Type Distribution:")
        report.append("-" * 60)
        total_props = sum(prop_types.values())
        
        for prop_type, count in sorted(prop_types.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_props * 100) if total_props > 0 else 0
            rec = analyzer.get_prop_type_recommendation(prop_type)
            status = "✅" if rec['recommendation'] in ['HIGHLY RECOMMENDED', 'PROCEED WITH CAUTION'] else "❌"
            report.append(f"  {status} {prop_type}: {count} picks ({pct:.0f}%) - {rec['recommendation']}")
        
        # Driver analysis
        report.append("\n\n🏎️ Top Drivers in Portfolio:")
        report.append("-" * 60)
        for driver, count in sorted(drivers.items(), key=lambda x: x[1], reverse=True)[:8]:
            rec = analyzer.get_driver_recommendation(driver)
            status = "✅" if rec['recommendation'] in ['TARGET', 'SELECTIVE'] else "⚠️"
            report.append(f"  {status} {driver}: {count} picks - {rec['recommendation']}")
        
        # Lineup size analysis
        report.append("\n\n📊 Lineup Size Distribution:")
        report.append("-" * 60)
        for size, count in sorted(lineup_sizes.items()):
            if size in analyzer.lineup_size_performance:
                perf = analyzer.lineup_size_performance[size]
                report.append(f"  • {size}-Pick: {count} parlays (Historical: {perf['win_rate']:.0%} win rate, ${perf['total_roi']:+d} ROI)")
            else:
                report.append(f"  • {size}-Pick: {count} parlays")
        
        # Risk assessment
        report.append("\n\n⚠️ RISK ASSESSMENT:")
        report.append("-" * 60)
        
        risks = []
        recommendations = []
        
        # Check for risky prop types
        if prop_types.get('Overtake Points', 0) > 0:
            risks.append(f"Portfolio contains {prop_types['Overtake Points']} Overtake props (0% historical success)")
            recommendations.append("Remove all Overtake props")
        
        if prop_types.get('F1 Points', 0) > 0:
            risks.append(f"Portfolio contains {prop_types['F1 Points']} F1 Points props (0% historical success)")
            recommendations.append("Replace F1 Points props with Starting Position")
        
        # Check Starting Position percentage
        starting_pos_pct = (prop_types.get('Starting Position', 0) / total_props * 100) if total_props > 0 else 0
        if starting_pos_pct < 50:
            risks.append(f"Only {starting_pos_pct:.0f}% Starting Position props (recommend 80%+)")
            recommendations.append("Increase Starting Position props to 80% of portfolio")
        
        # Check lineup sizes
        large_parlays = sum(count for size, count in lineup_sizes.items() if size > 3)
        if large_parlays > 0:
            risks.append(f"{large_parlays} parlays with 4+ picks (higher variance)")
            recommendations.append("Focus on 2-3 pick parlays for better ROI")
        
        if risks:
            for risk in risks:
                report.append(f"  ❌ {risk}")
        else:
            report.append("  ✅ Portfolio aligns well with winning patterns")
        
        # Recommendations
        if recommendations:
            report.append("\n\n💡 RECOMMENDATIONS TO IMPROVE PORTFOLIO:")
            report.append("-" * 60)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")
        
        # Individual bet analysis
        report.append("\n\n" + "="*80)
        report.append("DETAILED BET ANALYSIS")
        report.append("="*80)
        
        for i, bet in enumerate(bets[:10], 1):  # Show first 10 bets
            report.append(f"\n{'='*60}")
            report.append(f"BET {i}: {bet.get('type', 'Unknown').upper()}")
            report.append(f"{'='*60}")
            report.append(f"Stake: ${bet.get('stake', 0):.2f}")
            report.append(f"Potential Payout: ${bet.get('stake', 0) * bet.get('payout', 1):.2f}")
            report.append(f"Win Probability: {bet.get('probability', 0):.1%}")
            report.append(f"Expected Value: ${bet.get('expected_value', 0):.2f}")
            
            report.append("\nSelections:")
            for j, selection in enumerate(bet.get('selections', []), 1):
                driver = selection.get('driver', 'Unknown')
                prop = selection.get('prop', 'Unknown')
                direction = selection.get('direction', '')
                line = selection.get('line', 0)
                prob = selection.get('probability', 0)
                
                # Get recommendation
                prop_category = {'overtakes': 'Overtake Points', 'points': 'F1 Points', 
                               'starting_position': 'Starting Position'}.get(prop, prop)
                prop_rec = analyzer.get_prop_type_recommendation(prop_category)
                driver_rec = analyzer.get_driver_recommendation(driver)
                
                status = "✅" if (prop_rec['recommendation'] != 'AVOID' and 
                                driver_rec['recommendation'] != 'AVOID') else "⚠️"
                
                report.append(f"  {j}. {status} {driver} - {prop} {direction} {line}")
                report.append(f"     Probability: {prob:.1%}")
                report.append(f"     Prop Rating: {prop_rec['recommendation']}")
                report.append(f"     Driver Rating: {driver_rec['recommendation']}")
    
    # Strategy summary
    report.append("\n\n" + "="*80)
    report.append("OPTIMAL STRATEGY SUMMARY")
    report.append("="*80)
    
    for strategy in analyzer._generate_strategy_recommendations():
        report.append(f"  ✅ {strategy}")
    
    # Next steps
    report.append("\n\n📋 Next Steps for Improvement:")
    report.append("-" * 60)
    for step in analyzer._generate_next_steps():
        report.append(f"  • {step}")
    
    # Footer
    report.append("\n\n" + "="*100)
    report.append("End of Report")
    report.append("="*100)
    
    full_report = "\n".join(report)
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report)
        print(f"Report saved to: {output_path}")
    
    return full_report


def main():
    """Main function to generate Prize Picks report"""
    # Load portfolio
    portfolio_path = Path(__file__).parent / 'pipeline_outputs' / 'portfolio_v4_production.json'
    
    if not portfolio_path.exists():
        print(f"Portfolio file not found: {portfolio_path}")
        print("Creating sample portfolio for demonstration...")
        
        # Create sample portfolio
        portfolio = {
            "bets": [
                {
                    "type": "2-pick",
                    "selections": [
                        {"driver": "Max Verstappen", "prop": "starting_position", "direction": "UNDER", "line": 2.5, "probability": 0.75},
                        {"driver": "Lando Norris", "prop": "starting_position", "direction": "UNDER", "line": 4.5, "probability": 0.65}
                    ],
                    "probability": 0.49,
                    "stake": 50,
                    "payout": 3.0,
                    "expected_value": 73.5
                },
                {
                    "type": "3-pick",
                    "selections": [
                        {"driver": "Charles Leclerc", "prop": "starting_position", "direction": "UNDER", "line": 5.5, "probability": 0.60},
                        {"driver": "George Russell", "prop": "overtakes", "direction": "OVER", "line": 2.5, "probability": 0.55},
                        {"driver": "Lewis Hamilton", "prop": "points", "direction": "OVER", "line": 8.5, "probability": 0.70}
                    ],
                    "probability": 0.23,
                    "stake": 40,
                    "payout": 6.0,
                    "expected_value": 55.2
                }
            ],
            "metadata": {
                "generated": datetime.now().isoformat(),
                "race": "Next Race",
                "bankroll": 1000
            }
        }
    else:
        with open(portfolio_path, 'r') as f:
            portfolio = json.load(f)
    
    # Generate report
    output_path = Path(__file__).parent / 'pipeline_outputs' / f'prize_picks_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    report = generate_comprehensive_report(portfolio, output_path)
    
    # Also print to console
    print(report)


if __name__ == "__main__":
    main()