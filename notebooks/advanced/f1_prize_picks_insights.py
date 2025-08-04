#!/usr/bin/env python3
"""
F1 Prize Picks Pattern Analysis and Insights

This module provides pattern analysis and summary generation for Prize Picks
based on historical performance data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json


class PrizePicksPatternAnalyzer:
    """Analyzes patterns in Prize Picks historical performance"""
    
    def __init__(self):
        # Historical pattern data based on provided insights
        self.prop_type_performance = {
            'Starting Position': {'picks': 7, 'hits': 7, 'success_rate': 1.00, 'roi_impact': 210},
            '1st Pit Stop Time': {'picks': 9, 'hits': 5, 'success_rate': 0.56, 'roi_impact': 8},
            'F1 Points': {'picks': 9, 'hits': 0, 'success_rate': 0.00, 'roi_impact': -60},
            'Overtake Points': {'picks': 7, 'hits': 0, 'success_rate': 0.00, 'roi_impact': -40},
            'F1 Sprint Points': {'picks': 5, 'hits': 0, 'success_rate': 0.00, 'roi_impact': 0}
        }
        
        self.lineup_size_performance = {
            2: {'lineups': 3, 'wins': 1, 'win_rate': 0.33, 'total_roi': 30},
            3: {'lineups': 1, 'wins': 0, 'win_rate': 0.00, 'total_roi': -20},
            4: {'lineups': 3, 'wins': 0, 'win_rate': 0.00, 'total_roi': -40},
            5: {'lineups': 6, 'wins': 2, 'win_rate': 0.33, 'total_roi': 168}
        }
        
        self.driver_performance = {
            'Lando Norris': {'picks': 6, 'hits': 3, 'success_rate': 0.50, 'best_prop': 'Starting Position (2/2)'},
            'Max Verstappen': {'picks': 4, 'hits': 3, 'success_rate': 0.75, 'best_prop': 'Starting Position (2/2)'},
            'Lewis Hamilton': {'picks': 4, 'hits': 1, 'success_rate': 0.25, 'best_prop': 'Starting Position (1/1)'},
            'George Russell': {'picks': 4, 'hits': 2, 'success_rate': 0.50, 'best_prop': 'Starting Position (2/2)'},
            'Charles Leclerc': {'picks': 3, 'hits': 2, 'success_rate': 0.67, 'best_prop': 'Starting Position (2/2)'},
            'Andrea Kimi Antonelli': {'picks': 2, 'hits': 0, 'success_rate': 0.00, 'best_prop': 'Avoid overtakes'},
            'Lance Stroll': {'picks': 2, 'hits': 0, 'success_rate': 0.00, 'best_prop': 'Avoid overtakes'},
            'Fernando Alonso': {'picks': 2, 'hits': 0, 'success_rate': 0.00, 'best_prop': 'Avoid overtakes'}
        }
        
        self.team_performance = {
            'Red Bull': {'picks': 4, 'hits': 3, 'success_rate': 0.75},
            'McLaren': {'picks': 6, 'hits': 3, 'success_rate': 0.50},
            'Ferrari': {'picks': 3, 'hits': 2, 'success_rate': 0.67},
            'Mercedes': {'picks': 8, 'hits': 3, 'success_rate': 0.38},
            'Williams': {'picks': 2, 'hits': 1, 'success_rate': 0.50},
            'Aston Martin': {'picks': 4, 'hits': 0, 'success_rate': 0.00}
        }
    
    def generate_pattern_summary(self) -> Dict:
        """Generate comprehensive pattern analysis summary"""
        summary = {
            'prop_type_analysis': self._analyze_prop_types(),
            'lineup_size_analysis': self._analyze_lineup_sizes(),
            'driver_analysis': self._analyze_drivers(),
            'team_analysis': self._analyze_teams(),
            'key_insights': self._generate_key_insights(),
            'recommended_strategy': self._generate_strategy_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        return summary
    
    def _analyze_prop_types(self) -> Dict:
        """Analyze prop type performance"""
        analysis = []
        for prop_type, data in self.prop_type_performance.items():
            analysis.append({
                'prop_type': prop_type,
                'total_picks': data['picks'],
                'hits': data['hits'],
                'success_rate': f"{data['success_rate']:.0%}",
                'roi_impact': f"${data['roi_impact']:+d}" if data['roi_impact'] != 0 else "$0 (free plays)"
            })
        
        # Sort by success rate descending
        analysis.sort(key=lambda x: self.prop_type_performance[x['prop_type']]['success_rate'], reverse=True)
        return {'details': analysis, 'summary': 'Starting Position props dominate with 100% success rate'}
    
    def _analyze_lineup_sizes(self) -> Dict:
        """Analyze lineup size performance"""
        analysis = []
        for size, data in self.lineup_size_performance.items():
            analysis.append({
                'lineup_size': f"{size}-Pick",
                'lineups': data['lineups'],
                'wins': data['wins'],
                'win_rate': f"{data['win_rate']:.0%}",
                'total_roi': f"${data['total_roi']:+d}"
            })
        
        return {
            'details': analysis,
            'summary': '2-Pick lineups show best ROI despite lower win rate'
        }
    
    def _analyze_drivers(self) -> Dict:
        """Analyze driver performance"""
        analysis = []
        for driver, data in self.driver_performance.items():
            analysis.append({
                'driver': driver,
                'total_picks': data['picks'],
                'hits': data['hits'],
                'success_rate': f"{data['success_rate']:.0%}",
                'best_prop_type': data['best_prop']
            })
        
        # Sort by success rate descending
        analysis.sort(key=lambda x: self.driver_performance[x['driver']]['success_rate'], reverse=True)
        return {
            'details': analysis[:8],  # Top 8 drivers
            'summary': 'Max Verstappen leads with 75% success rate'
        }
    
    def _analyze_teams(self) -> Dict:
        """Analyze team performance"""
        analysis = []
        for team, data in self.team_performance.items():
            analysis.append({
                'team': team,
                'total_picks': data['picks'],
                'hits': data['hits'],
                'success_rate': f"{data['success_rate']:.0%}"
            })
        
        # Sort by success rate descending
        analysis.sort(key=lambda x: self.team_performance[x['team']]['success_rate'], reverse=True)
        return {
            'details': analysis,
            'summary': 'Red Bull leads team performance at 75% success rate'
        }
    
    def _generate_key_insights(self) -> Dict:
        """Generate key insights from pattern analysis"""
        return {
            'winning_patterns': [
                'Starting Position props = 100% success rate (7/7)',
                '2-pick lineups with starting position = profitable',
                'Conservative pit stop time overs = 56% success',
                'Red Bull/Ferrari drivers for starting position'
            ],
            'losing_patterns': [
                'Overtake props = 0% success rate (0/7)',
                'F1 Points props = 0% success rate (0/9)',
                'Mercedes drivers for overtakes = avoid',
                'Aston Martin drivers = avoid completely',
                '4+ pick lineups = higher variance'
            ]
        }
    
    def _generate_strategy_recommendations(self) -> List[str]:
        """Generate recommended strategy based on patterns"""
        return [
            'Focus 80% on Starting Position props',
            'Use 2-3 pick lineups maximum',
            'Target Red Bull, Ferrari, McLaren drivers',
            'Completely avoid Overtake and F1 Points props',
            'Research qualifying pace on Saturday'
        ]
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps for improvement"""
        return [
            'Track timing of successful picks (practice vs qualifying analysis)',
            'Note weather conditions for each race',
            'Add confidence ratings (1-5) for each pick',
            'Track research time spent per lineup',
            'Monitor which qualifying positions translate to wins'
        ]
    
    def format_insights_for_report(self) -> str:
        """Format insights for inclusion in Prize Picks report"""
        summary = self.generate_pattern_summary()
        
        report = []
        report.append("\n" + "="*80)
        report.append("PRIZE PICKS PATTERN ANALYSIS")
        report.append("="*80)
        
        # Prop Type Performance
        report.append("\n📊 Prop Type Performance:")
        report.append("-" * 60)
        report.append(f"{'Prop Type':<20} {'Picks':>8} {'Hits':>8} {'Success':>10} {'ROI Impact':>15}")
        report.append("-" * 60)
        
        for prop in summary['prop_type_analysis']['details']:
            report.append(f"{prop['prop_type']:<20} {prop['total_picks']:>8} {prop['hits']:>8} "
                         f"{prop['success_rate']:>10} {prop['roi_impact']:>15}")
        
        # Lineup Size Performance
        report.append("\n\n📈 Lineup Size Performance:")
        report.append("-" * 60)
        report.append(f"{'Size':<10} {'Lineups':>10} {'Wins':>8} {'Win Rate':>12} {'Total ROI':>15}")
        report.append("-" * 60)
        
        for lineup in summary['lineup_size_analysis']['details']:
            report.append(f"{lineup['lineup_size']:<10} {lineup['lineups']:>10} {lineup['wins']:>8} "
                         f"{lineup['win_rate']:>12} {lineup['total_roi']:>15}")
        
        # Key Insights
        report.append("\n\n🔑 Key Insights:")
        report.append("-" * 60)
        report.append("\n🟢 WINNING PATTERNS:")
        for pattern in summary['key_insights']['winning_patterns']:
            report.append(f"  • {pattern}")
        
        report.append("\n🔴 LOSING PATTERNS:")
        for pattern in summary['key_insights']['losing_patterns']:
            report.append(f"  • {pattern}")
        
        # Strategy Recommendations
        report.append("\n\n📋 RECOMMENDED STRATEGY:")
        report.append("-" * 60)
        for i, strategy in enumerate(summary['recommended_strategy'], 1):
            report.append(f"  {i}. {strategy}")
        
        return "\n".join(report)
    
    def get_prop_type_recommendation(self, prop_type: str) -> Dict:
        """Get recommendation for a specific prop type"""
        if prop_type in self.prop_type_performance:
            perf = self.prop_type_performance[prop_type]
            
            if perf['success_rate'] >= 0.7:
                recommendation = "HIGHLY RECOMMENDED"
                confidence = "HIGH"
            elif perf['success_rate'] >= 0.5:
                recommendation = "PROCEED WITH CAUTION"
                confidence = "MEDIUM"
            else:
                recommendation = "AVOID"
                confidence = "LOW"
            
            return {
                'prop_type': prop_type,
                'recommendation': recommendation,
                'confidence': confidence,
                'historical_success': f"{perf['success_rate']:.0%}",
                'roi_impact': perf['roi_impact']
            }
        
        return {
            'prop_type': prop_type,
            'recommendation': 'NO DATA',
            'confidence': 'UNKNOWN',
            'historical_success': 'N/A',
            'roi_impact': 0
        }
    
    def get_driver_recommendation(self, driver: str) -> Dict:
        """Get recommendation for a specific driver"""
        if driver in self.driver_performance:
            perf = self.driver_performance[driver]
            
            if perf['success_rate'] >= 0.6:
                recommendation = "TARGET"
            elif perf['success_rate'] >= 0.4:
                recommendation = "SELECTIVE"
            else:
                recommendation = "AVOID"
            
            return {
                'driver': driver,
                'recommendation': recommendation,
                'historical_success': f"{perf['success_rate']:.0%}",
                'best_prop': perf['best_prop'],
                'total_picks': perf['picks']
            }
        
        return {
            'driver': driver,
            'recommendation': 'NO DATA',
            'historical_success': 'N/A',
            'best_prop': 'Unknown',
            'total_picks': 0
        }


# Function to integrate insights into Prize Picks reports
def add_insights_to_prize_picks_report(portfolio: List[Dict], bankroll: float = 1000) -> str:
    """
    Add pattern analysis insights to Prize Picks portfolio report
    
    Args:
        portfolio: List of optimized parlays
        bankroll: Total bankroll amount
        
    Returns:
        Enhanced report with pattern insights
    """
    analyzer = PrizePicksPatternAnalyzer()
    
    # Start with pattern insights
    report = [analyzer.format_insights_for_report()]
    
    # Add portfolio-specific analysis
    report.append("\n\n" + "="*80)
    report.append("CURRENT PORTFOLIO ANALYSIS")
    report.append("="*80)
    
    # Analyze current portfolio against patterns
    prop_types_used = {}
    drivers_used = {}
    lineup_sizes = {}
    
    for parlay in portfolio:
        # Count lineup sizes
        n_picks = parlay.get('n_picks', len(parlay.get('picks', [])))
        lineup_sizes[n_picks] = lineup_sizes.get(n_picks, 0) + 1
        
        # Analyze picks
        if 'picks' in parlay:
            for _, pick in parlay['picks'].iterrows():
                prop_type = pick.get('bet_type', 'Unknown')
                driver = pick.get('driver', 'Unknown')
                
                prop_types_used[prop_type] = prop_types_used.get(prop_type, 0) + 1
                drivers_used[driver] = drivers_used.get(driver, 0) + 1
    
    # Prop type analysis
    report.append("\n📊 Portfolio Prop Type Distribution:")
    total_props = sum(prop_types_used.values())
    for prop_type, count in sorted(prop_types_used.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_props * 100) if total_props > 0 else 0
        rec = analyzer.get_prop_type_recommendation(prop_type)
        report.append(f"  • {prop_type}: {count} picks ({pct:.0f}%) - {rec['recommendation']}")
    
    # Driver analysis
    report.append("\n🏎️ Portfolio Driver Distribution:")
    for driver, count in sorted(drivers_used.items(), key=lambda x: x[1], reverse=True)[:5]:
        rec = analyzer.get_driver_recommendation(driver)
        report.append(f"  • {driver}: {count} picks - {rec['recommendation']} ({rec['historical_success']})")
    
    # Lineup size analysis
    report.append("\n📈 Portfolio Lineup Sizes:")
    for size, count in sorted(lineup_sizes.items()):
        if size in analyzer.lineup_size_performance:
            hist_win_rate = analyzer.lineup_size_performance[size]['win_rate']
            report.append(f"  • {size}-Pick: {count} parlays (Historical win rate: {hist_win_rate:.0%})")
        else:
            report.append(f"  • {size}-Pick: {count} parlays")
    
    # Risk assessment based on patterns
    report.append("\n\n⚠️ RISK ASSESSMENT:")
    report.append("-" * 60)
    
    risks = []
    if prop_types_used.get('Overtake Points', 0) > 0:
        risks.append("WARNING: Portfolio contains Overtake props (0% historical success)")
    if prop_types_used.get('F1 Points', 0) > 0:
        risks.append("WARNING: Portfolio contains F1 Points props (0% historical success)")
    
    starting_pos_pct = (prop_types_used.get('Starting Position', 0) / total_props * 100) if total_props > 0 else 0
    if starting_pos_pct < 50:
        risks.append(f"CAUTION: Only {starting_pos_pct:.0f}% Starting Position props (recommend 80%+)")
    
    if risks:
        for risk in risks:
            report.append(f"  ⚠️ {risk}")
    else:
        report.append("  ✅ Portfolio aligns well with winning patterns")
    
    # Final recommendations
    report.append("\n\n💡 RECOMMENDATIONS FOR THIS PORTFOLIO:")
    report.append("-" * 60)
    
    if starting_pos_pct < 80:
        report.append("  1. Increase Starting Position props to 80% of portfolio")
    if any(size > 3 for size in lineup_sizes.keys()):
        report.append("  2. Consider reducing to 2-3 pick parlays for better ROI")
    if 'Aston Martin' in str(drivers_used):
        report.append("  3. Remove Aston Martin driver picks (0% historical success)")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test the pattern analyzer
    analyzer = PrizePicksPatternAnalyzer()
    print(analyzer.format_insights_for_report())
    
    # Test prop type recommendation
    print("\n\nTesting prop type recommendations:")
    for prop in ['Starting Position', 'F1 Points', '1st Pit Stop Time']:
        rec = analyzer.get_prop_type_recommendation(prop)
        print(f"{prop}: {rec['recommendation']} ({rec['historical_success']})")
    
    # Test driver recommendation
    print("\n\nTesting driver recommendations:")
    for driver in ['Max Verstappen', 'Fernando Alonso', 'Lewis Hamilton']:
        rec = analyzer.get_driver_recommendation(driver)
        print(f"{driver}: {rec['recommendation']} ({rec['historical_success']})")