#!/usr/bin/env python3
"""F1 Risk Dashboard Module - Phase 4.3 Implementation

This module creates a comprehensive risk dashboard for F1 betting portfolios,
including visualizations and risk metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class F1RiskDashboard:
    """Comprehensive risk dashboard for F1 betting"""
    
    def __init__(self, bankroll: float = 1000):
        self.bankroll = bankroll
        self.risk_metrics = {}
        self.portfolio_history = []
        
    def calculate_risk_metrics(self, portfolio: Dict) -> Dict:
        """Calculate comprehensive risk metrics for a portfolio
        
        Args:
            portfolio: Betting portfolio with bets and stakes
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'bankroll': self.bankroll,
            'total_exposure': 0,
            'exposure_pct': 0,
            'expected_value': 0,
            'expected_roi': 0,
            'value_at_risk': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_probability': 0,
            'risk_score': 0,
            'diversification_score': 0,
            'bet_count': 0,
            'avg_odds': 0,
            'kelly_fraction': 0
        }
        
        if not portfolio or 'bets' not in portfolio:
            return metrics
            
        bets = portfolio['bets']
        if not bets:
            return metrics
            
        # Basic metrics
        metrics['bet_count'] = len(bets)
        metrics['total_exposure'] = sum(bet.get('stake', 0) for bet in bets)
        metrics['exposure_pct'] = (metrics['total_exposure'] / self.bankroll) * 100
        
        # Expected value and ROI
        total_ev = sum(bet.get('expected_value', 0) for bet in bets)
        metrics['expected_value'] = total_ev
        metrics['expected_roi'] = (total_ev / metrics['total_exposure'] * 100) if metrics['total_exposure'] > 0 else 0
        
        # Win probability (average across all bets)
        win_probs = [bet.get('probability', 0.5) for bet in bets]
        metrics['win_probability'] = np.mean(win_probs) if win_probs else 0
        
        # Average odds
        odds = [bet.get('payout', 2.0) for bet in bets]
        metrics['avg_odds'] = np.mean(odds) if odds else 0
        
        # Value at Risk (95% confidence)
        metrics['value_at_risk'] = self._calculate_var(bets, confidence=0.95)
        
        # Sharpe ratio (risk-adjusted returns)
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(bets)
        
        # Maximum potential drawdown
        metrics['max_drawdown'] = min(metrics['total_exposure'], self.bankroll)
        
        # Risk score (0-100, higher is riskier)
        metrics['risk_score'] = self._calculate_risk_score(metrics)
        
        # Kelly fraction used
        kelly_fractions = [bet.get('kelly_fraction', 0.25) for bet in bets]
        metrics['kelly_fraction'] = np.mean(kelly_fractions) if kelly_fractions else 0.25
        
        # Store in history
        self.portfolio_history.append(metrics)
        self.risk_metrics = metrics
        
        return metrics
    
    def _calculate_var(self, bets: List[Dict], confidence: float = 0.95) -> float:
        """Calculate Value at Risk
        
        Args:
            bets: List of bets
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR amount
        """
        if not bets:
            return 0
            
        # Simulate outcomes
        n_simulations = 10000
        outcomes = []
        
        for _ in range(n_simulations):
            total_return = 0
            
            for bet in bets:
                stake = bet.get('stake', 0)
                prob = bet.get('probability', 0.5)
                payout = bet.get('payout', 2.0)
                
                # Simulate bet outcome
                if np.random.random() < prob:
                    total_return += stake * (payout - 1)
                else:
                    total_return -= stake
                    
            outcomes.append(total_return)
        
        # Calculate VaR at confidence level
        var_percentile = (1 - confidence) * 100
        var_amount = np.percentile(outcomes, var_percentile)
        
        return abs(var_amount)  # Return as positive number
    
    def _calculate_sharpe_ratio(self, bets: List[Dict]) -> float:
        """Calculate Sharpe ratio for risk-adjusted returns
        
        Args:
            bets: List of bets
            
        Returns:
            Sharpe ratio
        """
        if not bets:
            return 0
            
        # Calculate expected returns and volatility
        returns = []
        
        for bet in bets:
            stake = bet.get('stake', 0)
            prob = bet.get('probability', 0.5)
            payout = bet.get('payout', 2.0)
            
            # Expected return
            expected = stake * (prob * payout - 1)
            
            # Variance
            win_return = stake * (payout - 1)
            loss_return = -stake
            variance = prob * (win_return - expected)**2 + (1-prob) * (loss_return - expected)**2
            
            returns.append({
                'expected': expected,
                'std': np.sqrt(variance)
            })
        
        # Portfolio statistics
        portfolio_expected = sum(r['expected'] for r in returns)
        portfolio_std = np.sqrt(sum(r['std']**2 for r in returns))  # Assuming independence
        
        # Sharpe ratio (assuming risk-free rate = 0)
        if portfolio_std > 0:
            sharpe = portfolio_expected / portfolio_std
        else:
            sharpe = 0
            
        return sharpe
    
    def _calculate_risk_score(self, metrics: Dict) -> float:
        """Calculate overall risk score (0-100)
        
        Args:
            metrics: Risk metrics dictionary
            
        Returns:
            Risk score
        """
        score = 0
        
        # Exposure risk (0-40 points)
        exposure_pct = metrics['exposure_pct']
        if exposure_pct > 50:
            score += 40
        elif exposure_pct > 25:
            score += 20 + (exposure_pct - 25) * 0.8
        else:
            score += exposure_pct * 0.8
            
        # Concentration risk (0-30 points)
        if metrics['bet_count'] == 1:
            score += 30
        elif metrics['bet_count'] < 3:
            score += 20
        elif metrics['bet_count'] < 5:
            score += 10
            
        # Odds risk (0-30 points)
        avg_odds = metrics['avg_odds']
        if avg_odds > 10:
            score += 30
        elif avg_odds > 5:
            score += 15 + (avg_odds - 5) * 3
        else:
            score += avg_odds * 3
            
        return min(100, score)
    
    def create_dashboard(self, portfolio: Dict, predictions: Dict,
                        save_path: Optional[str] = None):
        """Create comprehensive risk dashboard visualization
        
        Args:
            portfolio: Current betting portfolio
            predictions: Full predictions data
            save_path: Optional path to save dashboard
        """
        # Calculate current metrics
        metrics = self.calculate_risk_metrics(portfolio)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Title
        fig.suptitle('F1 Betting Risk Dashboard', fontsize=20, y=0.98)
        
        # 1. Portfolio Overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_portfolio_overview(ax1, metrics)
        
        # 2. Risk Gauge (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_risk_gauge(ax2, metrics['risk_score'])
        
        # 3. Exposure Breakdown (top right)
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_exposure_breakdown(ax3, portfolio)
        
        # 4. Expected Returns Distribution (middle left)
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_returns_distribution(ax4, portfolio)
        
        # 5. Win Probability by Bet Type (middle right)
        ax5 = fig.add_subplot(gs[1, 2:])
        self._plot_win_probabilities(ax5, portfolio)
        
        # 6. Value at Risk Analysis (bottom left)
        ax6 = fig.add_subplot(gs[2, :2])
        self._plot_var_analysis(ax6, metrics)
        
        # 7. Historical Performance (bottom right)
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_historical_performance(ax7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _plot_portfolio_overview(self, ax, metrics):
        """Plot portfolio overview metrics"""
        ax.axis('off')
        
        # Create text summary
        text = f"""
PORTFOLIO OVERVIEW
═══════════════════

Bankroll: ${metrics['bankroll']:,.2f}
Total Exposure: ${metrics['total_exposure']:.2f} ({metrics['exposure_pct']:.1f}%)
Number of Bets: {metrics['bet_count']}

Expected Value: ${metrics['expected_value']:.2f}
Expected ROI: {metrics['expected_roi']:.1f}%
Win Probability: {metrics['win_probability']*100:.1f}%

Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Value at Risk (95%): ${metrics['value_at_risk']:.2f}
Risk Score: {metrics['risk_score']:.0f}/100
        """
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_risk_gauge(self, ax, risk_score):
        """Plot risk score gauge"""
        # Create semi-circular gauge
        theta = np.linspace(np.pi, 0, 100)
        r = 1
        
        # Color gradient
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
        
        for i in range(len(theta)-1):
            ax.fill_between([theta[i], theta[i+1]], 0, r, 
                          color=colors[i], alpha=0.8)
        
        # Add needle
        angle = np.pi - (risk_score / 100) * np.pi
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Labels
        ax.text(0, -0.2, f'{risk_score:.0f}', ha='center', fontsize=24, fontweight='bold')
        ax.text(0, -0.35, 'Risk Score', ha='center', fontsize=12)
        
        # Risk levels
        ax.text(-1.1, 0, 'Low', ha='right', va='center', fontsize=10)
        ax.text(0, 1.1, 'Medium', ha='center', va='bottom', fontsize=10)
        ax.text(1.1, 0, 'High', ha='left', va='center', fontsize=10)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.5, 1.3)
        ax.axis('off')
        ax.set_title('Risk Level', fontsize=14, pad=20)
    
    def _plot_exposure_breakdown(self, ax, portfolio):
        """Plot exposure breakdown by bet type"""
        if not portfolio or 'bets' not in portfolio:
            ax.text(0.5, 0.5, 'No bets', ha='center', va='center')
            return
            
        # Group by bet type
        bet_types = {}
        for bet in portfolio['bets']:
            bet_type = bet.get('type', 'Unknown')
            stake = bet.get('stake', 0)
            
            if bet_type not in bet_types:
                bet_types[bet_type] = 0
            bet_types[bet_type] += stake
        
        # Create pie chart
        if bet_types:
            labels = list(bet_types.keys())
            sizes = list(bet_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax.set_title('Exposure by Bet Type', fontsize=14)
    
    def _plot_returns_distribution(self, ax, portfolio):
        """Plot expected returns distribution"""
        if not portfolio or 'bets' not in portfolio:
            ax.text(0.5, 0.5, 'No bets', ha='center', va='center')
            return
            
        # Simulate returns
        n_simulations = 10000
        returns = []
        
        for _ in range(n_simulations):
            total_return = 0
            
            for bet in portfolio['bets']:
                stake = bet.get('stake', 0)
                prob = bet.get('probability', 0.5)
                payout = bet.get('payout', 2.0)
                
                if np.random.random() < prob:
                    total_return += stake * (payout - 1)
                else:
                    total_return -= stake
                    
            returns.append(total_return)
        
        # Plot histogram
        ax.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Add statistics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        var_95 = np.percentile(returns, 5)
        
        ax.axvline(mean_return, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: ${mean_return:.2f}')
        ax.axvline(median_return, color='green', linestyle='--', linewidth=2,
                  label=f'Median: ${median_return:.2f}')
        ax.axvline(var_95, color='orange', linestyle='--', linewidth=2,
                  label=f'VaR (95%): ${var_95:.2f}')
        
        ax.set_xlabel('Return ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Expected Returns Distribution (10,000 simulations)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_win_probabilities(self, ax, portfolio):
        """Plot win probabilities by bet"""
        if not portfolio or 'bets' not in portfolio:
            ax.text(0.5, 0.5, 'No bets', ha='center', va='center')
            return
            
        # Extract bet details
        bet_labels = []
        probabilities = []
        stakes = []
        
        for i, bet in enumerate(portfolio['bets']):
            bet_labels.append(f"{bet.get('type', 'Bet')} #{i+1}")
            probabilities.append(bet.get('probability', 0.5))
            stakes.append(bet.get('stake', 0))
        
        # Create bar chart
        y_pos = np.arange(len(bet_labels))
        bars = ax.barh(y_pos, probabilities)
        
        # Color bars based on probability
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob >= 0.7:
                bar.set_color('green')
            elif prob >= 0.55:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
            
            # Add stake annotation
            ax.text(prob + 0.01, i, f'${stakes[i]:.0f}', 
                   va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bet_labels)
        ax.set_xlabel('Win Probability')
        ax.set_xlim(0, 1)
        ax.set_title('Win Probability by Bet (with stake size)', fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add reference line
        ax.axvline(0.55, color='black', linestyle='--', alpha=0.5,
                  label='Min recommended (55%)')
        ax.legend()
    
    def _plot_var_analysis(self, ax, metrics):
        """Plot Value at Risk analysis"""
        # Create VaR visualization
        scenarios = ['Best Case', 'Expected', 'VaR (95%)', 'Worst Case']
        values = [
            metrics['total_exposure'] * (metrics['avg_odds'] - 1),  # Best case
            metrics['expected_value'],  # Expected
            -metrics['value_at_risk'],  # VaR
            -metrics['total_exposure']  # Worst case
        ]
        
        colors = ['green', 'blue', 'orange', 'red']
        bars = ax.bar(scenarios, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${value:.0f}',
                   ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Profit/Loss ($)')
        ax.set_title('Scenario Analysis', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_historical_performance(self, ax):
        """Plot historical performance if available"""
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            ax.text(0.5, 0.5, 'Insufficient historical data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Historical Performance', fontsize=14)
            return
            
        # Extract time series data
        timestamps = [h['timestamp'] for h in self.portfolio_history]
        roi_values = [h['expected_roi'] for h in self.portfolio_history]
        risk_scores = [h['risk_score'] for h in self.portfolio_history]
        
        # Create dual axis plot
        ax2 = ax.twinx()
        
        # Plot ROI
        line1 = ax.plot(range(len(timestamps)), roi_values, 'b-o', 
                       label='Expected ROI (%)', linewidth=2)
        ax.set_ylabel('Expected ROI (%)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot risk score
        line2 = ax2.plot(range(len(timestamps)), risk_scores, 'r-s', 
                        label='Risk Score', linewidth=2)
        ax2.set_ylabel('Risk Score', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Labels
        ax.set_xlabel('Portfolio #')
        ax.set_title('Historical Performance Trends', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    def generate_risk_report(self) -> str:
        """Generate text risk report
        
        Returns:
            Formatted risk report string
        """
        if not self.risk_metrics:
            return "No risk metrics available. Run calculate_risk_metrics() first."
            
        m = self.risk_metrics
        
        report = f"""
F1 BETTING RISK REPORT
Generated: {m['timestamp']}
{'='*50}

PORTFOLIO SUMMARY
-----------------
Bankroll: ${m['bankroll']:,.2f}
Total Exposure: ${m['total_exposure']:.2f} ({m['exposure_pct']:.1f}%)
Number of Bets: {m['bet_count']}
Average Odds: {m['avg_odds']:.2f}x

EXPECTED RETURNS
----------------
Expected Value: ${m['expected_value']:.2f}
Expected ROI: {m['expected_roi']:.1f}%
Win Probability: {m['win_probability']*100:.1f}%
Sharpe Ratio: {m['sharpe_ratio']:.2f}

RISK METRICS
------------
Risk Score: {m['risk_score']:.0f}/100
Value at Risk (95%): ${m['value_at_risk']:.2f}
Maximum Drawdown: ${m['max_drawdown']:.2f}
Kelly Fraction: {m['kelly_fraction']*100:.0f}%

RISK ASSESSMENT
---------------
"""
        
        # Add risk assessment
        if m['risk_score'] < 30:
            report += "Risk Level: LOW - Conservative portfolio\n"
        elif m['risk_score'] < 60:
            report += "Risk Level: MODERATE - Balanced risk/reward\n"
        else:
            report += "Risk Level: HIGH - Aggressive portfolio\n"
            
        if m['exposure_pct'] > 30:
            report += "⚠️  Warning: High bankroll exposure\n"
            
        if m['sharpe_ratio'] < 0.5:
            report += "⚠️  Warning: Low risk-adjusted returns\n"
            
        if m['bet_count'] < 3:
            report += "⚠️  Warning: Low diversification\n"
            
        return report