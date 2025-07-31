"""
F1 ML Explainability Module

This module provides explainability functionality for F1 predictions and Prize Picks optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Any, Optional, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")


class PredictionExplainer:
    """
    Generate explanations for individual predictions
    """
    def __init__(self, model=None, feature_names=None, explainer=None):
        self.model = model
        self.feature_names = feature_names if feature_names is not None else []
        self.explainer = explainer
        
        if not SHAP_AVAILABLE and explainer is None:
            logger.warning("SHAP not available and no explainer provided. Explanations will be limited.")
    
    def explain_prediction(self, features, driver_name="Driver"):
        """
        Generate comprehensive explanation for a single prediction
        """
        if self.model is None:
            return self._get_dummy_explanation(driver_name)
            
        # Get prediction
        if hasattr(features, 'reshape'):
            features_reshaped = features.reshape(1, -1)
        else:
            features_reshaped = np.array(features).reshape(1, -1)
            
        prob = self.model.predict_proba(features_reshaped)[0, 1]
        
        # Get SHAP values if available
        shap_values = None
        if self.explainer and SHAP_AVAILABLE:
            try:
                shap_values = self.explainer.shap_values(features_reshaped)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_values = shap_values[0]
            except Exception as e:
                logger.warning(f"Could not compute SHAP values: {e}")
                shap_values = np.zeros(len(features))
        else:
            shap_values = np.zeros(len(features))
        
        # Create explanation
        explanation = {
            'driver': driver_name,
            'probability': prob,
            'prediction': 'Top 10' if prob > 0.5 else 'Outside Top 10',
            'confidence': self._calculate_confidence(prob),
            'top_factors': self._get_top_factors(features, shap_values),
            'risk_factors': self._get_risk_factors(features, shap_values)
        }
        
        return explanation
    
    def _get_dummy_explanation(self, driver_name):
        """Return a dummy explanation when model is not available"""
        return {
            'driver': driver_name,
            'probability': 0.5,
            'prediction': 'Unknown',
            'confidence': 'Low',
            'top_factors': [],
            'risk_factors': []
        }
    
    def _calculate_confidence(self, probability):
        """
        Calculate confidence level based on probability
        """
        distance_from_middle = abs(probability - 0.5)
        if distance_from_middle > 0.3:
            return 'High'
        elif distance_from_middle > 0.15:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_top_factors(self, features, shap_values, n=3):
        """
        Get top positive contributing factors
        """
        if shap_values is None or len(self.feature_names) == 0:
            return []
            
        positive_indices = np.where(shap_values > 0)[0]
        if len(positive_indices) == 0:
            return []
        
        sorted_indices = positive_indices[np.argsort(shap_values[positive_indices])[::-1]]
        
        factors = []
        for idx in sorted_indices[:n]:
            if idx < len(self.feature_names):
                factors.append({
                    'feature': self.feature_names[idx],
                    'value': float(features[idx]),
                    'impact': float(shap_values[idx])
                })
        
        return factors
    
    def _get_risk_factors(self, features, shap_values, n=3):
        """
        Get top negative contributing factors
        """
        if shap_values is None or len(self.feature_names) == 0:
            return []
            
        negative_indices = np.where(shap_values < 0)[0]
        if len(negative_indices) == 0:
            return []
        
        sorted_indices = negative_indices[np.argsort(shap_values[negative_indices])]
        
        factors = []
        for idx in sorted_indices[:n]:
            if idx < len(self.feature_names):
                factors.append({
                    'feature': self.feature_names[idx],
                    'value': float(features[idx]),
                    'impact': float(shap_values[idx])
                })
        
        return factors
    
    def generate_narrative(self, explanation):
        """
        Generate natural language explanation
        """
        narrative = f"**{explanation['driver']} - {explanation['prediction']} Prediction**\n\n"
        narrative += f"Probability: {explanation['probability']:.1%} (Confidence: {explanation['confidence']})\n\n"
        
        if explanation['top_factors']:
            narrative += "**Key Success Factors:**\n"
            for factor in explanation['top_factors']:
                narrative += f"- {factor['feature']}: {factor['value']:.2f} "
                narrative += f"(+{factor['impact']:.3f} impact)\n"
        
        if explanation['risk_factors']:
            narrative += "\n**Risk Factors:**\n"
            for factor in explanation['risk_factors']:
                narrative += f"- {factor['feature']}: {factor['value']:.2f} "
                narrative += f"({factor['impact']:.3f} impact)\n"
        
        return narrative


class PrizePicksExplainer:
    """
    Explain Prize Picks optimization decisions
    """
    def __init__(self):
        self.bet_type_descriptions = {
            'top_10': 'Finish in top 10 positions',
            'top_5': 'Finish in top 5 positions',
            'top_3': 'Finish on the podium',
            'points': 'Score championship points',
            'beat_teammate': 'Finish ahead of teammate',
            'h2h': 'Head-to-head matchup',
            'dnf': 'Did not finish (DNF)'
        }
    
    def explain_parlay(self, parlay):
        """
        Generate comprehensive explanation for a parlay
        """
        explanation = {
            'summary': self._generate_summary(parlay),
            'picks_analysis': self._analyze_picks(parlay),
            'correlation_analysis': self._analyze_correlation(parlay),
            'risk_assessment': self._assess_risk(parlay),
            'value_proposition': self._analyze_value(parlay)
        }
        
        return explanation
    
    def _generate_summary(self, parlay):
        """
        Generate executive summary
        """
        summary = f"**{parlay.get('n_picks', 0)}-Pick Parlay**\n\n"
        summary += f"- **Bet Amount**: ${parlay.get('bet_size', 0):.2f}\n"
        summary += f"- **Potential Payout**: ${parlay.get('bet_size', 0) * parlay.get('payout', 1):.2f} ({parlay.get('payout', 1)}x)\n"
        summary += f"- **Win Probability**: {parlay.get('adjusted_prob', 0):.1%}\n"
        summary += f"- **Expected Value**: ${parlay.get('expected_value', 0) * parlay.get('bet_size', 0):.2f}\n"
        summary += f"- **Kelly Stake**: {parlay.get('kelly_stake', 0):.1%} of bankroll\n"
        
        return summary
    
    def _analyze_picks(self, parlay):
        """
        Analyze individual picks
        """
        analysis = "**Individual Pick Analysis:**\n\n"
        
        picks = parlay.get('picks', [])
        for i, pick in enumerate(picks, 1):
            bet_type_desc = self.bet_type_descriptions.get(pick.get('bet_type', ''), pick.get('bet_type', 'Unknown'))
            analysis += f"**Pick {i}: {pick.get('driver', 'Unknown')} - {bet_type_desc}**\n"
            analysis += f"- True Probability: {pick.get('true_prob', 0):.1%}\n"
            analysis += f"- Implied Probability: {pick.get('implied_prob', 0):.1%}\n"
            analysis += f"- Edge: +{pick.get('edge', 0):.1%}\n"
            analysis += f"- Confidence: {pick.get('confidence', 0):.1%}\n\n"
        
        return analysis
    
    def _analyze_correlation(self, parlay):
        """
        Analyze correlation between picks
        """
        analysis = "**Correlation Analysis:**\n\n"
        
        correlation = parlay.get('correlation', 0)
        if correlation < 0.3:
            risk_level = "Low"
        elif correlation < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        analysis += f"- Overall Correlation: {correlation:.2f} ({risk_level} risk)\n"
        analysis += f"- Diversification: {'Good' if correlation < 0.3 else 'Could be improved'}\n"
        
        # Check for same driver multiple times
        picks = parlay.get('picks', [])
        drivers = [p.get('driver', '') for p in picks]
        if len(set(drivers)) < len(drivers):
            analysis += "- ⚠️ Multiple bets on same driver increases correlation risk\n"
        
        return analysis
    
    def _assess_risk(self, parlay):
        """
        Assess risk factors
        """
        analysis = "**Risk Assessment:**\n\n"
        
        # Calculate risk score
        risk_factors = []
        
        # Parlay size risk
        n_picks = parlay.get('n_picks', 0)
        if n_picks >= 4:
            risk_factors.append("Large parlay size (4+ picks) significantly reduces win probability")
        
        # Low probability picks
        picks = parlay.get('picks', [])
        low_prob_picks = [p for p in picks if p.get('true_prob', 0) < 0.5]
        if len(low_prob_picks) > 0:
            risk_factors.append(f"{len(low_prob_picks)} picks have <50% probability")
        
        # Correlation risk
        if parlay.get('correlation', 0) > 0.5:
            risk_factors.append("High correlation between picks")
        
        if risk_factors:
            analysis += "**Risk Factors:**\n"
            for factor in risk_factors:
                analysis += f"- {factor}\n"
        else:
            analysis += "✅ No major risk factors identified\n"
        
        # Overall risk rating
        risk_score = len(risk_factors) / 3 if len(risk_factors) > 0 else 0
        if risk_score < 0.33:
            risk_rating = "Low Risk"
        elif risk_score < 0.67:
            risk_rating = "Medium Risk"
        else:
            risk_rating = "High Risk"
        
        analysis += f"\n**Overall Risk Rating**: {risk_rating}\n"
        
        return analysis
    
    def _analyze_value(self, parlay):
        """
        Analyze value proposition
        """
        analysis = "**Value Analysis:**\n\n"
        
        ev = parlay.get('expected_value', 0)
        payout = parlay.get('payout', 1)
        adjusted_prob = parlay.get('adjusted_prob', 0)
        
        roi = (ev / 1) * 100  # ROI percentage
        
        analysis += f"- Expected ROI: {roi:.1f}%\n"
        analysis += f"- Breakeven Win Rate: {1/payout:.1%}\n"
        analysis += f"- Your Win Rate: {adjusted_prob:.1%}\n"
        analysis += f"- Edge over Breakeven: +{(adjusted_prob - 1/payout):.1%}\n"
        
        if ev > 0:
            analysis += "\n✅ **Positive Expected Value - Recommended Bet**\n"
        else:
            analysis += "\n❌ **Negative Expected Value - Not Recommended**\n"
        
        return analysis


def create_interactive_explanation(explanation, shap_values=None):
    """
    Create interactive Plotly visualization for prediction explanation
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot create interactive visualization.")
        return None
        
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Prediction Probability', 'Feature Contributions',
                       'Confidence Gauge', 'Risk Assessment'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'indicator'}, {'type': 'scatter'}]]
    )
    
    # 1. Prediction probability bar
    fig.add_trace(
        go.Bar(
            x=['Not Top 10', 'Top 10'],
            y=[1 - explanation['probability'], explanation['probability']],
            marker_color=['red', 'green'],
            text=[f"{(1-explanation['probability']):.1%}", f"{explanation['probability']:.1%}"],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Feature contributions
    all_factors = explanation.get('top_factors', []) + explanation.get('risk_factors', [])
    if all_factors:
        features = [f['feature'] for f in all_factors]
        impacts = [f['impact'] for f in all_factors]
        colors = ['green' if i > 0 else 'red' for i in impacts]
        
        fig.add_trace(
            go.Bar(
                x=impacts,
                y=features,
                orientation='h',
                marker_color=colors
            ),
            row=1, col=2
        )
    
    # 3. Confidence gauge
    confidence_value = {'High': 0.9, 'Medium': 0.6, 'Low': 0.3}.get(explanation.get('confidence', 'Low'), 0.3)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence_value,
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.33], 'color': "lightgray"},
                    {'range': [0.33, 0.67], 'color': "gray"},
                    {'range': [0.67, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ),
        row=2, col=1
    )
    
    # 4. Risk vs Reward scatter
    top_factors = explanation.get('top_factors', [])
    risk_factors = explanation.get('risk_factors', [])
    risk_score = len(risk_factors) / (len(risk_factors) + len(top_factors) + 1) if (risk_factors or top_factors) else 0.5
    reward_score = explanation.get('probability', 0.5)
    
    fig.add_trace(
        go.Scatter(
            x=[risk_score],
            y=[reward_score],
            mode='markers+text',
            marker=dict(size=20, color='blue'),
            text=[explanation.get('driver', 'Unknown')],
            textposition="top center"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"{explanation.get('driver', 'Unknown')} - Prize Picks Analysis",
        showlegend=False,
        height=800
    )
    
    # Update axes
    fig.update_xaxes(title_text="Probability", row=1, col=1)
    fig.update_xaxes(title_text="Impact", row=1, col=2)
    fig.update_xaxes(title_text="Risk Score", range=[0, 1], row=2, col=2)
    fig.update_yaxes(title_text="Reward (Win Probability)", range=[0, 1], row=2, col=2)
    
    return fig