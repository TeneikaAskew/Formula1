"""
F1 Explainability Engine Module

This module contains the PredictionExplainer and PrizePicksExplainer classes
for generating SHAP-based explanations of F1 predictions and betting recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')


class PredictionExplainer:
    """
    Explain individual F1 driver predictions using SHAP
    """
    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        if model is not None:
            # Initialize SHAP explainer
            if hasattr(model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(model)
            else:
                print("Warning: Model doesn't support probability predictions")
    
    def explain_prediction(self, features, driver_name, actual_position=None):
        """
        Generate explanation for a single prediction
        """
        if self.explainer is None:
            return "No model available for explanation"
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Handle multi-class output (use positive class for binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Get base value and prediction
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]  # Positive class
        
        prediction_prob = self.model.predict_proba(features)[0, 1]
        
        # Get top contributing features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'abs_shap': np.abs(shap_values[0])
        }).sort_values('abs_shap', ascending=False)
        
        # Generate explanation
        explanation = {
            'driver': driver_name,
            'prediction_probability': prediction_prob,
            'base_probability': 1 / (1 + np.exp(-base_value)),  # Convert log-odds to probability
            'confidence': self._calculate_confidence(prediction_prob),
            'top_factors': feature_importance.head(5).to_dict('records'),
            'narrative': self._generate_narrative(driver_name, prediction_prob, feature_importance)
        }
        
        if actual_position:
            explanation['actual_position'] = actual_position
            explanation['prediction_accuracy'] = self._evaluate_accuracy(prediction_prob, actual_position)
        
        return explanation
    
    def _calculate_confidence(self, prob):
        """
        Calculate confidence level based on prediction probability
        """
        # High confidence when probability is very high or very low
        distance_from_middle = abs(prob - 0.5)
        
        if distance_from_middle > 0.4:
            return "Very High"
        elif distance_from_middle > 0.3:
            return "High"
        elif distance_from_middle > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _generate_narrative(self, driver_name, prob, feature_importance):
        """
        Generate human-readable explanation narrative
        """
        top_factors = feature_importance.head(3)
        
        narrative = f"{driver_name} has a {prob:.1%} chance of finishing in the top 10. "
        
        # Describe top factors
        narratives = []
        for _, factor in top_factors.iterrows():
            feature = factor['feature']
            impact = factor['shap_value']
            
            if impact > 0:
                direction = "increases"
            else:
                direction = "decreases"
            
            # Humanize feature names
            feature_descriptions = {
                'avg_position_5': 'recent average position',
                'grid': 'starting position',
                'constructor_avg_position': 'team performance',
                'driver_track_avg': 'track history',
                'career_points': 'career success',
                'dnf_rate_5': 'recent reliability'
            }
            
            feature_desc = feature_descriptions.get(feature, feature)
            narratives.append(f"{feature_desc} {direction} chances")
        
        narrative += f"Key factors: {', '.join(narratives)}."
        
        return narrative
    
    def _evaluate_accuracy(self, prob, actual_position):
        """
        Evaluate prediction accuracy if actual result is known
        """
        predicted_top10 = prob > 0.5
        actual_top10 = actual_position <= 10
        
        if predicted_top10 == actual_top10:
            return "Correct"
        else:
            return "Incorrect"
    
    def create_shap_plot(self, features, driver_name, save_path=None):
        """
        Create SHAP waterfall plot for prediction
        """
        if self.explainer is None:
            return None
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
                data=features[0],
                feature_names=self.feature_names
            ),
            max_display=10
        )
        
        plt.title(f'SHAP Explanation for {driver_name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class PrizePicksExplainer:
    """
    Explain Prize Picks optimization decisions
    """
    def __init__(self):
        self.bet_type_descriptions = {
            'top_10': 'Finish in Top 10',
            'top_5': 'Finish in Top 5',
            'top_3': 'Podium Finish',
            'points': 'Score Points',
            'h2h': 'Head-to-Head Win',
            'beat_teammate': 'Beat Teammate',
            'grid_gain': 'Gain Positions',
            'dnf': 'Did Not Finish'
        }
    
    def explain_parlay(self, parlay_data, detailed=True):
        """
        Generate comprehensive explanation for a Prize Picks parlay
        """
        picks = parlay_data['picks']
        n_picks = parlay_data['n_picks']
        
        explanation = {
            'summary': self._generate_summary(parlay_data),
            'individual_picks': self._analyze_picks(picks),
            'correlation_analysis': self._analyze_correlation(parlay_data),
            'risk_assessment': self._assess_risk(parlay_data),
            'value_proposition': self._analyze_value(parlay_data)
        }
        
        if detailed:
            explanation['recommendations'] = self._generate_recommendations(parlay_data)
            explanation['alternative_options'] = self._suggest_alternatives(parlay_data)
        
        return explanation
    
    def _generate_summary(self, parlay_data):
        """
        Generate executive summary of the parlay
        """
        return {
            'total_picks': parlay_data['n_picks'],
            'combined_probability': parlay_data['adjusted_prob'],
            'expected_value': parlay_data['expected_value'],
            'potential_payout': parlay_data['payout'],
            'recommended_stake': parlay_data['kelly_stake'],
            'confidence_level': self._calculate_parlay_confidence(parlay_data)
        }
    
    def _analyze_picks(self, picks):
        """
        Analyze individual picks in the parlay
        """
        pick_analysis = []
        
        for _, pick in picks.iterrows():
            analysis = {
                'driver': pick['driver'],
                'bet_type': self.bet_type_descriptions.get(pick['bet_type'], pick['bet_type']),
                'probability': pick['true_prob'],
                'edge': pick['edge'],
                'confidence': pick.get('confidence', 0.7),
                'risk_level': self._calculate_pick_risk(pick)
            }
            pick_analysis.append(analysis)
        
        return pick_analysis
    
    def _analyze_correlation(self, parlay_data):
        """
        Analyze correlation between picks
        """
        correlation = parlay_data['correlation']
        
        if correlation < 0.2:
            risk_level = "Low"
            description = "Picks are well-diversified with minimal correlation"
        elif correlation < 0.5:
            risk_level = "Medium"
            description = "Moderate correlation between picks"
        else:
            risk_level = "High"
            description = "High correlation - picks are closely related"
        
        return {
            'correlation_score': correlation,
            'risk_level': risk_level,
            'description': description,
            'recommendation': self._get_correlation_recommendation(correlation)
        }
    
    def _assess_risk(self, parlay_data):
        """
        Comprehensive risk assessment
        """
        prob = parlay_data['adjusted_prob']
        ev = parlay_data['expected_value']
        kelly = parlay_data['kelly_stake']
        
        # Risk factors
        risk_score = 0
        risk_factors = []
        
        # Low probability
        if prob < 0.1:
            risk_score += 3
            risk_factors.append("Very low win probability")
        elif prob < 0.2:
            risk_score += 2
            risk_factors.append("Low win probability")
        
        # Negative EV
        if ev < 0:
            risk_score += 3
            risk_factors.append("Negative expected value")
        
        # High Kelly stake
        if kelly > 0.1:
            risk_score += 1
            risk_factors.append("Aggressive bet sizing")
        
        # Overall assessment
        if risk_score >= 5:
            overall_risk = "Very High"
        elif risk_score >= 3:
            overall_risk = "High"
        elif risk_score >= 1:
            overall_risk = "Medium"
        else:
            overall_risk = "Low"
        
        return {
            'overall_risk': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'break_even_probability': 1 / parlay_data['payout']
        }
    
    def _analyze_value(self, parlay_data):
        """
        Analyze value proposition
        """
        ev = parlay_data['expected_value']
        prob = parlay_data['adjusted_prob']
        payout = parlay_data['payout']
        
        # ROI calculation
        roi = (ev / 1.0) * 100  # Assuming unit stake
        
        # Value rating
        if roi > 20:
            value_rating = "Excellent"
        elif roi > 10:
            value_rating = "Good"
        elif roi > 5:
            value_rating = "Fair"
        elif roi > 0:
            value_rating = "Marginal"
        else:
            value_rating = "Poor"
        
        return {
            'expected_roi': roi,
            'value_rating': value_rating,
            'win_probability': prob,
            'payout_multiplier': payout,
            'true_odds': 1 / prob if prob > 0 else float('inf'),
            'implied_odds': payout
        }
    
    def _calculate_parlay_confidence(self, parlay_data):
        """
        Calculate overall confidence in the parlay
        """
        # Factors affecting confidence
        ev = parlay_data['expected_value']
        correlation = parlay_data['correlation']
        prob = parlay_data['adjusted_prob']
        
        confidence_score = 0
        
        # Positive EV
        if ev > 0.2:
            confidence_score += 3
        elif ev > 0.1:
            confidence_score += 2
        elif ev > 0:
            confidence_score += 1
        
        # Low correlation
        if correlation < 0.3:
            confidence_score += 2
        elif correlation < 0.5:
            confidence_score += 1
        
        # Reasonable probability
        if 0.1 < prob < 0.5:
            confidence_score += 1
        
        # Convert to rating
        if confidence_score >= 5:
            return "Very High"
        elif confidence_score >= 3:
            return "High"
        elif confidence_score >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_pick_risk(self, pick):
        """
        Calculate risk level for individual pick
        """
        edge = pick['edge']
        prob = pick['true_prob']
        
        if edge < 0.05 or prob < 0.3:
            return "High"
        elif edge < 0.1 or prob < 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _get_correlation_recommendation(self, correlation):
        """
        Get recommendation based on correlation
        """
        if correlation < 0.3:
            return "Good diversification - proceed with confidence"
        elif correlation < 0.5:
            return "Consider reducing correlation by swapping similar picks"
        else:
            return "High correlation risk - strongly consider alternative picks"
    
    def _generate_recommendations(self, parlay_data):
        """
        Generate actionable recommendations
        """
        recommendations = []
        
        # Based on EV
        if parlay_data['expected_value'] < 0:
            recommendations.append("Consider passing on this parlay due to negative expected value")
        elif parlay_data['expected_value'] > 0.2:
            recommendations.append("Strong value opportunity - consider full Kelly stake")
        
        # Based on correlation
        if parlay_data['correlation'] > 0.5:
            recommendations.append("Reduce correlation by diversifying bet types or drivers")
        
        # Based on probability
        if parlay_data['adjusted_prob'] < 0.1:
            recommendations.append("Very low win probability - consider fewer picks")
        
        # Stake sizing
        if parlay_data['kelly_stake'] > 0.1:
            recommendations.append("Consider reducing stake size for risk management")
        
        return recommendations
    
    def _suggest_alternatives(self, parlay_data):
        """
        Suggest alternative betting options
        """
        alternatives = []
        
        n_picks = parlay_data['n_picks']
        
        # Suggest different parlay sizes
        if n_picks > 3:
            alternatives.append({
                'option': f"Reduce to {n_picks-1} picks",
                'rationale': "Higher win probability with still attractive payout"
            })
        
        if n_picks < 5:
            alternatives.append({
                'option': f"Increase to {n_picks+1} picks",
                'rationale': "Higher potential payout if edge is strong"
            })
        
        # Suggest splitting
        if n_picks >= 4:
            alternatives.append({
                'option': "Split into two smaller parlays",
                'rationale': "Reduce variance while maintaining upside"
            })
        
        return alternatives


def create_interactive_explanation(explanation_data, plot_type='waterfall'):
    """
    Create interactive Plotly visualization for explanations
    """
    if plot_type == 'waterfall':
        # Create waterfall chart for SHAP values
        top_factors = pd.DataFrame(explanation_data['top_factors'])
        
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(top_factors) + ["total"],
            x=list(top_factors['feature']) + ["Prediction"],
            y=list(top_factors['shap_value']) + [sum(top_factors['shap_value'])],
            text=[f"{v:.3f}" for v in top_factors['shap_value']] + [f"{sum(top_factors['shap_value']):.3f}"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        
        fig.update_layout(
            title=f"Prediction Explanation for {explanation_data['driver']}",
            xaxis_title="Features",
            yaxis_title="SHAP Value (Impact on Prediction)",
            showlegend=False
        )
        
    elif plot_type == 'parlay_breakdown':
        # Create breakdown of parlay picks
        picks = pd.DataFrame(explanation_data['individual_picks'])
        
        fig = px.bar(
            picks,
            x='driver',
            y='probability',
            color='risk_level',
            title='Parlay Pick Breakdown',
            labels={'probability': 'Win Probability', 'driver': 'Driver'},
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="50% probability")
    
    return fig


# Export key components
__all__ = [
    'PredictionExplainer',
    'PrizePicksExplainer',
    'create_interactive_explanation'
]