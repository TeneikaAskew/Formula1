#!/usr/bin/env python3
"""F1 Probability Calibration Module - Phase 2.1 Implementation

This module implements probability calibration techniques including:
- Isotonic Regression
- Platt Scaling  
- Bayesian priors
- Cross-validation for calibration
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class F1ProbabilityCalibrator:
    """Calibrate F1 betting probabilities using various techniques"""
    
    def __init__(self):
        self.isotonic_calibrators = {}
        self.platt_calibrators = {}
        self.prior_rates = {}
        
    def fit_isotonic(self, probabilities: np.ndarray, actual_outcomes: np.ndarray, 
                     prop_type: str) -> IsotonicRegression:
        """Fit isotonic regression calibrator for a specific prop type
        
        Args:
            probabilities: Raw predicted probabilities
            actual_outcomes: Binary outcomes (0 or 1)
            prop_type: Type of prop (e.g., 'points', 'overtakes')
            
        Returns:
            Fitted IsotonicRegression model
        """
        # Ensure we have enough samples
        if len(probabilities) < 10:
            print(f"Warning: Not enough samples ({len(probabilities)}) for {prop_type} calibration")
            return None
            
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(probabilities, actual_outcomes)
        
        # Store calibrator
        self.isotonic_calibrators[prop_type] = iso_reg
        
        # Calculate prior rate
        self.prior_rates[prop_type] = np.mean(actual_outcomes)
        
        return iso_reg
    
    def fit_platt(self, probabilities: np.ndarray, actual_outcomes: np.ndarray,
                  prop_type: str) -> LogisticRegression:
        """Fit Platt scaling calibrator
        
        Args:
            probabilities: Raw predicted probabilities
            actual_outcomes: Binary outcomes (0 or 1)
            prop_type: Type of prop
            
        Returns:
            Fitted LogisticRegression model for Platt scaling
        """
        # Ensure we have enough samples
        if len(probabilities) < 10:
            print(f"Warning: Not enough samples ({len(probabilities)}) for {prop_type} Platt scaling")
            return None
            
        # Transform probabilities to log-odds for better calibration
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        log_odds = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
        
        # Fit logistic regression
        platt = LogisticRegression()
        platt.fit(log_odds, actual_outcomes)
        
        # Store calibrator
        self.platt_calibrators[prop_type] = platt
        
        return platt
    
    def calibrate_probability(self, raw_prob: float, prop_type: str, 
                            method: str = 'isotonic') -> float:
        """Calibrate a single probability
        
        Args:
            raw_prob: Raw predicted probability
            prop_type: Type of prop
            method: 'isotonic' or 'platt'
            
        Returns:
            Calibrated probability
        """
        # Ensure probability is bounded
        raw_prob = np.clip(raw_prob, 0.01, 0.99)
        
        if method == 'isotonic':
            if prop_type in self.isotonic_calibrators and self.isotonic_calibrators[prop_type]:
                calibrated = self.isotonic_calibrators[prop_type].predict([raw_prob])[0]
                return float(np.clip(calibrated, 0.01, 0.99))
        
        elif method == 'platt':
            if prop_type in self.platt_calibrators and self.platt_calibrators[prop_type]:
                epsilon = 1e-15
                raw_prob = np.clip(raw_prob, epsilon, 1 - epsilon)
                log_odds = np.log(raw_prob / (1 - raw_prob))
                calibrated = self.platt_calibrators[prop_type].predict_proba([[log_odds]])[0, 1]
                return float(np.clip(calibrated, 0.01, 0.99))
        
        # Fallback to raw probability if no calibrator available
        return raw_prob
    
    def apply_bayesian_prior(self, raw_prob: float, prop_type: str, 
                           sample_size: int, prior_strength: float = 10.0) -> float:
        """Apply Bayesian prior to probability estimate
        
        Uses Beta-Binomial conjugate prior to shrink estimates toward historical mean
        
        Args:
            raw_prob: Raw predicted probability
            prop_type: Type of prop
            sample_size: Number of historical samples used for prediction
            prior_strength: Strength of prior (equivalent sample size)
            
        Returns:
            Probability adjusted with Bayesian prior
        """
        if prop_type not in self.prior_rates:
            return raw_prob
            
        prior_rate = self.prior_rates[prop_type]
        
        # Beta prior parameters
        alpha_prior = prior_rate * prior_strength
        beta_prior = (1 - prior_rate) * prior_strength
        
        # Observed successes and failures
        observed_successes = raw_prob * sample_size
        observed_failures = (1 - raw_prob) * sample_size
        
        # Posterior parameters
        alpha_post = alpha_prior + observed_successes
        beta_post = beta_prior + observed_failures
        
        # Posterior mean
        posterior_prob = alpha_post / (alpha_post + beta_post)
        
        return float(np.clip(posterior_prob, 0.01, 0.99))
    
    def evaluate_calibration(self, probabilities: np.ndarray, 
                           actual_outcomes: np.ndarray, n_bins: int = 10) -> Dict:
        """Evaluate calibration quality
        
        Args:
            probabilities: Predicted probabilities
            actual_outcomes: Actual binary outcomes
            n_bins: Number of bins for calibration plot
            
        Returns:
            Dictionary with calibration metrics
        """
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_outcomes, probabilities, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate expected calibration error (ECE)
        bin_counts, _ = np.histogram(probabilities, bins=n_bins)
        ece = 0
        for i in range(len(fraction_of_positives)):
            if bin_counts[i] > 0:
                weight = bin_counts[i] / len(probabilities)
                ece += weight * abs(fraction_of_positives[i] - mean_predicted_value[i])
        
        # Calculate Brier score
        brier_score = np.mean((probabilities - actual_outcomes) ** 2)
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'ece': ece,
            'brier_score': brier_score
        }
    
    def cross_validate_calibration(self, probabilities: np.ndarray, 
                                 actual_outcomes: np.ndarray, 
                                 prop_type: str, n_folds: int = 5) -> np.ndarray:
        """Cross-validate calibration to avoid overfitting
        
        Args:
            probabilities: Raw predicted probabilities
            actual_outcomes: Actual binary outcomes
            prop_type: Type of prop
            n_folds: Number of cross-validation folds
            
        Returns:
            Cross-validated calibrated probabilities
        """
        calibrated_probs = np.zeros_like(probabilities)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(probabilities):
            # Train calibrator on fold
            train_probs = probabilities[train_idx]
            train_outcomes = actual_outcomes[train_idx]
            
            # Fit temporary isotonic calibrator
            temp_iso = IsotonicRegression(out_of_bounds='clip')
            temp_iso.fit(train_probs, train_outcomes)
            
            # Calibrate validation fold
            val_probs = probabilities[val_idx]
            calibrated_probs[val_idx] = temp_iso.predict(val_probs)
        
        return np.clip(calibrated_probs, 0.01, 0.99)
    
    def plot_calibration_curve(self, probabilities: np.ndarray, 
                             actual_outcomes: np.ndarray, 
                             title: str = "Calibration Plot") -> None:
        """Plot calibration curve
        
        Args:
            probabilities: Predicted probabilities
            actual_outcomes: Actual binary outcomes
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Actual calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_outcomes, probabilities, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', label='Model calibration')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_calibration_data(predictions_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create calibration datasets from historical predictions
    
    Args:
        predictions_df: DataFrame with columns like 'predicted_prob', 'actual_outcome', 'prop_type'
        
    Returns:
        Dictionary mapping prop types to (probabilities, outcomes) tuples
    """
    calibration_data = {}
    
    for prop_type in predictions_df['prop_type'].unique():
        prop_data = predictions_df[predictions_df['prop_type'] == prop_type]
        
        if len(prop_data) >= 10:  # Minimum samples for calibration
            probabilities = prop_data['predicted_prob'].values
            outcomes = prop_data['actual_outcome'].values
            calibration_data[prop_type] = (probabilities, outcomes)
    
    return calibration_data