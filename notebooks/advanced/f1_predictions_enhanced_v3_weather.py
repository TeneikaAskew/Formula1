#!/usr/bin/env python3
"""Enhanced F1 predictions v3 with weather integration - Interactive version"""

import sys
import os

# Import base v3 predictor
from f1_predictions_enhanced_v3 import F1PrizePicksPredictor
from f1_weather_integration import F1WeatherAnalyzer, integrate_weather_into_predictions
from pathlib import Path
import json


def main():
    """Run v3 predictions with weather integration - interactive version"""
    print("="*80)
    print("F1 PREDICTIONS V3 WITH WEATHER INTEGRATION")
    print("="*80)
    
    # Create base predictor - this will prompt for prop lines
    predictor = F1PrizePicksPredictor()
    
    # Create weather analyzer
    print("\nInitializing weather analysis...")
    weather_analyzer = F1WeatherAnalyzer(predictor.data)
    
    # Integrate weather into predictor
    predictor = integrate_weather_into_predictions(predictor, weather_analyzer)
    
    print(f"\nWeather data loaded for analysis")
    print("Weather will affect predictions for:")
    print("  • Overtakes (wet conditions increase opportunities)")
    print("  • DNF probability (weather-related incidents)")
    print("  • Points scoring (wet weather specialists)")
    print("  • Position variance (unpredictability in rain)")
    
    # Generate predictions with weather integration
    predictions = predictor.generate_all_predictions()
    
    # Save predictions
    output_path = Path("pipeline_outputs/enhanced_predictions_weather.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    print(f"\nPredictions with weather integration saved to: {output_path}")
    
    return predictions


if __name__ == "__main__":
    predictions = main()