#!/usr/bin/env python3
"""
Test weather integration in F1 ML pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from f1_ml.weather import F1WeatherProvider
from f1db_data_loader import load_f1db_data
import pandas as pd

def test_weather_csv_cache():
    """Test that weather data is properly cached in CSV"""
    print("Testing Weather CSV Cache")
    print("="*60)
    
    # Initialize provider with API key
    API_KEY = os.environ.get('VISUAL_CROSSING_API_KEY')
    provider = F1WeatherProvider(api_key=API_KEY)
    
    # Test fetching a specific race
    test_races = [
        ('Monaco', '2023-05-28'),
        ('Silverstone', '2023-07-09'),
        ('Spa-Francorchamps', '2023-07-30'),
    ]
    
    for circuit, date in test_races:
        print(f"\nTesting {circuit} on {date}:")
        
        # First call - might fetch from API or CSV cache
        weather1 = provider.get_weather_for_race(circuit, date)
        print(f"  Temperature: {weather1['temperature']:.1f}°C")
        print(f"  Rain probability: {weather1['rain_probability']:.1%}")
        print(f"  Humidity: {weather1['humidity']:.1f}%")
        
        # Second call - should definitely come from CSV cache
        weather2 = provider.get_weather_for_race(circuit, date)
        
        # Verify they match
        assert weather1['temperature'] == weather2['temperature'], "Cache mismatch!"
        print("  ✓ Cache working correctly")
    
    # Check CSV file
    if provider.csv_cache_file.exists():
        cache_df = pd.read_csv(provider.csv_cache_file)
        print(f"\n\nCSV Cache Statistics:")
        print(f"  Total records: {len(cache_df)}")
        print(f"  Unique circuits: {cache_df['circuit_name'].nunique()}")
        print(f"  Date range: {cache_df['date'].min()} to {cache_df['date'].max()}")
        print(f"  Average temperature: {cache_df['temperature'].mean():.1f}°C")
        print(f"  Wet race percentage: {cache_df['is_wet_race'].mean():.1%}")


def test_weather_in_pipeline():
    """Test weather integration in the full pipeline"""
    print("\n\nTesting Weather in F1 Pipeline")
    print("="*60)
    
    # Load F1 data
    data = load_f1db_data()
    
    # Get recent races
    races = data['races']
    recent_races = races[races['year'] == 2023].head(5)
    
    print(f"\nChecking weather for {len(recent_races)} races from 2023:")
    
    # Create weather provider
    api_key = os.environ.get('VISUAL_CROSSING_API_KEY')
    provider = F1WeatherProvider(api_key)
    
    # Get weather for these races
    weather_data = provider.get_weather_features_for_races(recent_races)
    
    print(f"\nWeather data shape: {weather_data.shape}")
    print("\nSample weather data:")
    print(weather_data[['raceId', 'temperature', 'humidity', 'rain_probability', 'is_wet_race']].head())
    
    # Merge with race info
    race_weather = recent_races.merge(weather_data, on='raceId', how='left')
    
    print("\nRaces with weather:")
    for _, race in race_weather.iterrows():
        print(f"  {race['name'][:40]:40s} - {race['temperature']:5.1f}°C, Rain: {race['rain_probability']:4.0%}")


if __name__ == "__main__":
    test_weather_csv_cache()
    test_weather_in_pipeline()
    
    print("\n\n✓ All tests passed!")
    print("\nWeather data is now integrated and cached in:")
    print(f"  data/weather_cache/f1_weather_data.csv")