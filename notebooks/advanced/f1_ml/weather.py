"""
F1 Weather Data Module

This module provides real weather data for F1 races using various weather APIs.
Supports both historical weather data (for training) and forecast data (for predictions).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class F1WeatherProvider:
    """
    Provides real weather data for F1 races
    
    Supports multiple weather APIs:
    - Visual Crossing (recommended for historical data)
    - OpenWeatherMap (good for current/forecast)
    - WeatherAPI.com (alternative option)
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = 'visual_crossing'):
        """
        Initialize weather provider
        
        Args:
            api_key: API key for weather service
            provider: Weather API provider ('visual_crossing', 'openweathermap', 'weatherapi')
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get(f'{provider.upper()}_API_KEY')
        self.cache_dir = Path('data/weather_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV cache file for persistent storage
        self.csv_cache_file = self.cache_dir / 'f1_weather_data.csv'
        self._load_csv_cache()
        
        # Circuit coordinates for weather lookup
        self.circuit_coordinates = {
            # Current circuits
            'bahrain': (26.0325, 50.5106),
            'jeddah': (21.6319, 39.1044),
            'albert_park': (-37.8497, 144.9680),
            'suzuka': (34.8431, 136.5411),
            'shanghai': (31.3389, 121.2198),
            'miami': (25.9581, -80.2389),
            'imola': (44.3439, 11.7167),
            'monaco': (43.7347, 7.4144),
            'catalunya': (41.5700, 2.2611),
            'villeneuve': (45.5000, -73.5228),
            'red_bull_ring': (47.2197, 14.7647),
            'silverstone': (52.0786, -1.0169),
            'hungaroring': (47.5789, 19.2486),
            'spa': (50.4372, 5.9714),
            'zandvoort': (52.3888, 4.5409),
            'monza': (45.6156, 9.2811),
            'marina_bay': (1.2914, 103.8644),
            'americas': (30.1328, -97.6411),
            'rodriguez': (19.4042, -99.0907),
            'interlagos': (-23.7036, -46.6997),
            'vegas': (36.1147, -115.1730),
            'yas_marina': (24.4672, 54.6031),
            # Additional circuits
            'baku': (40.3725, 49.8533),
            'losail': (25.4900, 51.4542),
            # Historic circuits
            'nurburgring': (50.3356, 6.9475),
            'istanbul': (40.9517, 29.4050),
            'sepang': (2.7606, 101.7381),
            'hockenheim': (49.3278, 8.5656),
            'magny_cours': (46.8639, 3.1633),
            'indianapolis': (39.7950, -86.2372),
            'jerez': (36.7083, -6.0342),
            'estoril': (38.7506, -9.3942),
            'adelaide': (-34.9275, 138.6172),
            'fuji': (35.3717, 138.9256),
        }
        
        # Provider endpoints
        self.endpoints = {
            'visual_crossing': {
                'historical': 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date}',
                'forecast': 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}'
            },
            'openweathermap': {
                'historical': 'https://api.openweathermap.org/data/3.0/onecall/timemachine',
                'current': 'https://api.openweathermap.org/data/2.5/weather',
                'forecast': 'https://api.openweathermap.org/data/2.5/forecast'
            },
            'weatherapi': {
                'historical': 'https://api.weatherapi.com/v1/history.json',
                'current': 'https://api.weatherapi.com/v1/current.json',
                'forecast': 'https://api.weatherapi.com/v1/forecast.json'
            }
        }
    
    def get_circuit_coordinates(self, circuit_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a circuit"""
        # Normalize circuit name
        normalized = circuit_name.lower().replace(' ', '_').replace('-', '_')
        
        # Try exact match first
        if normalized in self.circuit_coordinates:
            return self.circuit_coordinates[normalized]
        
        # Extract key location words from full Grand Prix name
        location_keywords = {
            'bahrain': 'bahrain',
            'saudi': 'jeddah',
            'australian': 'albert_park',
            'melbourne': 'albert_park',
            'azerbaijan': 'baku',
            'baku': 'baku',
            'miami': 'miami',
            'monaco': 'monaco',
            'spanish': 'catalunya',
            'spain': 'catalunya',
            'barcelona': 'catalunya',
            'canadian': 'villeneuve',
            'canada': 'villeneuve',
            'montreal': 'villeneuve',
            'austrian': 'red_bull_ring',
            'austria': 'red_bull_ring',
            'british': 'silverstone',
            'britain': 'silverstone',
            'hungarian': 'hungaroring',
            'hungary': 'hungaroring',
            'budapest': 'hungaroring',
            'belgian': 'spa',
            'belgium': 'spa',
            'dutch': 'zandvoort',
            'netherlands': 'zandvoort',
            'italian': 'monza',
            'italy': 'monza',
            'singapore': 'marina_bay',
            'japanese': 'suzuka',
            'japan': 'suzuka',
            'qatar': 'losail',
            'united states': 'americas',
            'usa': 'americas',
            'austin': 'americas',
            'mexican': 'rodriguez',
            'mexico': 'rodriguez',
            'brazilian': 'interlagos',
            'brazil': 'interlagos',
            'sao paulo': 'interlagos',
            'são paulo': 'interlagos',
            'las vegas': 'vegas',
            'abu dhabi': 'yas_marina',
            'chinese': 'shanghai',
            'china': 'shanghai',
            'emilia': 'imola',
            'imola': 'imola'
        }
        
        # Check for location keywords in circuit name
        name_lower = circuit_name.lower()
        for keyword, circuit_key in location_keywords.items():
            if keyword in name_lower:
                if circuit_key in self.circuit_coordinates:
                    return self.circuit_coordinates[circuit_key]
        
        # Try partial matches as last resort
        for key, coords in self.circuit_coordinates.items():
            if key in normalized or normalized in key:
                return coords
        
        logger.warning(f"No coordinates found for circuit: {circuit_name}")
        return None
    
    def _load_csv_cache(self):
        """Load weather data from CSV cache"""
        if self.csv_cache_file.exists():
            try:
                self.weather_cache_df = pd.read_csv(self.csv_cache_file)
                logger.info(f"Loaded {len(self.weather_cache_df)} weather records from CSV cache")
            except Exception as e:
                logger.warning(f"Failed to load CSV cache: {e}")
                self.weather_cache_df = pd.DataFrame()
        else:
            self.weather_cache_df = pd.DataFrame()
    
    def _save_to_csv_cache(self, weather_data: Dict, circuit_name: str, date: str):
        """Save weather data to CSV cache"""
        # Add circuit and date info
        weather_data['circuit_name'] = circuit_name
        weather_data['date'] = date
        weather_data['fetch_timestamp'] = datetime.now().isoformat()
        
        # Create new row
        new_row = pd.DataFrame([weather_data])
        
        # Append to cache
        if self.weather_cache_df.empty:
            self.weather_cache_df = new_row
        else:
            self.weather_cache_df = pd.concat([self.weather_cache_df, new_row], ignore_index=True)
        
        # Save to CSV
        try:
            self.weather_cache_df.to_csv(self.csv_cache_file, index=False)
            logger.info(f"Saved weather data to CSV cache for {circuit_name} on {date}")
        except Exception as e:
            logger.error(f"Failed to save to CSV cache: {e}")
    
    def _check_csv_cache(self, circuit_name: str, date: str) -> Optional[Dict]:
        """Check if weather data exists in CSV cache"""
        if self.weather_cache_df.empty:
            return None
        
        # Look for matching record
        mask = (self.weather_cache_df['circuit_name'] == circuit_name) & \
               (self.weather_cache_df['date'] == date)
        
        matches = self.weather_cache_df[mask]
        
        if not matches.empty:
            # Return most recent record
            record = matches.iloc[-1].to_dict()
            logger.info(f"Found weather data in CSV cache for {circuit_name} on {date}")
            
            # Remove metadata fields
            record.pop('circuit_name', None)
            record.pop('date', None)
            record.pop('fetch_timestamp', None)
            
            return record
        
        return None
    
    def _get_cache_key(self, location: str, date: str) -> str:
        """Generate cache key for weather data"""
        return f"{self.provider}_{location}_{date}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load weather data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save weather data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def fetch_visual_crossing_weather(self, lat: float, lon: float, date: str) -> Optional[Dict]:
        """Fetch weather from Visual Crossing API"""
        if not self.api_key:
            logger.error("No API key provided for Visual Crossing")
            return None
        
        try:
            location = f"{lat},{lon}"
            url = self.endpoints['visual_crossing']['historical'].format(
                location=location,
                date=date
            )
            
            params = {
                'key': self.api_key,
                'unitGroup': 'metric',
                'include': 'hours',
                'elements': 'datetime,temp,humidity,precip,precipprob,windspeed,conditions,description'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract race day data (assume race at 2 PM local)
            day_data = data['days'][0]
            race_hour_data = None
            if 'hours' in day_data:
                for hour in day_data['hours']:
                    if hour.get('datetime', '').startswith('14:'):  # 2 PM
                        race_hour_data = hour
                        break
            
            # Use race hour data if available, otherwise day average
            weather_data = race_hour_data or day_data
            
            return {
                'temperature': weather_data.get('temp', 20),
                'humidity': weather_data.get('humidity', 60),
                'wind_speed': weather_data.get('windspeed', 10),
                'precipitation': weather_data.get('precip', 0),
                'rain_probability': weather_data.get('precipprob', 0) / 100,
                'conditions': weather_data.get('conditions', ''),
                'description': weather_data.get('description', ''),
                'is_wet_race': weather_data.get('precip', 0) > 0.5 or 'rain' in weather_data.get('conditions', '').lower()
            }
            
        except Exception as e:
            logger.error(f"Visual Crossing API error: {e}")
            return None
    
    def fetch_openweathermap_weather(self, lat: float, lon: float, date: str) -> Optional[Dict]:
        """Fetch weather from OpenWeatherMap API"""
        if not self.api_key:
            logger.error("No API key provided for OpenWeatherMap")
            return None
        
        try:
            # Convert date to timestamp
            dt = datetime.strptime(date, '%Y-%m-%d')
            timestamp = int(dt.timestamp())
            
            # Check if historical or forecast
            days_diff = (dt - datetime.now()).days
            
            if days_diff < -5:  # Historical (requires paid subscription)
                url = self.endpoints['openweathermap']['historical']
                params = {
                    'lat': lat,
                    'lon': lon,
                    'dt': timestamp,
                    'appid': self.api_key,
                    'units': 'metric'
                }
            elif days_diff <= 5:  # Forecast
                url = self.endpoints['openweathermap']['forecast']
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'
                }
            else:
                logger.warning("Date too far in future for OpenWeatherMap")
                return None
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response based on endpoint
            if 'list' in data:  # Forecast endpoint
                # Find closest time to race (2 PM)
                target_hour = 14
                best_match = min(data['list'], 
                               key=lambda x: abs(datetime.fromtimestamp(x['dt']).hour - target_hour))
                weather = best_match
            else:  # Current or historical
                weather = data
            
            return {
                'temperature': weather['main']['temp'],
                'humidity': weather['main']['humidity'],
                'wind_speed': weather['wind']['speed'],
                'precipitation': weather.get('rain', {}).get('3h', 0),
                'rain_probability': 1.0 if weather.get('rain') else 0.0,
                'conditions': weather['weather'][0]['main'],
                'description': weather['weather'][0]['description'],
                'is_wet_race': 'rain' in weather['weather'][0]['main'].lower()
            }
            
        except Exception as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            return None
    
    def get_weather_for_race(self, circuit_name: str, date: str, use_cache: bool = True) -> Dict:
        """
        Get weather data for a specific race
        
        Args:
            circuit_name: Name of the circuit
            date: Date of the race (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with weather features
        """
        # Check CSV cache first - this is the primary cache
        if use_cache:
            csv_cached_data = self._check_csv_cache(circuit_name, date)
            if csv_cached_data:
                return csv_cached_data
        
        # Check JSON cache second (legacy)
        cache_key = self._get_cache_key(circuit_name, date)
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                # Also save to CSV cache for future use
                self._save_to_csv_cache(cached_data, circuit_name, date)
                return cached_data
        
        # Get circuit coordinates
        coords = self.get_circuit_coordinates(circuit_name)
        if not coords:
            logger.warning(f"Using default weather for unknown circuit: {circuit_name}")
            return self._get_default_weather(date)
        
        lat, lon = coords
        
        # Try to fetch from API
        weather_data = None
        
        if self.provider == 'visual_crossing':
            weather_data = self.fetch_visual_crossing_weather(lat, lon, date)
        elif self.provider == 'openweathermap':
            weather_data = self.fetch_openweathermap_weather(lat, lon, date)
        # Add more providers as needed
        
        if weather_data:
            # Add derived features
            weather_data['track_temp'] = weather_data['temperature'] + 10  # Rough approximation
            weather_data['weather_changeability'] = 0.2 if weather_data['rain_probability'] > 0.3 else 0.1
            
            # Cache the data in both formats
            self._save_to_cache(cache_key, weather_data)
            self._save_to_csv_cache(weather_data, circuit_name, date)
            return weather_data
        
        # Fallback to default weather
        logger.warning(f"Failed to fetch weather data, using defaults for {circuit_name} on {date}")
        return self._get_default_weather(date)
    
    def _get_default_weather(self, date: str) -> Dict:
        """Get default weather when API fails"""
        # Use reasonable defaults based on season
        dt = datetime.strptime(date, '%Y-%m-%d')
        month = dt.month
        
        # Seasonal temperature
        if month in [12, 1, 2]:  # Winter
            temp = 15
        elif month in [6, 7, 8]:  # Summer
            temp = 25
        else:  # Spring/Fall
            temp = 20
        
        return {
            'temperature': temp,
            'humidity': 60,
            'wind_speed': 10,
            'precipitation': 0,
            'rain_probability': 0.15,
            'conditions': 'Partly Cloudy',
            'description': 'Default weather data',
            'is_wet_race': False,
            'track_temp': temp + 10,
            'weather_changeability': 0.1
        }
    
    def get_weather_features_for_races(self, races_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get weather features for multiple races
        
        Args:
            races_df: DataFrame with race information (must have circuitId, date columns)
            
        Returns:
            DataFrame with weather features
        """
        weather_features = []
        
        # Get circuit mapping if available
        circuits_df = races_df[['circuitId', 'name']].drop_duplicates() if 'name' in races_df.columns else None
        
        for _, race in races_df.iterrows():
            race_id = race.get('raceId', race.get('id'))
            circuit_name = race.get('name', f"circuit_{race.get('circuitId', 'unknown')}")
            date = race.get('date')
            
            if pd.isna(date):
                logger.warning(f"No date for race {race_id}")
                weather = self._get_default_weather('2023-06-01')  # Default date
            else:
                # Convert date to string format
                if isinstance(date, pd.Timestamp):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)[:10]
                
                weather = self.get_weather_for_race(circuit_name, date_str)
            
            # Add race ID
            weather['raceId'] = race_id
            weather_features.append(weather)
        
        return pd.DataFrame(weather_features)


def get_f1_weather_features(df: pd.DataFrame, api_key: Optional[str] = None, 
                           provider: str = 'visual_crossing') -> pd.DataFrame:
    """
    Convenience function to get weather features for F1 races
    
    Args:
        df: DataFrame with race data
        api_key: API key for weather service
        provider: Weather API provider
        
    Returns:
        DataFrame with weather features
    """
    weather_provider = F1WeatherProvider(api_key=api_key, provider=provider)
    return weather_provider.get_weather_features_for_races(df)


# Example usage and testing
if __name__ == "__main__":
    # Test the weather provider
    provider = F1WeatherProvider()
    
    # Test getting coordinates
    print("Testing circuit coordinates:")
    test_circuits = ['silverstone', 'Monaco', 'Monza', 'spa-francorchamps']
    for circuit in test_circuits:
        coords = provider.get_circuit_coordinates(circuit)
        print(f"  {circuit}: {coords}")
    
    # Test getting weather for a specific race
    print("\nTesting weather fetch:")
    weather = provider.get_weather_for_race('silverstone', '2023-07-09')
    print(f"Silverstone weather: {weather}")
    
    # Note: To use real API data, set environment variable:
    # export VISUAL_CROSSING_API_KEY='your-api-key'
    # Or pass api_key parameter when initializing