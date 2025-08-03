# Weather API Setup for F1 ML Pipeline

This document explains how to set up real weather data integration for the F1 ML pipeline, replacing the synthetic weather data with actual historical and forecast weather information.

## Overview

The F1 ML pipeline now supports real weather data through multiple weather API providers. Weather is a crucial factor in F1 race outcomes, affecting:
- Tire strategy
- Car setup
- Race incidents
- Qualifying performance

## Supported Weather APIs

### 1. Visual Crossing (Recommended)
- **Best for**: Historical weather data
- **Free tier**: 1,000 records/day
- **Sign up**: https://www.visualcrossing.com/weather-api
- **Features**: Excellent historical data coverage, hourly data available

### 2. OpenWeatherMap
- **Best for**: Current and 5-day forecast
- **Free tier**: 1,000 calls/day
- **Sign up**: https://openweathermap.org/api
- **Note**: Historical data requires paid subscription

### 3. WeatherAPI.com
- **Best for**: Balance of historical and forecast
- **Free tier**: 1,000,000 calls/month
- **Sign up**: https://www.weatherapi.com/
- **Features**: Good historical coverage (last 7 days free)

## Setup Instructions

### Step 1: Get an API Key

1. Choose a weather API provider (Visual Crossing recommended)
2. Sign up for a free account
3. Generate an API key from your dashboard

### Step 2: Configure Environment Variable

Set your API key as an environment variable:

```bash
# Linux/Mac
export VISUAL_CROSSING_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set VISUAL_CROSSING_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:VISUAL_CROSSING_API_KEY="your-api-key-here"
```

Or add to your `.env` file:
```
VISUAL_CROSSING_API_KEY=your-api-key-here
```

### Step 3: Test the Integration

```python
from f1_ml.weather import F1WeatherProvider

# Initialize weather provider
weather_provider = F1WeatherProvider(
    api_key='your-api-key',  # or use environment variable
    provider='visual_crossing'
)

# Test getting weather for a specific race
weather = weather_provider.get_weather_for_race(
    circuit_name='Silverstone',
    date='2023-07-09'
)
print(weather)
```

## Using Weather Data in the Pipeline

The pipeline automatically uses real weather data when available:

```python
from f1_ml.features import F1FeatureStore

# The feature store will automatically use real weather if API key is set
feature_store = F1FeatureStore()
features = feature_store.prepare_features(data)
```

## Weather Features Provided

The weather module provides the following features:
- **temperature**: Air temperature in Celsius
- **track_temp**: Estimated track temperature
- **humidity**: Relative humidity percentage
- **wind_speed**: Wind speed in km/h
- **rain_probability**: Probability of rain (0-1)
- **is_wet_race**: Binary indicator for wet conditions
- **weather_changeability**: Likelihood of changing conditions

## Fallback Behavior

If the API is unavailable or no API key is provided, the system falls back to:
1. **Cached data**: Previously fetched weather is cached locally
2. **Historical averages**: Circuit-specific historical weather patterns
3. **No random/synthetic data**: All fallbacks use real historical patterns

## Cache Management

Weather data is automatically cached to:
- Reduce API calls
- Improve performance
- Provide offline capability

Cache location: `data/weather_cache/`

## Best Practices

1. **Development**: Use cached data to avoid API limits
2. **Production**: Set up automated cache refresh
3. **Backtesting**: Pre-fetch all historical weather data
4. **Real-time predictions**: Fetch weather 24-48 hours before race

## API Limits and Costs

| Provider | Free Tier | Historical Data | Cost |
|----------|-----------|-----------------|------|
| Visual Crossing | 1,000/day | Full history | $0.0001/record |
| OpenWeatherMap | 1,000/day | Paid only | $0.002/call |
| WeatherAPI | 1M/month | 7 days free | $0.000025/call |

## Troubleshooting

### No API Key Error
```
Warning: Failed to get real weather data: No API key provided
```
**Solution**: Set the environment variable or pass api_key parameter

### API Limit Exceeded
```
Error: API rate limit exceeded
```
**Solution**: Use cached data or upgrade API plan

### Circuit Not Found
```
Warning: No coordinates found for circuit: [name]
```
**Solution**: Circuit coordinates may need to be added to weather.py

## Example: Full Pipeline with Weather

```python
import os
from f1db_data_loader import load_f1db_data
from f1_ml.features import F1FeatureStore

# Set API key
os.environ['VISUAL_CROSSING_API_KEY'] = 'your-key'

# Load data
data = load_f1db_data()

# Create features with real weather
feature_store = F1FeatureStore()
features = feature_store.prepare_features(data)

# Weather data is now integrated into features
print("Weather features included:", 
      any(col for col in features.columns if 'weather' in col or 'rain' in col))
```

## Contributing

To add support for a new weather API:
1. Add provider to `F1WeatherProvider` class
2. Implement fetch method
3. Map response to standard format
4. Add to endpoints dictionary

## Future Enhancements

- [ ] Real-time weather updates during race
- [ ] Radar data integration
- [ ] Track surface temperature sensors
- [ ] Multi-session weather tracking
- [ ] Weather forecast uncertainty modeling