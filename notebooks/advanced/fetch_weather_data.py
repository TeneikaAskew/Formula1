#!/usr/bin/env python3
"""
Fetch and cache F1 weather data using Visual Crossing API
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from f1_ml.weather import F1WeatherProvider
from f1db_data_loader import load_f1db_data
import pandas as pd

# Set the API key
API_KEY = '852HYSUA4KW2NFS9FCCTYB9FJ'

def check_weather_status(weather_provider, target_races):
    """Check current status of weather data collection"""
    print("\n" + "="*60)
    print("Weather Data Status Check")
    print("="*60)
    
    print(f"\nTarget races (specified range): {len(target_races)}")
    
    # Check CSV cache
    if weather_provider.csv_cache_file.exists():
        cache_df = pd.read_csv(weather_provider.csv_cache_file)
        print(f"Cached weather records: {len(cache_df)}")
        
        # Show year distribution
        cache_df['year'] = pd.to_datetime(cache_df['date']).dt.year
        year_counts = cache_df['year'].value_counts().sort_index()
        print("\nCached data by year:")
        for year, count in year_counts.items():
            print(f"  {year}: {count} races")
        
        # Calculate remaining
        races_fetched = len(cache_df)
        races_remaining = len(target_races) - races_fetched
        
        print(f"\nProgress:")
        print(f"  ✓ Completed: {races_fetched} races ({races_fetched/len(target_races)*100:.1f}%)")
        print(f"  ⏳ Remaining: {races_remaining} races")
        
        # Calculate API credits needed
        # Visual Crossing uses credit-based pricing
        # Each API call costs credits based on data requested
        credits_per_race = 24  # Approximate based on your account showing 985 credits for 41 races
        credits_needed = races_remaining * credits_per_race
        days_needed = (credits_needed + 999) // 1000  # Ceiling division
        
        print(f"\nAPI Credits Estimate:")
        print(f"  - Credits per race: ~{credits_per_race} (based on historical usage)")
        print(f"  - Total credits needed: {credits_needed}")
        print(f"  - Free tier limit: 1000 credits/day")
        print(f"  - Days needed to complete: {days_needed}")
        
        return races_fetched, races_remaining, credits_needed
    else:
        print("No weather cache found!")
        print(f"Need to fetch all {len(target_races)} races")
        credits_needed = len(target_races) * 24
        days_needed = (credits_needed + 999) // 1000
        print(f"This will require approximately {days_needed} days with the free tier")
        return 0, len(target_races), credits_needed

def fetch_historical_weather_data():
    """Fetch weather data for recent F1 races"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fetch F1 weather data')
    parser.add_argument('--start-year', type=int, default=2020, help='Start year (default: 2020)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year (default: 2024)')
    parser.add_argument('--all', action='store_true', help='Fetch all available races')
    parser.add_argument('--status', action='store_true', help='Only show status, do not fetch')
    parser.add_argument('--max-credits', type=int, default=990, help='Maximum credits to use (default: 990)')
    args, unknown = parser.parse_known_args()
    
    # Initialize weather provider with the API key
    weather_provider = F1WeatherProvider(api_key=API_KEY, provider='visual_crossing')
    
    # Load F1 race data
    print("F1 Weather Data Fetcher")
    print("="*60)
    print("\nLoading F1 race data...")
    data = load_f1db_data()
    races_df = data['races']
    
    # Get races from specified range
    if args.all:
        recent_races = races_df.copy()
    else:
        recent_races = races_df[(races_df['year'] >= args.start_year) & (races_df['year'] <= args.end_year)]
    
    recent_races = recent_races.sort_values('date')
    
    print(f"\nFound {len(recent_races)} races from {recent_races['year'].min()}-{recent_races['year'].max()}")
    print(f"CSV cache location: {weather_provider.csv_cache_file}")
    
    # Check status first
    races_cached, races_remaining, credits_needed = check_weather_status(weather_provider, recent_races)
    
    # If status only mode, exit here
    if args.status:
        return
    
    # Calculate how many races we can fetch today
    credits_per_race = 24  # Based on observed usage
    max_races_today = args.max_credits // credits_per_race
    
    print(f"\n" + "="*60)
    print(f"Fetching Weather Data")
    print(f"="*60)
    print(f"Daily credit limit: 1000")
    print(f"Max credits to use: {args.max_credits}")
    print(f"Credits per race: ~{credits_per_race}")
    print(f"Max races to fetch today: {max_races_today}")
    
    # Fetch weather for each race
    weather_fetched = 0
    weather_cached = 0
    weather_errors = 0
    credits_used = 0
    
    for idx, (_, race) in enumerate(recent_races.iterrows(), 1):
        race_name = race.get('name', race.get('officialName', 'Unknown'))
        race_date = str(race['date'])[:10]  # YYYY-MM-DD format
        
        print(f"\n[{idx}/{len(recent_races)}] Checking weather for {race_name} on {race_date}...")
        
        # Check if already in cache
        cached_data = weather_provider._check_csv_cache(race_name, race_date)
        
        if cached_data:
            print(f"  ✓ Already cached")
            weather_cached += 1
        else:
            # Check if we have credits left
            if weather_fetched >= max_races_today:
                print(f"\n⚠️  Daily limit reached ({weather_fetched} races fetched, ~{credits_used} credits used)")
                print(f"Run again tomorrow to continue fetching remaining {races_remaining - weather_fetched} races")
                break
                
            # Fetch from API
            try:
                weather_data = weather_provider.get_weather_for_race(race_name, race_date)
                print(f"  ✓ Fetched from API - Temp: {weather_data['temperature']:.1f}°C, "
                      f"Rain prob: {weather_data['rain_probability']:.1%}")
                weather_fetched += 1
                credits_used = weather_fetched * credits_per_race
                
                # Rate limit handling
                import time
                time.sleep(1.0)  # 1 second delay between requests
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                weather_errors += 1
                
                # Handle rate limit errors specifically
                if "429" in str(e):
                    print("\n⚠️  API credit limit reached!")
                    print("The data fetched so far has been saved.")
                    print("Run again tomorrow to continue.")
                    break
    
    print(f"\n" + "="*60)
    print(f"Summary:")
    print(f"  - Weather data already cached: {weather_cached}")
    print(f"  - Weather data fetched from API: {weather_fetched}")
    print(f"  - Errors encountered: {weather_errors}")
    print(f"  - Total races processed: {weather_cached + weather_fetched + weather_errors}")
    
    # Calculate API usage
    print(f"\nAPI Credits Usage:")
    print(f"  - Credits used today: ~{credits_used} (at ~{credits_per_race} credits per race)")
    print(f"  - Daily credit limit: 1000")
    print(f"  - Credits remaining today: ~{1000 - credits_used}")
    
    # Calculate remaining work
    total_races = len(recent_races)
    races_complete = weather_cached + weather_fetched
    races_remaining_now = total_races - races_complete
    if races_remaining_now > 0:
        credits_needed = races_remaining_now * credits_per_race
        days_needed = (credits_needed + 999) // 1000  # Ceiling division
        print(f"\nRemaining work:")
        print(f"  - Races still to fetch: {races_remaining_now}")
        print(f"  - Credits needed: {credits_needed}")
        print(f"  - Days needed: {days_needed}")
        print(f"\nTo continue fetching, run:")
        print(f"  python fetch_weather_data.py --start-year {args.start_year} --end-year {args.end_year}")
    else:
        print("\n✅ All weather data fetched successfully!")
    
    # Show sample of cached data
    if weather_provider.csv_cache_file.exists():
        cache_df = pd.read_csv(weather_provider.csv_cache_file)
        print(f"\nTotal cached weather data: {len(cache_df)} records")
        
        if len(cache_df) > 0:
            print("\nSample records:")
            print(cache_df[['circuit_name', 'date', 'temperature', 'rain_probability', 'is_wet_race']].tail(5))
            
            # Show statistics
            print(f"\nWeather statistics:")
            print(f"  - Average temperature: {cache_df['temperature'].mean():.1f}°C")
            print(f"  - Average rain probability: {cache_df['rain_probability'].mean():.1%}")
            print(f"  - Wet races: {cache_df['is_wet_race'].sum()} ({cache_df['is_wet_race'].mean():.1%})")


def check_specific_race(circuit_name: str, date: str):
    """Check weather for a specific race"""
    weather_provider = F1WeatherProvider(api_key=API_KEY, provider='visual_crossing')
    
    print(f"\nChecking weather for {circuit_name} on {date}...")
    weather = weather_provider.get_weather_for_race(circuit_name, date)
    
    print("\nWeather data:")
    for key, value in weather.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    # Fetch historical data
    fetch_historical_weather_data()