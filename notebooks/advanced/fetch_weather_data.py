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
from datetime import datetime

def check_and_setup_api_key():
    """Check for API key and help user set it up if missing"""
    # Get API key from environment
    api_key = os.environ.get('VISUAL_CROSSING_API_KEY')
    
    # Check if already set and valid
    if api_key and api_key != 'your_visual_crossing_api_key_here':
        return api_key
    
    # Check for .env file
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('VISUAL_CROSSING_API_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key and key != 'your_visual_crossing_api_key_here':
                        os.environ['VISUAL_CROSSING_API_KEY'] = key
                        return key
    
    # No API key found - provide setup instructions
    print("\n" + "="*60)
    print("⚠️  VISUAL CROSSING API KEY NOT FOUND!")
    print("="*60)
    print("\nThis script requires a Visual Crossing API key to fetch weather data.")
    print("\nTo get your free API key:")
    print("1. Visit https://www.visualcrossing.com/")
    print("2. Sign up for a free account")
    print("3. Get your API key from the dashboard")
    print("   (Free tier includes 1000 API calls per day)")
    
    print("\n" + "-"*60)
    
    # Offer to set it up interactively
    setup_now = input("\nWould you like to set up your API key now? (y/n): ")
    if setup_now.lower() != 'y':
        print("\nTo set it up later, use one of these methods:")
        print("1. Export environment variable:")
        print("   export VISUAL_CROSSING_API_KEY=your_actual_api_key")
        print("2. Create .env file with your key")
        print("3. Pass it when running the script:")
        print("   VISUAL_CROSSING_API_KEY=your_key python fetch_weather_data.py")
        sys.exit(1)
    
    # Get API key from user
    api_key = input("\nEnter your Visual Crossing API key: ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)
    
    # Ask how to save it
    print("\nHow would you like to save the API key?")
    print("1. Create/update .env file (recommended)")
    print("2. Just use for this session")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        # Create or update .env file
        env_content = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if not line.startswith('VISUAL_CROSSING_API_KEY='):
                        env_content.append(line.rstrip())
        
        env_content.append(f'VISUAL_CROSSING_API_KEY={api_key}')
        
        with open(env_file, 'w') as f:
            f.write('\n'.join(env_content) + '\n')
        
        print(f"\n✓ Saved API key to {env_file}")
        print("The key will be automatically loaded in future runs.")
    
    # Set for current session
    os.environ['VISUAL_CROSSING_API_KEY'] = api_key
    print("\n✓ API key set for current session")
    print("="*60 + "\n")
    
    return api_key

# Get API key (with setup if needed)
API_KEY = check_and_setup_api_key()

def check_session_status(weather_provider, target_races, date_column, session_type):
    """Check status for a specific session type"""
    # Count valid dates for this session
    if date_column not in target_races.columns:
        return 0, 0, 0
        
    valid_dates = target_races[date_column].notna()
    total_sessions = valid_dates.sum()
    
    if total_sessions == 0:
        return 0, 0, 0
    
    # Count cached items by checking each race individually
    cached_count = 0
    for _, race in target_races[valid_dates].iterrows():
        circuit_name = race.get('circuit_name', race.get('circuitId', 'Unknown'))
        session_date = str(race[date_column])[:10]
        if weather_provider._check_csv_cache(circuit_name, session_date, session_type):
            cached_count += 1
    
    remaining = total_sessions - cached_count
    credits_needed = remaining * 24
    
    return cached_count, remaining, credits_needed

def check_all_sessions_status(weather_provider, target_races, session_configs, sessions_to_fetch):
    """Check status for all session types"""
    print("\n" + "="*60)
    print("Weather Data Status Check")
    print("="*60)
    
    total_cached = 0
    total_remaining = 0
    total_credits_needed = 0
    
    for session_type in sessions_to_fetch:
        print(f"\n{session_type.upper()} Sessions:")
        session_total_cached = 0
        session_total_remaining = 0
        
        for date_col, _ in session_configs[session_type]:
            cached, remaining, credits = check_session_status(
                weather_provider, target_races, date_col, session_type
            )
            
            if date_col in target_races.columns:
                total_possible = target_races[date_col].notna().sum()
                if total_possible > 0:
                    print(f"  {date_col}: {cached}/{total_possible} cached")
                    session_total_cached += cached
                    session_total_remaining += remaining
        
        total_cached += session_total_cached
        total_remaining += session_total_remaining
        total_credits_needed += session_total_remaining * 24
        
        print(f"  Total: {session_total_cached} cached, {session_total_remaining} remaining")
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total cached: {total_cached}")
    print(f"  Total remaining: {total_remaining}")
    print(f"  Credits needed: {total_credits_needed}")
    print(f"  Days needed: {(total_credits_needed + 999) // 1000}")

def check_weather_status(weather_provider, target_races):
    """Check current status of weather data collection (backward compatibility)"""
    return check_session_status(weather_provider, target_races, 'date', 'race')

def fetch_historical_weather_data():
    """Fetch weather data for recent F1 races and all session types"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fetch F1 weather data')
    parser.add_argument('--start-year', type=int, default=2020, help='Start year (default: 2020)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year (default: 2024)')
    parser.add_argument('--all', action='store_true', help='Fetch all available races')
    parser.add_argument('--status', action='store_true', help='Only show status, do not fetch')
    parser.add_argument('--max-credits', type=int, default=990, help='Maximum credits to use (default: 990)')
    parser.add_argument('--session', type=str, default='all', 
                       choices=['all', 'race', 'sprint', 'qualifying', 'practice'],
                       help='Session type to fetch (default: all)')
    args, unknown = parser.parse_known_args()
    
    # Initialize weather provider with the API key
    weather_provider = F1WeatherProvider(api_key=API_KEY, provider='visual_crossing')
    
    # Load F1 race data
    print("F1 Weather Data Fetcher")
    print("="*60)
    print("\nLoading F1 race data...")
    data = load_f1db_data()
    races_df = data['races']
    circuits_df = data['circuits']
    
    # Merge races with circuits to get circuit names and coordinates
    races_with_circuits = races_df.merge(
        circuits_df[['id', 'name', 'placeName', 'latitude', 'longitude']], 
        left_on='circuitId', 
        right_on='id', 
        suffixes=('', '_circuit')
    )
    races_with_circuits['circuit_name'] = races_with_circuits['name_circuit']
    
    # Get races from specified range
    if args.all:
        recent_races = races_with_circuits.copy()
    else:
        recent_races = races_with_circuits[(races_with_circuits['year'] >= args.start_year) & (races_with_circuits['year'] <= args.end_year)]
    
    recent_races = recent_races.sort_values('date')
    
    print(f"\nFound {len(recent_races)} races from {recent_races['year'].min()}-{recent_races['year'].max()}")
    
    # Define session types and their date columns
    session_configs = {
        'race': [('date', 'race')],
        'sprint': [('sprintRaceDate', 'sprint')],
        'qualifying': [
            ('qualifyingDate', 'qualifying'),
            ('qualifying1Date', 'qualifying'),
            ('qualifying2Date', 'qualifying'),
            ('sprintQualifyingDate', 'qualifying')
        ],
        'practice': [
            ('freePractice1Date', 'practice'),
            ('freePractice2Date', 'practice'),
            ('freePractice3Date', 'practice'),
            ('freePractice4Date', 'practice')
        ]
    }
    
    # Determine which sessions to fetch
    if args.session == 'all':
        sessions_to_fetch = list(session_configs.keys())
    else:
        sessions_to_fetch = [args.session]
    
    print(f"Session types to fetch: {', '.join(sessions_to_fetch)}")
    
    # Check status for all session types
    if args.status:
        check_all_sessions_status(weather_provider, recent_races, session_configs, sessions_to_fetch)
        return
    
    # Calculate how many items we can fetch today
    credits_per_fetch = 24  # Based on observed usage
    max_fetches_today = args.max_credits // credits_per_fetch
    
    print(f"\n" + "="*60)
    print(f"Fetching Weather Data")
    print(f"="*60)
    print(f"Daily credit limit: 1000")
    print(f"Max credits to use: {args.max_credits}")
    print(f"Credits per fetch: ~{credits_per_fetch}")
    print(f"Max fetches today: {max_fetches_today}")
    
    # Fetch weather for each session type
    weather_fetched = 0
    weather_cached = 0
    weather_errors = 0
    credits_used = 0
    
    # Track items to fetch
    fetch_items = []
    
    # Build list of all items to fetch
    for session_type in sessions_to_fetch:
        for date_col, session_label in session_configs[session_type]:
            if date_col not in recent_races.columns:
                continue
                
            # Get races with valid dates for this column
            valid_races = recent_races[recent_races[date_col].notna()]
            
            for _, race in valid_races.iterrows():
                race_name = race.get('name', race.get('officialName', 'Unknown'))
                circuit_name = race.get('circuit_name', race.get('circuitId', 'Unknown'))
                session_date = str(race[date_col])[:10]  # YYYY-MM-DD format
                
                # Check if already cached using circuit name
                if not weather_provider._check_csv_cache(circuit_name, session_date, session_type):
                    fetch_items.append({
                        'race_name': race_name,
                        'circuit_name': circuit_name,
                        'date': session_date,
                        'session_type': session_type,
                        'date_column': date_col,
                        'year': race['year'],
                        'latitude': race.get('latitude'),
                        'longitude': race.get('longitude')
                    })
    
    print(f"\nTotal items to fetch: {len(fetch_items)}")
    print(f"Items that can be fetched today: {min(len(fetch_items), max_fetches_today)}")
    
    # Sort by year and date to fetch oldest first
    fetch_items.sort(key=lambda x: (x['year'], x['date']))
    
    # Fetch weather data
    import time
    for idx, item in enumerate(fetch_items, 1):
        if weather_fetched >= max_fetches_today:
            print(f"\n⚠️  Daily limit reached ({weather_fetched} items fetched, ~{credits_used} credits used)")
            print(f"Run again tomorrow to continue fetching remaining {len(fetch_items) - weather_fetched} items")
            break
            
        race_name = item['race_name']
        circuit_name = item['circuit_name']
        session_date = item['date']
        session_type = item['session_type']
        date_column = item['date_column']
        
        print(f"\n[{idx}/{len(fetch_items)}] Fetching {session_type} weather for {circuit_name} ({race_name}) on {session_date}...")
        
        try:
            # If we have coordinates, use them directly
            if item.get('latitude') and item.get('longitude'):
                # Temporarily update coordinates for this circuit
                normalized = circuit_name.lower().replace(' ', '_').replace('-', '_')
                weather_provider.circuit_coordinates[normalized] = (item['latitude'], item['longitude'])
            
            weather_data = weather_provider.get_weather_for_race(circuit_name, session_date, session_type=session_type)
            print(f"  ✓ Fetched from API - Temp: {weather_data['temperature']:.1f}°C, "
                  f"Rain prob: {weather_data['rain_probability']:.1%}")
            weather_fetched += 1
            credits_used = weather_fetched * credits_per_fetch
            
            # Rate limit handling
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
    
    # Count cached items
    total_cached = 0
    for session_type in sessions_to_fetch:
        cache_file = weather_provider.csv_cache_files.get(session_type)
        if cache_file and cache_file.exists():
            cache_df = pd.read_csv(cache_file)
            total_cached += len(cache_df)
    
    print(f"\n" + "="*60)
    print(f"Summary:")
    print(f"  - Weather data already cached: {total_cached}")
    print(f"  - Weather data fetched from API: {weather_fetched}")
    print(f"  - Errors encountered: {weather_errors}")
    print(f"  - Items remaining to fetch: {len(fetch_items) - weather_fetched}")
    
    # Calculate API usage
    print(f"\nAPI Credits Usage:")
    print(f"  - Credits used today: ~{credits_used} (at ~{credits_per_fetch} credits per item)")
    print(f"  - Daily credit limit: 1000")
    print(f"  - Credits remaining today: ~{1000 - credits_used}")
    
    # Calculate remaining work
    items_remaining = len(fetch_items) - weather_fetched
    if items_remaining > 0:
        credits_needed = items_remaining * credits_per_fetch
        days_needed = (credits_needed + 999) // 1000  # Ceiling division
        print(f"\nRemaining work:")
        print(f"  - Items still to fetch: {items_remaining}")
        print(f"  - Credits needed: {credits_needed}")
        print(f"  - Days needed: {days_needed}")
        print(f"\nTo continue fetching, run:")
        print(f"  python fetch_weather_data.py --start-year {args.start_year} --end-year {args.end_year} --session {args.session}")
    else:
        print(f"\n✅ All {args.session} weather data fetched successfully!")
    
    # Show sample of cached data for each session type
    print("\nCached Data Summary:")
    for session_type in sessions_to_fetch:
        cache_file = weather_provider.csv_cache_files.get(session_type)
        if cache_file and cache_file.exists():
            cache_df = pd.read_csv(cache_file)
            print(f"\n{session_type.upper()} weather data: {len(cache_df)} records")
            
            if len(cache_df) > 0:
                # Show statistics
                print(f"  - Average temperature: {cache_df['temperature'].mean():.1f}°C")
                print(f"  - Average rain probability: {cache_df['rain_probability'].mean():.1%}")
                wet_count = cache_df['is_wet_race'].sum()
                print(f"  - Wet sessions: {wet_count} ({cache_df['is_wet_race'].mean():.1%})")


def check_specific_race(circuit_name: str, date: str):
    """Check weather for a specific race"""
    weather_provider = F1WeatherProvider(api_key=API_KEY, provider='visual_crossing')
    
    print(f"\nChecking weather for {circuit_name} on {date}...")
    weather = weather_provider.get_weather_for_race(circuit_name, date)
    
    print("\nWeather data:")
    for key, value in weather.items():
        print(f"  - {key}: {value}")


def main(year=None, force_update=False):
    """Main function for GitHub workflow integration"""
    import argparse
    
    # If called from workflow, use provided parameters
    if year:
        # Set up args for fetch_historical_weather_data
        import sys
        sys.argv = ['fetch_weather_data.py', '--start-year', str(year), '--end-year', str(year)]
        if force_update:
            sys.argv.append('--force')
    
    # Run the historical weather fetcher
    fetch_historical_weather_data()


if __name__ == "__main__":
    # Fetch historical data
    fetch_historical_weather_data()