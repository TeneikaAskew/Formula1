#!/usr/bin/env python3
"""Test Jolpica fetcher with exponential backoff"""

import sys
sys.path.append('/workspace')
from jolpica_laps_fetcher import JolpicaLapsFetcher
import logging

# Enable debug logging to see rate limit behavior
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize fetcher
fetcher = JolpicaLapsFetcher()

print("Testing Jolpica Fetcher with Exponential Backoff")
print("=" * 50)
print(f"Burst limit: {fetcher.BURST_LIMIT} requests/second")
print(f"Sustained limit: {fetcher.SUSTAINED_LIMIT} requests/hour")
print()

# Test fetching a few races to see rate limiting in action
try:
    # Try to fetch some 2021 races (should work after backoff)
    for round_num in range(1, 4):
        print(f"\nFetching 2021 Round {round_num}...")
        
        # Show rate limit status before
        status = fetcher.get_rate_limit_status()
        print(f"Before - Requests in last hour: {status['requests_last_hour']}, Backoff: {status['current_backoff']}s")
        
        # Fetch race
        race_data = fetcher.fetch_race_laps(2021, round_num)
        
        if race_data:
            print(f"✓ Successfully fetched {race_data['raceName']} - {len(race_data['laps'])} laps")
        else:
            print("✗ Failed to fetch race data")
        
        # Show rate limit status after
        status = fetcher.get_rate_limit_status()
        print(f"After - Requests in last hour: {status['requests_last_hour']}, Backoff: {status['current_backoff']}s")
        
except KeyboardInterrupt:
    print("\n\nTest interrupted by user")
except Exception as e:
    print(f"\nError: {e}")
    
# Final status
print("\n" + "=" * 50)
status = fetcher.get_rate_limit_status()
print("Final Rate Limit Status:")
for key, value in status.items():
    print(f"  {key}: {value}")