#!/usr/bin/env python3
"""
Test script for the new DHL API scraper
"""

from dhl_api_scraper import DHLAPIScraper
import pandas as pd

def test_api_scraper():
    """Test the DHL API scraper"""
    
    print("🧪 Testing DHL API Scraper")
    print("=" * 50)
    
    scraper = DHLAPIScraper()
    
    # Test 1: Get available events
    print("1️⃣ Testing event list retrieval...")
    events = scraper.get_available_events()
    
    if events:
        print(f"✅ Found {len(events)} events")
        print("Sample events:")
        for event in events[:3]:
            print(f"   ID: {event['id']} | Name: {event['name']}")
    else:
        print("❌ No events found")
        return False
    
    # Test 2: Get data for one event
    if events:
        test_event = events[0]
        print(f"\n2️⃣ Testing data extraction for: {test_event['name']}")
        
        pit_stops = scraper.get_event_data(test_event['id'], test_event['name'])
        
        if pit_stops:
            print(f"✅ Extracted {len(pit_stops)} pit stops")
            
            # Show sample data
            print("\nSample pit stop data:")
            for stop in pit_stops[:3]:
                print(f"   Pos {stop['position']}: {stop['driver']} ({stop['team']}) - {stop['time']}s")
                
            # Test race mapping
            print(f"\n3️⃣ Testing race mapping for: {test_event['name']}")
            race_id, circuit_id = scraper._map_dhl_race_to_f1db(test_event['name'], 2025)
            
            if race_id:
                print(f"✅ Mapped to race ID: {race_id}, circuit ID: {circuit_id}")
            else:
                print(f"⚠️  No race mapping found (circuit: {circuit_id})")
                
        else:
            print("❌ No pit stop data extracted")
            return False
    
    # Test 3: Check F1DB data loading
    print(f"\n4️⃣ Testing F1DB data loading...")
    if not scraper.f1db_data['drivers'].empty:
        print(f"✅ Loaded {len(scraper.f1db_data['drivers'])} drivers")
        print(f"✅ Loaded {len(scraper.f1db_data['races'])} races")
        print(f"✅ Loaded {len(scraper.f1db_data['circuits'])} circuits")
    else:
        print("❌ F1DB data not loaded properly")
        return False
    
    print(f"\n5️⃣ Testing driver mapping...")
    test_drivers = ['Verstappen', 'Hamilton', 'Leclerc']
    for driver in test_drivers:
        driver_id = scraper._map_driver_to_id(driver)
        if driver_id:
            print(f"✅ {driver} -> {driver_id}")
        else:
            print(f"❌ {driver} -> No mapping")
    
    print(f"\n6️⃣ Testing constructor mapping...")
    test_teams = ['Red Bull', 'Ferrari', 'Mercedes']
    for team in test_teams:
        constructor_id = scraper._map_team_to_constructor_id(team)
        if constructor_id:
            print(f"✅ {team} -> {constructor_id}")
        else:
            print(f"❌ {team} -> No mapping")
    
    print(f"\n✅ All tests completed successfully!")
    return True

def test_full_scrape():
    """Test a small scrape operation"""
    print(f"\n🚀 Testing full scrape (first 2 events only)...")
    
    scraper = DHLAPIScraper()
    
    # Get events and limit to first 2 for testing
    events = scraper.get_available_events()
    if not events:
        print("❌ No events available")
        return
    
    # Process only first 2 events for testing
    test_events = events[:2]
    print(f"Testing with {len(test_events)} events...")
    
    for event in test_events:
        pit_stops = scraper.get_event_data(event['id'], event['name'])
        
        if pit_stops:
            # Add mappings
            year = 2025
            race_id, circuit_id = scraper._map_dhl_race_to_f1db(event['name'], year)
            
            for stop in pit_stops:
                stop['driver_id'] = scraper._map_driver_to_id(stop['driver'], stop['team'])
                stop['constructor_id'] = scraper._map_team_to_constructor_id(stop['team'])
                stop['race_id'] = race_id
                stop['circuit_id'] = circuit_id
                stop['year'] = year
                
            scraper.all_data.extend(pit_stops)
    
    if scraper.all_data:
        print(f"✅ Successfully processed {len(scraper.all_data)} pit stops")
        
        # Show mapping stats
        df = pd.DataFrame(scraper.all_data)
        scraper.show_mapping_stats(df)
        
        # Save test data
        filepath = scraper.save_data("test_api_scrape", "csv")
        print(f"💾 Test data saved to: {filepath}")
        
    else:
        print("❌ No data collected")

if __name__ == "__main__":
    # Run basic tests
    if test_api_scraper():
        # If basic tests pass, try a small scrape
        test_full_scrape()
    else:
        print("❌ Basic tests failed")