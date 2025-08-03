#!/usr/bin/env python3
"""
Test script to verify table extraction from DHL pit stop page
"""

from dhl_current_season_scraper import CurrentSeasonScraper
import time

def test_single_race_extraction():
    """Test extraction from a single race to verify table data"""
    
    print("🧪 Testing DHL Table Extraction")
    print("=" * 50)
    
    # Create scraper with visible browser for debugging
    scraper = CurrentSeasonScraper(visible=True)
    
    try:
        scraper.setup_driver()
        
        # Navigate to the page
        print("📍 Loading DHL page...")
        scraper.driver.get(scraper.url)
        time.sleep(5)
        
        # Handle cookies
        scraper.handle_cookie_consent()
        
        # Try to extract data from current page (should be latest race)
        print("🔍 Extracting pit stop data...")
        pit_stops = scraper.extract_pit_stops()
        
        if pit_stops:
            print(f"\n✅ Successfully extracted {len(pit_stops)} pit stops!")
            
            # Show first few entries
            print("\n📊 Sample data (first 5 entries):")
            print("-" * 60)
            print(f"{'Pos':<4} {'Driver':<15} {'Team':<12} {'Time':<6} {'Lap':<4} {'Pts':<4}")
            print("-" * 60)
            
            for i, stop in enumerate(pit_stops[:5]):
                print(f"{stop.get('position', 'N/A'):<4} "
                      f"{stop.get('driver', 'N/A'):<15} "
                      f"{stop.get('team', 'N/A'):<12} "
                      f"{stop.get('time', 'N/A'):<6} "
                      f"{stop.get('lap', 'N/A'):<4} "
                      f"{stop.get('points', 'N/A'):<4}")
            
            if len(pit_stops) > 5:
                print(f"... and {len(pit_stops) - 5} more entries")
                
            # Show last few entries to verify we got the full table
            if len(pit_stops) > 10:
                print(f"\n📊 Last 3 entries (showing positions {pit_stops[-3]['position']}-{pit_stops[-1]['position']}):")
                print("-" * 60)
                for stop in pit_stops[-3:]:
                    print(f"{stop.get('position', 'N/A'):<4} "
                          f"{stop.get('driver', 'N/A'):<15} "
                          f"{stop.get('team', 'N/A'):<12} "
                          f"{stop.get('time', 'N/A'):<6} "
                          f"{stop.get('lap', 'N/A'):<4} "
                          f"{stop.get('points', 'N/A'):<4}")
            
            # Verify we have more than 10 entries (proving it's not just chart data)
            if len(pit_stops) > 10:
                print(f"\n✅ PASS: Got {len(pit_stops)} entries (more than chart's 10 limit)")
            else:
                print(f"\n⚠️  WARNING: Only got {len(pit_stops)} entries - might be chart data only!")
                
        else:
            print("❌ No pit stop data extracted!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Wait a bit so user can see the browser
        input("\nPress Enter to close browser...")
        scraper.driver.quit()

if __name__ == "__main__":
    test_single_race_extraction()