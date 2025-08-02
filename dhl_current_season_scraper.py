#!/usr/bin/env python3
"""
Simplified DHL F1 Pit Stop Scraper - Current Season Focus
Extracts pit stop data from the current F1 season
"""

import json
import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CurrentSeasonScraper:
    def __init__(self, visible=False):
        """Initialize scraper
        Args:
            visible: Show browser window (useful for debugging)
        """
        self.url = "https://inmotion.dhl/en/formula-1/fastest-pit-stop-award"
        self.data = []
        self.visible = visible
        
    def setup_driver(self):
        """Setup Chrome driver"""
        options = webdriver.ChromeOptions()
        if not self.visible:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)
        
    def extract_pit_stops(self):
        """Extract pit stop data from current page - prioritize table for complete data"""
        # First, check if we need to click on the Table tab
        try:
            # Look for tab buttons - try multiple selectors
            tab_selectors = [
                "a[href='#element_event_grid_row2_col0_table']",  # Specific tab href from your HTML
                "a[data-toggle='tab'][href*='table']",
                "li a[href*='table']",
                ".nav-tabs a:contains('Table')",
                ".nav-tabs li:nth-child(2) a"  # Second tab is usually Table
            ]
            
            table_tab = None
            for selector in tab_selectors:
                try:
                    if 'contains' in selector:
                        # Use XPath for contains
                        table_tab = self.driver.find_element(By.XPATH, "//a[contains(text(),'Table')]")
                    else:
                        table_tab = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if table_tab:
                        break
                except:
                    continue
                    
            if table_tab:
                # Check if tab is already active
                parent_li = table_tab.find_element(By.XPATH, "..")
                if 'active' not in parent_li.get_attribute('class'):
                    logger.info("Clicking on Table tab to see all drivers...")
                    self.driver.execute_script("arguments[0].click();", table_tab)  # Use JS click to avoid interception
                    time.sleep(2)  # Wait for tab content to load
                else:
                    logger.info("Table tab is already active")
        except Exception as e:
            logger.warning(f"Could not find or click table tab: {e}")
            
        # Try to extract from table (has all drivers)
        table_data = self.extract_from_table()
        if table_data:
            return table_data
            
        # Fallback to chart data if table not found
        try:
            canvas = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "canvas.f1-award-chart"))
            )
            
            json_data = canvas.get_attribute("data-chart-data")
            if json_data:
                stops = json.loads(json_data)
                logger.info(f"✓ Found {len(stops)} pit stops in chart data (top 10 only)")
                return stops
                
        except TimeoutException:
            logger.warning("Could not find any pit stop data")
            
        return []
        
    def extract_from_table(self):
        """Extract data from HTML table - contains ALL drivers"""
        try:
            # Give the page time to update
            time.sleep(3)
            
            # Find the table - simplified approach
            table = None
            try:
                table = self.driver.find_element(By.CSS_SELECTOR, "table.f1-award-table")
                if not table:
                    logger.warning("No table found")
                    return []
            except:
                logger.warning("Could not find f1-award-table")
                return []
            
            # Get all rows from tbody
            try:
                tbody = table.find_element(By.TAG_NAME, "tbody")
                rows = tbody.find_elements(By.TAG_NAME, "tr")
            except:
                logger.warning("Could not find tbody or rows")
                return []
                
            logger.info(f"Found {len(rows)} rows in table body")
            
            stops = []
            for idx, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    # Skip if not enough cells or all cells are empty
                    if len(cells) < 6:
                        continue
                        
                    # Check if cells have content
                    cell_texts = [cell.text.strip() for cell in cells]
                    if not any(cell_texts):  # Skip if all cells are empty
                        continue
                        
                    # Try to parse the data
                    try:
                        pos = int(cell_texts[0]) if cell_texts[0] else None
                        team = cell_texts[1]
                        driver = cell_texts[2]
                        time_str = cell_texts[3]
                        lap = int(cell_texts[4]) if cell_texts[4] else None
                        points_str = cell_texts[5].replace('*', '').replace(' ', '')
                        points = int(points_str) if points_str else 0
                        
                        if pos and driver and time_str:  # Minimum required fields
                            stop = {
                                "position": pos,
                                "team": team,
                                "driver": driver,
                                "time": float(time_str),
                                "lap": lap,
                                "points": points
                            }
                            stops.append(stop)
                            
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Row {idx} parse error: {e}")
                        logger.debug(f"Cell texts: {cell_texts}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Error processing row {idx}: {e}")
                    continue
                    
            logger.info(f"✓ Successfully extracted {len(stops)} pit stops from table")
            return stops
            
        except Exception as e:
            logger.error(f"Failed to extract table data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
            
    def handle_cookie_consent(self):
        """Handle cookie consent popup if present"""
        try:
            # Wait a bit to see if cookie consent appears
            time.sleep(2)
            
            # Try to find and remove the privacy iframe
            self.driver.execute_script("""
                var iframe = document.querySelector('iframe[src*="privacy"]');
                if (iframe) {
                    iframe.remove();
                    console.log('Privacy iframe removed');
                }
            """)
            
            # Also try to find any cookie accept buttons
            try:
                accept_buttons = [
                    "//button[contains(text(), 'Accept')]",
                    "//button[contains(text(), 'OK')]",
                    "//button[contains(@class, 'accept')]",
                    "//button[contains(@class, 'agree')]",
                    "//a[contains(text(), 'Accept')]"
                ]
                
                for xpath in accept_buttons:
                    try:
                        btn = self.driver.find_element(By.XPATH, xpath)
                        if btn.is_displayed():
                            btn.click()
                            logger.info("Clicked cookie consent button")
                            time.sleep(1)
                            break
                    except:
                        continue
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Cookie consent handling: {e}")
    
    def get_race_options(self):
        """Get all available races from the modal"""
        try:
            # Handle cookie consent first
            self.handle_cookie_consent()
            
            # Click the "Choose Event" button - use JavaScript to avoid interception
            choose_event_btn = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Choose Event')]"))
            )
            logger.info("Clicking 'Choose Event' button...")
            self.driver.execute_script("arguments[0].click();", choose_event_btn)
            
            # Wait for modal to appear
            time.sleep(2)
            
            # Wait for modal content to be visible
            self.wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.modal__content"))
            )
            
            # Get all race options from the modal
            race_options = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.form-item-radio input[type='radio']"))
            )
            
            races = []
            for option in race_options:
                try:
                    value = option.get_attribute('value')
                    # Get the label text (race name)
                    label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{option.get_attribute('id')}']")
                    # Extract just the circuit name (skip the flag span)
                    spans = label.find_elements(By.TAG_NAME, "span")
                    race_name = spans[-1].text.strip() if spans else label.text.strip()
                    if value and race_name:
                        races.append((value, race_name))
                        logger.debug(f"Found race: {race_name} (value: {value})")
                except Exception as e:
                    logger.warning(f"Error parsing race option: {e}")
                    continue
                
            logger.info(f"✓ Found {len(races)} races in modal")
            for idx, (val, name) in enumerate(races):
                logger.debug(f"  {idx+1}. {name}")
            
            # Close modal for now using either Cancel button or X button
            try:
                cancel_btn = self.driver.find_element(By.CSS_SELECTOR, "button.js-modal-close, button[aria-label='Close']")
                self.driver.execute_script("arguments[0].click();", cancel_btn)
            except:
                # Try the Cancel button specifically
                try:
                    cancel_btn = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Cancel')]")
                    self.driver.execute_script("arguments[0].click();", cancel_btn)
                except:
                    logger.warning("Could not find modal close button, continuing anyway")
            
            time.sleep(1)
            
            return races
            
        except Exception as e:
            logger.error(f"Error finding races: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
            
    def get_existing_races(self):
        """Get list of races we already have data for from the latest CSV file"""
        existing_races = set()
        
        try:
            # Find the latest CSV file in the data/dhl directory
            data_dir = Path("data/dhl")
            if not data_dir.exists():
                return existing_races
                
            # Get all CSV files and find the most recent one
            csv_files = list(data_dir.glob("*.csv"))
            if not csv_files:
                return existing_races
                
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Checking existing data from: {latest_file}")
            
            # Read the CSV and get unique race names
            df = pd.read_csv(latest_file)
            if 'race' in df.columns:
                existing_races = set(df['race'].unique())
                logger.info(f"Found {len(existing_races)} races in existing data")
                for race in sorted(existing_races):
                    logger.debug(f"  - {race}")
            
        except Exception as e:
            logger.warning(f"Could not read existing data: {e}")
            
        return existing_races
    
    def scrape_all_races(self):
        """Scrape data for all races in current season"""
        logger.info("\n🏎️  DHL F1 Pit Stop Data Scraper")
        logger.info("=" * 40)
        
        # Check for existing data
        existing_races = self.get_existing_races()
        
        self.setup_driver()
        
        try:
            logger.info("Loading DHL Formula 1 page...")
            self.driver.get(self.url)
            time.sleep(5)  # Give more time for page to fully load
            
            # Handle cookie consent immediately after page load
            self.handle_cookie_consent()
            
            # Wait for page elements to be present
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "canvas.f1-award-chart, table.f1-award-table"))
                )
            except:
                logger.warning("Page elements taking long to load...")
            
            # Get all available races
            races = self.get_race_options()
            
            if not races:
                logger.error("No races found in modal! Cannot continue.")
                return
                
            # Filter out races we already have data for
            races_to_process = []
            for value, name in races:
                if name not in existing_races:
                    races_to_process.append((value, name))
                else:
                    logger.info(f"Skipping {name} - already have data")
                    
            if not races_to_process:
                logger.info("✅ All races already processed! No new data to fetch.")
                return
                
            # Store the total number of races for tracking
            total_races = len(races_to_process)
            processed_races = 0
            
            logger.info(f"\n📊 Processing {total_races} new races (skipping {len(races) - total_races} existing)")
            logger.info("-" * 40)
            
            # Process each race
            for i, (value, name) in enumerate(races_to_process):
                logger.info(f"\nProcessing race {i+1}/{total_races}: {name}")
                
                try:
                    # Click Choose Event button again using JavaScript
                    choose_event_btn = self.wait.until(
                        EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Choose Event')]"))
                    )
                    self.driver.execute_script("arguments[0].click();", choose_event_btn)
                    time.sleep(2)  # Give more time for modal
                    
                    # Wait for modal to be visible
                    self.wait.until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, "div.modal__content"))
                    )
                    
                    # Select the race radio button using JavaScript
                    radio_btn = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, f"input[type='radio'][value='{value}']"))
                    )
                    self.driver.execute_script("arguments[0].click();", radio_btn)
                    time.sleep(0.5)  # Small pause after selection
                    
                    # Click Apply button using JavaScript
                    apply_btn = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "button.ok-btn"))
                    )
                    self.driver.execute_script("arguments[0].click();", apply_btn)
                    
                    # Wait for modal to close
                    time.sleep(2)
                    
                    # Wait for page to reload with new data
                    # Look for a change in the page - wait for the table to have content
                    try:
                        self.wait.until(
                            lambda driver: len(driver.find_elements(By.CSS_SELECTOR, "table.f1-award-table tbody tr")) > 0
                        )
                        # Additional wait for data to fully load
                        time.sleep(3)
                    except:
                        logger.warning(f"Timeout waiting for data to load for {name}")
                        time.sleep(5)  # Give it more time
                    
                    # Extract data
                    stops = self.extract_pit_stops()
                    
                    # Only process if we got data
                    if not stops:
                        logger.warning(f"No pit stop data found for {name}")
                        continue
                    
                    # Add race info to each stop
                    for stop in stops:
                        # Check if data is from table (simpler structure) or JSON (complex structure)
                        if 'driver' in stop and isinstance(stop['driver'], str):
                            # Table data - already has simple structure
                            formatted_stop = {
                                'position': stop.get('position'),
                                'driver': stop.get('driver'),
                                'team': stop.get('team'),
                                'time': stop.get('time'),
                                'lap': stop.get('lap'),
                                'points': stop.get('points'),
                                'race': name,  # Use the actual race name from the modal
                                'year': datetime.now().year
                            }
                        else:
                            # JSON data - needs parsing
                            formatted_stop = {
                                'position': stop.get('position', stop.get('id')),
                                'driver_number': stop.get('driverNr'),
                                'driver_tla': stop.get('tla'),
                                'driver': f"{stop.get('firstName', '')} {stop.get('lastName', '')}".strip(),
                                'team': stop.get('team'),
                                'time': stop.get('duration'),
                                'lap': stop.get('lap'),
                                'points': stop.get('points'),
                                'start_time': stop.get('startTime', {}).get('date') if isinstance(stop.get('startTime'), dict) else None,
                                'irregular': stop.get('irregular', False),
                                'race': name,
                                'year': datetime.now().year
                            }
                        self.data.append(formatted_stop)
                        
                    logger.info(f"✓ Extracted {len(stops)} pit stops from {name}")
                    processed_races += 1
                    
                    # Add a small delay between races to avoid overloading
                    if i < total_races - 1:  # Don't wait after the last race
                        time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"✗ Failed to process {name}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # Continue to next race even if this one fails
                    continue
            
            logger.info(f"\n📊 Processed {processed_races}/{total_races} races successfully")
                    
        except Exception as e:
            logger.error(f"Fatal error during scraping: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        finally:
            self.driver.quit()
            
        logger.info(f"\n✅ Successfully extracted {len(self.data)} pit stop records!")
        
    def save_data(self, filename=None, format='csv', append_mode=True):
        """Save extracted data"""
        if not self.data:
            logger.warning("No data to save!")
            return
            
        df_new = pd.DataFrame(self.data)
        
        # Clean up data - check which time field exists
        if 'time' in df_new.columns:
            df_new = df_new.dropna(subset=['time'])  # Remove entries without times
        elif 'duration' in df_new.columns:
            # Rename duration to time for consistency
            df_new['time'] = df_new['duration']
            df_new = df_new.dropna(subset=['time'])
        
        # Create output directory structure
        output_dir = Path("data/dhl")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we should append to existing file
        if append_mode and format == 'csv':
            # Find the latest CSV file
            csv_files = list(output_dir.glob("*.csv"))
            if csv_files:
                latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                
                # Read existing data
                df_existing = pd.read_csv(latest_file)
                
                # Combine with new data
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                
                # Remove duplicates based on race, driver, and position
                df_combined = df_combined.drop_duplicates(subset=['race', 'driver', 'position'], keep='last')
                
                # Sort by race and position
                sort_cols = []
                if 'race' in df_combined.columns:
                    sort_cols.append('race')
                if 'position' in df_combined.columns:
                    sort_cols.append('position')
                if sort_cols:
                    df_combined = df_combined.sort_values(sort_cols)
                
                # Save back to the same file
                df_combined.to_csv(latest_file, index=False)
                logger.info(f"📁 Updated existing file: {latest_file}")
                logger.info(f"   Added {len(df_new)} new records, total: {len(df_combined)} records")
                return latest_file
        
        # Otherwise create a new file
        # Sort by race and position if they exist
        sort_cols = []
        if 'race' in df_new.columns:
            sort_cols.append('race')
        if 'position' in df_new.columns:
            sort_cols.append('position')
        if sort_cols:
            df_new = df_new.sort_values(sort_cols)
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dhl_current_season_pitstops_{timestamp}"
            
        if format == 'csv':
            filepath = output_dir / f"{filename}.csv"
            df_new.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = output_dir / f"{filename}.json"
            df_new.to_json(filepath, orient='records', indent=2)
        else:
            filepath = output_dir / f"{filename}.xlsx"
            df_new.to_excel(filepath, index=False)
            
        logger.info(f"📁 Data saved to: {filepath}")
        return filepath
        
    def show_summary(self):
        """Display summary statistics"""
        if not self.data:
            return
            
        df = pd.DataFrame(self.data)
        df_clean = df.dropna(subset=['time'])
        
        print("\n📊 SUMMARY STATISTICS")
        print("=" * 40)
        print(f"Total pit stops: {len(df_clean)}")
        print(f"Races covered: {df_clean['race'].nunique()}")
        print(f"Fastest stop: {df_clean['time'].min():.2f}s")
        print(f"Average time: {df_clean['time'].mean():.2f}s")
        
        print("\n🏆 FASTEST STOPS BY RACE")
        print("-" * 40)
        fastest = df_clean.loc[df_clean.groupby('race')['time'].idxmin()]
        for _, row in fastest.iterrows():
            print(f"{row['race'][:25]:25} | {row['driver']:15} | {row['team']:12} | {row['time']:.2f}s")
            
        print("\n🏎️ TEAM AVERAGES")
        print("-" * 40)
        team_avg = df_clean.groupby('team')['time'].agg(['mean', 'min', 'count']).sort_values('mean')
        for team, stats in team_avg.iterrows():
            print(f"{team:15} | Avg: {stats['mean']:.3f}s | Best: {stats['min']:.2f}s | Stops: {int(stats['count'])}")

# Quick usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape current season DHL pit stop data")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--output", help="Output filename (without extension)")
    parser.add_argument("--format", choices=['csv', 'json', 'excel'], default='csv')
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-append", action="store_true", help="Create new file instead of appending to existing")
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run scraper
    scraper = CurrentSeasonScraper(visible=args.visible)
    scraper.scrape_all_races()
    scraper.save_data(filename=args.output, format=args.format, append_mode=not args.no_append)
    
    if args.summary:
        scraper.show_summary()
        
    print("\nReady for PrizePicks analysis! 🎯")