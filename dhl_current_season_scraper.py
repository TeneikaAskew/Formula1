#!/usr/bin/env python3
"""
Enhanced DHL F1 Pit Stop Scraper with F1DB ID Mapping
Extracts pit stop data with proper driver/constructor/race ID mapping
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
import os

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
        self.f1db_data = self._load_f1db_data()
        
    def _load_f1db_data(self):
        """Load F1DB reference data for mapping"""
        f1db_data = {
            'drivers': pd.DataFrame(),
            'constructors': pd.DataFrame(),
            'races': pd.DataFrame(),
            'circuits': pd.DataFrame()
        }
        
        # Try different data paths
        data_paths = ["data/f1db", "../data/f1db", "../../data/f1db"]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    # Load each file separately to handle missing files gracefully
                    for file_type in ['drivers', 'constructors', 'races', 'circuits']:
                        file_path = os.path.join(path, f'{file_type}.csv')
                        if os.path.exists(file_path):
                            f1db_data[file_type] = pd.read_csv(file_path)
                            logger.debug(f"Loaded {file_type} from {file_path}")
                    
                    if not f1db_data['drivers'].empty:
                        logger.info(f"✓ Loaded F1DB data from {path}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Error loading F1DB data from {path}: {e}")
        
        return f1db_data
    
    def _map_driver_to_id(self, driver_name, team_name=None):
        """Map driver name to F1DB driver ID"""
        if self.f1db_data['drivers'].empty or not driver_name:
            return None
        
        drivers = self.f1db_data['drivers']
        driver_name_lower = driver_name.lower().strip()
        
        # Special mappings for current drivers
        special_mappings = {
            'verstappen': 'max-verstappen',
            'perez': 'sergio-perez',
            'pérez': 'sergio-perez',
            'hamilton': 'lewis-hamilton',
            'russell': 'george-russell',
            'leclerc': 'charles-leclerc',
            'sainz': 'carlos-sainz-jr',
            'norris': 'lando-norris',
            'piastri': 'oscar-piastri',
            'alonso': 'fernando-alonso',
            'stroll': 'lance-stroll',
            'ocon': 'esteban-ocon',
            'gasly': 'pierre-gasly',
            'albon': 'alex-albon',
            'sargeant': 'logan-sargeant',
            'colapinto': 'franco-colapinto',
            'bottas': 'valtteri-bottas',
            'zhou': 'guanyu-zhou',
            'magnussen': 'kevin-magnussen',
            'hulkenberg': 'nico-hulkenberg',
            'hülkenberg': 'nico-hulkenberg',
            'ricciardo': 'daniel-ricciardo',
            'tsunoda': 'yuki-tsunoda',
            'lawson': 'liam-lawson',
            'bearman': 'oliver-bearman'
        }
        
        # Try last name mapping first
        parts = driver_name.split()
        if parts:
            last_name = parts[-1].lower()
            if last_name in special_mappings:
                return special_mappings[last_name]
        
        # Try exact match
        exact_match = drivers[drivers['name'].str.lower() == driver_name_lower]
        if not exact_match.empty:
            return exact_match.iloc[0]['id']
        
        # Try last name match in database
        if parts:
            last_name_match = drivers[drivers['lastName'].str.lower() == parts[-1].lower()]
            if not last_name_match.empty:
                return last_name_match.iloc[0]['id']
        
        return None
    
    def _map_team_to_constructor_id(self, team_name):
        """Map team name to F1DB constructor ID"""
        if not team_name:
            return None
        
        team_name_lower = team_name.lower().strip()
        
        # Current team mappings
        team_mappings = {
            'red bull': 'red-bull',
            'red bull racing': 'red-bull',
            'mercedes': 'mercedes',
            'mercedes amg': 'mercedes',
            'ferrari': 'ferrari',
            'mclaren': 'mclaren',
            'aston martin': 'aston-martin',
            'alpine': 'alpine',
            'williams': 'williams',
            'alphatauri': 'alphatauri',
            'rb': 'rb',  # AlphaTauri renamed to RB
            'visa rb': 'rb',
            'alfa romeo': 'alfa-romeo',
            'sauber': 'sauber',
            'haas': 'haas'
        }
        
        for key, value in team_mappings.items():
            if key in team_name_lower:
                return value
        
        return None
    
    def _map_race_to_id(self, race_name, year=None):
        """Map race name to F1DB race ID"""
        if self.f1db_data['races'].empty or not race_name:
            return None
        
        races = self.f1db_data['races']
        race_name_lower = race_name.lower().strip()
        year = year or datetime.now().year
        
        # Filter by year
        year_races = races[races['year'] == year]
        if year_races.empty:
            return None
        
        # Try matching by common race name patterns
        race_keywords = {
            'bahrain': 'bahrain',
            'saudi': 'saudi',
            'jeddah': 'saudi',
            'australia': 'australia',
            'melbourne': 'australia',
            'japan': 'japan',
            'suzuka': 'japan',
            'china': 'china',
            'shanghai': 'china',
            'miami': 'miami',
            'imola': 'emilia',
            'monaco': 'monaco',
            'spain': 'spain',
            'barcelona': 'spain',
            'canada': 'canada',
            'montreal': 'canada',
            'austria': 'austria',
            'spielberg': 'austria',
            'great britain': 'britain',
            'british': 'britain',
            'silverstone': 'britain',
            'hungary': 'hungary',
            'hungaroring': 'hungary',
            'belgium': 'belgium',
            'spa': 'belgium',
            'netherlands': 'netherlands',
            'dutch': 'netherlands',
            'zandvoort': 'netherlands',
            'italy': 'italy',
            'monza': 'italy',
            'singapore': 'singapore',
            'united states': 'states',
            'austin': 'states',
            'cota': 'states',
            'mexico': 'mexico',
            'brazil': 'brazil',
            'sao paulo': 'brazil',
            'interlagos': 'brazil',
            'las vegas': 'vegas',
            'qatar': 'qatar',
            'abu dhabi': 'abu-dhabi',
            'yas marina': 'abu-dhabi'
        }
        
        # Find matching keyword
        for keyword, search_term in race_keywords.items():
            if keyword in race_name_lower:
                for _, race in year_races.iterrows():
                    if search_term in str(race.get('officialName', '')).lower():
                        return race['id']
        
        return None
    
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
        """Extract pit stop data from current page - ALWAYS prioritize table for complete data"""
        # First, ALWAYS try to click on the Table tab to ensure we get ALL drivers
        try:
            # Wait a bit for page to fully load
            time.sleep(3)
            
            # Look for tab buttons - try multiple selectors
            tab_selectors = [
                "a[href='#element_event_grid_row2_col0_table']",  # Specific tab href
                "a[data-toggle='tab'][href*='table']",
                "li a[href*='table']",
                "ul.nav-tabs a[href*='table']",
                ".nav-tabs li:nth-child(2) a",  # Second tab is usually Table
                "a.nav-link[href*='table']"
            ]
            
            table_tab = None
            for selector in tab_selectors:
                try:
                    table_tab = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if table_tab:
                        logger.debug(f"Found table tab with selector: {selector}")
                        break
                except:
                    continue
            
            # If CSS selectors fail, try XPath
            if not table_tab:
                try:
                    table_tab = self.driver.find_element(By.XPATH, "//a[contains(text(),'Table')]")
                    logger.debug("Found table tab with XPath")
                except:
                    pass
                    
            if table_tab:
                # Always click the tab to ensure table is visible
                logger.info("Clicking on Table tab to see ALL drivers...")
                self.driver.execute_script("arguments[0].click();", table_tab)
                time.sleep(3)  # Give more time for table to load
            else:
                logger.warning("Could not find Table tab - data might be incomplete!")
                
        except Exception as e:
            logger.warning(f"Error handling table tab: {e}")
        
        # ONLY extract from table (has ALL drivers - complete data)
        table_data = self.extract_from_table()
        if table_data:
            logger.info(f"✅ Successfully extracted {len(table_data)} pit stops from TABLE (complete data)")
            return table_data
        
        # NO FALLBACK TO CHART DATA - we only want complete table data
        logger.error("❌ Table extraction failed - NO DATA EXTRACTED (chart data not allowed)")
        return []
        
    def extract_from_table(self):
        """Extract data from HTML table - contains ALL drivers"""
        try:
            # Give the page time to update after tab click
            time.sleep(2)
            
            # Try multiple table selectors
            table_selectors = [
                "table.f1-award-table",
                "table.table",
                "#element_event_grid_row2_col0_table table",
                "div[id*='table'] table",
                "div.tab-pane.active table",
                "table"  # Last resort - any table
            ]
            
            table = None
            for selector in table_selectors:
                try:
                    tables = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    # If multiple tables, find the one with pit stop data
                    for t in tables:
                        try:
                            # Check if table has tbody with rows
                            tbody = t.find_element(By.TAG_NAME, "tbody")
                            rows = tbody.find_elements(By.TAG_NAME, "tr")
                            if len(rows) > 5:  # Should have more than just a few rows
                                table = t
                                logger.debug(f"Found table with {len(rows)} rows using selector: {selector}")
                                break
                        except:
                            continue
                    if table:
                        break
                except:
                    continue
            
            if not table:
                logger.warning("Could not find pit stop table")
                return []
            
            # Get all rows from tbody
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")
                
            logger.info(f"Processing {len(rows)} rows from pit stop table...")
            
            stops = []
            for idx, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    # Skip if not enough cells
                    if len(cells) < 6:
                        continue
                        
                    # Check if cells have content
                    cell_texts = [cell.text.strip() for cell in cells]
                    if not any(cell_texts):  # Skip if all cells are empty
                        continue
                        
                    # Try to parse the data
                    try:
                        # Position (first column)
                        pos_text = cell_texts[0].strip()
                        if not pos_text or not pos_text.isdigit():
                            continue
                        pos = int(pos_text)
                        
                        # Team and Driver
                        team = cell_texts[1].strip()
                        driver = cell_texts[2].strip()
                        
                        # Time (must be a valid float)
                        time_str = cell_texts[3].strip()
                        if not time_str:
                            continue
                        time_val = float(time_str)
                        
                        # Lap
                        lap_text = cell_texts[4].strip()
                        lap = int(lap_text) if lap_text and lap_text.isdigit() else None
                        
                        # Points (handle asterisks and spaces)
                        points_str = cell_texts[5].replace('*', '').replace(' ', '').strip()
                        points = int(points_str) if points_str and points_str.isdigit() else 0
                        
                        # Validate essential fields
                        if pos and driver and time_val:
                            stop = {
                                "position": pos,
                                "team": team,
                                "driver": driver,
                                "time": time_val,
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
            
            if stops:
                logger.info(f"✅ Successfully extracted {len(stops)} pit stops from table")
            else:
                logger.warning("No valid pit stop data found in table")
                
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
                            driver_name = stop.get('driver')
                            team_name = stop.get('team')
                            year = datetime.now().year
                            
                            # Map IDs
                            driver_id = self._map_driver_to_id(driver_name, team_name)
                            constructor_id = self._map_team_to_constructor_id(team_name)
                            race_id = self._map_race_to_id(name, year)
                            
                            # Get circuit ID from race
                            circuit_id = None
                            if race_id and not self.f1db_data['races'].empty:
                                race_info = self.f1db_data['races'][self.f1db_data['races']['id'] == race_id]
                                if not race_info.empty:
                                    circuit_id = race_info.iloc[0].get('circuitId')
                            
                            formatted_stop = {
                                'position': stop.get('position'),
                                'driver': driver_name,
                                'driver_id': driver_id,
                                'team': team_name,
                                'constructor_id': constructor_id,
                                'time': stop.get('time'),
                                'lap': stop.get('lap'),
                                'points': stop.get('points'),
                                'race': name,
                                'race_id': race_id,
                                'circuit_id': circuit_id,
                                'year': year
                            }
                        else:
                            # JSON data - needs parsing
                            driver_name = f"{stop.get('firstName', '')} {stop.get('lastName', '')}".strip()
                            team_name = stop.get('team')
                            year = datetime.now().year
                            
                            # Map IDs
                            driver_id = self._map_driver_to_id(driver_name, team_name)
                            constructor_id = self._map_team_to_constructor_id(team_name)
                            race_id = self._map_race_to_id(name, year)
                            
                            # Get circuit ID from race
                            circuit_id = None
                            if race_id and not self.f1db_data['races'].empty:
                                race_info = self.f1db_data['races'][self.f1db_data['races']['id'] == race_id]
                                if not race_info.empty:
                                    circuit_id = race_info.iloc[0].get('circuitId')
                            
                            formatted_stop = {
                                'position': stop.get('position', stop.get('id')),
                                'driver_number': stop.get('driverNr'),
                                'driver_tla': stop.get('tla'),
                                'driver': driver_name,
                                'driver_id': driver_id,
                                'team': team_name,
                                'constructor_id': constructor_id,
                                'time': stop.get('duration'),
                                'lap': stop.get('lap'),
                                'points': stop.get('points'),
                                'start_time': stop.get('startTime', {}).get('date') if isinstance(stop.get('startTime'), dict) else None,
                                'irregular': stop.get('irregular', False),
                                'race': name,
                                'race_id': race_id,
                                'circuit_id': circuit_id,
                                'year': year
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
        
        # ID mapping statistics
        if 'driver_id' in df.columns:
            print("\n🔗 ID MAPPING STATISTICS")
            print("-" * 40)
            print(f"Mapped drivers: {df['driver_id'].notna().sum()}/{len(df)} ({df['driver_id'].notna().sum()/len(df)*100:.1f}%)")
            print(f"Mapped constructors: {df['constructor_id'].notna().sum()}/{len(df)} ({df['constructor_id'].notna().sum()/len(df)*100:.1f}%)")
            print(f"Mapped races: {df['race_id'].notna().sum()}/{len(df)} ({df['race_id'].notna().sum()/len(df)*100:.1f}%)")
            
            # Show unmapped drivers if any
            unmapped_drivers = df[df['driver_id'].isna()]['driver'].unique()
            if len(unmapped_drivers) > 0:
                print(f"\nUnmapped drivers: {', '.join(unmapped_drivers[:5])}")
                if len(unmapped_drivers) > 5:
                    print(f"... and {len(unmapped_drivers) - 5} more")
        
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