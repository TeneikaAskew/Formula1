#!/usr/bin/env python3
"""
DHL F1 Pit Stop Data Scraper
Extracts official pit stop times from DHL's F1 website for all races and years
"""

import json
import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
import argparse
from pathlib import Path
import re
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dhl_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DHLPitStopScraper:
    """Scraper for DHL F1 Pit Stop Award data"""
    
    def __init__(self, headless=True, wait_time=20):
        self.base_url = "https://inmotion.dhl/en/formula-1/fastest-pit-stop-award"
        self.wait_time = wait_time
        self.headless = headless
        self.driver = None
        self.all_data = []
        self.f1db_data = self._load_f1db_data()
        
    def _load_f1db_data(self):
        """Load F1DB reference data for mapping"""
        f1db_data = {
            'drivers': pd.DataFrame(),
            'constructors': pd.DataFrame(),
            'races': pd.DataFrame(),
            'circuits': pd.DataFrame(),
            'results': pd.DataFrame()
        }
        
        # Try different data paths
        data_paths = [
            "data/f1db",
            "../data/f1db",
            "../../data/f1db",
            "/workspace/data/f1db"
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    # Load each file separately to handle missing files gracefully
                    drivers_path = os.path.join(path, 'drivers.csv')
                    if os.path.exists(drivers_path):
                        f1db_data['drivers'] = pd.read_csv(drivers_path)
                        logger.info(f"Loaded drivers data from {drivers_path}")
                    
                    constructors_path = os.path.join(path, 'constructors.csv')
                    if os.path.exists(constructors_path):
                        f1db_data['constructors'] = pd.read_csv(constructors_path)
                        logger.info(f"Loaded constructors data from {constructors_path}")
                    
                    races_path = os.path.join(path, 'races.csv')
                    if os.path.exists(races_path):
                        f1db_data['races'] = pd.read_csv(races_path)
                        logger.info(f"Loaded races data from {races_path}")
                    
                    circuits_path = os.path.join(path, 'circuits.csv')
                    if os.path.exists(circuits_path):
                        f1db_data['circuits'] = pd.read_csv(circuits_path)
                        logger.info(f"Loaded circuits data from {circuits_path}")
                    
                    results_path = os.path.join(path, 'results.csv')
                    if os.path.exists(results_path):
                        f1db_data['results'] = pd.read_csv(results_path)
                        logger.info(f"Loaded results data from {results_path}")
                    
                    # If we loaded at least drivers, consider it successful
                    if not f1db_data['drivers'].empty:
                        logger.info(f"Successfully loaded F1DB data from {path}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load F1DB data from {path}: {e}")
        
        return f1db_data
        
    def setup_driver(self):
        """Initialize Selenium WebDriver with Chrome"""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, self.wait_time)
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
            
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
            
    def navigate_to_page(self):
        """Navigate to the DHL pit stop page"""
        logger.info(f"Navigating to {self.base_url}")
        self.driver.get(self.base_url)
        time.sleep(3)  # Allow page to load
        
    def find_year_selector(self):
        """Find and return year selector if available"""
        selectors = [
            "select[name='year']",
            "select#year",
            ".year-selector select",
            "select[data-year]",
            "//select[contains(@class, 'year')]"
        ]
        
        for selector in selectors:
            try:
                if selector.startswith("//"):
                    element = self.driver.find_element(By.XPATH, selector)
                else:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                return element
            except NoSuchElementException:
                continue
        return None
        
    def find_race_selector(self):
        """Find and return race/event selector"""
        selectors = [
            "select[name='event']",
            "select#event",
            ".event-selector select",
            "select[data-event]",
            "//select[contains(@class, 'event')]",
            "select.race-selector",
            "select[class*='race']"
        ]
        
        for selector in selectors:
            try:
                if selector.startswith("//"):
                    element = self.driver.find_element(By.XPATH, selector)
                else:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                return element
            except NoSuchElementException:
                continue
        return None
        
    def extract_table_data_only(self):
        """Extract pit stop data ONLY from table (complete data)"""
        try:
            # First, try to click the Table tab to ensure we see all data
            try:
                # Look for table tab and click it
                table_tab_selectors = [
                    "a[href*='table']",
                    "//a[contains(text(),'Table')]",
                    ".nav-tabs li:nth-child(2) a"
                ]
                
                for selector in table_tab_selectors:
                    try:
                        if selector.startswith("//"):
                            tab = self.driver.find_element(By.XPATH, selector)
                        else:
                            tab = self.driver.find_element(By.CSS_SELECTOR, selector)
                        
                        if tab:
                            logger.info("Clicking Table tab to ensure complete data")
                            self.driver.execute_script("arguments[0].click();", tab)
                            time.sleep(3)
                            break
                    except:
                        continue
            except:
                logger.warning("Could not find table tab - proceeding with current view")
            
            # Extract from table
            return self.extract_table_data()
                
        except Exception as e:
            logger.error(f"Failed to extract table data: {e}")
            return []
            
    def extract_table_data(self):
        """Extract data from HTML table - contains ALL drivers"""
        try:
            # Wait a bit for table to load
            time.sleep(2)
            
            # Try multiple table selectors
            table_selectors = [
                "table.f1-award-table",
                "table.table",
                "#element_event_grid_row2_col0_table table",
                "div[id*='table'] table",
                "div.tab-pane.active table",
                "table"  # Last resort
            ]
            
            table = None
            for selector in table_selectors:
                try:
                    tables = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for t in tables:
                        try:
                            tbody = t.find_element(By.TAG_NAME, "tbody")
                            rows = tbody.find_elements(By.TAG_NAME, "tr")
                            if len(rows) > 5:  # Should have many rows for pit stops
                                table = t
                                logger.info(f"Found table with {len(rows)} rows")
                                break
                        except:
                            continue
                    if table:
                        break
                except:
                    continue
            
            if not table:
                logger.error("Could not find pit stop table")
                return []
            
            # Extract data from table rows
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")
            
            data = []
            for idx, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 6:
                        continue
                        
                    cell_texts = [cell.text.strip() for cell in cells]
                    if not any(cell_texts):
                        continue
                    
                    # Parse the row data
                    try:
                        pos_text = cell_texts[0]
                        if not pos_text or not pos_text.isdigit():
                            continue
                            
                        pit_stop = {
                            "position": int(pos_text),
                            "team": cell_texts[1],
                            "driver": cell_texts[2],
                            "duration": float(cell_texts[3]),
                            "lap": int(cell_texts[4]) if cell_texts[4] and cell_texts[4].isdigit() else None,
                            "points": int(cell_texts[5].replace('*', '').strip()) if cell_texts[5].replace('*', '').strip().isdigit() else 0
                        }
                        data.append(pit_stop)
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Row {idx} parse error: {e}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Error processing row {idx}: {e}")
                    continue
                    
            logger.info(f"✅ Extracted {len(data)} pit stops from table")
            return data
            
        except Exception as e:
            logger.error(f"Failed to extract table data: {e}")
            return []
            
    def _map_driver_to_id(self, driver_name, team_name=None, year=None):
        """Map driver name to F1DB driver ID"""
        if self.f1db_data['drivers'].empty:
            return None
        
        drivers = self.f1db_data['drivers']
        driver_name_lower = driver_name.lower().strip()
        
        # Try exact match first
        exact_match = drivers[drivers['name'].str.lower() == driver_name_lower]
        if not exact_match.empty:
            return exact_match.iloc[0]['id']
        
        # Try last name match
        parts = driver_name.split()
        if len(parts) > 0:
            last_name = parts[-1].lower()
            
            # Special mappings for known variations
            special_mappings = {
                'hulkenberg': 'nico-hulkenberg',
                'hülkenberg': 'nico-hulkenberg',
                'sainz': 'carlos-sainz-jr',
                'perez': 'sergio-perez',
                'pérez': 'sergio-perez',
                'magnussen': 'kevin-magnussen',
                'leclerc': 'charles-leclerc',
                'zhou': 'guanyu-zhou',
                'schumacher': 'mick-schumacher',  # For recent years
                'verstappen': 'max-verstappen',
                'bottas': 'valtteri-bottas',
                'de vries': 'nyck-de-vries',
                'devries': 'nyck-de-vries'
            }
            
            if last_name in special_mappings:
                return special_mappings[last_name]
            
            # Try matching by last name in drivers table
            last_name_match = drivers[drivers['lastName'].str.lower() == last_name]
            
            # If multiple drivers with same last name, try to filter by active years
            if len(last_name_match) > 1 and year:
                # Get results for this year to find active drivers
                if not self.f1db_data['results'].empty:
                    results = self.f1db_data['results']
                    year_results = results[results['year'] == year]
                    active_driver_ids = year_results['driverId'].unique()
                    active_matches = last_name_match[last_name_match['id'].isin(active_driver_ids)]
                    if not active_matches.empty:
                        return active_matches.iloc[0]['id']
            
            if not last_name_match.empty:
                return last_name_match.iloc[0]['id']
            
            # Try partial matching on full name
            for _, driver in drivers.iterrows():
                if last_name in driver['lastName'].lower():
                    return driver['id']
        
        return None
    
    def _map_team_to_constructor_id(self, team_name, year=None):
        """Map team name to F1DB constructor ID"""
        if self.f1db_data['constructors'].empty or not team_name:
            return None
        
        constructors = self.f1db_data['constructors']
        team_name_lower = team_name.lower().strip()
        
        # Special mappings for known team variations
        team_mappings = {
            'red bull': 'red-bull',
            'red bull racing': 'red-bull',
            'aston martin': 'aston-martin',
            'aston martin f1': 'aston-martin',
            'mercedes': 'mercedes',
            'mercedes amg': 'mercedes',
            'ferrari': 'ferrari',
            'scuderia ferrari': 'ferrari',
            'mclaren': 'mclaren',
            'mclaren f1': 'mclaren',
            'alpine': 'alpine',
            'alpine f1': 'alpine',
            'alphatauri': 'alphatauri',
            'scuderia alphatauri': 'alphatauri',
            'alfa romeo': 'alfa-romeo',
            'haas': 'haas',
            'haas f1': 'haas',
            'williams': 'williams',
            'williams racing': 'williams',
            'sauber': 'sauber'
        }
        
        # Try mapped name first
        for key, value in team_mappings.items():
            if key in team_name_lower:
                return value
        
        # Try exact match
        exact_match = constructors[constructors['id'].str.lower() == team_name_lower]
        if not exact_match.empty:
            return exact_match.iloc[0]['id']
        
        # Try partial match on constructor name
        for _, constructor in constructors.iterrows():
            if team_name_lower in constructor['name'].lower() or constructor['name'].lower() in team_name_lower:
                return constructor['id']
        
        return None
    
    def _map_race_to_id(self, race_name, year=None):
        """Map race name to F1DB race ID"""
        if self.f1db_data['races'].empty or not race_name:
            return None
        
        races = self.f1db_data['races']
        race_name_lower = race_name.lower().strip()
        
        # Filter by year if provided
        if year:
            year_races = races[races['year'] == year]
        else:
            year_races = races
        
        # Try exact match on official name
        for _, race in year_races.iterrows():
            if race_name_lower in str(race.get('officialName', '')).lower():
                return race['id']
        
        # Try matching by circuit name
        if not self.f1db_data['circuits'].empty:
            circuits = self.f1db_data['circuits']
            for _, circuit in circuits.iterrows():
                circuit_name = circuit['name'].lower()
                if circuit_name in race_name_lower or any(word in race_name_lower for word in circuit_name.split() if len(word) > 4):
                    # Find races at this circuit
                    circuit_races = year_races[year_races['circuitId'] == circuit['id']]
                    if not circuit_races.empty:
                        return circuit_races.iloc[0]['id']
        
        # Try matching by common race name patterns
        race_keywords = ['british', 'italian', 'monaco', 'spanish', 'belgian', 'dutch', 'hungarian',
                        'singapore', 'japanese', 'mexican', 'brazilian', 'abu dhabi', 'saudi',
                        'bahrain', 'australian', 'canadian', 'french', 'austrian', 'azerbaijan',
                        'miami', 'las vegas', 'chinese', 'emilia', 'qatar', 'united states']
        
        for keyword in race_keywords:
            if keyword in race_name_lower:
                for _, race in year_races.iterrows():
                    if keyword in str(race.get('officialName', '')).lower():
                        return race['id']
        
        return None
    
    def process_race_data(self, race_name, year=None):
        """Process and format pit stop data for a specific race"""
        data = self.extract_table_data_only()
        
        # Get race ID
        race_id = self._map_race_to_id(race_name, year)
        
        # Get circuit ID from race
        circuit_id = None
        if race_id and not self.f1db_data['races'].empty:
            race_info = self.f1db_data['races'][self.f1db_data['races']['id'] == race_id]
            if not race_info.empty:
                circuit_id = race_info.iloc[0].get('circuitId')
        
        for pit_stop in data:
            # Extract driver name
            driver_full_name = f"{pit_stop.get('firstName', '')} {pit_stop.get('lastName', '')}".strip() or pit_stop.get("driver", "")
            team_name = pit_stop.get("team", "")
            
            # Map to IDs
            driver_id = self._map_driver_to_id(driver_full_name, team_name, year)
            constructor_id = self._map_team_to_constructor_id(team_name, year)
            
            # Standardize the data structure
            formatted_stop = {
                "year": year or datetime.now().year,
                "race": race_name,
                "race_id": race_id,
                "circuit_id": circuit_id,
                "position": pit_stop.get("position", pit_stop.get("id")),
                "driver_number": pit_stop.get("driverNr"),
                "driver_tla": pit_stop.get("tla"),
                "driver_first_name": pit_stop.get("firstName"),
                "driver_last_name": pit_stop.get("lastName"),
                "driver": driver_full_name,
                "driver_id": driver_id,
                "team": team_name,
                "constructor_id": constructor_id,
                "time": pit_stop.get("duration"),
                "lap": pit_stop.get("lap"),
                "points": pit_stop.get("points"),
                "irregular": pit_stop.get("irregular", False),
                "notes": pit_stop.get("notes", ""),
                "start_time": None
            }
            
            # Parse start time if available
            if "startTime" in pit_stop and isinstance(pit_stop["startTime"], dict):
                formatted_stop["start_time"] = pit_stop["startTime"].get("date")
                
            self.all_data.append(formatted_stop)
            
    def scrape_year(self, year=None):
        """Scrape all races for a specific year"""
        self.navigate_to_page()
        
        # Try to select year if selector exists
        if year:
            year_selector = self.find_year_selector()
            if year_selector:
                try:
                    select = Select(year_selector)
                    select.select_by_value(str(year))
                    time.sleep(2)  # Wait for page update
                    logger.info(f"Selected year: {year}")
                except Exception as e:
                    logger.warning(f"Could not select year {year}: {e}")
                    
        # Find race selector
        race_selector = self.find_race_selector()
        if not race_selector:
            logger.error("No race selector found")
            # Try to extract data from current page
            self.process_race_data("Current Race", year)
            return
            
        # Get all race options
        try:
            select = Select(race_selector)
            options = select.options
            logger.info(f"Found {len(options)} race options")
            
            # Process each race
            for i, option in enumerate(options):
                race_name = option.text.strip()
                if not race_name or race_name.lower() in ["select", "choose"]:
                    continue
                    
                logger.info(f"Processing race {i+1}/{len(options)}: {race_name}")
                
                try:
                    select.select_by_index(i)
                    time.sleep(2)  # Wait for data to load
                    self.process_race_data(race_name, year)
                    
                except Exception as e:
                    logger.error(f"Failed to process {race_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to process races: {e}")
            
    def scrape_all_available_data(self):
        """Scrape all available years and races"""
        self.setup_driver()
        
        try:
            # First, check if there's a year selector
            self.navigate_to_page()
            year_selector = self.find_year_selector()
            
            if year_selector:
                # Get all available years
                select = Select(year_selector)
                years = [option.get_attribute("value") for option in select.options 
                        if option.get_attribute("value") and option.get_attribute("value").isdigit()]
                logger.info(f"Found years: {years}")
                
                for year in years:
                    logger.info(f"Scraping year: {year}")
                    self.scrape_year(int(year))
            else:
                # No year selector, just scrape current season
                logger.info("No year selector found, scraping current season only")
                self.scrape_year()
                
        finally:
            self.close_driver()
            
    def save_data(self, format="csv", filename=None):
        """Save scraped data to file"""
        if not self.all_data:
            logger.warning("No data to save")
            return
            
        df = pd.DataFrame(self.all_data)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dhl_pitstops_{timestamp}"
            
        output_path = Path("data/dhl")
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            filepath = output_path / f"{filename}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
            
        elif format == "json":
            filepath = output_path / f"{filename}.json"
            df.to_json(filepath, orient="records", indent=2)
            logger.info(f"Saved {len(df)} records to {filepath}")
            
        elif format == "excel":
            filepath = output_path / f"{filename}.xlsx"
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Overall data
                df.to_excel(writer, sheet_name='All_Pit_Stops', index=False)
                
                # Summary by race
                if 'race' in df.columns and 'time' in df.columns:
                    summary = df.groupby('race')['time'].agg(['min', 'mean', 'count'])
                    summary.to_excel(writer, sheet_name='Race_Summary')
                
                # Mapping summary
                mapping_summary = pd.DataFrame({
                    'total_records': [len(df)],
                    'mapped_drivers': [df['driver_id'].notna().sum()],
                    'unmapped_drivers': [df['driver_id'].isna().sum()],
                    'mapped_constructors': [df['constructor_id'].notna().sum()],
                    'unmapped_constructors': [df['constructor_id'].isna().sum()],
                    'mapped_races': [df['race_id'].notna().sum()],
                    'unmapped_races': [df['race_id'].isna().sum()]
                })
                mapping_summary.to_excel(writer, sheet_name='Mapping_Summary', index=False)
                    
            logger.info(f"Saved {len(df)} records to {filepath}")
            
        return filepath
        
    def generate_summary(self):
        """Generate summary statistics"""
        if not self.all_data:
            return
            
        df = pd.DataFrame(self.all_data)
        
        print("\n=== PIT STOP DATA SUMMARY ===")
        print(f"Total pit stops: {len(df)}")
        
        if 'year' in df.columns:
            print(f"Years covered: {df['year'].min()} - {df['year'].max()}")
            
        if 'race' in df.columns:
            print(f"Total races: {df['race'].nunique()}")
            
        if 'time' in df.columns:
            print(f"\nFastest pit stop: {df['time'].min():.2f}s")
            print(f"Average pit stop: {df['time'].mean():.2f}s")
            
        # Mapping statistics
        print("\n=== DATA MAPPING SUMMARY ===")
        print(f"Mapped drivers: {df['driver_id'].notna().sum()}/{len(df)} ({df['driver_id'].notna().sum()/len(df)*100:.1f}%)")
        print(f"Mapped constructors: {df['constructor_id'].notna().sum()}/{len(df)} ({df['constructor_id'].notna().sum()/len(df)*100:.1f}%)")
        print(f"Mapped races: {df['race_id'].notna().sum()}/{len(df)} ({df['race_id'].notna().sum()/len(df)*100:.1f}%)")
        
        # Show unmapped drivers
        if 'driver_id' in df.columns:
            unmapped_drivers = df[df['driver_id'].isna()]['driver'].unique()
            if len(unmapped_drivers) > 0:
                print(f"\nUnmapped drivers ({len(unmapped_drivers)}): {', '.join(unmapped_drivers[:10])}")
                if len(unmapped_drivers) > 10:
                    print(f"... and {len(unmapped_drivers) - 10} more")
            
        if 'team' in df.columns and 'time' in df.columns:
            print("\n=== FASTEST BY TEAM ===")
            team_fastest = df.groupby('team')['time'].min().sort_values()
            for team, time in team_fastest.head(10).items():
                print(f"{team}: {time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Scrape DHL F1 Pit Stop Data")
    parser.add_argument("--year", type=int, help="Specific year to scrape")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--format", choices=["csv", "json", "excel"], default="csv", 
                      help="Output format")
    parser.add_argument("--output", help="Output filename (without extension)")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    
    args = parser.parse_args()
    
    scraper = DHLPitStopScraper(headless=args.headless)
    
    try:
        if args.year:
            scraper.setup_driver()
            scraper.scrape_year(args.year)
            scraper.close_driver()
        else:
            scraper.scrape_all_available_data()
            
        # Save data
        scraper.save_data(format=args.format, filename=args.output)
        
        # Show summary if requested
        if args.summary:
            scraper.generate_summary()
            
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise

if __name__ == "__main__":
    main()