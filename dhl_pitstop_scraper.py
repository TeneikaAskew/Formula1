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
            "select[contains(class, 'race')]"
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
        
    def extract_json_data(self):
        """Extract pit stop data from data-chart-data attribute"""
        try:
            # Wait for canvas element with data
            canvas = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "canvas.f1-award-chart"))
            )
            
            # Get JSON data from attribute
            json_str = canvas.get_attribute("data-chart-data")
            if json_str:
                data = json.loads(json_str)
                logger.info(f"Found {len(data)} pit stops in JSON data")
                return data
            else:
                logger.warning("No data-chart-data attribute found")
                return []
                
        except TimeoutException:
            logger.warning("Canvas element not found, trying table extraction")
            return self.extract_table_data()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON data: {e}")
            return []
            
    def extract_table_data(self):
        """Fallback: Extract data from HTML table"""
        try:
            table = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.f1-award-table"))
            )
            
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
            data = []
            
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 6:
                    pit_stop = {
                        "position": int(cells[0].text.strip()),
                        "team": cells[1].text.strip(),
                        "driver": cells[2].text.strip(),
                        "duration": float(cells[3].text.strip()),
                        "lap": int(cells[4].text.strip()),
                        "points": int(cells[5].text.strip())
                    }
                    data.append(pit_stop)
                    
            logger.info(f"Extracted {len(data)} pit stops from table")
            return data
            
        except TimeoutException:
            logger.error("No table found on page")
            return []
            
    def process_race_data(self, race_name, year=None):
        """Process and format pit stop data for a specific race"""
        data = self.extract_json_data()
        
        for pit_stop in data:
            # Standardize the data structure
            formatted_stop = {
                "year": year or datetime.now().year,
                "race": race_name,
                "position": pit_stop.get("position", pit_stop.get("id")),
                "driver_number": pit_stop.get("driverNr"),
                "driver_tla": pit_stop.get("tla"),
                "driver_first_name": pit_stop.get("firstName"),
                "driver_last_name": pit_stop.get("lastName"),
                "driver": f"{pit_stop.get('firstName', '')} {pit_stop.get('lastName', '')}".strip() or pit_stop.get("driver"),
                "team": pit_stop.get("team"),
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