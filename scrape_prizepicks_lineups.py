#!/usr/bin/env python3
"""
Scrape PrizePicks lineup history and save to CSV
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_driver(headless=True):
    """Set up Chrome driver with options"""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def login_to_prizepicks(driver, username, password, max_retries=3):
    """Login to PrizePicks with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Login attempt {attempt + 1}/{max_retries}")
            logger.info("Navigating to PrizePicks login page...")
            driver.get("https://app.prizepicks.com/")
            
            # Wait for login form to appear
            wait = WebDriverWait(driver, 30)
            
            try:
                # Click on login button if needed
                login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Log In')]")))
                login_button.click()
                time.sleep(2)
            except:
                logger.info("Login form might already be visible")
            
            # Enter credentials
            logger.info("Entering credentials...")
            email_input = wait.until(EC.presence_of_element_located((By.NAME, "email")))
            email_input.clear()
            email_input.send_keys(username)
            
            password_input = driver.find_element(By.NAME, "password")
            password_input.clear()
            password_input.send_keys(password)
            
            # Submit login form
            submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_button.click()
            
            # Wait for successful login by checking for user menu or lineups page
            time.sleep(5)
            logger.info("Login submitted, verifying success...")
            
            # Check if we're logged in by looking for user menu or lineups
            try:
                wait.until(lambda d: "my-lineups" in d.current_url or 
                          d.find_elements(By.CSS_SELECTOR, "button[aria-label*='User']"))
                logger.info("Login successful!")
                return True
            except TimeoutException:
                if attempt < max_retries - 1:
                    logger.warning("Login verification failed, retrying...")
                    time.sleep(3)
                else:
                    raise Exception("Login failed after all retries")
                    
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Login attempt failed: {str(e)}, retrying...")
                time.sleep(5)
            else:
                raise Exception(f"Login failed after {max_retries} attempts: {str(e)}")
    
    return False

def extract_lineup_data(driver):
    """Extract lineup data from the page"""
    lineups = []
    wait = WebDriverWait(driver, 20)
    
    # Navigate to past lineups
    logger.info("Navigating to past lineups...")
    driver.get("https://app.prizepicks.com/my-lineups/past-lineups")
    time.sleep(5)
    
    # Wait for lineups to load
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-lineup-id]")))
        logger.info("Lineups loaded successfully")
    except TimeoutException:
        logger.warning("No lineups found or page didn't load properly")
        return lineups
    
    # Extract summary data
    try:
        # Get summary stats
        summary_divs = driver.find_elements(By.CSS_SELECTOR, "div.heading-md.text-soClean-100")
        lineups_won = summary_divs[0].text if len(summary_divs) > 0 else "0"
        amount_won = summary_divs[1].text if len(summary_divs) > 1 else "$0.00"
        
        logger.info(f"Summary - Lineups Won: {lineups_won}, Amount Won: {amount_won}")
    except:
        lineups_won = "0"
        amount_won = "$0.00"
    
    # Process all pages of lineups
    page_num = 1
    all_processed_ids = set()
    
    while True:
        logger.info(f"Processing page {page_num}")
        
        # Get all lineup buttons on current page
        lineup_buttons = driver.find_elements(By.CSS_SELECTOR, "button[data-lineup-id]")
        new_lineups_found = 0
        
        for button in lineup_buttons:
            lineup_id = button.get_attribute('data-lineup-id')
            if lineup_id in all_processed_ids:
                continue
            
            all_processed_ids.add(lineup_id)
            new_lineups_found += 1
            
            try:
                lineup_data = {}
                
                # Get lineup ID
                lineup_data['lineup_id'] = lineup_id
            
                # Get date - look for the date header above this lineup
                date_element = button.find_element(By.XPATH, "./preceding::div[contains(@class, 'subheading-md')][1]")
                lineup_data['date'] = date_element.text
                
                # Get pick type and payout
                heading = button.find_element(By.CSS_SELECTOR, "p.heading-sm")
                pick_info = heading.text.split()
                lineup_data['pick_type'] = pick_info[0]  # e.g., "2-Pick"
                lineup_data['payout'] = heading.find_element(By.TAG_NAME, "span").text
                
                # Get entry fee and play type
                subheading = button.find_element(By.CSS_SELECTOR, "p.subheading-sm")
                entry_fee_span = subheading.find_element(By.CSS_SELECTOR, "span.text-soClean-100")
                lineup_data['entry_fee'] = entry_fee_span.text
                play_type_text = subheading.text.replace(entry_fee_span.text, "").strip()
                lineup_data['play_type'] = play_type_text  # e.g., "Power Play" or "Flex Play"
                
                # Get players
                players_element = button.find_element(By.CSS_SELECTOR, "p.body-sm.text-soClean-140")
                lineup_data['players'] = players_element.text
                
                # Get result (Win/Loss)
                result_div = button.find_element(By.CSS_SELECTOR, "div.label-sm")
                lineup_data['result'] = result_div.text
                
                # Get player win/loss status
                player_status = []
                player_imgs = button.find_elements(By.CSS_SELECTOR, "img.h-full.w-full")
                player_status_icons = button.find_elements(By.CSS_SELECTOR, "svg#player-headshot-status-badge")
                
                for i, img in enumerate(player_imgs):
                    player_name = img.get_attribute('alt')
                    # Check if player won or lost by looking at the border color class
                    parent_div = img.find_element(By.XPATH, "./..")
                    border_classes = parent_div.get_attribute('class')
                    if 'border-atlien-100' in border_classes:
                        status = 'Won'
                    elif 'border-roses' in border_classes:
                        status = 'Lost'
                    elif 'border-soClean-180' in border_classes:
                        status = 'Push'
                    else:
                        status = 'Unknown'
                    player_status.append(f"{player_name}:{status}")
                
                lineup_data['player_results'] = '; '.join(player_status)
                
                # Add to lineups list
                lineups.append(lineup_data)
                logger.info(f"Extracted lineup {lineup_data['lineup_id']}")
                
            except Exception as e:
                logger.error(f"Error extracting lineup {lineup_id}: {str(e)}")
                continue
        
        logger.info(f"Found {new_lineups_found} new lineups on page {page_num}")
        
        # Check if there's a next page
        try:
            # Look for pagination or scroll to load more
            # Try to scroll to bottom to load more lineups
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Check if new lineups loaded
            new_buttons = driver.find_elements(By.CSS_SELECTOR, "button[data-lineup-id]")
            if len(new_buttons) == len(lineup_buttons) or new_lineups_found == 0:
                logger.info("No more lineups to load")
                break
                
            page_num += 1
            
        except Exception as e:
            logger.info(f"Finished processing lineups: {str(e)}")
            break
    
    return lineups

def save_to_csv(lineups, output_path):
    """Save lineups to CSV file"""
    if not lineups:
        logger.warning("No lineups to save")
        return
    
    df = pd.DataFrame(lineups)
    
    # Add timestamp column
    df['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Reorder columns
    column_order = ['lineup_id', 'date', 'pick_type', 'entry_fee', 'play_type', 
                    'payout', 'result', 'players', 'player_results', 'scraped_at']
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} lineups to {output_path}")

def main():
    """Main function"""
    # Get credentials from environment or command line
    if 'PRIZE_PICKS' in os.environ:
        creds = os.environ['PRIZE_PICKS'].split(',')
        username = creds[0]
        password = creds[1]
    else:
        # For local testing
        username = input("Enter PrizePicks username: ")
        password = input("Enter PrizePicks password: ")
    
    # Set up output path
    output_dir = "data/prizepicks"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"prizepicks_lineups_{timestamp}.csv")
    
    # Determine if running in GitHub Actions
    is_github_actions = os.environ.get('GITHUB_ACTIONS', 'false').lower() == 'true'
    
    # Set up driver
    driver = setup_driver(headless=is_github_actions)
    
    try:
        # Login
        login_to_prizepicks(driver, username, password)
        
        # Extract data
        lineups = extract_lineup_data(driver)
        
        # Save to CSV
        save_to_csv(lineups, output_path)
        
        # Also save a "latest" version
        latest_path = os.path.join(output_dir, "prizepicks_lineups_latest.csv")
        save_to_csv(lineups, latest_path)
        
        logger.info("Scraping completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise
    finally:
        driver.quit()

if __name__ == "__main__":
    main()