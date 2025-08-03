#!/usr/bin/env python3
"""
Test script for PrizePicks scraper
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scraper():
    """Test the PrizePicks scraper"""
    
    # Check if selenium is installed
    try:
        import selenium
        logger.info(f"✓ Selenium installed (version {selenium.__version__})")
    except ImportError:
        logger.error("✗ Selenium not installed. Run: pip install selenium")
        return False
    
    # Check if pandas is installed
    try:
        import pandas
        logger.info(f"✓ Pandas installed (version {pandas.__version__})")
    except ImportError:
        logger.error("✗ Pandas not installed. Run: pip install pandas")
        return False
    
    # Check if Chrome driver is available
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        driver.quit()
        logger.info("✓ Chrome driver is available")
    except Exception as e:
        logger.error(f"✗ Chrome driver not available: {str(e)}")
        logger.info("Install Chrome and ChromeDriver: https://chromedriver.chromium.org/")
        return False
    
    # Check if credentials are available
    if 'PRIZE_PICKS' in os.environ:
        logger.info("✓ PRIZE_PICKS credentials found in environment")
    else:
        logger.warning("⚠ PRIZE_PICKS credentials not in environment, will prompt for input")
    
    # Check output directory
    output_dir = "data/prizepicks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"✓ Created output directory: {output_dir}")
    else:
        logger.info(f"✓ Output directory exists: {output_dir}")
    
    return True

def main():
    """Main test function"""
    logger.info("Testing PrizePicks scraper setup...")
    
    if test_scraper():
        logger.info("\nAll checks passed! You can run the scraper with:")
        logger.info("  python scrape_prizepicks_lineups.py")
        logger.info("\nFor GitHub Actions, set PRIZE_PICKS secret with format: username,password")
    else:
        logger.error("\nSome checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()