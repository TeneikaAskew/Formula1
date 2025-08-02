# DHL F1 Pit Stop Data Scraper - GitHub Action

## Overview
This GitHub Action automatically scrapes the official DHL F1 pit stop data every Thursday at 6 AM UTC.

## Features
- **Incremental Updates**: Only fetches data for races not already in the CSV
- **Automatic Commits**: Commits new data back to the repository
- **Manual Trigger**: Can be run manually via GitHub Actions UI
- **Smart Deduplication**: Prevents duplicate entries

## How It Works
1. Runs every Thursday at 6 AM UTC (typically after race weekends)
2. Checks existing CSV for races already scraped
3. Only fetches new race data from DHL website
4. Appends new data to existing CSV file
5. Commits changes if new data was found

## Manual Execution
To run manually:
1. Go to Actions tab in GitHub
2. Select "Scrape DHL F1 Pit Stop Data"
3. Click "Run workflow"

## Data Location
- Scraped data is saved to: `/data/dhl/`
- Format: CSV with columns: position, team, driver, time, lap, points, race, year

## Local Testing
Before relying on the GitHub Action, test locally:
```bash
# First run - gets all races
python dhl_current_season_scraper.py

# Subsequent runs - only gets new races
python dhl_current_season_scraper.py

# Force new file instead of appending
python dhl_current_season_scraper.py --no-append

# Debug mode
python dhl_current_season_scraper.py --debug
```

## Troubleshooting
- If the action fails, check the logs in the Actions tab
- Common issues:
  - Website structure changes
  - Chrome/ChromeDriver version mismatch
  - Network timeouts

## Customization
Edit `.github/workflows/scrape-dhl-pitstops.yml` to:
- Change schedule (cron expression)
- Add notifications
- Modify commit message format