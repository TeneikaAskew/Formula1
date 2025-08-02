# F1 Pit Stop Data Extraction Solutions

This repository contains two approaches to extract Formula 1 pit stop data from the official DHL Fastest Pit Stop Award website.

## 🎯 Quick Start

### Option 1: DHL Current Season Scraper (Recommended for Current Data)
```bash
# Install dependencies
pip install selenium pandas beautifulsoup4

# Run the scraper (saves to data/dhl/ as CSV by default)
python dhl_current_season_scraper.py

# Run with summary statistics
python dhl_current_season_scraper.py --summary

# Custom filename (still saves to data/dhl/)
python dhl_current_season_scraper.py --output "2025_pitstops"
```

### Option 2: Full DHL Scraper (All Features)
```bash
python dhl_pitstop_scraper.py --headless --format excel --summary
```

## 📊 Data Structure

All scrapers output similar data structure:

| Field | Description | Example |
|-------|-------------|---------|
| year | Season year | 2025 |
| race | Grand Prix name | Belgian Grand Prix |
| driver | Driver full name | Yuki Tsunoda |
| team | Constructor/Team | Red Bull |
| time/duration | Pit stop duration (seconds) | 2.17 |
| lap | Lap number | 13 |
| points | DHL points awarded | 25 |
| position | Ranking in that race | 1 |

## 🔧 Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install ChromeDriver (for DHL scrapers)
- **Windows**: Download from [ChromeDriver](https://chromedriver.chromium.org/)
- **Mac**: `brew install chromedriver`
- **Linux**: `sudo apt-get install chromium-chromedriver`

### 3. Data Directory Structure
The scrapers automatically save data to:
```
data/
└── dhl/
    ├── dhl_current_season_pitstops_20250802_141532.csv
    ├── openf1_pitstops_20250802_142015.csv
    └── ...
```
The directory will be created automatically when you run the scrapers.

## 📈 Usage Examples

### Extract Current Season for PrizePicks Analysis
```python
from dhl_current_season_scraper import CurrentSeasonScraper

# Run scraper
scraper = CurrentSeasonScraper(visible=False)
scraper.scrape_all_races()
scraper.save_data(format='csv')
scraper.show_summary()
```

### Get Historical Data via OpenF1
```python
from openf1_pitstop_extractor import OpenF1PitStopExtractor

# Extract multiple years
extractor = OpenF1PitStopExtractor()
data = extractor.extract_multiple_years(2020, 2024)
extractor.save_data(data, format='excel')
extractor.generate_report(data)
```

### Analyze Extracted Data
```python
import pandas as pd

# Load data
df = pd.read_csv('data/dhl/dhl_current_season_pitstops_20250802.csv')

# Find fastest stops by team
team_fastest = df.groupby('team')['time'].min().sort_values()
print("Fastest stops by team:")
print(team_fastest)

# Driver consistency analysis
driver_stats = df.groupby('driver')['time'].agg(['mean', 'std', 'count'])
consistent_drivers = driver_stats.sort_values('std').head(10)
print("\nMost consistent drivers:")
print(consistent_drivers)

# Race-by-race analysis
race_summary = df.groupby('race').agg({
    'time': ['min', 'mean'],
    'driver': 'count'
})
print("\nRace summary:")
print(race_summary)
```

## 🚀 Advanced Features

### Scheduling Automated Updates
```bash
# Add to crontab for daily updates during race season
0 6 * * * cd /path/to/project && python dhl_current_season_scraper.py --output "daily_update" >> scraper.log 2>&1
```

### Combining Multiple Data Sources
```python
# Merge DHL and OpenF1 data
dhl_df = pd.read_csv('dhl_pitstops.csv')
openf1_df = pd.read_csv('openf1_pitstops.csv')

# Standardize column names
dhl_df.rename(columns={'duration': 'time'}, inplace=True)

# Combine datasets
combined_df = pd.concat([dhl_df, openf1_df], ignore_index=True)
combined_df.drop_duplicates(subset=['year', 'race', 'driver', 'lap'], inplace=True)
```

## ⚠️ Important Notes

1. **Rate Limiting**: All scrapers include delays to respect server resources
2. **Data Quality**: DHL data is official FIA-verified timing data
3. **Year Availability**: 
   - DHL website: Current season primarily
   - OpenF1 API: 2018 onwards
   - For pre-2018 data, consider Ergast API

## 🐛 Troubleshooting

### ChromeDriver Issues
```bash
# Check Chrome version
google-chrome --version

# Ensure ChromeDriver matches Chrome version
```

### No Data Found
- Run with `--visible` flag to see browser interaction
- Check internet connection
- Verify website structure hasn't changed

### Timeout Errors
- Increase wait times in scraper configuration
- Check if website is accessible
- Try running during off-peak hours

## 📝 Requirements File

Create `requirements.txt`:
```
selenium>=4.0.0
pandas>=1.3.0
beautifulsoup4>=4.9.0
requests>=2.26.0
openpyxl>=3.0.0
lxml>=4.6.0
```

## 🎯 For PrizePicks Analysis

The scraped data is perfect for betting analysis:
- **Pit stop consistency** by driver/team
- **Track-specific performance** patterns
- **Weather impact** on pit stop times
- **Team improvement** trends over season

Remember to always verify data accuracy and comply with all applicable terms of service and regulations.