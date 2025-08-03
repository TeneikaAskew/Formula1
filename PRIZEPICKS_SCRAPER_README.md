# PrizePicks Lineup Scraper

This tool automatically scrapes your PrizePicks lineup history and saves it to CSV files.

## Features

- Logs into PrizePicks and extracts all past lineups
- Captures lineup details: ID, date, pick type, entry fee, play type, payout, result
- Tracks individual player performance (Win/Loss/Push)
- Saves data to timestamped CSV files
- Runs automatically via GitHub Actions (daily)

## Setup

### Local Testing

1. Install dependencies:
```bash
pip install selenium pandas
```

2. Install Chrome and ChromeDriver:
- Download Chrome: https://www.google.com/chrome/
- Download ChromeDriver: https://chromedriver.chromium.org/

3. Test the setup:
```bash
python test_prizepicks_scraper.py
```

4. Run the scraper:
```bash
python scrape_prizepicks_lineups.py
```

### GitHub Actions Setup

1. Add your PrizePicks credentials as a GitHub secret:
   - Go to your repository Settings > Secrets and variables > Actions
   - Click "New repository secret"
   - Name: `PRIZE_PICKS`
   - Value: `your_email@example.com,your_password` (comma-separated)

2. The workflow runs automatically:
   - Daily at 5:00 AM UTC (midnight EST)
   - Manual trigger: Actions tab > "Daily PrizePicks Lineup Scraper" > Run workflow

## Output

Data is saved to `data/prizepicks/` with two files:
- `prizepicks_lineups_YYYYMMDD_HHMMSS.csv` - Timestamped version
- `prizepicks_lineups_latest.csv` - Always contains the most recent data

## CSV Fields

- `lineup_id`: Unique identifier for the lineup
- `date`: Date of the lineup (e.g., "December 29, 2024")
- `pick_type`: Type of pick (e.g., "2-Pick", "3-Pick")
- `entry_fee`: Entry fee amount
- `play_type`: Play type (e.g., "Power Play", "Flex Play")
- `payout`: Potential payout amount
- `result`: Lineup result (Win/Loss)
- `players`: List of players in the lineup
- `player_results`: Individual player outcomes (e.g., "Player1:Won; Player2:Lost")
- `scraped_at`: Timestamp when data was scraped

## Troubleshooting

- **Login fails**: Check credentials in PRIZE_PICKS secret
- **No lineups found**: Verify you're logged in and have past lineups
- **Chrome driver issues**: Ensure Chrome and ChromeDriver versions match
- **GitHub Actions fails**: Check workflow logs in Actions tab