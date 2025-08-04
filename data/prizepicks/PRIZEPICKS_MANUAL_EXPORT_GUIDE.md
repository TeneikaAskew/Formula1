# PrizePicks Manual Data Export Guide

This guide explains how to manually export your PrizePicks data using browser developer tools. https://api.prizepicks.com/v1/entries?filter=settled

## Quick Steps

1. **Log into PrizePicks** and navigate to your past lineups
2. **Open Developer Tools** (F12)
3. **Go to Network tab**
4. **Refresh the page** (or navigate between pages to trigger API calls)
5. **Look for API calls** that return your lineup/wager data (usually JSON responses)
6. **Save the response** as `lineup.json` in `data/prizepicks/`
7. **Run the parser**: `python parse_prizepicks_wagers.py data/prizepicks/lineup.json`

## Detailed Instructions

### Step 1: Capture API Data

1. Open Chrome/Firefox and log into PrizePicks
2. Navigate to your past lineups page
3. Press **F12** to open Developer Tools
4. Click the **Network** tab
5. Click **Clear** button to clear existing requests
6. **Refresh the page** (F5)
7. Look for requests that contain:
   - `projections`
   - `lineups`
   - `entries`
   - API endpoints (usually JSON responses) - https://api.prizepicks.com/v1/entries?filter=settled

### Step 2: Save API Response

1. Click on the API request
2. Go to **Response** tab
3. Right-click and select **Copy Response**
4. Save to a file named `lineup.json`

### Step 3: Parse the Data

Run the parser with your saved file:

```bash
python parse_prizepicks_wagers.py data/prizepicks/lineup.json
```

This will:
- Parse all your wagers and player results
- Calculate win rates and ROI by sport and pick type
- Generate a detailed summary report
- Save everything to CSV for further analysis

## Output Files

The parser creates files in `data/prizepicks/`:
- `prizepicks_wagers_TIMESTAMP.csv` - All your wager data with profit/loss
- `prizepicks_wagers_latest.csv` - Always contains the most recent data
- `prizepicks_summary_TIMESTAMP.txt` - Detailed performance report

## Tips

- **Multiple Pages**: If you have multiple pages of lineups, capture each page's API response
- **Filter Network Tab**: Type "json" in the filter to see only JSON responses
- **Save Raw Data**: Always save the raw API response for reference
- **Check Headers**: Some data might be in request headers (Authorization, etc.)

## Data Fields Captured

The parser extracts:
- Lineup ID
- Date
- Sport
- Pick type (2-Pick, 3-Pick, etc.)
- Entry fee
- Play type (Flex Play/Power Play)
- Potential payout
- Result (Win/Loss)
- Actual payout received
- Profit/Loss for each entry
- Players in the lineup
- Individual player results (Won/Lost/Push)

## Troubleshooting

**No API data found:**
- Clear browser cache and retry
- Look for XHR/Fetch requests specifically
- Check "All" filter in Network tab

**Parser errors:**
- Ensure the API response is complete (not truncated)
- Check that the file is saved as UTF-8
- Try extracting smaller portions first

**Missing projections:**
- Some projections might be loaded dynamically
- Scroll through all lineups to trigger loading
- Check for pagination API calls