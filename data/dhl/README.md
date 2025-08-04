# DHL F1 Pit Stop Data

This directory contains pit stop data extracted from the official DHL F1 Fastest Pit Stop Award API endpoints.

## Data Overview

### Files

1. **`dhl_pitstops_integrated_[timestamp].csv`** - Main pit stop data file
   - Contains all pit stop times with F1DB IDs for easy integration
   - Columns: driver_id, constructor_id, race_id, event_name, event_date, time, position, points
   - Updated after each race weekend

2. **`dhl_events_all_years.csv`** - Complete event calendar
   - All F1 events from 2023-2025 in a single file
   - Maps DHL event IDs to F1DB race IDs
   - Used for freshness checks and race lookups
   - Automatically updated when new data is extracted

3. **`dhl_constructor_averages_[year].csv`** - Team pit stop averages
   - Average pit stop times by constructor
   - Used for team performance analysis

## Data Pipeline

### 1. Automated Updates
- GitHub Actions workflow runs every Thursday at 6 AM UTC
- Checks if new race data is available before extracting
- Only updates when races have occurred since last extraction

### 2. Freshness Check
The extractor checks:
- Latest event date in existing data
- Upcoming races from event calendar
- Whether today's date has passed the next race date (+ 1 day buffer)

### 3. Data Sources
- **2025**: `https://inmotion.dhl/api/f1-award-element-data/6367` (events) & `/6365` (drivers)
- **2024**: `https://inmotion.dhl/api/f1-award-element-data/6276` (events) & `/6273` (drivers)
- **2023**: `https://inmotion.dhl/api/f1-award-element-data/6284` (events) & `/6282` (drivers)

### 4. ID Mapping
The extractor maps DHL data to F1DB IDs:
- **Driver mapping**: Uses driver name fuzzy matching against F1DB drivers.csv
- **Constructor mapping**: Maps team names to F1DB constructor slugs
- **Race mapping**: Matches events by date and circuit to F1DB races.csv

## Manual Usage

### Extract all years (with freshness check):
```bash
python dhl_integrated_extractor.py
```

### Force update (skip freshness check):
```bash
python dhl_integrated_extractor.py --force
```

### Extract specific years:
```bash
python dhl_integrated_extractor.py --years 2024 2025
```

### Check if update needed (exit code 0 = yes, 1 = no):
```bash
python dhl_integrated_extractor.py --check-only
```

## Data Quality

### Coverage
- **2023**: 23 races (100% coverage)
- **2024**: 24 races (100% coverage) 
- **2025**: Updates after each race

### Mapping Success Rates
- Drivers: ~95-98% mapped to F1DB IDs
- Constructors: ~98-100% mapped
- Races: ~95-100% mapped

### Known Issues
- Some reserve/test drivers may not map correctly
- Historical team name changes require manual mapping updates
- Sprint race pit stops are included but not differentiated

## Integration with F1 Pipeline

The pit stop data integrates with the main F1 prediction pipeline:
- Used in feature engineering for pit stop performance metrics
- Provides constructor reliability indicators
- Historical pit stop times for race strategy modeling

## Updating Mappings

If driver/team mappings need updates:
1. Edit the mapping dictionaries in `dhl_integrated_extractor.py`
2. Add new team names to `team_mappings`
3. Add circuit variations to `circuit_mappings`
4. Re-run the extractor with `--force`

## GitHub Actions Workflow

Located at: `.github/workflows/extract-dhl-pitstops.yml`

The workflow:
1. Checks data freshness
2. Skips if no new races have occurred
3. Extracts new data if available
4. Commits and pushes updates automatically

To manually trigger: Go to Actions tab → "Extract DHL F1 Pit Stop Data" → "Run workflow"