# F1 Fantasy Data

This directory contains data fetched from the F1 Fantasy API, providing driver statistics and race-by-race performance details for the current season.

## Data Files

### 1. `driver_overview.csv`
Contains high-level statistics for all drivers in the current F1 season.

**Key columns:**
- `player_id`: Unique identifier for each driver
- `player_name`: Driver's full name
- `team_name`: Current F1 team
- `current_price`: Driver's price in F1 Fantasy (in millions)
- `fantasy_points`: Total fantasy points for the season
- `fantasy_avg`: Average fantasy points per race
- `price_change`: Price change since season start
- `points_per_million`: Value metric (fantasy points per million cost)
- `overtake_points`: Points earned from overtaking
- `podiums`: Number of podium finishes
- `dnfs`: Number of DNFs (Did Not Finish)
- `fastest_laps`: Number of fastest laps
- `driver_of_day`: Number of Driver of the Day awards

### 2. `driver_details.csv`
Contains race-by-race breakdown for each driver.

**Key columns:**
- `player_id`: Driver identifier (links to driver_overview)
- `player_name`: Driver's name
- `gameday_id`: Race number (1, 2, 3, etc.)
- `total_points`: Fantasy points earned in this race
- `race_position`: Final race position
- `quali_position`: Qualifying position
- `beat_teammate`: Whether driver beat their teammate
- `overtaking_count`: Number of overtakes in the race
- `fastest_lap`: Whether driver set fastest lap
- `driver_of_day`: Whether driver won Driver of the Day

### 3. `.f1_fantasy_metadata.json`
Metadata about the data extraction:
- `last_updated`: When the data was last fetched
- `num_drivers`: Number of drivers in the dataset
- `num_races`: Number of races included
- `api_version`: Version of the API used
- `update_frequency`: How often data is updated

## Data Updates

- **Frequency**: Weekly (every Tuesday at 7:00 AM UTC)
- **Method**: GitHub Actions workflow (`fetch-f1-fantasy-data.yml`)
- **Source**: F1 Fantasy API (https://fantasy.formula1.com)

## Manual Execution (Testing/Debugging)

To run the F1 Fantasy data fetcher manually:

### Option 1: From project root (recommended)
```bash
python notebooks/advanced/f1_fantasy_fetcher.py
# Data will automatically save to /data/f1_fantasy/
```

### Option 2: From notebooks/advanced directory
```bash
cd notebooks/advanced
python f1_fantasy_fetcher.py
# Data will automatically save to /data/f1_fantasy/
```

### Option 3: With custom settings
```bash
# From project root with longer API delay (be extra respectful during testing)
python notebooks/advanced/f1_fantasy_fetcher.py --api-delay 1.0
```

### Option 4: Force specific output directory (not recommended)
```bash
# Only use if you need a different location for testing
python notebooks/advanced/f1_fantasy_fetcher.py --output-dir /absolute/path/to/output
```

### Verify the data was fetched correctly
```bash
# Check file sizes and line counts
ls -la data/f1_fantasy/
wc -l data/f1_fantasy/*.csv

# Preview the data
head -5 data/f1_fantasy/driver_overview.csv
head -5 data/f1_fantasy/driver_details.csv

# Check metadata
cat data/f1_fantasy/.f1_fantasy_metadata.json
```

### Troubleshooting
- If you get connection errors, the API might be temporarily down
- If data seems outdated, check if there was a recent race
- The fetcher will show progress logs and a summary of top drivers
- Typical run time: 30-60 seconds for all drivers

## Usage in ML Pipeline

This data can be integrated with the main F1 ML pipeline to:

1. **Enhance predictions** with fantasy metrics
2. **Validate model outputs** against fantasy point scoring
3. **Identify value drivers** for betting/fantasy strategies
4. **Track driver form** through price changes and recent performance

### Example Usage

```python
import pandas as pd

# Load driver overview
drivers = pd.read_csv('data/f1_fantasy/driver_overview.csv')

# Get top value drivers
value_drivers = drivers.nlargest(10, 'points_per_million')

# Load race details
details = pd.read_csv('data/f1_fantasy/driver_details.csv')

# Get driver's recent form (last 5 races)
driver_form = details[details['player_name'] == 'Lando Norris'].tail(5)
```

## Data Quality

- All monetary values are in millions (e.g., 31.4 = $31.4M)
- Missing data is handled gracefully with appropriate defaults
- Player IDs remain consistent throughout the season
- Data is validated before committing to ensure required columns exist

## Integration Points

1. **F1PerformanceAnalyzer**: Can use fantasy points as additional validation metric
2. **F1FeatureStore**: Fantasy metrics can be added as features (price changes, form)
3. **PrizePicksOptimizer**: Fantasy ownership % can inform contrarian strategies
4. **Backtesting**: Fantasy points provide ground truth for model evaluation

## Important Notes

- This is unofficial use of the F1 Fantasy API
- Data should be used respectfully and not for commercial purposes
- API structure may change without notice
- Always verify critical data points against official sources