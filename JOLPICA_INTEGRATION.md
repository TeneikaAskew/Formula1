# Jolpica F1 API Integration Documentation

## Overview

This document describes the integration of Jolpica F1 API lap timing data into the F1 ML pipeline to enhance overtakes analysis and points predictions.

## Implementation Summary

### 1. Data Fetcher (`jolpica_laps_fetcher.py`)

Created a comprehensive data fetcher with the following features:
- **Pagination support**: Handles API's limit of 2000 records per request
- **Rate limiting protection**: Implements retry logic with 30-second delays on 429 errors
- **Incremental fetching**: Tracks already fetched races to avoid re-downloading
- **Data consolidation**: Creates unified Parquet and CSV files for analysis

### 2. Data Structure

```
data/jolpica/
├── laps/
│   ├── 2020/
│   │   ├── 2020_round_01_laps.json
│   │   ├── 2020_round_02_laps.json
│   │   └── ...
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   └── 2024/
├── all_laps_2020_2024.parquet  # Consolidated dataset
├── all_laps_2020_2024.csv      # CSV version
├── fetch_metadata.json          # Tracking fetched races
└── fetch_report.json            # Summary statistics
```

### 3. API Details

- **Base URL**: `http://api.jolpi.ca/ergast/f1/`
- **Endpoints Used**:
  - `/{year}.json` - Get season race list
  - `/{year}/{round}/laps.json` - Get lap timings for specific race
- **Rate Limits**: Approximately 8-10 requests before hitting 429 error
- **Data Coverage**: 1996-present (we fetched 2020-2024)

### 4. Data Schema

Each lap record contains:
```json
{
  "season": 2024,
  "round": 1,
  "raceName": "Bahrain Grand Prix",
  "circuitId": "bahrain",
  "date": "2024-03-02",
  "lap": 1,
  "driverId": "max_verstappen",
  "position": 1,
  "time": "1:37.284"
}
```

### 5. Current Status

Due to API rate limiting, we successfully fetched:
- **2020 Season**: 8 races (out of 17)
- **2021-2024**: Pending due to rate limits

Total records fetched: 8,039 lap timings

### 6. Usage in Overtakes Analysis

The lap data enables:
1. **True Overtake Detection**: Compare positions between consecutive laps
2. **Overtaking Patterns**: Identify where and when overtakes happen
3. **Driver Performance**: Track position changes throughout races
4. **Circuit Analysis**: Understand overtaking opportunities by track

### 7. Integration with F1 Performance Analysis

To integrate with `f1_performance_analysis.py`:

```python
# Load Jolpica laps data
jolpica_laps = pd.read_parquet('data/jolpica/all_laps_2020_2024.parquet')

# Calculate actual overtakes (position improvements)
def calculate_real_overtakes(jolpica_laps, race_id):
    race_laps = jolpica_laps[
        (jolpica_laps['season'] == race_year) & 
        (jolpica_laps['round'] == race_round)
    ].sort_values(['lap', 'position'])
    
    overtakes = []
    for lap in range(2, race_laps['lap'].max() + 1):
        prev_lap = race_laps[race_laps['lap'] == lap - 1]
        curr_lap = race_laps[race_laps['lap'] == lap]
        
        for driver in curr_lap['driverId'].unique():
            prev_pos = prev_lap[prev_lap['driverId'] == driver]['position'].values
            curr_pos = curr_lap[curr_lap['driverId'] == driver]['position'].values
            
            if prev_pos and curr_pos and curr_pos[0] < prev_pos[0]:
                overtakes.append({
                    'driver': driver,
                    'lap': lap,
                    'positions_gained': int(prev_pos[0] - curr_pos[0])
                })
    
    return pd.DataFrame(overtakes)
```

### 8. Recommendations

1. **Complete Data Fetch**: Run the fetcher during off-peak hours or implement exponential backoff
2. **Cache Strategy**: Use the consolidated Parquet file for fast loading
3. **Merge with F1DB**: Join on race/driver identifiers for comprehensive analysis
4. **Real-time Updates**: Schedule weekly fetches for latest race data

### 9. Running the Fetcher

```bash
# Fetch all data (with rate limit handling)
python jolpica_laps_fetcher.py

# Fetch specific season
python -c "
from jolpica_laps_fetcher import JolpicaLapsFetcher
fetcher = JolpicaLapsFetcher()
fetcher.fetch_season_laps(2024)
"
```

### 10. Next Steps

1. Complete fetching remaining seasons (2021-2024)
2. Integrate lap-by-lap analysis into overtakes calculations
3. Add lap timing analysis to predict race pace
4. Create visualizations of overtaking patterns
5. Enhance ML features with actual overtake counts

## Error Handling

The fetcher includes robust error handling:
- Automatic retry on rate limits (429 errors)
- Metadata tracking to resume interrupted fetches
- Error logging for debugging
- Graceful handling of missing data

## Performance Considerations

- Each race fetch takes 10-20 seconds (depending on race length)
- Full 5-year fetch estimated at 2-3 hours with rate limits
- Consolidated dataset is ~10MB in Parquet format
- Consider using the Parquet file for faster loading in analysis