# F1 Pit Stop Box Time Integration Guide (DEPRECATED)

> **Note**: FastF1 integration has been removed from this project. This document is kept for historical reference only.

## Overview

This guide previously documented how to integrate FastF1 pit stop box times with f1db data for use in PrizePicks and other fantasy F1 platforms that require actual pit stop durations.

## The Problem

The f1db dataset contains pit stop information but lacks the actual "box time" - the duration the car is stationary in the pit box. This is critical for:
- PrizePicks pit stop duration props
- Fantasy F1 scoring that includes pit stop performance
- Analysis of team pit crew performance

## The Solution: FastF1 Integration (REMOVED)

FastF1 was an open-source Python library that accessed official F1 timing data, including pit stop durations. This integration has been removed from the project.

### Installation (NO LONGER NEEDED)

```bash
# FastF1 is no longer used in this project
# pip install fastf1>=3.0.0  # DEPRECATED
```

### Key Components (REMOVED)

1. ~~**fastf1_box_time_extractor.py** - Main extraction module~~ (REMOVED)
2. ~~**Driver mapping** - Maps f1db driver IDs to FastF1 3-letter codes~~ (REMOVED)
3. ~~**Data enrichment pipeline** - Adds box times to existing f1db pit stops~~ (REMOVED)

## Usage Example

```python
from f1db_data_loader import load_f1db_data
from fastf1_box_time_extractor import F1BoxTimeExtractor

# Load f1db data
data = load_f1db_data()
pit_stops = data['pit_stops']
races = data['races']

# Create extractor and enrich pit stops
extractor = F1BoxTimeExtractor()
enriched_stops = extractor.enrich_f1db_pitstops(pit_stops, races, year_from=2020)

# Filter to stops with box times
stops_with_times = enriched_stops[enriched_stops['box_time_seconds'].notna()]
print(f"Pit stops with box times: {len(stops_with_times)}/{len(enriched_stops)}")

# Analyze box times
print(f"Average box time: {stops_with_times['box_time_seconds'].mean():.2f}s")
print(f"Fastest stop: {stops_with_times['box_time_seconds'].min():.2f}s")
```

## Data Mapping Challenges

### 1. Driver Identification
- **Challenge**: f1db uses driver IDs like "lewis-hamilton", FastF1 uses 3-letter codes like "HAM"
- **Solution**: Maintain a mapping dictionary in the extractor class
- **Maintenance**: Update mapping when new drivers join F1

### 2. Data Availability
- **Challenge**: FastF1 data quality varies by session and year
- **Solution**: Use multiple extraction methods as fallbacks
- **Coverage**: Best results from 2018 onwards

### 3. Race Identification
- **Challenge**: Race names may differ between f1db and FastF1
- **Solution**: Use flexible matching and year/round number as fallback

## Typical Box Time Ranges

For PrizePicks and fantasy purposes, typical F1 pit stop box times are:

- **Ultra-fast**: < 2.2 seconds (rare, perfect stops)
- **Fast**: 2.2 - 2.5 seconds (top teams, good execution)
- **Normal**: 2.5 - 3.0 seconds (standard stops)
- **Slow**: 3.0 - 4.0 seconds (minor issues)
- **Problem stops**: > 4.0 seconds (equipment issues, unsafe releases, etc.)

## Integration with ML Pipeline

To use enriched pit stop data in predictions:

```python
# In your feature engineering
def add_pit_stop_features(df, enriched_stops):
    # Average team pit stop performance
    team_avg_stops = enriched_stops.groupby(['constructorId', 'year'])['box_time_seconds'].agg(['mean', 'std'])
    
    # Driver-specific pit stop history
    driver_pit_performance = enriched_stops.groupby(['driverId', 'year'])['box_time_seconds'].agg(['mean', 'min', 'count'])
    
    # Merge features
    df = df.merge(team_avg_stops, on=['constructorId', 'year'], how='left')
    df = df.merge(driver_pit_performance, on=['driverId', 'year'], how='left')
    
    return df
```

## Caching Strategy

FastF1 downloads and caches data locally:

```python
# Enable caching (already done in the extractor)
import fastf1
fastf1.Cache.enable_cache("/workspace/data/fastf1_cache")
```

This prevents re-downloading data and speeds up subsequent runs.

## Limitations and Considerations

1. **Historical Coverage**: FastF1 data is most reliable from 2018 onwards
2. **Live Data**: FastF1 can access live timing during races (with slight delay)
3. **Data Gaps**: Not all sessions have complete pit stop timing data
4. **Circuit Variations**: Pit lane characteristics vary by circuit, affecting total pit time

## Example Analysis for PrizePicks

```python
# Get recent race pit stops for props analysis
def analyze_driver_pit_stops(enriched_stops, driver_id, last_n_races=5):
    driver_stops = enriched_stops[
        (enriched_stops['driverId'] == driver_id) & 
        (enriched_stops['box_time_seconds'].notna())
    ].sort_values('raceId', ascending=False)
    
    recent_stops = driver_stops.head(last_n_races * 2)  # Assuming ~2 stops per race
    
    if len(recent_stops) > 0:
        stats = {
            'avg_box_time': recent_stops['box_time_seconds'].mean(),
            'min_box_time': recent_stops['box_time_seconds'].min(),
            'max_box_time': recent_stops['box_time_seconds'].max(),
            'consistency': recent_stops['box_time_seconds'].std(),
            'sample_size': len(recent_stops)
        }
        
        # PrizePicks typically sets lines around 2.4-2.6 seconds
        stats['under_2.4_rate'] = (recent_stops['box_time_seconds'] < 2.4).mean()
        stats['under_2.6_rate'] = (recent_stops['box_time_seconds'] < 2.6).mean()
        
        return stats
    return None

# Example usage
lewis_stats = analyze_driver_pit_stops(enriched_stops, 'lewis-hamilton')
if lewis_stats:
    print(f"Hamilton avg box time: {lewis_stats['avg_box_time']:.2f}s")
    print(f"Under 2.6s rate: {lewis_stats['under_2.6_rate']:.1%}")
```

## Next Steps

1. **Build Historical Database**: Run enrichment on all historical f1db data
2. **Real-time Updates**: Set up pipeline to enrich new races as they happen
3. **Team Analysis**: Analyze pit crew performance trends by constructor
4. **ML Features**: Incorporate pit stop performance into race prediction models

## Troubleshooting

### Common Issues

1. **"No module named fastf1"**
   - Solution: `pip install fastf1>=3.0.0`

2. **"No pit stop data found"**
   - Check race name spelling matches FastF1 conventions
   - Try using round number instead of race name

3. **Low enrichment rate**
   - Update driver mapping for new/changed drivers
   - Check FastF1 data availability for specific races

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test single race
extractor = F1BoxTimeExtractor()
test_stops = extractor.extract_box_times_for_race(2023, 'Monaco')
print(test_stops)
```

## Data Quality Metrics

When evaluating enriched data:

- **Coverage**: Aim for >70% of pit stops having box times (2020+ races)
- **Validation**: Box times should typically be 2.0-4.0 seconds
- **Outliers**: Investigate any stops <1.8s or >5.0s for data quality

## Conclusion

Integrating FastF1 box times with f1db pit stop data provides the missing piece needed for accurate pit stop duration predictions and fantasy F1 applications. While there are challenges with data mapping and availability, the enrichment pipeline provides a robust solution for adding this critical data.