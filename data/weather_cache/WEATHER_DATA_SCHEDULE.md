# Weather Data Fetching Schedule

## Current Status (as of today)
- **Races fetched**: 41 out of 107 (38.3%)
- **Years covered**: 2023-2024 (complete)
- **Years missing**: 2020-2022
- **API used**: Visual Crossing Weather API

## API Limits & Cost Model
Visual Crossing uses a **credit-based** pricing model:
- **Free tier**: 1,000 credits per day
- **Cost per race**: Each race query costs ~24 credits (for hourly weather data)
- **Races per day**: Can fetch ~40 races per day (leaving buffer for safety)

Your account summary showed:
- 41 queries = 985 credits used
- ~24 credits per race query

## Fetching Schedule

### Remaining Work
- **Races to fetch**: 66 (all from 2020-2022)
- **Credits needed**: 1,584 (66 × 24)
- **Days needed**: 2 days

### Recommended Schedule

**Day 1** (when credit limit resets):
```bash
python fetch_weather_data.py --start-year 2020 --end-year 2024
```
- Will fetch ~41 races (uses ~984 credits)
- Automatically stops at 990 credits to leave a small buffer

**Day 2**:
```bash
python fetch_weather_data.py --start-year 2020 --end-year 2024
```
- Will fetch remaining ~25 races (uses ~600 credits)
- Completes all 2020-2022 data

## Usage Options

### Check status only (no API calls):
```bash
python fetch_weather_data.py --status
```

### Fetch with custom credit limit:
```bash
python fetch_weather_data.py --max-credits 500
```

### Fetch specific year range:
```bash
python fetch_weather_data.py --start-year 2020 --end-year 2022
```

## Current Weather Data

The pipeline currently has:
- ✅ Real weather data for 41 races (2023-2024)
- ⚠️ Historical averages for 66 races (2020-2022)
- No synthetic/random data - all fallbacks use realistic historical patterns

## After Completion

Once all weather data is fetched:
- 100% real weather data for all 2020-2024 races
- Improved model accuracy for historical backtesting
- More reliable weather-related feature importance