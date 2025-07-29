# F1 Pipeline GitHub Actions

This directory contains automated workflows for the F1 Machine Learning project. All workflows use the latest versions of GitHub Actions (v4) to avoid deprecation warnings.

## Workflows Overview

### 1. Daily F1 Data Update (`daily-f1-data-update.yml`)
- **Schedule**: Daily at 6:00 AM UTC
- **Purpose**: Updates F1DB data from the official repository
- **Features**:
  - Automatically checks if F1 season is complete
  - Stops running when no future races remain
  - Caches data for efficiency
  - Commits updates automatically
  - Creates GitHub issues for season completion notifications
- **Manual Trigger Options**:
  - `force_download`: Force re-download even if data is up to date

### 2. Daily F1 Pipeline Run (`daily-f1-pipeline.yml`)
- **Schedule**: Daily at 10:00 AM UTC
- **Purpose**: Regular pipeline execution to keep predictions updated
- **Features**:
  - Automatic F1DB data updates
  - Model retraining with latest data
  - Results archiving
  - Failure notifications via GitHub issues
- **Manual Trigger Options**:
  - `race_id`: Process specific race

### 3. Advanced Pipeline (`f1-pipeline-advanced.yml`)
- **Triggers**: 
  - Daily schedule at 10:00 AM UTC
  - Push to main branch
  - Pull requests
  - Manual dispatch
- **Features**:
  - Comprehensive testing before pipeline run
  - Validates notebooks
  - Backtesting capabilities
  - GitHub Pages deployment
  - MLflow tracking
  - Detailed notifications
- **Manual Trigger Options**:
  - `race_id`: Process specific race
  - `run_backtest`: Enable backtesting
  - `deploy_results`: Deploy to GitHub Pages

### 4. Race Weekend Predictions (`race-weekend-predictions.yml`)
- **Schedule**: 
  - Fridays at 12:00 PM UTC (Practice)
  - Saturdays at 10:00 AM UTC (Qualifying)
  - Sundays at 10:00 AM UTC (Race)
- **Features**:
  - Automatic race weekend detection
  - Session-specific predictions
  - Pull request creation with predictions
  - Social media-ready summaries
- **Manual Trigger Options**:
  - `session_type`: practice/qualifying/race
  - `update_predictions`: Update with latest data

### 5. Fetch Weather Data (`fetch-weather-data.yml`)
- **Schedule**: Daily at 8:00 AM UTC
- **Purpose**: Fetches weather data from Visual Crossing API
- **Features**:
  - Checks API rate limits before fetching
  - Caches weather data to avoid duplicate API calls
  - Creates issues if API key is not configured
  - Commits weather data updates
- **Manual Trigger Options**:
  - `year`: Year to fetch weather data for
  - `force_update`: Force update existing weather data

## Common Features

All workflows include:
- ✅ Latest GitHub Actions (v4) to avoid deprecation
- ✅ Proper error handling and notifications
- ✅ Caching for dependencies and data
- ✅ Manual trigger options via workflow_dispatch
- ✅ Artifact uploads for results
- ✅ Automated commits with github-actions[bot]

## Configuration Requirements

### Secrets Required
- `VISUAL_CROSSING_API_KEY`: For weather data fetching (get free key at https://www.visualcrossing.com/)
  - Free tier provides 1000 API calls per day
  - Required only for weather data workflow

### Automatic Secrets
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Data Dependencies
- F1DB data is automatically downloaded and updated
- Weather data requires API key configuration
- Pipeline expects data in `data/f1db/` directory

## Workflow Dependencies

```
daily-f1-data-update.yml
    ↓ (provides F1 race data)
daily-f1-pipeline.yml / f1-pipeline-advanced.yml
    ↓ (uses data for predictions)
race-weekend-predictions.yml (runs on race weekends)

fetch-weather-data.yml (independent, provides weather data)
```

## Caching Strategy

The workflows use GitHub Actions cache for:
- Python dependencies (`~/.cache/pip`)
- F1DB data (`data/f1db`)
- Weather data (`notebooks/advanced/data/weather_cache`)
- MLflow tracking data (`notebooks/advanced/mlflow`)
- Model artifacts

## Artifacts

Each run produces artifacts that are retained for:
- Pipeline results: 30 days
- MLflow tracking: 90 days
- Weather data: 30 days
- Error logs: 7 days
- Race predictions: 7 days

## Monitoring

### Success Notifications
- Commit comments on successful runs
- Pull requests with predictions
- GitHub Pages deployment
- Console output with statistics

### Failure Notifications
- GitHub Issues created on failure
- Error logs uploaded as artifacts
- Labels: `pipeline-failure`, `automated`, `weather-api`, `f1-data`

## Troubleshooting

### Common Issues and Fixes

1. **"No tests found" error**: ✅ Fixed - workflows now run actual test files (`test_*.py`)
2. **Deprecation warnings**: ✅ Fixed - all actions updated to v4
3. **YAML syntax errors**: ✅ Fixed - proper Python indentation in multi-line scripts
4. **Import Errors**: Ensure PYTHONPATH is set correctly
5. **Data Not Found**: Check F1DB cache is properly restored
6. **Weather API failures**: Configure `VISUAL_CROSSING_API_KEY` secret
7. **Memory Issues**: Ubuntu-latest provides 7GB RAM
8. **Timeout**: Default timeout is 6 hours

### Debug Mode

Enable debug logging by setting repository secret:
- Name: `ACTIONS_STEP_DEBUG`
- Value: `true`

## Local Testing

To test workflows locally, use [act](https://github.com/nektos/act):

```bash
# Test daily pipeline
act schedule -W .github/workflows/daily-f1-pipeline.yml

# Test with specific inputs
act workflow_dispatch -W .github/workflows/f1-pipeline-advanced.yml \
  -e '{"inputs":{"race_id":"monaco_2024","run_backtest":"true"}}'

# Test weather fetching
act workflow_dispatch -W .github/workflows/fetch-weather-data.yml \
  -e '{"inputs":{"year":"2024","force_update":"true"}}'
```

## Manual Triggers

All workflows support manual triggering via GitHub Actions UI:

1. Go to Actions tab
2. Select the workflow
3. Click "Run workflow"
4. Fill in optional parameters
5. Click "Run workflow" button

## Cost Optimization

- Workflows use caching extensively to reduce runtime
- Artifacts have retention policies to manage storage
- Tests run only when relevant files change
- Race weekend detection prevents unnecessary runs
- Season completion detection stops workflows when not needed
- Weather API calls are cached to avoid hitting rate limits

## Recent Updates

- **Fixed YAML syntax errors**: Python code properly indented in multi-line scripts
- **Updated to Actions v4**: All deprecated v3 actions updated to v4
- **Fixed pytest issues**: Tests now run on actual test files
- **Added weather workflow**: Separate workflow for weather data management
- **Improved error handling**: Better notifications and error reporting
- **Added season detection**: Workflows pause when F1 season is complete

## Future Enhancements

- [ ] Slack/Discord notifications
- [ ] Performance metrics dashboard
- [ ] A/B testing for strategies
- [ ] Multi-region deployment
- [ ] Container-based execution
- [ ] Real-time race data integration
- [ ] Enhanced weather prediction models