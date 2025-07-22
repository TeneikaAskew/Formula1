# F1 Pipeline GitHub Actions

This directory contains automated workflows for the F1 Prize Picks optimization pipeline.

## Workflows

### 1. Daily F1 Pipeline Run (`daily-f1-pipeline.yml`)
- **Schedule**: Runs daily at 10:00 AM UTC
- **Purpose**: Regular pipeline execution to keep predictions updated
- **Features**:
  - Automatic F1DB data updates
  - Model retraining with latest data
  - Results archiving
  - Failure notifications

### 2. Advanced Pipeline (`f1-pipeline-advanced.yml`)
- **Triggers**: 
  - Daily schedule
  - Push to main branch
  - Pull requests
  - Manual dispatch
- **Features**:
  - Comprehensive testing before pipeline run
  - Backtesting capabilities
  - GitHub Pages deployment
  - MLflow tracking
  - Detailed notifications

### 3. Race Weekend Predictions (`race-weekend-predictions.yml`)
- **Schedule**: 
  - Fridays at 12:00 PM UTC (Practice)
  - Saturdays at 10:00 AM UTC (Qualifying)
  - Sundays at 10:00 AM UTC (Race)
- **Features**:
  - Automatic race weekend detection
  - Session-specific predictions
  - Pull request creation with predictions
  - Social media-ready summaries

## Manual Triggers

All workflows support manual triggering via GitHub Actions UI:

1. Go to Actions tab
2. Select the workflow
3. Click "Run workflow"
4. Fill in optional parameters:
   - `race_id`: Process specific race
   - `run_backtest`: Enable backtesting
   - `deploy_results`: Deploy to GitHub Pages

## Configuration

### Secrets Required
No secrets required for basic operation. The workflows use `GITHUB_TOKEN` which is automatically provided.

### Caching
The workflows use GitHub Actions cache for:
- Python dependencies
- F1DB data
- MLflow tracking data
- Model artifacts

### Artifacts
Each run produces artifacts that are retained for:
- Pipeline results: 30 days
- MLflow tracking: 90 days
- Error logs: 7 days

## Monitoring

### Success Notifications
- Commit comments on successful runs
- Pull requests with predictions
- GitHub Pages deployment

### Failure Notifications
- GitHub Issues created on failure
- Error logs uploaded as artifacts
- Labels: `pipeline-failure`, `automated`

## Local Testing

To test workflows locally, use [act](https://github.com/nektos/act):

```bash
# Test daily pipeline
act schedule -W .github/workflows/daily-f1-pipeline.yml

# Test with specific inputs
act workflow_dispatch -W .github/workflows/f1-pipeline-advanced.yml \
  -e '{"inputs":{"race_id":"monaco_2024","run_backtest":"true"}}'
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Data Not Found**: Check F1DB cache is properly restored
3. **Memory Issues**: Ubuntu-latest provides 7GB RAM
4. **Timeout**: Default timeout is 6 hours

### Debug Mode

Enable debug logging by setting repository secret:
- Name: `ACTIONS_STEP_DEBUG`
- Value: `true`

## Cost Optimization

- Workflows use caching extensively to reduce runtime
- Artifacts have retention policies to manage storage
- Tests run only when relevant files change
- Race weekend detection prevents unnecessary runs

## Future Enhancements

- [ ] Slack/Discord notifications
- [ ] Performance metrics dashboard
- [ ] A/B testing for strategies
- [ ] Multi-region deployment
- [ ] Container-based execution