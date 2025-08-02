# How to View the F1 Risk Dashboard

## Overview
The F1 Risk Dashboard is a comprehensive visualization showing risk metrics, portfolio analysis, and betting recommendations. It's generated as a PNG image file.

## Methods to View the Dashboard

### 1. Run the Full Prediction System
```bash
python f1_predictions_enhanced_v4.py
```
This will:
- Generate predictions using ensemble methods
- Create optimal betting portfolio with Kelly Criterion
- Calculate risk metrics
- Generate and save the dashboard to `pipeline_outputs/risk_dashboard.png`

### 2. Run the Test Script
```bash
python test_dashboard.py
```
This runs the full system with example data and shows you where the dashboard is saved.

### 3. View Sample Dashboard
```bash
python view_dashboard_example.py
```
This creates a dashboard with sample data to show what it looks like.

### 4. View in Jupyter Notebook
```python
from IPython.display import Image, display

# After running predictions
display(Image("pipeline_outputs/risk_dashboard.png"))
```

### 5. Open with System Image Viewer

**Windows:**
```bash
start pipeline_outputs/risk_dashboard.png
```

**Mac:**
```bash
open pipeline_outputs/risk_dashboard.png
```

**Linux:**
```bash
xdg-open pipeline_outputs/risk_dashboard.png
```

## Dashboard Components

The dashboard contains 7 panels:

1. **Portfolio Overview** (Top Left)
   - Bankroll amount
   - Total exposure
   - Expected value and ROI
   - Risk metrics

2. **Risk Gauge** (Top Center)
   - Visual risk score (0-100)
   - Color-coded risk level

3. **Exposure Breakdown** (Top Right)
   - Pie chart showing bet distribution
   - Breakdown by parlay type

4. **Returns Distribution** (Middle Left)
   - Monte Carlo simulation (10,000 runs)
   - Shows probability of different outcomes
   - Mean, median, and Value at Risk

5. **Win Probabilities** (Middle Right)
   - Bar chart of each bet's win probability
   - Stake size annotations
   - Color-coded by confidence

6. **Scenario Analysis** (Bottom Left)
   - Best case scenario
   - Expected outcome
   - Value at Risk (95% confidence)
   - Worst case scenario

7. **Historical Performance** (Bottom Right)
   - Tracks ROI and risk over time
   - Shows trends in betting performance

## Understanding the Metrics

- **Risk Score (0-100)**: Overall portfolio risk
  - 0-30: Low risk (conservative)
  - 30-60: Moderate risk (balanced)
  - 60-100: High risk (aggressive)

- **Value at Risk (VaR)**: Maximum expected loss at 95% confidence
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
- **Expected ROI**: Projected return on investment

## Customizing the Dashboard

To create a dashboard with your own portfolio:

```python
from f1_risk_dashboard import F1RiskDashboard

# Create dashboard
dashboard = F1RiskDashboard(bankroll=5000)  # Your bankroll

# Your portfolio dict
portfolio = {
    'bets': [...],  # Your bets
    'total_stake': 500,
    'expected_value': 650
}

# Generate dashboard
metrics = dashboard.calculate_risk_metrics(portfolio)
dashboard.create_dashboard(portfolio, {}, save_path="my_dashboard.png")
```

## Tips

1. The dashboard updates automatically when you run new predictions
2. Historical data accumulates over multiple runs
3. Save dashboards with timestamps to track progress
4. Use the risk score to adjust your betting strategy
5. Monitor the Sharpe ratio for consistent risk-adjusted returns