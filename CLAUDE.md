# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Formula One Machine Learning Workshop - a comprehensive ML platform for F1 racing analytics serving constructor teams, betting markets, and fans/analysts. The project has evolved from a basic workshop into production-ready solutions with real-world applications.

## Development Commands

### Quick Start (Docker - Recommended)

**Windows PowerShell:**
```powershell
.\build-f1-claude.ps1 setup    # First time setup
.\build-f1-claude.ps1 jupyter  # Start Jupyter notebooks
.\build-f1-claude.ps1 claude   # Use Claude CLI
.\build-f1-claude.ps1 status   # Check service status
.\build-f1-claude.ps1 logs     # View logs
```

**Mac/Linux:**
```bash
./build-f1-claude.sh setup    # First time setup
./build-f1-claude.sh jupyter  # Start Jupyter notebooks
./build-f1-claude.sh claude   # Use Claude CLI
./build-f1-claude.sh status   # Check service status
./build-f1-claude.sh logs     # View logs
```

### Python Virtual Environment (Alternative)

**Windows:**
```batch
.\create_venv_windows.bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests and Pipeline

```bash
# Test setup (from notebooks/advanced)
python test_windows_setup.py

# Run F1 pipeline
python run_f1_pipeline.py              # Upcoming race
python run_f1_pipeline.py --race-id 1234  # Specific race
python run_f1_pipeline.py --backtest   # Backtesting mode
```

## Architecture and Key Components

### Data Flow
1. **Data Sources**: `/data/f1db/` contains comprehensive F1 CSV files (races, drivers, results, etc.)
2. **Feature Engineering**: 100+ engineered features including rolling averages, head-to-head stats, track-specific metrics
3. **Model Training**: Multiple models (Random Forest, XGBoost, LightGBM) with temporal validation
4. **Predictions**: Position predictions, DNF probability, points scoring, betting odds
5. **Optimization**: Kelly criterion for bet sizing, prize picks optimization

### Key Notebooks
- `notebooks/advanced/F1_Improved_RF_GB_Models.ipynb` - Core ML models with proper validation
- `notebooks/advanced/F1_Constructor_Driver_Evaluation.ipynb` - Driver evaluation system
- `notebooks/advanced/F1_Prize_Picks_Optimizer.ipynb` - Betting optimization with Kelly criterion
- `notebooks/advanced/F1_Pipeline_Integration.ipynb` - Full pipeline orchestration
- `notebooks/advanced/F1_Backtesting_Framework.ipynb` - Historical race prediction testing and validation

### Important Classes and Functions

**Data Pipeline (`notebooks/advanced/run_f1_pipeline.py`):**
- `F1PredictionsGenerator` - Main class for generating predictions
- Handles data loading, feature engineering, model training, and prediction generation
- Includes caching mechanism to avoid reprocessing

**Model Architecture:**
- Uses ensemble of Random Forest, XGBoost, and LightGBM
- Temporal validation to prevent data leakage
- Feature importance analysis with SHAP
- Calibrated probabilities for betting markets

**Backtesting Framework (`notebooks/advanced/F1_Backtesting_Framework.ipynb`):**
- `F1Backtester` - Class for running historical race predictions
- Simulates pre-race conditions by splitting data temporally
- Compares predictions with actual results
- Calculates accuracy metrics (position error, winner/podium/points accuracy)
- Batch testing capabilities for multiple races
- Driver-specific performance analysis

### Model Performance Metrics
- Winner Prediction: 65-70% accuracy
- Podium Prediction: 72-75% accuracy
- Points Finish: 78-82% accuracy
- Backtesting ROI: 15-25% (moderate Kelly strategy)

### Common Development Tasks

**Adding New Features:**
1. Update feature engineering in relevant notebooks
2. Ensure temporal consistency (no future data leakage)
3. Add to pipeline configuration if needed

**Model Improvements:**
1. Always use temporal validation (train on past, test on future)
2. Check for overfitting with backtesting
3. Update pipeline config with new model parameters

**Running Backtests:**
```python
# In notebooks or pipeline
python run_f1_pipeline.py --backtest --start-date 2023-01-01 --end-date 2023-12-31

# Using the Backtesting Framework notebook
# Open notebooks/advanced/F1_Backtesting_Framework.ipynb
# Features: temporal data splitting, single/batch race testing, accuracy metrics, visualizations
```

### Environment Variables
Create `.env` file with:
```
ANTHROPIC_API_KEY=your-api-key-here
```

### Service Ports
- Jupyter Lab: 8888
- API (future): 5000
- Streamlit (dev): 8501
- FastAPI (dev): 8000

### Data Directory Structure
The pipeline automatically searches for data in:
- `../../data/f1db` (from notebooks/advanced)
- `../data/f1db` (from notebooks)
- `data/f1db` (from workspace root)
- `./data/f1db` (current directory)

### Critical Notes
- Always maintain temporal integrity in ML models (no future data in training)
- The project uses real F1 data only - no synthetic/fallback data
- Models are optimized for both accuracy and calibration (important for betting)
- Pipeline includes automatic caching to speed up repeated runs