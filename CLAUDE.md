# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Formula One Predictive model - a comprehensive ML platform for F1 racing analytics serving constructor teams, betting markets, and fans/analysts. The project has evolved from a basic workshop into production-ready solutions with real-world applications.

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
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Pipeline and Analysis

```bash
# F1 Performance Analysis
python notebooks/advanced/f1_performance_analysis.py

# Enhanced Pipeline (from notebooks/advanced)
python run_enhanced_pipeline.py

# PrizePicks Data Analysis
python parse_prizepicks_wagers.py data/prizepicks/lineup.json
```

## Architecture and Key Components

### Data Flow
1. **Data Sources**: `/data/f1db/` contains comprehensive F1 CSV files (races, drivers, results, etc.)
2. **Feature Engineering**: 100+ engineered features including rolling averages, head-to-head stats, track-specific metrics
3. **Model Training**: Multiple models (Random Forest, XGBoost, LightGBM) with temporal validation
4. **Predictions**: Position predictions, DNF probability, points scoring, betting odds
5. **Optimization**: Kelly criterion for bet sizing, prize picks optimization

### Active Notebooks and Scripts
- `notebooks/advanced/F1DB_Data_Tutorial.ipynb` - F1 database exploration and analysis
- `notebooks/advanced/F1_Betting_Market_Models.ipynb` - Advanced market calibration techniques
- `notebooks/advanced/F1_Constructor_Driver_Evaluation.ipynb` - Constructor and driver evaluation
- `notebooks/advanced/F1_MLflow_Tracking.ipynb` - Machine learning experiment tracking
- `notebooks/advanced/F1_Prize_Picks_Optimizer.ipynb` - Betting optimization with Kelly criterion

### Core Python Scripts
- `notebooks/advanced/f1_performance_analysis.py` - Comprehensive F1 driver performance analysis
- `notebooks/advanced/run_enhanced_pipeline.py` - Enhanced F1 prediction pipeline
- `notebooks/advanced/f1_market_calibration.py` - Market calibration functions
- `parse_prizepicks_wagers.py` - PrizePicks betting data analysis
- `dhl_pitstop_scraper.py` - DHL pit stop data collection

### Documentation (Guides)
All documentation has been moved to `notebooks/advanced/guides/`:
- Architecture guides and summaries
- Usage instructions and tutorials
- Pipeline documentation
- Troubleshooting guides

### Important Classes and Functions

**F1 Performance Analysis (`notebooks/advanced/f1_performance_analysis.py`):**
- `F1PerformanceAnalyzer` - Main class for comprehensive F1 analysis
- Driver performance metrics across seasons and circuits
- Overtakes analysis with circuit-specific calculations
- Constructor and team performance evaluation
- Handles data from multiple F1DB sources with temporal consistency

**Enhanced Pipeline (`notebooks/advanced/run_enhanced_pipeline.py`):**
- Advanced F1 prediction pipeline with weather integration
- Multi-model ensemble predictions (Random Forest, XGBoost, LightGBM)
- Temporal validation and backtesting capabilities
- Automated report generation and model performance tracking

**PrizePicks Analysis (`parse_prizepicks_wagers.py`):**
- Parses PrizePicks API wager data from JSON format
- Calculates win rates, ROI, and profit/loss by sport and pick type
- Generates detailed performance summaries and CSV exports
- Handles player-level result tracking and analysis

**Market Calibration (`notebooks/advanced/f1_market_calibration.py`):**
- `OrdinalRegressionClassifier` - Handles ordered nature of F1 positions
- `calibrate_probabilities_isotonic()` - Isotonic regression for probability calibration
- `generate_betting_odds()` - Creates calibrated betting odds from predictions
- `kelly_criterion_bet_size()` - Optimal bet sizing with fractional Kelly

### Performance Analysis Features
- **Driver Performance**: Season and circuit-specific metrics for all current F1 drivers
- **Overtakes Analysis**: Circuit-specific overtaking statistics with previous race data
- **Constructor Evaluation**: Team performance across seasons and constructors
- **DHL Pit Stop Integration**: Real-time pit stop data with box time analysis
- **Temporal Consistency**: All analyses maintain proper data chronology

### Common Development Tasks

**Running F1 Performance Analysis:**
```bash
python notebooks/advanced/f1_performance_analysis.py
```
This generates comprehensive driver analysis with overtakes, performance metrics, and DHL pit stop data.

**PrizePicks Data Analysis:**
1. Export lineup data from PrizePicks (see PRIZEPICKS_MANUAL_EXPORT_GUIDE.md)
2. Save as `data/prizepicks/lineup.json`
3. Run: `python parse_prizepicks_wagers.py data/prizepicks/lineup.json`

**Enhanced Pipeline:**
```bash
cd notebooks/advanced
python run_enhanced_pipeline.py
```
Runs the full ML pipeline with weather integration and automated reporting.

### Environment Variables
Create `.env` file with:
```
ANTHROPIC_API_KEY=your-api-key-here
VISUAL_CROSSING_API_KEY=your_visual_crossing_api_key_here
```

### Service Ports
- Jupyter Lab: 8888
- API (future): 5000
- Streamlit (dev): 8501
- FastAPI (dev): 8000

### Data Directory Structure
```
/workspace/
├── data/
│   ├── f1db/                    # F1 database CSV files
│   ├── prizepicks/              # PrizePicks betting data
│   ├── dhl/                     # DHL pit stop data
│   ├── weather_cache/           # Weather API cache
│   └── f1_fantasy/              # F1 fantasy data
├── notebooks/advanced/
│   ├── guides/                  # All documentation
│   ├── data/                    # Notebook-specific data
│   └── pipeline_outputs/        # Pipeline results
```

### Critical Notes
- **Data Integrity**: Always maintain temporal consistency - no future data in training
- **Real Data Only**: Project uses actual F1 data, no synthetic fallbacks
- **Driver Consistency**: F1 performance analysis now shows all active drivers
- **Documentation**: All guides moved to `notebooks/advanced/guides/`
- **Clean Architecture**: Test files and utility scripts removed for clarity

### Recent Updates (2025-08-03)
- **Workspace Cleanup**: Removed all test files, utility scripts, and redundant Docker files
- **Documentation Organization**: Moved all markdown files to `notebooks/advanced/guides/`
- **Driver Analysis Fix**: Fixed `get_active_drivers()` to use driver_standings data properly
- **PrizePicks Integration**: Added comprehensive betting data analysis with API parser
- **Environment Simplification**: Removed redundant venv creation scripts