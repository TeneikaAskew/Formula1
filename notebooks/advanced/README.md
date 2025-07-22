# Advanced F1 Notebooks

These notebooks contain production-ready solutions for real-world F1 predictive applications.

## 🔄 F1DB Integration

These notebooks now support loading data from the official F1DB repository (https://github.com/f1db/f1db), which provides:
- Always up-to-date F1 data (updated after each race)
- Comprehensive datasets from 1950 to present
- Multiple format options (CSV, JSON, SQL)

See `F1DB_Data_Tutorial.ipynb` for details on using F1DB data.

## 📋 Requirements

These notebooks require the base packages plus additional dependencies:

```bash
pip install -r requirements-dev.txt
```

## 📚 Notebooks

1. **F1_Improved_Models.ipynb**
   - Fixes overfitting issues from original models
   - Implements proper temporal validation
   - Realistic performance metrics

2. **F1_Constructor_Driver_Evaluation.ipynb**
   - Driver evaluation system for constructor teams
   - ROI calculations for contract negotiations
   - Compatibility scoring

3. **F1_Betting_Market_Models.ipynb**
   - Calibrated probabilities for betting markets
   - Head-to-head predictions
   - DNF risk assessment

4. **Random_Forest_and_Gradient_Boosting.ipynb**
   - Advanced ensemble methods
   - SMOTE for class imbalance
   - Feature importance analysis

## 🚀 Quick Start

All notebooks automatically extract the F1 data on first run. Just open and run!