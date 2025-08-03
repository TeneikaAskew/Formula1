# CLAUDE.local.md

This file contains local/personal instructions for Claude Code that are not checked into version control.

## Local Development Notes

### Current Focus Areas
- F1 performance analysis with comprehensive driver metrics
- PrizePicks betting data analysis and ROI optimization
- DHL pit stop data integration for real-time insights

### Personal Workflow
1. **F1 Analysis**: Run `python notebooks/advanced/f1_performance_analysis.py` for latest driver performance
2. **PrizePicks**: Manual export from browser → `parse_prizepicks_wagers.py` for analysis
3. **Pipeline**: Use `run_enhanced_pipeline.py` for full ML predictions

### Key Insights from Recent Analysis
- F1 picks show +56.2% ROI (strongest performer)
- All 40+ drivers now appear consistently in performance analysis
- DHL pit stop data successfully integrated with overtakes analysis

### Environment Setup
- Using manual PrizePicks export method (no browser automation)
- Docker setup via `build-f1-claude.ps1` for development
- All documentation organized in `notebooks/advanced/guides/`

### Data Sources
- F1DB: Primary F1 historical data
- PrizePicks: Betting lineup performance via API export
- DHL: Real-time pit stop data with box times
- Weather: Visual Crossing API for race conditions

### Recent Optimizations (2025-08-03)
- Fixed driver consistency issues in performance analysis
- Cleaned up workspace (removed 14 test files, utility scripts)
- Streamlined PrizePicks workflow to API-based approach
- Organized all guides into dedicated folder structure

## Important Reminders
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary for achieving goals
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files unless explicitly requested