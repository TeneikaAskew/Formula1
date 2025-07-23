#!/usr/bin/env python3
"""
Run F1 Prize Picks Pipeline

Usage:
    python run_f1_pipeline.py              # Run for upcoming race
    python run_f1_pipeline.py --race-id 1234  # Run for specific race
    python run_f1_pipeline.py --backtest   # Run backtesting
    python run_f1_pipeline.py --schedule   # Schedule race weekend automation
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Add notebook directory to path
sys.path.append(str(Path(__file__).parent))

def run_master_notebook(race_id=None, mode='predict'):
    """Run the master pipeline notebook"""
    # Convert notebook to script first
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'script',
        'F1_Pipeline_Integration.ipynb'
    ])
    
    # Import and run
    from F1_Pipeline_Integration import F1PrizePipeline, PipelineConfig
    from F1_Pipeline_Integration import RaceWeekendAutomation, PerformanceMonitor
    
    # Load configuration
    config = PipelineConfig.load()
    
    # Initialize pipeline
    pipeline = F1PrizePipeline(config)
    
    if mode == 'schedule':
        # Run automation
        automation = RaceWeekendAutomation(pipeline)
        pipeline.load_data()
        
        upcoming = automation.get_race_schedule()
        if not upcoming.empty:
            next_race = upcoming.iloc[0]
            automation.schedule_race_analysis(next_race['raceId'], next_race['date'])
            print(f"Scheduled analyses for {next_race['name']}")
            automation.execute_scheduled_runs()
    elif mode == 'backtest':
        print("Running backtesting...")
        # Import and run backtesting notebook
        subprocess.run([
            sys.executable, '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            'F1_Backtesting_Framework.ipynb'
        ])
    else:
        # Normal prediction mode
        results = pipeline.run(race_id)
        
        if results:
            print("\nPipeline completed successfully!")
            print(f"Results saved to {config.output_dir}")
            
            # Track performance
            monitor = PerformanceMonitor(config.output_dir)
            monitor.load_metrics()
            monitor.track_predictions(results.get('predictions', pd.DataFrame()))
            monitor.save_metrics()
        else:
            print("\nPipeline failed. Check logs for details.")

def main():
    parser = argparse.ArgumentParser(description='Run F1 Prize Picks Pipeline')
    parser.add_argument('--race-id', type=int, help='Specific race ID to analyze')
    parser.add_argument('--backtest', action='store_true', 
                       help='Run backtesting instead of predictions')
    parser.add_argument('--schedule', action='store_true',
                       help='Schedule automated race weekend analyses')
    
    args = parser.parse_args()
    
    if args.backtest:
        run_master_notebook(mode='backtest')
    elif args.schedule:
        run_master_notebook(mode='schedule')
    else:
        run_master_notebook(race_id=args.race_id)

if __name__ == "__main__":
    main()
