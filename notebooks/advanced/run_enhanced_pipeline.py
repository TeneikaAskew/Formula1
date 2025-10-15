#!/usr/bin/env python3
"""
Enhanced F1 Pipeline Orchestrator

This script orchestrates the enhanced F1 prediction pipeline, managing:
- Shared data loading across all components
- Parallel execution where appropriate
- Configuration management
- Error handling and logging
"""

import os
import sys
import json
import yaml
import logging
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1db_data_loader import F1DBDataLoader


def resolve_config_path(config_path: str) -> Path:
    """Resolve a configuration path relative to this file.

    The function first attempts to resolve the provided path relative to the
    directory containing this script. If that file does not exist, the original
    user-specified path (expanded and resolved where possible) is returned so
    the caller can handle any missing-file errors with an informative path.
    """

    user_path = Path(config_path).expanduser()
    script_dir = Path(__file__).parent

    # Try resolving relative to the script directory first
    candidate = (script_dir / user_path).resolve(strict=False)
    if candidate.exists():
        return candidate

    # Fall back to the user-supplied path (resolved when possible)
    return user_path.resolve(strict=False)


class EnhancedF1Pipeline:
    """Orchestrates the enhanced F1 prediction pipeline"""
    
    def __init__(self, config_path: str = "pipeline_config_enhanced.yaml"):
        """Initialize the pipeline with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        self.logger.info(f"Configuration loaded from {self.resolved_config_path}")
        self.results = {}
        self.execution_times = {}
        self.data = None
        self.data_cache_file = None

    def _load_config(self) -> Dict:
        """Load pipeline configuration from YAML file"""
        self.resolved_config_path = resolve_config_path(self.config_path)
        config_path = self.resolved_config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create logger
        self.logger = logging.getLogger('EnhancedF1Pipeline')
        self.logger.setLevel(log_level)
        
        formatter = logging.Formatter(
            log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_config.get('file'):
            log_file = Path(log_config['file'])
            log_file.parent.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            if formatter:
                file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if not self.logger.handlers:
            # Ensure the logger has at least a null handler so logging calls are safe
            self.logger.addHandler(logging.NullHandler())
    
    def load_shared_data(self):
        """Load F1DB data once for sharing across all components"""
        self.logger.info("Loading F1DB data...")
        start_time = time.time()
        
        try:
            # Check if caching is enabled
            if self.config['data'].get('cache_enabled', True):
                cache_dir = Path(self.config['data'].get('cache_directory', '/tmp/f1db_cache'))
                cache_dir.mkdir(exist_ok=True)
                self.data_cache_file = cache_dir / 'f1db_data.pkl'
                
                # Check if cached data exists and is fresh
                if self.data_cache_file.exists():
                    cache_age_hours = (time.time() - self.data_cache_file.stat().st_mtime) / 3600
                    ttl_hours = self.config['data'].get('cache_ttl_hours', 24)
                    
                    if cache_age_hours < ttl_hours:
                        self.logger.info(f"Loading cached data (age: {cache_age_hours:.1f} hours)")
                        import pickle
                        with open(self.data_cache_file, 'rb') as f:
                            self.data = pickle.load(f)
                        self.execution_times['data_loading'] = time.time() - start_time
                        return
            
            # Load fresh data
            loader = F1DBDataLoader()
            self.data = loader.get_core_datasets()
            
            # Cache the data if enabled
            if self.config['data'].get('cache_enabled', True) and self.data_cache_file:
                self.logger.info("Caching data for future runs...")
                import pickle
                with open(self.data_cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
            
            self.execution_times['data_loading'] = time.time() - start_time
            self.logger.info(f"Data loading completed in {self.execution_times['data_loading']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def run_performance_analysis(self) -> Dict:
        """Run performance analysis component"""
        if not self.config['performance_analysis'].get('enabled', True):
            self.logger.info("Performance analysis is disabled in configuration")
            return {}
            
        self.logger.info("Running performance analysis...")
        start_time = time.time()
        
        try:
            # Create a temporary script to run with shared data
            script = """
import sys
import json
import pickle
from pathlib import Path

# Load shared data
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

# Import and run performance analysis
from f1_performance_analysis import F1PerformanceAnalyzer

analyzer = F1PerformanceAnalyzer(data)

# Run various analyses
next_race = analyzer.get_next_race()
active_drivers = analyzer.get_active_drivers()

# Format next race properly
if hasattr(next_race, 'to_dict'):
    next_race_dict = next_race.to_dict()
else:
    next_race_dict = {
        'name': str(next_race.get('officialName', 'Unknown')) if hasattr(next_race, 'get') else 'Unknown',
        'date': str(next_race.get('date', 'TBD')) if hasattr(next_race, 'get') else 'TBD',
        'circuit': str(next_race.get('circuitId', 'Unknown')) if hasattr(next_race, 'get') else 'Unknown'
    }

results = {
    'next_race': next_race_dict,
    'active_drivers': active_drivers.to_dict() if hasattr(active_drivers, 'to_dict') else [],
    'season_stats': analyzer.get_season_stats() if hasattr(analyzer, 'get_season_stats') else {}
}

# Save results
output_file = sys.argv[2]
Path(output_file).parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
"""
            
            # Save data for subprocess
            import pickle
            import tempfile
            temp_data_file = Path(tempfile.gettempdir()) / 'perf_analysis_data.pkl'
            with open(temp_data_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Run the analysis
            output_file = self.config['performance_analysis']['output_file']
            result = subprocess.run(
                [sys.executable, '-c', script, str(temp_data_file), output_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Performance analysis failed: {result.stderr}")
            
            # Load results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            self.execution_times['performance_analysis'] = time.time() - start_time
            self.logger.info(f"Performance analysis completed in {self.execution_times['performance_analysis']:.2f} seconds")
            
            # Clean up
            temp_data_file.unlink()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            if self.config['pipeline'].get('continue_on_error', True):
                return {'error': str(e)}
            raise
    
    def run_v3_predictions(self) -> Dict:
        """Run v3 predictions"""
        if not self.config['predictions_v3'].get('enabled', True):
            self.logger.info("V3 predictions are disabled in configuration")
            return {}
            
        self.logger.info("Running V3 predictions...")
        start_time = time.time()
        
        try:
            # Create environment with configuration
            env = os.environ.copy()
            env['F1_PIPELINE_CONFIG'] = json.dumps(self.config['predictions_v3'])
            
            # Run v3 predictions
            script = """
import os
import sys
import json
import pickle
from pathlib import Path

# Load shared data
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

# Get configuration
config = json.loads(os.environ.get('F1_PIPELINE_CONFIG', '{}'))

# Monkey patch the data loader to use shared data
class MockDataLoader:
    def get_core_datasets(self):
        return data

# Replace the import
import f1db_data_loader
f1db_data_loader.F1DBDataLoader = MockDataLoader

# Now import and run v3
from f1_predictions_enhanced_v3 import F1PrizePicksPredictor

# Create predictor with config
predictor = F1PrizePicksPredictor()

# Apply configuration
for key, value in config.get('default_lines', {}).items():
    if key in predictor.default_lines:
        predictor.default_lines[key] = value

predictor.calibration_enabled = config.get('calibration_enabled', True)
predictor.hierarchical_priors_enabled = config.get('hierarchical_priors_enabled', True)
predictor.contextual_enabled = config.get('contextual_enabled', True)

# Disable prompts
import io
sys.stdin = io.StringIO('1\\n')  # Choose default lines

# Generate predictions
predictions = predictor.generate_all_predictions()

# Save results
output_file = config.get('output_file', 'pipeline_outputs/enhanced_predictions_v3.json')
Path(output_file).parent.mkdir(exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(predictions, f, indent=2, default=str)

print("V3 predictions completed successfully")
"""
            
            # Save data for subprocess
            import pickle
            import tempfile
            temp_data_file = Path(tempfile.gettempdir()) / 'v3_data.pkl'
            with open(temp_data_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Run the predictions
            result = subprocess.run(
                [sys.executable, '-c', script, str(temp_data_file)],
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                raise Exception(f"V3 predictions failed: {result.stderr}")
            
            # Load results
            output_file = self.config['predictions_v3']['output_file']
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            self.execution_times['v3_predictions'] = time.time() - start_time
            self.logger.info(f"V3 predictions completed in {self.execution_times['v3_predictions']:.2f} seconds")
            
            # Clean up
            temp_data_file.unlink()
            
            return results
            
        except Exception as e:
            self.logger.error(f"V3 predictions failed: {e}")
            if self.config['pipeline'].get('continue_on_error', True):
                return {'error': str(e)}
            raise
    
    def run_v3_weather_predictions(self) -> Dict:
        """Run v3 weather predictions"""
        if not self.config['predictions_v3_weather'].get('enabled', True):
            self.logger.info("V3 weather predictions are disabled in configuration")
            return {}
            
        self.logger.info("Running V3 weather predictions...")
        start_time = time.time()
        
        try:
            # Create environment with configuration
            env = os.environ.copy()
            env['F1_PIPELINE_CONFIG'] = json.dumps(self.config['predictions_v3_weather'])
            
            # Create script to run v3 weather with shared data
            script = """
import os
import sys
import json
import pickle
from pathlib import Path

# Load shared data
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

# Get configuration
config = json.loads(os.environ.get('F1_PIPELINE_CONFIG', '{}'))

# Monkey patch the data loader to use shared data
class MockDataLoader:
    def get_core_datasets(self):
        return data

# Replace the import
import f1db_data_loader
f1db_data_loader.F1DBDataLoader = MockDataLoader

# Import and run v3 weather
from f1_predictions_enhanced_v3_weather import main

# Disable stdin prompts (v3_weather calls the base v3 which prompts)
import io
sys.stdin = io.StringIO('1\\n')  # Choose default lines

# Override output path if specified in config
original_main = main

def patched_main():
    predictions = original_main()
    
    # Save to configured output file
    output_file = config.get('output_file', 'pipeline_outputs/enhanced_predictions_v3_weather.json')
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    return predictions

# Run weather predictions
predictions = patched_main()
print("V3 weather predictions completed successfully")
"""
            
            # Save data for subprocess
            import pickle
            import tempfile
            temp_data_file = Path(tempfile.gettempdir()) / 'v3_weather_data.pkl'
            with open(temp_data_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Run the weather predictions
            result = subprocess.run(
                [sys.executable, '-c', script, str(temp_data_file)],
                capture_output=True,
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                raise Exception(f"V3 weather predictions failed: {result.stderr}")
            
            # Load results
            output_file = self.config['predictions_v3_weather']['output_file']
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            self.execution_times['v3_weather_predictions'] = time.time() - start_time
            self.logger.info(f"V3 weather predictions completed in {self.execution_times['v3_weather_predictions']:.2f} seconds")
            
            # Clean up
            temp_data_file.unlink()
            
            return results
            
        except Exception as e:
            self.logger.error(f"V3 weather predictions failed: {e}")
            if self.config['pipeline'].get('continue_on_error', True):
                return {'error': str(e)}
            raise
    
    def run_v4_predictions(self) -> Dict:
        """Run v4 production predictions"""
        if not self.config['predictions_v4'].get('enabled', True):
            self.logger.info("V4 predictions are disabled in configuration")
            return {}
            
        self.logger.info("Running V4 production predictions...")
        start_time = time.time()
        
        try:
            # Create script to run v4 with shared data
            script = f"""
import sys
import json
import pickle
from pathlib import Path

# Load shared data
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

# Monkey patch the data loader
class MockDataLoader:
    def get_core_datasets(self):
        return data

import f1db_data_loader
f1db_data_loader.F1DBDataLoader = MockDataLoader

# Run v4 production
from f1_predictions_v4_production import main

# Override sys.argv for argparse
import sys
sys.argv = ['f1_predictions_v4_production.py', '--bankroll', '{self.config['predictions_v4'].get('bankroll', 1000)}']

# Run main
main()
"""
            
            # Save data for subprocess
            import pickle
            import tempfile
            temp_data_file = Path(tempfile.gettempdir()) / 'v4_data.pkl'
            with open(temp_data_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Run the predictions
            result = subprocess.run(
                [sys.executable, '-c', script, str(temp_data_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"V4 predictions failed: {result.stderr}")
            
            # Load results
            output_file = self.config['predictions_v4']['output_file']
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            self.execution_times['v4_predictions'] = time.time() - start_time
            self.logger.info(f"V4 predictions completed in {self.execution_times['v4_predictions']:.2f} seconds")
            
            # Clean up
            temp_data_file.unlink()
            
            return results
            
        except Exception as e:
            self.logger.error(f"V4 predictions failed: {e}")
            if self.config['pipeline'].get('continue_on_error', True):
                return {'error': str(e)}
            raise
    
    def run_parallel_tasks(self) -> Dict:
        """Run tasks that can be executed in parallel"""
        self.logger.info("Starting parallel execution of prediction tasks...")
        
        tasks = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            # Submit v3 predictions
            if self.config['predictions_v3'].get('enabled', True):
                tasks.append(('v3_predictions', executor.submit(self.run_v3_predictions)))
            
            # Submit v3 weather predictions
            if self.config['predictions_v3_weather'].get('enabled', True):
                tasks.append(('v3_weather_predictions', executor.submit(self.run_v3_weather_predictions)))
            
            # Submit v4 predictions (can run in parallel since independent)
            if self.config['predictions_v4'].get('enabled', True):
                tasks.append(('v4_predictions', executor.submit(self.run_v4_predictions)))
            
            # Collect results
            results = {}
            for task_name, future in tasks:
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    self.logger.error(f"{task_name} failed: {e}")
                    if not self.config['pipeline'].get('continue_on_error', True):
                        raise
                    results[task_name] = {'error': str(e)}
            
            return results
    
    def generate_summary(self):
        """Generate summary report of all results"""
        if not self.config['summary'].get('enabled', True):
            return
            
        self.logger.info("Generating summary report...")
        
        summary = {
            'pipeline_name': self.config['pipeline']['name'],
            'pipeline_version': self.config['pipeline']['version'],
            'execution_date': datetime.now().isoformat(),
            'execution_times': self.execution_times,
            'total_execution_time': sum(self.execution_times.values()),
            'results_summary': {}
        }
        
        # Summarize each component's results
        for component, results in self.results.items():
            if isinstance(results, dict) and 'error' in results:
                summary['results_summary'][component] = {
                    'status': 'failed',
                    'error': results['error']
                }
            else:
                summary['results_summary'][component] = {
                    'status': 'success',
                    'records': len(results) if isinstance(results, (list, dict)) else 0
                }
        
        # Save summary
        output_file = Path(self.config['summary']['output_file'])
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Summary report saved to {output_file}")
        
        # Generate markdown report
        self.generate_markdown_report(summary)
    
    def generate_markdown_report(self, summary):
        """Generate a readable markdown report"""
        self.logger.info("Generating markdown report...")
        
        markdown_content = f"""# F1 Enhanced Pipeline Results

**Pipeline:** {summary['pipeline_name']} v{summary['pipeline_version']}  
**Execution Date:** {summary['execution_date']}  
**Total Execution Time:** {summary['total_execution_time']:.2f} seconds

## Component Status

"""
        
        for component, status in summary['results_summary'].items():
            status_icon = "✅" if status['status'] == 'success' else "❌"
            markdown_content += f"- {status_icon} **{component.replace('_', ' ').title()}**: {status['status']}\n"
            if status['status'] == 'failed':
                markdown_content += f"  - Error: {status.get('error', 'Unknown')}\n"
        
        markdown_content += f"\n## Execution Times\n\n"
        for component, time in summary['execution_times'].items():
            markdown_content += f"- **{component.replace('_', ' ').title()}**: {time:.2f}s\n"
        
        # Add prediction results if available
        markdown_content += self._format_prediction_results()
        
        # Save markdown report
        md_file = Path("pipeline_outputs/enhanced_pipeline_report.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Also print to console
        print("\n" + "="*80)
        print("F1 ENHANCED PIPELINE RESULTS")
        print("="*80)
        print(markdown_content)
        
        self.logger.info(f"Markdown report saved to {md_file}")
    
    def _format_prediction_results(self):
        """Format prediction results for markdown"""
        content = "\n## Prediction Results\n\n"
        
        # V4 Portfolio Results
        v4_file = Path("pipeline_outputs/portfolio_v4_production.json")
        if v4_file.exists():
            try:
                with open(v4_file, 'r') as f:
                    v4_data = json.load(f)
                
                content += "### V4 Production Portfolio\n\n"
                content += f"- **Total Bets**: {len(v4_data.get('bets', []))}\n"
                content += f"- **Total Stake**: ${v4_data.get('total_stake', 0)}\n"
                content += f"- **Expected Value**: ${v4_data.get('expected_value', 0):.2f}\n"
                content += f"- **Expected ROI**: {v4_data.get('expected_roi', 0):.1f}%\n\n"
                
                if v4_data.get('bets'):
                    content += "#### Top Bets:\n\n"
                    for i, bet in enumerate(v4_data['bets'][:3], 1):
                        content += f"{i}. **{bet['type']}** - ${bet['stake']} stake, {bet['payout']}x payout\n"
                        for selection in bet['selections']:
                            content += f"   - {selection['driver']}: {selection['prop']} {selection['direction']} {selection['line']}\n"
                
            except Exception as e:
                content += f"- V4 Portfolio: Error reading results ({e})\n"
        
        # V3 Results
        v3_file = Path("pipeline_outputs/enhanced_predictions_v3.json")
        if v3_file.exists():
            try:
                with open(v3_file, 'r') as f:
                    v3_data = json.load(f)
                
                content += "\n### V3 Enhanced Predictions\n\n"
                if 'predictions' in v3_data:
                    content += f"- **Total Predictions**: {len(v3_data['predictions'])}\n"
                    
                    # Show sample predictions
                    sample_preds = list(v3_data['predictions'].items())[:5]
                    content += "\n#### Sample Predictions:\n\n"
                    for driver, preds in sample_preds:
                        content += f"**{driver}:**\n"
                        for prop, values in preds.items():
                            if isinstance(values, dict) and 'over_probability' in values:
                                content += f"  - {prop}: {values['over_probability']:.1%} over\n"
                
            except Exception as e:
                content += f"- V3 Predictions: Error reading results ({e})\n"
        
        # Performance Analysis
        perf_file = Path("pipeline_outputs/performance_analysis_report.json")
        if perf_file.exists():
            try:
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                
                content += "\n### Performance Analysis\n\n"
                if 'next_race' in perf_data:
                    next_race = perf_data['next_race']
                    content += f"- **Next Race**: {next_race.get('name', 'Unknown')}\n"
                    content += f"- **Date**: {next_race.get('date', 'TBD')}\n"
                
                if 'active_drivers' in perf_data:
                    active_count = len(perf_data['active_drivers']) if isinstance(perf_data['active_drivers'], list) else 0
                    content += f"- **Active Drivers**: {active_count}\n"
                
            except Exception as e:
                content += f"- Performance Analysis: Error reading results ({e})\n"
        
        content += "\n---\n*Generated by Enhanced F1 Pipeline*\n"
        return content
    
    def run(self):
        """Run the complete enhanced pipeline"""
        self.logger.info(f"Starting {self.config['pipeline']['name']} v{self.config['pipeline']['version']}")
        pipeline_start = time.time()
        
        try:
            # Step 1: Load shared data
            self.load_shared_data()
            
            # Step 2: Run performance analysis (can be parallel but we'll run it first)
            if self.config['performance_analysis'].get('enabled', True):
                self.results['performance_analysis'] = self.run_performance_analysis()
            
            # Step 3: Run prediction tasks in parallel
            if self.config['pipeline'].get('parallel_execution', True):
                parallel_results = self.run_parallel_tasks()
                self.results.update(parallel_results)
            else:
                # Run sequentially
                self.results['v3_predictions'] = self.run_v3_predictions()
                self.results['v3_weather_predictions'] = self.run_v3_weather_predictions()
                self.results['v4_predictions'] = self.run_v4_predictions()
            
            # Step 4: Generate summary
            self.generate_summary()
            
            total_time = time.time() - pipeline_start
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            # Send notifications if configured
            self.send_notifications(success=True)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.send_notifications(success=False, error=str(e))
            raise
    
    def send_notifications(self, success: bool, error: str = None):
        """Send notifications if configured"""
        if not self.config['notifications'].get('enabled', False):
            return
            
        # Placeholder for notification logic
        # Could integrate with Slack, email, etc.
        pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced F1 Pipeline Orchestrator')
    parser.add_argument('--config', default='pipeline_config_enhanced.yaml',
                        help='Path to pipeline configuration file (resolved relative to this script by default)')
    parser.add_argument('--skip-weather', action='store_true',
                        help='Skip weather predictions')
    parser.add_argument('--sequential', action='store_true',
                        help='Run tasks sequentially instead of in parallel')
    parser.add_argument('--race-id', type=int,
                        help='Specific race ID to process (for future implementation)')
    
    args = parser.parse_args()
    
    # Note race_id for future implementation
    if args.race_id:
        print(f"Note: Race ID {args.race_id} specified but not yet implemented in enhanced pipeline")
        print("This will be added in a future version. Running for next race instead.")
    
    resolved_config_path = resolve_config_path(args.config)

    # Override config if needed
    if args.skip_weather or args.sequential:
        import yaml
        with open(resolved_config_path, 'r') as f:
            config = yaml.safe_load(f)

        if args.skip_weather:
            config['predictions_v3_weather']['enabled'] = False

        if args.sequential:
            config['pipeline']['parallel_execution'] = False

        # Save temporary config
        import tempfile
        temp_config = Path(tempfile.gettempdir()) / 'temp_pipeline_config.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)

        resolved_config_path = temp_config

    # Run pipeline
    print(f"Using configuration file: {resolved_config_path}")
    pipeline = EnhancedF1Pipeline(str(resolved_config_path))
    pipeline.run()


if __name__ == "__main__":
    main()