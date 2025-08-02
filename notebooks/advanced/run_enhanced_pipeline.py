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


class EnhancedF1Pipeline:
    """Orchestrates the enhanced F1 prediction pipeline"""
    
    def __init__(self, config_path: str = "pipeline_config_enhanced.yaml"):
        """Initialize the pipeline with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        self.results = {}
        self.execution_times = {}
        self.data = None
        self.data_cache_file = None
        
    def _load_config(self) -> Dict:
        """Load pipeline configuration from YAML file"""
        config_path = Path(self.config_path)
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
        
        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('file'):
            log_file = Path(log_config['file'])
            log_file.parent.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
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
results = {
    'next_race': analyzer.get_next_race(),
    'active_drivers': analyzer.get_active_drivers().to_dict() if hasattr(analyzer.get_active_drivers(), 'to_dict') else [],
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
                        help='Path to pipeline configuration file')
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
    
    # Override config if needed
    if args.skip_weather or args.sequential:
        import yaml
        with open(args.config, 'r') as f:
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
        
        args.config = str(temp_config)
    
    # Run pipeline
    pipeline = EnhancedF1Pipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()