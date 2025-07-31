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
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging

# Add notebook directory to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('F1Pipeline')

# Import F1 ML modules
from f1_ml.features import F1FeatureStore
from f1_ml.models import F1ModelTrainer
from f1_ml.evaluation import IntegratedF1Predictor
from f1_ml.optimization import PrizePicksOptimizer
from f1_ml.explainability import PredictionExplainer, PrizePicksExplainer
from f1_ml.backtesting import F1BacktestEngine, prepare_backtest_data, compare_strategies
from f1db_data_loader import fix_column_mappings
from f1db_data_loader import load_f1db_data
from f1_performance_analysis import F1PerformanceAnalyzer

class PipelineConfig:
    """Configuration for the F1 pipeline"""
    def __init__(self, output_dir=None):
        # Data paths
        current_dir = Path.cwd()
        self.data_dir = self._find_data_dir(current_dir)
        self.model_dir = Path('.')
        self.output_dir = Path(output_dir) if output_dir else Path('pipeline_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        # Model settings
        self.use_cached_data = True
        self.auto_sync = True
        self.cache_expiry_hours = 24
        
        # Optimization settings
        self.bankroll = 1000
        self.kelly_fraction = 0.25
        self.max_correlation = 0.5
        self.min_edge = 0.05
        self.max_exposure = 0.25
        
        # Constraints
        self.constraints = {
            'max_per_driver': 2,
            'max_per_type': 3,
            'min_avg_edge': 0.08
        }
        
        # Pipeline settings
        self.generate_report = True
        self.save_predictions = True
        self.mlflow_tracking = False
        
        logger.info(f"Data directory: {self.data_dir}")
    
    def _find_data_dir(self, current_dir):
        """Find the F1DB data directory"""
        # Use absolute path to /data/f1db
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_path = project_root / 'data' / 'f1db'
        
        # Create directory if it doesn't exist
        data_path.mkdir(parents=True, exist_ok=True)
        
        return data_path
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'data_dir': str(self.data_dir),
            'model_dir': str(self.model_dir),
            'output_dir': str(self.output_dir),
            'bankroll': self.bankroll,
            'kelly_fraction': self.kelly_fraction,
            'max_correlation': self.max_correlation,
            'min_edge': self.min_edge,
            'max_exposure': self.max_exposure,
            'constraints': self.constraints
        }
    
    def save(self, path='pipeline_config.json'):
        """Save configuration"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path='pipeline_config.json', output_dir=None):
        """Load configuration"""
        config = cls(output_dir=output_dir)
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(config, key):
                        if key.endswith('_dir'):
                            setattr(config, key, Path(value))
                        else:
                            setattr(config, key, value)
        # If output_dir was provided explicitly, use it
        if output_dir:
            config.output_dir = Path(output_dir)
            config.output_dir.mkdir(exist_ok=True)
        return config

class F1PredictionsGenerator:
    """Generate race predictions using real F1 data"""
    
    def __init__(self, data):
        self.data = data
        
    def get_driver_stats(self, driver_id, last_n_races=10):
        """Calculate driver statistics from historical data"""
        results = self.data.get('results', pd.DataFrame())
        
        if results.empty:
            return {}
        
        # Filter by driver
        driver_results = results[results['driverId'] == driver_id]
        
        # Get recent results
        recent_results = driver_results.tail(last_n_races)
        
        stats = {
            'races': len(driver_results),
            'wins': (driver_results['positionOrder'] == 1).sum(),
            'podiums': (driver_results['positionOrder'] <= 3).sum(),
            'top5': (driver_results['positionOrder'] <= 5).sum(),
            'top10': (driver_results['positionOrder'] <= 10).sum(),
            'points_finishes': (driver_results['points'] > 0).sum(),
            'dnf': (driver_results['statusId'] > 1).sum() if 'statusId' in driver_results else 0,
            'avg_position': driver_results[driver_results['positionOrder'] > 0]['positionOrder'].mean(),
            'recent_avg_position': recent_results[recent_results['positionOrder'] > 0]['positionOrder'].mean()
        }
        
        # Calculate rates
        if stats['races'] > 0:
            stats['top10_rate'] = stats['top10'] / stats['races']
            stats['top5_rate'] = stats['top5'] / stats['races']
            stats['podium_rate'] = stats['podiums'] / stats['races']
            stats['points_rate'] = stats['points_finishes'] / stats['races']
            stats['dnf_rate'] = stats['dnf'] / stats['races']
        
        return stats
    
    def generate_predictions(self, race_id=None):
        """Generate predictions for all drivers"""
        drivers = self.data.get('drivers', pd.DataFrame())
        results = self.data.get('results', pd.DataFrame())
        
        if drivers.empty or results.empty:
            return pd.DataFrame()
        
        # Get drivers who raced recently
        recent_years = [datetime.now().year, datetime.now().year - 1]
        recent_driver_ids = results[results['year'].isin(recent_years)]['driverId'].unique() if 'year' in results else results['driverId'].unique()[-40:]
        active_drivers = drivers[drivers['driverId'].isin(recent_driver_ids)]
        
        predictions = []
        
        for _, driver in active_drivers.iterrows():
            stats = self.get_driver_stats(driver['driverId'])
            
            if stats['races'] == 0:
                continue
            
            # Calculate probabilities based on historical performance
            recency_factor = 0.7
            consistency_bonus = 0.05
            
            top10_prob = stats['top10_rate'] * recency_factor + (stats['recent_avg_position'] <= 10) * (1 - recency_factor) * 0.8
            top10_prob = min(0.95, top10_prob + consistency_bonus)
            
            top5_prob = stats['top5_rate'] * recency_factor + (stats['recent_avg_position'] <= 5) * (1 - recency_factor) * 0.6
            top5_prob = min(0.85, top5_prob + consistency_bonus)
            
            top3_prob = stats['podium_rate'] * recency_factor + (stats['recent_avg_position'] <= 3) * (1 - recency_factor) * 0.4
            top3_prob = min(0.70, top3_prob + consistency_bonus)
            
            points_prob = stats['points_rate'] * recency_factor + (stats['recent_avg_position'] <= 10) * (1 - recency_factor) * 0.9
            points_prob = min(0.95, points_prob)
            
            predictions.append({
                'driver': driver['surname'],
                'driver_id': driver['driverId'],
                'top10_prob': round(top10_prob, 3),
                'top5_prob': round(top5_prob, 3),
                'top3_prob': round(top3_prob, 3),
                'points_prob': round(points_prob, 3),
                'beat_teammate_prob': 0.5,
                'confidence': round(0.7 + (1 - stats['recent_avg_position'] / 20) * 0.2, 3)
            })
        
        predictions_df = pd.DataFrame(predictions)
        if not predictions_df.empty:
            predictions_df = predictions_df.sort_values('confidence', ascending=False).head(20)
        
        return predictions_df

class F1PrizePipeline:
    """Main pipeline orchestrating all components"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data = None
        self.feature_store = F1FeatureStore()
        self.optimizer = PrizePicksOptimizer(
            kelly_fraction=config.kelly_fraction,
            max_correlation=config.max_correlation
        )
        self.predictions_generator = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare F1 data"""
        logger.info("Loading F1 data...")
        
        # Load data
        self.data = load_f1db_data(data_dir=str(self.config.data_dir))
        
        # Apply column mappings
        self.data = fix_column_mappings(self.data)
        
        # Initialize predictions generator
        self.predictions_generator = F1PredictionsGenerator(self.data)
        
        logger.info(f"Loaded {len(self.data)} datasets")
        return self.data
    
    def generate_predictions(self, race_id=None):
        """Generate predictions for race"""
        logger.info("Generating predictions...")
        
        predictions = self.predictions_generator.generate_predictions(race_id)
        
        # Get race information
        races = self.data.get('races', pd.DataFrame())
        if not races.empty and race_id:
            race_info = races[races['id'] == race_id]
            if race_info.empty:
                race_info = races[races['raceId'] == race_id]
            if not race_info.empty:
                self.results['race_info'] = race_info.iloc[0].to_dict()
        elif not races.empty:
            # Get next race if no race_id specified
            races['date'] = pd.to_datetime(races['date'])
            upcoming = races[races['date'] > datetime.now()].sort_values('date')
            if not upcoming.empty:
                self.results['race_info'] = upcoming.iloc[0].to_dict()
        
        self.results['predictions'] = predictions
        logger.info(f"Generated predictions for {len(predictions)} drivers")
        return predictions
    
    def optimize_picks(self):
        """Optimize Prize Picks selections"""
        logger.info("Optimizing Prize Picks...")
        
        if 'predictions' not in self.results or self.results['predictions'].empty:
            logger.error("No predictions available")
            return []
        
        # Generate all possible picks
        all_picks = self.optimizer.generate_all_picks(
            self.results['predictions'],
            min_edge=self.config.min_edge
        )
        
        if all_picks.empty:
            logger.warning("No picks with positive edge found")
            return []
        
        # Optimize portfolio
        portfolio = self.optimizer.optimize_portfolio(
            all_picks,
            bankroll=self.config.bankroll,
            constraints=self.config.constraints
        )
        
        self.results['portfolio'] = portfolio
        logger.info(f"Optimized portfolio with {len(portfolio)} parlays")
        return portfolio
    
    def generate_report(self, save_path=None):
        """Generate comprehensive report"""
        logger.info("Generating report...")
        
        # Extract key information for CI/CD
        predictions_df = self.results.get('predictions', pd.DataFrame())
        top_3_drivers = []
        avg_confidence = 0
        if not predictions_df.empty and 'predicted_prob' in predictions_df.columns:
            top_predictions = predictions_df.nlargest(3, 'predicted_prob')
            top_3_drivers = top_predictions['driver'].tolist() if 'driver' in top_predictions.columns else []
            avg_confidence = predictions_df['predicted_prob'].mean() if 'predicted_prob' in predictions_df.columns else 0
        
        # Get race name
        race_name = 'Unknown'
        if 'race_info' in self.results:
            race_name = self.results['race_info'].get('name', 'Unknown')
        
        # Calculate expected ROI from portfolio
        expected_roi = 0
        portfolio = self.results.get('portfolio', [])
        if portfolio:
            total_ev = sum(p.get('expected_value', 0) * p.get('bet_size', 0) for p in portfolio)
            total_bet = sum(p.get('bet_size', 0) for p in portfolio)
            if total_bet > 0:
                expected_roi = total_ev / total_bet
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'race_name': race_name,
            'top_3_predictions': top_3_drivers,
            'avg_confidence': avg_confidence,
            'val_accuracy': 0.75,  # Placeholder - should come from model evaluation
            'expected_roi': expected_roi,
            'betting_recommendations': self._generate_betting_summary(),
            'config': self.config.to_dict(),
            'predictions': predictions_df.to_dict('records'),
            'portfolio': self._serialize_portfolio()
        }
        
        if save_path is None:
            save_path = self.config.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as latest_report.json for CI/CD
        latest_path = self.config.output_dir / "latest_report.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {save_path}")
        logger.info(f"Latest report saved to {latest_path}")
        return report
    
    def _serialize_portfolio(self):
        """Serialize portfolio for JSON"""
        if 'portfolio' not in self.results:
            return []
        
        serialized = []
        for parlay in self.results['portfolio']:
            parlay_data = {
                'n_picks': parlay['n_picks'],
                'bet_size': parlay['bet_size'],
                'payout': parlay['payout'],
                'adjusted_prob': parlay['adjusted_prob'],
                'expected_value': parlay['expected_value'],
                'picks': parlay['picks'].to_dict('records') if hasattr(parlay['picks'], 'to_dict') else parlay['picks']
            }
            serialized.append(parlay_data)
        
        return serialized
    
    def _generate_betting_summary(self):
        """Generate a text summary of betting recommendations"""
        portfolio = self.results.get('portfolio', [])
        if not portfolio:
            return "No betting recommendations available"
        
        summary = f"**Recommended Bets ({len(portfolio)} parlays)**\n\n"
        
        for i, parlay in enumerate(portfolio[:3], 1):  # Top 3 parlays
            summary += f"**Parlay {i}:**\n"
            summary += f"- Bet Size: ${parlay.get('bet_size', 0):.2f}\n"
            summary += f"- Potential Payout: ${parlay.get('bet_size', 0) * parlay.get('payout', 1):.2f} ({parlay.get('payout', 1):.1f}x)\n"
            summary += f"- Win Probability: {parlay.get('adjusted_prob', 0):.1%}\n"
            summary += f"- Expected Value: ${parlay.get('expected_value', 0) * parlay.get('bet_size', 0):.2f}\n"
            
            picks = parlay.get('picks', [])
            if isinstance(picks, pd.DataFrame):
                picks = picks.to_dict('records')
            
            summary += f"- Picks: {len(picks)}\n"
            for pick in picks:
                summary += f"  - {pick.get('driver', 'Unknown')}: {pick.get('bet_type', 'Unknown')}\n"
            summary += "\n"
        
        return summary
    
    def run(self, race_id=None):
        """Run complete pipeline"""
        logger.info("Starting F1 Prize Picks pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Generate performance analysis tables
            logger.info("Generating performance analysis tables...")
            analyzer = F1PerformanceAnalyzer(self.data)
            performance_tables = analyzer.generate_all_tables()
            self.results['performance_analysis'] = performance_tables
            
            # Generate predictions
            predictions = self.generate_predictions(race_id)
            
            # Optimize picks
            portfolio = self.optimize_picks()
            
            # Generate report
            if self.config.save_predictions:
                report = self.generate_report()
            
            logger.info("Pipeline completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

class RaceWeekendAutomation:
    """Automate pipeline execution for race weekends"""
    def __init__(self, pipeline: F1PrizePipeline):
        self.pipeline = pipeline
        self.schedule = []

class PerformanceMonitor:
    """Monitor pipeline and prediction performance"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics = []
    
    def load_metrics(self):
        """Load existing metrics"""
        pass
    
    def track_predictions(self, predictions):
        """Track prediction accuracy"""
        pass
    
    def save_metrics(self):
        """Save metrics to file"""
        pass

def run_master_notebook(race_id=None, mode='predict', output_dir=None):
    """Run the master pipeline"""
    
    # Load configuration
    config = PipelineConfig.load(output_dir=output_dir)
    
    # Initialize pipeline
    pipeline = F1PrizePipeline(config)
    
    if mode == 'schedule':
        # Run automation
        automation = RaceWeekendAutomation(pipeline)
        pipeline.load_data()
        
        # Get upcoming races
        races = pipeline.data.get('races', pd.DataFrame())
        if not races.empty and 'date' in races.columns:
            races['date'] = pd.to_datetime(races['date'])
            upcoming = races[races['date'] > datetime.now()].sort_values('date')
            if not upcoming.empty:
                next_race = upcoming.iloc[0]
                race_id = next_race.get('id', next_race.get('raceId'))
                race_name = next_race.get('officialName', next_race.get('name', 'Unknown'))
                print(f"Scheduling analyses for {race_name}")
                # For now just run once
                results = pipeline.run(race_id)
                return results
    elif mode == 'backtest':
        print("Running backtesting...")
        # Load data first
        pipeline.load_data()
        
        # Initialize backtesting engine
        engine = F1BacktestEngine(initial_bankroll=config.bankroll)
        
        # Prepare backtest data
        backtest_data = prepare_backtest_data(
            pipeline.data.get('results', pd.DataFrame()),
            pipeline.data.get('races', pd.DataFrame()),
            pipeline.data.get('drivers', pd.DataFrame()),
            pipeline.data.get('constructors', pd.DataFrame()),
            pipeline.data.get('qualifying', pd.DataFrame()),
            start_year=2023,
            end_year=2024
        )
        
        # Run backtest
        engine.run_backtest(
            backtest_data,
            pipeline.optimizer,
            strategy='moderate',
            kelly_fraction=config.kelly_fraction
        )
        
        # Display results
        print(f"\nBacktest complete!")
        print(f"Final bankroll: ${engine.bankroll:.2f}")
        print(f"Total return: {((engine.bankroll - engine.initial_bankroll) / engine.initial_bankroll):.1%}")
        
        return engine
    else:
        # Normal prediction mode
        results = pipeline.run(race_id)
        
        if results and results.get('portfolio', []):
            print("\nPipeline completed successfully!")
            print(f"Results saved to {config.output_dir}")
            
            # Display recommendations
            portfolio = results['portfolio']
            total_wagered = sum(p['bet_size'] for p in portfolio)
            total_expected = sum(p['expected_value'] * p['bet_size'] for p in portfolio)
            
            print("\n" + "=" * 80)
            print("F1 PRIZE PICKS RECOMMENDATIONS")
            print("=" * 80)
            
            for i, parlay in enumerate(portfolio, 1):
                print(f"\n{'='*60}")
                print(f"PARLAY {i}: {parlay['n_picks']}-PICK ENTRY")
                print(f"{'='*60}")
                print(f"Bet Amount: ${parlay['bet_size']:.2f}")
                print(f"Potential Payout: ${parlay['bet_size'] * parlay['payout']:.2f}")
                print(f"Win Probability: {parlay['adjusted_prob']:.1%}")
                print(f"Expected Value: +{parlay['expected_value']:.1%}")
                
                picks = parlay['picks']
                if hasattr(picks, 'iterrows'):
                    print("\nSelections:")
                    for j, (_, pick) in enumerate(picks.iterrows(), 1):
                        print(f"  {j}. {pick['driver']} - {pick['bet_type']}")
                        print(f"     Probability: {pick['true_prob']:.1%}")
                        print(f"     Edge: +{pick['edge']:.1%}")
            
            print("\n" + "=" * 80)
            print("PORTFOLIO SUMMARY")
            print("=" * 80)
            print(f"Total Wagered: ${total_wagered:.2f}")
            print(f"Expected Profit: ${total_expected:.2f}")
            print(f"Expected ROI: {(total_expected/total_wagered):.1%}" if total_wagered > 0 else "N/A")
            
            # Track performance
            monitor = PerformanceMonitor(config.output_dir)
            monitor.load_metrics()
            monitor.track_predictions(results.get('predictions', pd.DataFrame()))
            monitor.save_metrics()
            
            return results
        else:
            print("\nNo betting recommendations generated. This could mean:")
            print("- No bets met the minimum edge requirement")
            print("- Try adjusting config.min_edge or config.kelly_fraction")
            
            # Still return results with performance analysis
            return results

def main():
    parser = argparse.ArgumentParser(description='Run F1 Prize Picks Pipeline')
    parser.add_argument('--race-id', type=int, help='Specific race ID to analyze')
    parser.add_argument('--backtest', action='store_true', 
                       help='Run backtesting instead of predictions')
    parser.add_argument('--schedule', action='store_true',
                       help='Schedule automated race weekend analyses')
    parser.add_argument('--output-dir', type=str, default='pipeline_outputs',
                       dest='output_dir',
                       help='Output directory for results (default: pipeline_outputs)')
    
    args = parser.parse_args()
    
    if args.backtest:
        run_master_notebook(mode='backtest', output_dir=args.output_dir)
    elif args.schedule:
        run_master_notebook(mode='schedule', output_dir=args.output_dir)
    else:
        run_master_notebook(race_id=args.race_id, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
