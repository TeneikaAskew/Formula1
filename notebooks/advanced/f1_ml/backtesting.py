"""
F1 Backtesting Module

This module provides comprehensive backtesting functionality for F1 betting strategies,
including the F1BacktestEngine class and associated helper functions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def prepare_backtest_data(results, races, drivers, constructors, qualifying=None, 
                         start_year=2023, end_year=2024):
    """
    Prepare data for backtesting period
    
    Parameters:
    -----------
    results : pd.DataFrame
        Race results data
    races : pd.DataFrame
        Race information
    drivers : pd.DataFrame
        Driver information
    constructors : pd.DataFrame
        Constructor information
    qualifying : pd.DataFrame
        Qualifying results (optional)
    start_year : int
        Start year for backtesting
    end_year : int
        End year for backtesting
        
    Returns:
    --------
    pd.DataFrame
        Prepared backtest data
    """
    # Merge core data
    df = results.merge(races[['raceId', 'year', 'round', 'circuitId', 'date', 'name']], on='raceId')
    df = df.merge(drivers[['driverId', 'surname']], on='driverId')
    df = df.merge(constructors[['constructorId', 'name']], on='constructorId', suffixes=('', '_constructor'))
    
    # Rename constructor name column
    if 'name_constructor' in df.columns:
        df = df.rename(columns={'name_constructor': 'constructor_name'})
    
    # Filter to backtest period
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Add qualifying data if available
    if qualifying is not None and not qualifying.empty:
        # Handle different column names
        qual_cols = qualifying.columns
        position_col = 'position' if 'position' in qual_cols else 'positionNumber'
        
        df = df.merge(
            qualifying[['raceId', 'driverId', position_col]].rename(columns={position_col: 'qualifying_position'}),
            on=['raceId', 'driverId'],
            how='left'
        )
    
    # Sort by date
    df = df.sort_values(['date', 'raceId', 'positionOrder'])
    
    return df


class F1BacktestEngine:
    """
    Backtesting engine for F1 Prize Picks strategies
    """
    def __init__(self, initial_bankroll=1000):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.results = []
        self.bets = []
        self.bankroll_history = [initial_bankroll]
        
    def calculate_bet_outcomes(self, race_data):
        """
        Calculate actual outcomes for different bet types
        
        Parameters:
        -----------
        race_data : pd.DataFrame
            Race results data
            
        Returns:
        --------
        dict
            Outcomes for each driver
        """
        outcomes = {}
        
        for _, driver_result in race_data.iterrows():
            driver = driver_result['surname']
            position = driver_result['positionOrder']
            
            # Calculate outcomes
            outcomes[driver] = {
                'top_10': position <= 10,
                'top_5': position <= 5,
                'top_3': position <= 3,
                'points': driver_result['points'] > 0,
                'dnf': driver_result.get('statusId', 1) > 1,
                'position': position
            }
            
            # Beat teammate
            teammate_data = race_data[
                (race_data['constructorId'] == driver_result['constructorId']) &
                (race_data['driverId'] != driver_result['driverId'])
            ]
            
            if not teammate_data.empty:
                teammate_position = teammate_data.iloc[0]['positionOrder']
                outcomes[driver]['beat_teammate'] = position < teammate_position
            else:
                outcomes[driver]['beat_teammate'] = True  # No teammate
        
        return outcomes
    
    def simulate_predictions(self, race_data, backtest_data, strategy='conservative'):
        """
        Simulate predictions based on historical performance
        
        Parameters:
        -----------
        race_data : pd.DataFrame
            Current race data
        backtest_data : pd.DataFrame
            All historical data
        strategy : str
            Strategy type: 'conservative', 'moderate', or 'aggressive'
            
        Returns:
        --------
        pd.DataFrame
            Simulated predictions
        """
        predictions = []
        
        # Get recent performance for each driver
        for driver_id in race_data['driverId'].unique():
            driver_name = race_data[race_data['driverId'] == driver_id]['surname'].iloc[0]
            
            # Look at last 5 races
            historical = backtest_data[
                (backtest_data['driverId'] == driver_id) &
                (backtest_data['date'] < race_data['date'].iloc[0])
            ].tail(5)
            
            if len(historical) >= 3:
                # Calculate probabilities based on historical performance
                top10_prob = (historical['positionOrder'] <= 10).mean()
                top5_prob = (historical['positionOrder'] <= 5).mean()
                top3_prob = (historical['positionOrder'] <= 3).mean()
                points_prob = (historical['points'] > 0).mean()
                
                # Adjust for strategy
                if strategy == 'conservative':
                    # Reduce probabilities by 10%
                    adjustment = 0.9
                elif strategy == 'aggressive':
                    # Increase probabilities by 5%
                    adjustment = 1.05
                else:
                    adjustment = 1.0
                
                predictions.append({
                    'driver': driver_name,
                    'top10_prob': min(0.95, top10_prob * adjustment),
                    'top5_prob': min(0.85, top5_prob * adjustment),
                    'top3_prob': min(0.70, top3_prob * adjustment),
                    'points_prob': min(0.95, points_prob * adjustment),
                    'beat_teammate_prob': 0.5,  # Default
                    'confidence': 0.7 + 0.05 * len(historical)  # Higher confidence with more data
                })
        
        return pd.DataFrame(predictions)
    
    def evaluate_parlay(self, picks, outcomes):
        """
        Evaluate if a parlay won
        
        Parameters:
        -----------
        picks : list
            List of picks in the parlay
        outcomes : dict
            Actual race outcomes
            
        Returns:
        --------
        bool
            Whether the parlay won
        """
        all_won = True
        
        for pick in picks:
            driver = pick['driver']
            bet_type = pick['bet_type']
            
            if driver not in outcomes:
                all_won = False
                break
            
            if bet_type in outcomes[driver]:
                if not outcomes[driver][bet_type]:
                    all_won = False
                    break
            else:
                all_won = False
                break
        
        return all_won
    
    def process_race(self, race_id, race_data, backtest_data, optimizer, 
                    strategy='conservative', kelly_fraction=0.25):
        """
        Process a single race for backtesting
        
        Parameters:
        -----------
        race_id : int
            Race ID
        race_data : pd.DataFrame
            Current race data
        backtest_data : pd.DataFrame
            All historical data
        optimizer : PrizePicksOptimizer
            Optimizer instance
        strategy : str
            Strategy type
        kelly_fraction : float
            Kelly criterion fraction
        """
        race_name = race_data['name'].iloc[0]
        race_date = race_data['date'].iloc[0]
        
        # Generate predictions
        predictions = self.simulate_predictions(race_data, backtest_data, strategy)
        
        if predictions.empty:
            return
        
        # Generate picks using optimizer
        optimizer.kelly_fraction = kelly_fraction
        picks = optimizer.generate_all_picks(predictions, min_edge=0.05)
        
        if picks.empty:
            return
        
        # Get actual outcomes
        outcomes = self.calculate_bet_outcomes(race_data)
        
        # Simulate some parlays
        race_bets = []
        
        # Try different parlay sizes
        for n_picks in [2, 3, 4]:
            if len(picks) >= n_picks:
                # Select top picks by edge
                top_picks = picks.nlargest(n_picks, 'edge')
                
                # Calculate combined probability (simplified)
                combined_prob = np.prod(top_picks['true_prob'])
                
                # Calculate bet size (simplified Kelly)
                from ..optimization import PrizePicksBetTypes
                payout = PrizePicksBetTypes.PAYOUTS[n_picks]
                bet_size = min(self.bankroll * 0.05, 50)  # Max 5% or $50
                
                # Check if parlay won
                won = self.evaluate_parlay(top_picks.to_dict('records'), outcomes)
                
                # Calculate profit/loss
                if won:
                    profit = bet_size * (payout - 1)
                else:
                    profit = -bet_size
                
                # Record bet
                bet_record = {
                    'race_id': race_id,
                    'race_name': race_name,
                    'race_date': race_date,
                    'n_picks': n_picks,
                    'bet_size': bet_size,
                    'payout': payout,
                    'combined_prob': combined_prob,
                    'won': won,
                    'profit': profit,
                    'bankroll_before': self.bankroll,
                    'picks': top_picks.to_dict('records')
                }
                
                race_bets.append(bet_record)
                self.bets.append(bet_record)
                
                # Update bankroll
                self.bankroll += profit
        
        # Record race result
        race_result = {
            'race_id': race_id,
            'race_name': race_name,
            'race_date': race_date,
            'n_bets': len(race_bets),
            'total_wagered': sum(b['bet_size'] for b in race_bets),
            'total_profit': sum(b['profit'] for b in race_bets),
            'bankroll_after': self.bankroll
        }
        
        self.results.append(race_result)
        self.bankroll_history.append(self.bankroll)
    
    def run_backtest(self, data, optimizer, strategy='conservative', kelly_fraction=0.25):
        """
        Run full backtest on historical data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Prepared backtest data
        optimizer : PrizePicksOptimizer
            Optimizer instance
        strategy : str
            Strategy type
        kelly_fraction : float
            Kelly criterion fraction
        """
        # Get unique races
        races = data.groupby('raceId').first().reset_index()
        races = races.sort_values('date')
        
        print(f"\nRunning backtest on {len(races)} races...")
        print(f"Strategy: {strategy}, Kelly fraction: {kelly_fraction}")
        
        # Process each race
        for _, race in tqdm(races.iterrows(), total=len(races)):
            race_id = race['raceId']
            race_data = data[data['raceId'] == race_id]
            
            self.process_race(race_id, race_data, data, optimizer, strategy, kelly_fraction)
        
        print(f"\nBacktest complete!")
        print(f"Final bankroll: ${self.bankroll:.2f}")
        print(f"Total return: {((self.bankroll - self.initial_bankroll) / self.initial_bankroll):.1%}")


def calculate_risk_metrics(bets_df, bankroll_history):
    """
    Calculate comprehensive risk metrics
    
    Parameters:
    -----------
    bets_df : pd.DataFrame
        DataFrame of all bets
    bankroll_history : np.array
        Array of bankroll values over time
        
    Returns:
    --------
    dict
        Risk metrics
    """
    # Maximum drawdown
    running_max = np.maximum.accumulate(bankroll_history)
    drawdown = (bankroll_history - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (simplified)
    if len(bets_df) > 1:
        returns = bets_df['profit'] / bets_df['bet_size']
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # Win/loss streaks
    wins = bets_df['won'].values
    
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    
    for won in wins:
        if won:
            if current_streak >= 0:
                current_streak += 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                current_streak = 1
        else:
            if current_streak <= 0:
                current_streak -= 1
                max_loss_streak = min(max_loss_streak, current_streak)
            else:
                current_streak = -1
    
    return {
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'max_win_streak': max_win_streak,
        'max_loss_streak': abs(max_loss_streak),
        'avg_win': bets_df[bets_df['won']]['profit'].mean() if any(bets_df['won']) else 0,
        'avg_loss': bets_df[~bets_df['won']]['profit'].mean() if any(~bets_df['won']) else 0,
        'profit_factor': abs(bets_df[bets_df['profit'] > 0]['profit'].sum() / 
                           bets_df[bets_df['profit'] < 0]['profit'].sum()) if any(bets_df['profit'] < 0) else np.inf
    }


def compare_strategies(backtest_data, optimizer, strategies=None):
    """
    Compare different betting strategies
    
    Parameters:
    -----------
    backtest_data : pd.DataFrame
        Prepared backtest data
    optimizer : PrizePicksOptimizer
        Optimizer instance
    strategies : list
        List of (strategy_name, kelly_fraction) tuples
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    if strategies is None:
        strategies = [
            ('conservative', 0.15),
            ('moderate', 0.25),
            ('aggressive', 0.35)
        ]
    
    strategy_results = []
    
    print("\nComparing different strategies...")
    
    for strategy_name, kelly_fraction in strategies:
        # Run backtest
        engine = F1BacktestEngine(initial_bankroll=1000)
        engine.run_backtest(
            backtest_data,
            optimizer,
            strategy=strategy_name,
            kelly_fraction=kelly_fraction
        )
        
        # Calculate metrics
        bets = pd.DataFrame(engine.bets)
        
        if not bets.empty:
            total_return = (engine.bankroll - engine.initial_bankroll) / engine.initial_bankroll
            win_rate = bets['won'].mean()
            roi = bets['profit'].sum() / bets['bet_size'].sum() if bets['bet_size'].sum() > 0 else 0
            
            risk_metrics = calculate_risk_metrics(bets, np.array(engine.bankroll_history))
            
            strategy_results.append({
                'strategy': strategy_name,
                'kelly_fraction': kelly_fraction,
                'final_bankroll': engine.bankroll,
                'total_return': total_return,
                'win_rate': win_rate,
                'roi': roi,
                'max_drawdown': risk_metrics['max_drawdown'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'n_bets': len(bets)
            })
    
    return pd.DataFrame(strategy_results)


# Export key components
__all__ = [
    'F1BacktestEngine',
    'prepare_backtest_data',
    'calculate_risk_metrics',
    'compare_strategies'
]