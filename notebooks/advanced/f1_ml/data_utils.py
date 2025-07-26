"""
F1 Data Utilities Module

This module provides utilities for handling F1DB data column mappings
and ensuring compatibility with the ML pipeline.
"""

import pandas as pd
from typing import Dict, Optional


def fix_column_mappings(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Fix column mappings to ensure compatibility with ML pipeline
    
    Args:
        data: Dictionary of DataFrames from F1DB
        
    Returns:
        Dictionary with fixed column mappings
    """
    # Fix results columns
    if 'results' in data:
        df = data['results']
        
        # Map grid column
        if 'grid' not in df.columns and 'gridPositionNumber' in df.columns:
            df['grid'] = df['gridPositionNumber']
            
        # Map position column
        if 'position' not in df.columns and 'positionNumber' in df.columns:
            df['position'] = df['positionNumber']
            
        # Create statusId from reasonRetired
        if 'statusId' not in df.columns and 'reasonRetired' in df.columns:
            # Map common retirement reasons to status IDs
            status_mapping = {
                'Finished': 1,
                'Accident': 2,
                'Collision': 3,
                'Engine': 4,
                'Gearbox': 5,
                'Transmission': 6,
                'Clutch': 7,
                'Hydraulics': 8,
                'Electrical': 9,
                '+1 Lap': 11,
                '+2 Laps': 12,
                '+3 Laps': 13,
                '+4 Laps': 14,
                '+5 Laps': 15,
                '+6 Laps': 16,
                '+7 Laps': 17,
                '+8 Laps': 18,
                '+9 Laps': 19,
                'Spun off': 20,
                'Radiator': 21,
                'Suspension': 22,
                'Brakes': 23,
                'Differential': 24,
                'Overheating': 25,
                'Mechanical': 26,
                'Tyre': 27,
                'Driver Seat': 28,
                'Puncture': 29,
                'Disqualified': 30,
                'Wheel': 31,
                'Fuel system': 32,
                'Throttle': 33,
                'Steering': 34,
                'Technical': 35,
                'Electronics': 36,
                'Broken wing': 37,
                'Heat shield fire': 38,
                'Exhaust': 39,
                'Oil leak': 40
            }
            
            # Create statusId column
            df['statusId'] = df['reasonRetired'].fillna('Finished').map(
                lambda x: status_mapping.get(x, 50)  # 50 for unknown
            )
            
        # Create status column if missing
        if 'status' not in df.columns:
            df['status'] = df['reasonRetired'].fillna('Finished')
            
        # Add DNF indicator
        df['dnf'] = (df['statusId'] > 1).astype(int)
        
        # Add win, podium, points columns
        df['win'] = (df['positionOrder'] == 1).astype(int)
        df['podium'] = (df['positionOrder'] <= 3).astype(int)
        df['points_finish'] = (df['points'] > 0).astype(int)
        
    # Fix races columns
    if 'races' in data:
        df = data['races']
        
        # Map id to raceId
        if 'raceId' not in df.columns and 'id' in df.columns:
            df['raceId'] = df['id']
            
        # Map officialName to name
        if 'name' not in df.columns and 'officialName' in df.columns:
            df['name'] = df['officialName']
            
    # Fix drivers columns
    if 'drivers' in data:
        df = data['drivers']
        
        # Map id to driverId
        if 'driverId' not in df.columns and 'id' in df.columns:
            df['driverId'] = df['id']
            
        # Map names
        if 'forename' not in df.columns and 'firstName' in df.columns:
            df['forename'] = df['firstName']
            
        if 'surname' not in df.columns and 'lastName' in df.columns:
            df['surname'] = df['lastName']
            
        if 'driverRef' not in df.columns and 'abbreviation' in df.columns:
            df['driverRef'] = df['abbreviation']
            
        if 'code' not in df.columns and 'abbreviation' in df.columns:
            df['code'] = df['abbreviation']
            
        if 'dob' not in df.columns and 'dateOfBirth' in df.columns:
            df['dob'] = df['dateOfBirth']
            
        if 'nationality' not in df.columns and 'nationalityCountryId' in df.columns:
            df['nationality'] = df['nationalityCountryId']
            
        if 'number' not in df.columns and 'permanentNumber' in df.columns:
            df['number'] = df['permanentNumber']
            
    # Fix constructors columns
    if 'constructors' in data:
        df = data['constructors']
        
        # Map id to constructorId
        if 'constructorId' not in df.columns and 'id' in df.columns:
            df['constructorId'] = df['id']
            
        if 'constructorRef' not in df.columns and 'name' in df.columns:
            df['constructorRef'] = df['name']
            
        if 'nationality' not in df.columns and 'countryId' in df.columns:
            df['nationality'] = df['countryId']
            
    return data


def merge_race_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge race-related data into a single DataFrame for analysis
    
    Args:
        data: Dictionary of DataFrames from F1DB
        
    Returns:
        Merged DataFrame with race results and related data
    """
    # Start with results
    if 'results' not in data:
        raise ValueError("No results data found")
        
    df = data['results'].copy()
    
    # Merge with races
    if 'races' in data:
        races_cols = ['raceId', 'year', 'round', 'circuitId', 'name', 'date']
        races_data = data['races'][races_cols].drop_duplicates(subset=['raceId'])
        df = df.merge(races_data, on='raceId', how='left', suffixes=('', '_race'))
        
        # Handle duplicate columns
        if 'year_race' in df.columns:
            df['year'] = df['year'].fillna(df['year_race'])
            df.drop('year_race', axis=1, inplace=True)
            
        if 'round_race' in df.columns:
            df['round'] = df['round'].fillna(df['round_race'])
            df.drop('round_race', axis=1, inplace=True)
    
    # Merge with drivers
    if 'drivers' in data:
        driver_cols = ['driverId', 'forename', 'surname', 'dob', 'nationality', 'driverRef']
        driver_data = data['drivers'][driver_cols].drop_duplicates(subset=['driverId'])
        df = df.merge(driver_data, on='driverId', how='left')
        
        # Calculate driver age at race time
        if 'date' in df.columns and 'dob' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['dob'] = pd.to_datetime(df['dob'])
            df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25
    
    # Merge with constructors
    if 'constructors' in data:
        const_cols = ['constructorId', 'name', 'nationality']
        const_data = data['constructors'][const_cols].drop_duplicates(subset=['constructorId'])
        const_data.columns = ['constructorId', 'constructor_name', 'constructor_nationality']
        df = df.merge(const_data, on='constructorId', how='left')
        
    # Sort by race and finishing position
    df = df.sort_values(['raceId', 'positionOrder'])
    
    return df


def get_recent_results(data: Dict[str, pd.DataFrame], n_years: int = 5) -> pd.DataFrame:
    """
    Get recent race results for analysis
    
    Args:
        data: Dictionary of DataFrames from F1DB
        n_years: Number of recent years to include
        
    Returns:
        DataFrame with recent results
    """
    df = merge_race_data(data)
    
    # Filter to recent years
    if 'year' in df.columns:
        max_year = df['year'].max()
        min_year = max_year - n_years + 1
        df = df[df['year'] >= min_year]
    
    return df


def calculate_driver_stats(results_df: pd.DataFrame, driver_id: str) -> Dict:
    """
    Calculate driver statistics from results
    
    Args:
        results_df: DataFrame with race results
        driver_id: Driver ID to calculate stats for
        
    Returns:
        Dictionary of driver statistics
    """
    driver_results = results_df[results_df['driverId'] == driver_id]
    
    if len(driver_results) == 0:
        return {}
    
    stats = {
        'driverId': driver_id,
        'races': len(driver_results),
        'wins': (driver_results['positionOrder'] == 1).sum(),
        'podiums': (driver_results['positionOrder'] <= 3).sum(),
        'points': driver_results['points'].sum(),
        'dnf_rate': driver_results['dnf'].mean() if 'dnf' in driver_results else 0,
        'avg_position': driver_results['positionOrder'].mean(),
        'avg_grid': driver_results['grid'].mean() if 'grid' in driver_results else None,
        'best_result': driver_results['positionOrder'].min(),
        'points_finishes': (driver_results['points'] > 0).sum()
    }
    
    return stats


# Export key functions
__all__ = [
    'fix_column_mappings',
    'merge_race_data',
    'get_recent_results',
    'calculate_driver_stats'
]