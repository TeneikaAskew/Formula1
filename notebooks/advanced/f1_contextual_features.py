#!/usr/bin/env python3
"""F1 Contextual Features Module - Phase 3.1 Implementation

This module implements contextual features including:
- Weather conditions (temperature, rain probability)
- Track characteristics (length, type, overtaking difficulty)
- Recent form (last 3 races performance)
- Momentum indicators
- Circuit-specific driver/team performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class F1ContextualFeatures:
    """Extract contextual features for enhanced predictions"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.data = data_dict
        self._initialize_track_characteristics()
        
    def _initialize_track_characteristics(self):
        """Initialize track characteristics database"""
        # Track characteristics based on historical data and expert knowledge
        self.track_characteristics = {
            # High-speed tracks (favor top teams)
            14: {'name': 'Monza', 'type': 'high_speed', 'overtaking': 'easy', 'dnf_risk': 'medium'},
            22: {'name': 'Spa', 'type': 'high_speed', 'overtaking': 'moderate', 'dnf_risk': 'high'},
            34: {'name': 'Silverstone', 'type': 'high_speed', 'overtaking': 'moderate', 'dnf_risk': 'medium'},
            
            # Street circuits (difficult overtaking)
            6: {'name': 'Monaco', 'type': 'street', 'overtaking': 'very_hard', 'dnf_risk': 'high'},
            26: {'name': 'Singapore', 'type': 'street', 'overtaking': 'hard', 'dnf_risk': 'high'},
            71: {'name': 'Baku', 'type': 'street', 'overtaking': 'easy', 'dnf_risk': 'very_high'},
            
            # Technical tracks (favor driver skill)
            27: {'name': 'Suzuka', 'type': 'technical', 'overtaking': 'hard', 'dnf_risk': 'medium'},
            32: {'name': 'Hungaroring', 'type': 'technical', 'overtaking': 'very_hard', 'dnf_risk': 'low'},
            
            # Mixed characteristics
            1: {'name': 'Melbourne', 'type': 'mixed', 'overtaking': 'moderate', 'dnf_risk': 'medium'},
            4: {'name': 'Catalunya', 'type': 'mixed', 'overtaking': 'hard', 'dnf_risk': 'low'},
            11: {'name': 'Hockenheim', 'type': 'mixed', 'overtaking': 'moderate', 'dnf_risk': 'medium'},
            18: {'name': 'Interlagos', 'type': 'mixed', 'overtaking': 'easy', 'dnf_risk': 'high'},
            
            # Modern tracks
            69: {'name': 'Austin', 'type': 'modern', 'overtaking': 'easy', 'dnf_risk': 'medium'},
            70: {'name': 'Sochi', 'type': 'modern', 'overtaking': 'moderate', 'dnf_risk': 'low'},
            73: {'name': 'Miami', 'type': 'street', 'overtaking': 'moderate', 'dnf_risk': 'medium'},
            75: {'name': 'Jeddah', 'type': 'street', 'overtaking': 'hard', 'dnf_risk': 'high'},
            77: {'name': 'Las Vegas', 'type': 'street', 'overtaking': 'easy', 'dnf_risk': 'medium'},
        }
        
        # Overtaking difficulty multipliers
        self.overtaking_multipliers = {
            'very_hard': 0.5,
            'hard': 0.7,
            'moderate': 1.0,
            'easy': 1.3,
            'very_easy': 1.5
        }
        
        # DNF risk multipliers
        self.dnf_risk_multipliers = {
            'very_low': 0.5,
            'low': 0.7,
            'medium': 1.0,
            'high': 1.3,
            'very_high': 1.6
        }
    
    def get_track_features(self, circuit_id: int) -> Dict:
        """Get track-specific features
        
        Args:
            circuit_id: Circuit ID
            
        Returns:
            Dictionary of track features
        """
        default_track = {
            'type': 'mixed',
            'overtaking': 'moderate',
            'dnf_risk': 'medium',
            'overtaking_multiplier': 1.0,
            'dnf_risk_multiplier': 1.0
        }
        
        if circuit_id not in self.track_characteristics:
            return default_track
            
        track = self.track_characteristics[circuit_id]
        
        return {
            'type': track['type'],
            'overtaking': track['overtaking'],
            'dnf_risk': track['dnf_risk'],
            'overtaking_multiplier': self.overtaking_multipliers.get(track['overtaking'], 1.0),
            'dnf_risk_multiplier': self.dnf_risk_multipliers.get(track['dnf_risk'], 1.0),
            'is_street_circuit': track['type'] == 'street',
            'is_high_speed': track['type'] == 'high_speed',
            'is_technical': track['type'] == 'technical'
        }
    
    def get_recent_form(self, driver_id: int, race_id: int, n_races: int = 3) -> Dict:
        """Get driver's recent form metrics
        
        Args:
            driver_id: Driver ID
            race_id: Current race ID (to find previous races)
            n_races: Number of recent races to consider
            
        Returns:
            Dictionary of recent form metrics
        """
        results = self.data.get('results', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty or races.empty:
            return self._get_default_form()
            
        # Get race date to find previous races
        current_race = races[races['id'] == race_id]
        if current_race.empty:
            return self._get_default_form()
            
        race_date = current_race.iloc[0]['date']
        
        # Get previous races
        previous_races = races[races['date'] < race_date].sort_values('date', ascending=False).head(n_races)
        
        if previous_races.empty:
            return self._get_default_form()
            
        # Get driver results in recent races
        recent_results = results[
            (results['driverId'] == driver_id) & 
            (results['raceId'].isin(previous_races['id']))
        ]
        
        if recent_results.empty:
            return self._get_default_form()
            
        # Calculate form metrics
        positions = recent_results['positionNumber'].dropna()
        points = recent_results['points'].fillna(0)
        dnfs = recent_results['positionText'].isin(['DNF', 'DNS', 'DSQ'])
        
        # Position trend (negative = improving)
        position_trend = 0
        if len(positions) >= 2:
            position_trend = np.polyfit(range(len(positions)), positions.values, 1)[0]
        
        # Points momentum
        points_momentum = points.iloc[-1] - points.mean() if len(points) > 1 else 0
        
        return {
            'avg_position': float(positions.mean()) if len(positions) > 0 else 15.0,
            'avg_points': float(points.mean()),
            'position_trend': float(position_trend),  # Negative = improving
            'points_momentum': float(points_momentum),
            'recent_dnf_rate': float(dnfs.mean()),
            'races_analyzed': len(recent_results),
            'last_race_position': float(positions.iloc[-1]) if len(positions) > 0 else 20.0,
            'consistency': float(positions.std()) if len(positions) > 1 else 5.0
        }
    
    def _get_default_form(self) -> Dict:
        """Get default form metrics when data unavailable"""
        return {
            'avg_position': 15.0,
            'avg_points': 0.0,
            'position_trend': 0.0,
            'points_momentum': 0.0,
            'recent_dnf_rate': 0.12,
            'races_analyzed': 0,
            'last_race_position': 20.0,
            'consistency': 5.0
        }
    
    def get_team_momentum(self, constructor_id: int, race_id: int, n_races: int = 5) -> Dict:
        """Get team momentum indicators
        
        Args:
            constructor_id: Constructor ID
            race_id: Current race ID
            n_races: Number of recent races
            
        Returns:
            Dictionary of team momentum metrics
        """
        results = self.data.get('results', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty or races.empty:
            return {'team_momentum': 0.0, 'reliability_trend': 1.0}
            
        # Get recent races
        current_race = races[races['id'] == race_id]
        if current_race.empty:
            return {'team_momentum': 0.0, 'reliability_trend': 1.0}
            
        race_date = current_race.iloc[0]['date']
        previous_races = races[races['date'] < race_date].sort_values('date', ascending=False).head(n_races)
        
        # Get team results
        team_results = results[
            (results['constructorId'] == constructor_id) & 
            (results['raceId'].isin(previous_races['id']))
        ]
        
        if team_results.empty:
            return {'team_momentum': 0.0, 'reliability_trend': 1.0}
            
        # Group by race to get team totals
        race_points = team_results.groupby('raceId')['points'].sum()
        
        # Calculate momentum (trend in points)
        momentum = 0
        if len(race_points) >= 2:
            momentum = np.polyfit(range(len(race_points)), race_points.values, 1)[0]
        
        # Reliability trend (DNF rate)
        dnf_rate = team_results['positionText'].isin(['DNF', 'DNS', 'DSQ']).mean()
        reliability = 1.0 - dnf_rate
        
        return {
            'team_momentum': float(momentum),
            'reliability_trend': float(reliability),
            'avg_team_points': float(race_points.mean()),
            'last_race_points': float(race_points.iloc[-1]) if len(race_points) > 0 else 0.0
        }
    
    def get_circuit_history(self, driver_id: int, circuit_id: int, years: int = 3) -> Dict:
        """Get driver's historical performance at specific circuit
        
        Args:
            driver_id: Driver ID
            circuit_id: Circuit ID
            years: Number of years to look back
            
        Returns:
            Dictionary of circuit-specific metrics
        """
        results = self.data.get('results', pd.DataFrame())
        races = self.data.get('races', pd.DataFrame())
        
        if results.empty or races.empty:
            return self._get_default_circuit_history()
            
        # Get races at this circuit
        circuit_races = races[
            (races['circuitId'] == circuit_id) & 
            (races['year'] >= races['year'].max() - years)
        ]
        
        if circuit_races.empty:
            return self._get_default_circuit_history()
            
        # Get driver results at circuit
        circuit_results = results[
            (results['driverId'] == driver_id) & 
            (results['raceId'].isin(circuit_races['id']))
        ]
        
        if circuit_results.empty:
            return self._get_default_circuit_history()
            
        positions = circuit_results['positionNumber'].dropna()
        points = circuit_results['points'].fillna(0)
        dnfs = circuit_results['positionText'].isin(['DNF', 'DNS', 'DSQ'])
        
        # Get overtakes at circuit
        grid = self.data.get('races_starting_grid_positions', pd.DataFrame())
        if not grid.empty:
            overtakes_list = []
            for _, result in circuit_results.iterrows():
                race_id = result['raceId']
                grid_pos = grid[(grid['driverId'] == driver_id) & (grid['raceId'] == race_id)]
                if not grid_pos.empty and pd.notna(result['positionNumber']):
                    overtakes = max(0, grid_pos.iloc[0]['positionNumber'] - result['positionNumber'])
                    overtakes_list.append(overtakes)
            
            avg_overtakes_at_circuit = np.mean(overtakes_list) if overtakes_list else 0
        else:
            avg_overtakes_at_circuit = 0
        
        return {
            'circuit_races': len(circuit_results),
            'avg_position_at_circuit': float(positions.mean()) if len(positions) > 0 else 15.0,
            'avg_points_at_circuit': float(points.mean()),
            'best_position_at_circuit': float(positions.min()) if len(positions) > 0 else 20.0,
            'dnf_rate_at_circuit': float(dnfs.mean()),
            'avg_overtakes_at_circuit': float(avg_overtakes_at_circuit),
            'circuit_affinity': 1.0 - (positions.mean() / 20.0) if len(positions) > 0 else 0.5
        }
    
    def _get_default_circuit_history(self) -> Dict:
        """Get default circuit history when data unavailable"""
        return {
            'circuit_races': 0,
            'avg_position_at_circuit': 15.0,
            'avg_points_at_circuit': 0.0,
            'best_position_at_circuit': 20.0,
            'dnf_rate_at_circuit': 0.12,
            'avg_overtakes_at_circuit': 0.0,
            'circuit_affinity': 0.5
        }
    
    def get_weather_adjustments(self, race_id: int, weather_data: Optional[Dict] = None) -> Dict:
        """Get weather-based adjustments
        
        Args:
            race_id: Race ID
            weather_data: Optional weather data (temperature, rain_probability)
            
        Returns:
            Dictionary of weather adjustments
        """
        # Default weather adjustments
        adjustments = {
            'rain_multiplier': 1.0,
            'temp_adjustment': 1.0,
            'chaos_factor': 1.0,
            'is_wet': False,
            'is_extreme_temp': False
        }
        
        if not weather_data:
            return adjustments
            
        # Rain probability adjustments
        rain_prob = weather_data.get('rain_probability', 0)
        if rain_prob > 0.7:
            adjustments['rain_multiplier'] = 1.5  # More overtakes in rain
            adjustments['chaos_factor'] = 1.8     # More unpredictability
            adjustments['is_wet'] = True
        elif rain_prob > 0.3:
            adjustments['rain_multiplier'] = 1.2
            adjustments['chaos_factor'] = 1.3
            
        # Temperature adjustments
        temp = weather_data.get('temperature', 20)
        if temp > 35:  # Very hot
            adjustments['temp_adjustment'] = 0.9  # Harder on tires
            adjustments['is_extreme_temp'] = True
        elif temp < 10:  # Very cold
            adjustments['temp_adjustment'] = 0.85  # Harder to get heat in tires
            adjustments['is_extreme_temp'] = True
            
        return adjustments
    
    def get_all_contextual_features(self, driver_id: int, constructor_id: int, 
                                   circuit_id: int, race_id: int,
                                   weather_data: Optional[Dict] = None) -> Dict:
        """Get all contextual features combined
        
        Args:
            driver_id: Driver ID
            constructor_id: Constructor ID
            circuit_id: Circuit ID
            race_id: Race ID
            weather_data: Optional weather data
            
        Returns:
            Dictionary of all contextual features
        """
        features = {}
        
        # Track features
        track_features = self.get_track_features(circuit_id)
        features.update({f'track_{k}': v for k, v in track_features.items()})
        
        # Recent form
        form_features = self.get_recent_form(driver_id, race_id)
        features.update({f'form_{k}': v for k, v in form_features.items()})
        
        # Team momentum
        team_features = self.get_team_momentum(constructor_id, race_id)
        features.update({f'team_{k}': v for k, v in team_features.items()})
        
        # Circuit history
        circuit_features = self.get_circuit_history(driver_id, circuit_id)
        features.update({f'circuit_{k}': v for k, v in circuit_features.items()})
        
        # Weather adjustments
        weather_features = self.get_weather_adjustments(race_id, weather_data)
        features.update({f'weather_{k}': v for k, v in weather_features.items()})
        
        # Composite features
        features['momentum_score'] = (
            0.4 * (1 - form_features['position_trend'] / 10) +  # Position improving
            0.3 * (team_features['team_momentum'] / 50) +       # Team improving
            0.3 * circuit_features['circuit_affinity']          # Good at circuit
        )
        
        features['risk_score'] = (
            0.3 * form_features['recent_dnf_rate'] +
            0.3 * (1 - team_features['reliability_trend']) +
            0.2 * track_features['dnf_risk_multiplier'] +
            0.2 * weather_features['chaos_factor']
        )
        
        features['overtaking_potential'] = (
            track_features['overtaking_multiplier'] *
            weather_features['rain_multiplier'] *
            (1 + circuit_features['avg_overtakes_at_circuit'] / 10)
        )
        
        return features