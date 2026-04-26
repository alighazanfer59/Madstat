"""
Day count tracking, resets, and freezes for trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class DayCounter:
    """Tracks day counts for trading strategy."""
    
    def __init__(self, reset_threshold: int = 3, freeze_threshold: int = 5):
        """
        Initialize DayCounter.
        
        Args:
            reset_threshold: Threshold for resetting counters
            freeze_threshold: Threshold for freezing counters
        """
        self.reset_threshold = reset_threshold
        self.freeze_threshold = freeze_threshold
        self.counters = {
            'up_days': 0,
            'down_days': 0,
            'inside_days': 0,
            'consecutive_up': 0,
            'consecutive_down': 0,
            'last_direction': None
        }
    
    def reset_counters(self):
        """Reset all counters."""
        self.counters = {
            'up_days': 0,
            'down_days': 0,
            'inside_days': 0,
            'consecutive_up': 0,
            'consecutive_down': 0,
            'last_direction': None
        }
    
    def update_counters(self, day_data: Dict) -> Dict:
        """
        Update counters based on day data.
        
        Args:
            day_data: Dictionary with day information
            
        Returns:
            Updated counters
        """
        current_direction = day_data.get('direction')
        
        # Check if counter needs to be reset
        if (self.counters['up_days'] >= self.reset_threshold or 
            self.counters['down_days'] >= self.reset_threshold or
            self.counters['inside_days'] >= self.reset_threshold):
            self.reset_counters()
        
        # Check if counter should be frozen
        if (self.counters['consecutive_up'] >= self.freeze_threshold or 
            self.counters['consecutive_down'] >= self.freeze_threshold):
            return self.counters
        
        # Update counters based on direction
        if current_direction == 'up':
            self.counters['up_days'] += 1
            self.counters['consecutive_up'] += 1
            self.counters['consecutive_down'] = 0
        elif current_direction == 'down':
            self.counters['down_days'] += 1
            self.counters['consecutive_down'] += 1
            self.counters['consecutive_up'] = 0
        else:  # inside day
            self.counters['inside_days'] += 1
            self.counters['consecutive_up'] = 0
            self.counters['consecutive_down'] = 0
        
        self.counters['last_direction'] = current_direction
        
        return self.counters
    
    def get_counter_status(self) -> Dict:
        """Get current counter status."""
        status = self.counters.copy()
        
        # Add status flags
        status['reset_triggered'] = (
            status['up_days'] >= self.reset_threshold or
            status['down_days'] >= self.reset_threshold or
            status['inside_days'] >= self.reset_threshold
        )
        
        status['frozen'] = (
            status['consecutive_up'] >= self.freeze_threshold or
            status['consecutive_down'] >= self.freeze_threshold
        )
        
        return status
    
    @staticmethod
    def analyze_day_type(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze day types and directions for each day.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with day type analysis
        """
        df = df.copy()
        
        # Determine day direction
        df['Day_Direction'] = np.where(df['Close'] > df['Open'], 'up',
                                     np.where(df['Close'] < df['Open'], 'down', 'inside'))
        
        # Calculate body percentage
        df['Body_Percentage'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])) * 100
        
        # Classify day strength
        df['Day_Strength'] = np.where(
            df['Body_Percentage'] >= 70, 'strong',
            np.where(df['Body_Percentage'] <= 30, 'weak', 'normal')
        )
        
        return df
    
    @staticmethod
    def track_daily_counts(df: pd.DataFrame, 
                          reset_threshold: int = 3,
                          freeze_threshold: int = 5) -> pd.DataFrame:
        """
        Track daily counts for the entire DataFrame.
        
        Args:
            df: DataFrame with OHLC data
            reset_threshold: Threshold for resetting counters
            freeze_threshold: Threshold for freezing counters
            
        Returns:
            DataFrame with daily count tracking
        """
        df = df.copy()
        
        # Analyze day types
        df = DayCounter.analyze_day_type(df)
        
        # Initialize counters
        counter = DayCounter(reset_threshold, freeze_threshold)
        
        # Track counts for each day
        counts = []
        
        for i in range(len(df)):
            day_data = {
                'direction': df.iloc[i]['Day_Direction'],
                'strength': df.iloc[i]['Day_Strength']
            }
            
            # Update counters
            counters = counter.update_counters(day_data)
            
            # Add to tracking
            counts.append({
                'Date': df.iloc[i]['Date'],
                'Day_Direction': df.iloc[i]['Day_Direction'],
                'Day_Strength': df.iloc[i]['Day_Strength'],
                'Up_Days': counters['up_days'],
                'Down_Days': counters['down_days'],
                'Inside_Days': counters['inside_days'],
                'Consecutive_Up': counters['consecutive_up'],
                'Consecutive_Down': counters['consecutive_down'],
                'Reset_Triggered': counter.get_counter_status()['reset_triggered'],
                'Frozen': counter.get_counter_status()['frozen']
            })
        
        # Convert to DataFrame and merge
        counts_df = pd.DataFrame(counts)
        df = pd.merge(df, counts_df, on='Date', how='left')
        
        return df
    
    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of day counts.
        
        Args:
            df: DataFrame with count tracking
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_days': len(df),
            'up_days': df['Up_Days'].max(),
            'down_days': df['Down_Days'].max(),
            'inside_days': df['Inside_Days'].max(),
            'max_consecutive_up': df['Consecutive_Up'].max(),
            'max_consecutive_down': df['Consecutive_Down'].max(),
            'reset_events': df['Reset_Triggered'].sum(),
            'freeze_events': df['Frozen'].sum()
        }
        
        return stats