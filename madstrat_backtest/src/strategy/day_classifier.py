"""
Day classification for GSD/RSD/GD/RD/Inside Day/FBR logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time

class DayClassifier:
    """Classifies trading days based on price action."""
    
    @staticmethod
    def calculate_body_range(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candle body range.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with body range columns
        """
        df = df.copy()
        df['Body_Range'] = abs(df['Close'] - df['Open'])
        df['Body_Percentage'] = (df['Body_Range'] / (df['High'] - df['Low'])) * 100
        return df
    
    @staticmethod
    def classify_day_type(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify day type: GSD, RSD, GD, RD, Inside Day.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with day classification
        """
        df = df.copy()
        
        # Calculate body range
        df = DayClassifier.calculate_body_range(df)
        
        # Calculate total range
        df['Total_Range'] = df['High'] - df['Low']
        
        # Classify day types
        conditions = [
            (df['Body_Percentage'] >= 70),  # Green Strong Day (GSD)
            (df['Body_Percentage'] <= 30) & (df['Close'] > df['Open']),  # Red Strong Day (RSD)
            (df['Body_Percentage'] >= 50) & (df['Close'] > df['Open']),  # Green Day (GD)
            (df['Body_Percentage'] >= 50) & (df['Close'] < df['Open']),  # Red Day (RD)
            (df['Body_Percentage'] <= 40),  # Inside Day
        ]
        
        choices = ['GSD', 'RSD', 'GD', 'RD', 'Inside Day']
        
        df['Day_Type'] = np.select(conditions, choices, default='Normal')
        
        return df
    
    @staticmethod
    def is_fbr_day(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Identify Failure Breakout (FBR) days.
        
        Args:
            df: DataFrame with OHLC data
            threshold: Threshold for FBR identification
            
        Returns:
            DataFrame with FBR indicator
        """
        df = df.copy()
        
        # Calculate range from previous day high/low
        df['Prev_High'] = df['High'].shift(1)
        df['Prev_Low'] = df['Low'].shift(1)
        
        # Check for failed breakouts
        df['Failed_Breakout_High'] = (df['High'] > df['Prev_High']) & \
                                    (df['Close'] < df['Prev_High'])
        df['Failed_Breakout_Low'] = (df['Low'] < df['Prev_Low']) & \
                                   (df['Close'] > df['Prev_Low'])
        
        df['FBR_Day'] = df['Failed_Breakout_High'] | df['Failed_Breakout_Low']
        
        return df
    
    @staticmethod
    def calculate_day_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate day strength score.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with day strength score
        """
        df = df.copy()
        
        # Calculate body range percentage
        df = DayClassifier.calculate_body_range(df)
        
        # Calculate close position within range
        df['Close_Position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])) * 100
        
        # Calculate strength score
        df['Strength_Score'] = (
            (df['Body_Percentage'] / 100) * 0.6 +  # Body weight: 60%
            (df['Close_Position'] / 100) * 0.4    # Close position weight: 40%
        )
        
        return df
    
    @staticmethod
    def classify_all_days(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete day classification.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all classification metrics
        """
        df = df.copy()
        
        # Classify day types
        df = DayClassifier.classify_day_type(df)
        
        # Identify FBR days
        df = DayClassifier.is_fbr_day(df)
        
        # Calculate day strength
        df = DayClassifier.calculate_day_strength(df)
        
        return df
    
    @staticmethod
    def get_day_stats(df: pd.DataFrame) -> Dict[str, int]:
        """
        Get statistics of day types.
        
        Args:
            df: DataFrame with day classifications
            
        Returns:
            Dictionary with day type counts
        """
        day_types = df['Day_Type'].value_counts()
        fbr_count = df['FBR_Day'].sum()
        
        stats = {
            'GSD': day_types.get('GSD', 0),
            'RSD': day_types.get('RSD', 0),
            'GD': day_types.get('GD', 0),
            'RD': day_types.get('RD', 0),
            'Inside Day': day_types.get('Inside Day', 0),
            'Normal': day_types.get('Normal', 0),
            'FBR': int(fbr_count)
        }
        
        return stats