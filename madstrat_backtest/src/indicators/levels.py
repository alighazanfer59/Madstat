"""
Price levels calculations for PDH, PDL, PD-EQ, PWH, PWL.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, time

class PriceLevelsCalculator:
    """Calculates various price levels for trading strategy."""
    
    @staticmethod
    def calculate_pdh_pdl(df: pd.DataFrame, timeframe: str = '1D') -> pd.DataFrame:
        """
        Calculate Previous Day High (PDH) and Previous Day Low (PDL.
        
        Args:
            df: DataFrame with OHLC data
            timeframe: Timeframe for calculation
            
        Returns:
            DataFrame with PDH and PDL columns
        """
        df = df.copy()
        
        if timeframe == '1D':
            # For daily data, use previous day's high and low
            df['PDH'] = df['high'].shift(1)
            df['PDL'] = df['low'].shift(1)
        else:
            # For intraday data, calculate previous day's high and low
            # Group by date and get max high and min low
            daily_stats = df.groupby(df['Date'].dt.date).agg({
                'High': 'max',
                'Low': 'min'
            }).rename(columns={'High': 'Daily_High', 'Low': 'Daily_Low'})
            
            # Merge with original dataframe
            df = df.merge(daily_stats, left_on=df['Date'].dt.date, right_index=True)
            
            # Shift to get previous day's values
            df['PDH'] = df['Daily_High'].shift(1)
            df['PDL'] = df['Daily_Low'].shift(1)
            
            # Drop temporary columns
            df.drop(['Daily_High', 'Daily_Low'], axis=1, inplace=True)
        
        return df
    
    @staticmethod
    def calculate_pd_eq(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Previous Day Equilibrium (PD-EQ) level.
        
        PD-EQ is the midpoint between PDH and PDL.
        
        Args:
            df: DataFrame with PDH and PDL columns
            
        Returns:
            DataFrame with PD-EQ column
        """
        df = df.copy()
        df['PD-EQ'] = (df['PDH'] + df['PDL']) / 2
        return df
    
    @staticmethod
    def calculate_pwh_pwl(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate Previous Week High (PWH) and Previous Week Low (PWL).
        
        Args:
            df: DataFrame with OHLC data
            window: Number of days to look back for week calculation
            
        Returns:
            DataFrame with PWH and PWL columns
        """
        df = df.copy()
        
        # Calculate rolling high and low for the specified window
        df['PWH'] = df['High'].rolling(window=window, min_periods=1).max()
        df['PWL'] = df['Low'].rolling(window=window, min_periods=1).min()
        
        return df
    
    @staticmethod
    def calculate_all_levels(df: pd.DataFrame, 
                            pwh_window: int = 5) -> pd.DataFrame:
        """
        Calculate all price levels at once.
        
        Args:
            df: DataFrame with OHLC data
            pwh_window: Window for PWH/PWL calculation
            
        Returns:
            DataFrame with all price level columns
        """
        df = df.copy()
        
        # Calculate PDH and PDL
        df = PriceLevelsCalculator.calculate_pdh_pdl(df)
        
        # Calculate PD-EQ
        df = PriceLevelsCalculator.calculate_pd_eq(df)
        
        # Calculate PWH and PWL
        df = PriceLevelsCalculator.calculate_pwh_pwl(df, pwh_window)
        
        return df
    
    @staticmethod
    def is_above_pdh(price: float, pdh: float) -> bool:
        """Check if price is above PDH."""
        return price > pdh
    
    @staticmethod
    def is_below_pdl(price: float, pdl: float) -> bool:
        """Check if price is below PDL."""
        return price < pdl
    
    @staticmethod
    def is_above_pd_eq(price: float, pd_eq: float) -> bool:
        """Check if price is above PD-EQ."""
        return price > pd_eq
    
    @staticmethod
    def is_below_pd_eq(price: float, pd_eq: float) -> bool:
        """Check if price is below PD-EQ."""
        return price < pd_eq