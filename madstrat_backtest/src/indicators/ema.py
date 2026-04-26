"""
EMA/SMA calculations for technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict

class EMA_SMA_Calculator:
    """Calculates EMA and SMA indicators."""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Price data series
            period: SMA period
            
        Returns:
            Series with SMA values
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price data series
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_multiple_emas(data: pd.DataFrame, price_col: str, 
                              periods: List[int] = [9, 18, 50]) -> pd.DataFrame:
        """
        Calculate multiple EMA columns for a DataFrame.
        
        Args:
            data: DataFrame with price data
            price_col: Name of price column ('Close', 'High', 'Low', etc.)
            periods: List of EMA periods to calculate
            
        Returns:
            DataFrame with additional EMA columns
        """
        df = data.copy()
        
        for period in periods:
            ema_col = f'EMA_{period}'
            df[ema_col] = EMA_SMA_Calculator.ema(df[price_col], period)
        
        return df
    
    @staticmethod
    def calculate_multiple_smas(data: pd.DataFrame, price_col: str, 
                              periods: List[int] = [9, 18, 50]) -> pd.DataFrame:
        """
        Calculate multiple SMA columns for a DataFrame.
        
        Args:
            data: DataFrame with price data
            price_col: Name of price column ('Close', 'High', 'Low', etc.)
            periods: List of SMA periods to calculate
            
        Returns:
            DataFrame with additional SMA columns
        """
        df = data.copy()
        
        for period in periods:
            sma_col = f'SMA_{period}'
            df[sma_col] = EMA_SMA_Calculator.sma(df[price_col], period)
        
        return df
    
    @staticmethod
    def ema_crossover_signals(data: pd.DataFrame, fast_period: int = 9, 
                            slow_period: int = 18) -> pd.Series:
        """
        Generate signals based on EMA crossover.
        
        Args:
            data: DataFrame with EMA columns
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Series with signals: 1 for buy, -1 for sell, 0 for neutral
        """
        fast_ema = f'EMA_{fast_period}'
        slow_ema = f'EMA_{slow_period}'
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: Fast EMA crosses above slow EMA
        buy_condition = (data[fast_ema] > data[slow_ema]) & \
                       (data[fast_ema].shift(1) <= data[slow_ema].shift(1))
        signals[buy_condition] = 1
        
        # Sell signal: Fast EMA crosses below slow EMA
        sell_condition = (data[fast_ema] < data[slow_ema]) & \
                        (data[fast_ema].shift(1) >= data[slow_ema].shift(1))
        signals[sell_condition] = -1
        
        return signals