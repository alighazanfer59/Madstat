"""
Average Daily Range (ADR) calculations for trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class ADRCalculator:
    """Calculates Average Daily Range for trading strategy."""
    
    @staticmethod
    def calculate_daily_range(df: pd.DataFrame, use_close: bool = True) -> pd.DataFrame:
        """
        Calculate daily range for each day.
        
        Args:
            df: DataFrame with OHLC data
            use_close: Whether to use Close price for range calculation
            
        Returns:
            DataFrame with Daily Range column
        """
        df = df.copy()
        
        # Group by date and calculate range for each day
        if use_close:
            # Range using Close price
            daily_stats = df.groupby(df['Date'].dt.date).agg({
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
            
            # Calculate daily range as difference between high and low
            daily_stats['Daily_Range'] = daily_stats['High'] - daily_stats['Low']
            
            # Merge back to original dataframe
            df = df.merge(daily_stats[['Daily_Range']], left_on=df['Date'].dt.date, right_index=True)
        else:
            # Range using Open price
            daily_stats = df.groupby(df['Date'].dt.date).agg({
                'High': 'max',
                'Low': 'min',
                'Open': 'first'
            })
            
            # Calculate daily range as difference between high and low
            daily_stats['Daily_Range'] = daily_stats['High'] - daily_stats['Low']
            
            # Merge back to original dataframe
            df = df.merge(daily_stats[['Daily_Range']], left_on=df['Date'].dt.date, right_index=True)
        
        return df
    
    @staticmethod
    def calculate_adr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Daily Range over specified period.
        
        Args:
            df: DataFrame with Daily Range column
            period: Number of days for ADR calculation
            
        Returns:
            DataFrame with ADR column
        """
        df = df.copy()
        
        # Calculate rolling average of daily range
        df['ADR'] = df['Daily_Range'].rolling(window=period, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def calculate_adr_levels(df: pd.DataFrame, 
                            atr_multiplier: float = 1.5,
                            use_close: bool = True) -> pd.DataFrame:
        """
        Calculate ADR-based levels for trading.
        
        Args:
            df: DataFrame with ADR column
            atr_multiplier: Multiplier for ADR levels
            use_close: Whether to use Close price for reference
            
        Returns:
            DataFrame with ADR level columns
        """
        df = df.copy()
        
        # Reference price (Close or Open)
        ref_price = df['Close'] if use_close else df['Open']
        
        # Calculate ADR levels
        df['ADR_Resistance'] = ref_price + (df['ADR'] * atr_multiplier)
        df['ADR_Support'] = ref_price - (df['ADR'] * atr_multiplier)
        
        # Calculate ADR levels with 0.5 multiplier
        df['ADR_Mid_Resistance'] = ref_price + (df['ADR'] * 0.5)
        df['ADR_Mid_Support'] = ref_price - (df['ADR'] * 0.5)
        
        return df
    
    @staticmethod
    def calculate_all_adr_metrics(df: pd.DataFrame, 
                                 adr_period: int = 14,
                                 atr_multiplier: float = 1.5,
                                 use_close: bool = True) -> pd.DataFrame:
        """
        Calculate all ADR metrics in one function.
        
        Args:
            df: DataFrame with OHLC data
            adr_period: Period for ADR calculation
            atr_multiplier: Multiplier for ADR levels
            use_close: Whether to use Close price for reference
            
        Returns:
            DataFrame with all ADR metrics
        """
        df = df.copy()
        
        # Calculate daily range
        df = ADRCalculator.calculate_daily_range(df, use_close)
        
        # Calculate ADR
        df = ADRCalculator.calculate_adr(df, adr_period)
        
        # Calculate ADR levels
        df = ADRCalculator.calculate_adr_levels(df, atr_multiplier, use_close)
        
        return df
    
    @staticmethod
    def get_adr_percentage(range_val: float, price: float) -> float:
        """
        Calculate ADR as percentage of price.
        
        Args:
            range_val: Daily range value
            price: Reference price
            
        Returns:
            ADR as percentage of price
        """
        if price == 0:
            return 0
        return (range_val / price) * 100
    
    @staticmethod
    def is_above_adr_resistance(price: float, adr_resistance: float) -> bool:
        """Check if price is above ADR resistance."""
        return price > adr_resistance
    
    @staticmethod
    def is_below_adr_support(price: float, adr_support: float) -> bool:
        """Check if price is below ADR support."""
        return price < adr_support
    
    @staticmethod
    def is_in_adr_range(price: float, 
                        adr_support: float, 
                        adr_resistance: float) -> bool:
        """Check if price is within ADR range."""
        return adr_support <= price <= adr_resistance