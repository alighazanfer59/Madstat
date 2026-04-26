"""
Data processor module for cleaning and aligning multi-timeframe data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os

class DataProcessor:
    """Handles cleaning and alignment of multi-timeframe data."""
    
    def __init__(self, processed_path: str = "data/processed"):
        """
        Initialize the DataProcessor.
        
        Args:
            processed_path: Path to store processed data files
        """
        self.processed_path = processed_path
        os.makedirs(processed_path, exist_ok=True)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df.sort_values('Date', inplace=True)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found:\n{missing_values}")
            # Forward fill missing values
            df.fillna(method='ffill', inplace=True)
            
            # If still missing (e.g., first rows), backfill
            df.fillna(method='bfill', inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(subset=['Date'], keep='first', inplace=True)
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def align_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple timeframes to a common timeline.
        
        Args:
            data_dict: Dictionary of timeframes and their DataFrames
            
        Returns:
            Dictionary of aligned DataFrames
        """
        # Get the earliest and latest dates across all timeframes
        all_dates = []
        for df in data_dict.values():
            all_dates.extend(df['Date'].tolist())
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        # Create a complete datetime index
        full_timeline = pd.date_range(start=start_date, end=end_date, freq='D')
        
        aligned_data = {}
        
        # Align each timeframe
        for timeframe, df in data_dict.items():
            # Set Date as index
            df.set_index('Date', inplace=True)
            
            # Resample according to timeframe
            if timeframe == '15m':
                resampled = df.resample('15T').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            elif timeframe == '1h':
                resampled = df.resample('H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            elif timeframe == '4h':
                resampled = df.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            else:
                # Default to hourly for other timeframes
                resampled = df.resample('H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
            
            # Drop any NaN values that might result from resampling
            resampled.dropna(inplace=True)
            
            # Reset index
            resampled.reset_index(inplace=True)
            
            # Store aligned data
            aligned_data[timeframe] = resampled
        
        return aligned_data
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame], symbol: str):
        """
        Save processed data to CSV files.
        
        Args:
            data_dict: Dictionary of timeframes and their DataFrames
            symbol: Trading symbol
        """
        for timeframe, df in data_dict.items():
            filename = f"{symbol}_{timeframe}_processed.csv"
            filepath = os.path.join(self.processed_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load previously processed data.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to load
            
        Returns:
            Dictionary of timeframes and their DataFrames
        """
        data_dict = {}
        
        for timeframe in timeframes:
            filename = f"{symbol}_{timeframe}_processed.csv"
            filepath = os.path.join(self.processed_path, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                data_dict[timeframe] = df
            else:
                print(f"Processed data file not found: {filepath}")
        
        return data_dict
    
    def process_multiple_timeframes(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Process data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to process
            
        Returns:
            Dictionary of timeframes and their processed DataFrames
        """
        # This would typically be called after fetching data
        # For now, just return an empty dictionary
        # In a real implementation, this would:
        # 1. Load raw data for each timeframe
        # 2. Clean each timeframe
        # 3. Align timeframes
        # 4. Save processed data
        return {}