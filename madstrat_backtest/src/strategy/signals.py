"""
Signal generator for D2/D3/MAD/EQ signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

class SignalType(Enum):
    """Types of trading signals."""
    D2_BUY = "D2_BUY"
    D2_SELL = "D2_SELL"
    D3_BUY = "D3_BUY"
    D3_SELL = "D3_SELL"
    MAD_BUY = "MAD_BUY"
    MAD_SELL = "MAD_SELL"
    EQ_BUY = "EQ_BUY"
    EQ_SELL = "EQ_SELL"
    NEUTRAL = "NEUTRAL"

class SignalGenerator:
    """Generates trading signals based on confluence scoring."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize SignalGenerator.
        
        Args:
            thresholds: Thresholds for signal generation
        """
        self.thresholds = thresholds or {
            'd2_strength': 0.5,
            'd3_strength': 0.7,
            'mad_strength': 0.6,
            'eq_strength': 0.4,
            'min_confidence': 0.3
        }
    
    def generate_d2_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate D2 signals based on price action and EMAs.
        
        Args:
            df: DataFrame with OHLC and EMA data
            
        Returns:
            DataFrame with D2 signals
        """
        df = df.copy()
        
        # D2 Buy: Close above EMA(9) and EMA(9) above EMA(18)
        df['D2_Buy_Condition'] = (
            (df['Close'] > df['EMA_9']) & 
            (df['EMA_9'] > df['EMA_18'])
        )
        
        # D2 Sell: Close below EMA(9) and EMA(9) below EMA(18)
        df['D2_Sell_Condition'] = (
            (df['Close'] < df['EMA_9']) & 
            (df['EMA_9'] < df['EMA_18'])
        )
        
        # Generate D2 signals
        df['D2_Signal'] = np.where(
            df['D2_Buy_Condition'],
            SignalType.D2_BUY.value,
            np.where(
                df['D2_Sell_Condition'],
                SignalType.D2_SELL.value,
                SignalType.NEUTRAL.value
            )
        )
        
        return df
    
    def generate_d3_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate D3 signals based on price action, EMAs, and day structure.
        
        Args:
            df: DataFrame with OHLC, EMA, and day structure data
            
        Returns:
            DataFrame with D3 signals
        """
        df = df.copy()
        
        # D3 Buy: D2 buy conditions + day structure bullish
        df['D3_Buy_Condition'] = (
            (df['D2_Buy_Condition']) &
            (df['Day_Type'].isin(['GSD', 'GD'])) &
            (df['Confluence_Score'] > self.thresholds['d3_strength'])
        )
        
        # D3 Sell: D2 sell conditions + day structure bearish
        df['D3_Sell_Condition'] = (
            (df['D2_Sell_Condition']) &
            (df['Day_Type'].isin(['RSD', 'RD'])) &
            (df['Confluence_Score'] < -self.thresholds['d3_strength'])
        )
        
        # Generate D3 signals
        df['D3_Signal'] = np.where(
            df['D3_Buy_Condition'],
            SignalType.D3_BUY.value,
            np.where(
                df['D3_Sell_Condition'],
                SignalType.D3_SELL.value,
                SignalType.NEUTRAL.value
            )
        )
        
        return df
    
    def generate_mad_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MAD (Moving Average Divergence) signals.
        
        Args:
            df: DataFrame with OHLC and EMA data
            
        Returns:
            DataFrame with MAD signals
        """
        df = df.copy()
        
        # Check for divergence between price and EMA(50)
        df['Price_vs_EMA50'] = df['Close'] - df['EMA_50']
        df['Price_vs_EMA50_Change'] = df['Price_vs_EMA50'].diff()
        
        # Bullish divergence: price makes higher low, EMA makes lower low
        df['Bullish_Divergence'] = (
            (df['Close'] > df['Close'].shift(2)) &
            (df['EMA_50'] < df['EMA_50'].shift(2)) &
            (df['Low'] < df['Low'].shift(1))
        )
        
        # Bearish divergence: price makes lower high, EMA makes higher high
        df['Bearish_Divergence'] = (
            (df['Close'] < df['Close'].shift(2)) &
            (df['EMA_50'] > df['EMA_50'].shift(2)) &
            (df['High'] > df['High'].shift(1))
        )
        
        # Generate MAD signals
        df['MAD_Signal'] = np.where(
            (df['Bullish_Divergence']) & (df['Confluence_Score'] > self.thresholds['mad_strength']),
            SignalType.MAD_BUY.value,
            np.where(
                (df['Bearish_Divergence']) & (df['Confluence_Score'] < -self.thresholds['mad_strength']),
                SignalType.MAD_SELL.value,
                SignalType.NEUTRAL.value
            )
        )
        
        return df
    
    def generate_eq_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate EQ (Equilibrium) signals based on PD-EQ and ADR.
        
        Args:
            df: DataFrame with price levels and ADR data
            
        Returns:
            DataFrame with EQ signals
        """
        df = df.copy()
        
        # EQ Buy: Price near PD-EQ + bullish confluence + within ADR
        df['EQ_Buy_Condition'] = (
            (abs(df['Close'] - df['PD-EQ']) / df['PD-EQ'] < 0.002) &  # Within 0.2% of PD-EQ
            (df['Confluence_Score'] > self.thresholds['eq_strength']) &
            (df['Close'] < df['ADR_Resistance']) &
            (df['Close'] > df['ADR_Support'])
        )
        
        # EQ Sell: Price near PD-EQ + bearish confluence + within ADR
        df['EQ_Sell_Condition'] = (
            (abs(df['Close'] - df['PD-EQ']) / df['PD-EQ'] < 0.002) &  # Within 0.2% of PD-EQ
            (df['Confluence_Score'] < -self.thresholds['eq_strength']) &
            (df['Close'] < df['ADR_Resistance']) &
            (df['Close'] > df['ADR_Support'])
        )
        
        # Generate EQ signals
        df['EQ_Signal'] = np.where(
            df['EQ_Buy_Condition'],
            SignalType.EQ_BUY.value,
            np.where(
                df['EQ_Sell_Condition'],
                SignalType.EQ_SELL.value,
                SignalType.NEUTRAL.value
            )
        )
        
        return df
    
    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signals and combine them.
        
        Args:
            df: DataFrame with all required data
            
        Returns:
            DataFrame with all signals
        """
        df = df.copy()
        
        # Generate individual signals
        df = self.generate_d2_signals(df)
        df = self.generate_d3_signals(df)
        df = self.generate_mad_signals(df)
        df = self.generate_eq_signals(df)
        
        # Combine signals with priority: D3 > MAD > EQ > D2
        def get_priority_signal(row):
            signals = [
                row.get('D3_Signal', SignalType.NEUTRAL.value),
                row.get('MAD_Signal', SignalType.NEUTRAL.value),
                row.get('EQ_Signal', SignalType.NEUTRAL.value),
                row.get('D2_Signal', SignalType.NEUTRAL.value)
            ]
            
            # Return first non-neutral signal
            for signal in signals:
                if signal != SignalType.NEUTRAL.value:
                    return signal
            
            return SignalType.NEUTRAL.value
        
        df['Combined_Signal'] = df.apply(get_priority_signal, axis=1)
        
        # Add signal metadata
        df['Signal_Strength'] = df['Confluence_Score']
        df['Signal_Confidence'] = abs(df['Confluence_Score'])
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of all signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Dictionary with signal summary
        """
        signal_counts = df['Combined_Signal'].value_counts()
        
        summary = {
            'total_signals': len(df),
            'd2_buy': signal_counts.get(SignalType.D2_BUY.value, 0),
            'd2_sell': signal_counts.get(SignalType.D2_SELL.value, 0),
            'd3_buy': signal_counts.get(SignalType.D3_BUY.value, 0),
            'd3_sell': signal_counts.get(SignalType.D3_SELL.value, 0),
            'mad_buy': signal_counts.get(SignalType.MAD_BUY.value, 0),
            'mad_sell': signal_counts.get(SignalType.MAD_SELL.value, 0),
            'eq_buy': signal_counts.get(SignalType.EQ_BUY.value, 0),
            'eq_sell': signal_counts.get(SignalType.EQ_SELL.value, 0),
            'neutral': signal_counts.get(SignalType.NEUTRAL.value, 0),
            'avg_strength': df['Signal_Strength'].mean(),
            'avg_confidence': df['Signal_Confidence'].mean()
        }
        
        return summary