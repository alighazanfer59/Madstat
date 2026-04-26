"""
Confluence scoring system for trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class ConfluenceScorer:
    """Scores confluence of trading signals."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ConfluenceScorer.
        
        Args:
            weights: Weights for different signal components
        """
        self.weights = weights or {
            'price_action': 0.3,
            'ema_crossover': 0.2,
            'price_levels': 0.2,
            'adr': 0.15,
            'day_structure': 0.15
        }
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_price_action_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action score.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with price action scores
        """
        df = df.copy()
        
        # Calculate body strength
        df['Body_Percentage'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])) * 100
        
        # Calculate close position
        df['Close_Position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])) * 100
        
        # Price action score based on close position and body strength
        df['Price_Action_Score'] = (
            (df['Close_Position'] / 100) * 0.6 +  # Close position weight: 60%
            (df['Body_Percentage'] / 100) * 0.4   # Body strength weight: 40%
        )
        
        return df
    
    def calculate_ema_crossover_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA crossover score.
        
        Args:
            df: DataFrame with EMA columns
            
        Returns:
            DataFrame with EMA crossover scores
        """
        df = df.copy()
        
        # Check for crossovers
        df['Fast_Above_Slow'] = df['EMA_9'] > df['EMA_18']
        df['Fast_Above_Slow_Shift'] = df['Fast_Above_Slow'].shift(1)
        
        # Identify crossovers
        df['Crossover_Up'] = (df['Fast_Above_Slow'] == True) & (df['Fast_Above_Slow_Shift'] == False)
        df['Crossover_Down'] = (df['Fast_Above_Slow'] == False) & (df['Fast_Above_Slow_Shift'] == True)
        
        # Score based on current position relative to EMAs
        df['EMA_Position'] = ((df['Close'] - df['EMA_9']) / df['EMA_9']) * 100
        
        # EMA crossover score
        df['EMA_Score'] = np.where(
            df['Crossover_Up'],
            1.0,  # Strong buy signal
            np.where(
                df['Crossover_Down'],
                -1.0,  # Strong sell signal
                np.tanh(df['EMA_Position'] / 5)  # Gradual score based on position
            )
        )
        
        return df
    
    def calculate_price_level_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price level score based on PDH, PDL, PD-EQ.
        
        Args:
            df: DataFrame with price levels
            
        Returns:
            DataFrame with price level scores
        """
        df = df.copy()
        
        # Calculate distances from key levels
        df['Dist_to_PDH'] = (df['High'] - df['PDH']) / df['PDH'] * 100
        df['Dist_to_PDL'] = (df['PDL'] - df['Low']) / df['PDL'] * 100
        df['Dist_to_PD_EQ'] = abs(df['Close'] - df['PD-EQ']) / df['PD-EQ'] * 100
        
        # Score based on proximity to levels
        df['Level_Score'] = np.where(
            df['Dist_to_PDH'] < 0.1,  # Very close to PDH
            0.5,  # Potential resistance
            np.where(
                df['Dist_to_PDL'] < 0.1,  # Very close to PDL
                -0.5,  # Potential support
                np.where(
                    df['Dist_to_PD_EQ'] < 0.05,  # Very close to PD-EQ
                    0,  # Neutral
                    np.tanh(-df['Dist_to_PD_EQ'] / 10)  # Gradual score based on distance
                )
            )
        )
        
        return df
    
    def calculate_adr_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADR-based score.
        
        Args:
            df: DataFrame with ADR data
            
        Returns:
            DataFrame with ADR scores
        """
        df = df.copy()
        
        # Check position relative to ADR levels
        df['Above_ADR_Resistance'] = df['Close'] > df['ADR_Resistance']
        df['Below_ADR_Support'] = df['Close'] < df['ADR_Support']
        df['In_ADR_Mid'] = (
            (df['Close'] > df['ADR_Mid_Support']) & 
            (df['Close'] < df['ADR_Mid_Resistance'])
        )
        
        # ADR score
        df['ADR_Score'] = np.where(
            df['Above_ADR_Resistance'],
            0.3,  # Potential resistance
            np.where(
                df['Below_ADR_Support'],
                -0.3,  # Potential support
                np.where(
                    df['In_ADR_Mid'],
                    0.1,  # Neutral slight bias
                    0  # Neutral
                )
            )
        )
        
        return df
    
    def calculate_day_structure_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate day structure score.
        
        Args:
            df: DataFrame with day classification
            
        Returns:
            DataFrame with day structure scores
        """
        df = df.copy()
        
        # Score based on day type
        day_scores = {
            'GSD': 0.4,    # Green Strong Day - bullish
            'RSD': -0.4,   # Red Strong Day - bearish
            'GD': 0.2,     # Green Day - slightly bullish
            'RD': -0.2,    # Red Day - slightly bearish
            'Inside Day': 0,  # Neutral
            'Normal': 0    # Neutral
        }
        
        df['Day_Structure_Score'] = df['Day_Type'].map(day_scores).fillna(0)
        
        # Adjust based on FBR days
        df['Day_Structure_Score'] = np.where(
            df['FBR_Day'],
            df['Day_Structure_Score'] * 1.5,  # Amplify score on FBR days
            df['Day_Structure_Score']
        )
        
        return df
    
    def calculate_confluence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall confluence score.
        
        Args:
            df: DataFrame with all component scores
            
        Returns:
            DataFrame with confluence scores
        """
        df = df.copy()
        
        # Calculate weighted score
        df['Confluence_Score'] = (
            df['Price_Action_Score'] * self.weights['price_action'] +
            df['EMA_Score'] * self.weights['ema_crossover'] +
            df['Level_Score'] * self.weights['price_levels'] +
            df['ADR_Score'] * self.weights['adr'] +
            df['Day_Structure_Score'] * self.weights['day_structure']
        )
        
        # Generate signals based on score
        df['Signal'] = np.where(
            df['Confluence_Score'] > 0.3,
            SignalType.BUY.value,
            np.where(
                df['Confluence_Score'] < -0.3,
                SignalType.SELL.value,
                SignalType.NEUTRAL.value
            )
        )
        
        # Add confidence level
        df['Confidence'] = abs(df['Confluence_Score'])
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of signals.
        
        Args:
            df: DataFrame with confluence scores
            
        Returns:
            Dictionary with signal summary
        """
        signal_counts = df['Signal'].value_counts()
        
        summary = {
            'total_signals': len(df),
            'buy_signals': signal_counts.get('BUY', 0),
            'sell_signals': signal_counts.get('SELL', 0),
            'neutral_signals': signal_counts.get('NEUTRAL', 0),
            'avg_confidence_buy': df.loc[df['Signal'] == 'BUY', 'Confidence'].mean(),
            'avg_confidence_sell': df.loc[df['Signal'] == 'SELL', 'Confidence'].mean(),
            'avg_confidence_neutral': df.loc[df['Signal'] == 'NEUTRAL', 'Confidence'].mean()
        }
        
        return summary