"""
Technical Features
==================

Classic technical analysis indicators as features.

Features:
1. Moving average crossovers
2. MACD
3. Volume-based indicators
4. Price patterns
5. Support/Resistance
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class TechnicalFeatures:
    """
    Generate technical analysis features.
    
    Philosophy:
    - Technical indicators capture market psychology
    - Volume confirms price moves
    - Classic patterns may have some predictive power
    """
    
    def __init__(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None,
        volume: pd.DataFrame = None,
        returns: pd.DataFrame = None
    ):
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.returns = returns if returns is not None else np.log(close / close.shift(1))
        self.features: Dict[str, pd.DataFrame] = {}
        
    def compute_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute all technical features."""
        if verbose:
            print("Computing technical features...")
            
        # 1. Moving average crossovers
        self._ma_crossovers()
        
        # 2. MACD
        self._macd()
        
        # 3. Volume features
        if self.volume is not None:
            self._volume_features()
            
        # 4. Price channel position
        if self.high is not None and self.low is not None:
            self._price_channel()
            
        # 5. Gap features
        self._gap_features()
        
        # 6. Money Flow Index
        if self.volume is not None and self.high is not None and self.low is not None:
            self._money_flow()
            
        if verbose:
            print(f"✅ Created {len(self.features)} technical features")
            
        return self.features
    
    def _ma_crossovers(self) -> None:
        """Moving average crossover signals."""
        # SMA crossovers
        sma_5 = self.close.rolling(5).mean()
        sma_10 = self.close.rolling(10).mean()
        sma_21 = self.close.rolling(21).mean()
        sma_50 = self.close.rolling(50).mean()
        sma_200 = self.close.rolling(200).mean()
        
        # Price vs MA
        self.features['price_vs_sma21'] = (self.close - sma_21) / (sma_21 + 1e-10)
        self.features['price_vs_sma50'] = (self.close - sma_50) / (sma_50 + 1e-10)
        self.features['price_vs_sma200'] = (self.close - sma_200) / (sma_200 + 1e-10)
        
        # MA crossovers (short vs long)
        self.features['ma_cross_5_21'] = (sma_5 - sma_21) / (sma_21 + 1e-10)
        self.features['ma_cross_21_50'] = (sma_21 - sma_50) / (sma_50 + 1e-10)
        self.features['ma_cross_50_200'] = (sma_50 - sma_200) / (sma_200 + 1e-10)
        
        # EMA crossovers
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        self.features['ema_cross_12_26'] = (ema_12 - ema_26) / (ema_26 + 1e-10)
        
    def _macd(self) -> None:
        """MACD indicator."""
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        
        # MACD line
        macd = ema_12 - ema_26
        
        # Signal line
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Histogram
        histogram = macd - signal
        
        # Normalize by price
        self.features['macd_norm'] = macd / (self.close + 1e-10)
        self.features['macd_signal_norm'] = signal / (self.close + 1e-10)
        self.features['macd_hist_norm'] = histogram / (self.close + 1e-10)
        
        # MACD momentum (change in histogram)
        self.features['macd_hist_change'] = histogram - histogram.shift(1)
        
    def _volume_features(self) -> None:
        """Volume-based indicators."""
        v = self.volume
        r = self.returns
        
        # Volume ratio (relative to average)
        vol_ma_21 = v.rolling(21).mean()
        self.features['volume_ratio'] = v / (vol_ma_21 + 1e-10)
        
        # Volume trend
        vol_ma_5 = v.rolling(5).mean()
        self.features['volume_trend'] = vol_ma_5 / (vol_ma_21 + 1e-10) - 1
        
        # On-Balance Volume direction
        obv_sign = np.sign(r) * v
        obv = obv_sign.cumsum()
        obv_ma = obv.rolling(21).mean()
        self.features['obv_trend'] = (obv - obv_ma) / (obv_ma.abs() + 1e-10)
        
        # Volume-price confirmation
        # High volume + up move = bullish
        self.features['vol_price_confirm'] = (
            self.features['volume_ratio'] * np.sign(r)
        ).rolling(5).mean()
        
        # Volume spike detection
        vol_std = v.rolling(21).std()
        self.features['volume_spike'] = (v - vol_ma_21) / (vol_std + 1e-10)
        
    def _price_channel(self) -> None:
        """Donchian channel and similar."""
        for window in [20, 55]:
            high_n = self.high.rolling(window).max()
            low_n = self.low.rolling(window).min()
            
            # Position in channel
            channel_pos = (self.close - low_n) / (high_n - low_n + 1e-10)
            self.features[f'channel_pos_{window}d'] = channel_pos * 2 - 1  # [-1, 1]
            
            # Channel width (volatility)
            channel_width = (high_n - low_n) / self.close
            self.features[f'channel_width_{window}d'] = channel_width
            
    def _gap_features(self) -> None:
        """Gap-related features."""
        # Overnight gap (if we had open, but we can estimate)
        # Using returns as proxy
        r = self.returns
        
        # Large move indicator
        vol_21 = r.rolling(21).std()
        move_zscore = r / (vol_21 + 1e-10)
        
        # Gap up/down (extreme moves)
        self.features['gap_up'] = (move_zscore > 2).astype(float).rolling(5).sum()
        self.features['gap_down'] = (move_zscore < -2).astype(float).rolling(5).sum()
        
        # Gap reversal pattern
        # If yesterday had extreme move, what's the tendency?
        prev_extreme = move_zscore.shift(1).abs() > 2
        self.features['post_gap_return'] = r.where(prev_extreme, 0).rolling(21).mean()
        
    def _money_flow(self) -> None:
        """Money Flow Index."""
        typical_price = (self.high + self.low + self.close) / 3
        raw_mf = typical_price * self.volume
        
        # Direction
        tp_diff = typical_price.diff()
        pos_mf = raw_mf.where(tp_diff > 0, 0).rolling(14).sum()
        neg_mf = raw_mf.where(tp_diff < 0, 0).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        self.features['mfi_14d'] = (mfi - 50) / 50  # Normalize to [-1, 1]


def compute_technical_features(
    close: pd.DataFrame,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    returns: pd.DataFrame = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function."""
    tf = TechnicalFeatures(close, high, low, volume, returns)
    return tf.compute_all()
