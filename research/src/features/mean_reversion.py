"""
Mean Reversion Features
=======================

Features based on mean reversion hypothesis.
Assets tend to revert to some equilibrium level.

Features:
1. Z-scores from moving averages
2. RSI (Relative Strength Index)
3. Bollinger Band position
4. Short-term reversal
5. Distance from highs/lows
"""

import pandas as pd
import numpy as np
from typing import Dict


class MeanReversionFeatures:
    """
    Generate mean reversion features.
    
    Philosophy:
    - Prices/returns tend to revert to a mean
    - Extreme moves are often followed by reversals
    - Different reversion speeds at different horizons
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        close: pd.DataFrame = None
    ):
        self.returns = returns
        self.close = close if close is not None else (1 + returns).cumprod()
        self.features: Dict[str, pd.DataFrame] = {}
        
    def compute_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute all mean reversion features."""
        if verbose:
            print("Computing mean reversion features...")
            
        # 1. Z-scores from moving averages
        self._zscore_features()
        
        # 2. RSI
        self._rsi_features()
        
        # 3. Bollinger Band features
        self._bollinger_features()
        
        # 4. Short-term reversal
        self._reversal_features()
        
        # 5. Distance from extremes
        self._extreme_distance()
        
        # 6. Return deviation from norm
        self._return_deviation()
        
        # 7. Stochastic oscillator
        self._stochastic()
        
        if verbose:
            print(f"✅ Created {len(self.features)} mean reversion features")
            
        return self.features
    
    def _zscore_features(self) -> None:
        """Z-score: how far price is from its moving average."""
        for window in [10, 21, 42, 63]:
            ma = self.close.rolling(window).mean()
            std = self.close.rolling(window).std()
            
            zscore = (self.close - ma) / (std + 1e-10)
            self.features[f'zscore_{window}d'] = zscore.clip(-4, 4)
            
    def _rsi_features(self) -> None:
        """Relative Strength Index - momentum/reversal indicator."""
        for window in [7, 14, 21]:
            # Separate gains and losses
            delta = self.close.diff()
            gains = delta.clip(lower=0)
            losses = (-delta).clip(lower=0)
            
            # Smoothed averages
            avg_gain = gains.rolling(window).mean()
            avg_loss = losses.rolling(window).mean()
            
            # RSI formula
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Normalize to [-1, 1] centered at 0
            self.features[f'rsi_{window}d'] = (rsi - 50) / 50
            
    def _bollinger_features(self) -> None:
        """Bollinger Band position."""
        for window in [20, 40]:
            ma = self.close.rolling(window).mean()
            std = self.close.rolling(window).std()
            
            upper = ma + 2 * std
            lower = ma - 2 * std
            
            # Position within band: -1 (at lower) to +1 (at upper)
            bb_pos = (self.close - lower) / (upper - lower + 1e-10) * 2 - 1
            self.features[f'bb_pos_{window}d'] = bb_pos.clip(-2, 2)
            
            # Bandwidth (volatility measure)
            bandwidth = (upper - lower) / (ma + 1e-10)
            self.features[f'bb_width_{window}d'] = bandwidth
            
    def _reversal_features(self) -> None:
        """Short-term reversal signals."""
        # 1-day reversal
        self.features['reversal_1d'] = -self.returns.shift(1)
        
        # 5-day reversal
        self.features['reversal_5d'] = -self.returns.rolling(5).sum().shift(1)
        
        # Reversal after extreme move
        ret_1d = self.returns.shift(1)
        vol_21 = self.returns.rolling(21).std()
        ret_zscore = ret_1d / (vol_21 + 1e-10)
        
        # Strong reversal signal when previous move was extreme
        self.features['extreme_reversal'] = -ret_zscore.clip(-3, 3)
        
    def _extreme_distance(self) -> None:
        """Distance from recent highs and lows."""
        for window in [21, 63, 252]:
            roll_max = self.close.rolling(window).max()
            roll_min = self.close.rolling(window).min()
            
            # Distance from high (0 = at high, 1 = at low)
            dist_high = (roll_max - self.close) / (roll_max - roll_min + 1e-10)
            self.features[f'dist_from_high_{window}d'] = dist_high
            
            # Days since high
            # (This is expensive, simplified version)
            is_max = (self.close == roll_max).astype(float)
            self.features[f'at_high_{window}d'] = is_max.rolling(5).mean()
            
    def _return_deviation(self) -> None:
        """How much does recent return deviate from historical norm?"""
        ret_5 = self.returns.rolling(5).sum()
        
        # Long-term average return
        ret_mean_252 = self.returns.rolling(252).mean() * 5  # Scale to 5-day
        ret_std_252 = self.returns.rolling(252).std() * np.sqrt(5)
        
        # Deviation from historical mean
        deviation = (ret_5 - ret_mean_252) / (ret_std_252 + 1e-10)
        self.features['ret_deviation_252d'] = deviation.clip(-4, 4)
        
    def _stochastic(self) -> None:
        """Stochastic oscillator (%K, %D)."""
        for window in [14, 21]:
            low_n = self.close.rolling(window).min()
            high_n = self.close.rolling(window).max()
            
            # %K: where price is in the range
            k = (self.close - low_n) / (high_n - low_n + 1e-10) * 100
            
            # %D: smoothed %K
            d = k.rolling(3).mean()
            
            # Normalize to [-1, 1]
            self.features[f'stoch_k_{window}d'] = (k - 50) / 50
            self.features[f'stoch_d_{window}d'] = (d - 50) / 50


def compute_mean_reversion_features(
    returns: pd.DataFrame,
    close: pd.DataFrame = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function."""
    mrf = MeanReversionFeatures(returns, close)
    return mrf.compute_all()
