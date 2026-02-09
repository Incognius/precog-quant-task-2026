"""
Momentum Features
=================

Price/return momentum-based predictive features.
These capture trending behavior and persistence in returns.

Features:
1. Cumulative returns at multiple horizons
2. Momentum (short-term vs long-term)
3. Acceleration (change in momentum)
4. Trend strength indicators
5. Time-series momentum (TSMOM)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class MomentumFeatures:
    """
    Generate momentum-based features for alpha generation.
    
    Philosophy: 
    - Momentum exists across multiple timeframes
    - Acceleration often precedes reversals
    - Cross-sectional momentum is different from time-series momentum
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Args:
            returns: DataFrame of log returns (date x asset)
        """
        self.returns = returns
        self.features: Dict[str, pd.DataFrame] = {}
        
    def compute_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute all momentum features."""
        if verbose:
            print("Computing momentum features...")
            
        # 1. Cumulative returns at different horizons
        self._cum_returns()
        
        # 2. Momentum spreads (short vs long)
        self._momentum_spreads()
        
        # 3. Acceleration
        self._acceleration()
        
        # 4. Trend strength
        self._trend_strength()
        
        # 5. Time-series momentum signal
        self._tsmom()
        
        # 6. Momentum quality (consistency)
        self._momentum_quality()
        
        if verbose:
            print(f"✅ Created {len(self.features)} momentum features")
            
        return self.features
    
    def _cum_returns(self) -> None:
        """Cumulative returns at various horizons."""
        horizons = [5, 10, 21, 42, 63]  # 1wk, 2wk, 1mo, 2mo, 3mo
        
        for h in horizons:
            self.features[f'ret_{h}d'] = self.returns.rolling(h).sum()
            
    def _momentum_spreads(self) -> None:
        """Momentum = short-term return - long-term return."""
        # Recent vs medium-term
        ret_5 = self.returns.rolling(5).sum()
        ret_21 = self.returns.rolling(21).sum()
        ret_63 = self.returns.rolling(63).sum()
        
        self.features['mom_5_21'] = ret_5 - ret_21
        self.features['mom_5_63'] = ret_5 - ret_63
        self.features['mom_21_63'] = ret_21 - ret_63
        
        # 12-1 momentum (skip most recent month) - classic factor
        ret_252 = self.returns.rolling(252).sum()
        self.features['mom_12_1'] = ret_252.shift(21) - ret_21
        
    def _acceleration(self) -> None:
        """Change in momentum (2nd derivative)."""
        mom_5 = self.returns.rolling(5).sum()
        mom_21 = self.returns.rolling(21).sum()
        
        # Short-term acceleration
        self.features['accel_5d'] = mom_5 - mom_5.shift(5)
        
        # Medium-term acceleration  
        self.features['accel_21d'] = mom_21 - mom_21.shift(21)
        
        # Momentum acceleration (change in momentum spread)
        mom_spread = mom_5 - mom_21
        self.features['mom_accel'] = mom_spread - mom_spread.shift(5)
        
    def _trend_strength(self) -> None:
        """Measure trend strength using directional indicators."""
        r = self.returns
        
        # Fraction of up days (smoothed)
        for window in [10, 21]:
            up_frac = (r > 0).rolling(window).mean()
            self.features[f'up_frac_{window}d'] = (up_frac - 0.5) * 2  # Center at 0
            
        # Consecutive direction days
        sign = np.sign(r)
        self.features['streak_sign'] = sign.rolling(5).sum() / 5
        
    def _tsmom(self) -> None:
        """
        Time-Series Momentum (TSMOM) - Moskowitz et al.
        Signal based on sign of past returns scaled by volatility.
        """
        # Lookback return
        ret_252 = self.returns.rolling(252).sum()
        
        # Volatility scaling
        vol_63 = self.returns.rolling(63).std() * np.sqrt(252)
        
        # TSMOM signal: sign(r) / vol (risk-parity scaling)
        tsmom_raw = np.sign(ret_252) * (ret_252.abs() / (vol_63 + 1e-10))
        self.features['tsmom'] = tsmom_raw.clip(-3, 3)  # Winsorize
        
        # Shorter horizon TSMOM
        ret_63 = self.returns.rolling(63).sum()
        vol_21 = self.returns.rolling(21).std() * np.sqrt(252)
        tsmom_short = np.sign(ret_63) * (ret_63.abs() / (vol_21 + 1e-10))
        self.features['tsmom_short'] = tsmom_short.clip(-3, 3)
        
    def _momentum_quality(self) -> None:
        """
        Momentum quality - consistency of the trend.
        High-quality momentum: steady gains. Low-quality: erratic.
        """
        # Path-dependent momentum quality
        # Ratio of realized return to theoretical max (sum of abs returns)
        ret_21 = self.returns.rolling(21).sum()
        sum_abs_21 = self.returns.abs().rolling(21).sum()
        
        # Momentum quality: how much of potential return was captured
        self.features['mom_quality_21d'] = ret_21 / (sum_abs_21 + 1e-10)
        
        # Information ratio of momentum (mean/std of rolling returns)
        ret_mean = self.returns.rolling(21).mean()
        ret_std = self.returns.rolling(21).std()
        self.features['mom_ir_21d'] = ret_mean / (ret_std + 1e-10)


def compute_momentum_features(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Convenience function to compute all momentum features."""
    mf = MomentumFeatures(returns)
    return mf.compute_all()
