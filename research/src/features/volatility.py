"""
Volatility Features
==================

Volatility-based features capturing risk regimes and vol dynamics.

Features:
1. Realized volatility at multiple horizons
2. Volatility ratios (term structure)
3. Volatility regime indicators
4. GARCH-inspired features
5. Range-based volatility estimators
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class VolatilityFeatures:
    """
    Generate volatility-based features.
    
    Philosophy:
    - Volatility is mean-reverting
    - Vol regime impacts expected returns
    - Range-based estimators are more efficient
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None,
        close: Optional[pd.DataFrame] = None,
        open_: Optional[pd.DataFrame] = None
    ):
        self.returns = returns
        self.high = high
        self.low = low
        self.close = close
        self.open = open_
        self.features: Dict[str, pd.DataFrame] = {}
        
    def compute_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute all volatility features."""
        if verbose:
            print("Computing volatility features...")
            
        # 1. Realized volatility
        self._realized_vol()
        
        # 2. Vol ratios (term structure)
        self._vol_ratios()
        
        # 3. Vol regime percentile
        self._vol_regime()
        
        # 4. Vol-of-vol
        self._vol_of_vol()
        
        # 5. Range-based estimators
        if self.high is not None and self.low is not None:
            self._range_based_vol()
            
        # 6. Volatility surprise
        self._vol_surprise()
        
        # 7. Intraday volatility proxy
        if all(x is not None for x in [self.high, self.low, self.close, self.open]):
            self._intraday_vol()
            
        if verbose:
            print(f"✅ Created {len(self.features)} volatility features")
            
        return self.features
    
    def _realized_vol(self) -> None:
        """Realized volatility at multiple horizons."""
        for window in [5, 10, 21, 42, 63]:
            vol = self.returns.rolling(window).std()
            # Annualize for interpretability
            self.features[f'vol_{window}d'] = vol * np.sqrt(252)
            
    def _vol_ratios(self) -> None:
        """Volatility term structure ratios."""
        vol_5 = self.returns.rolling(5).std()
        vol_21 = self.returns.rolling(21).std()
        vol_63 = self.returns.rolling(63).std()
        
        # Short/long vol ratio - measures vol regime change
        self.features['vol_ratio_5_21'] = vol_5 / (vol_21 + 1e-10)
        self.features['vol_ratio_5_63'] = vol_5 / (vol_63 + 1e-10)
        self.features['vol_ratio_21_63'] = vol_21 / (vol_63 + 1e-10)
        
        # Vol change
        self.features['vol_change_5d'] = vol_5 / (vol_5.shift(5) + 1e-10) - 1
        
    def _vol_regime(self) -> None:
        """Vol regime as percentile rank."""
        vol_21 = self.returns.rolling(21).std()
        
        # Percentile within rolling window
        def rolling_percentile(s, window=252):
            """Compute percentile rank within window."""
            return s.rolling(window).apply(
                lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 20 else 0.5,
                raw=False
            )
        
        self.features['vol_pctl_252d'] = rolling_percentile(vol_21, 252)
        
        # High/low vol regime flag (z-score)
        vol_mean = vol_21.rolling(252).mean()
        vol_std = vol_21.rolling(252).std()
        self.features['vol_zscore'] = (vol_21 - vol_mean) / (vol_std + 1e-10)
        
    def _vol_of_vol(self) -> None:
        """Volatility of volatility - uncertainty in risk."""
        vol_5 = self.returns.rolling(5).std()
        
        # Rolling std of short-term vol
        self.features['vol_of_vol_21d'] = vol_5.rolling(21).std()
        self.features['vol_of_vol_63d'] = vol_5.rolling(63).std()
        
    def _range_based_vol(self) -> None:
        """
        Range-based volatility estimators.
        More efficient than close-to-close vol.
        """
        # Parkinson (1980) - uses high/low
        log_hl = np.log(self.high / self.low)
        parkinson = log_hl ** 2 / (4 * np.log(2))
        
        for window in [5, 21]:
            pk_vol = np.sqrt(parkinson.rolling(window).mean() * 252)
            self.features[f'parkinson_vol_{window}d'] = pk_vol
            
        # Garman-Klass (if open/close available)
        if self.close is not None and self.open is not None:
            log_hl_sq = (np.log(self.high / self.low)) ** 2
            log_co_sq = (np.log(self.close / self.open)) ** 2
            gk = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
            
            for window in [5, 21]:
                gk_vol = np.sqrt(gk.rolling(window).mean() * 252)
                self.features[f'gk_vol_{window}d'] = gk_vol
                
    def _vol_surprise(self) -> None:
        """Volatility surprise: realized vs expected."""
        # Expected vol (trailing estimate)
        vol_21 = self.returns.rolling(21).std()
        vol_expected = vol_21.shift(1).rolling(5).mean()
        
        # Realized vol (most recent)
        vol_realized = self.returns.rolling(5).std()
        
        # Surprise = realized - expected
        self.features['vol_surprise'] = vol_realized - vol_expected
        
        # Normalized surprise
        self.features['vol_surprise_z'] = (vol_realized - vol_expected) / (vol_expected + 1e-10)
        
    def _intraday_vol(self) -> None:
        """
        Intraday volatility proxies using OHLC.
        """
        # Average True Range (ATR) - element-wise max across three components
        tr1 = self.high - self.low
        tr2 = (self.high - self.close.shift(1)).abs()
        tr3 = (self.low - self.close.shift(1)).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        for window in [5, 14, 21]:
            self.features[f'atr_{window}d'] = tr.rolling(window).mean()
            
        # ATR ratio
        self.features['atr_ratio_5_21'] = (
            tr.rolling(5).mean() / (tr.rolling(21).mean() + 1e-10)
        )


def compute_volatility_features(
    returns: pd.DataFrame,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    close: pd.DataFrame = None,
    open_: pd.DataFrame = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function."""
    vf = VolatilityFeatures(returns, high, low, close, open_)
    return vf.compute_all()
