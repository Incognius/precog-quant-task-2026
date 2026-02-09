"""
Long-Only Feature Engineering
==============================

Features designed to identify stocks that will OUTPERFORM:
- Momentum features (trend continuation)
- Quality features (fundamental strength)
- Value features (undervaluation)
- Growth features (earnings acceleration)

All features are:
1. Computed using ONLY past data (no look-ahead)
2. Cross-sectionally z-scored (daily)
3. Winsorized to [-3, +3]

Author: Precog Quant Research
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class LongFeatureEngineer:
    """
    Feature engineering for long-only strategies.
    
    Usage:
        engineer = LongFeatureEngineer(df)
        features = engineer.compute_all_features()
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with panel data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date'])
        
        # Compute returns if not present
        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df.groupby('ticker')['Close'].pct_change()
        
        self.feature_cols = []
        
    def _zscore_cross_sectional(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Cross-sectionally z-score and winsorize."""
        zscore = df.groupby('date')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        return zscore.clip(-3, 3)
    
    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================
    
    def compute_momentum_features(self) -> pd.DataFrame:
        """
        Compute momentum-based features.
        
        Features:
        - mom_5d, mom_10d, mom_21d, mom_63d: Simple momentum (cumret)
        - mom_vol_adj: Volatility-adjusted momentum (risk-adjusted)
        - mom_persistence: How consistent is the trend?
        - mom_vs_sector: Momentum relative to cross-sectional mean
        """
        df = self.df.copy()
        
        # Simple momentum at different horizons
        for window in [5, 10, 21, 63]:
            df[f'mom_{window}d'] = df.groupby('ticker')['returns'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).sum().shift(1)
            )
            self.feature_cols.append(f'mom_{window}d_zscore')
        
        # Volatility-adjusted momentum (Sharpe-like)
        for window in [21, 63]:
            cumret = df.groupby('ticker')['returns'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).sum()
            )
            vol = df.groupby('ticker')['returns'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).std()
            )
            df[f'mom_vol_adj_{window}d'] = (cumret / (vol * np.sqrt(window) + 1e-10)).shift(1)
            self.feature_cols.append(f'mom_vol_adj_{window}d_zscore')
        
        # Momentum persistence (consistency of trend)
        def trend_consistency(returns):
            if len(returns) < 10:
                return np.nan
            trend_sign = np.sign(returns.sum())
            return (np.sign(returns) == trend_sign).mean()
        
        df['mom_persistence_21d'] = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(21, min_periods=15).apply(trend_consistency, raw=False).shift(1)
        )
        self.feature_cols.append('mom_persistence_21d_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('mom_') and not c.endswith('_zscore')]:
            df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================
    
    def compute_volatility_features(self) -> pd.DataFrame:
        """
        Volatility-based features.
        
        Features:
        - realized_vol: Historical volatility
        - vol_of_vol: Volatility clustering
        - vol_ratio: Short-term vs long-term vol ratio
        - vol_skew: Asymmetry in volatility
        """
        df = self.df.copy()
        
        # Realized volatility at different horizons
        for window in [10, 21, 63]:
            df[f'realized_vol_{window}d'] = df.groupby('ticker')['returns'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).std().shift(1)
            )
            self.feature_cols.append(f'realized_vol_{window}d_zscore')
        
        # Vol ratio (short/long) - mean reversion signal
        df['vol_ratio_10_63'] = df['realized_vol_10d'] / (df['realized_vol_63d'] + 1e-10)
        self.feature_cols.append('vol_ratio_10_63_zscore')
        
        # Vol of vol (volatility clustering)
        df['vol_of_vol'] = df.groupby('ticker')['realized_vol_21d'].transform(
            lambda x: x.rolling(21, min_periods=15).std().shift(1)
        )
        self.feature_cols.append('vol_of_vol_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('realized_vol_') or c.startswith('vol_')]:
            if not col.endswith('_zscore'):
                df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # PRICE PATTERN FEATURES
    # =========================================================================
    
    def compute_price_pattern_features(self) -> pd.DataFrame:
        """
        Price pattern features from OHLC data.
        
        Features:
        - intraday_range: High-Low relative to Close
        - upper_shadow: Upper wick length
        - lower_shadow: Lower wick length
        - body_ratio: Body vs full range
        """
        df = self.df.copy()
        
        # Intraday volatility
        df['hl_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
        df['intraday_range_5d'] = df.groupby('ticker')['hl_range'].transform(
            lambda x: x.rolling(5, min_periods=3).mean().shift(1)
        )
        self.feature_cols.append('intraday_range_5d_zscore')
        
        # Upper shadow (bearish signal when large)
        df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + 1e-10)
        df['upper_shadow_5d'] = df.groupby('ticker')['upper_shadow'].transform(
            lambda x: x.rolling(5, min_periods=3).mean().shift(1)
        )
        self.feature_cols.append('upper_shadow_5d_zscore')
        
        # Lower shadow (bullish signal when large)
        df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['lower_shadow_5d'] = df.groupby('ticker')['lower_shadow'].transform(
            lambda x: x.rolling(5, min_periods=3).mean().shift(1)
        )
        self.feature_cols.append('lower_shadow_5d_zscore')
        
        # Body/Range ratio
        df['body_ratio'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['body_ratio_5d'] = df.groupby('ticker')['body_ratio'].transform(
            lambda x: x.rolling(5, min_periods=3).mean().shift(1)
        )
        self.feature_cols.append('body_ratio_5d_zscore')
        
        # Cross-sectionally z-score
        for col in ['intraday_range_5d', 'upper_shadow_5d', 'lower_shadow_5d', 'body_ratio_5d']:
            df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    
    def compute_volume_features(self) -> pd.DataFrame:
        """
        Volume-based features.
        
        Features:
        - volume_ma_ratio: Current volume vs moving average
        - volume_trend: Volume momentum
        - price_volume_corr: Correlation between price and volume changes
        """
        df = self.df.copy()
        
        # Volume MA ratio
        vol_ma = df.groupby('ticker')['Volume'].transform(
            lambda x: x.rolling(21, min_periods=15).mean()
        )
        df['volume_ma_ratio'] = (df['Volume'] / (vol_ma + 1e-10)).shift(1)
        self.feature_cols.append('volume_ma_ratio_zscore')
        
        # Volume trend
        df['volume_pct_change'] = df.groupby('ticker')['Volume'].pct_change()
        df['volume_trend_5d'] = df.groupby('ticker')['volume_pct_change'].transform(
            lambda x: x.rolling(5, min_periods=3).mean().shift(1)
        )
        self.feature_cols.append('volume_trend_5d_zscore')
        
        # Price-volume correlation (should be positive in healthy trends)
        def rolling_corr(group):
            return group['returns'].rolling(21, min_periods=15).corr(group['volume_pct_change']).shift(1)
        
        df['price_volume_corr'] = df.groupby('ticker').apply(rolling_corr).reset_index(level=0, drop=True)
        self.feature_cols.append('price_volume_corr_zscore')
        
        # Cross-sectionally z-score
        for col in ['volume_ma_ratio', 'volume_trend_5d', 'price_volume_corr']:
            df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # MEAN REVERSION FEATURES
    # =========================================================================
    
    def compute_mean_reversion_features(self) -> pd.DataFrame:
        """
        Mean reversion features.
        
        Features:
        - distance_from_ma: How far from moving average
        - rsi: Relative Strength Index
        - bb_position: Bollinger Band position
        """
        df = self.df.copy()
        
        # Distance from MA (mean reversion signal)
        for window in [10, 21, 50]:
            ma = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).mean()
            )
            df[f'distance_from_ma_{window}d'] = ((df['Close'] / (ma + 1e-10)) - 1).shift(1)
            self.feature_cols.append(f'distance_from_ma_{window}d_zscore')
        
        # RSI
        def compute_rsi(returns, window=14):
            gains = returns.clip(lower=0)
            losses = (-returns).clip(lower=0)
            avg_gain = gains.rolling(window, min_periods=window).mean()
            avg_loss = losses.rolling(window, min_periods=window).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14d'] = df.groupby('ticker')['returns'].transform(
            lambda x: compute_rsi(x, 14).shift(1)
        )
        self.feature_cols.append('rsi_14d_zscore')
        
        # Bollinger Band position
        for window in [21]:
            ma = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).mean()
            )
            std = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).std()
            )
            df[f'bb_position_{window}d'] = ((df['Close'] - ma) / (2 * std + 1e-10)).shift(1)
            self.feature_cols.append(f'bb_position_{window}d_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('distance_from_ma_') or c.startswith('rsi_') or c.startswith('bb_position_')]:
            if not col.endswith('_zscore'):
                df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # COMPUTE ALL FEATURES
    # =========================================================================
    
    def compute_all_features(self) -> pd.DataFrame:
        """
        Compute all long-only features.
        
        Returns
        -------
        pd.DataFrame with all features computed
        """
        # Reset feature cols
        self.feature_cols = []
        
        # Compute each feature family
        df = self.compute_momentum_features()
        self.df = df
        
        df = self.compute_volatility_features()
        self.df = df
        
        df = self.compute_price_pattern_features()
        self.df = df
        
        df = self.compute_volume_features()
        self.df = df
        
        df = self.compute_mean_reversion_features()
        self.df = df
        
        # Remove duplicates from feature_cols
        self.feature_cols = list(set(self.feature_cols))
        
        print(f"✅ Computed {len(self.feature_cols)} long features")
        return self.df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of computed feature columns."""
        return [c for c in self.feature_cols if c in self.df.columns]
    
    def save_features(self, output_path: str):
        """Save features to parquet file."""
        # Keep only essential columns + features
        cols_to_keep = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'returns']
        cols_to_keep += self.get_feature_columns()
        cols_to_keep = [c for c in cols_to_keep if c in self.df.columns]
        
        output_df = self.df[cols_to_keep].copy()
        output_df.to_parquet(output_path, index=False)
        print(f"✅ Saved features to {output_path}")
