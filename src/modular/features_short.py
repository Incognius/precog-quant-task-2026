"""
Short-Only Feature Engineering
===============================

Features designed to identify stocks that will UNDERPERFORM:
- Overextension features (too far from mean)
- Momentum exhaustion (trend weakening)
- Distribution patterns (smart money selling)
- Fundamental deterioration (quality declining)
- Liquidity warning (volume drying up)

WHY SHORTING IS DIFFERENT:
- Short squeezes are asymmetric risk
- Borrowing costs matter
- Timing is more critical
- Mean reversion is key

All features are:
1. Computed using ONLY past data (no look-ahead)
2. Cross-sectionally z-scored (daily)
3. Winsorized to [-3, +3]
4. Higher values = MORE shortable

Author: Precog Quant Research
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ShortFeatureEngineer:
    """
    Feature engineering for short-only strategies.
    
    IMPORTANT: All features are oriented so that HIGHER = MORE SHORTABLE
    This allows consistent signal interpretation (high signal = take position)
    
    Usage:
        engineer = ShortFeatureEngineer(df)
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
    # OVEREXTENSION FEATURES (Stocks stretched too far from fair value)
    # =========================================================================
    
    def compute_overextension_features(self) -> pd.DataFrame:
        """
        Identify stocks that are overextended above their trend.
        
        Features:
        - overext_vs_ma: How far above moving average (SHORT signal)
        - overext_vs_ath: Distance from all-time high (high = near ATH = risky)
        - overext_bb_upper: How far above upper Bollinger Band
        - overext_channel: Position within price channel
        """
        df = self.df.copy()
        
        # Distance above moving averages (positive = overextended = SHORT)
        for window in [21, 50, 100]:
            ma = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).mean()
            )
            # Higher = more overextended = more shortable
            df[f'short_overext_ma_{window}d'] = ((df['Close'] / (ma + 1e-10)) - 1).shift(1)
            self.feature_cols.append(f'short_overext_ma_{window}d_zscore')
        
        # Distance from 52-week high (closer to high = more shortable due to profit-taking)
        def dist_from_high(prices, window=252):
            rolling_high = prices.rolling(window, min_periods=int(window*0.7)).max()
            return prices / (rolling_high + 1e-10)  # 1 = at high, <1 = below
        
        df['short_near_52w_high'] = df.groupby('ticker')['Close'].transform(
            lambda x: dist_from_high(x, 252).shift(1)
        )
        self.feature_cols.append('short_near_52w_high_zscore')
        
        # Bollinger Band overextension (above upper band)
        for window in [21]:
            ma = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).mean()
            )
            std = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).std()
            )
            upper_band = ma + 2 * std
            # Positive when above upper band = SHORT signal
            df[f'short_above_bb_{window}d'] = ((df['Close'] - upper_band) / (std + 1e-10)).shift(1)
            self.feature_cols.append(f'short_above_bb_{window}d_zscore')
        
        # Price channel position (higher = near top of channel)
        for window in [21, 63]:
            high = df.groupby('ticker')['High'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).max()
            )
            low = df.groupby('ticker')['Low'].transform(
                lambda x: x.rolling(window, min_periods=int(window*0.7)).min()
            )
            # 1 = at high, 0 = at low
            df[f'short_channel_pos_{window}d'] = ((df['Close'] - low) / (high - low + 1e-10)).shift(1)
            self.feature_cols.append(f'short_channel_pos_{window}d_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('short_') and not c.endswith('_zscore')]:
            df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # MOMENTUM EXHAUSTION FEATURES (Trend losing steam)
    # =========================================================================
    
    def compute_exhaustion_features(self) -> pd.DataFrame:
        """
        Identify stocks where momentum is exhausting.
        
        Features:
        - mom_decel: Momentum deceleration (slowing down)
        - mom_divergence: Price making new highs but momentum not
        - rsi_overbought: RSI > 70 (overbought = SHORT signal)
        - volume_exhaustion: Declining volume on up moves
        """
        df = self.df.copy()
        
        # Momentum deceleration (short-term mom weaker than long-term)
        mom_5d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(5, min_periods=3).sum()
        )
        mom_21d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(21, min_periods=15).sum()
        )
        mom_63d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(63, min_periods=45).sum()
        )
        
        # Deceleration = short-term trailing long-term (for up-trending stocks)
        # Higher = more deceleration = SHORT
        df['short_mom_decel_5v21'] = (mom_21d / 4 - mom_5d).shift(1)  # Normalized
        df['short_mom_decel_21v63'] = (mom_63d / 3 - mom_21d).shift(1)
        self.feature_cols.append('short_mom_decel_5v21_zscore')
        self.feature_cols.append('short_mom_decel_21v63_zscore')
        
        # RSI overbought
        def compute_rsi(returns, window=14):
            gains = returns.clip(lower=0)
            losses = (-returns).clip(lower=0)
            avg_gain = gains.rolling(window, min_periods=window).mean()
            avg_loss = losses.rolling(window, min_periods=window).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14d'] = df.groupby('ticker')['returns'].transform(
            lambda x: compute_rsi(x, 14)
        )
        # Higher RSI = more overbought = SHORT
        df['short_rsi_overbought'] = df['rsi_14d'].shift(1)
        self.feature_cols.append('short_rsi_overbought_zscore')
        
        # Volume declining on up moves (exhaustion)
        df['volume_pct'] = df.groupby('ticker')['Volume'].pct_change()
        df['up_day'] = (df['returns'] > 0).astype(int)
        
        # Average volume change on up days (negative = exhaustion on up moves)
        def vol_on_up_days(group):
            vol_changes = group['volume_pct'].values
            up_days = group['up_day'].values
            result = pd.Series(index=group.index, dtype=float)
            for i in range(21, len(group)):
                window_vols = vol_changes[i-21:i]
                window_ups = up_days[i-21:i]
                up_vol = window_vols[window_ups == 1].mean() if window_ups.sum() > 0 else 0
                result.iloc[i] = up_vol
            return result
        
        df['short_volume_exhaust'] = df.groupby('ticker').apply(vol_on_up_days).reset_index(level=0, drop=True)
        df['short_volume_exhaust'] = -df['short_volume_exhaust'].shift(1)  # Flip: negative vol change = SHORT
        self.feature_cols.append('short_volume_exhaust_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('short_') and not c.endswith('_zscore')]:
            if f'{col}_zscore' not in df.columns:
                df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # DISTRIBUTION FEATURES (Smart money exiting)
    # =========================================================================
    
    def compute_distribution_features(self) -> pd.DataFrame:
        """
        Identify distribution patterns (selling into strength).
        
        Features:
        - accumulation_dist: A/D line divergence
        - obv_divergence: OBV not confirming price
        - volume_price_divergence: Price up, volume down
        - closing_weakness: Closing near lows despite up days
        """
        df = self.df.copy()
        
        # Money Flow Multiplier and A/D
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        mf_volume = clv * df['Volume']
        
        # A/D line (cumulative)
        df['ad_line'] = df.groupby('ticker').apply(
            lambda g: mf_volume.loc[g.index].cumsum()
        ).reset_index(level=0, drop=True)
        
        # A/D line rate of change (negative = distribution = SHORT)
        df['ad_roc'] = df.groupby('ticker')['ad_line'].pct_change(periods=21)
        price_roc = df.groupby('ticker')['Close'].pct_change(periods=21)
        
        # Divergence: price up but A/D down
        df['short_ad_divergence'] = (price_roc - df['ad_roc']).shift(1)
        self.feature_cols.append('short_ad_divergence_zscore')
        
        # Closing weakness: closing near lows on up days
        close_position = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        df['short_closing_weakness'] = df.groupby('ticker').apply(
            lambda g: (1 - close_position.loc[g.index]).rolling(10, min_periods=7).mean().shift(1)
        ).reset_index(level=0, drop=True)
        self.feature_cols.append('short_closing_weakness_zscore')
        
        # OBV divergence (simplified)
        df['obv_sign'] = np.sign(df['returns'])
        df['obv'] = df.groupby('ticker').apply(
            lambda g: (g['obv_sign'] * g['Volume']).cumsum()
        ).reset_index(level=0, drop=True)
        
        obv_roc = df.groupby('ticker')['obv'].pct_change(periods=21)
        df['short_obv_divergence'] = (price_roc - obv_roc).shift(1)
        self.feature_cols.append('short_obv_divergence_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('short_') and not c.endswith('_zscore')]:
            if f'{col}_zscore' not in df.columns:
                df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # VOLATILITY REGIME FEATURES (High vol = risky for shorts)
    # =========================================================================
    
    def compute_volatility_features(self) -> pd.DataFrame:
        """
        Volatility features for shorting.
        
        NOTE: High volatility stocks are RISKIER to short due to squeeze risk.
        These features help avoid or size short positions appropriately.
        
        Features:
        - vol_regime: Current vs historical vol
        - vol_expansion: Vol increasing (risky for shorts)
        - gap_risk: Large gap frequency
        """
        df = self.df.copy()
        
        # Volatility regime (high vol = dangerous for shorts)
        vol_21d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(21, min_periods=15).std()
        )
        vol_252d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(252, min_periods=200).std()
        )
        # Ratio > 1 = vol expanding = risky
        df['short_vol_regime'] = (vol_21d / (vol_252d + 1e-10)).shift(1)
        self.feature_cols.append('short_vol_regime_zscore')
        
        # Vol expansion rate
        df['short_vol_expansion'] = df.groupby('ticker')['short_vol_regime'].transform(
            lambda x: x.pct_change(periods=5).shift(1)
        )
        self.feature_cols.append('short_vol_expansion_zscore')
        
        # Gap risk (large overnight gaps are dangerous for shorts)
        df['overnight_gap'] = (df['Open'] / df.groupby('ticker')['Close'].shift(1) - 1).abs()
        df['short_gap_risk'] = df.groupby('ticker')['overnight_gap'].transform(
            lambda x: x.rolling(21, min_periods=15).mean().shift(1)
        )
        self.feature_cols.append('short_gap_risk_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('short_') and not c.endswith('_zscore')]:
            if f'{col}_zscore' not in df.columns:
                df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # SHORT SQUEEZE RISK FEATURES
    # =========================================================================
    
    def compute_squeeze_risk_features(self) -> pd.DataFrame:
        """
        Features to identify short squeeze risk.
        
        High squeeze risk = AVOID shorting
        
        Features:
        - recent_drawdown: Stocks that just crashed may squeeze
        - volume_spike: Volume spikes can trigger squeezes
        - reversal_momentum: Stocks showing reversal signs
        """
        df = self.df.copy()
        
        # Recent drawdown (stocks that crashed may squeeze)
        cumret_21d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(21, min_periods=15).sum()
        )
        # Very negative recent returns = squeeze risk = AVOID shorting
        # Flip sign: low value = safe to short
        df['short_squeeze_risk_dd'] = (-cumret_21d).shift(1)  # More negative = higher risk
        self.feature_cols.append('short_squeeze_risk_dd_zscore')
        
        # Volume spike (unusual volume = potential squeeze)
        vol_ma21 = df.groupby('ticker')['Volume'].transform(
            lambda x: x.rolling(21, min_periods=15).mean()
        )
        vol_ratio = df['Volume'] / (vol_ma21 + 1e-10)
        df['short_squeeze_risk_vol'] = df.groupby('ticker').apply(
            lambda g: vol_ratio.loc[g.index].rolling(5, min_periods=3).max().shift(1)
        ).reset_index(level=0, drop=True)
        self.feature_cols.append('short_squeeze_risk_vol_zscore')
        
        # 5-day reversal (stocks bouncing = squeeze risk)
        mom_5d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(5, min_periods=3).sum()
        )
        mom_21d = df.groupby('ticker')['returns'].transform(
            lambda x: x.rolling(21, min_periods=15).sum()
        )
        # Positive 5d after negative 21d = reversal = squeeze risk
        df['short_squeeze_risk_reversal'] = np.where(
            mom_21d < 0,
            mom_5d,  # Positive 5d in downtrend = bounce risk
            0
        )
        df['short_squeeze_risk_reversal'] = df['short_squeeze_risk_reversal'].shift(1)
        self.feature_cols.append('short_squeeze_risk_reversal_zscore')
        
        # Cross-sectionally z-score
        for col in [c for c in df.columns if c.startswith('short_') and not c.endswith('_zscore')]:
            if f'{col}_zscore' not in df.columns:
                df[f'{col}_zscore'] = self._zscore_cross_sectional(df, col)
        
        return df
    
    # =========================================================================
    # COMPUTE ALL FEATURES
    # =========================================================================
    
    def compute_all_features(self) -> pd.DataFrame:
        """
        Compute all short-only features.
        
        Returns
        -------
        pd.DataFrame with all features computed
        """
        # Reset feature cols
        self.feature_cols = []
        
        # Compute each feature family
        print("Computing overextension features...")
        df = self.compute_overextension_features()
        self.df = df
        
        print("Computing exhaustion features...")
        df = self.compute_exhaustion_features()
        self.df = df
        
        print("Computing distribution features...")
        df = self.compute_distribution_features()
        self.df = df
        
        print("Computing volatility features...")
        df = self.compute_volatility_features()
        self.df = df
        
        print("Computing squeeze risk features...")
        df = self.compute_squeeze_risk_features()
        self.df = df
        
        # Remove duplicates from feature_cols
        self.feature_cols = list(set(self.feature_cols))
        
        print(f"✅ Computed {len(self.feature_cols)} short features")
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
    
    def get_composite_short_signal(self) -> pd.Series:
        """
        Create a composite short signal from all features.
        
        Higher = more shortable
        """
        feature_cols = self.get_feature_columns()
        
        # Separate into signal features and risk features
        signal_features = [c for c in feature_cols if 'squeeze_risk' not in c and 'vol_regime' not in c]
        risk_features = [c for c in feature_cols if 'squeeze_risk' in c or 'vol_regime' in c]
        
        # Composite signal = mean of signal features
        signal = self.df[signal_features].mean(axis=1)
        
        # Risk adjustment = penalize high squeeze risk
        risk = self.df[risk_features].mean(axis=1)
        
        # Final signal = signal - risk_penalty
        adjusted_signal = signal - 0.3 * risk
        
        return adjusted_signal
