"""
Statistical Arbitrage Features
==============================

Features based on cross-asset relationships and statistical patterns.

Features:
1. Cross-sectional rank/percentile
2. Sector-relative performance (if sectors available)
3. Beta and idiosyncratic return
4. Correlation-based features
5. Lead-lag relationships
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class StatArbFeatures:
    """
    Generate statistical arbitrage features.
    
    Philosophy:
    - Assets don't move in isolation
    - Relative value matters
    - Beta-neutral positions
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        volume: pd.DataFrame = None
    ):
        self.returns = returns
        self.volume = volume
        self.features: Dict[str, pd.DataFrame] = {}
        
    def compute_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute all stat arb features."""
        if verbose:
            print("Computing statistical arbitrage features...")
            
        # 1. Cross-sectional ranks
        self._cross_sectional_ranks()
        
        # 2. Market beta and residuals
        self._beta_features()
        
        # 3. Return vs peers
        self._peer_relative()
        
        # 4. Dispersion features
        self._dispersion()
        
        # 5. Correlation features
        self._correlation_features()
        
        # 6. Co-movement features
        self._comovement()
        
        if verbose:
            print(f"✅ Created {len(self.features)} stat arb features")
            
        return self.features
    
    def _cross_sectional_ranks(self) -> None:
        """Cross-sectional percentile rank of various measures."""
        r = self.returns
        
        # Rank of recent returns
        for window in [5, 21, 63]:
            cumret = r.rolling(window).sum()
            # Rank across assets (0 = worst, 1 = best)
            rank = cumret.rank(axis=1, pct=True)
            self.features[f'cs_rank_ret_{window}d'] = rank - 0.5  # Center at 0
            
        # Rank of volatility
        vol_21 = r.rolling(21).std()
        vol_rank = vol_21.rank(axis=1, pct=True)
        self.features['cs_rank_vol_21d'] = vol_rank - 0.5
        
    def _beta_features(self) -> None:
        """Market beta and idiosyncratic returns."""
        r = self.returns
        
        # Market return (equal-weighted)
        market = r.mean(axis=1)
        market_var = market.rolling(63).var()
        
        # Rolling beta to market
        def rolling_beta(asset_returns, market_returns, window=63):
            """Compute rolling beta."""
            cov = asset_returns.rolling(window).cov(market_returns)
            var = market_returns.rolling(window).var()
            return cov / (var + 1e-10)
        
        betas = pd.DataFrame(index=r.index, columns=r.columns)
        for col in r.columns:
            betas[col] = rolling_beta(r[col], market, 63)
            
        self.features['beta_63d'] = betas
        
        # Beta change (momentum in systematic exposure)
        beta_short = pd.DataFrame(index=r.index, columns=r.columns)
        for col in r.columns:
            beta_short[col] = rolling_beta(r[col], market, 21)
        self.features['beta_change'] = beta_short - betas
        
        # Idiosyncratic return (residual)
        expected_ret = betas * market.values.reshape(-1, 1)
        idio_ret = r - expected_ret
        
        # Idiosyncratic return features
        self.features['idio_ret_1d'] = idio_ret
        self.features['idio_ret_5d'] = idio_ret.rolling(5).sum()
        self.features['idio_ret_21d'] = idio_ret.rolling(21).sum()
        
    def _peer_relative(self) -> None:
        """Performance relative to cross-sectional median."""
        r = self.returns
        
        # Median return across assets
        cs_median = r.median(axis=1)
        
        # Relative performance
        for window in [5, 21]:
            cumret = r.rolling(window).sum()
            cs_med_cumret = cs_median.rolling(window).sum()
            
            relative = cumret.sub(cs_med_cumret, axis=0)
            self.features[f'rel_to_median_{window}d'] = relative
            
    def _dispersion(self) -> None:
        """Cross-sectional dispersion features."""
        r = self.returns
        
        # Return dispersion (cross-sectional std)
        dispersion = r.std(axis=1)
        
        # Dispersion regime
        disp_mean = dispersion.rolling(63).mean()
        disp_std = dispersion.rolling(63).std()
        self.features['dispersion_zscore'] = (
            (dispersion - disp_mean) / (disp_std + 1e-10)
        ).values.reshape(-1, 1) * np.ones((1, r.shape[1]))
        self.features['dispersion_zscore'] = pd.DataFrame(
            self.features['dispersion_zscore'],
            index=r.index,
            columns=r.columns
        )
        
        # Correlation dispersion (average pairwise correlation)
        # Rolling correlation of each asset with market
        market = r.mean(axis=1)
        corr_with_mkt = r.rolling(21).apply(
            lambda x: x.corr(market.loc[x.index]) if len(x) > 5 else np.nan,
            raw=False
        )
        self.features['corr_with_market'] = corr_with_mkt
        
    def _correlation_features(self) -> None:
        """Correlation-based features."""
        r = self.returns
        market = r.mean(axis=1)
        
        # Rolling correlation with market
        corrs = pd.DataFrame(index=r.index, columns=r.columns)
        for col in r.columns:
            corrs[col] = r[col].rolling(42).corr(market)
            
        self.features['mkt_corr_42d'] = corrs
        
        # Correlation change
        corrs_short = pd.DataFrame(index=r.index, columns=r.columns)
        for col in r.columns:
            corrs_short[col] = r[col].rolling(21).corr(market)
            
        self.features['corr_change'] = corrs_short - corrs
        
    def _comovement(self) -> None:
        """Co-movement features - do assets move together?"""
        r = self.returns
        
        # Fraction of assets moving same direction
        up_pct = (r > 0).mean(axis=1)
        down_pct = (r < 0).mean(axis=1)
        
        # Extreme co-movement days
        extreme_up = (up_pct > 0.8).astype(float)
        extreme_down = (down_pct > 0.8).astype(float)
        
        # Rolling count of extreme days
        self.features['extreme_comovement_up'] = pd.DataFrame(
            extreme_up.rolling(21).sum().values.reshape(-1, 1) * np.ones((1, r.shape[1])),
            index=r.index, columns=r.columns
        )
        self.features['extreme_comovement_down'] = pd.DataFrame(
            extreme_down.rolling(21).sum().values.reshape(-1, 1) * np.ones((1, r.shape[1])),
            index=r.index, columns=r.columns
        )


def compute_stat_arb_features(
    returns: pd.DataFrame,
    volume: pd.DataFrame = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function."""
    saf = StatArbFeatures(returns, volume)
    return saf.compute_all()
