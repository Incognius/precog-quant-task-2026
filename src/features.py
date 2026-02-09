"""
Feature Engineering Library for Quantitative Trading
=====================================================

This module provides systematic feature engineering functions for:
- Trend Analysis (strength, consistency, breadth)
- Regime Detection (volatility, trend, combined)
- Momentum Features (multi-timeframe)
- Cross-Sectional Features (rank-based)
- Feature Interactions

These features are designed to be:
1. Cross-sectionally standardized (mean=0, std=1 across assets)
2. Forward-looking safe (all calculations use lagged data)
3. GPU-acceleratable where possible

Author: Precog Quant Task
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TREND FEATURE BLOCK
# =============================================================================

def compute_trend_block(returns: pd.DataFrame, 
                        lookback_short: int = 20,
                        lookback_medium: int = 60,
                        lookback_long: int = 120) -> Dict[str, pd.DataFrame]:
    """
    Compute trend-related features at asset and market level.
    
    Features:
    ---------
    trend_strength: Volatility-normalized cumulative return over lookback_medium days.
                   Measures how strong the trend is relative to noise.
                   Formula: cumret_{60d} / (vol_{60d} * sqrt(60))
                   
    trend_consistency: Hit-ratio of daily returns in trend direction.
                      Measures how "clean" the trend is (high = smooth trend).
                      Formula: count(sign(ret) == sign(trend)) / N
                      
    trend_breadth: Cross-sectional participation in the trend.
                  Measures what % of assets are trending in same direction.
                  Formula: count(cumret_{20d} > 0) / N_assets
                  
    trend_acceleration: Second derivative - is trend accelerating or decelerating?
                       Formula: mom_{20d} - mom_{60d}
                       
    trend_divergence: Asset's trend vs market trend.
                     Measures idiosyncratic trend component.
                     Formula: asset_trend - market_trend
    
    All features are lagged by 1 day to prevent lookahead bias.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns (index=dates, columns=assets)
    lookback_short, lookback_medium, lookback_long : int
        Lookback windows for trend calculations
        
    Returns:
    --------
    Dict[str, pd.DataFrame]: Dictionary of feature DataFrames
    """
    features = {}
    
    # ----- Asset-Level Trend Features -----
    
    # Cumulative returns at different horizons
    cumret_short = returns.rolling(window=lookback_short, min_periods=int(lookback_short * 0.7)).sum()
    cumret_medium = returns.rolling(window=lookback_medium, min_periods=int(lookback_medium * 0.7)).sum()
    cumret_long = returns.rolling(window=lookback_long, min_periods=int(lookback_long * 0.7)).sum()
    
    # Volatility at different horizons (annualized)
    vol_short = returns.rolling(window=lookback_short, min_periods=int(lookback_short * 0.7)).std()
    vol_medium = returns.rolling(window=lookback_medium, min_periods=int(lookback_medium * 0.7)).std()
    vol_long = returns.rolling(window=lookback_long, min_periods=int(lookback_long * 0.7)).std()
    
    # Trend Strength: vol-normalized cumulative return (t-stat of trend)
    features['trend_strength'] = (cumret_medium / (vol_medium * np.sqrt(lookback_medium) + 1e-10)).shift(1)
    features['trend_strength_short'] = (cumret_short / (vol_short * np.sqrt(lookback_short) + 1e-10)).shift(1)
    features['trend_strength_long'] = (cumret_long / (vol_long * np.sqrt(lookback_long) + 1e-10)).shift(1)
    
    # Trend Acceleration: change in momentum (second derivative)
    features['trend_acceleration'] = (cumret_short - cumret_medium / (lookback_medium / lookback_short)).shift(1)
    
    # ----- Market-Level Trend Features (broadcast to all assets) -----
    
    market_ret = returns.mean(axis=1)
    
    # Market trend strength
    market_cumret = market_ret.rolling(window=lookback_medium, min_periods=int(lookback_medium * 0.7)).sum()
    market_vol = market_ret.rolling(window=lookback_medium, min_periods=int(lookback_medium * 0.7)).std()
    market_trend_strength = market_cumret / (market_vol * np.sqrt(lookback_medium) + 1e-10)
    
    # Trend Consistency: fraction of days matching trend direction
    trend_sign = np.sign(market_trend_strength)
    pos_days = (market_ret > 0).rolling(window=lookback_medium, min_periods=int(lookback_medium * 0.7)).sum()
    neg_days = (market_ret < 0).rolling(window=lookback_medium, min_periods=int(lookback_medium * 0.7)).sum()
    
    # Consistency = fraction of days in trend direction
    trend_consistency = pd.Series(index=returns.index, dtype=float)
    trend_consistency[trend_sign > 0] = pos_days[trend_sign > 0] / lookback_medium
    trend_consistency[trend_sign < 0] = neg_days[trend_sign < 0] / lookback_medium
    trend_consistency[trend_sign == 0] = 0.5
    
    # Trend Breadth: % of assets with positive cumulative return
    trend_breadth = (cumret_short > 0).mean(axis=1)
    
    # Broadcast market features to panel format
    n_assets = returns.shape[1]
    features['trend_consistency'] = pd.DataFrame(
        np.tile(trend_consistency.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    features['trend_breadth'] = pd.DataFrame(
        np.tile(trend_breadth.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    features['market_trend_strength'] = pd.DataFrame(
        np.tile(market_trend_strength.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    # Trend Divergence: Asset trend vs Market trend
    features['trend_divergence'] = features['trend_strength'] - features['market_trend_strength']
    
    return features


def compute_regime_indicators(returns: pd.DataFrame,
                               vol_lookback: int = 20,
                               trend_lookback: int = 60,
                               percentile_threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
    """
    Compute market regime indicators for conditioning/filtering.
    
    Regimes:
    --------
    vol_regime: Binary indicator for high/low volatility environment.
               High vol = vol > median(historical vol)
               Values: 1 (high vol), 0 (low vol)
               
    trend_regime: Binary indicator for up/down trend.
                 Up = 60d cumret > 0
                 Values: 1 (uptrend), 0 (downtrend)
                 
    combined_regime: 4-state regime combining vol and trend.
                    Values: 0 (down+low), 1 (down+high), 2 (up+low), 3 (up+high)
                    
    regime_confidence: How "pure" is the regime (0-1)?
                      High values = clear regime, low = uncertain
                      
    vol_percentile: Rolling percentile of current volatility.
                   Values: [0, 1] where 1 = highest historical vol
                   
    trend_percentile: Rolling percentile of current trend.
                     Values: [0, 1] where 1 = strongest uptrend
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns (index=dates, columns=assets)
    vol_lookback : int
        Window for volatility calculation
    trend_lookback : int
        Window for trend calculation
    percentile_threshold : float
        Threshold for high/low regime classification
        
    Returns:
    --------
    Dict[str, pd.DataFrame]: Dictionary of regime indicators
    """
    indicators = {}
    
    # Market returns
    market_ret = returns.mean(axis=1)
    n_assets = returns.shape[1]
    
    # ----- Volatility Regime -----
    
    vol = market_ret.rolling(window=vol_lookback, min_periods=int(vol_lookback * 0.7)).std() * np.sqrt(252)
    
    # Expanding median for regime classification (avoid lookahead)
    vol_median = vol.expanding(min_periods=252).median()
    high_vol = (vol > vol_median).astype(float)
    
    # Volatility percentile (expanding)
    vol_percentile = vol.expanding(min_periods=252).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5
    )
    
    indicators['vol_regime'] = pd.DataFrame(
        np.tile(high_vol.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    indicators['vol_percentile'] = pd.DataFrame(
        np.tile(vol_percentile.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    indicators['vol_level'] = pd.DataFrame(
        np.tile(vol.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    # ----- Trend Regime -----
    
    cumret = market_ret.rolling(window=trend_lookback, min_periods=int(trend_lookback * 0.7)).sum()
    uptrend = (cumret > 0).astype(float)
    
    # Trend percentile (expanding)
    trend_percentile = cumret.expanding(min_periods=252).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5
    )
    
    indicators['trend_regime'] = pd.DataFrame(
        np.tile(uptrend.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    indicators['trend_percentile'] = pd.DataFrame(
        np.tile(trend_percentile.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    # ----- Combined Regime -----
    
    # 4-state: 0=down+low, 1=down+high, 2=up+low, 3=up+high
    combined = uptrend * 2 + high_vol
    indicators['combined_regime'] = pd.DataFrame(
        np.tile(combined.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    # ----- Regime Confidence -----
    
    # How far from the regime boundary (0.5 percentile)?
    vol_confidence = np.abs(vol_percentile - 0.5) * 2  # 0 at boundary, 1 at extremes
    trend_confidence = np.abs(trend_percentile - 0.5) * 2
    regime_confidence = (vol_confidence + trend_confidence) / 2
    
    indicators['regime_confidence'] = pd.DataFrame(
        np.tile(regime_confidence.values.reshape(-1, 1), (1, n_assets)),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    return indicators


# =============================================================================
# ADVANCED MOMENTUM BLOCK
# =============================================================================

def compute_advanced_momentum_block(returns: pd.DataFrame,
                                    panel: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
    """
    Advanced momentum features beyond simple cumulative returns.
    
    Features:
    ---------
    mom_risk_adjusted_{horizon}: Momentum normalized by own volatility.
                                 Sharpe-ratio style momentum signal.
                                 
    mom_relative_{horizon}: Cross-sectional rank of momentum.
                           Measures how stock ranks vs peers.
                           
    mom_change: Momentum of momentum (acceleration).
               Second derivative of price.
               
    momentum_quality: Combination of strength and consistency.
                     High = strong AND consistent momentum.
                     
    momentum_dispersion: Cross-sectional dispersion of momentum.
                        High = divergent opinions (market uncertainty).
    """
    features = {}
    
    horizons = [3, 5, 10, 20, 60]
    
    for h in horizons:
        # Raw momentum
        mom = returns.rolling(window=h, min_periods=int(h * 0.7)).sum()
        
        # Risk-adjusted momentum (Sharpe-ratio style)
        vol = returns.rolling(window=max(h, 20), min_periods=int(max(h, 20) * 0.7)).std()
        features[f'mom_risk_adj_{h}d'] = (mom / (vol * np.sqrt(h) + 1e-10)).shift(1)
        
        # Cross-sectional rank (0 to 1)
        features[f'mom_rank_{h}d'] = mom.rank(axis=1, pct=True).shift(1)
    
    # Momentum change (acceleration) - 5d vs 20d
    mom_5d = returns.rolling(5).sum()
    mom_20d = returns.rolling(20).sum()
    features['mom_change'] = (mom_5d - mom_20d / 4).shift(1)  # Normalized for horizon
    
    # Momentum quality: strength * consistency
    trend_strength = features['mom_risk_adj_20d']
    
    # Consistency: fraction of days with positive returns in momentum direction
    daily_sign = np.sign(returns)
    mom_sign = np.sign(mom_20d)
    agreement = (daily_sign == mom_sign.shift(1)).rolling(20).mean()
    momentum_consistency = agreement.shift(1)
    
    features['momentum_quality'] = (np.abs(trend_strength) * momentum_consistency).shift(1)
    
    # Cross-sectional momentum dispersion
    mom_dispersion = mom_20d.std(axis=1) / (mom_20d.abs().mean(axis=1) + 1e-10)
    features['momentum_dispersion'] = pd.DataFrame(
        np.tile(mom_dispersion.values.reshape(-1, 1), (1, returns.shape[1])),
        index=returns.index, columns=returns.columns
    ).shift(1)
    
    return features


# =============================================================================
# VOLATILITY REGIME BLOCK
# =============================================================================

def compute_volatility_regime_block(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Advanced volatility-based features for regime detection.
    
    Features:
    ---------
    vol_term_structure: Ratio of short to long-term vol.
                       > 1 = inverted (stress), < 1 = normal (calm)
                       
    vol_surprise: Unexpected volatility (actual vs expected).
                 Positive = vol spike, negative = vol compression.
                 
    vol_clustering: Rolling autocorrelation of squared returns.
                   High = vol clusters persist (GARCH-like).
                   
    vol_asymmetry: Difference between upside and downside vol.
                  Markets often show higher downside vol.
    """
    features = {}
    
    # Volatility at different horizons
    vol_5d = returns.rolling(window=5, min_periods=4).std()
    vol_20d = returns.rolling(window=20, min_periods=15).std()
    vol_60d = returns.rolling(window=60, min_periods=45).std()
    
    # Vol term structure (short / long)
    features['vol_term_structure'] = (vol_5d / (vol_60d + 1e-10)).shift(1)
    
    # Vol surprise: actual vs expected (using vol-of-vol)
    vol_expected = vol_20d.rolling(window=60, min_periods=45).mean()
    vol_std = vol_20d.rolling(window=60, min_periods=45).std()
    features['vol_surprise'] = ((vol_20d - vol_expected) / (vol_std + 1e-10)).shift(1)
    
    # Vol clustering: autocorrelation of squared returns
    sq_ret = returns ** 2
    vol_clustering = sq_ret.rolling(window=20, min_periods=15).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0
    )
    features['vol_clustering'] = vol_clustering.shift(1)
    
    # Vol asymmetry: downside vol / upside vol
    downside_vol = returns[returns < 0].rolling(window=60, min_periods=30).std()
    upside_vol = returns[returns > 0].rolling(window=60, min_periods=30).std()
    
    # Fill NaN for days without enough negative/positive returns
    downside_vol = downside_vol.fillna(method='ffill')
    upside_vol = upside_vol.fillna(method='ffill')
    
    features['vol_asymmetry'] = (downside_vol / (upside_vol + 1e-10)).shift(1)
    
    return features


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def compute_interaction_features(momentum_features: Dict[str, pd.DataFrame],
                                  regime_indicators: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Create interaction features between momentum and regime.
    
    These capture conditional alpha:
    - Momentum works differently in different regimes
    - Interactions can capture non-linear effects
    
    Features:
    ---------
    mom_x_high_vol: Momentum when vol is high (different alpha?)
    mom_x_uptrend: Momentum in uptrends (trend following?)
    mom_x_regime_confidence: Weight momentum by regime clarity
    """
    interactions = {}
    
    # Get key features
    mom_5d = momentum_features.get('mom_risk_adj_5d', momentum_features.get('mom_5d'))
    mom_20d = momentum_features.get('mom_risk_adj_20d', momentum_features.get('mom_20d'))
    
    vol_regime = regime_indicators.get('vol_regime')
    trend_regime = regime_indicators.get('trend_regime')
    regime_confidence = regime_indicators.get('regime_confidence')
    
    if mom_5d is not None and vol_regime is not None:
        # Momentum in high vol (might work better as reversal)
        interactions['mom_x_high_vol'] = mom_5d * vol_regime
        interactions['mom_x_low_vol'] = mom_5d * (1 - vol_regime)
    
    if mom_5d is not None and trend_regime is not None:
        # Momentum in uptrend (trend following)
        interactions['mom_x_uptrend'] = mom_5d * trend_regime
        interactions['mom_x_downtrend'] = mom_5d * (1 - trend_regime)
    
    if mom_20d is not None and regime_confidence is not None:
        # Confidence-weighted momentum
        interactions['mom_x_confidence'] = mom_20d * regime_confidence
    
    if mom_5d is not None and mom_20d is not None:
        # Cross-timeframe momentum interaction
        interactions['mom_cross_tf'] = mom_5d * np.sign(mom_20d)
    
    return interactions


# =============================================================================
# FEATURE STANDARDIZATION
# =============================================================================

def cross_sectional_standardize(feature: pd.DataFrame, 
                                  min_assets: int = 10) -> pd.DataFrame:
    """
    Cross-sectionally standardize a feature (mean=0, std=1 across assets each day).
    
    This ensures features are comparable across time and removes
    time-varying level effects.
    
    Parameters:
    -----------
    feature : pd.DataFrame
        Feature matrix (index=dates, columns=assets)
    min_assets : int
        Minimum assets required for standardization
        
    Returns:
    --------
    pd.DataFrame : Standardized feature
    """
    cs_mean = feature.mean(axis=1)
    cs_std = feature.std(axis=1)
    cs_std = cs_std.replace(0, np.nan)
    
    standardized = feature.sub(cs_mean, axis=0).div(cs_std, axis=0)
    
    # Set to NaN if not enough assets
    n_assets = feature.notna().sum(axis=1)
    standardized.loc[n_assets < min_assets] = np.nan
    
    return standardized


def standardize_feature_dict(features: Dict[str, pd.DataFrame],
                              min_assets: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Cross-sectionally standardize all features in a dictionary.
    """
    return {
        name: cross_sectional_standardize(feat, min_assets)
        for name, feat in features.items()
    }


# =============================================================================
# FEATURE COMBINATION UTILITIES
# =============================================================================

def combine_feature_blocks(*feature_dicts: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Combine multiple feature dictionaries into one.
    Handles name conflicts by prefixing with block number.
    """
    combined = {}
    
    for block_idx, feat_dict in enumerate(feature_dicts):
        for name, feat in feat_dict.items():
            if name in combined:
                # Handle name collision
                new_name = f"block{block_idx}_{name}"
                combined[new_name] = feat
            else:
                combined[name] = feat
    
    return combined


def select_features(features: Dict[str, pd.DataFrame],
                    include: Optional[List[str]] = None,
                    exclude: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Select subset of features by name.
    
    Parameters:
    -----------
    features : Dict
        Full feature dictionary
    include : List[str], optional
        List of feature names to include (if None, include all)
    exclude : List[str], optional
        List of feature names to exclude
        
    Returns:
    --------
    Dict : Filtered feature dictionary
    """
    if include is not None:
        features = {k: v for k, v in features.items() if k in include}
    
    if exclude is not None:
        features = {k: v for k, v in features.items() if k not in exclude}
    
    return features


# =============================================================================
# GPU-ACCELERATED FEATURES (PyTorch)
# =============================================================================

def compute_features_gpu(returns: pd.DataFrame, 
                         device: str = 'cuda') -> Dict[str, pd.DataFrame]:
    """
    GPU-accelerated feature computation using PyTorch.
    
    This is useful for very large datasets or real-time computation.
    Falls back to CPU if CUDA not available.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    device : str
        'cuda' or 'cpu'
        
    Returns:
    --------
    Dict[str, pd.DataFrame] : Features
    """
    try:
        import torch
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        # Convert to tensor
        ret_tensor = torch.tensor(returns.values, dtype=torch.float32, device=device)
        T, N = ret_tensor.shape
        
        features = {}
        
        # Momentum features (vectorized on GPU)
        for window in [5, 10, 20]:
            # Cumulative sum using unfold
            ret_unfolded = ret_tensor.unfold(0, window, 1)  # (T-window+1, N, window)
            mom = ret_unfolded.sum(dim=2)
            
            # Pad with NaN
            mom_padded = torch.full((T, N), float('nan'), device=device)
            mom_padded[window:] = mom[:-1]  # Shift by 1
            
            features[f'mom_{window}d_gpu'] = pd.DataFrame(
                mom_padded.cpu().numpy(),
                index=returns.index,
                columns=returns.columns
            )
        
        # Volatility (rolling std)
        for window in [20]:
            ret_unfolded = ret_tensor.unfold(0, window, 1)
            vol = ret_unfolded.std(dim=2)
            
            vol_padded = torch.full((T, N), float('nan'), device=device)
            vol_padded[window:] = vol[:-1]
            
            features[f'vol_{window}d_gpu'] = pd.DataFrame(
                vol_padded.cpu().numpy(),
                index=returns.index,
                columns=returns.columns
            )
        
        print(f"✅ GPU features computed on {device}")
        return features
        
    except ImportError:
        print("PyTorch not installed, using CPU implementation")
        return {}


# =============================================================================
# CONVENIENCE FUNCTION: ALL FEATURES
# =============================================================================

def compute_all_features(returns: pd.DataFrame,
                          panel: Optional[Dict[str, pd.DataFrame]] = None,
                          standardize: bool = True,
                          include_interactions: bool = True,
                          include_gpu: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Compute all available features in one call.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    panel : Dict, optional
        Panel data with OHLCV (for volume features)
    standardize : bool
        Whether to cross-sectionally standardize features
    include_interactions : bool
        Whether to compute interaction features
    include_gpu : bool
        Whether to use GPU acceleration (requires PyTorch)
        
    Returns:
    --------
    Dict[str, pd.DataFrame] : All features
    """
    print("Computing features...")
    
    # Core feature blocks
    trend_features = compute_trend_block(returns)
    print(f"  ✅ Trend features: {len(trend_features)}")
    
    regime_indicators = compute_regime_indicators(returns)
    print(f"  ✅ Regime indicators: {len(regime_indicators)}")
    
    momentum_features = compute_advanced_momentum_block(returns, panel)
    print(f"  ✅ Momentum features: {len(momentum_features)}")
    
    vol_features = compute_volatility_regime_block(returns)
    print(f"  ✅ Volatility features: {len(vol_features)}")
    
    # Combine
    all_features = combine_feature_blocks(
        trend_features,
        regime_indicators, 
        momentum_features,
        vol_features
    )
    
    # Interactions
    if include_interactions:
        interactions = compute_interaction_features(momentum_features, regime_indicators)
        all_features = combine_feature_blocks(all_features, interactions)
        print(f"  ✅ Interaction features: {len(interactions)}")
    
    # GPU features
    if include_gpu:
        gpu_features = compute_features_gpu(returns)
        all_features = combine_feature_blocks(all_features, gpu_features)
    
    # Standardize
    if standardize:
        all_features = standardize_feature_dict(all_features)
        print("  ✅ Features standardized (cross-sectional)")
    
    print(f"\n📊 Total features: {len(all_features)}")
    
    return all_features


# =============================================================================
# REGIME MASK FOR FILTERING
# =============================================================================

def create_regime_mask(returns: pd.DataFrame,
                        regime_type: str = 'high_vol_only',
                        custom_indicator: Optional[pd.Series] = None) -> pd.Series:
    """
    Create a regime mask for filtering trading signals.
    
    The mask is 1 when conditions are favorable, 0 otherwise.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    regime_type : str
        One of:
        - 'high_vol_only': Trade only in high volatility (momentum works better)
        - 'low_vol_only': Trade only in low volatility (mean reversion)
        - 'uptrend_only': Trade only in uptrends
        - 'downtrend_only': Trade only in downtrends
        - 'confident_only': Trade when regime is clear
        - 'custom': Use custom_indicator
    custom_indicator : pd.Series, optional
        Custom binary indicator for 'custom' regime_type
        
    Returns:
    --------
    pd.Series : Binary mask (1=trade, 0=no trade)
    """
    market_ret = returns.mean(axis=1)
    
    if regime_type == 'high_vol_only':
        vol = market_ret.rolling(20).std() * np.sqrt(252)
        vol_median = vol.expanding(min_periods=252).median()
        mask = (vol > vol_median).astype(float)
        
    elif regime_type == 'low_vol_only':
        vol = market_ret.rolling(20).std() * np.sqrt(252)
        vol_median = vol.expanding(min_periods=252).median()
        mask = (vol <= vol_median).astype(float)
        
    elif regime_type == 'uptrend_only':
        cumret = market_ret.rolling(60).sum()
        mask = (cumret > 0).astype(float)
        
    elif regime_type == 'downtrend_only':
        cumret = market_ret.rolling(60).sum()
        mask = (cumret <= 0).astype(float)
        
    elif regime_type == 'confident_only':
        # Trade when regime is clear (not transitioning)
        vol = market_ret.rolling(20).std() * np.sqrt(252)
        vol_pct = vol.expanding(min_periods=252).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5
        )
        
        cumret = market_ret.rolling(60).sum()
        trend_pct = cumret.expanding(min_periods=252).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5
        )
        
        # Confident = far from 0.5 percentile
        vol_confident = np.abs(vol_pct - 0.5) > 0.25
        trend_confident = np.abs(trend_pct - 0.5) > 0.25
        mask = (vol_confident | trend_confident).astype(float)
        
    elif regime_type == 'custom':
        if custom_indicator is None:
            raise ValueError("Must provide custom_indicator for 'custom' regime_type")
        mask = custom_indicator.astype(float)
        
    else:
        raise ValueError(f"Unknown regime_type: {regime_type}")
    
    return mask.shift(1)  # Lag by 1 to avoid lookahead


if __name__ == "__main__":
    # Test the module
    print("Feature Engineering Library loaded successfully!")
    print("\nAvailable functions:")
    print("  - compute_trend_block()")
    print("  - compute_regime_indicators()")
    print("  - compute_advanced_momentum_block()")
    print("  - compute_volatility_regime_block()")
    print("  - compute_interaction_features()")
    print("  - compute_all_features()")
    print("  - create_regime_mask()")
