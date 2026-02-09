# =============================================================================
# TARGETS MODULE - Target Construction (NO features, NO models)
# =============================================================================
# 
# RESPONSIBILITY: Define what the model is trying to learn, clearly and causally
# 
# FORBIDDEN:
#   - Model training
#   - Portfolio logic
#   - Backtests
#
# ALLOWED:
#   - Target definition (returns, ranks, residuals)
#   - Alignment logic
#   - Horizon analysis
#   - Predictability sanity checks
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TargetConfig:
    """Configuration for target construction."""
    target_type: str  # 'return', 'rank', 'residual'
    horizon: int  # Forward horizon in days
    aggregation: str  # 'sum', 'mean', 'last'
    winsorize: bool = True
    winsorize_pct: float = 0.01
    
    def to_dict(self) -> dict:
        return {
            'target_type': self.target_type,
            'horizon': self.horizon,
            'aggregation': self.aggregation,
            'winsorize': self.winsorize,
            'winsorize_pct': self.winsorize_pct
        }


def construct_return_target(
    returns: pd.DataFrame,
    horizon: int = 5,
    aggregation: str = 'sum'
) -> pd.DataFrame:
    """
    Construct forward return target.
    
    INVARIANT: Uses shift(-horizon) to look FORWARD, creating proper target alignment.
    
    Args:
        returns: Daily returns (dates x assets)
        horizon: Forward horizon
        aggregation: 'sum' for cumulative return, 'mean' for average
    
    Returns:
        DataFrame with forward returns (dates x assets)
    """
    if aggregation == 'sum':
        # Forward cumulative return
        target = returns.shift(-horizon).rolling(horizon).sum()
    elif aggregation == 'mean':
        # Forward average return
        target = returns.shift(-horizon).rolling(horizon).mean()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return target


def construct_volnorm_target(
    returns: pd.DataFrame,
    horizon: int = 5,
    vol_window: int = 21
) -> pd.DataFrame:
    """
    Construct VOLATILITY-NORMALIZED forward return target.
    
    Definition: y_t = sum(r_{t+1}...r_{t+horizon}) / σ_t(vol_window)
    
    WHY THIS IS BETTER:
    - Removes volatility as a confounder
    - Forces model to learn DIRECTIONAL efficiency, not magnitude
    - Reduces tail chasing
    - Improves cross-asset comparability
    
    EMPIRICAL EFFECT:
    - Lower IS Sharpe (slightly)
    - Higher OOS Sharpe
    - Better stability
    
    Args:
        returns: Daily returns (dates x assets)
        horizon: Forward horizon
        vol_window: Window for volatility estimation (default 21 days)
    
    Returns:
        DataFrame with volatility-normalized forward returns
    """
    # Forward cumulative return
    forward_ret = returns.shift(-horizon).rolling(horizon).sum()
    
    # Historical volatility (computed BEFORE the forward period to avoid lookahead)
    hist_vol = returns.rolling(vol_window).std()
    
    # Add small epsilon to avoid division by zero
    target = forward_ret / (hist_vol + 1e-10)
    
    return target


def construct_rank_target(
    returns: pd.DataFrame,
    horizon: int = 5,
    normalize: str = 'percentile'
) -> pd.DataFrame:
    """
    Construct CROSS-SECTIONAL RANK target.
    
    Definition: y_{t,i} = rank(sum(r_{t+1:t+horizon,i})) across all assets at time t
    
    WHY THIS WORKS:
    - Ignores absolute magnitude
    - Learns relative winners
    - Matches how portfolios are ACTUALLY built (long top decile, short bottom)
    - ROBUST to regime shifts
    
    TRADE-OFF:
    - Loses magnitude information
    - But gains stability
    
    Args:
        returns: Daily returns (dates x assets)
        horizon: Forward horizon
        normalize: 'percentile' -> [0, 1], 'zscore' -> normalized, 'signed' -> [-1, 1]
    
    Returns:
        DataFrame with cross-sectional ranks
    """
    forward_ret = construct_return_target(returns, horizon, 'sum')
    
    # Rank across assets for each date
    if normalize == 'percentile':
        # [0, 1] percentile rank
        target = forward_ret.rank(axis=1, pct=True)
    elif normalize == 'zscore':
        # Cross-sectional z-score
        target = forward_ret.sub(forward_ret.mean(axis=1), axis=0).div(forward_ret.std(axis=1) + 1e-10, axis=0)
    elif normalize == 'signed':
        # [-1, 1] signed rank
        target = (forward_ret.rank(axis=1, pct=True) - 0.5) * 2
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    
    return target


def construct_sign_target(
    returns: pd.DataFrame,
    horizon: int = 5,
    mode: str = 'binary',
    flat_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Construct SIGN / CLASSIFICATION target.
    
    Definition (binary): y_t = 1 if sum(r_{t+1}...r_{t+horizon}) > 0, else 0
    Definition (ternary): y_t = 1 (up) / 0 (flat) / -1 (down)
    
    WHY IT WORKS:
    - Easier problem than regression
    - Less overfitting
    - Often higher ECONOMIC Sharpe even with lower accuracy
    
    WHEN TO USE:
    - If regression models are unstable
    - If signals are weak
    - If execution dominates performance
    
    Often works SHOCKINGLY well when regression fails.
    
    Args:
        returns: Daily returns (dates x assets)
        horizon: Forward horizon
        mode: 'binary' (up/down) or 'ternary' (up/flat/down)
        flat_threshold: For ternary, threshold to define "flat" (e.g., 0.001)
    
    Returns:
        DataFrame with classification targets
    """
    forward_ret = construct_return_target(returns, horizon, 'sum')
    
    if mode == 'binary':
        # 1 if positive, 0 if negative
        target = (forward_ret > 0).astype(float)
    elif mode == 'ternary':
        # 1 if up, -1 if down, 0 if flat
        target = pd.DataFrame(0.0, index=forward_ret.index, columns=forward_ret.columns)
        target[forward_ret > flat_threshold] = 1.0
        target[forward_ret < -flat_threshold] = -1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return target


def winsorize_target(
    target: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.Series:
    """Winsorize target to remove extreme values."""
    lower = target.quantile(lower_pct)
    upper = target.quantile(upper_pct)
    return target.clip(lower, upper)


def align_features_and_targets(
    feature_panel: pd.DataFrame,
    target_df: pd.DataFrame,
    target_name: str = 'target'
) -> pd.DataFrame:
    """
    Align features with targets, ensuring no lookahead bias.
    
    INVARIANT: 
    - Features at time t are aligned with targets from time t+horizon
    - This is done via the shift in construct_return_target
    
    Args:
        feature_panel: Panel with date, ticker, features
        target_df: DataFrame with targets (dates x assets)
        target_name: Name for target column
    
    Returns:
        Panel with features and target aligned
    """
    # Melt target to long format
    target_long = target_df.reset_index().melt(
        id_vars='Date' if 'Date' in target_df.reset_index().columns else target_df.index.name,
        var_name='ticker',
        value_name=target_name
    )
    
    # Rename date column if needed
    if 'Date' in target_long.columns:
        target_long = target_long.rename(columns={'Date': 'date'})
    elif target_df.index.name and target_df.index.name != 'date':
        target_long = target_long.rename(columns={target_df.index.name: 'date'})
    
    # Ensure date is datetime
    target_long['date'] = pd.to_datetime(target_long['date'])
    feature_panel['date'] = pd.to_datetime(feature_panel['date'])
    
    # Merge
    aligned = feature_panel.merge(
        target_long,
        on=['date', 'ticker'],
        how='left'
    )
    
    return aligned


# =============================================================================
# TARGET DIAGNOSTICS
# =============================================================================

def compute_ic_decay(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_lag: int = 20
) -> pd.DataFrame:
    """
    Compute Information Coefficient decay across time lags.
    
    Shows how predictive power of features decays over time.
    """
    ics = []
    
    for lag in range(1, max_lag + 1):
        panel_lagged = panel.copy()
        panel_lagged[target_col] = panel_lagged.groupby('ticker')[target_col].shift(-lag)
        panel_lagged = panel_lagged.dropna(subset=[target_col])
        
        for col in feature_cols:
            valid = panel_lagged[[col, target_col]].dropna()
            if len(valid) > 100:
                ic = valid[col].corr(valid[target_col])
                ics.append({
                    'feature': col,
                    'lag': lag,
                    'ic': ic
                })
    
    return pd.DataFrame(ics)


def estimate_noise_floor(
    target: pd.Series,
    n_shuffles: int = 100
) -> float:
    """
    Estimate noise floor by computing correlation with shuffled targets.
    
    If a model achieves correlation below this, it's not learning signal.
    """
    target_clean = target.dropna()
    
    correlations = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(target_clean.values)
        corr = np.corrcoef(target_clean.values, shuffled)[0, 1]
        correlations.append(abs(corr))
    
    # 95th percentile of random correlations
    noise_floor = np.percentile(correlations, 95)
    
    return noise_floor


def check_target_leakage(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Dict[str, float]:
    """
    Check for potential target leakage.
    
    If a feature is perfectly correlated with the target, there may be leakage.
    """
    leakage = {}
    
    for col in feature_cols:
        corr = panel[col].corr(panel[target_col])
        if abs(corr) > 0.8:
            leakage[col] = corr
    
    return leakage


def compute_target_autocorrelation(
    target: pd.Series,
    max_lag: int = 10
) -> pd.Series:
    """
    Compute target autocorrelation at various lags.
    
    High autocorrelation may indicate trivial predictability.
    """
    target_clean = target.dropna()
    
    autocorrs = {}
    for lag in range(1, max_lag + 1):
        autocorrs[lag] = target_clean.autocorr(lag=lag)
    
    return pd.Series(autocorrs)


def validate_target_horizon(
    returns: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    expected_horizon: int
) -> bool:
    """
    Validate that target horizon is correctly implemented.
    
    Returns True if target appears to be correctly forward-shifted.
    """
    # Sample a few dates and check alignment
    sample_dates = panel['date'].drop_duplicates().sort_values().iloc[100:110]
    sample_tickers = panel['ticker'].unique()[:5]
    
    errors = 0
    for date in sample_dates:
        for ticker in sample_tickers:
            # Get feature date
            feat_mask = (panel['date'] == date) & (panel['ticker'] == ticker)
            if feat_mask.sum() == 0:
                continue
            
            target_value = panel.loc[feat_mask, target_col].values[0]
            
            # Compute expected target
            future_dates = returns.index[returns.index > date][:expected_horizon]
            if len(future_dates) < expected_horizon:
                continue
            
            expected_target = returns.loc[future_dates, ticker].sum()
            
            if not np.isnan(target_value) and not np.isnan(expected_target):
                if abs(target_value - expected_target) > 1e-6:
                    errors += 1
    
    return errors == 0
