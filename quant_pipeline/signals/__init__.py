# =============================================================================
# SIGNALS MODULE - Signal Interpretation & Risk Mapping
# =============================================================================
# 
# RESPONSIBILITY: Translate raw model predictions into economic meaning
# 
# FORBIDDEN:
#   - Full portfolio backtests
#   - Sharpe optimization
#
# ALLOWED:
#   - Signal scaling
#   - Ranking analysis
#   - Turnover analysis
#   - Capacity & concentration diagnostics
#   - Exposure analysis
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SignalConfig:
    """Configuration for signal construction."""
    scaling_method: str  # 'zscore', 'rank', 'winsorize', 'none'
    zscore_window: int = 21
    winsorize_pct: float = 0.05
    
    def to_dict(self) -> dict:
        return {
            'scaling_method': self.scaling_method,
            'zscore_window': self.zscore_window,
            'winsorize_pct': self.winsorize_pct
        }


def scale_signal_zscore(
    predictions: pd.Series,
    dates: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Scale predictions to z-scores within rolling window.
    
    INVARIANT: Uses only past data for z-score computation.
    """
    df = pd.DataFrame({'date': dates, 'pred': predictions})
    
    # Group by date and compute cross-sectional z-score
    def zscore_cs(x):
        return (x - x.mean()) / x.std() if x.std() > 0 else 0
    
    df['signal'] = df.groupby('date')['pred'].transform(zscore_cs)
    
    return df['signal']


def scale_signal_rank(
    predictions: pd.Series,
    dates: pd.Series
) -> pd.Series:
    """
    Scale predictions to cross-sectional ranks.
    
    Returns percentile rank [0, 1] for each date.
    """
    df = pd.DataFrame({'date': dates, 'pred': predictions})
    df['signal'] = df.groupby('date')['pred'].rank(pct=True)
    
    return df['signal']


def compute_signal_turnover(
    signals: pd.DataFrame,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    signal_col: str = 'signal',
    threshold: float = 0.0
) -> pd.Series:
    """
    Compute daily turnover implied by signal changes.
    
    Turnover = sum of absolute weight changes.
    """
    # Pivot to get signal matrix (dates x assets)
    signal_matrix = signals.pivot(index=date_col, columns=ticker_col, values=signal_col)
    
    # Normalize to weights (assuming equal risk per position)
    long_mask = signal_matrix > threshold
    short_mask = signal_matrix < -threshold
    
    n_long = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)
    
    weights = signal_matrix.copy()
    for date in weights.index:
        if n_long[date] > 0:
            weights.loc[date, long_mask.loc[date]] = 1 / n_long[date]
        if n_short[date] > 0:
            weights.loc[date, short_mask.loc[date]] = -1 / n_short[date]
        weights.loc[date, ~(long_mask.loc[date] | short_mask.loc[date])] = 0
    
    # Compute turnover
    weight_changes = weights.diff().abs()
    turnover = weight_changes.sum(axis=1)
    
    return turnover


def compute_signal_concentration(
    signals: pd.DataFrame,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    signal_col: str = 'signal'
) -> pd.DataFrame:
    """
    Compute signal concentration metrics.
    
    Returns:
        - Top N concentration
        - HHI (Herfindahl-Hirschman Index)
    """
    results = []
    
    for date, group in signals.groupby(date_col):
        signal_abs = group[signal_col].abs()
        signal_sorted = signal_abs.sort_values(ascending=False)
        
        total = signal_abs.sum()
        if total > 0:
            # Top 5 concentration
            top5_conc = signal_sorted.head(5).sum() / total
            
            # Top 10 concentration
            top10_conc = signal_sorted.head(10).sum() / total
            
            # HHI
            weights = signal_abs / total
            hhi = (weights ** 2).sum()
        else:
            top5_conc = 0
            top10_conc = 0
            hhi = 0
        
        results.append({
            'date': date,
            'top5_concentration': top5_conc,
            'top10_concentration': top10_conc,
            'hhi': hhi
        })
    
    return pd.DataFrame(results)


def compute_signal_distribution_by_date(
    signals: pd.DataFrame,
    date_col: str = 'date',
    signal_col: str = 'signal'
) -> pd.DataFrame:
    """
    Compute signal distribution statistics by date.
    """
    stats = signals.groupby(date_col)[signal_col].agg([
        'mean', 'std', 'min', 'max', 'median',
        lambda x: (x > 0).sum(),  # n_positive
        lambda x: (x < 0).sum(),  # n_negative
    ])
    stats.columns = ['mean', 'std', 'min', 'max', 'median', 'n_positive', 'n_negative']
    
    return stats


def analyze_turnover_vs_threshold(
    signals: pd.DataFrame,
    thresholds: List[float],
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    signal_col: str = 'signal'
) -> pd.DataFrame:
    """
    Analyze how turnover changes with signal threshold.
    
    Useful for determining optimal entry/exit thresholds.
    """
    results = []
    
    for threshold in thresholds:
        turnover = compute_signal_turnover(
            signals, date_col, ticker_col, signal_col, threshold
        )
        
        results.append({
            'threshold': threshold,
            'mean_turnover': turnover.mean(),
            'std_turnover': turnover.std(),
            'max_turnover': turnover.max()
        })
    
    return pd.DataFrame(results)


# =============================================================================
# EXPOSURE ANALYSIS
# =============================================================================

def compute_net_exposure(
    signals: pd.DataFrame,
    date_col: str = 'date',
    signal_col: str = 'signal'
) -> pd.Series:
    """
    Compute net market exposure (sum of signals).
    
    Net exposure > 0 indicates long bias.
    Net exposure < 0 indicates short bias.
    """
    return signals.groupby(date_col)[signal_col].sum()


def compute_gross_exposure(
    signals: pd.DataFrame,
    date_col: str = 'date',
    signal_col: str = 'signal'
) -> pd.Series:
    """
    Compute gross exposure (sum of absolute signals).
    
    Measures total leverage.
    """
    return signals.groupby(date_col)[signal_col].apply(lambda x: x.abs().sum())


def compute_long_short_ratio(
    signals: pd.DataFrame,
    date_col: str = 'date',
    signal_col: str = 'signal'
) -> pd.Series:
    """
    Compute ratio of long to short exposure.
    """
    def ratio(x):
        long_exp = x[x > 0].sum()
        short_exp = abs(x[x < 0].sum())
        return long_exp / short_exp if short_exp > 0 else np.inf
    
    return signals.groupby(date_col)[signal_col].apply(ratio)


# =============================================================================
# SIGNAL STABILITY ANALYSIS
# =============================================================================

def compute_signal_autocorrelation(
    signals: pd.DataFrame,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    signal_col: str = 'signal',
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Compute signal autocorrelation at various lags.
    
    High autocorrelation = stable signal = lower turnover.
    Low autocorrelation = noisy signal = higher turnover.
    """
    # Pivot to matrix
    signal_matrix = signals.pivot(index=date_col, columns=ticker_col, values=signal_col)
    
    results = []
    for lag in range(1, max_lag + 1):
        # Compute cross-sectional correlation between t and t-lag
        shifted = signal_matrix.shift(lag)
        
        # Correlation by date
        daily_corrs = []
        for date in signal_matrix.index[lag:]:
            current = signal_matrix.loc[date].dropna()
            lagged = shifted.loc[date].dropna()
            common = current.index.intersection(lagged.index)
            
            if len(common) > 10:
                corr = current[common].corr(lagged[common])
                if not np.isnan(corr):
                    daily_corrs.append(corr)
        
        if len(daily_corrs) > 0:
            results.append({
                'lag': lag,
                'mean_autocorr': np.mean(daily_corrs),
                'std_autocorr': np.std(daily_corrs)
            })
    
    return pd.DataFrame(results)


def compute_signal_rank_stability(
    signals: pd.DataFrame,
    date_col: str = 'date',
    ticker_col: str = 'ticker',
    signal_col: str = 'signal',
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compute how stable the top N assets are over time.
    
    Returns overlap percentage between consecutive days.
    """
    # Get top N by signal each day
    top_assets = {}
    for date, group in signals.groupby(date_col):
        sorted_group = group.nlargest(top_n, signal_col)
        top_assets[date] = set(sorted_group[ticker_col])
    
    # Compute overlap
    dates = sorted(top_assets.keys())
    overlaps = []
    
    for i in range(1, len(dates)):
        prev_set = top_assets[dates[i-1]]
        curr_set = top_assets[dates[i]]
        overlap = len(prev_set & curr_set) / top_n
        overlaps.append({
            'date': dates[i],
            'overlap': overlap
        })
    
    return pd.DataFrame(overlaps)
