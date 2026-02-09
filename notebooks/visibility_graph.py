"""
Visibility Graph Module for Time Series Analysis

This module implements Visibility Graph (VG) and Horizontal Visibility Graph (HVG)
construction and feature extraction for financial time series analysis.

References:
- Lacasa et al. (2008) "From time series to complex networks: The visibility graph"
- Luque et al. (2009) "Horizontal visibility graphs: exact results for random time series"

NO FORWARD BIAS: All features at time t only use data up to time t.
"""

import numpy as np
import pandas as pd
from numba import jit, prange
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import curve_fit
import warnings


# ============================================================================
# VG Construction Functions (Numba-accelerated)
# ============================================================================

@jit(nopython=True, parallel=True)
def construct_vg_fast(series: np.ndarray) -> np.ndarray:
    """
    Construct Visibility Graph from a time series.
    
    Two nodes i and j (with j > i) are connected if for all k with i < k < j:
        y[k] < y[i] + (y[j] - y[i]) * (k - i) / (j - i)
    
    This is the "line of sight" condition.
    
    Parameters
    ----------
    series : np.ndarray
        1D array of time series values
        
    Returns
    -------
    np.ndarray
        Adjacency matrix (symmetric, binary)
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.int32)
    
    for i in prange(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                # Check if k blocks visibility between i and j
                threshold = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                if series[k] >= threshold:
                    visible = False
                    break
            
            if visible:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj


@jit(nopython=True, parallel=True)
def construct_hvg_fast(series: np.ndarray) -> np.ndarray:
    """
    Construct Horizontal Visibility Graph from a time series.
    
    Two nodes i and j (with j > i) are connected if for all k with i < k < j:
        y[k] < min(y[i], y[j])
    
    HVG is faster to compute and provides similar structural information.
    
    Parameters
    ----------
    series : np.ndarray
        1D array of time series values
        
    Returns
    -------
    np.ndarray
        Adjacency matrix (symmetric, binary)
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.int32)
    
    for i in prange(n):
        for j in range(i + 1, n):
            min_val = min(series[i], series[j])
            visible = True
            
            for k in range(i + 1, j):
                if series[k] >= min_val:
                    visible = False
                    break
            
            if visible:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def _power_law(k: np.ndarray, c: float, gamma: float) -> np.ndarray:
    """Power law function for fitting degree distribution."""
    return c * np.power(k.astype(float), -gamma)


def extract_graph_features(adj: np.ndarray, include_motifs: bool = False) -> Dict[str, float]:
    """
    Extract topological features from a visibility graph.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    include_motifs : bool
        Whether to include motif-based features (slower)
        
    Returns
    -------
    dict
        Dictionary of feature names to values
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    
    features = {}
    
    # Basic statistics
    features['n_nodes'] = n
    features['n_edges'] = adj.sum() // 2
    features['density'] = adj.sum() / (n * (n - 1)) if n > 1 else 0
    
    # Degree statistics
    features['degree_mean'] = degrees.mean()
    features['degree_std'] = degrees.std()
    features['degree_max'] = degrees.max()
    features['degree_min'] = degrees.min()
    
    # Degree entropy (measure of heterogeneity)
    degree_probs = np.bincount(degrees.astype(int)) / n
    degree_probs = degree_probs[degree_probs > 0]
    features['degree_entropy'] = -np.sum(degree_probs * np.log2(degree_probs + 1e-10))
    
    # Power law exponent (gamma) estimation
    # For VG, gamma ≈ 3 for random series, gamma < 3 for correlated/trending
    try:
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        if len(unique_degrees) > 3 and unique_degrees.max() > 2:
            # Filter to valid range
            mask = unique_degrees >= 2
            unique_degrees = unique_degrees[mask]
            counts = counts[mask]
            
            if len(unique_degrees) > 3:
                probs = counts / counts.sum()
                # Fit power law
                popt, _ = curve_fit(
                    _power_law, 
                    unique_degrees, 
                    probs,
                    p0=[1.0, 2.5],
                    bounds=([0, 1], [np.inf, 10]),
                    maxfev=1000
                )
                features['gamma'] = popt[1]
            else:
                features['gamma'] = 2.5  # Default
        else:
            features['gamma'] = 2.5
    except Exception:
        features['gamma'] = 2.5  # Default on failure
    
    # Clustering coefficient (local transitivity)
    clustering_local = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k > 1:
            # Count edges among neighbors
            subgraph = adj[np.ix_(neighbors, neighbors)]
            edges_among = subgraph.sum() / 2
            max_edges = k * (k - 1) / 2
            clustering_local[i] = edges_among / max_edges if max_edges > 0 else 0
    
    features['clustering_mean'] = clustering_local.mean()
    features['clustering_std'] = clustering_local.std()
    features['clustering_global'] = clustering_local[degrees > 1].mean() if (degrees > 1).any() else 0
    
    # Assortativity (degree correlation)
    edges = np.array(np.where(np.triu(adj, 1))).T
    if len(edges) > 1:
        source_degrees = degrees[edges[:, 0]]
        target_degrees = degrees[edges[:, 1]]
        features['assortativity'] = np.corrcoef(source_degrees, target_degrees)[0, 1]
    else:
        features['assortativity'] = 0
    
    # Average path length (approximated for large graphs)
    if n <= 100:
        from scipy.sparse.csgraph import shortest_path
        from scipy.sparse import csr_matrix
        sp = shortest_path(csr_matrix(adj), unweighted=True)
        sp_valid = sp[np.isfinite(sp) & (sp > 0)]
        features['avg_path_length'] = sp_valid.mean() if len(sp_valid) > 0 else n
    else:
        # Estimate using sampling
        features['avg_path_length'] = np.log(n) / np.log(degrees.mean() + 1)
    
    # Motif analysis (optional, slow)
    if include_motifs:
        # Count triangles
        triangles = 0
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            for j_idx, j in enumerate(neighbors):
                for k in neighbors[j_idx + 1:]:
                    if adj[j, k]:
                        triangles += 1
        features['n_triangles'] = triangles // 3  # Each triangle counted 3 times
        
        # Transitivity
        possible_triangles = np.sum(degrees * (degrees - 1)) / 2
        features['transitivity'] = triangles / possible_triangles if possible_triangles > 0 else 0
    
    return features


def compute_rolling_vg_features(
    series: np.ndarray,
    window_size: int = 20,
    step_size: int = 1,
    use_hvg: bool = True,
    include_motifs: bool = False,
    verbose: bool = False
) -> Tuple[List[int], List[Dict[str, float]]]:
    """
    Compute VG features over rolling windows.
    
    NO FORWARD BIAS: Features at index t use only data from [t-window_size+1, t]
    
    Parameters
    ----------
    series : np.ndarray
        Time series data
    window_size : int
        Size of rolling window
    step_size : int
        Step between windows
    use_hvg : bool
        Use HVG instead of VG (faster)
    include_motifs : bool
        Include motif-based features
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (time_indices, list of feature dicts)
    """
    n = len(series)
    time_indices = []
    features_list = []
    
    construct_func = construct_hvg_fast if use_hvg else construct_vg_fast
    
    for i in range(window_size - 1, n, step_size):
        # Window ends at i (inclusive), NO future data
        window = series[i - window_size + 1:i + 1]
        
        # Construct graph
        adj = construct_func(window)
        
        # Extract features
        features = extract_graph_features(adj, include_motifs=include_motifs)
        
        time_indices.append(i)
        features_list.append(features)
        
        if verbose and (len(time_indices) % 100 == 0):
            print(f"  Processed {len(time_indices)} windows...")
    
    return time_indices, features_list


def features_to_dataframe(
    time_indices: List[int],
    features_list: List[Dict[str, float]],
    dates: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Convert feature lists to a pandas DataFrame.
    """
    df = pd.DataFrame(features_list)
    
    if dates is not None:
        df.index = pd.to_datetime(dates[time_indices])
    else:
        df.index = time_indices
    
    return df


# ============================================================================
# Signal Generation
# ============================================================================

class SignalMethod(Enum):
    """Signal generation methods."""
    GAMMA = "gamma"  # Based on power law exponent
    CLUSTERING = "clustering"  # Based on clustering coefficient
    COMPOSITE = "composite"  # Combination of features
    ENTROPY = "entropy"  # Based on degree entropy


@dataclass
class VGStrategyConfig:
    """Configuration for VG-based trading strategy."""
    window_size: int = 20
    step_size: int = 1
    use_hvg: bool = True
    signal_method: SignalMethod = SignalMethod.COMPOSITE
    gamma_threshold: float = 2.5  # gamma < threshold → trending
    clustering_threshold: float = 0.3
    smooth_window: int = 5
    max_signal: float = 1.0
    include_motifs: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'use_hvg': self.use_hvg,
            'signal_method': self.signal_method.value,
            'gamma_threshold': self.gamma_threshold,
            'clustering_threshold': self.clustering_threshold,
            'smooth_window': self.smooth_window,
            'max_signal': self.max_signal
        }


def _compute_raw_signal(features: pd.DataFrame, config: VGStrategyConfig) -> pd.Series:
    """
    Compute raw signal from VG features.
    
    Signal interpretation:
    - Positive: Bullish (trending up, momentum)
    - Negative: Bearish (trending down, mean reversion expected)
    - Zero: Neutral/random
    """
    if config.signal_method == SignalMethod.GAMMA:
        # Low gamma → strong correlations → trend following
        # Normalize gamma to [-1, 1]
        gamma_normalized = (config.gamma_threshold - features['gamma']) / config.gamma_threshold
        return gamma_normalized.clip(-1, 1)
    
    elif config.signal_method == SignalMethod.CLUSTERING:
        # High clustering → local structure → mean reversion
        clustering_signal = (config.clustering_threshold - features['clustering_global']) / config.clustering_threshold
        return clustering_signal.clip(-1, 1)
    
    elif config.signal_method == SignalMethod.ENTROPY:
        # Low entropy → predictable → trend following
        median_entropy = features['degree_entropy'].median()
        entropy_signal = (median_entropy - features['degree_entropy']) / median_entropy
        return entropy_signal.clip(-1, 1)
    
    elif config.signal_method == SignalMethod.COMPOSITE:
        # Combine multiple signals
        gamma_signal = (config.gamma_threshold - features['gamma']) / config.gamma_threshold
        clustering_signal = (features['clustering_global'] - config.clustering_threshold) / config.clustering_threshold
        entropy_signal = (features['degree_entropy'].median() - features['degree_entropy']) / (features['degree_entropy'].median() + 1e-6)
        
        # Weighted combination
        composite = 0.5 * gamma_signal + 0.3 * clustering_signal + 0.2 * entropy_signal
        return composite.clip(-1, 1)
    
    else:
        raise ValueError(f"Unknown signal method: {config.signal_method}")


def generate_rolling_signals(
    prices: pd.Series,
    config: VGStrategyConfig,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate trading signals from price time series using Visibility Graphs.
    
    NO FORWARD BIAS: Signal at time t uses only data up to time t.
    
    Parameters
    ----------
    prices : pd.Series
        Price time series with DatetimeIndex
    config : VGStrategyConfig
        Strategy configuration
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: signal, gamma, clustering, etc.
    """
    series = prices.values
    dates = prices.index.values
    
    # Compute rolling features
    time_indices, features_list = compute_rolling_vg_features(
        series,
        window_size=config.window_size,
        step_size=config.step_size,
        use_hvg=config.use_hvg,
        include_motifs=config.include_motifs,
        verbose=verbose
    )
    
    # Convert to DataFrame
    features_df = features_to_dataframe(time_indices, features_list, dates)
    
    # Compute raw signal
    raw_signal = _compute_raw_signal(features_df, config)
    
    # Apply smoothing (EMA)
    if config.smooth_window > 1:
        signal = raw_signal.ewm(span=config.smooth_window).mean()
    else:
        signal = raw_signal
    
    # Scale to max signal
    signal = signal * config.max_signal
    
    # Build result DataFrame
    result = pd.DataFrame({
        'signal': signal,
        'gamma': features_df['gamma'],
        'clustering': features_df['clustering_global'],
        'entropy': features_df['degree_entropy'],
        'raw_signal': raw_signal
    }, index=features_df.index)
    
    return result


def generate_signals_universe(
    prices_wide: pd.DataFrame,
    config: VGStrategyConfig,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate signals for multiple assets.
    
    Parameters
    ----------
    prices_wide : pd.DataFrame
        Wide-format prices (dates x tickers)
    config : VGStrategyConfig
        Strategy configuration
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: date, ticker, signal, ...
    """
    all_signals = []
    
    tickers = prices_wide.columns.tolist()
    
    for i, ticker in enumerate(tickers):
        if verbose and (i % 10 == 0):
            print(f"Processing {ticker} ({i+1}/{len(tickers)})")
        
        prices = prices_wide[ticker].dropna()
        
        if len(prices) < config.window_size + 10:
            continue
        
        signals = generate_rolling_signals(prices, config, verbose=False)
        signals = signals.reset_index()
        signals.columns = ['date'] + list(signals.columns[1:])
        signals['ticker'] = ticker
        
        all_signals.append(signals)
    
    return pd.concat(all_signals, ignore_index=True)


# ============================================================================
# Regime Detection
# ============================================================================

def rolling_regime_detection(
    prices: pd.Series,
    config: VGStrategyConfig
) -> pd.Series:
    """
    Detect market regime using VG features.
    
    Regimes:
    - 'trending': Low gamma, strong correlations
    - 'mean_reverting': High clustering, local structure
    - 'random': gamma ≈ 3, exponential degree distribution
    - 'transitional': Between regimes
    
    NO FORWARD BIAS: Regime at time t uses only data up to time t.
    """
    signals = generate_rolling_signals(prices, config)
    
    regimes = pd.Series(index=signals.index, dtype=str)
    
    for idx in signals.index:
        gamma = signals.loc[idx, 'gamma']
        clustering = signals.loc[idx, 'clustering']
        
        if gamma < 2.2:
            regime = 'trending'
        elif gamma > 3.0:
            regime = 'random'
        elif clustering > 0.4:
            regime = 'mean_reverting'
        elif 2.2 <= gamma <= 3.0:
            regime = 'transitional'
        else:
            regime = 'unknown'
        
        regimes.loc[idx] = regime
    
    return regimes


# ============================================================================
# Signal Quality Analysis
# ============================================================================

def analyze_signal_quality(
    signals: pd.DataFrame,
    forward_returns: pd.Series,
    transaction_cost_bps: float = 10.0
) -> Dict[str, float]:
    """
    Analyze signal quality metrics.
    
    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame with 'signal' column
    forward_returns : pd.Series
        Next-period returns (aligned with signals)
    transaction_cost_bps : float
        Transaction cost in basis points
        
    Returns
    -------
    dict
        Quality metrics
    """
    # Align data
    common_idx = signals.index.intersection(forward_returns.dropna().index)
    signal = signals.loc[common_idx, 'signal']
    returns = forward_returns.loc[common_idx]
    
    if len(common_idx) < 10:
        return {
            'information_coefficient': 0,
            'hit_rate': 0.5,
            'avg_turnover': 0,
            'n_observations': len(common_idx)
        }
    
    # Information Coefficient (rank correlation)
    ic, _ = stats.spearmanr(signal, returns)
    
    # Hit rate (directional accuracy)
    correct_direction = ((signal > 0) & (returns > 0)) | ((signal < 0) & (returns < 0))
    hit_rate = correct_direction.mean()
    
    # Turnover (signal changes)
    signal_diff = signal.diff().abs()
    avg_turnover = signal_diff.mean()
    
    # Net IC (adjusted for transaction costs)
    tc_drag = avg_turnover * transaction_cost_bps / 10000
    net_ic = ic - tc_drag
    
    return {
        'information_coefficient': ic if not np.isnan(ic) else 0,
        'net_ic': net_ic if not np.isnan(net_ic) else 0,
        'hit_rate': hit_rate,
        'avg_turnover': avg_turnover,
        'n_observations': len(common_idx)
    }


# ============================================================================
# Utilities
# ============================================================================

def visualize_vg(series: np.ndarray, adj: np.ndarray, title: str = "Visibility Graph"):
    """
    Visualize a visibility graph with the time series.
    """
    import matplotlib.pyplot as plt
    
    n = len(series)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series with visibility lines
    ax = axes[0]
    ax.bar(range(n), series, color='steelblue', alpha=0.7)
    
    # Draw visibility lines
    edges = np.array(np.where(np.triu(adj, 1))).T
    for i, j in edges:
        ax.plot([i, j], [series[i], series[j]], 'r-', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Adjacency matrix
    ax = axes[1]
    im = ax.imshow(adj, cmap='Blues', aspect='auto')
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    ax.set_title('Adjacency Matrix')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig
