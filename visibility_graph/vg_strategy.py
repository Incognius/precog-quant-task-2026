"""
Visibility Graph Strategy Module

Trading signal generation based on VG/HVG features.

Strategies:
1. Heuristic: Use graph metrics (degree exponent, clustering) directly
2. ML: Use GNN predictions
3. Hybrid: Combine both
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum

from .vg_core import construct_hvg_fast, construct_vg_fast
from .vg_features import extract_graph_features, compute_rolling_vg_features


# =============================================================================
# CONFIGURATION
# =============================================================================

class SignalMethod(Enum):
    """Signal generation methods."""
    DEGREE_EXPONENT = "degree_exponent"  # Based on power law exponent
    CLUSTERING = "clustering"  # Based on clustering coefficient
    ENTROPY = "entropy"  # Based on degree entropy
    COMPOSITE = "composite"  # Weighted combination
    GNN = "gnn"  # Graph Neural Network


@dataclass
class VGStrategyConfig:
    """Configuration for VG-based strategy."""
    # Window parameters
    window_size: int = 20
    step_size: int = 1
    use_hvg: bool = True  # HVG is faster and often sufficient
    
    # Signal generation
    signal_method: SignalMethod = SignalMethod.COMPOSITE
    
    # Thresholds for heuristic methods
    gamma_threshold: float = 2.5  # Power law exponent (gamma < 2.5 = trending)
    clustering_threshold: float = 0.3  # Clustering coef (higher = more structured)
    entropy_threshold: float = 2.0  # Degree entropy
    
    # Signal smoothing
    smooth_window: int = 5  # Moving average for signal smoothing
    
    # Position sizing
    max_signal: float = 1.0  # Clip signals to [-max, max]
    
    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'use_hvg': self.use_hvg,
            'signal_method': self.signal_method.value,
            'gamma_threshold': self.gamma_threshold,
            'clustering_threshold': self.clustering_threshold,
            'entropy_threshold': self.entropy_threshold,
            'smooth_window': self.smooth_window,
            'max_signal': self.max_signal
        }


# =============================================================================
# SIGNAL GENERATION - HEURISTIC
# =============================================================================

def signal_from_degree_exponent(
    gamma: float,
    gamma_threshold: float = 2.5,
    invert: bool = False
) -> float:
    """
    Generate signal based on power law exponent.
    
    Theory:
    - gamma ~ 2-2.5: Strong persistence (trending market)
    - gamma ~ 3+: More random behavior
    
    Signal interpretation:
    - Low gamma -> Trend following (go with momentum)
    - High gamma -> Mean reversion
    
    Parameters
    ----------
    gamma : float
        Power law exponent
    gamma_threshold : float
        Threshold for neutral signal
    invert : bool
        If True, invert the signal logic
    
    Returns
    -------
    float
        Signal in [-1, 1]
    """
    if np.isnan(gamma):
        return 0.0
    
    # Normalize gamma to signal
    # gamma < threshold -> positive (trending regime, follow trend)
    # gamma > threshold -> negative (random regime, mean revert)
    signal = (gamma_threshold - gamma) / gamma_threshold
    signal = np.clip(signal, -1, 1)
    
    if invert:
        signal = -signal
    
    return float(signal)


def signal_from_clustering(
    clustering: float,
    clustering_threshold: float = 0.3
) -> float:
    """
    Generate signal based on clustering coefficient.
    
    High clustering = structured patterns = potentially predictable
    Low clustering = random = hard to predict
    
    Parameters
    ----------
    clustering : float
        Global clustering coefficient
    clustering_threshold : float
        Threshold
    
    Returns
    -------
    float
        Signal in [0, 1] (higher = more confident in prediction)
    """
    if np.isnan(clustering):
        return 0.5
    
    # Normalize
    signal = clustering / (clustering + clustering_threshold)
    return float(signal)


def signal_from_entropy(
    entropy: float,
    entropy_threshold: float = 2.0
) -> float:
    """
    Generate signal based on degree entropy.
    
    High entropy = uniform degree distribution = random
    Low entropy = concentrated (hubs) = structured
    
    Returns
    -------
    float
        Signal in [-1, 1]
    """
    if np.isnan(entropy):
        return 0.0
    
    # Lower entropy = more structure = stronger signal
    signal = (entropy_threshold - entropy) / entropy_threshold
    signal = np.clip(signal, -1, 1)
    
    return float(signal)


def generate_composite_signal(
    features: Dict[str, float],
    config: VGStrategyConfig,
    momentum: Optional[float] = None
) -> float:
    """
    Generate composite signal from multiple VG features.
    
    Parameters
    ----------
    features : dict
        Graph features from extract_graph_features
    config : VGStrategyConfig
        Strategy configuration
    momentum : float, optional
        Recent price momentum (for direction)
    
    Returns
    -------
    float
        Signal in [-1, 1]
    """
    # Get individual signals
    gamma_signal = signal_from_degree_exponent(
        features.get('gamma', np.nan),
        config.gamma_threshold
    )
    
    clustering_signal = signal_from_clustering(
        features.get('clustering_global', np.nan),
        config.clustering_threshold
    )
    
    entropy_signal = signal_from_entropy(
        features.get('degree_entropy', np.nan),
        config.entropy_threshold
    )
    
    # Composite: weighted average
    # Gamma is the main driver, clustering and entropy modulate confidence
    regime_signal = gamma_signal * 0.5 + entropy_signal * 0.3 + (clustering_signal - 0.5) * 0.4
    
    # If momentum is provided, use it for direction
    if momentum is not None:
        # Regime signal determines magnitude, momentum determines direction
        direction = np.sign(momentum)
        signal = regime_signal * direction
    else:
        signal = regime_signal
    
    return float(np.clip(signal, -config.max_signal, config.max_signal))


# =============================================================================
# ROLLING SIGNAL GENERATION
# =============================================================================

def generate_rolling_signals(
    prices: pd.Series,
    config: VGStrategyConfig,
    volumes: Optional[pd.Series] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate rolling VG-based trading signals.
    
    NO LOOKAHEAD BIAS: Signal at time t only uses data up to time t.
    
    Parameters
    ----------
    prices : pd.Series
        Price series with datetime index
    config : VGStrategyConfig
        Strategy configuration
    volumes : pd.Series, optional
        Volume series
    verbose : bool
        Print progress
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, signal, and intermediate features
    """
    price_values = prices.values.astype(np.float64)
    dates = prices.index
    
    # Compute rolling VG features
    time_indices, features_list = compute_rolling_vg_features(
        price_values,
        window_size=config.window_size,
        step_size=config.step_size,
        use_hvg=config.use_hvg,
        include_motifs=False,  # Skip for speed
        verbose=verbose
    )
    
    # Compute momentum for direction
    returns = prices.pct_change()
    momentum = returns.rolling(window=config.window_size).mean()
    
    # Generate signals
    signals = []
    
    for idx, features in zip(time_indices, features_list):
        date = dates[idx]
        mom = momentum.iloc[idx] if idx < len(momentum) else 0
        
        if config.signal_method == SignalMethod.DEGREE_EXPONENT:
            signal = signal_from_degree_exponent(features.get('gamma', np.nan), config.gamma_threshold)
        elif config.signal_method == SignalMethod.CLUSTERING:
            signal = 2 * signal_from_clustering(features.get('clustering_global', np.nan), config.clustering_threshold) - 1
        elif config.signal_method == SignalMethod.ENTROPY:
            signal = signal_from_entropy(features.get('degree_entropy', np.nan), config.entropy_threshold)
        elif config.signal_method == SignalMethod.COMPOSITE:
            signal = generate_composite_signal(features, config, mom)
        else:
            signal = 0.0
        
        signals.append({
            'date': date,
            'signal_raw': signal,
            'gamma': features.get('gamma', np.nan),
            'clustering': features.get('clustering_global', np.nan),
            'entropy': features.get('degree_entropy', np.nan),
            'assortativity': features.get('assortativity', np.nan),
            'momentum': mom
        })
    
    df = pd.DataFrame(signals)
    df.set_index('date', inplace=True)
    
    # Smooth signal
    df['signal'] = df['signal_raw'].rolling(window=config.smooth_window, min_periods=1).mean()
    df['signal'] = df['signal'].clip(-config.max_signal, config.max_signal)
    
    return df


# =============================================================================
# MULTI-ASSET SIGNAL GENERATION
# =============================================================================

def generate_signals_universe(
    prices_df: pd.DataFrame,
    config: VGStrategyConfig,
    volumes_df: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate VG signals for a universe of assets.
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with columns as tickers and index as dates
    config : VGStrategyConfig
        Strategy configuration
    volumes_df : pd.DataFrame, optional
        Volume DataFrame (same structure as prices)
    verbose : bool
        Print progress
    
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: date, ticker, signal
    """
    all_signals = []
    
    tickers = prices_df.columns.tolist()
    
    for i, ticker in enumerate(tickers):
        if verbose:
            print(f"Processing {ticker} ({i+1}/{len(tickers)})")
        
        prices = prices_df[ticker].dropna()
        volumes = volumes_df[ticker].dropna() if volumes_df is not None else None
        
        if len(prices) < config.window_size + 10:
            continue
        
        signals = generate_rolling_signals(prices, config, volumes, verbose=False)
        signals = signals[['signal']].reset_index()
        signals['ticker'] = ticker
        
        all_signals.append(signals)
    
    result = pd.concat(all_signals, ignore_index=True)
    result = result[['date', 'ticker', 'signal']]
    
    return result


# =============================================================================
# SIGNAL ANALYSIS
# =============================================================================

def analyze_signal_quality(
    signals: pd.DataFrame,
    returns: pd.Series,
    transaction_cost_bps: float = 10.0
) -> Dict[str, float]:
    """
    Analyze quality of generated signals.
    
    Parameters
    ----------
    signals : pd.DataFrame
        Signal DataFrame with 'signal' column
    returns : pd.Series
        Forward returns corresponding to signal dates
    transaction_cost_bps : float
        Transaction cost in basis points
    
    Returns
    -------
    dict
        Signal quality metrics
    """
    # Align signals and returns
    aligned = pd.concat([signals['signal'], returns], axis=1, join='inner')
    aligned.columns = ['signal', 'return']
    
    # Signal-return correlation (Information Coefficient)
    ic = aligned['signal'].corr(aligned['return'])
    
    # Hit rate (signal direction matches return direction)
    aligned['signal_direction'] = np.sign(aligned['signal'])
    aligned['return_direction'] = np.sign(aligned['return'])
    hit_rate = (aligned['signal_direction'] == aligned['return_direction']).mean()
    
    # Turnover
    signal_changes = aligned['signal'].diff().abs()
    avg_turnover = signal_changes.mean()
    
    # Transaction cost impact
    tc_rate = transaction_cost_bps / 10000
    tc_drag = avg_turnover * tc_rate * 252  # Annualized
    
    # Signal autocorrelation (stability)
    signal_autocorr = aligned['signal'].autocorr()
    
    return {
        'information_coefficient': float(ic),
        'hit_rate': float(hit_rate),
        'avg_turnover': float(avg_turnover),
        'annual_tc_drag': float(tc_drag),
        'signal_autocorr': float(signal_autocorr),
        'signal_mean': float(aligned['signal'].mean()),
        'signal_std': float(aligned['signal'].std())
    }


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_market_regime(
    features: Dict[str, float],
    config: VGStrategyConfig
) -> str:
    """
    Detect market regime based on VG features.
    
    Returns
    -------
    str
        One of: 'trending', 'mean_reverting', 'random', 'transitional'
    """
    gamma = features.get('gamma', np.nan)
    clustering = features.get('clustering_global', np.nan)
    entropy = features.get('degree_entropy', np.nan)
    
    if np.isnan(gamma) or np.isnan(clustering):
        return 'unknown'
    
    # Trending: low gamma (strong persistence), moderate clustering
    if gamma < 2.5 and clustering > 0.2:
        return 'trending'
    
    # Mean reverting: high gamma, low clustering
    if gamma > 3.0 and clustering < 0.2:
        return 'mean_reverting'
    
    # Random: high gamma, high entropy
    if gamma > 2.8 and entropy > 2.5:
        return 'random'
    
    return 'transitional'


def rolling_regime_detection(
    prices: pd.Series,
    config: VGStrategyConfig
) -> pd.Series:
    """
    Detect market regime on a rolling basis.
    
    Returns
    -------
    pd.Series
        Series of regime labels
    """
    price_values = prices.values.astype(np.float64)
    dates = prices.index
    
    time_indices, features_list = compute_rolling_vg_features(
        price_values,
        window_size=config.window_size,
        step_size=config.step_size,
        use_hvg=config.use_hvg
    )
    
    regimes = []
    for idx, features in zip(time_indices, features_list):
        regime = detect_market_regime(features, config)
        regimes.append({'date': dates[idx], 'regime': regime})
    
    return pd.DataFrame(regimes).set_index('date')['regime']
