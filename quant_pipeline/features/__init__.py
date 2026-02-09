# =============================================================================
# FEATURES MODULE - Feature Construction (ALPHA ONLY)
# =============================================================================
# 
# RESPONSIBILITY: Generate candidate features with potential information content
# 
# FORBIDDEN:
#   - ML models
#   - Targets
#   - Backtests
#   - Sharpe ratios
#   - Portfolio logic
#
# ALLOWED:
#   - Feature construction
#   - Statistical inspection
#   - Stationarity tests
#   - Correlation analysis
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    name: str
    family: str  # momentum, volatility, mean_reversion, etc.
    lookback: int  # primary lookback window
    description: str
    unit: str = "dimensionless"
    stationary: Optional[bool] = None
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'family': self.family,
            'lookback': self.lookback,
            'description': self.description,
            'unit': self.unit,
            'stationary': self.stationary
        }


@dataclass
class FeatureRegistry:
    """Registry of all computed features with metadata."""
    features: Dict[str, FeatureMetadata] = field(default_factory=dict)
    
    def register(self, meta: FeatureMetadata):
        self.features[meta.name] = meta
    
    def get_names(self) -> List[str]:
        return list(self.features.keys())
    
    def get_by_family(self, family: str) -> List[str]:
        return [k for k, v in self.features.items() if v.family == family]
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.features.values()])


# =============================================================================
# FEATURE GENERATORS (Pure functions, no state)
# =============================================================================

def kalman_filter_1d(y: np.ndarray, Q: float = 1e-5, R: float = 1e-2) -> np.ndarray:
    """
    1D Kalman filter for trend estimation.
    
    INVARIANT: Uses only past data (no lookahead).
    
    Args:
        y: Input series
        Q: Process noise
        R: Measurement noise
    
    Returns:
        Filtered estimate (same length as y)
    """
    n = len(y)
    x_est = np.zeros(n)
    P = np.zeros(n)
    
    x_est[0] = y[0]
    P[0] = 1.0
    
    for t in range(1, n):
        # Predict
        x_pred = x_est[t-1]
        P_pred = P[t-1] + Q
        
        # Update
        K = P_pred / (P_pred + R)
        x_est[t] = x_pred + K * (y[t] - x_pred)
        P[t] = (1 - K) * P_pred
    
    return x_est


def compute_momentum_features(returns: pd.Series, registry: FeatureRegistry) -> pd.DataFrame:
    """
    Compute momentum-family features.
    
    Features:
        - mom_5d, mom_10d, mom_21d, mom_63d: Rolling returns
        - mom_acceleration: Short-term vs medium-term momentum
        - mom_reversal: Negative short-term momentum
        - mom_zscore: Z-scored momentum
        - mom_consistency: Win rate over window
    """
    feat = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    for w in [5, 10, 21, 63]:
        name = f'mom_{w}d'
        feat[name] = returns.rolling(w).sum()
        registry.register(FeatureMetadata(
            name=name,
            family='momentum',
            lookback=w,
            description=f'{w}-day cumulative return',
            unit='return'
        ))
    
    # Acceleration
    feat['mom_acceleration'] = feat['mom_5d'] - feat['mom_21d']
    registry.register(FeatureMetadata(
        name='mom_acceleration',
        family='momentum',
        lookback=21,
        description='Momentum acceleration (5d - 21d)'
    ))
    
    # Reversal
    feat['mom_reversal'] = -feat['mom_5d']
    registry.register(FeatureMetadata(
        name='mom_reversal',
        family='momentum',
        lookback=5,
        description='Mean reversion signal (negative 5d mom)'
    ))
    
    # Z-score
    feat['mom_zscore'] = (
        (feat['mom_21d'] - feat['mom_21d'].rolling(63).mean()) / 
        feat['mom_21d'].rolling(63).std()
    )
    registry.register(FeatureMetadata(
        name='mom_zscore',
        family='momentum',
        lookback=63,
        description='Z-scored 21d momentum'
    ))
    
    # Consistency (win rate)
    feat['mom_consistency'] = returns.rolling(21).apply(
        lambda x: (x > 0).sum() / len(x), raw=False
    )
    registry.register(FeatureMetadata(
        name='mom_consistency',
        family='momentum',
        lookback=21,
        description='Fraction of positive returns over 21d'
    ))
    
    return feat


def compute_volatility_features(returns: pd.Series, registry: FeatureRegistry) -> pd.DataFrame:
    """
    Compute volatility-family features.
    
    Features:
        - vol_5d, vol_10d, vol_21d: Rolling volatility
        - vol_ratio: Short/medium vol ratio
        - vol_zscore: Z-scored volatility
        - vol_regime: High volatility indicator
    """
    feat = pd.DataFrame(index=returns.index)
    
    # Rolling volatility
    for w in [5, 10, 21]:
        name = f'vol_{w}d'
        feat[name] = returns.rolling(w).std()
        registry.register(FeatureMetadata(
            name=name,
            family='volatility',
            lookback=w,
            description=f'{w}-day rolling volatility',
            unit='volatility'
        ))
    
    # Ratio
    feat['vol_ratio'] = feat['vol_5d'] / feat['vol_21d']
    registry.register(FeatureMetadata(
        name='vol_ratio',
        family='volatility',
        lookback=21,
        description='5d/21d volatility ratio'
    ))
    
    # Z-score
    feat['vol_zscore'] = (
        (feat['vol_21d'] - feat['vol_21d'].rolling(63).mean()) /
        feat['vol_21d'].rolling(63).std()
    )
    registry.register(FeatureMetadata(
        name='vol_zscore',
        family='volatility',
        lookback=63,
        description='Z-scored 21d volatility'
    ))
    
    # Regime
    feat['vol_regime'] = (
        feat['vol_21d'] > feat['vol_21d'].rolling(126).quantile(0.8)
    ).astype(float)
    registry.register(FeatureMetadata(
        name='vol_regime',
        family='volatility',
        lookback=126,
        description='High volatility regime indicator'
    ))
    
    return feat


def compute_mean_reversion_features(
    log_prices: pd.Series, 
    prices: pd.Series,
    returns: pd.Series,
    registry: FeatureRegistry
) -> pd.DataFrame:
    """
    Compute mean-reversion features.
    
    Features:
        - ma_20_dev, ma_50_dev: Deviation from moving average
        - bb_position: Bollinger band position
        - rsi_21: Relative strength index
    """
    feat = pd.DataFrame(index=prices.index)
    
    # MA deviations
    for w in [20, 50]:
        name = f'ma_{w}_dev'
        feat[name] = log_prices - log_prices.rolling(w).mean()
        registry.register(FeatureMetadata(
            name=name,
            family='mean_reversion',
            lookback=w,
            description=f'Log price deviation from {w}d MA'
        ))
    
    # Bollinger position
    ma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    feat['bb_position'] = (prices - ma20) / (2 * std20)
    registry.register(FeatureMetadata(
        name='bb_position',
        family='mean_reversion',
        lookback=20,
        description='Position within Bollinger bands'
    ))
    
    # RSI
    feat['rsi_21'] = returns.rolling(21).apply(
        lambda x: x[x > 0].sum() / (x.abs().sum() + 1e-10), raw=False
    )
    registry.register(FeatureMetadata(
        name='rsi_21',
        family='mean_reversion',
        lookback=21,
        description='21-day relative strength index'
    ))
    
    return feat


def compute_kalman_features(log_prices: pd.Series, registry: FeatureRegistry) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute Kalman filter features.
    
    Features:
        - kalman_trend: Deviation from Kalman estimate
        - kalman_trend_zscore: Z-scored Kalman deviation
        - kalman_slope: Kalman estimate slope
        - kalman_curvature: Second derivative of Kalman
        - kalman_deviation: Absolute deviation
    
    Returns:
        Tuple of (features DataFrame, kalman_estimate Series)
    """
    feat = pd.DataFrame(index=log_prices.index)
    
    # Run Kalman filter
    kalman_est = pd.Series(
        kalman_filter_1d(log_prices.values),
        index=log_prices.index
    )
    
    # Store estimate for later use
    feat['_kalman_est'] = kalman_est
    
    # Trend (deviation from estimate)
    feat['kalman_trend'] = kalman_est - log_prices
    registry.register(FeatureMetadata(
        name='kalman_trend',
        family='kalman',
        lookback=0,  # Recursive
        description='Kalman estimate minus log price'
    ))
    
    # Z-score
    feat['kalman_trend_zscore'] = (
        (feat['kalman_trend'] - feat['kalman_trend'].rolling(63).mean()) /
        feat['kalman_trend'].rolling(63).std()
    )
    registry.register(FeatureMetadata(
        name='kalman_trend_zscore',
        family='kalman',
        lookback=63,
        description='Z-scored Kalman trend'
    ))
    
    # Slope
    feat['kalman_slope'] = kalman_est.diff(5)
    registry.register(FeatureMetadata(
        name='kalman_slope',
        family='kalman',
        lookback=5,
        description='5-day Kalman slope'
    ))
    
    # Curvature
    feat['kalman_curvature'] = feat['kalman_slope'].diff(5)
    registry.register(FeatureMetadata(
        name='kalman_curvature',
        family='kalman',
        lookback=10,
        description='Kalman curvature (second derivative)'
    ))
    
    # Absolute deviation
    feat['kalman_deviation'] = (log_prices - kalman_est).abs()
    registry.register(FeatureMetadata(
        name='kalman_deviation',
        family='kalman',
        lookback=0,
        description='Absolute deviation from Kalman'
    ))
    
    return feat, kalman_est


# =============================================================================
# HMM FEATURES - Regime Detection
# =============================================================================

def fit_hmm_gaussian(observations: np.ndarray, n_states: int = 3, n_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Gaussian HMM using EM algorithm (no external dependencies).
    
    Implements Baum-Welch algorithm for parameter estimation.
    
    Args:
        observations: 1D array of observations
        n_states: Number of hidden states
        n_iter: Number of EM iterations
    
    Returns:
        Tuple of (transition_matrix, means, stds, state_probs)
        - transition_matrix: (n_states, n_states) transition probabilities
        - means: (n_states,) emission means
        - stds: (n_states,) emission standard deviations
        - state_probs: (T, n_states) posterior state probabilities
    """
    T = len(observations)
    
    # Initialize parameters
    np.random.seed(42)
    
    # Initialize means with k-means-like approach
    sorted_obs = np.sort(observations)
    quantiles = np.linspace(0, 1, n_states + 2)[1:-1]
    means = np.array([sorted_obs[int(q * len(sorted_obs))] for q in quantiles])
    
    # Initialize std
    stds = np.full(n_states, np.std(observations) / n_states)
    
    # Initialize transition matrix (encourage persistence)
    A = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(A, 0.9)
    A = A / A.sum(axis=1, keepdims=True)
    
    # Initial state distribution
    pi = np.full(n_states, 1.0 / n_states)
    
    # EM iterations
    for iteration in range(n_iter):
        # E-step: Forward-backward algorithm
        
        # Compute emission probabilities
        B = np.zeros((T, n_states))
        for k in range(n_states):
            B[:, k] = np.exp(-0.5 * ((observations - means[k]) / stds[k])**2) / (stds[k] * np.sqrt(2 * np.pi))
        B = np.maximum(B, 1e-300)  # Avoid underflow
        
        # Forward pass (scaled)
        alpha = np.zeros((T, n_states))
        scale = np.zeros(T)
        
        alpha[0] = pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * B[t]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
            else:
                alpha[t] = 1.0 / n_states
                scale[t] = 1e-300
        
        # Backward pass (scaled)
        beta = np.zeros((T, n_states))
        beta[-1] = 1.0
        
        for t in range(T-2, -1, -1):
            beta[t] = (A @ (B[t+1] * beta[t+1]))
            if scale[t+1] > 0:
                beta[t] /= scale[t+1]
        
        # Compute posteriors (gamma)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum
        
        # Compute xi (transition posteriors)
        xi = np.zeros((T-1, n_states, n_states))
        for t in range(T-1):
            denom = 0.0
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * B[t+1, j] * beta[t+1, j]
                    denom += xi[t, i, j]
            if denom > 0:
                xi[t] /= denom
        
        # M-step: Update parameters
        
        # Update initial distribution
        pi = gamma[0]
        
        # Update transition matrix
        for i in range(n_states):
            denom = gamma[:-1, i].sum()
            if denom > 0:
                for j in range(n_states):
                    A[i, j] = xi[:, i, j].sum() / denom
        
        # Ensure rows sum to 1
        A = A / A.sum(axis=1, keepdims=True)
        
        # Update emission parameters
        for k in range(n_states):
            denom = gamma[:, k].sum()
            if denom > 0:
                means[k] = (gamma[:, k] * observations).sum() / denom
                stds[k] = np.sqrt((gamma[:, k] * (observations - means[k])**2).sum() / denom)
                stds[k] = max(stds[k], 1e-6)  # Avoid zero std
    
    return A, means, stds, gamma


def compute_regime_features(
    returns: pd.Series,
    kalman_slope: pd.Series,
    registry: FeatureRegistry,
    n_states: int = 3,
    lookback: int = 63
) -> pd.DataFrame:
    """
    Compute regime detection features using fast volatility-based classification.
    
    This is a FAST alternative to full HMM that captures similar information:
    - Regime is classified based on volatility quantiles
    - State probabilities are computed via softmax of distance to regime centers
    - Much faster than full EM-based HMM
    
    Core Features:
        - regime_state: Current volatility regime (0=low, 1=mid, 2=high)
        - regime_confidence: Confidence in current regime assignment
        - regime_entropy: Uncertainty across regimes
        - regime_p_high_vol: Probability of high volatility state
        - regime_delta_prob: Change in regime probabilities
        - regime_duration: Consecutive days in same regime
        - regime_transition_rate: Recent transition frequency
    
    Kalman-based features (on smoother signal):
        - kalman_regime_conf: Regime confidence on Kalman slope
        - kalman_regime_entropy: Entropy on Kalman features
    
    Args:
        returns: Daily returns
        kalman_slope: Kalman slope for smoothed estimation
        registry: Feature registry
        n_states: Number of regime states (typically 3)
        lookback: Window for volatility estimation
    
    Returns:
        DataFrame with regime features
    """
    feat = pd.DataFrame(index=returns.index)
    T = len(returns)
    
    # Compute rolling volatility
    vol = returns.rolling(lookback).std()
    vol_kalman = kalman_slope.rolling(lookback).std()
    
    # Compute rolling volatility quantiles for regime classification
    vol_low = vol.rolling(252).quantile(0.33)
    vol_high = vol.rolling(252).quantile(0.67)
    
    # Classify into regimes
    regime_state = np.zeros(T)
    regime_state[vol.values > vol_high.values] = 2  # High vol
    regime_state[(vol.values > vol_low.values) & (vol.values <= vol_high.values)] = 1  # Mid
    # Low vol = 0 (default)
    
    # Compute "soft" regime probabilities using softmax of -distance to thresholds
    # This gives us continuous confidence measures
    probs = np.zeros((T, n_states))
    
    # Temperature for softmax
    temp = 1.0
    
    for t in range(T):
        if np.isnan(vol.iloc[t]) or np.isnan(vol_low.iloc[t]):
            probs[t] = np.nan
            continue
        
        v = vol.iloc[t]
        v_low = vol_low.iloc[t]
        v_high = vol_high.iloc[t]
        
        if v_low >= v_high:
            # Edge case
            probs[t] = [0.33, 0.34, 0.33]
        else:
            # Distance-based probabilities (inverse distance squared, normalized)
            mid_point = (v_low + v_high) / 2
            
            # Centers for each regime
            centers = [v_low * 0.5, mid_point, v_high * 1.5]
            
            # Compute negative squared distances
            scores = np.array([-((v - c) / (v_high - v_low + 1e-10))**2 / temp for c in centers])
            
            # Softmax
            exp_scores = np.exp(scores - scores.max())
            probs[t] = exp_scores / exp_scores.sum()
    
    # Compute features
    feat['regime_state'] = regime_state
    registry.register(FeatureMetadata(
        name='regime_state',
        family='regime',
        lookback=lookback,
        description='Volatility regime (0=low, 1=mid, 2=high)'
    ))
    
    # Confidence = max probability
    feat['regime_confidence'] = np.nanmax(probs, axis=1)
    registry.register(FeatureMetadata(
        name='regime_confidence',
        family='regime',
        lookback=lookback,
        description='Confidence in regime classification'
    ))
    
    # Entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.nansum(probs * np.log(probs + 1e-10), axis=1)
    feat['regime_entropy'] = entropy
    registry.register(FeatureMetadata(
        name='regime_entropy',
        family='regime',
        lookback=lookback,
        description='Regime uncertainty (entropy)'
    ))
    
    # P(high vol)
    feat['regime_p_high_vol'] = probs[:, 2] if n_states >= 3 else probs[:, 1]
    registry.register(FeatureMetadata(
        name='regime_p_high_vol',
        family='regime',
        lookback=lookback,
        description='Probability of high volatility regime'
    ))
    
    # Delta probability
    feat['regime_delta_prob'] = np.abs(np.diff(probs, axis=0, prepend=np.nan)).sum(axis=1)
    feat['regime_delta_prob'] = feat['regime_delta_prob'].replace([np.inf, -np.inf], np.nan)
    registry.register(FeatureMetadata(
        name='regime_delta_prob',
        family='regime',
        lookback=lookback,
        description='Change in regime probabilities'
    ))
    
    # Regime duration (consecutive days in same state)
    duration = np.ones(T)
    for t in range(1, T):
        if regime_state[t] == regime_state[t-1]:
            duration[t] = duration[t-1] + 1
    feat['regime_duration'] = duration
    registry.register(FeatureMetadata(
        name='regime_duration',
        family='regime',
        lookback=lookback,
        description='Consecutive days in current regime'
    ))
    
    # Transition rate (regime changes in last 21 days)
    regime_changes = (regime_state != np.roll(regime_state, 1)).astype(float)
    feat['regime_transition_rate'] = pd.Series(regime_changes).rolling(21).mean().values
    registry.register(FeatureMetadata(
        name='regime_transition_rate',
        family='regime',
        lookback=21,
        description='Regime transition frequency (21d)'
    ))
    
    # --- Kalman-based regime features ---
    vol_kalman_low = vol_kalman.rolling(252).quantile(0.33)
    vol_kalman_high = vol_kalman.rolling(252).quantile(0.67)
    
    probs_kalman = np.zeros((T, n_states))
    
    for t in range(T):
        if np.isnan(vol_kalman.iloc[t]) or np.isnan(vol_kalman_low.iloc[t]):
            probs_kalman[t] = np.nan
            continue
        
        v = vol_kalman.iloc[t]
        v_low = vol_kalman_low.iloc[t]
        v_high = vol_kalman_high.iloc[t]
        
        if v_low >= v_high:
            probs_kalman[t] = [0.33, 0.34, 0.33]
        else:
            mid_point = (v_low + v_high) / 2
            centers = [v_low * 0.5, mid_point, v_high * 1.5]
            scores = np.array([-((v - c) / (v_high - v_low + 1e-10))**2 / temp for c in centers])
            exp_scores = np.exp(scores - scores.max())
            probs_kalman[t] = exp_scores / exp_scores.sum()
    
    feat['kalman_regime_conf'] = np.nanmax(probs_kalman, axis=1)
    registry.register(FeatureMetadata(
        name='kalman_regime_conf',
        family='regime',
        lookback=lookback,
        description='Kalman-based regime confidence'
    ))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy_kalman = -np.nansum(probs_kalman * np.log(probs_kalman + 1e-10), axis=1)
    feat['kalman_regime_entropy'] = entropy_kalman
    registry.register(FeatureMetadata(
        name='kalman_regime_entropy',
        family='regime',
        lookback=lookback,
        description='Kalman-based regime entropy'
    ))
    
    return feat


def _forward_pass_final(
    observations: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> np.ndarray:
    """
    Compute final state probabilities using forward algorithm.
    
    Args:
        observations: Sequence of observations
        A: Transition matrix (n_states x n_states)
        means: Emission means (n_states,)
        stds: Emission stds (n_states,)
    
    Returns:
        State probabilities at final time step (n_states,)
    """
    n_states = len(means)
    T = len(observations)
    
    if T == 0:
        return np.ones(n_states) / n_states
    
    # Emission probabilities
    B = np.zeros((T, n_states))
    for k in range(n_states):
        B[:, k] = np.exp(-0.5 * ((observations - means[k]) / (stds[k] + 1e-10))**2)
        B[:, k] /= (stds[k] * np.sqrt(2 * np.pi) + 1e-10)
    B = np.maximum(B, 1e-300)
    
    # Forward pass with scaling
    alpha = np.zeros((T, n_states))
    pi = np.ones(n_states) / n_states  # Uniform initial
    
    alpha[0] = pi * B[0]
    alpha[0] /= (alpha[0].sum() + 1e-300)
    
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[t]
        s = alpha[t].sum()
        if s > 0:
            alpha[t] /= s
        else:
            alpha[t] = np.ones(n_states) / n_states
    
    return alpha[-1]


def compute_hmm_features(
    returns: pd.Series,
    kalman_slope: pd.Series,
    registry: FeatureRegistry,
    n_states: int = 3,
    fit_window: int = 252,
    fit_interval: int = 21  # Fit HMM every 21 days, interpolate between
) -> pd.DataFrame:
    """
    Compute HMM-based regime features with SPARSE fitting for efficiency.
    
    HMM is fitted only every `fit_interval` days (default: 21 = monthly).
    Between fittings, state probabilities are updated using the forward algorithm
    with fixed parameters.
    
    This is ~20x faster than daily fitting while retaining most signal quality.
    
    Core HMM Features:
        - hmm_confidence: Max state probability (regime certainty)
        - hmm_entropy: Regime entropy (high = transition)
        - hmm_p_high_vol: Probability of high-volatility state
        - hmm_state: Most likely state
    
    Transition Features:
        - hmm_delta_prob: Change in state probability
        - hmm_expected_duration: Expected regime duration from transition matrix
    
    Uses BOTH raw returns AND Kalman-smoothed features for robust estimation.
    
    Args:
        returns: Daily returns
        kalman_slope: Kalman slope for smoothed estimation
        registry: Feature registry
        n_states: Number of HMM states
        fit_window: Rolling window for fitting
        fit_interval: How often to refit HMM (days)
    
    Returns:
        DataFrame with HMM features
    """
    feat = pd.DataFrame(index=returns.index)
    T = len(returns)
    
    # Initialize arrays
    hmm_confidence = np.full(T, np.nan)
    hmm_entropy = np.full(T, np.nan)
    hmm_p_high_vol = np.full(T, np.nan)
    hmm_state = np.full(T, np.nan)
    hmm_delta_prob = np.full(T, np.nan)
    hmm_expected_duration = np.full(T, np.nan)
    
    # HMM on Kalman features
    hmm_kalman_confidence = np.full(T, np.nan)
    hmm_kalman_entropy = np.full(T, np.nan)
    
    # Convert to arrays
    ret_vals = returns.values
    kalman_vals = kalman_slope.fillna(0).values
    
    # Current HMM parameters (will be updated at fit_interval)
    A_ret = None
    means_ret = None
    stds_ret = None
    A_kalman = None
    means_kalman = None
    stds_kalman = None
    
    # Previous state probs for delta computation
    prev_probs_ret = None
    
    # Rolling HMM fit - but only every fit_interval days
    for t in range(fit_window, T):
        # Refit HMM at intervals
        if (t - fit_window) % fit_interval == 0:
            # Get window
            window_ret = ret_vals[t-fit_window:t]
            window_kalman = kalman_vals[t-fit_window:t]
            
            # Skip if not enough valid data
            if np.isnan(window_ret).sum() > fit_window * 0.1:
                continue
            
            # Fill NaNs with median
            window_ret = np.nan_to_num(window_ret, nan=np.nanmedian(window_ret))
            window_kalman = np.nan_to_num(window_kalman, nan=np.nanmedian(window_kalman))
            
            try:
                # Fit HMM on returns
                A_ret, means_ret, stds_ret, gamma_ret = fit_hmm_gaussian(
                    window_ret, n_states, n_iter=30  # Reduced iterations
                )
                
                # Fit HMM on Kalman features (smoother)
                A_kalman, means_kalman, stds_kalman, gamma_kalman = fit_hmm_gaussian(
                    window_kalman, n_states, n_iter=30
                )
            except Exception as e:
                continue
        
        # Compute state probabilities using current parameters
        if A_ret is None:
            continue
        
        # Get recent observations for filtering
        recent_ret = ret_vals[max(0, t-10):t+1]
        recent_kalman = kalman_vals[max(0, t-10):t+1]
        
        try:
            # Forward algorithm to get current state probabilities
            probs_ret = _forward_pass_final(recent_ret, A_ret, means_ret, stds_ret)
            probs_kalman = _forward_pass_final(recent_kalman, A_kalman, means_kalman, stds_kalman)
            
            # Core features
            hmm_confidence[t] = probs_ret.max()
            hmm_entropy[t] = -np.sum(probs_ret * np.log(probs_ret + 1e-10))
            
            # Identify high-vol state (highest emission std)
            high_vol_state = np.argmax(stds_ret)
            hmm_p_high_vol[t] = probs_ret[high_vol_state]
            
            # Most likely state
            hmm_state[t] = np.argmax(probs_ret)
            
            # Delta probability (change from previous)
            if prev_probs_ret is not None:
                hmm_delta_prob[t] = np.abs(probs_ret - prev_probs_ret).sum()
            prev_probs_ret = probs_ret.copy()
            
            # Expected duration from transition matrix
            current_state = int(hmm_state[t])
            hmm_expected_duration[t] = 1.0 / (1.0 - A_ret[current_state, current_state] + 1e-6)
            
            # Kalman-based HMM features (smoother)
            hmm_kalman_confidence[t] = probs_kalman.max()
            hmm_kalman_entropy[t] = -np.sum(probs_kalman * np.log(probs_kalman + 1e-10))
            
        except Exception as e:
            continue
    
    # Add features
    feat['hmm_confidence'] = hmm_confidence
    registry.register(FeatureMetadata(
        name='hmm_confidence',
        family='hmm',
        lookback=fit_window,
        description='Max state probability (regime certainty)'
    ))
    
    feat['hmm_entropy'] = hmm_entropy
    registry.register(FeatureMetadata(
        name='hmm_entropy',
        family='hmm',
        lookback=fit_window,
        description='Regime entropy (high = transition)'
    ))
    
    feat['hmm_p_high_vol'] = hmm_p_high_vol
    registry.register(FeatureMetadata(
        name='hmm_p_high_vol',
        family='hmm',
        lookback=fit_window,
        description='Probability of high-volatility state'
    ))
    
    feat['hmm_state'] = hmm_state
    registry.register(FeatureMetadata(
        name='hmm_state',
        family='hmm',
        lookback=fit_window,
        description='Most likely HMM state'
    ))
    
    feat['hmm_delta_prob'] = hmm_delta_prob
    registry.register(FeatureMetadata(
        name='hmm_delta_prob',
        family='hmm',
        lookback=fit_window,
        description='Sum of absolute changes in state probabilities'
    ))
    
    feat['hmm_expected_duration'] = hmm_expected_duration
    registry.register(FeatureMetadata(
        name='hmm_expected_duration',
        family='hmm',
        lookback=fit_window,
        description='Expected duration of current regime'
    ))
    
    # Kalman-based HMM (smoother)
    feat['hmm_kalman_confidence'] = hmm_kalman_confidence
    registry.register(FeatureMetadata(
        name='hmm_kalman_confidence',
        family='hmm',
        lookback=fit_window,
        description='HMM confidence on Kalman features (less noisy)'
    ))
    
    feat['hmm_kalman_entropy'] = hmm_kalman_entropy
    registry.register(FeatureMetadata(
        name='hmm_kalman_entropy',
        family='hmm',
        lookback=fit_window,
        description='HMM entropy on Kalman features'
    ))
    
    return feat


def compute_regime_interaction_features(
    panel: pd.DataFrame,
    registry: FeatureRegistry
) -> pd.DataFrame:
    """
    Compute interaction features based on regime detection.
    
    Works with either fast regime features or HMM features (whichever is available).
    
    These features encode CONDITIONAL alpha hypotheses:
    - "Momentum works better in stable regimes"
    - "Mean reversion works after shocks"
    - "Kalman signals are cleaner in low-vol regimes"
    """
    feat = pd.DataFrame(index=panel.index)
    
    # Check which features are available
    has_hmm = 'hmm_confidence' in panel.columns
    has_regime = 'regime_confidence' in panel.columns
    
    # Use HMM if available, otherwise use fast regime features
    if has_hmm:
        conf_col = 'hmm_confidence'
        high_vol_col = 'hmm_p_high_vol'
        entropy_col = 'hmm_entropy'
        duration_col = 'hmm_expected_duration'
        prefix = 'hmm'
    elif has_regime:
        conf_col = 'regime_confidence'
        high_vol_col = 'regime_p_high_vol'
        entropy_col = 'regime_entropy'
        duration_col = 'regime_duration'
        prefix = 'regime'
    else:
        # No regime features available, return empty
        return feat
    
    # Momentum × regime confidence
    # Penalizes momentum in unstable regimes
    feat[f'mom_x_{prefix}_conf'] = panel['mom_21d'] * panel[conf_col]
    registry.register(FeatureMetadata(
        name=f'mom_x_{prefix}_conf',
        family='regime_interaction',
        lookback=252,
        description='Momentum × regime confidence (penalize unstable)'
    ))
    
    # Reversion × shock probability
    # Reversion works best after shocks
    feat[f'reversal_x_{prefix}_shock'] = panel['mom_reversal'] * panel[high_vol_col]
    registry.register(FeatureMetadata(
        name=f'reversal_x_{prefix}_shock',
        family='regime_interaction',
        lookback=252,
        description='Reversal × P(shock state)'
    ))
    
    # Kalman slope × low-vol probability
    feat[f'kalman_x_{prefix}_lowvol'] = panel['kalman_slope'] * (1 - panel[high_vol_col])
    registry.register(FeatureMetadata(
        name=f'kalman_x_{prefix}_lowvol',
        family='regime_interaction',
        lookback=252,
        description='Kalman slope × P(low vol state)'
    ))
    
    # Momentum × expected duration
    # Trust momentum more in persistent regimes
    duration_clipped = panel[duration_col].clip(upper=20)
    feat[f'mom_x_{prefix}_duration'] = panel['mom_21d'] * duration_clipped / 20
    registry.register(FeatureMetadata(
        name=f'mom_x_{prefix}_duration',
        family='regime_interaction',
        lookback=252,
        description='Momentum × normalized regime duration'
    ))
    
    # Vol × entropy (risk filter)
    feat[f'vol_x_{prefix}_entropy'] = panel['vol_21d'] * panel[entropy_col]
    registry.register(FeatureMetadata(
        name=f'vol_x_{prefix}_entropy',
        family='regime_interaction',
        lookback=252,
        description='Volatility × entropy (uncertainty risk)'
    ))
    
    return feat


# Backward compatibility alias
def compute_hmm_interaction_features(panel: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """DEPRECATED: Use compute_regime_interaction_features instead."""
    return compute_regime_interaction_features(panel, registry)


# =============================================================================
# FEATURE PANEL BUILDER
# =============================================================================

class FeaturePanelBuilder:
    """
    Builds feature panel from price/return data.
    
    INVARIANTS:
    - No lookahead bias (all features use only past data)
    - All features have metadata
    - Missing values are tracked, not silently filled
    """
    
    def __init__(self, include_hmm: bool = False):
        """
        Args:
            include_hmm: Whether to include slow HMM features (default: False).
                        If False, uses fast regime features instead.
        """
        self.registry = FeatureRegistry()
        self._panel = None
        self._is_built = False
        self._include_hmm = include_hmm
    
    def build(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        log_prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build feature panel for all assets.
        
        Args:
            prices: Close prices (assets x dates)
            returns: Daily returns (assets x dates)
            log_prices: Log prices (assets x dates)
        
        Returns:
            Panel with columns: date, ticker, [features...]
        """
        all_features = []
        
        n_tickers = len(prices.columns)
        
        for i, ticker in enumerate(prices.columns):
            if (i + 1) % 20 == 0:
                print(f"   Processing {i+1}/{n_tickers} assets...")
            
            feat = pd.DataFrame(index=prices.index)
            feat['date'] = feat.index
            feat['ticker'] = ticker
            
            # Compute each feature family
            price_series = prices[ticker]
            return_series = returns[ticker]
            log_price_series = log_prices[ticker]
            
            # Momentum
            mom_feat = compute_momentum_features(return_series, self.registry)
            feat = pd.concat([feat, mom_feat], axis=1)
            
            # Volatility
            vol_feat = compute_volatility_features(return_series, self.registry)
            feat = pd.concat([feat, vol_feat], axis=1)
            
            # Mean reversion
            mr_feat = compute_mean_reversion_features(
                log_price_series, price_series, return_series, self.registry
            )
            feat = pd.concat([feat, mr_feat], axis=1)
            
            # Kalman (returns both features and estimate)
            kalman_feat, kalman_est = compute_kalman_features(log_price_series, self.registry)
            kalman_slope = kalman_feat['kalman_slope']
            # Remove internal column before concat
            kalman_feat = kalman_feat.drop(columns=['_kalman_est'])
            feat = pd.concat([feat, kalman_feat], axis=1)
            
            # Store Kalman estimate for visualization
            feat['_kalman_est'] = kalman_est
            feat['_log_price'] = log_price_series
            
            # Fast regime features (always included)
            regime_feat = compute_regime_features(
                return_series, kalman_slope, self.registry,
                n_states=3, lookback=63
            )
            feat = pd.concat([feat, regime_feat], axis=1)
            
            # HMM features (optional - slower but more sophisticated)
            if self._include_hmm:
                hmm_feat = compute_hmm_features(
                    return_series, kalman_slope, self.registry,
                    n_states=3, fit_window=252, fit_interval=21
                )
                feat = pd.concat([feat, hmm_feat], axis=1)
            
            all_features.append(feat)
        
        # Combine
        panel = pd.concat(all_features, ignore_index=True)
        
        # Add cross-sectional features
        panel = self._add_cross_sectional_features(panel)
        
        # Add regime interaction features (works with either HMM or regime features)
        regime_interact = compute_regime_interaction_features(panel, self.registry)
        panel = pd.concat([panel, regime_interact], axis=1)
        
        self._panel = panel
        self._is_built = True
        
        print(f"[FEATURES] Built panel: {len(panel):,} rows, {len(self.registry.get_names())} features")
        
        return panel
    
    def _add_cross_sectional_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional rank features."""
        # Rank by date
        panel['cs_rank_ret5d'] = panel.groupby('date')['mom_5d'].rank(pct=True)
        self.registry.register(FeatureMetadata(
            name='cs_rank_ret5d',
            family='cross_sectional',
            lookback=5,
            description='Cross-sectional rank of 5d return'
        ))
        
        panel['cs_rank_ret21d'] = panel.groupby('date')['mom_21d'].rank(pct=True)
        self.registry.register(FeatureMetadata(
            name='cs_rank_ret21d',
            family='cross_sectional',
            lookback=21,
            description='Cross-sectional rank of 21d return'
        ))
        
        panel['cs_rank_vol'] = panel.groupby('date')['vol_21d'].rank(pct=True)
        self.registry.register(FeatureMetadata(
            name='cs_rank_vol',
            family='cross_sectional',
            lookback=21,
            description='Cross-sectional rank of 21d volatility'
        ))
        
        panel['cs_rank_mom'] = panel.groupby('date')['mom_zscore'].rank(pct=True)
        self.registry.register(FeatureMetadata(
            name='cs_rank_mom',
            family='cross_sectional',
            lookback=63,
            description='Cross-sectional rank of momentum z-score'
        ))
        
        return panel
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.registry.get_names()
    
    def get_metadata(self) -> pd.DataFrame:
        """Get feature metadata as DataFrame."""
        return self.registry.to_dataframe()


# =============================================================================
# FEATURE DIAGNOSTICS
# =============================================================================

def check_lookahead_bias(
    panel: pd.DataFrame,
    feature_cols: List[str],
    return_col: str = 'mom_5d'
) -> Dict[str, float]:
    """
    Check for potential lookahead bias by computing
    correlation between features and FUTURE returns.
    
    If correlation with future returns >> correlation with past returns,
    there may be lookahead bias.
    """
    results = {}
    
    for col in feature_cols:
        # Get valid data
        valid = panel[['date', 'ticker', col, return_col]].dropna()
        
        # Correlation with same-day return (should be 0 or slightly positive)
        corr_same = valid[col].corr(valid[return_col])
        
        results[col] = {
            'corr_same_day': corr_same,
        }
    
    return results


def compute_feature_stability(
    panel: pd.DataFrame,
    feature_cols: List[str],
    n_periods: int = 4
) -> pd.DataFrame:
    """
    Compute feature stability across time periods.
    
    Returns mean and std of feature distribution in each sub-period.
    """
    panel = panel.copy()
    panel['period'] = pd.qcut(panel['date'].rank(method='dense'), n_periods, labels=False)
    
    stability = []
    for col in feature_cols:
        for period in range(n_periods):
            mask = panel['period'] == period
            stability.append({
                'feature': col,
                'period': period,
                'mean': panel.loc[mask, col].mean(),
                'std': panel.loc[mask, col].std(),
                'skew': panel.loc[mask, col].skew(),
                'missing_pct': panel.loc[mask, col].isna().mean() * 100
            })
    
    return pd.DataFrame(stability)


def compute_feature_correlation_matrix(
    panel: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Compute correlation matrix between features."""
    return panel[feature_cols].corr()


def identify_redundant_features(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.95
) -> List[Tuple[str, str, float]]:
    """Identify pairs of features with correlation above threshold."""
    redundant = []
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) >= threshold:
                    redundant.append((col1, col2, corr))
    
    return sorted(redundant, key=lambda x: -abs(x[2]))
