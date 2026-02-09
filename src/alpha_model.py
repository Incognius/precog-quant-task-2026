"""
Reliability-Weighted Alpha Model
================================

Mathematical Framework (from research paper):

1. Model outputs class probabilities: p_t^(i) = (p_up, p_down, p_hold)
2. Raw score: s_t^(i) = p_up - p_down
3. Hit ratio (per-asset accuracy): h^(i) = rolling accuracy on calibration window
4. Reliability-weighted score: r_t^(i) = h^(i) * s_t^(i)
5. Dollar-neutral alpha: α_t^(i) = r_t^(i) - mean(r_t)

This ensures sum(α_t) = 0 (dollar neutrality).

Author: Quantitative Research Pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LABELING: 3-CLASS TARGET (UP / DOWN / HOLD)
# =============================================================================

def create_3class_labels(returns: pd.DataFrame, 
                         threshold: float = 0.005,
                         forward_days: int = 1) -> pd.DataFrame:
    """
    Create 3-class labels: Up (2), Hold (1), Down (0).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns panel (dates x assets)
    threshold : float
        Threshold for Up/Down classification (default: 0.5%)
    forward_days : int
        Forward return horizon for labels
        
    Returns:
    --------
    pd.DataFrame : Labels (0=Down, 1=Hold, 2=Up)
    """
    # Forward returns
    if forward_days == 1:
        fwd_returns = returns.shift(-1)
    else:
        fwd_returns = returns.rolling(forward_days).sum().shift(-forward_days)
    
    # Create labels
    labels = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=int)
    labels[:] = 1  # Default: Hold
    labels[fwd_returns > threshold] = 2   # Up
    labels[fwd_returns < -threshold] = 0  # Down
    
    return labels


def compute_sample_weights(dates: pd.DatetimeIndex, 
                          decay_halflife: int = 63,
                          recent_boost_days: int = 21,
                          recent_boost_factor: float = 2.0) -> np.ndarray:
    """
    Compute time-decay sample weights for training.
    
    Recent samples get higher weight (recency bias for regime adaptation).
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        Training dates
    decay_halflife : int
        Halflife for exponential decay (days)
    recent_boost_days : int
        Number of recent days to boost
    recent_boost_factor : float
        Boost factor for recent days
        
    Returns:
    --------
    np.ndarray : Sample weights
    """
    n = len(dates)
    
    # Exponential decay from most recent
    days_ago = np.arange(n-1, -1, -1)  # n-1, n-2, ..., 0
    decay = np.exp(-np.log(2) * days_ago / decay_halflife)
    
    # Boost recent days
    boost = np.ones(n)
    boost[-recent_boost_days:] = recent_boost_factor
    
    weights = decay * boost
    weights = weights / weights.sum()  # Normalize
    
    return weights


# =============================================================================
# HIT RATIO COMPUTATION
# =============================================================================

class HitRatioTracker:
    """
    Track per-asset hit ratios (accuracy) over a rolling calibration window.
    
    h^(i) = (1/K) * sum(1{ŷ_τ == y_τ}) for τ in calibration set
    """
    
    def __init__(self, calibration_window: int = 63):
        """
        Parameters:
        -----------
        calibration_window : int
            Number of past predictions to use for hit ratio
        """
        self.calibration_window = calibration_window
        self.predictions_history = {}  # asset -> list of (pred_class, true_class)
        self.hit_ratios = {}  # asset -> current hit ratio
        
    def update(self, date: pd.Timestamp, 
               pred_classes: pd.Series, 
               true_classes: pd.Series):
        """
        Update hit ratios with new predictions.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date
        pred_classes : pd.Series
            Predicted classes (0, 1, 2) per asset
        true_classes : pd.Series
            True classes (0, 1, 2) per asset
        """
        for asset in pred_classes.index:
            if pd.isna(pred_classes[asset]) or pd.isna(true_classes[asset]):
                continue
                
            pred = int(pred_classes[asset])
            true = int(true_classes[asset])
            
            if asset not in self.predictions_history:
                self.predictions_history[asset] = []
            
            self.predictions_history[asset].append((pred, true))
            
            # Keep only calibration window
            if len(self.predictions_history[asset]) > self.calibration_window:
                self.predictions_history[asset] = \
                    self.predictions_history[asset][-self.calibration_window:]
            
            # Compute hit ratio
            history = self.predictions_history[asset]
            hits = sum(1 for p, t in history if p == t)
            self.hit_ratios[asset] = hits / len(history)
    
    def get_hit_ratios(self, assets: pd.Index) -> pd.Series:
        """
        Get current hit ratios for assets.
        
        Returns 0.33 (random baseline) for assets with no history.
        """
        ratios = pd.Series(index=assets, dtype=float)
        for asset in assets:
            ratios[asset] = self.hit_ratios.get(asset, 1/3)  # Random baseline
        return ratios
    
    def reset(self):
        """Reset tracker."""
        self.predictions_history = {}
        self.hit_ratios = {}


# =============================================================================
# RELIABILITY-WEIGHTED ALPHA MODEL
# =============================================================================

class ReliabilityWeightedAlphaModel:
    """
    Reliability-Weighted Alpha Model.
    
    Algorithm:
    1. Compute h^(i) for all assets from calibration windows
    2. At time t, form raw score: s_t^(i) = p_up - p_down
    3. Compute reliability-weighted score: r_t^(i) = h^(i) * s_t^(i)
    4. Center cross-sectionally: α_t^(i) = r_t^(i) - mean(r_t)
    
    This ensures dollar neutrality: sum(α_t) = 0
    """
    
    def __init__(self, 
                 calibration_window: int = 63,
                 min_hit_ratio: float = 0.0,
                 smoothing_halflife: Optional[int] = None):
        """
        Parameters:
        -----------
        calibration_window : int
            Rolling window for hit ratio computation
        min_hit_ratio : float
            Minimum hit ratio to include asset (0 = include all)
        smoothing_halflife : int, optional
            Signal smoothing halflife for turnover control
        """
        self.calibration_window = calibration_window
        self.min_hit_ratio = min_hit_ratio
        self.smoothing_halflife = smoothing_halflife
        
        self.hit_tracker = HitRatioTracker(calibration_window)
        self.prev_alpha = None
        
    def compute_alpha(self, 
                      probabilities: Dict[str, pd.Series],
                      true_classes: Optional[pd.Series] = None,
                      date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Compute reliability-weighted alpha for a single timestamp.
        
        Parameters:
        -----------
        probabilities : Dict[str, pd.Series]
            'up': P(up) per asset
            'down': P(down) per asset
            'hold': P(hold) per asset
        true_classes : pd.Series, optional
            True classes for updating hit ratios (for next period)
        date : pd.Timestamp, optional
            Current date
            
        Returns:
        --------
        pd.Series : Alpha signals (dollar-neutral)
        """
        p_up = probabilities['up']
        p_down = probabilities['down']
        assets = p_up.index
        
        # Step 1: Get predicted classes (argmax)
        prob_df = pd.DataFrame({
            'down': probabilities['down'],
            'hold': probabilities['hold'],
            'up': probabilities['up']
        })
        pred_classes = prob_df.idxmax(axis=1).map({'down': 0, 'hold': 1, 'up': 2})
        
        # Step 2: Get hit ratios
        hit_ratios = self.hit_tracker.get_hit_ratios(assets)
        
        # Step 3: Raw score: s_t = p_up - p_down
        raw_score = p_up - p_down
        
        # Step 4: Reliability-weighted score: r_t = h * s_t
        reliability_score = hit_ratios * raw_score
        
        # Step 5: Filter by min hit ratio
        if self.min_hit_ratio > 0:
            mask = hit_ratios >= self.min_hit_ratio
            reliability_score = reliability_score[mask]
        
        # Step 6: Cross-sectional centering (dollar neutrality)
        # α_t = r_t - mean(r_t)
        alpha = reliability_score - reliability_score.mean()
        
        # Step 7: Optional smoothing
        if self.smoothing_halflife is not None and self.prev_alpha is not None:
            decay = np.exp(-np.log(2) / self.smoothing_halflife)
            common_assets = alpha.index.intersection(self.prev_alpha.index)
            alpha_smooth = alpha.copy()
            alpha_smooth[common_assets] = (
                decay * self.prev_alpha[common_assets] + 
                (1 - decay) * alpha[common_assets]
            )
            alpha = alpha_smooth
        
        self.prev_alpha = alpha.copy()
        
        # Update hit ratios for next period
        if true_classes is not None and date is not None:
            self.hit_tracker.update(date, pred_classes, true_classes)
        
        return alpha
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        hit_ratios = pd.Series(self.hit_tracker.hit_ratios)
        return {
            'mean_hit_ratio': hit_ratios.mean() if len(hit_ratios) > 0 else 0,
            'std_hit_ratio': hit_ratios.std() if len(hit_ratios) > 0 else 0,
            'min_hit_ratio': hit_ratios.min() if len(hit_ratios) > 0 else 0,
            'max_hit_ratio': hit_ratios.max() if len(hit_ratios) > 0 else 0,
            'n_assets_tracked': len(hit_ratios),
        }
    
    def reset(self):
        """Reset model state."""
        self.hit_tracker.reset()
        self.prev_alpha = None


# =============================================================================
# WALK-FORWARD CLASSIFICATION PIPELINE
# =============================================================================

def run_walk_forward_classification(
    features: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    returns: pd.DataFrame,
    model_type: str = 'lightgbm',
    training_window: int = 252,
    retrain_interval: int = 21,
    calibration_window: int = 63,
    decay_halflife: int = 63,
    use_sample_weights: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run walk-forward classification with reliability-weighted alpha.
    
    Parameters:
    -----------
    features : Dict[str, pd.DataFrame]
        Feature DataFrames (name -> dates x assets)
    labels : pd.DataFrame
        3-class labels (dates x assets)
    returns : pd.DataFrame
        Daily returns for backtesting
    model_type : str
        'lightgbm', 'xgboost', or 'mlp'
    training_window : int
        Training window size (days)
    retrain_interval : int
        Retrain every N days
    calibration_window : int
        Window for hit ratio computation
    decay_halflife : int
        Halflife for time-decay sample weights
    use_sample_weights : bool
        Whether to use time-decay weights
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict : Results with predictions, alphas, backtest, metrics
    """
    # Lazy imports
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
    
    # Prepare aligned data
    feature_names = list(features.keys())
    first_feat = features[feature_names[0]]
    all_dates = first_feat.index
    assets = first_feat.columns
    n_dates = len(all_dates)
    n_assets = len(assets)
    n_features = len(feature_names)
    
    if verbose:
        print(f"Walk-Forward Classification Pipeline")
        print(f"  Dates: {n_dates}, Assets: {n_assets}, Features: {n_features}")
        print(f"  Training window: {training_window}, Retrain interval: {retrain_interval}")
    
    # Pre-stack features into 3D array: (dates, assets, features)
    X_3d = np.zeros((n_dates, n_assets, n_features))
    for i, fname in enumerate(feature_names):
        X_3d[:, :, i] = features[fname].values
    
    # Labels
    y_2d = labels.values
    
    # Initialize alpha model
    alpha_model = ReliabilityWeightedAlphaModel(
        calibration_window=calibration_window,
        smoothing_halflife=5
    )
    
    # Storage
    all_alphas = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    all_probs_up = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    all_probs_down = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    all_probs_hold = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    # Walk-forward loop
    start_idx = training_window + 50
    current_model = None
    
    for t_idx in range(start_idx, n_dates):
        day_offset = t_idx - start_idx
        
        # Retrain check
        if day_offset % retrain_interval == 0 or current_model is None:
            # Training data
            train_start = max(0, t_idx - training_window)
            train_end = t_idx
            
            # Flatten training data
            X_train_list = []
            y_train_list = []
            weight_list = []
            
            train_dates = all_dates[train_start:train_end]
            
            # Compute sample weights
            if use_sample_weights:
                sample_weights = compute_sample_weights(
                    train_dates, 
                    decay_halflife=decay_halflife,
                    recent_boost_days=retrain_interval
                )
            
            for t in range(train_start, train_end):
                X_day = X_3d[t]
                y_day = y_2d[t]
                
                # Filter valid samples
                valid = ~(np.isnan(X_day).any(axis=1) | np.isnan(y_day))
                
                if valid.sum() > 0:
                    X_train_list.append(X_day[valid])
                    y_train_list.append(y_day[valid])
                    
                    if use_sample_weights:
                        w = sample_weights[t - train_start]
                        weight_list.extend([w] * valid.sum())
            
            if len(X_train_list) == 0:
                continue
            
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            weights = np.array(weight_list) if use_sample_weights else None
            
            # Train model
            if model_type == 'lightgbm' and HAS_LGB:
                current_model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_class=3,
                    objective='multiclass',
                    verbose=-1,
                    n_jobs=-1
                )
                current_model.fit(X_train, y_train, sample_weight=weights)
            else:
                # Fallback to sklearn
                from sklearn.linear_model import LogisticRegression
                current_model = LogisticRegression(
                    max_iter=500,
                    C=1.0,
                    solver='lbfgs'  # Supports multiclass natively
                )
                current_model.fit(X_train, y_train, sample_weight=weights)
            
            if verbose and day_offset % (retrain_interval * 5) == 0:
                print(f"  Retrained at t={t_idx}/{n_dates} ({all_dates[t_idx].strftime('%Y-%m-%d')})")
        
        # Predict for current day
        X_pred = X_3d[t_idx]
        valid_mask = ~np.isnan(X_pred).any(axis=1)
        
        if valid_mask.sum() > 0 and current_model is not None:
            # Get probabilities
            probs = current_model.predict_proba(X_pred[valid_mask])
            
            # Store probabilities
            p_down = pd.Series(index=assets[valid_mask], data=probs[:, 0])
            p_hold = pd.Series(index=assets[valid_mask], data=probs[:, 1])
            p_up = pd.Series(index=assets[valid_mask], data=probs[:, 2])
            
            all_probs_down.loc[all_dates[t_idx], assets[valid_mask]] = probs[:, 0]
            all_probs_hold.loc[all_dates[t_idx], assets[valid_mask]] = probs[:, 1]
            all_probs_up.loc[all_dates[t_idx], assets[valid_mask]] = probs[:, 2]
            
            # Get true labels for hit ratio update (from previous day)
            if t_idx > 0:
                prev_labels = pd.Series(index=assets, data=y_2d[t_idx-1])
            else:
                prev_labels = None
            
            # Compute alpha
            alpha = alpha_model.compute_alpha(
                probabilities={'up': p_up, 'down': p_down, 'hold': p_hold},
                true_classes=prev_labels,
                date=all_dates[t_idx]
            )
            
            all_alphas.loc[all_dates[t_idx], alpha.index] = alpha.values
    
    # Clean up
    all_alphas = all_alphas.dropna(how='all')
    
    # Convert alpha to weights (normalize)
    weights = all_alphas.div(all_alphas.abs().sum(axis=1), axis=0)
    weights = weights.fillna(0)
    
    # Clip extreme weights
    weights = weights.clip(-0.1, 0.1)
    weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0)
    
    if verbose:
        print(f"\n✅ Walk-forward completed")
        print(f"   Alpha shape: {all_alphas.shape}")
        diag = alpha_model.get_diagnostics()
        print(f"   Mean hit ratio: {diag['mean_hit_ratio']:.4f}")
        print(f"   Assets tracked: {diag['n_assets_tracked']}")
    
    return {
        'alphas': all_alphas,
        'weights': weights,
        'probabilities': {
            'up': all_probs_up,
            'down': all_probs_down,
            'hold': all_probs_hold,
        },
        'diagnostics': alpha_model.get_diagnostics(),
        'alpha_model': alpha_model,
    }


# =============================================================================
# INFORMATION COEFFICIENT (IC) COMPUTATION
# =============================================================================

def compute_ic(predictions: pd.DataFrame, 
               forward_returns: pd.DataFrame) -> pd.Series:
    """
    Compute Information Coefficient (rank correlation) per day.
    
    IC_t = rank_corr(signal_t, return_{t+1})
    """
    common_idx = predictions.index.intersection(forward_returns.index)
    
    ic_series = pd.Series(index=common_idx, dtype=float)
    
    for date in common_idx:
        pred = predictions.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()
        common = pred.index.intersection(fwd.index)
        
        if len(common) > 10:
            ic_series[date] = pred[common].corr(fwd[common], method='spearman')
    
    return ic_series


def compute_hit_rate(predictions: pd.DataFrame,
                     labels: pd.DataFrame) -> float:
    """
    Compute overall hit rate (3-class accuracy).
    """
    pred_classes = predictions.apply(lambda x: 2 if x > 0 else (0 if x < 0 else 1), axis=0)
    
    common_idx = predictions.index.intersection(labels.index)
    
    hits = 0
    total = 0
    
    for date in common_idx:
        pred = pred_classes.loc[date].dropna()
        true = labels.loc[date].dropna()
        common = pred.index.intersection(true.index)
        
        if len(common) > 0:
            hits += (pred[common] == true[common]).sum()
            total += len(common)
    
    return hits / total if total > 0 else 0
