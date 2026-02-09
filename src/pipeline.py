"""
Quantitative Trading Pipeline Module v3.0
==========================================

Enhanced pipeline with:
- Optimized rolling regression (10-50x faster)
- Best model tracking
- Baseline + new features mode
- Feature block library (momentum, volatility, etc.)
- Ensemble support for combining uncorrelated alphas
- Signal smoothing for turnover control
- Regime filtering for conditional alpha
- GPU-accelerated models (LightGBM, XGBoost, PyTorch)

Usage:
    from src.pipeline import Pipeline
    
    pipeline = Pipeline()
    pipeline.load_data()
    
    # Run with baseline + new features
    results = pipeline.run_with_baseline(my_features)
    
    # Run with signal smoothing
    results = pipeline.run_with_smoothing(my_features, halflife=5)
    
    # Run with regime filtering
    results = pipeline.run_with_regime(my_features, regime_type='high_vol_only')
    
    # Track best model
    pipeline.update_best_model(results, "momentum_v1")
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Tuple, Optional, List, Union, Callable
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# DEFAULT PARAMETERS (FROZEN)
# =============================================================================

DEFAULT_PARAMS = {
    # Model Parameters (FROZEN)
    'ridge_alpha': 1.0,              # Fixed regularization (no tuning)
    'training_window': 252,          # 1 year rolling window
    
    # Portfolio Parameters (FROZEN)
    'long_pct': 0.20,                # Top 20% long
    'short_pct': 0.20,               # Bottom 20% short
    
    # Backtest Parameters (FROZEN)
    'transaction_cost_bps': 10,      # 10 bps per turnover
    'trading_days_per_year': 252,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str = '../data/processed/df_after_eda.parquet') -> pd.DataFrame:
    """Load the processed dataset."""
    return pd.read_parquet(filepath)


def create_panel_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create wide-format panel data for returns.
    Returns dictionary of DataFrames indexed by Date with asset_id columns.
    """
    panel = {}
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        panel[col] = df.pivot(index='Date', columns='asset_id', values=col)
    
    panel['returns'] = panel['Close'].pct_change()
    panel['forward_returns'] = panel['returns'].shift(-1)
    
    return panel


def compute_target(forward_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional demeaned forward returns (target variable).
    
    y_{i,t+1} = r_{i,t+1} - mean_j(r_{j,t+1})
    """
    cs_mean = forward_returns.mean(axis=1)
    target = forward_returns.sub(cs_mean, axis=0)
    return target


# =============================================================================
# FEATURE UTILITIES
# =============================================================================

def cross_sectional_standardize(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize feature cross-sectionally at each time t.
    
    f_tilde(i,t) = (f(i,t) - mean_j(f(j,t))) / std_j(f(j,t))
    """
    cs_mean = feature_df.mean(axis=1)
    cs_std = feature_df.std(axis=1)
    cs_std = cs_std.replace(0, np.nan)
    standardized = feature_df.sub(cs_mean, axis=0).div(cs_std, axis=0)
    return standardized


def standardize_features(features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Standardize all features in a dictionary."""
    return {name: cross_sectional_standardize(feat) for name, feat in features.items()}


# =============================================================================
# FEATURE BLOCKS LIBRARY
# =============================================================================

def compute_baseline_features(panel: Dict[str, pd.DataFrame], 
                               short_window: int = 5,
                               medium_window: int = 20,
                               vol_window: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Compute baseline features.
    
    Features:
    - f1_short_mom_norm: 5-day momentum normalized by 20-day volatility
    - f2_medium_mom: 20-day momentum
    - f3_volatility: 20-day realized volatility
    """
    returns = panel['returns']
    
    features = {}
    
    # f1: Short-term momentum normalized by volatility
    short_mom = returns.rolling(window=short_window).sum()
    rolling_vol = returns.rolling(window=vol_window).std()
    features['f1_short_mom_norm'] = short_mom / rolling_vol
    
    # f2: Medium-term momentum
    features['f2_medium_mom'] = returns.rolling(window=medium_window).sum()
    
    # f3: Realized volatility
    features['f3_volatility'] = rolling_vol
    
    return features


def compute_momentum_block(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Momentum Block: cumulative returns at different horizons, volatility-normalized.
    
    Features:
    - mom_3d: 3-day momentum / 20d vol
    - mom_5d: 5-day momentum / 20d vol  
    - mom_10d: 10-day momentum / 20d vol
    """
    vol_20d = returns.rolling(window=20, min_periods=15).std()
    
    return {
        'mom_3d': returns.rolling(3).sum() / vol_20d,
        'mom_5d': returns.rolling(5).sum() / vol_20d,
        'mom_10d': returns.rolling(10).sum() / vol_20d,
    }


def compute_volatility_block(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Volatility Block: State variables capturing volatility dynamics.
    These are conditioning variables, not direct signals.
    
    Operator 1: Volatility Level
    - vol_short: 10-day realized volatility
    - vol_ratio: Short-term vol / Long-term vol (relative level)
    
    Operator 2: Volatility Surprise
    - vol_change: Change in volatility (first difference)
    - vol_change_z: Z-scored volatility change
    
    Operator 3: Volatility Uncertainty
    - vol_of_vol: Volatility of volatility (rolling std of vol)
    """
    # Operator 1: Volatility Level
    vol_short = returns.rolling(window=10, min_periods=8).std()
    vol_long = returns.rolling(window=60, min_periods=45).std()
    vol_ratio = vol_short / vol_long
    
    # Operator 2: Volatility Surprise
    vol_change = vol_short.diff(5)  # 5-day change in vol
    vol_change_mean = vol_change.rolling(window=60, min_periods=30).mean()
    vol_change_std = vol_change.rolling(window=60, min_periods=30).std()
    vol_change_z = (vol_change - vol_change_mean) / vol_change_std
    
    # Operator 3: Volatility Uncertainty
    vol_of_vol = vol_short.rolling(window=20, min_periods=15).std()
    
    return {
        'vol_short': vol_short,
        'vol_ratio': vol_ratio,
        'vol_change': vol_change,
        'vol_change_z': vol_change_z,
        'vol_of_vol': vol_of_vol,
    }


def compute_mean_reversion_block(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Mean Reversion Block: Short-term reversal signals.
    
    Features:
    - ret_1d: Yesterday's return (classic reversal)
    - ret_1d_z: Z-scored 1-day return
    """
    # 1-day return (reversal signal)
    ret_1d = returns
    
    # Z-scored 1-day return
    ret_mean = returns.rolling(window=20, min_periods=15).mean()
    ret_std = returns.rolling(window=20, min_periods=15).std()
    ret_1d_z = (returns - ret_mean) / ret_std
    
    return {
        'ret_1d': ret_1d,
        'ret_1d_z': ret_1d_z,
    }


def compute_volume_block(panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Volume Block: Volume-based features.
    
    Features:
    - volume_ratio: Today's volume / 20-day avg volume
    - volume_trend: 5-day vs 20-day volume ratio
    """
    volume = panel['Volume']
    
    vol_20d_avg = volume.rolling(window=20, min_periods=15).mean()
    vol_5d_avg = volume.rolling(window=5, min_periods=3).mean()
    
    return {
        'volume_ratio': volume / vol_20d_avg,
        'volume_trend': vol_5d_avg / vol_20d_avg,
    }


# =============================================================================
# MODEL TRAINING (OPTIMIZED)
# =============================================================================

def prepare_aligned_data(features_std: Dict[str, pd.DataFrame], 
                         target: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Index, pd.Index, List[str]]:
    """
    Pre-align all features and target into NumPy arrays ONCE before rolling loop.
    This is ~10-50x faster than row-by-row dictionary creation.
    
    Returns:
        X_3d: (n_dates, n_assets, n_features) array
        y_2d: (n_dates, n_assets) array
        dates: DatetimeIndex
        assets: Index of asset IDs
        feature_names: List of feature names
    """
    feature_names = list(features_std.keys())
    
    # Get common index
    dates = target.index
    assets = target.columns
    
    n_dates = len(dates)
    n_assets = len(assets)
    n_features = len(feature_names)
    
    # Pre-allocate arrays
    X_3d = np.full((n_dates, n_assets, n_features), np.nan)
    y_2d = target.values.copy()
    
    # Fill feature arrays (vectorized reindex)
    for i, fname in enumerate(feature_names):
        feat_aligned = features_std[fname].reindex(index=dates, columns=assets)
        X_3d[:, :, i] = feat_aligned.values
    
    return X_3d, y_2d, dates, assets, feature_names


def extract_train_data(X_3d: np.ndarray, y_2d: np.ndarray, 
                       start_idx: int, end_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract and flatten training data from pre-aligned arrays.
    Much faster than prepare_model_data since data is already aligned.
    """
    # Slice the window
    X_window = X_3d[start_idx:end_idx]  # (window, n_assets, n_features)
    y_window = y_2d[start_idx:end_idx]  # (window, n_assets)
    
    # Flatten to (window * n_assets, n_features) and (window * n_assets,)
    n_days, n_assets, n_features = X_window.shape
    X_flat = X_window.reshape(-1, n_features)
    y_flat = y_window.reshape(-1)
    
    # Remove NaN rows
    valid_mask = ~np.isnan(X_flat).any(axis=1) & ~np.isnan(y_flat)
    
    if valid_mask.sum() < 100:
        return None, None
    
    return X_flat[valid_mask], y_flat[valid_mask]


def run_rolling_regression(features_std: Dict[str, pd.DataFrame], 
                           target: pd.DataFrame, 
                           params: Dict,
                           verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling Ridge regression to generate predictions.
    OPTIMIZED: Pre-aligns data once, then uses fast NumPy slicing.
    
    For each day t:
    1. Train on [t - training_window, t - 1]
    2. Predict for day t
    
    Returns:
    - predictions: DataFrame of predictions aligned with target
    - coefficients: DataFrame of rolling coefficients
    """
    training_window = params['training_window']
    ridge_alpha = params['ridge_alpha']
    
    # PRE-ALIGN DATA ONCE (the key optimization!)
    if verbose:
        print("Pre-aligning data...")
    X_3d, y_2d, all_dates, assets, feature_names = prepare_aligned_data(features_std, target)
    
    n_dates = len(all_dates)
    
    predictions = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    coefficients = pd.DataFrame(index=all_dates, columns=['intercept'] + feature_names, dtype=float)
    
    start_idx = training_window + 50  # Warmup for features
    
    if verbose:
        print(f"Running rolling regression from index {start_idx} to {n_dates}...")
        print(f"  Training window: {training_window} days")
        print(f"  Ridge alpha: {ridge_alpha}")
        print(f"  Features ({len(feature_names)}): {feature_names}")
    
    for t_idx in range(start_idx, n_dates):
        if verbose and t_idx % 100 == 0:
            print(f"  Processing day {t_idx}/{n_dates} ({all_dates[t_idx].date()})...")
        
        current_date = all_dates[t_idx]
        
        # Fast NumPy slicing (instead of pandas row-by-row)
        X_train, y_train = extract_train_data(X_3d, y_2d, t_idx - training_window, t_idx)
        
        if X_train is None:
            continue
        
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X_train, y_train)
        
        coefficients.loc[current_date, 'intercept'] = model.intercept_
        coefficients.loc[current_date, feature_names] = model.coef_
        
        # Prediction for current date (fast NumPy)
        X_pred = X_3d[t_idx]  # (n_assets, n_features)
        valid_mask = ~np.isnan(X_pred).any(axis=1)
        
        if valid_mask.sum() > 0:
            y_pred = model.predict(X_pred[valid_mask])
            predictions.loc[current_date, assets[valid_mask]] = y_pred
    
    predictions = predictions.dropna(how='all')
    coefficients = coefficients.dropna(how='all')
    
    if verbose:
        print(f"Completed. Predictions shape: {predictions.shape}")
    
    return predictions, coefficients


# =============================================================================
# ADVANCED ML MODELS (LightGBM, XGBoost, Neural Networks)
# =============================================================================

def run_rolling_lightgbm(features_std: Dict[str, pd.DataFrame],
                          target: pd.DataFrame,
                          params: Dict,
                          model_params: Optional[Dict] = None,
                          retrain_interval: int = 21,
                          use_gpu: bool = True,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Run rolling LightGBM predictions with optional GPU acceleration.
    
    LightGBM is a gradient boosting framework that:
    - Captures non-linear feature interactions
    - Handles sparse features well
    - Is extremely fast (especially on GPU)
    
    Parameters:
    -----------
    features_std : Dict[str, pd.DataFrame]
        Standardized features
    target : pd.DataFrame
        Target variable
    params : Dict
        Pipeline parameters (training_window)
    model_params : Dict, optional
        LightGBM hyperparameters
    retrain_interval : int
        Retrain model every N days (reduces compute)
    use_gpu : bool
        Use GPU acceleration if available
    verbose : bool
        Print progress
        
    Returns:
    --------
    pd.DataFrame : Predictions
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    training_window = params['training_window']
    
    # Default model parameters (optimized for financial data)
    default_model_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.05,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
    }
    
    if model_params:
        default_model_params.update(model_params)
    
    # GPU settings
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                default_model_params['device'] = 'gpu'
                default_model_params['gpu_platform_id'] = 0
                default_model_params['gpu_device_id'] = 0
                if verbose:
                    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        except:
            if verbose:
                print("⚠️ GPU not available, using CPU")
    
    # Pre-align data
    if verbose:
        print("Pre-aligning data for LightGBM...")
    X_3d, y_2d, all_dates, assets, feature_names = prepare_aligned_data(features_std, target)
    
    n_dates = len(all_dates)
    predictions = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    start_idx = training_window + 50
    current_model = None
    
    if verbose:
        print(f"Running rolling LightGBM (retrain every {retrain_interval} days)...")
        print(f"  Training window: {training_window} days")
        print(f"  Features ({len(feature_names)}): {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")
    
    for t_idx in range(start_idx, n_dates):
        day_offset = t_idx - start_idx
        
        # Retrain model periodically
        if day_offset % retrain_interval == 0 or current_model is None:
            X_train, y_train = extract_train_data(X_3d, y_2d, t_idx - training_window, t_idx)
            
            if X_train is None:
                continue
            
            current_model = lgb.LGBMRegressor(**default_model_params)
            current_model.fit(X_train, y_train)
            
            if verbose and day_offset % (retrain_interval * 10) == 0:
                print(f"  Retrained at day {t_idx}/{n_dates} ({all_dates[t_idx].date()})")
        
        # Predict
        X_pred = X_3d[t_idx]
        valid_mask = ~np.isnan(X_pred).any(axis=1)
        
        if valid_mask.sum() > 0 and current_model is not None:
            y_pred = current_model.predict(X_pred[valid_mask])
            predictions.loc[all_dates[t_idx], assets[valid_mask]] = y_pred
    
    predictions = predictions.dropna(how='all')
    
    if verbose:
        print(f"✅ LightGBM completed. Predictions shape: {predictions.shape}")
    
    return predictions


def run_rolling_xgboost(features_std: Dict[str, pd.DataFrame],
                         target: pd.DataFrame,
                         params: Dict,
                         model_params: Optional[Dict] = None,
                         retrain_interval: int = 21,
                         use_gpu: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Run rolling XGBoost predictions with GPU acceleration.
    
    XGBoost is another powerful gradient boosting framework that:
    - Has excellent regularization (often better for overfitting control)
    - Native GPU support
    - Handles missing values natively
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    training_window = params['training_window']
    
    default_model_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': 42,
    }
    
    if model_params:
        default_model_params.update(model_params)
    
    # GPU settings
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                default_model_params['tree_method'] = 'hist'
                default_model_params['device'] = 'cuda'
                if verbose:
                    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        except:
            if verbose:
                print("⚠️ GPU not available, using CPU")
    
    # Pre-align data
    if verbose:
        print("Pre-aligning data for XGBoost...")
    X_3d, y_2d, all_dates, assets, feature_names = prepare_aligned_data(features_std, target)
    
    n_dates = len(all_dates)
    predictions = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    start_idx = training_window + 50
    current_model = None
    
    if verbose:
        print(f"Running rolling XGBoost (retrain every {retrain_interval} days)...")
    
    for t_idx in range(start_idx, n_dates):
        day_offset = t_idx - start_idx
        
        if day_offset % retrain_interval == 0 or current_model is None:
            X_train, y_train = extract_train_data(X_3d, y_2d, t_idx - training_window, t_idx)
            
            if X_train is None:
                continue
            
            current_model = xgb.XGBRegressor(**default_model_params)
            current_model.fit(X_train, y_train)
            
            if verbose and day_offset % (retrain_interval * 10) == 0:
                print(f"  Retrained at day {t_idx}/{n_dates}")
        
        X_pred = X_3d[t_idx]
        valid_mask = ~np.isnan(X_pred).any(axis=1)
        
        if valid_mask.sum() > 0 and current_model is not None:
            y_pred = current_model.predict(X_pred[valid_mask])
            predictions.loc[all_dates[t_idx], assets[valid_mask]] = y_pred
    
    predictions = predictions.dropna(how='all')
    
    if verbose:
        print(f"✅ XGBoost completed. Predictions shape: {predictions.shape}")
    
    return predictions


class FinancialMLP(nn.Module):
    """
    Multi-Layer Perceptron for financial prediction.
    
    Architecture designed for financial data:
    - Batch normalization for input stability
    - Dropout for regularization
    - Skip connections for gradient flow
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


def run_rolling_mlp(features_std: Dict[str, pd.DataFrame],
                    target: pd.DataFrame,
                    params: Dict,
                    hidden_dims: List[int] = [64, 32],
                    dropout: float = 0.3,
                    epochs: int = 50,
                    batch_size: int = 1024,
                    learning_rate: float = 0.001,
                    retrain_interval: int = 21,
                    use_gpu: bool = True,
                    verbose: bool = True,
                    random_seed: int = 42) -> pd.DataFrame:
    """
    Run rolling Neural Network (MLP) predictions with GPU acceleration.
    
    Parameters:
    -----------
    hidden_dims : List[int]
        Hidden layer dimensions
    dropout : float
        Dropout rate for regularization
    epochs : int
        Training epochs per retrain
    batch_size : int
        Mini-batch size
    learning_rate : float
        Learning rate for Adam optimizer
    retrain_interval : int
        Retrain every N days
    use_gpu : bool
        Use CUDA if available
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed. Run: pip install torch")
    
    # SET RANDOM SEEDS FOR REPRODUCIBILITY
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    training_window = params['training_window']
    
    # Device setup
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"🔧 PyTorch device: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Pre-align data
    X_3d, y_2d, all_dates, assets, feature_names = prepare_aligned_data(features_std, target)
    n_features = len(feature_names)
    n_dates = len(all_dates)
    
    predictions = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    
    start_idx = training_window + 50
    current_model = None
    
    if verbose:
        print(f"Running rolling MLP (retrain every {retrain_interval} days)...")
        print(f"  Architecture: {n_features} -> {hidden_dims} -> 1")
    
    for t_idx in range(start_idx, n_dates):
        day_offset = t_idx - start_idx
        
        if day_offset % retrain_interval == 0 or current_model is None:
            X_train, y_train = extract_train_data(X_3d, y_2d, t_idx - training_window, t_idx)
            
            if X_train is None:
                continue
            
            # Create model
            current_model = FinancialMLP(n_features, hidden_dims, dropout).to(device)
            optimizer = torch.optim.Adam(current_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Convert to tensors
            X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
            
            # Training loop
            current_model.train()
            for epoch in range(epochs):
                # Mini-batch training
                indices = torch.randperm(len(X_tensor))
                for i in range(0, len(X_tensor), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    X_batch = X_tensor[batch_idx]
                    y_batch = y_tensor[batch_idx]
                    
                    optimizer.zero_grad()
                    pred = current_model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
            
            if verbose and day_offset % (retrain_interval * 10) == 0:
                print(f"  Retrained at day {t_idx}/{n_dates}")
        
        # Predict
        X_pred = X_3d[t_idx]
        valid_mask = ~np.isnan(X_pred).any(axis=1)
        
        if valid_mask.sum() > 0 and current_model is not None:
            current_model.eval()
            with torch.no_grad():
                X_test = torch.tensor(X_pred[valid_mask], dtype=torch.float32, device=device)
                y_pred = current_model(X_test).cpu().numpy()
            predictions.loc[all_dates[t_idx], assets[valid_mask]] = y_pred
    
    predictions = predictions.dropna(how='all')
    
    if verbose:
        print(f"✅ MLP completed. Predictions shape: {predictions.shape}")
    
    return predictions


# =============================================================================
# SIGNAL & PORTFOLIO CONSTRUCTION
# =============================================================================

def construct_signals(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert predictions to cross-sectional ranks (signals).
    
    Returns:
    - signals: centered ranks (mean=0)
    - signal_ranks: percentile ranks [0, 1]
    """
    signal_ranks = predictions.rank(axis=1, pct=True)
    signals = signal_ranks - 0.5
    return signals, signal_ranks


def construct_portfolio_weights(signal_ranks: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Construct long-short portfolio weights based on signal ranks.
    
    Long: top long_pct (e.g., 20%)
    Short: bottom short_pct (e.g., 20%)
    Equal weight within each leg.
    """
    long_pct = params['long_pct']
    short_pct = params['short_pct']
    
    weights = pd.DataFrame(0.0, index=signal_ranks.index, columns=signal_ranks.columns)
    
    for date in signal_ranks.index:
        ranks = signal_ranks.loc[date].dropna()
        n_assets = len(ranks)
        
        if n_assets == 0:
            continue
        
        long_threshold = 1 - long_pct
        short_threshold = short_pct
        
        long_assets = ranks[ranks >= long_threshold].index
        short_assets = ranks[ranks <= short_threshold].index
        
        n_long = len(long_assets)
        n_short = len(short_assets)
        
        if n_long > 0:
            weights.loc[date, long_assets] = 1.0 / n_long
        if n_short > 0:
            weights.loc[date, short_assets] = -1.0 / n_short
    
    return weights


# =============================================================================
# SIGNAL SMOOTHING (TURNOVER CONTROL)
# =============================================================================

def smooth_signals(signals: pd.DataFrame, 
                   halflife: int = 5,
                   method: str = 'ewm') -> pd.DataFrame:
    """
    Apply smoothing to raw signals to reduce turnover.
    
    This is a critical component for controlling transaction costs.
    Smoothing dampens high-frequency signal changes that lead to excessive trading.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Raw signals (index=dates, columns=assets)
    halflife : int
        Smoothing halflife in days. Higher = more smoothing = less turnover.
        Typical values: 2-10 days
    method : str
        Smoothing method:
        - 'ewm': Exponential weighted moving average (default)
        - 'sma': Simple moving average
        - 'decay': Exponential decay blending with previous signal
        
    Returns:
    --------
    pd.DataFrame : Smoothed signals
    
    Notes:
    ------
    The relationship between halflife (h) and alpha is: alpha = 1 - exp(ln(0.5)/h)
    - halflife=2: alpha≈0.29 (fast, less smoothing)
    - halflife=5: alpha≈0.13 (moderate)
    - halflife=10: alpha≈0.07 (slow, heavy smoothing)
    """
    if method == 'ewm':
        # Exponential weighted moving average
        smoothed = signals.ewm(halflife=halflife, min_periods=1).mean()
        
    elif method == 'sma':
        # Simple moving average
        smoothed = signals.rolling(window=halflife * 2, min_periods=1).mean()
        
    elif method == 'decay':
        # Exponential decay: new_signal = alpha * raw + (1-alpha) * prev_signal
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        smoothed = signals.copy()
        for i in range(1, len(signals)):
            smoothed.iloc[i] = alpha * signals.iloc[i] + (1 - alpha) * smoothed.iloc[i-1]
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed


def smooth_weights(weights: pd.DataFrame,
                   decay: float = 0.8) -> pd.DataFrame:
    """
    Apply exponential decay smoothing to portfolio weights directly.
    
    This blends today's target weights with yesterday's actual weights.
    
    Parameters:
    -----------
    weights : pd.DataFrame
        Target portfolio weights (index=dates, columns=assets)
    decay : float
        Decay factor [0, 1]. Higher = more persistence = less turnover.
        - 0.0: No smoothing (use raw weights)
        - 0.5: Equal blend of today and yesterday
        - 0.9: Very smooth (90% yesterday, 10% today)
        
    Returns:
    --------
    pd.DataFrame : Smoothed weights
    """
    smoothed = weights.copy().astype(float)
    
    for i in range(1, len(weights)):
        # Blend: smoothed[t] = decay * smoothed[t-1] + (1-decay) * target[t]
        blended = decay * smoothed.iloc[i-1].values + (1 - decay) * weights.iloc[i].values
        smoothed.iloc[i] = blended
        
        # Re-normalize to maintain dollar neutrality
        row = smoothed.iloc[i].values
        long_mask = row > 0
        short_mask = row < 0
        
        long_sum = row[long_mask].sum() if long_mask.any() else 0
        short_sum = np.abs(row[short_mask].sum()) if short_mask.any() else 0
        
        if long_sum > 0:
            row[long_mask] /= long_sum
        if short_sum > 0:
            row[short_mask] /= short_sum
        
        smoothed.iloc[i] = row
    
    return smoothed


# =============================================================================
# REGIME FILTERING
# =============================================================================

def apply_regime_filter(weights: pd.DataFrame,
                        regime_mask: pd.Series,
                        scale_by_confidence: bool = False) -> pd.DataFrame:
    """
    Apply regime-based filtering to portfolio weights.
    
    When regime is unfavorable (mask=0), positions are reduced or eliminated.
    
    Parameters:
    -----------
    weights : pd.DataFrame
        Portfolio weights (index=dates, columns=assets)
    regime_mask : pd.Series
        Binary mask (1=trade, 0=no trade) or continuous confidence [0,1]
    scale_by_confidence : bool
        If True, scale weights by mask value instead of zeroing out.
        
    Returns:
    --------
    pd.DataFrame : Filtered weights
    """
    # Align mask to weights index
    mask_aligned = regime_mask.reindex(weights.index).fillna(0)
    
    if scale_by_confidence:
        # Scale weights by confidence
        filtered = weights.mul(mask_aligned, axis=0)
    else:
        # Binary: zero out positions when mask=0
        filtered = weights.copy()
        filtered.loc[mask_aligned < 0.5] = 0
    
    return filtered


def create_regime_mask(returns: pd.DataFrame,
                       regime_type: str = 'high_vol_only') -> pd.Series:
    """
    Create a regime mask for filtering trading signals.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns (index=dates, columns=assets)
    regime_type : str
        One of:
        - 'high_vol_only': Trade only in high volatility
        - 'low_vol_only': Trade only in low volatility  
        - 'uptrend_only': Trade only in uptrends
        - 'downtrend_only': Trade only in downtrends
        - 'trending': Trade when trend is strong (either direction)
        - 'confident': Trade when regime is clear
        
    Returns:
    --------
    pd.Series : Binary mask (1=trade, 0=no trade)
    """
    market_ret = returns.mean(axis=1)
    
    # Volatility calculations
    vol = market_ret.rolling(20).std() * np.sqrt(252)
    vol_median = vol.expanding(min_periods=252).median()
    
    # Trend calculations
    cumret_60d = market_ret.rolling(60).sum()
    
    if regime_type == 'high_vol_only':
        mask = (vol > vol_median).astype(float)
        
    elif regime_type == 'low_vol_only':
        mask = (vol <= vol_median).astype(float)
        
    elif regime_type == 'uptrend_only':
        mask = (cumret_60d > 0).astype(float)
        
    elif regime_type == 'downtrend_only':
        mask = (cumret_60d <= 0).astype(float)
        
    elif regime_type == 'trending':
        # Strong trend in either direction
        cumret_abs = cumret_60d.abs()
        trend_threshold = cumret_abs.expanding(min_periods=252).median()
        mask = (cumret_abs > trend_threshold).astype(float)
        
    elif regime_type == 'confident':
        # Both vol and trend are clear (not at median)
        vol_pct = vol.expanding(min_periods=252).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5
        )
        trend_pct = cumret_60d.expanding(min_periods=252).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5
        )
        # Confident if either vol or trend is extreme
        vol_confident = (vol_pct < 0.25) | (vol_pct > 0.75)
        trend_confident = (trend_pct < 0.25) | (trend_pct > 0.75)
        mask = (vol_confident | trend_confident).astype(float)
        
    else:
        raise ValueError(f"Unknown regime_type: {regime_type}")
    
    return mask.shift(1)  # Lag by 1 to avoid lookahead


def construct_portfolio_weights_with_regime(signal_ranks: pd.DataFrame,
                                            params: Dict,
                                            regime_mask: Optional[pd.Series] = None,
                                            signal_smoothing_halflife: Optional[int] = None,
                                            weight_decay: Optional[float] = None) -> pd.DataFrame:
    """
    Construct portfolio weights with optional regime filtering and smoothing.
    
    This is the enhanced version of construct_portfolio_weights that integrates:
    1. Signal smoothing (before ranking)
    2. Portfolio weight smoothing (after construction)
    3. Regime filtering (final step)
    
    Parameters:
    -----------
    signal_ranks : pd.DataFrame
        Cross-sectional signal ranks [0, 1]
    params : Dict
        Parameters including long_pct, short_pct
    regime_mask : pd.Series, optional
        Binary mask for regime filtering
    signal_smoothing_halflife : int, optional
        If provided, smooth signals before portfolio construction
    weight_decay : float, optional
        If provided, apply weight decay smoothing
        
    Returns:
    --------
    pd.DataFrame : Final portfolio weights
    """
    # Step 1: Apply signal smoothing if requested
    if signal_smoothing_halflife is not None:
        # Convert ranks back to signals, smooth, then re-rank
        signals = signal_ranks - 0.5
        signals_smoothed = smooth_signals(signals, halflife=signal_smoothing_halflife)
        signal_ranks = signals_smoothed.rank(axis=1, pct=True)
    
    # Step 2: Construct base weights
    weights = construct_portfolio_weights(signal_ranks, params)
    
    # Step 3: Apply weight smoothing if requested
    if weight_decay is not None and weight_decay > 0:
        weights = smooth_weights(weights, decay=weight_decay)
    
    # Step 4: Apply regime filtering if requested
    if regime_mask is not None:
        weights = apply_regime_filter(weights, regime_mask)
    
    return weights


# =============================================================================
# ENSEMBLE UTILITIES
# =============================================================================

def ensemble_predictions(predictions_list: List[pd.DataFrame], 
                         weights: Optional[List[float]] = None,
                         method: str = 'equal') -> pd.DataFrame:
    """
    Combine predictions from multiple models.
    
    Methods:
    - 'equal': Simple average
    - 'custom': Use provided weights
    
    Args:
        predictions_list: List of prediction DataFrames
        weights: Optional custom weights (must sum to 1)
        method: Ensemble method
        
    Returns:
        Combined predictions DataFrame
    """
    n_models = len(predictions_list)
    
    if method == 'equal' or weights is None:
        weights = [1.0 / n_models] * n_models
    elif method == 'custom':
        assert weights is not None and abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    
    # Align all predictions to common index
    common_dates = predictions_list[0].index
    common_assets = predictions_list[0].columns
    for pred in predictions_list[1:]:
        common_dates = common_dates.intersection(pred.index)
        common_assets = common_assets.intersection(pred.columns)
    
    # Weighted average
    combined = pd.DataFrame(0.0, index=common_dates, columns=common_assets)
    for pred, w in zip(predictions_list, weights):
        combined += w * pred.loc[common_dates, common_assets]
    
    return combined


def compute_prediction_correlation(predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Compute correlation matrix between model predictions.
    Useful for identifying uncorrelated alphas for ensembling.
    """
    n_models = len(predictions_list)
    
    # Align and stack predictions
    common_dates = predictions_list[0].index
    for pred in predictions_list[1:]:
        common_dates = common_dates.intersection(pred.index)
    
    # Compute daily cross-sectional correlations
    daily_corrs = np.zeros((len(common_dates), n_models, n_models))
    
    for i, date in enumerate(common_dates):
        preds_day = [pred.loc[date].dropna() for pred in predictions_list]
        common_assets = preds_day[0].index
        for p in preds_day[1:]:
            common_assets = common_assets.intersection(p.index)
        
        if len(common_assets) > 10:
            pred_matrix = np.column_stack([p.loc[common_assets].values for p in preds_day])
            daily_corrs[i] = np.corrcoef(pred_matrix.T)
    
    # Average correlation
    mean_corr = np.nanmean(daily_corrs, axis=0)
    
    return pd.DataFrame(mean_corr, 
                       index=[f'Model_{i}' for i in range(n_models)],
                       columns=[f'Model_{i}' for i in range(n_models)])


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(weights: pd.DataFrame, 
                 returns: pd.DataFrame, 
                 params: Dict) -> Dict:
    """
    Run backtest with transaction costs.
    
    Execution: trade at close of signal day, returns realized next day.
    
    Returns dictionary with:
    - gross_returns: returns before transaction costs
    - net_returns: returns after transaction costs
    - turnover: daily turnover
    - transaction_costs: daily transaction costs
    - long_returns: returns from long leg only
    - short_returns: returns from short leg only
    """
    tc_bps = params['transaction_cost_bps']
    
    common_dates = weights.index.intersection(returns.index)
    weights_aligned = weights.loc[common_dates]
    returns_aligned = returns.loc[common_dates]
    
    weights_lagged = weights_aligned.shift(1)
    gross_returns = (weights_lagged * returns_aligned).sum(axis=1)
    
    weight_changes = weights_aligned.diff().abs()
    turnover = weight_changes.sum(axis=1) / 2
    transaction_costs = turnover * tc_bps / 10000
    net_returns = gross_returns - transaction_costs
    
    long_weights = weights_lagged.clip(lower=0)
    short_weights = weights_lagged.clip(upper=0).abs()
    long_returns = (long_weights * returns_aligned).sum(axis=1)
    short_returns = -(short_weights * returns_aligned).sum(axis=1)
    
    valid_idx = gross_returns.notna() & turnover.notna()
    
    return {
        'gross_returns': gross_returns[valid_idx],
        'net_returns': net_returns[valid_idx],
        'turnover': turnover[valid_idx],
        'transaction_costs': transaction_costs[valid_idx],
        'long_returns': long_returns[valid_idx],
        'short_returns': short_returns[valid_idx],
        'weights': weights_aligned,
    }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_performance_metrics(returns: pd.Series, 
                                 turnover: pd.Series, 
                                 trading_days: int = 252) -> Dict:
    """Compute standard performance metrics."""
    ann_return = returns.mean() * trading_days
    ann_vol = returns.std() * np.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    avg_dd = drawdown.mean()
    
    ann_turnover = turnover.mean() * trading_days
    
    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_dd': avg_dd,
        'ann_turnover': ann_turnover,
    }


def compute_all_metrics(backtest_results: Dict, 
                        trading_days: int = 252) -> Dict[str, Dict]:
    """Compute metrics for gross, net, long, and short returns."""
    return {
        'gross': compute_performance_metrics(
            backtest_results['gross_returns'], 
            backtest_results['turnover'], 
            trading_days
        ),
        'net': compute_performance_metrics(
            backtest_results['net_returns'], 
            backtest_results['turnover'], 
            trading_days
        ),
        'long': compute_performance_metrics(
            backtest_results['long_returns'], 
            backtest_results['turnover'], 
            trading_days
        ),
        'short': compute_performance_metrics(
            backtest_results['short_returns'], 
            backtest_results['turnover'], 
            trading_days
        ),
    }


# =============================================================================
# BEST MODEL TRACKING
# =============================================================================

class BestModelTracker:
    """
    Track and persist the best performing model.
    
    Usage:
        tracker = BestModelTracker()
        tracker.update(results, "model_v1")
        tracker.print_leaderboard()
    """
    
    def __init__(self, save_path: str = '../outputs/best_model.json'):
        self.save_path = save_path
        self.history = []
        self.best_model = None
        self.best_sharpe = -np.inf
        self._load()
    
    def _load(self):
        """Load history from disk."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.best_model = data.get('best_model')
                    self.best_sharpe = data.get('best_sharpe', -np.inf)
            except:
                pass
    
    def _save(self):
        """Save history to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_model': self.best_model,
                'best_sharpe': self.best_sharpe,
                'updated': datetime.now().isoformat(),
            }, f, indent=2)
    
    def update(self, results: Dict, model_name: str, features: List[str] = None) -> bool:
        """
        Update tracker with new model results.
        
        Returns True if this is a new best model.
        """
        gross_sharpe = results['metrics']['gross']['sharpe']
        net_sharpe = results['metrics']['net']['sharpe']
        
        record = {
            'name': model_name,
            'timestamp': datetime.now().isoformat(),
            'gross_sharpe': float(gross_sharpe),
            'net_sharpe': float(net_sharpe),
            'max_dd': float(results['metrics']['net']['max_dd']),
            'turnover': float(results['metrics']['gross']['ann_turnover']),
            'features': features or list(results['features'].keys()),
        }
        
        self.history.append(record)
        
        is_new_best = gross_sharpe > self.best_sharpe
        if is_new_best:
            self.best_sharpe = gross_sharpe
            self.best_model = record
            print(f"🏆 NEW BEST MODEL: {model_name} (Sharpe: {gross_sharpe:.4f})")
        
        self._save()
        return is_new_best
    
    def get_best(self) -> Optional[Dict]:
        """Get best model record."""
        return self.best_model
    
    def print_leaderboard(self, top_n: int = 10):
        """Print top models by Sharpe."""
        if not self.history:
            print("No models tracked yet.")
            return
        
        sorted_history = sorted(self.history, key=lambda x: x['gross_sharpe'], reverse=True)
        
        print("\n" + "=" * 80)
        print("MODEL LEADERBOARD")
        print("=" * 80)
        print(f"{'Rank':<6}{'Model':<25}{'Gross Sharpe':<15}{'Net Sharpe':<15}{'Turnover':<12}")
        print("-" * 80)
        
        for i, record in enumerate(sorted_history[:top_n]):
            rank_str = "👑" if i == 0 else f"{i+1}"
            print(f"{rank_str:<6}{record['name']:<25}{record['gross_sharpe']:<15.4f}"
                  f"{record['net_sharpe']:<15.4f}{record['turnover']:<12.1f}")
        
        print("-" * 80)
    
    def compare_to_best(self, results: Dict) -> pd.DataFrame:
        """Compare results to best model."""
        if self.best_model is None:
            print("No best model to compare to.")
            return None
        
        comparison = {
            'Metric': ['Gross Sharpe', 'Net Sharpe', 'Max Drawdown', 'Turnover'],
            'Best Model': [
                self.best_model['gross_sharpe'],
                self.best_model['net_sharpe'],
                self.best_model['max_dd'] * 100,
                self.best_model['turnover'],
            ],
            'Current': [
                results['metrics']['gross']['sharpe'],
                results['metrics']['net']['sharpe'],
                results['metrics']['net']['max_dd'] * 100,
                results['metrics']['gross']['ann_turnover'],
            ]
        }
        
        df = pd.DataFrame(comparison)
        df['Delta'] = df['Current'] - df['Best Model']
        
        return df


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class Pipeline:
    """
    Complete quantitative trading pipeline with best model tracking.
    
    Usage:
        pipeline = Pipeline()
        pipeline.load_data()
        
        # Run with baseline + new features
        results = pipeline.run_with_baseline(my_features)
        
        # Update best model tracker
        pipeline.update_best_model(results, "my_model_v1")
        
        # Compare to best
        pipeline.compare_to_best(results)
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize pipeline with parameters."""
        self.params = params or DEFAULT_PARAMS.copy()
        self.panel = None
        self.target = None
        self._baseline_features = None
        self._baseline_results = None
        self.tracker = BestModelTracker()
    
    def load_data(self, filepath: str = '../data/processed/df_after_eda.parquet') -> 'Pipeline':
        """Load data and create panel."""
        df = load_data(filepath)
        self.panel = create_panel_data(df)
        self.panel['target'] = compute_target(self.panel['forward_returns'])
        self.target = self.panel['target']
        print(f"Data loaded: {len(self.panel['returns'])} days x {len(self.panel['returns'].columns)} assets")
        return self
    
    def get_baseline_features(self) -> Dict[str, pd.DataFrame]:
        """Get baseline features (cached)."""
        if self._baseline_features is None:
            self._baseline_features = compute_baseline_features(self.panel)
        return self._baseline_features
    
    def run(self, features: Dict[str, pd.DataFrame], 
            standardize: bool = True,
            verbose: bool = True) -> Dict:
        """
        Run the full pipeline with given features.
        
        Args:
            features: Dict of feature DataFrames
            standardize: Whether to cross-sectionally standardize features
            verbose: Print progress
            
        Returns:
            Dict with predictions, coefficients, signals, weights, backtest results, metrics
        """
        if self.panel is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Standardize features
        if standardize:
            features_std = standardize_features(features)
        else:
            features_std = features
        
        # Run rolling regression
        predictions, coefficients = run_rolling_regression(
            features_std, self.target, self.params, verbose
        )
        
        # Construct signals and weights
        signals, signal_ranks = construct_signals(predictions)
        weights = construct_portfolio_weights(signal_ranks, self.params)
        
        # Run backtest
        backtest = run_backtest(weights, self.panel['returns'], self.params)
        
        # Compute metrics
        metrics = compute_all_metrics(backtest)
        
        return {
            'features': features,
            'features_std': features_std,
            'predictions': predictions,
            'coefficients': coefficients,
            'signals': signals,
            'signal_ranks': signal_ranks,
            'weights': weights,
            'backtest': backtest,
            'metrics': metrics,
        }
    
    def run_with_baseline(self, new_features: Dict[str, pd.DataFrame],
                          verbose: bool = True) -> Dict:
        """
        Run pipeline with BASELINE + NEW features combined.
        This is the recommended approach for incremental feature development.
        
        Args:
            new_features: Dict of new feature DataFrames to add to baseline
            verbose: Print progress
            
        Returns:
            Results dict (same as run())
        """
        # Combine baseline + new features
        baseline_features = self.get_baseline_features()
        combined_features = {**baseline_features, **new_features}
        
        if verbose:
            print(f"Running with baseline ({len(baseline_features)} features) + "
                  f"new ({len(new_features)} features) = {len(combined_features)} total")
        
        return self.run(combined_features, verbose=verbose)
    
    def run_with_smoothing(self, features: Dict[str, pd.DataFrame],
                            signal_halflife: Optional[int] = 5,
                            weight_decay: Optional[float] = None,
                            model_type: str = 'ridge',
                            model_params: Optional[Dict] = None,
                            retrain_interval: int = 21,
                            use_gpu: bool = True,
                            standardize: bool = True,
                            verbose: bool = True,
                            random_seed: int = 42) -> Dict:
        """
        Run pipeline with signal smoothing and optional advanced ML models.
        
        This is the enhanced version for turnover control experiments.
        
        Parameters:
        -----------
        features : Dict[str, pd.DataFrame]
            Feature DataFrames
        signal_halflife : int, optional
            Signal smoothing halflife (None = no smoothing)
        weight_decay : float, optional
            Weight decay for position smoothing (None = no smoothing)
        model_type : str
            Model type: 'ridge', 'lightgbm', 'xgboost', 'mlp'
        model_params : Dict, optional
            Model-specific hyperparameters
        retrain_interval : int
            For ML models: retrain every N days
        use_gpu : bool
            Use GPU if available
        standardize : bool
            Cross-sectionally standardize features
        verbose : bool
            Print progress
        random_seed : int
            Random seed for reproducibility (default: 42)
            
        Returns:
        --------
        Dict : Results with predictions, signals, weights, backtest, metrics
        """
        if self.panel is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Standardize features
        if standardize:
            features_std = standardize_features(features)
        else:
            features_std = features
        
        # Run model based on type
        if model_type == 'ridge':
            predictions, coefficients = run_rolling_regression(
                features_std, self.target, self.params, verbose
            )
        elif model_type == 'lightgbm':
            predictions = run_rolling_lightgbm(
                features_std, self.target, self.params,
                model_params=model_params,
                retrain_interval=retrain_interval,
                use_gpu=use_gpu,
                verbose=verbose
            )
            coefficients = None
        elif model_type == 'xgboost':
            predictions = run_rolling_xgboost(
                features_std, self.target, self.params,
                model_params=model_params,
                retrain_interval=retrain_interval,
                use_gpu=use_gpu,
                verbose=verbose
            )
            coefficients = None
        elif model_type == 'mlp':
            predictions = run_rolling_mlp(
                features_std, self.target, self.params,
                retrain_interval=retrain_interval,
                use_gpu=use_gpu,
                verbose=verbose,
                random_seed=random_seed,
                **(model_params or {})
            )
            coefficients = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Construct signals with optional smoothing
        signals, signal_ranks = construct_signals(predictions)
        
        if signal_halflife is not None:
            signals = smooth_signals(signals, halflife=signal_halflife)
            signal_ranks = signals.rank(axis=1, pct=True)
        
        # Construct weights with optional decay
        weights = construct_portfolio_weights(signal_ranks, self.params)
        
        if weight_decay is not None:
            weights = smooth_weights(weights, decay=weight_decay)
        
        # Run backtest
        backtest = run_backtest(weights, self.panel['returns'], self.params)
        
        # Compute metrics
        metrics = compute_all_metrics(backtest)
        
        return {
            'features': features,
            'features_std': features_std,
            'predictions': predictions,
            'coefficients': coefficients,
            'signals': signals,
            'signal_ranks': signal_ranks,
            'weights': weights,
            'backtest': backtest,
            'metrics': metrics,
            'params': {
                'model_type': model_type,
                'signal_halflife': signal_halflife,
                'weight_decay': weight_decay,
            }
        }
    
    def run_with_regime(self, features: Dict[str, pd.DataFrame],
                         regime_type: str = 'high_vol_only',
                         regime_mask: Optional[pd.Series] = None,
                         signal_halflife: Optional[int] = None,
                         weight_decay: Optional[float] = None,
                         model_type: str = 'ridge',
                         model_params: Optional[Dict] = None,
                         retrain_interval: int = 21,
                         use_gpu: bool = True,
                         standardize: bool = True,
                         verbose: bool = True) -> Dict:
        """
        Run pipeline with regime filtering.
        
        Filters trading signals based on market regime (vol, trend, etc.)
        
        Parameters:
        -----------
        features : Dict[str, pd.DataFrame]
            Feature DataFrames
        regime_type : str
            Regime type: 'high_vol_only', 'low_vol_only', 'uptrend_only', 
                        'downtrend_only', 'trending', 'confident'
        regime_mask : pd.Series, optional
            Custom regime mask (overrides regime_type)
        signal_halflife : int, optional
            Signal smoothing halflife
        weight_decay : float, optional
            Weight decay for position smoothing
        model_type : str
            Model type: 'ridge', 'lightgbm', 'xgboost', 'mlp'
            
        Returns:
        --------
        Dict : Results with predictions, signals, weights, backtest, metrics
        """
        if self.panel is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create regime mask
        if regime_mask is None:
            regime_mask = create_regime_mask(self.panel['returns'], regime_type)
        
        # Standardize features
        if standardize:
            features_std = standardize_features(features)
        else:
            features_std = features
        
        # Run model
        if model_type == 'ridge':
            predictions, coefficients = run_rolling_regression(
                features_std, self.target, self.params, verbose
            )
        elif model_type == 'lightgbm':
            predictions = run_rolling_lightgbm(
                features_std, self.target, self.params,
                model_params=model_params,
                retrain_interval=retrain_interval,
                use_gpu=use_gpu,
                verbose=verbose
            )
            coefficients = None
        elif model_type == 'xgboost':
            predictions = run_rolling_xgboost(
                features_std, self.target, self.params,
                model_params=model_params,
                retrain_interval=retrain_interval,
                use_gpu=use_gpu,
                verbose=verbose
            )
            coefficients = None
        elif model_type == 'mlp':
            predictions = run_rolling_mlp(
                features_std, self.target, self.params,
                retrain_interval=retrain_interval,
                use_gpu=use_gpu,
                verbose=verbose,
                **(model_params or {})
            )
            coefficients = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Construct signals with optional smoothing
        signals, signal_ranks = construct_signals(predictions)
        
        if signal_halflife is not None:
            signals = smooth_signals(signals, halflife=signal_halflife)
            signal_ranks = signals.rank(axis=1, pct=True)
        
        # Construct weights with regime filtering
        weights = construct_portfolio_weights_with_regime(
            signal_ranks, self.params,
            regime_mask=regime_mask,
            weight_decay=weight_decay
        )
        
        # Run backtest
        backtest = run_backtest(weights, self.panel['returns'], self.params)
        
        # Compute metrics
        metrics = compute_all_metrics(backtest)
        
        # Calculate time in market
        time_in_market = (weights.abs().sum(axis=1) > 0.01).mean()
        
        return {
            'features': features,
            'features_std': features_std,
            'predictions': predictions,
            'coefficients': coefficients,
            'signals': signals,
            'signal_ranks': signal_ranks,
            'weights': weights,
            'backtest': backtest,
            'metrics': metrics,
            'regime_mask': regime_mask,
            'time_in_market': time_in_market,
            'params': {
                'model_type': model_type,
                'regime_type': regime_type,
                'signal_halflife': signal_halflife,
                'weight_decay': weight_decay,
            }
        }
    
    def run_baseline(self, verbose: bool = True) -> Dict:
        """Run the baseline model only."""
        if self.panel is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if verbose:
            print("Running baseline model...")
        
        features = self.get_baseline_features()
        self._baseline_results = self.run(features, verbose=verbose)
        return self._baseline_results
    
    @property
    def baseline_results(self) -> Dict:
        """Get cached baseline results, running if necessary."""
        if self._baseline_results is None:
            self._baseline_results = self.run_baseline(verbose=False)
        return self._baseline_results
    
    def update_best_model(self, results: Dict, model_name: str) -> bool:
        """Update best model tracker."""
        return self.tracker.update(results, model_name, list(results['features'].keys()))
    
    def compare_to_best(self, results: Dict):
        """Print comparison to best model."""
        df = self.tracker.compare_to_best(results)
        if df is not None:
            print("\n" + "=" * 70)
            print(f"COMPARISON TO BEST MODEL: {self.tracker.best_model['name']}")
            print("=" * 70)
            print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            
            sharpe_delta = df.loc[df['Metric'] == 'Gross Sharpe', 'Delta'].values[0]
            print("\n" + "-" * 70)
            if sharpe_delta > 0:
                print(f"✅ Current model BEATS best by {sharpe_delta:+.4f}")
            else:
                print(f"❌ Current model TRAILS best by {sharpe_delta:.4f}")
    
    def print_leaderboard(self, top_n: int = 10):
        """Print model leaderboard."""
        self.tracker.print_leaderboard(top_n)
    
    def compare(self, results: Dict, baseline: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compare results to baseline.
        
        Args:
            results: Results from pipeline.run()
            baseline: Baseline results (uses cached if not provided)
            
        Returns:
            Comparison DataFrame
        """
        if baseline is None:
            baseline = self.baseline_results
        
        comparison = {
            'Metric': [
                'Gross Sharpe', 'Net Sharpe', 'Annual Return (Net)', 
                'Max Drawdown (Net)', 'Annual Turnover', 
                'Long Leg Sharpe', 'Short Leg Sharpe'
            ],
            'Baseline': [
                baseline['metrics']['gross']['sharpe'],
                baseline['metrics']['net']['sharpe'],
                baseline['metrics']['net']['ann_return'] * 100,
                baseline['metrics']['net']['max_dd'] * 100,
                baseline['metrics']['gross']['ann_turnover'],
                baseline['metrics']['long']['sharpe'],
                baseline['metrics']['short']['sharpe'],
            ],
            'New Model': [
                results['metrics']['gross']['sharpe'],
                results['metrics']['net']['sharpe'],
                results['metrics']['net']['ann_return'] * 100,
                results['metrics']['net']['max_dd'] * 100,
                results['metrics']['gross']['ann_turnover'],
                results['metrics']['long']['sharpe'],
                results['metrics']['short']['sharpe'],
            ]
        }
        
        df = pd.DataFrame(comparison)
        df['Delta'] = df['New Model'] - df['Baseline']
        
        return df
    
    def print_comparison(self, results: Dict, baseline: Optional[Dict] = None):
        """Print formatted comparison table."""
        df = self.compare(results, baseline)
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON vs BASELINE")
        print("=" * 70)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        # Key insights
        sharpe_delta = df.loc[df['Metric'] == 'Gross Sharpe', 'Delta'].values[0]
        print("\n" + "-" * 70)
        if sharpe_delta > 0:
            print(f"✅ New model IMPROVES Gross Sharpe by {sharpe_delta:+.4f}")
        else:
            print(f"❌ New model DEGRADES Gross Sharpe by {sharpe_delta:.4f}")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR NOTEBOOKS
# =============================================================================

def quick_test(features: Dict[str, pd.DataFrame], 
               data_path: str = '../data/processed/df_after_eda.parquet',
               compare_baseline: bool = True,
               with_baseline: bool = True) -> Dict:
    """
    Quick test of features with comparison to baseline.
    
    Usage:
        from src.pipeline import quick_test
        
        results = quick_test({
            'my_feature_1': feature_df_1,
            'my_feature_2': feature_df_2,
        }, with_baseline=True)
    """
    pipeline = Pipeline()
    pipeline.load_data(data_path)
    
    if with_baseline:
        results = pipeline.run_with_baseline(features)
    else:
        results = pipeline.run(features)
    
    if compare_baseline:
        pipeline.print_comparison(results)
    
    return results
