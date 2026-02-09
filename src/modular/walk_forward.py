"""
Walk-Forward Model Training
============================

Proper walk-forward training framework with:
- Expanding or rolling window training
- Embargo period to prevent look-ahead
- Monthly retraining (configurable)
- Multiple model support (Ridge, LGBM, MLP)
- Proper timing: signal at T close → trade at T+1 open

CRITICAL TIMING:
- Features at time T use data up to and including T close
- Signal generated at T close
- Position taken at T+1 open
- Return earned from T+1 open to T+2 open (or T close to T+1 close)

Author: Precog Quant Research
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr
import warnings
import pickle
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# Optional imports
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class WalkForwardTrainer:
    """
    Walk-forward training framework.
    
    Key principles:
    1. Train on past, predict on future (no look-ahead)
    2. Embargo period between train and test
    3. Retrain periodically (not every day - too expensive)
    4. Track IC over time to detect model decay
    
    Usage:
        trainer = WalkForwardTrainer(df, feature_cols, target_col)
        predictions = trainer.train_predict(
            model_type='lgbm',
            retrain_freq_days=21,  # Monthly
            min_train_days=252,
            embargo_days=5
        )
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'fwd_ret_5d',
        date_col: str = 'date',
        ticker_col: str = 'ticker'
    ):
        """
        Initialize trainer.
        
        Parameters
        ----------
        df : pd.DataFrame
            Panel data with features and target
        feature_cols : List[str]
            List of feature column names
        target_col : str
            Target column (forward returns)
        """
        self.df = df.copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values([date_col, ticker_col])
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col
        self.ticker_col = ticker_col
        
        self.dates = sorted(self.df[date_col].unique())
        self.trained_models = {}
        self.diagnostics = []
        
        # Validate
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features[:5]}...")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
    
    def _create_model(self, model_type: str, **params) -> object:
        """Create a model instance."""
        if model_type == 'ridge':
            return Ridge(alpha=params.get('alpha', 1.0))
        elif model_type == 'lasso':
            return Lasso(alpha=params.get('alpha', 0.001))
        elif model_type == 'lgbm':
            if not HAS_LGBM:
                raise ImportError("LightGBM not installed")
            return lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.03),
                num_leaves=params.get('num_leaves', 31),
                min_child_samples=params.get('min_child_samples', 100),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 0.1),
                verbose=-1,
                force_col_wise=True
            )
        elif model_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (64, 32)),
                alpha=params.get('alpha', 0.01),
                learning_rate_init=params.get('learning_rate_init', 0.001),
                max_iter=params.get('max_iter', 200),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectionally z-score the target."""
        df = df.copy()
        df['target_zscore'] = df.groupby(self.date_col)[self.target_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        df['target_zscore'] = df['target_zscore'].clip(-3, 3)
        return df
    
    def train_predict(
        self,
        model_type: str = 'lgbm',
        model_params: Optional[Dict] = None,
        retrain_freq_days: int = 21,
        min_train_days: int = 252,
        embargo_days: int = 5,
        expanding_window: bool = True,
        max_train_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Walk-forward training and prediction.
        
        Parameters
        ----------
        model_type : str
            Model type ('ridge', 'lgbm', 'mlp')
        model_params : Dict
            Model hyperparameters
        retrain_freq_days : int
            Days between retraining (default: 21 = monthly)
        min_train_days : int
            Minimum training window (default: 252 = 1 year)
        embargo_days : int
            Gap between train and test to prevent look-ahead
        expanding_window : bool
            If True, use expanding window; if False, rolling window
        max_train_days : int, optional
            Maximum training window for rolling (ignored if expanding)
        
        Returns
        -------
        pd.DataFrame with columns: ['date', 'ticker', 'prediction', 'signal_zscore']
        """
        model_params = model_params or {}
        
        # Prepare data
        df = self._prepare_target(self.df)
        df = df.dropna(subset=self.feature_cols + ['target_zscore'])
        
        print(f"Walk-forward training: {model_type.upper()}")
        print(f"  Retrain frequency: {retrain_freq_days} days")
        print(f"  Min training window: {min_train_days} days")
        print(f"  Embargo: {embargo_days} days")
        print(f"  Window type: {'Expanding' if expanding_window else 'Rolling'}")
        
        # Determine retrain dates
        dates = list(sorted(df[self.date_col].unique()))
        start_idx = min_train_days + embargo_days
        
        if start_idx >= len(dates):
            raise ValueError(f"Not enough data: need {start_idx} days, have {len(dates)}")
        
        retrain_dates = dates[start_idx::retrain_freq_days]
        print(f"  Retrain dates: {len(retrain_dates)}")
        
        all_predictions = []
        scaler = StandardScaler()
        current_model = None
        last_retrain_idx = -1
        
        for retrain_date in retrain_dates:
            retrain_idx = dates.index(retrain_date)
            
            # Training window
            if expanding_window:
                train_start_idx = 0
            else:
                train_start_idx = max(0, retrain_idx - embargo_days - (max_train_days or min_train_days))
            
            train_end_idx = retrain_idx - embargo_days
            
            train_dates = dates[train_start_idx:train_end_idx]
            train_df = df[df[self.date_col].isin(train_dates)]
            
            if len(train_df) < min_train_days * 0.5:  # Need at least half the expected data
                continue
            
            # Test window (until next retrain or end)
            next_retrain_idx = min(retrain_idx + retrain_freq_days, len(dates))
            test_dates = dates[retrain_idx:next_retrain_idx]
            test_df = df[df[self.date_col].isin(test_dates)]
            
            if len(test_df) == 0:
                continue
            
            # Prepare features
            X_train = train_df[self.feature_cols].values
            y_train = train_df['target_zscore'].values
            X_test = test_df[self.feature_cols].values
            
            # Handle NaN in features
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)
            
            # Scale
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self._create_model(model_type, **model_params)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            preds = model.predict(X_test_scaled)
            
            # Store predictions
            pred_df = test_df[[self.date_col, self.ticker_col]].copy()
            pred_df['prediction'] = preds
            all_predictions.append(pred_df)
            
            # Diagnostics
            if self.target_col in test_df.columns:
                valid_mask = ~test_df['target_zscore'].isna()
                if valid_mask.sum() > 10:
                    ic = spearmanr(preds[valid_mask], test_df.loc[valid_mask.values, 'target_zscore'].values)[0]
                    self.diagnostics.append({
                        'retrain_date': retrain_date,
                        'train_size': len(train_df),
                        'test_size': len(test_df),
                        'ic': ic
                    })
            
            current_model = model
            last_retrain_idx = retrain_idx
        
        if not all_predictions:
            raise ValueError("No predictions generated")
        
        # Combine predictions
        predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Cross-sectionally z-score the signal
        predictions['signal_zscore'] = predictions.groupby(self.date_col)['prediction'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        ).clip(-3, 3)
        
        # Summary
        avg_ic = np.nanmean([d['ic'] for d in self.diagnostics]) if self.diagnostics else np.nan
        print(f"\n✅ Walk-forward training complete")
        print(f"  Predictions: {len(predictions):,}")
        print(f"  Average IC: {avg_ic:.4f}")
        
        self.trained_models[model_type] = current_model
        
        return predictions
    
    def train_ensemble(
        self,
        model_configs: Dict[str, Dict],
        retrain_freq_days: int = 21,
        min_train_days: int = 252,
        embargo_days: int = 5,
        ensemble_method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Train multiple models and create ensemble.
        
        Parameters
        ----------
        model_configs : Dict[str, Dict]
            Dictionary of model_name -> {'type': ..., 'params': ...}
        ensemble_method : str
            'mean', 'rank_mean', or 'ic_weighted'
        
        Returns
        -------
        pd.DataFrame with individual and ensemble predictions
        """
        all_preds = {}
        
        for name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")
            
            preds = self.train_predict(
                model_type=config['type'],
                model_params=config.get('params', {}),
                retrain_freq_days=retrain_freq_days,
                min_train_days=min_train_days,
                embargo_days=embargo_days
            )
            all_preds[name] = preds
        
        # Merge predictions
        base_cols = [self.date_col, self.ticker_col]
        merged = all_preds[list(all_preds.keys())[0]][base_cols].copy()
        
        for name, preds in all_preds.items():
            merged = merged.merge(
                preds[[self.date_col, self.ticker_col, 'signal_zscore']].rename(
                    columns={'signal_zscore': f'signal_{name}'}
                ),
                on=base_cols,
                how='outer'
            )
        
        # Create ensemble
        signal_cols = [c for c in merged.columns if c.startswith('signal_')]
        
        if ensemble_method == 'mean':
            merged['signal_ensemble'] = merged[signal_cols].mean(axis=1)
        elif ensemble_method == 'rank_mean':
            # Rank within each date, then average ranks
            for col in signal_cols:
                merged[f'{col}_rank'] = merged.groupby(self.date_col)[col].rank(pct=True)
            rank_cols = [f'{c}_rank' for c in signal_cols]
            merged['signal_ensemble'] = merged[rank_cols].mean(axis=1)
        elif ensemble_method == 'ic_weighted':
            # Weight by IC (from diagnostics)
            ics = {}
            for name in all_preds.keys():
                model_diags = [d for d in self.diagnostics if d.get('model') == name]
                ics[name] = np.nanmean([d['ic'] for d in model_diags]) if model_diags else 1.0
            total_ic = sum(max(0, ic) for ic in ics.values())
            for name in all_preds.keys():
                weight = max(0, ics[name]) / total_ic if total_ic > 0 else 1/len(all_preds)
                merged[f'signal_{name}'] *= weight
            merged['signal_ensemble'] = merged[signal_cols].sum(axis=1)
        
        # Z-score ensemble
        merged['signal_ensemble_zscore'] = merged.groupby(self.date_col)['signal_ensemble'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        ).clip(-3, 3)
        
        print(f"\n✅ Ensemble created: {len(merged):,} predictions")
        
        return merged
    
    def get_diagnostics(self) -> pd.DataFrame:
        """Return training diagnostics as DataFrame."""
        return pd.DataFrame(self.diagnostics)
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: str):
        """Save predictions to parquet."""
        predictions.to_parquet(output_path, index=False)
        print(f"✅ Saved predictions to {output_path}")
    
    def save_model(self, model_type: str, output_path: str):
        """Save trained model."""
        if model_type not in self.trained_models:
            raise ValueError(f"Model '{model_type}' not trained")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.trained_models[model_type], f)
        print(f"✅ Saved model to {output_path}")


class WalkForwardTrainerV2:
    """
    Enhanced walk-forward trainer with fewer retrains.
    
    Key improvements:
    - Quarterly retraining (less overfit)
    - Model decay detection
    - Automatic feature importance tracking
    - Signal decay/smoothing
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'fwd_ret_5d',
        retrain_freq: str = 'quarterly'  # 'monthly', 'quarterly', 'yearly'
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.retrain_freq = retrain_freq
        
        # Map frequency to days
        self.freq_map = {
            'monthly': 21,
            'quarterly': 63,
            'yearly': 252
        }
    
    def compute_optimal_retrain_freq(self) -> int:
        """
        Analyze IC decay to find optimal retrain frequency.
        
        Returns days between retrains.
        """
        # TODO: Implement IC decay analysis
        # For now, return quarterly
        return self.freq_map.get(self.retrain_freq, 63)
    
    def train_with_signal_decay(
        self,
        model_type: str = 'lgbm',
        decay_halflife: int = 10
    ) -> pd.DataFrame:
        """
        Train model and apply signal decay between retrains.
        
        This smooths predictions and reduces turnover.
        """
        # Get base predictions
        trainer = WalkForwardTrainer(
            self.df, self.feature_cols, self.target_col
        )
        
        preds = trainer.train_predict(
            model_type=model_type,
            retrain_freq_days=self.freq_map.get(self.retrain_freq, 63)
        )
        
        # Apply exponential decay to signal
        alpha = 1 - np.exp(-np.log(2) / decay_halflife)
        
        preds = preds.sort_values(['ticker', 'date'])
        preds['signal_smoothed'] = preds.groupby('ticker')['signal_zscore'].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )
        
        # Re-z-score after smoothing
        preds['signal_smoothed_zscore'] = preds.groupby('date')['signal_smoothed'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        ).clip(-3, 3)
        
        return preds
