"""
Model Training Framework
========================

Walk-forward training with proper evaluation and model persistence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import os
from datetime import datetime
import pickle


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = 'lightgbm'  # 'lightgbm', 'linear', 'mlp'
    
    # Walk-forward settings
    training_window: int = 504  # 2 years
    validation_window: int = 63  # 3 months (within training for hyperparam)
    retrain_interval: int = 21  # Monthly retrain
    
    # Label settings
    label_type: str = '3class'  # '3class', 'regression'
    label_threshold: float = 0.005  # For 3-class
    forward_horizon: int = 1  # 1-day forward
    
    # Sample weighting
    use_sample_weights: bool = True
    decay_halflife: int = 63
    
    # LightGBM defaults
    lgbm_params: Dict = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'min_child_samples': 20,
        'device': 'gpu',  # Enable GPU acceleration
        'gpu_use_dp': False,  # Use float32 for speed
    })
    
    # Minimum samples for training (allow shorter windows)
    min_train_samples: int = 5000


class ExperimentLogger:
    """Log experiment results."""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.experiments: List[Dict] = []
        
    def log(self, experiment: Dict) -> None:
        """Log a single experiment."""
        experiment['timestamp'] = datetime.now().isoformat()
        self.experiments.append(experiment)
        
        # Save immediately
        self._save()
        
    def _save(self) -> None:
        """Save all experiments to JSON."""
        path = os.path.join(self.save_dir, 'experiment_log.json')
        with open(path, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
            
    def get_best(self, metric: str = 'net_sharpe') -> Dict:
        """Get best experiment by metric."""
        if not self.experiments:
            return None
        return max(self.experiments, key=lambda x: x.get('metrics', {}).get(metric, -np.inf))


class WalkForwardTrainer:
    """
    Walk-forward model training with proper evaluation.
    
    Key principles:
    - NEVER look ahead
    - Retrain periodically to adapt to regime changes
    - Track per-asset and aggregate metrics
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models: Dict[str, Any] = {}  # Saved models by date
        self.predictions: Dict[str, pd.DataFrame] = {}
        self.feature_importance: List[pd.DataFrame] = []
        
    def create_labels(
        self,
        returns: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.005
    ) -> pd.DataFrame:
        """
        Create 3-class labels: Up (2), Hold (1), Down (0).
        
        Labels are based on FORWARD returns (shifted).
        """
        fwd_ret = returns.shift(-horizon)
        
        labels = pd.DataFrame(1, index=returns.index, columns=returns.columns)  # Default: Hold
        labels[fwd_ret > threshold] = 2   # Up
        labels[fwd_ret < -threshold] = 0  # Down
        
        return labels
    
    def compute_sample_weights(
        self,
        dates: pd.DatetimeIndex,
        halflife: int = 63
    ) -> np.ndarray:
        """
        Time-decay sample weights.
        More recent samples get higher weight.
        """
        n = len(dates)
        decay = np.exp(-np.log(2) / halflife * np.arange(n)[::-1])
        return decay / decay.sum() * n
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weights: np.ndarray = None
    ) -> Any:
        """Train LightGBM classifier with GPU support."""
        import lightgbm as lgb
        
        params = self.config.lgbm_params.copy()
        
        # Try GPU first, fallback to CPU if not available
        try:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except Exception as e:
            if 'GPU' in str(e).upper() or 'device' in str(e).lower():
                print("⚠️ GPU not available, falling back to CPU")
                params.pop('device', None)
                params.pop('gpu_use_dp', None)
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                raise e
        
        return model
    
    def run_walk_forward(
        self,
        features: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        train_start: str = None,
        train_end: str = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run walk-forward training and prediction.
        
        Returns:
            Dict with predictions, metrics, feature importance
        """
        # Create labels
        labels = self.create_labels(
            returns,
            horizon=self.config.forward_horizon,
            threshold=self.config.label_threshold
        )
        
        # Get feature names and data
        feature_names = list(features.keys())
        n_features = len(feature_names)
        
        # Align all data
        if train_start:
            for name in features:
                features[name] = features[name].loc[train_start:]
            labels = labels.loc[train_start:]
            returns = returns.loc[train_start:]
            
        if train_end:
            for name in features:
                features[name] = features[name].loc[:train_end]
            labels = labels.loc[:train_end]
            returns = returns.loc[:train_end]
        
        dates = returns.index
        assets = returns.columns
        n_dates = len(dates)
        n_assets = len(assets)
        
        # Build feature matrix
        if verbose:
            print(f"Building feature matrix: {n_dates} dates x {n_assets} assets x {n_features} features")
        
        # Store all predictions
        all_preds = pd.DataFrame(index=dates, columns=assets, dtype=float)
        all_probs_up = pd.DataFrame(index=dates, columns=assets, dtype=float)
        all_probs_down = pd.DataFrame(index=dates, columns=assets, dtype=float)
        
        feature_importance_list = []
        retrain_dates = []
        
        # Walk-forward loop
        train_window = self.config.training_window
        retrain_interval = self.config.retrain_interval
        min_train_samples = getattr(self.config, 'min_train_samples', 5000)
        
        current_model = None
        steps_since_retrain = 0
        
        # Calculate actual minimum start (allow smaller windows if period is short)
        min_history = min(train_window, max(63, n_dates // 4))  # At least 63 days or 1/4 of period
        
        for t in range(min_history, n_dates):
            current_date = dates[t]
            
            # Check if need to retrain
            if current_model is None or steps_since_retrain >= retrain_interval:
                # Get training data - use all available history up to train_window
                train_start_idx = max(0, t - train_window)
                train_end_idx = t
                
                # Ensure we have at least some history
                actual_window = train_end_idx - train_start_idx
                if actual_window < 21:  # Skip if less than 1 month
                    steps_since_retrain += 1
                    continue
                
                train_dates = dates[train_start_idx:train_end_idx]
                
                # Build training matrix
                X_train = []
                y_train = []
                weights_train = []
                
                sample_weights = self.compute_sample_weights(
                    train_dates, self.config.decay_halflife
                ) if self.config.use_sample_weights else None
                
                for i, d in enumerate(train_dates):
                    for asset in assets:
                        # Get features for this (date, asset)
                        x = [features[f].loc[d, asset] for f in feature_names]
                        y = labels.loc[d, asset]
                        
                        if not np.isnan(y) and not any(np.isnan(x)):
                            X_train.append(x)
                            y_train.append(int(y))
                            if sample_weights is not None:
                                weights_train.append(sample_weights[i])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                weights_train = np.array(weights_train) if sample_weights is not None else None
                
                # Train model
                if len(np.unique(y_train)) >= 2:
                    current_model = self.train_lightgbm(X_train, y_train, weights_train)
                    
                    # Store feature importance
                    fi = pd.DataFrame({
                        'feature': feature_names,
                        'importance': current_model.feature_importances_,
                        'date': current_date
                    })
                    feature_importance_list.append(fi)
                    retrain_dates.append(current_date)
                    
                    if verbose and len(retrain_dates) % 10 == 1:
                        print(f"  Retrained at {current_date.strftime('%Y-%m-%d')}, "
                              f"samples={len(X_train)}, classes={np.unique(y_train, return_counts=True)}")
                
                steps_since_retrain = 0
            
            # Predict for current date
            if current_model is not None:
                for asset in assets:
                    x = [features[f].loc[current_date, asset] for f in feature_names]
                    
                    if not any(np.isnan(x)):
                        x = np.array(x).reshape(1, -1)
                        probs = current_model.predict_proba(x)[0]
                        
                        # Get probability for each class
                        classes = current_model.classes_
                        prob_dict = {int(c): p for c, p in zip(classes, probs)}
                        
                        all_probs_down.loc[current_date, asset] = prob_dict.get(0, 0)
                        all_probs_up.loc[current_date, asset] = prob_dict.get(2, 0)
                        all_preds.loc[current_date, asset] = np.argmax(probs)
            
            steps_since_retrain += 1
        
        # Aggregate feature importance
        if feature_importance_list:
            all_fi = pd.concat(feature_importance_list)
            avg_fi = all_fi.groupby('feature')['importance'].mean().sort_values(ascending=False)
        else:
            avg_fi = pd.Series()
        
        return {
            'predictions': all_preds,
            'probs_up': all_probs_up,
            'probs_down': all_probs_down,
            'labels': labels,
            'feature_importance': avg_fi,
            'retrain_dates': retrain_dates,
        }
    
    def compute_alpha(
        self,
        probs_up: pd.DataFrame,
        probs_down: pd.DataFrame,
        hit_ratios: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Compute reliability-weighted, dollar-neutral alpha.
        
        Alpha = (p_up - p_down) * hit_ratio - cross_sectional_mean
        """
        raw_score = probs_up - probs_down
        
        if hit_ratios is not None:
            weighted_score = raw_score * hit_ratios
        else:
            weighted_score = raw_score
        
        # Dollar neutrality: subtract cross-sectional mean
        cs_mean = weighted_score.mean(axis=1)
        alpha = weighted_score.sub(cs_mean, axis=0)
        
        return alpha


def compute_hit_ratio(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    window: int = 63
) -> pd.DataFrame:
    """
    Compute rolling hit ratio per asset.
    
    Hit = predicted class matches true class
    """
    hits = (predictions == labels).astype(float)
    hit_ratio = hits.rolling(window, min_periods=20).mean()
    
    # Fill NaN with prior (random = 1/3)
    return hit_ratio.fillna(1/3)
