# =============================================================================
# MODELS MODULE - Model Training (NO strategy, NO backtest)
# =============================================================================
# 
# RESPONSIBILITY: Evaluate whether the model understands the data-generating process
# 
# FORBIDDEN:
#   - Position sizing
#   - Backtests
#   - Transaction costs
#   - Sharpe optimization
#
# ALLOWED:
#   - Model training
#   - Cross-validation
#   - Feature importance
#   - Residual analysis
#   - Regime diagnostics
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pickle
from pathlib import Path
import json
import time
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_type: str  # 'lightgbm', 'xgboost', 'ridge', 'ensemble'
    params: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'model_type': self.model_type,
            'params': self.params
        }


@dataclass
class TrainingResult:
    """Result of model training."""
    model_name: str
    train_corr: float
    val_corr: float
    train_time: float
    n_train: int
    n_val: int
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'train_corr': self.train_corr,
            'val_corr': self.val_corr,
            'train_time': self.train_time,
            'n_train': self.n_train,
            'n_val': self.n_val
        }


class ModelTrainer:
    """
    Trains models with proper validation.
    
    INVARIANTS:
    - Time-based train/val split (no future leakage)
    - Models saved with full configuration
    - No backtest or Sharpe computation
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, TrainingResult] = {}
        self.scaler = None
    
    def prepare_data(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/val data with time-based split.
        
        INVARIANT: Split is strictly temporal - val dates are all after train dates.
        """
        from sklearn.preprocessing import StandardScaler
        
        # Sort by date
        panel = panel.sort_values('date')
        
        # Get unique dates
        unique_dates = sorted(panel['date'].unique())
        split_idx = int(len(unique_dates) * train_ratio)
        split_date = unique_dates[split_idx]
        
        # Split
        train_mask = panel['date'] < split_date
        val_mask = panel['date'] >= split_date
        
        X_train = panel.loc[train_mask, feature_cols].values
        y_train = panel.loc[train_mask, target_col].values
        dates_train = panel.loc[train_mask, 'date'].values
        
        X_val = panel.loc[val_mask, feature_cols].values
        y_val = panel.loc[val_mask, target_col].values
        dates_val = panel.loc[val_mask, 'date'].values
        
        # Scale features (fit on train only)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"[DATA] Train: {len(X_train):,} samples ({unique_dates[0]} to {split_date})")
        print(f"[DATA] Val: {len(X_val):,} samples ({split_date} to {unique_dates[-1]})")
        
        return X_train_scaled, y_train, dates_train, X_val_scaled, y_val, dates_val
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelConfig
    ) -> TrainingResult:
        """Train LightGBM model."""
        import lightgbm as lgb
        
        print(f"\n[TRAIN] Training {config.name}...")
        
        params = config.params.copy()
        params['random_state'] = self.random_seed
        params['verbose'] = -1
        
        t_start = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        train_time = time.time() - t_start
        
        # Predictions
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        
        # Correlations
        train_corr = np.corrcoef(pred_train, y_train)[0, 1]
        val_corr = np.corrcoef(pred_val, y_val)[0, 1]
        
        # Feature importance
        importance = dict(zip(
            [f'f{i}' for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        
        self.models[config.name] = model
        
        result = TrainingResult(
            model_name=config.name,
            train_corr=train_corr,
            val_corr=val_corr,
            train_time=train_time,
            n_train=len(X_train),
            n_val=len(X_val),
            feature_importance=importance
        )
        
        self.results[config.name] = result
        
        print(f"   Time: {train_time:.2f}s")
        print(f"   Train corr: {train_corr:.4f}")
        print(f"   Val corr: {val_corr:.4f}")
        
        return result
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelConfig
    ) -> TrainingResult:
        """Train XGBoost model."""
        import xgboost as xgb
        
        print(f"\n[TRAIN] Training {config.name}...")
        
        params = config.params.copy()
        params['random_state'] = self.random_seed
        params['verbosity'] = 0
        
        t_start = time.time()
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        train_time = time.time() - t_start
        
        # Predictions
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        
        # Correlations
        train_corr = np.corrcoef(pred_train, y_train)[0, 1]
        val_corr = np.corrcoef(pred_val, y_val)[0, 1]
        
        # Feature importance
        importance = dict(zip(
            [f'f{i}' for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        
        self.models[config.name] = model
        
        result = TrainingResult(
            model_name=config.name,
            train_corr=train_corr,
            val_corr=val_corr,
            train_time=train_time,
            n_train=len(X_train),
            n_val=len(X_val),
            feature_importance=importance
        )
        
        self.results[config.name] = result
        
        print(f"   Time: {train_time:.2f}s")
        print(f"   Train corr: {train_corr:.4f}")
        print(f"   Val corr: {val_corr:.4f}")
        
        return result
    
    def train_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: ModelConfig
    ) -> TrainingResult:
        """Train Ridge regression model."""
        from sklearn.linear_model import Ridge
        
        print(f"\n[TRAIN] Training {config.name}...")
        
        params = config.params.copy()
        
        t_start = time.time()
        model = Ridge(**params)
        model.fit(X_train, y_train)
        train_time = time.time() - t_start
        
        # Predictions
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        
        # Correlations
        train_corr = np.corrcoef(pred_train, y_train)[0, 1]
        val_corr = np.corrcoef(pred_val, y_val)[0, 1]
        
        # Feature importance (coefficients)
        importance = dict(zip(
            [f'f{i}' for i in range(X_train.shape[1])],
            np.abs(model.coef_)
        ))
        
        self.models[config.name] = model
        
        result = TrainingResult(
            model_name=config.name,
            train_corr=train_corr,
            val_corr=val_corr,
            train_time=train_time,
            n_train=len(X_train),
            n_val=len(X_val),
            feature_importance=importance
        )
        
        self.results[config.name] = result
        
        print(f"   Time: {train_time:.2f}s")
        print(f"   Train corr: {train_corr:.4f}")
        print(f"   Val corr: {val_corr:.4f}")
        
        return result
    
    def predict(
        self,
        X: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """Generate predictions from a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def save_models(self, output_dir: Path):
        """Save all models and configurations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_dir / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"   Saved: {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = output_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"   Saved: {scaler_path}")
        
        # Save results
        results_df = pd.DataFrame([r.to_dict() for r in self.results.values()])
        results_df.to_csv(output_dir / "training_results.csv", index=False)
        print(f"   Saved: training_results.csv")
    
    def load_model(self, model_path: Path) -> Any:
        """Load a saved model."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def compute_daily_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray
) -> pd.Series:
    """
    Compute Information Coefficient by date.
    
    IC = correlation(prediction, target) per day
    """
    daily_ics = {}
    
    for date in np.unique(dates):
        mask = dates == date
        if mask.sum() > 5:
            pred_day = predictions[mask]
            target_day = targets[mask]
            ic = np.corrcoef(pred_day, target_day)[0, 1]
            if not np.isnan(ic):
                daily_ics[date] = ic
    
    return pd.Series(daily_ics)


def compute_information_ratio(daily_ics: pd.Series) -> float:
    """
    Compute Information Ratio.
    
    IR = mean(IC) / std(IC)
    """
    return daily_ics.mean() / daily_ics.std() if daily_ics.std() > 0 else 0


def analyze_residuals(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze prediction residuals.
    
    Checks for:
    - Residual autocorrelation (should be low)
    - Residual normality
    - Heteroskedasticity
    """
    residuals = targets - predictions
    
    # Autocorrelation
    residual_series = pd.Series(residuals)
    autocorr_1 = residual_series.autocorr(lag=1)
    autocorr_5 = residual_series.autocorr(lag=5)
    
    # Normality (Jarque-Bera)
    from scipy import stats
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    
    # Mean by time period
    df = pd.DataFrame({
        'date': dates,
        'residual': residuals
    })
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_mean = df.groupby('month')['residual'].mean()
    
    return {
        'autocorr_lag1': autocorr_1,
        'autocorr_lag5': autocorr_5,
        'jb_stat': jb_stat,
        'jb_pvalue': jb_pvalue,
        'monthly_mean_std': monthly_mean.std(),
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals)
    }


def compute_conditional_ic(
    predictions: np.ndarray,
    targets: np.ndarray,
    conditioning_var: np.ndarray,
    n_buckets: int = 5
) -> pd.DataFrame:
    """
    Compute IC conditioned on another variable.
    
    Useful for checking if model performance varies by regime.
    """
    # Create buckets
    valid_mask = ~np.isnan(conditioning_var)
    buckets = pd.qcut(conditioning_var[valid_mask], n_buckets, labels=False, duplicates='drop')
    
    results = []
    for bucket in range(n_buckets):
        bucket_mask = (buckets == bucket)
        pred_bucket = predictions[valid_mask][bucket_mask]
        target_bucket = targets[valid_mask][bucket_mask]
        
        if len(pred_bucket) > 10:
            ic = np.corrcoef(pred_bucket, target_bucket)[0, 1]
            results.append({
                'bucket': bucket,
                'ic': ic,
                'n': len(pred_bucket)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# WALK-FORWARD TRAINING FRAMEWORK
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for Walk-Forward Training."""
    initial_train_days: int = 252  # 1 year minimum training
    retrain_frequency: int = 21    # Retrain every ~month
    min_train_days: int = 126      # Minimum 6 months for training
    expanding_window: bool = True  # True = expanding, False = rolling
    max_train_days: int = 504      # Max training window (2 years) if rolling
    
    # Sample weighting
    use_decay_weights: bool = True
    decay_halflife: int = 63       # Halflife in trading days (~3 months)
    min_weight: float = 0.1        # Minimum weight for oldest samples
    
    # Validation
    embargo_days: int = 5          # Gap between train and predict to avoid leakage
    
    # Models
    model_type: str = 'lightgbm'   # 'lightgbm', 'xgboost', 'ridge'


@dataclass
class WalkForwardResult:
    """Result from one walk-forward fold."""
    train_start: str
    train_end: str
    predict_start: str
    predict_end: str
    n_train: int
    n_predict: int
    train_ic: float
    predict_ic: float
    model_params: Dict[str, Any]


class WalkForwardTrainer:
    """
    Walk-Forward Model Training for Financial Data.
    
    This is the PROPER way to train models on financial time series:
    1. Train on historical data up to time T
    2. Predict on future data T+embargo to T+retrain_freq
    3. Move forward, retrain, repeat
    
    Key features:
    - Expanding or rolling training window
    - Sample decay weights (recent data matters more)
    - Embargo period to prevent lookahead bias
    - Periodic retraining to capture regime shifts
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.fold_results: List[WalkForwardResult] = []
        self.all_predictions: List[pd.DataFrame] = []
        self.models: List[Any] = []
        self.scalers: List[Any] = []
        
    def compute_sample_weights(self, dates: pd.Series) -> np.ndarray:
        """
        Compute exponential decay weights based on date.
        
        More recent samples get higher weight.
        w(t) = exp(-λ * days_ago) where λ = ln(2) / halflife
        """
        if not self.config.use_decay_weights:
            return np.ones(len(dates))
        
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values
        
        # Exponential decay: w = exp(-λ * t) where halflife = ln(2) / λ
        decay_rate = np.log(2) / self.config.decay_halflife
        weights = np.exp(-decay_rate * days_ago)
        
        # Apply minimum weight floor
        weights = np.maximum(weights, self.config.min_weight)
        
        # Normalize to mean = 1
        weights = weights / weights.mean()
        
        return weights
    
    def get_training_dates(self, all_dates: np.ndarray) -> List[Tuple[str, str, str, str]]:
        """
        Generate (train_start, train_end, predict_start, predict_end) tuples.
        
        Returns list of date boundaries for each walk-forward fold.
        """
        unique_dates = np.sort(np.unique(all_dates))
        n_dates = len(unique_dates)
        
        folds = []
        
        # Start after initial training period
        current_idx = self.config.initial_train_days
        
        while current_idx < n_dates - self.config.embargo_days:
            # Training window
            if self.config.expanding_window:
                train_start_idx = 0
            else:
                train_start_idx = max(0, current_idx - self.config.max_train_days)
            
            train_end_idx = current_idx - 1
            
            # Check minimum training days
            if train_end_idx - train_start_idx < self.config.min_train_days:
                current_idx += self.config.retrain_frequency
                continue
            
            # Prediction window (after embargo)
            predict_start_idx = current_idx + self.config.embargo_days
            predict_end_idx = min(
                current_idx + self.config.retrain_frequency + self.config.embargo_days - 1,
                n_dates - 1
            )
            
            if predict_start_idx >= n_dates:
                break
                
            folds.append((
                str(unique_dates[train_start_idx]),
                str(unique_dates[train_end_idx]),
                str(unique_dates[predict_start_idx]),
                str(unique_dates[predict_end_idx])
            ))
            
            current_idx += self.config.retrain_frequency
        
        return folds
    
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
        model_params: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Train a single fold with the specified model type."""
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model_type = self.config.model_type.lower()
        
        if model_type == 'lightgbm':
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'verbosity': -1,
                'force_row_wise': True,
                'n_jobs': -1,
                **model_params
            }
            
            train_data = lgb.Dataset(
                X_scaled, 
                label=y_train, 
                weight=weights,
                free_raw_data=False
            )
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params.get('n_estimators', 100)
            )
            
        elif model_type == 'xgboost':
            import xgboost as xgb
            
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'verbosity': 0,
                'n_jobs': -1,
                **model_params
            }
            
            dtrain = xgb.DMatrix(X_scaled, label=y_train, weight=weights)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get('n_estimators', 100)
            )
            
        elif model_type == 'ridge':
            from sklearn.linear_model import Ridge
            
            # Ridge doesn't support sample weights directly in sklearn
            # We'll weight by sqrt(w) * X, sqrt(w) * y transformation
            sqrt_weights = np.sqrt(weights).reshape(-1, 1)
            X_weighted = X_scaled * sqrt_weights
            y_weighted = y_train * sqrt_weights.flatten()
            
            model = Ridge(alpha=model_params.get('alpha', 1.0))
            model.fit(X_weighted, y_weighted)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model, scaler
    
    def predict_fold(
        self,
        model: Any,
        scaler: Any,
        X: np.ndarray
    ) -> np.ndarray:
        """Generate predictions for a fold."""
        X_scaled = scaler.transform(X)
        
        model_type = self.config.model_type.lower()
        
        if model_type == 'lightgbm':
            return model.predict(X_scaled)
        elif model_type == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_scaled)
            return model.predict(dtest)
        elif model_type == 'ridge':
            return model.predict(X_scaled)
        else:
            return model.predict(X_scaled)
    
    def run(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_params: Dict[str, Any] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Execute walk-forward training.
        
        Parameters:
        -----------
        panel_df : DataFrame with columns ['date', 'ticker'] + features + target
        feature_cols : List of feature column names
        target_col : Name of target column
        model_params : Model-specific parameters
        verbose : Print progress
        
        Returns:
        --------
        DataFrame with out-of-sample predictions for entire period
        """
        model_params = model_params or {}
        
        # Ensure date is datetime
        panel_df = panel_df.copy()
        panel_df['date'] = pd.to_datetime(panel_df['date'])
        
        # Get fold boundaries
        all_dates = panel_df['date'].values
        folds = self.get_training_dates(all_dates)
        
        if verbose:
            print(f"=" * 60)
            print(f"WALK-FORWARD TRAINING")
            print(f"=" * 60)
            print(f"Model: {self.config.model_type}")
            print(f"Window: {'Expanding' if self.config.expanding_window else 'Rolling'}")
            print(f"Retrain frequency: {self.config.retrain_frequency} days")
            print(f"Decay halflife: {self.config.decay_halflife} days")
            print(f"Embargo: {self.config.embargo_days} days")
            print(f"Number of folds: {len(folds)}")
            print(f"-" * 60)
        
        self.fold_results = []
        self.all_predictions = []
        
        for i, (train_start, train_end, pred_start, pred_end) in enumerate(folds):
            # Get training data
            train_mask = (
                (panel_df['date'] >= train_start) & 
                (panel_df['date'] <= train_end)
            )
            train_df = panel_df[train_mask].copy()
            
            # Get prediction data
            pred_mask = (
                (panel_df['date'] >= pred_start) & 
                (panel_df['date'] <= pred_end)
            )
            pred_df = panel_df[pred_mask].copy()
            
            if len(train_df) < 100 or len(pred_df) == 0:
                continue
            
            # Prepare features and targets
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_pred = pred_df[feature_cols].values
            y_pred = pred_df[target_col].values
            
            # Remove NaN rows
            train_valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            pred_valid = ~np.isnan(X_pred).any(axis=1)
            
            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            train_dates = train_df[train_valid]['date']
            
            X_pred = X_pred[pred_valid]
            y_pred_actual = y_pred[pred_valid]
            pred_df_valid = pred_df[pred_valid].copy()
            
            if len(X_train) < 100:
                continue
            
            # Compute sample weights for training
            weights = self.compute_sample_weights(train_dates)
            
            # Train model
            model, scaler = self.train_fold(X_train, y_train, weights, model_params)
            
            # Generate predictions
            predictions = self.predict_fold(model, scaler, X_pred)
            
            # Compute ICs
            train_preds = self.predict_fold(model, scaler, X_train)
            train_ic = np.corrcoef(train_preds, y_train)[0, 1] if len(y_train) > 1 else 0
            pred_ic = np.corrcoef(predictions, y_pred_actual)[0, 1] if len(y_pred_actual) > 1 else 0
            
            # Store results
            result = WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                predict_start=pred_start,
                predict_end=pred_end,
                n_train=len(X_train),
                n_predict=len(X_pred),
                train_ic=train_ic if not np.isnan(train_ic) else 0,
                predict_ic=pred_ic if not np.isnan(pred_ic) else 0,
                model_params=model_params
            )
            self.fold_results.append(result)
            
            # Store predictions with metadata
            pred_df_valid['prediction'] = predictions
            pred_df_valid['fold'] = i
            self.all_predictions.append(pred_df_valid[['date', 'ticker', target_col, 'prediction', 'fold']])
            
            # Store model
            self.models.append(model)
            self.scalers.append(scaler)
            
            if verbose and (i % 5 == 0 or i == len(folds) - 1):
                print(f"Fold {i+1:3d}/{len(folds)}: "
                      f"Train IC={train_ic:.4f}, "
                      f"OOS IC={pred_ic:.4f}, "
                      f"N_train={len(X_train):,}, "
                      f"N_pred={len(X_pred):,}")
        
        # Combine all predictions
        if self.all_predictions:
            combined_preds = pd.concat(self.all_predictions, ignore_index=True)
        else:
            combined_preds = pd.DataFrame()
        
        if verbose:
            print(f"-" * 60)
            self.print_summary()
        
        return combined_preds
    
    def print_summary(self):
        """Print walk-forward training summary."""
        if not self.fold_results:
            print("No results to summarize.")
            return
        
        train_ics = [r.train_ic for r in self.fold_results]
        pred_ics = [r.predict_ic for r in self.fold_results]
        
        print(f"\nWALK-FORWARD SUMMARY:")
        print(f"  Total Folds: {len(self.fold_results)}")
        print(f"  Training IC:  Mean={np.mean(train_ics):.4f}, Std={np.std(train_ics):.4f}")
        print(f"  OOS IC:       Mean={np.mean(pred_ics):.4f}, Std={np.std(pred_ics):.4f}")
        print(f"  IC Ratio:     {np.mean(pred_ics)/np.mean(train_ics):.2%}" if np.mean(train_ics) != 0 else "")
        print(f"  Hit Rate:     {np.mean([ic > 0 for ic in pred_ics]):.1%} of folds with positive OOS IC")
        
        # Information Ratio
        if np.std(pred_ics) > 0:
            ir = np.mean(pred_ics) / np.std(pred_ics)
            print(f"  Info Ratio:   {ir:.3f}")
    
    def get_results_df(self) -> pd.DataFrame:
        """Get fold results as DataFrame."""
        return pd.DataFrame([
            {
                'fold': i,
                'train_start': r.train_start,
                'train_end': r.train_end,
                'predict_start': r.predict_start,
                'predict_end': r.predict_end,
                'n_train': r.n_train,
                'n_predict': r.n_predict,
                'train_ic': r.train_ic,
                'predict_ic': r.predict_ic
            }
            for i, r in enumerate(self.fold_results)
        ])
    
    def plot_ic_evolution(self, ax=None):
        """Plot IC evolution over time."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        
        results_df = self.get_results_df()
        
        ax.bar(results_df['fold'], results_df['predict_ic'], 
               alpha=0.7, label='OOS IC', color='steelblue')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=results_df['predict_ic'].mean(), color='red', 
                   linestyle='--', label=f"Mean OOS IC: {results_df['predict_ic'].mean():.4f}")
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Information Coefficient')
        ax.set_title('Walk-Forward OOS IC by Fold')
        ax.legend()
        
        return ax
