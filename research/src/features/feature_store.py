"""
Feature Store
=============

Central feature management: generation, storage, and access.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class FeatureConfig:
    """Feature computation settings."""
    # Feature categories to compute
    compute_momentum: bool = True
    compute_volatility: bool = True
    compute_mean_reversion: bool = True
    compute_stat_arb: bool = True
    compute_technical: bool = True
    
    # Standardization
    standardize: bool = True
    standardize_method: str = 'zscore'  # 'zscore', 'rank', 'robust'
    winsorize_pct: float = 0.01
    
    # Quality thresholds
    max_nan_pct: float = 0.15
    min_variance: float = 1e-10


class FeatureStore:
    """
    Central hub for feature management.
    
    Handles:
    - Feature generation from raw data
    - Quality filtering
    - Cross-sectional standardization
    - Feature analysis (correlation, IC)
    - Persistence
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.raw_features: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, pd.DataFrame] = {}
        self.feature_metadata: Dict[str, Dict] = {}
        
    def compute_all_features(
        self,
        returns: pd.DataFrame,
        close: pd.DataFrame = None,
        high: pd.DataFrame = None,
        low: pd.DataFrame = None,
        open_: pd.DataFrame = None,
        volume: pd.DataFrame = None,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all configured features.
        
        Args:
            returns: Log returns (date x asset)
            close: Close prices
            high: High prices
            low: Low prices
            open_: Open prices
            volume: Volume
            verbose: Print progress
            
        Returns:
            Dictionary of standardized features
        """
        if verbose:
            print("="*60)
            print("COMPUTING ALL FEATURES")
            print("="*60)
            
        all_features = {}
        
        # 1. Momentum features
        if self.config.compute_momentum:
            from .momentum import MomentumFeatures
            mf = MomentumFeatures(returns)
            all_features.update(mf.compute_all(verbose))
            
        # 2. Volatility features
        if self.config.compute_volatility:
            from .volatility import VolatilityFeatures
            vf = VolatilityFeatures(returns, high, low, close, open_)
            all_features.update(vf.compute_all(verbose))
            
        # 3. Mean reversion features
        if self.config.compute_mean_reversion:
            from .mean_reversion import MeanReversionFeatures
            mrf = MeanReversionFeatures(returns, close)
            all_features.update(mrf.compute_all(verbose))
            
        # 4. Stat arb features
        if self.config.compute_stat_arb:
            from .stat_arb import StatArbFeatures
            saf = StatArbFeatures(returns, volume)
            all_features.update(saf.compute_all(verbose))
            
        # 5. Technical features
        if self.config.compute_technical and close is not None:
            from .technical import TechnicalFeatures
            tf = TechnicalFeatures(close, high, low, volume, returns)
            all_features.update(tf.compute_all(verbose))
            
        self.raw_features = all_features
        
        if verbose:
            print(f"\n📊 Total raw features: {len(all_features)}")
            
        # Clean and standardize
        self.features = self._clean_and_standardize(all_features, verbose)
        
        return self.features
    
    def _clean_and_standardize(
        self,
        features: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Clean features and apply cross-sectional standardization."""
        if verbose:
            print("\nCleaning and standardizing features...")
            
        clean_features = {}
        removed = []
        
        for name, feat in features.items():
            # Check NaN percentage
            nan_pct = feat.isna().mean().mean()
            if nan_pct > self.config.max_nan_pct:
                removed.append((name, f"NaN={nan_pct:.1%}"))
                continue
                
            # Check variance
            var = feat.var().mean()
            if var < self.config.min_variance:
                removed.append((name, f"Low variance"))
                continue
                
            # Clean NaNs
            clean_df = feat.ffill().bfill().fillna(0)
            
            # Cross-sectional standardization
            if self.config.standardize:
                clean_df = self._standardize(clean_df)
                
            # Store metadata
            self.feature_metadata[name] = {
                'nan_pct': nan_pct,
                'mean_variance': var,
                'category': self._infer_category(name),
            }
            
            clean_features[name] = clean_df
            
        if verbose:
            print(f"✅ Kept {len(clean_features)} features, removed {len(removed)}")
            if removed and len(removed) <= 10:
                for name, reason in removed:
                    print(f"   ❌ {name}: {reason}")
                    
        return clean_features
    
    def _standardize(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional standardization."""
        if self.config.standardize_method == 'zscore':
            cs_mean = feat.mean(axis=1)
            cs_std = feat.std(axis=1).replace(0, 1)
            result = feat.sub(cs_mean, axis=0).div(cs_std, axis=0)
            
        elif self.config.standardize_method == 'rank':
            result = feat.rank(axis=1, pct=True) - 0.5
            
        elif self.config.standardize_method == 'robust':
            cs_median = feat.median(axis=1)
            cs_mad = (feat.sub(cs_median, axis=0)).abs().median(axis=1).replace(0, 1)
            result = feat.sub(cs_median, axis=0).div(cs_mad, axis=0)
        else:
            result = feat
            
        # Winsorize
        if self.config.winsorize_pct > 0:
            clip_val = 4.0  # ~4 sigma
            result = result.clip(-clip_val, clip_val)
            
        return result.fillna(0)
    
    def _infer_category(self, name: str) -> str:
        """Infer feature category from name."""
        if any(x in name.lower() for x in ['ret_', 'mom_', 'accel', 'tsmom', 'streak']):
            return 'momentum'
        elif any(x in name.lower() for x in ['vol_', 'atr', 'gk_', 'parkinson']):
            return 'volatility'
        elif any(x in name.lower() for x in ['rsi', 'zscore', 'bb_', 'reversal', 'stoch']):
            return 'mean_reversion'
        elif any(x in name.lower() for x in ['cs_', 'beta', 'idio', 'rel_', 'corr', 'disp']):
            return 'stat_arb'
        elif any(x in name.lower() for x in ['ma_', 'macd', 'volume', 'channel', 'mfi', 'obv']):
            return 'technical'
        else:
            return 'other'
    
    def get_feature_matrix(
        self,
        date_range: Tuple[str, str] = None,
        feature_names: List[str] = None
    ) -> Tuple[np.ndarray, pd.Index, pd.Index, List[str]]:
        """
        Get feature matrix in (N, T, F) format for modeling.
        
        Returns:
            X: Feature array (n_assets, n_dates, n_features)
            assets: Asset index
            dates: Date index
            feature_names: List of feature names
        """
        if feature_names is None:
            feature_names = list(self.features.keys())
            
        # Get reference shape
        ref_feat = self.features[feature_names[0]]
        
        if date_range:
            ref_feat = ref_feat.loc[date_range[0]:date_range[1]]
            
        dates = ref_feat.index
        assets = ref_feat.columns
        
        # Build 3D array
        n_assets = len(assets)
        n_dates = len(dates)
        n_features = len(feature_names)
        
        X = np.zeros((n_assets, n_dates, n_features))
        
        for i, fname in enumerate(feature_names):
            feat = self.features[fname]
            if date_range:
                feat = feat.loc[date_range[0]:date_range[1]]
            X[:, :, i] = feat[assets].values.T
            
        return X, assets, dates, feature_names
    
    def analyze_correlations(self) -> pd.DataFrame:
        """Compute feature correlation matrix."""
        # Stack features into 2D for correlation
        stacked = {}
        for name, feat in self.features.items():
            stacked[name] = feat.values.flatten()
            
        corr_df = pd.DataFrame(stacked).corr()
        return corr_df
    
    def compute_feature_ic(
        self,
        forward_returns: pd.DataFrame,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Compute Information Coefficient for each feature.
        
        IC = correlation between feature and forward returns
        """
        fwd_ret = forward_returns.shift(-horizon)
        
        ic_stats = []
        for name, feat in self.features.items():
            # Daily IC
            daily_ic = feat.corrwith(fwd_ret, axis=1)
            
            ic_stats.append({
                'feature': name,
                'category': self.feature_metadata.get(name, {}).get('category', 'unknown'),
                'mean_ic': daily_ic.mean(),
                'std_ic': daily_ic.std(),
                'ic_ir': daily_ic.mean() / (daily_ic.std() + 1e-10),
                'pct_positive': (daily_ic > 0).mean(),
                't_stat': daily_ic.mean() / (daily_ic.std() / np.sqrt(len(daily_ic)) + 1e-10),
            })
            
        return pd.DataFrame(ic_stats).set_index('feature').sort_values('ic_ir', ascending=False)
    
    def save(self, path: str) -> None:
        """Save features to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save features as parquet
        for name, feat in self.features.items():
            feat.to_parquet(os.path.join(path, f"{name}.parquet"))
            
        # Save metadata
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)
            
    def load(self, path: str) -> None:
        """Load features from disk."""
        import glob
        
        # Load features
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        for pfile in parquet_files:
            name = os.path.splitext(os.path.basename(pfile))[0]
            self.features[name] = pd.read_parquet(pfile)
            
        # Load metadata
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.feature_metadata = json.load(f)
    
    def summary(self) -> pd.DataFrame:
        """Get summary of all features."""
        rows = []
        for name, feat in self.features.items():
            meta = self.feature_metadata.get(name, {})
            rows.append({
                'feature': name,
                'category': meta.get('category', 'unknown'),
                'nan_pct': meta.get('nan_pct', 0),
                'mean': feat.mean().mean(),
                'std': feat.std().mean(),
            })
        return pd.DataFrame(rows).set_index('feature')
