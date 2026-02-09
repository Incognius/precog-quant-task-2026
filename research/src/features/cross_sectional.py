"""
Cross-Sectional Features
========================

Features that are computed by comparing assets cross-sectionally.

Features:
1. Cross-sectional standardization
2. Industry/Sector neutralization (placeholder)
3. Factor exposures
4. Relative strength
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class CrossSectionalFeatures:
    """
    Generate cross-sectional normalized features.
    
    Philosophy:
    - Raw features need cross-sectional standardization for comparability
    - Dollar-neutral portfolios require centered alphas
    - Outliers should be handled robustly
    """
    
    def __init__(self):
        self.features: Dict[str, pd.DataFrame] = {}
        
    def standardize(
        self,
        feature: pd.DataFrame,
        method: str = 'zscore',
        winsorize_pct: float = 0.01
    ) -> pd.DataFrame:
        """
        Cross-sectionally standardize a feature.
        
        Args:
            feature: DataFrame (date x asset)
            method: 'zscore', 'rank', or 'robust'
            winsorize_pct: Percentile for winsorization
            
        Returns:
            Standardized feature
        """
        if method == 'zscore':
            cs_mean = feature.mean(axis=1)
            cs_std = feature.std(axis=1).replace(0, 1)
            result = feature.sub(cs_mean, axis=0).div(cs_std, axis=0)
            
        elif method == 'rank':
            result = feature.rank(axis=1, pct=True) - 0.5
            
        elif method == 'robust':
            # Median and MAD (median absolute deviation)
            cs_median = feature.median(axis=1)
            cs_mad = (feature.sub(cs_median, axis=0)).abs().median(axis=1)
            cs_mad = cs_mad.replace(0, 1)
            result = feature.sub(cs_median, axis=0).div(cs_mad, axis=0)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Winsorize
        if winsorize_pct > 0:
            lower = result.quantile(winsorize_pct, axis=1)
            upper = result.quantile(1 - winsorize_pct, axis=1)
            result = result.clip(lower, upper, axis=0)
            
        return result
    
    def compute_relative_strength(
        self,
        returns: pd.DataFrame,
        windows: List[int] = [5, 21, 63]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute relative strength features.
        
        Relative strength = asset return rank vs cross-section
        """
        features = {}
        
        for window in windows:
            cumret = returns.rolling(window).sum()
            rs = self.standardize(cumret, method='rank')
            features[f'rel_strength_{window}d'] = rs
            
        return features
    
    def demean(self, feature: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional demean (for dollar neutrality)."""
        cs_mean = feature.mean(axis=1)
        return feature.sub(cs_mean, axis=0)
    
    def compute_pca_factors(
        self,
        returns: pd.DataFrame,
        n_components: int = 5,
        window: int = 252
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling PCA factor exposures.
        
        Returns factor loadings for each asset over time.
        """
        from sklearn.decomposition import PCA
        
        features = {}
        
        # Rolling PCA is expensive, do it sparingly
        exposures = {f'pca_factor_{i+1}': [] for i in range(n_components)}
        dates = []
        
        for i in range(window, len(returns), 21):  # Monthly
            window_data = returns.iloc[i-window:i].dropna(axis=1, how='any')
            
            if window_data.shape[1] < n_components + 5:
                continue
                
            pca = PCA(n_components=n_components)
            pca.fit(window_data.T)  # Assets as samples
            
            # Get loadings (how each date loads on factors)
            loadings = pca.transform(window_data.T)
            
            # Store
            dates.append(returns.index[i])
            for j in range(n_components):
                exp_dict = dict(zip(window_data.columns, loadings[:, j]))
                exposures[f'pca_factor_{j+1}'].append(exp_dict)
                
        # Convert to DataFrames
        for fname, exp_list in exposures.items():
            if exp_list:
                df = pd.DataFrame(exp_list, index=dates)
                df = df.reindex(returns.index).ffill()
                features[fname] = df
                
        return features


def standardize_features(
    features: Dict[str, pd.DataFrame],
    method: str = 'zscore',
    winsorize_pct: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """Cross-sectionally standardize all features."""
    csf = CrossSectionalFeatures()
    result = {}
    
    for name, feat in features.items():
        result[name] = csf.standardize(feat, method, winsorize_pct)
        
    return result
