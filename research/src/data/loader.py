"""
Data Loader Module
==================

Handles loading and preprocessing of the 10-year daily OHLCV data for 66 assets.

Data Split Strategy (for ~2500 data points = 10 years):
- Training: 2016-2021 (~6 years, ~1500 points)
- Validation: 2022-2023 (~2 years, ~500 points) - for hyperparameter tuning
- Holdout/Test: 2024-2026 (~2 years, ~500 points) - NEVER touched until final eval

Walk-Forward within Training:
- Expanding window with periodic retraining
- Use validation for hyperparameter selection
- Final models tested only on holdout
"""

import os
import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""
    data_dir: str = "../data/raw/assets"
    
    # Split dates (adjusted for 2016-2026 data)
    train_start: str = "2016-01-01"
    train_end: str = "2021-12-31"      # ~6 years
    val_start: str = "2022-01-01"
    val_end: str = "2023-12-31"        # ~2 years validation
    test_start: str = "2024-01-01"
    test_end: str = "2026-12-31"       # ~2 years holdout (NEVER touch until final)
    
    # Walk-forward settings
    initial_train_window: int = 252    # 1 year minimum
    retrain_interval: int = 21         # Monthly retrain
    
    # Data quality
    min_history_days: int = 252        # Require 1 year history
    max_nan_pct: float = 0.05          # Max 5% NaN per asset


class DataLoader:
    """
    Load and manage the 66-asset daily OHLCV dataset.
    
    Provides:
    - Panel data (date x asset) for returns, prices, volume
    - Clean train/val/test splits
    - Walk-forward iterators
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.assets: Dict[str, pd.DataFrame] = {}
        self.panel: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict = {}
        
    def load_all(self, data_dir: Optional[str] = None) -> None:
        """Load all asset CSVs from directory."""
        data_dir = data_dir or self.config.data_dir
        paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        
        if not paths:
            raise FileNotFoundError(f"No CSVs found in {data_dir}")
        
        print(f"Loading {len(paths)} assets from {data_dir}...")
        
        for path in paths:
            asset_name = os.path.splitext(os.path.basename(path))[0]
            df = self._load_single(path)
            if df is not None:
                self.assets[asset_name] = df
        
        print(f"✅ Loaded {len(self.assets)} assets")
        self._build_panel()
        self._compute_metadata()
        
    def _load_single(self, path: str) -> Optional[pd.DataFrame]:
        """Load and clean single asset CSV."""
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Parse date
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date').drop_duplicates(subset=['date'])
            df = df.set_index('date')
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    return None
                    
            return df[required]
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _build_panel(self) -> None:
        """Build panel data from individual assets."""
        if not self.assets:
            return
            
        # Build returns panel
        close_data = {}
        volume_data = {}
        
        for asset, df in self.assets.items():
            close_data[asset] = df['close']
            volume_data[asset] = df['volume']
        
        close_panel = pd.DataFrame(close_data)
        volume_panel = pd.DataFrame(volume_data)
        
        # Compute log returns
        returns_panel = np.log(close_panel / close_panel.shift(1))
        
        # Store panels
        self.panel = {
            'close': close_panel,
            'returns': returns_panel,
            'volume': volume_panel,
            'open': pd.DataFrame({a: df['open'] for a, df in self.assets.items()}),
            'high': pd.DataFrame({a: df['high'] for a, df in self.assets.items()}),
            'low': pd.DataFrame({a: df['low'] for a, df in self.assets.items()}),
        }
        
        print(f"📊 Panel shape: {close_panel.shape[0]} days x {close_panel.shape[1]} assets")
        print(f"   Date range: {close_panel.index[0].strftime('%Y-%m-%d')} to {close_panel.index[-1].strftime('%Y-%m-%d')}")
        
    def _compute_metadata(self) -> None:
        """Compute dataset metadata."""
        returns = self.panel['returns']
        
        self.metadata = {
            'n_assets': len(self.assets),
            'n_days': len(returns),
            'date_start': returns.index[0],
            'date_end': returns.index[-1],
            'years': len(returns) / 252,
            'nan_pct': returns.isna().mean().mean() * 100,
        }
        
    def get_split(self, split: str = 'train') -> Dict[str, pd.DataFrame]:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', 'test', or 'train_val' (train + val combined)
            
        Returns:
            Dictionary with panel data for the split period
        """
        if split == 'train':
            start, end = self.config.train_start, self.config.train_end
        elif split == 'val':
            start, end = self.config.val_start, self.config.val_end
        elif split == 'test':
            start, end = self.config.test_start, self.config.test_end
        elif split == 'train_val':
            start, end = self.config.train_start, self.config.val_end
        else:
            raise ValueError(f"Unknown split: {split}")
            
        result = {}
        for name, panel in self.panel.items():
            result[name] = panel.loc[start:end].copy()
            
        return result
    
    def get_returns(self, split: str = 'train') -> pd.DataFrame:
        """Convenience method to get returns for a split."""
        return self.get_split(split)['returns']
    
    def walk_forward_indices(
        self,
        train_window: Optional[int] = None,
        test_window: int = 21,
        gap: int = 1,
        expanding: bool = True,
        split: str = 'train_val'
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate walk-forward train/test indices.
        
        Args:
            train_window: Training window size (None for expanding)
            test_window: Test window size (default 21 days = 1 month)
            gap: Gap between train end and test start (for label leakage prevention)
            expanding: If True, use expanding window; else rolling
            split: Which split to generate indices for
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        returns = self.get_returns(split)
        dates = returns.index
        n = len(dates)
        
        train_window = train_window or self.config.initial_train_window
        
        indices = []
        test_start = train_window + gap
        
        while test_start + test_window <= n:
            test_end = test_start + test_window
            
            if expanding:
                train_idx = dates[:test_start - gap]
            else:
                train_idx = dates[test_start - gap - train_window:test_start - gap]
                
            test_idx = dates[test_start:test_end]
            
            indices.append((train_idx, test_idx))
            test_start += test_window
            
        return indices
    
    def summary(self) -> pd.DataFrame:
        """Return summary statistics for each asset."""
        returns = self.panel['returns']
        
        stats = []
        for asset in returns.columns:
            r = returns[asset].dropna()
            stats.append({
                'asset': asset,
                'n_days': len(r),
                'ann_return': r.mean() * 252 * 100,
                'ann_vol': r.std() * np.sqrt(252) * 100,
                'sharpe': (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0,
                'max_dd': self._max_drawdown(r) * 100,
                'nan_pct': returns[asset].isna().mean() * 100,
            })
            
        return pd.DataFrame(stats).set_index('asset')
    
    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """Compute max drawdown from returns series."""
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()


if __name__ == "__main__":
    # Quick test
    loader = DataLoader()
    loader.load_all("../../data/raw/assets")
    print(loader.metadata)
    print(loader.summary().head())
