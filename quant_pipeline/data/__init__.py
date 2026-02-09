# =============================================================================
# DATA MODULE - Loading, Validation, Splitting
# =============================================================================
# 
# RESPONSIBILITY: Data I/O and integrity checks
# FORBIDDEN: Feature engineering, targets, models
#
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import hashlib
import json

class DataConfig:
    """Immutable configuration for data loading."""
    
    def __init__(
        self,
        data_dir: str,
        oos_start: str = "2024-01-01",
        min_history_days: int = 252,
        random_seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.oos_start = pd.Timestamp(oos_start)
        self.min_history_days = min_history_days
        self.random_seed = random_seed
        
        # Computed at load time
        self._load_timestamp = None
        self._data_hash = None
    
    def to_dict(self) -> dict:
        return {
            'data_dir': str(self.data_dir),
            'oos_start': str(self.oos_start.date()),
            'min_history_days': self.min_history_days,
            'random_seed': self.random_seed,
            'load_timestamp': self._load_timestamp,
            'data_hash': self._data_hash
        }
    
    def __repr__(self):
        return f"DataConfig(oos_start={self.oos_start.date()}, seed={self.random_seed})"


class DataLoader:
    """
    Loads and validates raw price data.
    
    INVARIANTS:
    - Data is sorted by date
    - No future data leaks into IS period
    - Missing data is explicitly handled
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._raw_data: Dict[str, pd.DataFrame] = {}
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._is_loaded = False
    
    def load(self) -> "DataLoader":
        """Load all asset files."""
        asset_files = sorted(self.config.data_dir.glob("Asset_*.csv"))
        
        if len(asset_files) == 0:
            raise ValueError(f"No asset files found in {self.config.data_dir}")
        
        print(f"[DATA] Loading {len(asset_files)} assets...")
        
        for f in asset_files:
            ticker = f.stem
            df = pd.read_csv(f, parse_dates=['Date'])
            df = df.set_index('Date').sort_index()
            
            # Validate required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{ticker} missing columns: {missing}")
            
            self._raw_data[ticker] = df
        
        # Build price matrix
        self._prices = pd.DataFrame({
            k: v['Close'] for k, v in self._raw_data.items()
        }).dropna()
        
        # Compute returns
        self._returns = self._prices.pct_change()
        
        # Compute data hash for reproducibility
        self.config._data_hash = self._compute_hash()
        self.config._load_timestamp = datetime.now().isoformat()
        
        self._is_loaded = True
        print(f"[DATA] Loaded: {len(self._prices)} days, {len(self._prices.columns)} assets")
        print(f"[DATA] Date range: {self._prices.index[0].date()} to {self._prices.index[-1].date()}")
        print(f"[DATA] Hash: {self.config._data_hash[:16]}...")
        
        return self
    
    def _compute_hash(self) -> str:
        """Compute hash of data for reproducibility tracking."""
        data_str = self._prices.to_csv()
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get_prices(self) -> pd.DataFrame:
        """Get price matrix."""
        self._check_loaded()
        return self._prices.copy()
    
    def get_returns(self) -> pd.DataFrame:
        """Get returns matrix."""
        self._check_loaded()
        return self._returns.copy()
    
    def get_raw(self, ticker: str) -> pd.DataFrame:
        """Get raw OHLCV for a specific ticker."""
        self._check_loaded()
        return self._raw_data[ticker].copy()
    
    def get_tickers(self) -> list:
        """Get list of tickers."""
        self._check_loaded()
        return list(self._prices.columns)
    
    def _check_loaded(self):
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")


class DataSplitter:
    """
    Splits data into IS/OOS with strict temporal separation.
    
    INVARIANTS:
    - OOS period is NEVER seen during any IS computation
    - Split is deterministic given config
    - No lookahead bias possible
    """
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.config = loader.config
        self._split_done = False
    
    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into IS/OOS.
        
        Returns:
            prices_is, prices_oos, returns_is, returns_oos
        """
        prices = self.loader.get_prices()
        returns = self.loader.get_returns()
        
        # Strict temporal split
        is_mask = prices.index < self.config.oos_start
        oos_mask = prices.index >= self.config.oos_start
        
        prices_is = prices.loc[is_mask].copy()
        prices_oos = prices.loc[oos_mask].copy()
        returns_is = returns.loc[is_mask].copy()
        returns_oos = returns.loc[oos_mask].copy()
        
        # Validate split
        self._validate_split(prices_is, prices_oos)
        
        self._split_done = True
        
        print(f"[SPLIT] IS: {prices_is.index[0].date()} to {prices_is.index[-1].date()} ({len(prices_is)} days)")
        print(f"[SPLIT] OOS: {prices_oos.index[0].date()} to {prices_oos.index[-1].date()} ({len(prices_oos)} days)")
        print(f"[SPLIT] Ratio: {len(prices_is)/(len(prices_is)+len(prices_oos))*100:.1f}% IS / {len(prices_oos)/(len(prices_is)+len(prices_oos))*100:.1f}% OOS")
        
        return prices_is, prices_oos, returns_is, returns_oos
    
    def _validate_split(self, prices_is: pd.DataFrame, prices_oos: pd.DataFrame):
        """Validate no temporal overlap."""
        if len(prices_is) == 0:
            raise ValueError("IS period is empty")
        if len(prices_oos) == 0:
            raise ValueError("OOS period is empty")
        
        # Check no overlap
        if prices_is.index[-1] >= prices_oos.index[0]:
            raise ValueError(f"Temporal overlap detected: IS ends {prices_is.index[-1]}, OOS starts {prices_oos.index[0]}")
        
        # Check minimum history
        if len(prices_is) < self.config.min_history_days:
            raise ValueError(f"IS period too short: {len(prices_is)} < {self.config.min_history_days}")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_no_future_data(df: pd.DataFrame, reference_date: pd.Timestamp) -> bool:
    """Check that no data exists beyond reference date."""
    if df.index.max() > reference_date:
        raise ValueError(f"Future data detected: max date {df.index.max()} > {reference_date}")
    return True


def validate_sorted(df: pd.DataFrame) -> bool:
    """Check that dataframe is sorted by index."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data is not sorted by date")
    return True


def compute_data_quality_report(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Generate data quality diagnostics."""
    return {
        'n_assets': len(prices.columns),
        'n_days': len(prices),
        'date_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
        'missing_pct': (prices.isna().sum().sum() / prices.size * 100),
        'zero_return_pct': ((returns == 0).sum().sum() / returns.size * 100),
        'extreme_return_pct': ((returns.abs() > 0.20).sum().sum() / returns.size * 100),
        'mean_daily_return': returns.mean().mean() * 100,
        'mean_daily_vol': returns.std().mean() * 100,
    }
