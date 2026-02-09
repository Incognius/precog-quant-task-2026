"""
Backtesting Framework
=====================

WorldQuant-style backtester with proper metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Backtest settings."""
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 10  # 10 bps = 0.1%
    execution_lag: int = 1  # Days between signal and execution
    min_assets: int = 10  # Minimum assets to trade
    max_position_pct: float = 0.10  # Max 10% per asset
    

class Backtester:
    """
    WorldQuant-style backtester.
    
    Features:
    - Unit-gross (L1-normalized) positions
    - Transaction costs
    - Execution lag
    - Additive PnL (no compounding)
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
    def run(
        self,
        alpha: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Dict:
        """
        Run backtest on alpha signal.
        
        Args:
            alpha: Alpha signal (date x asset), should be dollar-neutral
            returns: Forward returns (date x asset)
            
        Returns:
            Dict with returns, metrics, diagnostics
        """
        # Align data
        common_dates = alpha.index.intersection(returns.index)
        common_assets = alpha.columns.intersection(returns.columns)
        
        alpha = alpha.loc[common_dates, common_assets].copy()
        returns = returns.loc[common_dates, common_assets].copy()
        
        # Apply execution lag
        weights = alpha.shift(self.config.execution_lag)
        
        # Fill NaN weights with 0 (no position)
        weights = weights.fillna(0)
        
        # L1 normalize (unit gross)
        weights_norm = self._l1_normalize(weights)
        
        # Compute portfolio returns
        portfolio_returns = (weights_norm * returns.fillna(0)).sum(axis=1)
        
        # Compute turnover
        weights_prev = weights_norm.shift(1).fillna(0)
        turnover = (weights_norm - weights_prev).abs().sum(axis=1) * 0.5
        
        # Transaction costs
        tc = turnover * (self.config.transaction_cost_bps / 10000)
        
        # Net returns
        net_returns = portfolio_returns - tc
        
        # Compute metrics
        metrics = self._compute_metrics(portfolio_returns, net_returns, turnover)
        
        return {
            'gross_returns': portfolio_returns,
            'net_returns': net_returns,
            'turnover': turnover,
            'tc': tc,
            'weights': weights_norm,
            'metrics': metrics,
        }
    
    def _l1_normalize(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Normalize weights to unit gross."""
        gross = weights.abs().sum(axis=1)
        gross = gross.replace(0, 1)  # Avoid division by zero
        return weights.div(gross, axis=0)
    
    def _compute_metrics(
        self,
        gross_returns: pd.Series,
        net_returns: pd.Series,
        turnover: pd.Series
    ) -> Dict:
        """Compute performance metrics."""
        def sharpe(r: pd.Series, periods: int = 252) -> float:
            if r.std() == 0:
                return 0
            return r.mean() / r.std() * np.sqrt(periods)
        
        def max_drawdown(r: pd.Series) -> float:
            cum = (1 + r).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak
            return dd.min()
        
        def calmar(r: pd.Series) -> float:
            ann_ret = r.mean() * 252
            mdd = abs(max_drawdown(r))
            return ann_ret / mdd if mdd > 0 else 0
        
        return {
            'gross_sharpe': sharpe(gross_returns),
            'net_sharpe': sharpe(net_returns),
            'gross_ann_return': gross_returns.mean() * 252,
            'net_ann_return': net_returns.mean() * 252,
            'gross_ann_vol': gross_returns.std() * np.sqrt(252),
            'net_ann_vol': net_returns.std() * np.sqrt(252),
            'max_drawdown': max_drawdown(net_returns),
            'calmar': calmar(net_returns),
            'avg_turnover': turnover.mean(),
            'ann_turnover': turnover.mean() * 252,
            'total_return': (1 + net_returns).prod() - 1,
            'n_days': len(net_returns),
        }


def analyze_regime_performance(
    returns: pd.Series,
    market_returns: pd.Series,
    vol_regime: pd.Series = None
) -> Dict:
    """
    Analyze performance by market regime.
    
    Regimes:
    - Bull (market up > 1%)
    - Bear (market down > 1%) 
    - Sideways (in between)
    - High vol
    - Low vol
    """
    # Market regime
    rolling_mkt = market_returns.rolling(21).sum()
    
    bull = rolling_mkt > 0.02
    bear = rolling_mkt < -0.02
    sideways = ~bull & ~bear
    
    # Vol regime
    if vol_regime is None:
        vol_21 = market_returns.rolling(21).std() * np.sqrt(252)
        vol_median = vol_21.rolling(252).median()
        high_vol = vol_21 > vol_median
        low_vol = vol_21 <= vol_median
    else:
        high_vol = vol_regime > 0.5
        low_vol = vol_regime <= 0.5
    
    def regime_stats(mask):
        r = returns[mask].dropna()
        if len(r) < 10:
            return {'sharpe': np.nan, 'return': np.nan, 'n_days': len(r)}
        return {
            'sharpe': r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0,
            'return': r.mean() * 252,
            'n_days': len(r),
            'win_rate': (r > 0).mean(),
        }
    
    return {
        'bull': regime_stats(bull),
        'bear': regime_stats(bear),
        'sideways': regime_stats(sideways),
        'high_vol': regime_stats(high_vol),
        'low_vol': regime_stats(low_vol),
    }


def compute_win_loss_stats(returns: pd.Series) -> Dict:
    """Compute detailed win/loss statistics."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    return {
        'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0,
        'avg_win': wins.mean() if len(wins) > 0 else 0,
        'avg_loss': losses.mean() if len(losses) > 0 else 0,
        'win_loss_ratio': abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0,
        'profit_factor': abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0,
        'max_consecutive_wins': _max_consecutive(returns > 0),
        'max_consecutive_losses': _max_consecutive(returns < 0),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
    }


def _max_consecutive(mask: pd.Series) -> int:
    """Count max consecutive True values."""
    groups = (~mask).cumsum()
    return mask.groupby(groups).sum().max() if len(mask) > 0 else 0
