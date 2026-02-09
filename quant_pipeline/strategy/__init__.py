"""
Strategy Module - Position sizing, risk management, and backtesting

This module contains:
- Position sizing algorithms
- Risk constraints
- Backtesting engine
- Performance metrics (Sharpe ONLY computed here)

IMPORTANT: This is the ONLY module where Sharpe ratio is computed.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class PositionSizingMethod(Enum):
    """Position sizing methods."""
    EQUAL_WEIGHT = "equal_weight"
    SIGNAL_PROPORTIONAL = "signal_proportional"
    VOL_PARITY = "vol_parity"
    RANK_WEIGHTED = "rank_weighted"


@dataclass
class StrategyConfig:
    """Configuration for strategy and backtest."""
    # Capital
    initial_capital: float = 1_000_000.0
    
    # Position sizing
    sizing_method: PositionSizingMethod = PositionSizingMethod.SIGNAL_PROPORTIONAL
    max_position_size: float = 0.10  # Max 10% per position
    leverage: float = 1.0  # 1.0 = no leverage
    
    # Long/short
    long_only: bool = False
    neutralize: bool = True  # Force net exposure = 0
    
    # Transaction costs
    transaction_cost_bps: float = 10.0  # 10 bps = 0.10%
    
    # Risk constraints
    max_turnover_daily: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    
    # Signal threshold
    signal_threshold: float = 0.0  # Only trade if |signal| > threshold
    
    def to_dict(self) -> dict:
        return {
            'initial_capital': self.initial_capital,
            'sizing_method': self.sizing_method.value,
            'max_position_size': self.max_position_size,
            'leverage': self.leverage,
            'long_only': self.long_only,
            'neutralize': self.neutralize,
            'transaction_cost_bps': self.transaction_cost_bps,
            'max_turnover_daily': self.max_turnover_daily,
            'stop_loss_pct': self.stop_loss_pct,
            'signal_threshold': self.signal_threshold
        }


# =============================================================================
# POSITION SIZING
# =============================================================================

def compute_positions_equal_weight(
    signals: pd.DataFrame,
    config: StrategyConfig
) -> pd.DataFrame:
    """
    Equal weight position sizing.
    
    Long top N, short bottom N (if not long_only).
    """
    signals = signals.copy()
    
    # Apply threshold
    signals.loc[signals['signal'].abs() < config.signal_threshold, 'signal'] = 0
    
    # Rank signals within each date
    signals['rank'] = signals.groupby('date')['signal'].rank(pct=True)
    
    # Count active positions
    n_assets = signals.groupby('date')['ticker'].transform('count')
    
    if config.long_only:
        # Long top 50%
        signals['position'] = np.where(signals['rank'] > 0.5, 1.0, 0.0)
    else:
        # Long top 50%, short bottom 50%
        signals['position'] = np.where(
            signals['rank'] > 0.5, 1.0,
            np.where(signals['rank'] <= 0.5, -1.0, 0.0)
        )
    
    # Normalize to sum to leverage
    pos_sum = signals.groupby('date')['position'].transform(lambda x: x.abs().sum())
    signals['position'] = signals['position'] / pos_sum * config.leverage
    
    # Apply max position constraint
    signals['position'] = signals['position'].clip(
        -config.max_position_size,
        config.max_position_size
    )
    
    return signals


def compute_positions_signal_proportional(
    signals: pd.DataFrame,
    config: StrategyConfig
) -> pd.DataFrame:
    """
    Position size proportional to signal strength.
    """
    signals = signals.copy()
    
    # Apply threshold
    mask = signals['signal'].abs() < config.signal_threshold
    signals.loc[mask, 'signal'] = 0
    
    if config.long_only:
        signals['position'] = signals['signal'].clip(lower=0)
    else:
        signals['position'] = signals['signal']
    
    # Neutralize if required
    if config.neutralize and not config.long_only:
        mean_pos = signals.groupby('date')['position'].transform('mean')
        signals['position'] = signals['position'] - mean_pos
    
    # Normalize to leverage
    pos_sum = signals.groupby('date')['position'].transform(lambda x: x.abs().sum())
    pos_sum = pos_sum.replace(0, 1)  # Avoid division by zero
    signals['position'] = signals['position'] / pos_sum * config.leverage
    
    # Apply max position constraint
    signals['position'] = signals['position'].clip(
        -config.max_position_size,
        config.max_position_size
    )
    
    return signals


def compute_positions_vol_parity(
    signals: pd.DataFrame,
    volatility: pd.Series,
    config: StrategyConfig
) -> pd.DataFrame:
    """
    Position size inversely proportional to volatility.
    
    Higher volatility = smaller position.
    """
    signals = signals.copy()
    
    # Apply threshold
    mask = signals['signal'].abs() < config.signal_threshold
    signals.loc[mask, 'signal'] = 0
    
    # Inverse volatility
    inv_vol = 1.0 / volatility.replace(0, np.nan)
    inv_vol = inv_vol.fillna(inv_vol.median())
    
    if config.long_only:
        signals['position'] = signals['signal'].clip(lower=0) * inv_vol
    else:
        signals['position'] = signals['signal'] * inv_vol
    
    # Neutralize if required
    if config.neutralize and not config.long_only:
        mean_pos = signals.groupby('date')['position'].transform('mean')
        signals['position'] = signals['position'] - mean_pos
    
    # Normalize to leverage
    pos_sum = signals.groupby('date')['position'].transform(lambda x: x.abs().sum())
    pos_sum = pos_sum.replace(0, 1)
    signals['position'] = signals['position'] / pos_sum * config.leverage
    
    # Apply max position constraint
    signals['position'] = signals['position'].clip(
        -config.max_position_size,
        config.max_position_size
    )
    
    return signals


def compute_positions(
    signals: pd.DataFrame,
    config: StrategyConfig,
    volatility: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute positions based on configuration.
    """
    if config.sizing_method == PositionSizingMethod.EQUAL_WEIGHT:
        return compute_positions_equal_weight(signals, config)
    elif config.sizing_method == PositionSizingMethod.SIGNAL_PROPORTIONAL:
        return compute_positions_signal_proportional(signals, config)
    elif config.sizing_method == PositionSizingMethod.VOL_PARITY:
        if volatility is None:
            raise ValueError("Vol parity requires volatility input")
        return compute_positions_vol_parity(signals, volatility, config)
    else:
        return compute_positions_signal_proportional(signals, config)


# =============================================================================
# BACKTESTING
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a backtest."""
    # Portfolio values
    portfolio_values: pd.Series
    daily_returns: pd.Series
    
    # Position information
    positions: pd.DataFrame
    daily_turnover: pd.Series
    
    # Returns decomposition
    gross_returns: pd.Series
    transaction_costs: pd.Series
    
    # Config used
    config: StrategyConfig
    
    def __post_init__(self):
        """Compute summary statistics."""
        self.total_return = (self.portfolio_values.iloc[-1] / 
                            self.portfolio_values.iloc[0]) - 1
        self.cagr = self._compute_cagr()
        self.sharpe = self._compute_sharpe()
        self.sortino = self._compute_sortino()
        self.max_drawdown = self._compute_max_drawdown()
        self.avg_drawdown = self._compute_avg_drawdown()
        self.total_turnover = self.daily_turnover.sum()
        self.total_tcosts = self.transaction_costs.sum()
    
    def _compute_cagr(self) -> float:
        """Compute compound annual growth rate."""
        n_years = len(self.daily_returns) / 252
        return (1 + self.total_return) ** (1 / n_years) - 1
    
    def _compute_sharpe(self) -> float:
        """Compute annualized Sharpe ratio."""
        if self.daily_returns.std() == 0:
            return 0.0
        return (self.daily_returns.mean() / self.daily_returns.std()) * np.sqrt(252)
    
    def _compute_sortino(self) -> float:
        """Compute Sortino ratio (downside deviation)."""
        downside = self.daily_returns[self.daily_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.inf
        return (self.daily_returns.mean() / downside.std()) * np.sqrt(252)
    
    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown."""
        cummax = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - cummax) / cummax
        return drawdown.min()
    
    def _compute_avg_drawdown(self) -> float:
        """Compute average drawdown."""
        cummax = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - cummax) / cummax
        return drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            'total_return': float(self.total_return),
            'cagr': float(self.cagr),
            'sharpe': float(self.sharpe),
            'sortino': float(self.sortino),
            'max_drawdown': float(self.max_drawdown),
            'avg_drawdown': float(self.avg_drawdown),
            'total_turnover': float(self.total_turnover),
            'total_transaction_costs': float(self.total_tcosts),
            'n_days': len(self.daily_returns)
        }


def run_backtest(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    config: StrategyConfig
) -> BacktestResult:
    """
    Run vectorized backtest.
    
    Parameters:
    -----------
    positions : DataFrame with columns ['date', 'ticker', 'position']
    returns : DataFrame with columns ['date', 'ticker', 'return']
    config : Strategy configuration
    
    Returns:
    --------
    BacktestResult
    """
    # Merge positions with returns
    merged = positions.merge(
        returns[['date', 'ticker', 'return']],
        on=['date', 'ticker'],
        how='left'
    )
    merged['return'] = merged['return'].fillna(0)
    
    # Compute daily portfolio return (position-weighted)
    merged['weighted_return'] = merged['position'] * merged['return']
    daily_gross_returns = merged.groupby('date')['weighted_return'].sum()
    
    # Compute turnover
    positions_wide = positions.pivot(
        index='date', columns='ticker', values='position'
    ).fillna(0)
    
    position_changes = positions_wide.diff().abs().fillna(0)
    daily_turnover = position_changes.sum(axis=1)
    
    # Compute transaction costs
    tc_rate = config.transaction_cost_bps / 10000
    daily_tcosts = daily_turnover * tc_rate
    
    # Net returns
    daily_returns = daily_gross_returns - daily_tcosts
    
    # Portfolio value
    portfolio_values = (1 + daily_returns).cumprod() * config.initial_capital
    
    return BacktestResult(
        portfolio_values=portfolio_values,
        daily_returns=daily_returns,
        positions=positions,
        daily_turnover=daily_turnover,
        gross_returns=daily_gross_returns,
        transaction_costs=daily_tcosts,
        config=config
    )


# =============================================================================
# BENCHMARK
# =============================================================================

def compute_benchmark_equal_weight(
    returns: pd.DataFrame,
    config: StrategyConfig
) -> BacktestResult:
    """
    Compute equal-weight buy-and-hold benchmark.
    """
    # Equal weight all assets
    n_assets = returns.groupby('date')['ticker'].transform('count')
    
    # Daily return is mean across all assets
    daily_returns = returns.groupby('date')['return'].mean()
    
    # Transaction costs: one-time entry
    tc_rate = config.transaction_cost_bps / 10000
    
    # Create artificial positions for benchmark
    positions = returns[['date', 'ticker']].copy()
    positions['position'] = 1.0 / positions.groupby('date')['ticker'].transform('count')
    
    # Portfolio value
    portfolio_values = (1 + daily_returns).cumprod() * config.initial_capital
    
    # Turnover is zero (buy and hold)
    daily_turnover = pd.Series(0, index=daily_returns.index)
    daily_tcosts = pd.Series(0, index=daily_returns.index)
    
    return BacktestResult(
        portfolio_values=portfolio_values,
        daily_returns=daily_returns,
        positions=positions,
        daily_turnover=daily_turnover,
        gross_returns=daily_returns,
        transaction_costs=daily_tcosts,
        config=config
    )


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 63
) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(252)


def compute_drawdown_series(
    portfolio_values: pd.Series
) -> pd.Series:
    """Compute drawdown time series."""
    cummax = portfolio_values.cummax()
    return (portfolio_values - cummax) / cummax


def compute_monthly_returns(
    daily_returns: pd.Series
) -> pd.Series:
    """Compute monthly returns."""
    return daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)


def compute_yearly_returns(
    daily_returns: pd.Series
) -> pd.Series:
    """Compute yearly returns."""
    return daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)


def compute_hit_rate(
    daily_returns: pd.Series
) -> float:
    """Compute win rate (% of positive days)."""
    return (daily_returns > 0).mean()


def compute_profit_factor(
    daily_returns: pd.Series
) -> float:
    """Compute profit factor (gross profits / gross losses)."""
    profits = daily_returns[daily_returns > 0].sum()
    losses = -daily_returns[daily_returns < 0].sum()
    if losses == 0:
        return np.inf
    return profits / losses


def compare_to_benchmark(
    strategy: BacktestResult,
    benchmark: BacktestResult
) -> dict:
    """
    Compare strategy to benchmark.
    """
    return {
        'strategy_sharpe': strategy.sharpe,
        'benchmark_sharpe': benchmark.sharpe,
        'sharpe_diff': strategy.sharpe - benchmark.sharpe,
        'strategy_return': strategy.total_return,
        'benchmark_return': benchmark.total_return,
        'excess_return': strategy.total_return - benchmark.total_return,
        'strategy_max_dd': strategy.max_drawdown,
        'benchmark_max_dd': benchmark.max_drawdown,
        'strategy_turnover': strategy.total_turnover,
        'strategy_tcosts': strategy.total_tcosts
    }


# =============================================================================
# OOS VALIDATION
# =============================================================================

def run_oos_backtest(
    signals_oos: pd.DataFrame,
    returns_oos: pd.DataFrame,
    config: StrategyConfig
) -> BacktestResult:
    """
    Run out-of-sample backtest.
    
    This is the FINAL validation - no changes allowed after seeing OOS results.
    """
    # Compute positions
    positions = compute_positions(signals_oos, config)
    
    # Run backtest
    return run_backtest(positions, returns_oos, config)


def compute_is_oos_comparison(
    is_result: BacktestResult,
    oos_result: BacktestResult
) -> dict:
    """
    Compare in-sample to out-of-sample performance.
    
    Key diagnostic: How much does performance decay?
    """
    sharpe_decay = (is_result.sharpe - oos_result.sharpe) / is_result.sharpe if is_result.sharpe != 0 else 0
    
    return {
        'is_sharpe': is_result.sharpe,
        'oos_sharpe': oos_result.sharpe,
        'sharpe_decay_pct': sharpe_decay,
        'is_return': is_result.total_return,
        'oos_return': oos_result.total_return,
        'is_max_dd': is_result.max_drawdown,
        'oos_max_dd': oos_result.max_drawdown,
        'is_turnover': is_result.total_turnover,
        'oos_turnover': oos_result.total_turnover
    }
