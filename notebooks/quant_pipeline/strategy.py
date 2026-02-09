"""
Quant Pipeline - Strategy Module

Implements backtesting engine and portfolio construction for quantitative strategies.

NO FORWARD BIAS: All computations at time t use only data available at time t.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import warnings


# ============================================================================
# Configuration Classes
# ============================================================================

class PositionSizingMethod(Enum):
    """Position sizing methods."""
    EQUAL_WEIGHT = "equal_weight"
    SIGNAL_PROPORTIONAL = "signal_proportional"
    VOLATILITY_SCALED = "volatility_scaled"
    RISK_PARITY = "risk_parity"


@dataclass
class StrategyConfig:
    """Configuration for backtesting."""
    initial_capital: float = 1_000_000.0
    sizing_method: PositionSizingMethod = PositionSizingMethod.SIGNAL_PROPORTIONAL
    max_position_size: float = 0.10  # Max weight per asset
    leverage: float = 1.0
    long_only: bool = False
    neutralize: bool = True  # Dollar neutral (long = short)
    transaction_cost_bps: float = 10.0
    vol_lookback: int = 20  # For vol-scaled sizing
    rebalance_freq: int = 1  # 1 = daily, 5 = weekly, etc.
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'initial_capital': self.initial_capital,
            'sizing_method': self.sizing_method.value,
            'max_position_size': self.max_position_size,
            'leverage': self.leverage,
            'long_only': self.long_only,
            'neutralize': self.neutralize,
            'transaction_cost_bps': self.transaction_cost_bps
        }


@dataclass
class BacktestResult:
    """Results from backtesting."""
    sharpe: float
    total_return: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    avg_drawdown: float
    turnover: float  # Annualized
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    portfolio_values: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'sharpe': self.sharpe,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'annual_volatility': self.annual_volatility,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'turnover': self.turnover,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio
        }


# ============================================================================
# Position Computation
# ============================================================================

def compute_positions(
    signal_data: pd.DataFrame,
    config: StrategyConfig,
    volatility_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute portfolio positions from signals.
    
    Parameters
    ----------
    signal_data : pd.DataFrame
        Long-format with columns: date, ticker, signal
    config : StrategyConfig
        Strategy configuration
    volatility_data : pd.DataFrame, optional
        Volatility estimates for vol-scaled sizing
        
    Returns
    -------
    pd.DataFrame
        Long-format with columns: date, ticker, position
    """
    # Pivot to wide format for cross-sectional normalization
    signals_wide = signal_data.pivot(index='date', columns='ticker', values='signal')
    
    if config.sizing_method == PositionSizingMethod.EQUAL_WEIGHT:
        # Equal weight for non-zero signals
        positions_wide = np.sign(signals_wide)
        row_sums = positions_wide.abs().sum(axis=1)
        positions_wide = positions_wide.div(row_sums, axis=0).fillna(0)
    
    elif config.sizing_method == PositionSizingMethod.SIGNAL_PROPORTIONAL:
        # Proportional to signal strength
        positions_wide = signals_wide.copy()
        
        # Cross-sectional normalization
        row_sums = positions_wide.abs().sum(axis=1)
        positions_wide = positions_wide.div(row_sums, axis=0).fillna(0)
    
    elif config.sizing_method == PositionSizingMethod.VOLATILITY_SCALED:
        # Inverse volatility weighting
        if volatility_data is None:
            # Estimate from returns
            warnings.warn("No volatility data provided, using equal weight")
            positions_wide = signals_wide.copy()
            row_sums = positions_wide.abs().sum(axis=1)
            positions_wide = positions_wide.div(row_sums, axis=0).fillna(0)
        else:
            vol_wide = volatility_data.pivot(index='date', columns='ticker', values='volatility')
            inv_vol = 1.0 / vol_wide.clip(lower=0.01)
            positions_wide = signals_wide * inv_vol
            row_sums = positions_wide.abs().sum(axis=1)
            positions_wide = positions_wide.div(row_sums, axis=0).fillna(0)
    
    else:
        raise ValueError(f"Unknown sizing method: {config.sizing_method}")
    
    # Apply constraints
    if config.long_only:
        positions_wide = positions_wide.clip(lower=0)
        # Renormalize
        row_sums = positions_wide.sum(axis=1)
        positions_wide = positions_wide.div(row_sums, axis=0).fillna(0)
    
    if config.neutralize and not config.long_only:
        # Dollar neutral: long weights sum to +0.5, short to -0.5
        long_weights = positions_wide.clip(lower=0)
        short_weights = positions_wide.clip(upper=0)
        
        long_sums = long_weights.sum(axis=1)
        short_sums = short_weights.abs().sum(axis=1)
        
        long_weights = long_weights.div(long_sums * 2, axis=0).fillna(0)
        short_weights = -short_weights.abs().div(short_sums * 2, axis=0).fillna(0)
        
        positions_wide = long_weights + short_weights
    
    # Cap individual positions
    positions_wide = positions_wide.clip(-config.max_position_size, config.max_position_size)
    
    # Apply leverage
    positions_wide = positions_wide * config.leverage
    
    # Convert back to long format
    positions_long = positions_wide.reset_index().melt(
        id_vars='date',
        var_name='ticker',
        value_name='position'
    )
    
    return positions_long


# ============================================================================
# Backtesting Engine
# ============================================================================

def run_backtest(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    config: StrategyConfig
) -> BacktestResult:
    """
    Run backtest simulation.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Long-format with columns: date, ticker, position
    returns : pd.DataFrame
        Long-format with columns: date, ticker, return
    config : StrategyConfig
        Strategy configuration
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    # Merge positions with returns
    merged = positions.merge(returns, on=['date', 'ticker'], how='inner')
    
    # Pivot to wide format
    positions_wide = merged.pivot(index='date', columns='ticker', values='position')
    returns_wide = merged.pivot(index='date', columns='ticker', values='return')
    
    # Align
    common_cols = positions_wide.columns.intersection(returns_wide.columns)
    positions_wide = positions_wide[common_cols].fillna(0)
    returns_wide = returns_wide[common_cols].fillna(0)
    
    # Compute daily portfolio return
    # Position at time t earns return from t to t+1
    positions_shifted = positions_wide.shift(1)  # Positions known at start of day
    daily_gross_returns = (positions_shifted * returns_wide).sum(axis=1)
    
    # Compute turnover
    position_changes = (positions_wide - positions_wide.shift(1)).abs()
    daily_turnover = position_changes.sum(axis=1).fillna(0)
    
    # Transaction costs (apply to changes)
    tc_bps = config.transaction_cost_bps
    daily_tc = daily_turnover * tc_bps / 10000
    
    # Net returns
    daily_net_returns = daily_gross_returns - daily_tc
    daily_net_returns = daily_net_returns.dropna()
    
    # Compute portfolio value
    portfolio_values = config.initial_capital * (1 + daily_net_returns).cumprod()
    
    # Compute metrics
    n_days = len(daily_net_returns)
    annual_factor = 252
    
    avg_return = daily_net_returns.mean()
    std_return = daily_net_returns.std()
    
    sharpe = avg_return / std_return * np.sqrt(annual_factor) if std_return > 0 else 0
    
    total_return = portfolio_values.iloc[-1] / config.initial_capital - 1
    annual_return = (1 + total_return) ** (annual_factor / n_days) - 1 if n_days > 0 else 0
    annual_volatility = std_return * np.sqrt(annual_factor)
    
    # Drawdown
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    
    # Turnover (annualized)
    avg_daily_turnover = daily_turnover.mean()
    annualized_turnover = avg_daily_turnover * annual_factor
    
    # Win rate
    wins = (daily_net_returns > 0).sum()
    total_days = (daily_net_returns != 0).sum()
    win_rate = wins / total_days if total_days > 0 else 0.5
    
    # Profit factor
    gross_profit = daily_net_returns[daily_net_returns > 0].sum()
    gross_loss = daily_net_returns[daily_net_returns < 0].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    return BacktestResult(
        sharpe=sharpe,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        max_drawdown=max_drawdown,
        avg_drawdown=avg_drawdown,
        turnover=annualized_turnover,
        win_rate=win_rate,
        profit_factor=profit_factor,
        calmar_ratio=calmar_ratio,
        portfolio_values=portfolio_values,
        daily_returns=daily_net_returns,
        positions=positions_wide
    )


# ============================================================================
# Benchmark Computation
# ============================================================================

def compute_benchmark_equal_weight(
    returns: pd.DataFrame,
    config: StrategyConfig
) -> BacktestResult:
    """
    Compute equal-weight benchmark.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Long-format with columns: date, ticker, return
    config : StrategyConfig
        Strategy configuration (for initial capital)
        
    Returns
    -------
    BacktestResult
        Benchmark results
    """
    # Pivot to wide
    returns_wide = returns.pivot(index='date', columns='ticker', values='return')
    
    # Equal weight all assets
    n_assets = returns_wide.shape[1]
    positions_wide = pd.DataFrame(
        1.0 / n_assets,
        index=returns_wide.index,
        columns=returns_wide.columns
    )
    
    # Convert to long format
    positions_long = positions_wide.reset_index().melt(
        id_vars='date',
        var_name='ticker',
        value_name='position'
    )
    
    # Run backtest with reduced TC (benchmark is buy-and-hold)
    benchmark_config = StrategyConfig(
        initial_capital=config.initial_capital,
        transaction_cost_bps=1.0  # Minimal TC for benchmark
    )
    
    return run_backtest(positions_long, returns, benchmark_config)


def compare_to_benchmark(
    strategy_result: BacktestResult,
    benchmark_result: BacktestResult
) -> Dict[str, float]:
    """
    Compare strategy to benchmark.
    """
    return {
        'sharpe_diff': strategy_result.sharpe - benchmark_result.sharpe,
        'return_diff': strategy_result.total_return - benchmark_result.total_return,
        'risk_diff': strategy_result.annual_volatility - benchmark_result.annual_volatility,
        'drawdown_diff': strategy_result.max_drawdown - benchmark_result.max_drawdown,
        'information_ratio': (strategy_result.annual_return - benchmark_result.annual_return) / 
                            (strategy_result.annual_volatility + 1e-6) if strategy_result.annual_volatility > 0 else 0
    }


# ============================================================================
# Utility Functions
# ============================================================================

def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 63  # ~3 months
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    return rolling_mean / rolling_std * np.sqrt(252)


def compute_drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    """
    Compute drawdown series from portfolio values.
    """
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    return drawdown


def compute_position_summary(positions: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics of positions.
    """
    return {
        'avg_long_exposure': positions[positions > 0].sum(axis=1).mean(),
        'avg_short_exposure': positions[positions < 0].abs().sum(axis=1).mean(),
        'avg_gross_exposure': positions.abs().sum(axis=1).mean(),
        'avg_net_exposure': positions.sum(axis=1).mean(),
        'n_long_avg': (positions > 0).sum(axis=1).mean(),
        'n_short_avg': (positions < 0).sum(axis=1).mean()
    }
