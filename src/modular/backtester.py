"""
Proper Backtester
==================

A rigorous, bias-free backtesting engine.

CRITICAL TIMING RULES:
1. Signal at time T (using data up to T close)
2. Decision made after T close
3. Trade executed at T+1 open (in practice)
4. For daily simulation: earn return from T to T+1 close

NO LOOK-AHEAD BIAS GUARANTEES:
- Position on day T determined by signal from day T-1
- Never use future data for any calculation
- Proper handling of rebalancing dates

Author: Precog Quant Research
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 10  # 10 bps = 0.10%
    rebalance_freq_days: int = 21  # Monthly
    top_pct: float = 0.20  # Long top 20%
    bottom_pct: float = 0.20  # Short bottom 20%
    max_position_size: float = 0.10  # Max 10% per position
    stop_loss_pct: Optional[float] = None  # Optional stop loss
    signal_col: str = 'signal_zscore'
    return_col: str = 'fwd_ret_1d'  # 1-day forward return


@dataclass
class BacktestResults:
    """Results from backtesting."""
    daily_returns: pd.DataFrame
    metrics: Dict
    positions: pd.DataFrame
    trades: pd.DataFrame


class Backtester:
    """
    Rigorous backtesting engine.
    
    Usage:
        bt = Backtester(predictions_df, config)
        results = bt.run_long_only()
        results = bt.run_long_short()
        results = bt.run_short_only()
        
        bt.print_metrics(results)
        bt.plot_results(results)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: ['date', 'ticker', signal_col, return_col]
        config : BacktestConfig
            Backtesting configuration
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['date', 'ticker'])
        
        self.config = config or BacktestConfig()
        
        # Validate
        required_cols = ['date', 'ticker', self.config.signal_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if self.config.return_col not in df.columns:
            print(f"⚠️ Return column '{self.config.return_col}' not found, will need to be added")
        
        self.dates = sorted(self.df['date'].unique())
    
    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        rets = results_df['net_return'].dropna()
        
        if len(rets) == 0:
            return {}
        
        # Cumulative return
        cumulative = (1 + rets).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized metrics
        n_years = len(rets) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_return / (ann_vol + 1e-10)
        
        # Drawdown
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (rets > 0).mean()
        
        # Turnover
        total_turnover = results_df['turnover'].sum()
        ann_turnover = total_turnover / max(n_years, 0.1)
        
        # Sortino ratio (downside vol)
        downside_rets = rets[rets < 0]
        downside_vol = downside_rets.std() * np.sqrt(252) if len(downside_rets) > 0 else 0
        sortino = ann_return / (downside_vol + 1e-10)
        
        # Calmar ratio (return / max dd)
        calmar = ann_return / (abs(max_dd) + 1e-10)
        
        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'ann_turnover': ann_turnover,
            'n_days': len(rets),
            'n_years': n_years
        }
    
    def run_long_only(
        self,
        signal_col: Optional[str] = None,
        top_pct: Optional[float] = None,
        rebalance_freq: Optional[int] = None
    ) -> BacktestResults:
        """
        Run long-only backtest.
        
        TIMING:
        - Day T: Rank stocks by signal (computed using data up to T)
        - Day T: Select top N% for long portfolio
        - Day T+1: Earn return from T to T+1 (using fwd_ret_1d at T)
        
        Parameters
        ----------
        signal_col : str
            Signal column (default: config.signal_col)
        top_pct : float
            Percentage of stocks to long (default: config.top_pct)
        rebalance_freq : int
            Days between rebalancing (default: config.rebalance_freq_days)
        """
        signal_col = signal_col or self.config.signal_col
        top_pct = top_pct or self.config.top_pct
        rebalance_freq = rebalance_freq or self.config.rebalance_freq_days
        return_col = self.config.return_col
        tc_bps = self.config.transaction_cost_bps
        
        df = self.df.copy()
        dates = df['date'].unique()
        
        results = []
        positions_list = []
        current_positions = None
        last_rebal_idx = -rebalance_freq
        
        for i, date in enumerate(dates):
            day_df = df[df['date'] == date].set_index('ticker')
            
            # Rebalance?
            if i - last_rebal_idx >= rebalance_freq or current_positions is None:
                # Get signals
                signals = day_df[signal_col].dropna()
                n_long = max(1, int(len(signals) * top_pct))
                
                # Select top N%
                ranked = signals.sort_values(ascending=False)
                new_positions = set(ranked.head(n_long).index)
                
                # Calculate turnover
                if current_positions is not None:
                    exited = len(current_positions - new_positions)
                    entered = len(new_positions - current_positions)
                    turnover = (exited + entered) / 2  # One-sided
                else:
                    turnover = n_long
                
                current_positions = new_positions
                last_rebal_idx = i
                
                # Transaction cost
                tc = turnover / n_long * tc_bps / 10000 if n_long > 0 else 0
            else:
                turnover = 0
                tc = 0
            
            # Calculate portfolio return
            if current_positions and return_col in day_df.columns:
                available = [t for t in current_positions if t in day_df.index]
                if available:
                    # Equal weight
                    port_return = day_df.loc[available, return_col].mean()
                else:
                    port_return = 0
            else:
                port_return = 0
            
            net_return = port_return - tc
            
            results.append({
                'date': date,
                'gross_return': port_return,
                'tc': tc,
                'net_return': net_return,
                'turnover': turnover,
                'n_positions': len(current_positions) if current_positions else 0
            })
            
            # Track positions
            if current_positions:
                for ticker in current_positions:
                    positions_list.append({
                        'date': date,
                        'ticker': ticker,
                        'side': 'long',
                        'weight': 1.0 / len(current_positions)
                    })
        
        results_df = pd.DataFrame(results)
        results_df['cumulative'] = (1 + results_df['net_return']).cumprod()
        
        positions_df = pd.DataFrame(positions_list) if positions_list else pd.DataFrame()
        
        metrics = self._calculate_metrics(results_df)
        
        return BacktestResults(
            daily_returns=results_df,
            metrics=metrics,
            positions=positions_df,
            trades=pd.DataFrame()  # TODO: Track individual trades
        )
    
    def run_short_only(
        self,
        signal_col: Optional[str] = None,
        bottom_pct: Optional[float] = None,
        rebalance_freq: Optional[int] = None
    ) -> BacktestResults:
        """
        Run short-only backtest.
        
        IMPORTANT: For short-only, we SHORT stocks with HIGHEST signal
        (assuming signal is oriented as "higher = more shortable")
        
        If your signal is oriented as "higher = better performance",
        flip it before using this function.
        
        TIMING:
        - Day T: Rank stocks by signal
        - Day T: Select top N% for SHORT portfolio (highest shortability)
        - Day T+1: Profit = -return (we make money when stocks go down)
        """
        signal_col = signal_col or self.config.signal_col
        bottom_pct = bottom_pct or self.config.bottom_pct
        rebalance_freq = rebalance_freq or self.config.rebalance_freq_days
        return_col = self.config.return_col
        tc_bps = self.config.transaction_cost_bps
        
        df = self.df.copy()
        dates = df['date'].unique()
        
        results = []
        positions_list = []
        current_positions = None
        last_rebal_idx = -rebalance_freq
        
        for i, date in enumerate(dates):
            day_df = df[df['date'] == date].set_index('ticker')
            
            # Rebalance?
            if i - last_rebal_idx >= rebalance_freq or current_positions is None:
                signals = day_df[signal_col].dropna()
                n_short = max(1, int(len(signals) * bottom_pct))
                
                # For short-only: select HIGHEST signal (most shortable)
                ranked = signals.sort_values(ascending=False)
                new_positions = set(ranked.head(n_short).index)
                
                if current_positions is not None:
                    exited = len(current_positions - new_positions)
                    entered = len(new_positions - current_positions)
                    turnover = (exited + entered) / 2
                else:
                    turnover = n_short
                
                current_positions = new_positions
                last_rebal_idx = i
                
                tc = turnover / n_short * tc_bps / 10000 if n_short > 0 else 0
            else:
                turnover = 0
                tc = 0
            
            # Short return = -underlying return
            if current_positions and return_col in day_df.columns:
                available = [t for t in current_positions if t in day_df.index]
                if available:
                    underlying_return = day_df.loc[available, return_col].mean()
                    port_return = -underlying_return  # SHORT profit
                else:
                    port_return = 0
            else:
                port_return = 0
            
            net_return = port_return - tc
            
            results.append({
                'date': date,
                'gross_return': port_return,
                'tc': tc,
                'net_return': net_return,
                'turnover': turnover,
                'n_positions': len(current_positions) if current_positions else 0
            })
            
            if current_positions:
                for ticker in current_positions:
                    positions_list.append({
                        'date': date,
                        'ticker': ticker,
                        'side': 'short',
                        'weight': -1.0 / len(current_positions)
                    })
        
        results_df = pd.DataFrame(results)
        results_df['cumulative'] = (1 + results_df['net_return']).cumprod()
        
        positions_df = pd.DataFrame(positions_list) if positions_list else pd.DataFrame()
        
        metrics = self._calculate_metrics(results_df)
        
        return BacktestResults(
            daily_returns=results_df,
            metrics=metrics,
            positions=positions_df,
            trades=pd.DataFrame()
        )
    
    def run_long_short(
        self,
        signal_col: Optional[str] = None,
        top_pct: Optional[float] = None,
        bottom_pct: Optional[float] = None,
        rebalance_freq: Optional[int] = None
    ) -> BacktestResults:
        """
        Run long-short backtest.
        
        TIMING:
        - Day T: Rank stocks by signal
        - Day T: Long top N%, Short bottom N%
        - Day T+1: Return = 0.5 * long_return - 0.5 * short_return
        """
        signal_col = signal_col or self.config.signal_col
        top_pct = top_pct or self.config.top_pct
        bottom_pct = bottom_pct or self.config.bottom_pct
        rebalance_freq = rebalance_freq or self.config.rebalance_freq_days
        return_col = self.config.return_col
        tc_bps = self.config.transaction_cost_bps
        
        df = self.df.copy()
        dates = df['date'].unique()
        
        results = []
        positions_list = []
        current_longs = None
        current_shorts = None
        last_rebal_idx = -rebalance_freq
        
        for i, date in enumerate(dates):
            day_df = df[df['date'] == date].set_index('ticker')
            
            # Rebalance?
            if i - last_rebal_idx >= rebalance_freq or current_longs is None:
                signals = day_df[signal_col].dropna()
                n_long = max(1, int(len(signals) * top_pct))
                n_short = max(1, int(len(signals) * bottom_pct))
                
                ranked = signals.sort_values(ascending=False)
                new_longs = set(ranked.head(n_long).index)
                new_shorts = set(ranked.tail(n_short).index)
                
                if current_longs is not None:
                    longs_changed = len(new_longs - current_longs) + len(current_longs - new_longs)
                    shorts_changed = len(new_shorts - current_shorts) + len(current_shorts - new_shorts)
                    turnover = (longs_changed + shorts_changed) / 2
                else:
                    turnover = n_long + n_short
                
                current_longs = new_longs
                current_shorts = new_shorts
                last_rebal_idx = i
                
                tc = turnover / (n_long + n_short) * tc_bps / 10000 if (n_long + n_short) > 0 else 0
            else:
                turnover = 0
                tc = 0
            
            # Calculate return
            if current_longs and current_shorts and return_col in day_df.columns:
                avail_longs = [t for t in current_longs if t in day_df.index]
                avail_shorts = [t for t in current_shorts if t in day_df.index]
                
                if avail_longs and avail_shorts:
                    long_ret = day_df.loc[avail_longs, return_col].mean()
                    short_ret = day_df.loc[avail_shorts, return_col].mean()
                    port_return = 0.5 * long_ret - 0.5 * short_ret  # Market neutral
                else:
                    port_return = 0
            else:
                port_return = 0
            
            net_return = port_return - tc
            
            results.append({
                'date': date,
                'gross_return': port_return,
                'tc': tc,
                'net_return': net_return,
                'turnover': turnover,
                'n_longs': len(current_longs) if current_longs else 0,
                'n_shorts': len(current_shorts) if current_shorts else 0
            })
            
            # Track positions
            if current_longs:
                for ticker in current_longs:
                    positions_list.append({
                        'date': date, 'ticker': ticker, 'side': 'long',
                        'weight': 0.5 / len(current_longs)
                    })
            if current_shorts:
                for ticker in current_shorts:
                    positions_list.append({
                        'date': date, 'ticker': ticker, 'side': 'short',
                        'weight': -0.5 / len(current_shorts)
                    })
        
        results_df = pd.DataFrame(results)
        results_df['cumulative'] = (1 + results_df['net_return']).cumprod()
        
        positions_df = pd.DataFrame(positions_list) if positions_list else pd.DataFrame()
        
        metrics = self._calculate_metrics(results_df)
        
        return BacktestResults(
            daily_returns=results_df,
            metrics=metrics,
            positions=positions_df,
            trades=pd.DataFrame()
        )
    
    def print_metrics(self, results: BacktestResults, title: str = "Backtest Results"):
        """Print formatted metrics."""
        m = results.metrics
        
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        print(f"  Total Return:    {m['total_return']*100:>8.1f}%")
        print(f"  Annual Return:   {m['ann_return']*100:>8.1f}%")
        print(f"  Annual Vol:      {m['ann_vol']*100:>8.1f}%")
        print(f"  Sharpe Ratio:    {m['sharpe']:>8.2f}")
        print(f"  Sortino Ratio:   {m['sortino']:>8.2f}")
        print(f"  Calmar Ratio:    {m['calmar']:>8.2f}")
        print(f"  Max Drawdown:    {m['max_dd']*100:>8.1f}%")
        print(f"  Win Rate:        {m['win_rate']*100:>8.1f}%")
        print(f"  Annual Turnover: {m['ann_turnover']:>8.0f}")
        print(f"  Trading Days:    {m['n_days']:>8d}")
        print(f"{'='*60}\n")
    
    def compare_strategies(
        self,
        strategies: Dict[str, BacktestResults]
    ) -> pd.DataFrame:
        """Compare multiple strategies."""
        comparison = []
        
        for name, results in strategies.items():
            m = results.metrics
            comparison.append({
                'Strategy': name,
                'Return': f"{m['ann_return']*100:.1f}%",
                'Vol': f"{m['ann_vol']*100:.1f}%",
                'Sharpe': f"{m['sharpe']:.2f}",
                'MaxDD': f"{m['max_dd']*100:.1f}%",
                'Turnover': f"{m['ann_turnover']:.0f}"
            })
        
        return pd.DataFrame(comparison)


def validate_no_lookahead(df: pd.DataFrame, signal_col: str, return_col: str) -> bool:
    """
    Validate that there's no look-ahead bias in the signal.
    
    Checks:
    1. Signal at T should not correlate perfectly with return at T
    2. Signal should be computed before return is realized
    """
    # Check correlation between signal[T] and return[T]
    # If correlation is very high (>0.9), likely look-ahead bias
    
    corr = df.groupby('date').apply(
        lambda g: spearmanr(g[signal_col], g[return_col])[0] 
        if len(g) > 5 else np.nan
    ).mean()
    
    if abs(corr) > 0.5:
        print(f"⚠️ WARNING: High correlation ({corr:.2f}) between signal and same-day return")
        print("   This suggests possible look-ahead bias!")
        return False
    
    print(f"✅ No obvious look-ahead bias detected (corr: {corr:.4f})")
    return True
