# Systematic Model Improvement Plan

## Status: ✅ IMPLEMENTED

## Goal
Improve the Net Sharpe Ratio of the quantitative trading strategy. Previous experiments (Notebook 05) showed high Gross Sharpe (0.68) but negative Net Sharpe due to excessive turnover (441x). We aim to capture the alpha while reducing costs.

## Completed Changes

### 1. Feature Engineering Library (`src/features.py`) ✅
Created comprehensive feature engineering module with:
- `compute_trend_block()`: Trend strength, consistency, breadth, acceleration, divergence
- `compute_regime_indicators()`: Vol regime, trend regime, combined regime, confidence
- `compute_advanced_momentum_block()`: Multi-horizon risk-adjusted momentum
- `compute_volatility_regime_block()`: Vol term structure, surprise, clustering, asymmetry
- `compute_interaction_features()`: Momentum x Regime interactions
- `create_regime_mask()`: Pre-built regime filters for strategy conditioning
- GPU-accelerated features (PyTorch)

### 2. Enhanced Pipeline (`src/pipeline.py`) ✅
Updated to v3.0 with:

**Turnover Control Functions:**
- `smooth_signals(signals, halflife, method)`: EMA/SMA/decay smoothing
- `smooth_weights(weights, decay)`: Position-level smoothing with normalization

**Regime Filtering:**
- `apply_regime_filter(weights, regime_mask)`: Apply regime-based filtering
- `create_regime_mask(returns, regime_type)`: Create masks for high/low vol, trend, etc.
- `construct_portfolio_weights_with_regime()`: Integrated portfolio construction

**Advanced ML Models (GPU):**
- `run_rolling_lightgbm()`: LightGBM with GPU support
- `run_rolling_xgboost()`: XGBoost with CUDA acceleration
- `run_rolling_mlp()`: PyTorch neural network

**Pipeline Class Methods:**
- `run_with_smoothing()`: Run with signal/weight smoothing + any model type
- `run_with_regime()`: Run with regime filtering + smoothing + any model

### 3. Strategy Optimization Notebook (`notebooks/06_strategy_optimization.ipynb`) ✅
Created comprehensive experiment notebook with:
- **Experiment 1**: Signal Smoothing Grid Search (halflife 2-10, decay 0.5-0.9)
- **Experiment 2**: Regime Filtering (high_vol, low_vol, uptrend, downtrend, trending, confident)
- **Experiment 3**: Advanced ML Models (Ridge, LightGBM, XGBoost, MLP) with GPU
- **Experiment 4**: Combined optimal configuration
- Full visualizations and leaderboard tracking

## Success Criteria
- **Net Sharpe > 0.5** (currently negative)
- **Turnover < 100x** (currently ~400x)

## How to Run

1. **Configure Python environment** (if not done):
   ```python
   # Install dependencies
   pip install lightgbm xgboost torch
   ```

2. **Run the optimization notebook**:
   - Open `notebooks/06_strategy_optimization.ipynb`
   - Run all cells sequentially
   - The notebook will run experiments and save results to `outputs/`

3. **Check results**:
   - Leaderboard saved to `outputs/results/optimization_leaderboard.csv`
   - Figures saved to `outputs/figures/`
   - Experiment log saved to `outputs/optimization_experiment_log.json`

## Key Files Modified/Created

| File | Status | Description |
|------|--------|-------------|
| `src/features.py` | NEW | Feature engineering library |
| `src/pipeline.py` | MODIFIED | v3.0 with smoothing, regime, ML |
| `notebooks/06_strategy_optimization.ipynb` | NEW | Experiment notebook |

## Technical Details

### Signal Smoothing Parameters
```python
# Recommended configurations:
halflife = 5  # days (moderate smoothing)
halflife = 10  # days (heavy smoothing)
weight_decay = 0.7 - 0.9  # position persistence
```

### Regime Filter Options
- `high_vol_only`: Trade when vol > median (momentum often works better)
- `low_vol_only`: Trade when vol < median (mean reversion)
- `uptrend_only`: Trade when 60d market return > 0
- `downtrend_only`: Trade when 60d market return < 0
- `trending`: Trade when trend is strong (either direction)
- `confident`: Trade when regime is clear (not transitioning)

### Model Options
- `ridge`: Fast, interpretable, baseline
- `lightgbm`: Non-linear, captures interactions, GPU-accelerated
- `xgboost`: Strong regularization, GPU-accelerated
- `mlp`: Neural network, PyTorch, GPU-accelerated
