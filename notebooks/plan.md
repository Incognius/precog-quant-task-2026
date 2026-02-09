# Systematic Model Improvement Plan

## Status: NB5 COMPLETE ✅

### Progress Summary

| Notebook | Status | Key Achievements |
|----------|--------|------------------|
| NB1 | ✅ Complete | Data cleaning, 100 assets, 1418 days |
| NB2 | ✅ Complete | EDA, regime detection (low/normal/high vol) |
| NB3 | ✅ Complete | Baseline model, Ridge regression |
| NB4 | ✅ Complete | Signal diagnostics, regime-adaptive smoothing |
| NB5 | ✅ Complete | Multi-strategy backtest, 0.95 Sharpe achieved |

### NB5 Key Results
- **Best Strategy**: Long_Only_Weekly (Sharpe 0.95, beats EW benchmark 0.86)
- **IS/OOS**: Improved OOS (0.84 → 1.12 Sharpe) - genuine alpha!
- **Regime Performance**: Low vol (2.35), Normal (0.66), High vol (0.08)
- **Max Drawdown**: -28.9% (COVID crash, 340-day recovery)
- **Turnover**: 12x/year (reasonable)
- **Best SL/TP**: 10% stop-loss improves max DD to -18.5%

### Remaining Work
1. Part 4: Statistical Arbitrage Overlay (pairs trading, cointegration)
2. Final report compilation
3. Presentation preparation

---

## Original Goal
Improve the Net Sharpe Ratio of the quantitative trading strategy. Previous experiments (Notebook 05) showed high Gross Sharpe (0.68) but negative Net Sharpe due to excessive turnover (441x). We aim to capture the alpha while reducing costs.

## Solution Implemented
- **Fixed turnover calculation**: Actual turnover is 12-18x/year, not 441x
- **Regime-adaptive smoothing**: Different EMA halflifes per regime
- **Long-only weekly rebalance**: Reduces turnover while maintaining Sharpe
- **Vol-targeting**: 16% target with 2.0x max leverage

## Proposed Changes
1. Systematize Feature Engineering
Move the successful "Trend Regime" features from 
05_model_experiments.ipynb
 into a reusable library.

[NEW] 
src/features.py
Implement compute_trend_block:
trend_strength: Volatility-normalized cumulative return.
trend_consistency: Hit-ratio of trend direction.
trend_breadth: Cross-sectional participation.
Implement compute_regime_indicators:
vol_regime: High/Low volatility classification.
trend_regime: Up/Down trend classification.
[MODIFY] 
src/pipeline.py
Import new feature blocks.
Add support for "Regime Signal" in 
run
 and 
run_backtest
.
2. Implement Turnover Control & Regime Filtering
Add mechanisms to dampen trading frequency and avoid unfavorable market conditions.

[MODIFY] 
src/pipeline.py
Signal Smoothing: Implement smooth_signals(signals, alpha=0.1) to apply EMA smoothing to raw predictions before ranking. This reduces signal noise and turnover.
Regime Filtering: Update 
construct_portfolio_weights
 to accept a regime_mask.
If regime_mask[t] == 0 (Unfavorable), force weights to 0 (Cash).
Or scale weights by regime_confidence.
3. Strategy Optimization Experiment
Create a new notebook to run the improved strategy.

[NEW] 
notebooks/06_strategy_optimization.ipynb
Load data and baseline features.
Generate new Trend/Regime features using src.features.
Experiment 1: Signal Smoothing. Grid search smoothing parameters (e.g., halflife 2, 5, 10 days) to find optimal Turnove/Sharpe trade-off.
Experiment 2: Regime Gating. Test "High Volatility Only" and "Trend Following" filters.
Experiment 3: Combined. Best Smoothing + Best Regime.
Track results using 
BestModelTracker
.
Verification Plan
Automated Tests
We will rely on the 06_strategy_optimization.ipynb to serve as the verification suite.
It will explicitly calculate and print:
Gross Sharpe
Net Sharpe
Annual Turnover
Success Criteria:
Net Sharpe > 0.5 (currently negative).
Turnover < 100x (currently ~400x).
Manual Verification
Inspect the cumulative return plots in the notebook to ensure the strategy isn't just sitting in cash (check "Time in Market" metric).