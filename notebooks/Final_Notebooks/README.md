# Final Pipeline - Quantitative Trading System

## Overview

This folder contains the final, modular pipeline for the Precog Quant Task 2026. Each notebook operates on the output of the previous stage, ensuring clean separation of concerns and easy debugging.

## Pipeline Flow

```
Raw Data (CSVs)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Feature Engineering                                 │
│ Input:  data/raw/assets/*.csv                               │
│ Output: data/processed/stage1_features.parquet              │
│         outputs/stage1_feature_metadata.json                │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1.5: Target Creation                                   │
│ Input:  data/processed/stage1_features.parquet              │
│ Output: data/processed/stage1_5_targets.parquet             │
│         outputs/stage1_5_target_metadata.json               │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Model Training                                      │
│ Input:  data/processed/stage1_5_targets.parquet             │
│ Output: data/processed/stage2_predictions.parquet           │
│         outputs/stage2_model_diagnostics.json               │
│         outputs/models/final_model/                         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Backtesting                                         │
│ Input:  data/processed/stage2_predictions.parquet           │
│ Output: outputs/results/stage3_backtest_results.parquet     │
│         outputs/stage3_strategy_config.json                 │
│         outputs/figures/final/*.png                         │
└─────────────────────────────────────────────────────────────┘
```

## Notebooks

### `Stage1_Feature_Engineering.ipynb`
**Purpose**: Transform raw price data into predictive features

**Features Generated**:
- **Momentum**: 5d, 10d, 21d, 63d, 126d, 252d returns
- **Volatility**: Rolling std, ATR, Vol-of-Vol
- **Regime**: Vol regime (low/normal/high), Trend regime
- **Kalman**: Filtered price, velocity, deviation
- **Technical**: RSI, MACD, Bollinger Bands, Volume ratios

**Key Outputs**:
- Cross-sectionally z-scored features
- IC analysis for each feature

---

### `Stage1_5_Target_Creation.ipynb`
**Purpose**: Create multiple target types for model training

**Targets Created**:
- **Raw**: 1d, 5d, 21d forward returns
- **Smoothed**: EMA-smoothed (reduces noise)
- **Ternary**: Up/Neutral/Down classification
- **Probability**: Continuous [0,1] confidence scores
- **Risk-Adjusted**: Sharpe-like forward targets

**Key Analysis**:
- Turnover comparison (smooth targets = lower turnover)
- Feature-target IC comparison

---

### `Stage2_Model_Training.ipynb`
**Purpose**: Train and evaluate predictive models

**Models**:
- Ridge Regression
- Lasso Regression
- Elastic Net
- Ensemble (IC-weighted combination)

**Training Method**:
- Walk-forward validation
- 252-day initial training window
- 21-day retrain frequency
- 5-day embargo between train/test
- Exponential decay weighting (63-day half-life)

**Key Outputs**:
- IS/OOS IC comparison
- Regime-based performance analysis
- Best model selection

---

### `Stage3_Backtesting.ipynb`
**Purpose**: Backtest trading strategies and optimize parameters

**Strategies**:
- Long-only (signal-weighted)
- Long-short (market neutral)
- Top-N equal-weight

**Risk Management**:
- Transaction costs (10 bps)
- Stop-loss (configurable)
- Volatility targeting (20% annualized)
- Position limits (10% max per asset)

**Grid Search**:
- Rebalance frequency
- Stop-loss thresholds
- Vol targets

**Final Output**:
- IS vs OOS comparison
- Alpha vs EW benchmark
- Performance visualizations

---

## Configuration

All pipeline parameters are centralized in `pipeline_config.py`:

```python
# Data split
IS_START = '2016-01-01'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'

# Model training
INITIAL_TRAIN_DAYS = 252
RETRAIN_FREQUENCY = 21
EMBARGO_DAYS = 5
DECAY_HALFLIFE = 63

# Backtesting
TRANSACTION_COST_BPS = 10
STOP_LOSS_PCT = 0.03
VOL_TARGET = 0.20
```

---

## How to Run

1. **Ensure raw data exists**: `data/raw/assets/Asset_*.csv`

2. **Run notebooks in order**:
   ```
   Stage1_Feature_Engineering.ipynb
   Stage1_5_Target_Creation.ipynb
   Stage2_Model_Training.ipynb
   Stage3_Backtesting.ipynb
   ```

3. **Check outputs** in `data/processed/` and `outputs/`

---

## Key Design Decisions

1. **Modular Pipeline**: Each stage reads parquet from previous stage, making debugging easy
2. **IS/OOS Split**: 2016-2023 for training/validation, 2024-2026 for true out-of-sample
3. **No Data Leakage**: Embargo period prevents look-ahead bias
4. **Cross-sectional Normalization**: Features z-scored within each day for fair ranking
5. **Walk-Forward Validation**: Mimics real trading with periodic retraining

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

---

## Author

Precog Quant Task 2026 Submission
