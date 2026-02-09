# Quant Pipeline - Modular Architecture

## Overview

This is a complete redesign of the quant research workflow into **clean, modular, testable notebooks** where each stage is independently verifiable.

## Design Philosophy

> *"A correct pipeline with mediocre performance is infinitely more valuable than an impressive backtest that cannot be reproduced."*

### Core Principles

1. **Separation of Concerns** - Each notebook has ONE responsibility
2. **No Duplication** - Shared code lives in importable modules
3. **Explicit Checkpoints** - Every stage has failure tests
4. **Delayed Backtest** - No backtest until final stage
5. **Delayed Sharpe** - No Sharpe until strategy layer

---

## Package Structure

```
quant_pipeline/
├── __init__.py              # Global seed, package init
├── data/
│   └── __init__.py          # DataConfig, DataLoader, DataSplitter
├── features/
│   └── __init__.py          # FeaturePanelBuilder, feature generators
├── targets/
│   └── __init__.py          # TargetConfig, target construction
├── models/
│   └── __init__.py          # ModelTrainer, ModelConfig
├── signals/
│   └── __init__.py          # Signal scaling, turnover, exposure
├── strategy/
│   └── __init__.py          # Position sizing, backtesting, metrics
└── diagnostics/
    └── __init__.py          # DiagnosticSuite, test functions
```

---

## Notebook Flow

```
NB1 (Features) → NB2 (Targets) → NB3 (Model) → NB4 (Signals) → NB5 (Backtest)
                                                                    ↑
                                              HV (Hypothesis Validation)
```

### Hypothesis Validation Notebook
**File**: `notebooks/HV_hypothesis_validation.ipynb`

**Purpose**: Validate methodology choices with rigorous statistical testing.

**Tests**:
1. **H1: Kalman vs Raw** - Is HMM on Kalman features superior?
2. **H2: Label Permutation** - Does model capture real alpha?
3. **H3: Time Permutation** - Does timing skill exist?
4. **H4: CS Permutation** - Does ranking skill exist?

---

### Notebook 1: Data & Feature Engineering
**File**: `notebooks/NB1_data_features.ipynb`

**Allowed**:
- ✅ Load raw data
- ✅ Data cleaning & validation
- ✅ Feature engineering (momentum, volatility, Kalman, etc.)
- ✅ Feature diagnostics

**Forbidden**:
- ❌ ML models
- ❌ Targets/labels
- ❌ Backtests
- ❌ Sharpe

**Outputs**:
- `features_is.parquet`
- `features_oos.parquet`
- `feature_metadata.csv`

---

### Notebook 2: Target Construction
**File**: `notebooks/NB2_targets.ipynb`

**Allowed**:
- ✅ Target construction (forward returns)
- ✅ Label sanity checks
- ✅ IC decay analysis
- ✅ Noise floor estimation

**Forbidden**:
- ❌ Model training
- ❌ Portfolio logic
- ❌ Backtests

**Outputs**:
- `targets_is.parquet`
- `target_config.json`
- `noise_floor.txt`

---

### Notebook 3: Model Training
**File**: `notebooks/NB3_model_training.ipynb`

**Allowed**:
- ✅ Model training (LightGBM, XGBoost, Ridge)
- ✅ Cross-validation (temporal)
- ✅ Information Coefficient analysis
- ✅ Feature importance

**Forbidden**:
- ❌ Position sizing
- ❌ Full portfolio backtest
- ❌ Transaction costs
- ❌ Sharpe optimization

**Outputs**:
- `predictions_is.parquet`
- `trained_models/*.pkl`
- `ic_stats.json`

---

### Notebook 4: Signal Interpretation
**File**: `notebooks/NB4_signals.ipynb`

**Allowed**:
- ✅ Signal scaling (rank, z-score)
- ✅ Turnover analysis
- ✅ Concentration analysis
- ✅ Exposure analysis

**Forbidden**:
- ❌ Full portfolio backtest
- ❌ Sharpe optimization

**Outputs**:
- `signals_is.parquet`
- `signal_config.json`
- `signal_stats.json`

---

### Notebook 5: Strategy & Backtest
**File**: `notebooks/NB5_strategy_backtest.ipynb`

**THIS IS THE ONLY NOTEBOOK WHERE SHARPE IS COMPUTED**

**Allowed**:
- ✅ Position sizing
- ✅ Risk constraints
- ✅ Full backtest simulation
- ✅ Transaction costs
- ✅ Performance metrics (Sharpe, Sortino, etc.)
- ✅ Benchmark comparison
- ✅ OOS validation

**Outputs**:
- `strategy_config.json`
- `is_results.json`
- `oos_results.json`
- `benchmark_comparison.json`

---

## Critical Rules

### 1. Checkpoint Pattern
```python
# At the end of each stage:
if not diagnostics.passed():
    raise RuntimeError("Diagnostics failed - do not proceed")
```

### 2. No Future Leakage
- Features use only past data
- Targets use strictly forward data
- No overlap in train/val/test

### 3. Reproducibility
```python
from quant_pipeline import set_global_seed
set_global_seed(42)  # Called at start of every notebook
```

### 4. OOS is Sacred
Once OOS is revealed:
- **NO changes to the pipeline**
- Document failures, don't fix them
- This prevents overfitting to test set

---

## Module Documentation

### Data Module
- `DataConfig`: Configuration dataclass for data loading
- `DataLoader`: Loads raw asset files
- `DataSplitter`: Time-based train/val/test splits

### Features Module
- `FeaturePanelBuilder`: Constructs feature panel
- `compute_momentum_features()`: Returns, rolling stats
- `compute_volatility_features()`: Parkinson, GARCH-like
- `compute_kalman_features()`: Kalman filter smoothing
- `compute_regime_features()`: Fast volatility-based regime detection
- `fit_hmm_gaussian()`: Pure Python Gaussian HMM (Baum-Welch)
- `compute_hmm_features()`: HMM regime features (optional, slower)
- `compute_regime_interaction_features()`: Conditional alpha features

#### Regime Features (Fast - Default)

**Core Regime Features:**
| Feature | Formula | Description |
|---------|---------|-------------|
| `regime_confidence` | $\max_k P(\text{state} = k)$ | Regime certainty (softmax) |
| `regime_entropy` | $-\sum P(k) \log P(k)$ | High = transition uncertainty |
| `regime_p_high_vol` | $P(\text{high vol state})$ | Stress probability |
| `regime_duration` | consecutive days | Regime persistence |
| `regime_transition_rate` | 21d rolling changes/days | Transition frequency |

**Interaction Features (where alpha usually lives):**
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `mom_x_regime_conf` | mom × confidence | Penalize unstable regimes |
| `reversal_x_regime_shock` | reversal × P(shock) | Reversion after shocks |
| `kalman_x_regime_lowvol` | slope × P(low vol) | Kalman reliable in calm markets |

#### HMM Features (Optional - Slower)

Enable with `FeaturePanelBuilder(include_hmm=True)`:

| Feature | Formula | Description |
|---------|---------|-------------|
| `hmm_confidence` | $\max_k P(z_t = k \| y_{1:t})$ | Regime certainty |
| `hmm_entropy` | $-\sum P(z_t=k) \log P(z_t=k)$ | High = transition |
| `hmm_p_high_vol` | $P(\text{high vol state})$ | Stress probability |
| `hmm_expected_duration` | $1/(1-a_{kk})$ | Regime persistence |

### Targets Module
- `TargetConfig`: Configuration for target construction
- `construct_return_target()`: Forward N-day returns
- `construct_rank_target()`: Cross-sectional ranks
- `compute_ic_decay()`: IC vs horizon analysis

### Models Module
- `ModelConfig`: Configuration for model training
- `ModelTrainer`: Trains and validates models
- `compute_daily_ic()`: Information coefficient
- `analyze_residuals()`: Residual diagnostics

### Signals Module
- `SignalConfig`: Configuration for signal scaling
- `scale_signal_zscore()`: Rolling z-score
- `scale_signal_rank()`: Cross-sectional rank
- `compute_signal_turnover()`: Daily turnover

### Strategy Module
- `StrategyConfig`: Position sizing, risk constraints
- `compute_positions()`: Convert signals to positions
- `run_backtest()`: Vectorized backtester
- `BacktestResult`: Performance metrics container

### Diagnostics Module
- `DiagnosticSuite`: Collection of tests
- `DiagnosticResult`: Single test result
- `TestResult`: PASS / WARN / FAIL enum

---

## Running the Pipeline

```bash
# 1. Configure environment
cd "c:\Users\ponna\OneDrive\Desktop\Precog Task"
pip install -r requirements.txt

# 2. Run notebooks in order
# Open each notebook and run all cells
# DO NOT skip notebooks
# DO NOT run out of order

# Order:
# 1. NB1_data_features.ipynb
# 2. NB2_targets.ipynb
# 3. NB3_model_training.ipynb
# 4. NB4_signals.ipynb
# 5. NB5_strategy_backtest.ipynb
```

---

## Random Seed

All randomness is controlled via:

```python
from quant_pipeline import set_global_seed
set_global_seed(42)
```

This sets seeds for:
- `numpy`
- `random`
- Environment variable `PYTHONHASHSEED`

---

## Files Generated

### Data Files (data/processed/)
- `features_is.parquet` - In-sample features
- `features_oos.parquet` - Out-of-sample features
- `targets_is.parquet` - In-sample targets
- `predictions_is.parquet` - Model predictions
- `signals_is.parquet` - Scaled signals

### Config Files (data/processed/)
- `target_config.json`
- `ic_stats.json`
- `signal_config.json`
- `signal_stats.json`

### Backtest Results (outputs/backtest/)
- `strategy_config.json`
- `is_results.json`
- `oos_results.json`
- `benchmark_comparison.json`
- `is_daily_returns.csv`

### Figures (outputs/figures/)
- `eda/` - Exploratory data analysis
- `features/` - Feature diagnostics
- `model/` - Model diagnostics
- `backtest/` - Performance plots

---

## Author Notes

This architecture was designed after struggling with:
1. Results changing between runs
2. Bugs that were impossible to localize
3. LLM-generated all-in-one notebooks

The solution: **Modularity with explicit boundaries**.

Every decision is documented. Every stage can be verified independently.

*"The problem is rarely the model. It's usually the data."*
