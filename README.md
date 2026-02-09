# Precog Quant Task 2026 - Algorithmic Trading Pipeline

**Author:** Ponnambalam V
**Task:** Quantitative Trading Strategy Development  

---

## Project Summary

This repository implements an end-to-end algorithmic trading pipeline for a universe of **100 anonymized stocks** spanning 2016-2026. The final strategy achieves:

| Metric | In-Sample (2016-2023) | Out-of-Sample (2024-2026) |
|--------|----------------------|---------------------------|
| **Sharpe Ratio** | 3.47 | **2.10** |
| **Annual Return** | 28.5% | 31.5% |
| **Max Drawdown** | -12.8% | -11.2% |
| **Annual Volatility** | 15.2% | 15.0% |
| **vs Equal-Weight** | +124% improvement | - |

---

## Task Completion Status

| Part | Task | Status |
|------|------|--------|
| 1 | Feature Engineering & Data Cleaning | Complete |
| 2 | Model Training & Strategy Formulation | Complete |
| 3 | Backtesting & Performance Analysis | Complete |
| 4 | Statistical Arbitrage Overlay | Complete |

---

## Final Strategy Summary

### Model: LightGBM Ensemble with Volatility Targeting

**Best Model Configuration:**
```
Model:              LightGBM (Gradient Boosting)
n_estimators:       100
max_depth:          4
num_leaves:         15
learning_rate:      0.05
min_child_samples:  50
subsample:          0.8
colsample_bytree:   0.8
reg_alpha:          0.1
reg_lambda:         0.1
```

**Ensemble Weights:**
- LightGBM: 50%
- Ridge Regression: 30%
- Random Forest: 20%

### Information Coefficients (IC)

| Model | IC Mean | IC Std | IC IR | Hit Rate |
|-------|---------|--------|-------|----------|
| **LightGBM** | **0.032** | 0.06 | **0.52** | 71% |
| Ridge | 0.021 | 0.05 | 0.42 | 78% |
| **Ensemble** | **0.035** | 0.05 | **0.70** | - |

### Strategy Configuration

| Parameter | Value |
|-----------|-------|
| Strategy Type | Long-Short Equal Weight |
| Long Leg | Top 10% by signal (20 assets) |
| Short Leg | Bottom 10% by signal (20 assets) |
| Rebalancing | Weekly (every 5 trading days) |
| Transaction Costs | 10 bps per trade |
| Volatility Target | 15% annualized |
| Max Leverage | 2.0x |
| Position Sizing | Equal weight (1/20 per leg) |

### Feature Set (38 Features)

| Family | Count | Features |
|--------|-------|----------|
| **Momentum** | 8 | mom_5d, mom_10d, mom_21d, mom_63d, mom_accel, mom_reversal, mom_zscore, mom_consistency |
| **Volatility** | 6 | vol_5d, vol_10d, vol_21d, vol_ratio, vol_zscore, vol_regime |
| **Mean Reversion** | 5 | ma_20_dev, ma_50_dev, bb_position, rsi_14, rsi_21 |
| **Kalman Filter** | 5 | kalman_trend, kalman_slope, kalman_curvature, kalman_residual, kalman_zscore |
| **Regime** | 6 | regime_state, regime_conf, regime_entropy, regime_duration, regime_transition, regime_prob_high |
| **Cross-Sectional** | 4 | cs_rank_ret, cs_rank_vol, cs_rank_mom, cs_zscore_ret |
| **Interaction** | 4 | mom_x_regime, vol_x_regime, kalman_x_conf, reversal_x_shock |

**Top Features by Importance:**
1. kalman_slope (18.2%) - Trend direction
2. mom_reversal (12.5%) - Short-term reversal
3. vol_ratio (11.8%) - Vol regime change
4. regime_conf (9.2%) - Confidence in state
5. cs_rank_ret (8.5%) - Relative performance

---

## Directory Structure

```
Precog Task/
|-- data/
|   |-- raw/assets/            # Raw OHLCV CSVs (100 assets) - NOT COMMITTED
|   +-- processed/             # Intermediate parquet files - NOT COMMITTED
|
|-- notebooks/
|   |-- Final_Notebooks/       # FINAL PIPELINE (run these)
|   |   |-- Stage1_Feature_Engineering.ipynb
|   |   |-- Stage1_5_Target_Creation.ipynb
|   |   |-- Stage2_v2_Model_Training.ipynb
|   |   |-- Stage3_v2_Backtesting.ipynb
|   |   |-- Stage4_Final.ipynb
|   |   |-- StatArb.ipynb
|   |   +-- pipeline_config.py
|   |
|   |-- NB1-NB6*.ipynb         # Earlier iteration experiments
|   |-- 01-08*.ipynb           # Initial exploration notebooks
|   +-- Research_*.ipynb       # Research experiments
|
|-- outputs/
|   |-- figures/               # Generated visualizations
|   |   |-- stage4/           # Final strategy figures
|   |   |-- final/            # Performance plots
|   |   |-- eda/              # Exploratory analysis
|   |   +-- hmm/              # Regime detection
|   |-- models/                # Saved model artifacts
|   +-- results/               # Backtest results and metrics
|
|-- reports/                   # PDF reports (methodology and results)
|   |-- Final_Strategy_Comprehensive.pdf  # Main strategy report
|   |-- Iteration_3_Comprehensive.pdf     # NB pipeline documentation
|   +-- [other technical reports]
|
|-- src/                       # Utility modules
|   |-- data/                  # Data processing utilities
|   |-- models/                # Model architectures
|   |-- backtesting/           # Backtest engine
|   +-- utils/                 # Visualization helpers
|
+-- requirements.txt           # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Incognius/precog-quant-task-2026.git
cd precog-quant-task-2026
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Visit: https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026
   - Download all CSV files
   - Place them in `data/raw/assets/` directory
   - Ensure files are named `Asset_001.csv` through `Asset_100.csv`

### Running the Final Strategy Pipeline

Execute notebooks in `notebooks/Final_Notebooks/` in sequence:

```bash
cd notebooks/Final_Notebooks

# 1. Feature Engineering (~5 min)
jupyter notebook Stage1_Feature_Engineering.ipynb

# 2. Target Creation (~2 min)
jupyter notebook Stage1_5_Target_Creation.ipynb

# 3. Model Training (~10 min)
jupyter notebook Stage2_v2_Model_Training.ipynb

# 4. Backtesting (~5 min)
jupyter notebook Stage3_v2_Backtesting.ipynb

# 5. Final Strategy with Vol Targeting (~5 min)
jupyter notebook Stage4_Final.ipynb

# 6. Statistical Arbitrage Analysis (optional)
jupyter notebook StatArb.ipynb
```

---

## Pipeline Configuration

All parameters are in `notebooks/Final_Notebooks/pipeline_config.py`:

```python
# DATA SPLIT
IS_START = '2016-01-01'     # In-sample start
IS_END = '2023-12-31'       # In-sample end  
OOS_START = '2024-01-01'    # Out-of-sample start

# MODEL TRAINING
INITIAL_TRAIN_DAYS = 252    # 1 year initial window
RETRAIN_FREQUENCY = 21      # Monthly retraining
EMBARGO_DAYS = 5            # Gap to prevent leakage
DECAY_HALFLIFE = 63         # Sample weight decay

# BACKTESTING
TRANSACTION_COST_BPS = 10   # 10 basis points per trade
VOL_TARGET = 0.15           # 15% annualized volatility
MAX_LEVERAGE = 2.0          # Maximum leverage cap
REBALANCE_DAYS = 5          # Weekly rebalancing

# STRATEGY
TOP_PCT = 0.10              # Long top 10%
BOT_PCT = 0.10              # Short bottom 10%
```

---

## Other Experiments

### Exploration Notebooks (notebooks/)

| Notebook | Description |
|----------|-------------|
| 01-08*.ipynb | Initial exploration and baseline strategies |
| NB1-NB6*.ipynb | Second iteration with walk-forward validation |
| Research_*.ipynb | Visibility graph and alternative approaches |
| Short_Only_Strategy.ipynb | Short-only strategy analysis |
| Model_Stress_Test.ipynb | Regime-conditional stress testing |

### Research Directories

| Directory | Purpose |
|-----------|---------|
| forward-bias/ | Analysis of look-ahead bias detection |
| research/ | Academic literature implementations |
| visibility_graph/ | Time series to graph transformations |
| quant_pipeline/ | Alternative pipeline architecture |

---

## Reports

All reports are in `reports/` as PDFs:

| Report | Description |
|--------|-------------|
| Final_Strategy_Comprehensive.pdf | **Main report**: Complete final strategy documentation |
| Iteration_3_Comprehensive.pdf | NB1-NB6 pipeline methodology |
| research_methodology_comprehensive.pdf | Detailed research journey |
| kalman_filter_technical.pdf | Kalman filter feature derivation |
| hmm_regime_technical.pdf | HMM regime detection methodology |
| ic_analysis_technical.pdf | Information Coefficient analysis |

---

## Results Summary

### Performance vs Benchmark

| Strategy | OOS Sharpe | Annual Return | Max Drawdown |
|----------|------------|---------------|--------------|
| **LGBM + Vol15%** | **2.10** | **31.5%** | **-11.2%** |
| Equal-Weight Buy and Hold | 0.94 | 12.8% | -35.2% |
| Long-Only Top 20% | 1.45 | 18.2% | -22.4% |

### Key Findings

1. **Kalman filter features dominate** - 18% importance, smooth trend detection
2. **Short-term reversal > raw momentum** - Contrarian signals work better
3. **Volatility targeting critical** - Improves Sharpe by 19%, cuts drawdown 50%
4. **Weekly rebalancing optimal** - Balances signal freshness vs transaction costs
5. **Statistical arbitrage fails** - Negative Sharpe after 10 bps costs

---

## Dependencies

Core dependencies (see requirements.txt for full list):
```
numpy>=1.21.0
pandas>=1.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
scipy>=1.7.0
pyarrow>=6.0.0
jupyter>=1.0.0
tqdm>=4.62.0
```

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/iamspace/precog-quant-task-2026
- LightGBM: Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- Kalman Filter: Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Information Coefficient: Grinold, R. & Kahn, R. (1999). "Active Portfolio Management"

---

## Contact

For questions regarding this submission, please contact through the Precog recruitment portal.

---

*Last Updated: February 2026*
