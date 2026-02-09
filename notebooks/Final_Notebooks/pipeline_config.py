"""
================================================================================
PIPELINE CONFIGURATION
================================================================================
Central configuration for the entire quantitative trading pipeline.
All notebooks import from this file for consistency.

Author: Precog Quant Task 2026
Last Updated: 2025-02-07
================================================================================
"""

from pathlib import Path
import pandas as pd

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent  # Precog Task folder
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "assets"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures" / "final"

# Create directories if they don't exist
for d in [PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================
# IS Period: 2016-01-01 to 2023-12-31 (Training)
# OOS Period: 2024-01-01 to 2026-12-31 (True Out-of-Sample)
IS_START = pd.Timestamp('2016-01-01')
IS_END = pd.Timestamp('2023-12-31')
OOS_START = pd.Timestamp('2024-01-01')
OOS_END = pd.Timestamp('2026-12-31')

# Feature warm-up period (for computing rolling features)
WARMUP_DAYS = 252  # 1 year

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
# Momentum lookbacks
MOMENTUM_WINDOWS = [5, 10, 21, 63, 126, 252]

# Volatility lookbacks
VOLATILITY_WINDOWS = [5, 21, 63]

# Vol-of-Vol lookbacks
VOL_OF_VOL_WINDOWS = [21, 63]

# Moving average windows for mean reversion
MA_WINDOWS = [21, 63, 126]

# Kalman filter parameters
KALMAN_OBSERVATION_COV = 1.0
KALMAN_TRANSITION_COV = 0.01

# ============================================================================
# TARGET CONFIGURATION
# ============================================================================
# Forward return horizons
TARGET_HORIZONS = [1, 5, 21]  # days

# Ternary target thresholds (percentile-based)
TERNARY_UP_THRESHOLD = 0.6   # top 40% = UP
TERNARY_DOWN_THRESHOLD = 0.4  # bottom 40% = DOWN

# Smoothing parameters
TARGET_SMOOTH_WINDOW = 5  # days for exponential smoothing

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Walk-forward parameters
INITIAL_TRAIN_DAYS = 252  # 1 year initial training
RETRAIN_FREQUENCY = 21    # days between retrains
EMBARGO_DAYS = 5          # gap between train and test
EXPANDING_WINDOW = True   # expanding vs rolling window
DECAY_HALFLIFE = 63       # observation weight half-life

# Model hyperparameters
RIDGE_ALPHA = 100.0
LASSO_ALPHA = 0.01
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 100
}

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================
INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST_BPS = 10  # 10 basis points = 0.10%
MAX_POSITION_SIZE = 0.10   # 10% max per asset
REBALANCE_FREQUENCY = 21   # days

# Stop-loss parameters
STOP_LOSS_PCT = 0.03       # 3% stop-loss
TRAILING_STOP_PCT = 0.03   # 3% trailing stop

# Volatility targeting
VOL_TARGET = 0.20          # 20% annualized
MAX_LEVERAGE = 3.0

# ============================================================================
# OUTPUT FILE NAMES (for inter-notebook communication)
# ============================================================================
# Stage 1 outputs
FEATURES_PARQUET = PROCESSED_DATA_DIR / "stage1_features.parquet"
FEATURE_METADATA_JSON = OUTPUT_DIR / "stage1_feature_metadata.json"

# Stage 1.5 outputs
TARGETS_PARQUET = PROCESSED_DATA_DIR / "stage1_5_targets.parquet"
TARGET_METADATA_JSON = OUTPUT_DIR / "stage1_5_target_metadata.json"

# Stage 2 outputs
PREDICTIONS_PARQUET = PROCESSED_DATA_DIR / "stage2_predictions.parquet"
MODEL_WEIGHTS_DIR = MODELS_DIR / "final_model"
MODEL_DIAGNOSTICS_JSON = OUTPUT_DIR / "stage2_model_diagnostics.json"

# Stage 3 outputs
BACKTEST_RESULTS_PARQUET = RESULTS_DIR / "stage3_backtest_results.parquet"
STRATEGY_CONFIG_JSON = OUTPUT_DIR / "stage3_strategy_config.json"
FINAL_REPORT_DIR = BASE_DIR / "reports" / "final"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_raw_data():
    """Load all raw asset CSVs into a single DataFrame."""
    import pandas as pd
    from pathlib import Path
    
    all_data = []
    for f in sorted(RAW_DATA_DIR.glob("Asset_*.csv")):
        df = pd.read_csv(f, parse_dates=['Date'])
        df['ticker'] = f.stem  # Asset_001, etc.
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['ticker', 'Date']).reset_index(drop=True)
    combined = combined.rename(columns={'Date': 'date'})
    return combined

def get_is_oos_mask(df, date_col='date'):
    """Return boolean masks for IS and OOS periods."""
    is_mask = (df[date_col] >= IS_START) & (df[date_col] <= IS_END)
    oos_mask = (df[date_col] >= OOS_START) & (df[date_col] <= OOS_END)
    return is_mask, oos_mask

def print_stage_header(stage_name, stage_num):
    """Print a formatted stage header."""
    print("=" * 80)
    print(f"STAGE {stage_num}: {stage_name.upper()}")
    print("=" * 80)
    print()
