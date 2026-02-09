# =============================================================================
# QUANT PIPELINE - Modular, Auditable Research Framework
# =============================================================================
# 
# "A correct pipeline with mediocre performance is infinitely more valuable 
#  than an impressive backtest that cannot be reproduced."
#
# Structure:
#   quant_pipeline/
#   ├── data/         - Data loading, validation, splitting
#   ├── features/     - Feature construction (NO targets, NO models)
#   ├── targets/      - Target construction (NO features, NO models)
#   ├── models/       - Model training (NO backtests, NO Sharpe)
#   ├── signals/      - Signal interpretation (NO portfolio optimization)
#   ├── strategy/     - Strategy construction (FINAL backtest here only)
#   ├── diagnostics/  - Testing & validation utilities
#   └── utils/        - Shared utilities
#
# =============================================================================

__version__ = "0.1.0"
__author__ = "Quant Research Pipeline"

# Enforce reproducibility
import numpy as np
import random
import os

def set_global_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"[SEED] Global random seed set to {seed}")

# Set seed on import
set_global_seed(42)
