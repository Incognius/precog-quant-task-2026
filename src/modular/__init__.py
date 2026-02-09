"""
Modular Pipeline Components
===========================

Stage 1: Feature Engineering (features_long.py, features_short.py)
Stage 2: Walk-Forward Training (walk_forward.py)
Stage 3: Backtesting (backtester.py)

Pipeline flow:
    raw_data → Stage1 → features.parquet → Stage2 → predictions.parquet → Stage3 → results.parquet
"""

from .features_long import LongFeatureEngineer
from .features_short import ShortFeatureEngineer
from .walk_forward import WalkForwardTrainer
from .backtester import Backtester

__all__ = [
    'LongFeatureEngineer',
    'ShortFeatureEngineer', 
    'WalkForwardTrainer',
    'Backtester'
]
