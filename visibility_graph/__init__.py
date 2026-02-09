"""
Visibility Graph Strategy Module

This module implements Visibility Graph (VG) and Horizontal Visibility Graph (HVG)
based trading strategies for financial time series analysis.

Components:
- vg_core: Core VG/HVG construction algorithms
- vg_features: Feature extraction from graphs
- vg_gnn: Graph Neural Network models
- vg_strategy: Trading signal generation
"""

from .vg_core import (
    construct_visibility_graph,
    construct_horizontal_visibility_graph,
    construct_vg_fast,
    construct_hvg_fast,
    construct_weighted_vg,
    construct_directed_vg,
    adjacency_to_edge_index,
    validate_vg_properties
)
from .vg_features import (
    extract_graph_features,
    compute_rolling_vg_features,
    features_to_dataframe,
    compute_degree_stats,
    fit_power_law,
    compute_clustering_coefficient,
    count_triads
)
from .vg_strategy import (
    VGStrategyConfig,
    SignalMethod,
    generate_rolling_signals,
    generate_signals_universe,
    analyze_signal_quality,
    detect_market_regime,
    rolling_regime_detection
)

__all__ = [
    # Core
    'construct_visibility_graph',
    'construct_horizontal_visibility_graph',
    'construct_vg_fast',
    'construct_hvg_fast',
    'construct_weighted_vg',
    'construct_directed_vg',
    'adjacency_to_edge_index',
    'validate_vg_properties',
    # Features
    'extract_graph_features',
    'compute_rolling_vg_features',
    'features_to_dataframe',
    'compute_degree_stats',
    'fit_power_law',
    'compute_clustering_coefficient',
    'count_triads',
    # Strategy
    'VGStrategyConfig',
    'SignalMethod',
    'generate_rolling_signals',
    'generate_signals_universe',
    'analyze_signal_quality',
    'detect_market_regime',
    'rolling_regime_detection'
]

