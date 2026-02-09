# =============================================================================
# DIAGNOSTICS MODULE - Testing & Validation Utilities
# =============================================================================
# 
# RESPONSIBILITY: Provide testing and validation utilities for all stages
# 
# This module defines HOW each stage can fail and HOW we detect that failure.
#
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class TestResult(Enum):
    """Result of a diagnostic test."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    test_name: str
    result: TestResult
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        symbol = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[self.result.value]
        return f"{symbol} {self.test_name}: {self.message}"


class DiagnosticSuite:
    """Collection of diagnostic results."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[DiagnosticResult] = []
    
    def add(self, result: DiagnosticResult):
        self.results.append(result)
    
    def passed(self) -> bool:
        return all(r.result != TestResult.FAIL for r in self.results)
    
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.result == TestResult.PASS)
    
    def n_warn(self) -> int:
        return sum(1 for r in self.results if r.result == TestResult.WARN)
    
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.result == TestResult.FAIL)
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"DIAGNOSTIC SUITE: {self.name}",
            f"{'='*60}",
        ]
        
        for r in self.results:
            lines.append(str(r))
        
        lines.append(f"\nSummary: {self.n_pass()} passed, {self.n_warn()} warnings, {self.n_fail()} failed")
        lines.append(f"Overall: {'PASS' if self.passed() else 'FAIL'}")
        lines.append("="*60)
        
        return "\n".join(lines)
    
    def print_summary(self):
        print(self.summary())


# =============================================================================
# DATA DIAGNOSTICS
# =============================================================================

def diagnose_data_quality(
    prices: pd.DataFrame,
    returns: pd.DataFrame
) -> DiagnosticSuite:
    """Run data quality diagnostics."""
    suite = DiagnosticSuite("Data Quality")
    
    # Test 1: Missing values
    missing_pct = prices.isna().sum().sum() / prices.size * 100
    if missing_pct > 5:
        suite.add(DiagnosticResult(
            "Missing Values",
            TestResult.FAIL,
            f"Too many missing values: {missing_pct:.2f}%",
            {'missing_pct': missing_pct}
        ))
    elif missing_pct > 1:
        suite.add(DiagnosticResult(
            "Missing Values",
            TestResult.WARN,
            f"Some missing values: {missing_pct:.2f}%",
            {'missing_pct': missing_pct}
        ))
    else:
        suite.add(DiagnosticResult(
            "Missing Values",
            TestResult.PASS,
            f"Missing values acceptable: {missing_pct:.2f}%"
        ))
    
    # Test 2: Extreme returns
    extreme_pct = (returns.abs() > 0.50).sum().sum() / returns.size * 100
    if extreme_pct > 0.1:
        suite.add(DiagnosticResult(
            "Extreme Returns",
            TestResult.WARN,
            f"High percentage of extreme returns (>50%): {extreme_pct:.4f}%",
            {'extreme_pct': extreme_pct}
        ))
    else:
        suite.add(DiagnosticResult(
            "Extreme Returns",
            TestResult.PASS,
            f"Extreme returns acceptable: {extreme_pct:.4f}%"
        ))
    
    # Test 3: Zero returns
    zero_pct = (returns == 0).sum().sum() / returns.size * 100
    if zero_pct > 20:
        suite.add(DiagnosticResult(
            "Zero Returns",
            TestResult.WARN,
            f"High percentage of zero returns: {zero_pct:.2f}%",
            {'zero_pct': zero_pct}
        ))
    else:
        suite.add(DiagnosticResult(
            "Zero Returns",
            TestResult.PASS,
            f"Zero returns acceptable: {zero_pct:.2f}%"
        ))
    
    # Test 4: Data sorted
    is_sorted = prices.index.is_monotonic_increasing
    if not is_sorted:
        suite.add(DiagnosticResult(
            "Data Sorted",
            TestResult.FAIL,
            "Data is NOT sorted by date"
        ))
    else:
        suite.add(DiagnosticResult(
            "Data Sorted",
            TestResult.PASS,
            "Data is sorted by date"
        ))
    
    return suite


# =============================================================================
# FEATURE DIAGNOSTICS
# =============================================================================

def diagnose_features(
    panel: pd.DataFrame,
    feature_cols: List[str],
    n_periods: int = 4
) -> DiagnosticSuite:
    """Run feature diagnostics."""
    suite = DiagnosticSuite("Feature Quality")
    
    # Test 1: Missing values per feature
    for col in feature_cols:
        missing_pct = panel[col].isna().mean() * 100
        if missing_pct > 20:
            suite.add(DiagnosticResult(
                f"Missing: {col}",
                TestResult.FAIL,
                f"Too many missing: {missing_pct:.1f}%"
            ))
        elif missing_pct > 10:
            suite.add(DiagnosticResult(
                f"Missing: {col}",
                TestResult.WARN,
                f"Some missing: {missing_pct:.1f}%"
            ))
    
    # Test 2: Constant features
    for col in feature_cols:
        std = panel[col].std()
        if std < 1e-10:
            suite.add(DiagnosticResult(
                f"Constant: {col}",
                TestResult.FAIL,
                f"Feature is constant (std={std:.2e})"
            ))
    
    # Test 3: Inf values
    for col in feature_cols:
        inf_pct = np.isinf(panel[col]).mean() * 100
        if inf_pct > 0:
            suite.add(DiagnosticResult(
                f"Infinite: {col}",
                TestResult.FAIL,
                f"Feature has inf values: {inf_pct:.2f}%"
            ))
    
    # Test 4: Feature stability
    panel_copy = panel.copy()
    panel_copy['period'] = pd.qcut(
        panel_copy['date'].rank(method='dense'), 
        n_periods, 
        labels=False
    )
    
    for col in feature_cols:
        means = panel_copy.groupby('period')[col].mean()
        mean_range = means.max() - means.min()
        overall_std = panel_copy[col].std()
        
        if overall_std > 0:
            instability = mean_range / overall_std
            if instability > 2.0:
                suite.add(DiagnosticResult(
                    f"Unstable: {col}",
                    TestResult.WARN,
                    f"Feature mean shifts significantly across time (ratio={instability:.2f})"
                ))
    
    # Test 5: High correlation (redundancy)
    corr_matrix = panel[feature_cols].corr()
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols):
            if i < j:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.98:
                    suite.add(DiagnosticResult(
                        f"Redundant: {col1} vs {col2}",
                        TestResult.WARN,
                        f"Features highly correlated: {corr:.3f}"
                    ))
    
    # Summary
    if suite.n_fail() == 0 and suite.n_warn() == 0:
        suite.add(DiagnosticResult(
            "Overall Feature Quality",
            TestResult.PASS,
            f"All {len(feature_cols)} features passed basic checks"
        ))
    
    return suite


def diagnose_lookahead_bias(
    panel: pd.DataFrame,
    feature_cols: List[str],
    future_return_col: str
) -> DiagnosticSuite:
    """
    Diagnose potential lookahead bias.
    
    If a feature is highly correlated with future returns at time t,
    but the feature is computed at time t, there may be lookahead bias.
    """
    suite = DiagnosticSuite("Lookahead Bias Check")
    
    suspicious = []
    
    for col in feature_cols:
        # Correlation with future return
        corr = panel[col].corr(panel[future_return_col])
        
        if abs(corr) > 0.5:
            suspicious.append((col, corr))
            suite.add(DiagnosticResult(
                f"Suspicious: {col}",
                TestResult.WARN,
                f"High correlation with future return: {corr:.3f}"
            ))
    
    if len(suspicious) == 0:
        suite.add(DiagnosticResult(
            "Lookahead Bias",
            TestResult.PASS,
            "No features have suspiciously high correlation with future returns"
        ))
    
    return suite


# =============================================================================
# TARGET DIAGNOSTICS
# =============================================================================

def diagnose_target(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> DiagnosticSuite:
    """Diagnose target quality."""
    suite = DiagnosticSuite("Target Quality")
    
    target = panel[target_col]
    
    # Test 1: Target distribution
    skew = target.skew()
    kurt = target.kurtosis()
    
    if abs(skew) > 3:
        suite.add(DiagnosticResult(
            "Target Skewness",
            TestResult.WARN,
            f"Target highly skewed: {skew:.2f}"
        ))
    else:
        suite.add(DiagnosticResult(
            "Target Skewness",
            TestResult.PASS,
            f"Target skewness acceptable: {skew:.2f}"
        ))
    
    # Test 2: Target noise
    target_std = target.std()
    target_mean = target.mean()
    snr = abs(target_mean) / target_std if target_std > 0 else 0
    
    if snr < 0.01:
        suite.add(DiagnosticResult(
            "Target SNR",
            TestResult.WARN,
            f"Target has very low signal-to-noise: {snr:.4f}"
        ))
    else:
        suite.add(DiagnosticResult(
            "Target SNR",
            TestResult.PASS,
            f"Target SNR: {snr:.4f}"
        ))
    
    # Test 3: Target autocorrelation (trivial predictability)
    target_series = target.dropna()
    if len(target_series) > 10:
        autocorr = target_series.autocorr(lag=1)
        if autocorr > 0.5:
            suite.add(DiagnosticResult(
                "Target Autocorrelation",
                TestResult.WARN,
                f"Target highly autocorrelated (lag=1): {autocorr:.3f} - may indicate trivial predictability"
            ))
        else:
            suite.add(DiagnosticResult(
                "Target Autocorrelation",
                TestResult.PASS,
                f"Target autocorrelation acceptable: {autocorr:.3f}"
            ))
    
    return suite


# =============================================================================
# MODEL DIAGNOSTICS
# =============================================================================

def diagnose_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None
) -> DiagnosticSuite:
    """Diagnose model prediction quality (NO Sharpe, NO backtest)."""
    suite = DiagnosticSuite("Model Predictions")
    
    # Test 1: Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if corr < 0.01:
        suite.add(DiagnosticResult(
            "Prediction Correlation",
            TestResult.FAIL,
            f"Prediction correlation near zero: {corr:.4f}"
        ))
    elif corr < 0.05:
        suite.add(DiagnosticResult(
            "Prediction Correlation",
            TestResult.WARN,
            f"Prediction correlation low: {corr:.4f}"
        ))
    else:
        suite.add(DiagnosticResult(
            "Prediction Correlation",
            TestResult.PASS,
            f"Prediction correlation: {corr:.4f}"
        ))
    
    # Test 2: Prediction distribution
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    
    if pred_std < true_std * 0.1:
        suite.add(DiagnosticResult(
            "Prediction Variance",
            TestResult.WARN,
            f"Predictions have much lower variance than target ({pred_std:.4f} vs {true_std:.4f})"
        ))
    else:
        suite.add(DiagnosticResult(
            "Prediction Variance",
            TestResult.PASS,
            f"Prediction variance: {pred_std:.4f} (target: {true_std:.4f})"
        ))
    
    # Test 3: Residual autocorrelation
    residuals = y_true - y_pred
    if len(residuals) > 10:
        residual_ac = pd.Series(residuals).autocorr(lag=1)
        if abs(residual_ac) > 0.3:
            suite.add(DiagnosticResult(
                "Residual Autocorrelation",
                TestResult.WARN,
                f"Residuals are autocorrelated: {residual_ac:.3f}"
            ))
        else:
            suite.add(DiagnosticResult(
                "Residual Autocorrelation",
                TestResult.PASS,
                f"Residual autocorrelation acceptable: {residual_ac:.3f}"
            ))
    
    # Test 4: IC (Information Coefficient) by time period
    if dates is not None:
        unique_dates = np.unique(dates)
        daily_ics = []
        for d in unique_dates:
            mask = dates == d
            if mask.sum() > 5:
                ic = np.corrcoef(y_true[mask], y_pred[mask])[0, 1]
                if not np.isnan(ic):
                    daily_ics.append(ic)
        
        if len(daily_ics) > 10:
            ic_mean = np.mean(daily_ics)
            ic_std = np.std(daily_ics)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            
            suite.add(DiagnosticResult(
                "Information Ratio",
                TestResult.PASS if icir > 0.05 else TestResult.WARN,
                f"IC: {ic_mean:.4f}, IR: {icir:.4f}"
            ))
    
    return suite


def diagnose_overfitting(
    train_corr: float,
    val_corr: float,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None
) -> DiagnosticSuite:
    """Diagnose potential overfitting."""
    suite = DiagnosticSuite("Overfitting Check")
    
    # Test 1: Train vs Val correlation gap
    gap = train_corr - val_corr
    if gap > 0.10:
        suite.add(DiagnosticResult(
            "Correlation Gap",
            TestResult.FAIL,
            f"Large train-val gap: {train_corr:.4f} vs {val_corr:.4f} (gap={gap:.4f})"
        ))
    elif gap > 0.05:
        suite.add(DiagnosticResult(
            "Correlation Gap",
            TestResult.WARN,
            f"Moderate train-val gap: {train_corr:.4f} vs {val_corr:.4f} (gap={gap:.4f})"
        ))
    else:
        suite.add(DiagnosticResult(
            "Correlation Gap",
            TestResult.PASS,
            f"Train-val gap acceptable: {train_corr:.4f} vs {val_corr:.4f}"
        ))
    
    # Test 2: Val performance
    if val_corr < 0.02:
        suite.add(DiagnosticResult(
            "Validation Performance",
            TestResult.FAIL,
            f"Validation correlation too low: {val_corr:.4f}"
        ))
    elif val_corr < 0.05:
        suite.add(DiagnosticResult(
            "Validation Performance",
            TestResult.WARN,
            f"Validation correlation marginal: {val_corr:.4f}"
        ))
    else:
        suite.add(DiagnosticResult(
            "Validation Performance",
            TestResult.PASS,
            f"Validation correlation acceptable: {val_corr:.4f}"
        ))
    
    return suite
