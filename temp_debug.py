import pandas as pd
import numpy as np

targets = pd.read_parquet('data/processed/stage1_5_targets.parquet')

print('=== TARGET_SMOOTH ANALYSIS ===')
sample = targets[targets.ticker == targets.ticker.unique()[0]].copy()
sample = sample.sort_values('date')

# Correlations
print('Correlations with fwd_ret_1d:')
corr_same = sample['target_smooth'].corr(sample['fwd_ret_1d'])
print(f'  target_smooth (same day): {corr_same:.4f}')

sample['target_smooth_lag1'] = sample['target_smooth'].shift(1)
corr_lag1 = sample['target_smooth_lag1'].corr(sample['fwd_ret_1d'])
print(f'  target_smooth_lag1 (yesterday): {corr_lag1:.4f}')

# Lead-lag analysis
print()
print('Lead-lag analysis (single asset):')
for lag in range(-5, 6):
    shifted = sample['target_smooth'].shift(lag)
    corr = shifted.corr(sample['fwd_ret_1d'])
    s = '+' if lag >= 0 else ''
    print(f'  target_smooth(t{s}{lag}) vs fwd_ret_1d(t): {corr:.4f}')

# Cross-sectional analysis
print()
print('=== CROSS-SECTIONAL ANALYSIS ===')
merged = targets[['date', 'ticker', 'fwd_ret_1d', 'target_smooth']].copy()

# Get predictive IC
merged['fwd_ret_1d_next'] = merged.groupby('ticker')['fwd_ret_1d'].shift(-1)

def get_rank_ic(grp):
    return grp['target_smooth'].corr(grp['fwd_ret_1d'], method='spearman')

ic_same = merged.groupby('date').apply(get_rank_ic).mean()
print(f'Rank IC (target_smooth vs same-day fwd_ret_1d): {ic_same:.4f}')

def get_pred_ic(grp):
    return grp['target_smooth'].corr(grp['fwd_ret_1d_next'], method='spearman')

ic_pred = merged.dropna().groupby('date').apply(get_pred_ic).mean()
print(f'PREDICTIVE IC (target_smooth vs NEXT-day fwd_ret_1d): {ic_pred:.4f}')

# Now check the relationship with the prediction
print()
print('=== CHECKING MODEL PREDICTIONS ===')
pred = pd.read_parquet('data/processed/stage2_predictions.parquet')

# Merge
pred_tgt = pred.merge(
    targets[['date', 'ticker', 'target_smooth']], 
    on=['date', 'ticker'], 
    how='left'
)

# Signal predicts target_smooth well (we know this)
print(f'Signal vs target_smooth: {pred_tgt["signal_zscore"].corr(pred_tgt["target_smooth"]):.4f}')

# But does target_smooth predict next-day returns?
pred_tgt['fwd_ret_1d_next'] = pred_tgt.groupby('ticker')['fwd_ret_1d'].shift(-1)
print(f'target_smooth vs NEXT fwd_ret_1d: {pred_tgt["target_smooth"].corr(pred_tgt["fwd_ret_1d_next"]):.4f}')

# THE KEY: target_smooth includes TODAY's fwd_ret_1d in the EMA
# So it correlates with today's return, but NOT with tomorrow's
print()
print('=== THE PROBLEM ===')
print('target_smooth = EMA(fwd_ret_1d) looking BACKWARD')
print('It includes today fwd_ret_1d in the weighted average')
print('So predicting target_smooth well means predicting PAST trend, not FUTURE')
