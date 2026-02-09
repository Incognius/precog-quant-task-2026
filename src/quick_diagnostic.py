"""Quick diagnostic to check feature-return correlations"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load processed data
df = pd.read_parquet('data/processed/df_after_eda.parquet')

# Standardize columns
df = df.rename(columns={
    'Date': 'date', 
    'asset_id': 'ticker',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker', 'date'])

# Create simple forward returns
df['fwd_ret_1d'] = df.groupby('ticker')['close'].pct_change(1).shift(-1)
df['fwd_ret_5d'] = df.groupby('ticker')['close'].pct_change(5).shift(-5)

print('Feature-Return Correlations')
print('='*60)
print(f'{"Feature":<25} {"1-day IC":>15} {"5-day IC":>15}')
print('-'*60)

# Test momentum at different horizons
for days in [1, 5, 10, 20, 60, 120]:
    df[f'mom_{days}d'] = df.groupby('ticker')['close'].pct_change(days)
    df['norm_feat'] = df.groupby('date')[f'mom_{days}d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    ic_1d = df['norm_feat'].corr(df['fwd_ret_1d'])
    ic_5d = df['norm_feat'].corr(df['fwd_ret_5d'])
    print(f'{"Momentum " + str(days) + "d":<25} {ic_1d:>15.4f} {ic_5d:>15.4f}')

# Test reversal (negative momentum)
for days in [1, 5, 10]:
    df[f'rev_{days}d'] = -df.groupby('ticker')['close'].pct_change(days)
    df['norm_feat'] = df.groupby('date')[f'rev_{days}d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    ic_1d = df['norm_feat'].corr(df['fwd_ret_1d'])
    ic_5d = df['norm_feat'].corr(df['fwd_ret_5d'])
    print(f'{"Reversal " + str(days) + "d":<25} {ic_1d:>15.4f} {ic_5d:>15.4f}')

# Test volatility
for days in [10, 20, 60]:
    df[f'vol_{days}d'] = df.groupby('ticker')['close'].pct_change().rolling(days).std()
    df['norm_feat'] = df.groupby('date')[f'vol_{days}d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    ic_1d = df['norm_feat'].corr(df['fwd_ret_1d'])
    ic_5d = df['norm_feat'].corr(df['fwd_ret_5d'])
    print(f'{"Volatility " + str(days) + "d":<25} {ic_1d:>15.4f} {ic_5d:>15.4f}')

# Test volume features
df['volume_ratio'] = df['volume'] / df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
df['norm_feat'] = df.groupby('date')['volume_ratio'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)
ic_1d = df['norm_feat'].corr(df['fwd_ret_1d'])
ic_5d = df['norm_feat'].corr(df['fwd_ret_5d'])
print(f'{"Volume ratio (vs 20d MA)":<25} {ic_1d:>15.4f} {ic_5d:>15.4f}')

# Test RSI-like feature
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

df['rsi_14'] = df.groupby('ticker')['close'].transform(lambda x: calc_rsi(x, 14))
df['norm_feat'] = df.groupby('date')['rsi_14'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-10)
)
ic_1d = df['norm_feat'].corr(df['fwd_ret_1d'])
ic_5d = df['norm_feat'].corr(df['fwd_ret_5d'])
print(f'{"RSI 14":<25} {ic_1d:>15.4f} {ic_5d:>15.4f}')

# Test mean reversion (distance from MA)
for days in [20, 60, 120]:
    df[f'dist_ma_{days}'] = df['close'] / df.groupby('ticker')['close'].transform(lambda x: x.rolling(days).mean()) - 1
    df['norm_feat'] = df.groupby('date')[f'dist_ma_{days}'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    ic_1d = df['norm_feat'].corr(df['fwd_ret_1d'])
    ic_5d = df['norm_feat'].corr(df['fwd_ret_5d'])
    print(f'{"Dist from MA " + str(days) + "d":<25} {ic_1d:>15.4f} {ic_5d:>15.4f}')

print('='*60)
print("\nConclusion: Look for ICs significantly different from 0")
print("           IC > 0.02 is weak but usable")
print("           IC > 0.05 is good")
print("           IC > 0.10 is excellent")
