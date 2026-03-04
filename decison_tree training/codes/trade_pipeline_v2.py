import pandas as pd
import numpy as np
import os

# ==========================================
# 1. HYPER-PARAMETERS (Optimized)
# ==========================================
Z_THRESH       = 1.0      # Hunter Frequency
VOL_P_LIMIT    = 30        # Hunter Volatility       # Be more selective about volatility (was 30)
Z_WINDOW       = 60        
VOL_WINDOW     = 20        
XM_SPREAD_COST = 0.000175  
RISK_PER_TRADE = 100.0

# ==========================================
# 2. DATA LOAD & ALIGNMENT
# ==========================================
df_oof = pd.read_csv('cv_predictions_oof.csv')
df_feat = pd.read_csv('xauusd_train_pruned.csv')

df_oof['Date'] = pd.to_datetime(df_oof['Date'])
df_feat['Date'] = pd.to_datetime(df_feat['Date'])

# Merge to get predictions and market data in one place
df = pd.merge(df_oof[df_oof['has_prediction'] == True], 
              df_feat[['Date', 'Close_XAUUSD']], on='Date', how='left')

# ==========================================
# 3. SIGNAL PROCESSING (The Fixes)
# ==========================================
# [Fix 1] Zero-Crossing/Rolling Baseline
df['pred_mean'] = df['oof_prediction'].rolling(Z_WINDOW).mean()
df['pred_std']  = df['oof_prediction'].rolling(Z_WINDOW).std()
df['pred_z']    = (df['oof_prediction'] - df['pred_mean']) / df['pred_std']

# [Fix 2] Volatility Filter (Avoid Flat Markets)
df['mkt_vol']   = df['y_next_log_return'].rolling(VOL_WINDOW).std()
vol_threshold   = np.nanpercentile(df['mkt_vol'], VOL_P_LIMIT)

# ==========================================
# 4. VOLATILITY-ADJUSTED POSITION SIZING
# ==========================================
# Logic: Wider Stop Loss in high vol = Smaller Position Size
# We use Market Volatility as a proxy for ATR
df['stop_loss_dist'] = df['mkt_vol'] * 2.0  # 2-Sigma Stop
df['pos_size_units'] = RISK_PER_TRADE / df['stop_loss_dist']

# ==========================================
# 5. EXECUTION LOGIC
# ==========================================
df['signal'] = 0
# Condition: High Conviction AND Sufficient Volatility
mask_long  = (df['pred_z'] > Z_THRESH)  & (df['mkt_vol'] > vol_threshold)
mask_short = (df['pred_z'] < -Z_THRESH) & (df['mkt_vol'] > vol_threshold)

df.loc[mask_long,  'signal'] = 1
df.loc[mask_short, 'signal'] = -1

# [Fix 3] Friday Exit Logic
df['day_of_week'] = df['Date'].dt.dayofweek # 4 = Friday
# Force exit/no-trade on Fridays to avoid Sunday Gap
df.loc[df['day_of_week'] == 4, 'signal'] = 0

# ==========================================
# 6. PERFORMANCE CALCULATION
# ==========================================
# Realistic Return = (Direction * Market Return) - Spread Cost
df['raw_ret'] = df['signal'] * df['y_next_log_return']
df['net_ret'] = np.where(df['signal'] != 0, df['raw_ret'] - XM_SPREAD_COST, 0)

# Metrics
trades_df = df[df['signal'] != 0].copy()
win_rate = (trades_df['net_ret'] > 0).mean()
avg_profit = trades_df['net_ret'].mean()

print(f"--- PIPELINE V2 RESULTS ---")
print(f"Total Trades: {len(trades_df)}")
print(f"Directional Acc: {(trades_df['raw_ret'] > 0).mean()*100:.2f}%")
print(f"Real Win Rate:   {win_rate*100:.2f}% (After XM Spread)")
print(f"Avg Profit/Trade: {avg_profit:.6f} log units")
print(f"Final Prediction Z: {df['pred_z'].iloc[-1]:.4f}")

# Save the hunter signals for your 4H entry check
df[df['signal'] != 0].to_csv('risk_signals.csv', index=False)