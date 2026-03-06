import os
import pickle
import warnings
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred

warnings.filterwarnings("ignore")

DAYS_BACK            = 500
PROB_THRESHOLD       = 0.50
CONVICTION_THRESHOLD = 0.8
PRED_Z_LOOKBACK      = 252
FRED_API_KEY         = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES         = ['DFII10', 'DFII5', 'DGS2', 'FEDFUNDS']

BASE_FEATURES = [
    'Macro_Fast', 'Market_State', 'Close_XAUUSD', 'LogReturn_ZScore',
    'Close_Returns', 'Return_ZScore', 'Close_USDJPY', 'Log_Returns',
    'BB_Middle', 'Return_Percentile', 'MACD_Signal',
    'Distance_From_AllTimeHigh', 'Close_EURUSD', 'Pct_From_AllTimeHigh',
    'Volume_Percentile'
]

CALIB_FEATURES = ['oof_prediction', 'pred_z', 'abs_pred_z', 'Macro_Fast', 'Market_State']


def fetch_data(start, end):
    print("Fetching Yahoo Finance data...")

    TICKER_MAP = {
        'GC=F':     'Close_XAUUSD',
        'EURUSD=X': 'Close_EURUSD',
        'JPY=X':    'Close_USDJPY',
        '^TNX':     'TNX_Raw',
    }
    raw = yf.download(list(TICKER_MAP.keys()), start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw['Close'].rename(columns=TICKER_MAP)
        volume   = raw['Volume']['GC=F'].rename('Volume')
    else:
        close_df = raw[['Close']].rename(columns={'Close': 'Close_XAUUSD'})
        volume   = raw['Volume'].rename('Volume')

    print("Fetching FRED macro data...")
    fred  = Fred(api_key=FRED_API_KEY)
    macro = pd.DataFrame(
        {s: fred.get_series(s, start, end) for s in MACRO_SERIES}
    )
    macro.index = pd.to_datetime(macro.index)

    df = close_df.join(macro, how='left').join(volume, how='left').sort_index()
    df = df.ffill().dropna(subset=['Close_XAUUSD'])  # FRED is sparse; left-join then ffill
    print(f"Raw data: {len(df)} rows  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def engineer_features(df):
    print("Engineering features...")

    z_cols = []
    for col in MACRO_SERIES:
        df[f"{col}_delta"] = df[col].diff()
        for feat in [col, f"{col}_delta"]:
            roll = df[feat].shift(1).rolling(252)
            df[f"{feat}_z"] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(f"{feat}_z")
    df['Macro_Fast'] = df[z_cols].mean(axis=1)

    df['Close_Returns'] = df['Close_XAUUSD'].pct_change()
    df['Log_Returns']   = np.log(df['Close_XAUUSD'] / df['Close_XAUUSD'].shift(1))

    ema50   = df['Close_XAUUSD'].ewm(span=50,  adjust=False).mean()
    ema200  = df['Close_XAUUSD'].ewm(span=200, adjust=False).mean()
    mkt_vol = df['Close_Returns'].shift(1).rolling(20).std()  # causal: t-20..t-1 per spec
    vol_med = mkt_vol.expanding().median()
    df['Market_State'] = 0
    df.loc[(ema50 > ema200) & (mkt_vol < vol_med), 'Market_State'] = 1
    df.loc[(ema50 < ema200) & (mkt_vol > vol_med), 'Market_State'] = -1

    r20 = df['Log_Returns'].rolling(20)
    df['LogReturn_ZScore'] = (df['Log_Returns'] - r20.mean()) / r20.std()

    c20 = df['Close_Returns'].rolling(20)
    df['Return_ZScore'] = (df['Close_Returns'] - c20.mean()) / c20.std()

    df['BB_Middle'] = df['Close_XAUUSD'].rolling(20).mean()

    df['Return_Percentile'] = df['Close_Returns'].rolling(100).rank(pct=True)
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill()   # guard: GC=F vol delayed 24h
    df['Volume_Percentile'] = df['Volume'].rolling(100).rank(pct=True)

    macd = (df['Close_XAUUSD'].ewm(span=12, adjust=False).mean()
          - df['Close_XAUUSD'].ewm(span=26, adjust=False).mean())
    df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()

    ath = df['Close_XAUUSD'].expanding().max()
    df['Distance_From_AllTimeHigh'] = ath - df['Close_XAUUSD']
    df['Pct_From_AllTimeHigh']      = (df['Distance_From_AllTimeHigh'] / ath) * 100

    clean = df[BASE_FEATURES].dropna()
    print(f"Feature matrix ready: {len(clean)} rows")
    return clean


def load_artefacts(base_dir):
    print("Loading model artefacts...")

    def load(name):
        with open(os.path.join(base_dir, name), 'rb') as f:
            return pickle.load(f)

    base_model  = load("cv_best_fold_model.pkl")
    calibrator  = load("calibrator.pkl")
    oof_history = pd.read_csv(
        os.path.join(base_dir, "cv_predictions_oof.csv"),
        index_col=0, parse_dates=True
    )
    print(f"OOF history: {len(oof_history)} rows")
    return base_model, calibrator, oof_history


def run_inference(feature_df, base_model, calibrator, oof_history):
    print("Running inference...")

    today      = feature_df.iloc[[-1]].copy()
    today_date = today.index[0].date()

    oof_pred = float(base_model.predict(today[BASE_FEATURES].values)[0])

    pred_col   = next((c for c in oof_history.columns if 'pred' in c.lower()), oof_history.columns[0])
    hist       = oof_history[pred_col].dropna().tail(PRED_Z_LOOKBACK)
    pred_z     = float((oof_pred - hist.mean()) / hist.std()) if hist.std() != 0 else 0.0
    abs_pred_z = abs(pred_z)

    macro_fast   = float(today['Macro_Fast'].iloc[0])
    market_state = int(today['Market_State'].iloc[0])
    market_state_str = str(market_state)  # OHE trained on string categories '-1','0','1'

    calib_input = pd.DataFrame(
        [[oof_pred, pred_z, abs_pred_z, macro_fast, market_state_str]],
        columns=CALIB_FEATURES
    )
    calib_input['Market_State'] = calib_input['Market_State'].astype(str)
    prob = float(calibrator.predict_proba(calib_input)[0][1])

    signal = "NO SIGNAL"
    if prob >= PROB_THRESHOLD and abs_pred_z >= CONVICTION_THRESHOLD:
        signal = "BUY" if oof_pred > 0 else "SELL"

    return {
        "date":           str(today_date),
        "signal":         signal,
        "oof_prediction": round(oof_pred, 6),
        "pred_z":         round(pred_z, 4),
        "abs_pred_z":     round(abs_pred_z, 4),
        "prob_success":   round(prob, 4),
        "macro_fast":     round(macro_fast, 4),
        "market_state":   market_state,
        "Close_XAUUSD":   round(float(today['Close_XAUUSD'].iloc[0]), 2),
    }


def main(artefact_dir="."):
    print(f"GOLD XAUUSD INFERENCE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    end   = datetime.now()
    start = end - timedelta(days=DAYS_BACK)

    df       = fetch_data(start, end)
    features = engineer_features(df)
    base_model, calibrator, oof_history = load_artefacts(artefact_dir)
    result   = run_inference(features, base_model, calibrator, oof_history)

    label = 'Bull' if result['market_state'] == 1 else 'Bear' if result['market_state'] == -1 else 'Neutral'
    print()
    print(f"DATE         : {result['date']}")
    print(f"XAUUSD CLOSE : {result['Close_XAUUSD']:.2f}")
    print(f"MACRO FAST   : {result['macro_fast']:+.4f}")
    print(f"MARKET STATE : {result['market_state']} ({label})")
    print(f"BASE PRED    : {result['oof_prediction']:+.6f}")
    print(f"PRED Z-SCORE : {result['pred_z']:+.4f}")
    print(f"PROB SUCCESS : {result['prob_success']:.4f}")
    print(f"SIGNAL       : {result['signal']}")
    if result['signal'] == 'NO SIGNAL':
        if result['abs_pred_z'] < CONVICTION_THRESHOLD:
            print(f"  reason: |pred_z| {result['abs_pred_z']:.2f} < {CONVICTION_THRESHOLD}")
        if result['prob_success'] < PROB_THRESHOLD:
            print(f"  reason: prob {result['prob_success']:.4f} < {PROB_THRESHOLD}")
    print()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".",
                        help="Directory containing pkl and csv artefacts")
    args = parser.parse_args()
    main(artefact_dir=args.dir)