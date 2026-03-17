import os, pickle, warnings, time, requests
import numpy as np
import pandas as pd
import streamlit as st
from fredapi import Fred
from datetime import datetime, timedelta, timezone
import yfinance as yf

try:
    from qwen_briefing import run_briefing, update_outcome, load_memory
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

warnings.filterwarnings("ignore")

ARTEFACT_DIR      = os.path.dirname(os.path.abspath(__file__))
FRED_API_KEY      = "219d0c44b2e3b4a8b690c3f69b91a5bb"
MACRO_SERIES      = ["DFII10", "DFII5", "DGS2", "FEDFUNDS"]
DAYS_BACK         = 520
PRED_Z_LOOKBACK   = 252
PROB_THRESHOLD    = 0.45
Z_THRESHOLD       = 0.6
SMC_SWING         = 5
SR_TOL            = 0.0015
NY_TZ             = timezone(timedelta(hours=-5))
MOROCCO_TZ        = timezone(timedelta(hours=1))
SETTLEMENT_NY     = 13.5
MAINTENANCE_START = 21.25
MAINTENANCE_END   = 22.0

BASE_FEATURES = [
    "Close_Returns","Log_Returns","EURUSD_Returns","USDJPY_Returns",
    "BB_PctB","Price_Over_EMA50","Price_Over_EMA200","MACD_Signal_Norm",
    "LogReturn_ZScore","Return_ZScore","Return_Percentile","Volume_Percentile",
    "Pct_From_AllTimeHigh","Bull_Trend","Macro_Fast",
]
CALIB_FEATURES = [
    "prediction_value","abs_prediction",
    "Bull_Trend","Macro_Fast","BB_PctB","Price_Over_EMA200",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
:root{
    --bg:#05070a;--surf:#0c0f14;--surf2:#111520;--border:#1c2030;
    --border2:#252a38;--text:#e2e8f0;--muted:#3d4a5c;--muted2:#5a6a80;
    --accent:#f5c842;--accent2:#e6a800;--buy:#10d988;--sell:#ff4d6a;
    --mono:'JetBrains Mono',monospace;--sans:'Space Grotesk',sans-serif;
    --glow-buy:0 0 24px rgba(16,217,136,0.18);
    --glow-sell:0 0 24px rgba(255,77,106,0.18);
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],
section.main,.main .block-container{background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
body{background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:2.4rem 1.6rem 6rem!important;max-width:960px!important;}
*{box-sizing:border-box;}
.app-header{display:flex;align-items:center;justify-content:space-between;
    padding:0 0 1.4rem;border-bottom:1px solid var(--border);margin-bottom:2.2rem;}
.app-logo{display:flex;align-items:center;gap:0.75rem;}
.logo-hex{width:34px;height:34px;background:var(--accent);
    clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    display:flex;align-items:center;justify-content:center;
    font-family:var(--mono);font-size:0.6rem;font-weight:700;color:#000;flex-shrink:0;}
.logo-text{font-family:var(--mono);font-size:0.78rem;font-weight:600;letter-spacing:0.2em;color:var(--accent);text-transform:uppercase;}
.logo-sub{font-family:var(--mono);font-size:0.56rem;color:var(--muted2);letter-spacing:0.12em;margin-top:2px;}
.header-right{text-align:right;}
.header-ts{font-family:var(--mono);font-size:0.58rem;color:var(--muted2);letter-spacing:0.08em;}
.header-status{font-family:var(--mono);font-size:0.56rem;letter-spacing:0.12em;margin-top:3px;}
.section-label{font-family:var(--mono);font-size:0.56rem;letter-spacing:0.22em;
    text-transform:uppercase;color:var(--muted2);padding:0 0 0.6rem;
    border-bottom:1px solid var(--border);margin:1.8rem 0 1rem;}
.kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
    background:var(--border);border:1px solid var(--border);border-radius:2px;overflow:hidden;margin-bottom:1px;}
.kpi-strip-2{grid-template-columns:repeat(2,1fr);}
.kpi-strip-3{grid-template-columns:repeat(3,1fr);}
.kpi-cell{background:var(--surf);padding:1.1rem 1rem;}
.kpi-lbl{font-family:var(--mono);font-size:0.52rem;color:var(--muted2);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;}
.kpi-val{font-family:var(--mono);font-size:1.1rem;font-weight:600;color:var(--text);}
.kpi-val.gold{color:var(--accent);}
.kpi-val.bull{color:var(--buy);}
.kpi-val.bear{color:var(--sell);}
.kpi-val.muted{color:var(--muted2);}
.kpi-sub{font-family:var(--mono);font-size:0.52rem;color:var(--muted);margin-top:0.25rem;}
.sig-banner{position:relative;border-radius:3px;overflow:hidden;margin:1.4rem 0;
    padding:1.8rem 2rem 1.8rem 2.4rem;display:flex;align-items:center;
    justify-content:space-between;background:var(--surf);}
.sig-banner::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;}
.sb-buy{border:1px solid rgba(16,217,136,0.3);box-shadow:var(--glow-buy);}
.sb-buy::before{background:var(--buy);}
.sb-sell{border:1px solid rgba(255,77,106,0.3);box-shadow:var(--glow-sell);}
.sb-sell::before{background:var(--sell);}
.sb-none{border:1px solid var(--border2);}
.sb-none::before{background:var(--muted);}
.sig-label{font-family:var(--mono);font-size:2.6rem;font-weight:700;letter-spacing:0.04em;line-height:1;}
.sb-buy .sig-label{color:var(--buy);text-shadow:0 0 30px rgba(16,217,136,0.5);}
.sb-sell .sig-label{color:var(--sell);text-shadow:0 0 30px rgba(255,77,106,0.5);}
.sb-none .sig-label{color:var(--muted2);}
.sig-sub{font-family:var(--mono);font-size:0.6rem;color:var(--muted2);letter-spacing:0.12em;text-transform:uppercase;margin-top:0.5rem;}
.sig-right{text-align:right;}
.sig-prob{font-family:var(--mono);font-size:1.4rem;font-weight:600;}
.sb-buy .sig-prob{color:var(--buy);}
.sb-sell .sig-prob{color:var(--sell);}
.sb-none .sig-prob{color:var(--muted2);}
.sig-prob-lbl{font-family:var(--mono);font-size:0.56rem;color:var(--muted2);letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem;}
.sig-reason{font-family:var(--mono);font-size:0.6rem;color:var(--muted2);line-height:1.9;margin-top:0.5rem;}
.dtbl{background:var(--surf);border:1px solid var(--border);border-radius:2px;overflow:hidden;}
.dtbl-row{display:flex;justify-content:space-between;align-items:center;
    padding:0.55rem 1.2rem;border-bottom:1px solid var(--border);}
.dtbl-row:last-child{border-bottom:none;}
.dtbl-row:hover{background:var(--surf2);}
.dtbl-k{font-family:var(--mono);font-size:0.62rem;color:var(--muted2);letter-spacing:0.08em;}
.dtbl-v{font-family:var(--mono);font-size:0.68rem;font-weight:500;color:var(--text);}
.dtbl-v.buy{color:var(--buy);}
.dtbl-v.sell{color:var(--sell);}
.dtbl-v.gold{color:var(--accent);}
.dtbl-v.muted{color:var(--muted2);}
.smc-row{display:grid;grid-template-columns:100px 90px 70px 1fr 36px 64px;
    align-items:center;padding:6px 10px;border-radius:4px;margin-bottom:2px;}
.smc-row:hover{background:var(--surf);}
.smc-price{font-family:var(--mono);font-size:0.72rem;font-weight:500;color:var(--text);}
.smc-tag{font-family:var(--mono);font-size:0.55rem;padding:2px 6px;border-radius:3px;border:1px solid;display:inline-block;letter-spacing:0.05em;}
.smc-dir{font-family:var(--mono);font-size:0.58rem;color:var(--muted2);}
.smc-zone{font-family:var(--mono);font-size:0.55rem;color:var(--muted);}
.smc-hits{font-family:var(--mono);font-size:0.55rem;color:var(--muted2);text-align:center;}
.smc-dist{font-family:var(--mono);font-size:0.65rem;font-weight:500;text-align:right;}
.smc-cur{display:flex;align-items:center;gap:10px;margin:10px 0;padding:0 10px;}
.smc-cur-line{flex:1;height:1px;background:var(--border2);}
.smc-cur-price{font-family:var(--mono);font-size:1.1rem;font-weight:500;color:var(--text);}
.smc-zone-hdr{font-family:var(--mono);font-size:0.5rem;letter-spacing:0.2em;color:var(--muted);padding:4px 10px 3px;}
.smc-col-hdr{display:grid;grid-template-columns:100px 90px 70px 1fr 36px 64px;
    padding:4px 10px 6px;font-family:var(--mono);font-size:0.5rem;letter-spacing:0.12em;color:var(--muted);border-bottom:1px solid var(--border);margin-bottom:4px;}
.fresh-row{display:grid;grid-template-columns:140px 110px 60px 1fr;gap:0;
    padding:6px 1.2rem;border-bottom:1px solid var(--border);}
.fresh-row:last-child{border-bottom:none;}
.fresh-row:hover{background:var(--surf2);}
.fresh-k{font-family:var(--mono);font-size:0.6rem;color:var(--muted2);}
.fresh-v{font-family:var(--mono);font-size:0.6rem;color:var(--text);}
.fresh-status{font-family:var(--mono);font-size:0.6rem;}
.candle-status{font-family:var(--mono);font-size:0.62rem;padding:10px 14px;
    border-radius:3px;margin:0.6rem 0;}
.cs-ok{background:rgba(16,217,136,0.06);border:1px solid rgba(16,217,136,0.2);color:var(--buy);}
.cs-warn{background:rgba(245,200,66,0.06);border:1px solid rgba(245,200,66,0.2);color:var(--accent);}
.cs-danger{background:rgba(255,77,106,0.06);border:1px solid rgba(255,77,106,0.2);color:var(--sell);}
.feat-table-wrap{overflow-x:auto;overflow-y:auto;max-height:420px;}
.feat-table{border-collapse:collapse;font-family:var(--mono);font-size:0.58rem;width:100%;}
.feat-table th{background:var(--bg);color:var(--muted2);padding:5px 8px;border-bottom:1px solid var(--border);letter-spacing:0.08em;white-space:nowrap;position:sticky;top:0;}
.feat-table td{padding:4px 8px;border-bottom:1px solid var(--border);color:var(--text);white-space:nowrap;}
.feat-table tr:hover td{background:var(--surf2);}
.feat-table tr.stats-row td{color:var(--accent);background:var(--surf);font-weight:500;}
.stSpinner>div{border-top-color:var(--accent)!important;}
[data-testid="stStatusWidget"]{display:none!important;}
</style>
"""

def _kpi(lbl, val, cls="", sub=""):
    s = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f'<div class="kpi-cell"><div class="kpi-lbl">{lbl}</div><div class="kpi-val {cls}">{val}</div>{s}</div>'

def _section(lbl):
    return f'<div class="section-label">{lbl}</div>'

def _row(k, v, cls=""):
    return f'<div class="dtbl-row"><span class="dtbl-k">{k}</span><span class="dtbl-v {cls}">{v}</span></div>'


def _now_ny():
    return datetime.now(NY_TZ)

def _now_morocco():
    return datetime.now(MOROCCO_TZ)

def _is_candle_settled():
    ny  = _now_ny()
    h   = ny.hour + ny.minute / 60.0
    dow = ny.weekday()
    if dow == 5:
        return True, "Saturday — market closed, Friday candle confirmed"
    if dow == 6:
        if h < 17.0:
            return True, "Sunday before 5pm NY — Friday candle confirmed"
        return False, "Sunday after 5pm NY — new session open, still Friday candle"
    if h >= SETTLEMENT_NY:
        return True, "Past 1:30 PM NY — today's candle is closed and confirmed"
    return False, f"Before settlement ({ny.strftime('%H:%M')} NY) — will use yesterday's candle"

def dist(price, current):
    pct  = (price - current) / current * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"



def _fetch_yf(ticker, start, end, retries=3):
    for i in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if not df.empty:
                return df
        except Exception:
            if i < retries - 1:
                time.sleep(1)
    return pd.DataFrame()


def fetch_fred_data(start, end):
    fred_obj = Fred(api_key=FRED_API_KEY)
    series, ages = {}, {}
    for s in MACRO_SERIES:
        try:
            data = fred_obj.get_series(s, start, end)
            series[s] = data
        except Exception:
            local = os.path.join(ARTEFACT_DIR, f"{s}.csv")
            if os.path.exists(local):
                df   = pd.read_csv(local, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data = df[df.columns[0]].replace(".", np.nan).astype(float)
                data = data[(data.index >= str(start)) & (data.index <= str(end))]
                series[s] = data
            else:
                raise FileNotFoundError(f"No data for {s}")
        last = data.dropna().index[-1] if not data.dropna().empty else None
        if last is not None:
            bday_age = len(pd.bdate_range(start=last, end=pd.Timestamp.today().normalize())) - 1
            ages[s]  = (last.strftime("%Y-%m-%d"), bday_age)
        else:
            ages[s] = ("unknown", 99)
    macro = pd.DataFrame(series)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    return macro, ages


def fetch_all_daily():
    end   = datetime.today()
    start = end - timedelta(days=DAYS_BACK)
    fetch_log = {}

    gold = _fetch_yf("GC=F", start, end)
    fetch_log["XAU/USD"] = ("yfinance GC=F", "ok")

    eur = _fetch_yf("EURUSD=X", start, end)
    fetch_log["EURUSD"] = ("yfinance EURUSD=X", "ok")
    jpy = _fetch_yf("JPY=X", start, end)
    fetch_log["USDJPY"] = ("yfinance JPY=X", "ok")
    macro, fred_ages = fetch_fred_data(start, end)
    fetch_log["FRED"] = ("FRED API", "ok")

    prices = pd.DataFrame({
        "Close_XAUUSD":  gold["Close"],
        "Volume_XAUUSD": gold.get("Volume", pd.Series(dtype=float)),
        "Close_EURUSD":  eur["Close"],
        "Close_USDJPY":  jpy["Close"],
    })
    full_idx = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="B")
    prices   = prices.reindex(full_idx)
    macro    = macro.reindex(full_idx)
    df       = prices.join(macro, how="left")

    fill_report = {}
    for col in df.columns:
        nans = int(df[col].isna().sum())
        if nans > 0:
            gap_sizes, in_gap, g = [], False, 0
            for v in df[col]:
                if pd.isna(v):
                    in_gap = True; g += 1
                else:
                    if in_gap: gap_sizes.append(g)
                    in_gap = False; g = 0
            fill_report[col] = {"nan_filled": nans, "max_gap_days": max(gap_sizes) if gap_sizes else 0}

    df = df.ffill().bfill()

    settled, _ = _is_candle_settled()
    today_n    = pd.Timestamp.today().normalize()
    candle_note = ""
    if df.index[-1] >= today_n:
        if settled:
            candle_note = f"Today's candle ({df.index[-1].date()}) is confirmed closed — kept"
        else:
            candle_note = f"Dropped today's unsettled candle ({df.index[-1].date()}) — using yesterday"
            df = df.iloc[:-1]

    df.dropna(subset=["Close_XAUUSD"], inplace=True)
    df.index.name = "Date"
    return df, fred_ages, fill_report, fetch_log, candle_note


def engineer(df):
    out  = pd.DataFrame(index=df.index)
    gold = df["Close_XAUUSD"]

    out["Close_Returns"]  = gold.pct_change()
    out["Log_Returns"]    = np.log(gold / gold.shift(1))
    out["EURUSD_Returns"] = df["Close_EURUSD"].pct_change()
    out["USDJPY_Returns"] = df["Close_USDJPY"].pct_change()

    sma20 = gold.rolling(20).mean()
    std20 = gold.rolling(20).std()
    upper = sma20 + 2*std20; lower = sma20 - 2*std20
    out["BB_PctB"] = (gold - lower) / (upper - lower)

    ema50  = gold.ewm(span=50,  adjust=False).mean()
    ema200 = gold.ewm(span=200, adjust=False).mean()
    out["Price_Over_EMA50"]  = gold / ema50
    out["Price_Over_EMA200"] = gold / ema200
    out["Bull_Trend"]        = (ema50 - ema200) / ema200

    macd = gold.ewm(span=12,adjust=False).mean() - gold.ewm(span=26,adjust=False).mean()
    out["MACD_Signal_Norm"] = macd.ewm(span=9,adjust=False).mean() / gold

    r20 = out["Log_Returns"].rolling(20)
    out["LogReturn_ZScore"] = (out["Log_Returns"] - r20.mean()) / r20.std()
    c20 = out["Close_Returns"].rolling(20)
    out["Return_ZScore"]    = (out["Close_Returns"] - c20.mean()) / c20.std()

    out["Return_Percentile"] = out["Close_Returns"].rolling(100).rank(pct=True)
    vol = df["Volume_XAUUSD"].replace(0, np.nan).ffill()
    out["Volume_Percentile"] = vol.rolling(100).rank(pct=True)

    ath = gold.expanding().max()
    out["Pct_From_AllTimeHigh"] = (ath - gold) / ath

    z_cols = []
    for col in MACRO_SERIES:
        df[f"{col}_delta"] = df[col].diff()
        for feat in [col, f"{col}_delta"]:
            roll = df[feat].shift(1).rolling(252)
            out[f"{feat}_z"] = (df[feat] - roll.mean()) / roll.std()
            z_cols.append(f"{feat}_z")
    out["Macro_Fast"] = (out[z_cols].mean(axis=1)
                         .replace([np.inf,-np.inf], np.nan)
                         .ffill().bfill().clip(-5, 5))
    out.drop(columns=z_cols, inplace=True)
    out["Close_XAUUSD"] = gold
    return out.dropna(subset=BASE_FEATURES)


def run_ml(feat_df):
    def _load(name):
        path = os.path.join(ARTEFACT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found")
        with open(path, "rb") as f:
            return pickle.load(f)

    model      = _load("cv_best_fold_model.pkl")
    calibrator = _load("calibrator.pkl")
    oof        = pd.read_csv(os.path.join(ARTEFACT_DIR, "cv_predictions_oof.csv"),
                             index_col=0, parse_dates=True)
    today    = feat_df.iloc[[-1]].copy()
    pred_val = float(model.predict(today[BASE_FEATURES].values)[0])
    abs_pred = abs(pred_val)
    hist     = oof["oof_prediction"].dropna().tail(PRED_Z_LOOKBACK)
    h_std    = hist.std()
    pred_z   = float((pred_val - hist.mean()) / h_std) if h_std > 0 else 0.0

    calib_in = pd.DataFrame([[pred_val, abs_pred,
        float(today["Bull_Trend"].iloc[0]), float(today["Macro_Fast"].iloc[0]),
        float(today["BB_PctB"].iloc[0]),    float(today["Price_Over_EMA200"].iloc[0]),
    ]], columns=CALIB_FEATURES)
    prob = float(calibrator.predict_proba(calib_in)[0][1])

    signal = "NO SIGNAL"
    if prob >= PROB_THRESHOLD and abs(pred_z) >= Z_THRESHOLD:
        signal = "BUY" if pred_val > 0 else "SELL"

    return dict(signal=signal, prob=prob, pred_val=pred_val, pred_z=pred_z,
                abs_pred_z=abs(pred_z),
                bull_trend=float(today["Bull_Trend"].iloc[0]),
                macro_fast=float(today["Macro_Fast"].iloc[0]),
                bb_pctb=float(today["BB_PctB"].iloc[0]),
                ema200=float(today["Price_Over_EMA200"].iloc[0]),
                close=float(today["Close_XAUUSD"].iloc[0]))


def weekly_range(feat_df):
    try:
        raw = _fetch_yf("GC=F", datetime.today()-timedelta(days=14), datetime.today())
        if not raw.empty:
            raw = raw[raw.index < pd.Timestamp.today().normalize()].tail(7)
            return float(raw["High"].max()), float(raw["Low"].min()), \
                   f"{raw.index[0].date()} to {raw.index[-1].date()}"
    except Exception:
        pass
    c = feat_df["Close_XAUUSD"].tail(7)
    return float(c.max()), float(c.min()), "approx from close prices"


def intraday_range():
    try:
        raw = yf.download("GC=F", start=datetime.today().strftime("%Y-%m-%d"),
                          interval="1h", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            return None, None, 0
        return float(raw["High"].max()), float(raw["Low"].min()), len(raw)
    except Exception:
        return None, None, 0


def smc_4h(current):
    empty = {k: [] for k in ["bos_bull","bos_bear","choch_bull","choch_bear","ob_bull","ob_bear","sr"]}
    try:
        end   = datetime.utcnow()
        start = end - timedelta(days=58)
        raw   = yf.download("GC=F", start=start, end=end,
                             interval="1h", auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if raw.empty:
            return empty

        ohlc           = raw["Close"].resample("4h").ohlc()
        ohlc.columns   = ["Open","High","Low","Close"]
        ohlc["Volume"] = raw["Volume"].resample("4h").sum()
        ohlc           = ohlc.dropna().iloc[:-1]

        highs  = ohlc["High"].values
        lows   = ohlc["Low"].values
        opens  = ohlc["Open"].values
        closes = ohlc["Close"].values
        n, L   = len(ohlc), SMC_SWING

        hl  = np.zeros(n, dtype=int)
        lvl = np.full(n, np.nan)
        for i in range(L, n-L):
            wh = np.concatenate([highs[i-L:i], highs[i+1:i+L+1]])
            wl = np.concatenate([lows[i-L:i],  lows[i+1:i+L+1]])
            if highs[i] > wh.max():   hl[i]=1;  lvl[i]=highs[i]
            elif lows[i] < wl.min():  hl[i]=-1; lvl[i]=lows[i]

        sh_list = [(i, lvl[i]) for i in range(n) if hl[i]==1]
        sl_list = [(i, lvl[i]) for i in range(n) if hl[i]==-1]
        trend = 0; last_sh = last_sl = None
        res = {k: [] for k in ["bos_bull","bos_bear","choch_bull","choch_bear","ob_bull","ob_bear"]}

        for i in range(1, n):
            for si, sv in sh_list:
                if si < i and (last_sh is None or si > last_sh[0]): last_sh = (si, sv)
            for si, sv in sl_list:
                if si < i and (last_sl is None or si > last_sl[0]): last_sl = (si, sv)

            if last_sh and closes[i] > last_sh[1]:
                res["bos_bull" if trend==1 else "choch_bull"].append(
                    {"price": round(last_sh[1],1), "when": ohlc.index[i].strftime("%m-%d %H:%M")})
                for j in range(last_sh[0]-1, max(0,last_sh[0]-30), -1):
                    if closes[j] < opens[j]:
                        if not any(lows[k]<lows[j] for k in range(j+1,min(j+40,n))):
                            res["ob_bull"].append({
                                "top": round(highs[j],1), "bottom": round(lows[j],1),
                                "mid": round((highs[j]+lows[j])/2,1),
                                "when": ohlc.index[j].strftime("%m-%d %H:%M")})
                        break
                trend=1; last_sh=None

            elif last_sl and closes[i] < last_sl[1]:
                res["bos_bear" if trend==-1 else "choch_bear"].append(
                    {"price": round(last_sl[1],1), "when": ohlc.index[i].strftime("%m-%d %H:%M")})
                for j in range(last_sl[0]-1, max(0,last_sl[0]-30), -1):
                    if closes[j] > opens[j]:
                        if not any(highs[k]>highs[j] for k in range(j+1,min(j+40,n))):
                            res["ob_bear"].append({
                                "top": round(highs[j],1), "bottom": round(lows[j],1),
                                "mid": round((highs[j]+lows[j])/2,1),
                                "when": ohlc.index[j].strftime("%m-%d %H:%M")})
                        break
                trend=-1; last_sl=None

        all_prices = [lvl[i] for i in range(n) if hl[i]!=0 and not np.isnan(lvl[i])]
        used, sr   = [False]*len(all_prices), []
        for i, p in enumerate(all_prices):
            if used[i]: continue
            nearby = [p]
            for j in range(i+1, len(all_prices)):
                if not used[j] and abs(all_prices[j]-p)/p < SR_TOL:
                    nearby.append(all_prices[j]); used[j]=True
            used[i] = True
            if len(nearby) >= 2:
                mid = round(np.mean(nearby), 1)
                if not any(abs(mid-s["price"])/mid < SR_TOL*2 for s in sr):
                    sr.append({"price": mid, "hits": len(nearby)})

        res["sr"] = sorted(sr, key=lambda x: x["price"])
        return res
    except Exception as e:
        return empty


def _render_qwen(sig, r, current, w_high, w_low, i_high, i_low, deduped, raw_df):
    if not QWEN_AVAILABLE:
        return

    st.markdown(_section("QWEN3 MARKET BRIEFING"), unsafe_allow_html=True)
    st.markdown("""
    <style>
    .qwen-card{background:#0c0f14;border:1px solid #1c2030;border-radius:6px;overflow:hidden;margin-bottom:12px;}
    .qwen-card-hdr{padding:10px 14px 8px;border-bottom:1px solid #1c2030;font-family:'JetBrains Mono',monospace;font-size:0.58rem;letter-spacing:0.16em;text-transform:uppercase;}
    .qwen-card-body{padding:12px 14px;font-family:'Space Grotesk',sans-serif;font-size:0.82rem;color:#9ca8b8;line-height:1.8;}
    .qwen-trade{background:#0c0f14;border:1px solid rgba(16,217,136,0.25);border-radius:6px;padding:14px;margin-bottom:12px;}
    .qwen-trade-hdr{font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#10d988;letter-spacing:0.16em;margin-bottom:10px;}
    .qwen-mem{background:#111520;border:1px solid #1c2030;border-radius:4px;padding:10px 14px;margin-bottom:12px;}
    .qwen-mem-hdr{font-family:'JetBrains Mono',monospace;font-size:0.52rem;color:#3d4a5c;letter-spacing:0.14em;margin-bottom:6px;}
    .qwen-mem-row{display:grid;grid-template-columns:90px 80px 56px 64px 1fr;font-family:'JetBrains Mono',monospace;font-size:0.56rem;color:#5a6a80;padding:3px 0;border-bottom:1px solid #1c2030;}
    .qwen-mem-row:last-child{border-bottom:none;}
    </style>
    """, unsafe_allow_html=True)

    memory    = load_memory()
    past_runs = memory.get("runs", [])
    if past_runs:
        mem_rows = ""
        for run in reversed(past_runs[-10:]):
            oc     = run.get("outcome", "pending")
            oc_col = "#10d988" if oc=="win" else "#ff4d6a" if oc=="loss" else "#3d4a5c"
            sc_col = "var(--buy)" if run["signal"]=="BUY" else "var(--sell)" if run["signal"]=="SELL" else "var(--muted2)"
            mem_rows += (f'<div class="qwen-mem-row">'
                         f'<span>{run["date"]}</span>'
                         f'<span style="color:{sc_col}">{run["signal"]}</span>'
                         f'<span>{run["prob"]:.2f}</span>'
                         f'<span style="color:{oc_col}">{oc}</span>'
                         f'<span style="color:#3d4a5c">{run.get("world_view_note","")[:55]}</span>'
                         f'</div>')
        st.markdown(
            f'<div class="qwen-mem"><div class="qwen-mem-hdr">MEMORY LOG — LAST 10 RUNS</div>'
            f'<div class="qwen-mem-row" style="color:#252a38"><span>DATE</span><span>SIGNAL</span>'
            f'<span>PROB</span><span>OUTCOME</span><span>NOTE</span></div>'
            f'{mem_rows}</div>',
            unsafe_allow_html=True)

    if r and isinstance(r, dict):
        payload = {
            "signal":        sig,
            "prob":          r.get("prob", 0),
            "pred_z":        r.get("pred_z", 0),
            "bull_trend":    r.get("bull_trend", 0),
            "macro_fast":    r.get("macro_fast", 0),
            "bb_pctb":       r.get("bb_pctb", 0),
            "ema200":        r.get("ema200", 1),
            "close":         current,
            "weekly_high":   w_high,
            "weekly_low":    w_low,
            "intraday_high": i_high,
            "intraday_low":  i_low,
            "date":          str(raw_df.index[-1].date()),
        }
    else:
        payload = {"signal":"NO SIGNAL","prob":0,"pred_z":0,"bull_trend":0,
                   "macro_fast":0,"bb_pctb":0,"ema200":1,"close":current,
                   "weekly_high":w_high,"weekly_low":w_low,
                   "intraday_high":i_high,"intraday_low":i_low,
                   "date":str(raw_df.index[-1].date())}

    if st.button("RUN QWEN3 BRIEFING", use_container_width=True):
        status = st.empty()

        def cb(msg):
            status.markdown(f'<div class="candle-status cs-warn">{msg}</div>',
                            unsafe_allow_html=True)

        result, _ = run_briefing(payload, deduped, cb)
        status.empty()

        if "error" in result:
            st.markdown(f'<div class="candle-status cs-danger">Qwen error: {result["error"]}</div>',
                        unsafe_allow_html=True)
            return

        cards = [
            ("GLOBAL MARKET CONTEXT",  result.get("market_context",""),    "var(--accent)"),
            ("FEATURE READING",        result.get("feature_reading",""),   "var(--muted2)"),
            ("SMC STRUCTURE ANALYSIS", result.get("smc_analysis",""),      "var(--muted2)"),
            ("ASSET CONNECTIONS",      result.get("asset_connections",""), "var(--muted2)"),
            ("FORWARD OUTLOOK",        result.get("forward_outlook",""),   "#7F77DD"),
        ]
        for title, body, color in cards:
            if body:
                st.markdown(
                    f'<div class="qwen-card">'
                    f'<div class="qwen-card-hdr" style="color:{color}">{title}</div>'
                    f'<div class="qwen-card-body">{body}</div>'
                    f'</div>', unsafe_allow_html=True)

        trade = result.get("trade_scenario")
        if trade and isinstance(trade, dict) and trade.get("entry_zone"):
            st.markdown(
                f'<div class="qwen-trade">'
                f'<div class="qwen-trade-hdr">TRADE SCENARIO  (prob >= 60%)</div>'
                f'<div class="qwen-card-body">'
                f'<b style="color:var(--text)">Entry zone:</b> {trade.get("entry_zone","")}<br><br>'
                f'<b style="color:var(--text)">Reasoning:</b> {trade.get("reasoning","")}<br><br>'
                f'<b style="color:var(--text)">What to watch:</b> {trade.get("what_to_watch","")}'
                f'</div></div>', unsafe_allow_html=True)

        wv = result.get("world_view_update","")
        if wv:
            st.markdown(
                f'<div class="qwen-card">'
                f'<div class="qwen-card-hdr" style="color:#534AB7">WORLD VIEW (saved to memory)</div>'
                f'<div class="qwen-card-body">{wv}</div>'
                f'</div>', unsafe_allow_html=True)

    with st.expander("Update past signal outcome"):
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            od = st.text_input("Date (YYYY-MM-DD)", value=str(raw_df.index[-1].date()))
        with c2:
            ov = st.selectbox("Outcome", ["win","loss"])
        with c3:
            st.write("")
            st.write("")
            if st.button("Save"):
                update_outcome(od, ov)
                st.success(f"Saved {ov} for {od}")


def main():
    st.set_page_config(page_title="Gold Signal", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)

    now = datetime.now()
    mor = _now_morocco()
    ny  = _now_ny()
    mor_h = mor.hour + mor.minute / 60.0
    settled, reason = _is_candle_settled()

    status_col = "var(--buy)" if settled and not (MAINTENANCE_START <= mor_h < MAINTENANCE_END) else "var(--accent)" if not settled else "var(--sell)"
    st.markdown(f"""
    <div class="app-header">
      <div class="app-logo">
        <div class="logo-hex">XAU</div>
        <div>
          <div class="logo-text">XAUUSD &nbsp; Signal Validator</div>
          <div class="logo-sub">Data integrity · SMC levels · ML inference</div>
        </div>
      </div>
      <div class="header-right">
        <div class="header-ts">{now.strftime('%Y-%m-%d &nbsp; %H:%M:%S')}</div>
        <div class="header-status" style="color:{status_col}">
          Morocco {mor.strftime('%H:%M')} &nbsp;·&nbsp; NY {ny.strftime('%H:%M')}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    cs_cls = "cs-ok" if settled and not (MAINTENANCE_START <= mor_h < MAINTENANCE_END) else "cs-danger" if (MAINTENANCE_START <= mor_h < MAINTENANCE_END) else "cs-warn"
    st.markdown(f'<div class="candle-status {cs_cls}">{reason}</div>', unsafe_allow_html=True)

    with st.expander("Safe run windows"):
        st.markdown("""<div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted2);line-height:2.2">
        <span style="color:var(--buy)">BEST</span> &nbsp; 6:30 PM – 9:15 PM Morocco &nbsp;·&nbsp; settled + FRED fresh<br>
        <span style="color:var(--accent)">OK</span> &nbsp;&nbsp; 10:00 PM – 5:00 AM Morocco &nbsp;·&nbsp; yesterday candle confirmed<br>
        <span style="color:var(--sell)">AVOID</span> 9:15 PM – 10:00 PM Morocco &nbsp;·&nbsp; CME maintenance<br>
        <span style="color:var(--muted2)">FRED</span> &nbsp; DFII10/DFII5/DGS2 ready ~8:30 PM Morocco &nbsp;·&nbsp; FEDFUNDS changes on FOMC days only
        </div>""", unsafe_allow_html=True)

    st.markdown(_section("FETCHING DATA"), unsafe_allow_html=True)
    with st.spinner("Fetching all data sources..."):
        raw_df, fred_ages, fill_report, fetch_log, candle_note = fetch_all_daily()

    fetch_rows = "".join(_row(src, f"{method} &nbsp;<span style='color:var(--{'buy' if s=='ok' else 'accent' if s=='fallback' else 'muted2'})'>{s}</span>")
                         for src, (method, s) in fetch_log.items())
    if candle_note:
        fetch_rows += _row("Candle", candle_note, "gold")
    fetch_rows += _row("Last closed candle", str(raw_df.index[-1].date()), "gold")
    st.markdown(f'<div class="dtbl">{fetch_rows}</div>', unsafe_allow_html=True)

    st.markdown(_section("DATA FRESHNESS"), unsafe_allow_html=True)
    gold_last = raw_df.index[-1]
    gold_bday = len(pd.bdate_range(start=gold_last, end=pd.Timestamp.today().normalize())) - 1
    fr_html   = '<div class="dtbl">'
    fr_html  += f'<div class="fresh-row"><span class="fresh-k">XAU/USD</span><span class="fresh-v">{gold_last.date()}</span><span class="fresh-v">{gold_bday}bd</span><span class="fresh-status" style="color:var(--buy)">OK</span></div>'
    for s, (d, age) in fred_ages.items():
        if s == "FEDFUNDS":
            note = "OK — only changes on FOMC days, current value is valid"
            col  = "var(--buy)"
        elif age <= 1:
            note = "OK"
            col  = "var(--buy)"
        else:
            note = f"OK — {age}bd old (weekend/holiday gap, normal)"
            col  = "var(--buy)"
        fr_html += f'<div class="fresh-row"><span class="fresh-k">{s}</span><span class="fresh-v">{d}</span><span class="fresh-v">{age}bd</span><span class="fresh-status" style="color:{col}">{note}</span></div>'
    fr_html += '</div>'
    st.markdown(fr_html, unsafe_allow_html=True)

    if fill_report:
        st.markdown(_section("FILL REPORT"), unsafe_allow_html=True)
        fill_rows = "".join(
            _row(col, f"{v['nan_filled']} NaN filled &nbsp;·&nbsp; max gap {v['max_gap_days']}d &nbsp;<span style='color:var(--muted2);font-size:0.55rem'>{'holiday/weekend — normal' if v['max_gap_days'] <= 3 else 'check source'}</span>", "muted")
            for col, v in fill_report.items()
        )
        st.markdown(f'<div class="dtbl">{fill_rows}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--mono);font-size:0.55rem;color:var(--muted);padding:6px 0">NaN fills are expected. Gold does not trade on weekends or US holidays. Max gap of 1-3 days is always normal. Macro series (FRED) only update on business days.</div>', unsafe_allow_html=True)

    st.markdown(_section("FEATURE ENGINEERING"), unsafe_allow_html=True)
    with st.spinner("Engineering features..."):
        feat_df = engineer(raw_df.copy())
    st.markdown(f'<div class="dtbl">{_row("Rows", str(len(feat_df)))}{_row("Features", str(len(BASE_FEATURES)))}</div>', unsafe_allow_html=True)

    st.markdown(_section("ML INFERENCE"), unsafe_allow_html=True)
    try:
        r       = run_ml(feat_df)
        sig     = r["signal"]
        current = r["close"]
        bt_lbl  = "BULL" if r["bull_trend"]>0.02 else "BEAR" if r["bull_trend"]<-0.02 else "NEUTRAL"
        mf_lbl  = "TIGHT" if r["macro_fast"]>0.5 else "EASY" if r["macro_fast"]<-0.5 else "NEUTRAL"
        sc      = {"BUY":"sb-buy","SELL":"sb-sell","NO SIGNAL":"sb-none"}[sig]
        vc      = {"BUY":"buy","SELL":"sell","NO SIGNAL":"muted"}[sig]

        reason_html = ""
        if sig == "NO SIGNAL":
            parts = []
            if abs(r["pred_z"]) < Z_THRESHOLD:
                parts.append(f"|z|={abs(r['pred_z']):.2f} < {Z_THRESHOLD}")
            if r["prob"] < PROB_THRESHOLD:
                parts.append(f"prob={r['prob']:.3f} < {PROB_THRESHOLD}")
            reason_html = f'<div class="sig-reason">{" &nbsp;|&nbsp; ".join(parts)}</div>'

        st.markdown(f"""
        <div class="sig-banner {sc}">
          <div>
            <div class="sig-label">{sig}</div>
            <div class="sig-sub">XAU/USD &nbsp;·&nbsp; {raw_df.index[-1].date()} &nbsp;·&nbsp; Daily close</div>
            {reason_html}
          </div>
          <div class="sig-right">
            <div class="sig-prob">{r['prob']:.1%}</div>
            <div class="sig-prob-lbl">Win Probability</div>
          </div>
        </div>""", unsafe_allow_html=True)

        rc = "muted" if bt_lbl=="NEUTRAL" else ("bull" if bt_lbl=="BULL" else "bear")
        mc = "muted" if mf_lbl=="NEUTRAL" else ("sell" if mf_lbl=="TIGHT" else "bull")
        rows = [
            ("Close",         f"${r['close']:,.2f}",         "gold"),
            ("Win prob",      f"{r['prob']:.4f}",             vc),
            ("Pred Z-score",  f"{r['pred_z']:+.4f}",          "bull" if r["pred_z"]>0 else "bear"),
            ("Pred value",    f"{r['pred_val']:+.8f}",        ""),
            ("Bull Trend",    f"{r['bull_trend']:+.4f}  [{bt_lbl}]", rc),
            ("Macro Fast",    f"{r['macro_fast']:+.4f}  [{mf_lbl}]", mc),
            ("BB PctB",       f"{r['bb_pctb']:.4f}",          ""),
            ("EMA200 ratio",  f"{r['ema200']:.4f}",            "bull" if r["ema200"]>1 else "bear"),
            ("Prob gate",     f">= {PROB_THRESHOLD}",         "muted"),
            ("Z gate",        f">= {Z_THRESHOLD}",            "muted"),
        ]
        st.markdown(f'<div class="dtbl">{"".join(_row(k,v,c) for k,v,c in rows)}</div>', unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.markdown(f'<div class="candle-status cs-warn">Model files not found: {e} — showing data validation only</div>', unsafe_allow_html=True)
        current = float(feat_df["Close_XAUUSD"].iloc[-1])
        sig     = "NO SIGNAL"

    st.markdown(_section("WEEKLY RANGE — LAST 7 CLOSED DAILY CANDLES"), unsafe_allow_html=True)
    w_high, w_low, w_dates = weekly_range(feat_df)
    w_mid = (w_high + w_low) / 2
    pos   = "upper half — near weekly resistance" if current > w_mid else "lower half — near weekly support"
    st.markdown(f"""
    <div class="kpi-strip">
      {_kpi("Weekly High",  f"${w_high:,.1f}", "bear", dist(w_high, current))}
      {_kpi("Weekly Mid",   f"${w_mid:,.1f}",  "",     dist(w_mid,  current))}
      {_kpi("Weekly Low",   f"${w_low:,.1f}",  "bull", dist(w_low,  current))}
      {_kpi("Range",        f"${w_high-w_low:,.1f}", "muted")}
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="dtbl">{_row("Period", w_dates)}{_row("Position", pos)}</div>', unsafe_allow_html=True)

    st.markdown(_section("INTRADAY RANGE — TODAY FROM 00:00"), unsafe_allow_html=True)
    i_high, i_low, n_bars = intraday_range()
    if i_high and i_low:
        i_mid = (i_high + i_low) / 2
        st.markdown(f"""
        <div class="kpi-strip">
          {_kpi("Intraday High", f"${i_high:,.1f}", "bear", dist(i_high, current))}
          {_kpi("Intraday Mid",  f"${i_mid:,.1f}",  "",     dist(i_mid,  current))}
          {_kpi("Intraday Low",  f"${i_low:,.1f}",  "bull", dist(i_low,  current))}
          {_kpi("1H bars today", str(n_bars), "muted")}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="dtbl">{_row("Intraday", "No data yet — market not open")}</div>', unsafe_allow_html=True)

    st.markdown(_section("4H SMC LEVELS"), unsafe_allow_html=True)
    with st.spinner("Computing 4H SMC levels..."):
        smc = smc_4h(current)

    all_levels = []
    for ev in smc["bos_bull"]:
        all_levels.append({"price": ev["price"], "type": "BOS", "dir": "Bull", "col": "#10d988", "zone": f"formed {ev['when']}", "hits": 1})
    for ev in smc["choch_bull"]:
        all_levels.append({"price": ev["price"], "type": "CHoCH", "dir": "Bull", "col": "#5DCAA5", "zone": f"formed {ev['when']}", "hits": 1})
    for ob in smc["ob_bull"]:
        all_levels.append({"price": ob["mid"], "type": "OB", "dir": "Bull", "col": "#10d988", "zone": f"${ob['bottom']:,.1f}–${ob['top']:,.1f}", "hits": 1})
    for ev in smc["bos_bear"]:
        all_levels.append({"price": ev["price"], "type": "BOS", "dir": "Bear", "col": "#ff4d6a", "zone": f"formed {ev['when']}", "hits": 1})
    for ev in smc["choch_bear"]:
        all_levels.append({"price": ev["price"], "type": "CHoCH", "dir": "Bear", "col": "#F0997B", "zone": f"formed {ev['when']}", "hits": 1})
    for ob in smc["ob_bear"]:
        all_levels.append({"price": ob["mid"], "type": "OB", "dir": "Bear", "col": "#ff4d6a", "zone": f"${ob['bottom']:,.1f}–${ob['top']:,.1f}", "hits": 1})
    for sr in smc["sr"]:
        all_levels.append({"price": sr["price"], "type": "S/R", "dir": "Neutral", "col": "#f5c842", "zone": "", "hits": sr["hits"]})

    seen, deduped = [], []
    for lv in sorted(all_levels, key=lambda x: x["price"], reverse=True):
        if not any(abs(lv["price"]-s)/lv["price"] < SR_TOL*2 for s in seen):
            seen.append(lv["price"]); deduped.append(lv)

    resistance = sorted([l for l in deduped if l["price"] > current], key=lambda x: x["price"])
    support    = sorted([l for l in deduped if l["price"] < current], key=lambda x: x["price"], reverse=True)

    def smc_row(lv):
        dc  = "#ff4d6a" if lv["price"] > current else "#10d988"
        hits_s = f"{lv['hits']}x" if lv["hits"] > 1 else ""
        return (f'<div class="smc-row" style="background:{lv["col"]}08">'
                f'<span class="smc-price">${lv["price"]:,.1f}</span>'
                f'<span class="smc-tag" style="color:{lv["col"]};border-color:{lv["col"]}40">{lv["type"]}</span>'
                f'<span class="smc-dir" style="color:{lv["col"]}">{lv["dir"]}</span>'
                f'<span class="smc-zone">{lv["zone"]}</span>'
                f'<span class="smc-hits">{hits_s}</span>'
                f'<span class="smc-dist" style="color:{dc}">{dist(lv["price"],current)}</span>'
                f'</div>')

    smc_html = (f'<div class="smc-col-hdr"><span>PRICE</span><span>TYPE</span><span>DIR</span>'
                f'<span>ZONE / FORMED</span><span>HITS</span><span style="text-align:right">DIST</span></div>')
    smc_html += '<div class="smc-zone-hdr">RESISTANCE</div>'
    smc_html += "".join(smc_row(l) for l in resistance[:8])
    smc_html += (f'<div class="smc-cur"><div class="smc-cur-line"></div>'
                 f'<div><div class="smc-cur-price">${current:,.1f}</div>'
                 f'<div style="font-family:var(--mono);font-size:0.5rem;color:var(--muted2);margin-top:1px">CURRENT &nbsp;·&nbsp; ML: <span style="color:var(--{"buy" if sig=="BUY" else "sell" if sig=="SELL" else "muted2"})">{sig}</span></div></div>'
                 f'<div class="smc-cur-line"></div></div>')
    smc_html += '<div class="smc-zone-hdr">SUPPORT</div>'
    smc_html += "".join(smc_row(l) for l in support[:8])
    st.markdown(smc_html, unsafe_allow_html=True)

    st.markdown(_section("FEATURE TABLE — LAST 252 ROWS"), unsafe_allow_html=True)
    tail  = feat_df[["Close_XAUUSD"] + BASE_FEATURES].tail(PRED_Z_LOOKBACK).copy()
    means = tail.mean()
    stds  = tail.std()

    th = "".join(f"<th>{c}</th>" for c in ["Date","Close"] + BASE_FEATURES)
    mean_tds = f"<td>MEAN</td><td>{means['Close_XAUUSD']:.2f}</td>" + "".join(f"<td>{means[f]:.4f}</td>" for f in BASE_FEATURES)
    std_tds  = f"<td>STD</td><td>{stds['Close_XAUUSD']:.2f}</td>"  + "".join(f"<td>{stds[f]:.4f}</td>"  for f in BASE_FEATURES)
    data_rows = ""
    for date, row in tail.iterrows():
        tds = f"<td>{date.strftime('%Y-%m-%d')}</td><td>{row['Close_XAUUSD']:,.1f}</td>"
        tds += "".join(f"<td>{row[f]:.4f}</td>" if not np.isnan(row[f]) else "<td>—</td>" for f in BASE_FEATURES)
        data_rows += f"<tr>{tds}</tr>"

    table_html = (f'<div class="feat-table-wrap"><table class="feat-table">'
                  f'<thead><tr>{th}</tr></thead>'
                  f'<tbody>'
                  f'<tr class="stats-row">{mean_tds}</tr>'
                  f'<tr class="stats-row">{std_tds}</tr>'
                  f'{data_rows}'
                  f'</tbody></table></div>')
    st.markdown(table_html, unsafe_allow_html=True)

    _render_qwen(sig, r if "r" in dir() else None,
                 current, w_high, w_low, i_high, i_low,
                 deduped, raw_df)


if __name__ == "__main__":
    main()