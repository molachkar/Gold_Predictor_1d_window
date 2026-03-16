"""
SMC Chart Component — XAUUSD 4H
================================
Detects and visualises on a Plotly chart:
  - BOS  (Break of Structure)
  - CHoCH (Change of Character)
  - Order Blocks (last opposing candle before BOS)
  - Key S/R Levels (swing highs/lows respected 2+ times)

Drop-in for Gold inference.py:
  from smc_chart import render_smc_chart
  render_smc_chart()          # call anywhere in main()

Dependencies (add to requirements.txt):
  smartmoneyconcepts>=0.0.17
  plotly>=5.0
  yfinance>=0.2
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── constants ──────────────────────────────────────────────────────────────────
TICKER        = "GC=F"
INTERVAL      = "4h"
LOOKBACK_DAYS = 90          # how many days of 4H data to fetch
SWING_LENGTH  = 5           # bars each side to qualify as a swing high/low
SR_TOLERANCE  = 0.0015      # 0.15% price band to cluster S/R levels
SR_MIN_HITS   = 2           # minimum touches to call a level "key"
OB_EXTEND_BARS = 40         # how far right to extend order block boxes

# colours — match your existing dark theme
C_BG          = "#05070a"
C_SURF        = "#0c0f14"
C_BORDER      = "#1c2030"
C_TEXT        = "#e2e8f0"
C_MUTED       = "#5a6a80"
C_GOLD        = "#f5c842"
C_BUY         = "#10d988"
C_SELL        = "#ff4d6a"
C_BOS_BULL    = "#10d988"
C_BOS_BEAR    = "#ff4d6a"
C_CHOCH_BULL  = "#5DCAA5"
C_CHOCH_BEAR  = "#F0997B"
C_OB_BULL     = "rgba(16,217,136,0.15)"
C_OB_BEAR     = "rgba(255,77,106,0.15)"
C_OB_BULL_BD  = "rgba(16,217,136,0.6)"
C_OB_BEAR_BD  = "rgba(255,77,106,0.6)"
C_SR          = "rgba(245,200,66,0.25)"
C_SR_BD       = "rgba(245,200,66,0.7)"


# ── data fetching ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)   # refresh every 15 min
def fetch_4h(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch 4H OHLCV for GC=F.
    yfinance returns up to 60 days of 1h data and ~730 days of 1d.
    For 4H we fetch 1H and resample — most reliable approach.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=min(lookback_days, 59))   # yfinance 1h cap

    raw = yf.download(
        TICKER, start=start, end=end,
        interval="1h", auto_adjust=False, progress=False
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    if raw.empty:
        return pd.DataFrame()

    # resample 1H → 4H
    ohlc = raw["Close"].resample("4h").ohlc()
    ohlc.columns = ["Open", "High", "Low", "Close"]
    ohlc["Volume"] = raw["Volume"].resample("4h").sum()
    ohlc.dropna(inplace=True)

    # drop incomplete current candle
    ohlc = ohlc.iloc[:-1]
    return ohlc


# ── swing high / low detection ─────────────────────────────────────────────────
def find_swings(df: pd.DataFrame, length: int = SWING_LENGTH) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      HighLow : 1 = swing high, -1 = swing low, 0 = neither
      Level   : price of the swing point (NaN if not a swing)
    Classic pivot: high[i] > all highs in [i-length, i+length] except itself.
    Uses only confirmed (closed) candles — no look-ahead.
    """
    highs = df["High"].values
    lows  = df["Low"].values
    n     = len(df)
    hl    = np.zeros(n, dtype=int)
    lvl   = np.full(n, np.nan)

    for i in range(length, n - length):
        window_h = np.concatenate([highs[i-length:i], highs[i+1:i+length+1]])
        window_l = np.concatenate([lows[i-length:i],  lows[i+1:i+length+1]])
        if highs[i] > window_h.max():
            hl[i]  = 1
            lvl[i] = highs[i]
        elif lows[i] < window_l.min():
            hl[i]  = -1
            lvl[i] = lows[i]

    out = pd.DataFrame({"HighLow": hl, "Level": lvl}, index=df.index)
    return out


# ── BOS / CHoCH detection ──────────────────────────────────────────────────────
def find_bos_choch(df: pd.DataFrame, swings: pd.DataFrame):
    """
    Scans left-to-right through confirmed swing points.
    Tracks current trend (last BOS direction).
    BOS   = close breaks the SAME direction as current trend through last swing.
    CHoCH = close breaks the OPPOSITE direction through last swing.

    Returns list of dicts:
      { type, direction, level, bar_index, broken_index }
    """
    events   = []
    closes   = df["Close"].values
    highs    = df["High"].values
    lows     = df["Low"].values
    idx      = df.index
    n        = len(df)

    swing_highs = [(i, swings["Level"].iloc[i])
                   for i in range(len(swings))
                   if swings["HighLow"].iloc[i] == 1]
    swing_lows  = [(i, swings["Level"].iloc[i])
                   for i in range(len(swings))
                   if swings["HighLow"].iloc[i] == -1]

    trend = 0   # 1 = bull, -1 = bear, 0 = unknown

    # last unbroken swing high and low
    last_sh = None
    last_sl = None

    for i in range(1, n):
        # update last swing points seen so far (causal)
        for si, sv in swing_highs:
            if si < i:
                if last_sh is None or si > last_sh[0]:
                    last_sh = (si, sv)
        for si, sv in swing_lows:
            if si < i:
                if last_sl is None or si > last_sl[0]:
                    last_sl = (si, sv)

        # check bullish break (close above last swing high)
        if last_sh is not None and closes[i] > last_sh[1]:
            ev_type = "BOS" if trend == 1 else "CHoCH"
            events.append({
                "type":          ev_type,
                "direction":     1,            # bullish
                "level":         last_sh[1],
                "bar_index":     last_sh[0],
                "broken_index":  i,
                "broken_time":   idx[i],
                "level_time":    idx[last_sh[0]],
            })
            trend   = 1
            last_sh = None   # consumed — wait for new swing

        # check bearish break (close below last swing low)
        elif last_sl is not None and closes[i] < last_sl[1]:
            ev_type = "BOS" if trend == -1 else "CHoCH"
            events.append({
                "type":          ev_type,
                "direction":     -1,           # bearish
                "level":         last_sl[1],
                "bar_index":     last_sl[0],
                "broken_index":  i,
                "broken_time":   idx[i],
                "level_time":    idx[last_sl[0]],
            })
            trend   = -1
            last_sl = None

    return events


# ── order block detection ──────────────────────────────────────────────────────
def find_order_blocks(df: pd.DataFrame, bos_events: list) -> list:
    """
    For each BOS/CHoCH event, find the order block:
      Bullish BOS  → scan left from the broken swing high
                     find the LAST bearish candle (Close < Open) before the break
      Bearish BOS  → scan left from the broken swing low
                     find the LAST bullish candle (Close > Open) before the break

    Returns list of dicts:
      { direction, top, bottom, start_time, end_time, mitigated }
    """
    opens  = df["Open"].values
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    idx    = df.index
    n      = len(df)
    obs    = []

    for ev in bos_events:
        bi = ev["broken_index"]
        if bi < 2:
            continue

        if ev["direction"] == 1:          # bullish BOS → find last bearish candle
            ob_idx = None
            for j in range(bi - 1, max(0, bi - 30), -1):
                if closes[j] < opens[j]:  # bearish candle
                    ob_idx = j
                    break
            if ob_idx is None:
                continue
            top    = highs[ob_idx]
            bottom = lows[ob_idx]
            color  = "bull"
        else:                             # bearish BOS → find last bullish candle
            ob_idx = None
            for j in range(bi - 1, max(0, bi - 30), -1):
                if closes[j] > opens[j]:  # bullish candle
                    ob_idx = j
                    break
            if ob_idx is None:
                continue
            top    = highs[ob_idx]
            bottom = lows[ob_idx]
            color  = "bear"

        # check if already mitigated (price traded back through the OB zone)
        end_idx   = min(ob_idx + OB_EXTEND_BARS, n - 1)
        mitigated = False
        for k in range(ob_idx + 1, end_idx + 1):
            if color == "bull" and lows[k] < bottom:
                mitigated = True
                end_idx   = k
                break
            if color == "bear" and highs[k] > top:
                mitigated = True
                end_idx   = k
                break

        obs.append({
            "direction":  color,
            "top":        top,
            "bottom":     bottom,
            "start_time": idx[ob_idx],
            "end_time":   idx[end_idx],
            "mitigated":  mitigated,
            "ob_index":   ob_idx,
        })

    # deduplicate — keep newest OB per price zone
    seen = []
    deduped = []
    for ob in sorted(obs, key=lambda x: x["ob_index"], reverse=True):
        zone_mid = (ob["top"] + ob["bottom"]) / 2
        duplicate = any(
            abs(zone_mid - s) / zone_mid < SR_TOLERANCE for s in seen
        )
        if not duplicate:
            seen.append(zone_mid)
            deduped.append(ob)

    return deduped


# ── key S/R level detection ────────────────────────────────────────────────────
def find_key_levels(df: pd.DataFrame, swings: pd.DataFrame,
                    tol: float = SR_TOLERANCE,
                    min_hits: int = SR_MIN_HITS) -> list:
    """
    Clusters swing highs and lows that are within `tol` of each other.
    A cluster with >= min_hits touches becomes a key level.
    Returns list of { price, hits, top, bottom }.
    """
    swing_prices = swings.loc[swings["HighLow"] != 0, "Level"].dropna().values

    if len(swing_prices) < min_hits:
        return []

    clusters = []
    used     = np.zeros(len(swing_prices), dtype=bool)

    for i, p in enumerate(swing_prices):
        if used[i]:
            continue
        nearby = [p]
        for j in range(i + 1, len(swing_prices)):
            if not used[j] and abs(swing_prices[j] - p) / p < tol:
                nearby.append(swing_prices[j])
                used[j] = True
        used[i] = True
        if len(nearby) >= min_hits:
            mid    = np.mean(nearby)
            spread = mid * tol
            clusters.append({
                "price":  mid,
                "hits":   len(nearby),
                "top":    mid + spread,
                "bottom": mid - spread,
            })

    return clusters


# ── Plotly chart builder ───────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, swings: pd.DataFrame,
                bos_events: list, obs: list, key_levels: list,
                ml_signal: str = "NO SIGNAL") -> go.Figure:

    fig = go.Figure()

    # ── candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="XAU/USD 4H",
        increasing_line_color=C_BUY,
        decreasing_line_color=C_SELL,
        increasing_fillcolor=C_BUY,
        decreasing_fillcolor=C_SELL,
        line_width=1,
        whiskerwidth=0.4,
    ))

    # ── key S/R levels ────────────────────────────────────────────────────────
    for lvl in key_levels:
        fig.add_hrect(
            y0=lvl["bottom"], y1=lvl["top"],
            fillcolor=C_SR, line_color=C_SR_BD, line_width=0.8,
            annotation_text=f"  S/R {lvl['price']:,.0f}  ({lvl['hits']} hits)",
            annotation_font_size=9,
            annotation_font_color=C_GOLD,
            annotation_position="right",
        )

    # ── order blocks ──────────────────────────────────────────────────────────
    for ob in obs:
        if ob["mitigated"]:
            continue   # only show active (unmitigated) OBs
        fill = C_OB_BULL if ob["direction"] == "bull" else C_OB_BEAR
        bd   = C_OB_BULL_BD if ob["direction"] == "bull" else C_OB_BEAR_BD
        label = "Bull OB" if ob["direction"] == "bull" else "Bear OB"
        fig.add_shape(
            type="rect",
            x0=ob["start_time"], x1=ob["end_time"],
            y0=ob["bottom"],     y1=ob["top"],
            fillcolor=fill, line_color=bd, line_width=0.8,
        )
        fig.add_annotation(
            x=ob["start_time"], y=ob["top"],
            text=f"  {label} {ob['top']:,.0f}–{ob['bottom']:,.0f}",
            font=dict(size=9, color=C_BUY if ob["direction"] == "bull" else C_SELL),
            showarrow=False, xanchor="left", yanchor="bottom",
        )

    # ── BOS / CHoCH lines ─────────────────────────────────────────────────────
    shown_labels = set()
    for ev in bos_events[-30:]:   # last 30 events max to avoid clutter
        is_bull  = ev["direction"] == 1
        is_bos   = ev["type"] == "BOS"
        color    = (C_BOS_BULL if is_bos else C_CHOCH_BULL) if is_bull else \
                   (C_BOS_BEAR if is_bos else C_CHOCH_BEAR)
        label    = ev["type"]
        dash     = "solid" if is_bos else "dot"

        fig.add_shape(
            type="line",
            x0=ev["level_time"], x1=ev["broken_time"],
            y0=ev["level"],      y1=ev["level"],
            line=dict(color=color, width=1.2, dash=dash),
        )
        # label only if not already shown for this event type
        label_key = f"{label}_{is_bull}"
        if label_key not in shown_labels:
            fig.add_annotation(
                x=ev["broken_time"], y=ev["level"],
                text=f" {label}",
                font=dict(size=9, color=color),
                showarrow=False, xanchor="left",
                yanchor="bottom" if is_bull else "top",
            )
            shown_labels.add(label_key)

    # ── swing high / low dots ─────────────────────────────────────────────────
    sh = swings[swings["HighLow"] == 1]
    sl = swings[swings["HighLow"] == -1]

    fig.add_trace(go.Scatter(
        x=sh.index, y=df.loc[sh.index, "High"] * 1.001,
        mode="markers",
        marker=dict(symbol="triangle-down", size=6, color=C_SELL, opacity=0.7),
        name="Swing High", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=sl.index, y=df.loc[sl.index, "Low"] * 0.999,
        mode="markers",
        marker=dict(symbol="triangle-up", size=6, color=C_BUY, opacity=0.7),
        name="Swing Low", showlegend=False,
    ))

    # ── ML signal badge ───────────────────────────────────────────────────────
    badge_color = C_BUY if ml_signal == "BUY" else C_SELL if ml_signal == "SELL" else C_MUTED
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=f"<b>ML Signal: {ml_signal}</b>",
        font=dict(size=13, color=badge_color),
        bgcolor=C_SURF, bordercolor=badge_color, borderwidth=1,
        showarrow=False, xanchor="left", yanchor="top",
    )

    # ── layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(color=C_TEXT, family="JetBrains Mono, monospace", size=11),
        xaxis=dict(
            showgrid=True, gridcolor=C_BORDER, gridwidth=0.5,
            tickfont=dict(size=10, color=C_MUTED),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(
            showgrid=True, gridcolor=C_BORDER, gridwidth=0.5,
            tickfont=dict(size=10, color=C_MUTED),
            tickformat=",.0f",
            side="right",
        ),
        margin=dict(l=10, r=60, t=40, b=10),
        legend=dict(
            bgcolor=C_SURF, bordercolor=C_BORDER, borderwidth=0.5,
            font=dict(size=10, color=C_MUTED),
            orientation="h", x=0, y=-0.04,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C_SURF, bordercolor=C_BORDER,
            font=dict(color=C_TEXT, size=11),
        ),
        height=580,
        title=dict(
            text="XAU/USD · 4H · Smart Money Structure",
            font=dict(size=12, color=C_MUTED),
            x=0.5,
        ),
    )

    # hide weekend gaps
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[22, 1], pattern="hour"),  # dead hours in futures
    ])

    return fig


# ── summary stats helper ───────────────────────────────────────────────────────
def _smc_summary(bos_events: list, obs: list, key_levels: list) -> dict:
    recent = bos_events[-1] if bos_events else None
    active_obs = [o for o in obs if not o["mitigated"]]
    bull_obs   = [o for o in active_obs if o["direction"] == "bull"]
    bear_obs   = [o for o in active_obs if o["direction"] == "bear"]
    last_type  = recent["type"] if recent else "—"
    last_dir   = "Bullish" if recent and recent["direction"] == 1 else \
                 "Bearish" if recent else "—"
    return dict(
        last_event   = f"{last_type} ({last_dir})",
        active_obs   = len(active_obs),
        bull_obs     = len(bull_obs),
        bear_obs     = len(bear_obs),
        key_levels   = len(key_levels),
        total_events = len(bos_events),
    )


# ── main render function (call from Gold inference.py) ─────────────────────────
def render_smc_chart(ml_signal: str = "NO SIGNAL",
                     lookback_days: int = LOOKBACK_DAYS):
    """
    Call this from anywhere inside your Streamlit app:

        from smc_chart import render_smc_chart
        render_smc_chart(ml_signal=r["signal"])
    """

    # ── section header (matches your existing CSS style) ──────────────────────
    st.markdown(
        '<div class="section-label">XAU/USD · 4H · SMART MONEY STRUCTURE</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading 4H candles and computing SMC structure..."):
        df = fetch_4h(lookback_days)

    if df.empty or len(df) < SWING_LENGTH * 2 + 5:
        st.warning("Not enough 4H data returned — try again in a few minutes.")
        return

    # ── compute all layers ────────────────────────────────────────────────────
    swings     = find_swings(df, SWING_LENGTH)
    bos_events = find_bos_choch(df, swings)
    obs        = find_order_blocks(df, bos_events)
    key_levels = find_key_levels(df, swings)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    s = _smc_summary(bos_events, obs, key_levels)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last structure event", s["last_event"])
    c2.metric("Active order blocks",  s["active_obs"],
              delta=f"+{s['bull_obs']} bull  -{s['bear_obs']} bear")
    c3.metric("Key S/R levels",       s["key_levels"])
    c4.metric("Total BOS/CHoCH",      s["total_events"])

    # ── chart ─────────────────────────────────────────────────────────────────
    fig = build_chart(df, swings, bos_events, obs, key_levels, ml_signal)
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom":     True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png", "filename": "xauusd_4h_smc", "scale": 2
        },
    })

    # ── legend explainer ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:18px;font-family:'JetBrains Mono',monospace;
                font-size:0.58rem;color:#5a6a80;margin-top:0.5rem;padding:0 0.2rem">
      <span><span style="color:#10d988">▲</span> Swing Low &nbsp;
            <span style="color:#ff4d6a">▼</span> Swing High</span>
      <span><span style="color:#10d988">——</span> Bullish BOS</span>
      <span><span style="color:#ff4d6a">——</span> Bearish BOS</span>
      <span><span style="color:#5DCAA5">·····</span> Bullish CHoCH</span>
      <span><span style="color:#F0997B">·····</span> Bearish CHoCH</span>
      <span><span style="color:#10d988;opacity:0.6">▬</span> Bull Order Block</span>
      <span><span style="color:#ff4d6a;opacity:0.6">▬</span> Bear Order Block</span>
      <span><span style="color:#f5c842;opacity:0.6">▬</span> Key S/R Level</span>
    </div>
    """, unsafe_allow_html=True)