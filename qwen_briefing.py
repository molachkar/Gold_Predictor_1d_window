import os, json, requests
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
import xml.etree.ElementTree as ET

SAMBANOVA_KEY  = "cf4f3348-46b8-4b52-aa62-e9febd509c98"
SAMBANOVA_URL  = "https://api.sambanova.ai/v1/chat/completions"
MODEL          = "Qwen3-235B"
MEMORY_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_memory.json")
MAX_MEMORY     = 30
TRADE_PROB_MIN = 0.60

NEWS_FEEDS = {
    "XAU/USD":  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
    "S&P 500":  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    "Oil":      "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL%3DF&region=US&lang=en-US",
    "Nasdaq":   "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US",
    "Bitcoin":  "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
    "Silver":   "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SI%3DF&region=US&lang=en-US",
}


def _fetch_headlines(url, max_items=5):
    try:
        req  = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            raw = resp.read()
        root    = ET.fromstring(raw)
        channel = root.find("channel")
        if channel is None:
            return []
        items  = []
        cutoff = datetime.utcnow() - timedelta(days=5)
        for item in channel.findall("item")[:max_items * 2]:
            title   = (item.findtext("title") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            if not title:
                continue
            try:
                pub_dt = datetime.strptime(pub_raw, "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass
            items.append(title)
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def fetch_all_headlines():
    result = {}
    for asset, url in NEWS_FEEDS.items():
        result[asset] = _fetch_headlines(url)
    return result


def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "world_view": "",
        "runs": [],
        "created": datetime.now().strftime("%Y-%m-%d"),
    }


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def _build_prompt(signal_data, smc_data, headlines, memory):
    sig     = signal_data.get("signal", "NO SIGNAL")
    prob    = signal_data.get("prob", 0)
    pred_z  = signal_data.get("pred_z", 0)
    bt      = signal_data.get("bull_trend", 0)
    mf      = signal_data.get("macro_fast", 0)
    bb      = signal_data.get("bb_pctb", 0)
    ema200  = signal_data.get("ema200", 1)
    close   = signal_data.get("close", 0)
    w_high  = signal_data.get("weekly_high", 0)
    w_low   = signal_data.get("weekly_low", 0)
    i_high  = signal_data.get("intraday_high")
    i_low   = signal_data.get("intraday_low")
    date    = signal_data.get("date", datetime.now().strftime("%Y-%m-%d"))

    bt_lbl = "BULL" if bt > 0.02 else "BEAR" if bt < -0.02 else "NEUTRAL"
    mf_lbl = "TIGHT" if mf > 0.5 else "EASY" if mf < -0.5 else "NEUTRAL"

    res_levels = [l for l in smc_data if l["price"] > close]
    sup_levels = [l for l in smc_data if l["price"] < close]
    nearest_res = res_levels[0] if res_levels else None
    nearest_sup = sup_levels[0] if sup_levels else None

    news_block = ""
    for asset, titles in headlines.items():
        if titles:
            news_block += f"\n{asset}:\n" + "\n".join(f"  - {t}" for t in titles)

    past_runs = memory.get("runs", [])[-10:]
    memory_block = ""
    if past_runs:
        memory_block = "\nYour previous observations (last 10 runs):\n"
        for run in past_runs:
            outcome = run.get("outcome", "pending")
            memory_block += f"  {run['date']}  signal={run['signal']}  prob={run['prob']:.2f}  outcome={outcome}\n"
            if run.get("world_view_note"):
                memory_block += f"    note: {run['world_view_note']}\n"

    world_view = memory.get("world_view", "")
    world_view_block = f"\nYour current world view (built over time):\n{world_view}\n" if world_view else ""

    trade_instruction = ""
    if sig in ("BUY", "SELL") and prob >= TRADE_PROB_MIN:
        trade_instruction = f"""
Since win probability is {prob:.1%} (above the 60% threshold), also provide a trade_scenario section with:
- entry_zone: the specific price zone to watch for entry based on nearest {'bull OB or support' if sig == 'BUY' else 'bear OB or resistance'}
- reasoning: why this zone makes sense given the signal direction and SMC structure
- what_to_watch: one or two things that would invalidate this setup
Do NOT suggest stop loss or take profit levels. Focus only on entry context and reasoning."""

    prompt = f"""You are a professional gold market analyst with deep knowledge of macro economics, central bank policy, geopolitics, and technical market structure. You have persistent memory of your previous analysis runs.

Today's date: {date}

=== MARKET DATA ===
XAU/USD close: ${close:,.2f}
ML Signal: {sig}
Win probability: {prob:.4f} ({prob:.1%})
Pred Z-score: {pred_z:+.4f}
Bull Trend: {bt:+.4f} [{bt_lbl}]
Macro Fast: {mf:+.4f} [{mf_lbl}]
BB PctB: {bb:.4f}
EMA200 ratio: {ema200:.4f}
Weekly High: ${w_high:,.1f}  |  Weekly Low: ${w_low:,.1f}
{f'Intraday High: ${i_high:,.1f}  |  Intraday Low: ${i_low:,.1f}' if i_high else 'Intraday: not yet available'}

=== NEAREST SMC LEVELS ===
Nearest resistance: {f"${nearest_res['price']:,.1f} ({nearest_res['type']} {nearest_res['dir']})" if nearest_res else "none found"}
Nearest support: {f"${nearest_sup['price']:,.1f} ({nearest_sup['type']} {nearest_sup['dir']})" if nearest_sup else "none found"}
All levels above: {', '.join(f"${l['price']:,.1f} {l['type']}" for l in res_levels[:5])}
All levels below: {', '.join(f"${l['price']:,.1f} {l['type']}" for l in sup_levels[:5])}

=== MARKET HEADLINES (last 5 days) ===
{news_block if news_block else "No headlines available"}
{world_view_block}
{memory_block}

=== YOUR TASK ===
Respond ONLY with valid JSON, no markdown, no backticks, no preamble.

{trade_instruction}

Return this exact structure:
{{
  "market_context": "2-3 sentences explaining what is happening in the global economy right now and why it matters for gold. Reference specific events from headlines. Use your knowledge of geopolitics and central bank activity.",
  "feature_reading": "2-3 sentences explaining what the ML model features are saying in plain English. Explain Bull Trend, Macro Fast, BB PctB and pred_z in human terms. Do not just repeat the numbers.",
  "smc_analysis": "2-3 sentences explaining how the current SMC structure relates to the signal direction. Which levels are most important right now and why.",
  "asset_connections": "2 sentences on how the other 5 assets (S&P, Oil, Nasdaq, Bitcoin, Silver) are currently positioned relative to gold.",
  "forward_outlook": "2 sentences on what could happen next and what key events or levels to watch.",
  "trade_scenario": {{"entry_zone": "", "reasoning": "", "what_to_watch": ""}} or null if prob < 0.60 or no signal,
  "world_view_update": "One paragraph (3-5 sentences) updating your running world view on global macro, central banks, and gold's place in the current regime. This will be saved and shown to you on the next run.",
  "run_note": "One sentence summary of this run for your memory log."
}}"""

    return prompt


def _call_qwen(prompt):
    try:
        r = requests.post(
            SAMBANOVA_URL,
            headers={
                "Authorization": f"Bearer {SAMBANOVA_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      MODEL,
                "messages":   [{"role": "user", "content": prompt}],
                "max_tokens": 1200,
                "temperature": 0.3,
            },
            timeout=60,
        )
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        if raw.startswith("<think>"):
            end_think = raw.find("</think>")
            if end_think != -1:
                raw = raw[end_think + 8:].strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "raw": r.text[:300] if "r" in dir() else "no response"}


def run_briefing(signal_data, smc_levels, status_callback=None):
    if status_callback:
        status_callback("Fetching market headlines...")
    headlines = fetch_all_headlines()

    if status_callback:
        status_callback("Loading memory...")
    memory = load_memory()

    if status_callback:
        status_callback("Calling Qwen3-235B...")
    prompt  = _build_prompt(signal_data, smc_levels, headlines, memory)
    result  = _call_qwen(prompt)

    if "error" not in result:
        run_entry = {
            "date":            signal_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            "signal":          signal_data.get("signal", "NO SIGNAL"),
            "prob":            round(signal_data.get("prob", 0), 4),
            "outcome":         "pending",
            "world_view_note": result.get("run_note", ""),
        }
        runs = memory.get("runs", [])
        runs.append(run_entry)
        if len(runs) > MAX_MEMORY:
            runs = runs[-MAX_MEMORY:]
        memory["runs"]       = runs
        memory["world_view"] = result.get("world_view_update", memory.get("world_view", ""))
        memory["last_run"]   = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_memory(memory)

    return result, headlines


def update_outcome(date_str, outcome):
    memory = load_memory()
    for run in memory.get("runs", []):
        if run["date"] == date_str and run["outcome"] == "pending":
            run["outcome"] = outcome
            break
    save_memory(memory)