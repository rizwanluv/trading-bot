"""
Telegram trading assistant powered by Delta Exchange and Gemini.

Features:
- Live price tracking (/price)
- Automatic level analysis: Pivots, Fibs, S/R zones, Trend bias (/levels, /analysis)
- Custom price alerts with auto-detection & background monitoring (/alert, /alerts, /delalert, /clearalerts)
- Multi-timeframe candle entry scanner: 1m, 5m, 15m candle setups with Entry, SL, TP, R:R (/entry, /scan)
- Automated candle entry alerts: background scanning of 1m, 5m, 15m candle closes (/watch, /unwatch, /watchers)
- AI trading assistant with live market context
"""
import os
import sys
import re
import asyncio
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

# Load .env file if present
def load_env_file(filepath: str = ".env"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            print(f"Notice: Could not load .env file: {e}")

load_env_file()

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from market_data import (
    DEFAULT_SYMBOL,
    SECONDARY_SYMBOL,
    POPULAR_SYMBOLS,
    resolve_symbol,
    get_symbol_display_name,
    get_market_overview,
    get_ticker,
    get_level_analysis,
    format_level_analysis_message,
    get_candle_entry,
    get_multi_timeframe_entry,
    format_entry_analysis_message,
    get_gautam_jha_analysis,
    format_gautam_jha_message,
)
from alerts_manager import AlertManager
from delta_client import DeltaClient, DEFAULT_BASE_URL
from auto_trader import AutoTrader
from news_analysis import get_news_sentiment
from orderbook_analysis import analyze_orderbook
from amd_scalper import (
    analyze_amd_scalp,
    format_amd_scalp_html_report,
    calculate_risk_managed_plan,
)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Gemini Setup
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
SUPPORTED_GEMINI_MODELS = [
    "gemini-2.5-flash",       # Google's latest multimodal price-performance model (Default)
    "gemini-2.5-pro",         # Google's latest deep-reasoning frontier model
    "gemini-2.5-flash-lite",  # Google's ultra-fast lightweight model
    "gemini-2.0-flash",       # Stable 2.0 release
    "gemini-1.5-flash",       # Legacy fallback
]
MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()


def get_gemini_model() -> str:
    """Get active Gemini model name from environment or fallback default."""
    global MODEL
    return os.environ.get("GEMINI_MODEL", MODEL or DEFAULT_GEMINI_MODEL).strip()


def save_gemini_credentials(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    env_path: str = ".env",
) -> bool:
    """Save Gemini API Key and Model to .env file and update current process environment."""
    global MODEL
    if api_key:
        clean_key = api_key.strip()
        os.environ["GEMINI_API_KEY"] = clean_key
    if model:
        clean_model = model.strip()
        os.environ["GEMINI_MODEL"] = clean_model
        MODEL = clean_model

    lines = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read existing {env_path}: {e}")

    keys_set = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if api_key and stripped.startswith("GEMINI_API_KEY="):
            new_lines.append(f"GEMINI_API_KEY={api_key.strip()}\n")
            keys_set.add("GEMINI_API_KEY")
        elif model and stripped.startswith("GEMINI_MODEL="):
            new_lines.append(f"GEMINI_MODEL={model.strip()}\n")
            keys_set.add("GEMINI_MODEL")
        else:
            new_lines.append(line)

    if api_key and "GEMINI_API_KEY" not in keys_set:
        new_lines.append(f"GEMINI_API_KEY={api_key.strip()}\n")
    if model and "GEMINI_MODEL" not in keys_set:
        new_lines.append(f"GEMINI_MODEL={model.strip()}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except Exception as e:
        logger.error(f"Failed to write Gemini credentials to {env_path}: {e}")
        return False


SYSTEM = """You are an expert price-action trader following Gautam Jha style strictly.

Core Principles:
- Always do top-down analysis starting from the Daily timeframe.
- Daily Open (DO) is a high-probability liquidity level (many algos reverse when daily candle flips color).
- Prefer clear trending moves. Avoid FOMO and choppy markets.
- Entries and exits are based on candle behavior and key liquidity levels, not lagging indicators.
- Liquidity is taken at obvious highs/lows and previous day levels (PDH / PDL). The reaction after the grab is more important than the grab itself!

When analyzing a chart or replying, always structure your analysis clearly:

**Chart Context**
- Instrument + Timeframe
- Current price location relative to Daily Open, Previous Day High (PDH), Previous Day Low (PDL)

**Key Liquidity Levels**
- Daily Open (DO)
- Previous Day High (PDH) / Previous Day Low (PDL)
- Obvious liquidity pools / equal highs/lows

**Market Structure**
- Clear Trend / Pullback / Range / Choppy
- Is the daily candle currently green or red?

**Trade Idea(s)** (Give maximum 1-2 high quality ideas only)
Use one of these three styles:
1. **Break-and-Go (Momentum)**: For strong trends. Enter when a new candle breaks the high/low of previous candle in trend direction.
2. **Retrace-to-Level**: Wait for price to pull back to Daily Open or key liquidity level. Enter when a candle in trend direction forms at that level.
3. **Level Reversal / Continuation**: Price returns to previous strong level or sweeps PDH/PDL. Look for rejection or continuation candle.

For every trade idea clearly write:
- Direction (Long / Short)
- Exact Entry condition
- Stop Loss placement
- Target / Exit rule (TP1 & TP2 with Risk:Reward)
- Why this setup is high probability

**Risk Reminder**
Educational purpose only. Not financial advice. Always manage risk.

Important Rules:
- If market is choppy or ranging → clearly state "No high-probability setup, better to wait"
- Never invent levels that are not visible
- Prefer quality over quantity
- Keep replies concise, structured, and mobile-friendly."""

# ====================== CATEGORIZED 18-AGENT SYSTEM ======================
SYSTEM_PROMPT_18_AGENTS = """
You are an elite institutional trading desk with 18 specialized AI agents organized in clear categories.
Be strict and focus on high accuracy. Prefer "No Trade" when confluence is weak.

### Agent Categories:

**1. Price Action Core**
- Gautam Jha Analyst: Daily Open (DO), candle behavior, Break-and-Go, Retrace-to-Level, Level Reversal
- Order Block / Supply-Demand Agent: Fresh Order Blocks and strong Supply/Demand zones
- Fair Value Gap (FVG) Agent: Imbalances and Fair Value Gaps
- Breaker / Mitigation Agent: Breaker blocks and mitigation of previous orders

**2. Liquidity & Sessions**
- Liquidity & Session Specialist: Liquidity grabs, equal highs/lows, PDH/PDL, Asia/London/New York sessions
- Psychological Levels Agent: Round numbers and major psychological levels (e.g. 00, 50, 000)

**3. Market Context**
- Higher Timeframe Trend Agent: Daily & Weekly bias and structure
- Multi-Timeframe Alignment Agent: Checks if lower TF aligns with higher TF
- Correlated Markets Agent: BTC vs DXY, Nasdaq, Gold etc.
- Volatility & Range Agent: Current volatility and range condition (expansion vs compression)

**4. News & Sentiment**
- News & US News Agent: Impact of news, especially US high-impact events (CPI, NFP, FOMC)
- Market Sentiment Agent: Risk-on/Risk-off, strength of move, FOMO detection

**5. Momentum & Strength**
- Volume & Momentum Agent: Candle strength, momentum, volume characteristics
- Institutional Intent Agent: Possible institutional behavior and smart money flow

**6. Decision Layer**
- Confluence Agent: Gives Confluence Score /10 and lists strong vs weak factors
- Risk Manager: Stop quality, Risk-Reward (min 1:1.5), overall risk (can reject)
- Contrarian Agent: Challenges the main bias (Devil’s Advocate)
- Final Decision Maker: Makes the final call. Very strict.

### Response Format (Follow Exactly):

**1. Price Action Core**
- Gautam Jha Analyst: 
- Order Block / Supply-Demand: 
- Fair Value Gap (FVG): 
- Breaker / Mitigation: 

**2. Liquidity & Sessions**
- Liquidity & Session Specialist: 
- Psychological Levels: 

**3. Market Context**
- Higher Timeframe Trend: 
- Multi-Timeframe Alignment: 
- Correlated Markets: 
- Volatility & Range: 

**4. News & Sentiment**
- News & US News: 
- Market Sentiment: 

**5. Momentum & Strength**
- Volume & Momentum: 
- Institutional Intent: 

**6. Decision Layer**
- Confluence Agent:
  - Score: X/10
  - Strong Factors:
  - Weak/Missing Factors:
- Risk Manager: Approve / Caution / Reject — Reason:
- Contrarian Agent: 
- Final Decision:
  - Direction: Long / Short / No Trade
  - Setup Name:
  - Entry Condition:
  - Stop Loss:
  - Target:
  - Confidence: High / Medium / Low
  - Confluence Score: X/10
  - Main Reason:

**Risk Reminder**
Educational purpose only. Not financial advice. Manage your risk.

Strict Rules:
- Confluence Score below 7 → Prefer No Trade
- If Risk Manager rejects → Final Decision = No Trade
- Only give trade when multiple strong factors clearly align
- Be honest and professional
"""

# Try importing google.genai if available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False

# Global state
alert_manager = AlertManager("alerts_store.json")
delta_client = DeltaClient()
auto_trader = AutoTrader(store_file="autotrade_store.json", delta_client=delta_client)
histories: Dict[int, List[Dict[str, str]]] = {}


def generate_ai_reply(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """Generate response from Gemini via SDK or direct REST API with auto-fallback."""
    api_key = os.environ.get("GEMINI_API_KEY")
    active_model = get_gemini_model()
    sys_instruction = system_prompt or SYSTEM
    if not api_key:
        return (
            "⚠️ <b>GEMINI_API_KEY is not set.</b>\n\n"
            f"• <b>Active Model:</b> <code>{active_model}</code>\n\n"
            "You can still use all market features:\n"
            "• <code>/price [SYMBOL]</code> — Live prices\n"
            "• <code>/levels [SYMBOL]</code> — Automatic Level Analysis\n"
            "• <code>/alert [SYMBOL] &lt;PRICE&gt;</code> — Set price alerts\n"
            "• <code>/entry [SYMBOL]</code> — 1m, 5m, 15m candle entry scan\n"
            "• <code>/watch [SYMBOL]</code> — Automatic candle entry alerts\n\n"
            "To chat with AI, connect your key with <code>/setgemini &lt;API_KEY&gt;</code>."
        )

    candidate_models = [active_model] + [m for m in SUPPORTED_GEMINI_MODELS if m != active_model]

    if HAS_GENAI_SDK:
        for m_name in candidate_models:
            try:
                client = genai.Client(api_key=api_key)
                formatted_contents = [
                    types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
                    for m in messages
                ]
                reply = client.models.generate_content(
                    model=m_name,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction, max_output_tokens=800
                    ),
                )
                if reply.text:
                    if m_name != active_model:
                        save_gemini_credentials(model=m_name)
                    return reply.text
            except Exception as e:
                logger.warning(f"google-genai SDK call failed with {m_name}: {e}. Trying fallback.")

    # REST fallback
    last_err = ""
    for m_name in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {"role": m["role"], "parts": [{"text": m["text"]}]}
                    for m in messages
                ],
                "systemInstruction": {"parts": [{"text": sys_instruction}]},
                "generationConfig": {"maxOutputTokens": 800},
            }
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        if m_name != active_model:
                            save_gemini_credentials(model=m_name)
                        return parts[0].get("text", "No response text.")
                return "No response received from Gemini API."
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:100]}"
                logger.warning(f"REST call with {m_name} failed ({last_err}). Trying fallback.")
        except Exception as e:
            last_err = str(e)
            logger.warning(f"REST call error with {m_name}: {e}")

    return f"Error contacting Gemini API ({active_model}): {last_err}"


def generate_ai_vision_reply(
    prompt_text: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    system_prompt: Optional[str] = None,
) -> str:
    """Analyze chart photo using Gemini Multimodal Vision following 18-Agent Institutional Desk system with auto-fallback."""
    api_key = os.environ.get("GEMINI_API_KEY")
    active_model = get_gemini_model()
    sys_instruction = system_prompt or SYSTEM_PROMPT_18_AGENTS
    if not api_key:
        return (
            "⚠️ <b>GEMINI_API_KEY is not set.</b>\n\n"
            f"• <b>Active Model:</b> <code>{active_model}</code>\n\n"
            "To enable 18-Agent chart screenshot analysis with Gautam Jha strategy, "
            "set your key via <code>/setgemini &lt;API_KEY&gt;</code>."
        )

    candidate_models = [active_model] + [m for m in SUPPORTED_GEMINI_MODELS if m != active_model]

    if HAS_GENAI_SDK:
        for m_name in candidate_models:
            try:
                client = genai.Client(api_key=api_key)
                img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                text_part = types.Part.from_text(text=prompt_text)
                reply = client.models.generate_content(
                    model=m_name,
                    contents=[types.Content(parts=[text_part, img_part])],
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        max_output_tokens=1500,
                    ),
                )
                if reply.text:
                    if m_name != active_model:
                        save_gemini_credentials(model=m_name)
                    return reply.text
            except Exception as e:
                logger.warning(f"google-genai vision call with {m_name} failed: {e}. Trying fallback.")

    # REST fallback
    import base64
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    last_err = ""
    for m_name in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt_text},
                            {"inline_data": {"mime_type": mime_type, "data": b64_img}},
                        ]
                    }
                ],
                "systemInstruction": {"parts": [{"text": sys_instruction}]},
                "generationConfig": {"maxOutputTokens": 1500},
            }
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        if m_name != active_model:
                            save_gemini_credentials(model=m_name)
                        return parts[0].get("text", "No response text.")
                return "No response received from Gemini Vision API."
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:100]}"
                logger.warning(f"REST vision call with {m_name} failed ({last_err}). Trying fallback.")
        except Exception as e:
            last_err = str(e)
            logger.warning(f"REST vision error with {m_name}: {e}")

    return f"Error contacting Gemini Vision API ({active_model}): {last_err}"


def generate_compact_ai_insight(prompt_text: str, max_tokens: int = 250) -> str:
    """
    Generate ultra-concise AI strategic insights strictly capping tokens to minimize API consumption.
    Consumes minimal tokens (<100 input prompt) and caps output at max_tokens (default 250).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    active_model = get_gemini_model()
    if not api_key:
        return "⚠️ <b>GEMINI_API_KEY is not set.</b> Connect your key with <code>/setgemini &lt;KEY&gt;</code> to enable AI insights."

    candidate_models = [active_model] + [m for m in SUPPORTED_GEMINI_MODELS if m != active_model]
    system_instruction = (
        "You are an elite quantitative trading researcher. "
        "Provide ultra-concise, sharp (<70 words total) bulleted insights. No filler words."
    )

    if HAS_GENAI_SDK:
        for m_name in candidate_models:
            try:
                client = genai.Client(api_key=api_key)
                reply = client.models.generate_content(
                    model=m_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=max_tokens,
                    ),
                )
                if reply.text:
                    if m_name != active_model:
                        save_gemini_credentials(model=m_name)
                    return reply.text
            except Exception as e:
                logger.warning(f"SDK compact insight failed with {m_name}: {e}")

    # REST fallback
    for m_name in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        if m_name != active_model:
                            save_gemini_credentials(model=m_name)
                        return parts[0].get("text", "")
        except Exception as e:
            logger.warning(f"REST compact insight failed with {m_name}: {e}")

    return "⚠️ Unable to contact Gemini API for AI insight."


# ==================== 18-Agent Institutional Desk & Hub Engines ====================

def generate_18_agents_analysis(symbol: str, timeframe: str = "15m") -> str:
    """
    Generate complete 18-Agent Categorized Institutional Desk Analysis.
    Combines Price Action Core, Liquidity & Sessions, Market Context, News & Sentiment,
    Momentum & Strength, and Decision Layer.
    Uses Google Gemini multimodal/reasoning model when API key is set,
    with an instantaneous deterministic institutional synthesis fallback (0 tokens).
    """
    sym = resolve_symbol(symbol)
    tf = timeframe.lower() if timeframe.lower() in ("1m", "5m", "15m", "30m", "1h") else "15m"

    try:
        ticker = get_ticker(sym)
        mark = ticker["mark_price"] or ticker["close"]
        chg = mark - ticker["open"] if ticker["open"] > 0 else 0.0
        chg_pct = (chg / ticker["open"] * 100.0) if ticker["open"] > 0 else 0.0
    except Exception:
        ticker = {"mark_price": 0.0, "close": 0.0, "high": 0.0, "low": 0.0, "volume": 0.0, "open": 0.0}
        mark, chg, chg_pct = 0.0, 0.0, 0.0

    try:
        levels = get_level_analysis(sym)
        pivots = levels.get("pivots", {})
        pivot = pivots.get("pivot", mark)
        r1 = pivots.get("r1", mark * 1.01)
        s1 = pivots.get("s1", mark * 0.99)
        trend_bias = levels.get("trend_bias", "NEUTRAL")
    except Exception:
        pivot, r1, s1, trend_bias = mark, mark * 1.01, mark * 0.99, "NEUTRAL"

    try:
        gj_analysis = get_gautam_jha_analysis(sym)
        gj = gj_analysis.get("gautam_jha", {})
        do_val = gj.get("daily_open", mark)
        color = gj.get("daily_candle_color", "Neutral")
        pdh_val = gj.get("pdh", mark * 1.02)
        pdl_val = gj.get("pdl", mark * 0.98)
        pdh_swept = gj.get("pdh_swept", False)
        pdl_swept = gj.get("pdl_swept", False)
        struct = gj_analysis.get("market_structure", "Normal Range")
    except Exception:
        do_val, color, pdh_val, pdl_val, pdh_swept, pdl_swept, struct = mark, "Neutral", mark * 1.02, mark * 0.98, False, False, "Normal Range"

    try:
        multi = get_multi_timeframe_entry(sym, ["1m", "5m", "15m"])
        tfs_data = multi.get("timeframes", {})
        c1m = tfs_data.get("1m", {})
        c5m = tfs_data.get("5m", {})
        c15m = tfs_data.get("15m", {})
    except Exception:
        c1m, c5m, c15m = {}, {}, {}

    try:
        ob = analyze_orderbook(sym)
        imb_ratio = ob.get("imbalance_ratio", 1.0)
        imb_bias = ob.get("imbalance_bias", "NEUTRAL")
        micro_price = ob.get("micro_price", mark)
        buy_wall = ob.get("top_bid_wall", {}).get("price", mark * 0.99)
        sell_wall = ob.get("top_ask_wall", {}).get("price", mark * 1.01)
    except Exception:
        imb_ratio, imb_bias, micro_price, buy_wall, sell_wall = 1.0, "NEUTRAL", mark, mark * 0.99, mark * 1.01

    try:
        news = get_news_sentiment(sym)
        news_score = news.get("sentiment_score", 0.0)
        news_label = news.get("sentiment_label", "NEUTRAL")
        headlines = [h.get("title", "") for h in news.get("headlines", [])[:3]]
    except Exception:
        news_score, news_label, headlines = 0.0, "NEUTRAL", []

    try:
        conf = auto_trader.confluence_engine.calculate_confluence(sym)
        conf_score = conf.get("confluence_score", 50)
        conf_sig = conf.get("bias_signal", "NEUTRAL")
    except Exception:
        conf_score, conf_sig = 50, "NEUTRAL"

    # AI query if Gemini key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        prompt_lines = [
            f"Live Institutional Market Data for contract #{sym} ({tf}):",
            f"- Price: ${mark:,.2f} (24h: {chg_pct:+.2f}%) | High: ${ticker.get('high', 0):,.2f} | Low: ${ticker.get('low', 0):,.2f}",
            f"- Gautam Jha Daily Open: ${do_val:,.2f} (Daily Candle: {color})",
            f"- PDH: ${pdh_val:,.2f} (Swept: {pdh_swept}) | PDL: ${pdl_val:,.2f} (Swept: {pdl_swept})",
            f"- Market Structure: {struct} | Higher TF Bias: {trend_bias}",
            f"- Key Levels: Pivot ${pivot:,.2f}, Resistance ${r1:,.2f}, Support ${s1:,.2f}",
            f"- 1m Candle: {c1m.get('signal', 'NEUTRAL')} ({c1m.get('pattern', 'None')}, RSI {c1m.get('rsi')})",
            f"- 5m Candle: {c5m.get('signal', 'NEUTRAL')} ({c5m.get('pattern', 'None')}, RSI {c5m.get('rsi')})",
            f"- 15m Candle: {c15m.get('signal', 'NEUTRAL')} ({c15m.get('pattern', 'None')}, RSI {c15m.get('rsi')})",
            f"- Delta L2 Order Book: Imbalance Ratio {imb_ratio:.2f} ({imb_bias}), Micro-Price ${micro_price:,.2f}, Buy Wall ${buy_wall:,.2f}, Sell Wall ${sell_wall:,.2f}",
            f"- News Sentiment: {news_score:+.2f} ({news_label}), Recent Headlines: {'; '.join(headlines)}",
            f"- Confluence Engine Score: {conf_score}% {conf_sig}",
            "",
            "Conduct full 18-Agent Institutional Desk Analysis following the exact 6-category response format.",
            "Strict Rules: Score < 7 means No Trade; If Risk Manager rejects -> No Trade.",
        ]
        user_prompt = "\n".join(prompt_lines)
        ai_resp = generate_ai_reply([{"role": "user", "text": user_prompt}], system_prompt=SYSTEM_PROMPT_18_AGENTS)
        if ai_resp and not ai_resp.startswith("⚠️ <b>GEMINI_API_KEY is not set") and not ai_resp.startswith("Error contacting"):
            return ai_resp

    # Deterministic 18-Agent Institutional Report (0 tokens / offline)
    score_10 = round(conf_score / 10.0)
    score_10 = max(1, min(10, score_10))

    strong_factors = []
    weak_factors = []

    if color in ("Green", "Red"):
        strong_factors.append(f"Gautam Jha Daily Open color alignment ({color})")
    if pdh_swept or pdl_swept:
        strong_factors.append("Liquidity sweep confirmed on previous day extremes")
    if imb_bias != "NEUTRAL":
        strong_factors.append(f"L2 Order Book volume imbalance ({imb_bias})")
    else:
        weak_factors.append("Order book bid/ask volume neutral")

    if abs(news_score) >= 0.15:
        strong_factors.append(f"Macro news sentiment backing direction ({news_label})")
    else:
        weak_factors.append("Lack of strong directional news catalyst")

    if c15m.get("signal") == conf_sig:
        strong_factors.append("15m candle pattern confirms macro bias")
    else:
        weak_factors.append("Multi-timeframe candle divergence")

    if not strong_factors:
        strong_factors.append("None significant")
    if not weak_factors:
        weak_factors.append("None significant")

    # Risk Manager Decision
    sl_dist = abs(mark - s1) if conf_sig == "BUY" else abs(mark - r1)
    if sl_dist == 0 or sl_dist > (mark * 0.05):
        sl_dist = mark * 0.015

    sl_price = (mark - sl_dist) if conf_sig == "BUY" else (mark + sl_dist)
    tp1_price = (mark + sl_dist * 1.5) if conf_sig == "BUY" else (mark - sl_dist * 1.5)
    tp2_price = (mark + sl_dist * 2.5) if conf_sig == "BUY" else (mark - sl_dist * 2.5)

    if score_10 >= 7 and conf_sig in ("BUY", "SELL"):
        rm_status = "Approve"
        rm_reason = f"Favorable Risk:Reward (1:1.5 to 1:2.5+), clean invalidation at ${sl_price:,.2f}."
        final_dir = "Long" if conf_sig == "BUY" else "Short"
        setup_name = f"GJ_{'Break_And_Go' if pdh_swept or pdl_swept else 'DO_Continuation'}"
        entry_cond = f"Enter on candle confirmation above ${mark:,.2f}" if conf_sig == "BUY" else f"Enter on candle confirmation below ${mark:,.2f}"
        confidence = "High" if score_10 >= 8 else "Medium"
        main_reason = f"Confluence score of {score_10}/10 with {len(strong_factors)} institutional factors aligning."
    elif score_10 >= 6 and conf_sig in ("BUY", "SELL"):
        rm_status = "Caution"
        rm_reason = "Marginal confluence score; smaller size recommended."
        final_dir = "No Trade"
        setup_name = "Wait for Confirmation"
        entry_cond = "Wait for clear retest or 15m breakout candle"
        confidence = "Low"
        main_reason = f"Confluence score {score_10}/10 is below strict institutional threshold (7/10)."
    else:
        rm_status = "Reject"
        rm_reason = "Weak confluence or choppy structure; risk of whipsaw."
        final_dir = "No Trade"
        setup_name = "Chop / Range Wait"
        entry_cond = "No entry; wait for liquidity sweep or Daily Open flip"
        confidence = "Low"
        main_reason = "Confluence below threshold and Risk Manager rejected setup."

    psych_round = round(mark / 100.0) * 100.0 if "BTC" in sym else round(mark / 10.0) * 10.0
    contrarian_bias = "Bearish rejection risk if price retests higher resistance" if conf_sig == "BUY" else "Bullish short-squeeze risk if sellers fail to break support"

    report = (
        f"🏛️ **18-Agent Institutional Desk Analysis**\n"
        f"Instrument: `#{sym}` | Timeframe: `{tf.upper()}` | Mark: `${mark:,.2f}`\n\n"
        f"**1. Price Action Core**\n"
        f"- Gautam Jha Analyst: Daily Open at ${do_val:,.2f} ({color} candle). Bias is {conf_sig}.\n"
        f"- Order Block / Supply-Demand: Nearest Supply at ${r1:,.2f}, Demand at ${s1:,.2f}.\n"
        f"- Fair Value Gap (FVG): Imbalance zone between ${s1:,.2f} and ${r1:,.2f}.\n"
        f"- Breaker / Mitigation: Mitigation level anchored near Daily Pivot ${pivot:,.2f}.\n\n"
        f"**2. Liquidity & Sessions**\n"
        f"- Liquidity & Session Specialist: PDH ${pdh_val:,.2f} ({'Swept 🎯' if pdh_swept else 'Intact'}), PDL ${pdl_val:,.2f} ({'Swept 🎯' if pdl_swept else 'Intact'}).\n"
        f"- Psychological Levels: Major psychological round number at `${psych_round:,.0f}`.\n\n"
        f"**3. Market Context**\n"
        f"- Higher Timeframe Trend: {trend_bias} trend bias on Daily structure.\n"
        f"- Multi-Timeframe Alignment: 1m ({c1m.get('signal', 'NEUTRAL')}), 5m ({c5m.get('signal', 'NEUTRAL')}), 15m ({c15m.get('signal', 'NEUTRAL')}).\n"
        f"- Correlated Markets: Risk sentiment alignment tracking DXY and macro flows.\n"
        f"- Volatility & Range: 24h High ${ticker.get('high', 0):,.2f} | Low ${ticker.get('low', 0):,.2f}.\n\n"
        f"**4. News & Sentiment**\n"
        f"- News & US News: Financial sentiment score {news_score:+.2f} ({news_label}).\n"
        f"- Market Sentiment: {news_label} sentiment environment; watching macro releases.\n\n"
        f"**5. Momentum & Strength**\n"
        f"- Volume & Momentum: 15m RSI={c15m.get('rsi', 50.0)}, candle pattern: {c15m.get('pattern', 'Standard')}.\n"
        f"- Institutional Intent: L2 Order Book Imbalance ratio {imb_ratio:.2f} ({imb_bias}).\n\n"
        f"**6. Decision Layer**\n"
        f"- Confluence Agent:\n"
        f"  - Score: {score_10}/10\n"
        f"  - Strong Factors: {'; '.join(strong_factors)}\n"
        f"  - Weak/Missing Factors: {'; '.join(weak_factors)}\n"
        f"- Risk Manager: {rm_status} — Reason: {rm_reason}\n"
        f"- Contrarian Agent: {contrarian_bias}.\n"
        f"- Final Decision:\n"
        f"  - Direction: {final_dir}\n"
        f"  - Setup Name: {setup_name}\n"
        f"  - Entry Condition: {entry_cond}\n"
        f"  - Stop Loss: ${sl_price:,.2f}\n"
        f"  - Target: ${tp1_price:,.2f} (TP1 1:1.5) | ${tp2_price:,.2f} (TP2 1:2.5+)\n"
        f"  - Confidence: {confidence}\n"
        f"  - Confluence Score: {score_10}/10\n"
        f"  - Main Reason: {main_reason}\n\n"
        f"**Risk Reminder**\n"
        f"Educational purpose only. Not financial advice. Manage your risk.\n"
    )
    return report


def format_symbol_hub_overview(symbol: str) -> str:
    """Format all-in-one comprehensive hub snapshot for BTC or Gold combining all market dimensions."""
    sym = resolve_symbol(symbol)
    name = get_symbol_display_name(sym)
    cmd_prefix = "/gold" if ("XAU" in sym or "GOLD" in sym) else "/btc"

    try:
        t = get_ticker(sym)
        mark = t["mark_price"] or t["close"]
        chg = mark - t["open"] if t["open"] > 0 else 0.0
        chg_pct = (chg / t["open"] * 100.0) if t["open"] > 0 else 0.0
        sign = "+" if chg >= 0 else ""
        color_emoji = "🟢" if chg >= 0 else "🔴"
    except Exception:
        mark, chg, chg_pct, sign, color_emoji = 0.0, 0.0, 0.0, "", "⚪"
        t = {"high": 0.0, "low": 0.0, "volume": 0.0}

    # Liquidity & GJ
    try:
        gj_analysis = get_gautam_jha_analysis(sym)
        gj = gj_analysis.get("gautam_jha", {})
        do_val = gj.get("daily_open", 0.0)
        do_col = gj.get("daily_candle_color", "Neutral")
        pdh_val = gj.get("pdh", 0.0)
        pdl_val = gj.get("pdl", 0.0)
        pdh_s = "Swept 🎯" if gj.get("pdh_swept") else "Intact"
        pdl_s = "Swept 🎯" if gj.get("pdl_swept") else "Intact"
        struct = gj_analysis.get("market_structure", "Normal")
    except Exception:
        do_val, do_col, pdh_val, pdl_val, pdh_s, pdl_s, struct = 0.0, "Neutral", 0.0, 0.0, "N/A", "N/A", "N/A"

    # Order book
    try:
        ob = analyze_orderbook(sym)
        ob_ratio = ob.get("imbalance_ratio", 1.0)
        ob_bias = ob.get("imbalance_bias", "NEUTRAL")
        buy_wall = ob.get("top_bid_wall", {}).get("price", 0.0)
        ob_str = f"Ratio <code>{ob_ratio:.2f} ({ob_bias})</code> | Wall: <code>${buy_wall:,.2f}</code>"
    except Exception:
        ob_str = "Available via Delta L2"

    # News
    try:
        news = get_news_sentiment(sym)
        news_score = news.get("sentiment_score", 0.0)
        news_label = news.get("sentiment_label", "NEUTRAL")
        news_str = f"<code>{news_score:+.2f} ({news_label})</code>"
    except Exception:
        news_str = "Neutral (0.00)"

    # Confluence
    try:
        conf = auto_trader.confluence_engine.calculate_confluence(sym)
        conf_score = conf.get("confluence_score", 50)
        conf_sig = conf.get("bias_signal", "NEUTRAL")
        conf_cat = conf.get("bias_category", "NEUTRAL")
        conf_str = f"<b>{conf_score}% {conf_sig}</b> ({conf_cat})"
    except Exception:
        conf_str = "50% NEUTRAL"

    header_icon = "🥇" if ("XAU" in sym or "GOLD" in sym) else "🪙"

    msg = (
        f"{header_icon} <b>{name.upper()} (<code>#{sym}</code>) ALL-IN-ONE HUB</b>\n\n"
        f"💵 <b>Price:</b> <code>${mark:,.2f}</code> | 24h: {color_emoji} <code>{sign}{chg_pct:.2f}%</code>\n"
        f"📊 <b>24h Range:</b> High <code>${t.get('high', 0):,.2f}</code> | Low <code>${t.get('low', 0):,.2f}</code>\n"
        f"🚪 <b>Daily Open:</b> <code>${do_val:,.2f}</code> ({do_col}) | Structure: <b>{struct}</b>\n"
        f"🎯 <b>Liquidity:</b> PDH <code>${pdh_val:,.2f}</code> ({pdh_s}) | PDL <code>${pdl_val:,.2f}</code> ({pdl_s})\n"
        f"📦 <b>Order Book:</b> {ob_str}\n"
        f"📰 <b>News Sentiment:</b> {news_str}\n"
        f"🎯 <b>Master Confluence:</b> {conf_str}\n\n"
        f"⚡ <b>1-Command Working Modes:</b>\n"
        f"• <code>{cmd_prefix} price</code> — Live ticker & volume\n"
        f"• <code>{cmd_prefix} levels</code> — Automatic S/R, Pivots & Fibs\n"
        f"• <code>{cmd_prefix} gj</code> — Gautam Jha liquidity & sweeps\n"
        f"• <code>{cmd_prefix} entry</code> — 1m, 5m, 15m candle setups\n"
        f"• <code>{cmd_prefix} watch</code> — Automatic candle entry alerts\n"
        f"• <code>{cmd_prefix} book</code> — Live L2 depth, imbalance & walls\n"
        f"• <code>{cmd_prefix} news</code> — Breaking news & sentiment score\n"
        f"• <code>{cmd_prefix} confluence</code> — Master 5-strategy score & plan\n"
        f"• <code>{cmd_prefix} analyze</code> — 18-Agent Institutional Desk\n"
        f"• <code>{cmd_prefix} buy [sz]</code> — Execute Buy order with auto SL/TP\n"
        f"• <code>{cmd_prefix} sell [sz]</code> — Execute Sell order with auto SL/TP"
    )
    return msg


# ==================== Command Handlers ====================

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Start command intro."""
    msg = (
        "🤖 <b>Welcome to Gemini Trading Assistant & Auto-Trade Bot!</b>\n\n"
        "Your intelligent institutional algorithmic assistant powered by Delta Exchange live data, "
        "<b>18-Agent Institutional Desk Analysis</b>, automatic level analysis, price alerts, "
        "multi-timeframe candle scanner, <b>Gautam Jha Liquidity strategy</b>, and "
        "<b>Automated Order Execution</b> for <b>🥇 Gold (XAU/USD)</b> and <b>🪙 Bitcoin (BTC/USD)</b>.\n\n"
        "🏛️ <b>6 Master Unified Command Hubs (1 Command, Multiple Working Types):</b>\n"
        "• <code>/btc [action]</code> — 11-in-1 Bitcoin Hub (price, levels, gj, entry, watch, book, news, confluence, analyze, trade)\n"
        "• <code>/gold [action]</code> — 11-in-1 Gold Hub (price, levels, gj, entry, watch, book, news, confluence, analyze, trade)\n"
        "• <code>/trade [action]</code> — Master Trading Hub (dashboard, on, off, live, paper, pos, close, bal, manual trade)\n"
        "• <code>/alert [action]</code> — Master Alerts Hub (on, off, list, del, clear, price alert, watch)\n"
        "• <code>/analyze [sym]</code> — 18-Agent Categorized Institutional Desk Analysis\n"
        "• <code>/keys [action]</code> — API & Bot Config Hub (check, set, base, gemini, model)\n"
        "• 📸 <b>Send Chart Screenshot:</b> 18-Agent Institutional Multimodal Vision Analysis!\n\n"
        "Type <code>/list</code> to view all commands or <code>/help</code> for detailed guides."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def list_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show all bot commands in a quick, clean reference list featuring unified hubs."""
    msg = (
        "📜 <b>UNIFIED MULTI-WORKING COMMAND HUBS:</b>\n"
        "<i>(Minimum Commands — Maximum Working Types!)</i>\n\n"
        "🏛️ <b>0. Master Institutional Dashboard (/status or /dashboard):</b>\n"
        "• <code>/status</code> (or <code>/dashboard</code>) — Single-card institutional command center\n"
        "  └ Real-time bot state, live market session killzone, open positions with live PnL, risk rules, self-learning health, and API connectivity in 1 view!\n\n"
        "🪙 <b>1. Bitcoin All-in-One Hub (/btc):</b>\n"
        "• <code>/btc</code> — Comprehensive all-in-one Bitcoin card\n"
        "• <code>/btc price</code> — Live ticker & volume\n"
        "• <code>/btc levels</code> — Pivots, S/R & Fibs\n"
        "• <code>/btc gj</code> — Gautam Jha liquidity & sweeps\n"
        "• <code>/btc entry</code> — 1m, 5m, 15m candle setups\n"
        "• <code>/btc watch</code> — Auto candle entry alerts\n"
        "• <code>/btc book</code> — L2 order book depth & walls\n"
        "• <code>/btc news</code> — Breaking news & sentiment score\n"
        "• <code>/btc confluence</code> — Master 5-strategy score & plan\n"
        "• <code>/btc analyze</code> — 18-Agent Institutional Desk\n"
        "• <code>/btc buy [sz]</code> | <code>/btc sell [sz]</code> — Execute order with auto SL/TP\n\n"
        "🥇 <b>2. Gold All-in-One Hub (/gold or /xau):</b>\n"
        "• <code>/gold</code> — Comprehensive all-in-one Gold card\n"
        "• <code>/gold price</code> | <code>/gold levels</code> | <code>/gold gj</code> | <code>/gold entry</code>\n"
        "• <code>/gold watch</code> | <code>/gold book</code> | <code>/gold news</code> | <code>/gold confluence</code>\n"
        "• <code>/gold analyze</code> — 18-Agent Institutional Desk\n"
        "• <code>/gold buy [sz]</code> | <code>/gold sell [sz]</code> — Execute order with auto SL/TP\n\n"
        "💼 <b>3. Master Trading & Portfolio Hub (/trade):</b>\n"
        "• <code>/trade</code> — Portfolio & bot dashboard (balance, mode, win rate)\n"
        "• <code>/trade status</code> — Detailed single-screen institutional command center 🏛️\n"
        "• <code>/trade on</code> (or <code>start</code>) — START automated trading bot 🟢\n"
        "• <code>/trade off</code> (or <code>stop</code>) — STOP / pause automated trading 🔴\n"
        "• <code>/trade be [on|off]</code> — Toggle Breakeven Stop-Loss on TP1 (Risk-Free Trades) 🛡️\n"
        "• <code>/trade trail [on|off|pct]</code> — Toggle Dynamic Trailing Stop-Loss ⚡\n"
        "• <code>/trade size &lt;VAL&gt;</code> (or <code>/size &lt;VAL&gt;</code>) — Set Auto-Trade Lot Size (e.g. <code>/trade size 0.05</code>) 🎯\n"
        "• <code>/trade size &lt;SYM&gt; &lt;VAL&gt;</code> — Set Pair Override (e.g. <code>/trade size BTC 0.01</code>)\n"
        "• <code>/trade live</code> | <code>/trade paper</code> — Switch execution mode\n"
        "• <code>/trade pos</code> — View active open positions & live PnL\n"
        "• <code>/trade close [id|all]</code> — Close position(s) at market\n"
        "• <code>/trade bal</code> — Wallet & account balances\n"
        "• <code>/trade amd [sym]</code> — Execute/analyze 1m/5m/15m AMD Scalp\n"
        "• <code>/trade maxpos &lt;N&gt;</code> — Set max concurrent trades (e.g. 5)\n"
        "• <code>/trade risk &lt;PCT&gt;</code> — Set capital risk per trade (e.g. 1.5%)\n"
        "• <code>/trade multi [on|off]</code> — Toggle multi-trade per symbol\n"
        "• <code>/trade confluence &lt;sym&gt;</code> — Execute master confluence trade\n"
        "• <code>/trade &lt;sym&gt; &lt;buy|sell&gt; [sz]</code> — Manual trade execution\n"
        "• <code>/trade learn</code> — Self-learning performance & insights\n\n"
        "🔔 <b>4. Master Alerts Hub (/alert):</b>\n"
        "• <code>/alert</code> — Alerts overview & active list\n"
        "• <code>/alert on</code> — Turn ON automatic market alerts for BTC & Gold 🟢\n"
        "• <code>/alert off</code> — Turn OFF automatic alerts 🔴\n"
        "• <code>/alert list</code> — List your active price alerts\n"
        "• <code>/alert &lt;sym&gt; &lt;price&gt;</code> — Set price alert (e.g. <code>/alert btc 85000</code>)\n"
        "• <code>/alert &lt;price&gt;</code> — Set price alert (auto-detects symbol)\n"
        "• <code>/alert del &lt;ID&gt;</code> — Remove an alert by ID\n"
        "• <code>/alert clear</code> — Clear all price alerts\n"
        "• <code>/alert watch &lt;sym&gt;</code> — Watch candle closes\n\n"
        "⚡ <b>5. AMD Scalp Trading Desk (/amd or /scalp):</b>\n"
        "• <code>/amd [symbol]</code> — 1m, 5m, 15m Multi-Timeframe Institutional Scalper\n"
        "• Accumulation range detection + Judas Swing liquidity sweep + 1m MSS trigger\n"
        "• Automated Stop-Loss, Take-Profit (TP1/TP2), and 1.5% capital risk management\n"
        "• <code>/amd trade [symbol]</code> — Instant auto-scalp execution\n\n"
        "🏛️ <b>6. 18-Agent Institutional Desk (/analyze):</b>\n"
        "• <code>/analyze [symbol] [tf]</code> (or <code>/analysis</code>) — Deep 18-agent categorized report\n"
        "• Categories: Price Action Core, Liquidity & Sessions, Market Context, News & Sentiment, Momentum & Strength, Decision Layer\n\n"
        "🔐 <b>7. Master Keys & Bot Config (/keys):</b>\n"
        "• <code>/keys</code> — Connection status & balances overview\n"
        "• <code>/keys check</code> — Test live Delta Exchange connection\n"
        "• <code>/keys set &lt;KEY&gt; &lt;SECRET&gt;</code> — Connect Delta API keys\n"
        "• <code>/keys base [india|global]</code> — Switch Delta endpoint\n"
        "• <code>/keys gemini &lt;KEY&gt;</code> — Set Google Gemini API Key\n"
        "• <code>/keys model [flash|pro|lite]</code> — Switch Gemini AI Model\n\n"
        "📸 <b>Chart Photo Analysis:</b>\n"
        "• <i>Send Chart Photo</i> — Instant 18-Agent Multimodal Vision Analysis\n\n"
        "💡 <b>Direct Shortcuts & Aliases:</b>\n"
        "• <code>/status</code>, <code>/dashboard</code>, <code>/dash</code>\n"
        "• <code>/btc</code>, <code>/btclevels</code>, <code>/btcgj</code>, <code>/btcentry</code>, <code>/btcwatch</code>, <code>/btcamd</code>\n"
        "• <code>/gold</code>, <code>/xau</code>, <code>/xauusd</code>, <code>/goldlevels</code>, <code>/goldgj</code>, <code>/goldentry</code>, <code>/goldwatch</code>, <code>/goldamd</code>\n"
        "• <code>/amd</code>, <code>/scalp</code>, <code>/multitrade</code>\n"
        "• <code>/price</code>, <code>/levels</code>, <code>/analysis</code>, <code>/gj</code>, <code>/liquidity</code>\n"
        "• <code>/alert</code>, <code>/alerts</code>, <code>/delalert</code>, <code>/clearalerts</code>\n"
        "• <code>/entry</code>, <code>/scan</code>, <code>/watch</code>, <code>/unwatch</code>, <code>/watchers</code>\n"
        "• <code>/autotrade</code>, <code>/starttrade</code>, <code>/stoptrade</code>, <code>/trade</code>, <code>/positions</code>, <code>/closeposition</code>, <code>/balance</code>, <code>/mode</code>\n"
        "• <code>/alertson</code>, <code>/alertsoff</code>, <code>/setkey</code>, <code>/setsecret</code>, <code>/setkeys</code>, <code>/keys</code>\n"
        "• <code>/list</code>, <code>/help</code>, <code>/start</code>, <code>/reset</code>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Help command with usage examples."""
    msg = (
        "📖 <b>Trading Assistant & Auto-Trade Guide:</b>\n\n"
        "<b>1. Master Unified Commands (Recommended):</b>\n"
        "• <code>/btc</code> — All-in-one Bitcoin hub (run with <code>price</code>, <code>levels</code>, <code>gj</code>, <code>book</code>, <code>news</code>, <code>confluence</code>, <code>analyze</code>, <code>buy</code>, <code>sell</code>)\n"
        "• <code>/gold</code> — All-in-one Gold hub (same 11 working modes)\n"
        "• <code>/trade</code> — Trading control hub (<code>on</code>, <code>off</code>, <code>live</code>, <code>paper</code>, <code>pos</code>, <code>close</code>, <code>bal</code>, <code>learn</code>)\n"
        "• <code>/alert</code> — Alerts hub (<code>on</code>, <code>off</code>, <code>list</code>, <code>del</code>, <code>clear</code>, <code>watch</code>, or <code>&lt;price&gt;</code>)\n"
        "• <code>/analyze [sym] [tf]</code> — 18-Agent Institutional Desk analysis\n"
        "• <code>/keys</code> — Bot configuration (<code>check</code>, <code>set</code>, <code>base</code>, <code>gemini</code>, <code>model</code>)\n\n"
        "<b>2. 18-Agent Institutional Desk System:</b>\n"
        "• Sends chart or text through 18 specialized agents in 6 categories:\n"
        "  1. Price Action Core (Gautam Jha DO, Order Blocks, FVGs, Breakers)\n"
        "  2. Liquidity & Sessions (PDH/PDL sweeps, session ranges, round numbers)\n"
        "  3. Market Context (Higher TF bias, Multi-TF alignment, Correlated markets)\n"
        "  4. News & Sentiment (US high-impact news, market sentiment)\n"
        "  5. Momentum & Strength (Volume characteristics, institutional intent)\n"
        "  6. Decision Layer (Strict Confluence Score /10 & Risk Manager veto)\n\n"
        "<b>3. Automated & Manual Trading:</b>\n"
        "• <code>/trade on</code> — Start automated trading bot 🟢\n"
        "• <code>/trade off</code> — Pause automated trading bot 🔴\n"
        "• <code>/trade btc buy 1</code> — Buy Bitcoin with automatic Stop Loss & Take Profit\n"
        "• <code>/trade pos</code> — View live positions & unrealized PnL\n"
        "• <code>/trade close all</code> — Close all positions immediately\n\n"
        "<b>4. Delta Exchange API Setup:</b>\n"
        "• <code>/keys set &lt;API_KEY&gt; &lt;API_SECRET&gt;</code> — Connect Delta credentials\n"
        "• <code>/keys check</code> — Verify connection live & display wallet balance\n"
        "• <code>/keys base [india|global]</code> — Switch Delta India (.exchange) vs Global (.com)\n\n"
        "<b>5. Chart Screenshot Upload:</b>\n"
        "• Upload any chart screenshot with caption (e.g. <code>BTC 15m</code>) for instant 18-agent categorized analysis!"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def price_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Live price command: shows market overview if no args, or specific ticker if symbol given."""
    if not ctx.args:
        msg = get_market_overview()
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    symbol = resolve_symbol(ctx.args[0])
    try:
        t = get_ticker(symbol)
        mark = t["mark_price"] or t["close"]
        chg = mark - t["open"] if t["open"] > 0 else 0.0
        chg_pct = (chg / t["open"] * 100.0) if t["open"] > 0 else 0.0
        sign = "+" if chg >= 0 else ""
        color_emoji = "🟢" if chg >= 0 else "🔴"

        name = get_symbol_display_name(symbol)
        if "XAU" in symbol or "GOLD" in symbol:
            shortcuts_str = "💡 <i>Shortcuts: <code>/goldlevels</code> | <code>/goldgj</code> | <code>/goldentry</code> | <code>/goldwatch</code></i>"
        elif "BTC" in symbol:
            shortcuts_str = "💡 <i>Shortcuts: <code>/btclevels</code> | <code>/btcgj</code> | <code>/btcentry</code> | <code>/btcwatch</code></i>"
        else:
            shortcuts_str = f"💡 <i>Shortcuts: <code>/levels {symbol}</code> | <code>/gj {symbol}</code> | <code>/entry {symbol}</code></i>"

        msg = (
            f"<b>{name}</b> (<code>#{t['symbol']}</code>) <b>Live Ticker</b>\n\n"
            f"💵 <b>Price:</b> <code>${mark:,.2f}</code>\n"
            f"📊 <b>24h Change:</b> {color_emoji} <code>{sign}{chg:,.2f} ({sign}{chg_pct:.2f}%)</code>\n"
            f"🔺 <b>24h High:</b> <code>${t['high']:,.2f}</code>\n"
            f"🔻 <b>24h Low:</b> <code>${t['low']:,.2f}</code>\n"
            f"🚪 <b>24h Open:</b> <code>${t['open']:,.2f}</code>\n"
            f"📦 <b>24h Volume:</b> <code>{t['volume']:,.0f}</code>\n\n"
            f"{shortcuts_str}"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error fetching price for {symbol}: {e}")


async def levels_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Automatic level analysis command."""
    symbol = resolve_symbol(ctx.args[0]) if ctx.args else DEFAULT_SYMBOL
    try:
        analysis = get_level_analysis(symbol)
        msg = format_level_analysis_message(analysis)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error computing levels for {symbol}: {e}")


async def gj_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Dedicated Gautam Jha Liquidity Level Analysis command."""
    symbol = resolve_symbol(ctx.args[0]) if ctx.args else DEFAULT_SYMBOL
    try:
        analysis = get_gautam_jha_analysis(symbol)
        msg = format_gautam_jha_message(analysis)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error computing Gautam Jha analysis for {symbol}: {e}")


# ==================== Unified Multi-Working Bitcoin Hub ====================
async def btc_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Bitcoin (BTC) Unified Multi-Working Command Hub.
    Usage:
    • /btc — Comprehensive all-in-one market snapshot
    • /btc [price|levels|gj|entry|watch|book|news|confluence|analyze|buy|sell]
    """
    args = ctx.args or []
    if not args:
        msg = format_symbol_hub_overview("BTCUSD")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    sub = args[0].lower().strip()
    rest = args[1:]

    if sub in ("price", "p", "ticker", "quote"):
        ctx.args = ["BTCUSD"]
        await price_cmd(update, ctx)
    elif sub in ("levels", "level", "lvl", "pivots", "sr"):
        ctx.args = ["BTCUSD"]
        await levels_cmd(update, ctx)
    elif sub in ("gj", "gautam", "liquidity", "pa"):
        ctx.args = ["BTCUSD"]
        await gj_cmd(update, ctx)
    elif sub in ("entry", "entries", "scan", "candle", "candles"):
        ctx.args = ["BTCUSD"]
        await entry_cmd(update, ctx)
    elif sub in ("watch", "watcher", "watchers"):
        tfs = rest if rest else ["1m,5m,15m"]
        ctx.args = ["BTCUSD"] + tfs
        await watch_cmd(update, ctx)
    elif sub in ("unwatch", "stopwatch"):
        ctx.args = ["BTCUSD"]
        await unwatch_cmd(update, ctx)
    elif sub in ("book", "orderbook", "depth", "l2"):
        ctx.args = ["BTCUSD"]
        await orderbook_cmd(update, ctx)
    elif sub in ("news", "sentiment"):
        ctx.args = ["BTCUSD"] + rest
        await news_cmd(update, ctx)
    elif sub in ("confluence", "combine", "master"):
        ctx.args = ["BTCUSD"] + rest
        await confluence_cmd(update, ctx)
    elif sub in ("analyze", "analysis", "desk", "agents", "18"):
        ctx.args = ["BTCUSD"] + rest
        await analyze_cmd(update, ctx)
    elif sub in ("amd", "scalp", "po3"):
        ctx.args = ["BTCUSD"] + rest
        await amd_cmd(update, ctx)
    elif sub in ("buy", "long"):
        size = rest[0] if rest else None
        ctx.args = ["BTCUSD", "buy"] + ([size] if size else [])
        await trade_cmd(update, ctx)
    elif sub in ("sell", "short"):
        size = rest[0] if rest else None
        ctx.args = ["BTCUSD", "sell"] + ([size] if size else [])
        await trade_cmd(update, ctx)
    else:
        msg = format_symbol_hub_overview("BTCUSD")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def btc_levels_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /btclevels -> BTC Level Analysis."""
    ctx.args = ["BTCUSD"]
    await levels_cmd(update, ctx)


async def btc_gj_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /btcgj -> BTC Gautam Jha Liquidity."""
    ctx.args = ["BTCUSD"]
    await gj_cmd(update, ctx)


async def btc_entry_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /btcentry -> BTC 1m, 5m, 15m candle entry scan."""
    ctx.args = ["BTCUSD"]
    await entry_cmd(update, ctx)


async def btc_watch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /btcwatch -> Enable automated candle alerts for BTC."""
    tfs = ctx.args if ctx.args else ["1m,5m,15m"]
    ctx.args = ["BTCUSD"] + tfs
    await watch_cmd(update, ctx)


# ==================== Unified Multi-Working Gold Hub ====================
async def gold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Gold (XAU/USD) Unified Multi-Working Command Hub.
    Usage:
    • /gold — Comprehensive all-in-one market snapshot
    • /gold [price|levels|gj|entry|watch|book|news|confluence|analyze|buy|sell]
    """
    args = ctx.args or []
    if not args:
        msg = format_symbol_hub_overview("XAUTUSD")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    sub = args[0].lower().strip()
    rest = args[1:]

    if sub in ("price", "p", "ticker", "quote"):
        ctx.args = ["XAUTUSD"]
        await price_cmd(update, ctx)
    elif sub in ("levels", "level", "lvl", "pivots", "sr"):
        ctx.args = ["XAUTUSD"]
        await levels_cmd(update, ctx)
    elif sub in ("gj", "gautam", "liquidity", "pa"):
        ctx.args = ["XAUTUSD"]
        await gj_cmd(update, ctx)
    elif sub in ("entry", "entries", "scan", "candle", "candles"):
        ctx.args = ["XAUTUSD"]
        await entry_cmd(update, ctx)
    elif sub in ("watch", "watcher", "watchers"):
        tfs = rest if rest else ["1m,5m,15m"]
        ctx.args = ["XAUTUSD"] + tfs
        await watch_cmd(update, ctx)
    elif sub in ("unwatch", "stopwatch"):
        ctx.args = ["XAUTUSD"]
        await unwatch_cmd(update, ctx)
    elif sub in ("book", "orderbook", "depth", "l2"):
        ctx.args = ["XAUTUSD"]
        await orderbook_cmd(update, ctx)
    elif sub in ("news", "sentiment"):
        ctx.args = ["XAUTUSD"] + rest
        await news_cmd(update, ctx)
    elif sub in ("confluence", "combine", "master"):
        ctx.args = ["XAUTUSD"] + rest
        await confluence_cmd(update, ctx)
    elif sub in ("analyze", "analysis", "desk", "agents", "18"):
        ctx.args = ["XAUTUSD"] + rest
        await analyze_cmd(update, ctx)
    elif sub in ("amd", "scalp", "po3"):
        ctx.args = ["XAUTUSD"] + rest
        await amd_cmd(update, ctx)
    elif sub in ("buy", "long"):
        size = rest[0] if rest else None
        ctx.args = ["XAUTUSD", "buy"] + ([size] if size else [])
        await trade_cmd(update, ctx)
    elif sub in ("sell", "short"):
        size = rest[0] if rest else None
        ctx.args = ["XAUTUSD", "sell"] + ([size] if size else [])
        await trade_cmd(update, ctx)
    else:
        msg = format_symbol_hub_overview("XAUTUSD")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def gold_levels_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /goldlevels -> Gold Level Analysis."""
    ctx.args = ["XAUTUSD"]
    await levels_cmd(update, ctx)


async def gold_gj_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /goldgj -> Gold Gautam Jha Liquidity."""
    ctx.args = ["XAUTUSD"]
    await gj_cmd(update, ctx)


async def gold_entry_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /goldentry -> Gold 1m, 5m, 15m candle entry scan."""
    ctx.args = ["XAUTUSD"]
    await entry_cmd(update, ctx)


async def gold_watch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /goldwatch -> Enable automated candle alerts for Gold."""
    tfs = ctx.args if ctx.args else ["1m,5m,15m"]
    ctx.args = ["XAUTUSD"] + tfs
    await watch_cmd(update, ctx)


# ==================== 18-Agent Institutional Desk Command ====================
async def analyze_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    18-Agent Categorized Institutional Desk Analysis (/analyze [symbol] [timeframe]).
    Evaluates:
    1. Price Action Core (Gautam Jha DO, Order Blocks, FVGs, Breakers)
    2. Liquidity & Sessions (PDH/PDL sweeps, session range, round levels)
    3. Market Context (Higher TF structure, Multi-TF alignment, Correlated markets)
    4. News & Sentiment (Macro events, financial news sentiment)
    5. Momentum & Strength (Volume characteristics, smart money flow)
    6. Decision Layer (Strict Confluence Score & Risk Manager veto)
    """
    args = ctx.args or []
    sym_raw = args[0] if args else DEFAULT_SYMBOL
    tf = args[1].lower() if len(args) > 1 and args[1].lower() in ("1m", "5m", "15m", "30m", "1h") else "15m"
    symbol = resolve_symbol(sym_raw)

    await update.message.reply_text(
        f"🏛️ <b>18 Institutional Agents Analyzing #{symbol}...</b>\n"
        "Price Action → Liquidity → Context → News → Momentum → Decision\nPlease wait ⏳",
        parse_mode=ParseMode.HTML,
    )

    try:
        report = generate_18_agents_analysis(symbol, tf)
        if len(report) > 4000:
            parts = [report[i:i+3800] for i in range(0, len(report), 3800)]
            for p in parts:
                await update.message.reply_text(p)
        else:
            await update.message.reply_text(report)
    except Exception as e:
        logger.error(f"Error in 18-agent analysis for {symbol}: {e}")
        await update.message.reply_text(f"❌ Error during 18-agent analysis: {e}")


def _parse_alert_args(args: List[str]) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """Parse alert command arguments flexibly with smart symbol inference."""
    if not args:
        return None, None, None, (
            "Usage: <code>/alert [SYMBOL] [CONDITION] &lt;PRICE&gt;</code>\n"
            "Examples:\n"
            "• <code>/alert btc 85000</code>\n"
            "• <code>/alert gold 4180</code>\n"
            "• <code>/alert 85000</code> (auto-detects BTC)\n"
            "• <code>/alert 4180</code> (auto-detects Gold)\n"
            "• <code>/alert BTCUSD &lt;= 84500</code>"
        )

    symbol = None
    target_price = None
    condition = None

    tokens = list(args)
    for tok in list(tokens):
        if tok in (">=", ">", "above"):
            condition = ">="
            tokens.remove(tok)
        elif tok in ("<=", "<", "below"):
            condition = "<="
            tokens.remove(tok)

    for tok in list(tokens):
        clean_tok = tok.replace("$", "").replace(",", "")
        try:
            target_price = float(clean_tok)
            tokens.remove(tok)
            break
        except ValueError:
            pass

    if target_price is None:
        return None, None, None, "❌ Could not parse price. Example: <code>/alert btc 85000</code> or <code>/alert gold 4180</code>"

    if tokens:
        symbol = resolve_symbol(tokens[0])
    else:
        # Auto-detect BTC vs Gold based on price magnitude
        if target_price >= 15000:
            symbol = "BTCUSD"
        elif 1000 <= target_price < 15000:
            symbol = "XAUTUSD"
        else:
            symbol = DEFAULT_SYMBOL

    return symbol, target_price, condition, None


async def set_alert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Master Alerts Multi-Working Hub.
    Usage:
    • /alert (alerts center & active alerts list)
    • /alert on | /alert off (toggle automatic candle & liquidity alerts)
    • /alert list (list active alerts)
    • /alert del <id> | /alert clear
    • /alert watch <sym> [tfs] | /alert unwatch <sym>
    • /alert <sym> <price> (or /alert <price>)
    """
    args = ctx.args or []
    if not args:
        chat_id = update.effective_chat.id
        alerts = alert_manager.get_chat_alerts(chat_id)
        watchers = alert_manager.get_chat_entry_watchers(chat_id)
        auto_alerts_on = len(watchers) > 0

        alerts_summary = f"<code>{len(alerts)}</code> active" if alerts else "None active"
        watchers_summary = f"<code>{len(watchers)}</code> active" if watchers else "None active"

        msg = (
            "🔔 <b>MASTER ALERTS CONTROL CENTER</b>\n\n"
            f"• <b>Auto Market Alerts:</b> {'🟢 ACTIVE (BTC & Gold)' if auto_alerts_on else '🔴 OFF'}\n"
            f"• <b>Active Price Alerts:</b> {alerts_summary}\n"
            f"• <b>Candle Watchers:</b> {watchers_summary}\n\n"
            "⚡ <b>1-Command Working Modes:</b>\n"
            "• <code>/alert on</code> — Turn ON automatic alerts for BTC & Gold 🟢\n"
            "• <code>/alert off</code> — Turn OFF automatic alerts 🔴\n"
            "• <code>/alert list</code> — View all active price alerts\n"
            "• <code>/alert &lt;price&gt;</code> — Set price alert (e.g. <code>/alert 85000</code>)\n"
            "• <code>/alert &lt;sym&gt; &lt;price&gt;</code> — Set price alert (e.g. <code>/alert gold 4180</code>)\n"
            "• <code>/alert del &lt;ID&gt;</code> — Remove an alert by ID\n"
            "• <code>/alert clear</code> — Clear all price alerts\n"
            "• <code>/alert watch &lt;sym&gt;</code> — Enable candle alerts on closes\n"
            "• <code>/alert unwatch &lt;sym&gt;</code> — Disable candle alerts"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    sub = args[0].lower().strip()
    rest = args[1:]

    if sub in ("on", "start", "enable"):
        await auto_alert_on_cmd(update, ctx)
    elif sub in ("off", "stop", "disable"):
        await auto_alert_off_cmd(update, ctx)
    elif sub in ("list", "show", "all"):
        ctx.args = []
        await list_alerts_cmd(update, ctx)
    elif sub in ("del", "delete", "remove"):
        ctx.args = rest
        await delete_alert_cmd(update, ctx)
    elif sub in ("clear", "clearall", "reset"):
        await clear_alerts_cmd(update, ctx)
    elif sub in ("watch", "watcher"):
        ctx.args = rest
        await watch_cmd(update, ctx)
    elif sub in ("unwatch", "stopwatch"):
        ctx.args = rest
        await unwatch_cmd(update, ctx)
    elif sub in ("watchers",):
        await list_watchers_cmd(update, ctx)
    else:
        # Price alert mode
        symbol, target_price, condition, err = _parse_alert_args(args)
        if err:
            await update.message.reply_text(err, parse_mode=ParseMode.HTML)
            return

        chat_id = update.effective_chat.id
        try:
            ticker = get_ticker(symbol)
            curr_price = ticker["mark_price"] or ticker["close"]
        except Exception as e:
            await update.message.reply_text(f"❌ Could not verify symbol {symbol}: {e}")
            return

        alert = alert_manager.add_price_alert(
            chat_id=chat_id,
            symbol=symbol,
            target_price=target_price,
            condition=condition,
            current_price=curr_price,
        )

        cond_str = "rises to or crosses above" if alert["condition"] in (">=", ">") else "drops to or crosses below"
        diff_pct = ((target_price - curr_price) / curr_price * 100.0) if curr_price > 0 else 0.0

        msg = (
            f"✅ <b>Price Alert Set (#A{alert['id']})</b>\n\n"
            f"🪙 <b>Symbol:</b> <code>#{symbol}</code>\n"
            f"🎯 <b>Target Price:</b> <code>${target_price:,.2f}</code>\n"
            f"📍 <b>Current Price:</b> <code>${curr_price:,.2f}</code> ({diff_pct:+.2f}% away)\n"
            f"🔔 <b>Trigger:</b> When price {cond_str} <code>${target_price:,.2f}</code>\n\n"
            f"<i>I will notify you automatically when this price is hit. Use <code>/alert list</code> to manage.</i>"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def auto_alert_on_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Enable automatic market alerts for BTC and Gold (/alertson, /autoalert on)."""
    chat_id = update.effective_chat.id
    # Add watchers for BTCUSD and XAUTUSD on 5m and 15m
    alert_manager.add_entry_watcher(chat_id, "BTCUSD", ["5m", "15m"])
    alert_manager.add_entry_watcher(chat_id, "XAUTUSD", ["5m", "15m"])

    msg = (
        "🔔 <b>AUTOMATIC ALERTS: TURNED ON!</b> 🟢\n\n"
        "Now automatically scanning every candle close for:\n"
        "• 🪙 <b>Bitcoin (#BTCUSD)</b> — 5m & 15m timeframes\n"
        "• 🥇 <b>Gold (#XAUTUSD)</b> — 5m & 15m timeframes\n\n"
        "🎯 <b>What's Monitored:</b>\n"
        "• Gautam Jha Daily Open (DO) color flip reversals & continuations\n"
        "• Previous Day High (PDH) & Low (PDL) sweeps\n"
        "• Multi-timeframe Pin Bars, Hammers, Shooting Stars & Engulfing setups\n"
        "• Complete trade plans: Entry, Stop Loss, TP1 (1:1.5), TP2 (1:2.5+)\n\n"
        "⚡ <i>You will receive instant notifications whenever high-probability setups form!</i>\n\n"
        "🛑 <i>To turn off anytime: <code>/alertsoff</code> or <code>/autoalert off</code></i>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def auto_alert_off_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Disable automatic market alerts (/alertsoff, /autoalert off)."""
    chat_id = update.effective_chat.id
    count = alert_manager.remove_entry_watcher(chat_id)
    msg = (
        "🔕 <b>AUTOMATIC ALERTS: TURNED OFF.</b> 🔴\n\n"
        "All automatic candle and setup scanners have been paused for this chat.\n\n"
        "💡 <i>Your custom price alerts (if any) remain saved.</i>\n"
        "To turn automatic alerts back on anytime, type <code>/alertson</code>."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def auto_alert_toggle_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Command handler for /autoalert or /autoalerts [on|off]."""
    args = ctx.args or []
    if not args:
        chat_id = update.effective_chat.id
        watchers = alert_manager.get_chat_entry_watchers(chat_id)
        if watchers:
            tfs_list = [f"#{w['symbol']} ({', '.join([tf.upper() for tf in w['timeframes']])})" for w in watchers]
            await update.message.reply_text(
                "🔔 <b>Automatic Alerts: ACTIVE 🟢</b>\n\n"
                "Currently monitoring:\n" + "\n".join(f"• {t}" for t in tfs_list) +
                "\n\n<b>Controls:</b>\n"
                "• <code>/alertsoff</code> — Turn alerts OFF\n"
                "• <code>/alertson</code> — Turn alerts ON",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                "🔕 <b>Automatic Alerts: OFF 🔴</b>\n\n"
                "To turn automatic candle and liquidity alerts ON:\n"
                "• Type <code>/alertson</code> or <code>/autoalert on</code>",
                parse_mode=ParseMode.HTML,
            )
        return

    sub = args[0].lower().strip()
    if sub in ("on", "start", "enable", "1", "true"):
        await auto_alert_on_cmd(update, ctx)
    elif sub in ("off", "stop", "disable", "0", "false"):
        await auto_alert_off_cmd(update, ctx)
    else:
        await update.message.reply_text("Usage: <code>/autoalert on</code> or <code>/autoalert off</code>", parse_mode=ParseMode.HTML)


async def list_alerts_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List active price alerts or toggle automatic alerts on/off."""
    if ctx.args:
        sub = ctx.args[0].lower().strip()
        if sub in ("on", "start", "enable", "1"):
            await auto_alert_on_cmd(update, ctx)
            return
        elif sub in ("off", "stop", "disable", "0"):
            await auto_alert_off_cmd(update, ctx)
            return

    chat_id = update.effective_chat.id
    alerts = alert_manager.get_chat_alerts(chat_id)

    if not alerts:
        await update.message.reply_text(
            "📭 <b>You have no active price alerts.</b>\n\n"
            "• Set one with: <code>/alert &lt;PRICE&gt;</code> or <code>/alert &lt;SYMBOL&gt; &lt;PRICE&gt;</code>\n"
            "• Turn on automatic candle alerts: <code>/alertson</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    lines = [f"📋 <b>Your Active Price Alerts ({len(alerts)}):</b>\n"]
    for a in alerts:
        sym = a["symbol"]
        try:
            t = get_ticker(sym)
            curr = t["mark_price"] or t["close"]
            dist_pct = ((a["target_price"] - curr) / curr * 100.0) if curr > 0 else 0.0
            curr_str = f" | Current: <code>${curr:,.2f}</code> ({dist_pct:+.2f}%)"
        except Exception:
            curr_str = ""

        cond_sign = "≥" if a["condition"] in (">=", ">") else "≤"
        lines.append(
            f"• <b>[#A{a['id']}]</b> <code>#{sym}</code> {cond_sign} <code>${a['target_price']:,.2f}</code>{curr_str}"
        )

    lines.append("\n🗑️ <i>To remove an alert: <code>/delalert &lt;ID&gt;</code> or <code>/clearalerts</code></i>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def delete_alert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Delete a price alert by ID."""
    if not ctx.args:
        await update.message.reply_text("Usage: <code>/delalert &lt;ID&gt;</code> (e.g. <code>/delalert 1</code>)", parse_mode=ParseMode.HTML)
        return

    raw_id = ctx.args[0].replace("#A", "").replace("A", "").replace("#", "")
    try:
        alert_id = int(raw_id)
    except ValueError:
        await update.message.reply_text("❌ Invalid alert ID. Example: <code>/delalert 1</code>", parse_mode=ParseMode.HTML)
        return

    chat_id = update.effective_chat.id
    removed = alert_manager.remove_alert(chat_id, alert_id)
    if removed:
        await update.message.reply_text(f"🗑️ Alert <b>#A{alert_id}</b> removed.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"❌ Alert #A{alert_id} not found in your active alerts.", parse_mode=ParseMode.HTML)


async def clear_alerts_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clear all price alerts for this chat."""
    chat_id = update.effective_chat.id
    count = alert_manager.clear_chat_alerts(chat_id)
    await update.message.reply_text(f"🗑️ Cleared <b>{count}</b> price alert(s).", parse_mode=ParseMode.HTML)


async def entry_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Instant multi-timeframe candle entry scanner."""
    symbol = resolve_symbol(ctx.args[0]) if ctx.args else DEFAULT_SYMBOL
    await update.message.reply_text(f"⏳ Analyzing 1m, 5m, and 15m candles for #{symbol}...")
    try:
        multi_data = get_multi_timeframe_entry(symbol, ["1m", "5m", "15m"])
        msg = format_entry_analysis_message(multi_data)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error scanning candles for {symbol}: {e}")


def _parse_watch_args(args: List[str]) -> Tuple[str, List[str]]:
    """Parse watcher arguments."""
    valid_tfs = {"1m", "5m", "15m", "30m", "1h"}
    symbol = DEFAULT_SYMBOL
    timeframes = ["1m", "5m", "15m"]

    if not args:
        return symbol, timeframes

    for arg in args:
        parts = [p.strip().lower() for p in arg.split(",")]
        matched_tfs = [p for p in parts if p in valid_tfs]
        if matched_tfs:
            timeframes = matched_tfs
        else:
            symbol = resolve_symbol(arg)

    return symbol, timeframes


async def watch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Enable automated candle entry alerts on 1m, 5m, 15m candle closes."""
    symbol, timeframes = _parse_watch_args(ctx.args or [])
    chat_id = update.effective_chat.id

    try:
        # verify symbol
        get_ticker(symbol)
    except Exception as e:
        await update.message.reply_text(f"❌ Could not verify symbol {symbol}: {e}")
        return

    watcher = alert_manager.add_entry_watcher(chat_id, symbol, timeframes)
    tfs_str = ", ".join([tf.upper() for tf in timeframes])

    msg = (
        f"🔔 <b>Automated Candle Entry Alerts Enabled!</b>\n\n"
        f"🪙 <b>Symbol:</b> <code>#{symbol}</code>\n"
        f"⏱️ <b>Monitored Timeframes:</b> <code>{tfs_str}</code>\n"
        f"🎯 <b>Detection:</b> Pin Bars, Engulfing, Breakouts, S/R Bounces, RSI Reversals\n\n"
        f"<i>When a completed candle triggers an entry setup with confirmed SL & TP, "
        f"you will receive an instant alert!</i>\n\n"
        f"🛑 <i>To disable: <code>/unwatch {symbol}</code></i>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def unwatch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Stop automated candle entry monitoring."""
    chat_id = update.effective_chat.id
    symbol = ctx.args[0] if ctx.args else None
    count = alert_manager.remove_entry_watcher(chat_id, symbol)

    if count > 0:
        sym_str = f"for #{symbol.upper()}" if symbol and symbol.lower() != "all" else "for all symbols"
        await update.message.reply_text(f"🔕 Stopped automated candle alerts {sym_str}.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("ℹ️ No active entry watchers found to remove.", parse_mode=ParseMode.HTML)


async def list_watchers_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List active candle entry watchers."""
    chat_id = update.effective_chat.id
    watchers = alert_manager.get_chat_entry_watchers(chat_id)

    if not watchers:
        await update.message.reply_text(
            "📭 <b>You have no active candle entry watchers.</b>\n\n"
            "Start one with <code>/watch XAUTUSD</code> or <code>/watch BTCUSD 5m,15m</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    lines = [f"🔭 <b>Active Candle Entry Scanners ({len(watchers)}):</b>\n"]
    for w in watchers:
        tfs = ", ".join([tf.upper() for tf in w["timeframes"]])
        lines.append(f"• <code>#{w['symbol']}</code> — Timeframes: <code>{tfs}</code> (Auto-alerting)")

    lines.append("\n🛑 <i>To disable: <code>/unwatch [SYMBOL]</code></i>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clear AI chat history."""
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text("🧹 Conversation history cleared.")


def _detect_symbol_from_text(text: Optional[str], default: str = DEFAULT_SYMBOL) -> str:
    """Helper to detect symbol mentioned in a chat message or image caption."""
    if not text:
        return default
    upper = text.upper().replace("/", "")
    if "GOLD" in upper or "XAU" in upper or "XAUT" in upper:
        return "XAUTUSD"
    if "BTC" in upper or "BITCOIN" in upper or "XBT" in upper:
        return "BTCUSD"
    if "ETH" in upper or "ETHEREUM" in upper:
        return "ETHUSD"
    if "SOL" in upper or "SOLANA" in upper:
        return "SOLUSD"
    for word in text.split():
        clean_word = word.strip("$#,!?.").upper().replace("/", "")
        if clean_word.endswith("USD") and len(clean_word) >= 5:
            return resolve_symbol(clean_word)
    return default


async def chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Chat with AI assistant with live market context injected, and route plain-text quick commands."""
    chat_id = update.effective_chat.id
    text = update.message.text
    if not text:
        return

    clean_text = text.strip()
    lower_text = clean_text.lower()

    # 1. Automatic detection of pasted Delta API credentials
    # e.g. "delta new api 17sY... api secret 0SeM..."
    match_key = re.search(r'(?:api[_\s-]*key|delta(?:\s+new)?\s+api)[:\s]+([A-Za-z0-9_-]{8,})', clean_text, re.IGNORECASE)
    match_sec = re.search(r'(?:api[_\s-]*secret|secret)[:\s]+([A-Za-z0-9_-]{15,})', clean_text, re.IGNORECASE)
    if match_key and match_sec:
        k = match_key.group(1).strip()
        s = match_sec.group(1).strip()
        save_delta_credentials(api_key=k, api_secret=s)
        await update.message.reply_text(
            "🔑 <b>Delta API credentials detected and saved!</b>\nTesting connection now...",
            parse_mode=ParseMode.HTML,
        )
        await keys_cmd(update, ctx)
        return

    # Detection of pasted Google Gemini API key
    match_gemini = re.search(r'(?:gemini[_\s-]*api[_\s-]*key|google[_\s-]*api[_\s-]*key|gemini[_\s-]*key)[:\s]+([A-Za-z0-9_-]{25,})', clean_text, re.IGNORECASE)
    if match_gemini:
        gkey = match_gemini.group(1).strip()
        save_gemini_credentials(api_key=gkey)
        await update.message.reply_text(
            f"🔑 <b>Google Gemini API Key Saved!</b>\n• Active Model: <code>{get_gemini_model()}</code> 🟢",
            parse_mode=ParseMode.HTML,
        )
        return

    # Google Gemini model queries & switches
    if lower_text in ("model", "models", "gemini model", "google model", "check model", "what model", "ai model", "gemini"):
        ctx.args = []
        await gemini_model_cmd(update, ctx)
        return
    if ("update" in lower_text or "upgrade" in lower_text or "latest" in lower_text) and "model" in lower_text:
        ctx.args = ["latest"]
        await gemini_model_cmd(update, ctx)
        return
    if lower_text.startswith("model ") or lower_text.startswith("set model ") or lower_text.startswith("switch model "):
        parts = clean_text.split()
        ctx.args = [parts[-1]] if len(parts) > 1 else []
        await gemini_model_cmd(update, ctx)
        return

    # 2. Plain-text command routing (without leading '/')
    # Keys / connection checks
    if lower_text in (
        "keys check", "check keys", "check key", "keys", "key",
        "test keys", "test key", "key test", "check api", "check api key",
        "api status", "delta keys", "delta key", "verify keys", "keys status"
    ) or lower_text.startswith("keys check") or lower_text.startswith("check key") or lower_text.startswith("test key"):
        await keys_cmd(update, ctx)
        return

    # Start / Stop Trading commands
    if lower_text in (
        "start trade", "start trading", "start bot", "trade on", "start autotrade",
        "start auto trade", "trading start", "trade start", "trading on", "autotrade on"
    ):
        await start_trade_cmd(update, ctx)
        return
    if lower_text in (
        "stop trade", "stop trading", "stop bot", "trade off", "stop autotrade",
        "stop auto trade", "trading stop", "trade stop", "trading off", "autotrade off"
    ):
        await stop_trade_cmd(update, ctx)
        return

    # Alerts toggle
    if lower_text in (
        "alerts on", "alert on", "auto alert on", "auto alerts on",
        "turn on alerts", "enable alerts", "alerts start"
    ):
        await auto_alert_on_cmd(update, ctx)
        return
    if lower_text in (
        "alerts off", "alert off", "auto alert off", "auto alerts off",
        "turn off alerts", "disable alerts", "alerts stop"
    ):
        await auto_alert_off_cmd(update, ctx)
        return

    # Institutional Status & Dashboard
    if lower_text in (
        "status", "dashboard", "dash", "bot status", "bot overview", "overview",
        "desk status", "system status", "bot health"
    ):
        await status_cmd(update, ctx)
        return

    # Close positions
    if lower_text in ("close all", "closeall", "exit all", "close all positions", "close positions"):
        await close_all_cmd(update, ctx)
        return

    # Breakeven SL toggles
    if lower_text in ("be on", "breakeven on", "enable breakeven", "turn on breakeven"):
        ctx.args = ["be", "on"]
        await trade_cmd(update, ctx)
        return
    if lower_text in ("be off", "breakeven off", "disable breakeven", "turn off breakeven"):
        ctx.args = ["be", "off"]
        await trade_cmd(update, ctx)
        return

    # Trailing Stop-Loss toggles
    if lower_text in ("trail on", "trailing on", "trailing stop on", "enable trailing", "enable trail"):
        ctx.args = ["trail", "on"]
        await trade_cmd(update, ctx)
        return
    if lower_text in ("trail off", "trailing off", "trailing stop off", "disable trailing", "disable trail"):
        ctx.args = ["trail", "off"]
        await trade_cmd(update, ctx)
        return

    # Auto-Trade Lot Size queries and changes
    if lower_text in ("lot size", "lotsize", "size", "check size", "check lot size", "trade size", "what size", "auto trade size", "auto trade lot size"):
        ctx.args = ["size"]
        await trade_cmd(update, ctx)
        return

    # Per-symbol lot size: "btc size 0.01", "gold lot size 0.5", "size btc 0.01", "lot size btc 0.01"
    m_sym_size = re.match(r'^(?:set\s+)?(btc|bitcoin|gold|xau|eth|sol)\s+(?:lot\s*size|size|lot)\s*(?:to\s*|=)?\s*([0-9.]+)$', lower_text)
    if m_sym_size:
        ctx.args = ["size", m_sym_size.group(1), m_sym_size.group(2)]
        await trade_cmd(update, ctx)
        return

    m_sym_size2 = re.match(r'^(?:set\s+)?(?:lot\s*size|size|lot)\s+(btc|bitcoin|gold|xau|eth|sol)\s*(?:to\s*|=)?\s*([0-9.]+)$', lower_text)
    if m_sym_size2:
        ctx.args = ["size", m_sym_size2.group(1), m_sym_size2.group(2)]
        await trade_cmd(update, ctx)
        return

    # Global lot size: "set auto trade lot size 0.05", "set lot size 0.05", "lot size 0.05", "size 0.05", "lot 0.05"
    m_global_size = re.match(r'^(?:set\s+)?(?:auto\s+trade\s+)?(?:lot\s*size|size|lot)\s*(?:to\s*|=)?\s*([0-9.]+)(?:\s*(?:lot|lots|units?))?$', lower_text)
    if m_global_size:
        ctx.args = ["size", m_global_size.group(1)]
        await trade_cmd(update, ctx)
        return

    # Natural language with "lot size" (e.g. "set auto trade lot size and its manual change...", "auto trade lot size 0.02")
    if "lot size" in lower_text or "auto trade lot" in lower_text:
        num_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', lower_text)
        if num_match:
            ctx.args = ["size", num_match.group(1)]
        else:
            ctx.args = ["size"]
        await trade_cmd(update, ctx)
        return

    # Help & commands list
    if lower_text in ("help", "commands", "cmd", "cmds", "list", "all commands"):
        await list_cmd(update, ctx)
        return

    # Quick Buy / Sell plain text orders (e.g. "buy btc", "sell btc 0.1", "buy gold")
    quick_order_match = re.match(r'^(buy|sell|long|short)\s+(btc|bitcoin|gold|xau|eth|sol)(?:\s+([0-9.]+))?$', lower_text)
    if quick_order_match:
        q_side = quick_order_match.group(1)
        q_sym = quick_order_match.group(2)
        q_size = quick_order_match.group(3)
        args_to_pass = [q_sym, q_side]
        if q_size:
            args_to_pass.append(q_size)
        ctx.args = args_to_pass
        await trade_cmd(update, ctx)
        return

    # Balance / Positions / PnL
    if lower_text in ("balance", "my balance", "check balance", "wallet", "wallet balance", "bal"):
        await balance_cmd(update, ctx)
        return
    if lower_text in ("positions", "my positions", "open positions", "position", "pos", "pnl", "profit", "live pnl"):
        await positions_cmd(update, ctx)
        return

    # Self-learning & self-improvement queries
    if lower_text in (
        "learn", "learning", "self learn", "self learning", "self improve",
        "how to improve", "learning stats", "learning report", "learning engine",
        "strategy report", "insights", "self improvement", "bot learning",
        "self learning and self improve", "self learning and self improve its own",
    ) or "self learn" in lower_text or "self improve" in lower_text or "learning report" in lower_text:
        ctx.args = []
        await learn_cmd(update, ctx)
        return

    if lower_text in ("learn ai", "ai insights", "ai insight", "strategy ai", "ai improve", "ai learning"):
        ctx.args = ["ai"]
        await learn_cmd(update, ctx)
        return

    # News analysis queries
    if lower_text in (
        "news", "news analysis", "crypto news", "market news", "gold news", "btc news",
        "bitcoin news", "latest news", "breaking news", "check news", "sentiment", "macro",
    ) or "news analysis" in lower_text:
        sym = "XAUTUSD" if ("gold" in lower_text or "xau" in lower_text) else "BTCUSD"
        ctx.args = [sym]
        await news_cmd(update, ctx)
        return

    # Orderbook analysis queries (including common typo 'oderbook')
    if lower_text in (
        "orderbook", "order book", "orderbook analysis", "order book analysis", "oderbook", "oderbook analysis",
        "depth", "depth analysis", "market depth", "l2 depth", "order flow", "bid ask",
        "liquidity walls", "buy walls", "sell walls", "book",
    ) or "orderbook" in lower_text or "order book" in lower_text or "oderbook" in lower_text:
        sym = "XAUTUSD" if ("gold" in lower_text or "xau" in lower_text) else "BTCUSD"
        ctx.args = [sym]
        await orderbook_cmd(update, ctx)
        return

    # Confluence & combined strategy queries
    if lower_text in (
        "confluence", "master strategy", "every strategy combined", "combined strategy",
        "all strategies", "all strategy", "combine strategy", "combined trade",
        "confluence trade", "master trade", "strategy combined",
    ) or "every strategy combined" in lower_text or "combined strategy" in lower_text or "confluence trade" in lower_text:
        sym = "XAUTUSD" if ("gold" in lower_text or "xau" in lower_text) else "BTCUSD"
        ctx.args = [sym]
        await confluence_cmd(update, ctx)
        return

    # 18-Agent Institutional Desk queries
    if lower_text in (
        "analysis", "analyze", "deep analysis", "18 agents", "18 agent", "desk",
        "institutional analysis", "chart analysis", "market analysis"
    ) or lower_text.startswith("analyze") or lower_text.startswith("analysis"):
        sym = "XAUTUSD" if ("gold" in lower_text or "xau" in lower_text) else "BTCUSD"
        ctx.args = [sym]
        await analyze_cmd(update, ctx)
        return

    # AMD Scalp & Multi-Timeframe (1m, 5m, 15m) queries
    if lower_text in (
        "amd", "scalp", "amd scalp", "amd trading", "po3", "power of 3",
        "accumulation manipulation distribution", "scalping", "amd trade",
        "1m 5m 15m", "1m 5m", "scalp trade",
    ) or "amd scalp" in lower_text or "scalp trading" in lower_text or "amd trade" in lower_text:
        sym = "XAUTUSD" if ("gold" in lower_text or "xau" in lower_text) else "BTCUSD"
        ctx.args = [sym]
        await amd_cmd(update, ctx)
        return

    # Multi-trade status & control
    if lower_text in (
        "multiple trade", "multi trade", "auto multiple trade", "multiple trades",
        "multi trades", "concurrent trades",
    ) or "multiple trade" in lower_text or "multi trade" in lower_text:
        ctx.args = ["multi"]
        await trade_cmd(update, ctx)
        return

    # Master hub quick access
    if lower_text in ("btc", "bitcoin"):
        ctx.args = []
        await btc_cmd(update, ctx)
        return
    if lower_text in ("gold", "xau"):
        ctx.args = []
        await gold_cmd(update, ctx)
        return
    if lower_text in ("trade", "trading"):
        ctx.args = []
        await trade_cmd(update, ctx)
        return
    if lower_text in ("alert", "alerts"):
        ctx.args = []
        await set_alert_cmd(update, ctx)
        return

    # Mode switch
    if lower_text in ("mode paper", "paper mode", "paper trading"):
        ctx.args = ["paper"]
        await mode_cmd(update, ctx)
        return
    if lower_text in ("mode live", "live mode", "live trading"):
        ctx.args = ["live"]
        await mode_cmd(update, ctx)
        return

    # Detect symbol from text if mentioned, else default
    symbol = _detect_symbol_from_text(text, DEFAULT_SYMBOL)

    # Build rich market context
    try:
        ticker = get_ticker(symbol)
        levels = get_level_analysis(symbol)
        multi_entry = get_multi_timeframe_entry(symbol, ["1m", "5m", "15m"])

        gj = levels.get("gautam_jha", {})
        ctx_summary = (
            f"[MARKET DATA FOR {symbol}]\n"
            f"Price: {ticker['mark_price']} | 24h High: {ticker['high']} | 24h Low: {ticker['low']}\n"
            f"Trend Bias: {levels['trend_bias']}\n"
            f"Gautam Jha Liquidity: Daily Open={gj.get('daily_open')} ({gj.get('daily_candle_color')}), "
            f"PDH={gj.get('pdh')} (Swept: {gj.get('pdh_swept')}), PDL={gj.get('pdl')} (Swept: {gj.get('pdl_swept')})\n"
            f"Key Levels: Pivot={levels['pivots']['pivot']}, R1={levels['pivots']['r1']}, S1={levels['pivots']['s1']}, "
            f"Nearest Res={levels['nearest_resistance']['name']} ({levels['nearest_resistance']['price']}), "
            f"Nearest Sup={levels['nearest_support']['name']} ({levels['nearest_support']['price']})\n"
            f"1m Candle: RSI={multi_entry['timeframes'].get('1m', {}).get('rsi')}, Pattern={multi_entry['timeframes'].get('1m', {}).get('pattern')}, Signal={multi_entry['timeframes'].get('1m', {}).get('signal')}\n"
            f"5m Candle: RSI={multi_entry['timeframes'].get('5m', {}).get('rsi')}, Pattern={multi_entry['timeframes'].get('5m', {}).get('pattern')}, Signal={multi_entry['timeframes'].get('5m', {}).get('signal')}\n"
            f"15m Candle: RSI={multi_entry['timeframes'].get('15m', {}).get('rsi')}, Pattern={multi_entry['timeframes'].get('15m', {}).get('pattern')}, Signal={multi_entry['timeframes'].get('15m', {}).get('signal')}\n\n"
        )
    except Exception as e:
        ctx_summary = f"[Market Data Notice: {e}]\n\n"

    hist = histories.setdefault(chat_id, [])
    user_content = f"{ctx_summary}User: {text}"
    hist.append({"role": "user", "text": user_content})
    hist[:] = hist[-16:]

    answer = generate_ai_reply(hist)
    hist.append({"role": "model", "text": answer})
    await update.message.reply_text(answer)


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Analyze chart photo or screenshot using Gemini Multimodal Vision & 18-Agent Institutional Desk system."""
    if not update.message or not update.message.photo:
        return

    await update.message.reply_text(
        "🏛️ 18 Institutional Agents Analyzing Chart...\n"
        "Price Action → Liquidity → Context → News → Momentum → Decision\nPlease wait ⏳"
    )

    photo = update.message.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    caption = update.message.caption or "Deep categorized analysis required. Follow the category structure. Only high confluence trades allowed."

    # Detect symbol from caption if present
    symbol = _detect_symbol_from_text(caption, DEFAULT_SYMBOL)

    # Enrich prompt with live market liquidity data if available
    context_note = ""
    try:
        gj_analysis = get_gautam_jha_analysis(symbol)
        gj = gj_analysis["gautam_jha"]
        ob = analyze_orderbook(symbol)
        news = get_news_sentiment(symbol)
        conf = auto_trader.confluence_engine.calculate_confluence(symbol)
        context_note = (
            f"\n[Live Delta Exchange Institutional Data for #{symbol}]\n"
            f"• Current Mark Price: ${gj_analysis['mark_price']:,.2f}\n"
            f"• Gautam Jha Daily Open: ${gj['daily_open']:,.2f} ({gj['daily_candle_color']} candle)\n"
            f"• PDH: ${gj['pdh']:,.2f} (Swept: {gj['pdh_swept']}) | PDL: ${gj['pdl']:,.2f} (Swept: {gj['pdl_swept']})\n"
            f"• Market Structure: {gj_analysis['market_structure']}\n"
            f"• L2 Order Book Imbalance: {ob['imbalance_ratio']:.2f} ({ob['imbalance_bias']})\n"
            f"• News Sentiment: {news['sentiment_score']:+.2f} ({news['sentiment_label']})\n"
            f"• Master Confluence Score: {conf['confluence_score']}% {conf['bias_signal']}\n"
        )
    except Exception:
        pass

    full_prompt = (
        f"User request: {caption}\n"
        f"{context_note}\n"
        f"Conduct a rigorous 18-Agent Institutional Desk Analysis following the exact 6-category structure:\n"
        f"1. Price Action Core (Gautam Jha, Order Block, FVG, Breaker)\n"
        f"2. Liquidity & Sessions (Grabs, PDH/PDL, round numbers)\n"
        f"3. Market Context (Higher TF, Multi-TF alignment, Correlations)\n"
        f"4. News & Sentiment (US high-impact events, Sentiment)\n"
        f"5. Momentum & Strength (Volume, Institutional Intent)\n"
        f"6. Decision Layer (Strict Confluence Score /10 & Risk Manager approval/reject)\n"
        f"Strict Rules: Confluence Score < 7 means No Trade; If Risk Manager rejects -> No Trade."
    )

    reply = generate_ai_vision_reply(full_prompt, bytes(image_bytes), system_prompt=SYSTEM_PROMPT_18_AGENTS)
    await update.message.reply_text(reply)


# ==================== Trading & Delta Exchange Commands ====================

def save_delta_credentials(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    base_url: Optional[str] = None,
    env_path: str = ".env",
) -> bool:
    """Save Delta Exchange credentials to .env file and update current process environment."""
    if api_key:
        os.environ["DELTA_API_KEY"] = api_key
        delta_client.api_key = api_key
    if api_secret:
        os.environ["DELTA_API_SECRET"] = api_secret
        delta_client.api_secret = api_secret
    if base_url:
        os.environ["DELTA_BASE_URL"] = base_url
        delta_client.base_url = base_url.rstrip("/")

    lines = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Could not read existing {env_path}: {e}")

    keys_set = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if api_key and stripped.startswith("DELTA_API_KEY="):
            new_lines.append(f"DELTA_API_KEY={api_key}\n")
            keys_set.add("DELTA_API_KEY")
        elif api_secret and stripped.startswith("DELTA_API_SECRET="):
            new_lines.append(f"DELTA_API_SECRET={api_secret}\n")
            keys_set.add("DELTA_API_SECRET")
        elif base_url and stripped.startswith("DELTA_BASE_URL="):
            new_lines.append(f"DELTA_BASE_URL={base_url}\n")
            keys_set.add("DELTA_BASE_URL")
        else:
            new_lines.append(line)

    if api_key and "DELTA_API_KEY" not in keys_set:
        new_lines.append(f"DELTA_API_KEY={api_key}\n")
    if api_secret and "DELTA_API_SECRET" not in keys_set:
        new_lines.append(f"DELTA_API_SECRET={api_secret}\n")
    if base_url and "DELTA_BASE_URL" not in keys_set:
        new_lines.append(f"DELTA_BASE_URL={base_url}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except Exception as e:
        logger.error(f"Failed to write keys to {env_path}: {e}")
        return False


def save_delta_keys_to_env(api_key: str, api_secret: str, base_url: Optional[str] = None, env_path: str = ".env") -> bool:
    """Backward compatibility wrapper."""
    return save_delta_credentials(api_key=api_key, api_secret=api_secret, base_url=base_url, env_path=env_path)


async def gemini_model_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View or switch Google Gemini AI model (/model [MODEL_NAME])."""
    args = ctx.args or []
    current_model = get_gemini_model()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))

    if not args:
        models_list = "\n".join([
            f"  • <code>{m}</code> {'🟢 <b>(Active)</b>' if m == current_model else ''}"
            for m in SUPPORTED_GEMINI_MODELS
        ])
        msg = (
            "🤖 <b>Google Gemini AI Model Configuration</b>\n\n"
            f"• <b>Active Model:</b> <code>{current_model}</code> ⚡\n"
            f"• <b>API Key:</b> {'🟢 Configured' if has_key else '⚠️ Not Set (Use /setgemini <KEY>)'}\n\n"
            f"<b>Supported Models:</b>\n{models_list}\n\n"
            "<b>To Switch Models:</b>\n"
            "• <code>/model gemini-2.5-flash</code> — Latest multimodal & price-performance (Default)\n"
            "• <code>/model gemini-2.5-pro</code> — Deep reasoning & advanced analysis\n"
            "• <code>/model gemini-2.5-flash-lite</code> — Ultra low-latency speed\n"
            "• <code>/model gemini-2.0-flash</code> — 2.0 generation\n\n"
            "💡 <i>Shortcuts work too: <code>/model pro</code>, <code>/model flash</code>, or <code>/model lite</code></i>"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    req = args[0].strip().lower()
    if req in ("flash", "2.5-flash", "fast", "latest"):
        target_model = "gemini-2.5-flash"
    elif req in ("pro", "2.5-pro", "reasoning", "deep"):
        target_model = "gemini-2.5-pro"
    elif req in ("lite", "flash-lite", "2.5-lite", "2.5-flash-lite"):
        target_model = "gemini-2.5-flash-lite"
    elif req in ("2.0", "2.0-flash", "2.0flash"):
        target_model = "gemini-2.0-flash"
    elif req in ("1.5", "1.5-flash"):
        target_model = "gemini-1.5-flash"
    else:
        target_model = args[0].strip()

    save_gemini_credentials(model=target_model)

    await update.message.reply_text(
        f"✅ <b>Google Gemini Model Updated!</b>\n\n"
        f"• <b>New Model:</b> <code>{target_model}</code> 🚀\n"
        f"• <b>Saved to:</b> <code>.env</code> file\n\n"
        "Ask any question or upload a chart screenshot to analyze with the new model!",
        parse_mode=ParseMode.HTML,
    )


async def set_gemini_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set Google Gemini API Key (/setgemini <API_KEY>)."""
    args = ctx.args or []
    current_model = get_gemini_model()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))

    if not args:
        await update.message.reply_text(
            "🔑 <b>Google Gemini API Key Setup</b>\n\n"
            f"• <b>Status:</b> {'🟢 Configured' if has_key else '⚠️ Not Set'}\n"
            f"• <b>Active Model:</b> <code>{current_model}</code>\n\n"
            "<b>Usage:</b>\n"
            "<code>/setgemini &lt;API_KEY&gt;</code>\n\n"
            "💡 <b>How to get your free key:</b>\n"
            "1. Visit https://aistudio.google.com\n"
            "2. Click <b>Get API key → Create API key</b>\n"
            "3. Send <code>/setgemini YOUR_KEY</code> to this bot",
            parse_mode=ParseMode.HTML,
        )
        return

    api_key = args[0].strip()
    save_gemini_credentials(api_key=api_key)

    await update.message.reply_text(
        "✅ <b>Google Gemini API Key Saved!</b>\n\n"
        f"• <b>Status:</b> 🟢 Active & Configured\n"
        f"• <b>Model:</b> <code>{current_model}</code>\n"
        f"• <b>Storage:</b> Persisted to local <code>.env</code>\n\n"
        "🚀 <i>You can now chat directly with the AI assistant and upload chart photos for Gautam Jha vision analysis!</i>",
        parse_mode=ParseMode.HTML,
    )


async def gemini_status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View Google Gemini AI status and model (/gemini, /ai)."""
    await gemini_model_cmd(update, ctx)


async def set_key_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set Delta Exchange API Key (/setkey <API_KEY>)."""
    args = ctx.args or []
    if not args:
        masked = delta_client.get_masked_key()
        await update.message.reply_text(
            "🔑 <b>Delta Exchange API Key</b>\n\n"
            f"<b>Current Key:</b> <code>{masked}</code>\n\n"
            "<b>Usage:</b>\n"
            "<code>/setkey &lt;API_KEY&gt;</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/setkey ovRwsM4ZGWkK2JI67yOIiTdu2BWnhg</code>\n\n"
            "💡 <i>To set your secret as well: <code>/setsecret &lt;SECRET&gt;</code></i>",
            parse_mode=ParseMode.HTML,
        )
        return

    api_key = args[0].strip()
    save_delta_credentials(api_key=api_key)
    masked = delta_client.get_masked_key()

    has_secret = bool(delta_client.api_secret)
    secret_note = "🟢 Configured" if has_secret else "⚠️ Not Set (Required for live execution: <code>/setsecret &lt;SECRET&gt;</code>)"

    await update.message.reply_text(
        "✅ <b>Delta Exchange API Key Saved!</b>\n\n"
        f"• <b>API Key:</b> <code>{masked}</code>\n"
        f"• <b>API Secret:</b> {secret_note}\n"
        f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n\n"
        + ("🚀 <b>Ready for live trading! Use <code>/mode live</code> and <code>/starttrade</code></b>" if has_secret else
           "💡 <i>Next step: Send your API Secret using <code>/setsecret &lt;API_SECRET&gt;</code> to enable live trading!</i>"),
        parse_mode=ParseMode.HTML,
    )


async def set_secret_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set Delta Exchange API Secret (/setsecret <API_SECRET>)."""
    args = ctx.args or []
    if not args:
        has_secret = bool(delta_client.api_secret)
        await update.message.reply_text(
            "🔐 <b>Delta Exchange API Secret</b>\n\n"
            f"<b>Status:</b> {'Configured [Protected] 🟢' if has_secret else 'Not Set ⚠️'}\n\n"
            "<b>Usage:</b>\n"
            "<code>/setsecret &lt;API_SECRET&gt;</code>\n\n"
            "🔒 <i>Your secret is stored securely in local <code>.env</code> file.</i>\n"
            "<i>(Tip: delete your Telegram message after sending to protect your secret)</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    api_secret = args[0].strip()
    save_delta_credentials(api_secret=api_secret)

    await update.message.reply_text(
        "✅ <b>Delta Exchange API Secret Saved!</b>\n\n"
        f"• <b>API Key:</b> <code>{delta_client.get_masked_key()}</code>\n"
        f"• <b>API Secret:</b> Configured & Active 🟢\n"
        f"• <b>Connection Status:</b> {'Ready for Live Trading ⚡' if delta_client.is_configured() else 'API Key needed: /setkey <KEY>'}\n\n"
        "🎯 <b>Next Steps:</b>\n"
        "• <code>/balance</code> — Check live wallet balance\n"
        "• <code>/mode live</code> — Switch to live execution\n"
        "• <code>/starttrade</code> — Start automatic trading engine",
        parse_mode=ParseMode.HTML,
    )


async def set_keys_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set Delta Exchange API Key and Secret (/setkeys <API_KEY> <API_SECRET>)."""
    args = ctx.args or []
    if len(args) < 2:
        masked = delta_client.get_masked_key()
        await update.message.reply_text(
            "🔑 <b>Delta Exchange API Key Setup</b>\n\n"
            f"<b>Status:</b> {('Configured (' + masked + ')') if delta_client.is_configured() else 'Not Configured ⚠️'}\n\n"
            "<b>Usage:</b>\n"
            "• <code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>\n"
            "• Or set individually: <code>/setkey &lt;KEY&gt;</code> & <code>/setsecret &lt;SECRET&gt;</code>\n\n"
            "<b>Example:</b>\n"
            "<code>/setkeys d_key_123456789 secret_abcdef123456789</code>\n\n"
            "💡 <b>How to get keys:</b>\n"
            "1. Log into your Delta Exchange India or Global account\n"
            "2. Go to <b>Settings → API Keys → Create New API Key</b>\n"
            "3. Enable <b>Trading / Orders</b> permissions\n"
            "4. Copy your API Key & API Secret and send them here\n\n"
            "🔒 <i>Your keys are stored securely in local <code>.env</code> file.</i>\n"
            "<i>(Tip: delete your Telegram message after sending to protect your secret)</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    api_key = args[0].strip()
    api_secret = args[1].strip()

    ok = save_delta_keys_to_env(api_key, api_secret)
    masked = delta_client.get_masked_key()

    if ok:
        await update.message.reply_text(
            "✅ <b>Delta Exchange API Keys Saved!</b>\n\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Status:</b> Configured & Active\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Current Mode:</b> <code>{auto_trader.mode.upper()}</code>\n\n"
            "🎯 <b>Next Steps:</b>\n"
            "• Use <code>/balance</code> to view wallet balance\n"
            "• Use <code>/mode live</code> to switch to live trading\n"
            "• Use <code>/starttrade</code> to start automated trading\n"
            "• Use <code>/trade BTC buy</code> for manual execution",
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.message.reply_text(
            "⚠️ <i>Error saving keys to .env file, but keys are active for current session.</i>",
            parse_mode=ParseMode.HTML,
        )


async def set_base_url_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Set or switch Delta Exchange base URL (/setbaseurl [india|global|<URL>])."""
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "🌐 <b>Delta Exchange Base URL Configuration</b>\n\n"
            f"• <b>Current URL:</b> <code>{delta_client.base_url}</code>\n\n"
            "<b>Usage:</b>\n"
            "• <code>/setbaseurl india</code> — Delta India (https://api.india.delta.exchange)\n"
            "• <code>/setbaseurl global</code> — Delta Global (https://api.delta.exchange)\n"
            "• <code>/setbaseurl &lt;URL&gt;</code> — Custom API endpoint",
            parse_mode=ParseMode.HTML,
        )
        return

    arg = args[0].strip().lower()
    if arg in ("india", "in", "ind"):
        new_url = "https://api.india.delta.exchange"
    elif arg in ("global", "com", "world", "int", "international"):
        new_url = "https://api.delta.exchange"
    elif arg.startswith("http://") or arg.startswith("https://"):
        new_url = args[0].strip()
    else:
        new_url = f"https://{args[0].strip()}"

    delta_client.set_base_url(new_url)
    save_delta_credentials(base_url=new_url)

    await update.message.reply_text(
        f"✅ <b>Delta Exchange URL updated:</b> <code>{new_url}</code>\n\n"
        "Run <code>/keys</code> or <code>/checkkeys</code> to test connection.",
        parse_mode=ParseMode.HTML,
    )


async def keys_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Master Keys & Bot Configuration Hub.
    Usage:
    • /keys (connection status & credentials overview)
    • /keys check (test live Delta Exchange connection)
    • /keys set <key> <secret> (save Delta credentials)
    • /keys base [india|global] (switch Delta endpoint)
    • /keys gemini <api_key> (save Google Gemini API Key)
    • /keys model [pro|flash|lite] (switch Gemini model)
    """
    args = ctx.args or []
    if args:
        sub = args[0].lower().strip()
        rest = args[1:]
        if sub in ("set", "save", "add"):
            if len(rest) >= 2:
                ctx.args = rest[:2]
                await set_keys_cmd(update, ctx)
                return
            elif len(rest) == 1:
                ctx.args = [rest[0]]
                await set_key_cmd(update, ctx)
                return
            else:
                await update.message.reply_text("Usage: <code>/keys set &lt;KEY&gt; &lt;SECRET&gt;</code>", parse_mode=ParseMode.HTML)
                return
        elif sub in ("secret",):
            ctx.args = rest
            await set_secret_cmd(update, ctx)
            return
        elif sub in ("base", "baseurl", "url"):
            ctx.args = rest
            await set_base_url_cmd(update, ctx)
            return
        elif sub in ("gemini", "google", "ai"):
            ctx.args = rest
            await set_gemini_cmd(update, ctx)
            return
        elif sub in ("model", "switchmodel"):
            ctx.args = rest
            await gemini_model_cmd(update, ctx)
            return
        # If sub is check or test, fall through to live check below

    is_cfg = delta_client.is_configured()
    masked = delta_client.get_masked_key()
    mode = auto_trader.mode.upper()
    at_status = "🟢 ACTIVE" if auto_trader.enabled else "🔴 OFF"
    active_model = get_gemini_model()
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    if not is_cfg:
        msg = (
            "🔐 <b>MASTER KEYS & BOT CONFIGURATION HUB</b>\n\n"
            "<b>Delta Exchange API Status:</b> 🔴 <b>NOT CONFIGURED</b>\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Trading Mode:</b> <code>{mode}</code> (Paper Trading is active)\n"
            f"• <b>Auto-Trading:</b> {at_status}\n\n"
            f"<b>Google Gemini AI:</b> {'🟢 Configured' if has_gemini else '⚠️ Not Set'}\n"
            f"• <b>Active Model:</b> <code>{active_model}</code> 🚀\n\n"
            "⚡ <b>1-Command Working Modes:</b>\n"
            "• <code>/keys check</code> — Test live Delta Exchange connection\n"
            "• <code>/keys set &lt;KEY&gt; &lt;SECRET&gt;</code> — Connect Delta API keys\n"
            "• <code>/keys base [india|global]</code> — Switch Delta India vs Global\n"
            "• <code>/keys gemini &lt;KEY&gt;</code> — Connect Google Gemini API Key\n"
            "• <code>/keys model [flash|pro|lite]</code> — Switch Gemini model\n\n"
            "💡 <i>Tip: You can also paste your keys directly into the chat:</i>\n"
            "<code>delta new api YOUR_KEY api secret YOUR_SECRET</code>"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    # Perform live connection check safely in a thread (Python 3.8+ compatible)
    loop = asyncio.get_event_loop()
    test_res = await loop.run_in_executor(None, delta_client.test_connection)

    if test_res.get("success"):
        balances = test_res.get("balances", [])
        bal_lines = []
        for b in balances[:5]:
            asset = b.get("asset_symbol", "USD")
            amt = float(b.get("balance", 0.0))
            avail = float(b.get("available_balance", amt))
            if amt > 0:
                bal_lines.append(f"  • <b>{asset}:</b> {amt:.4f} (Avail: {avail:.4f})")
        bal_text = "\n".join(bal_lines) if bal_lines else "  • 0.00 USD (Deposit funds on Delta to trade live)"

        drift = getattr(delta_client, "_server_time_offset", 0.0)
        msg = (
            "🔐 <b>MASTER KEYS & BOT CONFIGURATION HUB</b>\n\n"
            "<b>Delta Exchange API Status:</b> 🟢 <b>CONNECTED & VERIFIED</b>\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Clock Sync:</b> <code>{drift:+.2f}s</code> (Auto-synchronized)\n"
            f"• <b>Trading Mode:</b> <code>{mode}</code>\n"
            f"• <b>Auto-Trading:</b> {at_status}\n\n"
            f"<b>Google Gemini AI:</b> {'🟢 Configured' if has_gemini else '⚠️ Not Set'}\n"
            f"• <b>Active Model:</b> <code>{active_model}</code> 🚀\n\n"
            f"💰 <b>Live Balances:</b>\n{bal_text}\n\n"
            "⚡ <b>1-Command Working Modes:</b>\n"
            "• <code>/keys check</code> — Re-test live connection & refresh balances\n"
            "• <code>/keys set &lt;KEY&gt; &lt;SECRET&gt;</code> — Update Delta credentials\n"
            "• <code>/keys base [india|global]</code> — Switch Delta endpoint\n"
            "• <code>/keys gemini &lt;KEY&gt;</code> — Set Gemini API Key\n"
            "• <code>/keys model [flash|pro|lite]</code> — Switch Gemini Model\n\n"
            "🎯 <b>Ready Commands:</b>\n"
            "• <code>/trade on</code> — Start automatic trading bot\n"
            "• <code>/trade off</code> — Pause automatic trading\n"
            "• <code>/trade live</code> — Switch to live orders\n"
            "• <code>/trade pos</code> — View active open positions"
        )
    else:
        err_msg = test_res.get("error", "Unknown connection error")
        hint = ""
        if "invalid_api_key" in err_msg.lower():
            hint = (
                "\n\n⚠️ <b>Troubleshooting Delta 'invalid_api_key':</b>\n"
                "1. <b>Platform Mismatch:</b> If your account is on Delta Global (.com), run:\n"
                "   <code>/keys base global</code>\n"
                "   If your account is on Delta India (.exchange), run:\n"
                "   <code>/keys base india</code>\n"
                "2. <b>IP Whitelist:</b> If you set IP restrictions when creating the key on Delta, requests from this bot will be rejected. Make sure IP restriction is disabled.\n"
                "3. <b>Permissions:</b> Verify in Delta settings that <b>Read</b> and <b>Trade</b> permissions are checked.\n"
                "4. <b>Re-enter Keys:</b> Update keys anytime with:\n"
                "   <code>/keys set &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>"
            )
        elif "expired_signature" in err_msg.lower():
            drift = getattr(delta_client, "_server_time_offset", 0.0)
            hint = (
                f"\n\n⚠️ <b>Clock Drift:</b> Synchronized offset ({drift:+.2f}s). Run <code>/keys check</code> again to retry."
            )

        msg = (
            "🔐 <b>MASTER KEYS & BOT CONFIGURATION HUB</b>\n\n"
            "<b>Delta Exchange API Status:</b> 🔴 <b>CONNECTION FAILED</b>\n\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Error:</b> <code>{err_msg}</code>"
            f"{hint}\n\n"
            "💡 <b>To update your API keys:</b>\n"
            "<code>/keys set &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>"
        )

    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def mode_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Switch or check trading mode (/mode [paper|live])."""
    args = ctx.args or []
    if not args:
        mode_text = (
            f"🎮 <b>PAPER TRADING</b> (Simulated with $10,000 virtual balance)"
            if auto_trader.mode == "paper"
            else f"⚡ <b>LIVE TRADING</b> (Connected to Delta Exchange)"
        )
        await update.message.reply_text(
            f"⚙️ <b>Current Trading Mode:</b> {mode_text}\n\n"
            f"• <b>Auto-Trader:</b> {'🟢 ON' if auto_trader.enabled else '🔴 OFF'}\n"
            f"• <b>Delta Keys:</b> {('Configured' if delta_client.is_configured() else 'Not Configured')}\n\n"
            "<b>To change mode:</b>\n"
            "• <code>/mode paper</code> — Safe simulated trading ($10,000 demo cash)\n"
            "• <code>/mode live</code> — Real orders via Delta Exchange API",
            parse_mode=ParseMode.HTML,
        )
        return

    req_mode = args[0].lower().strip()
    if req_mode in ("live", "real"):
        if not delta_client.is_configured():
            await update.message.reply_text(
                "⚠️ <b>Cannot switch to LIVE mode yet!</b>\n\n"
                "Delta Exchange API keys are not configured.\n"
                "Please configure them first using:\n"
                "<code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>",
                parse_mode=ParseMode.HTML,
            )
            return
        auto_trader.set_mode("live")
        await update.message.reply_text(
            "⚡ <b>Trading Mode Switched to LIVE!</b>\n\n"
            "⚠️ <i>Orders will now be executed on your Delta Exchange account with real capital.</i>\n"
            "Use <code>/balance</code> to check your wallet balance and <code>/positions</code> to inspect positions.",
            parse_mode=ParseMode.HTML,
        )
    elif req_mode in ("paper", "demo", "sim", "virtual"):
        auto_trader.set_mode("paper")
        await update.message.reply_text(
            "🎮 <b>Trading Mode Switched to PAPER!</b>\n\n"
            "All trades are simulated with virtual balance. No real funds are risked.",
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.message.reply_text(
            "Usage: <code>/mode paper</code> or <code>/mode live</code>",
            parse_mode=ParseMode.HTML,
        )


async def start_trade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Start automatic trading engine (/starttrade, /tradeon, /startbot, /autotrade on)."""
    chat_id = update.effective_chat.id
    if auto_trader.mode == "live" and not delta_client.is_configured():
        await update.message.reply_text(
            "⚠️ <b>Delta Exchange API keys are not set for LIVE mode!</b>\n\n"
            "Please configure your keys first using:\n"
            "<code>/setkey &lt;API_KEY&gt;</code> and <code>/setsecret &lt;API_SECRET&gt;</code>\n"
            "(or <code>/setkeys &lt;KEY&gt; &lt;SECRET&gt;</code>)\n\n"
            "Or switch to safe paper trading with $10,000 demo funds:\n"
            "<code>/mode paper</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    auto_trader.set_enabled(True)
    auto_trader.add_subscriber(chat_id)

    mode_badge = (
        "🎮 <b>PAPER SIMULATION</b> ($10,000 demo capital, zero risk)"
        if auto_trader.mode == "paper"
        else "⚡ <b>LIVE DELTA EXCHANGE</b> (Real account execution)"
    )

    msg = (
        "🟢 <b>AUTOMATIC TRADING ENGINE: STARTED!</b> 🚀\n\n"
        f"• <b>Status:</b> 🟢 <b>ACTIVE / RUNNING</b>\n"
        f"• <b>Execution Mode:</b> {mode_badge}\n"
        f"• <b>Account Balance:</b> <code>${auto_trader.balance:,.2f}</code>\n"
        "• <b>Scanned Markets:</b> 🪙 Bitcoin (#BTCUSD) & 🥇 Gold (#XAUTUSD)\n"
        "• <b>Timeframes:</b> 5m & 15m\n"
        "• <b>Strategy Rules:</b>\n"
        "  - Gautam Jha Daily Open (DO) color flip reactions\n"
        "  - Previous Day High (PDH) & Low (PDL) sweeps\n"
        "  - Momentum Break-and-Go & Level Continuations\n"
        "  - Automatic Stop Loss (SL) & dual Take Profit (TP1 1:1.5, TP2 1:2.5+)\n\n"
        "🔔 <i>You will receive instant alerts whenever trades open or exit.</i>\n\n"
        "🛑 <i>To stop automatic trading anytime: <code>/stoptrade</code> or <code>/autotrade off</code></i>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def stop_trade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Stop automatic trading engine (/stoptrade, /tradeoff, /stopbot, /autotrade off)."""
    auto_trader.set_enabled(False)
    open_count = len(auto_trader.get_open_positions())
    msg = (
        "🔴 <b>AUTOMATIC TRADING ENGINE: STOPPED & PAUSED.</b> 🛑\n\n"
        "No new automated trades will be entered.\n\n"
        f"• <b>Active Positions:</b> {open_count}\n"
        f"• <b>Balance:</b> <code>${auto_trader.balance:,.2f}</code> ({auto_trader.mode.upper()} mode)\n\n"
        "💡 <i>Existing open positions remain managed and will exit when SL or TP is reached.</i>\n"
        "To view or close positions manually: <code>/positions</code> or <code>/closeall</code>.\n"
        "To restart automatic trading: <code>/starttrade</code> or <code>/autotrade on</code>."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def autotrade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Enable or disable automated strategy execution (/autotrade [on|off])."""
    args = ctx.args or []
    if not args:
        chat_id = update.effective_chat.id
        status_str = "🟢 <b>ACTIVE / RUNNING</b>" if auto_trader.enabled else "🔴 <b>DISABLED / PAUSED</b>"
        mode_str = auto_trader.mode.upper()
        summary = auto_trader.get_account_summary()
        await update.message.reply_text(
            f"🤖 <b>Gautam Jha Automated Trading Engine</b>\n\n"
            f"• <b>Status:</b> {status_str}\n"
            f"• <b>Mode:</b> <code>{mode_str}</code>\n"
            f"• <b>Subscribed for Alerts:</b> {'Yes 🟢' if chat_id in auto_trader.subscribers else 'No'}\n"
            f"• <b>Open Positions:</b> {summary['open_positions_count']}\n"
            f"• <b>Balance:</b> <code>${summary['balance']:,.2f}</code>\n"
            f"• <b>Win Rate:</b> {summary['win_rate_pct']}%\n"
            f"• <b>Total Trades:</b> {summary['total_trades']}\n\n"
            "<b>Controls:</b>\n"
            "• <code>/starttrade</code> (or <code>/autotrade on</code>) — Start auto trading\n"
            "• <code>/stoptrade</code> (or <code>/autotrade off</code>) — Stop auto trading\n"
            "• <code>/mode paper</code> / <code>/mode live</code> — Switch mode",
            parse_mode=ParseMode.HTML,
        )
        return

    subcmd = args[0].lower().strip()
    if subcmd in ("on", "start", "enable", "1", "true"):
        await start_trade_cmd(update, ctx)
    elif subcmd in ("off", "stop", "disable", "0", "false"):
        await stop_trade_cmd(update, ctx)
    else:
        await update.message.reply_text("Usage: <code>/starttrade</code> or <code>/stoptrade</code>", parse_mode=ParseMode.HTML)


async def learn_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Self-learning and self-improving strategy dashboard (/learn [ai|reset|unsuppress <SETUP>]).
    Displays real-time performance, dynamic setup weights, auto-suppression, and drawdown controls (0 tokens).
    """
    args = ctx.args or []
    if args:
        subcmd = args[0].lower().strip()
        if subcmd == "ai":
            await update.message.reply_text("🤖 Generating token-efficient AI strategic reflection...", parse_mode=ParseMode.HTML)
            insight = auto_trader.get_learning_ai_insight(generate_compact_ai_insight)
            await update.message.reply_text(insight, parse_mode=ParseMode.HTML)
            return
        elif subcmd == "reset":
            auto_trader.reset_learning()
            await update.message.reply_text(
                "🔄 <b>Self-Learning Memory Reset!</b>\nAll learned weights, streaks, and setup statuses have been reset to default baseline.",
                parse_mode=ParseMode.HTML,
            )
            return
        elif subcmd in ("unsuppress", "enable", "restore") and len(args) > 1:
            target_setup = " ".join(args[1:])
            ok = auto_trader.learning_engine.unsuppress_setup(target_setup)
            if ok:
                await update.message.reply_text(
                    f"✅ Setup <b>{target_setup}</b> has been unsuppressed and restored to ACTIVE status.",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await update.message.reply_text(
                    f"⚠️ Setup <b>{target_setup}</b> not found in learning registry.",
                    parse_mode=ParseMode.HTML,
                )
            return

    # Default: 0-token instant local analytics
    report = auto_trader.get_learning_report()
    await update.message.reply_text(report, parse_mode=ParseMode.HTML)


async def insights_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View self-learning strategy insights (/insights)."""
    await learn_cmd(update, ctx)


async def news_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View breaking news & sentiment analysis (/news [SYMBOL] [ai])."""
    args = ctx.args or []
    sym = "BTCUSD"
    is_ai = False

    for a in args:
        a_clean = a.lower().strip()
        if a_clean == "ai":
            is_ai = True
        elif a_clean in ("btc", "bitcoin", "btcusd"):
            sym = "BTCUSD"
        elif a_clean in ("gold", "xau", "xaut", "xautusd"):
            sym = "XAUTUSD"
        elif a_clean in ("eth", "ethereum"):
            sym = "ETHUSD"
        else:
            sym = resolve_symbol(a)

    if is_ai:
        await update.message.reply_text("🤖 Generating token-efficient AI macro synthesis...", parse_mode=ParseMode.HTML)
        news_data = get_news_sentiment(sym)
        headlines_summary = "\n".join(f"- {a['title']} ({a['label']})" for a in news_data.get("articles", [])[:4])
        prompt = (
            f"Asset: {news_data['asset_name']}. Headlines:\n{headlines_summary}\n"
            "In 2-3 concise bullets (<60 words), state: 1) Core market driver, 2) Sentiment impact on price."
        )
        ai_resp = generate_compact_ai_insight(prompt, max_tokens=180)
        await update.message.reply_text(
            f"📰 <b>AI Macro Sentiment Synthesis:</b>\n\n{ai_resp}",
            parse_mode=ParseMode.HTML,
        )
        return

    report = auto_trader.get_news_report(sym)
    await update.message.reply_text(report, parse_mode=ParseMode.HTML)


async def btc_news_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Bitcoin news (/btcnews)."""
    ctx.args = ["BTCUSD"] + (ctx.args or [])
    await news_cmd(update, ctx)


async def gold_news_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Gold news (/goldnews)."""
    ctx.args = ["XAUTUSD"] + (ctx.args or [])
    await news_cmd(update, ctx)


async def orderbook_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View live L2 Order Book depth, imbalance & liquidity walls (/orderbook [SYMBOL])."""
    args = ctx.args or []
    sym = resolve_symbol(args[0]) if args else "BTCUSD"
    report = auto_trader.get_orderbook_report(sym)
    await update.message.reply_text(report, parse_mode=ParseMode.HTML)


async def btc_book_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Bitcoin order book (/btcbook)."""
    ctx.args = ["BTCUSD"]
    await orderbook_cmd(update, ctx)


async def gold_book_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Gold order book (/goldbook)."""
    ctx.args = ["XAUTUSD"]
    await orderbook_cmd(update, ctx)


async def amd_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Multi-Timeframe AMD Scalp Trading Command (1m, 5m, 15m).
    Analyzes Accumulation (A), Manipulation Judas Sweeps (M), and Distribution triggers (D)
    with automated SL, TP1, TP2, and capital risk management.
    Usage:
    • /amd [symbol] — 1m/5m/15m AMD Scalp analysis
    • /amd trade [symbol] — Execute AMD scalp setup immediately
    • /scalp [symbol]
    """
    args = ctx.args or []
    is_trade = False
    symbol_arg = DEFAULT_SYMBOL

    filtered_args = []
    for a in args:
        if a.lower().strip() in ("trade", "execute", "run", "buy", "sell"):
            is_trade = True
        else:
            filtered_args.append(a)

    if filtered_args:
        symbol_arg = filtered_args[0]

    sym = resolve_symbol(symbol_arg)

    try:
        if is_trade:
            await update.message.reply_text(
                f"⚡ <b>Analyzing 1m/5m/15m AMD Scalp setup for {sym}...</b>",
                parse_mode=ParseMode.HTML,
            )
            res = auto_trader.execute_amd_trade(sym, force=False)
            if res.get("status") == "executed":
                trade = res["trade"]
                side_tag = "🟢 LONG" if trade["side"] == "buy" else "🔴 SHORT"
                msg = (
                    f"🚀 <b>AMD SCALP TRADE EXECUTED!</b> 🎯\n\n"
                    f"🪙 <b>Symbol:</b> <code>#{trade['symbol']}</code>\n"
                    f"🚦 <b>Side:</b> {side_tag}\n"
                    f"📦 <b>Size:</b> <code>{trade['size']}</code>\n"
                    f"⚡ <b>Entry:</b> <code>${trade['entry_price']:,.2f}</code>\n"
                    f"🛑 <b>Stop Loss:</b> <code>${trade['sl_price']:,.2f}</code>\n"
                    f"🎯 <b>Take Profit 1:</b> <code>${trade['tp1_price']:,.2f}</code> (Range Target)\n"
                    f"🎯 <b>Take Profit 2:</b> <code>${trade['tp2_price']:,.2f}</code> (Expansion Target)\n"
                    f"⚖️ <b>Risk/Reward:</b> <code>{trade.get('rrr', '1:2')}</code>\n"
                    f"💡 <b>Strategy:</b> <code>{trade.get('strategy')}</code>\n"
                    f"⚙️ <b>Mode:</b> <code>{trade.get('mode', '').upper()}</code>\n"
                    f"🆔 <b>Position ID:</b> <code>{trade.get('position_id')}</code>"
                )
                await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
            else:
                reason = res.get("reason", "No executable trigger.")
                report = auto_trader.get_amd_report(sym)
                await update.message.reply_text(
                    f"⏸️ <b>AMD Scalp Execution Skipped:</b> {reason}\n\n{report}",
                    parse_mode=ParseMode.HTML,
                )
        else:
            report = auto_trader.get_amd_report(sym)
            await update.message.reply_text(report, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error in amd_cmd for {sym}: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error analyzing AMD scalp for {sym}: {e}")


async def btc_amd_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /btcamd -> Bitcoin 1m/5m/15m AMD Scalp."""
    ctx.args = ["BTCUSD"] + (ctx.args or [])
    await amd_cmd(update, ctx)


async def gold_amd_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /goldamd -> Gold 1m/5m/15m AMD Scalp."""
    ctx.args = ["XAUTUSD"] + (ctx.args or [])
    await amd_cmd(update, ctx)


async def multi_trade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /multitrade -> Multi-Trade configuration & positions status."""
    ctx.args = ["multi"] + (ctx.args or [])
    await trade_cmd(update, ctx)


async def confluence_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Evaluate all strategies combined (Gautam Jha + Candles + OrderBook + News + Self-Learning).
    Usage: /confluence [SYMBOL] or /confluence trade [SYMBOL]
    """
    args = ctx.args or []
    is_trade = False
    sym = "BTCUSD"

    filtered_args = []
    for a in args:
        if a.lower().strip() in ("trade", "execute", "run"):
            is_trade = True
        else:
            filtered_args.append(a)

    if filtered_args:
        sym = resolve_symbol(filtered_args[0])

    if is_trade:
        await update.message.reply_text(
            f"⚡ Evaluating master confluence and executing trade for <b>{sym}</b>...",
            parse_mode=ParseMode.HTML,
        )
        result = auto_trader.execute_confluence_trade(sym, force=False)
        if result.get("status") == "executed":
            trade = result["trade"]
            conf = result["confluence"]
            await update.message.reply_text(
                f"🚀 <b>CONFLUENCE TRADE EXECUTED!</b>\n\n"
                f"• <b>Contract:</b> <code>#{trade['symbol']}</code>\n"
                f"• <b>Mode:</b> <code>{trade['mode'].upper()}</code>\n"
                f"• <b>Side:</b> <b>{trade['side'].upper()}</b>\n"
                f"• <b>Size:</b> <code>{trade['size']}</code>\n"
                f"• <b>Entry Price:</b> <code>${trade['entry_price']:,.2f}</code>\n"
                f"• <b>Stop Loss:</b> <code>${trade['sl']:,.2f}</code>\n"
                f"• <b>Take Profit 1:</b> <code>${trade['tp1']:,.2f}</code> (1:1.5)\n"
                f"• <b>Take Profit 2:</b> <code>${trade['tp2']:,.2f}</code> (1:2.5+)\n"
                f"• <b>Confluence Score:</b> <code>{conf['confluence_score']}% {conf['bias_signal']}</code>\n\n"
                "<i>Position is now actively tracked by exit monitor!</i>",
                parse_mode=ParseMode.HTML,
            )
        else:
            reason = result.get("reason", "Confluence criteria not satisfied.")
            await update.message.reply_text(
                f"⏸️ <b>Confluence Trade Skipped:</b>\n{reason}\n\n"
                "Run <code>/confluence</code> to inspect individual strategy scores.",
                parse_mode=ParseMode.HTML,
            )
        return

    report = auto_trader.get_confluence_report(sym)
    await update.message.reply_text(report, parse_mode=ParseMode.HTML)


async def confluence_trade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Directly execute trade when master confluence confirms (/confluencetrade [SYMBOL])."""
    ctx.args = ["trade"] + (ctx.args or [])
    await confluence_cmd(update, ctx)


async def btc_confluence_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Bitcoin confluence analysis (/btcconfluence)."""
    ctx.args = ["BTCUSD"] + (ctx.args or [])
    await confluence_cmd(update, ctx)


async def gold_confluence_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut for Gold confluence analysis (/goldconfluence)."""
    ctx.args = ["XAUTUSD"] + (ctx.args or [])
    await confluence_cmd(update, ctx)


async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Single-card institutional command center dashboard (/status, /dashboard, /dash)."""
    msg = auto_trader.format_institutional_dashboard()
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def size_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Directly configure or check auto-trading lot size (/size, /lotsize [VAL] [SYMBOL])."""
    ctx.args = ["size"] + (ctx.args or [])
    await trade_cmd(update, ctx)


async def trade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Master Trading & Portfolio Multi-Working Hub.
    Usage:
    • /trade (portfolio & bot dashboard)
    • /trade status (detailed institutional dashboard)
    • /trade on | /trade off (start / stop automated trading)
    • /trade be [on|off] (toggle breakeven stop loss)
    • /trade trail [on|off|pct] (toggle trailing stop loss)
    • /trade live | /trade paper (switch mode)
    • /trade pos (view open positions & PnL)
    • /trade close <id|all> (close positions)
    • /trade bal (view wallet balances)
    • /trade confluence <sym> (execute master confluence trade)
    • /trade <sym> <buy|sell> [size] (manual trade)
    • /trade learn [ai|reset] (self-learning dashboard)
    """
    args = ctx.args or []
    if not args:
        summary = auto_trader.get_account_summary()
        status_str = "🟢 ACTIVE / RUNNING" if auto_trader.enabled else "🔴 DISABLED / PAUSED"
        mode_str = auto_trader.mode.upper()
        top_setup = auto_trader.learning_engine.get_top_setup()
        top_setup_str = f"{top_setup['name']} ({top_setup['win_rate_pct']}%)" if top_setup else "Calibrating..."

        msg = (
            "💼 <b>MASTER TRADING & PORTFOLIO HUB</b>\n\n"
            f"• <b>Bot Status:</b> {status_str}\n"
            f"• <b>Execution Mode:</b> <code>{mode_str}</code> "
            + ("(Connected to Delta Exchange ⚡)" if auto_trader.mode == "live" else "($10,000 Paper Demo 🎮)") + "\n"
            f"• <b>Account Balance:</b> <code>${auto_trader.balance:,.2f}</code>\n"
            f"• <b>Open Positions:</b> <code>{summary['open_positions_count']}/{summary.get('max_open_positions', 5)}</code>\n"
            f"• <b>Risk Per Trade:</b> <code>{summary.get('risk_per_trade_pct', 0.015)*100:.1f}%</code>\n"
            f"• <b>Win Rate:</b> <code>{summary['win_rate_pct']}%</code> ({summary['total_trades']} closed trades)\n"
            f"• <b>Top Learned Setup:</b> <code>{top_setup_str}</code>\n\n"
            "⚡ <b>1-Command Working Modes:</b>\n"
            "• <code>/trade status</code> — Comprehensive institutional dashboard 🏛️\n"
            "• <code>/trade on</code> (or <code>/trade start</code>) — Start auto trading 🟢\n"
            "• <code>/trade off</code> (or <code>/trade stop</code>) — Stop / pause auto trading 🔴\n"
            "• <code>/trade be [on|off]</code> — Toggle Breakeven SL on TP1 🛡️\n"
            "• <code>/trade trail [on|off]</code> — Toggle Dynamic Trailing SL ⚡\n"
            "• <code>/trade size &lt;VAL&gt;</code> — Set Auto-Trade Lot Size (e.g. <code>/trade size 0.05</code>) 🎯\n"
            "• <code>/trade amd [btc|gold]</code> — 1m/5m/15m AMD Scalp execution\n"
            "• <code>/trade maxpos &lt;N&gt;</code> — Set max concurrent positions (e.g. 5)\n"
            "• <code>/trade risk &lt;PCT&gt;</code> — Set capital risk per trade (e.g. 1.5%)\n"
            "• <code>/trade multi [on|off]</code> — Toggle multi-trade per symbol\n"
            "• <code>/trade live</code> / <code>/trade paper</code> — Switch execution mode\n"
            "• <code>/trade pos</code> — View active open positions & live PnL\n"
            "• <code>/trade close all</code> — Close all positions at market\n"
            "• <code>/trade close &lt;ID&gt;</code> — Close position by ID\n"
            "• <code>/trade bal</code> — Delta Exchange & Paper wallet balances\n"
            "• <code>/trade btc buy 1</code> — Buy Bitcoin with auto SL/TP\n"
            "• <code>/trade gold sell 0.05</code> — Short Gold with auto SL/TP\n"
            "• <code>/trade confluence btc</code> — Execute master confluence trade\n"
            "• <code>/trade learn</code> — Self-learning performance & insights"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    sub = args[0].lower().strip()
    rest = args[1:]

    # Sub-action routing
    if sub in ("on", "start", "enable"):
        await start_trade_cmd(update, ctx)
    elif sub in ("off", "stop", "pause", "disable"):
        await stop_trade_cmd(update, ctx)
    elif sub in ("live", "real"):
        ctx.args = ["live"]
        await mode_cmd(update, ctx)
    elif sub in ("paper", "demo", "sim"):
        ctx.args = ["paper"]
        await mode_cmd(update, ctx)
    elif sub in ("pos", "position", "positions", "open"):
        await positions_cmd(update, ctx)
    elif sub in ("close", "closeposition", "closeall"):
        if rest and rest[0].lower() == "all":
            await close_all_cmd(update, ctx)
        elif rest:
            ctx.args = [rest[0]]
            await close_position_cmd(update, ctx)
        else:
            await close_all_cmd(update, ctx)
    elif sub in ("bal", "balance", "wallet"):
        await balance_cmd(update, ctx)
    elif sub in ("learn", "learning", "insights"):
        ctx.args = rest
        await learn_cmd(update, ctx)
    elif sub in ("confluence", "combine", "master"):
        sym = rest[0] if rest else "BTCUSD"
        ctx.args = ["trade", sym]
        await confluence_cmd(update, ctx)
    elif sub in ("amd", "scalp", "po3"):
        ctx.args = rest
        await amd_cmd(update, ctx)
    elif sub in ("maxpos", "limit", "maxpositions"):
        if rest:
            try:
                cnt = int(rest[0])
                new_cnt = auto_trader.set_max_positions(cnt)
                await update.message.reply_text(f"✅ Max concurrent open positions set to <b>{new_cnt}</b>.", parse_mode=ParseMode.HTML)
            except Exception as e:
                await update.message.reply_text(f"⚠️ Invalid number: {e}")
        else:
            await update.message.reply_text(f"📊 Current max concurrent positions: <b>{auto_trader.data.get('max_open_positions', 5)}</b>\nTo update: <code>/trade maxpos &lt;N&gt;</code>", parse_mode=ParseMode.HTML)
    elif sub in ("risk", "riskpct"):
        if rest:
            try:
                raw_pct = float(rest[0].replace("%", ""))
                dec_pct = raw_pct / 100.0 if raw_pct >= 1.0 else raw_pct
                new_pct = auto_trader.set_risk_per_trade(dec_pct)
                await update.message.reply_text(f"✅ Capital risk per trade set to <b>{new_pct*100:.1f}%</b>.", parse_mode=ParseMode.HTML)
            except Exception as e:
                await update.message.reply_text(f"⚠️ Invalid risk percent: {e}")
        else:
            pct = auto_trader.data.get("risk_per_trade_pct", 0.015)
            await update.message.reply_text(f"🛡️ Current capital risk per trade: <b>{pct*100:.1f}%</b>\nTo update: <code>/trade risk &lt;PCT%&gt;</code>", parse_mode=ParseMode.HTML)
    elif sub in ("status", "dash", "dashboard"):
        await status_cmd(update, ctx)
    elif sub in ("be", "breakeven"):
        if rest and rest[0].lower() in ("off", "disable", "false", "0"):
            auto_trader.set_auto_breakeven(False)
            await update.message.reply_text("⚪ <b>Breakeven Stop-Loss DISABLED.</b>\nTrades will exit completely at TP1.", parse_mode=ParseMode.HTML)
        elif rest and rest[0].lower() in ("on", "enable", "true", "1"):
            auto_trader.set_auto_breakeven(True)
            await update.message.reply_text("🛡️ <b>Breakeven Stop-Loss ENABLED 🟢</b>\nWhen TP1 is hit, Stop-Loss automatically moves to Entry price to lock in a 100% risk-free trade while runner continues towards TP2.", parse_mode=ParseMode.HTML)
        else:
            curr = "🟢 ENABLED" if auto_trader.data.get("auto_breakeven") else "⚪ DISABLED"
            await update.message.reply_text(f"🛡️ Breakeven Protection: <b>{curr}</b>\n• To enable: <code>/trade be on</code>\n• To disable: <code>/trade be off</code>", parse_mode=ParseMode.HTML)
    elif sub in ("trail", "trailing", "trailingstop"):
        if rest and rest[0].lower() in ("off", "disable", "false", "0"):
            auto_trader.set_trailing_sl(False)
            await update.message.reply_text("⚪ <b>Dynamic Trailing Stop-Loss DISABLED.</b>", parse_mode=ParseMode.HTML)
        else:
            pct = None
            if rest and rest[0].replace("%", "").replace(".", "").isdigit():
                try:
                    raw = float(rest[0].replace("%", ""))
                    pct = raw / 100.0 if raw >= 1.0 else raw
                except Exception:
                    pass
            res = auto_trader.set_trailing_sl(True, pct=pct)
            await update.message.reply_text(f"⚡ <b>Dynamic Trailing Stop-Loss ENABLED 🟢</b>\nTrailing offset: <b>{res['trailing_pct']*100:.1f}%</b> behind peak price.", parse_mode=ParseMode.HTML)
    elif sub in ("multi", "multitrade", "multiple"):
        if rest and rest[0].lower() in ("off", "disable", "false", "0"):
            val = auto_trader.set_allow_multiple_per_symbol(False)
        else:
            val = auto_trader.set_allow_multiple_per_symbol(True)
        status_text = "🟢 ENABLED (Multiple trades per symbol allowed)" if val else "🔴 DISABLED (1 trade per symbol limit)"
        await update.message.reply_text(f"🔄 Multi-Trade Per Symbol: <b>{status_text}</b>", parse_mode=ParseMode.HTML)
    elif sub in ("size", "lotsize", "lot", "qty"):
        # Auto-trade lot size configuration & overrides
        if not rest:
            cur_size = auto_trader.get_lot_size()
            cur_mode = auto_trader.data.get("lot_size_mode", "custom").upper()
            sym_overrides = auto_trader.data.get("symbol_lot_sizes", {})
            sym_text = "\n".join([f"  • <b>{k}:</b> <code>{v}</code>" for k, v in sym_overrides.items()]) if sym_overrides else "  • <i>None (all symbols follow global size)</i>"

            msg = (
                "🎯 <b>AUTO-TRADE LOT SIZE CONFIGURATION</b>\n\n"
                f"• <b>Global Lot Size:</b> <code>{cur_size}</code>\n"
                f"• <b>Sizing Mode:</b> <code>{cur_mode}</code> (Bot strictly follows this lot size)\n"
                f"• <b>Per-Symbol Overrides:</b>\n{sym_text}\n\n"
                "⚡ <b>Commands to Change Lot Size:</b>\n"
                "• <code>/trade size &lt;VAL&gt;</code> — Set global lot size (e.g. <code>/trade size 0.05</code>)\n"
                "• <code>/trade size &lt;SYM&gt; &lt;VAL&gt;</code> — Set per-pair lot size (e.g. <code>/trade size BTC 0.01</code>)\n"
                "• <code>/trade size &lt;SYM&gt; reset</code> — Clear override for pair\n"
                "• <code>/size &lt;VAL&gt;</code> — Fast alias"
            )
            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
            return

        def _try_float(val_str):
            try:
                return float(str(val_str).replace("x", "").replace("lots", "").replace("lot", "").strip())
            except (ValueError, AttributeError):
                return None

        # Mode configuration (/trade size mode custom / risk_pct)
        if rest[0].lower() in ("mode", "type") and len(rest) > 1:
            try:
                new_m = auto_trader.set_lot_size_mode(rest[1])
                await update.message.reply_text(f"✅ Lot size mode updated to <b>{new_m.upper()}</b>.", parse_mode=ParseMode.HTML)
            except Exception as e:
                await update.message.reply_text(f"⚠️ Invalid mode: {e}")
            return

        # Case 1: Single numeric argument -> /trade size 0.05
        val_first = _try_float(rest[0])
        if val_first is not None and len(rest) == 1:
            new_sz = auto_trader.set_lot_size(val_first)
            await update.message.reply_text(
                f"✅ <b>Global Auto-Trade Lot Size set to {new_sz}!</b> 🎯\n"
                f"All auto-trading scanners (AMD Scalp, Master Confluence, Gautam Jha, Candle Entry) and manual trades will strictly follow this lot size.",
                parse_mode=ParseMode.HTML,
            )
            return

        # Case 2: Size first, symbol second -> /trade size 0.01 btc
        if val_first is not None and len(rest) > 1:
            sym = resolve_symbol(rest[1])
            new_sz = auto_trader.set_lot_size(val_first, symbol=sym)
            await update.message.reply_text(
                f"✅ <b>Lot Size for {sym} set to {new_sz}!</b> 🎯\n"
                f"Auto-trader will strictly use <code>{new_sz}</code> for all trades on {sym}.",
                parse_mode=ParseMode.HTML,
            )
            return

        # Case 3: Symbol first -> /trade size btc 0.01 OR /trade size btc reset
        sym_candidate = resolve_symbol(rest[0])
        if len(rest) > 1:
            if rest[1].lower() in ("reset", "clear", "remove", "default"):
                removed = auto_trader.remove_symbol_lot_size(sym_candidate)
                if removed:
                    await update.message.reply_text(f"✅ Lot size override for <b>{sym_candidate}</b> cleared. Reverted to global: <code>{auto_trader.get_lot_size()}</code>.", parse_mode=ParseMode.HTML)
                else:
                    await update.message.reply_text(f"ℹ️ No override was active for <b>{sym_candidate}</b>.", parse_mode=ParseMode.HTML)
                return

            val_sec = _try_float(rest[1])
            if val_sec is not None:
                new_sz = auto_trader.set_lot_size(val_sec, symbol=sym_candidate)
                await update.message.reply_text(
                    f"✅ <b>Lot Size for {sym_candidate} set to {new_sz}!</b> 🎯\n"
                    f"Auto-trader will strictly use <code>{new_sz}</code> for all trades on {sym_candidate}.",
                    parse_mode=ParseMode.HTML,
                )
                return

        # Query single symbol: e.g. /trade size btc
        sym_sz = auto_trader.get_lot_size(sym_candidate)
        await update.message.reply_text(f"📊 Active lot size for <b>{sym_candidate}</b>: <code>{sym_sz}</code>\nTo change: <code>/trade size {sym_candidate} &lt;VAL&gt;</code>", parse_mode=ParseMode.HTML)
    else:
        # Manual Trade execution:
        # Format 1: /trade <symbol> <side> [size]
        # Format 2: /trade <side> <symbol> [size]
        if sub in ("buy", "sell", "long", "short"):
            side = sub
            if not rest:
                await update.message.reply_text("Usage: <code>/trade buy &lt;SYMBOL&gt; [size]</code>", parse_mode=ParseMode.HTML)
                return
            sym_raw = rest[0]
            size_arg = rest[1] if len(rest) > 1 else None
        else:
            sym_raw = sub
            if not rest:
                await update.message.reply_text("Usage: <code>/trade &lt;SYMBOL&gt; &lt;buy|sell&gt; [size]</code>", parse_mode=ParseMode.HTML)
                return
            side = rest[0].lower().strip()
            size_arg = rest[1] if len(rest) > 1 else None

        if side not in ("buy", "sell", "long", "short"):
            await update.message.reply_text("⚠️ Action must be <code>buy</code> or <code>sell</code>.", parse_mode=ParseMode.HTML)
            return

        if side == "long":
            side = "buy"
        elif side == "short":
            side = "sell"

        size = None
        if size_arg:
            try:
                size = float(size_arg)
            except ValueError:
                await update.message.reply_text("⚠️ Invalid size. Please specify a numeric amount.", parse_mode=ParseMode.HTML)
                return

        symbol = resolve_symbol(sym_raw)
        await update.message.reply_text(f"⏳ Executing {side.upper()} order for {symbol} ({auto_trader.mode.upper()} mode)...")

        res = auto_trader.execute_trade(
            symbol=symbol,
            side=side,
            size=size,
            trade_type="manual",
        )

        if res.get("status") in ("executed", "simulated", "filled", "open"):
            emoji = "🟢 LONG" if side == "buy" else "🔴 SHORT"
            msg = (
                f"✅ <b>TRADE EXECUTED!</b>\n\n"
                f"• <b>Symbol:</b> <code>#{symbol}</code>\n"
                f"• <b>Action:</b> {emoji}\n"
                f"• <b>Size:</b> <code>{res.get('size')}</code>\n"
                f"• <b>Entry Price:</b> <code>${res.get('entry_price', 0):,.2f}</code>\n"
                f"• <b>Stop Loss:</b> <code>${res.get('sl_price', 0):,.2f}</code>\n"
                f"• <b>Take Profit 1:</b> <code>${res.get('tp1_price', 0):,.2f}</code>\n"
                f"• <b>Take Profit 2:</b> <code>${res.get('tp2_price', 0):,.2f}</code>\n"
                f"• <b>Mode:</b> <code>{res.get('mode', '').upper()}</code>\n"
                f"• <b>Position ID:</b> <code>{res.get('position_id', 'N/A')}</code>\n\n"
                "📊 <i>Track with <code>/trade pos</code> or close with <code>/trade close {id}</code></i>"
            )
            await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        else:
            err = res.get("error", "Unknown error")
            await update.message.reply_text(
                f"❌ <b>Trade Execution Failed:</b> {err}\n\n"
                f"Mode: <code>{auto_trader.mode.upper()}</code>",
                parse_mode=ParseMode.HTML,
            )


async def positions_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View active positions and PnL (/positions)."""
    if auto_trader.mode == "live":
        live_res = delta_client.get_positions()
        live_positions = live_res.get("result", []) if live_res.get("success") else []
        local_positions = auto_trader.get_open_positions()

        if not live_positions and not local_positions:
            await update.message.reply_text(
                "📋 <b>No Open Positions (LIVE Mode)</b>\n\n"
                "Use <code>/trade &lt;SYMBOL&gt; &lt;buy|sell&gt;</code> or <code>/autotrade on</code> to open trades.",
                parse_mode=ParseMode.HTML,
            )
            return

        lines = [f"⚡ <b>Open Positions (LIVE Mode):</b>\n"]
        for p in live_positions:
            prod_id = p.get("product_id")
            size = p.get("size", 0)
            entry = float(p.get("entry_price") or 0)
            mark = float(p.get("mark_price") or 0)
            upnl = float(p.get("unrealized_pnl") or 0)
            side = "LONG" if size > 0 else "SHORT"
            sym = "BTCUSD" if prod_id == 27 else ("XAUTUSD" if prod_id == 131253 else f"Product {prod_id}")
            lines.append(
                f"• <b>{sym}</b> [{side}] Size: {abs(size)}\n"
                f"  Entry: <code>${entry:,.2f}</code> | Current: <code>${mark:,.2f}</code>\n"
                f"  PnL: <b>{'🟢' if upnl >= 0 else '🔴'} ${upnl:+,.2f}</b>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
        return

    # Paper mode
    positions = auto_trader.get_open_positions()
    if not positions:
        await update.message.reply_text(
            "📋 <b>No Open Positions (PAPER Mode)</b>\n\n"
            f"Virtual Balance: <code>${auto_trader.balance:,.2f}</code>\n"
            "Use <code>/trade BTCUSD buy</code> or <code>/autotrade on</code> to get started!",
            parse_mode=ParseMode.HTML,
        )
        return

    lines = [f"🎮 <b>Active Paper Positions ({len(positions)}):</b>\n"]
    for p in positions:
        sym = p["symbol"]
        side = p["side"].upper()
        size = p["size"]
        entry = p["entry_price"]
        sl = p.get("sl_price", 0)
        tp1 = p.get("tp1_price", 0)
        tp2 = p.get("tp2_price", 0)
        pid = p["id"]

        try:
            t = get_ticker(sym)
            curr = t["mark_price"] or t["close"]
            pnl_mult = 1 if side == "BUY" else -1
            pnl_usd = (curr - entry) * size * pnl_mult
            pnl_pct = ((curr - entry) / entry * 100.0) * pnl_mult
            pnl_str = f"{'🟢' if pnl_usd >= 0 else '🔴'} ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)"
            curr_str = f"${curr:,.2f}"
        except Exception:
            curr_str = "N/A"
            pnl_str = "N/A"

        lines.append(
            f"• <b>{sym}</b> [{side}] | Size: <code>{size}</code>\n"
            f"  Entry: <code>${entry:,.2f}</code> | Current: <code>{curr_str}</code>\n"
            f"  Unrealized PnL: <b>{pnl_str}</b>\n"
            f"  Targets: SL <code>${sl:,.2f}</code> | TP1 <code>${tp1:,.2f}</code> | TP2 <code>${tp2:,.2f}</code>\n"
            f"  ID: <code>{pid}</code> (Close: <code>/closeposition {pid}</code>)\n"
        )

    lines.append("💡 <i>Close all with <code>/closeall</code></i>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def close_position_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Close an open position by ID (/closeposition <id>)."""
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "Usage: <code>/closeposition &lt;position_id&gt;</code>\n"
            "View active IDs with <code>/positions</code>, or close all with <code>/closeall</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    pid = args[0].strip()
    res = auto_trader.close_position(pid)
    if res.get("status") == "closed":
        pnl = res.get("pnl", 0.0)
        pnl_pct = res.get("pnl_pct", 0.0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        await update.message.reply_text(
            f"✅ <b>Position Closed:</b> <code>{pid}</code>\n\n"
            f"• <b>Symbol:</b> <code>{res.get('symbol')}</code>\n"
            f"• <b>Exit Price:</b> <code>${res.get('exit_price', 0):,.2f}</code>\n"
            f"• <b>Realized PnL:</b> {emoji} <b>${pnl:+,.2f} ({pnl_pct:+.2f}%)</b>\n"
            f"• <b>New Balance:</b> <code>${auto_trader.balance:,.2f}</code>",
            parse_mode=ParseMode.HTML,
        )
    else:
        err = res.get("error", "Position not found or could not be closed")
        await update.message.reply_text(f"⚠️ {err}", parse_mode=ParseMode.HTML)


async def close_all_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Close all open positions (/closeall)."""
    closed = auto_trader.close_all_positions()
    if not closed:
        await update.message.reply_text("📋 No open positions to close.", parse_mode=ParseMode.HTML)
        return

    total_pnl = sum(p.get("pnl", 0.0) for p in closed)
    emoji = "🟢" if total_pnl >= 0 else "🔴"
    await update.message.reply_text(
        f"✅ <b>Closed {len(closed)} Position(s)!</b>\n\n"
        f"• <b>Total Realized PnL:</b> {emoji} <b>${total_pnl:+,.2f}</b>\n"
        f"• <b>New Balance:</b> <code>${auto_trader.balance:,.2f}</code>",
        parse_mode=ParseMode.HTML,
    )


async def balance_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View account balances and equity (/balance)."""
    summary = auto_trader.get_account_summary()

    if auto_trader.mode == "live":
        if not delta_client.is_configured():
            await update.message.reply_text(
                "⚠️ Delta Exchange API keys are not configured. Run <code>/setkeys &lt;KEY&gt; &lt;SECRET&gt;</code>.",
                parse_mode=ParseMode.HTML,
            )
            return

        bal_res = delta_client.get_wallet_balances()
        if bal_res.get("success"):
            balances = bal_res.get("result", [])
            lines = ["⚡ <b>Delta Exchange Wallet Balances (LIVE)</b>\n"]
            if balances:
                for b in balances:
                    asset = b.get("asset_symbol", "USDT")
                    bal = float(b.get("balance", 0))
                    avail = float(b.get("available_balance", 0))
                    lines.append(f"• <b>{asset}:</b> {bal:,.4f} (Avail: {avail:,.4f})")
            else:
                lines.append("• No balances returned.")
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return
        else:
            err = bal_res.get("error", {}).get("message") or bal_res.get("error", "Failed to fetch balances")
            await update.message.reply_text(f"⚠️ Error fetching Delta Exchange balances: {err}", parse_mode=ParseMode.HTML)
            return

    # Paper mode
    pnl = summary["total_realized_pnl"]
    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
    msg = (
        f"🎮 <b>Paper Trading Account Balance</b>\n\n"
        f"• <b>Cash Balance:</b> <code>${summary['balance']:,.2f}</code>\n"
        f"• <b>Initial Balance:</b> <code>${summary['initial_balance']:,.2f}</code>\n"
        f"• <b>Estimated Equity:</b> <code>${summary['equity']:,.2f}</code>\n"
        f"• <b>Total Realized PnL:</b> {pnl_emoji} <b>${pnl:+,.2f}</b>\n"
        f"• <b>Open Positions:</b> {summary['open_positions_count']}\n"
        f"• <b>Win Rate:</b> {summary['win_rate_pct']}%\n"
        f"• <b>Total Trades:</b> {summary['total_trades']}\n\n"
        "💡 <i>To trade with real funds on Delta Exchange:</i>\n"
        "1. <code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>\n"
        "2. <code>/mode live</code>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def orders_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View open orders on Delta Exchange (/orders)."""
    if auto_trader.mode != "live":
        open_pos = auto_trader.get_open_positions()
        if not open_pos:
            await update.message.reply_text("📋 No active paper orders or positions.", parse_mode=ParseMode.HTML)
            return
        msg = f"🎮 <b>Paper Trading Active Positions ({len(open_pos)}):</b>\n\n"
        for p in open_pos:
            msg += f"• #{p['symbol']} [{p['side'].upper()}] Entry: ${p['entry_price']:,.2f} | SL: ${p.get('sl_price', 0):,.2f} | TP: ${p.get('tp1_price', 0):,.2f}\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
        return

    if not delta_client.is_configured():
        await update.message.reply_text("⚠️ Delta keys not set. Run <code>/setkeys &lt;KEY&gt; &lt;SECRET&gt;</code>.", parse_mode=ParseMode.HTML)
        return

    orders_res = delta_client.get_open_orders()
    if orders_res.get("success"):
        orders = orders_res.get("result", [])
        if not orders:
            await update.message.reply_text("📋 No open orders on Delta Exchange.", parse_mode=ParseMode.HTML)
            return
        lines = [f"⚡ <b>Open Orders on Delta Exchange ({len(orders)}):</b>\n"]
        for o in orders:
            lines.append(
                f"• Order #{o.get('id')}: {o.get('order_type')} {o.get('side')} "
                f"Size: {o.get('size')} Price: ${float(o.get('limit_price') or 0):,.2f}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    else:
        err = orders_res.get("error", "Failed to retrieve orders")
        await update.message.reply_text(f"⚠️ Error: {err}", parse_mode=ParseMode.HTML)


async def cancel_orders_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Cancel open orders on Delta Exchange (/cancelorders [symbol])."""
    if auto_trader.mode != "live":
        await update.message.reply_text(
            "ℹ️ In paper mode, use <code>/closeall</code> or <code>/closeposition &lt;id&gt;</code> to close positions.",
            parse_mode=ParseMode.HTML,
        )
        return

    if not delta_client.is_configured():
        await update.message.reply_text("⚠️ Delta keys not set.", parse_mode=ParseMode.HTML)
        return

    args = ctx.args or []
    prod_id = None
    if args:
        sym = resolve_symbol(args[0])
        prod_id = delta_client.get_product_id(sym)

    res = delta_client.cancel_all_orders(product_id=prod_id)
    if res.get("success"):
        await update.message.reply_text("✅ All open orders on Delta Exchange cancelled successfully.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(f"⚠️ Cancel failed: {res.get('error')}", parse_mode=ParseMode.HTML)


# ==================== Background Tasks ====================

async def price_alert_loop(application):
    """Periodically check active price alerts."""
    logger.info("Price alert background loop started.")
    while True:
        try:
            alerts = alert_manager.price_alerts
            if alerts:
                distinct_symbols = set(a["symbol"] for a in alerts)
                prices = {}
                for sym in distinct_symbols:
                    try:
                        t = get_ticker(sym)
                        prices[sym] = t["mark_price"] or t["close"]
                    except Exception as err:
                        logger.warning(f"Error fetching ticker for alert {sym}: {err}")

                triggered = alert_manager.check_price_alerts(prices)
                for alert in triggered:
                    chat_id = alert["chat_id"]
                    sym = alert["symbol"]
                    target = alert["target_price"]
                    trig_price = alert.get("trigger_price", target)
                    created_pr = alert.get("created_price", 0.0)
                    cond = alert["condition"]

                    cond_text = "Crossed Above or Reached" if cond in (">=", ">") else "Dropped Below or Reached"
                    diff_pct = ((trig_price - created_pr) / created_pr * 100.0) if created_pr > 0 else 0.0

                    msg = (
                        f"🚨 <b>PRICE ALERT TRIGGERED!</b> 🚨\n\n"
                        f"🪙 <b>Symbol:</b> <code>#{sym}</code>\n"
                        f"🎯 <b>Target:</b> <code>${target:,.2f}</code>\n"
                        f"⚡ <b>Current Price:</b> <code>${trig_price:,.2f}</code>\n"
                        f"🔔 <b>Condition:</b> {cond_text} <code>${target:,.2f}</code>\n"
                        f"📈 <b>Price when set:</b> <code>${created_pr:,.2f}</code> ({diff_pct:+.2f}%)\n\n"
                        f"🔍 <i>Check <code>/levels {sym}</code> or <code>/entry {sym}</code> for trade setup!</i>"
                    )
                    try:
                        await application.bot.send_message(
                            chat_id=chat_id,
                            text=msg,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception as send_err:
                        logger.error(f"Failed to send price alert to {chat_id}: {send_err}")

        except Exception as e:
            logger.error(f"Error in price_alert_loop: {e}")

        await asyncio.sleep(12)


async def entry_alert_loop(application):
    """Periodically check 1m, 5m, 15m candle entry setups for watched symbols."""
    logger.info("Candle entry alert background scanner started.")
    while True:
        try:
            watchers = alert_manager.get_all_entry_watchers()
            for watcher in watchers:
                chat_id = watcher["chat_id"]
                sym = watcher["symbol"]
                tfs = watcher.get("timeframes", ["1m", "5m", "15m"])

                for tf in tfs:
                    try:
                        analysis = get_candle_entry(sym, timeframe=tf)
                        if analysis.get("has_setup") and analysis.get("candle_time"):
                            candle_time = analysis["candle_time"]
                            last_alert = watcher.get("last_alert_time", {}).get(tf, 0)

                            if candle_time > last_alert:
                                alert_manager.update_watcher_alert_time(chat_id, sym, tf, candle_time)

                                tp = analysis.get("trade_plan", {})
                                sig = analysis.get("signal", "ENTRY SETUP")
                                emoji = "🟢" if "BUY" in sig or "LONG" in sig else "🔴"
                                reasons_str = ", ".join(analysis.get("reasons", []))

                                msg = (
                                    f"🎯 <b>CANDLE ENTRY ALERT ({tf.upper()})!</b> 🎯\n\n"
                                    f"🪙 <b>Symbol:</b> <code>#{sym}</code>\n"
                                    f"⏱️ <b>Timeframe:</b> <code>{tf.upper()}</code>\n"
                                    f"🚦 <b>Signal:</b> {emoji} <b>{sig}</b>\n"
                                    f"🕯️ <b>Pattern:</b> {analysis.get('pattern', 'N/A')}\n"
                                    f"📊 <b>Technical:</b> RSI <code>{analysis.get('rsi')}</code> | Vol <code>{analysis.get('vol_ratio')}x</code>\n"
                                    f"🔍 <b>Triggers:</b> {reasons_str}\n\n"
                                    f"📋 <b>Suggested Trade Plan:</b>\n"
                                    f"• <b>Entry:</b> <code>${tp.get('entry', 0):,.2f}</code>\n"
                                    f"• <b>Stop Loss:</b> <code>${tp.get('sl', 0):,.2f}</code> (Risk: ${tp.get('risk', 0):,.2f})\n"
                                    f"• <b>Take Profit 1:</b> <code>${tp.get('tp1', 0):,.2f}</code> (1:1.5)\n"
                                    f"• <b>Take Profit 2:</b> <code>${tp.get('tp2', 0):,.2f}</code> (1:2.5)\n"
                                    f"• <b>R:R Ratio:</b> {tp.get('rrr', '1:1.5+')}\n\n"
                                    f"⚠️ <i>Always apply position sizing and risk management!</i>"
                                )
                                try:
                                    await application.bot.send_message(
                                        chat_id=chat_id,
                                        text=msg,
                                        parse_mode=ParseMode.HTML,
                                    )
                                except Exception as send_err:
                                    logger.error(f"Failed to send entry alert to {chat_id}: {send_err}")
                    except Exception as tf_err:
                        logger.warning(f"Error checking {sym} {tf}: {tf_err}")

        except Exception as e:
            logger.error(f"Error in entry_alert_loop: {e}")

        await asyncio.sleep(25)


async def auto_trade_loop(application):
    """Background engine: monitors open positions for exits and auto-executes high-probability setups."""
    logger.info("Automated trading background loop started.")
    symbols_to_scan = auto_trader.data.get("symbols", ["BTCUSD", "XAUTUSD"])

    while True:
        try:
            # 1. Check open positions for Stop Loss or Take Profit triggers
            exits = auto_trader.check_open_positions_for_exits()
            for exit_info in exits:
                pnl = exit_info.get("pnl", 0.0)
                pnl_pct = exit_info.get("pnl_pct", 0.0)
                emoji = "🟢 TAKE PROFIT HIT! 🚀" if pnl >= 0 else "🛑 STOP LOSS TRIGGERED"
                msg = (
                    f"{emoji}\n\n"
                    f"🪙 <b>Symbol:</b> <code>#{exit_info.get('symbol')}</code>\n"
                    f"⚡ <b>Reason:</b> <code>{exit_info.get('exit_reason')}</code>\n"
                    f"🎯 <b>Exit Price:</b> <code>${exit_info.get('exit_price', 0):,.2f}</code>\n"
                    f"📈 <b>Entry Price:</b> <code>${exit_info.get('entry_price', 0):,.2f}</code>\n"
                    f"💰 <b>Realized PnL:</b> <b>${pnl:+,.2f} ({pnl_pct:+.2f}%)</b>\n"
                    f"💼 <b>Balance:</b> <code>${auto_trader.balance:,.2f}</code> ({auto_trader.mode.upper()} mode)"
                )
                for chat_id in auto_trader.subscribers:
                    try:
                        await application.bot.send_message(
                            chat_id=chat_id,
                            text=msg,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception as send_err:
                        logger.warning(f"Failed to send exit notification to {chat_id}: {send_err}")

            # 2. If autotrade is enabled, scan for high-probability setups
            if auto_trader.enabled:
                new_trades = auto_trader.scan_and_auto_trade(symbols=symbols_to_scan)
                for tr in new_trades:
                    side_emoji = "🟢 LONG" if tr.get("side") == "buy" else "🔴 SHORT"
                    trade_msg = (
                        f"🤖 <b>AUTO-TRADE EXECUTED!</b> 🎯\n\n"
                        f"🪙 <b>Symbol:</b> <code>#{tr.get('symbol')}</code>\n"
                        f"🚦 <b>Side:</b> {side_emoji}\n"
                        f"📦 <b>Size:</b> <code>{tr.get('size')}</code>\n"
                        f"⚡ <b>Entry:</b> <code>${tr.get('entry_price', 0):,.2f}</code>\n"
                        f"🛑 <b>Stop Loss:</b> <code>${tr.get('sl_price', 0):,.2f}</code>\n"
                        f"🎯 <b>Take Profit 1:</b> <code>${tr.get('tp1_price', 0):,.2f}</code>\n"
                        f"🎯 <b>Take Profit 2:</b> <code>${tr.get('tp2_price', 0):,.2f}</code>\n"
                        f"💡 <b>Strategy:</b> <code>{tr.get('strategy', 'Auto Execution')}</code> ({tr.get('reason')})\n"
                        f"⚙️ <b>Mode:</b> <code>{tr.get('mode', '').upper()}</code>\n"
                        f"🆔 <b>Position ID:</b> <code>{tr.get('position_id')}</code>"
                    )
                    for chat_id in auto_trader.subscribers:
                        try:
                            await application.bot.send_message(
                                chat_id=chat_id,
                                text=trade_msg,
                                parse_mode=ParseMode.HTML,
                            )
                        except Exception as send_err:
                            logger.warning(f"Failed to send auto-trade notification to {chat_id}: {send_err}")

        except Exception as e:
            logger.error(f"Error in auto_trade_loop: {e}")

        await asyncio.sleep(18)


async def post_init(application):
    """Start background async monitoring tasks after bot initialization."""
    asyncio.create_task(price_alert_loop(application))
    asyncio.create_task(entry_alert_loop(application))
    asyncio.create_task(auto_trade_loop(application))
    logger.info("Background alert & auto-trading tasks successfully scheduled.")


def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("ERROR: TELEGRAM_TOKEN environment variable is not set.", file=sys.stderr)
        print("Please export TELEGRAM_TOKEN=... and rerun.", file=sys.stderr)
        sys.exit(1)

    app = ApplicationBuilder().token(token).post_init(post_init).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("commands", list_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("dashboard", status_cmd))
    app.add_handler(CommandHandler("dash", status_cmd))

    # Delta Exchange & Automated Trading Commands
    app.add_handler(CommandHandler("autotrade", autotrade_cmd))
    app.add_handler(CommandHandler("starttrade", start_trade_cmd))
    app.add_handler(CommandHandler("stoptrade", stop_trade_cmd))
    app.add_handler(CommandHandler("tradeon", start_trade_cmd))
    app.add_handler(CommandHandler("tradeoff", stop_trade_cmd))
    app.add_handler(CommandHandler("startbot", start_trade_cmd))
    app.add_handler(CommandHandler("stopbot", stop_trade_cmd))
    app.add_handler(CommandHandler("trading", autotrade_cmd))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CommandHandler("trade", trade_cmd))
    app.add_handler(CommandHandler("size", size_cmd))
    app.add_handler(CommandHandler("lotsize", size_cmd))
    app.add_handler(CommandHandler("lot", size_cmd))
    app.add_handler(CommandHandler("positions", positions_cmd))
    app.add_handler(CommandHandler("closeposition", close_position_cmd))
    app.add_handler(CommandHandler("closeall", close_all_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("orders", orders_cmd))
    app.add_handler(CommandHandler("cancelorders", cancel_orders_cmd))
    app.add_handler(CommandHandler("learn", learn_cmd))
    app.add_handler(CommandHandler("learning", learn_cmd))
    app.add_handler(CommandHandler("selflearn", learn_cmd))
    app.add_handler(CommandHandler("selflearning", learn_cmd))
    app.add_handler(CommandHandler("insights", insights_cmd))
    app.add_handler(CommandHandler("insight", insights_cmd))
    app.add_handler(CommandHandler("confluence", confluence_cmd))
    app.add_handler(CommandHandler("combine", confluence_cmd))
    app.add_handler(CommandHandler("master", confluence_cmd))
    app.add_handler(CommandHandler("confluencetrade", confluence_trade_cmd))
    app.add_handler(CommandHandler("orderbook", orderbook_cmd))
    app.add_handler(CommandHandler("book", orderbook_cmd))
    app.add_handler(CommandHandler("depth", orderbook_cmd))
    app.add_handler(CommandHandler("news", news_cmd))
    app.add_handler(CommandHandler("btcnews", btc_news_cmd))
    app.add_handler(CommandHandler("goldnews", gold_news_cmd))
    app.add_handler(CommandHandler("btcbook", btc_book_cmd))
    app.add_handler(CommandHandler("goldbook", gold_book_cmd))
    app.add_handler(CommandHandler("btcconfluence", btc_confluence_cmd))
    app.add_handler(CommandHandler("goldconfluence", gold_confluence_cmd))
    app.add_handler(CommandHandler("setkey", set_key_cmd))
    app.add_handler(CommandHandler("setsecret", set_secret_cmd))
    app.add_handler(CommandHandler("setkeys", set_keys_cmd))
    app.add_handler(CommandHandler("keys", keys_cmd))
    app.add_handler(CommandHandler("checkkeys", keys_cmd))
    app.add_handler(CommandHandler("checkkey", keys_cmd))
    app.add_handler(CommandHandler("testkeys", keys_cmd))
    app.add_handler(CommandHandler("testkey", keys_cmd))
    app.add_handler(CommandHandler("key", keys_cmd))
    app.add_handler(CommandHandler("bot", keys_cmd))
    app.add_handler(CommandHandler("config", keys_cmd))
    app.add_handler(CommandHandler("setbaseurl", set_base_url_cmd))
    app.add_handler(CommandHandler("model", gemini_model_cmd))
    app.add_handler(CommandHandler("geminimodel", gemini_model_cmd))
    app.add_handler(CommandHandler("googlemodel", gemini_model_cmd))
    app.add_handler(CommandHandler("setgemini", set_gemini_cmd))
    app.add_handler(CommandHandler("setgeminikey", set_gemini_cmd))
    app.add_handler(CommandHandler("gemini", gemini_status_cmd))
    app.add_handler(CommandHandler("ai", gemini_status_cmd))

    # BTC Shortcuts
    app.add_handler(CommandHandler("btc", btc_cmd))
    app.add_handler(CommandHandler("btclevels", btc_levels_cmd))
    app.add_handler(CommandHandler("btcgj", btc_gj_cmd))
    app.add_handler(CommandHandler("btcentry", btc_entry_cmd))
    app.add_handler(CommandHandler("btcwatch", btc_watch_cmd))

    # Gold Shortcuts
    app.add_handler(CommandHandler("gold", gold_cmd))
    app.add_handler(CommandHandler("xau", gold_cmd))
    app.add_handler(CommandHandler("xauusd", gold_cmd))
    app.add_handler(CommandHandler("goldlevels", gold_levels_cmd))
    app.add_handler(CommandHandler("xaulevels", gold_levels_cmd))
    app.add_handler(CommandHandler("xauusdlevels", gold_levels_cmd))
    app.add_handler(CommandHandler("goldgj", gold_gj_cmd))
    app.add_handler(CommandHandler("xaugj", gold_gj_cmd))
    app.add_handler(CommandHandler("xauusdgj", gold_gj_cmd))
    app.add_handler(CommandHandler("goldentry", gold_entry_cmd))
    app.add_handler(CommandHandler("xauentry", gold_entry_cmd))
    app.add_handler(CommandHandler("xauusdentry", gold_entry_cmd))
    app.add_handler(CommandHandler("goldwatch", gold_watch_cmd))
    app.add_handler(CommandHandler("xauwatch", gold_watch_cmd))
    app.add_handler(CommandHandler("xauusdwatch", gold_watch_cmd))

    # 18-Agent Institutional Desk Analysis
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("analysis", analyze_cmd))
    app.add_handler(CommandHandler("desk", analyze_cmd))
    app.add_handler(CommandHandler("agents", analyze_cmd))

    # AMD Scalp (1m, 5m, 15m) Commands & Multi-Trade
    app.add_handler(CommandHandler("amd", amd_cmd))
    app.add_handler(CommandHandler("scalp", amd_cmd))
    app.add_handler(CommandHandler("btcamd", btc_amd_cmd))
    app.add_handler(CommandHandler("goldamd", gold_amd_cmd))
    app.add_handler(CommandHandler("xauamd", gold_amd_cmd))
    app.add_handler(CommandHandler("multitrade", multi_trade_cmd))

    # General Market Commands
    app.add_handler(CommandHandler("price", price_cmd))
    app.add_handler(CommandHandler("levels", levels_cmd))
    app.add_handler(CommandHandler("gj", gj_cmd))
    app.add_handler(CommandHandler("liquidity", gj_cmd))
    app.add_handler(CommandHandler("alertson", auto_alert_on_cmd))
    app.add_handler(CommandHandler("alertsoff", auto_alert_off_cmd))
    app.add_handler(CommandHandler("autoalert", auto_alert_toggle_cmd))
    app.add_handler(CommandHandler("autoalerts", auto_alert_toggle_cmd))
    app.add_handler(CommandHandler("alert", set_alert_cmd))
    app.add_handler(CommandHandler("alerts", set_alert_cmd))
    app.add_handler(CommandHandler("delalert", delete_alert_cmd))
    app.add_handler(CommandHandler("clearalerts", clear_alerts_cmd))
    app.add_handler(CommandHandler("entry", entry_cmd))
    app.add_handler(CommandHandler("scan", entry_cmd))
    app.add_handler(CommandHandler("watch", watch_cmd))
    app.add_handler(CommandHandler("unwatch", unwatch_cmd))
    app.add_handler(CommandHandler("watchers", list_watchers_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("Bot starting polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
