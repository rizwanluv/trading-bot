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

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Gemini Setup
MODEL = "gemini-2.5-flash"
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


def generate_ai_reply(messages: List[Dict[str, str]]) -> str:
    """Generate response from Gemini via SDK or direct REST API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠️ <b>GEMINI_API_KEY is not set.</b>\n\n"
            "You can still use all market features:\n"
            "• <code>/price [SYMBOL]</code> — Live prices\n"
            "• <code>/levels [SYMBOL]</code> — Automatic Level Analysis\n"
            "• <code>/alert [SYMBOL] &lt;PRICE&gt;</code> — Set price alerts\n"
            "• <code>/entry [SYMBOL]</code> — 1m, 5m, 15m candle entry scan\n"
            "• <code>/watch [SYMBOL]</code> — Automatic candle entry alerts\n\n"
            "To chat with AI, set the <code>GEMINI_API_KEY</code> environment variable."
        )

    if HAS_GENAI_SDK:
        try:
            client = genai.Client(api_key=api_key)
            formatted_contents = [
                types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
                for m in messages
            ]
            reply = client.models.generate_content(
                model=MODEL,
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM, max_output_tokens=800
                ),
            )
            if reply.text:
                return reply.text
        except Exception as e:
            logger.warning(f"google-genai SDK call failed: {e}. Falling back to REST API.")

    # REST fallback
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"role": m["role"], "parts": [{"text": m["text"]}]}
                for m in messages
            ],
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "generationConfig": {"maxOutputTokens": 800},
        }
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "No response text.")
        return "No response received from Gemini API."
    except Exception as e:
        return f"Error contacting Gemini API: {e}"


def generate_ai_vision_reply(prompt_text: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Analyze chart photo using Gemini Multimodal Vision following Gautam Jha strategy."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠️ <b>GEMINI_API_KEY is not set.</b>\n\n"
            "To enable chart screenshot analysis with Gautam Jha strategy, "
            "please set the <code>GEMINI_API_KEY</code> environment variable."
        )

    if HAS_GENAI_SDK:
        try:
            client = genai.Client(api_key=api_key)
            img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            text_part = types.Part.from_text(text=prompt_text)
            reply = client.models.generate_content(
                model=MODEL,
                contents=[types.Content(parts=[text_part, img_part])],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    max_output_tokens=1200,
                ),
            )
            if reply.text:
                return reply.text
        except Exception as e:
            logger.warning(f"google-genai vision call failed: {e}. Falling back to REST API.")

    # REST fallback
    try:
        import base64
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {"inline_data": {"mime_type": mime_type, "data": b64_img}},
                    ]
                }
            ],
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "generationConfig": {"maxOutputTokens": 1200},
        }
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "No response text.")
        return "No response received from Gemini Vision API."
    except Exception as e:
        return f"Error contacting Gemini Vision API: {e}"


# ==================== Command Handlers ====================

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Start command intro."""
    msg = (
        "🤖 <b>Welcome to Gemini Trading Assistant & Auto-Trade Bot!</b>\n\n"
        "Your intelligent algorithmic assistant powered by Delta Exchange live data, automatic level "
        "analysis, price alerts, multi-timeframe candle scanner, <b>Gautam Jha Liquidity strategy</b>, "
        "and <b>Automated Order Execution</b> for <b>🥇 Gold (XAU/USD)</b> and <b>🪙 Bitcoin (BTC/USD)</b>.\n\n"
        "🔥 <b>Key Capabilities:</b>\n"
        "• <b>🤖 Automated Trading:</b> <code>/autotrade on [live|paper]</code> (Executes high-probability setups)\n"
        "• <b>⚡ Manual Trading:</b> <code>/trade btc buy 1</code> | <code>/trade gold sell 1</code> (Auto SL & TP)\n"
        "• <b>📈 Positions & Balances:</b> <code>/positions</code> | <code>/balance</code> | <code>/closeall</code>\n"
        "• <b>🔑 Delta API Keys:</b> <code>/setkeys &lt;KEY&gt; &lt;SECRET&gt;</code> | <code>/keys</code>\n"
        "• <b>🥇 Gold (XAU/USD):</b> <code>/gold</code>, <code>/goldlevels</code>, <code>/goldgj</code>, <code>/goldentry</code>, <code>/goldwatch</code>\n"
        "• <b>🪙 Bitcoin (BTC):</b> <code>/btc</code>, <code>/btclevels</code>, <code>/btcgj</code>, <code>/btcentry</code>, <code>/btcwatch</code>\n"
        "• <b>🌐 Live Market Overview:</b> <code>/price</code> (Live Gold & BTC overview)\n"
        "• <b>🚨 Price Alerts:</b> <code>/alert gold 4180</code> or <code>/alert btc 85000</code>\n"
        "• <b>📸 Chart Photo Scanner:</b> Send any chart photo for instant Gautam Jha analysis!\n\n"
        "Type <code>/list</code> to view all commands or <code>/help</code> for detailed instructions."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def list_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show all bot commands in a quick, clean reference list."""
    msg = (
        "📜 <b>ALL BOT COMMANDS:</b>\n\n"
        "🤖 <b>Automated & Manual Trading:</b>\n"
        "• <code>/starttrade</code> (or <code>/tradeon</code>) — <b>START</b> automated trading bot 🟢\n"
        "• <code>/stoptrade</code> (or <code>/tradeoff</code>) — <b>STOP</b> / pause automated trading 🔴\n"
        "• <code>/autotrade [on|off]</code> — Automated bot control & performance\n"
        "• <code>/trade &lt;SYMBOL&gt; &lt;BUY/SELL&gt; [SIZE]</code> — Execute order with auto SL/TP\n"
        "• <code>/positions</code> — View active open positions & unrealized PnL\n"
        "• <code>/closeposition &lt;ID&gt;</code> (or <code>/closeall</code>) — Close position at market\n"
        "• <code>/balance</code> — View Delta Exchange wallet & paper balance\n"
        "• <code>/mode [live|paper]</code> — Switch between Live & Paper trading\n\n"
        "🔑 <b>Delta Exchange API Keys:</b>\n"
        "• <code>/setkey &lt;KEY&gt;</code> — Set Delta Exchange API Key\n"
        "• <code>/setsecret &lt;SECRET&gt;</code> — Set Delta Exchange API Secret\n"
        "• <code>/setkeys &lt;KEY&gt; &lt;SECRET&gt;</code> — Connect Delta credentials\n"
        "• <code>/keys</code> (or <code>/checkkeys</code>) — Check & verify API connection live\n"
        "• <code>/setbaseurl [india|global]</code> — Switch Delta India vs Global\n"
        "• <code>/orders</code> — View active open orders on Delta\n"
        "• <code>/cancelorders [SYMBOL]</code> — Cancel working orders\n\n"
        "🔔 <b>Automatic Market Alerts:</b>\n"
        "• <code>/alertson</code> — <b>TURN ON</b> automatic alerts for BTC & Gold 🟢\n"
        "• <code>/alertsoff</code> — <b>TURN OFF</b> automatic market alerts 🔴\n"
        "• <code>/autoalert [on|off]</code> — Automatic alerts toggle\n\n"
        "🥇 <b>Gold (XAU/USD) Shortcuts:</b>\n"
        "• <code>/gold</code> (or <code>/xau</code>, <code>/xauusd</code>) — Live Gold ticker & 24h stats\n"
        "• <code>/goldlevels</code> (or <code>/xaulevels</code>) — Gold Automatic Level Analysis\n"
        "• <code>/goldgj</code> (or <code>/xaugj</code>) — Gold Gautam Jha Liquidity (DO, PDH, PDL sweeps)\n"
        "• <code>/goldentry</code> (or <code>/xauentry</code>) — Gold 1m, 5m, 15m candle entry scan\n"
        "• <code>/goldwatch</code> (or <code>/xauwatch</code>) — Turn ON automated candle alerts for Gold\n\n"
        "🪙 <b>Bitcoin (BTC) Shortcuts:</b>\n"
        "• <code>/btc</code> — Live BTC ticker & 24h stats\n"
        "• <code>/btclevels</code> — BTC Automatic Level Analysis (Pivots, Fibs, S/R)\n"
        "• <code>/btcgj</code> — BTC Gautam Jha Liquidity (DO, PDH, PDL sweeps)\n"
        "• <code>/btcentry</code> — BTC 1m, 5m, 15m candle entry scan\n"
        "• <code>/btcwatch</code> — Turn ON automated candle alerts for BTC\n\n"
        "💹 <b>Market Data & Any Symbol:</b>\n"
        "• <code>/price</code> — Live overview of Gold & BTC\n"
        "• <code>/price [SYMBOL]</code> — Live ticker for any coin (e.g. <code>/price ETH</code>)\n\n"
        "📊 <b>Level & Liquidity Analysis:</b>\n"
        "• <code>/levels [SYMBOL]</code> (or <code>/analysis</code>) — S/R, Pivots, Fibs, DO\n"
        "• <code>/gj [SYMBOL]</code> (or <code>/liquidity</code>) — Gautam Jha Price Action\n\n"
        "🚨 <b>Custom Price Alerts:</b>\n"
        "• <code>/alert [SYMBOL] &lt;PRICE&gt;</code> — Set price alert (e.g. <code>/alert gold 4180</code>)\n"
        "• <code>/alerts</code> — List your active price alerts\n"
        "• <code>/delalert &lt;ID&gt;</code> — Remove an alert by ID\n"
        "• <code>/clearalerts</code> — Clear all your active price alerts\n\n"
        "🎯 <b>Candle Scanner & Watchers:</b>\n"
        "• <code>/entry [SYMBOL]</code> (or <code>/scan</code>) — Scan 1m, 5m, 15m candles\n"
        "• <code>/watch [SYMBOL] [tfs]</code> — Turn ON automated candle alerts\n"
        "• <code>/unwatch [SYMBOL]</code> — Turn OFF automated candle alerts\n"
        "• <code>/watchers</code> — List active candle scanners\n\n"
        "📸 <b>Chart Photo Analysis:</b>\n"
        "• <i>Send Chart Photo</i> — Upload any screenshot for Gautam Jha vision analysis\n\n"
        "🤖 <b>Bot Controls & Chat:</b>\n"
        "• <code>/list</code> — Show this full commands list\n"
        "• <code>/help</code> — Detailed instructions & examples\n"
        "• <code>/start</code> — Introduction & welcome overview\n"
        "• <code>/reset</code> — Clear AI conversation history\n"
        "• <i>Any text</i> — Chat directly with the AI trading assistant"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Help command with usage examples."""
    msg = (
        "📖 <b>Trading Assistant & Auto-Trade Guide:</b>\n\n"
        "<b>1. Automated Trading (Auto-Trade Engine):</b>\n"
        "• <code>/autotrade on [symbol] [live|paper]</code> — Turn on auto-trading\n"
        "  <i>Examples: <code>/autotrade on btc paper</code> | <code>/autotrade on gold live</code> | <code>/autotrade on</code></i>\n"
        "• <code>/autotrade off</code> — Turn off auto-trading\n"
        "• <code>/autotrade status</code> — View performance, win-rate & open trades\n"
        "• <code>/mode [live|paper]</code> — Switch between Live Delta Exchange and Paper simulation\n\n"
        "<b>2. Manual Trading & Positions:</b>\n"
        "• <code>/trade &lt;SYMBOL&gt; &lt;BUY/SELL&gt; [SIZE]</code> — Immediate trade with auto SL & TP\n"
        "  <i>Examples: <code>/trade btc buy 1</code> | <code>/trade gold sell 1</code></i>\n"
        "• <code>/positions</code> — View all open positions, mark prices & real-time PnL\n"
        "• <code>/closeposition &lt;SYMBOL&gt;</code> — Close position at market (or <code>/closeall</code>)\n"
        "• <code>/balance</code> — View Delta Exchange wallet balances & paper portfolio\n\n"
        "<b>3. Delta Exchange API Configuration:</b>\n"
        "• <code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code> — Set your Delta credentials directly\n"
        "• <code>/keys</code> — Check connection status & permissions\n"
        "• <code>/orders</code> — View working open orders on Delta\n"
        "• <code>/cancelorders</code> — Cancel working orders\n\n"
        "<b>4. 🥇 Gold (XAU/USD) Shortcuts:</b>\n"
        "• <code>/gold</code> (or <code>/xau</code>, <code>/xauusd</code>) — Live Gold ticker\n"
        "• <code>/goldlevels</code> (or <code>/xaulevels</code>) — Gold key levels (Pivots, Fibs, S/R)\n"
        "• <code>/goldgj</code> (or <code>/xaugj</code>) — Gold Gautam Jha Liquidity (DO, PDH, PDL sweeps)\n"
        "• <code>/goldentry</code> (or <code>/xauentry</code>) — Gold 1m, 5m, 15m candle entry scan\n"
        "• <code>/goldwatch</code> (or <code>/xauwatch</code>) — Automated candle alerts for Gold\n\n"
        "<b>5. 🪙 Bitcoin (BTC) Shortcuts:</b>\n"
        "• <code>/btc</code> — Live BTC ticker & 24h stats\n"
        "• <code>/btclevels</code> — BTC key levels (Pivots, Fibs, S/R)\n"
        "• <code>/btcgj</code> — BTC Gautam Jha Liquidity (DO, PDH, PDL)\n"
        "• <code>/btcentry</code> — BTC 1m, 5m, 15m candle entry scan\n"
        "• <code>/btcwatch</code> — Automated candle alerts for BTC\n\n"
        "<b>6. Market Data, Alerts & Analysis:</b>\n"
        "• <code>/price</code> — Live comparison overview (Gold & BTC)\n"
        "• <code>/levels [SYMBOL]</code> — S/R, Pivots, Fibs for any coin\n"
        "• <code>/gj [SYMBOL]</code> — Gautam Jha analysis for any coin\n"
        "• <code>/alert [SYMBOL] &lt;PRICE&gt;</code> — Custom price alerts\n"
        "• <code>/entry [SYMBOL]</code> — 1m, 5m, 15m candle setups\n"
        "• <code>/watch [SYMBOL]</code> — Automated background candle alerts\n"
        "• <b>📸 Send Chart Screenshot</b> — Gautam Jha AI vision analysis"
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


# Shortcuts for Bitcoin (BTC)
async def btc_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /btc -> Live BTC ticker."""
    ctx.args = ["BTCUSD"]
    await price_cmd(update, ctx)


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


# Shortcuts for Gold (XAU)
async def gold_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Shortcut: /gold or /xau -> Live Gold ticker."""
    ctx.args = ["XAUTUSD"]
    await price_cmd(update, ctx)


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
    """Set price alert command."""
    symbol, target_price, condition, err = _parse_alert_args(ctx.args or [])
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
        f"<i>I will notify you automatically when this price is hit. Use <code>/alerts</code> to manage.</i>"
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

    # Balance / Positions
    if lower_text in ("balance", "my balance", "check balance", "wallet", "wallet balance"):
        await balance_cmd(update, ctx)
        return
    if lower_text in ("positions", "my positions", "open positions", "position"):
        await positions_cmd(update, ctx)
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
    """Analyze chart photo or screenshot using Gemini Multimodal Vision & Gautam Jha strategy."""
    if not update.message or not update.message.photo:
        return

    await update.message.reply_text("🔍 Analyzing chart screenshot with Gautam Jha price-action strategy... ⏳")

    photo = update.message.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    caption = update.message.caption or "Analyze this chart using Gautam Jha style."

    # Detect symbol from caption if present
    symbol = _detect_symbol_from_text(caption, DEFAULT_SYMBOL)

    # Enrich prompt with live market liquidity data if available
    context_note = ""
    try:
        gj_analysis = get_gautam_jha_analysis(symbol)
        gj = gj_analysis["gautam_jha"]
        context_note = (
            f"\n[Live Delta Exchange Data for {symbol}]\n"
            f"Current Price: {gj_analysis['mark_price']} | Daily Open: {gj['daily_open']} ({gj['daily_candle_color']})\n"
            f"PDH: {gj['pdh']} (Swept: {gj['pdh_swept']}) | PDL: {gj['pdl']} (Swept: {gj['pdl_swept']})\n"
            f"Market Structure: {gj_analysis['market_structure']}\n"
        )
    except Exception:
        pass

    full_prompt = (
        f"User request: {caption}\n"
        f"{context_note}\n"
        f"Analyze this chart image thoroughly using Gautam Jha rules: identify Daily Open, PDH, PDL, "
        f"liquidity pools, market structure, and provide 1-2 high-probability trade setups (Break-and-Go, "
        f"Retrace-to-Level, or Level Reversal) with Entry, Stop Loss, and Targets."
    )

    reply = generate_ai_vision_reply(full_prompt, bytes(image_bytes))
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
    """View Delta Exchange API connection status & perform live handshake (/keys, /checkkeys, /testkeys)."""
    is_cfg = delta_client.is_configured()
    masked = delta_client.get_masked_key()
    mode = auto_trader.mode.upper()
    at_status = "🟢 ACTIVE" if auto_trader.enabled else "🔴 OFF"

    if not is_cfg:
        msg = (
            "🔐 <b>Delta Exchange API Status: 🔴 NOT CONFIGURED</b>\n\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Trading Mode:</b> <code>{mode}</code> (Paper Trading is active)\n"
            f"• <b>Auto-Trading:</b> {at_status}\n\n"
            "🔑 <b>How to set your Delta Exchange API keys:</b>\n"
            "• Use command: <code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>\n"
            "• Or individual commands:\n"
            "  - <code>/setkey &lt;KEY&gt;</code>\n"
            "  - <code>/setsecret &lt;SECRET&gt;</code>\n\n"
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
            "🔐 <b>Delta Exchange API Status: 🟢 CONNECTED & VERIFIED</b>\n\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Clock Sync:</b> <code>{drift:+.2f}s</code> (Auto-synchronized)\n"
            f"• <b>Trading Mode:</b> <code>{mode}</code>\n"
            f"• <b>Auto-Trading:</b> {at_status}\n\n"
            f"💰 <b>Live Balances:</b>\n{bal_text}\n\n"
            "🎯 <b>Ready Commands:</b>\n"
            "• <code>/starttrade</code> — Start automatic trading bot\n"
            "• <code>/stoptrade</code> — Pause automatic trading\n"
            "• <code>/mode live</code> — Switch to live orders\n"
            "• <code>/mode paper</code> — Switch back to paper mode ($10,000 demo)\n"
            "• <code>/balance</code> — Refresh live balance"
        )
    else:
        err_msg = test_res.get("error", "Unknown connection error")
        hint = ""
        if "invalid_api_key" in err_msg.lower():
            hint = (
                "\n\n⚠️ <b>Troubleshooting Delta 'invalid_api_key':</b>\n"
                "1. <b>Platform Mismatch:</b> If your account is on Delta Global (.com), run:\n"
                "   <code>/setbaseurl global</code>\n"
                "   If your account is on Delta India (.exchange), run:\n"
                "   <code>/setbaseurl india</code>\n"
                "2. <b>IP Whitelist:</b> If you set IP restrictions when creating the key on Delta, requests from this bot will be rejected. Make sure IP restriction is disabled.\n"
                "3. <b>Permissions:</b> Verify in Delta settings that <b>Read</b> and <b>Trade</b> permissions are checked.\n"
                "4. <b>Re-enter Keys:</b> Update keys anytime with:\n"
                "   <code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>"
            )
        elif "expired_signature" in err_msg.lower():
            drift = getattr(delta_client, "_server_time_offset", 0.0)
            hint = (
                f"\n\n⚠️ <b>Clock Drift:</b> Synchronized offset ({drift:+.2f}s). Run <code>/checkkeys</code> again to retry."
            )

        msg = (
            "🔐 <b>Delta Exchange API Status: 🔴 CONNECTION FAILED</b>\n\n"
            f"• <b>API Key:</b> <code>{masked}</code>\n"
            f"• <b>Base URL:</b> <code>{delta_client.base_url}</code>\n"
            f"• <b>Error:</b> <code>{err_msg}</code>"
            f"{hint}\n\n"
            "💡 <b>To update your API keys:</b>\n"
            "<code>/setkeys &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>"
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


async def trade_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Execute a manual trade (/trade <SYMBOL> <buy|sell> [size])."""
    args = ctx.args or []
    if len(args) < 2:
        await update.message.reply_text(
            "⚡ <b>Manual Trade Execution</b>\n\n"
            "<b>Usage:</b>\n"
            "<code>/trade &lt;SYMBOL&gt; &lt;buy|sell&gt; [size]</code>\n\n"
            "<b>Examples:</b>\n"
            "• <code>/trade BTC buy</code> (Buy BTC with default size)\n"
            "• <code>/trade GOLD sell 0.05</code> (Short Gold)\n"
            "• <code>/trade BTCUSD buy 0.01</code>\n\n"
            f"• <b>Current Mode:</b> <code>{auto_trader.mode.upper()}</code>\n"
            "<i>(Auto-calculates entry, SL, TP1, and TP2 based on Gautam Jha levels!)</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    sym_raw = args[0]
    side = args[1].lower().strip()
    if side not in ("buy", "sell", "long", "short"):
        await update.message.reply_text("⚠️ Side must be <code>buy</code> or <code>sell</code>.", parse_mode=ParseMode.HTML)
        return

    if side == "long":
        side = "buy"
    elif side == "short":
        side = "sell"

    size = None
    if len(args) >= 3:
        try:
            size = float(args[2])
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
            "📊 <i>Track with <code>/positions</code> or close with <code>/closeposition {id}</code></i>"
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
    symbols_to_scan = ["BTCUSD", "XAUTUSD"]

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
                        f"🎯 <b>Take Profit 1:</b> <code>${tr.get('tp1_price', 0):,.2f}</code> (1:1.5)\n"
                        f"🎯 <b>Take Profit 2:</b> <code>${tr.get('tp2_price', 0):,.2f}</code> (1:2.5)\n"
                        f"💡 <b>Strategy:</b> Gautam Jha Liquidity ({tr.get('reason')})\n"
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
    app.add_handler(CommandHandler("positions", positions_cmd))
    app.add_handler(CommandHandler("closeposition", close_position_cmd))
    app.add_handler(CommandHandler("closeall", close_all_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("orders", orders_cmd))
    app.add_handler(CommandHandler("cancelorders", cancel_orders_cmd))
    app.add_handler(CommandHandler("setkey", set_key_cmd))
    app.add_handler(CommandHandler("setsecret", set_secret_cmd))
    app.add_handler(CommandHandler("setkeys", set_keys_cmd))
    app.add_handler(CommandHandler("keys", keys_cmd))
    app.add_handler(CommandHandler("checkkeys", keys_cmd))
    app.add_handler(CommandHandler("checkkey", keys_cmd))
    app.add_handler(CommandHandler("testkeys", keys_cmd))
    app.add_handler(CommandHandler("testkey", keys_cmd))
    app.add_handler(CommandHandler("key", keys_cmd))
    app.add_handler(CommandHandler("setbaseurl", set_base_url_cmd))

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

    # General Market Commands
    app.add_handler(CommandHandler("price", price_cmd))
    app.add_handler(CommandHandler("levels", levels_cmd))
    app.add_handler(CommandHandler("analysis", levels_cmd))
    app.add_handler(CommandHandler("gj", gj_cmd))
    app.add_handler(CommandHandler("liquidity", gj_cmd))
    app.add_handler(CommandHandler("alertson", auto_alert_on_cmd))
    app.add_handler(CommandHandler("alertsoff", auto_alert_off_cmd))
    app.add_handler(CommandHandler("autoalert", auto_alert_toggle_cmd))
    app.add_handler(CommandHandler("autoalerts", auto_alert_toggle_cmd))
    app.add_handler(CommandHandler("alert", set_alert_cmd))
    app.add_handler(CommandHandler("alerts", list_alerts_cmd))
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
