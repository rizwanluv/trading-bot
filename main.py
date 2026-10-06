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
        "🤖 <b>Welcome to Gemini Trading Assistant!</b>\n\n"
        "Your intelligent assistant combining Delta Exchange market intelligence, automatic level "
        "analysis, price alerts, multi-timeframe candle scanner, and <b>Gautam Jha Price-Action & Liquidity strategy</b>.\n\n"
        "🔥 <b>Key Features:</b>\n"
        "• <b>Live Prices:</b> <code>/price XAUTUSD</code>\n"
        "• <b>Automatic Levels:</b> <code>/levels XAUTUSD</code> (Pivots, Fibs, S/R, DO, PDH, PDL)\n"
        "• <b>Gautam Jha Liquidity:</b> <code>/gj XAUTUSD</code> (Daily Open, PDH/PDL sweeps, 3 Trade Styles)\n"
        "• <b>Price Alerts:</b> <code>/alert 4180</code> or <code>/alert BTCUSD 86000</code>\n"
        "• <b>Candle Entry Scan:</b> <code>/entry XAUTUSD</code> (1m, 5m, 15m setups)\n"
        "• <b>Automated Entry Alerts:</b> <code>/watch XAUTUSD</code> (Notifies on candle close)\n"
        "• <b>📸 Chart Photo Scanner:</b> Send any chart photo/screenshot for instant Gautam Jha analysis!\n"
        "• <b>Manage Alerts:</b> <code>/alerts</code>, <code>/delalert &lt;ID&gt;</code>, <code>/watchers</code>\n"
        "• <b>AI Analysis:</b> Send any question to get precise trade plans grounded in live levels!\n\n"
        "Type <code>/list</code> to view all commands or <code>/help</code> for full instructions."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def list_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show all bot commands in a quick, clean reference list."""
    msg = (
        "📜 <b>ALL BOT COMMANDS:</b>\n\n"
        "💹 <b>Market Data & Prices</b>\n"
        "• <code>/price [SYMBOL]</code> — Live ticker, 24h high/low, open, and change %\n\n"
        "📊 <b>Level & Liquidity Analysis</b>\n"
        "• <code>/levels [SYMBOL]</code> — Automatic Level Analysis (Pivots, Fibs, S/R, DO)\n"
        "• <code>/analysis [SYMBOL]</code> — Alias for /levels\n"
        "• <code>/gj [SYMBOL]</code> — Gautam Jha Price-Action Analysis (DO, PDH, PDL, sweeps)\n"
        "• <code>/liquidity [SYMBOL]</code> — Alias for /gj\n\n"
        "🚨 <b>Price Alerts</b>\n"
        "• <code>/alert [SYMBOL] &lt;PRICE&gt;</code> — Set price alert (auto >= or <=)\n"
        "• <code>/alerts</code> — List your active price alerts\n"
        "• <code>/delalert &lt;ID&gt;</code> — Remove an alert by ID\n"
        "• <code>/clearalerts</code> — Clear all your active price alerts\n\n"
        "🎯 <b>1m, 5m, 15m Candle Entry Alerts</b>\n"
        "• <code>/entry [SYMBOL]</code> — Instant scan on 1m, 5m, 15m candle setups\n"
        "• <code>/scan [SYMBOL]</code> — Alias for /entry\n"
        "• <code>/watch [SYMBOL] [tfs]</code> — Turn ON automated candle entry alerts\n"
        "• <code>/unwatch [SYMBOL]</code> — Turn OFF automated candle entry alerts\n"
        "• <code>/watchers</code> — List active candle scanners\n\n"
        "📸 <b>Chart Photo Analysis</b>\n"
        "• <i>Send Chart Photo</i> — Upload any screenshot for Gautam Jha vision analysis\n\n"
        "🤖 <b>Bot Controls & Chat</b>\n"
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
        "📖 <b>Trading Assistant Commands:</b>\n\n"
        "<b>1. Price & Market Data:</b>\n"
        "• <code>/price [SYMBOL]</code> — Live ticker (default: XAUTUSD)\n"
        "  <i>Example: /price BTCUSD</i>\n\n"
        "<b>2. Automatic Level & Liquidity Analysis:</b>\n"
        "• <code>/levels [SYMBOL]</code> (or <code>/analysis</code>) — Calculates Daily Pivots, "
        "Fibonacci Retracements, Nearest Support & Resistance, Daily Open, and Trend Bias.\n"
        "• <code>/gj [SYMBOL]</code> (or <code>/liquidity</code>) — Pure <b>Gautam Jha Strategy</b>: "
        "Daily Open (DO), Previous Day High (PDH), Previous Day Low (PDL), liquidity sweep status, and 3 trade setups.\n"
        "  <i>Example: /gj XAUTUSD</i>\n\n"
        "<b>3. Price Alerts:</b>\n"
        "• <code>/alert &lt;PRICE&gt;</code> — Set alert on default symbol (auto-detects above/below)\n"
        "• <code>/alert &lt;SYMBOL&gt; &lt;PRICE&gt;</code> — Set alert for specific symbol\n"
        "• <code>/alert &lt;SYMBOL&gt; &gt;= &lt;PRICE&gt;</code> — Explicit condition\n"
        "  <i>Examples: /alert 4180 | /alert BTCUSD 86500</i>\n"
        "• <code>/alerts</code> — View all your active price alerts\n"
        "• <code>/delalert &lt;ID&gt;</code> — Delete alert by ID\n"
        "• <code>/clearalerts</code> — Clear all price alerts\n\n"
        "<b>4. 1m, 5m, 15m Candle Entry Alerts:</b>\n"
        "• <code>/entry [SYMBOL]</code> (or <code>/scan</code>) — Instantly scan 1m, 5m, and 15m "
        "candles for Break-and-Go, Retrace-to-DO, Pin Bars, Engulfing, and S/R Bounces with Entry, SL, TP1, TP2!\n"
        "• <code>/watch [SYMBOL] [tfs]</code> — Enable <b>automatic entry alerts</b>! Sends notification "
        "when an entry setup forms on a candle close.\n"
        "  <i>Examples: /watch XAUTUSD | /watch BTCUSD 5m,15m</i>\n"
        "• <code>/unwatch [SYMBOL]</code> — Stop entry alerts (or <code>/unwatch all</code>)\n"
        "• <code>/watchers</code> — View active entry scanners\n\n"
        "<b>5. Chart Photos & AI Chat:</b>\n"
        "• <b>📸 Send Chart Screenshot</b> — Upload any chart photo (add caption e.g. <code>XAUUSD 15m</code>) "
        "and get a complete Gautam Jha top-down liquidity breakdown!\n"
        "• <code>/reset</code> — Clear AI conversation history\n"
        "• <i>Any text</i> — Chat with assistant (sees live market data, DO, PDH, PDL, and candle signals)"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def price_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Live price command."""
    symbol = ctx.args[0].upper() if ctx.args else DEFAULT_SYMBOL
    try:
        t = get_ticker(symbol)
        mark = t["mark_price"] or t["close"]
        chg = mark - t["open"] if t["open"] > 0 else 0.0
        chg_pct = (chg / t["open"] * 100.0) if t["open"] > 0 else 0.0
        sign = "+" if chg >= 0 else ""

        msg = (
            f"🪙 <b>#{t['symbol']} Live Ticker</b>\n\n"
            f"💵 <b>Price:</b> <code>${mark:,.2f}</code>\n"
            f"📊 <b>24h Change:</b> <code>{sign}{chg:,.2f} ({sign}{chg_pct:.2f}%)</code>\n"
            f"🔺 <b>24h High:</b> <code>${t['high']:,.2f}</code>\n"
            f"🔻 <b>24h Low:</b> <code>${t['low']:,.2f}</code>\n"
            f"🚪 <b>24h Open:</b> <code>${t['open']:,.2f}</code>\n"
            f"📦 <b>24h Volume:</b> <code>{t['volume']:,.0f}</code>"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error fetching price for {symbol}: {e}")


async def levels_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Automatic level analysis command."""
    symbol = ctx.args[0].upper() if ctx.args else DEFAULT_SYMBOL
    try:
        analysis = get_level_analysis(symbol)
        msg = format_level_analysis_message(analysis)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error computing levels for {symbol}: {e}")


async def gj_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Dedicated Gautam Jha Liquidity Level Analysis command."""
    symbol = ctx.args[0].upper() if ctx.args else DEFAULT_SYMBOL
    try:
        analysis = get_gautam_jha_analysis(symbol)
        msg = format_gautam_jha_message(analysis)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error computing Gautam Jha analysis for {symbol}: {e}")


def _parse_alert_args(args: List[str]) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """Parse alert command arguments flexibly."""
    if not args:
        return None, None, None, (
            "Usage: <code>/alert [SYMBOL] [CONDITION] &lt;PRICE&gt;</code>\n"
            "Examples:\n"
            "• <code>/alert 4180</code>\n"
            "• <code>/alert XAUTUSD 4180</code>\n"
            "• <code>/alert BTCUSD &lt;= 85000</code>"
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
        return None, None, None, "❌ Could not parse price. Example: <code>/alert 4180</code> or <code>/alert BTCUSD 86000</code>"

    symbol = tokens[0].upper() if tokens else DEFAULT_SYMBOL
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


async def list_alerts_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List active price alerts."""
    chat_id = update.effective_chat.id
    alerts = alert_manager.get_chat_alerts(chat_id)

    if not alerts:
        await update.message.reply_text(
            "📭 <b>You have no active price alerts.</b>\n\n"
            "Set one with <code>/alert &lt;PRICE&gt;</code> or <code>/alert &lt;SYMBOL&gt; &lt;PRICE&gt;</code>.",
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
    symbol = ctx.args[0].upper() if ctx.args else DEFAULT_SYMBOL
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
            symbol = arg.upper()

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


async def chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Chat with AI assistant with live market context injected."""
    chat_id = update.effective_chat.id
    text = update.message.text
    if not text:
        return

    # Detect symbol from text if mentioned, else default
    symbol = DEFAULT_SYMBOL
    for word in text.split():
        clean_word = word.strip("$#,!?.").upper()
        if clean_word.endswith("USD") and len(clean_word) >= 5:
            symbol = clean_word
            break

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
    symbol = DEFAULT_SYMBOL
    for word in caption.split():
        clean_word = word.strip("$#,!?.").upper()
        if clean_word.endswith("USD") and len(clean_word) >= 5:
            symbol = clean_word
            break

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


async def post_init(application):
    """Start background async monitoring tasks after bot initialization."""
    asyncio.create_task(price_alert_loop(application))
    asyncio.create_task(entry_alert_loop(application))
    logger.info("Background alert tasks successfully scheduled.")


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
    app.add_handler(CommandHandler("price", price_cmd))
    app.add_handler(CommandHandler("levels", levels_cmd))
    app.add_handler(CommandHandler("analysis", levels_cmd))
    app.add_handler(CommandHandler("gj", gj_cmd))
    app.add_handler(CommandHandler("liquidity", gj_cmd))
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
