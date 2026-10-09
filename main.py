"""
Telegram trading assistant powered by Gemini.

Setup:
  pip install -r requirements.txt
  export GEMINI_API_KEY=...      # from https://aistudio.google.com/apikey
  export TELEGRAM_TOKEN=...      # from @BotFather (or TELEGRAM_BOT_TOKEN)
Run:
  python main.py
  python main.py --check         # run system diagnostics

Commands:
  /start            intro and command list
  /price [SYMBOL]   live price from Delta Exchange (default XAUTUSD)
  /reset            clear conversation
  /help             show command guide
  any text          chat with the assistant (it sees the latest price)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from auto_trade import (
    get_auto_trader,
    AutoTrader,
    AutoTradeConfig,
    PinpointTradePlan,
    generate_pinpoint_plan,
    format_pinpoint_report,
    execute_pinpoint_plan,
    get_levels_report,
    calculate_risk_reward,
    make_modern_meter,
    TradeLevelAlert,
    ExchangeApiClient,
)

try:
    from health_server import start_health_server
except ImportError:
    try:
        from .health_server import start_health_server  # type: ignore
    except Exception:
        start_health_server = None

# Optional import for google-genai SDK (Python >= 3.9/3.10)
try:
    from google import genai
    from google.genai import types as genai_types

    _HAS_GENAI = True
except (ImportError, Exception):
    _HAS_GENAI = False
    genai = None
    genai_types = None

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
)
logger = logging.getLogger("trading_bot")

MODEL = os.getenv("GEMINI_MODEL") or os.getenv("MODEL") or "gemini-2.5-flash"
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "XAUTUSD")
DELTA_API = os.getenv("DELTA_API", "https://api.india.delta.exchange/v2/tickers")
BINANCE_API = os.getenv("BINANCE_API", "https://api.binance.com/api/v3")

# Fallback candidate models if primary model is unavailable
_raw_candidate_models = [
    MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
_seen: set[str] = set()
CANDIDATE_MODELS = [
    m for m in _raw_candidate_models if m and not (m in _seen or _seen.add(m))
]

SYSTEM = """You are a trading assistant on Telegram. Focus on technical analysis,
risk management, position sizing, and trade planning. Keep replies short and
mobile-friendly. When asked for entries, give entry, stop loss, take profit and
risk:reward, and state your assumptions. You are not a financial advisor; markets
are uncertain and the user makes the final decision. Never promise profits."""

SYMBOL_ALIASES: Dict[str, str] = {
    "BTC": "BTCUSD",
    "BITCOIN": "BTCUSD",
    "BTCUSDT": "BTCUSD",
    "ETH": "ETHUSD",
    "ETHEREUM": "ETHUSD",
    "ETHUSDT": "ETHUSD",
    "SOL": "SOLUSD",
    "SOLANA": "SOLUSD",
    "SOLUSDT": "SOLUSD",
    "XRP": "XRPUSD",
    "RIPPLE": "XRPUSD",
    "XRPUSDT": "XRPUSD",
    "XAU": "XAUTUSD",
    "XAUUSD": "XAUTUSD",
    "GOLD": "XAUTUSD",
    "PAXG": "XAUTUSD",
}

# Per-chat conversation histories: chat_id -> list of {"role": "...", "parts": [{"text": "..."}]}
histories: Dict[int, List[Dict[str, Any]]] = {}

# Per-chat cached latest trade plans: chat_id -> PinpointTradePlan
latest_trade_plans: Dict[int, PinpointTradePlan] = {}


def load_dotenv(filepath: str = ".env") -> None:
    """Load key-value pairs from .env into os.environ if not already present."""
    if not os.path.exists(filepath):
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        logger.warning("Could not read .env file: %s", exc)


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment variables."""
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("API_KEY_GEMINI")
    )


def get_telegram_token() -> Optional[str]:
    """Retrieve Telegram Bot Token from environment variables."""
    return (
        os.getenv("TELEGRAM_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("BOT_TOKEN")
    )


def get_binance_price(symbol: str) -> str:
    """Fetch live ticker data from Binance REST API."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
    target = ExchangeApiClient.format_binance_symbol(cleaned)
    try:
        r = requests.get(f"{BINANCE_API}/ticker/24hr?symbol={target}", timeout=10)
        r.raise_for_status()
        t = r.json()
        last_price = t.get("lastPrice", "0.0")
        change_pct = t.get("priceChangePercent", "0.0")
        high_price = t.get("highPrice", "0.0")
        low_price = t.get("lowPrice", "0.0")
        volume = t.get("volume", "0.0")
        return (
            f"{target} (Binance): last {last_price} ({change_pct}%) | "
            f"high {high_price} | low {low_price} | vol {volume}"
        )
    except Exception as e:
        return f"Binance price unavailable ({e})"


def get_binance_ticker_card(symbol: str) -> str:
    """Fetch live ticker data from Binance REST API and format as a rich HTML card."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
    target = ExchangeApiClient.format_binance_symbol(cleaned)
    try:
        r = requests.get(f"{BINANCE_API}/ticker/24hr?symbol={target}", timeout=10)
        r.raise_for_status()
        t = r.json()
        last_price = float(t.get("lastPrice", 0.0) or 0.0)
        high_price = float(t.get("highPrice", 0.0) or 0.0)
        low_price = float(t.get("lowPrice", 0.0) or 0.0)
        open_price = float(t.get("openPrice", 0.0) or 0.0)
        change_pct = float(t.get("priceChangePercent", 0.0) or 0.0)
        volume = float(t.get("volume", 0.0) or 0.0)
        quote_vol = float(t.get("quoteVolume", 0.0) or 0.0)
        bid = float(t.get("bidPrice", 0.0) or 0.0)
        ask = float(t.get("askPrice", 0.0) or 0.0)

        range_span = high_price - low_price
        range_pct = ((last_price - low_price) / range_span * 100.0) if range_span > 0 else 50.0
        range_meter = make_modern_meter(range_pct, width=10, fill_char="■", empty_char="░")
        arrow = "🟢 +" if change_pct >= 0 else "🔴 "

        if "BTC" in target:
            display_name = "Bitcoin (BTC)"
            icon = "⚡"
        elif "ETH" in target:
            display_name = "Ethereum (ETH)"
            icon = "🔷"
        elif "SOL" in target:
            display_name = "Solana (SOL)"
            icon = "🟣"
        elif "XRP" in target:
            display_name = "Ripple (XRP)"
            icon = "💧"
        elif "PAXG" in target or "XAU" in target:
            display_name = "PAX Gold (XAU)"
            icon = "🥇"
        else:
            display_name = target
            icon = "🟡"

        trend_badge = "🟢 <b>BULLISH</b>" if change_pct >= 0.5 else ("🔴 <b>BEARISH</b>" if change_pct <= -0.5 else "⚪ <b>RANGE</b>")
        quote_vol_str = f"${quote_vol/1_000_000:,.1f}M" if quote_vol >= 1_000_000 else f"${quote_vol:,.0f}"

        return (
            f"🟡 <b>{display_name} Binance Ticker</b> [{trend_badge}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Pair</b>: <code>{target}</code>\n"
            f"• <b>Last Price</b>: <code>${last_price:,.2f}</code>\n"
            f"• <b>24h Change</b>: {arrow}{change_pct:.2f}%\n"
            f"• <b>24h Range</b>: <code>[{range_meter}]</code>\n"
            f"  Low: <code>${low_price:,.2f}</code> ➔ High: <code>${high_price:,.2f}</code>\n"
            f"• <b>24h Volume</b>: <code>{volume:,.2f}</code> ({quote_vol_str})\n"
            f"• <b>Order Book</b>: Bid <code>${bid:,.2f}</code> | Ask <code>${ask:,.2f}</code>\n"
            f"• <b>Exchange</b>: Binance Public REST API\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 Actions: /entry {target} | /levels {target} | /price {target}</i>"
        )
    except Exception as e:
        return f"⚠️ Binance price unavailable for {symbol.upper()} ({e})"


def get_price(symbol: str) -> str:
    """Fetch live ticker data from Delta Exchange API (or Binance if prefixed)."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
    if cleaned.startswith("BINANCE"):
        parts = symbol.strip().split()
        b_sym = parts[1] if len(parts) > 1 else "BTC"
        return get_binance_price(b_sym)

    target = SYMBOL_ALIASES.get(cleaned, cleaned)
    try:
        r = requests.get(f"{DELTA_API}/{target}", timeout=10)
        r.raise_for_status()
        data = r.json()
        t = data.get("result")
        if not t:
            return f"Price unavailable: symbol '{symbol.upper()}' not found on Delta Exchange."
        mark_price = t.get("mark_price")
        close_price = t.get("close")
        high_price = t.get("high")
        low_price = t.get("low")
        open_price = t.get("open")
        return (
            f"{target}: mark {mark_price} | last {close_price} | "
            f"high {high_price} | low {low_price} | open {open_price}"
        )
    except Exception as e:
        return f"Price unavailable ({e})"


def get_ticker_card(symbol: str) -> str:
    """Fetch live ticker data from Delta Exchange or Binance API and format as a rich HTML card."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
    if cleaned.startswith("BINANCE"):
        parts = symbol.strip().split()
        b_sym = parts[1] if len(parts) > 1 else "BTC"
        return get_binance_ticker_card(b_sym)

    target = SYMBOL_ALIASES.get(cleaned, cleaned)
    try:
        r = requests.get(f"{DELTA_API}/{target}", timeout=10)
        r.raise_for_status()
        data = r.json()
        t = data.get("result")
        if not t:
            b_card = get_binance_ticker_card(cleaned)
            if not b_card.startswith("⚠️ Binance price unavailable"):
                return b_card
            return f"❌ Symbol <b>{symbol.upper()}</b> not found on Delta Exchange or Binance."
        mark_price = float(t.get("mark_price") or 0.0)
        close_price = float(t.get("close") or 0.0)
        high_price = float(t.get("high") or 0.0)
        low_price = float(t.get("low") or 0.0)
        open_price = float(t.get("open") or 0.0)
        volume = float(t.get("volume") or 0.0)

        if open_price > 0:
            change_pct = ((close_price - open_price) / open_price) * 100.0
        else:
            change_pct = 0.0

        spread = abs(mark_price - close_price) if (mark_price > 0 and close_price > 0) else 0.0
        range_span = high_price - low_price
        range_pct = ((close_price - low_price) / range_span * 100.0) if range_span > 0 else 50.0
        range_meter = make_modern_meter(range_pct, width=10, fill_char="■", empty_char="░")
        arrow = "🟢 +" if change_pct >= 0 else "🔴 "

        if "BTC" in target:
            display_name = "Bitcoin (BTC)"
            icon = "⚡"
        elif "XAU" in target:
            display_name = "Gold (XAU)"
            icon = "🥇"
        elif "ETH" in target:
            display_name = "Ethereum (ETH)"
            icon = "🔷"
        elif "SOL" in target:
            display_name = "Solana (SOL)"
            icon = "🟣"
        elif "XRP" in target:
            display_name = "Ripple (XRP)"
            icon = "💧"
        else:
            display_name = target
            icon = "📊"

        if change_pct >= 0.5:
            trend_badge = "🟢 <b>BULLISH</b>"
        elif change_pct <= -0.5:
            trend_badge = "🔴 <b>BEARISH</b>"
        else:
            trend_badge = "⚪ <b>RANGE</b>"

        return (
            f"{icon} <b>{display_name} Market Ticker</b> [{trend_badge}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Symbol</b>: <code>{target}</code>\n"
            f"• <b>Last Price</b>: <code>${close_price:,.2f}</code>\n"
            f"• <b>Mark Price</b>: <code>${mark_price:,.2f}</code> (Spread: ${spread:,.2f})\n"
            f"• <b>24h Change</b>: {arrow}{change_pct:.2f}%\n"
            f"• <b>24h Range</b>: <code>[{range_meter}]</code>\n"
            f"  Low: <code>${low_price:,.2f}</code> ➔ High: <code>${high_price:,.2f}</code>\n"
            f"• <b>24h Volume</b>: <code>{volume:,.2f}</code>\n"
            f"• <b>Exchange</b>: Delta Exchange API\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 Actions: /entry {target} | /levels {target} | /alert {target} {close_price:,.0f}</i>"
        )
    except Exception as e:
        b_card = get_binance_ticker_card(cleaned)
        if not b_card.startswith("⚠️ Binance price unavailable"):
            return b_card
        return f"⚠️ Price unavailable for {symbol.upper()} ({e})"


def call_gemini_rest(
    contents: List[Dict[str, Any]],
    system_instruction: str = SYSTEM,
    max_output_tokens: int = 800,
) -> str:
    """Call Google Gemini REST API directly using requests."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in environment or .env file."
        )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    last_error = "Unknown error"
    for model_name in CANDIDATE_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            data = resp.json()
            if resp.status_code == 200:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    return text or "No response, try again."
                return "No response candidates returned."

            error_data = data.get("error", {})
            err_msg = error_data.get("message", f"HTTP {resp.status_code}")
            last_error = f"{resp.status_code}: {err_msg}"

            # Only retry if the model was not found
            if resp.status_code == 404:
                logger.warning("Model %s returned 404, attempting fallback...", model_name)
                continue

            break
        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning("Network failure contacting Gemini with %s: %s", model_name, exc)

    raise RuntimeError(last_error)


def call_gemini(
    contents: List[Dict[str, Any]],
    system_instruction: str = SYSTEM,
    max_output_tokens: int = 800,
) -> str:
    """Unified entry point for Gemini generation (SDK if installed, REST otherwise)."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in environment or .env file."
        )

    # Use google-genai SDK if available
    if _HAS_GENAI and genai is not None and genai_types is not None:
        try:
            client = genai.Client(api_key=api_key)
            genai_contents = []
            for item in contents:
                role = item.get("role", "user")
                parts = []
                for p in item.get("parts", []):
                    parts.append(genai_types.Part.from_text(text=p.get("text", "")))
                genai_contents.append(genai_types.Content(role=role, parts=parts))

            reply = client.models.generate_content(
                model=MODEL,
                contents=genai_contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                ),
            )
            if reply.text:
                return reply.text
        except Exception as exc:
            logger.warning("google.genai SDK error (%s), using REST fallback.", exc)

    return call_gemini_rest(
        contents=contents,
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
    )


async def reply_safely(update: Update, text: str, parse_mode: Optional[str] = None) -> None:
    """Send reply respecting Telegram's 4096 character limit with clean line-break chunking."""
    msg = update.message or update.effective_message
    if not msg:
        return
    max_chunk = 4000
    kwargs = {"parse_mode": parse_mode} if parse_mode else {}
    if len(text) <= max_chunk:
        try:
            await msg.reply_text(text, **kwargs)
        except Exception:
            await msg.reply_text(text)
        return

    # Intelligent chunking respecting line breaks
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_chunk:
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0
            while len(line) > max_chunk:
                chunks.append(line[:max_chunk])
                line = line[max_chunk:]
        current_chunk.append(line)
        current_len += len(line)
    if current_chunk:
        chunks.append("".join(current_chunk))

    for chunk in chunks:
        try:
            await msg.reply_text(chunk, **kwargs)
        except Exception:
            await msg.reply_text(chunk)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await reply_safely(
        update,
        "⚡ <b>PRECISION TRADING ASSISTANT & AUTO-TRADER</b>\n"
        "Trading assistant ready. The Combined Ensemble System integrates Technical Indicators, "
        "ITB Machine Learning, and AI Adaptive Reinforcement for maximum precision.\n\n"
        "📊 <b>1. Market Analysis & Planning</b>\n"
        "• /market — Global market overview of major assets\n"
        "• /binance [SYM] — Live Binance ticker card, 24h metrics & order book\n"
        "• /scan [SYM] — Ultimate Master Scan (Deliberation, Trend, Levels, Entry)\n"
        "• /discussion [SYM] (/consensus) — 5-Layer Inter-Engine Deliberation forum & debate\n"
        "• /entry [SYM] — Pinpoint precise Entry, Stop Loss & Take Profit targets\n"
        "• /alert [SYM] [PRICE] — Set automatic price notifications\n\n"
        "🤖 <b>2. Auto-Trading Ensemble (ITB + Indicators + AI)</b>\n"
        "• /autotrade [on|off|status] — Toggle ensemble auto-trading\n"
        "• /symbols [add|rm] [SYM] — Manage actively traded pairs\n"
        "• /strategy — View the Combined Ensemble Strategy logic\n"
        "• /tpsl [TP] [SL] — Set global Take Profit & Stop Loss multipliers\n"
        "• /trailing [on|off] — Toggle dynamic trailing stop protection\n\n"
        "⚡ <b>3. Manual Execution & Positions</b>\n"
        "• /buy | /sell [LOTS] — Instant market orders\n"
        "• /position — View live open trades and unrealized PnL\n"
        "• /close [all|ID] — Close active positions instantly\n"
        "• /pnl — Performance dashboard & historical trades\n\n"
        "💼 <b>4. Funds & System Configuration</b>\n"
        "• /mode [paper|live] — Switch between simulated paper funds & real API execution\n"
        "• /capital — View and manage trading equity\n"
        "• /api set [binance|delta] [KEY] [SECRET] — Configure live exchange credentials\n"
        "• /api switch [binance|delta] — Switch active live trading exchange\n"
        "• /api ping — Test public exchange REST connectivity\n\n"
        "<i>💡 Pro Tip: All advanced legacy commands (e.g., /risk, /lotsize, /calc, /analyze) are still active for power users!</i>\n\n"
        "<i>💬 Send any message to converse with the AI market analyst.</i>",
        parse_mode="HTML",
    )


async def help_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, ctx)


async def autotrade_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    trader = get_auto_trader()
    chat_id = update.effective_chat.id if update.effective_chat else None
    args = [a.lower() for a in (ctx.args or [])]

    if not args or args[0] in ("status", "info"):
        await reply_safely(update, trader.get_status_text(), parse_mode="HTML")
        return

    sub = args[0]
    if sub in ("on", "start", "enable"):
        msg = trader.enable(chat_id=chat_id)
        syms_str = ", ".join(trader.config.symbols) if trader.config.symbols else trader.config.symbol
        reply = (
            f"🟢 <b>{msg}</b>\n\n"
            f"• <b>Monitored Pairs</b>: {syms_str}\n"
            f"• <b>Max Positions</b>: {trader.config.max_positions} (Max/pair: {trader.config.max_positions_per_symbol})\n"
            f"• <b>Lot Size</b>: {trader.config.lot_size}\n"
            f"• <b>TP</b>: {trader.config.tp_value} ({trader.config.tp_mode.upper()})\n"
            f"• <b>SL</b>: {trader.config.sl_value} ({trader.config.sl_mode.upper()})\n\n"
            f"Both BTC & Gold (and configured pairs) are actively scanned. "
            f"Trade signals and executions will be sent directly to this chat."
        )
        await reply_safely(update, reply, parse_mode="HTML")
    elif sub in ("off", "stop", "disable"):
        msg = trader.disable()
        await reply_safely(update, f"🔴 <b>{msg}</b>", parse_mode="HTML")
    elif sub == "close":
        if len(args) > 1:
            target = args[1]
            if target == "all":
                closed_msgs = trader.close_all_positions(reason="MANUAL")
                if closed_msgs:
                    await reply_safely(update, "🔄 <b>Closed All Positions:</b>\n\n" + "\n\n".join(closed_msgs), parse_mode="HTML")
                else:
                    await reply_safely(update, "No open positions to close.")
            elif trader.normalize_symbol(target) in [p.symbol for p in trader.positions]:
                closed_msgs = trader.close_positions_by_symbol(target, reason="MANUAL")
                await reply_safely(update, "\n\n".join(closed_msgs), parse_mode="HTML")
            else:
                res = trader.close_position_by_id(target, reason="MANUAL")
                if res:
                    await reply_safely(update, res, parse_mode="HTML")
                else:
                    await reply_safely(update, f"❌ Position <code>{target}</code> not found.", parse_mode="HTML")
        else:
            if not trader.positions:
                await reply_safely(update, "No active position to close.")
            elif len(trader.positions) == 1:
                res = trader.close_current_position(reason="MANUAL")
                await reply_safely(update, res or "Closed.", parse_mode="HTML")
            else:
                lines = ["⚠️ <b>Multiple Open Positions:</b> Specify which one to close:"]
                for p in trader.positions:
                    lines.append(f"• <code>/autotrade close {p.id}</code> ({p.direction} {p.symbol})")
                lines.append("• <code>/autotrade close all</code> (Close all open trades)")
                await reply_safely(update, "\n".join(lines), parse_mode="HTML")
    elif sub in ("symbol", "pair") and len(args) > 1:
        msg = trader.set_symbol(args[1])
        await reply_safely(update, msg)
    elif sub in ("symbols", "pairs"):
        if len(args) > 1:
            if args[1] == "both":
                ok, msg = trader.set_symbols(["BTCUSD", "XAUTUSD"])
            else:
                ok, msg = trader.set_symbols(args[1:])
            await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
        else:
            syms_str = ", ".join(trader.config.symbols)
            await reply_safely(update, f"Monitored symbols: {syms_str}")
    elif sub == "max" and len(args) > 1:
        try:
            total_m = int(args[1])
            per_m = int(args[2]) if len(args) > 2 else 1
            ok, msg = trader.set_max_positions(total_m, per_m)
            await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
        except ValueError:
            await reply_safely(update, "Usage: /autotrade max <total_positions> [per_pair_limit]")
    else:
        await reply_safely(
            update,
            "Usage:\n"
            "• /autotrade on - Turn auto trade ON (BTC & Gold)\n"
            "• /autotrade off - Turn auto trade OFF\n"
            "• /autotrade status - Show trading dashboard\n"
            "• /autotrade close [ID|SYM|all] - Close position(s)\n"
            "• /autotrade symbols [both|btc|gold] - Set active trading pairs\n"
            "• /autotrade max <total> [per_sym] - Configure max positions",
        )


async def symbols_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """View or configure multi-asset auto-trading pairs (BTC, Gold, etc.)."""
    trader = get_auto_trader()
    args = ctx.args or []

    if not args:
        syms_str = ", ".join(f"<code>{s}</code>" for s in trader.config.symbols)
        await reply_safely(
            update,
            f"🪙 <b>Active Auto-Trading Pairs</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Monitored Pairs</b>: {syms_str}\n"
            f"• <b>Max Positions Total</b>: <code>{trader.config.max_positions}</code>\n"
            f"• <b>Max Positions Per Pair</b>: <code>{trader.config.max_positions_per_symbol}</code>\n"
            f"• <b>Auto-Trading</b>: {'🟢 ENABLED' if trader.config.enabled else '🔴 DISABLED'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Commands:</b>\n"
            f"• <code>/symbols both</code> — Monitor & trade both BTC and Gold\n"
            f"• <code>/symbols btc gold eth</code> — Set custom active list\n"
            f"• <code>/symbols add sol</code> — Add Solana to trading list\n"
            f"• <code>/symbols remove xautusd</code> — Remove Gold from trading list\n"
            f"• <code>/symbols btc</code> — Monitor & trade Bitcoin only\n"
            f"• <code>/symbols gold</code> — Monitor & trade Gold only",
            parse_mode="HTML",
        )
        return

    sub = args[0].lower()
    if sub == "both":
        ok, msg = trader.set_symbols(["BTCUSD", "XAUTUSD"])
        await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
    elif sub == "add" and len(args) > 1:
        ok, msg = trader.add_symbol(args[1])
        await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
    elif sub in ("remove", "rm", "del") and len(args) > 1:
        ok, msg = trader.remove_symbol(args[1])
        await reply_safely(update, f"ℹ️ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
    elif sub in ("btc", "bitcoin") and len(args) == 1:
        ok, msg = trader.set_symbols(["BTCUSD"])
        await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
    elif sub in ("gold", "xau", "xautusd") and len(args) == 1:
        ok, msg = trader.set_symbols(["XAUTUSD"])
        await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")
    else:
        ok, msg = trader.set_symbols(args)
        await reply_safely(update, f"✅ <b>{msg}</b>" if ok else f"❌ {msg}", parse_mode="HTML")


async def close_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Close an active trade by ID, symbol, or all open positions."""
    trader = get_auto_trader()
    args = ctx.args or []

    if not trader.positions:
        await reply_safely(update, "💼 No active positions to close.")
        return

    if not args:
        if len(trader.positions) == 1:
            res = trader.close_current_position(reason="MANUAL")
            await reply_safely(update, res or "Closed.", parse_mode="HTML")
            return

        lines = [
            "⚠️ <b>Multiple Open Positions Found:</b>",
            "Specify which position to close, or close all:\n",
        ]
        for p in trader.positions:
            lines.append(f"• <code>/close {p.id}</code> — {p.direction} {p.symbol} @ {p.entry_price:.2f}")
        lines.append("\n• <code>/close all</code> — Close all open positions at market")
        lines.append("• <code>/close btc</code> — Close Bitcoin positions")
        lines.append("• <code>/close gold</code> — Close Gold positions")
        await reply_safely(update, "\n".join(lines), parse_mode="HTML")
        return

    target = args[0].strip()
    if target.lower() == "all":
        closed_msgs = trader.close_all_positions(reason="MANUAL")
        if closed_msgs:
            msg = f"🔄 <b>Closed {len(closed_msgs)} Position(s):</b>\n\n" + "\n\n".join(closed_msgs)
            await reply_safely(update, msg, parse_mode="HTML")
        else:
            await reply_safely(update, "No positions were closed.")
        return

    sym_normalized = trader.normalize_symbol(target)
    matched_sym = [p for p in trader.positions if p.symbol == sym_normalized]
    if matched_sym:
        closed_msgs = trader.close_positions_by_symbol(sym_normalized, reason="MANUAL")
        msg = f"🔄 <b>Closed {len(closed_msgs)} position(s) for {sym_normalized}:</b>\n\n" + "\n\n".join(closed_msgs)
        await reply_safely(update, msg, parse_mode="HTML")
        return

    res = trader.close_position_by_id(target, reason="MANUAL")
    if res:
        await reply_safely(update, res, parse_mode="HTML")
    else:
        await reply_safely(
            update,
            f"❌ Position <code>{target}</code> not found.\n"
            f"Use <code>/position</code> to view active positions and IDs, or <code>/close all</code> to exit all.",
            parse_mode="HTML",
        )


async def lotsize_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    trader = get_auto_trader()
    args = ctx.args or []

    if not args:
        lot_mode_desc = (
            "Fixed lot size"
            if trader.config.lot_mode in ("fixed", "manual")
            else f"Risk {trader.config.risk_pct}% sizing"
        )
        report = (
            f"📦 <b>Current Lot Size Configuration</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Active Mode</b>: {'🤖 <b>AUTO</b> (Dynamic Risk)' if trader.config.lot_mode in ('auto', 'risk_pct', 'risk') else '👤 <b>MANUAL</b> (Fixed Lots)'}\n"
            f"• <b>Mode</b>: {lot_mode_desc}\n"
            f"• <b>Manual Lot Size</b>: <code>{trader.config.lot_size}</code>\n"
            f"• <b>Auto Risk Sizing</b>: <code>{trader.config.risk_pct}%</code> equity\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Change lot size:</b>\n"
            f"• <code>/lotsize auto</code> (Enable dynamic auto lot sizing)\n"
            f"• <code>/lotsize auto 2%</code> (Auto sizing with 2% risk)\n"
            f"• <code>/lotsize manual 0.05</code> (Enable manual fixed lot size)\n"
            f"• <code>/lotsize 0.05</code> (Set fixed lot size to 0.05)\n"
            f"• <code>/lotsize 1.0</code> (Set fixed lot size to 1.0)\n"
            f"• <code>/lotsize risk 2%</code> (Set risk-based sizing to 2% equity)"
        )
        await reply_safely(update, report, parse_mode="HTML")
        return

    sub = args[0].lower().strip()

    # 1. /lotsize auto [risk%]
    if sub in ("auto", "dynamic"):
        risk_pct = None
        if len(args) > 1:
            try:
                risk_pct = float(args[1].replace("%", "").strip())
            except ValueError:
                await reply_safely(
                    update,
                    "Error: Invalid risk percentage. Example: <code>/lotsize auto 2%</code>",
                    parse_mode="HTML",
                )
                return
        ok, msg = trader.set_auto_lot(risk_pct=risk_pct)
        if ok:
            risk_usd = trader.config.equity * (trader.config.risk_pct / 100.0)
            await reply_safely(
                update,
                f"🤖 <b>Auto Lot Sizing Activated</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Mode</b>: Dynamic Risk-Based\n"
                f"• <b>Risk per Trade</b>: <code>{trader.config.risk_pct}%</code> equity (${risk_usd:,.2f})\n"
                f"• <b>Dynamic Formula</b>: <i>Lots = (Equity × Risk%) / SL Distance</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>💡 To change risk: <code>/lotsize auto 2%</code> or <code>/lotsize risk 2%</code>\n"
                f"💡 To switch to manual: <code>/lotsize manual 0.05</code></i>",
                parse_mode="HTML",
            )
        else:
            await reply_safely(update, f"❌ {msg}")
        return

    # 2. /lotsize manual [size] or /lotsize fixed [size]
    if sub in ("manual", "fixed"):
        lot_size = None
        if len(args) > 1:
            try:
                lot_size = float(args[1].strip())
            except ValueError:
                await reply_safely(
                    update,
                    "Error: Invalid lot size. Example: <code>/lotsize manual 0.05</code>",
                    parse_mode="HTML",
                )
                return
        ok, msg = trader.set_manual_lot(size=lot_size)
        if ok:
            await reply_safely(
                update,
                f"👤 <b>Manual Lot Sizing Activated</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• <b>Mode</b>: Fixed Volume\n"
                f"• <b>Lot Size</b>: <code>{trader.config.lot_size} lots</code> per trade\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>💡 To change volume: <code>/lotsize manual 0.05</code> or <code>/lotsize 0.05</code>\n"
                f"💡 To switch to auto: <code>/lotsize auto</code></i>",
                parse_mode="HTML",
            )
        else:
            await reply_safely(update, f"❌ {msg}")
        return

    # 3. Check for risk % format: /lotsize risk 2%
    if sub == "risk" and len(args) > 1:
        try:
            val = float(args[1].replace("%", "").strip())
            trader.config.risk_pct = val
            ok, msg = trader.set_lot_size(val, mode="risk_pct")
            await reply_safely(update, f"✅ {msg}")
        except ValueError:
            await reply_safely(update, "Error: Invalid risk percentage. Example: /lotsize risk 1.5")
        return

    # 4. /lotsize set <size>
    if sub == "set" and len(args) > 1:
        try:
            val = float(args[1].strip())
            ok, msg = trader.set_manual_lot(size=val)
            if ok:
                await reply_safely(update, f"✅ {msg}")
            else:
                await reply_safely(update, f"❌ {msg}")
        except ValueError:
            await reply_safely(update, "Error: Invalid lot size. Example: /lotsize set 0.05")
        return

    # 5. Direct numeric lot size or percentage
    raw = args[0].replace("%", "").strip()
    try:
        val = float(raw)
        if "%" in args[0]:
            trader.config.risk_pct = val
            ok, msg = trader.set_lot_size(val, mode="risk_pct")
        else:
            ok, msg = trader.set_lot_size(val, mode="fixed")
        if ok:
            await reply_safely(update, f"✅ {msg}")
        else:
            await reply_safely(update, f"❌ {msg}")
    except ValueError:
        await reply_safely(
            update,
            "Error: Invalid lot size format.\nUsage:\n"
            "• <code>/lotsize auto [risk%]</code> (e.g. /lotsize auto 2%)\n"
            "• <code>/lotsize manual [size]</code> (e.g. /lotsize manual 0.05)\n"
            "• <code>/lotsize 0.05</code> or <code>/lotsize risk 2%</code>",
            parse_mode="HTML",
        )


async def tpsl_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    trader = get_auto_trader()
    args = ctx.args or []

    if not args:
        await reply_safely(
            update,
            f"🎯 <b>Current Strategy TP / SL Configuration</b>\n"
            f"• <b>Take Profit</b>: <code>{trader.config.tp_value}</code> (mode: {trader.config.tp_mode.upper()})\n"
            f"• <b>Stop Loss</b>: <code>{trader.config.sl_value}</code> (mode: {trader.config.sl_mode.upper()})\n\n"
            f"<b>How to set on strategy:</b>\n"
            f"• <code>/tpsl 2.0 1.0</code> - Risk:Reward (2.0R TP, 1.0R SL)\n"
            f"• <code>/tpsl rr 2.5 1.0</code> - R:R ratio\n"
            f"• <code>/tpsl pts 40 20</code> - Price distance (40 pts TP, 20 pts SL)\n"
            f"• <code>/tpsl pct 1.5 0.75</code> - Percentage (1.5% TP, 0.75% SL)\n"
            f"• <code>/tpsl atr 3.0 1.5</code> - ATR multiple (3.0x ATR TP, 1.5x ATR SL)",
            parse_mode="HTML",
        )
        return

    # Parse format: /tpsl [mode] [tp] [sl] OR /tpsl [tp] [sl]
    mode = "rr"
    sl_mode = "swing"

    try:
        if args[0].lower() in ("rr", "pts", "pct", "atr"):
            mode = args[0].lower()
            sl_mode = mode
            if len(args) < 3:
                await reply_safely(update, f"Usage: /tpsl {mode} <tp_value> <sl_value>")
                return
            tp_val = float(args[1])
            sl_val = float(args[2])
        elif len(args) >= 2:
            tp_val = float(args[0])
            sl_val = float(args[1])
            mode = "rr"
            sl_mode = "swing"
        else:
            await reply_safely(update, "Usage: /tpsl <tp_value> <sl_value> or /tpsl <mode> <tp> <sl>")
            return

        ok, msg = trader.set_tp_sl(tp_val, sl_val, tp_mode=mode, sl_mode=sl_mode)
        if ok:
            await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
        else:
            await reply_safely(update, f"❌ {msg}")
    except ValueError:
        await reply_safely(update, "Error: TP and SL values must be numbers. Example: /tpsl 2.0 1.0")


async def price(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    symbol = ctx.args[0] if ctx.args else DEFAULT_SYMBOL
    await reply_safely(update, get_price(symbol))



async def scan_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Ultimate Master Scan combining 3-Engine Ensemble, MTF Trend, Levels, and Pinpoint Entry."""
    args = ctx.args or []
    symbol = args[0] if args else "BTCUSD"
    trader = get_auto_trader()
    
    msg = await reply_safely(
        update,
        f"🔄 <b>Scanning {symbol.upper()}...</b>\nGathering 3-Engine Ensemble Confluence, MTF Trend, and Smart Money levels...",
        parse_mode="HTML"
    )
    
    try:
        ensemble = trader.generate_ensemble_report(symbol)
        trend = trader.generate_mtf_trend_report(symbol)
        levels = trader.generate_order_blocks(symbol)
        entry = trader.generate_pinpoint_entry(symbol)
        
        part1 = f"🌐 <b>ULTIMATE MASTER SCAN: {symbol.upper()}</b>\n\n{ensemble}\n\n{trend}"
        part2 = f"{levels}\n\n{entry}"
        
        if len(part1 + "\n\n" + part2) <= 3900:
            if hasattr(msg, "edit_text"):
                await msg.edit_text(part1 + "\n\n" + part2, parse_mode="HTML")
            else:
                await reply_safely(update, part1 + "\n\n" + part2, parse_mode="HTML")
        else:
            if hasattr(msg, "edit_text"):
                await msg.edit_text(part1, parse_mode="HTML")
            else:
                await reply_safely(update, part1, parse_mode="HTML")
            await reply_safely(update, part2, parse_mode="HTML")
    except Exception as e:
        logger.error("Scan failed for %s: %s", symbol, e, exc_info=True)
        if hasattr(msg, "edit_text"):
            await msg.edit_text(f"⚠️ Scan failed: {e}")
        else:
            await reply_safely(update, f"⚠️ Scan failed: {e}")

async def market_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Combined Market Overview with Real-Time 3-Engine Ensemble Signals."""
    trader = get_auto_trader()
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "XAUTUSD"]
    msg = await reply_safely(update, "🔄 <b>Fetching Global Market Overview & Ensemble Confluence...</b>", parse_mode="HTML")
    
    lines = [
        "🌍 <b>GLOBAL MARKET OVERVIEW</b>",
        "<i>Real-time quotes & Combined 3-Engine Ensemble consensus</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]
    for sym in symbols:
        try:
            df = trader.fetch_candles(sym, count=120)
            if df is not None and not df.empty:
                current = float(df['close'].iloc[-1])
                open_p = float(df['open'].iloc[0])
                pct = ((current - open_p) / open_p) * 100.0
                icon = "🟢" if pct >= 0 else "🔴"
                ens = trader.evaluate_ensemble(sym)
                v_word = ens["verdict"].split()[0]
                v_badge = ("🟢 " + v_word) if "BUY" in ens["verdict"] else (("🔴 " + v_word) if "SELL" in ens["verdict"] else "⚪ NEUTRAL")
                lines.append(f"• <b>{sym.replace('USD','')}</b>: <code>${current:,.2f}</code> {icon} {pct:+.2f}% | <b>{v_badge}</b>")
        except Exception as e:
            logger.debug("Market overview error for %s: %s", sym, e)
            
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("<i>💡 Use /scan [SYM] for master scan or /entry [SYM] for pinpoint plan.</i>")
    report = "\n".join(lines)
    if hasattr(msg, "edit_text"):
        await msg.edit_text(report, parse_mode="HTML")
    else:
        await reply_safely(update, report, parse_mode="HTML")

async def btc_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Bitcoin market ticker and stats card."""
    card = get_ticker_card("BTCUSD")
    await reply_safely(update, card, parse_mode="HTML")


async def gold_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Gold market ticker and stats card."""
    card = get_ticker_card("XAUTUSD")
    await reply_safely(update, card, parse_mode="HTML")


async def binance_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Binance market ticker and stats card for symbol (default BTC)."""
    args = ctx.args or []
    sym = args[0] if args else "BTC"
    card = get_binance_ticker_card(sym)
    await reply_safely(update, card, parse_mode="HTML")


async def symbol_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """View or switch active auto-trade symbol."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        card = get_ticker_card(trader.config.symbol)
        await reply_safely(
            update,
            f"🔄 <b>Active Auto-Trade Symbol:</b> <code>{trader.config.symbol}</code>\n\n"
            f"{card}\n\n"
            f"<b>Switch symbol:</b>\n"
            f"• <code>/symbol btc</code> (Trade Bitcoin BTCUSD)\n"
            f"• <code>/symbol gold</code> (Trade Gold XAUTUSD)\n"
            f"• <code>/symbol [ANY_SYMBOL]</code> (e.g. ETHUSD, SOLUSD)",
            parse_mode="HTML",
        )
        return

    sym_req = args[0]
    msg = trader.set_symbol(sym_req)
    card = get_ticker_card(trader.config.symbol)
    await reply_safely(
        update,
        f"✅ <b>{msg}</b>\n\n{card}",
        parse_mode="HTML",
    )


async def eth_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Ethereum market ticker and stats card."""
    card = get_ticker_card("ETHUSD")
    await reply_safely(update, card, parse_mode="HTML")


async def sol_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Solana market ticker and stats card."""
    card = get_ticker_card("SOLUSD")
    await reply_safely(update, card, parse_mode="HTML")


async def xrp_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Ripple (XRP) market ticker and stats card."""
    card = get_ticker_card("XRPUSD")
    await reply_safely(update, card, parse_mode="HTML")



async def trend_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Run a multi-timeframe trend scanner."""
    args = ctx.args or []
    symbol = args[0] if args else "BTCUSD"
    trader = get_auto_trader()
    report = trader.generate_mtf_trend_report(symbol)
    await reply_safely(update, report, parse_mode="HTML")

async def analyze_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Run 10-indicator technical analysis engine and report signals."""
    trader = get_auto_trader()
    symbol = ctx.args[0] if ctx.args else trader.config.symbol
    report = trader.analyze_market(symbol)
    await reply_safely(update, report, parse_mode="HTML")


async def position_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show active position dashboard or idle status."""
    trader = get_auto_trader()
    text = trader.get_position_text()
    await reply_safely(update, text, parse_mode="HTML")


async def pnl_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show historical trading performance and closed trades PnL."""
    trader = get_auto_trader()
    report = trader.get_performance_report()
    await reply_safely(update, report, parse_mode="HTML")


async def trailing_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Configure dynamic trailing stop loss."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        state = "🟢 ENABLED (ON)" if trader.config.trailing_sl else "⚪ DISABLED (OFF)"
        await reply_safely(
            update,
            f"🛡️ <b>Trailing Stop Loss Status</b>\n"
            f"• <b>Status</b>: {state}\n\n"
            f"<b>Usage:</b>\n"
            f"• <code>/trailing on</code> - Enable trailing SL (locks in profits at 1R+)\n"
            f"• <code>/trailing off</code> - Disable trailing SL",
            parse_mode="HTML",
        )
        return

    sub = args[0].lower()
    if sub in ("on", "enable", "start", "1", "true"):
        msg = trader.set_trailing_sl(True)
        await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
    elif sub in ("off", "disable", "stop", "0", "false"):
        msg = trader.set_trailing_sl(False)
        await reply_safely(update, f"⚪ <b>{msg}</b>", parse_mode="HTML")
    else:
        await reply_safely(update, "Usage: /trailing on or /trailing off")


async def risk_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Configure maximum daily loss drawdown protection limit."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        await reply_safely(
            update,
            f"🛡️ <b>Capital Risk Protection</b>\n"
            f"• <b>Max Daily Loss Limit</b>: <code>{trader.config.max_daily_loss_pct}%</code> equity\n"
            f"• <b>Account Equity</b>: <code>${trader.config.equity:,.2f}</code>\n"
            f"• <b>Max Daily Drawdown</b>: <code>${trader.config.equity * trader.config.max_daily_loss_pct / 100.0:,.2f}</code>\n\n"
            f"<i>When daily loss reaches this threshold, auto-trading pauses automatically.</i>\n\n"
            f"<b>Change limit:</b>\n"
            f"• <code>/risk 2.5</code> (Set max daily drawdown to 2.5%)\n"
            f"• <code>/risk 5.0</code> (Set max daily drawdown to 5.0%)",
            parse_mode="HTML",
        )
        return

    raw = args[0].replace("%", "").strip()
    try:
        val = float(raw)
        ok, msg = trader.set_max_daily_loss(val)
        if ok:
            await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
        else:
            await reply_safely(update, f"❌ {msg}")
    except ValueError:
        await reply_safely(update, "Error: Invalid number. Example: /risk 3.0")


async def itb_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Intelligent Trading Bot (ITB) Machine Learning analysis, backtest, and training."""
    trader = get_auto_trader()
    args = ctx.args or []

    if not args:
        # Default: live analysis on primary symbol
        card = trader.get_itb_analysis()
        await reply_safely(update, card, parse_mode="HTML")
        return

    sub = args[0].lower()
    if sub in ("backtest", "bt", "sim", "simulate"):
        # /itb backtest [SYM] [COUNT]
        sym = args[1] if len(args) > 1 and not args[1].isdigit() else trader.config.symbol
        count = 150
        for a in args[1:]:
            if a.isdigit():
                count = max(40, min(2000, int(a)))
                break
        await reply_safely(
            update,
            f"⏳ <i>Running ITB backtest simulation for {sym.upper()} ({count} candles)...</i>",
            parse_mode="HTML",
        )
        report = trader.run_itb_backtest(symbol=sym, count=count)
        await reply_safely(update, report, parse_mode="HTML")
        return

    if sub in ("train", "fit", "learn"):
        # /itb train [SYM] [COUNT]
        sym = args[1] if len(args) > 1 and not args[1].isdigit() else trader.config.symbol
        count = 200
        for a in args[1:]:
            if a.isdigit():
                count = max(60, min(3000, int(a)))
                break
        await reply_safely(
            update,
            f"🧠 <i>Fitting ITB ML regression weights on {sym.upper()} ({count} candles)...</i>",
            parse_mode="HTML",
        )
        report = trader.train_itb_model(symbol=sym, count=count)
        await reply_safely(update, report, parse_mode="HTML")
        return

    # /itb [SYM]
    target_sym = trader.normalize_symbol(args[0])
    card = trader.get_itb_analysis(target_sym)
    await reply_safely(update, card, parse_mode="HTML")


async def discussion_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inter-engine deliberation forum dialogue where all 5 layers cross-examine each other."""
    trader = get_auto_trader()
    args = ctx.args or []
    target_sym = trader.normalize_symbol(args[0]) if args else trader.config.symbol
    report = trader.generate_deliberation_report(target_sym)
    await reply_safely(update, report, parse_mode="HTML")


async def strategy_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """View or configure active automated trading strategy."""
    trader = get_auto_trader()
    args = ctx.args or []
    if args:
        ok, msg = trader.set_strategy_type(args[0])
        await reply_safely(update, msg, parse_mode="HTML")
        return

    w_itb = getattr(trader.config, "weight_itb", 0.35) * 100.0
    w_pro = getattr(trader.config, "weight_pro", 0.35) * 100.0
    w_ai = getattr(trader.config, "weight_ai", 0.30) * 100.0
    cycles = getattr(trader.config, "learning_cycles", 0)
    last_retrain = getattr(trader.config, "last_retrain_time", "") or "Continuous background active"

    await reply_safely(
        update,
        f"🎯 <b>TRADING STRATEGY CONFIGURATION & ARCHITECTURE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Active System</b>: 🌟 <b>MULTI-LAYER DELIBERATIVE ENSEMBLE</b>\n"
        f"• <b>Deliberation Mode</b>: 🗣️ Continuous Mutual Cross-Examination\n"
        f"• <b>Autonomous Learning</b>: 🔄 Online Background Retraining (Cycle #{cycles})\n"
        f"• <b>Last Retrain</b>: <code>{last_retrain}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Dynamic Self-Learned Allocation:</b>\n"
        f"1. 🤖 <b>ITB Machine Learning</b>: <code>{w_itb:.1f}%</code>\n"
        f"   • Ridge regression forward trajectory & statistical moments\n"
        f"2. 📊 <b>Indicators Pro</b>: <code>{w_pro:.1f}%</code>\n"
        f"   • EMA trend filter, RSI momentum, ADX velocity & Supertrend\n"
        f"3. 🧠 <b>AI Learning Bot</b>: <code>{w_ai:.1f}%</code>\n"
        f"   • Permanent memory expectancy & dynamic regime calibration\n"
        f"4. 🧱 <b>SMC & Structure</b>: Institutional Order Blocks & FVGs\n"
        f"5. 🛡️ <b>Risk Guardian</b>: Cross-layer vetoes & conviction sizing arbiter\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>💡 All engines run together with mutual dependency. View live debate: <code>/discussion</code> or <code>/consensus</code></i>",
        parse_mode="HTML",
    )


async def mode_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Switch or inspect trading mode (paper vs live)."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        mode_icon = "📄 PAPER TRADING" if trader.config.trading_mode == "paper" else "🚨 LIVE TRADING"
        mode_color = "🟢" if trader.config.trading_mode == "paper" else "🔴"
        api_st = trader.exchange_client.get_masked_status()
        await reply_safely(
            update,
            f"🔄 <b>TRADING MODE CONFIGURATION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Active Mode</b>: {mode_color} <b>{mode_icon}</b>\n"
            f"• <b>Current Equity</b>: <code>${trader.config.equity:,.2f}</code>\n"
            f"• <b>Exchange API</b>: <code>{api_st['exchange'].upper()}</code> ({api_st['status']})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Switch Modes:</b>\n"
            f"• <code>/mode paper</code> — Simulated funds, real market prices, zero risk\n"
            f"• <code>/mode live</code> — Real orders dispatched via configured exchange API\n\n"
            f"<i>💡 To use live mode, configure API credentials first via <code>/api set</code></i>",
            parse_mode="HTML",
        )
        return

    req_mode = args[0].lower()
    ok, msg = trader.set_trading_mode(req_mode)
    await reply_safely(update, msg, parse_mode="HTML")


async def capital_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Manage paper trading funds and capital (set, add/deposit, reduce/withdraw, reset)."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        report = trader.get_capital_report()
        await reply_safely(update, report, parse_mode="HTML")
        return

    sub = args[0].lower()
    # Check if user typed: /capital 100 or /capital 500 directly
    try:
        direct_amt = float(sub.replace("$", "").replace(",", ""))
        ok, msg = trader.set_paper_capital(direct_amt)
        if ok:
            await reply_safely(update, f"✅ {msg}", parse_mode="HTML")
        else:
            await reply_safely(update, f"❌ {msg}")
        return
    except ValueError:
        pass

    if sub in ("set", "base"):
        if len(args) < 2:
            await reply_safely(
                update,
                "Usage: <code>/capital set &lt;amount&gt;</code>\nExample: <code>/capital set 100</code>",
                parse_mode="HTML",
            )
            return
        try:
            amt = float(args[1].replace("$", "").replace(",", ""))
            ok, msg = trader.set_paper_capital(amt)
            if ok:
                await reply_safely(update, f"✅ {msg}", parse_mode="HTML")
            else:
                await reply_safely(update, f"❌ {msg}")
        except ValueError:
            await reply_safely(update, "Error: Amount must be a valid number. Example: /capital set 100")

    elif sub in ("add", "deposit", "plus", "+"):
        if len(args) < 2:
            await reply_safely(
                update,
                "Usage: <code>/capital add &lt;amount&gt;</code> (or <code>/deposit &lt;amount&gt;</code>)\nExample: <code>/capital add 50</code>",
                parse_mode="HTML",
            )
            return
        try:
            amt = float(args[1].replace("$", "").replace(",", ""))
            ok, msg = trader.add_funds(amt)
            if ok:
                await reply_safely(update, msg, parse_mode="HTML")
            else:
                await reply_safely(update, f"❌ {msg}")
        except ValueError:
            await reply_safely(update, "Error: Amount must be a valid number. Example: /capital add 50")

    elif sub in ("reduce", "withdraw", "minus", "sub", "-"):
        if len(args) < 2:
            await reply_safely(
                update,
                "Usage: <code>/capital reduce &lt;amount&gt;</code> (or <code>/withdraw &lt;amount&gt;</code>)\nExample: <code>/capital reduce 25</code>",
                parse_mode="HTML",
            )
            return
        try:
            amt = float(args[1].replace("$", "").replace(",", ""))
            ok, msg = trader.reduce_funds(amt)
            if ok:
                await reply_safely(update, msg, parse_mode="HTML")
            else:
                await reply_safely(update, f"❌ {msg}")
        except ValueError:
            await reply_safely(update, "Error: Amount must be a valid number. Example: /capital reduce 25")

    elif sub in ("reset", "default"):
        target_amt = 100.0
        if len(args) > 1:
            try:
                target_amt = float(args[1].replace("$", "").replace(",", ""))
            except ValueError:
                pass
        ok, msg = trader.reset_funds(target_amt)
        await reply_safely(update, f"✅ {msg}", parse_mode="HTML")

    elif sub in ("status", "report", "show"):
        report = trader.get_capital_report()
        await reply_safely(update, report, parse_mode="HTML")

    else:
        await reply_safely(
            update,
            "<b>Capital Management Usage:</b>\n"
            "• <code>/capital</code> — View capital & funds dashboard\n"
            "• <code>/capital set 100</code> — Set base paper capital to $100\n"
            "• <code>/capital add 50</code> (or <code>/deposit 50</code>) — Add / deposit funds\n"
            "• <code>/capital reduce 20</code> (or <code>/withdraw 20</code>) — Reduce / withdraw funds\n"
            "• <code>/capital reset</code> — Reset back to default $100.00",
            parse_mode="HTML",
        )


async def deposit_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick deposit / add funds shortcut."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        await reply_safely(
            update,
            f"💰 <b>Deposit / Add Funds</b>\n\n"
            f"Current Balance: <code>${trader.config.equity:,.2f}</code>\n\n"
            f"<b>Usage:</b>\n"
            f"• <code>/deposit 50</code> — Deposit $50 into paper capital\n"
            f"• <code>/deposit 100</code> — Deposit $100 into paper capital",
            parse_mode="HTML",
        )
        return
    try:
        amt = float(args[0].replace("$", "").replace(",", ""))
        ok, msg = trader.add_funds(amt)
        if ok:
            await reply_safely(update, msg, parse_mode="HTML")
        else:
            await reply_safely(update, f"❌ {msg}")
    except ValueError:
        await reply_safely(update, "Error: Amount must be a valid number. Example: /deposit 50")


async def withdraw_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick withdraw / reduce funds shortcut."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args:
        await reply_safely(
            update,
            f"💸 <b>Withdraw / Reduce Funds</b>\n\n"
            f"Current Balance: <code>${trader.config.equity:,.2f}</code>\n\n"
            f"<b>Usage:</b>\n"
            f"• <code>/withdraw 25</code> — Reduce $25 from paper capital\n"
            f"• <code>/withdraw 50</code> — Reduce $50 from paper capital",
            parse_mode="HTML",
        )
        return
    try:
        amt = float(args[0].replace("$", "").replace(",", ""))
        ok, msg = trader.reduce_funds(amt)
        if ok:
            await reply_safely(update, msg, parse_mode="HTML")
        else:
            await reply_safely(update, f"❌ {msg}")
    except ValueError:
        await reply_safely(update, "Error: Amount must be a valid number. Example: /withdraw 25")


async def api_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Live exchange API configuration, authentication test, and balance query."""
    trader = get_auto_trader()
    args = ctx.args or []
    if not args or args[0].lower() in ("status", "info"):
        report = trader.get_api_status_report()
        await reply_safely(update, report, parse_mode="HTML")
        return

    sub = args[0].lower()
    if sub == "test":
        ok, msg, data = trader.exchange_client.test_connection()
        if ok:
            bal_str = ""
            ok_bal, bal_val, _ = trader.exchange_client.get_balance()
            if ok_bal:
                bal_str = f"\n• <b>Live Wallet Balance</b>: <code>${bal_val:,.2f} USDT</code>"
            await reply_safely(
                update,
                f"✅ <b>Exchange API Connected Successfully</b>\n"
                f"• Exchange: <code>{trader.config.live_exchange.upper()}</code>\n"
                f"• Result: <i>{msg}</i>{bal_str}\n\n"
                f"<i>Ready for live trading. Switch mode with /mode live</i>",
                parse_mode="HTML",
            )
        else:
            await reply_safely(
                update,
                f"❌ <b>Exchange API Connection Failed</b>\n"
                f"• Exchange: <code>{trader.config.live_exchange.upper()}</code>\n"
                f"• Error: <i>{msg}</i>\n\n"
                f"Check your API Key and Secret with <code>/api set</code>",
                parse_mode="HTML",
            )
    elif sub in ("switch", "use"):
        if len(args) >= 2:
            target_ex = args[1].lower()
            ok, msg = trader.switch_exchange(target_ex)
            await reply_safely(update, msg, parse_mode="HTML")
        else:
            await reply_safely(
                update,
                "Usage: <code>/api switch binance</code> or <code>/api switch delta</code>",
                parse_mode="HTML",
            )
    elif sub in ("ping", "public"):
        ok, msg = trader.exchange_client.test_public_connection()
        icon = "🟢" if ok else "🔴"
        await reply_safely(
            update,
            f"{icon} <b>Exchange Public API Status</b>\n"
            f"• Exchange: <code>{trader.config.live_exchange.upper()}</code>\n"
            f"• Result: <i>{msg}</i>",
            parse_mode="HTML",
        )
    elif sub == "clear":
        msg = trader.clear_exchange_api()
        await reply_safely(update, f"🧹 {msg}")
    elif sub == "set":
        # /api set <exchange> <key> <secret> OR /api set <key> <secret>
        if len(args) == 4:
            ex = args[1]
            key = args[2]
            sec = args[3]
        elif len(args) == 3:
            ex = trader.config.live_exchange
            key = args[1]
            sec = args[2]
        else:
            await reply_safely(
                update,
                "Usage:\n"
                "• <code>/api set binance &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>\n"
                "• <code>/api set delta &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>\n"
                "• <code>/api switch [binance|delta]</code>\n"
                "• <code>/api ping</code>",
                parse_mode="HTML",
            )
            return
        ok, msg = trader.set_exchange_api(ex, key, sec)
        await reply_safely(update, msg, parse_mode="HTML")
    else:
        report = trader.get_api_status_report()
        await reply_safely(update, report, parse_mode="HTML")


async def entry_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze pinpoint trade entry, precision SL (Swing + ATR buffer), and multi-tier TP targets."""
    trader = get_auto_trader()
    args = ctx.args or []
    symbol = trader.config.symbol
    direction_override = None

    if len(args) == 1:
        arg_up = args[0].upper()
        if arg_up in ("BUY", "LONG", "SELL", "SHORT"):
            direction_override = arg_up
        else:
            symbol = args[0]
    elif len(args) >= 2:
        symbol = args[0]
        direction_override = args[1].upper()

    plan = generate_pinpoint_plan(symbol, direction_override=direction_override, trader=trader)
    if update.effective_chat:
        latest_trade_plans[update.effective_chat.id] = plan

    report = format_pinpoint_report(plan)
    await reply_safely(update, report, parse_mode="HTML")


async def execute_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute a pinpoint trade plan immediately (at market, limit, or breakout)."""
    trader = get_auto_trader()
    chat_id = update.effective_chat.id if update.effective_chat else 0
    args = ctx.args or []

    cmd_name = ""
    if update.message and update.message.text:
        cmd_parts = update.message.text.split()
        if cmd_parts:
            cmd_name = cmd_parts[0].lower().lstrip("/").split("@")[0]

    plan = latest_trade_plans.get(chat_id)
    entry_mode = "market"
    lot_override = None

    target_sym = None
    for a in args:
        a_lower = a.lower()
        if a_lower in ("limit", "pullback"):
            entry_mode = "limit"
        elif a_lower in ("breakout", "bo"):
            entry_mode = "breakout"
        elif a_lower in ("market", "now"):
            entry_mode = "market"
        else:
            try:
                lot_override = float(a)
            except ValueError:
                norm = trader.normalize_symbol(a)
                if norm in ("BTCUSD", "XAUTUSD", "ETHUSD", "SOLUSD", "XRPUSD") or any(
                    k in a.upper() for k in ("BTC", "ETH", "SOL", "XRP", "XAU", "GOLD")
                ):
                    target_sym = norm

    exec_sym = target_sym or trader.config.symbol
    if cmd_name in ("buy", "long"):
        if not plan or plan.direction != "LONG" or (target_sym and plan.symbol != target_sym):
            plan = generate_pinpoint_plan(exec_sym, direction_override="LONG", trader=trader)
            latest_trade_plans[chat_id] = plan
    elif cmd_name in ("sell", "short"):
        if not plan or plan.direction != "SHORT" or (target_sym and plan.symbol != target_sym):
            plan = generate_pinpoint_plan(exec_sym, direction_override="SHORT", trader=trader)
            latest_trade_plans[chat_id] = plan
    elif not plan or (target_sym and plan.symbol != target_sym):
        plan = generate_pinpoint_plan(exec_sym, trader=trader)
        latest_trade_plans[chat_id] = plan

    ok, msg = execute_pinpoint_plan(plan, lot_override=lot_override, entry_mode=entry_mode, trader=trader)
    await reply_safely(update, msg, parse_mode="HTML")


async def calc_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Position sizing and Risk:Reward ratio calculator."""
    trader = get_auto_trader()
    args = ctx.args or []

    if not args:
        plan = latest_trade_plans.get(update.effective_chat.id if update.effective_chat else 0)
        if plan:
            res = calculate_risk_reward(
                entry=plan.market_entry,
                sl=plan.stop_loss,
                tp=plan.take_profit_2,
                equity=trader.config.equity,
                symbol=plan.symbol,
            )
            await reply_safely(update, res, parse_mode="HTML")
            return

        await reply_safely(
            update,
            "📐 <b>Position Sizing & Risk:Reward Calculator</b>\n\n"
            "<b>Usage:</b>\n"
            "• <code>/calc &lt;entry&gt; &lt;sl&gt;</code> - Calculate R:R and lot sizing\n"
            "• <code>/calc &lt;entry&gt; &lt;sl&gt; &lt;tp&gt;</code> - Calculate specific target R:R\n"
            "• <code>/calc &lt;entry&gt; &lt;sl&gt; &lt;tp&gt; &lt;risk_usd&gt;</code> - Custom risk budget\n\n"
            "<b>Examples:</b>\n"
            "• <code>/calc 81700 81200</code> (BTC Long / Short)\n"
            "• <code>/calc 81700 81200 82800 150</code>\n"
            "• <code>/calc 4135 4115 4175 100</code> (Gold XAU)",
            parse_mode="HTML",
        )
        return

    try:
        raw_args = list(args)
        symbol = trader.config.symbol
        try:
            float(raw_args[0])
        except ValueError:
            symbol = trader.normalize_symbol(raw_args.pop(0))

        if len(raw_args) < 2:
            raise IndexError("Entry and Stop Loss are required.")

        entry = float(raw_args[0])
        sl = float(raw_args[1])
        tp = float(raw_args[2]) if len(raw_args) > 2 else None
        risk_usd = float(raw_args[3]) if len(raw_args) > 3 else None
        res = calculate_risk_reward(
            entry=entry,
            sl=sl,
            tp=tp,
            risk_dollars=risk_usd,
            equity=trader.config.equity,
            symbol=symbol,
        )
        await reply_safely(update, res, parse_mode="HTML")
    except (IndexError, ValueError):
        await reply_safely(
            update,
            "❌ Invalid parameters. Numbers required.\nExample: <code>/calc 81700 81200 82700</code> or <code>/calc BTC 81700 81200</code>",
            parse_mode="HTML",
        )


async def levels_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Institutional Order Blocks (OB) & Fair Value Gaps (FVG) scanner."""
    trader = get_auto_trader()
    args = ctx.args or []
    symbol = args[0] if args else trader.config.symbol
    report = get_levels_report(symbol, trader=trader)
    await reply_safely(update, report, parse_mode="HTML")


async def alert_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Trade level alert manager: set price alerts, list active alerts, or remove alerts."""
    trader = get_auto_trader()
    args = ctx.args or []

    if not args or args[0].lower() in ("list", "view", "status"):
        report = trader.get_alerts_report()
        await reply_safely(update, report, parse_mode="HTML")
        return

    sub = args[0].lower()
    if sub in ("del", "delete", "rm", "remove"):
        if len(args) < 2:
            await reply_safely(
                update,
                "⚠️ Usage: <code>/alert del &lt;ALERT_ID&gt;</code>\nExample: <code>/alert del ALT_BTC_123456</code>",
                parse_mode="HTML",
            )
            return
        alert_id = args[1].strip()
        ok, _ = trader.remove_alert(alert_id)
        if ok:
            await reply_safely(
                update,
                f"✅ Alert <code>{alert_id}</code> removed successfully.",
                parse_mode="HTML",
            )
        else:
            await reply_safely(
                update,
                f"❌ Alert <code>{alert_id}</code> not found.\nUse <code>/alert list</code> to inspect active alerts.",
                parse_mode="HTML",
            )
        return

    if sub in ("clear", "reset", "wipe"):
        sym_filter = args[1].strip() if len(args) > 1 else None
        cleared_cnt = trader.clear_alerts(symbol=sym_filter)
        scope = f"for {sym_filter.upper()}" if sym_filter else "across all symbols"
        await reply_safely(
            update,
            f"🧹 Cleared <b>{cleared_cnt}</b> trade level alert(s) {scope}.",
            parse_mode="HTML",
        )
        return

    # Flexible syntax:
    # 1. /alert BTC 85000 [above|below] [note...]
    # 2. /alert 85000 [above|below] [note...] (uses active trader symbol)
    symbol = trader.config.symbol
    target_price: Optional[float] = None
    cond_arg: Optional[str] = None
    note_words: List[str] = []

    try:
        val = float(args[0].replace("$", "").replace(",", ""))
        target_price = val
        if len(args) > 1:
            cond_arg = args[1].lower()
            note_words = args[2:]
    except ValueError:
        symbol = trader.normalize_symbol(args[0])
        if len(args) > 1:
            try:
                target_price = float(args[1].replace("$", "").replace(",", ""))
            except ValueError:
                target_price = None
            if len(args) > 2:
                cond_arg = args[2].lower()
                note_words = args[3:]
        else:
            target_price = None

    if target_price is None or target_price <= 0:
        await reply_safely(
            update,
            "⚠️ <b>Trade Alert Usage Guide:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "• <code>/alert BTC 85000 above</code> — Alert when BTC rises above $85,000\n"
            "• <code>/alert GOLD 4200 below</code> — Alert when Gold drops below $4,200\n"
            "• <code>/alert 82500</code> — Alert at $82,500 (auto-detect condition)\n"
            "• <code>/alert list</code> (or <code>/alerts</code>) — View active alerts\n"
            "• <code>/alert del &lt;ID&gt;</code> — Delete alert by ID\n"
            "• <code>/alert clear</code> — Wipe all active alerts\n"
            "━━━━━━━━━━━━━━━━━━━━━━",
            parse_mode="HTML",
        )
        return

    condition = "AUTO"
    if cond_arg in ("above", "cross_above", ">", ">="):
        condition = "CROSS_ABOVE"
    elif cond_arg in ("below", "cross_below", "<", "<="):
        condition = "CROSS_BELOW"
    elif cond_arg in ("touch", "at", "=="):
        condition = "TOUCH"
    elif cond_arg:
        note_words = [cond_arg] + note_words
        condition = "AUTO"

    note = " ".join(note_words).strip()
    chat_id = update.effective_chat.id if (update and update.effective_chat) else None

    ok, msg, alt = trader.add_alert(
        symbol=symbol,
        target_price=target_price,
        condition=condition,
        note=note,
        chat_id=chat_id,
        one_shot=True,
    )
    if not ok or not alt:
        await reply_safely(update, f"❌ Failed to create alert: {msg}")
        return

    cond_badge = "🟢 CROSS ABOVE" if alt.condition == "CROSS_ABOVE" else ("🔴 CROSS BELOW" if alt.condition == "CROSS_BELOW" else alt.condition)
    note_str = f"• <b>Note</b>: <i>{alt.note}</i>\n" if alt.note else ""

    await reply_safely(
        update,
        f"🔔 <b>Trade Level Alert Set</b> [🟢 ACTIVE]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Alert ID</b>: <code>{alt.id}</code>\n"
        f"• <b>Symbol</b>: <code>{alt.symbol}</code>\n"
        f"• <b>Target Level</b>: <code>${alt.target_price:,.2f}</code>\n"
        f"• <b>Trigger Rule</b>: {cond_badge}\n"
        f"{note_str}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>💡 You will receive an instant notification when {alt.symbol} reaches this level.</i>",
        parse_mode="HTML",
    )


async def menu_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Interactive command directory and quick dashboard."""
    trader = get_auto_trader()
    status_str = "🟢 AUTO-TRADING ON" if trader.config.enabled else "🔴 AUTO-TRADING OFF"
    mode_str = "📄 PAPER" if trader.config.trading_mode == "paper" else "🚨 LIVE"
    pos_count = len(trader.positions)
    pos_str = f"{pos_count} active position(s)" if pos_count > 0 else "0 open"
    symbols_str = ", ".join(trader.config.symbols) if trader.config.symbols else trader.config.symbol
    strat_str = "Deliberative Multi-Layer Ensemble (5 Engines)"
    await reply_safely(
        update,
        f"🎛️ <b>TRADING BOT COMMAND MENU</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Engine Status</b>: <b>{status_str}</b>\n"
        f"• <b>Trading Mode</b>: <b>{mode_str}</b> (Equity: <code>${trader.config.equity:,.2f}</code>)\n"
        f"• <b>Monitored Pairs</b>: <code>{symbols_str}</code>\n"
        f"• <b>Active Strategy</b>: <code>{strat_str}</code>\n"
        f"• <b>Positions</b>: <i>{pos_str} (Max {trader.config.max_positions})</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>📊 1. Market & Quotes:</b>\n"
        f"• /market — Global overview of BTC, ETH, SOL, XRP, GOLD\n        • /btc — Bitcoin ticker card & 24h stats\n"
        f"• /gold — Gold ticker card & 24h stats\n"
        f"• /eth | /sol | /xrp — Ethereum, Solana & Ripple tickers\n"
        f"• /scan [SYM] — Ultimate Master Scan (Deliberation + Trend + Levels)\n"
        f"• /discussion [SYM] (/consensus) — 5-Layer Deliberation forum & cross-critique\n"
        f"• /analyze [SYM] — 10-indicator confluence report\n        • /trend [SYM] — Multi-timeframe trend scanner\n"
        f"• /price [SYM] — Live price check\n\n"
        f"<b>🎯 2. Pinpoint Trade Planning:</b>\n"
        f"• /entry [SYM] — Pinpoint entry, precision SL & TPs\n"
        f"• /levels [SYM] — Order blocks & FVGs scanner\n"
        f"• /calc [ENTRY] [SL] [TP] — Position sizing & R:R\n\n"
        f"<b>⚡ 3. Auto-Trading & Execution:</b>\n"
        f"• /itb [SYM] — Live Intelligent Indicator & feature moments\n"
        f"• /itb backtest [SYM] [N] — ITB simulated backtest\n"
        f"• /itb train [SYM] — Fit ML ridge regression weights\n"
        f"• /strategy — View Combined Ensemble status\n"
        f"• /autotrade on|off — Toggle automated trading\n"
        f"• /execute — One-tap execute pinpoint trade plan\n"
        f"• /buy | /sell — Instant market execution\n"
        f"• /position — Open positions & unrealized PnL\n"
        f"• /close [ID|SYM|all] — Close open trade(s)\n"
        f"• /pnl — Performance & closed trades history\n\n"
        f"<b>💼 4. Paper/Live Mode & Capital:</b>\n"
        f"• /mode [paper|live] — Switch trading mode\n"
        f"• /capital (/funds) — View capital dashboard\n"
        f"• /capital set 100 — Set paper capital ($100 default)\n"
        f"• /deposit [AMT] — Add / deposit funds to balance\n"
        f"• /withdraw [AMT] — Reduce / withdraw funds\n"
        f"• /capital reset — Reset paper funds to $100\n\n"
        f"<b>🔌 5. Live Exchange API System:</b>\n"
        f"• /api — View API status & connectivity\n"
        f"• /api set delta &lt;KEY&gt; &lt;SECRET&gt; — Setup Delta API\n"
        f"• /api test — Test auth & query wallet balance\n"
        f"• /api clear — Clear API & return to paper mode\n\n"
        f"<b>🛡️ 6. Risk & Strategy:</b>\n"
        f"• /strategy — Inspect Combined Ensemble\n"
        f"• /symbols — View or configure active pairs\n"
        f"• /symbol [SYM] — Switch primary symbol\n"
        f"• /lotsize [SIZE|risk %] — Configure lot sizing\n"
        f"• /tpsl [TP] [SL] — Set strategy TP & SL\n"
        f"• /trailing on|off — Trailing stop loss guard\n"
        f"• /risk [PCT] — Daily drawdown safety limit\n\n"
        f"<b>🔔 7. Trade Level Alerts:</b>\n"
        f"• /alert [SYM] [PRICE] [above|below] — Set price target alert\n"
        f"• /alert list (/alerts) — View active trade alerts\n"
        f"• /alert del &lt;ID&gt; — Remove alert by ID\n"
        f"• /alert clear [SYM] — Clear all active alerts\n\n"
        f"<b>⚙️ 8. Assistant & Settings:</b>\n"
        f"• /help — Full command guide\n"
        f"• /reset — Clear AI conversation",
        parse_mode="HTML",
    )


async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    histories.pop(update.effective_chat.id, None)
    await reply_safely(update, "Conversation cleared.")


async def chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text or not update.effective_chat:
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    chat_id = update.effective_chat.id
    hist = histories.setdefault(chat_id, [])

    context = f"[Market data] {get_price(DEFAULT_SYMBOL)}\n\n"
    user_turn = {
        "role": "user",
        "parts": [{"text": context + user_text}],
    }
    hist.append(user_turn)

    # Retain the last 20 messages and ensure conversation begins with user role
    hist[:] = hist[-20:]
    while hist and hist[0].get("role") != "user":
        hist.pop(0)

    try:
        answer = call_gemini(hist)
        hist.append({"role": "model", "parts": [{"text": answer}]})
    except Exception as e:
        # Prevent dialogue context corruption on errors
        if hist and hist[-1] == user_turn:
            hist.pop()
        answer = f"Error: {e}"

    await reply_safely(update, answer)


def run_diagnostics() -> bool:
    """Validate system configuration, network connectivity, and credentials."""
    load_dotenv()
    print("=" * 60)
    print(" TRADING BOT SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print(f" Python Version   : {sys.version.split()[0]}")
    print(f" Default Model    : {MODEL}")
    print(f" Default Symbol   : {DEFAULT_SYMBOL}")
    print(f" GenAI Mode       : {'google-genai SDK' if _HAS_GENAI else 'REST API (requests)'}")

    # 1. Delta API Check
    print("\n[1] Testing Delta Exchange API...")
    price_output = get_price(DEFAULT_SYMBOL)
    delta_ok = not price_output.startswith("Price unavailable")
    print(f"    Ticker Result : {price_output}")
    print(f"    Status        : {'OK' if delta_ok else 'FAILED'}")

    # 2. Binance API Check
    print("\n[2] Testing Binance REST API...")
    binance_output = get_binance_price("BTC")
    binance_ok = not binance_output.startswith("Binance price unavailable")
    print(f"    Ticker Result : {binance_output}")
    print(f"    Status        : {'OK' if binance_ok else 'FAILED'}")

    # 3. Telegram Token Check
    print("\n[3] Checking Telegram Token...")
    token = get_telegram_token()
    if token:
        masked = token[:6] + "..." + token[-4:] if len(token) > 10 else "***"
        print(f"    Token         : {masked}")
        print("    Status        : CONFIGURED")
    else:
        print("    Status        : MISSING (set TELEGRAM_TOKEN or TELEGRAM_BOT_TOKEN)")

    # 4. Gemini API Key Check
    print("\n[4] Checking Gemini API Key...")
    api_key = get_gemini_api_key()
    if api_key:
        masked_k = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"    API Key       : {masked_k}")
        print("    Status        : CONFIGURED")
    else:
        print("    Status        : MISSING (set GEMINI_API_KEY in environment or .env)")

    # 5. Active Live Exchange Credentials Check
    print("\n[5] Checking Exchange Credentials...")
    trader = get_auto_trader()
    ex_status = trader.exchange_client.get_masked_status()
    print(f"    Active Exch   : {ex_status['exchange']}")
    print(f"    API Key       : {ex_status['api_key']}")
    print(f"    Status        : {ex_status['status']}")

    print("=" * 60)
    all_ready = bool(token and api_key and (delta_ok or binance_ok))
    print(f" OVERALL STATUS   : {'READY TO RUN' if all_ready else 'CONFIGURATION PENDING'}")
    print("=" * 60)
    return all_ready


async def auto_trade_worker(app: Any) -> None:
    """Periodic background worker checking market and executing strategy."""
    trader = get_auto_trader()
    trader.is_running = True
    logger.info("Auto-trade background worker started.")
    try:
        while trader.is_running:
            try:
                notifications = trader.step()
                if notifications and trader.config.notify_chat_id:
                    for note in notifications:
                        if not trader.is_running:
                            break
                        try:
                            # Verify app is still active before attempting to send message
                            if hasattr(app, "updater") and app.updater and not app.updater.running:
                                break
                            await app.bot.send_message(
                                chat_id=trader.config.notify_chat_id,
                                text=note,
                                parse_mode="HTML",
                            )
                        except Exception as note_err:
                            logger.warning("Failed to send auto-trade notification: %s", note_err)
            except Exception as step_err:
                logger.warning("Error in auto-trade step: %s", step_err)

            try:
                await asyncio.sleep(trader.config.poll_seconds)
            except asyncio.CancelledError:
                break
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Auto-trade background worker stopped.")


async def on_post_init(application: Any) -> None:
    asyncio.create_task(auto_trade_worker(application))
    if start_health_server is not None:
        try:
            asyncio.create_task(start_health_server())
        except Exception as exc:
            logger.warning("Could not schedule health server: %s", exc)


async def on_post_shutdown(application: Any) -> None:
    trader = get_auto_trader()
    trader.is_running = False



def main() -> None:
    load_dotenv()

    if "--check" in sys.argv or "--test" in sys.argv:
        run_diagnostics()
        return

    telegram_token = get_telegram_token()
    if not telegram_token:
        print(
            "Error: TELEGRAM_TOKEN (or TELEGRAM_BOT_TOKEN) is not set.\n"
            "Please configure your Telegram bot token:\n"
            "  export TELEGRAM_TOKEN=your_token_here\n"
            "or create a .env file with TELEGRAM_TOKEN=your_token_here\n"
            "Run diagnostics with: python main.py --check",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = get_gemini_api_key()
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY is not set. /price will work, but AI chat will report that the key is missing."
        )

    logger.info("Initializing Telegram bot application...")
    app = (
        ApplicationBuilder()
        .token(telegram_token)
        .post_init(on_post_init)
        .post_shutdown(on_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("pro", scan_command))
    app.add_handler(CommandHandler("market", market_command))
    app.add_handler(CommandHandler("markets", market_command))

    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("btc", btc_command))
    app.add_handler(CommandHandler("gold", gold_command))
    app.add_handler(CommandHandler("xau", gold_command))
    app.add_handler(CommandHandler("eth", eth_command))
    app.add_handler(CommandHandler("sol", sol_command))
    app.add_handler(CommandHandler("xrp", xrp_command))
    app.add_handler(CommandHandler("binance", binance_command))
    app.add_handler(CommandHandler("entry", entry_command))
    app.add_handler(CommandHandler("pinpoint", entry_command))
    app.add_handler(CommandHandler("execute", execute_command))
    app.add_handler(CommandHandler("buy", execute_command))
    app.add_handler(CommandHandler("sell", execute_command))
    app.add_handler(CommandHandler("calc", calc_command))
    app.add_handler(CommandHandler("size", calc_command))
    app.add_handler(CommandHandler("levels", levels_command))
    app.add_handler(CommandHandler("ob", levels_command))
    app.add_handler(CommandHandler("trend", trend_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("signal", analyze_command))
    app.add_handler(CommandHandler("itb", itb_command))
    app.add_handler(CommandHandler("intelligent", itb_command))
    app.add_handler(CommandHandler("strategy", strategy_command))
    app.add_handler(CommandHandler("strat", strategy_command))
    app.add_handler(CommandHandler("discussion", discussion_command))
    app.add_handler(CommandHandler("consensus", discussion_command))
    app.add_handler(CommandHandler("deliberate", discussion_command))
    app.add_handler(CommandHandler("forum", discussion_command))
    app.add_handler(CommandHandler("symbol", symbol_command))
    app.add_handler(CommandHandler("symbols", symbols_command))
    app.add_handler(CommandHandler("pairs", symbols_command))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("autotrade", autotrade_command))
    app.add_handler(CommandHandler("close", close_command))
    app.add_handler(CommandHandler("position", position_command))
    app.add_handler(CommandHandler("positions", position_command))
    app.add_handler(CommandHandler("pos", position_command))
    app.add_handler(CommandHandler("pnl", pnl_command))
    app.add_handler(CommandHandler("performance", pnl_command))
    app.add_handler(CommandHandler("trades", pnl_command))
    app.add_handler(CommandHandler("trailing", trailing_command))
    app.add_handler(CommandHandler("risk", risk_command))
    app.add_handler(CommandHandler("lotsize", lotsize_command))
    app.add_handler(CommandHandler("lot", lotsize_command))
    app.add_handler(CommandHandler("tpsl", tpsl_command))
    app.add_handler(CommandHandler("mode", mode_command))
    app.add_handler(CommandHandler("capital", capital_command))
    app.add_handler(CommandHandler("funds", capital_command))
    app.add_handler(CommandHandler("deposit", deposit_command))
    app.add_handler(CommandHandler("withdraw", withdraw_command))
    app.add_handler(CommandHandler("api", api_command))
    app.add_handler(CommandHandler("alert", alert_command))
    app.add_handler(CommandHandler("alerts", alert_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Exception while handling update: %s", context.error, exc_info=context.error)
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "⚠️ An unexpected error occurred while processing your request. The event has been logged."
                )
            except Exception:
                pass

    app.add_error_handler(global_error_handler)

    logger.info("Bot starting polling. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
