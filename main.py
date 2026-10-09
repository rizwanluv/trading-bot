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
)

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
    "ETHUSDT": "ETHUSD",
    "SOL": "SOLUSD",
    "SOLUSDT": "SOLUSD",
    "XRP": "XRPUSD",
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


def get_price(symbol: str) -> str:
    """Fetch live ticker data from Delta Exchange API."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
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
    """Fetch live ticker data from Delta Exchange API and format as a rich HTML card."""
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
    target = SYMBOL_ALIASES.get(cleaned, cleaned)
    try:
        r = requests.get(f"{DELTA_API}/{target}", timeout=10)
        r.raise_for_status()
        data = r.json()
        t = data.get("result")
        if not t:
            return f"❌ Symbol <b>{symbol.upper()}</b> not found on Delta Exchange."
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

        arrow = "🟢 +" if change_pct >= 0 else "🔴 "
        display_name = "Bitcoin (BTC)" if "BTC" in target else ("Gold (XAU)" if "XAU" in target else target)
        icon = "⚡" if "BTC" in target else ("🥇" if "XAU" in target else "📊")

        return (
            f"{icon} <b>{display_name} Market Ticker</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Symbol</b>: <code>{target}</code>\n"
            f"• <b>Mark Price</b>: <code>${mark_price:,.2f}</code>\n"
            f"• <b>Last Price</b>: <code>${close_price:,.2f}</code>\n"
            f"• <b>24h Change</b>: {arrow}{change_pct:.2f}%\n"
            f"• <b>24h High</b>: <code>${high_price:,.2f}</code>\n"
            f"• <b>24h Low</b>: <code>${low_price:,.2f}</code>\n"
            f"• <b>24h Volume</b>: <code>{volume:,.2f}</code>\n"
            f"• <b>Exchange</b>: Delta Exchange\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 Use /autotrade symbol {target} to trade this pair</i>"
        )
    except Exception as e:
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
    """Send reply respecting Telegram's 4096 character limit."""
    if not update.message:
        return
    max_chunk = 4000
    kwargs = {"parse_mode": parse_mode} if parse_mode else {}
    if len(text) <= max_chunk:
        try:
            await update.message.reply_text(text, **kwargs)
        except Exception:
            await update.message.reply_text(text)
    else:
        for i in range(0, len(text), max_chunk):
            try:
                await update.message.reply_text(text[i : i + max_chunk], **kwargs)
            except Exception:
                await update.message.reply_text(text[i : i + max_chunk])


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await reply_safely(
        update,
        "Trading assistant ready. Real-time analysis and automated trading for Bitcoin (BTC) & Gold (XAU).\n\n"
        "Market & Analysis Commands:\n"
        "• /btc - Live Bitcoin (BTC) ticker, 24h stats & range\n"
        "• /gold - Live Gold (XAU) ticker, 24h stats & range\n"
        "• /eth - Live Ethereum (ETH) ticker & stats\n"
        "• /sol - Live Solana (SOL) ticker & stats\n"
        "• /entry [SYM] - Pinpoint Entry, Precision SL & Multi-tier TP Targets\n"
        "• /levels [SYM] - Smart Money Order Blocks (OB) & Fair Value Gaps (FVG)\n"
        "• /calc [ENTRY] [SL] [TP] - Position Sizing & Risk:Reward Calculator\n"
        "• /analyze [SYMBOL] - Full technical analysis & signal report\n"
        "• /price [SYMBOL] - Price check on any Delta Exchange pair\n"
        "• /symbols [both|btc|gold|add|rm] - Configure active auto-trade pairs\n"
        "• /symbol [SYMBOL] - View or switch primary trading symbol\n\n"
        "Auto-Trade & Execution Controls:\n"
        "• /execute [LOTS|limit] - One-Tap Execution of Pinpoint Trade Plan\n"
        "• /buy - Quick Instant Long Market Order\n"
        "• /sell - Quick Instant Short Market Order\n"
        "• /autotrade [on|off|status|close|symbols|max] - Control automated trading\n"
        "• /position (/positions) - Live active positions dashboard & unrealized PnL\n"
        "• /close [ID|SYM|all] - Close specific position, symbol, or all open trades\n"
        "• /pnl - Performance report & closed trades history\n"
        "• /lotsize [SIZE|risk %] - View or update order lot sizing\n"
        "• /tpsl [TP] [SL] [MODE] - Set strategy Take Profit and Stop Loss\n"
        "• /trailing [on|off] - Dynamic trailing stop loss protection\n"
        "• /risk [PCT] - Max daily loss risk limit protection\n\n"
        "Assistant Commands:\n"
        "• /menu - Quick command directory\n"
        "• /reset - Clear conversation history\n"
        "• /help - Show this guide\n\n"
        "Send any message to chat with market context.",
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
                msg = trader.set_symbols(["BTCUSD", "XAUTUSD"])
            else:
                msg = trader.set_symbols(args[1:])
            await reply_safely(update, msg)
        else:
            syms_str = ", ".join(trader.config.symbols)
            await reply_safely(update, f"Monitored symbols: {syms_str}")
    elif sub == "max" and len(args) > 1:
        try:
            total_m = int(args[1])
            per_m = int(args[2]) if len(args) > 2 else 1
            msg = trader.set_max_positions(total_m, per_m)
            await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
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
        msg = trader.set_symbols(["BTCUSD", "XAUTUSD"])
        await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
    elif sub == "add" and len(args) > 1:
        msg = trader.add_symbol(args[1])
        await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
    elif sub in ("remove", "rm", "del") and len(args) > 1:
        msg = trader.remove_symbol(args[1])
        await reply_safely(update, f"ℹ️ <b>{msg}</b>", parse_mode="HTML")
    elif sub in ("btc", "bitcoin") and len(args) == 1:
        msg = trader.set_symbols(["BTCUSD"])
        await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
    elif sub in ("gold", "xau", "xautusd") and len(args) == 1:
        msg = trader.set_symbols(["XAUTUSD"])
        await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")
    else:
        msg = trader.set_symbols(args)
        await reply_safely(update, f"✅ <b>{msg}</b>", parse_mode="HTML")


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
        lot_mode_desc = "Fixed lot size" if trader.config.lot_mode == "fixed" else f"Risk {trader.config.risk_pct}% sizing"
        await reply_safely(
            update,
            f"📦 <b>Current Lot Size Configuration</b>\n"
            f"• <b>Lot Size</b>: <code>{trader.config.lot_size}</code>\n"
            f"• <b>Mode</b>: {lot_mode_desc}\n\n"
            f"<b>Change lot size:</b>\n"
            f"• <code>/lotsize 0.05</code> (Set fixed lot size to 0.05)\n"
            f"• <code>/lotsize 1.0</code> (Set fixed lot size to 1.0)\n"
            f"• <code>/lotsize risk 2%</code> (Set risk-based sizing to 2% equity)",
            parse_mode="HTML",
        )
        return

    # Check for risk % format
    if args[0].lower() == "risk" and len(args) > 1:
        try:
            val = float(args[1].replace("%", "").strip())
            trader.config.risk_pct = val
            ok, msg = trader.set_lot_size(val, mode="risk_pct")
            await reply_safely(update, f"✅ {msg}")
        except ValueError:
            await reply_safely(update, "Error: Invalid risk percentage. Example: /lotsize risk 1.5")
        return

    # Numeric lot size
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
            "Error: Invalid lot size format.\nUsage: /lotsize 0.05 or /lotsize risk 2%",
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


async def btc_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Bitcoin market ticker and stats card."""
    card = get_ticker_card("BTCUSD")
    await reply_safely(update, card, parse_mode="HTML")


async def gold_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show live Gold market ticker and stats card."""
    card = get_ticker_card("XAUTUSD")
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
            cmd_name = cmd_parts[0].lower().lstrip("/")

    plan = latest_trade_plans.get(chat_id)
    entry_mode = "market"
    lot_override = None

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
                pass

    if cmd_name in ("buy", "long"):
        if not plan or plan.direction != "LONG":
            plan = generate_pinpoint_plan(trader.config.symbol, direction_override="LONG", trader=trader)
            latest_trade_plans[chat_id] = plan
    elif cmd_name in ("sell", "short"):
        if not plan or plan.direction != "SHORT":
            plan = generate_pinpoint_plan(trader.config.symbol, direction_override="SHORT", trader=trader)
            latest_trade_plans[chat_id] = plan
    elif not plan:
        plan = generate_pinpoint_plan(trader.config.symbol, trader=trader)
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
        entry = float(args[0])
        sl = float(args[1])
        tp = float(args[2]) if len(args) > 2 else None
        risk_usd = float(args[3]) if len(args) > 3 else None
        symbol = trader.config.symbol
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
            "❌ Invalid parameters. Numbers required.\nExample: <code>/calc 81700 81200 82700</code>",
            parse_mode="HTML",
        )


async def levels_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Institutional Order Blocks (OB) & Fair Value Gaps (FVG) scanner."""
    trader = get_auto_trader()
    args = ctx.args or []
    symbol = args[0] if args else trader.config.symbol
    report = get_levels_report(symbol, trader=trader)
    await reply_safely(update, report, parse_mode="HTML")


async def menu_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Interactive command directory and quick dashboard."""
    trader = get_auto_trader()
    status_str = "🟢 AUTO-TRADING ON" if trader.config.enabled else "🔴 AUTO-TRADING OFF"
    pos_count = len(trader.positions)
    pos_str = f"{pos_count} active position(s)" if pos_count > 0 else "No open positions"
    symbols_str = ", ".join(trader.config.symbols) if trader.config.symbols else trader.config.symbol
    await reply_safely(
        update,
        f"🎛️ <b>TRADING BOT COMMAND MENU</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Engine Status</b>: <b>{status_str}</b>\n"
        f"• <b>Monitored Pairs</b>: <code>{symbols_str}</code>\n"
        f"• <b>Positions</b>: <i>{pos_str} (Max {trader.config.max_positions})</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Market & Analysis:</b>\n"
        f"• /entry [SYM] - Pinpoint Entry, Precision SL & TP Targets\n"
        f"• /levels [SYM] - Order Blocks (OB) & FVGs Scanner\n"
        f"• /calc [ENTRY] [SL] - Position Sizing & R:R Calculator\n"
        f"• /btc - Bitcoin Ticker Card\n"
        f"• /gold - Gold Ticker Card\n"
        f"• /eth - Ethereum Ticker Card\n"
        f"• /sol - Solana Ticker Card\n"
        f"• /analyze [SYM] - Technical Analysis & Confluence\n"
        f"• /symbols - View/Set Active Auto-Trade Pairs\n"
        f"• /symbol [SYM] - Switch Primary Symbol\n\n"
        f"<b>Auto-Trading & Execution:</b>\n"
        f"• /execute - One-Tap Execute Pinpoint Plan\n"
        f"• /buy | /sell - Instant Market Orders\n"
        f"• /autotrade on|off - Start/Stop Automated Trades\n"
        f"• /position (/positions) - Live Active Positions Dashboard\n"
        f"• /close [ID|SYM|all] - Close Position(s)\n"
        f"• /pnl - Performance Report & Trades History\n"
        f"• /lotsize [SIZE] - Set Order Sizing\n"
        f"• /tpsl [TP] [SL] - Set Strategy Targets\n"
        f"• /trailing on|off - Dynamic Trailing Stop Loss\n"
        f"• /risk [PCT] - Max Daily Drawdown Protection\n\n"
        f"<b>General:</b>\n"
        f"• /price [SYM] - Custom Pair Price Check\n"
        f"• /reset - Clear Dialogue History\n"
        f"• /help - Full Guide",
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

    # 2. Telegram Token Check
    print("\n[2] Checking Telegram Token...")
    token = get_telegram_token()
    if token:
        masked = token[:6] + "..." + token[-4:] if len(token) > 10 else "***"
        print(f"    Token         : {masked}")
        print("    Status        : CONFIGURED")
    else:
        print("    Status        : MISSING (set TELEGRAM_TOKEN or TELEGRAM_BOT_TOKEN)")

    # 3. Gemini API Key Check
    print("\n[3] Checking Gemini API Key...")
    api_key = get_gemini_api_key()
    if api_key:
        masked_k = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"    API Key       : {masked_k}")
        print("    Status        : CONFIGURED")
    else:
        print("    Status        : MISSING (set GEMINI_API_KEY in environment or .env)")

    print("=" * 60)
    all_ready = bool(token and api_key and delta_ok)
    print(f" OVERALL STATUS   : {'READY TO RUN' if all_ready else 'CONFIGURATION PENDING'}")
    print("=" * 60)
    return all_ready


async def auto_trade_worker(app: Any) -> None:
    """Periodic background worker checking market and executing strategy."""
    trader = get_auto_trader()
    trader.is_running = True
    logger.info("Auto-trade background worker started.")
    while trader.is_running:
        try:
            notifications = trader.step()
            if notifications and trader.config.notify_chat_id:
                for note in notifications:
                    try:
                        await app.bot.send_message(
                            chat_id=trader.config.notify_chat_id,
                            text=note,
                            parse_mode="HTML",
                        )
                    except Exception as note_err:
                        logger.warning("Failed to send auto-trade notification: %s", note_err)
        except Exception as step_err:
            logger.warning("Error in auto-trade step: %s", step_err)

        await asyncio.sleep(trader.config.poll_seconds)


async def on_post_init(application: Any) -> None:
    asyncio.create_task(auto_trade_worker(application))


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
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("btc", btc_command))
    app.add_handler(CommandHandler("gold", gold_command))
    app.add_handler(CommandHandler("xau", gold_command))
    app.add_handler(CommandHandler("eth", eth_command))
    app.add_handler(CommandHandler("sol", sol_command))
    app.add_handler(CommandHandler("entry", entry_command))
    app.add_handler(CommandHandler("pinpoint", entry_command))
    app.add_handler(CommandHandler("execute", execute_command))
    app.add_handler(CommandHandler("buy", execute_command))
    app.add_handler(CommandHandler("sell", execute_command))
    app.add_handler(CommandHandler("calc", calc_command))
    app.add_handler(CommandHandler("size", calc_command))
    app.add_handler(CommandHandler("levels", levels_command))
    app.add_handler(CommandHandler("ob", levels_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("signal", analyze_command))
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
    app.add_handler(CommandHandler("tpsl", tpsl_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("Bot starting polling. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
