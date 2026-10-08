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

from auto_trade import get_auto_trader, AutoTrader, AutoTradeConfig

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
        "Trading assistant ready. Ask me about setups, risk, or use live market & auto-trade commands.\n\n"
        "Commands:\n"
        "• /price [SYMBOL] - Live price from Delta Exchange (default: XAUTUSD)\n"
        "• /autotrade [on|off|status|close] - Control automated trading\n"
        "• /lotsize [SIZE] - View or update order lot size\n"
        "• /tpsl [TP] [SL] [MODE] - Set strategy Take Profit and Stop Loss\n"
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
        reply = (
            f"🟢 <b>{msg}</b>\n\n"
            f"• <b>Symbol</b>: {trader.config.symbol}\n"
            f"• <b>Lot Size</b>: {trader.config.lot_size}\n"
            f"• <b>TP</b>: {trader.config.tp_value} ({trader.config.tp_mode.upper()})\n"
            f"• <b>SL</b>: {trader.config.sl_value} ({trader.config.sl_mode.upper()})\n\n"
            f"Trade signals and executions will be sent directly to this chat."
        )
        await reply_safely(update, reply, parse_mode="HTML")
    elif sub in ("off", "stop", "disable"):
        msg = trader.disable()
        await reply_safely(update, f"🔴 <b>{msg}</b>", parse_mode="HTML")
    elif sub == "close":
        res = trader.close_current_position(reason="MANUAL")
        if res:
            await reply_safely(update, res, parse_mode="HTML")
        else:
            await reply_safely(update, "No active position to close.")
    elif sub in ("symbol", "pair") and len(args) > 1:
        msg = trader.set_symbol(args[1])
        await reply_safely(update, msg)
    else:
        await reply_safely(
            update,
            "Usage:\n"
            "• /autotrade on - Turn auto trade ON\n"
            "• /autotrade off - Turn auto trade OFF\n"
            "• /autotrade status - Show trading dashboard\n"
            "• /autotrade close - Close open position manually\n"
            "• /autotrade symbol [SYMBOL] - Change traded symbol",
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
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("autotrade", autotrade_command))
    app.add_handler(CommandHandler("lotsize", lotsize_command))
    app.add_handler(CommandHandler("tpsl", tpsl_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("Bot starting polling. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
