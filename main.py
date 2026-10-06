"""
Telegram trading assistant powered by Gemini.

Setup:
  pip install google-genai python-telegram-bot requests
  export GEMINI_API_KEY=AQ.Ab8RN6Ixz0Zy3PRY52JIaYnhixast7Qd0kCYsFB3dKjdxG7Utw
  export TELEGRAM_TOKEN=8993862862:AAGaQeh2yx0ICO4y040FySjK2_ASJ8x4vx8
Run:
  python trading_bot_gemini.py

Commands:
  /start            intro
  /price [SYMBOL]   live price from Delta Exchange (default XAUTUSD)
  /reset            clear conversation
  any text          chat with the assistant (it sees the latest price)
"""
import os
import requests
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters,
)

MODEL = "gemini-2.5-flash"  # check AI Studio for the current model name
DEFAULT_SYMBOL = "XAUTUSD"
DELTA_API = "https://api.india.delta.exchange/v2/tickers"  # use api.delta.exchange for global

SYSTEM = """You are a trading assistant on Telegram. Focus on technical analysis,
risk management, position sizing, and trade planning. Keep replies short and
mobile-friendly. When asked for entries, give entry, stop loss, take profit and
risk:reward, and state your assumptions. You are not a financial advisor; markets
are uncertain and the user makes the final decision. Never promise profits."""

gemini = genai.Client()  # reads GEMINI_API_KEY
histories: dict[int, list] = {}


def get_price(symbol: str) -> str:
    try:
        r = requests.get(f"{DELTA_API}/{symbol.upper()}", timeout=10)
        r.raise_for_status()
        t = r.json()["result"]
        return (f"{symbol.upper()}: mark {t.get('mark_price')} | last {t.get('close')} | "
                f"high {t.get('high')} | low {t.get('low')} | open {t.get('open')}")
    except Exception as e:
        return f"Price unavailable ({e})"


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Trading assistant ready. Ask me about setups, risk, or use /price XAUTUSD."
    )


async def price(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    symbol = ctx.args[0] if ctx.args else DEFAULT_SYMBOL
    await update.message.reply_text(get_price(symbol))


async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text("Conversation cleared.")


async def chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    hist = histories.setdefault(chat_id, [])
    text = update.message.text

    context = f"[Market data] {get_price(DEFAULT_SYMBOL)}\n\n"
    hist.append(types.Content(role="user", parts=[types.Part(text=context + text)]))
    hist[:] = hist[-20:]  # keep last 20 messages

    try:
        reply = gemini.models.generate_content(
            model=MODEL,
            contents=hist,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, max_output_tokens=800
            ),
        )
        answer = reply.text or "No response, try again."
    except Exception as e:
        answer = f"Error: {e}"

    hist.append(types.Content(role="model", parts=[types.Part(text=answer)]))
    await update.message.reply_text(answer)


def main():
    app = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.run_polling()


if __name__ == "__main__":
    main()
