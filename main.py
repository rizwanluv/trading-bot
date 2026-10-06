"""
Telegram trading assistant: Claude + Gemini + OpenAI answer at the same time.
A model is enabled only if its API key is set, so you can start with one or two.

Railway Variables:
  TELEGRAM_TOKEN      (required)
  ANTHROPIC_API_KEY   (Claude)
  GEMINI_API_KEY      (Gemini)
  OPENAI_API_KEY      (OpenAI)
  GROQ_API_KEY        (Groq, free tier)
Optional model overrides: CLAUDE_MODEL, GEMINI_MODEL, OPENAI_MODEL, GROQ_MODEL

Commands: /start  /price [SYMBOL]  /models  /reset
"""
import os
import asyncio
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters,
)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # check OpenAI docs for current name
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # or llama-3.3-70b-versatile

DEFAULT_SYMBOL = "XAUTUSD"
DELTA_API = "https://api.india.delta.exchange/v2/tickers"  # use api.delta.exchange for global

SYSTEM = """You are a trading assistant on Telegram. Focus on technical analysis,
risk management, position sizing, and trade planning. Keep replies short and
mobile-friendly (under 150 words). When asked for entries, give entry, stop loss,
take profit and risk:reward, and state your assumptions. You are not a financial
advisor; markets are uncertain and the user makes the final decision. Never
promise profits."""

# ---------- providers (each takes neutral history: [{"role","text"}]) ----------
providers = {}  # name -> function(hist) -> str


def setup_providers():
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()

        def ask_claude(hist):
            msgs = [{"role": h["role"], "content": h["text"]} for h in hist]
            r = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=700, system=SYSTEM, messages=msgs
            )
            return r.content[0].text

        providers["Claude"] = ask_claude

    if os.getenv("AQ.Ab8RN6Ixz0Zy3PRY52JIaYnhixast7Qd0kCYsFB3dKjdxG7Utw"):
        from google import genai
        from google.genai import types
        gem = genai.Client()

        def ask_gemini(hist):
            contents = [
                types.Content(
                    role="model" if h["role"] == "assistant" else "user",
                    parts=[types.Part(text=h["text"])],
                )
                for h in hist
            ]
            r = gem.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM, max_output_tokens=700
                ),
            )
            return r.text or "No response."

        providers["Gemini"] = ask_gemini

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        oa = OpenAI()

        def ask_openai(hist):
            msgs = [{"role": "system", "content": SYSTEM}] + [
                {"role": h["role"], "content": h["text"]} for h in hist
            ]
            r = oa.chat.completions.create(
                model=OPENAI_MODEL, messages=msgs, max_tokens=700
            )
            return r.choices[0].message.content

        providers["OpenAI"] = ask_openai

    if os.getenv("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"):
        from openai import OpenAI
        groq = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

        def ask_groq(hist):
            msgs = [{"role": "system", "content": SYSTEM}] + [
                {"role": h["role"], "content": h["text"]} for h in hist
            ]
            r = groq.chat.completions.create(
                model=GROQ_MODEL, messages=msgs, max_tokens=1000
            )
            return r.choices[0].message.content or "No response."

        providers["Groq"] = ask_groq


setup_providers()

# histories[chat_id][model_name] = neutral history list
histories: dict[int, dict[str, list]] = {}


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
    names = ", ".join(providers) or "none (add API keys)"
    await update.message.reply_text(
        f"Trading assistant ready.\nActive models: {names}\n"
        "Ask anything, or use /price XAUTUSD."
    )


async def models(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Active: " + (", ".join(providers) or "none")
    )


async def price(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    symbol = ctx.args[0] if ctx.args else DEFAULT_SYMBOL
    await update.message.reply_text(get_price(symbol))


async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text("Conversation cleared.")


async def ask_one(chat_id: int, name: str, fn, text: str):
    hist = histories.setdefault(chat_id, {}).setdefault(name, [])
    hist.append({"role": "user", "text": text})
    hist[:] = hist[-20:]
    try:
        answer = await asyncio.to_thread(fn, hist)
    except Exception as e:
        hist.pop()  # keep user/assistant alternation valid
        return name, f"Error: {str(e)[:300]}"
    hist.append({"role": "assistant", "text": answer})
    return name, answer


async def chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not providers:
        await update.message.reply_text("No AI keys set. Add API keys in Railway Variables.")
        return

    chat_id = update.effective_chat.id
    text = f"[Market data] {get_price(DEFAULT_SYMBOL)}\n\n{update.message.text}"

    await ctx.bot.send_chat_action(chat_id, "typing")
    tasks = [asyncio.create_task(ask_one(chat_id, n, f, text)) for n, f in providers.items()]

    # Send each answer as soon as that model finishes
    for done in asyncio.as_completed(tasks):
        name, answer = await done
        await update.message.reply_text(f"[{name}]\n{answer}"[:4000])


def main():
    app = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.run_polling()


if __name__ == "__main__":
    main()
