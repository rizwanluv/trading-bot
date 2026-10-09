import re

with open('main.py', 'r') as f:
    content = f.read()

# Add imports
import_health = "import asyncio\nfrom health_server import start_health_server\n"
if "from health_server" not in content:
    content = content.replace("import logging", import_health + "import logging")

# Add scan command
scan_cmd = """
async def scan_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    \"\"\"Ultimate Master Scan combining Trend, Analyze, Levels, and Entry.\"\"\"
    args = ctx.args or []
    symbol = args[0] if args else "BTCUSD"
    trader = get_auto_trader()
    
    msg = await reply_safely(update, f"🔄 <b>Scanning {symbol.upper()}...</b>\\nGathering Multi-Timeframe data, Confluence, and Smart Money levels...", parse_mode="HTML")
    
    try:
        trend = trader.generate_mtf_trend_report(symbol)
        analyze = trader.generate_trade_analysis(symbol)
        levels = trader.generate_order_blocks(symbol)
        entry = trader.generate_pinpoint_entry(symbol)
        
        # Combine them cleanly
        combined = f"🌐 <b>ULTIMATE MASTER SCAN: {symbol.upper()}</b>\\n\\n"
        combined += trend + "\\n\\n"
        combined += analyze + "\\n\\n"
        combined += levels + "\\n\\n"
        combined += entry
        
        # Telegram limit is 4096. If it's too long, split it.
        if len(combined) > 4000:
            await msg.edit_text(trend + "\\n\\n" + analyze, parse_mode="HTML")
            await reply_safely(update, levels + "\\n\\n" + entry, parse_mode="HTML")
        else:
            await msg.edit_text(combined, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"⚠️ Scan failed: {e}")

async def market_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    \"\"\"Combined Market Overview.\"\"\"
    trader = get_auto_trader()
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "XAUTUSD"]
    msg = await reply_safely(update, "🔄 <b>Fetching Global Market Overview...</b>", parse_mode="HTML")
    
    lines = ["🌍 <b>GLOBAL MARKET OVERVIEW</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
    for sym in symbols:
        try:
            df = trader.fetch_candles(sym, count=120)
            if df is not None and not df.empty:
                current = df['close'].iloc[-1]
                open_p = df['open'].iloc[0]
                pct = ((current - open_p) / open_p) * 100
                icon = "🟢" if pct >= 0 else "🔴"
                lines.append(f"• <b>{sym.replace('USD','')}</b>: <code>${current:,.2f}</code> {icon} {pct:+.2f}%")
        except:
            pass
            
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("<i>Use /scan [SYM] for deep analysis.</i>")
    await msg.edit_text("\\n".join(lines), parse_mode="HTML")

"""

content = content.replace("async def btc_command(", scan_cmd + "async def btc_command(")

# Add to handlers
handlers = """
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("pro", scan_command))
    app.add_handler(CommandHandler("market", market_command))
    app.add_handler(CommandHandler("markets", market_command))
"""
content = content.replace("app.add_handler(CommandHandler(\"start\", start))", "app.add_handler(CommandHandler(\"start\", start))" + handlers)

# Add health server to async startup
# We can do this in the `if __name__ == "__main__":` block, but python-telegram-bot v20 Application runs its own event loop via `run_polling()`.
# To run a background task in PTB v20, we can use `PostInit` callback.
