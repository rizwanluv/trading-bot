import re
with open('main.py', 'r') as f:
    content = f.read()

trend_cmd = """
async def trend_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    \"\"\"Run a multi-timeframe trend scanner.\"\"\"
    args = ctx.args or []
    symbol = args[0] if args else "BTCUSD"
    trader = get_auto_trader()
    report = trader.generate_mtf_trend_report(symbol)
    await reply_safely(update, report, parse_mode="HTML")

"""

content = content.replace("async def analyze_command(", trend_cmd + "async def analyze_command(")
content = content.replace("app.add_handler(CommandHandler(\"analyze\", analyze_command))", "app.add_handler(CommandHandler(\"trend\", trend_command))\n    app.add_handler(CommandHandler(\"analyze\", analyze_command))")

with open('main.py', 'w') as f:
    f.write(content)
