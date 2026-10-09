with open('main.py', 'r') as f:
    content = f.read()

find_strat = """async def strategy_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    \"\"\"View or switch active automated trading strategy.\"\"\"
    trader = get_auto_trader()
    args = ctx.args or []

    if not args:
        current = trader.config.strategy_type
        names = {
            "itb_ml": "🤖 <b>Intelligent Trading Bot (ITB Machine Learning)</b>",
            "indicators_pro": "📊 <b>Indicators Pro (Multi-Indicator Confluence)</b>",
            "ai_learning": "🧠 <b>AI Bot Learning (Adaptive Learning)</b>",
        }
        active_name = names.get(current, current)
        await reply_safely(
            update,
            f"🎯 <b>TRADING STRATEGY CONFIGURATION</b>\\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\\n"
            f"• <b>Active Strategy</b>: {active_name}\\n\\n"
            f"<b>Available Strategies:</b>\\n"
            f"1. <b>ITB Machine Learning</b> (<code>/strategy itb</code>)\\n"
            f"   • Rolling trend slope, skewness & kurtosis moments\\n"
            f"   • Ridge regression combined Intelligent Indicator\\n"
            f"   • Dynamic ATR stops & automated execution\\n\\n"
            f"2. <b>Indicators Pro</b> (<code>/strategy indicators</code>)\\n"
            f"   • EMA trend filter, RSI momentum & Bollinger channels\\n"
            f"   • Pinpoint pullback and breakout detection\\n\\n"
            f"3. <b>AI Learning Bot</b> (<code>/strategy ai</code>)\\n"
            f"   • Adaptive reward-weighted reinforcement\\n\\n"
            f"<i>Switch strategy: <code>/strategy itb</code> | <code>/strategy indicators</code></i>",
            parse_mode="HTML",
        )
    else:
        new_strat = args[0].lower()
        ok, msg = trader.set_strategy_type(new_strat)
        await reply_safely(update, msg, parse_mode="HTML")"""
        
replace_strat = """async def strategy_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    \"\"\"View active automated trading strategy.\"\"\"
    await reply_safely(
        update,
        f"🎯 <b>TRADING STRATEGY CONFIGURATION</b>\\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\\n"
        f"• <b>Active Strategy</b>: 🌟 <b>COMBINED ENSEMBLE SYSTEM</b>\\n\\n"
        f"<b>Active Engines (Working Together):</b>\\n"
        f"1. <b>ITB Machine Learning</b> 🤖\\n"
        f"   • Rolling trend slope, skewness & kurtosis moments\\n"
        f"   • Ridge regression prediction logic\\n\\n"
        f"2. <b>Indicators Pro</b> 📊\\n"
        f"   • EMA trend filter, RSI momentum & Bollinger channels\\n"
        f"   • Pinpoint pullback and breakout detection\\n\\n"
        f"3. <b>AI Learning Bot</b> 🧠\\n"
        f"   • Adaptive reward-weighted reinforcement\\n\\n"
        f"<i>The bot now runs all three engines simultaneously. It will only execute trades when the algorithms achieve confluence (agree on direction).</i>",
        parse_mode="HTML",
    )"""

if find_strat in content:
    content = content.replace(find_strat, replace_strat)
else:
    print("Warning: find_strat not found")

with open('main.py', 'w') as f:
    f.write(content)
