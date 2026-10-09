import re

with open('auto_trade.py', 'r') as f:
    content = f.read()

scan_method = """
    def generate_master_scan(self, symbol: str) -> str:
        \"\"\"Generates a combined master scan (Trend, Confluence, Levels, Entry).\"\"\"
        lines = [f"🌐 <b>ULTIMATE MASTER SCAN: {symbol.upper()}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
        
        # 1. MTF Trend (condensed)
        timeframes = [("15m", "15"), ("1h", "60"), ("4h", "240"), ("1D", "1D")]
        bullish = 0
        total = len(timeframes)
        trend_lines = []
        for label, res in timeframes:
            try:
                df = self.fetch_candles(symbol, count=60, resolution=res)
                if df is not None and len(df) > 20:
                    closes = df['close']
                    ema20 = closes.ewm(span=20).mean().iloc[-1]
                    ema50 = closes.ewm(span=50).mean().iloc[-1]
                    rsi = IndicatorEngine.rsi(closes).iloc[-1]
                    cp = closes.iloc[-1]
                    if ema20 > ema50 and cp > ema20 and rsi > 50:
                        trend_lines.append(f"  • {label}: 🟢 Bull")
                        bullish += 1
                    elif ema20 < ema50 and cp < ema20 and rsi < 50:
                        trend_lines.append(f"  • {label}: 🔴 Bear")
                    else:
                        trend_lines.append(f"  • {label}: ⚪ Neut")
            except:
                pass
                
        score = (bullish / total) * 100 if total else 50
        meter = make_modern_meter(score, width=10)
        lines.append(f"<b>1. MTF Trend</b> <code>[{meter}]</code>")
        lines.extend(trend_lines)
        lines.append("")
        
        # 2. Confluence (Analyze)
        try:
            df_1m = self.fetch_candles(symbol, count=120, resolution="1")
            snap = IndicatorsProStrategy.analyze_confluence(df_1m)
            lines.append(f"<b>2. Tech Confluence</b> (Score: {snap.score:+.1f})")
            lines.append(f"  • Trend: {snap.trend_direction.name}")
            lines.append(f"  • Volatility: {snap.volatility_state.name}")
            lines.append(f"  • Signal: {snap.overall_bias.name}")
            lines.append("")
        except:
            lines.append("<b>2. Tech Confluence</b>: ⚠️ Data Error\\n")
            
        # 3. Order Blocks (Levels)
        try:
            obs = [] # we can just call self.generate_order_blocks but it returns a formatted string. 
            # let's just use the string.
        except:
            pass
            
        # Instead of redefining, let's just call the existing methods and combine them, but split by lines to truncate.
        return "\\n".join(lines)
"""
# I'll just combine the output of the 4 functions and truncate/format.
