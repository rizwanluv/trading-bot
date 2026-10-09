import re
with open('auto_trade.py', 'r') as f:
    content = f.read()

mtf_method = """
    def generate_mtf_trend_report(self, symbol: str) -> str:
        \"\"\"Generate a Multi-Timeframe Trend Confluence report.\"\"\"
        timeframes = [("1m", "1"), ("5m", "5"), ("15m", "15"), ("1h", "60"), ("4h", "240"), ("1D", "1D")]
        
        lines = [f"📊 <b>MTF Trend Scanner: {symbol.upper()}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
        bullish_count = 0
        bearish_count = 0
        total_tf = len(timeframes)
        
        for label, res in timeframes:
            try:
                df = self.fetch_candles(symbol, count=100, resolution=res)
                if df is None or df.empty or len(df) < 50:
                    lines.append(f"• <b>{label}</b>: ⚠️ Insufficient Data")
                    continue
                    
                closes = df['close']
                ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
                rsi = IndicatorEngine.rsi(closes, period=14).iloc[-1]
                current_price = closes.iloc[-1]
                
                if ema20 > ema50 and current_price > ema20 and rsi > 50:
                    trend = "🟢 BULLISH"
                    bullish_count += 1
                elif ema20 < ema50 and current_price < ema20 and rsi < 50:
                    trend = "🔴 BEARISH"
                    bearish_count += 1
                else:
                    trend = "⚪ NEUTRAL"
                    
                lines.append(f"• <b>{label}</b>: {trend} (RSI: {rsi:.1f})")
            except Exception as e:
                logger.error("MTF scan error for %s on %s: %s", symbol, res, e)
                lines.append(f"• <b>{label}</b>: ⚠️ Error")
                
        score = (bullish_count / total_tf) * 100
        b_score = (bearish_count / total_tf) * 100
        
        # Determine overall trend
        if score >= 80:
            overall = "🚀 STRONG UPTREND"
            meter_val = score
        elif score >= 60:
            overall = "📈 UPTREND"
            meter_val = score
        elif b_score >= 80:
            overall = "📉 STRONG DOWNTREND"
            meter_val = 100 - b_score
        elif b_score >= 60:
            overall = "🩸 DOWNTREND"
            meter_val = 100 - b_score
        else:
            overall = "⚖️ CHOPPY / CONSOLIDATING"
            meter_val = 50
            
        meter = make_modern_meter(meter_val, width=12)
        
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"<b>Overall Alignment</b>: {overall}")
        lines.append(f"<b>Bullish vs Bearish</b>: {bullish_count} vs {bearish_count}")
        lines.append(f"<code>[{meter}]</code>")
        return "\\n".join(lines)
"""

content = content.replace("def check_trade_level_alerts(", mtf_method + "\n    def check_trade_level_alerts(")

with open('auto_trade.py', 'w') as f:
    f.write(content)
