def generate_mtf_trend_report(self, symbol: str) -> str:
    """Generate a Multi-Timeframe Trend Confluence report."""
    timeframes = [("1m", "1"), ("5m", "5"), ("15m", "15"), ("1h", "60"), ("4h", "240"), ("1D", "1D")]
    
    lines = [f"📊 <b>MTF Trend Scanner: {symbol.upper()}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
    bullish_count = 0
    total_tf = len(timeframes)
    
    for label, res in timeframes:
        try:
            df = self.fetch_candles(symbol, count=60, resolution=res)
            if df.empty or len(df) < 50:
                lines.append(f"• <b>{label}</b>: ⚠️ Insufficient Data")
                continue
                
            closes = df['close']
            ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
            rsi = self._calculate_rsi(closes, period=14).iloc[-1]
            current_price = closes.iloc[-1]
            
            if ema20 > ema50 and current_price > ema20 and rsi > 50:
                trend = "🟢 BULLISH"
                bullish_count += 1
            elif ema20 < ema50 and current_price < ema20 and rsi < 50:
                trend = "🔴 BEARISH"
            else:
                trend = "⚪ NEUTRAL"
                
            lines.append(f"• <b>{label}</b>: {trend} (RSI: {rsi:.1f})")
        except Exception as e:
            lines.append(f"• <b>{label}</b>: ⚠️ Error")
            
    score = (bullish_count / total_tf) * 100
    meter = make_modern_meter(score, width=12)
    
    if score >= 80:
        overall = "🚀 STRONG UPTREND"
    elif score >= 60:
        overall = "📈 UPTREND"
    elif score <= 20:
        overall = "📉 STRONG DOWNTREND"
    elif score <= 40:
        overall = "🩸 DOWNTREND"
    else:
        overall = "⚖️ CHOPPY / CONSOLIDATING"
        
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"<b>Overall Alignment</b>: {overall}")
    lines.append(f"<b>Bullish Confluence</b>: {score:.0f}%")
    lines.append(f"<code>[{meter}]</code>")
    return "\n".join(lines)
