"""
Market data fetcher and technical analysis engine using Delta Exchange API.
"""
import time
import requests
from typing import Dict, Any, List, Optional
from indicators import TechnicalAnalysis

DELTA_API = "https://api.india.delta.exchange/v2"
DEFAULT_SYMBOL = "BTCUSD"
SECONDARY_SYMBOL = "XAUTUSD"
POPULAR_SYMBOLS = ["BTCUSD", "XAUTUSD"]

SYMBOL_ALIASES = {
    "BTC": "BTCUSD",
    "BITCOIN": "BTCUSD",
    "BTCUSD": "BTCUSD",
    "BTCUSDT": "BTCUSD",
    "XAU": "XAUTUSD",
    "XAUT": "XAUTUSD",
    "GOLD": "XAUTUSD",
    "XAUTUSD": "XAUTUSD",
    "ETH": "ETHUSD",
    "ETHEREUM": "ETHUSD",
    "ETHUSD": "ETHUSD",
    "SOL": "SOLUSD",
    "SOLANA": "SOLUSD",
    "SOLUSD": "SOLUSD",
}


def resolve_symbol(symbol: Optional[str] = None, default: str = DEFAULT_SYMBOL) -> str:
    """Normalize user input symbol to canonical Delta Exchange symbol (e.g. BTC -> BTCUSD, GOLD -> XAUTUSD)."""
    if not symbol:
        return default
    cleaned = symbol.strip().upper().replace("$", "").replace("#", "")
    if cleaned in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[cleaned]
    if cleaned.endswith("USD"):
        return cleaned
    return f"{cleaned}USD"


def get_market_overview(symbols: Optional[List[str]] = None) -> str:
    """Return a live market overview comparing Bitcoin (BTC) and Gold (XAU)."""
    if not symbols:
        symbols = POPULAR_SYMBOLS

    lines = ["🪙 <b>MARKET OVERVIEW (Live):</b>\n"]
    for sym in symbols:
        try:
            t = get_ticker(sym)
            mark = t["mark_price"] or t["close"]
            chg = mark - t["open"] if t["open"] > 0 else 0.0
            chg_pct = (chg / t["open"] * 100.0) if t["open"] > 0 else 0.0
            sign = "+" if chg >= 0 else ""
            color_emoji = "🟢" if chg >= 0 else "🔴"

            name = "Bitcoin (BTC)" if "BTC" in sym else ("Gold (XAU)" if "XAU" in sym else sym)
            lines.append(
                f"{color_emoji} <b>{name}</b> (<code>#{t['symbol']}</code>)\n"
                f"• <b>Price:</b> <code>${mark:,.2f}</code> ({sign}{chg_pct:.2f}%)\n"
                f"• <b>24h Range:</b> <code>${t['low']:,.2f}</code> — <code>${t['high']:,.2f}</code>\n"
            )
        except Exception as e:
            lines.append(f"• <code>#{sym}</code>: Error fetching price ({e})\n")

    lines.append(
        "💡 <i>Quick shortcuts:</i>\n"
        "• <code>/btc</code> — Live BTC ticker | <code>/btclevels</code> — BTC key levels\n"
        "• <code>/btcgj</code> — BTC liquidity | <code>/btcentry</code> — BTC candle setups\n"
        "• <code>/gold</code> — Live Gold ticker | <code>/goldlevels</code> — Gold levels"
    )
    return "\n".join(lines)


def get_ticker(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Fetch live ticker data from Delta Exchange."""
    sym = resolve_symbol(symbol)
    url = f"{DELTA_API}/tickers/{sym}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("success") and "result" in data:
            res = data["result"]
            return {
                "symbol": sym,
                "mark_price": float(res.get("mark_price", 0.0) or 0.0),
                "close": float(res.get("close", 0.0) or 0.0),
                "high": float(res.get("high", 0.0) or 0.0),
                "low": float(res.get("low", 0.0) or 0.0),
                "open": float(res.get("open", 0.0) or 0.0),
                "volume": float(res.get("volume", 0.0) or 0.0),
                "raw": res,
            }
    except Exception as e:
        # Fallback to global delta exchange if india endpoint fails
        try:
            fallback_url = f"https://api.delta.exchange/v2/tickers/{sym}"
            r = requests.get(fallback_url, timeout=10)
            r.raise_for_status()
            res = r.json()["result"]
            return {
                "symbol": sym,
                "mark_price": float(res.get("mark_price", 0.0) or 0.0),
                "close": float(res.get("close", 0.0) or 0.0),
                "high": float(res.get("high", 0.0) or 0.0),
                "low": float(res.get("low", 0.0) or 0.0),
                "open": float(res.get("open", 0.0) or 0.0),
                "volume": float(res.get("volume", 0.0) or 0.0),
                "raw": res,
            }
        except Exception:
            raise RuntimeError(f"Could not fetch ticker for {sym}: {e}")
    raise RuntimeError(f"Ticker data unavailable for {sym}")


def get_candles(symbol: str = DEFAULT_SYMBOL, resolution: str = "5m", count: int = 60) -> List[Dict[str, Any]]:
    """
    Fetch historical candles for given resolution ('1m', '5m', '15m', '1h', '1d').
    Returns candles in chronological order (oldest to newest).
    """
    sym = resolve_symbol(symbol)
    now = int(time.time())
    res_seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }.get(resolution, 300)

    start = now - (count + 15) * res_seconds
    url = f"{DELTA_API}/history/candles?symbol={sym}&resolution={resolution}&start={start}&end={now}"

    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        if data.get("success") and "result" in data:
            candles = data["result"]
            # Delta returns descending order (latest first). Reverse to chronological.
            return candles[::-1]
    except Exception as e:
        # Fallback to global endpoint
        try:
            fallback_url = f"https://api.delta.exchange/v2/history/candles?symbol={sym}&resolution={resolution}&start={start}&end={now}"
            r = requests.get(fallback_url, timeout=12)
            r.raise_for_status()
            candles = r.json()["result"]
            return candles[::-1]
        except Exception:
            raise RuntimeError(f"Could not fetch {resolution} candles for {sym}: {e}")

    return []


def get_level_analysis(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """
    Perform automatic level analysis:
    - Daily Classic & Fibonacci Pivot Points
    - Swing Highs & Lows
    - Nearest immediate Support and Resistance zones
    - Fibonacci Retracements
    - Trend Bias & 24h Range status
    """
    sym = resolve_symbol(symbol)
    ticker = get_ticker(sym)
    mark = ticker["mark_price"] or ticker["close"]
    high_24h = ticker["high"]
    low_24h = ticker["low"]
    close = ticker["close"]
    open_24h = ticker["open"]

    # Pivot points based on 24h high, low, close
    pivots = TechnicalAnalysis.calc_pivot_points(high_24h, low_24h, close)
    fib_retracements = TechnicalAnalysis.calc_fib_retracements(high_24h, low_24h)

    # Gautam Jha Liquidity Levels from daily candles
    try:
        daily_candles = get_candles(symbol, resolution="1d", count=5)
        gj = TechnicalAnalysis.calc_gautam_jha_levels(daily_candles, mark)
    except Exception:
        gj = {
            "daily_open": open_24h or mark,
            "pdh": high_24h,
            "pdl": low_24h,
            "pdc": close,
            "today_high": high_24h,
            "today_low": low_24h,
            "daily_candle_color": "GREEN" if mark >= (open_24h or mark) else "RED",
            "dist_do": mark - (open_24h or mark),
            "dist_do_pct": ((mark - open_24h) / open_24h * 100.0) if open_24h > 0 else 0.0,
            "pdh_swept": False,
            "pdl_swept": False,
        }

    # Collect all key levels to determine nearest support & resistance
    key_levels = [
        ("R3", pivots["r3"]),
        ("R2", pivots["r2"]),
        ("R1", pivots["r1"]),
        ("Pivot", pivots["pivot"]),
        ("S1", pivots["s1"]),
        ("S2", pivots["s2"]),
        ("S3", pivots["s3"]),
        ("24h High", high_24h),
        ("24h Low", low_24h),
        ("Fib 61.8%", fib_retracements["61.8%"]),
        ("Fib 50.0%", fib_retracements["50.0%"]),
        ("Fib 38.2%", fib_retracements["38.2%"]),
        ("Daily Open", gj["daily_open"]),
        ("PDH", gj["pdh"]),
        ("PDL", gj["pdl"]),
    ]

    resistances = sorted([(name, lvl) for name, lvl in key_levels if lvl > mark], key=lambda x: x[1])
    supports = sorted([(name, lvl) for name, lvl in key_levels if lvl < mark], key=lambda x: x[1], reverse=True)

    nearest_res_name, nearest_res_val = resistances[0] if resistances else ("24h High", high_24h)
    nearest_sup_name, nearest_sup_val = supports[0] if supports else ("24h Low", low_24h)

    res_dist = nearest_res_val - mark
    res_dist_pct = (res_dist / mark * 100.0) if mark > 0 else 0.0

    sup_dist = mark - nearest_sup_val
    sup_dist_pct = (sup_dist / mark * 100.0) if mark > 0 else 0.0

    # Trend Bias
    if mark > pivots["r1"]:
        trend_bias = "🟢 Strong Bullish (Above R1)"
    elif mark > pivots["pivot"]:
        trend_bias = "🟢 Bullish Bias (Above Pivot)"
    elif mark < pivots["s1"]:
        trend_bias = "🔴 Strong Bearish (Below S1)"
    elif mark < pivots["pivot"]:
        trend_bias = "🔴 Bearish Bias (Below Pivot)"
    else:
        trend_bias = "🟡 Neutral / Consolidating at Pivot"

    # 24h Change
    change_24h = mark - open_24h if open_24h > 0 else 0.0
    change_pct = (change_24h / open_24h * 100.0) if open_24h > 0 else 0.0

    # Visual Range Bar
    total_range = high_24h - low_24h
    if total_range > 0:
        position_pct = max(0.0, min(100.0, (mark - low_24h) / total_range * 100.0))
        dots = int(position_pct / 10)
        range_bar = "[" + "=" * dots + "📍" + "-" * (10 - dots) + f"] ({position_pct:.1f}%)"
    else:
        range_bar = "N/A"

    return {
        "symbol": ticker["symbol"],
        "mark_price": mark,
        "high_24h": high_24h,
        "low_24h": low_24h,
        "open_24h": open_24h,
        "change_24h": change_24h,
        "change_pct": change_pct,
        "trend_bias": trend_bias,
        "range_bar": range_bar,
        "pivots": pivots,
        "fib_retracements": fib_retracements,
        "gautam_jha": gj,
        "nearest_resistance": {"name": nearest_res_name, "price": nearest_res_val, "distance": res_dist, "distance_pct": res_dist_pct},
        "nearest_support": {"name": nearest_sup_name, "price": nearest_sup_val, "distance": sup_dist, "distance_pct": sup_dist_pct},
    }


def format_level_analysis_message(analysis: Dict[str, Any]) -> str:
    """Format level analysis dict into a clean Telegram HTML message."""
    sym = analysis["symbol"]
    mark = analysis["mark_price"]
    chg = analysis["change_pct"]
    chg_sign = "+" if chg >= 0 else ""
    pivots = analysis["pivots"]
    fibs = analysis["fib_retracements"]
    gj = analysis.get("gautam_jha", {})
    nr = analysis["nearest_resistance"]
    ns = analysis["nearest_support"]

    pdh_tag = " ⚡<i>(Swept)</i>" if gj.get("pdh_swept") else ""
    pdl_tag = " ⚡<i>(Swept)</i>" if gj.get("pdl_swept") else ""
    color_emoji = "🟢" if gj.get("daily_candle_color") == "GREEN" else "🔴"

    msg = (
        f"📊 <b>AUTOMATIC LEVEL ANALYSIS: #{sym}</b>\n\n"
        f"💵 <b>Current Price:</b> <code>${mark:,.2f}</code> ({chg_sign}{chg:.2f}%)\n"
        f"📈 <b>24h Range:</b> <code>${analysis['low_24h']:,.2f}</code> — <code>${analysis['high_24h']:,.2f}</code>\n"
        f"🧭 <b>Range Position:</b> {analysis['range_bar']}\n"
        f"⚡ <b>Trend Bias:</b> {analysis['trend_bias']}\n\n"
        f"💧 <b>Gautam Jha Liquidity Levels:</b>\n"
        f"• <b>Daily Open (DO):</b> <code>${gj.get('daily_open', 0):,.2f}</code> ({color_emoji} {gj.get('daily_candle_color')} {gj.get('dist_do_pct', 0):+.2f}%)\n"
        f"• <b>Prev Day High (PDH):</b> <code>${gj.get('pdh', 0):,.2f}</code>{pdh_tag}\n"
        f"• <b>Prev Day Low (PDL):</b> <code>${gj.get('pdl', 0):,.2f}</code>{pdl_tag}\n\n"
        f"🎯 <b>Immediate Key Boundaries:</b>\n"
        f"🔺 <b>Resistance:</b> <code>${nr['price']:,.2f}</code> ({nr['name']}) | +{nr['distance_pct']:.2f}%\n"
        f"📍 <b>Price:</b> <code>${mark:,.2f}</code>\n"
        f"🔻 <b>Support:</b> <code>${ns['price']:,.2f}</code> ({ns['name']}) | -{ns['distance_pct']:.2f}%\n\n"
        f"📐 <b>Classic Pivot Points:</b>\n"
        f"• <b>R3:</b> <code>${pivots['r3']:,.2f}</code>\n"
        f"• <b>R2:</b> <code>${pivots['r2']:,.2f}</code>\n"
        f"• <b>R1:</b> <code>${pivots['r1']:,.2f}</code>\n"
        f"• <b>PIVOT (P):</b> <code>${pivots['pivot']:,.2f}</code>\n"
        f"• <b>S1:</b> <code>${pivots['s1']:,.2f}</code>\n"
        f"• <b>S2:</b> <code>${pivots['s2']:,.2f}</code>\n"
        f"• <b>S3:</b> <code>${pivots['s3']:,.2f}</code>\n\n"
        f"🌀 <b>Fibonacci Retracements (24h Range):</b>\n"
        f"• <b>61.8% (Golden):</b> <code>${fibs['61.8%']:,.2f}</code>\n"
        f"• <b>50.0% (Equilibrium):</b> <code>${fibs['50.0%']:,.2f}</code>\n"
        f"• <b>38.2%:</b> <code>${fibs['38.2%']:,.2f}</code>\n\n"
        f"💡 <i>Use <code>/entry {sym}</code> for candle entries or <code>/gj {sym}</code> for Gautam Jha price-action analysis.</i>"
    )
    return msg


def get_gautam_jha_analysis(symbol: str = DEFAULT_SYMBOL) -> Dict[str, Any]:
    """Perform top-down Gautam Jha Price-Action & Liquidity Analysis."""
    sym = resolve_symbol(symbol)
    level_data = get_level_analysis(sym)
    gj = level_data["gautam_jha"]
    mark = level_data["mark_price"]
    candles_15m = get_candles(sym, resolution="15m", count=40)
    atr = TechnicalAnalysis.calc_atr(candles_15m, period=14) if len(candles_15m) >= 15 else (mark * 0.002)

    # Check for immediate Gautam Jha trade setup on 15m and 5m
    setup_15m = None
    if len(candles_15m) >= 3:
        setup_15m = TechnicalAnalysis.detect_gautam_jha_setup(candles_15m[-2], candles_15m[-3], gj, atr)

    # Market structure evaluation
    daily_color = gj["daily_candle_color"]
    if gj["pdh_swept"] and mark < gj["pdh"]:
        market_structure = "Liquidity Grab at PDH (Bearish Rejection Reaction)"
    elif gj["pdl_swept"] and mark > gj["pdl"]:
        market_structure = "Liquidity Grab at PDL (Bullish Bounce Reaction)"
    elif daily_color == "GREEN" and mark > gj["daily_open"]:
        market_structure = f"Bullish Trend (Expanding Green Daily candle, +{gj['dist_do_pct']:.2f}% from DO)"
    elif daily_color == "RED" and mark < gj["daily_open"]:
        market_structure = f"Bearish Trend (Expanding Red Daily candle, {gj['dist_do_pct']:.2f}% from DO)"
    else:
        market_structure = "Consolidating around Daily Open"

    return {
        "symbol": sym,
        "mark_price": mark,
        "gautam_jha": gj,
        "market_structure": market_structure,
        "setup": setup_15m,
        "atr": atr,
    }


def format_gautam_jha_message(analysis: Dict[str, Any]) -> str:
    """Format Gautam Jha analysis into structured Telegram HTML message."""
    sym = analysis["symbol"]
    mark = analysis["mark_price"]
    gj = analysis["gautam_jha"]
    do = gj["daily_open"]
    pdh = gj["pdh"]
    pdl = gj["pdl"]
    color_emoji = "🟢" if gj["daily_candle_color"] == "GREEN" else "🔴"
    pdh_swept_str = "Swept ⚡ (Liquidity grabbed)" if gj["pdh_swept"] else "Untouched"
    pdl_swept_str = "Swept ⚡ (Liquidity grabbed)" if gj["pdl_swept"] else "Untouched"

    lines = [
        f"🎯 <b>GAUTAM JHA PRICE-ACTION ANALYSIS: #{sym}</b>\n",
        f"📊 <b>Chart Context:</b>",
        f"• <b>Instrument:</b> #{sym}",
        f"• <b>Current Price:</b> <code>${mark:,.2f}</code>",
        f"• <b>Location vs DO:</b> {color_emoji} {gj['dist_do_pct']:+.2f}% from Daily Open",
        f"• <b>Daily Candle:</b> {color_emoji} <b>{gj['daily_candle_color']}</b>\n",
        f"💧 <b>Key Liquidity Levels:</b>",
        f"• <b>Daily Open (DO):</b> <code>${do:,.2f}</code> (Major algo flip level)",
        f"• <b>Previous Day High (PDH):</b> <code>${pdh:,.2f}</code> [{pdh_swept_str}]",
        f"• <b>Previous Day Low (PDL):</b> <code>${pdl:,.2f}</code> [{pdl_swept_str}]\n",
        f"⚡ <b>Market Structure:</b>",
        f"• <b>Status:</b> {analysis['market_structure']}\n",
        f"💡 <b>Trade Idea (Gautam Jha Style):</b>",
    ]

    setup = analysis.get("setup")
    if setup:
        direction_emoji = "🟢" if setup["direction"] == "LONG" else "🔴"
        lines.extend([
            f"• <b>Setup Type:</b> <b>{setup['type']}</b> ({setup['quality']} Quality)",
            f"• <b>Direction:</b> {direction_emoji} <b>{setup['direction']}</b>",
            f"• <b>Reason:</b> {setup['reason']}",
            f"• <b>Entry:</b> <code>${setup['entry']:,.2f}</code>",
            f"• <b>Stop Loss:</b> <code>${setup['sl']:,.2f}</code> (Risk: ${setup['risk']:,.2f})",
            f"• <b>TP1 (1:1.5):</b> <code>${setup['tp1']:,.2f}</code>",
            f"• <b>TP2 (1:2.5):</b> <code>${setup['tp2']:,.2f}</code>",
            f"• <b>R:R Ratio:</b> {setup['rrr']}",
        ])
    else:
        # Give tactical guidance based on current location
        if gj["daily_candle_color"] == "GREEN":
            lines.extend([
                f"• <b>Strategy:</b> Look for <b>Break-and-Go</b> on 5m/15m or <b>Retrace-to-DO</b> (${do:,.2f}).",
                f"• <b>Entry Rule:</b> Wait for a green trend candle to break previous high or hold DO.",
                f"• <b>Invalidation:</b> If daily candle flips RED below <code>${do:,.2f}</code>, abort longs.",
            ])
        else:
            lines.extend([
                f"• <b>Strategy:</b> Look for <b>Break-and-Go</b> downward or <b>Retrace-to-DO</b> retest.",
                f"• <b>Entry Rule:</b> Wait for a red trend candle to break previous low or reject DO.",
                f"• <b>Invalidation:</b> If daily candle flips GREEN above <code>${do:,.2f}</code>, abort shorts.",
            ])

    lines.extend([
        f"\n⚠️ <b>Risk Reminder:</b> Educational analysis only. Reaction after liquidity sweep is key. Manage risk!"
    ])

    return "\n".join(lines)


def get_candle_entry(symbol: str = DEFAULT_SYMBOL, timeframe: str = "5m") -> Dict[str, Any]:
    """
    Analyze candlestick price action, indicators, and key levels for a specific timeframe (1m, 5m, 15m).
    Detects Long / Short entry opportunities with defined Entry, Stop Loss, and Take Profit targets.
    """
    sym = resolve_symbol(symbol)
    candles = get_candles(sym, resolution=timeframe, count=60)
    if len(candles) < 25:
        return {
            "symbol": sym,
            "timeframe": timeframe,
            "has_setup": False,
            "signal": "NEUTRAL",
            "reason": "Insufficient candle data",
        }

    # candles[-1] is active open candle; candles[-2] is the latest completed candle
    curr_closed = candles[-2]
    prev_closed = candles[-3]
    active_candle = candles[-1]

    closes = [float(c["close"]) for c in candles]
    vols = [float(c.get("volume", 0.0) or 0.0) for c in candles]

    # Indicators
    rsi = TechnicalAnalysis.calc_rsi(closes, period=14)
    ema9 = TechnicalAnalysis.calc_ema(closes, period=9)
    ema21 = TechnicalAnalysis.calc_ema(closes, period=21)
    atr = TechnicalAnalysis.calc_atr(candles, period=14)

    # Volume surge
    vol_sample = vols[-21:-1]
    avg_vol = sum(vol_sample) / float(len(vol_sample)) if vol_sample else 1.0
    closed_vol = float(curr_closed.get("volume", 0.0) or 0.0)
    vol_ratio = (closed_vol / avg_vol) if avg_vol > 0 else 1.0

    # Pattern detection on closed candle
    pattern_info = TechnicalAnalysis.detect_candle_pattern(curr_closed, prev_closed)
    pattern_name = pattern_info["pattern"]
    pattern_sentiment = pattern_info["sentiment"]

    c = float(curr_closed["close"])
    o = float(curr_closed["open"])
    h = float(curr_closed["high"])
    l = float(curr_closed["low"])

    # Level context & Gautam Jha Liquidity
    gj_levels = {}
    try:
        level_data = get_level_analysis(symbol)
        pivots = level_data["pivots"]
        pivot_p = pivots["pivot"]
        s1 = pivots["s1"]
        r1 = pivots["r1"]
        gj_levels = level_data.get("gautam_jha", {})
    except Exception:
        level_data = None
        pivot_p, s1, r1 = c, c * 0.99, c * 1.01

    gj_setup = TechnicalAnalysis.detect_gautam_jha_setup(curr_closed, prev_closed, gj_levels, atr) if gj_levels else None

    signal = "NEUTRAL"
    reasons = []
    confluence_score = 0

    # Check Bullish Confluences
    bull_confluences = []
    if gj_setup and gj_setup["direction"] == "LONG":
        bull_confluences.append(f"Gautam Jha: {gj_setup['type']} ({gj_setup['reason']})")
        confluence_score += 3
    if "Hammer" in pattern_name or "Bullish Engulfing" in pattern_name:
        bull_confluences.append(f"{pattern_name}")
        confluence_score += 2
    elif pattern_sentiment == "BULLISH":
        bull_confluences.append("Bullish candle close")
        confluence_score += 1

    if rsi < 35:
        bull_confluences.append(f"RSI oversold reversal ({rsi:.1f})")
        confluence_score += 2
    elif 40 <= rsi <= 60 and c > ema9:
        bull_confluences.append(f"RSI bullish momentum ({rsi:.1f})")
        confluence_score += 1

    if c > ema9 and ema9 > ema21:
        bull_confluences.append("Trend alignment (Price > EMA9 > EMA21)")
        confluence_score += 1

    # Bounce off support
    if l <= s1 <= max(o, c) or abs(l - s1) / c < 0.002:
        bull_confluences.append(f"Bounce at Support S1 (${s1:,.2f})")
        confluence_score += 2
    elif l <= pivot_p <= max(o, c) and c > pivot_p:
        bull_confluences.append(f"Holding above Pivot (${pivot_p:,.2f})")
        confluence_score += 1

    if vol_ratio >= 1.5:
        bull_confluences.append(f"High volume ({vol_ratio:.1f}x avg)")
        confluence_score += 1

    # Check Bearish Confluences
    bear_confluences = []
    bear_score = 0
    if gj_setup and gj_setup["direction"] == "SHORT":
        bear_confluences.append(f"Gautam Jha: {gj_setup['type']} ({gj_setup['reason']})")
        bear_score += 3
    if "Shooting Star" in pattern_name or "Bearish Engulfing" in pattern_name:
        bear_confluences.append(f"{pattern_name}")
        bear_score += 2
    elif pattern_sentiment == "BEARISH":
        bear_confluences.append("Bearish candle close")
        bear_score += 1

    if rsi > 65:
        bear_confluences.append(f"RSI overbought rejection ({rsi:.1f})")
        bear_score += 2
    elif 40 <= rsi <= 60 and c < ema9:
        bear_confluences.append(f"RSI bearish momentum ({rsi:.1f})")
        bear_score += 1

    if c < ema9 and ema9 < ema21:
        bear_confluences.append("Trend alignment (Price < EMA9 < EMA21)")
        bear_score += 1

    # Rejection at resistance
    if h >= r1 >= min(o, c) or abs(h - r1) / c < 0.002:
        bear_confluences.append(f"Rejection at Resistance R1 (${r1:,.2f})")
        bear_score += 2
    elif h >= pivot_p >= min(o, c) and c < pivot_p:
        bear_confluences.append(f"Failing below Pivot (${pivot_p:,.2f})")
        bear_score += 1

    if vol_ratio >= 1.5:
        bear_confluences.append(f"High volume ({vol_ratio:.1f}x avg)")
        bear_score += 1

    # Determine Signal
    trade_plan = {}
    has_setup = False

    if confluence_score >= 3 and confluence_score > bear_score:
        signal = "BUY / LONG"
        has_setup = True
        reasons = bull_confluences

        entry_price = round(c, 2)
        # Stop loss below candle low and recent swing
        sl_buffer = max(atr * 0.75, entry_price * 0.001)
        raw_sl = min(l, float(prev_closed["low"])) - sl_buffer
        sl_price = round(raw_sl, 2)
        risk = round(entry_price - sl_price, 2)
        if risk <= 0 or risk < (entry_price * 0.0005):
            risk = round(max(atr, entry_price * 0.002), 2)
            sl_price = round(entry_price - risk, 2)

        tp1_price = round(entry_price + (1.5 * risk), 2)
        tp2_price = round(entry_price + (2.5 * risk), 2)

        trade_plan = {
            "direction": "LONG",
            "entry": entry_price,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "risk": risk,
            "reward_tp1": round(1.5 * risk, 2),
            "reward_tp2": round(2.5 * risk, 2),
            "rrr": "1:1.5 (TP1) / 1:2.5 (TP2)",
        }

    elif bear_score >= 3 and bear_score > confluence_score:
        signal = "SELL / SHORT"
        has_setup = True
        reasons = bear_confluences

        entry_price = round(c, 2)
        sl_buffer = max(atr * 0.75, entry_price * 0.001)
        raw_sl = max(h, float(prev_closed["high"])) + sl_buffer
        sl_price = round(raw_sl, 2)
        risk = round(sl_price - entry_price, 2)
        if risk <= 0 or risk < (entry_price * 0.0005):
            risk = round(max(atr, entry_price * 0.002), 2)
            sl_price = round(entry_price + risk, 2)

        tp1_price = round(entry_price - (1.5 * risk), 2)
        tp2_price = round(entry_price - (2.5 * risk), 2)

        trade_plan = {
            "direction": "SHORT",
            "entry": entry_price,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "risk": risk,
            "reward_tp1": round(1.5 * risk, 2),
            "reward_tp2": round(2.5 * risk, 2),
            "rrr": "1:1.5 (TP1) / 1:2.5 (TP2)",
        }

    else:
        signal = "NEUTRAL"
        has_setup = False
        reasons = [f"No high-probability trigger. RSI {rsi:.1f}, pattern: {pattern_name}"]

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "candle_time": curr_closed["time"],
        "close_price": c,
        "pattern": pattern_name,
        "rsi": rsi,
        "atr": atr,
        "ema9": round(ema9, 2),
        "ema21": round(ema21, 2),
        "vol_ratio": round(vol_ratio, 2),
        "signal": signal,
        "has_setup": has_setup,
        "reasons": reasons,
        "trade_plan": trade_plan,
    }


def get_multi_timeframe_entry(symbol: str = DEFAULT_SYMBOL, timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze 1m, 5m, and 15m candles simultaneously."""
    sym = resolve_symbol(symbol)
    if not timeframes:
        timeframes = ["1m", "5m", "15m"]

    results = {}
    for tf in timeframes:
        try:
            results[tf] = get_candle_entry(sym, timeframe=tf)
        except Exception as e:
            results[tf] = {"timeframe": tf, "has_setup": False, "signal": "ERROR", "reason": str(e)}

    return {"symbol": sym, "timeframes": results}


def format_entry_analysis_message(multi_data: Dict[str, Any]) -> str:
    """Format 1m, 5m, 15m candle entry analysis into Telegram HTML message."""
    sym = multi_data["symbol"]
    lines = [f"🎯 <b>CANDLE ENTRY SCANNER: #{sym}</b>\n<i>(Analyzing 1m, 5m, 15m timeframes)</i>\n"]

    for tf in ["1m", "5m", "15m"]:
        data = multi_data["timeframes"].get(tf)
        if not data:
            continue

        lines.append(f"━━━━━━━━━━━━━━━━━━━")
        lines.append(f"⏱️ <b>Timeframe: {tf.upper()}</b>")

        sig = data.get("signal", "NEUTRAL")
        if "BUY" in sig:
            lines.append(f"🚦 <b>Signal:</b> 🟢 <b>LONG / BUY SETUP</b>")
        elif "SELL" in sig:
            lines.append(f"🚦 <b>Signal:</b> 🔴 <b>SHORT / SELL SETUP</b>")
        else:
            lines.append(f"🚦 <b>Signal:</b> ⚪ <b>NEUTRAL / AWAITING SETUP</b>")

        if "rsi" in data:
            lines.append(
                f"📊 <b>Technical:</b> RSI <code>{data['rsi']}</code> | ATR <code>{data['atr']}</code> | Vol <code>{data['vol_ratio']}x</code>"
            )
            lines.append(f"🕯️ <b>Pattern:</b> {data['pattern']}")

        reasons = data.get("reasons", [])
        if reasons:
            lines.append(f"🔍 <b>Triggers:</b> {', '.join(reasons)}")

        if data.get("has_setup") and data.get("trade_plan"):
            tp = data["trade_plan"]
            lines.append(f"\n📋 <b>Actionable Trade Plan:</b>")
            lines.append(f"• <b>Entry:</b> <code>${tp['entry']:,.2f}</code>")
            lines.append(f"• <b>Stop Loss:</b> <code>${tp['sl']:,.2f}</code> (Risk: ${tp['risk']:,.2f})")
            lines.append(f"• <b>Take Profit 1:</b> <code>${tp['tp1']:,.2f}</code> (1:1.5)")
            lines.append(f"• <b>Take Profit 2:</b> <code>${tp['tp2']:,.2f}</code> (1:2.5)")
            lines.append(f"• <b>R:R Ratio:</b> {tp['rrr']}")

        lines.append("")

    lines.append("🔔 <i>To get automatic alerts when a setup forms:</i>")
    lines.append(f"<code>/watch {sym}</code>")

    return "\n".join(lines)
