"""
Technical analysis indicators and candlestick pattern detection.
Zero-dependency, pure Python implementation for high performance and portability.
"""
from typing import List, Dict, Any, Optional, Tuple


class TechnicalAnalysis:
    """Calculates technical indicators, support/resistance levels, and candlestick patterns."""

    @staticmethod
    def calc_sma(values: List[float], period: int) -> float:
        if not values:
            return 0.0
        if len(values) < period:
            return sum(values) / len(values)
        return sum(values[-period:]) / float(period)

    @staticmethod
    def calc_ema(values: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if not values:
            return 0.0
        if len(values) < period:
            return sum(values) / len(values)
        k = 2.0 / (period + 1.0)
        ema = sum(values[:period]) / float(period)
        for v in values[period:]:
            ema = v * k + ema * (1.0 - k)
        return ema

    @staticmethod
    def calc_rsi(closes: List[float], period: int = 14) -> float:
        """Calculate standard Wilder's Relative Strength Index."""
        if len(closes) < period + 1:
            return 50.0

        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))

        if len(gains) < period:
            return 50.0

        avg_gain = sum(gains[:period]) / float(period)
        avg_loss = sum(losses[:period]) / float(period)

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1.0) + gains[i]) / float(period)
            avg_loss = (avg_loss * (period - 1.0) + losses[i]) / float(period)

        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return round(rsi, 2)

    @staticmethod
    def calc_atr(candles: List[Dict[str, Any]], period: int = 14) -> float:
        """Calculate Average True Range from candle dictionaries."""
        if len(candles) < 2:
            return 0.0

        trs = []
        for i in range(1, len(candles)):
            h = float(candles[i]["high"])
            l = float(candles[i]["low"])
            prev_c = float(candles[i - 1]["close"])
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)

        if not trs:
            return 0.0
        if len(trs) < period:
            return sum(trs) / len(trs)

        atr = sum(trs[:period]) / float(period)
        for tr in trs[period:]:
            atr = (atr * (period - 1.0) + tr) / float(period)

        return round(atr, 4)

    @staticmethod
    def calc_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Classic and Fibonacci Pivot Points from High, Low, Close.
        """
        p = (high + low + close) / 3.0
        diff = high - low

        # Classic Pivots
        r1 = 2.0 * p - low
        s1 = 2.0 * p - high
        r2 = p + diff
        s2 = p - diff
        r3 = high + 2.0 * (p - low)
        s3 = low - 2.0 * (high - p)

        # Fibonacci Pivots
        fib_r1 = p + 0.382 * diff
        fib_r2 = p + 0.618 * diff
        fib_r3 = p + 1.000 * diff
        fib_s1 = p - 0.382 * diff
        fib_s2 = p - 0.618 * diff
        fib_s3 = p - 1.000 * diff

        return {
            "pivot": round(p, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
            "fib_r1": round(fib_r1, 2),
            "fib_r2": round(fib_r2, 2),
            "fib_r3": round(fib_r3, 2),
            "fib_s1": round(fib_s1, 2),
            "fib_s2": round(fib_s2, 2),
            "fib_s3": round(fib_s3, 2),
        }

    @staticmethod
    def calc_fib_retracements(swing_high: float, swing_low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels for a given swing high and low."""
        rng = swing_high - swing_low
        return {
            "100.0%": round(swing_high, 2),
            "78.6%": round(swing_low + 0.786 * rng, 2),
            "61.8%": round(swing_low + 0.618 * rng, 2),
            "50.0%": round(swing_low + 0.500 * rng, 2),
            "38.2%": round(swing_low + 0.382 * rng, 2),
            "23.6%": round(swing_low + 0.236 * rng, 2),
            "0.0%": round(swing_low, 2),
        }

    @staticmethod
    def find_swing_levels(candles: List[Dict[str, Any]], lookback: int = 30) -> Tuple[float, float]:
        """Find recent swing high and swing low from historical candles."""
        if not candles:
            return 0.0, 0.0
        sample = candles[-lookback:] if len(candles) >= lookback else candles
        highs = [float(c["high"]) for c in sample]
        lows = [float(c["low"]) for c in sample]
        return round(max(highs), 2), round(min(lows), 2)

    @staticmethod
    def detect_candle_pattern(
        curr_candle: Dict[str, Any],
        prev_candle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect candlestick patterns on a completed candle:
        - Bullish Hammer / Pin Bar
        - Bearish Shooting Star / Pin Bar
        - Bullish Engulfing
        - Bearish Engulfing
        - Bullish / Bearish Momentum Candle
        - Doji
        """
        o = float(curr_candle["open"])
        c = float(curr_candle["close"])
        h = float(curr_candle["high"])
        l = float(curr_candle["low"])

        body = abs(c - o)
        rng = h - l
        if rng <= 0:
            return {"pattern": "Flat / Inactive", "sentiment": "NEUTRAL", "strength": 0}

        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        is_bullish = c > o
        is_bearish = c < o

        pattern = "Normal Candle"
        sentiment = "BULLISH" if is_bullish else ("BEARISH" if is_bearish else "NEUTRAL")
        strength = 1

        # Doji
        if body <= 0.10 * rng:
            return {
                "pattern": "Doji (Indecision)",
                "sentiment": "NEUTRAL",
                "strength": 1,
                "body": body,
                "range": rng,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
            }

        # Hammer / Bullish Pin Bar: long lower shadow, small upper shadow
        if lower_wick >= 1.8 * body and upper_wick <= 0.35 * rng and body >= 0.08 * rng:
            pattern = "Bullish Hammer / Pin Bar Rejection"
            sentiment = "BULLISH"
            strength = 3

        # Shooting Star / Bearish Pin Bar: long upper shadow, small lower shadow
        elif upper_wick >= 1.8 * body and lower_wick <= 0.35 * rng and body >= 0.08 * rng:
            pattern = "Bearish Shooting Star / Pin Bar Rejection"
            sentiment = "BEARISH"
            strength = 3

        # Engulfing patterns
        elif prev_candle:
            p_o = float(prev_candle["open"])
            p_c = float(prev_candle["close"])
            p_body = abs(p_c - p_o)
            p_bearish = p_c < p_o
            p_bullish = p_c > p_o

            if p_bearish and is_bullish and c >= p_o and o <= p_c and body > p_body:
                pattern = "Bullish Engulfing"
                sentiment = "BULLISH"
                strength = 3
            elif p_bullish and is_bearish and c <= p_o and o >= p_c and body > p_body:
                pattern = "Bearish Engulfing"
                sentiment = "BEARISH"
                strength = 3
            elif is_bullish and body >= 0.70 * rng and body > 1.2 * p_body:
                pattern = "Strong Bullish Expansion"
                sentiment = "BULLISH"
                strength = 2
            elif is_bearish and body >= 0.70 * rng and body > 1.2 * p_body:
                pattern = "Strong Bearish Expansion"
                sentiment = "BEARISH"
                strength = 2

        return {
            "pattern": pattern,
            "sentiment": sentiment,
            "strength": strength,
            "body": body,
            "range": rng,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
        }

    @staticmethod
    def calc_gautam_jha_levels(daily_candles: List[Dict[str, Any]], current_price: float) -> Dict[str, Any]:
        """
        Calculate Gautam Jha Liquidity Levels:
        - Daily Open (DO): high probability level where many algos reverse on color flip
        - Previous Day High (PDH) & Previous Day Low (PDL): primary liquidity pools
        - Daily candle color (GREEN / RED) and distance from DO
        - Liquidity Sweep state (PDH swept, PDL swept)
        """
        if len(daily_candles) < 2:
            return {
                "daily_open": current_price,
                "pdh": current_price,
                "pdl": current_price,
                "pdc": current_price,
                "today_high": current_price,
                "today_low": current_price,
                "daily_candle_color": "NEUTRAL",
                "dist_do": 0.0,
                "dist_do_pct": 0.0,
                "pdh_swept": False,
                "pdl_swept": False,
            }

        today = daily_candles[-1]
        yesterday = daily_candles[-2]

        daily_open = float(today["open"])
        today_high = float(today["high"])
        today_low = float(today["low"])
        pdh = float(yesterday["high"])
        pdl = float(yesterday["low"])
        pdc = float(yesterday["close"])

        daily_color = "GREEN" if current_price >= daily_open else "RED"
        dist_do = current_price - daily_open
        dist_do_pct = (dist_do / daily_open * 100.0) if daily_open > 0 else 0.0

        pdh_swept = today_high > pdh
        pdl_swept = today_low < pdl

        return {
            "daily_open": round(daily_open, 2),
            "pdh": round(pdh, 2),
            "pdl": round(pdl, 2),
            "pdc": round(pdc, 2),
            "today_high": round(today_high, 2),
            "today_low": round(today_low, 2),
            "daily_candle_color": daily_color,
            "dist_do": round(dist_do, 2),
            "dist_do_pct": round(dist_do_pct, 2),
            "pdh_swept": pdh_swept,
            "pdl_swept": pdl_swept,
        }

    @staticmethod
    def detect_gautam_jha_setup(
        curr_candle: Dict[str, Any],
        prev_candle: Dict[str, Any],
        gj_levels: Dict[str, Any],
        atr: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Identify Gautam Jha's 3 core trade styles based on price action & liquidity:
        1. Break-and-Go (Momentum): strong trend, new candle breaks high/low of previous candle
        2. Retrace-to-Level: price tests Daily Open or key liquidity level and prints trend candle
        3. Level Reversal / Liquidity Grab: price sweeps PDH/PDL or key level and rejects back
        """
        c = float(curr_candle["close"])
        o = float(curr_candle["open"])
        h = float(curr_candle["high"])
        l = float(curr_candle["low"])

        p_c = float(prev_candle["close"])
        p_o = float(prev_candle["open"])
        p_h = float(prev_candle["high"])
        p_l = float(prev_candle["low"])

        body = abs(c - o)
        rng = h - l or 0.01
        is_bull = c > o
        is_bear = c < o

        daily_color = gj_levels.get("daily_candle_color", "NEUTRAL")
        do = gj_levels.get("daily_open", c)
        pdh = gj_levels.get("pdh", c)
        pdl = gj_levels.get("pdl", c)

        # 1. Level Reversal / Liquidity Grab
        # Swept PDH liquidity then closed back below PDH
        if h >= pdh and c < pdh and is_bear:
            sl_price = round(h + max(0.5 * atr, c * 0.001), 2)
            risk = round(sl_price - c, 2)
            return {
                "type": "Level Reversal / Liquidity Grab",
                "direction": "SHORT",
                "reason": f"Swept PDH (${pdh:,.2f}) buy-side liquidity and rejected back below PDH",
                "entry": round(c, 2),
                "sl": sl_price,
                "tp1": round(c - 1.5 * risk, 2),
                "tp2": round(c - 2.5 * risk, 2),
                "risk": risk,
                "rrr": "1:1.5 / 1:2.5",
                "quality": "HIGH",
            }

        # Swept PDL liquidity then closed back above PDL
        if l <= pdl and c > pdl and is_bull:
            sl_price = round(l - max(0.5 * atr, c * 0.001), 2)
            risk = round(c - sl_price, 2)
            return {
                "type": "Level Reversal / Liquidity Grab",
                "direction": "LONG",
                "reason": f"Swept PDL (${pdl:,.2f}) sell-side liquidity and rejected back above PDL",
                "entry": round(c, 2),
                "sl": sl_price,
                "tp1": round(c + 1.5 * risk, 2),
                "tp2": round(c + 2.5 * risk, 2),
                "risk": risk,
                "rrr": "1:1.5 / 1:2.5",
                "quality": "HIGH",
            }

        # 2. Retrace-to-Level (Daily Open test & reaction)
        if abs(l - do) / do < 0.002 and is_bull and daily_color == "GREEN":
            sl_price = round(min(l, do) - max(0.5 * atr, c * 0.001), 2)
            risk = round(c - sl_price, 2)
            return {
                "type": "Retrace-to-Level",
                "direction": "LONG",
                "reason": f"Pullback to Daily Open (${do:,.2f}) held, daily candle remains Green",
                "entry": round(c, 2),
                "sl": sl_price,
                "tp1": round(c + 1.5 * risk, 2),
                "tp2": round(c + 2.5 * risk, 2),
                "risk": risk,
                "rrr": "1:1.5 / 1:2.5",
                "quality": "HIGH",
            }

        if abs(h - do) / do < 0.002 and is_bear and daily_color == "RED":
            sl_price = round(max(h, do) + max(0.5 * atr, c * 0.001), 2)
            risk = round(sl_price - c, 2)
            return {
                "type": "Retrace-to-Level",
                "direction": "SHORT",
                "reason": f"Retest of Daily Open (${do:,.2f}) rejected, daily candle remains Red",
                "entry": round(c, 2),
                "sl": sl_price,
                "tp1": round(c - 1.5 * risk, 2),
                "tp2": round(c - 2.5 * risk, 2),
                "risk": risk,
                "rrr": "1:1.5 / 1:2.5",
                "quality": "HIGH",
            }

        # 3. Break-and-Go (Momentum)
        if daily_color == "GREEN" and is_bull and c > p_h and body >= 0.55 * rng:
            sl_price = round(min(l, p_l) - max(0.5 * atr, c * 0.001), 2)
            risk = round(c - sl_price, 2)
            return {
                "type": "Break-and-Go (Momentum)",
                "direction": "LONG",
                "reason": f"Broke above previous candle high (${p_h:,.2f}) with strong green expansion in Green Daily trend",
                "entry": round(c, 2),
                "sl": sl_price,
                "tp1": round(c + 1.5 * risk, 2),
                "tp2": round(c + 2.5 * risk, 2),
                "risk": risk,
                "rrr": "1:1.5 / 1:2.5",
                "quality": "MODERATE",
            }

        if daily_color == "RED" and is_bear and c < p_l and body >= 0.55 * rng:
            sl_price = round(max(h, p_h) + max(0.5 * atr, c * 0.001), 2)
            risk = round(sl_price - c, 2)
            return {
                "type": "Break-and-Go (Momentum)",
                "direction": "SHORT",
                "reason": f"Broke below previous candle low (${p_l:,.2f}) with strong red expansion in Red Daily trend",
                "entry": round(c, 2),
                "sl": sl_price,
                "tp1": round(c - 1.5 * risk, 2),
                "tp2": round(c - 2.5 * risk, 2),
                "risk": risk,
                "rrr": "1:1.5 / 1:2.5",
                "quality": "MODERATE",
            }

        return None

