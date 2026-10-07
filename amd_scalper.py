"""
AMD (Accumulation - Manipulation - Distribution) Scalping Engine.
Analyzes 1m, 5m, and 15m candlesticks to detect high-probability institutional liquidity sweeps,
Judas swings, Market Structure Shifts (MSS), and executes sniper scalp trades with
automated Stop-Loss, Take-Profit (TP1/TP2), and dynamic risk management.

Zero LLM tokens required for continuous mathematical AMD calculations.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from indicators import TechnicalAnalysis
from market_data import (
    resolve_symbol,
    get_symbol_display_name,
    get_ticker,
    get_candles,
    get_level_analysis,
    DEFAULT_SYMBOL,
)

logger = logging.getLogger(__name__)


def detect_accumulation_range(candles_5m: List[Dict[str, Any]], window: int = 10) -> Optional[Dict[str, Any]]:
    """
    Detect Accumulation phase on 5m timeframe.
    Examines the consolidation window for range bounds and compression.
    """
    if len(candles_5m) < window + 2:
        return None

    # Exclude the active candle (candles[-1]) and latest completed candle (candles[-2] which might be manipulation)
    eval_candles = candles_5m[-(window + 2):-2]
    if len(eval_candles) < window:
        return None

    highs = [float(c["high"]) for c in eval_candles]
    lows = [float(c["low"]) for c in eval_candles]
    closes = [float(c["close"]) for c in eval_candles]

    range_high = max(highs)
    range_low = min(lows)
    range_eq = round((range_high + range_low) / 2.0, 2)
    range_height = range_high - range_low

    if range_height <= 0:
        return None

    atr = TechnicalAnalysis.calc_atr(candles_5m, period=14) or (range_eq * 0.002)

    # Tightness / Compression: Range height should generally be within 1.0 to 4.5 ATRs
    compression_ratio = range_height / atr if atr > 0 else 2.0
    is_compressed = compression_ratio <= 5.0

    return {
        "range_high": round(range_high, 2),
        "range_low": round(range_low, 2),
        "range_eq": range_eq,
        "range_height": round(range_height, 2),
        "compression_ratio": round(compression_ratio, 2),
        "is_compressed": is_compressed,
        "candles_count": len(eval_candles),
    }


def detect_manipulation_sweep(
    candles_5m: List[Dict[str, Any]],
    range_info: Dict[str, Any],
    atr_5m: float,
) -> Optional[Dict[str, Any]]:
    """
    Detect Manipulation (Judas Swing / Liquidity Sweep) on 5m.
    Checks if recent candles swept outside the accumulation range and showed rejection.
    """
    if len(candles_5m) < 3 or not range_info:
        return None

    # Inspect the last 2 completed 5m candles
    cand_curr = candles_5m[-2]
    cand_prev = candles_5m[-3]

    c_curr = float(cand_curr["close"])
    o_curr = float(cand_curr["open"])
    h_curr = float(cand_curr["high"])
    l_curr = float(cand_curr["low"])

    c_prev = float(cand_prev["close"])
    h_prev = float(cand_prev["high"])
    l_prev = float(cand_prev["low"])

    range_high = range_info["range_high"]
    range_low = range_info["range_low"]

    # 1. Bearish Manipulation -> Bullish Setup (Sell-side liquidity sweep below Range Low)
    # Price dips below range_low, but closes back above or rejects aggressively
    swept_low = min(l_curr, l_prev)
    if swept_low < range_low:
        # Check for rejection or close back above range_low or lower wick presence
        rejection_wick = min(o_curr, c_curr) - l_curr
        body_size = abs(c_curr - o_curr)
        is_rejection = (c_curr >= range_low * 0.999) or (rejection_wick > body_size * 0.8) or (c_curr > o_curr)

        if is_rejection:
            sweep_depth = round(range_low - swept_low, 2)
            return {
                "type": "BEARISH_SWEEP_REJECTION",
                "bias": "BULLISH",
                "phase": "MANIPULATION",
                "sweep_level": swept_low,
                "range_boundary": range_low,
                "sweep_depth": sweep_depth,
                "reason": f"Judas Swing swept sell stops at ${swept_low:,.2f} below Range Low (${range_low:,.2f})",
            }

    # 2. Bullish Manipulation -> Bearish Setup (Buy-side liquidity sweep above Range High)
    # Price spikes above range_high, but closes back below or rejects aggressively
    swept_high = max(h_curr, h_prev)
    if swept_high > range_high:
        rejection_wick = h_curr - max(o_curr, c_curr)
        body_size = abs(c_curr - o_curr)
        is_rejection = (c_curr <= range_high * 1.001) or (rejection_wick > body_size * 0.8) or (c_curr < o_curr)

        if is_rejection:
            sweep_depth = round(swept_high - range_high, 2)
            return {
                "type": "BULLISH_SWEEP_REJECTION",
                "bias": "BEARISH",
                "phase": "MANIPULATION",
                "sweep_level": swept_high,
                "range_boundary": range_high,
                "sweep_depth": sweep_depth,
                "reason": f"Judas Swing swept buy stops at ${swept_high:,.2f} above Range High (${range_high:,.2f})",
            }

    return None


def detect_1m_distribution_trigger(
    candles_1m: List[Dict[str, Any]],
    bias: str,
    manipulation_sweep_level: float,
    atr_1m: float,
) -> Optional[Dict[str, Any]]:
    """
    Detect 1m Sniper Execution Trigger:
    - Market Structure Shift (MSS)
    - Displacement candle breaking 1m swing points in the direction of Distribution
    - Fair Value Gap (FVG) or EMA9 momentum
    """
    if len(candles_1m) < 10:
        return None

    # Latest completed 1m candle and recent 1m swing points
    curr = candles_1m[-2]
    prev = candles_1m[-3]

    closes = [float(c["close"]) for c in candles_1m[-15:-1]]
    ema9_1m = TechnicalAnalysis.calc_ema(closes, period=9)

    c = float(curr["close"])
    o = float(curr["open"])
    h = float(curr["high"])
    l = float(curr["low"])

    body = abs(c - o)
    candle_range = max(h - l, 0.0001)
    body_ratio = body / candle_range

    # Local 1m swing points over the last 6 candles prior to curr
    prior_highs = [float(c["high"]) for c in candles_1m[-8:-3]]
    prior_lows = [float(c["low"]) for c in candles_1m[-8:-3]]
    recent_swing_high = max(prior_highs) if prior_highs else h
    recent_swing_low = min(prior_lows) if prior_lows else l

    if bias == "BULLISH":
        # Bullish Distribution: Price displaces up
        # MSS: Close breaks above recent 1m swing high or strong green expansion > EMA9
        is_green = c > o
        breaks_swing = c >= recent_swing_high
        above_ema = c > ema9_1m
        strong_body = is_green and (body_ratio >= 0.5 or body >= atr_1m * 0.7)

        if is_green and (breaks_swing or (above_ema and strong_body)):
            trigger_type = "MSS_BREAK" if breaks_swing else "MOMENTUM_DISPLACEMENT"
            return {
                "triggered": True,
                "side": "buy",
                "trigger_type": trigger_type,
                "entry_price": c,
                "invalidation_level": manipulation_sweep_level,
                "mss_level": recent_swing_high,
                "description": f"1m {trigger_type}: Energetic bullish displacement ({c:,.2f} > 1m EMA9 {ema9_1m:,.2f})",
            }

    elif bias == "BEARISH":
        # Bearish Distribution: Price displaces down
        # MSS: Close breaks below recent 1m swing low or strong red expansion < EMA9
        is_red = c < o
        breaks_swing = c <= recent_swing_low
        below_ema = c < ema9_1m
        strong_body = is_red and (body_ratio >= 0.5 or body >= atr_1m * 0.7)

        if is_red and (breaks_swing or (below_ema and strong_body)):
            trigger_type = "MSS_BREAK" if breaks_swing else "MOMENTUM_DISPLACEMENT"
            return {
                "triggered": True,
                "side": "sell",
                "trigger_type": trigger_type,
                "entry_price": c,
                "invalidation_level": manipulation_sweep_level,
                "mss_level": recent_swing_low,
                "description": f"1m {trigger_type}: Energetic bearish displacement ({c:,.2f} < 1m EMA9 {ema9_1m:,.2f})",
            }

    return None


def calculate_risk_managed_plan(
    symbol: str,
    side: str,
    entry_price: float,
    invalidation_level: float,
    range_target: float,
    external_target: float,
    atr_1m: float,
    balance: float = 10000.0,
    risk_pct: float = 0.015,
) -> Dict[str, Any]:
    """
    Compute mathematically sound, structure-based Stop-Loss, Take-Profit targets,
    and automatic position size based on capital risk management.
    """
    sym = resolve_symbol(symbol)
    side_lower = side.lower()

    # Dynamic SL Buffer: 0.5x 1m ATR or min 0.08%
    sl_buffer = max(atr_1m * 0.5, entry_price * 0.0008)

    if side_lower in ("buy", "long"):
        sl_price = round(invalidation_level - sl_buffer, 2)
        risk = round(entry_price - sl_price, 2)
        if risk <= 0:
            risk = round(max(atr_1m * 1.5, entry_price * 0.002), 2)
            sl_price = round(entry_price - risk, 2)

        # TP1: Target opposite range or minimum 1:1.8 RRR
        tp1_structural = round(range_target, 2)
        tp1_min_rrr = round(entry_price + (1.8 * risk), 2)
        tp1_price = max(tp1_structural, tp1_min_rrr)

        # TP2: External liquidity or 1:3.0 RRR
        tp2_price = round(max(external_target, entry_price + (3.0 * risk)), 2)

        reward_tp1 = round(tp1_price - entry_price, 2)
        reward_tp2 = round(tp2_price - entry_price, 2)

    else:  # sell / short
        sl_price = round(invalidation_level + sl_buffer, 2)
        risk = round(sl_price - entry_price, 2)
        if risk <= 0:
            risk = round(max(atr_1m * 1.5, entry_price * 0.002), 2)
            sl_price = round(entry_price + risk, 2)

        # TP1: Target opposite range or minimum 1:1.8 RRR
        tp1_structural = round(range_target, 2)
        tp1_min_rrr = round(entry_price - (1.8 * risk), 2)
        tp1_price = min(tp1_structural, tp1_min_rrr)

        # TP2: External liquidity or 1:3.0 RRR
        tp2_price = round(min(external_target, entry_price - (3.0 * risk)), 2)

        reward_tp1 = round(entry_price - tp1_price, 2)
        reward_tp2 = round(entry_price - tp2_price, 2)

    rrr_tp1 = round(reward_tp1 / risk, 2) if risk > 0 else 1.8
    rrr_tp2 = round(reward_tp2 / risk, 2) if risk > 0 else 3.0

    # Auto Position Sizing based on Capital Risk Management
    capital_at_risk = balance * risk_pct  # e.g. 1.5% of $10,000 = $150
    calculated_size = round(capital_at_risk / risk, 4) if risk > 0 else 1.0

    # Sanitize sizes based on asset conventions
    if "BTC" in sym:
        # Min 0.001 BTC, Max cap 5.0
        final_size = max(0.001, min(round(calculated_size, 3), 5.0))
    elif "XAU" in sym or "GOLD" in sym:
        # Tether Gold contracts
        final_size = max(0.01, min(round(calculated_size, 2), 25.0))
    elif "ETH" in sym:
        final_size = max(0.01, min(round(calculated_size, 2), 50.0))
    else:
        final_size = max(0.1, min(round(calculated_size, 1), 100.0))

    return {
        "side": "buy" if side_lower in ("buy", "long") else "sell",
        "entry": round(entry_price, 2),
        "sl": round(sl_price, 2),
        "tp1": round(tp1_price, 2),
        "tp2": round(tp2_price, 2),
        "risk_points": risk,
        "reward_tp1": reward_tp1,
        "reward_tp2": reward_tp2,
        "rrr": f"1:{rrr_tp1} (TP1) / 1:{rrr_tp2} (TP2)",
        "rrr_numeric": rrr_tp1,
        "capital_at_risk": round(capital_at_risk, 2),
        "suggested_size": final_size,
    }


def analyze_amd_scalp(
    symbol: str = DEFAULT_SYMBOL,
    balance: float = 10000.0,
    risk_pct: float = 0.015,
) -> Dict[str, Any]:
    """
    Complete Multi-Timeframe AMD Scalp Analysis across 1m, 5m, and 15m candles.
    Detects Accumulation, Manipulation (Judas Swing), and Distribution triggers.
    """
    sym = resolve_symbol(symbol)
    ticker = get_ticker(sym)
    mark_price = float(ticker["mark_price"] or ticker["close"])

    # 1. Fetch multi-timeframe candles
    candles_15m = get_candles(sym, resolution="15m", count=40)
    candles_5m = get_candles(sym, resolution="5m", count=40)
    candles_1m = get_candles(sym, resolution="1m", count=40)

    if len(candles_15m) < 15 or len(candles_5m) < 15 or len(candles_1m) < 15:
        return {
            "symbol": sym,
            "has_setup": False,
            "is_executable": False,
            "phase": "INSUFFICIENT_DATA",
            "reason": "Need at least 15 candles for 1m, 5m, and 15m timeframes.",
        }

    # 2. 15m Higher-Timeframe Trend & Bias
    c15_closes = [float(c["close"]) for c in candles_15m]
    ema9_15m = TechnicalAnalysis.calc_ema(c15_closes, period=9)
    ema21_15m = TechnicalAnalysis.calc_ema(c15_closes, period=21)
    atr_15m = TechnicalAnalysis.calc_atr(candles_15m, period=14) or (mark_price * 0.002)

    if ema9_15m > ema21_15m and mark_price >= ema21_15m:
        bias_15m = "BULLISH"
    elif ema9_15m < ema21_15m and mark_price <= ema21_15m:
        bias_15m = "BEARISH"
    else:
        bias_15m = "NEUTRAL"

    # Reference levels from 15m
    h15 = max(float(c["high"]) for c in candles_15m[-12:])
    l15 = min(float(c["low"]) for c in candles_15m[-12:])

    # 3. 5m Accumulation Range Analysis
    atr_5m = TechnicalAnalysis.calc_atr(candles_5m, period=14) or (mark_price * 0.0015)
    range_info = detect_accumulation_range(candles_5m, window=10)

    if not range_info:
        return {
            "symbol": sym,
            "has_setup": False,
            "is_executable": False,
            "phase": "NO_ACCUMULATION",
            "15m_bias": bias_15m,
            "mark_price": mark_price,
            "reason": "No established 5m consolidation/accumulation range found.",
        }

    # 4. 5m Manipulation (Judas Swing) Detection
    manipulation = detect_manipulation_sweep(candles_5m, range_info, atr_5m)

    # 5. 1m Sniper Distribution Trigger Detection
    atr_1m = TechnicalAnalysis.calc_atr(candles_1m, period=14) or (mark_price * 0.0008)
    trigger = None
    if manipulation:
        trigger = detect_1m_distribution_trigger(
            candles_1m=candles_1m,
            bias=manipulation["bias"],
            manipulation_sweep_level=manipulation["sweep_level"],
            atr_1m=atr_1m,
        )

    # 6. Build Trade Plan if triggered
    trade_plan = None
    is_executable = False
    has_setup = False
    current_phase = "ACCUMULATION"

    if manipulation:
        current_phase = "MANIPULATION"
        if trigger and trigger.get("triggered"):
            current_phase = "DISTRIBUTION"
            has_setup = True
            is_executable = True

            # Targets:
            # If Long: TP1 is Range High, TP2 is 15m High
            # If Short: TP1 is Range Low, TP2 is 15m Low
            range_target = range_info["range_high"] if trigger["side"] == "buy" else range_info["range_low"]
            external_target = h15 if trigger["side"] == "buy" else l15

            trade_plan = calculate_risk_managed_plan(
                symbol=sym,
                side=trigger["side"],
                entry_price=mark_price,
                invalidation_level=manipulation["sweep_level"],
                range_target=range_target,
                external_target=external_target,
                atr_1m=atr_1m,
                balance=balance,
                risk_pct=risk_pct,
            )

    return {
        "symbol": sym,
        "mark_price": mark_price,
        "has_setup": has_setup,
        "is_executable": is_executable,
        "phase": current_phase,
        "direction": trigger["side"].upper() if trigger else (manipulation["bias"] if manipulation else "NEUTRAL"),
        "timeframe_alignment": {
            "15m_bias": bias_15m,
            "15m_ema9": round(ema9_15m, 2),
            "15m_ema21": round(ema21_15m, 2),
            "15m_atr": round(atr_15m, 2),
            "5m_compression": range_info["is_compressed"],
            "1m_atr": round(atr_1m, 2),
        },
        "range_levels": range_info,
        "manipulation": manipulation,
        "trigger": trigger,
        "trade_plan": trade_plan,
        "timestamp": int(time.time()),
    }


def format_amd_scalp_html_report(analysis: Dict[str, Any]) -> str:
    """Format rich HTML card for Telegram detailing 1m, 5m, 15m AMD Scalp analysis."""
    sym = analysis.get("symbol", "BTCUSD")
    display_name = get_symbol_display_name(sym)
    mark = analysis.get("mark_price", 0.0)
    phase = analysis.get("phase", "SCANNING")
    has_setup = analysis.get("has_setup", False)
    plan = analysis.get("trade_plan")

    phase_emojis = {
        "ACCUMULATION": "📦 ACCUMULATION (Range Building)",
        "MANIPULATION": "⚡ MANIPULATION (Judas Sweep Detected)",
        "DISTRIBUTION": "🚀 DISTRIBUTION (Execution Active)",
        "NO_ACCUMULATION": "🔍 SCANNING FOR ACCUMULATION",
    }
    phase_banner = phase_emojis.get(phase, f"📊 PHASE: {phase}")

    tf = analysis.get("timeframe_alignment", {})
    rng = analysis.get("range_levels", {})
    manip = analysis.get("manipulation")
    trig = analysis.get("trigger")

    lines = [
        f"⚡ <b>AMD SCALP TRADING DESK: {display_name}</b> (<code>#{sym}</code>)\n"
        f"<i>(Multi-Timeframe 1m • 5m • 15m Institutional Engine)</i>\n",
        f"💰 <b>Current Price:</b> <code>${mark:,.2f}</code>",
        f"🎯 <b>AMD Cycle Status:</b> <b>{phase_banner}</b>\n",
        "📊 <b>Multi-Timeframe Structure:</b>",
        f"• <b>15m Higher-TF Bias:</b> <code>{tf.get('15m_bias', 'NEUTRAL')}</code> (EMA9: ${tf.get('15m_ema9', 0):,.2f})",
    ]

    if rng:
        lines.extend([
            f"• <b>5m Accumulation Range:</b> [<code>${rng.get('range_low', 0):,.2f}</code> ↔ <code>${rng.get('range_high', 0):,.2f}</code>]",
            f"  └ <i>Mid Equilibrium:</i> <code>${rng.get('range_eq', 0):,.2f}</code> | Height: <code>${rng.get('range_height', 0):,.2f}</code>",
        ])

    if manip:
        lines.append(f"• <b>5m Judas Manipulation:</b> ⚠️ <code>{manip.get('reason')}</code>")
    else:
        lines.append("• <b>5m Judas Manipulation:</b> ⏳ <i>Watching range extremes for stop hunt sweeps</i>")

    if trig:
        lines.append(f"• <b>1m Sniper Trigger:</b> ✅ <code>{trig.get('description')}</code>")
    else:
        lines.append("• <b>1m Sniper Trigger:</b> ⏳ <i>Waiting for 1m Market Structure Shift (MSS) displacement</i>")

    lines.append("")

    if has_setup and plan:
        side_tag = "🟢 LONG EXECUTION" if plan["side"] == "buy" else "🔴 SHORT EXECUTION"
        lines.extend([
            f"🚀 <b>AUTO-SCALP TRADE PLAN READY ({side_tag}):</b>",
            f"• <b>Entry:</b> <code>${plan['entry']:,.2f}</code>",
            f"• <b>🛑 Auto Stop Loss:</b> <code>${plan['sl']:,.2f}</code> (Risk: <code>${plan['risk_points']:,.2f}</code>)",
            f"• <b>🎯 Take Profit 1:</b> <code>${plan['tp1']:,.2f}</code> (Range Target)",
            f"• <b>🎯 Take Profit 2:</b> <code>${plan['tp2']:,.2f}</code> (Expansion Target)",
            f"• <b>⚖️ Risk/Reward:</b> <code>{plan['rrr']}</code>",
            f"• <b>🛡️ Capital at Risk (1.5%):</b> <code>${plan['capital_at_risk']:,.2f}</code>",
            f"• <b>📦 Auto-Sized Position:</b> <code>{plan['suggested_size']}</code> contracts",
            "",
            "💡 <i>To trigger this scalp immediately:</i>\n"
            f"<code>/trade {sym} {plan['side']} {plan['suggested_size']}</code>",
        ])
    else:
        lines.extend([
            "🛡️ <b>Risk Management Gate:</b>",
            "• Auto-Execution requires: 5m Judas Sweep + 1m MSS Displacement + Min 1:1.8 RRR.",
            "• Capital Risk Cap: 1.5% per scalp trade.",
            "",
            "💡 <i>Auto-Trading will trigger automatically when 1m/5m/15m criteria align!</i>",
        ])

    return "\n".join(lines)
