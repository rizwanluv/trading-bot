"""
Multi-Confluence Master Strategy Engine.
Combines all 5 core trading strategies into a unified high-probability decision model:
1. Gautam Jha Liquidity & Sweeps (30%)
2. Multi-Timeframe Candlestick Engine (25%)
3. Order Book (L2 Depth) Imbalance & Walls (20%)
4. News & Macro Financial Sentiment (15%)
5. Self-Learning Strategy Scoring & Adaptive Sizing (10% + Gatekeeper)

Zero LLM tokens required for continuous mathematical confluence calculations.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from market_data import (
    resolve_symbol,
    get_symbol_display_name,
    get_ticker,
    get_gautam_jha_analysis,
    get_multi_timeframe_entry,
)
from orderbook_analysis import analyze_orderbook
from news_analysis import get_news_sentiment
from self_learning import LearningEngine, normalize_setup_name

logger = logging.getLogger(__name__)

CONFLUENCE_THRESHOLD = 65.0  # Minimum 65% confluence required for trade execution


class ConfluenceEngine:
    """Master Multi-Strategy Combiner and Quantitative Trade Executor."""

    def __init__(self, learning_engine: Optional[LearningEngine] = None):
        self.learning_engine = learning_engine or LearningEngine()

    def evaluate_confluence(
        self,
        symbol: str = "BTCUSD",
        learning_engine: Optional[LearningEngine] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all 5 strategy layers and calculate master confluence score.
        """
        sym = resolve_symbol(symbol)
        engine = learning_engine or self.learning_engine

        ticker = get_ticker(sym)
        mark_price = float(ticker["mark_price"] or ticker["close"])

        # ---------------- 1. Gautam Jha Liquidity Layer (30% weight) ----------------
        gj_score_long = 0.0
        gj_score_short = 0.0
        gj_detail = "Neutral / Scanning"
        gj_setup_name = None
        gj_plan = {}

        try:
            gj = get_gautam_jha_analysis(sym)
            setup = gj.get("setup", {})
            if setup and setup.get("type"):
                gj_setup_name = f"Gautam Jha {setup.get('type')}"
                direction = setup.get("direction", "NEUTRAL").upper()
                gj_plan = {
                    "entry": setup.get("entry", mark_price),
                    "sl": setup.get("sl"),
                    "tp1": setup.get("tp1"),
                    "tp2": setup.get("tp2"),
                }
                if direction in ("BUY", "LONG"):
                    gj_score_long = 30.0
                    gj_detail = f"Bullish ({setup.get('type')})"
                elif direction in ("SELL", "SHORT"):
                    gj_score_short = 30.0
                    gj_detail = f"Bearish ({setup.get('type')})"
            else:
                # Secondary DO bias
                do_bias = gj.get("do_bias", "NEUTRAL")
                if "BULLISH" in do_bias:
                    gj_score_long = 15.0
                    gj_detail = "Above Daily Open (Bullish DO bias)"
                elif "BEARISH" in do_bias:
                    gj_score_short = 15.0
                    gj_detail = "Below Daily Open (Bearish DO bias)"
        except Exception as e:
            logger.warning(f"Error evaluating GJ layer for {sym}: {e}")

        # ---------------- 2. Candlestick Patterns Layer (25% weight) ----------------
        candle_score_long = 0.0
        candle_score_short = 0.0
        candle_detail = "Neutral / No pattern"
        candle_plan = {}

        try:
            multi_tf = get_multi_timeframe_entry(sym, ["5m", "15m"])
            tf15 = multi_tf.get("timeframes", {}).get("15m", {})
            tf5 = multi_tf.get("timeframes", {}).get("5m", {})

            # 15m pattern carries 15 points, 5m carries 10 points
            if tf15.get("has_setup"):
                sig15 = tf15.get("signal", "NEUTRAL")
                pat15 = tf15.get("pattern", "Setup")
                if "BUY" in sig15 or "LONG" in sig15:
                    candle_score_long += 15.0
                    candle_detail = f"15m {pat15}"
                    candle_plan = tf15.get("trade_plan", {})
                elif "SELL" in sig15 or "SHORT" in sig15:
                    candle_score_short += 15.0
                    candle_detail = f"15m {pat15}"
                    candle_plan = tf15.get("trade_plan", {})

            if tf5.get("has_setup"):
                sig5 = tf5.get("signal", "NEUTRAL")
                pat5 = tf5.get("pattern", "Setup")
                if "BUY" in sig5 or "LONG" in sig5:
                    candle_score_long += 10.0
                    candle_detail += f" + 5m {pat5}"
                    if not candle_plan:
                        candle_plan = tf5.get("trade_plan", {})
                elif "SELL" in sig5 or "SHORT" in sig5:
                    candle_score_short += 10.0
                    candle_detail += f" + 5m {pat5}"
                    if not candle_plan:
                        candle_plan = tf5.get("trade_plan", {})
        except Exception as e:
            logger.warning(f"Error evaluating Candle layer for {sym}: {e}")

        # ---------------- 3. Order Book L2 Depth Layer (20% weight) ----------------
        book_score_long = 0.0
        book_score_short = 0.0
        book_detail = "Balanced Book"
        book_data = {}

        try:
            book_data = analyze_orderbook(sym, depth_levels=15)
            imb = book_data.get("imbalance_ratio", 0.0)
            bias = book_data.get("bias", "NEUTRAL")

            if "BULLISH" in bias:
                # Up to 20 points depending on strength
                pts = 20.0 if "STRONG" in bias else 15.0
                book_score_long = pts
                book_detail = f"Bid Stacking ({imb:+.1%} imbalance, floor supported)"
            elif "BEARISH" in bias:
                pts = 20.0 if "STRONG" in bias else 15.0
                book_score_short = pts
                book_detail = f"Ask Stacking ({imb:+.1%} imbalance, ceiling resistance)"
            else:
                book_detail = f"Neutral ({imb:+.1%} imbalance)"
        except Exception as e:
            logger.warning(f"Error evaluating Orderbook layer for {sym}: {e}")

        # ---------------- 4. News & Macro Sentiment Layer (15% weight) ----------------
        news_score_long = 0.0
        news_score_short = 0.0
        news_detail = "Neutral"
        news_data = {}

        try:
            news_data = get_news_sentiment(sym, limit=8)
            n_score = news_data.get("sentiment_score", 0.0)
            n_label = news_data.get("sentiment_label", "NEUTRAL")

            if "BULLISH" in n_label:
                pts = 15.0 if "STRONG" in n_label else 10.0
                news_score_long = pts
                news_detail = f"Bullish sentiment ({n_score:+.2f})"
            elif "BEARISH" in n_label:
                pts = 15.0 if "STRONG" in n_label else 10.0
                news_score_short = pts
                news_detail = f"Bearish sentiment ({n_score:+.2f})"
            else:
                news_detail = f"Neutral ({n_score:+.2f})"
        except Exception as e:
            logger.warning(f"Error evaluating News layer for {sym}: {e}")

        # ---------------- 5. Self-Learning Strategy Layer (10% + Gate) ----------------
        active_setup_name = gj_setup_name or "Candle_Entry"
        can_execute, self_learning_reason = engine.can_execute(active_setup_name, sym)

        learning_params = engine.get_adapted_parameters(active_setup_name, sym, default_size=1.0)
        setup_score = learning_params.get("setup_score", 1.0)
        size_mult = learning_params.get("size_multiplier", 1.0)
        sl_buffer_mult = learning_params.get("sl_buffer_multiplier", 1.0)

        learning_score_pts = 0.0
        if can_execute:
            # Setup is active or prioritized
            learning_score_pts = 10.0 if setup_score >= 1.0 else 5.0
            learning_detail = f"Approved (Score {setup_score:.2f}, Sizing {size_mult:.2f}x)"
        else:
            learning_detail = f"VETOED / Suppressed: {self_learning_reason}"

        # ---------------- Confluence Synthesis ----------------
        total_long_score = round(
            gj_score_long + candle_score_long + book_score_long + news_score_long + (learning_score_pts if gj_score_long > gj_score_short else 0),
            1,
        )
        total_short_score = round(
            gj_score_short + candle_score_short + book_score_short + news_score_short + (learning_score_pts if gj_score_short > gj_score_long else 0),
            1,
        )

        if total_long_score >= CONFLUENCE_THRESHOLD and total_long_score > total_short_score:
            bias_signal = "STRONG_BUY" if total_long_score >= 75.0 else "BUY"
            dominant_score = total_long_score
            dominant_direction = "BUY"
        elif total_short_score >= CONFLUENCE_THRESHOLD and total_short_score > total_long_score:
            bias_signal = "STRONG_SELL" if total_short_score >= 75.0 else "SELL"
            dominant_score = total_short_score
            dominant_direction = "SELL"
        else:
            bias_signal = "NEUTRAL"
            dominant_score = max(total_long_score, total_short_score)
            dominant_direction = "NEUTRAL"

        # Determine Execution Decision
        is_executable = False
        decision_reason = ""

        if not can_execute:
            decision_reason = f"Execution blocked: {self_learning_reason}"
        elif dominant_direction == "NEUTRAL" or dominant_score < CONFLUENCE_THRESHOLD:
            decision_reason = f"Confluence score ({dominant_score}%) is below minimum execution threshold ({CONFLUENCE_THRESHOLD}%)."
        elif dominant_direction == "BUY" and "STRONG_BEARISH" in news_data.get("sentiment_label", ""):
            decision_reason = "Blocked by macro risk: News sentiment is strongly bearish."
        elif dominant_direction == "SELL" and "STRONG_BULLISH" in news_data.get("sentiment_label", ""):
            decision_reason = "Blocked by macro risk: News sentiment is strongly bullish."
        else:
            is_executable = True
            decision_reason = f"High confluence confirmed ({dominant_score}% {bias_signal}). Ready for execution."

        # Compute Unified Trade Plan
        unified_plan = self._build_unified_trade_plan(
            symbol=sym,
            direction=dominant_direction,
            mark_price=mark_price,
            gj_plan=gj_plan,
            candle_plan=candle_plan,
            book_data=book_data,
            sl_buffer_mult=sl_buffer_mult,
            size_mult=size_mult,
        )

        return {
            "symbol": sym,
            "mark_price": mark_price,
            "dominant_direction": dominant_direction,  # BUY | SELL | NEUTRAL
            "bias_signal": bias_signal,                # STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
            "confluence_score": dominant_score,        # 0.0 to 100.0%
            "long_score": total_long_score,
            "short_score": total_short_score,
            "is_executable": is_executable,
            "decision_reason": decision_reason,
            "layers": {
                "gautam_jha": {"detail": gj_detail, "long_pts": gj_score_long, "short_pts": gj_score_short, "weight": 30},
                "candlestick": {"detail": candle_detail, "long_pts": candle_score_long, "short_pts": candle_score_short, "weight": 25},
                "orderbook": {"detail": book_detail, "long_pts": book_score_long, "short_pts": book_score_short, "weight": 20},
                "news": {"detail": news_detail, "long_pts": news_score_long, "short_pts": news_score_short, "weight": 15},
                "self_learning": {"detail": learning_detail, "can_execute": can_execute, "weight": 10},
            },
            "trade_plan": unified_plan,
            "timestamp": int(time.time()),
        }

    def _build_unified_trade_plan(
        self,
        symbol: str,
        direction: str,
        mark_price: float,
        gj_plan: Dict[str, Any],
        candle_plan: Dict[str, Any],
        book_data: Dict[str, Any],
        sl_buffer_mult: float = 1.0,
        size_mult: float = 1.0,
    ) -> Dict[str, Any]:
        """Synthesize unified entry, SL, TP1, and TP2 from all inputs."""
        if direction not in ("BUY", "SELL"):
            return {}

        # 1. Base SL selection from plans
        base_sl = gj_plan.get("sl") or candle_plan.get("sl")
        if not base_sl:
            # 1.0% default fallback
            base_sl = mark_price * 0.99 if direction == "BUY" else mark_price * 1.01

        # Check orderbook wall for protective SL placement
        if direction == "BUY" and book_data.get("bid_walls"):
            # If there's a strong buy wall, place stop just behind it
            wall_price = book_data["bid_walls"][0]["price"]
            if wall_price < mark_price:
                base_sl = min(base_sl, wall_price * 0.998)
        elif direction == "SELL" and book_data.get("sell_walls"):
            wall_price = book_data["sell_walls"][0]["price"]
            if wall_price > mark_price:
                base_sl = max(base_sl, wall_price * 1.002)

        # Apply Self-Learning adaptive SL buffer
        risk_dist = abs(mark_price - base_sl) * sl_buffer_mult
        effective_sl = round(mark_price - risk_dist, 2) if direction == "BUY" else round(mark_price + risk_dist, 2)

        # 2. Take Profit targets (1:1.5 TP1 and 1:2.5+ TP2)
        effective_risk = abs(mark_price - effective_sl)
        if direction == "BUY":
            tp1 = round(mark_price + (effective_risk * 1.5), 2)
            tp2 = round(mark_price + (effective_risk * 2.5), 2)
        else:
            tp1 = round(mark_price - (effective_risk * 1.5), 2)
            tp2 = round(mark_price - (effective_risk * 2.5), 2)

        # Sizing
        base_size = 0.01 if "BTC" in symbol else 1.0
        final_size = max(0.001, round(base_size * size_mult, 4))

        return {
            "symbol": symbol,
            "side": "buy" if direction == "BUY" else "sell",
            "entry": mark_price,
            "sl": effective_sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk_amount": round(effective_risk, 2),
            "size": final_size,
            "rrr": "1:1.5 / 1:2.5+",
        }

    def execute_confluence_trade(
        self,
        symbol: str,
        auto_trader,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate confluence and, if approved, immediately execute via AutoTrader!
        """
        analysis = self.evaluate_confluence(symbol, auto_trader.learning_engine)
        if not analysis["is_executable"] and not force:
            return {
                "status": "skipped",
                "reason": analysis["decision_reason"],
                "confluence": analysis,
            }

        plan = analysis["trade_plan"]
        if not plan:
            if force:
                forced_dir = "BUY" if analysis["long_score"] >= analysis["short_score"] else "SELL"
                plan = self._build_unified_trade_plan(
                    symbol=symbol,
                    direction=forced_dir,
                    mark_price=analysis["mark_price"],
                    gj_plan={},
                    candle_plan={},
                    book_data={},
                )
            else:
                return {
                    "status": "error",
                    "reason": "Could not formulate valid trade plan for execution.",
                    "confluence": analysis,
                }

        strategy_desc = f"Master Confluence ({analysis['confluence_score']}% {analysis['bias_signal']})"

        # Execute via AutoTrader
        trade_record = auto_trader.execute_trade(
            symbol=symbol,
            side=plan["side"],
            size=plan["size"],
            sl_price=plan["sl"],
            tp1_price=plan["tp1"],
            tp2_price=plan["tp2"],
            strategy=strategy_desc,
            trade_type="confluence_master",
            reason=analysis["decision_reason"],
        )

        return {
            "status": "executed",
            "trade": trade_record,
            "confluence": analysis,
        }


def format_confluence_html_report(res: Dict[str, Any]) -> str:
    """Format rich Telegram HTML report for 5-strategy confluence."""
    sym = res["symbol"]
    disp = get_symbol_display_name(sym)
    score = res["confluence_score"]
    signal = res["bias_signal"]
    direction = res["dominant_direction"]
    layers = res["layers"]
    plan = res.get("trade_plan", {})

    if "BUY" in signal:
        sig_emoji = "🟢"
        meter = ("🟩" * int(score / 10)) + ("⬜" * (10 - int(score / 10)))
    elif "SELL" in signal:
        sig_emoji = "🔴"
        meter = ("🟥" * int(score / 10)) + ("⬜" * (10 - int(score / 10)))
    else:
        sig_emoji = "⚪"
        meter = ("🟨" * int(score / 10)) + ("⬜" * (10 - int(score / 10)))

    lines = [
        f"🎯 <b>MULTI-STRATEGY MASTER CONFLUENCE</b>",
        f"<i>Gautam Jha + Candles + OrderBook + News + Self-Learning</i>\n",
        f"• <b>Contract:</b> {disp}",
        f"• <b>Current Mark Price:</b> <code>${res['mark_price']:,.2f}</code>",
        f"• <b>Consensus Score:</b> {sig_emoji} <b>{score}% {signal.replace('_', ' ')}</b>",
        f"• <b>Meter:</b> <code>{meter}</code>",
        f"• <b>Long Confluence:</b> <code>{res['long_score']}%</code> | <b>Short:</b> <code>{res['short_score']}%</code>\n",
        "<b>🔍 STRATEGY LAYERS BREAKDOWN:</b>",
        f"1. <b>Gautam Jha Liquidity (30%):</b> {layers['gautam_jha']['detail']}",
        f"2. <b>Candlestick Patterns (25%):</b> {layers['candlestick']['detail']}",
        f"3. <b>Order Book L2 Depth (20%):</b> {layers['orderbook']['detail']}",
        f"4. <b>News Sentiment (15%):</b> {layers['news']['detail']}",
        f"5. <b>Self-Learning Risk (10%):</b> {layers['self_learning']['detail']}\n",
        f"📋 <b>EXECUTION DECISION:</b>",
        f"• <b>Status:</b> {'🟢 EXECUTABLE' if res['is_executable'] else '⏸️ WAIT FOR CONFLUENCE'}",
        f"• <i>{res['decision_reason']}</i>",
    ]

    if plan:
        lines.extend([
            "",
            "⚡ <b>SUGGESTED CONFLUENCE TRADE PLAN:</b>",
            f"• <b>Action:</b> <b>{direction}</b> (Size: <code>{plan['size']}</code>)",
            f"• <b>Entry:</b> <code>${plan['entry']:,.2f}</code>",
            f"• <b>Stop Loss:</b> <code>${plan['sl']:,.2f}</code> (Risk: ${plan['risk_amount']:,.2f})",
            f"• <b>Take Profit 1:</b> <code>${plan['tp1']:,.2f}</code> (1:1.5)",
            f"• <b>Take Profit 2:</b> <code>${plan['tp2']:,.2f}</code> (1:2.5+)",
            f"• <b>Risk-to-Reward:</b> {plan['rrr']}",
        ])

    lines.append("\n💡 <i>To execute this confluence trade now, run <code>/confluence trade [SYMBOL]</code></i>")
    return "\n".join(lines)
