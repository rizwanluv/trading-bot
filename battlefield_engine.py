"""
AI Battlefield Debate & Arbiter Engine.
Orchestrates an autonomous multi-agent debate:
- Agent 1: Aggressive Bull Trader (Long Thesis)
- Agent 2: Strict Bear Risk Manager (Short / Trap Counter-Thesis)
- Agent 3: Trading Desk Arbiter / Judge (Strict Structured Verdict & Confidence Scoring)

Validates automated trades and executes on Delta Exchange when confidence threshold is met.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

# Check if google-genai SDK is available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import ccxt.async_support as ccxt_async
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

from market_data import resolve_symbol, get_ticker, DELTA_API


# ==================== Market Snapshot Extraction ====================

def get_market_snapshot_sync(symbol: str = "BTCUSD", timeframe: str = "15m") -> Dict[str, Any]:
    """
    Fetches recent candles and derives technical context (Direction, High/Low, Session VWAP, Volume).
    Uses native Delta Exchange API with ticker fallbacks.
    """
    sym = resolve_symbol(symbol)
    
    snapshot: Dict[str, Any] = {}

    # 1. Native Delta Exchange fetch
    try:
        url = f"{DELTA_API}/chart/history"
        res = requests.get(url, params={"symbol": sym, "resolution": timeframe}, timeout=8)
        if res.status_code == 200:
            data = res.json()
            result = data.get("result", {})
            closes = result.get("c", [])
            opens = result.get("o", [])
            highs = result.get("h", [])
            lows = result.get("l", [])
            volumes = result.get("v", [])

            if closes and len(closes) >= 5:
                curr_price = float(closes[-1])
                open_price = float(opens[-1]) if opens else curr_price
                recent_high = max([float(x) for x in highs[-5:]]) if highs else curr_price
                recent_low = min([float(x) for x in lows[-5:]]) if lows else curr_price
                recent_vol = sum([float(x) for x in volumes[-3:]]) if volumes else 0.0

                # VWAP approximation
                total_vol = sum([float(x) for x in volumes]) if volumes else 0.0
                if total_vol > 0:
                    typical_vol = sum([((float(h) + float(l) + float(c)) / 3.0) * float(v) for h, l, c, v in zip(highs, lows, closes, volumes)])
                    session_vwap = round(typical_vol / total_vol, 2)
                else:
                    session_vwap = curr_price

                snapshot = {
                    "symbol": sym,
                    "current_price": curr_price,
                    "candle_direction": "Green" if curr_price >= open_price else "Red",
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "session_vwap": session_vwap,
                    "recent_volume": round(recent_vol, 2)
                }
    except Exception as e:
        logger.debug(f"Native Delta snapshot fetch failed: {e}")

    # Fallback ticker if native fetch failed
    if not snapshot:
        try:
            t = get_ticker(sym)
            curr = t.get("mark_price") or t.get("close") or 0.0
            snapshot = {
                "symbol": sym,
                "current_price": curr,
                "candle_direction": "Green" if curr >= (t.get("open") or curr) else "Red",
                "recent_high": t.get("high") or curr,
                "recent_low": t.get("low") or curr,
                "session_vwap": curr,
                "recent_volume": round(t.get("volume") or 0.0, 2)
            }
        except Exception as t_err:
            logger.warning(f"Fallback ticker snapshot error: {t_err}")
            snapshot = {
                "symbol": sym,
                "current_price": 0.0,
                "candle_direction": "Neutral",
                "recent_high": 0.0,
                "recent_low": 0.0,
                "session_vwap": 0.0,
                "recent_volume": 0.0
            }

    # 2. Real-Time Macro News Sentiment Analysis
    news_info = {"score": 0.0, "label": "NEUTRAL", "catalyst": "Normal financial flow"}
    try:
        from news_analysis import get_news_sentiment
        ns = get_news_sentiment(sym, limit=5)
        news_info["score"] = float(ns.get("sentiment_score", 0.0))
        news_info["label"] = str(ns.get("sentiment_label", "NEUTRAL"))
        headlines = [h.get("title", "") for h in ns.get("headlines", [])[:2]]
        news_info["catalyst"] = " | ".join(headlines) if headlines else "Normal macro sentiment"
    except Exception as n_err:
        logger.debug(f"News sentiment snapshot notice: {n_err}")
    snapshot["news"] = news_info

    # 3. World Big Institute Track (Delta L2 Depth, Imbalance Ratio, Whale Walls & Gautam Jha Sweeps)
    inst_info = {
        "imbalance_ratio": 0.0,
        "imbalance_bias": "NEUTRAL",
        "whale_bid_wall": 0.0,
        "whale_ask_wall": 0.0,
        "pdh_swept": False,
        "pdl_swept": False,
        "daily_open": 0.0,
        "do_color": "Neutral",
    }
    try:
        from orderbook_analysis import analyze_orderbook
        ob = analyze_orderbook(sym, depth_levels=15)
        inst_info["imbalance_ratio"] = float(ob.get("imbalance_ratio", 0.0))
        inst_info["imbalance_bias"] = str(ob.get("imbalance_bias", "NEUTRAL"))
        inst_info["whale_bid_wall"] = float(ob.get("top_bid_wall", {}).get("price", 0.0) or 0.0)
        inst_info["whale_ask_wall"] = float(ob.get("top_ask_wall", {}).get("price", 0.0) or 0.0)
    except Exception as ob_err:
        logger.debug(f"Orderbook snapshot notice: {ob_err}")

    try:
        from market_data import get_gautam_jha_analysis
        gj_res = get_gautam_jha_analysis(sym)
        gj = gj_res.get("gautam_jha", {})
        inst_info["pdh_swept"] = bool(gj.get("pdh_swept"))
        inst_info["pdl_swept"] = bool(gj.get("pdl_swept"))
        inst_info["daily_open"] = float(gj.get("daily_open", 0.0) or 0.0)
        inst_info["do_color"] = str(gj.get("daily_candle_color", "Neutral"))
    except Exception as gj_err:
        logger.debug(f"Gautam Jha snapshot notice: {gj_err}")
    snapshot["institutional"] = inst_info

    return snapshot


async def get_market_snapshot(symbol: str = "BTCUSD", timeframe: str = "15m") -> Dict[str, Any]:
    """Async wrapper for get_market_snapshot_sync."""
    return get_market_snapshot_sync(symbol, timeframe=timeframe)


# ==================== Multi-Agent Debate & Arbiter Decision ====================

def _clean_json_str(raw_text: str) -> str:
    """Strip markdown code block fences and trailing text."""
    if not raw_text:
        return "{}"
    t = raw_text.strip()
    if t.startswith("```"):
        import re
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def run_battlefield_sync(
    data: dict,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Executes autonomous Bull vs. Bear debate and Judge decision synchronously.
    Returns: (bull_case, bear_case, decision_dict)
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    candidate_models = [model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    news = data.get("news", {})
    inst = data.get("institutional", {})
    news_str = f"Sentiment {news.get('label', 'NEUTRAL')} ({news.get('score', 0.0):+.2f}) - {news.get('catalyst', 'Normal financial flow')}"
    inst_str = (
        f"L2 Imbalance {inst.get('imbalance_bias', 'NEUTRAL')} (Ratio {inst.get('imbalance_ratio', 0.0):+.2f}), "
        f"Whale Walls (Bid ${inst.get('whale_bid_wall', 0.0):,.2f} / Ask ${inst.get('whale_ask_wall', 0.0):,.2f}), "
        f"Liquidity Sweeps: PDH Swept={inst.get('pdh_swept')}, PDL Swept={inst.get('pdl_swept')}, "
        f"Daily Open=${inst.get('daily_open', 0.0):,.2f} ({inst.get('do_color', 'Neutral')})"
    )

    bull_prompt = f"""You are an aggressive Bull trader.
Market technicals: Price=${data.get('current_price')}, VWAP=${data.get('session_vwap')}, Candle={data.get('candle_direction')}.
News Analysis: {news_str}.
World Big Institute Track: {inst_str}.
Give a strictly factual, 2-sentence case to go LONG right now citing price action, institutional order flow, or news catalysts."""

    # 1. Generate Bull Thesis
    bull_resp = ""
    bear_resp = ""
    judge_resp_dict: Dict[str, Any] = {
        "verdict": "NO_TRADE",
        "confidence": 5,
        "reasoning": "Neutral market equilibrium.",
        "entry_price": data.get("current_price"),
        "stop_loss": None,
        "take_profit": None
    }

    # Attempt via SDK
    if HAS_GENAI_SDK and key:
        for m_name in candidate_models:
            try:
                client = genai.Client(api_key=key)
                r_bull = client.models.generate_content(model=m_name, contents=bull_prompt)
                if r_bull and r_bull.text:
                    bull_resp = r_bull.text.strip()

                    bear_prompt = f"""You are a strict Bear risk manager.
Market technicals: Price=${data.get('current_price')}, VWAP=${data.get('session_vwap')}, Candle={data.get('candle_direction')}.
News Analysis: {news_str}.
World Big Institute Track: {inst_str}.
The Bull argues: "{bull_resp}".
Refute the Bull directly in 2 sentences. Why is this an institutional trap, liquidity sweep, or short entry?"""
                    r_bear = client.models.generate_content(model=m_name, contents=bear_prompt)
                    bear_resp = r_bear.text.strip() if r_bear and r_bear.text else "High resistance overhead."

                    judge_prompt = f"""You are the Institutional Trading Desk Arbiter.
Market Technicals: Price=${data.get('current_price')}, VWAP=${data.get('session_vwap')}, Candle={data.get('candle_direction')}
News Sentiment: {news_str}
World Big Institute Track: {inst_str}
Bull Thesis: {bull_resp}
Bear Counter: {bear_resp}

Decide strictly. Return ONLY valid JSON matching this schema:
{{
  "verdict": "BUY" | "SELL" | "NO_TRADE",
  "confidence": <integer from 1 to 10>,
  "reasoning": "<1-2 sentence core reason for the winner incorporating institutional order flow, news, and price structure>",
  "entry_price": <number or null>,
  "stop_loss": <number or null>,
  "take_profit": <number or null>
}}"""
                    r_judge = client.models.generate_content(
                        model=m_name,
                        contents=judge_prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    if r_judge and r_judge.text:
                        raw = _clean_json_str(r_judge.text)
                        judge_resp_dict = json.loads(raw)
                        return bull_resp, bear_resp, judge_resp_dict
            except Exception as e:
                logger.warning(f"SDK battlefield call failed on {m_name}: {e}. Trying fallback.")

    # Attempt via REST API
    if HAS_REQUESTS and key:
        for m_name in candidate_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={key}"
                
                # Bull
                p_bull = {"contents": [{"parts": [{"text": bull_prompt}]}], "generationConfig": {"maxOutputTokens": 200}}
                res_b = requests.post(url, json=p_bull, timeout=12)
                if res_b.status_code == 200:
                    candidates = res_b.json().get("candidates", [])
                    if candidates:
                        bull_resp = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

                if not bull_resp:
                    bull_resp = f"Price is holding above session VWAP with {news.get('label', 'NEUTRAL')} news alignment."

                # Bear
                bear_prompt = f"""You are a strict Bear risk manager.
Market technicals: Price=${data.get('current_price')}, VWAP=${data.get('session_vwap')}, Candle={data.get('candle_direction')}.
News Analysis: {news_str}.
World Big Institute Track: {inst_str}.
The Bull argues: "{bull_resp}".
Refute the Bull directly in 2 sentences. Why is this an institutional trap, liquidity sweep, or short entry?"""
                p_bear = {"contents": [{"parts": [{"text": bear_prompt}]}], "generationConfig": {"maxOutputTokens": 200}}
                res_be = requests.post(url, json=p_bear, timeout=12)
                if res_be.status_code == 200:
                    candidates = res_be.json().get("candidates", [])
                    if candidates:
                        bear_resp = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

                if not bear_resp:
                    bear_resp = f"Risk of liquidity sweep near ${data.get('recent_high', 0):,.2f}; orderbook distribution detected."

                # Judge
                judge_prompt = f"""You are the Institutional Trading Desk Arbiter.
Market Technicals: Price=${data.get('current_price')}, VWAP=${data.get('session_vwap')}, Candle={data.get('candle_direction')}
News Sentiment: {news_str}
World Big Institute Track: {inst_str}
Bull Thesis: {bull_resp}
Bear Counter: {bear_resp}

Decide strictly. Return ONLY valid JSON matching this schema:
{{
  "verdict": "BUY" | "SELL" | "NO_TRADE",
  "confidence": <integer from 1 to 10>,
  "reasoning": "<1-2 sentence core reason for the winner incorporating institutional order flow, news, and price structure>",
  "entry_price": <number or null>,
  "stop_loss": <number or null>,
  "take_profit": <number or null>
}}"""
                p_judge = {
                    "contents": [{"parts": [{"text": judge_prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "maxOutputTokens": 300
                    }
                }
                res_j = requests.post(url, json=p_judge, timeout=15)
                if res_j.status_code == 200:
                    candidates = res_j.json().get("candidates", [])
                    if candidates:
                        raw_j = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                        judge_resp_dict = json.loads(_clean_json_str(raw_j))
                        return bull_resp, bear_resp, judge_resp_dict
            except Exception as rest_err:
                logger.warning(f"REST battlefield call error with {m_name}: {rest_err}")

    # Deterministic Institutional Fallback (Synthesizing Technicals, News Sentiment, and World Big Institute Flow)
    curr = data.get("current_price", 0.0)
    vwap = data.get("session_vwap", curr)
    direction = data.get("candle_direction", "Green")
    n_score = float(news.get("score", 0.0))
    imb_ratio = float(inst.get("imbalance_ratio", 0.0))
    pdh_sw = bool(inst.get("pdh_swept"))
    pdl_sw = bool(inst.get("pdl_swept"))

    bull_score = 0
    bear_score = 0
    bull_factors = []
    bear_factors = []

    if curr > vwap and direction == "Green":
        bull_score += 4
        bull_factors.append(f"Holding firmly above session VWAP (${vwap:,.2f}) with bullish candle expansion")
    elif curr < vwap and direction == "Red":
        bear_score += 4
        bear_factors.append(f"Trading below session VWAP (${vwap:,.2f}) with bearish distribution candles")

    if n_score >= 0.15:
        bull_score += 2
        bull_factors.append(f"Bullish news sentiment catalyst ({news.get('label', 'BULLISH')})")
    elif n_score <= -0.15:
        bear_score += 2
        bear_factors.append(f"Bearish news sentiment catalyst ({news.get('label', 'BEARISH')})")

    if imb_ratio >= 0.10:
        bull_score += 2
        bull_factors.append(f"Institutional bid accumulation in Delta L2 depth (imbalance {imb_ratio:+.2f})")
    elif imb_ratio <= -0.10:
        bear_score += 2
        bear_factors.append(f"Institutional ask distribution in Delta L2 depth (imbalance {imb_ratio:+.2f})")

    if pdl_sw:
        bull_score += 1
        bull_factors.append("Sell-side liquidity swept below PDL (smart money stop run)")
    if pdh_sw:
        bear_score += 1
        bear_factors.append("Buy-side liquidity swept above PDH (overhead liquidity grabbed)")

    if bull_score >= 4 and bull_score > bear_score:
        bull_resp = f"Institutions accumulating above VWAP (${vwap:,.2f}) backed by {news.get('label', 'neutral')} news."
        bear_resp = f"Overhead resistance near ${data.get('recent_high', 0):,.2f}; watch for orderbook absorption failure."
        conf = min(10, 8 + max(0, (bull_score - bear_score - 4) // 2))
        judge_resp_dict = {
            "verdict": "BUY",
            "confidence": conf,
            "reasoning": f"Bullish institutional confluence: {'; '.join(bull_factors)}.",
            "entry_price": curr,
            "stop_loss": round(curr * 0.992, 2),
            "take_profit": round(curr * 1.018, 2),
        }
    elif bear_score >= 4 and bear_score > bull_score:
        bull_resp = f"Oversold bounce potential near support ${data.get('recent_low', 0):,.2f}."
        bear_resp = f"Institutional selling pressure below VWAP (${vwap:,.2f}) confirmed by {news.get('label', 'neutral')} news."
        conf = min(10, 8 + max(0, (bear_score - bull_score - 4) // 2))
        judge_resp_dict = {
            "verdict": "SELL",
            "confidence": conf,
            "reasoning": f"Bearish institutional distribution: {'; '.join(bear_factors)}.",
            "entry_price": curr,
            "stop_loss": round(curr * 1.008, 2),
            "take_profit": round(curr * 0.982, 2),
        }
    else:
        bull_resp = "Consolidating near key pivot; awaiting institutional breakout."
        bear_resp = "Order flow equilibrium; lack of directional institutional imbalance."
        judge_resp_dict = {
            "verdict": "NO_TRADE",
            "confidence": 4,
            "reasoning": "Mixed signals between macro news, orderbook depth, and candle structure.",
            "entry_price": curr,
            "stop_loss": None,
            "take_profit": None,
        }

    return bull_resp, bear_resp, judge_resp_dict


async def run_battlefield(
    data: dict,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """Async wrapper for run_battlefield_sync."""
    return run_battlefield_sync(data, model=model, api_key=api_key)


# ==================== CCXT Order Execution Helper ====================

def get_ccxt_exchange(api_key: Optional[str] = None, api_secret: Optional[str] = None):
    """Instantiate CCXT Delta Exchange client if credentials are provided."""
    if not HAS_CCXT:
        return None
    key = api_key or os.getenv("EXCHANGE_API_KEY") or os.getenv("DELTA_API_KEY")
    secret = api_secret or os.getenv("EXCHANGE_SECRET_KEY") or os.getenv("DELTA_API_SECRET")
    if not key or not secret:
        return None
    try:
        import ccxt.async_support as ccxt_async
        return ccxt_async.delta({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
        })
    except Exception as e:
        logger.warning(f"Failed to instantiate CCXT delta client: {e}")
        return None


async def execute_trade_order_ccxt(symbol: str, side: str, amount: float) -> Dict[str, Any]:
    """Places market order directly on exchange using CCXT async client."""
    exchange = get_ccxt_exchange()
    if not exchange:
        return {"error": "CCXT exchange credentials not configured"}
    try:
        order = await exchange.create_order(
            symbol=symbol,
            type="market",
            side="buy" if side.upper() == "BUY" else "sell",
            amount=amount
        )
        return order
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            await exchange.close()
        except Exception:
            pass


# ==================== High-Level Evaluation & Trade Execution Gate ====================

async def evaluate_and_execute_battlefield(
    symbol: str,
    auto_trader_instance,
    timeframe: str = "15m",
    min_confidence: int = 8,
    force: bool = False,
    broadcast_channel_id: Optional[str] = None,
    bot_instance=None
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Fetches market snapshot.
    2. Runs Bull vs. Bear debate & Judge verdict.
    3. If confidence >= min_confidence and verdict in (BUY, SELL), executes trade on Delta Exchange.
    4. Dispatches real-time battlefield status to Telegram.
    """
    sym = resolve_symbol(symbol)
    data = await get_market_snapshot(sym, timeframe=timeframe)
    bull_case, bear_case, decision = await run_battlefield(data)

    verdict = decision.get("verdict", "NO_TRADE")
    confidence = int(decision.get("confidence") or 0)
    sl = decision.get("stop_loss")
    tp = decision.get("take_profit")
    reasoning = decision.get("reasoning", "No reason provided.")

    action_status = "⏸️ WAITING (No trade executed)"
    order_res = {}

    # Execution Gate: Bull vs Bear Debate Arbiter Validation
    news_obj = data.get("news", {})
    inst_obj = data.get("institutional", {})
    should_trade = force or (verdict in ["BUY", "SELL"] and confidence >= min_confidence)
    if should_trade and verdict in ["BUY", "SELL"]:
        side = "buy" if verdict == "BUY" else "sell"
        try:
            executed_pos = auto_trader_instance.execute_trade(
                symbol=sym,
                side=side,
                sl_price=sl,
                tp1_price=tp,
                strategy=f"Battlefield Arbiter ({verdict})",
                trade_type="battlefield",
                reason=f"Debate Confidence {confidence}/10: {reasoning}",
                confidence=f"{confidence}/10",
                news_sentiment=news_obj,
                institutional_flow=inst_obj,
            )
            order_res = {"executed": True, "position": executed_pos}
            sz = executed_pos.get("size")
            ent = executed_pos.get("entry_price", 0)
            t_mode = executed_pos.get("mode", "paper").upper()
            action_status = f"⚡ <b>EXECUTED {verdict} ORDER ({t_mode})</b>\n• <i>Size:</i> <code>{sz}</code> | <i>Entry:</i> <code>${ent:,.2f}</code> | <i>SL:</i> <code>${executed_pos.get('sl_price', 0):,.2f}</code> | <i>TP:</i> <code>${executed_pos.get('tp1_price', 0):,.2f}</code>"
        except Exception as exec_err:
            order_res = {"executed": False, "error": str(exec_err)}
            action_status = f"⚠️ <b>Execution Notice:</b> <code>{exec_err}</code>"

    # Format battle log HTML message
    side_emoji = "🟢" if verdict == "BUY" else ("🔴" if verdict == "SELL" else "⚪")
    pdh_str = "Swept 🎯" if inst_obj.get("pdh_swept") else "Intact"
    pdl_str = "Swept 🎯" if inst_obj.get("pdl_swept") else "Intact"
    whale_bid = inst_obj.get("whale_bid_wall", 0.0)
    whale_ask = inst_obj.get("whale_ask_wall", 0.0)

    message = (
        f"⚔️ <b>AUTO-BATTLEFIELD ARBITER: #{sym}</b>\n\n"
        f"🟢 <b>Bull Thesis:</b>\n<i>{bull_case}</i>\n\n"
        f"🔴 <b>Bear Counter:</b>\n<i>{bear_case}</i>\n\n"
        f"⚖️ <b>Judge Decision:</b> <code>{verdict}</code> {side_emoji} (Confidence: <b>{confidence}/10</b>)\n"
        f"• <b>Reason:</b> {reasoning}\n"
        f"• <b>SL / TP:</b> <code>{f'${sl:,.2f}' if sl else 'Auto ATR'}</code> / <code>{f'${tp:,.2f}' if tp else 'Auto 1:2.0'}</code>\n\n"
        f"📰 <b>News Analysis:</b> <code>{news_obj.get('label', 'NEUTRAL')} ({news_obj.get('score', 0.0):+.2f})</code>\n"
        f"   └ <i>{news_obj.get('catalyst', 'Normal flow')}</i>\n\n"
        f"🏛️ <b>World Big Institute Track:</b>\n"
        f"• <b>Delta L2 Imbalance:</b> <code>{inst_obj.get('imbalance_bias', 'NEUTRAL')} ({inst_obj.get('imbalance_ratio', 0.0):+.2f})</code>\n"
        f"• <b>Whale Walls:</b> Bid <code>${whale_bid:,.2f}</code> | Ask <code>${whale_ask:,.2f}</code>\n"
        f"• <b>Liquidity Sweeps:</b> PDH: <b>{pdh_str}</b> | PDL: <b>{pdl_str}</b>\n\n"
        f"{action_status}"
    )

    # Broadcast to Telegram if bot_instance is available
    if bot_instance:
        recipients = set()
        if broadcast_channel_id:
            recipients.add(broadcast_channel_id)
        if hasattr(auto_trader_instance, "subscribers") and auto_trader_instance.subscribers:
            recipients.update(auto_trader_instance.subscribers)

        for cid in recipients:
            try:
                await bot_instance.send_message(
                    chat_id=cid,
                    text=message,
                    parse_mode="HTML"
                )
            except Exception as send_err:
                logger.warning(f"Failed to send battlefield alert to {cid}: {send_err}")

    return {
        "symbol": sym,
        "market_data": data,
        "bull_thesis": bull_case,
        "bear_counter": bear_case,
        "decision": decision,
        "action_status": action_status,
        "order_result": order_res,
        "formatted_message": message
    }
