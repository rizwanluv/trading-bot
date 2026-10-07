"""
Auto Trading Engine supporting both Simulated Paper Trading and Live Delta Exchange Execution.
Integrates with Gautam Jha Liquidity Strategy and Multi-Timeframe Candlestick Engine.
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from delta_client import DeltaClient
from market_data import get_ticker, resolve_symbol, get_multi_timeframe_entry, get_gautam_jha_analysis
from self_learning import LearningEngine
from confluence_engine import ConfluenceEngine, format_confluence_html_report
from orderbook_analysis import analyze_orderbook, format_orderbook_html_report
from news_analysis import get_news_sentiment, format_news_html_report
from amd_scalper import (
    analyze_amd_scalp,
    format_amd_scalp_html_report,
    calculate_risk_managed_plan,
)

logger = logging.getLogger(__name__)


class AutoTrader:
    """Automated trade execution engine with risk management and state persistence."""

    def __init__(
        self,
        store_file: str = "autotrade_store.json",
        delta_client: Optional[DeltaClient] = None,
        mode: Optional[str] = None,
        default_size: Optional[float] = None,
        learning_engine: Optional[LearningEngine] = None,
        confluence_engine: Optional[ConfluenceEngine] = None,
    ):
        self.store_file = store_file
        self.delta_client = delta_client or DeltaClient()
        self.learning_engine = learning_engine or LearningEngine()
        self.confluence_engine = confluence_engine or ConfluenceEngine(learning_engine=self.learning_engine)
        init_size = float(default_size or os.environ.get("DEFAULT_ORDER_SIZE", 1))
        self.data: Dict[str, Any] = {
            "enabled": False,
            "mode": (mode or os.environ.get("TRADING_MODE", "paper")).lower(),
            "symbols": ["BTCUSD", "XAUTUSD"],
            "timeframes": ["1m", "5m", "15m"],
            "default_size": init_size,
            "lot_size": init_size,
            "symbol_lot_sizes": {},
            "lot_size_mode": "custom",  # 'custom' strictly executes user's configured lot size
            "max_open_positions": 5,
            "risk_per_trade_pct": 0.015,  # 1.5% auto capital risk per trade
            "allow_multiple_per_symbol": False,
            "auto_tp_sl": True,
            "auto_breakeven": False,      # Auto-shift SL to Breakeven when TP1 is reached
            "trailing_sl": False,         # Dynamic Trailing Stop-Loss
            "trailing_pct": 0.01,         # 1.0% trail distance
            "trailing_activation_pct": 0.012,  # 1.2% profit before trail kicks in
            "paper_balance": 10000.0,  # $10,000 initial virtual capital
            "battlefield_validation": True,    # Run Bull vs Bear debate before automated execution
            "battlefield_min_confidence": 8,   # Minimum Arbiter confidence (1-10) to execute
            "backup_channel_id": os.getenv("BACKUP_CHANNEL_ID", ""),
            "positions": {},           # position_id -> position dict
            "history": [],             # list of closed trades
            "stats": {"wins": 0, "losses": 0, "total_pnl": 0.0},
            "subscribers": [],
        }
        self.load()
        if mode:
            self.data["mode"] = mode.lower()
        if default_size is not None:
            self.data["default_size"] = float(default_size)
            self.data["lot_size"] = float(default_size)
            self.data["lot_size_mode"] = "custom"
        if "lot_size" not in self.data:
            self.data["lot_size"] = float(self.data.get("default_size", 1.0))
        if "lot_size_mode" not in self.data:
            self.data["lot_size_mode"] = "custom"
        if "symbol_lot_sizes" not in self.data:
            self.data["symbol_lot_sizes"] = {}
        if "battlefield_validation" not in self.data:
            self.data["battlefield_validation"] = True
        if "battlefield_min_confidence" not in self.data:
            self.data["battlefield_min_confidence"] = 8
        if "backup_channel_id" not in self.data:
            self.data["backup_channel_id"] = os.getenv("BACKUP_CHANNEL_ID", "")

        # Bootstrap self-learning engine if it has no trades but history is present
        if self.data.get("history") and self.learning_engine.overall.total_trades == 0:
            self.learning_engine.bootstrap_from_history(self.data["history"])

    def load(self):
        """Load state from disk."""
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.data.update(saved)
            except Exception as e:
                logger.error(f"Failed to load {self.store_file}: {e}")

    def save(self):
        """Save state to disk."""
        try:
            with open(self.store_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {self.store_file}: {e}")

    # ==================== Configuration & Settings ====================

    @property
    def is_enabled(self) -> bool:
        return bool(self.data.get("enabled", False))

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    @enabled.setter
    def enabled(self, val: bool):
        self.set_enabled(val)

    def set_enabled(self, val: bool):
        self.data["enabled"] = bool(val)
        self.save()

    @property
    def mode(self) -> str:
        return self.data.get("mode", "paper").lower()

    @mode.setter
    def mode(self, val: str):
        self.set_mode(val)

    def set_mode(self, mode: str) -> str:
        """Set trading mode ('paper' or 'live')."""
        clean_mode = mode.lower().strip()
        if clean_mode not in ("paper", "live"):
            raise ValueError("Mode must be either 'paper' or 'live'.")
        if clean_mode == "live" and not self.delta_client.is_configured():
            raise ValueError("Cannot switch to 'live' mode: Delta API Key & Secret are not configured.")
        self.data["mode"] = clean_mode
        self.save()
        return clean_mode

    @property
    def balance(self) -> float:
        return float(self.data.get("paper_balance", 10000.0))

    @balance.setter
    def balance(self, val: float):
        self.data["paper_balance"] = round(float(val), 2)
        self.save()

    @property
    def subscribers(self) -> List[int]:
        return self.data.setdefault("subscribers", [])

    def add_subscriber(self, chat_id: int):
        """Register a Telegram chat ID to receive trade execution and exit alerts."""
        subs = self.data.setdefault("subscribers", [])
        if chat_id not in subs:
            subs.append(chat_id)
            self.save()

    def remove_subscriber(self, chat_id: int):
        """Unregister a Telegram chat ID from alerts."""
        subs = self.data.setdefault("subscribers", [])
        if chat_id in subs:
            subs.remove(chat_id)
            self.save()

    @property
    def positions(self) -> List[Dict[str, Any]]:
        return list(self.data.get("positions", {}).values())

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return list(self.data.get("positions", {}).values())

    def toggle(
        self,
        enabled: Optional[bool] = None,
        symbol: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Toggle auto-trading ON/OFF and configure parameters."""
        if enabled is not None:
            self.data["enabled"] = enabled

        if mode is not None:
            self.set_mode(mode)

        if symbol:
            sym = resolve_symbol(symbol)
            symbols = self.data.setdefault("symbols", [])
            if enabled and sym not in symbols:
                symbols.append(sym)
            elif not enabled and sym in symbols:
                symbols.remove(sym)

        self.save()
        return {
            "enabled": self.data["enabled"],
            "mode": self.data["mode"],
            "symbols": self.data["symbols"],
            "size": self.data["default_size"],
            "max_open_positions": self.data.get("max_open_positions", 5),
            "risk_per_trade_pct": self.data.get("risk_per_trade_pct", 0.015),
        }

    def set_max_positions(self, count: int) -> int:
        """Set maximum allowable concurrent open positions."""
        c = max(1, min(int(count), 20))
        self.data["max_open_positions"] = c
        self.save()
        return c

    def set_risk_per_trade(self, risk_pct: float) -> float:
        """Set capital risk percentage per trade (e.g. 0.015 for 1.5%)."""
        val = max(0.001, min(float(risk_pct), 0.10))
        self.data["risk_per_trade_pct"] = val
        self.save()
        return val

    def set_allow_multiple_per_symbol(self, allow: bool) -> bool:
        """Enable or disable multiple concurrent positions on the same symbol."""
        self.data["allow_multiple_per_symbol"] = bool(allow)
        self.save()
        return self.data["allow_multiple_per_symbol"]

    def set_auto_breakeven(self, enabled: bool) -> bool:
        """Enable or disable moving Stop Loss to Breakeven when TP1 is hit."""
        self.data["auto_breakeven"] = bool(enabled)
        self.save()
        return self.data["auto_breakeven"]

    def set_trailing_sl(self, enabled: bool, pct: Optional[float] = None) -> Dict[str, Any]:
        """Configure dynamic trailing stop-loss."""
        self.data["trailing_sl"] = bool(enabled)
        if pct is not None:
            self.data["trailing_pct"] = max(0.002, min(float(pct), 0.10))
        self.save()
        return {
            "trailing_sl": self.data["trailing_sl"],
            "trailing_pct": self.data.get("trailing_pct", 0.01),
        }

    def set_lot_size(self, size: float, symbol: Optional[str] = None) -> float:
        """
        Set auto-trade lot size.
        If symbol is provided, sets a per-symbol override (e.g. BTCUSD -> 0.01).
        If symbol is None, sets the global default lot size.
        """
        val = max(0.0001, round(float(size), 4))
        if symbol:
            sym = resolve_symbol(symbol)
            sym_sizes = self.data.setdefault("symbol_lot_sizes", {})
            sym_sizes[sym] = val
        else:
            self.data["lot_size"] = val
            self.data["default_size"] = val
        self.data["lot_size_mode"] = "custom"
        self.save()
        return val

    def get_lot_size(self, symbol: Optional[str] = None) -> float:
        """Get configured lot size for a specific symbol or the global default."""
        if symbol:
            sym = resolve_symbol(symbol)
            sym_sizes = self.data.get("symbol_lot_sizes", {})
            if sym in sym_sizes:
                return float(sym_sizes[sym])
        return float(self.data.get("lot_size", self.data.get("default_size", 1.0)))

    def get_effective_lot_size(self, symbol: str) -> float:
        """Get effective active lot size to execute for the given symbol."""
        return self.get_lot_size(symbol)

    def set_lot_size_mode(self, mode: str) -> str:
        """
        Set lot size mode:
        'custom': strictly follows user-configured lot size (default)
        'risk_pct': dynamically calculates lot size from risk percentage and SL distance
        """
        clean = mode.lower().strip()
        if clean in ("custom", "fixed"):
            mode_val = "custom"
        elif clean in ("risk_pct", "risk", "auto"):
            mode_val = "risk_pct"
        else:
            raise ValueError("Lot size mode must be 'custom' or 'risk_pct'.")
        self.data["lot_size_mode"] = mode_val
        self.save()
        return mode_val

    def remove_symbol_lot_size(self, symbol: str) -> bool:
        """Remove a per-symbol lot size override, reverting back to global lot size."""
        sym = resolve_symbol(symbol)
        sym_sizes = self.data.get("symbol_lot_sizes", {})
        if sym in sym_sizes:
            del sym_sizes[sym]
            self.save()
            return True
        return False

    def calculate_risk_position_size(
        self,
        symbol: str,
        entry_price: float,
        sl_price: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ) -> float:
        """Calculate position size according to capital risk management."""
        sym = resolve_symbol(symbol)
        pct = float(risk_pct if risk_pct is not None else self.data.get("risk_per_trade_pct", 0.015))
        balance = self.balance if self.mode == "paper" else 10000.0
        risk_capital = balance * pct
        if sl_price is None:
            dist = entry_price * 0.01
        else:
            dist = abs(entry_price - sl_price)

        if dist <= 0:
            return float(self.data.get("default_size", 1.0))

        raw_size = risk_capital / dist
        if "BTC" in sym:
            return max(0.001, min(round(raw_size, 3), 5.0))
        elif "XAU" in sym or "GOLD" in sym:
            return max(0.01, min(round(raw_size, 2), 25.0))
        elif "ETH" in sym:
            return max(0.01, min(round(raw_size, 2), 50.0))
        else:
            return max(0.1, min(round(raw_size, 1), 100.0))

    # ==================== Trade Execution ====================

    def execute_trade(
        self,
        symbol: str,
        side: str,
        size: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        tp2: Optional[float] = None,
        strategy: Optional[str] = None,
        mode: Optional[str] = None,
        sl_price: Optional[float] = None,
        tp1_price: Optional[float] = None,
        tp2_price: Optional[float] = None,
        trade_type: str = "manual",
        reason: str = "",
        confidence: Optional[Any] = None,
        news_sentiment: Optional[Any] = None,
        institutional_flow: Optional[Any] = None,
        learning_notes: Optional[str] = None,
        prior_trades_analyzed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute trade either in PAPER mode (simulation) or LIVE mode (Delta Exchange).
        """
        sym = resolve_symbol(symbol)
        side_clean = side.upper().strip()
        if side_clean not in ("BUY", "SELL", "LONG", "SHORT"):
            raise ValueError("Side must be BUY, SELL, LONG, or SHORT.")

        order_side = "buy" if side_clean in ("BUY", "LONG") else "sell"
        trade_mode = (mode or self.data.get("mode", "paper")).lower()

        # Check maximum open positions limit
        max_positions = int(self.data.get("max_open_positions", 5))
        if len(self.data.get("positions", {})) >= max_positions:
            raise ValueError(f"Max open positions limit ({max_positions}) reached. Close an active position before opening a new trade.")

        # Check existing position for this symbol
        allow_multi_sym = self.data.get("allow_multiple_per_symbol", False)
        for pos in self.data.get("positions", {}).values():
            if pos.get("symbol") == sym and not allow_multi_sym:
                raise ValueError(f"An open position already exists for {sym}. Close it first before opening a new trade.")
            elif pos.get("symbol") == sym and allow_multi_sym:
                if pos.get("side") == order_side and pos.get("strategy") == (strategy or reason or trade_type):
                    raise ValueError(f"An identical position already exists for {sym}.")

        # Get current price
        ticker = get_ticker(sym)
        mark_price = float(ticker["mark_price"] or ticker["close"])

        # Resolve SL and TP parameters
        effective_sl = sl_price if sl_price is not None else sl
        effective_tp1 = tp1_price if tp1_price is not None else tp
        effective_tp2 = tp2_price if tp2_price is not None else tp2

        # Auto-calculate default SL/TP if not provided (1.0% SL, 1.5% TP)
        if effective_sl is None:
            effective_sl = mark_price * 0.99 if order_side == "buy" else mark_price * 1.01
        if effective_tp1 is None:
            effective_tp1 = mark_price * 1.015 if order_side == "buy" else mark_price * 0.985
        if effective_tp2 is None:
            effective_tp2 = effective_tp1 * 1.01 if order_side == "buy" else effective_tp1 * 0.99

        # Determine order size:
        # If explicitly passed > 0, use provided size.
        # Otherwise, if lot_size_mode is "custom" (default), use user's configured lot size.
        # If lot_size_mode is "risk_pct", calculate dynamic risk-based size.
        if size is not None and float(size) > 0:
            order_size = float(size)
        elif self.data.get("lot_size_mode", "custom") == "custom":
            order_size = self.get_effective_lot_size(sym)
        else:
            order_size = self.calculate_risk_position_size(sym, mark_price, effective_sl)

        risk_amount = abs(mark_price - effective_sl)
        reward_amount = abs(effective_tp1 - mark_price)
        rrr = f"1:{(reward_amount / risk_amount):.2f}" if risk_amount > 0 else "1:1.5"

        strat_name = strategy or reason or trade_type
        position_id = f"{sym}_{int(time.time() * 1000)}"

        # News analysis summary string
        if isinstance(news_sentiment, dict):
            news_desc = f"{news_sentiment.get('label', 'NEUTRAL')} ({news_sentiment.get('score', 0.0):+.2f})"
        elif news_sentiment:
            news_desc = str(news_sentiment)
        else:
            news_desc = "Neutral (0.00)"

        # Institutional flow summary string
        if isinstance(institutional_flow, dict):
            imb_ratio = institutional_flow.get("imbalance_ratio", 0.0)
            imb_bias = institutional_flow.get("imbalance_bias", "NEUTRAL")
            inst_desc = f"Imbalance {imb_bias} ({imb_ratio:+.2f})"
        elif institutional_flow:
            inst_desc = str(institutional_flow)
        else:
            inst_desc = "Delta L2 Depth Balanced"

        position_record = {
            "id": position_id,
            "position_id": position_id,
            "symbol": sym,
            "mode": trade_mode,
            "side": order_side,
            "size": order_size,
            "entry_price": mark_price,
            "sl": round(effective_sl, 2),
            "sl_price": round(effective_sl, 2),
            "tp1": round(effective_tp1, 2),
            "tp1_price": round(effective_tp1, 2),
            "tp2": round(effective_tp2, 2),
            "tp2_price": round(effective_tp2, 2),
            "rrr": rrr,
            "strategy": strat_name,
            "reason": reason or strat_name,
            "confidence": str(confidence) if confidence else "8/10",
            "news_summary": news_desc,
            "institutional_flow": inst_desc,
            "learning_notes": learning_notes or "Prior trades evaluated & risk filters passed",
            "status": "open",
            "opened_at": int(time.time()),
            "order_id": None,
        }

        if trade_mode == "live":
            if not self.delta_client.is_configured():
                raise ValueError("Delta API credentials are not set. Cannot place live orders.")

            order_res = self.delta_client.place_order(
                symbol=sym,
                size=int(order_size) if order_size >= 1 else order_size,
                side=order_side,
                order_type="market_order",
                stop_loss=effective_sl,
                take_profit=effective_tp1,
            )
            position_record["order_id"] = order_res.get("id")
            position_record["live_response"] = order_res

        # Record position
        self.data.setdefault("positions", {})[position_id] = position_record
        self.save()

        return position_record

    def close_position(
        self,
        symbol_or_id: str,
        reason: str = "Manual Close",
        exit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Close an active position by symbol or position_id and record trade result."""
        # Find position by ID or Symbol
        found_key = None
        pos = None

        if symbol_or_id in self.data.get("positions", {}):
            found_key = symbol_or_id
            pos = self.data["positions"][found_key]
        else:
            resolved_sym = None
            try:
                resolved_sym = resolve_symbol(symbol_or_id)
            except Exception:
                pass

            for k, p in self.data.get("positions", {}).items():
                if p.get("position_id") == symbol_or_id or p.get("id") == symbol_or_id:
                    found_key = k
                    pos = p
                    break
                if resolved_sym and p.get("symbol") == resolved_sym:
                    found_key = k
                    pos = p
                    break

        if not pos or not found_key:
            return {"status": "error", "error": f"No active position found for {symbol_or_id}."}

        sym = pos["symbol"]
        if exit_price is None:
            ticker = get_ticker(sym)
            exit_price = float(ticker["mark_price"] or ticker["close"])
        else:
            exit_price = float(exit_price)

        entry_price = float(pos["entry_price"])
        size = float(pos["size"])
        side = pos["side"].lower()

        # Calculate PnL
        if side in ("buy", "long"):
            pnl = (exit_price - entry_price) * size
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        else:
            pnl = (entry_price - exit_price) * size
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0 if entry_price > 0 else 0.0

        is_win = pnl > 0
        trade_mode = pos.get("mode", "paper")

        # Execute on Delta if live
        if trade_mode == "live" and self.delta_client.is_configured():
            try:
                self.delta_client.close_position(sym)
            except Exception as e:
                logger.warning(f"Error sending market close to Delta for {sym}: {e}")

        # Update paper balance if paper
        if trade_mode == "paper":
            self.data["paper_balance"] = round(self.data.get("paper_balance", 10000.0) + pnl, 2)

        # Record in history
        history_item = {
            "id": len(self.data["history"]) + 1,
            "position_id": pos.get("position_id", found_key),
            "status": "closed",
            "symbol": sym,
            "mode": trade_mode,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "is_win": is_win,
            "strategy": pos.get("strategy", "Auto Execution"),
            "opened_at": pos.get("opened_at", int(time.time())),
            "closed_at": int(time.time()),
        }
        self.data["history"].append(history_item)

        # Update stats
        stats = self.data.setdefault("stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
        if is_win:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["total_pnl"] = round(stats.get("total_pnl", 0.0) + pnl, 2)

        # Remove position
        del self.data["positions"][found_key]
        self.save()

        # Update self-learning engine (0 tokens, deterministic local optimization)
        try:
            self.learning_engine.record_trade_outcome(history_item)
            lr_sum = self.learning_engine.get_learning_summary()
            if is_win:
                learned_lesson = f"Win (+${pnl:,.2f}) on {history_item.get('strategy')}. Win rate updated to {lr_sum.get('overall_win_rate', 50.0)}%. Setup reinforced."
            else:
                learned_lesson = f"Loss (-${abs(pnl):,.2f}) on {history_item.get('strategy')}. SL buffer expanded & risk multiplier adjusted."
            history_item["learned_lesson"] = learned_lesson
        except Exception as e:
            logger.error(f"Error updating self-learning engine on trade close: {e}")
            history_item["learned_lesson"] = "Trade outcome evaluated and stored."

        try:
            from memory_manager import memory_manager
            memory_manager.record_trade_reflection(history_item)
        except Exception as m_err:
            logger.debug(f"Memory manager trade reflection notice: {m_err}")

        return history_item

    def close_all_positions(self, reason: str = "Manual Close All") -> List[Dict[str, Any]]:
        """Close all currently active positions."""
        results = []
        for pos_key in list(self.data.get("positions", {}).keys()):
            try:
                closed = self.close_position(pos_key, reason=reason)
                if closed.get("status") == "closed":
                    results.append(closed)
            except Exception as e:
                logger.error(f"Error closing position {pos_key}: {e}")
        return results

    def analyze_prior_trades(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze prior trade outcomes from history before taking next trade:
        - Reviews recent win rate, win/loss streak, and exit triggers
        - Synthesizes dynamic risk parameter adaptations
        """
        history = self.data.get("history", [])
        if symbol:
            sym_clean = resolve_symbol(symbol)
            trades = [t for t in history if t.get("symbol") == sym_clean]
        else:
            trades = history

        total_trades = len(trades)
        window = trades[-5:] if total_trades >= 5 else trades
        wins = sum(1 for t in window if t.get("is_win", False))
        losses = len(window) - wins
        win_rate = round((wins / len(window) * 100.0), 1) if window else 0.0

        last_trade = window[-1] if window else None
        if last_trade:
            res_str = "WIN" if last_trade.get("is_win") else "LOSS"
            last_desc = f"{res_str} ({last_trade.get('pnl', 0):+,.2f}) [{last_trade.get('exit_reason', '')}]"
        else:
            last_desc = "None (Initial Trade)"

        if losses >= 3:
            advice = "Loss streak detected; throttling size to 0.7x and widening SL buffer."
            size_mod = 0.7
            sl_buffer = 1.15
        elif wins >= 3:
            advice = "Strong win streak; setup prioritized with 1.15x multiplier."
            size_mod = 1.15
            sl_buffer = 1.0
        else:
            advice = "Stable equilibrium; standard risk parameters applied."
            size_mod = 1.0
            sl_buffer = 1.0

        return {
            "total_analyzed": total_trades,
            "window_size": len(window),
            "recent_wins": wins,
            "recent_losses": losses,
            "recent_win_rate": win_rate,
            "last_trade_summary": last_desc,
            "adaptive_advice": advice,
            "size_modifier": size_mod,
            "sl_buffer": sl_buffer,
            "summary_text": f"Prior {len(window)} trades: {wins}W/{losses}L ({win_rate}% WR) | Next trade: {advice}",
        }

    def _is_vetoed_by_battlefield(self, symbol: str, side: str) -> bool:
        """
        Runs Bull vs Bear debate arbiter before execution.
        If the Arbiter rules decisively against the trade direction (confidence >= 7),
        vetoes the entry to protect capital.
        """
        if not self.data.get("battlefield_validation", True):
            return False
        try:
            from battlefield_engine import get_market_snapshot_sync, run_battlefield_sync
            data = get_market_snapshot_sync(symbol, timeframe="15m")
            _, _, decision = run_battlefield_sync(data)
            verdict = decision.get("verdict", "NO_TRADE")
            confidence = int(decision.get("confidence") or 0)
            order_side = side.lower()
            if order_side in ("buy", "long") and verdict == "SELL" and confidence >= 7:
                logger.info(f"⚔️ Battlefield Arbiter vetoed {symbol} LONG entry (Bear thesis won: Conf {confidence}/10 - {decision.get('reasoning')})")
                return True
            elif order_side in ("sell", "short") and verdict == "BUY" and confidence >= 7:
                logger.info(f"⚔️ Battlefield Arbiter vetoed {symbol} SHORT entry (Bull thesis won: Conf {confidence}/10 - {decision.get('reasoning')})")
                return True
        except Exception as e:
            logger.debug(f"Battlefield veto evaluation notice: {e}")
        return False

    # ==================== Automated Background Scanners ====================

    def check_signals_and_auto_execute(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Analyze symbol for high-probability Gautam Jha & candle setups and auto-execute if enabled.
        """
        if not self.is_enabled:
            return None

        sym = resolve_symbol(symbol)
        if sym not in self.data.get("symbols", []):
            return None

        # Check if already in trade for this symbol
        allow_multi_sym = self.data.get("allow_multiple_per_symbol", False)
        if not allow_multi_sym:
            for pos in self.data.get("positions", {}).values():
                if pos.get("symbol") == sym:
                    return None

        max_positions = int(self.data.get("max_open_positions", 5))
        if len(self.data.get("positions", {})) >= max_positions:
            return None

        # Continuous Self-Learning: Evaluate previous trade outcomes before next trade
        prior_analysis = self.analyze_prior_trades(sym)
        learning_note = prior_analysis.get("summary_text", "Prior trades evaluated")

        # News Analysis for Auto Trade
        news_data = None
        try:
            news_data = get_news_sentiment(sym, limit=5)
        except Exception:
            news_data = {"score": 0.0, "label": "NEUTRAL", "catalyst": "Normal financial flow"}

        # World Big Institute Tracking (Delta L2 Depth Imbalance & Gautam Jha Liquidity)
        inst_flow = None
        try:
            ob = analyze_orderbook(sym, depth_levels=15)
            inst_flow = {
                "imbalance_ratio": ob.get("imbalance_ratio", 0.0),
                "imbalance_bias": ob.get("imbalance_bias", "NEUTRAL"),
                "whale_bid_wall": ob.get("top_bid_wall", {}).get("price", 0.0),
                "whale_ask_wall": ob.get("top_ask_wall", {}).get("price", 0.0),
            }
        except Exception:
            inst_flow = {"imbalance_ratio": 0.0, "imbalance_bias": "NEUTRAL"}

        # 1. High-Probability AMD Scalp Strategy (1m, 5m, 15m Multi-Timeframe)
        try:
            amd_res = analyze_amd_scalp(
                sym,
                balance=self.balance,
                risk_pct=float(self.data.get("risk_per_trade_pct", 0.015)),
            )
            if amd_res.get("has_setup") and amd_res.get("is_executable") and amd_res.get("trade_plan"):
                plan = amd_res["trade_plan"]
                if self._is_vetoed_by_battlefield(sym, plan["side"]):
                    logger.info(f"AMD Scalp entry vetoed by Battlefield Arbiter for {sym}")
                else:
                    strat_desc = f"AMD Scalp ({amd_res.get('phase', 'DISTRIBUTION')} {plan['side'].upper()})"
                    can_run, lr_reason = self.learning_engine.can_execute(strat_desc, sym)
                    if can_run:
                        base_sz = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else plan["suggested_size"]
                        adapted_params = self.learning_engine.get_adapted_parameters(
                            strat_desc, sym, default_size=base_sz
                        )
                        final_size = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else adapted_params.get("recommended_size", plan["suggested_size"])
                        sl_mult = adapted_params.get("sl_buffer_multiplier", 1.0)
                        final_sl = plan["sl"]
                        if sl_mult > 1.0:
                            dist = abs(plan["entry"] - final_sl)
                            if plan["side"] == "buy":
                                final_sl = round(plan["entry"] - (dist * sl_mult), 2)
                            else:
                                final_sl = round(plan["entry"] + (dist * sl_mult), 2)

                        return self.execute_trade(
                            symbol=sym,
                            side=plan["side"],
                            size=final_size,
                            sl_price=final_sl,
                            tp1_price=plan["tp1"],
                            tp2_price=plan["tp2"],
                            strategy=strat_desc,
                            trade_type="amd_scalp",
                            reason=f"1m/5m/15m AMD Trigger ({plan.get('rrr', '1:2')})",
                            confidence=f"{int(amd_res.get('confidence', 8))}/10",
                            news_sentiment=news_data,
                            institutional_flow=inst_flow,
                            learning_notes=learning_note,
                        )
                    else:
                        logger.info(f"Self-Learning Filter skipped {sym} ({strat_desc}): {lr_reason}")
        except Exception as e:
            logger.warning(f"Error evaluating AMD Scalp for {sym}: {e}")

        # 2. Master Multi-Strategy Confluence Check (Every Strategy Combined)
        try:
            conf = self.confluence_engine.evaluate_confluence(sym, self.learning_engine)
            if conf.get("is_executable") and conf.get("confluence_score", 0) >= 70.0:
                plan = conf.get("trade_plan", {})
                if plan:
                    if self._is_vetoed_by_battlefield(sym, plan["side"]):
                        logger.info(f"Confluence entry vetoed by Battlefield Arbiter for {sym}")
                    else:
                        desc = f"Master Confluence ({conf['confluence_score']}% {conf['bias_signal']})"
                        c_size = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else plan["size"]
                        return self.execute_trade(
                            symbol=sym,
                            side=plan["side"],
                            size=c_size,
                            sl_price=plan["sl"],
                            tp1_price=plan["tp1"],
                            tp2_price=plan["tp2"],
                            strategy=desc,
                            trade_type="confluence_master",
                            reason=conf.get("decision_reason", "Confluence alignment"),
                            confidence=f"{int(conf.get('confluence_score', 80) / 10)}/10",
                            news_sentiment=news_data,
                            institutional_flow=inst_flow,
                            learning_notes=learning_note,
                        )
        except Exception as e:
            logger.warning(f"Error evaluating Confluence setup for {sym}: {e}")

        # 3. Check Gautam Jha liquidity setup on 15m
        try:
            gj_analysis = get_gautam_jha_analysis(sym)
            setup = gj_analysis.get("setup")
            if setup and setup.get("type"):
                side = setup.get("direction", "LONG").lower()
                if self._is_vetoed_by_battlefield(sym, side):
                    logger.info(f"Gautam Jha entry vetoed by Battlefield Arbiter for {sym}")
                else:
                    sl = setup.get("sl")
                    tp1 = setup.get("tp1")
                    tp2 = setup.get("tp2")
                    desc = f"Gautam Jha {setup.get('type')}"

                    # Self-learning gate check
                    can_run, reason = self.learning_engine.can_execute(desc, sym)
                    if not can_run:
                        logger.info(f"Self-Learning Filter skipped {sym} ({desc}): {reason}")
                    else:
                        base_sz = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else float(self.data.get("default_size", 1.0))
                        adapted_params = self.learning_engine.get_adapted_parameters(
                            desc, sym, default_size=base_sz
                        )
                        adapted_size = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else adapted_params.get("recommended_size", self.data.get("default_size", 1.0))
                        sl_mult = adapted_params.get("sl_buffer_multiplier", 1.0)
                        if sl is not None and sl_mult > 1.0:
                            try:
                                ticker = get_ticker(sym)
                                mark = float(ticker["mark_price"] or ticker["close"])
                                dist = abs(mark - sl)
                                if side in ("buy", "long"):
                                    sl = round(mark - (dist * sl_mult), 2)
                                else:
                                    sl = round(mark + (dist * sl_mult), 2)
                            except Exception:
                                pass

                        return self.execute_trade(
                            symbol=sym,
                            side=side,
                            size=adapted_size,
                            sl_price=sl,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            strategy=desc,
                            trade_type="gautam_jha",
                            reason=f"Gautam Jha Liquidity Sweep ({side.upper()})",
                            confidence="8/10",
                            news_sentiment=news_data,
                            institutional_flow=inst_flow,
                            learning_notes=learning_note,
                        )
        except Exception as e:
            logger.warning(f"Error checking GJ setup for {sym}: {e}")

        # 4. Check 5m & 15m Candle Entry signals
        try:
            multi_entry = get_multi_timeframe_entry(sym, ["5m", "15m"])
            for tf in ["15m", "5m"]:
                tf_data = multi_entry["timeframes"].get(tf, {})
                if tf_data.get("has_setup") and tf_data.get("trade_plan"):
                    tp_plan = tf_data["trade_plan"]
                    sig = tf_data.get("signal", "NEUTRAL")
                    if "BUY" in sig or "LONG" in sig:
                        side = "buy"
                    elif "SELL" in sig or "SHORT" in sig:
                        side = "sell"
                    else:
                        continue

                    if self._is_vetoed_by_battlefield(sym, side):
                        logger.info(f"Candle entry vetoed by Battlefield Arbiter for {sym}")
                        continue

                    desc = f"{tf.upper()} {tf_data.get('pattern', 'Candle Setup')}"

                    # Self-learning gate check
                    can_run, reason = self.learning_engine.can_execute(desc, sym)
                    if not can_run:
                        logger.info(f"Self-Learning Filter skipped {sym} ({desc}): {reason}")
                        continue

                    base_sz = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else float(self.data.get("default_size", 1.0))
                    adapted_params = self.learning_engine.get_adapted_parameters(
                        desc, sym, default_size=base_sz
                    )
                    adapted_size = self.get_effective_lot_size(sym) if self.data.get("lot_size_mode", "custom") == "custom" else adapted_params.get("recommended_size", self.data.get("default_size", 1.0))
                    sl_mult = adapted_params.get("sl_buffer_multiplier", 1.0)
                    sl = tp_plan.get("sl")
                    if sl is not None and sl_mult > 1.0:
                        try:
                            ticker = get_ticker(sym)
                            mark = float(ticker["mark_price"] or ticker["close"])
                            dist = abs(mark - sl)
                            if side in ("buy", "long"):
                                sl = round(mark - (dist * sl_mult), 2)
                            else:
                                sl = round(mark + (dist * sl_mult), 2)
                        except Exception:
                            pass

                    return self.execute_trade(
                        symbol=sym,
                        side=side,
                        size=adapted_size,
                        sl_price=sl,
                        tp1_price=tp_plan.get("tp1"),
                        tp2_price=tp_plan.get("tp2"),
                        strategy=desc,
                        trade_type="candle_entry",
                        reason=f"{tf.upper()} {tf_data.get('pattern', 'Candle Setup')}",
                        confidence="7/10",
                        news_sentiment=news_data,
                        institutional_flow=inst_flow,
                        learning_notes=learning_note,
                    )
        except Exception as e:
            logger.warning(f"Error checking Candle Entry for {sym}: {e}")

        return None

    def scan_and_auto_trade(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scan specified symbols and auto-execute setups."""
        if not self.is_enabled:
            return []
        syms = symbols or self.data.get("symbols", ["BTCUSD", "XAUTUSD"])
        executed = []
        for sym in syms:
            trade = self.check_signals_and_auto_execute(sym)
            if trade:
                executed.append(trade)
        return executed

    def check_open_positions_for_exits(self, prices: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Check open positions against live prices for Take-Profit and Stop-Loss triggers.
        """
        if prices is None:
            prices = {}
            for pos in list(self.data.get("positions", {}).values()):
                sym = pos["symbol"]
                if sym not in prices:
                    try:
                        t = get_ticker(sym)
                        prices[sym] = float(t["mark_price"] or t["close"])
                    except Exception as e:
                        logger.warning(f"Failed to fetch ticker for {sym}: {e}")

        closed_trades = []
        for pos_key, pos in list(self.data.get("positions", {}).items()):
            sym = pos["symbol"]
            curr_price = prices.get(sym)
            if curr_price is None:
                continue

            side = pos["side"].lower()
            entry = float(pos.get("entry_price", curr_price))
            sl = float(pos.get("sl_price") or pos.get("sl", 0))
            tp1 = float(pos.get("tp1_price") or pos.get("tp1", 0))
            tp2 = float(pos.get("tp2_price") or pos.get("tp2", tp1))

            # 1. Dynamic Trailing Stop-Loss calculation
            if self.data.get("trailing_sl", False):
                trail_pct = float(self.data.get("trailing_pct", 0.01))
                act_pct = float(self.data.get("trailing_activation_pct", 0.012))
                if side in ("buy", "long") and curr_price >= entry * (1.0 + act_pct):
                    cand_sl = round(curr_price * (1.0 - trail_pct), 2)
                    if cand_sl > sl:
                        pos["sl_price"] = cand_sl
                        pos["sl"] = cand_sl
                        pos["is_trailing"] = True
                        sl = cand_sl
                        self.save()
                elif side in ("sell", "short") and curr_price <= entry * (1.0 - act_pct):
                    cand_sl = round(curr_price * (1.0 + trail_pct), 2)
                    if sl == 0 or cand_sl < sl:
                        pos["sl_price"] = cand_sl
                        pos["sl"] = cand_sl
                        pos["is_trailing"] = True
                        sl = cand_sl
                        self.save()

            # 2. Breakeven Stop-Loss shift when TP1 is hit
            if self.data.get("auto_breakeven", False):
                if side in ("buy", "long"):
                    if curr_price >= tp1 and tp1 > 0 and not pos.get("tp1_hit", False) and tp2 > tp1:
                        pos["tp1_hit"] = True
                        pos["sl_price"] = entry
                        pos["sl"] = entry
                        pos["is_breakeven"] = True
                        sl = entry
                        self.save()
                elif side in ("sell", "short"):
                    if curr_price <= tp1 and tp1 > 0 and not pos.get("tp1_hit", False) and tp2 < tp1:
                        pos["tp1_hit"] = True
                        pos["sl_price"] = entry
                        pos["sl"] = entry
                        pos["is_breakeven"] = True
                        sl = entry
                        self.save()

            hit_exit = False
            exit_reason = ""

            if side in ("buy", "long"):
                if curr_price >= tp2 and tp2 > 0:
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP2 Hit)"
                elif curr_price >= tp1 and tp1 > 0 and not self.data.get("auto_breakeven", False):
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP1 Hit)"
                elif curr_price <= sl and sl > 0:
                    hit_exit = True
                    exit_reason = "BREAKEVEN Stop Hit" if pos.get("is_breakeven") else "STOP LOSS Triggered"
            elif side in ("sell", "short"):
                if curr_price <= tp2 and tp2 > 0:
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP2 Hit)"
                elif curr_price <= tp1 and tp1 > 0 and not self.data.get("auto_breakeven", False):
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP1 Hit)"
                elif curr_price >= sl and sl > 0:
                    hit_exit = True
                    exit_reason = "BREAKEVEN Stop Hit" if pos.get("is_breakeven") else "STOP LOSS Triggered"

            if hit_exit:
                try:
                    closed = self.close_position(pos_key, reason=exit_reason, exit_price=curr_price)
                    if closed.get("status") == "closed":
                        closed_trades.append(closed)
                except Exception as e:
                    logger.error(f"Error auto-closing position for {sym}: {e}")

        return closed_trades

    # ==================== Reporting & Stats ====================

    def get_summary(self) -> Dict[str, Any]:
        """Get complete auto-trader dashboard summary."""
        stats = self.data.get("stats", {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        return {
            "enabled": self.is_enabled,
            "mode": self.mode,
            "symbols": self.data.get("symbols", []),
            "open_positions": self.data.get("positions", {}),
            "open_positions_count": len(self.data.get("positions", {})),
            "max_open_positions": int(self.data.get("max_open_positions", 5)),
            "risk_per_trade_pct": float(self.data.get("risk_per_trade_pct", 0.015)),
            "lot_size": float(self.data.get("lot_size", self.data.get("default_size", 1.0))),
            "default_size": float(self.data.get("default_size", 1.0)),
            "lot_size_mode": self.data.get("lot_size_mode", "custom"),
            "symbol_lot_sizes": dict(self.data.get("symbol_lot_sizes", {})),
            "allow_multiple_per_symbol": bool(self.data.get("allow_multiple_per_symbol", False)),
            "auto_breakeven": bool(self.data.get("auto_breakeven", False)),
            "trailing_sl": bool(self.data.get("trailing_sl", False)),
            "trailing_pct": float(self.data.get("trailing_pct", 0.01)),
            "trailing_activation_pct": float(self.data.get("trailing_activation_pct", 0.012)),
            "paper_balance": self.data.get("paper_balance", 10000.0),
            "balance": self.data.get("paper_balance", 10000.0),
            "initial_balance": 10000.0,
            "equity": self.data.get("paper_balance", 10000.0),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl": stats.get("total_pnl", 0.0),
            "total_realized_pnl": stats.get("total_pnl", 0.0),
            "delta_configured": self.delta_client.is_configured(),
            "masked_key": self.delta_client.get_masked_key(),
            "learning": self.learning_engine.get_learning_summary(),
            "battlefield_validation": bool(self.data.get("battlefield_validation", True)),
            "battlefield_min_confidence": int(self.data.get("battlefield_min_confidence", 8)),
            "backup_channel_id": self.data.get("backup_channel_id", os.getenv("BACKUP_CHANNEL_ID", "")),
        }

    def get_account_summary(self) -> Dict[str, Any]:
        """Alias for get_summary."""
        return self.get_summary()

    def get_positions_with_pnl(self) -> List[Dict[str, Any]]:
        """Return all open positions enriched with real-time mark price and unrealized PnL."""
        results = []
        for pos_key, pos in self.data.get("positions", {}).items():
            pos_copy = dict(pos)
            sym = pos["symbol"]
            try:
                t = get_ticker(sym)
                curr = float(t["mark_price"] or t["close"])
                pos_copy["current_price"] = curr
                entry = float(pos["entry_price"])
                size = float(pos["size"])
                side = pos["side"].lower()
                if side in ("buy", "long"):
                    u_pnl = (curr - entry) * size
                    u_pct = ((curr - entry) / entry) * 100.0 if entry > 0 else 0.0
                else:
                    u_pnl = (entry - curr) * size
                    u_pct = ((entry - curr) / entry) * 100.0 if entry > 0 else 0.0
                pos_copy["unrealized_pnl"] = round(u_pnl, 2)
                pos_copy["unrealized_pnl_pct"] = round(u_pct, 2)
            except Exception:
                pos_copy["current_price"] = pos.get("entry_price")
                pos_copy["unrealized_pnl"] = 0.0
                pos_copy["unrealized_pnl_pct"] = 0.0
            results.append(pos_copy)
        return results

    def format_institutional_dashboard(self) -> str:
        """Format single-screen institutional command center dashboard."""
        from market_data import get_market_session, get_symbol_display_name
        data = self.get_summary()
        open_positions = self.get_positions_with_pnl()
        session = get_market_session()

        status_tag = "🟢 <b>ACTIVE & SCANNING</b>" if data["enabled"] else "🔴 <b>STOPPED / PAUSED</b>"
        mode_tag = "⚡ <b>LIVE (DELTA)</b>" if data["mode"] == "live" else "📝 <b>SIMULATED PAPER</b>"

        bal = data["paper_balance"]
        tot_pnl = data["total_pnl"]
        pnl_sign = "+" if tot_pnl >= 0 else ""
        pnl_emoji = "🟢" if tot_pnl >= 0 else "🔴"
        win_rate = data["win_rate"]

        lines = [
            "🏛️ <b>INSTITUTIONAL TRADING DESK DASHBOARD</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"🤖 <b>Bot Status:</b> {status_tag} | Mode: {mode_tag}",
            f"🌐 <b>Session:</b> {session['emoji']} <b>{session['session']}</b> ({session['liquidity']} Liquidity)",
            f"   └ <i>{session['description']}</i>\n",
            "💼 <b>PORTFOLIO & CAPITAL:</b>",
            f"• <b>Account Balance:</b> <code>${bal:,.2f}</code>",
            f"• <b>Realized PnL:</b> {pnl_emoji} <code>{pnl_sign}${tot_pnl:,.2f}</code>",
            f"• <b>Historical Win Rate:</b> <code>{win_rate:.1f}%</code> ({data['wins']}W / {data['losses']}L / {data['total_trades']} Total)\n",
        ]

        # Open Positions
        max_pos = data["max_open_positions"]
        lines.append(f"📈 <b>ACTIVE POSITIONS ({len(open_positions)}/{max_pos}):</b>")
        if not open_positions:
            lines.append("• <i>No active positions. Scanning 1m/5m/15m AMD & Confluence setups...</i>\n")
        else:
            for p in open_positions:
                sym_display = get_symbol_display_name(p["symbol"])
                side_badge = "🟢 LONG" if p["side"].lower() in ("buy", "long") else "🔴 SHORT"
                u_pnl = p.get("unrealized_pnl", 0.0)
                u_pct = p.get("unrealized_pnl_pct", 0.0)
                u_sign = "+" if u_pnl >= 0 else ""
                u_emoji = "🟢" if u_pnl >= 0 else "🔴"

                be_badge = " [🛡️ BE]" if p.get("is_breakeven") else ""
                trail_badge = " [⚡ TRAIL]" if p.get("is_trailing") else ""

                lines.append(
                    f"• {side_badge} <b>{sym_display}</b>{be_badge}{trail_badge}\n"
                    f"  Entry: <code>${p['entry_price']:,.2f}</code> | Mark: <code>${p.get('current_price', p['entry_price']):,.2f}</code>\n"
                    f"  PnL: {u_emoji} <b>{u_sign}${u_pnl:,.2f} ({u_sign}{u_pct:.2f}%)</b> | RRR: <code>{p.get('rrr', '1:2')}</code>\n"
                    f"  SL: <code>${p.get('sl_price', 0):,.2f}</code> | TP1: <code>${p.get('tp1_price', 0):,.2f}</code> | TP2: <code>${p.get('tp2_price', 0):,.2f}</code>"
                )
            lines.append("")

        # Risk Rules
        risk_pct = data["risk_per_trade_pct"] * 100.0
        be_status = "🟢 ACTIVE" if data.get("auto_breakeven") else "⚪ OFF"
        trail_status = f"🟢 ACTIVE ({data.get('trailing_pct', 0.01)*100:.1f}%)" if data.get("trailing_sl") else "⚪ OFF"
        multi_status = "🟢 ALLOWED" if data["allow_multiple_per_symbol"] else "⚪ 1 PER PAIR"

        lot_size = data.get("lot_size", 1.0)
        lot_mode = data.get("lot_size_mode", "custom").upper()
        sym_sizes = data.get("symbol_lot_sizes", {})
        if sym_sizes:
            overrides = ", ".join([f"{k}:{v}" for k, v in sym_sizes.items()])
            lot_display = f"<code>{lot_size}</code> (Overrides: {overrides})"
        else:
            lot_display = f"<code>{lot_size}</code> (Global)"

        lines.extend([
            "🛡️ <b>RISK MANAGEMENT RULES:</b>",
            f"• <b>Auto-Trade Lot Size:</b> {lot_display} [Mode: <code>{lot_mode}</code>]",
            f"• <b>Capital Risk / Trade:</b> <code>{risk_pct:.1f}%</code>",
            f"• <b>Breakeven SL on TP1:</b> {be_status} (Protects wins into risk-free trades)",
            f"• <b>Trailing Stop-Loss:</b> {trail_status}",
            f"• <b>Symbol Allocation:</b> {multi_status}",
            f"• <b>Watched Pairs:</b> {', '.join(data['symbols'])}\n",
        ])

        # Self-learning
        learn = data.get("learning", {})
        l_score = learn.get("overall_score", 1.0)
        l_supp = len(learn.get("suppressed_setups", []))
        lines.extend([
            "🧠 <b>SELF-LEARNING DESK:</b>",
            f"• <b>Desk Health Score:</b> <code>{l_score:.2f}x</code> | Suppressed Setups: <code>{l_supp}</code>",
            f"• <b>Optimization:</b> Deterministic (0 LLM tokens)\n",
        ])

        # APIs
        delta_stat = "🟢 Connected" if data["delta_configured"] else "⚠️ Not Connected (Paper Mode Active)"
        lines.extend([
            "🔑 <b>API CONNECTIVITY:</b>",
            f"• <b>Delta Exchange:</b> {delta_stat}",
            "",
            "⚡ <b>QUICK ACTIONS:</b>",
            "• <code>/trade on</code> | <code>/trade off</code> — Toggle Bot",
            "• <code>/trade size 0.05</code> | <code>/size btc 0.01</code> — Change Lot Size",
            "• <code>/trade be on</code> — Toggle Breakeven Protection",
            "• <code>/trade trail on</code> — Toggle Trailing Stop",
            "• <code>/btc</code> | <code>/gold</code> | <code>/amd</code> — Deep Hubs",
        ])

        return "\n".join(lines)

    def get_learning_report(self) -> str:
        """Get formatted HTML report from self-learning engine (0 tokens)."""
        return self.learning_engine.format_html_report()

    def get_learning_ai_insight(self, gemini_caller) -> str:
        """Get compact AI insight on strategy performance (< 250 tokens)."""
        return self.learning_engine.generate_ai_insight(gemini_caller)

    def reset_learning(self):
        """Reset self-learning state."""
        self.learning_engine.reset()

    def evaluate_confluence(self, symbol: str) -> Dict[str, Any]:
        """Evaluate all 5 strategy layers for a symbol."""
        return self.confluence_engine.evaluate_confluence(symbol, self.learning_engine)

    def execute_confluence_trade(self, symbol: str, force: bool = False) -> Dict[str, Any]:
        """Execute a trade combining all strategies (Technicals, Order Book, News, Learning)."""
        return self.confluence_engine.execute_confluence_trade(symbol, self, force=force)

    def get_confluence_report(self, symbol: str) -> str:
        """Get formatted HTML report for all combined strategies."""
        res = self.evaluate_confluence(symbol)
        return format_confluence_html_report(res)

    def get_orderbook_report(self, symbol: str) -> str:
        """Get formatted HTML report for L2 Order Book depth."""
        book = analyze_orderbook(symbol)
        return format_orderbook_html_report(book)

    def get_news_report(self, symbol: str) -> str:
        """Get formatted HTML report for News and Macro Sentiment."""
        news = get_news_sentiment(symbol)
        return format_news_html_report(news)

    def get_amd_report(self, symbol: str) -> str:
        """Get formatted HTML report for 1m, 5m, 15m AMD Scalp analysis."""
        analysis = analyze_amd_scalp(
            symbol,
            balance=self.balance,
            risk_pct=float(self.data.get("risk_per_trade_pct", 0.015)),
        )
        return format_amd_scalp_html_report(analysis)

    def execute_amd_trade(self, symbol: str, force: bool = False) -> Dict[str, Any]:
        """Manually trigger AMD Scalp execution if setup is valid or force is requested."""
        analysis = analyze_amd_scalp(
            symbol,
            balance=self.balance,
            risk_pct=float(self.data.get("risk_per_trade_pct", 0.015)),
        )
        if not analysis.get("has_setup") and not force:
            return {
                "status": "rejected",
                "reason": f"No valid AMD Scalp trigger found ({analysis.get('phase')}).",
                "analysis": analysis,
            }

        plan = analysis.get("trade_plan")
        if not plan:
            sym = resolve_symbol(symbol)
            t = get_ticker(sym)
            mark = float(t["mark_price"] or t["close"])
            plan = calculate_risk_managed_plan(
                symbol=sym,
                side="buy" if analysis.get("direction") == "LONG" else "sell",
                entry_price=mark,
                invalidation_level=mark * 0.99,
                range_target=mark * 1.02,
                external_target=mark * 1.03,
                atr_1m=mark * 0.001,
                balance=self.balance,
                risk_pct=float(self.data.get("risk_per_trade_pct", 0.015)),
            )

        strat_desc = f"AMD Scalp ({analysis.get('phase', 'DISTRIBUTION')} {plan['side'].upper()})"
        amd_size = self.get_effective_lot_size(symbol) if self.data.get("lot_size_mode", "custom") == "custom" else plan["suggested_size"]
        trade = self.execute_trade(
            symbol=symbol,
            side=plan["side"],
            size=amd_size,
            sl_price=plan["sl"],
            tp1_price=plan["tp1"],
            tp2_price=plan["tp2"],
            strategy=strat_desc,
            trade_type="amd_scalp",
            reason=f"AMD Scalp Trigger ({plan.get('rrr', '1:2')})",
        )
        return {
            "status": "executed",
            "trade": trade,
            "analysis": analysis,
        }

    def set_battlefield_config(
        self,
        enabled: Optional[bool] = None,
        min_confidence: Optional[int] = None
    ) -> Dict[str, Any]:
        """Configure Battlefield multi-agent debate parameters."""
        if enabled is not None:
            self.data["battlefield_validation"] = bool(enabled)
        if min_confidence is not None:
            self.data["battlefield_min_confidence"] = max(1, min(10, int(min_confidence)))
        self.save()
        return {
            "battlefield_validation": self.data.get("battlefield_validation", True),
            "battlefield_min_confidence": int(self.data.get("battlefield_min_confidence", 8)),
        }

    def set_backup_channel_id(self, channel_id: str) -> str:
        """Configure dedicated backup channel and persist across runtime and store."""
        clean_id = str(channel_id).strip()
        self.data["backup_channel_id"] = clean_id
        self.save()
        try:
            from backup import set_backup_channel_id as persist_backup_channel
            persist_backup_channel(clean_id)
        except Exception as e:
            logger.warning(f"Notice: Could not persist backup channel via backup module: {e}")
        return clean_id

    def execute_battlefield_trade(
        self,
        symbol: str,
        force: bool = False,
        min_confidence: int = 8
    ) -> Dict[str, Any]:
        """
        Runs Bull vs Bear debate arbiter and executes trade on Delta Exchange if confidence >= min_confidence.
        """
        from battlefield_engine import get_market_snapshot_sync, run_battlefield_sync
        sym = resolve_symbol(symbol)
        data = get_market_snapshot_sync(sym, timeframe="15m")
        bull, bear, decision = run_battlefield_sync(data)
        verdict = decision.get("verdict", "NO_TRADE")
        confidence = int(decision.get("confidence") or 0)
        sl = decision.get("stop_loss")
        tp = decision.get("take_profit")
        reason = decision.get("reasoning", "")

        executed_trade = None
        should_trade = force or (verdict in ["BUY", "SELL"] and confidence >= min_confidence)
        if should_trade and verdict in ["BUY", "SELL"]:
            side = "buy" if verdict == "BUY" else "sell"
            executed_trade = self.execute_trade(
                symbol=sym,
                side=side,
                sl_price=sl,
                tp1_price=tp,
                strategy=f"Battlefield Arbiter ({verdict})",
                trade_type="battlefield",
                reason=f"Confidence {confidence}/10: {reason}",
            )

        return {
            "symbol": sym,
            "market_data": data,
            "bull_thesis": bull,
            "bear_counter": bear,
            "decision": decision,
            "executed": executed_trade is not None,
            "trade": executed_trade,
        }



