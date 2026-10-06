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

logger = logging.getLogger(__name__)


class AutoTrader:
    """Automated trade execution engine with risk management and state persistence."""

    def __init__(
        self,
        store_file: str = "autotrade_store.json",
        delta_client: Optional[DeltaClient] = None,
        mode: Optional[str] = None,
        default_size: Optional[float] = None,
    ):
        self.store_file = store_file
        self.delta_client = delta_client or DeltaClient()
        self.data: Dict[str, Any] = {
            "enabled": False,
            "mode": (mode or os.environ.get("TRADING_MODE", "paper")).lower(),
            "symbols": ["BTCUSD", "XAUTUSD"],
            "timeframes": ["5m", "15m"],
            "default_size": float(default_size or os.environ.get("DEFAULT_ORDER_SIZE", 1)),
            "max_open_positions": 2,
            "paper_balance": 10000.0,  # $10,000 initial virtual capital
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
        }

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
    ) -> Dict[str, Any]:
        """
        Execute trade either in PAPER mode (simulation) or LIVE mode (Delta Exchange).
        """
        sym = resolve_symbol(symbol)
        side_clean = side.upper().strip()
        if side_clean not in ("BUY", "SELL", "LONG", "SHORT"):
            raise ValueError("Side must be BUY, SELL, LONG, or SHORT.")

        order_side = "buy" if side_clean in ("BUY", "LONG") else "sell"
        order_size = float(size or self.data.get("default_size", 1.0))
        trade_mode = (mode or self.data.get("mode", "paper")).lower()

        # Check existing position for this symbol
        for pos in self.data.get("positions", {}).values():
            if pos.get("symbol") == sym:
                raise ValueError(f"An open position already exists for {sym}. Close it first before opening a new trade.")

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

        risk_amount = abs(mark_price - effective_sl)
        reward_amount = abs(effective_tp1 - mark_price)
        rrr = f"1:{(reward_amount / risk_amount):.2f}" if risk_amount > 0 else "1:1.5"

        strat_name = strategy or reason or trade_type
        position_id = f"{sym}_{int(time.time() * 1000)}"

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
        for pos in self.data.get("positions", {}).values():
            if pos.get("symbol") == sym:
                return None

        if len(self.data.get("positions", {})) >= self.data.get("max_open_positions", 2):
            return None

        # 1. Check Gautam Jha liquidity setup on 15m
        try:
            gj_analysis = get_gautam_jha_analysis(sym)
            setup = gj_analysis.get("setup")
            if setup and setup.get("type"):
                side = setup.get("direction", "LONG").lower()
                sl = setup.get("sl")
                tp1 = setup.get("tp1")
                tp2 = setup.get("tp2")
                desc = f"Gautam Jha {setup.get('type')}"

                return self.execute_trade(
                    symbol=sym,
                    side=side,
                    sl_price=sl,
                    tp1_price=tp1,
                    tp2_price=tp2,
                    strategy=desc,
                )
        except Exception as e:
            logger.warning(f"Error checking GJ setup for {sym}: {e}")

        # 2. Check 5m & 15m Candle Entry signals
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

                    desc = f"{tf.upper()} {tf_data.get('pattern', 'Candle Setup')}"
                    return self.execute_trade(
                        symbol=sym,
                        side=side,
                        sl_price=tp_plan.get("sl"),
                        tp1_price=tp_plan.get("tp1"),
                        tp2_price=tp_plan.get("tp2"),
                        strategy=desc,
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
            sl = float(pos.get("sl_price") or pos.get("sl", 0))
            tp1 = float(pos.get("tp1_price") or pos.get("tp1", 0))
            tp2 = float(pos.get("tp2_price") or pos.get("tp2", tp1))

            hit_exit = False
            exit_reason = ""

            if side in ("buy", "long"):
                if curr_price >= tp2 and tp2 > 0:
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP2 Hit)"
                elif curr_price >= tp1 and tp1 > 0:
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP1 Hit)"
                elif curr_price <= sl and sl > 0:
                    hit_exit = True
                    exit_reason = "STOP LOSS Triggered"
            elif side in ("sell", "short"):
                if curr_price <= tp2 and tp2 > 0:
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP2 Hit)"
                elif curr_price <= tp1 and tp1 > 0:
                    hit_exit = True
                    exit_reason = "TAKE PROFIT (TP1 Hit)"
                elif curr_price >= sl and sl > 0:
                    hit_exit = True
                    exit_reason = "STOP LOSS Triggered"

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
