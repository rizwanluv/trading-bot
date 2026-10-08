"""
Auto Trade & Runner Module for Delta Exchange & Telegram Bot
============================================================
Single unified entry point providing:
1. Automated trading engine for Telegram bot & live market monitoring:
   - Dynamic ON/OFF controls
   - Configurable lot sizing (fixed & risk-based)
   - Strategy TP & SL management (RR, points, percentage, ATR, swing)
   - Real-time Delta Exchange ticker & candle feed with failover
   - Open position lifecycle tracking and PnL calculation
2. Standalone & CLI Multi-Strategy Runner:
   - backtest               Backtest Indicators-Pro strategy on demo data
   - backtest-ai            Quick learning smoke-test of AI bot (demo data, temp memory)
   - live --strategy ai     Live/paper trading with AI Learning bot
   - live --strategy pro    Live/paper trading with Indicators-Pro strategy
   - both                   Backtest first, then start AI live bot

EXAMPLES
--------
  python auto_trade.py backtest --bars 5000
  python auto_trade.py live --strategy pro --symbol XAUUSD --source demo
  python auto_trade.py live --strategy ai --source binance --poll 30
  python auto_trade.py both --bars 3000

ENV VARS
--------
  BOT_SYMBOL, BOT_DATA_SOURCE, BOT_EXCHANGE, BOT_POLL, BOT_EQUITY,
  EXCHANGE_API_KEY, EXCHANGE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Educational only. Not financial advice.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import time as time_module
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from trading_strategy_indicators_pro import Direction, IndicatorsProStrategy, IndicatorEngine

logger = logging.getLogger("trading_bot.auto_trade")

BASE = os.path.dirname(os.path.abspath(__file__))
AI_FILE = os.path.join(BASE, "ai_bot_learning.py")
PRO_FILE = os.path.join(BASE, "trading_strategy_indicators_pro.py")
CONFIG_FILE_PATH = os.path.join(BASE, "auto_trade_config.json")
TRADES_HISTORY_PATH = os.path.join(BASE, "trades_history.json")
DELTA_CHART_API = os.getenv(
    "DELTA_CHART_API", "https://api.india.delta.exchange/v2/chart/history"
)
DELTA_TICKER_API = os.getenv(
    "DELTA_TICKER_API", "https://api.india.delta.exchange/v2/tickers"
)


# ==================================================================
# 1. AUTO TRADE CONFIGURATION & POSITION MODELS
# ==================================================================


@dataclass
class AutoTradeConfig:
    enabled: bool = False
    symbol: str = "BTCUSD"
    lot_size: float = 0.01
    lot_mode: str = "fixed"  # "fixed" or "risk_pct"
    risk_pct: float = 1.0  # used if lot_mode is "risk_pct"
    tp_value: float = 2.0  # multiplier, points, or percentage
    sl_value: float = 1.0  # multiplier, points, or percentage
    tp_mode: str = "rr"  # "rr", "pts", "pct", "atr"
    sl_mode: str = "swing"  # "swing", "pts", "pct", "atr"
    trailing_sl: bool = False
    max_daily_loss_pct: float = 3.0
    strategy_type: str = "indicators_pro"  # "indicators_pro" or "ai_learning"
    poll_seconds: int = 15
    equity: float = 10000.0
    notify_chat_id: Optional[int] = None
    config_file: str = CONFIG_FILE_PATH
    trades_history_file: str = TRADES_HISTORY_PATH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "lot_mode": self.lot_mode,
            "risk_pct": self.risk_pct,
            "tp_value": self.tp_value,
            "sl_value": self.sl_value,
            "tp_mode": self.tp_mode,
            "sl_mode": self.sl_mode,
            "trailing_sl": self.trailing_sl,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "strategy_type": self.strategy_type,
            "poll_seconds": self.poll_seconds,
            "equity": self.equity,
            "notify_chat_id": self.notify_chat_id,
        }

    def save(self) -> None:
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("Saved auto-trade configuration to %s", self.config_file)
        except Exception as exc:
            logger.warning("Could not save auto-trade config: %s", exc)

    @classmethod
    def load(cls, path: str = CONFIG_FILE_PATH) -> AutoTradeConfig:
        cfg = cls(config_file=path)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                logger.info("Loaded auto-trade configuration from %s", path)
            except Exception as exc:
                logger.warning("Could not read auto-trade config %s: %s", path, exc)
        return cfg


@dataclass
class AutoTradePosition:
    id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    lot_size: float
    entry_time: str
    strategy: str
    reason: str
    highest_price: float
    lowest_price: float
    status: str = "OPEN"  # "OPEN" or "CLOSED"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None  # "TP1", "TP2", "SL", "MANUAL"
    pnl: float = 0.0

    def current_pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            diff = current_price - self.entry_price
        else:
            diff = self.entry_price - current_price
        return round(diff * self.lot_size, 2)


# ==================================================================
# 2. AUTO TRADER CORE ENGINE (FOR BOT & MONITORING)
# ==================================================================


class AutoTrader:
    """
    Automated trading engine with live Delta Exchange market feed,
    strategy signal generation, position lifecycle management, and
    interactive configuration.
    """

    def __init__(self, config: Optional[AutoTradeConfig] = None):
        self.config = config or AutoTradeConfig.load()
        self.position: Optional[AutoTradePosition] = None
        self.closed_trades: List[AutoTradePosition] = self._load_trades_history()
        self.is_running: bool = False
        self._strategy_pro: Optional[IndicatorsProStrategy] = None
        self._init_strategy()

    def _load_trades_history(self) -> List[AutoTradePosition]:
        history_path = getattr(self.config, "trades_history_file", TRADES_HISTORY_PATH)
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trades = []
                for item in data:
                    if isinstance(item, dict):
                        trades.append(AutoTradePosition(**item))
                logger.info("Loaded %d historical trades from %s", len(trades), history_path)
                return trades
            except Exception as exc:
                logger.warning("Could not read trades history %s: %s", history_path, exc)
        return []

    def _save_trades_history(self) -> None:
        history_path = getattr(self.config, "trades_history_file", TRADES_HISTORY_PATH)
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump([asdict(t) for t in self.closed_trades[-100:]], f, indent=2)
            logger.info("Saved %d trades to history %s", len(self.closed_trades), history_path)
        except Exception as exc:
            logger.warning("Could not save trades history %s: %s", history_path, exc)

    def _init_strategy(self) -> None:
        self._strategy_pro = IndicatorsProStrategy(
            symbol=self.config.symbol,
            risk_per_trade=self.config.risk_pct,
            tp_mode=self.config.tp_mode,
            sl_mode=self.config.sl_mode,
            tp_value=self.config.tp_value,
            sl_value=self.config.sl_value,
            lot_size=self.config.lot_size,
            lot_mode=self.config.lot_mode,
        )

    def enable(self, chat_id: Optional[int] = None) -> str:
        self.config.enabled = True
        if chat_id is not None:
            self.config.notify_chat_id = chat_id
        self.config.save()
        return "Auto Trade is now ENABLED (ON). Monitoring market for strategy signals."

    def disable(self) -> str:
        self.config.enabled = False
        self.config.save()
        return "Auto Trade is now DISABLED (OFF). No new automatic trades will be executed."

    def toggle(
        self, state: Optional[bool] = None, chat_id: Optional[int] = None
    ) -> Tuple[bool, str]:
        if state is None:
            new_state = not self.config.enabled
        else:
            new_state = bool(state)

        if new_state:
            msg = self.enable(chat_id)
        else:
            msg = self.disable()
        return new_state, msg

    def set_lot_size(self, size: float, mode: str = "fixed") -> Tuple[bool, str]:
        if size <= 0:
            return False, "Error: Lot size must be greater than 0."

        self.config.lot_size = float(size)
        self.config.lot_mode = mode.lower()
        if self._strategy_pro:
            self._strategy_pro.set_lot_size(self.config.lot_size, self.config.lot_mode)
        self.config.save()

        mode_desc = (
            "Fixed lot size" if self.config.lot_mode == "fixed" else "Risk % lot sizing"
        )
        return True, f"Lot size updated: {self.config.lot_size} ({mode_desc})"

    def set_tp_sl(
        self,
        tp_val: float,
        sl_val: float,
        tp_mode: str = "rr",
        sl_mode: str = "swing",
    ) -> Tuple[bool, str]:
        if tp_val <= 0 or sl_val <= 0:
            return False, "Error: TP and SL values must be positive."

        valid_modes = ("rr", "pts", "pct", "atr", "swing")
        if tp_mode not in valid_modes or sl_mode not in valid_modes:
            return False, f"Invalid mode. Supported modes: {', '.join(valid_modes)}"

        self.config.tp_value = float(tp_val)
        self.config.sl_value = float(sl_val)
        self.config.tp_mode = tp_mode
        self.config.sl_mode = sl_mode

        if self._strategy_pro:
            self._strategy_pro.set_tp_sl(
                tp_val=self.config.tp_value,
                sl_val=self.config.sl_value,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
            )
        self.config.save()

        return (
            True,
            f"Strategy TP/SL updated:\n• Take Profit: {tp_val} (mode: {tp_mode})\n• Stop Loss: {sl_val} (mode: {sl_mode})",
        )

    def set_symbol(self, symbol: str) -> str:
        cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
        alias_map = {
            "BTC": "BTCUSD",
            "BITCOIN": "BTCUSD",
            "BTCUSDT": "BTCUSD",
            "XAU": "XAUTUSD",
            "GOLD": "XAUTUSD",
            "PAXG": "XAUTUSD",
            "XAUUSD": "XAUTUSD",
            "ETH": "ETHUSD",
            "ETHUSDT": "ETHUSD",
            "SOL": "SOLUSD",
            "SOLUSDT": "SOLUSD",
            "XRP": "XRPUSD",
            "XRPUSDT": "XRPUSD",
        }
        target = alias_map.get(cleaned, cleaned)
        self.config.symbol = target
        if self._strategy_pro:
            self._strategy_pro.set_symbol(target)
        self.config.save()
        return f"Auto trade symbol set to: {self.config.symbol}"

    def set_strategy_type(self, strategy_type: str) -> Tuple[bool, str]:
        st = strategy_type.strip().lower()
        if st not in ("indicators_pro", "ai_learning"):
            return False, "Unsupported strategy. Use 'indicators_pro' or 'ai_learning'."
        self.config.strategy_type = st
        self.config.save()
        return True, f"Strategy switched to: {st}"

    def set_trailing_sl(self, enabled: bool) -> str:
        self.config.trailing_sl = bool(enabled)
        self.config.save()
        state = "ENABLED (ON)" if self.config.trailing_sl else "DISABLED (OFF)"
        return f"Trailing Stop Loss is now {state}."

    def set_max_daily_loss(self, pct: float) -> Tuple[bool, str]:
        if pct <= 0 or pct > 50:
            return False, "Error: Max daily loss limit must be between 0.1% and 50%."
        self.config.max_daily_loss_pct = float(pct)
        self.config.save()
        return True, f"Max daily loss risk limit updated to: {self.config.max_daily_loss_pct}% equity."

    def fetch_candles(self, symbol: str, count: int = 120) -> pd.DataFrame:
        """Fetch 1m candle series from Delta Exchange API with fallback to synthetic data."""
        cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
        alias_map = {
            "BTC": "BTCUSD",
            "BITCOIN": "BTCUSD",
            "BTCUSDT": "BTCUSD",
            "XAU": "XAUTUSD",
            "GOLD": "XAUTUSD",
            "PAXG": "XAUTUSD",
            "XAUUSD": "XAUTUSD",
        }
        target = alias_map.get(cleaned, cleaned)
        now = int(time.time())
        from_ts = now - (count + 30) * 60
        url = f"{DELTA_CHART_API}?symbol={target}&resolution=1&from={from_ts}&to={now}"

        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                payload = resp.json()
                res = payload.get("result", {})
                if res and "c" in res and len(res["c"]) > 0:
                    df = pd.DataFrame(
                        {
                            "open": res["o"],
                            "high": res["h"],
                            "low": res["l"],
                            "close": res["c"],
                            "volume": res.get("v", [100.0] * len(res["c"])),
                        },
                        index=pd.to_datetime(res["t"], unit="s", utc=True),
                    ).astype(float)
                    return df
        except Exception as exc:
            logger.warning("Delta candle fetch error for %s: %s", target, exc)

        # Fallback synthetic series for offline resilience and tests
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=count, freq="1min")
        is_btc = "BTC" in target
        base = 82000.0 if is_btc else (4135.0 if "XAU" in target else 2650.0)
        scale = 10.0 if is_btc else 0.4
        spread = 15.0 if is_btc else 1.0
        p = base + np.arange(count, dtype=float) * scale
        return pd.DataFrame(
            {
                "open": p,
                "high": p + spread,
                "low": p - spread,
                "close": p + (scale * 0.5),
                "volume": 500.0,
            },
            index=dates,
        )

    def close_current_position(
        self, current_price: Optional[float] = None, reason: str = "MANUAL"
    ) -> Optional[str]:
        if not self.position:
            return None

        pos = self.position
        price = current_price if current_price is not None else pos.entry_price
        pnl = pos.current_pnl(price)

        pos.status = "CLOSED"
        pos.exit_price = price
        pos.exit_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        pos.exit_reason = reason
        pos.pnl = pnl

        self.config.equity += pnl
        self.config.save()
        self.closed_trades.append(pos)
        self._save_trades_history()
        self.position = None

        if self._strategy_pro:
            self._strategy_pro.update(pnl)

        sign = "+" if pnl >= 0 else ""
        return (
            f"🔄 Position Closed ({reason})\n"
            f"• Pair: {pos.symbol}\n"
            f"• Direction: {pos.direction}\n"
            f"• Entry: {pos.entry_price:.2f} | Exit: {price:.2f}\n"
            f"• Lot Size: {pos.lot_size}\n"
            f"• Realized PnL: {sign}${pnl:.2f}\n"
            f"• New Account Equity: ${self.config.equity:.2f}"
        )

    def step(self) -> List[str]:
        """
        Runs one evaluation cycle:
        1. Checks active position against market for TP/SL triggers.
        2. If no position is open and auto trade is ON, evaluates strategy for entry.
        Returns a list of notification strings for any significant events.
        """
        notifications: List[str] = []
        df1 = self.fetch_candles(self.config.symbol, count=120)
        if df1.empty:
            return notifications

        last_candle = df1.iloc[-1]
        curr_high = float(last_candle["high"])
        curr_low = float(last_candle["low"])
        curr_close = float(last_candle["close"])

        # 1. Manage Active Position
        if self.position is not None:
            pos = self.position
            pos.highest_price = max(pos.highest_price, curr_high)
            pos.lowest_price = min(pos.lowest_price, curr_low)

            if pos.direction == "LONG":
                # Check Take Profit
                if curr_high >= pos.take_profit_1:
                    msg = self.close_current_position(pos.take_profit_1, reason="TP1")
                    if msg:
                        notifications.append(f"🎯 <b>TAKE PROFIT HIT</b>\n{msg}")
                # Check Stop Loss
                elif curr_low <= pos.stop_loss:
                    msg = self.close_current_position(pos.stop_loss, reason="SL")
                    if msg:
                        notifications.append(f"🛑 <b>STOP LOSS HIT</b>\n{msg}")

            elif pos.direction == "SHORT":
                # Check Take Profit
                if curr_low <= pos.take_profit_1:
                    msg = self.close_current_position(pos.take_profit_1, reason="TP1")
                    if msg:
                        notifications.append(f"🎯 <b>TAKE PROFIT HIT</b>\n{msg}")
                # Check Stop Loss
                elif curr_high >= pos.stop_loss:
                    msg = self.close_current_position(pos.stop_loss, reason="SL")
                    if msg:
                        notifications.append(f"🛑 <b>STOP LOSS HIT</b>\n{msg}")

            # Trailing Stop Loss dynamic update
            if self.position is not None and self.config.trailing_sl:
                pos = self.position
                risk_amt = abs(pos.entry_price - pos.stop_loss)
                if pos.direction == "LONG" and curr_close > pos.entry_price + risk_amt:
                    trail_target = round(curr_close - risk_amt, 2)
                    if trail_target > pos.stop_loss:
                        old_sl = pos.stop_loss
                        pos.stop_loss = trail_target
                        notifications.append(
                            f"🛡️ <b>Trailing Stop Moved Up:</b> SL updated from {old_sl:.2f} ➔ <b>{pos.stop_loss:.2f}</b>"
                        )
                elif pos.direction == "SHORT" and curr_close < pos.entry_price - risk_amt:
                    trail_target = round(curr_close + risk_amt, 2)
                    if trail_target < pos.stop_loss:
                        old_sl = pos.stop_loss
                        pos.stop_loss = trail_target
                        notifications.append(
                            f"🛡️ <b>Trailing Stop Moved Down:</b> SL updated from {old_sl:.2f} ➔ <b>{pos.stop_loss:.2f}</b>"
                        )

        # Risk Management: check max daily loss limit
        today_losses = sum(
            t.pnl for t in self.closed_trades
            if t.pnl < 0 and t.exit_time and datetime.now(timezone.utc).strftime("%Y-%m-%d") in str(t.exit_time)
        )
        max_allowed_loss = (self.config.equity * self.config.max_daily_loss_pct / 100.0)
        if abs(today_losses) >= max_allowed_loss and self.config.enabled:
            self.config.enabled = False
            self.config.save()
            notifications.append(
                f"🛑 <b>Risk Guard Triggered:</b> Daily loss reached {self.config.max_daily_loss_pct}% "
                f"(${abs(today_losses):.2f}). Auto-trade paused to preserve capital."
            )

        # 2. Check for New Entry if no active position and auto-trade is ON
        if self.position is None and self.config.enabled:
            daily = (
                df1.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                .dropna()
            )
            if len(daily) < 2:
                # Synthetic daily support
                spread_d = 400.0 if "BTC" in self.config.symbol else 10.0
                daily = pd.DataFrame(
                    [
                        {
                            "open": curr_close - (spread_d * 0.5),
                            "high": curr_close + spread_d,
                            "low": curr_close - spread_d,
                            "close": curr_close,
                        },
                        {
                            "open": curr_close,
                            "high": curr_close + (spread_d * 0.5),
                            "low": curr_close - (spread_d * 0.5),
                            "close": curr_close,
                        },
                    ],
                    index=pd.date_range(
                        end=datetime.now(timezone.utc), periods=2, freq="1D"
                    ),
                )

            if self._strategy_pro:
                sig = self._strategy_pro.generate_signal(df1, daily)
                if sig is not None and sig.direction != Direction.FLAT:
                    entry = sig.entry
                    stop = sig.stop
                    tp1 = sig.tp1
                    tp2 = sig.tp2

                    # Calculate Lot Size
                    if self.config.lot_mode == "fixed":
                        lots = self.config.lot_size
                    else:
                        lots = self._strategy_pro.size(
                            self.config.equity, entry, stop, sig.atr
                        )

                    pos_id = f"TRADE_{int(time.time())}"
                    new_pos = AutoTradePosition(
                        id=pos_id,
                        symbol=self.config.symbol,
                        direction=sig.direction.name,
                        entry_price=round(entry, 2),
                        stop_loss=round(stop, 2),
                        take_profit_1=round(tp1, 2),
                        take_profit_2=round(tp2, 2) if tp2 else None,
                        lot_size=round(lots, 4),
                        entry_time=datetime.now(timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S UTC"
                        ),
                        strategy="Indicators Pro",
                        reason=sig.reason,
                        highest_price=entry,
                        lowest_price=entry,
                    )
                    self.position = new_pos
                    notifications.append(
                        f"🚀 <b>AUTO TRADE OPENED</b>\n"
                        f"• Symbol: {new_pos.symbol}\n"
                        f"• Direction: <b>{new_pos.direction}</b>\n"
                        f"• Entry: {new_pos.entry_price:.2f}\n"
                        f"• Lot Size: {new_pos.lot_size}\n"
                        f"• Take Profit: {new_pos.take_profit_1:.2f}\n"
                        f"• Stop Loss: {new_pos.stop_loss:.2f}\n"
                        f"• Strategy: {new_pos.strategy}\n"
                        f"• Setup: {sig.setup.value}"
                    )

        return notifications

    def get_status_text(self) -> str:
        status_icon = (
            "🟢 <b>ACTIVE (ON)</b>" if self.config.enabled else "🔴 <b>DISABLED (OFF)</b>"
        )
        lot_mode_str = (
            "Fixed"
            if self.config.lot_mode == "fixed"
            else f"Risk {self.config.risk_pct}%"
        )

        text = [
            "⚡ <b>AUTO TRADING DASHBOARD</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"• <b>Status</b>: {status_icon}",
            f"• <b>Symbol</b>: {self.config.symbol}",
            f"• <b>Strategy</b>: {self.config.strategy_type.replace('_', ' ').title()}",
            f"• <b>Lot Size</b>: {self.config.lot_size} ({lot_mode_str})",
            f"• <b>Take Profit</b>: {self.config.tp_value} ({self.config.tp_mode.upper()})",
            f"• <b>Stop Loss</b>: {self.config.sl_value} ({self.config.sl_mode.upper()})",
            f"• <b>Trailing Stop</b>: {'🟢 ON' if self.config.trailing_sl else '⚪ OFF'}",
            f"• <b>Daily Risk Guard</b>: {self.config.max_daily_loss_pct}% equity",
            f"• <b>Account Equity</b>: ${self.config.equity:.2f}",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if self.position:
            pos = self.position
            text.extend(
                [
                    "📊 <b>Active Position:</b>",
                    f"  • {pos.direction} {pos.symbol} @ {pos.entry_price:.2f}",
                    f"  • Lot: {pos.lot_size}",
                    f"  • TP: {pos.take_profit_1:.2f} | SL: {pos.stop_loss:.2f}",
                    f"  • Opened: {pos.entry_time}",
                ]
            )
        else:
            text.append("📊 <b>Active Position</b>: None")

        text.append("━━━━━━━━━━━━━━━━━━━━━━")
        total_closed = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in self.closed_trades)
        win_rate = (len(wins) / total_closed * 100) if total_closed > 0 else 0.0

        text.extend(
            [
                "📈 <b>Trade Performance:</b>",
                f"  • Total Closed: {total_closed}",
                f"  • Win Rate: {win_rate:.1f}% ({len(wins)}/{total_closed})",
                f"  • Net PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}",
            ]
        )

        return "\n".join(text)

    def get_position_text(self) -> str:
        """Return formatted dashboard text of current open position or idle status."""
        if not self.position:
            return (
                "💼 <b>Active Position Dashboard</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "• <b>Status</b>: No active open position\n"
                f"• <b>Trading Pair</b>: <code>{self.config.symbol}</code>\n"
                f"• <b>Auto-Trading</b>: {'🟢 ENABLED' if self.config.enabled else '🔴 DISABLED'}\n"
                f"• <b>Account Equity</b>: <code>${self.config.equity:,.2f}</code>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>💡 Market is continuously scanned for high-probability setups.</i>"
            )

        pos = self.position
        df = self.fetch_candles(pos.symbol, count=5)
        current_price = float(df["close"].iloc[-1]) if not df.empty else pos.entry_price
        pnl = pos.current_pnl(current_price)
        pnl_pct = (
            ((current_price - pos.entry_price) / pos.entry_price * 100.0)
            if pos.direction == "LONG"
            else ((pos.entry_price - current_price) / pos.entry_price * 100.0)
        )
        sign = "+" if pnl >= 0 else ""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"

        return (
            f"💼 <b>Active Position: {pos.symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Direction</b>: <b>{pos.direction}</b> {'🟢' if pos.direction == 'LONG' else '🔴'}\n"
            f"• <b>Entry Price</b>: <code>${pos.entry_price:,.2f}</code>\n"
            f"• <b>Current Price</b>: <code>${current_price:,.2f}</code>\n"
            f"• <b>Unrealized PnL</b>: {pnl_emoji} <b>{sign}${pnl:,.2f}</b> ({sign}{pnl_pct:.2f}%)\n"
            f"• <b>Take Profit</b>: <code>${pos.take_profit_1:,.2f}</code>\n"
            f"• <b>Stop Loss</b>: <code>${pos.stop_loss:,.2f}</code>\n"
            f"• <b>Lot Size</b>: <code>{pos.lot_size}</code>\n"
            f"• <b>Strategy</b>: {pos.strategy}\n"
            f"• <b>Opened At</b>: {pos.entry_time}\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<i>💡 Use /autotrade close to close this position manually at market price.</i>"
        )

    def get_performance_report(self) -> str:
        """Generate comprehensive performance analytics and PnL breakdown."""
        total = len(self.closed_trades)
        if total == 0:
            return (
                "📈 <b>Trading Performance & PnL Report</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "• <b>Total Closed Trades</b>: 0\n"
                f"• <b>Current Equity</b>: <code>${self.config.equity:,.2f}</code>\n"
                f"• <b>Auto-Trading</b>: {'🟢 ENABLED' if self.config.enabled else '🔴 DISABLED'}\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>No closed trades recorded yet. Run /autotrade on to begin.</i>"
            )

        wins = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        win_rate = (len(wins) / total) * 100.0 if total > 0 else 0.0
        net_pnl = sum(t.pnl for t in self.closed_trades)
        win_sum = sum(t.pnl for t in wins)
        loss_sum = abs(sum(t.pnl for t in losses))
        profit_factor = (win_sum / loss_sum) if loss_sum > 0 else (99.0 if win_sum > 0 else 1.0)
        sign = "+" if net_pnl >= 0 else ""
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"

        recent_lines = []
        for t in self.closed_trades[-5:]:
            t_sign = "+" if t.pnl >= 0 else ""
            t_icon = "🟢" if t.pnl >= 0 else "🔴"
            recent_lines.append(
                f"  • {t.symbol} {t.direction}: {t_icon} {t_sign}${t.pnl:.2f} ({t.exit_reason or 'CLOSED'})"
            )
        recent_text = "\n".join(recent_lines)

        return (
            f"📈 <b>Trading Performance & PnL Report</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Total Trades</b>: <code>{total}</code> (Wins: {len(wins)} | Losses: {len(losses)})\n"
            f"• <b>Win Rate</b>: <b>{win_rate:.1f}%</b>\n"
            f"• <b>Realized Net PnL</b>: {pnl_emoji} <b>{sign}${net_pnl:,.2f}</b>\n"
            f"• <b>Profit Factor</b>: <code>{profit_factor:.2f}</code>\n"
            f"• <b>Account Equity</b>: <code>${self.config.equity:,.2f}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Recent Closed Trades:</b>\n"
            f"{recent_text}\n"
            "━━━━━━━━━━━━━━━━━━━━━━"
        )

    def analyze_market(self, symbol: str) -> str:
        """
        Runs full 10-indicator technical analysis engine on the specified symbol
        and generates an actionable technical report with confluence score,
        key indicator readings, and recommended trade setups.
        """
        cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
        alias_map = {
            "BTC": "BTCUSD",
            "BITCOIN": "BTCUSD",
            "BTCUSDT": "BTCUSD",
            "ETH": "ETHUSD",
            "ETHUSDT": "ETHUSD",
            "SOL": "SOLUSD",
            "SOLUSDT": "SOLUSD",
            "XRP": "XRPUSD",
            "XRPUSDT": "XRPUSD",
            "XAU": "XAUTUSD",
            "GOLD": "XAUTUSD",
            "PAXG": "XAUTUSD",
            "XAUUSD": "XAUTUSD",
        }
        target = alias_map.get(cleaned, cleaned)
        df1 = self.fetch_candles(target, count=120)
        if df1.empty:
            return f"⚠️ Unable to fetch market candles for <b>{target}</b>."

        eng = IndicatorEngine()
        snap = eng.compute(df1)
        curr_price = float(df1["close"].iloc[-1])
        recent_high = float(df1["high"].tail(20).max())
        recent_low = float(df1["low"].tail(20).min())

        score = snap.score
        if score >= 2.0:
            rec = "STRONG BUY 🟢"
            direction = "LONG"
            risk = max(curr_price - recent_low, curr_price * 0.005)
            stop = curr_price - risk
            tp1 = curr_price + (risk * 2.0)
            tp2 = curr_price + (risk * 3.5)
        elif score >= 0.7:
            rec = "BUY 🟢"
            direction = "LONG"
            risk = max(curr_price - recent_low, curr_price * 0.005)
            stop = curr_price - risk
            tp1 = curr_price + (risk * 2.0)
            tp2 = curr_price + (risk * 3.0)
        elif score <= -2.0:
            rec = "STRONG SELL 🔴"
            direction = "SHORT"
            risk = max(recent_high - curr_price, curr_price * 0.005)
            stop = curr_price + risk
            tp1 = curr_price - (risk * 2.0)
            tp2 = curr_price - (risk * 3.5)
        elif score <= -0.7:
            rec = "SELL 🔴"
            direction = "SHORT"
            risk = max(recent_high - curr_price, curr_price * 0.005)
            stop = curr_price + risk
            tp1 = curr_price - (risk * 2.0)
            tp2 = curr_price - (risk * 3.0)
        else:
            rec = "NEUTRAL / RANGE ⚪"
            direction = "HOLD"
            risk = curr_price * 0.008
            stop = curr_price - risk
            tp1 = curr_price + (risk * 2.0)
            tp2 = curr_price + (risk * 3.0)

        # Status strings
        if snap.rsi >= 70:
            rsi_txt = f"{snap.rsi:.1f} (Overbought ⚠️)"
        elif snap.rsi <= 30:
            rsi_txt = f"{snap.rsi:.1f} (Oversold 🟢)"
        else:
            rsi_txt = f"{snap.rsi:.1f} (Neutral ⚪)"

        if snap.ema9 > snap.ema21 > snap.ema50:
            ema_txt = "Bullish Alignment (9 > 21 > 50) 🟢"
        elif snap.ema9 < snap.ema21 < snap.ema50:
            ema_txt = "Bearish Alignment (9 < 21 < 50) 🔴"
        else:
            ema_txt = "Consolidation / Mixed ⚪"

        st_txt = "Bullish Uptrend 🟢" if snap.supertrend_dir == 1 else "Bearish Downtrend 🔴"
        macd_txt = "Bullish Cross 🟢" if snap.macd_hist > 0 else "Bearish Cross 🔴"
        adx_txt = f"{snap.adx:.1f} ({'Strong Trend' if snap.adx > 25 else 'Ranging'})"

        return (
            f"📊 <b>TECHNICAL ANALYSIS: {target}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Signal</b>: <b>{rec}</b>\n"
            f"• <b>Confluence Score</b>: <code>{score:+.2f} / 5.0</code>\n"
            f"• <b>Current Price</b>: <code>${curr_price:,.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Indicator Readings:</b>\n"
            f"• <b>RSI (14)</b>: {rsi_txt}\n"
            f"• <b>EMAs</b>: {ema_txt}\n"
            f"• <b>Supertrend</b>: {st_txt}\n"
            f"• <b>MACD Hist</b>: {macd_txt} (<code>{snap.macd_hist:+.2f}</code>)\n"
            f"• <b>ADX (14)</b>: <code>{adx_txt}</code>\n"
            f"• <b>Bollinger Bands</b>: <code>${snap.bb_lower:,.1f}</code> – <code>${snap.bb_upper:,.1f}</code>\n"
            f"• <b>Stochastic K/D</b>: <code>{snap.stoch_k:.1f} / {snap.stoch_d:.1f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Suggested Setup ({direction}):</b>\n"
            f"• <b>Entry Target</b>: <code>${curr_price:,.2f}</code>\n"
            f"• <b>Stop Loss</b>: <code>${stop:,.2f}</code>\n"
            f"• <b>Take Profit 1</b>: <code>${tp1:,.2f}</code>\n"
            f"• <b>Take Profit 2</b>: <code>${tp2:,.2f}</code>\n"
            f"• <b>Risk:Reward Ratio</b>: <code>2.0 : 1</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 Quick switch: /symbol {target} | Auto trade: /autotrade on</i>"
        )


_GLOBAL_AUTO_TRADER: Optional[AutoTrader] = None


def get_auto_trader() -> AutoTrader:
    global _GLOBAL_AUTO_TRADER
    if _GLOBAL_AUTO_TRADER is None:
        _GLOBAL_AUTO_TRADER = AutoTrader()
    return _GLOBAL_AUTO_TRADER


# ==================================================================
# 3. MULTI-STRATEGY RUNNER & CLI FUNCTIONS (INLINED RUNNER)
# ==================================================================


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_backtest(args):
    print("=" * 64)
    print("  MODE: BACKTEST  -  trading_strategy_indicators_pro.py")
    print("=" * 64)
    pro = load_module("pro_strat", PRO_FILE)

    sym = getattr(args, "symbol", "BTCUSD")
    dfs = pro.make_data(args.bars, symbol=sym)
    strat = pro.IndicatorsProStrategy(
        symbol=sym,
        risk_per_trade=args.risk,
        min_quality=args.min_q,
        min_indicator_score=2.0,
        ai_min_conf=0.66,
        min_rr=2.1,
        use_adx_filter=True,
        min_adx=20,
    )
    if getattr(args, "lotsize", None) is not None:
        strat.set_lot_size(args.lotsize, "fixed")
    if getattr(args, "tp", None) is not None and getattr(args, "sl", None) is not None:
        strat.set_tp_sl(
            args.tp,
            args.sl,
            tp_mode=getattr(args, "tp_mode", "rr"),
            sl_mode=getattr(args, "sl_mode", "swing"),
        )
    bt = pro.Backtester(strat, capital=args.equity)
    stats = bt.run(*dfs)
    bt.report(stats)
    return stats


def run_backtest_ai(args):
    print("=" * 64)
    print("  MODE: BACKTEST-AI  -  ai_bot_learning.py (demo data, temp memory)")
    print("=" * 64)
    ai = load_module("ai_strat", AI_FILE)

    tmp_mem = os.path.join(tempfile.gettempdir(), "ai_bot_smoke_memory.json")
    if os.path.exists(tmp_mem):
        os.remove(tmp_mem)

    sym = getattr(args, "symbol", "BTCUSD")
    bot = ai.AIBotLearning(memory_path=tmp_mem, symbol=sym)
    if getattr(args, "lotsize", None) is not None:
        bot.set_lot_size(args.lotsize, "fixed")
    if getattr(args, "tp", None) is not None and getattr(args, "sl", None) is not None:
        bot.set_tp_sl(
            args.tp,
            args.sl,
            tp_mode=getattr(args, "tp_mode", "rr"),
            sl_mode=getattr(args, "sl_mode", "swing"),
        )
    pro = load_module("pro_strat", PRO_FILE)  # reuse demo data generator
    dfs = pro.make_data(args.bars, symbol=sym)

    equity = args.equity
    peak = equity
    pos = None
    trades = 0
    wins = 0
    pnl_sum = 0.0
    curve = [equity]
    df1, df5, df15, df1h, daily = dfs
    bot.peak_eq = equity

    for i in range(300, len(df1) - 1):
        w1 = df1.iloc[: i + 1]
        ts = df1.index[i]
        w5 = df5[df5.index <= ts]
        w15 = df15[df15.index <= ts]
        w1h = df1h[df1h.index <= ts]
        wd = daily[daily.index <= ts]
        candle = df1.iloc[i]

        if pos is not None:
            ep, closed = bot.manage(pos, candle)
            if ep > 0:
                pnl = (
                    (ep - pos.entry)
                    if pos.direction == ai.Direction.LONG
                    else (pos.entry - ep)
                ) * pos.remaining
                equity += pnl
                bot.on_trade_closed(pos, ep, pnl)
                trades += 1
                wins += 1 if pnl > 0 else 0
                pnl_sum += pnl
                if closed:
                    pos = None

        if pos is None and len(wd) >= 2:
            sig = bot.generate_signal(
                w1,
                wd,
                w5 if len(w5) else None,
                w15 if len(w15) else None,
                w1h if len(w1h) else None,
            )
            if sig:
                size = bot.size(equity, sig.entry, sig.stop, sig.atr)
                if size > 0:
                    pos = ai.Position(
                        sig.direction,
                        sig.entry,
                        sig.stop,
                        size,
                        sig.tp1,
                        sig.tp2,
                        size,
                        sig.timestamp,
                        sig.setup.value,
                        abs(sig.entry - sig.stop),
                        sig.entry,
                        sig.entry,
                        0,
                        signal_meta={
                            "quality": sig.quality,
                            "regime": sig.regime,
                            "session": sig.session,
                            "rsi": sig.rsi,
                            "adx": sig.adx,
                            "ind_score": sig.ind_score,
                            "ai_conf": sig.ai_conf,
                            "ai_dir": sig.ai_dir.name,
                            "rel_vol": sig.rel_vol,
                        },
                        trail_mult=bot.trail_mult,
                    )
        peak = max(peak, equity)
        curve.append(equity)

    eq = np.array(curve)
    dd = float((np.maximum.accumulate(eq) - eq).max() / peak * 100)
    print("\n" + "=" * 64)
    print("   AI BOT LEARNING - DEMO BACKTEST REPORT")
    print("=" * 64)
    print(f"  trades   : {trades}")
    print(f"  winrate  : {wins/trades*100 if trades else 0:.1f}%")
    print(f"  net pnl  : {pnl_sum:.2f}")
    print(f"  return   : {(equity/args.equity-1)*100:.2f}%")
    print(f"  max dd   : {dd:.2f}%")
    print(f"  equity   : {equity:.2f}")
    print("=" * 64)
    print(bot.mem.report())
    return {"trades": trades, "equity": round(equity, 2)}


def run_live_ai(args):
    print("=" * 64)
    print("  MODE: LIVE  -  ai_bot_learning.py (AI Learning Bot)")
    print("=" * 64)
    ai = load_module("ai_strat", AI_FILE)
    os.environ.setdefault("BOT_SYMBOL", args.symbol)
    os.environ.setdefault("BOT_DATA_SOURCE", args.source)
    os.environ.setdefault("BOT_EXCHANGE", args.exchange)
    os.environ.setdefault("BOT_POLL", str(args.poll))
    os.environ.setdefault("BOT_EQUITY", str(args.equity))
    ai.auto_start_live()


class LiveProBot:
    def __init__(self, pro, ai_mod, args):
        self.pro = pro
        self.ai_mod = ai_mod
        self.args = args
        self.strat = pro.IndicatorsProStrategy(
            symbol=args.symbol,
            risk_per_trade=args.risk,
            min_quality=args.min_q,
            min_indicator_score=2.0,
            ai_min_conf=0.66,
            min_rr=2.1,
            use_adx_filter=True,
            min_adx=20,
        )
        if getattr(args, "lotsize", None) is not None:
            self.strat.set_lot_size(args.lotsize, "fixed")
        if getattr(args, "tp", None) is not None and getattr(args, "sl", None) is not None:
            self.strat.set_tp_sl(
                args.tp,
                args.sl,
                tp_mode=getattr(args, "tp_mode", "rr"),
                sl_mode=getattr(args, "sl_mode", "swing"),
            )
        self.feed = ai_mod.MarketDataFeed(
            symbol=args.symbol, source=args.source, exchange_id=args.exchange
        )
        self.equity = args.equity
        self.strat.peak_eq = args.equity
        self.pos = None
        self.running = False
        self.errors = 0

    def _open(self, sig):
        msg = (
            f"INDICATORS PRO SIGNAL\n{sig.direction.name} @ {sig.entry:.2f}\n"
            f"SL {sig.stop:.2f} | TP1 {sig.tp1:.2f}\nQ:{sig.quality:.1f} | {sig.reason}"
        )
        print(f"\n{'='*55}\n  {msg}\n{'='*55}")
        self.ai_mod.send_telegram(msg)
        size = self.strat.size(self.equity, sig.entry, sig.stop, sig.atr)
        if size > 0:
            self.pos = self.pro.Position(
                sig.direction,
                sig.entry,
                sig.stop,
                size,
                sig.tp1,
                sig.tp2,
                size,
                sig.timestamp,
                sig.setup.value,
                abs(sig.entry - sig.stop),
                sig.entry,
                sig.entry,
                0,
            )
            print(f"  Position opened | size {size:.4f}")

    def _close(self, exit_price, ts):
        pnl = (
            (exit_price - self.pos.entry)
            if self.pos.direction == self.pro.Direction.LONG
            else (self.pos.entry - exit_price)
        ) * self.pos.remaining
        self.equity += pnl
        self.strat.update(pnl)
        self.ai_mod.send_telegram(
            f"PRO Closed {self.pos.direction.name} | PnL {pnl:.2f} | Eq {self.equity:.2f}"
        )
        print(f"[Live-Pro] Closed | PnL {pnl:.2f} | Equity {self.equity:.2f}")

    def step(self):
        """One iteration. Returns False when the bot should stop."""
        data = self.feed.get_multi_tf(limit_1m=700)
        if not data or "1m" not in data or data["1m"].empty:
            print("[Live-Pro] No data, retry...")
            return True
        df1 = data["1m"]
        df5, df15, df1h = data.get("5m"), data.get("15m"), data.get("1h")
        daily = data.get("daily", pd.DataFrame())
        if len(daily) < 2:
            daily = (
                df1.resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                .dropna()
            )

        candle = df1.iloc[-1]
        if self.pos is not None:
            ep, closed = self.strat.manage(self.pos, candle)
            if ep > 0:
                self._close(ep, df1.index[-1])
                if closed:
                    self.pos = None

        if self.pos is None and len(daily) >= 2:
            sig = self.strat.generate_signal(
                df1,
                daily,
                df1,
                df5 if df5 is not None and len(df5) else None,
                df15 if df15 is not None and len(df15) else None,
                df1h if df1h is not None and len(df1h) else None,
            )
            if sig:
                self._open(sig)
        return True

    def run(self, once=False):
        print(
            f"\n[Live-Pro] Indicators Pro | {self.args.symbol} | {self.feed.source} "
            f"| poll {self.args.poll}s | equity {self.equity}"
        )
        print("[Live-Pro] Auto-running. Ctrl+C to stop.\n")
        self.running = True
        while self.running:
            try:
                self.step()
                self.errors = 0
                if once:
                    break
                time_module.sleep(self.args.poll)
            except KeyboardInterrupt:
                print("\n[Live-Pro] Stopped")
                self.running = False
            except Exception as e:
                self.errors += 1
                print(f"[Live-Pro] Error ({self.errors}): {e}")
                traceback.print_exc()
                time_module.sleep(min(120, 30 * self.errors))


def run_live_pro(args):
    print("=" * 64)
    print("  MODE: LIVE  -  trading_strategy_indicators_pro.py (Indicators Pro)")
    print("=" * 64)
    pro = load_module("pro_strat", PRO_FILE)
    ai_mod = load_module("ai_strat", AI_FILE)
    bot = LiveProBot(pro, ai_mod, args)
    bot.run(once=args.once)


# ==================================================================
# 4. MAIN CLI PARSER & RUNNER
# ==================================================================


def main():
    p = argparse.ArgumentParser(
        description="Auto Trade Runner - runs both strategy files"
    )
    p.add_argument(
        "mode",
        choices=["backtest", "backtest-ai", "live", "both"],
        help="backtest | backtest-ai | live | both",
    )
    p.add_argument(
        "--strategy",
        choices=["ai", "pro"],
        default="ai",
        help="Which strategy for live mode (default: ai)",
    )
    p.add_argument("--symbol", default=os.getenv("BOT_SYMBOL", "XAUUSD"))
    p.add_argument(
        "--source",
        default=os.getenv("BOT_DATA_SOURCE", "demo"),
        help="demo | binance | ccxt | yfinance | mt5 | delta",
    )
    p.add_argument("--exchange", default=os.getenv("BOT_EXCHANGE", "binance"))
    p.add_argument("--poll", type=int, default=int(os.getenv("BOT_POLL", "30")))
    p.add_argument(
        "--equity",
        type=float,
        default=float(os.getenv("BOT_EQUITY", "10000")),
    )
    p.add_argument("--bars", type=int, default=5000, help="demo bars for backtest")
    p.add_argument("--risk", type=float, default=0.17, help="risk %% per trade")
    p.add_argument("--min-q", type=float, default=4.6, help="min signal quality")
    p.add_argument(
        "--once", action="store_true", help="live mode: run one iteration then exit"
    )
    p.add_argument(
        "--lotsize", type=float, default=None, help="lot size for trades"
    )
    p.add_argument("--tp", type=float, default=None, help="Take Profit value")
    p.add_argument("--sl", type=float, default=None, help="Stop Loss value")
    p.add_argument(
        "--tp-mode", default="rr", help="TP mode: rr | pts | pct | atr"
    )
    p.add_argument(
        "--sl-mode", default="swing", help="SL mode: swing | pts | pct | atr"
    )
    args = p.parse_args()

    for f in (AI_FILE, PRO_FILE):
        if not os.path.exists(f):
            sys.exit(f"[Runner] Missing file: {f}  (keep all files in workspace)")

    print(f"[Runner] Loaded: {os.path.basename(AI_FILE)}")
    print(f"[Runner] Loaded: {os.path.basename(PRO_FILE)}")

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "backtest-ai":
        run_backtest_ai(args)
    elif args.mode == "live":
        run_live_pro(args) if args.strategy == "pro" else run_live_ai(args)
    elif args.mode == "both":
        run_backtest(args)
        print("\n[Runner] Backtest done. Starting AI live bot...\n")
        run_live_ai(args)


if __name__ == "__main__":
    main()
