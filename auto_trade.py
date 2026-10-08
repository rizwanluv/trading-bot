"""
Auto Trade Module for Delta Exchange & Telegram Bot
===================================================
Provides automated strategy execution, dynamic ON/OFF controls,
configurable lot sizing, customizable TP/SL parameters, and live position tracking.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from trading_strategy_indicators_pro import Direction, IndicatorsProStrategy

logger = logging.getLogger("trading_bot.auto_trade")

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "auto_trade_config.json")
DELTA_CHART_API = os.getenv(
    "DELTA_CHART_API", "https://api.india.delta.exchange/v2/chart/history"
)
DELTA_TICKER_API = os.getenv(
    "DELTA_TICKER_API", "https://api.india.delta.exchange/v2/tickers"
)


@dataclass
class AutoTradeConfig:
    enabled: bool = False
    symbol: str = "XAUTUSD"
    lot_size: float = 0.01
    lot_mode: str = "fixed"  # "fixed" or "risk_pct"
    risk_pct: float = 1.0  # used if lot_mode is "risk_pct"
    tp_value: float = 2.0  # multiplier, points, or percentage
    sl_value: float = 1.0  # multiplier, points, or percentage
    tp_mode: str = "rr"  # "rr", "pts", "pct", "atr"
    sl_mode: str = "swing"  # "swing", "pts", "pct", "atr"
    strategy_type: str = "indicators_pro"  # "indicators_pro" or "ai_learning"
    poll_seconds: int = 15
    equity: float = 10000.0
    notify_chat_id: Optional[int] = None
    config_file: str = CONFIG_FILE_PATH

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


class AutoTrader:
    """
    Automated trading engine with live Delta Exchange market feed,
    strategy signal generation, position lifecycle management, and
    interactive configuration.
    """

    def __init__(self, config: Optional[AutoTradeConfig] = None):
        self.config = config or AutoTradeConfig.load()
        self.position: Optional[AutoTradePosition] = None
        self.closed_trades: List[AutoTradePosition] = []
        self.is_running: bool = False
        self._strategy_pro: Optional[IndicatorsProStrategy] = None
        self._init_strategy()

    def _init_strategy(self) -> None:
        self._strategy_pro = IndicatorsProStrategy(
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

    def toggle(self, state: Optional[bool] = None, chat_id: Optional[int] = None) -> Tuple[bool, str]:
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

        mode_desc = "Fixed lot size" if self.config.lot_mode == "fixed" else "Risk % lot sizing"
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
        self.config.symbol = cleaned
        self.config.save()
        return f"Auto trade symbol set to: {self.config.symbol}"

    def set_strategy_type(self, strategy_type: str) -> Tuple[bool, str]:
        st = strategy_type.strip().lower()
        if st not in ("indicators_pro", "ai_learning"):
            return False, "Unsupported strategy. Use 'indicators_pro' or 'ai_learning'."
        self.config.strategy_type = st
        self.config.save()
        return True, f"Strategy switched to: {st}"

    def fetch_candles(self, symbol: str, count: int = 120) -> pd.DataFrame:
        """Fetch 1m candle series from Delta Exchange API with fallback to synthetic data."""
        cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
        now = int(time.time())
        from_ts = now - (count + 30) * 60
        url = f"{DELTA_CHART_API}?symbol={cleaned}&resolution=1&from={from_ts}&to={now}"

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
            logger.warning("Delta candle fetch error for %s: %s", cleaned, exc)

        # Fallback synthetic series for offline resilience and tests
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=count, freq="1min")
        base = 2650.0 if "XAU" in cleaned else 80000.0
        p = base + pd.Series(range(count)) * 0.1
        return pd.DataFrame(
            {
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p + 0.2,
                "volume": 500.0,
            },
            index=dates,
        )

    def close_current_position(self, current_price: Optional[float] = None, reason: str = "MANUAL") -> Optional[str]:
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

        # 2. Check for New Entry if no active position and auto-trade is ON
        if self.position is None and self.config.enabled:
            daily = df1.resample("1D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(daily) < 2:
                # Synthetic daily support
                daily = pd.DataFrame(
                    [
                        {"open": curr_close - 5, "high": curr_close + 10, "low": curr_close - 10, "close": curr_close},
                        {"open": curr_close, "high": curr_close + 5, "low": curr_close - 5, "close": curr_close},
                    ],
                    index=pd.date_range(end=datetime.now(timezone.utc), periods=2, freq="1D"),
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
                        lots = self._strategy_pro.size(self.config.equity, entry, stop, sig.atr)

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
                        entry_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
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
        status_icon = "🟢 <b>ACTIVE (ON)</b>" if self.config.enabled else "🔴 <b>DISABLED (OFF)</b>"
        lot_mode_str = "Fixed" if self.config.lot_mode == "fixed" else f"Risk {self.config.risk_pct}%"

        text = [
            "⚡ <b>AUTO TRADING DASHBOARD</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"• <b>Status</b>: {status_icon}",
            f"• <b>Symbol</b>: {self.config.symbol}",
            f"• <b>Strategy</b>: {self.config.strategy_type.replace('_', ' ').title()}",
            f"• <b>Lot Size</b>: {self.config.lot_size} ({lot_mode_str})",
            f"• <b>Take Profit</b>: {self.config.tp_value} ({self.config.tp_mode.upper()})",
            f"• <b>Stop Loss</b>: {self.config.sl_value} ({self.config.sl_mode.upper()})",
            f"• <b>Account Equity</b>: ${self.config.equity:.2f}",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if self.position:
            pos = self.position
            text.extend([
                "📊 <b>Active Position:</b>",
                f"  • {pos.direction} {pos.symbol} @ {pos.entry_price:.2f}",
                f"  • Lot: {pos.lot_size}",
                f"  • TP: {pos.take_profit_1:.2f} | SL: {pos.stop_loss:.2f}",
                f"  • Opened: {pos.entry_time}",
            ])
        else:
            text.append("📊 <b>Active Position</b>: None")

        text.append("━━━━━━━━━━━━━━━━━━━━━━")
        total_closed = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in self.closed_trades)
        win_rate = (len(wins) / total_closed * 100) if total_closed > 0 else 0.0

        text.extend([
            "📈 <b>Trade Performance:</b>",
            f"  • Total Closed: {total_closed}",
            f"  • Win Rate: {win_rate:.1f}% ({len(wins)}/{total_closed})",
            f"  • Net PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}",
        ])

        return "\n".join(text)


_GLOBAL_AUTO_TRADER: Optional[AutoTrader] = None


def get_auto_trader() -> AutoTrader:
    global _GLOBAL_AUTO_TRADER
    if _GLOBAL_AUTO_TRADER is None:
        _GLOBAL_AUTO_TRADER = AutoTrader()
    return _GLOBAL_AUTO_TRADER
