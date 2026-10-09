"""
PRACTICAL RISK MANAGER – AUTOTRADE READY
=======================================
Risk + lot-size module for automated trading.

Features:
- Auto-adjust risk (drawdown, loss streak, expectancy)
- RF (Risk Factor) auto-adjust – scales risk & break-even thresholds
- Progressive auto break-even (lock profit → full BE → trail)
- Minimum risk floor + maximum risk cap
- Lot size calculation (standard / mini / micro)
- Instrument profiles (XAUUSD, Forex, Crypto)
- Practical TP ladder (1.5R / 2.5R / 4R)
- Partial TP + breakeven + trailing
- Daily & weekly loss limits
- No max-trades limit
- Autotrade helpers: can_trade, size, lots, TPs, trail, BE, update

Educational only. Not financial advice.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple
from enum import Enum
from datetime import date
import json
import os
import math


# ============================================================
# INSTRUMENT PROFILES (for lot sizing)
# ============================================================

INSTRUMENTS = {
    "XAUUSD":  {"contract_size": 100, "pip_size": 0.01, "pip_value_per_lot": 1.0, "min_lot": 0.01, "lot_step": 0.01},
    "XAUTUSD": {"contract_size": 100, "pip_size": 0.01, "pip_value_per_lot": 1.0, "min_lot": 0.01, "lot_step": 0.01},
    "GOLD":    {"contract_size": 100, "pip_size": 0.01, "pip_value_per_lot": 1.0, "min_lot": 0.01, "lot_step": 0.01},
    "GC=F":    {"contract_size": 100, "pip_size": 0.01, "pip_value_per_lot": 1.0, "min_lot": 0.01, "lot_step": 0.01},
    "PAXGUSDT":{"contract_size": 100, "pip_size": 0.01, "pip_value_per_lot": 1.0, "min_lot": 0.01, "lot_step": 0.01},
    "EURUSD":  {"contract_size": 100000, "pip_size": 0.0001, "pip_value_per_lot": 10.0, "min_lot": 0.01, "lot_step": 0.01},
    "GBPUSD":  {"contract_size": 100000, "pip_size": 0.0001, "pip_value_per_lot": 10.0, "min_lot": 0.01, "lot_step": 0.01},
    "USDJPY":  {"contract_size": 100000, "pip_size": 0.01,   "pip_value_per_lot": 9.0,  "min_lot": 0.01, "lot_step": 0.01},
    "AUDUSD":  {"contract_size": 100000, "pip_size": 0.0001, "pip_value_per_lot": 10.0, "min_lot": 0.01, "lot_step": 0.01},
    "USDCAD":  {"contract_size": 100000, "pip_size": 0.0001, "pip_value_per_lot": 10.0, "min_lot": 0.01, "lot_step": 0.01},
    "USDCHF":  {"contract_size": 100000, "pip_size": 0.0001, "pip_value_per_lot": 10.0, "min_lot": 0.01, "lot_step": 0.01},
    "BTCUSD":  {"contract_size": 1, "pip_size": 0.1, "pip_value_per_lot": 0.1, "min_lot": 0.001, "lot_step": 0.001},
    "BTCUSDT": {"contract_size": 1, "pip_size": 0.1, "pip_value_per_lot": 0.1, "min_lot": 0.001, "lot_step": 0.001},
    "ETHUSD":  {"contract_size": 1, "pip_size": 0.01, "pip_value_per_lot": 0.01, "min_lot": 0.01, "lot_step": 0.01},
    "ETHUSDT": {"contract_size": 1, "pip_size": 0.01, "pip_value_per_lot": 0.01, "min_lot": 0.01, "lot_step": 0.01},
    "SOLUSD":  {"contract_size": 1, "pip_size": 0.01, "pip_value_per_lot": 0.01, "min_lot": 0.01, "lot_step": 0.01},
    "SOLUSDT": {"contract_size": 1, "pip_size": 0.01, "pip_value_per_lot": 0.01, "min_lot": 0.01, "lot_step": 0.01},
    "XRPUSD":  {"contract_size": 1, "pip_size": 0.0001, "pip_value_per_lot": 0.0001, "min_lot": 0.1, "lot_step": 0.1},
    "XRPUSDT": {"contract_size": 1, "pip_size": 0.0001, "pip_value_per_lot": 0.0001, "min_lot": 0.1, "lot_step": 0.1},
    "XAUUSDT": {"contract_size": 1, "pip_size": 0.01, "pip_value_per_lot": 0.01, "min_lot": 0.01, "lot_step": 0.01},
    "DEFAULT": {"contract_size": 100000, "pip_size": 0.0001, "pip_value_per_lot": 10.0, "min_lot": 0.01, "lot_step": 0.01},
}


class RiskMode(Enum):
    NORMAL = "normal"
    REDUCED = "reduced"
    RECOVERY = "recovery"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskConfig:
    base_risk_pct: float = 0.25
    min_risk_pct: float = 0.10
    max_risk_pct: float = 0.50
    max_daily_loss_pct: float = 1.5
    max_weekly_loss_pct: float = 4.0
    max_consecutive_losses: int = 3
    tp1_r: float = 1.5
    tp2_r: float = 2.5
    tp3_r: float = 4.0
    tp1_close_pct: float = 0.50
    tp2_close_pct: float = 0.30
    trail_after_tp1: bool = True
    trail_atr_mult: float = 2.0
    trail_r_mult: float = 1.5
    # Auto break-even (progressive)
    auto_be: bool = True
    be_at_r: float = 1.0          # move SL to exact BE when price reaches +1R
    be_lock_r: float = 0.5        # early lock: move SL to +0.1R when price reaches +0.5R
    be_lock_offset_r: float = 0.1 # small profit locked at early stage
    be_buffer_r: float = 0.05     # buffer past entry so BE is slightly in profit
    dd_reduce_at: float = 0.05
    dd_reduce_factor: float = 0.55
    dd_hard_at: float = 0.12
    dd_hard_factor: float = 0.30
    win_streak_boost: float = 1.12
    max_boost_streak: int = 3
    expectancy_boost_above: float = 0.25
    expectancy_cut_below: float = 0.05
    min_rr: float = 1.5
    max_lot: float = 10.0
    min_lot_override: Optional[float] = None
    # RF (Risk Factor) – master multiplier for risk & BE sensitivity
    # 1.0 = normal | <1 safer | >1 more aggressive
    rf: float = 1.0
    rf_min: float = 0.50
    rf_max: float = 1.50
    rf_auto: bool = True              # auto-adjust RF from performance
    rf_up_on_exp: float = 0.05        # raise RF when expectancy is good
    rf_down_on_loss: float = 0.08     # lower RF after consecutive losses


@dataclass
class TradeRiskResult:
    size: float
    lots: float
    risk_amount: float
    risk_pct_used: float
    stop_distance: float
    stop_pips: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    mode: str
    reason: str
    symbol: str = ""
    pip_value: float = 0.0


@dataclass
class RiskState:
    equity: float = 10000.0
    peak_equity: float = 10000.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    win_streak: int = 0
    last_day: Optional[str] = None
    last_week: Optional[str] = None
    expectancy: float = 0.0
    recent_winrate: float = 0.5
    total_trades: int = 0
    total_wins: int = 0
    mode: str = "normal"
    rf: float = 1.0


class RiskManager:
    def __init__(
        self,
        equity: float = 10000.0,
        symbol: str = "XAUUSD",
        config: Optional[RiskConfig] = None,
        state_file: str = "risk_state.json",
    ):
        self.config = config or RiskConfig()
        self.symbol = symbol.upper()
        if not os.path.isabs(state_file):
            state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), state_file)
        self.state_file = state_file
        self.state = RiskState(equity=equity, peak_equity=equity)
        self.profile = INSTRUMENTS.get(self.symbol, INSTRUMENTS["DEFAULT"])
        self._load_state()
        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        if not getattr(self.state, "rf", None):
            self.state.rf = self.config.rf

    def set_symbol(self, symbol: str):
        self.symbol = symbol.upper()
        self.profile = INSTRUMENTS.get(self.symbol, INSTRUMENTS["DEFAULT"])

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
            except Exception:
                pass

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def _check_period_reset(self):
        today = date.today().isoformat()
        week = date.today().strftime("%Y-W%W")
        if self.state.last_day != today:
            self.state.last_day = today
            self.state.daily_pnl = 0.0
        if self.state.last_week != week:
            self.state.last_week = week
            self.state.weekly_pnl = 0.0

    def current_risk_pct(self) -> Tuple[float, str]:
        self._check_period_reset()
        cfg = self.config
        s = self.state
        risk = cfg.base_risk_pct
        reasons = []

        if s.peak_equity > 0:
            dd = (s.peak_equity - s.equity) / s.peak_equity
            if dd >= cfg.dd_hard_at:
                risk *= cfg.dd_hard_factor
                reasons.append(f"hard-DD {dd:.1%}")
                s.mode = RiskMode.RECOVERY.value
            elif dd >= cfg.dd_reduce_at:
                risk *= cfg.dd_reduce_factor
                reasons.append(f"soft-DD {dd:.1%}")
                s.mode = RiskMode.REDUCED.value
            else:
                s.mode = RiskMode.NORMAL.value

        if s.consecutive_losses >= 2:
            risk *= 0.70
            reasons.append(f"{s.consecutive_losses} losses")
            s.mode = RiskMode.RECOVERY.value

        if s.win_streak > 0 and s.mode == RiskMode.NORMAL.value:
            boost = cfg.win_streak_boost ** min(s.win_streak, cfg.max_boost_streak)
            risk *= boost
            reasons.append(f"streak x{boost:.2f}")

        if s.total_trades >= 3:
            if s.expectancy >= cfg.expectancy_boost_above and s.mode == RiskMode.NORMAL.value:
                risk *= 1.10
                reasons.append("good-exp")
                s.mode = RiskMode.AGGRESSIVE.value
            elif s.expectancy < cfg.expectancy_cut_below:
                risk *= 0.75
                reasons.append("low-exp")

        # Apply RF (Risk Factor)
        rf = getattr(s, "rf", cfg.rf) if cfg.rf_auto else cfg.rf
        rf = max(cfg.rf_min, min(cfg.rf_max, rf))
        s.rf = rf
        risk *= rf
        if abs(rf - 1.0) > 0.01:
            reasons.append(f"RF×{rf:.2f}")

        risk = max(cfg.min_risk_pct, min(cfg.max_risk_pct, risk))
        return round(risk, 4), (" | ".join(reasons) if reasons else "base")

    def can_trade(self) -> Tuple[bool, str]:
        self._check_period_reset()
        s = self.state
        cfg = self.config
        if s.consecutive_losses >= cfg.max_consecutive_losses:
            return False, f"{s.consecutive_losses} consecutive losses – pause"
        if s.equity > 0 and (-s.daily_pnl / s.equity * 100) >= cfg.max_daily_loss_pct:
            return False, "daily loss limit hit"
        if s.equity > 0 and (-s.weekly_pnl / s.equity * 100) >= cfg.max_weekly_loss_pct:
            return False, "weekly loss limit hit"
        return True, "ok"

    def calc_lots(self, risk_amount: float, stop_distance: float) -> Tuple[float, float, float]:
        p = self.profile
        pip_size = p["pip_size"]
        pip_value = p["pip_value_per_lot"]
        min_lot = self.config.min_lot_override or p["min_lot"]
        step = p["lot_step"]
        max_lot = self.config.max_lot

        if stop_distance <= 0 or pip_size <= 0:
            return 0.0, 0.0, pip_value

        stop_pips = stop_distance / pip_size
        if stop_pips <= 0 or pip_value <= 0:
            return 0.0, stop_pips, pip_value

        lots = risk_amount / (stop_pips * pip_value)
        lots = math.floor(lots / step) * step
        lots = max(min_lot, min(max_lot, lots))
        if lots < min_lot:
            lots = 0.0
        return round(lots, 4), round(stop_pips, 2), pip_value

    def position_size(
        self,
        entry: float,
        stop: float,
        equity: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> TradeRiskResult:
        if symbol:
            self.set_symbol(symbol)
        if equity is not None:
            self.state.equity = equity
            self.state.peak_equity = max(self.state.peak_equity, equity)

        ok, why = self.can_trade()
        if not ok:
            return TradeRiskResult(
                size=0.0, lots=0.0, risk_amount=0.0, risk_pct_used=0.0,
                stop_distance=0.0, stop_pips=0.0, tp1=0.0, tp2=0.0, tp3=None,
                mode=self.state.mode, reason=f"blocked: {why}", symbol=self.symbol
            )

        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return TradeRiskResult(
                size=0.0, lots=0.0, risk_amount=0.0, risk_pct_used=0.0,
                stop_distance=0.0, stop_pips=0.0, tp1=0.0, tp2=0.0, tp3=None,
                mode=self.state.mode, reason="invalid stop", symbol=self.symbol
            )

        risk_pct, reason = self.current_risk_pct()
        risk_amount = self.state.equity * (risk_pct / 100.0)
        lots, stop_pips, pip_value = self.calc_lots(risk_amount, stop_dist)
        unit_size = risk_amount / stop_dist if stop_dist > 0 else 0.0
        direction = "long" if entry > stop else "short"
        tps = self.take_profits(entry, stop, direction)

        return TradeRiskResult(
            size=round(unit_size, 6),
            lots=lots,
            risk_amount=round(risk_amount, 2),
            risk_pct_used=risk_pct,
            stop_distance=round(stop_dist, 6),
            stop_pips=stop_pips,
            tp1=tps["tp1"],
            tp2=tps["tp2"],
            tp3=tps.get("tp3"),
            mode=self.state.mode,
            reason=reason,
            symbol=self.symbol,
            pip_value=pip_value,
        )

    def take_profits(self, entry: float, stop: float, direction: str = "long") -> Dict[str, float]:
        risk = abs(entry - stop)
        cfg = self.config
        if direction.lower() in ("long", "buy", "1"):
            return {
                "tp1": round(entry + risk * cfg.tp1_r, 6),
                "tp2": round(entry + risk * cfg.tp2_r, 6),
                "tp3": round(entry + risk * cfg.tp3_r, 6),
                "tp1_r": cfg.tp1_r, "tp2_r": cfg.tp2_r, "tp3_r": cfg.tp3_r,
                "tp1_close_pct": cfg.tp1_close_pct, "tp2_close_pct": cfg.tp2_close_pct,
            }
        return {
            "tp1": round(entry - risk * cfg.tp1_r, 6),
            "tp2": round(entry - risk * cfg.tp2_r, 6),
            "tp3": round(entry - risk * cfg.tp3_r, 6),
            "tp1_r": cfg.tp1_r, "tp2_r": cfg.tp2_r, "tp3_r": cfg.tp3_r,
            "tp1_close_pct": cfg.tp1_close_pct, "tp2_close_pct": cfg.tp2_close_pct,
        }

    def meets_min_rr(self, entry: float, stop: float, tp: float) -> bool:
        risk = abs(entry - stop)
        reward = abs(tp - entry)
        if risk <= 0:
            return False
        return (reward / risk) >= self.config.min_rr

    def auto_breakeven_stop(
        self,
        direction: str,
        entry: float,
        current_stop: float,
        highest: float,
        lowest: float,
        initial_risk: float,
    ) -> Tuple[float, str]:
        """
        Progressive auto break-even:
          1) At +be_lock_r (default 0.5R) → lock small profit (entry ± be_lock_offset_r)
          2) At +be_at_r (default 1.0R)   → full BE with small buffer
        Returns (new_stop, reason). Never loosens the stop.
        """
        if not self.config.auto_be or initial_risk <= 0:
            return current_stop, ""

        cfg = self.config
        is_long = direction.lower() in ("long", "buy", "1")
        # Favourable excursion in R
        if is_long:
            mfe = (highest - entry) / initial_risk
        else:
            mfe = (entry - lowest) / initial_risk

        # RF scales BE sensitivity: higher RF → earlier BE (more protective when aggressive)
        rf = getattr(self.state, "rf", cfg.rf)
        rf = max(cfg.rf_min, min(cfg.rf_max, rf))
        be_at = cfg.be_at_r / max(rf, 0.5)          # aggressive RF → BE sooner
        be_lock = cfg.be_lock_r / max(rf, 0.5)

        new_stop = current_stop
        reason = ""

        # Stage 2: full breakeven (+ buffer) when MFE >= be_at (RF-adjusted)
        if mfe >= be_at:
            if is_long:
                target = entry + initial_risk * cfg.be_buffer_r
                if target > new_stop:
                    new_stop = target
                    reason = f"Auto BE @ {mfe:.2f}R (full)"
            else:
                target = entry - initial_risk * cfg.be_buffer_r
                if target < new_stop:
                    new_stop = target
                    reason = f"Auto BE @ {mfe:.2f}R (full)"

        # Stage 1: early lock when MFE >= be_lock_r (only if not already past full BE)
        elif mfe >= be_lock:
            if is_long:
                target = entry + initial_risk * cfg.be_lock_offset_r
                if target > new_stop:
                    new_stop = target
                    reason = f"Auto BE lock @ {mfe:.2f}R (+{cfg.be_lock_offset_r}R)"
            else:
                target = entry - initial_risk * cfg.be_lock_offset_r
                if target < new_stop:
                    new_stop = target
                    reason = f"Auto BE lock @ {mfe:.2f}R (+{cfg.be_lock_offset_r}R)"

        return new_stop, reason

    def trailing_stop(
        self, direction: str, entry: float, current_stop: float,
        highest: float, lowest: float, initial_risk: float,
        atr: Optional[float] = None, tp1_hit: bool = False,
    ) -> float:
        # Always apply progressive auto-BE first
        be_stop, _ = self.auto_breakeven_stop(
            direction, entry, current_stop, highest, lowest, initial_risk
        )
        current_stop = be_stop

        if not tp1_hit and not self.config.trail_after_tp1:
            return current_stop

        cfg = self.config
        dist = (atr * cfg.trail_atr_mult) if (atr and atr > 0) else (initial_risk * cfg.trail_r_mult)
        if direction.lower() in ("long", "buy", "1"):
            new_stop = highest - dist
            # After TP1, also respect BE level
            floor = entry if tp1_hit else current_stop
            return max(current_stop, floor, new_stop)
        new_stop = lowest + dist
        ceiling = entry if tp1_hit else current_stop
        return min(current_stop, ceiling, new_stop)

    def manage_position(
        self, direction: str, entry: float, stop: float, tp1: float, tp2: Optional[float],
        remaining_pct: float, highest: float, lowest: float, high: float, low: float,
        initial_risk: float, atr: Optional[float] = None, partials_done: int = 0,
    ) -> Dict:
        actions = {
            "close_pct": 0.0, "exit_price": 0.0, "new_stop": stop,
            "partials_done": partials_done, "fully_closed": False, "reason": "",
            "highest": max(highest, high), "lowest": min(lowest, low),
            "be_applied": False,
        }
        is_long = direction.lower() in ("long", "buy", "1")
        highest = actions["highest"]
        lowest = actions["lowest"]

        # --- Progressive auto break-even (runs every tick/candle) ---
        be_stop, be_reason = self.auto_breakeven_stop(
            direction, entry, stop, highest, lowest, initial_risk
        )
        if be_reason:
            stop = be_stop
            actions["new_stop"] = stop
            actions["be_applied"] = True
            actions["reason"] = be_reason

        # TP1
        if partials_done == 0:
            hit = (high >= tp1) if is_long else (low <= tp1)
            if hit:
                actions["close_pct"] = self.config.tp1_close_pct
                actions["exit_price"] = tp1
                # Move to BE with buffer on TP1
                buf = initial_risk * self.config.be_buffer_r
                actions["new_stop"] = (entry + buf) if is_long else (entry - buf)
                actions["partials_done"] = 1
                actions["be_applied"] = True
                actions["reason"] = "TP1 hit → partial + auto BE"
                return actions

        # TP2
        if partials_done == 1 and tp2 is not None:
            hit = (high >= tp2) if is_long else (low <= tp2)
            if hit:
                actions["close_pct"] = self.config.tp2_close_pct
                actions["exit_price"] = tp2
                actions["partials_done"] = 2
                actions["new_stop"] = self.trailing_stop(
                    direction, entry, stop, highest, lowest, initial_risk, atr, True
                )
                actions["reason"] = "TP2 hit → partial + trail"
                return actions

        # Trail after TP1 / after full BE
        if partials_done >= 1 or actions["be_applied"]:
            stop = self.trailing_stop(
                direction, entry, stop, highest, lowest, initial_risk, atr,
                tp1_hit=(partials_done >= 1),
            )
            actions["new_stop"] = stop

        # Stop hit
        hit_stop = (low <= stop) if is_long else (high >= stop)
        if hit_stop:
            actions["close_pct"] = remaining_pct
            actions["exit_price"] = stop
            actions["fully_closed"] = True
            actions["reason"] = "Stop hit"
            return actions

        actions["new_stop"] = stop
        return actions

    def update_after_trade(self, pnl: float, r_multiple: float = 0.0):
        self._check_period_reset()
        s = self.state
        s.equity += pnl
        s.peak_equity = max(s.peak_equity, s.equity)
        s.daily_pnl += pnl
        s.weekly_pnl += pnl
        s.total_trades += 1
        if pnl > 0:
            s.consecutive_losses = 0
            s.win_streak += 1
            s.total_wins += 1
        else:
            s.consecutive_losses += 1
            s.win_streak = 0
        if s.total_trades > 0:
            s.recent_winrate = s.total_wins / s.total_trades
            s.expectancy = (r_multiple * 0.3 + s.expectancy * 0.7) if s.expectancy else r_multiple * 0.5

        # Auto-adjust RF (Risk Factor)
        cfg = self.config
        if cfg.rf_auto:
            rf = getattr(s, "rf", cfg.rf)
            if pnl > 0 and s.expectancy >= cfg.expectancy_boost_above:
                rf = min(cfg.rf_max, rf + cfg.rf_up_on_exp)
            elif s.consecutive_losses >= 2:
                rf = max(cfg.rf_min, rf - cfg.rf_down_on_loss)
            elif pnl < 0:
                rf = max(cfg.rf_min, rf - cfg.rf_down_on_loss * 0.5)
            s.rf = round(rf, 3)

        self._save_state()

    def autotrade_plan(
        self, entry: float, stop: float, direction: str = "long",
        equity: Optional[float] = None, symbol: Optional[str] = None,
    ) -> Dict:
        res = self.position_size(entry, stop, equity=equity, symbol=symbol)
        if res.lots <= 0 and res.size <= 0:
            return {"ok": False, "reason": res.reason, "lots": 0.0, "size": 0.0}
        if not self.meets_min_rr(entry, stop, res.tp1):
            return {"ok": False, "reason": f"TP1 RR < {self.config.min_rr}", "lots": 0.0, "size": 0.0}
        return {
            "ok": True,
            "symbol": res.symbol,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "lots": res.lots,
            "size": res.size,
            "risk_amount": res.risk_amount,
            "risk_pct": res.risk_pct_used,
            "stop_pips": res.stop_pips,
            "tp1": res.tp1,
            "tp2": res.tp2,
            "tp3": res.tp3,
            "tp1_close_pct": self.config.tp1_close_pct,
            "tp2_close_pct": self.config.tp2_close_pct,
            "mode": res.mode,
            "reason": res.reason,
        }

    def summary(self) -> str:
        risk, reason = self.current_risk_pct()
        ok, why = self.can_trade()
        s = self.state
        dd = (s.peak_equity - s.equity) / s.peak_equity * 100 if s.peak_equity else 0
        rf = getattr(s, "rf", self.config.rf)
        return (
            f"Eq:{s.equity:.2f} Peak:{s.peak_equity:.2f} DD:{dd:.1f}% | "
            f"Risk:{risk:.2f}% RF×{rf:.2f} ({reason}) Mode:{s.mode} | "
            f"DayPnL:{s.daily_pnl:.2f} | CanTrade:{ok} ({why}) | Symbol:{self.symbol}"
        )

    def get_risk_dashboard(self) -> str:
        """Format an HTML dashboard for Telegram displaying real-time risk manager metrics."""
        risk_pct, reason = self.current_risk_pct()
        ok, can_trade_why = self.can_trade()
        s = self.state
        cfg = self.config
        dd = (s.peak_equity - s.equity) / s.peak_equity * 100.0 if s.peak_equity > 0 else 0.0
        rf = getattr(s, "rf", cfg.rf)
        status_icon = "🟢 <b>ACTIVE / READY</b>" if ok else f"🔴 <b>PAUSED</b> ({can_trade_why})"
        mode_badge = s.mode.upper()

        risk_dollars = s.equity * (risk_pct / 100.0)

        return (
            f"🛡️ <b>PRACTICAL RISK MANAGER DASHBOARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Status</b>: {status_icon}\n"
            f"• <b>Risk Mode</b>: <code>{mode_badge}</code> (RF ×{rf:.2f})\n"
            f"• <b>Current Equity</b>: <code>${s.equity:,.2f}</code> (Peak: <code>${s.peak_equity:,.2f}</code>)\n"
            f"• <b>Current Drawdown</b>: <code>{dd:.1f}%</code>\n"
            f"• <b>Per-Trade Risk</b>: <code>{risk_pct:.2f}%</code> (${risk_dollars:,.2f}) [{reason}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Performance & Safety Guard:</b>\n"
            f"• <b>Consecutive Losses</b>: <code>{s.consecutive_losses}</code> / {cfg.max_consecutive_losses} max\n"
            f"• <b>Win Streak</b>: <code>{s.win_streak}</code> | Total Trades: <code>{s.total_trades}</code>\n"
            f"• <b>Recent Expectancy</b>: <code>{s.expectancy:+.2f}R</code>\n"
            f"• <b>Today PnL</b>: <code>${s.daily_pnl:+,.2f}</code> (Max Loss: {cfg.max_daily_loss_pct}%)\n"
            f"• <b>Weekly PnL</b>: <code>${s.weekly_pnl:+,.2f}</code> (Max Loss: {cfg.max_weekly_loss_pct}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Auto Break-Even & Laddering:</b>\n"
            f"• <b>Progressive Auto-BE</b>: {'🟢 ON' if cfg.auto_be else '⚪ OFF'}\n"
            f"  Stage 1: Lock +{cfg.be_lock_offset_r}R at +{cfg.be_lock_r}R MFE\n"
            f"  Stage 2: Full BE at +{cfg.be_at_r}R MFE\n"
            f"• <b>TP Targets</b>: TP1 ({cfg.tp1_r}R) | TP2 ({cfg.tp2_r}R) | TP3 ({cfg.tp3_r}R)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 Commands: /risk [PCT] | /lotsize [auto|manual]</i>"
        )


def lots_for_risk(equity: float, risk_pct: float, entry: float, stop: float, symbol: str = "XAUUSD") -> float:
    profile = INSTRUMENTS.get(symbol.upper(), INSTRUMENTS["DEFAULT"])
    risk_amount = equity * (risk_pct / 100.0)
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return 0.0
    stop_pips = stop_dist / profile["pip_size"]
    if stop_pips <= 0 or profile["pip_value_per_lot"] <= 0:
        return 0.0
    lots = risk_amount / (stop_pips * profile["pip_value_per_lot"])
    step = profile["lot_step"]
    lots = math.floor(lots / step) * step
    return max(profile["min_lot"], min(10.0, round(lots, 4)))


if __name__ == "__main__":
    print("=" * 62)
    print("  RISK MANAGER – Auto BE + RF + Lot Size")
    print("=" * 62)

    rm = RiskManager(equity=10000.0, symbol="XAUUSD")
    plan = rm.autotrade_plan(entry=2650.0, stop=2640.0, direction="long")
    print("\n--- Autotrade Plan (XAUUSD) ---")
    for k, v in plan.items():
        print(f"  {k:<16}: {v}")

    print("\n--- State ---")
    print(rm.summary())

    # Demo progressive auto break-even
    print("\n--- Auto Break-Even demo (long, risk=10) ---")
    entry, stop, risk = 2650.0, 2640.0, 10.0
    # price moves to +0.5R
    be1, r1 = rm.auto_breakeven_stop("long", entry, stop, highest=2655.0, lowest=2640.0, initial_risk=risk)
    print(f"  At +0.5R (high=2655): stop → {be1:.2f}  [{r1}]")
    # price moves to +1.0R
    be2, r2 = rm.auto_breakeven_stop("long", entry, be1, highest=2660.0, lowest=2640.0, initial_risk=risk)
    print(f"  At +1.0R (high=2660): stop → {be2:.2f}  [{r2}]")

    rm.update_after_trade(pnl=37.5, r_multiple=1.5)
    print("\n--- After +1.5R win (RF may rise) ---")
    print(rm.summary())

    rm.update_after_trade(pnl=-25.0, r_multiple=-1.0)
    rm.update_after_trade(pnl=-25.0, r_multiple=-1.0)
    print("\n--- After 2 losses (RF & risk auto-reduced) ---")
    print(rm.summary())
    plan2 = rm.autotrade_plan(entry=2650.0, stop=2640.0)
    print(f"  New lots: {plan2.get('lots')} | risk %: {plan2.get('risk_pct')} | {plan2.get('reason')}")

    print("\nRisk manager ready for autotrade.")
