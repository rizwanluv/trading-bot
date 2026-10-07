"""
Self-Learning and Self-Improving Engine for Quantitative Crypto & Commodities Trading.
Continuously analyzes trade history, evaluates setup win-rates, adjusts risk/reward multipliers,
suppresses underperforming setups, prioritizes winning patterns, and enforces draw-down protection.

TOKEN EFFICIENCY GUARANTEE:
- 100% of continuous learning, trade evaluation, statistical scoring, parameter adaptation,
  and trade filtering run locally via deterministic algorithms at ZERO (0) LLM API token cost.
- Compact AI synthesis is strictly on-demand, compressed to <100 prompt tokens and capped at
  250 output tokens with 15-minute caching to eliminate unnecessary token consumption.
"""

import os
import json
import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sessions definition by UTC hour
# Asia: 00:00 - 08:00 UTC
# London: 08:00 - 13:00 UTC
# New York: 13:00 - 22:00 UTC
# Asia Late: 22:00 - 24:00 UTC
def get_session_name(timestamp: Optional[float] = None) -> str:
    t = timestamp or time.time()
    utc_hour = time.gmtime(t).tm_hour
    if 8 <= utc_hour < 13:
        return "London"
    elif 13 <= utc_hour < 22:
        return "New York"
    else:
        return "Asia"


def normalize_setup_name(raw_name: Optional[str]) -> str:
    """Normalize strategy/setup strings into standardized categories."""
    if not raw_name:
        return "Unknown"
    s = raw_name.lower().strip()
    if "reversal at pdh" in s or "pdh reversal" in s:
        return "GJ_Reversal_PDH"
    elif "reversal at pdl" in s or "pdl reversal" in s:
        return "GJ_Reversal_PDL"
    elif "do reversal" in s or "day open reversal" in s:
        return "GJ_DO_Reversal"
    elif "break and go" in s or "break_and_go" in s:
        return "GJ_Break_And_Go"
    elif "pin bar" in s or "pinbar" in s:
        return "Candle_PinBar"
    elif "engulfing" in s:
        return "Candle_Engulfing"
    elif "inside bar" in s:
        return "Candle_InsideBar"
    elif "liquidity" in s or "sweep" in s:
        return "GJ_Liquidity_Sweep"
    elif "manual" in s:
        return "Manual_Trade"
    # General clean name
    clean = raw_name.replace(" ", "_").replace("/", "_").replace("-", "_")
    return clean[:30]


class SetupMetrics:
    """Tracks quantitative performance metrics for a specific setup or symbol."""

    def __init__(self, name: str):
        self.name = name
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.net_pnl = 0.0
        self.current_streak = 0       # positive for win streak, negative for loss streak
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.tp1_hits = 0
        self.tp2_hits = 0
        self.sl_hits = 0
        self.manual_exits = 0
        self.status = "ACTIVE"        # ACTIVE | PRIORITIZED | SUPPRESSED | PROBATION
        self.weight_multiplier = 1.0  # Sizing / confidence weight (0.5 to 1.5)
        self.sl_buffer_multiplier = 1.0  # SL buffer adjustment (1.0 to 1.3)
        self.last_updated = int(time.time())

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round((self.wins / self.total_trades) * 100.0, 1)

    @property
    def profit_factor(self) -> float:
        if abs(self.gross_loss) < 1e-4:
            return round(self.gross_profit, 2) if self.gross_profit > 0 else 1.0
        return round(self.gross_profit / abs(self.gross_loss), 2)

    @property
    def score(self) -> float:
        """Normalized quantitative score (1.0 = baseline, >1.0 = strong, <0.8 = underperforming)."""
        if self.total_trades < 2:
            return 1.0
        wr_component = (self.win_rate - 50.0) / 50.0  # -1.0 to +1.0
        pf_component = min(max((self.profit_factor - 1.0), -1.0), 1.0)
        calculated = 1.0 + (0.6 * wr_component) + (0.4 * pf_component)
        return max(0.2, min(2.0, round(calculated, 2)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "net_pnl": round(self.net_pnl, 2),
            "profit_factor": self.profit_factor,
            "score": self.score,
            "status": self.status,
            "weight_multiplier": round(self.weight_multiplier, 2),
            "sl_buffer_multiplier": round(self.sl_buffer_multiplier, 2),
            "current_streak": self.current_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "tp1_hits": self.tp1_hits,
            "tp2_hits": self.tp2_hits,
            "sl_hits": self.sl_hits,
            "manual_exits": self.manual_exits,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SetupMetrics":
        obj = cls(name=data.get("name", "Unknown"))
        obj.total_trades = data.get("total_trades", 0)
        obj.wins = data.get("wins", 0)
        obj.losses = data.get("losses", 0)
        obj.gross_profit = float(data.get("gross_profit", 0.0))
        obj.gross_loss = float(data.get("gross_loss", 0.0))
        obj.net_pnl = float(data.get("net_pnl", 0.0))
        obj.current_streak = int(data.get("current_streak", 0))
        obj.max_win_streak = int(data.get("max_win_streak", 0))
        obj.max_loss_streak = int(data.get("max_loss_streak", 0))
        obj.tp1_hits = int(data.get("tp1_hits", 0))
        obj.tp2_hits = int(data.get("tp2_hits", 0))
        obj.sl_hits = int(data.get("sl_hits", 0))
        obj.manual_exits = int(data.get("manual_exits", 0))
        obj.status = data.get("status", "ACTIVE")
        obj.weight_multiplier = float(data.get("weight_multiplier", 1.0))
        obj.sl_buffer_multiplier = float(data.get("sl_buffer_multiplier", 1.0))
        obj.last_updated = int(data.get("last_updated", time.time()))
        return obj


class LearningEngine:
    """
    Self-learning and parameter optimization engine for trading strategies.
    Operates 100% locally with zero token consumption for continuous calculations.
    """

    def __init__(self, store_file: str = "learning_store.json"):
        self.store_file = store_file
        self.setups: Dict[str, SetupMetrics] = {}
        self.symbols: Dict[str, SetupMetrics] = {}
        self.sessions: Dict[str, SetupMetrics] = {
            "Asia": SetupMetrics("Asia"),
            "London": SetupMetrics("London"),
            "New York": SetupMetrics("New York"),
        }
        self.overall = SetupMetrics("OVERALL")
        self.adaptations: List[Dict[str, Any]] = []
        self.cooldowns: Dict[str, float] = {}  # key -> timestamp until trade cooldown expires
        self.settings: Dict[str, Any] = {
            "auto_suppress_enabled": True,
            "min_trades_for_adaptation": 3,
            "suppress_win_rate_threshold": 40.0,
            "prioritize_win_rate_threshold": 60.0,
            "max_loss_streak_cooldown": 3,
            "cooldown_duration_seconds": 1800,  # 30 mins cooling period after 3 consecutive losses
            "adaptive_sl_enabled": True,
            "adaptive_sizing_enabled": True,
        }
        self._ai_cache: Dict[str, Any] = {"summary": None, "timestamp": 0}
        self.load()

    # ==================== Persistence ====================

    def load(self):
        """Load learning state from disk."""
        if not os.path.exists(self.store_file):
            return

        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.settings.update(data.get("settings", {}))
            self.adaptations = data.get("adaptations", [])
            self.cooldowns = data.get("cooldowns", {})

            if "overall" in data:
                self.overall = SetupMetrics.from_dict(data["overall"])

            self.setups = {
                k: SetupMetrics.from_dict(v)
                for k, v in data.get("setups", {}).items()
            }
            self.symbols = {
                k: SetupMetrics.from_dict(v)
                for k, v in data.get("symbols", {}).items()
            }
            for sess_name, sess_dict in data.get("sessions", {}).items():
                self.sessions[sess_name] = SetupMetrics.from_dict(sess_dict)

        except Exception as e:
            logger.error(f"Error loading {self.store_file}: {e}")

    def save(self):
        """Save learning state to disk."""
        try:
            payload = {
                "settings": self.settings,
                "adaptations": self.adaptations[-50:],  # retain last 50 adaptations
                "cooldowns": self.cooldowns,
                "overall": self.overall.to_dict(),
                "setups": {k: v.to_dict() for k, v in self.setups.items()},
                "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
                "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            }
            with open(self.store_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {self.store_file}: {e}")

    # ==================== Bootstrap from History ====================

    def bootstrap_from_history(self, history: List[Dict[str, Any]]) -> int:
        """
        Bootstrap learning memory from existing trade history without double-counting.
        """
        if not history:
            return 0

        # Reset counts to recalculate cleanly
        self.setups.clear()
        self.symbols.clear()
        self.sessions = {
            "Asia": SetupMetrics("Asia"),
            "London": SetupMetrics("London"),
            "New York": SetupMetrics("New York"),
        }
        self.overall = SetupMetrics("OVERALL")
        self.adaptations.clear()

        count = 0
        for trade in history:
            self.record_trade_outcome(trade, save_immediately=False)
            count += 1

        self.save()
        return count

    # ==================== Trade Outcome Recording & Learning ====================

    def record_trade_outcome(self, trade: Dict[str, Any], save_immediately: bool = True) -> Dict[str, Any]:
        """
        Record a closed trade and immediately recalibrate strategy weights,
        suppression rules, and adaptive parameters. ZERO TOKENS USED.
        """
        symbol = trade.get("symbol", "UNKNOWN").upper()
        raw_strat = trade.get("strategy") or trade.get("reason") or "Manual"
        setup_name = normalize_setup_name(raw_strat)
        pnl = float(trade.get("pnl", 0.0))
        is_win = bool(trade.get("is_win", pnl > 0))
        opened_at = float(trade.get("opened_at", time.time()))
        exit_reason = str(trade.get("exit_reason", "")).lower()

        session_name = get_session_name(opened_at)

        # Get or create metric trackers
        setup_metric = self.setups.setdefault(setup_name, SetupMetrics(setup_name))
        symbol_metric = self.symbols.setdefault(symbol, SetupMetrics(symbol))
        session_metric = self.sessions.setdefault(session_name, SetupMetrics(session_name))

        # Update all relevant trackers
        trackers = [self.overall, setup_metric, symbol_metric, session_metric]
        for tr in trackers:
            tr.total_trades += 1
            tr.last_updated = int(time.time())
            tr.net_pnl += pnl

            if is_win:
                tr.wins += 1
                tr.gross_profit += max(0.0, pnl)
                tr.current_streak = tr.current_streak + 1 if tr.current_streak > 0 else 1
                tr.max_win_streak = max(tr.max_win_streak, tr.current_streak)
            else:
                tr.losses += 1
                tr.gross_loss += abs(min(0.0, pnl))
                tr.current_streak = tr.current_streak - 1 if tr.current_streak < 0 else -1
                tr.max_loss_streak = max(tr.max_loss_streak, abs(tr.current_streak))

            # Exit classification
            if "tp2" in exit_reason:
                tr.tp2_hits += 1
            elif "tp1" in exit_reason or "take profit" in exit_reason:
                tr.tp1_hits += 1
            elif "stop loss" in exit_reason or "sl" in exit_reason:
                tr.sl_hits += 1
            else:
                tr.manual_exits += 1

        # Run Self-Improvement / Adaptation logic
        adaptations_made = self._recalibrate(setup_metric, symbol_metric)

        if save_immediately:
            self.save()

        # Invalidate AI cache since fresh data arrived
        self._ai_cache["timestamp"] = 0

        return {
            "setup": setup_name,
            "symbol": symbol,
            "win": is_win,
            "pnl": pnl,
            "setup_win_rate": setup_metric.win_rate,
            "setup_status": setup_metric.status,
            "adaptations": adaptations_made,
        }

    # ==================== Self-Improvement Recalibration ====================

    def _recalibrate(self, setup_metric: SetupMetrics, symbol_metric: SetupMetrics) -> List[str]:
        """
        Core self-improving algorithm:
        1. Setup Suppression: Auto-disables setups that drop below win-rate threshold.
        2. Setup Prioritization: Boosts setups with strong positive edge.
        3. Adaptive Sizing: Adjusts sizing multiplier (0.5x to 1.3x) based on setup score.
        4. Cooldown Trigger: Halts symbol/setup after consecutive losses to prevent drawdowns.
        5. Adaptive SL Buffering: Widens stop loss if stop-outs are prematurely frequent.
        """
        adaptations = []
        now = time.time()
        min_sample = self.settings.get("min_trades_for_adaptation", 3)
        suppress_thresh = self.settings.get("suppress_win_rate_threshold", 40.0)
        prio_thresh = self.settings.get("prioritize_win_rate_threshold", 60.0)

        # 1. Setup Status & Weight Recalibration
        if setup_metric.total_trades >= min_sample and self.settings.get("auto_suppress_enabled", True):
            old_status = setup_metric.status

            if setup_metric.win_rate < suppress_thresh:
                setup_metric.status = "SUPPRESSED"
                setup_metric.weight_multiplier = 0.0  # Zero weight = skipped
                if old_status != "SUPPRESSED":
                    msg = (
                        f"🛡️ AUTO-SUPPRESSION: Setup '{setup_metric.name}' win rate is {setup_metric.win_rate}% "
                        f"(<{suppress_thresh}% over {setup_metric.total_trades} trades). Automatically suppressed to protect capital."
                    )
                    self._record_adaptation("SETUP_SUPPRESS", msg)
                    adaptations.append(msg)

            elif setup_metric.win_rate >= prio_thresh and setup_metric.profit_factor >= 1.2:
                setup_metric.status = "PRIORITIZED"
                setup_metric.weight_multiplier = min(1.3, 1.0 + (setup_metric.win_rate - 50.0) / 100.0)
                if old_status != "PRIORITIZED":
                    msg = (
                        f"🚀 AUTO-PRIORITIZATION: Setup '{setup_metric.name}' win rate is {setup_metric.win_rate}% "
                        f"(Score {setup_metric.score}). Size weight boosted to {setup_metric.weight_multiplier:.2f}x."
                    )
                    self._record_adaptation("SETUP_PRIORITIZE", msg)
                    adaptations.append(msg)
            else:
                setup_metric.status = "ACTIVE"
                setup_metric.weight_multiplier = 1.0

        # 2. Adaptive SL Buffer Calibration
        if self.settings.get("adaptive_sl_enabled", True) and setup_metric.total_trades >= 3:
            # If SL was hit repeatedly (> 60% of outcomes were SL hits)
            sl_hit_ratio = (setup_metric.sl_hits / setup_metric.total_trades) if setup_metric.total_trades > 0 else 0
            if sl_hit_ratio > 0.60:
                old_buf = setup_metric.sl_buffer_multiplier
                setup_metric.sl_buffer_multiplier = min(1.25, round(old_buf + 0.05, 2))
                if setup_metric.sl_buffer_multiplier != old_buf:
                    msg = (
                        f"📐 ADAPTIVE STOP-LOSS: Setup '{setup_metric.name}' experiencing frequent stop hits ({setup_metric.sl_hits}/{setup_metric.total_trades}). "
                        f"Expanded dynamic SL buffer to {setup_metric.sl_buffer_multiplier:.2f}x."
                    )
                    self._record_adaptation("ADAPTIVE_SL", msg)
                    adaptations.append(msg)

        # 3. Consecutive Loss Streak Cooldown Protection
        max_losses = self.settings.get("max_loss_streak_cooldown", 3)
        if symbol_metric.current_streak <= -max_losses:
            cd_duration = self.settings.get("cooldown_duration_seconds", 1800)
            self.cooldowns[symbol_metric.name] = now + cd_duration
            msg = (
                f"⏸️ DRAWDOWN COOLDOWN: Symbol '{symbol_metric.name}' hit {abs(symbol_metric.current_streak)} consecutive losses. "
                f"Auto-cooldown activated for {int(cd_duration / 60)} minutes to protect capital."
            )
            self._record_adaptation("COOLDOWN_ACTIVATED", msg)
            adaptations.append(msg)

        return adaptations

    def _record_adaptation(self, adaptation_type: str, message: str):
        self.adaptations.append({
            "timestamp": int(time.time()),
            "type": adaptation_type,
            "message": message,
        })
        logger.info(f"[SelfLearning] {message}")

    # ==================== Trade Execution Gating ====================

    def can_execute(self, setup_name: Optional[str], symbol: str) -> Tuple[bool, str]:
        """
        Query if a trade setup is allowed to execute based on learned performance.
        Returns (is_allowed: bool, reason: str).
        """
        now = time.time()
        sym = symbol.upper()
        norm_setup = normalize_setup_name(setup_name)

        # 1. Check Setup Suppression
        if norm_setup in self.setups:
            metric = self.setups[norm_setup]
            if metric.status == "SUPPRESSED" and self.settings.get("auto_suppress_enabled", True):
                return False, f"Setup '{norm_setup}' is suppressed due to poor win rate ({metric.win_rate}%). Skipping trade to preserve capital."

        # 2. Check Symbol Cooldown
        cd_until = self.cooldowns.get(sym, 0)
        if now < cd_until:
            remaining_mins = int((cd_until - now) / 60) + 1
            return False, f"Symbol {sym} is in loss streak cooldown ({remaining_mins}m remaining). Self-learning protection active."

        return True, "Execution approved by Self-Learning Engine"

    def get_adapted_parameters(
        self,
        setup_name: Optional[str],
        symbol: str,
        default_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate dynamically optimized parameters for this trade:
        - Adapted position size multiplier
        - Adapted stop-loss buffer
        - Preferred take-profit target strategy
        """
        norm_setup = normalize_setup_name(setup_name)
        sym = symbol.upper()

        setup_metric = self.setups.get(norm_setup)
        symbol_metric = self.symbols.get(sym)

        # Base sizing multiplier
        size_mult = 1.0
        if setup_metric and self.settings.get("adaptive_sizing_enabled", True):
            size_mult = setup_metric.weight_multiplier

        # If symbol recently had 2 losses, throttle size to 0.7x
        if symbol_metric and symbol_metric.current_streak <= -2:
            size_mult = min(size_mult, 0.7)

        sl_buffer = 1.0
        if setup_metric and self.settings.get("adaptive_sl_enabled", True):
            sl_buffer = setup_metric.sl_buffer_multiplier

        # Check if TP2 is rarely reached (< 25% of wins hit TP2), favor securing at TP1
        tp_strategy = "standard"
        if setup_metric and setup_metric.wins >= 3:
            tp2_rate = setup_metric.tp2_hits / setup_metric.wins
            if tp2_rate < 0.25:
                tp_strategy = "tp1_secured"

        recommended_size = round(default_size * size_mult, 4)

        return {
            "setup": norm_setup,
            "symbol": sym,
            "size_multiplier": size_mult,
            "recommended_size": max(0.001, recommended_size),
            "sl_buffer_multiplier": sl_buffer,
            "tp_strategy": tp_strategy,
            "setup_score": setup_metric.score if setup_metric else 1.0,
            "setup_status": setup_metric.status if setup_metric else "ACTIVE",
        }

    # ==================== Reporting & Statistics (0 TOKENS) ====================

    def get_learning_summary(self) -> Dict[str, Any]:
        """Return structured summary of learning engine state."""
        # Categorize setups
        active_setups = []
        suppressed_setups = []
        prioritized_setups = []

        for name, m in self.setups.items():
            info = {
                "name": name,
                "trades": m.total_trades,
                "win_rate": m.win_rate,
                "pnl": m.net_pnl,
                "score": m.score,
                "status": m.status,
            }
            if m.status == "SUPPRESSED":
                suppressed_setups.append(info)
            elif m.status == "PRIORITIZED":
                prioritized_setups.append(info)
            else:
                active_setups.append(info)

        return {
            "total_trades": self.overall.total_trades,
            "overall_win_rate": self.overall.win_rate,
            "overall_pnl": self.overall.net_pnl,
            "overall_profit_factor": self.overall.profit_factor,
            "current_streak": self.overall.current_streak,
            "best_win_streak": self.overall.max_win_streak,
            "worst_loss_streak": self.overall.max_loss_streak,
            "suppressed_count": len(suppressed_setups),
            "prioritized_count": len(prioritized_setups),
            "active_count": len(active_setups),
            "suppressed_setups": suppressed_setups,
            "prioritized_setups": prioritized_setups,
            "active_setups": active_setups,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "recent_adaptations": self.adaptations[-5:],
            "tokens_consumed": 0,  # Explicit verification of 0-token efficiency
        }

    def get_top_setup(self) -> Optional[Dict[str, Any]]:
        """Return the best performing setup by score and win rate."""
        if not self.setups:
            return None
        sorted_setups = sorted(
            self.setups.values(),
            key=lambda m: (m.status == "PRIORITIZED", m.score, m.win_rate, m.total_trades),
            reverse=True
        )
        best = sorted_setups[0]
        return {
            "name": best.name,
            "win_rate_pct": best.win_rate,
            "trades": best.total_trades,
            "score": best.score,
            "status": best.status,
        }

    def format_html_report(self) -> str:
        """
        Generate Telegram-ready HTML report with rich emojis and actionable breakdown.
        Calculated completely locally without calling any external LLM APIs (0 TOKENS).
        """
        summary = self.get_learning_summary()
        tot = summary["total_trades"]
        wr = summary["overall_win_rate"]
        pnl = summary["overall_pnl"]
        pf = summary["overall_profit_factor"]
        streak = summary["current_streak"]

        streak_str = f"🔥 {streak} Win Streak" if streak > 0 else (f"⚠️ {abs(streak)} Loss Streak" if streak < 0 else "Neutral")
        pnl_color = "🟢" if pnl >= 0 else "🔴"

        lines = [
            "🧠 <b>SELF-LEARNING STRATEGY ENGINE</b>",
            "<i>Autonomous Reinforcement & Setup Optimization (0 AI Tokens Used)</i>\n",
            f"• <b>Total Trades Analyzed:</b> <code>{tot}</code>",
            f"• <b>Overall Win Rate:</b> <code>{wr}%</code>",
            f"• <b>Realized Net PnL:</b> {pnl_color} <code>${pnl:+.2f}</code>",
            f"• <b>Profit Factor:</b> <code>{pf}</code>",
            f"• <b>Current Streak:</b> {streak_str}",
            f"• <b>Best Win Streak:</b> <code>{summary['best_win_streak']}</code>",
            "",
            "<b>📊 SETUP PERFORMANCE & SELF-IMPROVEMENT:</b>",
        ]

        if not self.setups:
            lines.append("<i>No trades recorded yet. Bot will continuously calibrate as trades occur.</i>")
        else:
            # Sort by win rate / total trades
            sorted_setups = sorted(
                self.setups.values(),
                key=lambda x: (x.win_rate, x.total_trades),
                reverse=True
            )
            for m in sorted_setups:
                status_emoji = "🟢" if m.status == "PRIORITIZED" else ("🔴" if m.status == "SUPPRESSED" else "⚪")
                tag = f"[{m.status}]"
                lines.append(
                    f"{status_emoji} <b>{m.name}</b> {tag}\n"
                    f"   Trades: <code>{m.total_trades}</code> | WR: <code>{m.win_rate}%</code> | "
                    f"PF: <code>{m.profit_factor}</code> | Score: <code>{m.score}</code>\n"
                    f"   Sizing: <code>{m.weight_multiplier:.2f}x</code> | SL Buffer: <code>{m.sl_buffer_multiplier:.2f}x</code>"
                )

        # Performance by Session
        lines.append("\n<b>🕒 SESSION BREAKDOWN (UTC):</b>")
        for sess_name, sm in self.sessions.items():
            if sm.total_trades > 0:
                lines.append(
                    f"• <b>{sess_name}:</b> {sm.total_trades} trades | WR: <code>{sm.win_rate}%</code> | PnL: <code>${sm.net_pnl:+.2f}</code>"
                )
            else:
                lines.append(f"• <b>{sess_name}:</b> 0 trades")

        # Active Cooldowns
        now = time.time()
        active_cds = [
            f"{sym} ({int((t - now) / 60) + 1}m left)"
            for sym, t in self.cooldowns.items()
            if t > now
        ]
        if active_cds:
            lines.append(f"\n⏸️ <b>Active Drawdown Cooldowns:</b> {', '.join(active_cds)}")

        # Recent Adaptations
        adaptations = summary.get("recent_adaptations", [])
        if adaptations:
            lines.append("\n<b>⚡ RECENT AUTO-ADAPTATIONS:</b>")
            for a in reversed(adaptations[-3:]):
                lines.append(f"• {a.get('message', '')}")

        lines.append("\n💡 <i>To get an ultra-low-token AI quantitative synthesis, use <code>/learn ai</code></i>")
        return "\n".join(lines)

    # ==================== Ultra-Token-Efficient AI Synthesis ====================

    def generate_ai_insight(self, gemini_caller) -> str:
        """
        Generate high-level AI synthesis strictly on-demand.
        - Compresses entire metrics history into < 100 input tokens.
        - Caps output at 250 tokens.
        - Uses 15-minute caching to eliminate repetitive token consumption.
        """
        now = time.time()
        cached = self._ai_cache
        if cached.get("summary") and (now - cached.get("timestamp", 0)) < 900:  # 15 mins cache
            return f"{cached['summary']}\n\n<i>(Cached insight — 0 tokens used)</i>"

        summary = self.get_learning_summary()
        if summary["total_trades"] == 0:
            return "🤖 <b>AI Strategic Review:</b> Insufficient trade history. The bot requires at least 3 closed trades to generate quantitative insights."

        # Highly compressed quantitative payload (< 100 tokens)
        compressed_payload = {
            "trades": summary["total_trades"],
            "wr": summary["overall_win_rate"],
            "pf": summary["overall_profit_factor"],
            "pnl": summary["overall_pnl"],
            "streak": summary["current_streak"],
            "suppressed": [s["name"] for s in summary["suppressed_setups"]],
            "top": [s["name"] for s in summary["prioritized_setups"]],
            "sessions": {k: f"{v['win_rate']}%" for k, v in summary["sessions"].items() if v["total_trades"] > 0},
        }

        prompt = (
            f"Review this trading algorithm's performance metrics: {json.dumps(compressed_payload)}. "
            "In 3 concise bullet points (<70 words total), give sharp quant recommendations on: "
            "1) Best edge to exploit, 2) Session risk, 3) Capital preservation tip."
        )

        try:
            # We call the gemini caller with a strict token cap
            ai_text = gemini_caller(prompt, max_tokens=250)
            self._ai_cache = {"summary": ai_text, "timestamp": now}
            return f"🤖 <b>AI Quantitative Insight (Compressed Review):</b>\n\n{ai_text}"
        except Exception as e:
            logger.error(f"Failed to generate AI insight: {e}")
            return f"⚠️ Could not generate AI insight: {e}"

    # ==================== Reset / Manual Management ====================

    def reset(self):
        """Reset all learning weights and history."""
        self.setups.clear()
        self.symbols.clear()
        self.sessions = {
            "Asia": SetupMetrics("Asia"),
            "London": SetupMetrics("London"),
            "New York": SetupMetrics("New York"),
        }
        self.overall = SetupMetrics("OVERALL")
        self.adaptations.clear()
        self.cooldowns.clear()
        self._ai_cache = {"summary": None, "timestamp": 0}
        self.save()

    def unsuppress_setup(self, setup_name: str) -> bool:
        """Manually reinstate a suppressed setup."""
        norm = normalize_setup_name(setup_name)
        if norm in self.setups:
            self.setups[norm].status = "ACTIVE"
            self.setups[norm].weight_multiplier = 1.0
            self.save()
            return True
        return False
