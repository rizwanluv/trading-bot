"""
Price alerts and candle entry watchers management with file persistence.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple


class AlertManager:
    """Manages active price alerts and automated candle entry watchers."""

    def __init__(self, storage_path: str = "alerts_store.json"):
        self.storage_path = storage_path
        self.price_alerts: List[Dict[str, Any]] = []
        self.entry_watchers: Dict[str, Dict[str, Any]] = {}  # key: f"{chat_id}:{symbol}"
        self.next_alert_id = 1
        self.load()

    def load(self):
        """Load alerts and watchers from storage file."""
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.price_alerts = data.get("price_alerts", [])
                self.entry_watchers = data.get("entry_watchers", {})
                self.next_alert_id = data.get("next_alert_id", 1)
                # Ensure next_alert_id is higher than any existing alert
                if self.price_alerts:
                    existing_ids = [a.get("id", 0) for a in self.price_alerts if isinstance(a.get("id"), int)]
                    if existing_ids:
                        self.next_alert_id = max(max(existing_ids) + 1, self.next_alert_id)
        except Exception as e:
            print(f"Warning: Could not load alerts from {self.storage_path}: {e}")

    def save(self):
        """Persist current alerts and watchers to storage file."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "price_alerts": self.price_alerts,
                        "entry_watchers": self.entry_watchers,
                        "next_alert_id": self.next_alert_id,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: Could not save alerts to {self.storage_path}: {e}")

    # ==================== Price Alerts ====================

    def add_price_alert(
        self,
        chat_id: int,
        symbol: str,
        target_price: float,
        condition: Optional[str] = None,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add a new price alert.
        If condition is None, auto-detects:
        - target > current -> '>=' (alert when price rises to target)
        - target < current -> '<=' (alert when price drops to target)
        """
        sym = symbol.upper()
        if condition is None:
            if current_price is not None:
                condition = ">=" if target_price >= current_price else "<="
            else:
                condition = ">="

        alert_id = self.next_alert_id
        self.next_alert_id += 1

        alert = {
            "id": alert_id,
            "chat_id": chat_id,
            "symbol": sym,
            "target_price": round(float(target_price), 4),
            "condition": condition,
            "created_price": round(float(current_price), 4) if current_price else 0.0,
            "created_time": time.time(),
        }
        self.price_alerts.append(alert)
        self.save()
        return alert

    def get_chat_alerts(self, chat_id: int) -> List[Dict[str, Any]]:
        """Get all pending price alerts for a specific chat."""
        return [a for a in self.price_alerts if a["chat_id"] == chat_id]

    def remove_alert(self, chat_id: int, alert_id: int) -> bool:
        """Remove a specific alert by ID."""
        initial_len = len(self.price_alerts)
        self.price_alerts = [a for a in self.price_alerts if not (a["chat_id"] == chat_id and a["id"] == alert_id)]
        if len(self.price_alerts) < initial_len:
            self.save()
            return True
        return False

    def clear_chat_alerts(self, chat_id: int) -> int:
        """Clear all alerts for a specific chat."""
        initial_len = len(self.price_alerts)
        self.price_alerts = [a for a in self.price_alerts if a["chat_id"] != chat_id]
        cleared_count = initial_len - len(self.price_alerts)
        if cleared_count > 0:
            self.save()
        return cleared_count

    def check_price_alerts(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check which price alerts have met their condition.
        Triggered alerts are removed from active list and returned.
        """
        triggered = []
        remaining = []

        for alert in self.price_alerts:
            sym = alert["symbol"]
            curr = current_prices.get(sym)
            if curr is None:
                remaining.append(alert)
                continue

            target = alert["target_price"]
            cond = alert["condition"]
            is_hit = False

            if cond in (">=", ">") and curr >= target:
                is_hit = True
            elif cond in ("<=", "<") and curr <= target:
                is_hit = True

            if is_hit:
                triggered_alert = dict(alert)
                triggered_alert["trigger_price"] = curr
                triggered.append(triggered_alert)
            else:
                remaining.append(alert)

        if triggered:
            self.price_alerts = remaining
            self.save()

        return triggered

    # ==================== Entry Watchers ====================

    def add_entry_watcher(
        self,
        chat_id: int,
        symbol: str,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add or update an automatic candle entry watcher for a chat.
        Defaults to ['1m', '5m', '15m'].
        """
        sym = symbol.upper()
        if not timeframes:
            timeframes = ["1m", "5m", "15m"]

        key = f"{chat_id}:{sym}"
        watcher = {
            "chat_id": chat_id,
            "symbol": sym,
            "timeframes": timeframes,
            "last_alert_time": {},  # tf -> candle_time
            "created_time": time.time(),
        }
        self.entry_watchers[key] = watcher
        self.save()
        return watcher

    def remove_entry_watcher(self, chat_id: int, symbol: Optional[str] = None) -> int:
        """
        Remove entry watcher for a symbol, or all watchers for chat if symbol is None or 'all'.
        """
        keys_to_remove = []
        sym = symbol.upper() if symbol and symbol.lower() != "all" else None

        for key, watcher in self.entry_watchers.items():
            if watcher["chat_id"] == chat_id:
                if sym is None or watcher["symbol"] == sym:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.entry_watchers[key]

        if keys_to_remove:
            self.save()
        return len(keys_to_remove)

    def get_chat_entry_watchers(self, chat_id: int) -> List[Dict[str, Any]]:
        """Get all entry watchers for a specific chat."""
        return [w for w in self.entry_watchers.values() if w["chat_id"] == chat_id]

    def get_all_entry_watchers(self) -> List[Dict[str, Any]]:
        """Get all active entry watchers across all chats."""
        return list(self.entry_watchers.values())

    def update_watcher_alert_time(self, chat_id: int, symbol: str, timeframe: str, candle_time: int):
        """Record that an alert has been delivered for this candle timestamp to prevent duplicate alerts."""
        key = f"{chat_id}:{symbol.upper()}"
        if key in self.entry_watchers:
            self.entry_watchers[key].setdefault("last_alert_time", {})[timeframe] = candle_time
            self.save()
