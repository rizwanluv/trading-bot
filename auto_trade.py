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
import math
import os
import sys
import hashlib
import hmac
import tempfile
import time
import time as time_module
import urllib.parse
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from trading_strategy_indicators_pro import Direction, IndicatorsProStrategy, IndicatorEngine
from ai_bot_learning import AIBotLearning, TradeRecord
from itb_engine import (
    ITBStrategy,
    ITBPredictor,
    ITBBacktester,
    ITBFeatureGenerator,
    ITBPredictionResult,
    format_itb_card,
    format_backtest_report,
)
try:
    from risk_manager import RiskManager, RiskConfig, INSTRUMENTS, TradeRiskResult
except ImportError:
    RiskManager = None
    RiskConfig = None
    INSTRUMENTS = {}
    TradeRiskResult = None

logger = logging.getLogger("trading_bot.auto_trade")

BASE = os.path.dirname(os.path.abspath(__file__))
AI_FILE = os.path.join(BASE, "ai_bot_learning.py")
PRO_FILE = os.path.join(BASE, "trading_strategy_indicators_pro.py")
CONFIG_FILE_PATH = os.path.join(BASE, "auto_trade_config.json")
TRADES_HISTORY_PATH = os.path.join(BASE, "trades_history.json")
OPEN_POSITIONS_PATH = os.path.join(BASE, "open_positions.json")
ALERTS_FILE_PATH = os.path.join(BASE, "trade_alerts.json")
RISK_STATE_PATH = os.path.join(BASE, "risk_state.json")
DELTA_CHART_API = os.getenv(
    "DELTA_CHART_API", "https://api.india.delta.exchange/v2/chart/history"
)
DELTA_TICKER_API = os.getenv(
    "DELTA_TICKER_API", "https://api.india.delta.exchange/v2/tickers"
)


def make_modern_meter(
    pct: float, width: int = 10, fill_char: str = "■", empty_char: str = "░"
) -> str:
    """Build a modern unicode progress/meter bar."""
    try:
        clamped = max(0.0, min(100.0, float(pct)))
    except (ValueError, TypeError):
        clamped = 0.0
    filled_len = int(round((clamped / 100.0) * width))
    empty_len = width - filled_len
    return f"{fill_char * filled_len}{empty_char * empty_len}"


# ==================================================================
# 1. AUTO TRADE CONFIGURATION & POSITION MODELS
# ==================================================================


@dataclass
class AutoTradeConfig:
    enabled: bool = False
    symbol: str = "BTCUSD"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSD", "XAUTUSD"])
    max_positions: int = 5
    max_positions_per_symbol: int = 1
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
    # Mode & Funds: Paper trade capital default $100
    trading_mode: str = "paper"  # "paper" or "live"
    paper_capital: float = 100.0  # Base paper trading capital ($100 default)
    equity: float = 100.0  # Active trading funds / account equity (default $100.00)
    initial_paper_capital: float = 100.0
    total_deposited: float = 0.0
    total_withdrawn: float = 0.0
    # Live exchange credentials
    live_exchange: str = "delta"  # "delta", "binance"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    delta_api_key: str = ""
    delta_api_secret: str = ""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_market: str = "auto"  # "auto", "spot", "futures"
    notify_chat_id: Optional[int] = None
    config_file: str = CONFIG_FILE_PATH
    trades_history_file: str = TRADES_HISTORY_PATH
    open_positions_file: str = OPEN_POSITIONS_PATH
    alerts_file: str = ALERTS_FILE_PATH
    risk_state_file: str = RISK_STATE_PATH
    auto_breakeven: bool = True
    risk_factor: float = 1.0
    # Dynamic Multi-Layer Ensemble & Background Self-Learning
    weight_itb: float = 0.35
    weight_pro: float = 0.35
    weight_ai: float = 0.30
    auto_learn_enabled: bool = True
    auto_retrain_interval_seconds: int = 300
    last_retrain_time: str = ""
    learning_cycles: int = 0
    engine_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.engine_stats:
            self.engine_stats = {
                "itb": {"wins": 0, "total": 0},
                "pro": {"wins": 0, "total": 0},
                "ai": {"wins": 0, "total": 0},
                "smc": {"wins": 0, "total": 0},
            }
        if self.config_file != CONFIG_FILE_PATH:
            base, ext = os.path.splitext(self.config_file)
            if self.open_positions_file == OPEN_POSITIONS_PATH:
                self.open_positions_file = f"{base}_open_positions{ext}"
            if self.alerts_file == ALERTS_FILE_PATH:
                self.alerts_file = f"{base}_alerts{ext}"
            if self.risk_state_file == RISK_STATE_PATH:
                self.risk_state_file = f"{base}_risk_state{ext}"
        if not self.delta_api_key and self.live_exchange == "delta" and self.exchange_api_key:
            self.delta_api_key = self.exchange_api_key
            self.delta_api_secret = self.exchange_api_secret
        elif not self.binance_api_key and self.live_exchange == "binance" and self.exchange_api_key:
            self.binance_api_key = self.exchange_api_key
            self.binance_api_secret = self.exchange_api_secret
        if not self.symbols:
            self.symbols = [self.symbol, "XAUTUSD"] if "XAU" not in self.symbol else ["BTCUSD", self.symbol]
        elif self.symbol not in self.symbols:
            self.symbols.insert(0, self.symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "symbol": self.symbol,
            "symbols": self.symbols,
            "max_positions": self.max_positions,
            "max_positions_per_symbol": self.max_positions_per_symbol,
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
            "trading_mode": self.trading_mode,
            "paper_capital": self.paper_capital,
            "equity": self.equity,
            "initial_paper_capital": self.initial_paper_capital,
            "total_deposited": self.total_deposited,
            "total_withdrawn": self.total_withdrawn,
            "live_exchange": self.live_exchange,
            "exchange_api_key": self.exchange_api_key,
            "exchange_api_secret": self.exchange_api_secret,
            "delta_api_key": self.delta_api_key,
            "delta_api_secret": self.delta_api_secret,
            "binance_api_key": self.binance_api_key,
            "binance_api_secret": self.binance_api_secret,
            "binance_market": self.binance_market,
            "notify_chat_id": self.notify_chat_id,
            "alerts_file": self.alerts_file,
            "weight_itb": self.weight_itb,
            "weight_pro": self.weight_pro,
            "weight_ai": self.weight_ai,
            "auto_learn_enabled": self.auto_learn_enabled,
            "auto_retrain_interval_seconds": self.auto_retrain_interval_seconds,
            "last_retrain_time": self.last_retrain_time,
            "learning_cycles": self.learning_cycles,
            "auto_breakeven": self.auto_breakeven,
            "risk_factor": self.risk_factor,
            "engine_stats": self.engine_stats,
        }

    def save(self) -> None:
        try:
            tmp_file = f"{self.config_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_file, self.config_file)
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
                if not getattr(cfg, "symbols", None):
                    cfg.symbols = list(dict.fromkeys([cfg.symbol, "XAUTUSD"]))
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
    mode: str = "paper"  # "paper" or "live"
    exchange_order_id: Optional[str] = None
    initial_stop_loss: Optional[float] = None
    engine_votes: Optional[Dict[str, str]] = None
    entry_rsi: float = 50.0
    entry_adx: float = 25.0
    entry_score: float = 0.0
    ai_conf: float = 0.50
    itb_score: float = 0.0
    regime: str = "NORMAL"
    session: str = "active"
    rel_vol: float = 1.0
    conviction_mult: float = 1.0
    quality_score: float = 3.5
    mtf_alignment: int = 1

    def current_pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            diff = current_price - self.entry_price
        else:
            diff = self.entry_price - current_price
        return round(diff * self.lot_size, 2)


@dataclass
class TradeLevelAlert:
    id: str
    symbol: str
    target_price: float
    condition: str = "AUTO"  # "CROSS_ABOVE", "CROSS_BELOW", "TOUCH", "AUTO"
    alert_type: str = "CUSTOM"  # "CUSTOM", "TP_PROXIMITY", "SL_WARNING", "BREAKEVEN", "ORDER_BLOCK", "FVG", "PINPOINT_ENTRY"
    note: str = ""
    created_at: str = ""
    triggered: bool = False
    triggered_at: Optional[str] = None
    chat_id: Optional[int] = None
    one_shot: bool = True


class ExchangeApiClient:
    """
    Unified client for Live Exchange Trading and Account Management.
    Supports Delta Exchange (India / Global) and Binance REST APIs.
    Handles HMAC SHA256 request signing, balance queries, order execution, and connectivity testing.
    """

    def __init__(
        self,
        exchange: str = "delta",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        market: str = "auto",
    ):
        self.exchange = (exchange or "delta").strip().lower()
        self.market = (market or os.getenv("BINANCE_MARKET", "auto")).lower()
        if self.exchange == "binance":
            self.api_key = (api_key or os.getenv("BINANCE_API_KEY") or os.getenv("EXCHANGE_API_KEY") or "").strip()
            self.api_secret = (api_secret or os.getenv("BINANCE_API_SECRET") or os.getenv("EXCHANGE_API_SECRET") or "").strip()
        else:
            self.api_key = (api_key or os.getenv("EXCHANGE_API_KEY") or os.getenv("DELTA_API_KEY") or "").strip()
            self.api_secret = (api_secret or os.getenv("EXCHANGE_API_SECRET") or os.getenv("DELTA_API_SECRET") or "").strip()
        self.testnet = testnet

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def mask_key(self, key: str) -> str:
        if not key:
            return "Not Configured"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    def get_masked_status(self) -> Dict[str, str]:
        mkt_tag = f" ({self.market.upper()})" if self.exchange == "binance" else ""
        return {
            "exchange": f"{self.exchange.upper()}{mkt_tag}",
            "api_key": self.mask_key(self.api_key),
            "api_secret": "***Configured***" if self.api_secret else "Not Configured",
            "status": "CONFIGURED 🟢" if self.is_configured else "NOT SET 🔴",
            "market": self.market,
        }

    @staticmethod
    def format_binance_symbol(symbol: str) -> str:
        s = symbol.strip().upper().replace("/", "").replace("-", "")
        if s in ("BTC", "BITCOIN", "BTCUSD"):
            return "BTCUSDT"
        if s in ("ETH", "ETHEREUM", "ETHUSD"):
            return "ETHUSDT"
        if s in ("SOL", "SOLANA", "SOLUSD"):
            return "SOLUSDT"
        if s in ("XRP", "RIPPLE", "XRPUSD"):
            return "XRPUSDT"
        if s in ("XAU", "GOLD", "XAUUSD", "XAUTUSD"):
            return "PAXGUSDT"
        if s.endswith("USD"):
            return s[:-3] + "USDT"
        if not (s.endswith("USDT") or s.endswith("BUSD") or s.endswith("FDUSD") or s.endswith("BTC")):
            return f"{s}USDT"
        return s

    def _sign_delta_request(self, method: str, path: str, payload_str: str, timestamp: str) -> str:
        msg = method.upper() + timestamp + path + payload_str
        return hmac.new(self.api_secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

    def test_public_connection(self) -> Tuple[bool, str]:
        """Verify public API connectivity to the active exchange without credentials."""
        if self.exchange == "binance":
            try:
                resp = requests.get("https://api.binance.com/api/v3/ping", timeout=6)
                if resp.status_code == 200:
                    return True, "Binance Public REST API is reachable (HTTP 200 OK)."
                return False, f"Binance Ping returned HTTP {resp.status_code}"
            except Exception as e:
                return False, f"Binance Public API unreachable: {e}"
        else:
            try:
                resp = requests.get(f"{DELTA_TICKER_API}/BTCUSD", timeout=6)
                if resp.status_code == 200:
                    return True, "Delta Exchange Public REST API is reachable (HTTP 200 OK)."
                return False, f"Delta Ping returned HTTP {resp.status_code}"
            except Exception as e:
                return False, f"Delta Public API unreachable: {e}"

    def test_connection(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Verify API credentials and return connectivity status with account info."""
        if not self.is_configured:
            pub_ok, pub_msg = self.test_public_connection()
            status_note = f" (Public API: {pub_msg})" if pub_ok else ""
            return False, f"Exchange API Key or Secret is not configured. Use /api set {self.exchange} <key> <secret>{status_note}", {}

        if self.exchange == "delta":
            base_url = "https://cdn.testnet.delta.exchange" if self.testnet else "https://api.india.delta.exchange"
            path = "/v2/wallet/balances"
            ts = str(int(time.time()))
            sig = self._sign_delta_request("GET", path, "", ts)
            headers = {
                "api-key": self.api_key,
                "timestamp": ts,
                "signature": sig,
                "Content-Type": "application/json",
                "User-Agent": "TradingBot/1.0",
            }
            try:
                resp = requests.get(f"{base_url}{path}", headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    balances = data.get("result", [])
                    usdt_bal = 0.0
                    for b in balances:
                        if b.get("asset_symbol") in ("USDT", "USD"):
                            usdt_bal += float(b.get("balance", 0.0))
                    return True, f"Connected to Delta Exchange successfully. Balance: ${usdt_bal:,.2f} USDT", {"balance": usdt_bal, "raw": data}
                elif resp.status_code in (401, 403):
                    return False, f"Delta Exchange Authentication failed (HTTP {resp.status_code}): Invalid API Key or Secret.", {}
                else:
                    return False, f"Delta Exchange returned HTTP {resp.status_code}: {resp.text[:200]}", {}
            except Exception as e:
                return False, f"Network error connecting to Delta Exchange: {e}", {}

        elif self.exchange == "binance":
            ts = int(time.time() * 1000)
            query = f"timestamp={ts}&recvWindow=5000"
            sig = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
            headers = {"X-MBX-APIKEY": self.api_key, "User-Agent": "TradingBot/1.0"}

            check_futures_first = (getattr(self, "market", "auto") == "futures")
            endpoints = []
            if check_futures_first:
                endpoints.append(("futures", "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com", "/fapi/v2/account"))
                endpoints.append(("spot", "https://testnet.binance.vision" if self.testnet else "https://api.binance.com", "/api/v3/account"))
            else:
                endpoints.append(("spot", "https://testnet.binance.vision" if self.testnet else "https://api.binance.com", "/api/v3/account"))
                endpoints.append(("futures", "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com", "/fapi/v2/account"))

            last_err = ""
            for mkt_type, base_url, path in endpoints:
                try:
                    resp = requests.get(f"{base_url}{path}?{query}&signature={sig}", headers=headers, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        usdt_bal = 0.0
                        if mkt_type == "futures":
                            usdt_bal = float(data.get("availableBalance", 0.0) or data.get("totalWalletBalance", 0.0))
                        else:
                            for b in data.get("balances", []):
                                if b.get("asset") in ("USDT", "USD"):
                                    usdt_bal += float(b.get("free", 0.0))
                        self.market = mkt_type
                        return True, f"Connected to Binance {mkt_type.title()} successfully. Available: ${usdt_bal:,.2f} USDT", {"balance": usdt_bal, "raw": data, "market": mkt_type}
                    elif resp.status_code in (401, 403, 400):
                        last_err = f"Binance {mkt_type.title()} Auth failed (HTTP {resp.status_code}): Invalid Key or Permissions."
                    else:
                        last_err = f"Binance {mkt_type.title()} error (HTTP {resp.status_code}): {resp.text[:200]}"
                except Exception as e:
                    last_err = f"Network error connecting to Binance {mkt_type.title()}: {e}"

            return False, last_err or "Binance authentication failed.", {}

        return False, f"Exchange '{self.exchange}' is not supported. Supported: delta, binance", {}

    def get_balance(self) -> Tuple[bool, float, str]:
        ok, msg, data = self.test_connection()
        if ok and "balance" in data:
            return True, float(data["balance"]), msg
        return False, 0.0, msg

    def get_ticker_price(self, symbol: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Fetch live ticker price for symbol from the configured exchange."""
        if self.exchange == "binance":
            b_sym = self.format_binance_symbol(symbol)
            endpoints = [
                f"https://api.binance.com/api/v3/ticker/24hr?symbol={b_sym}",
                f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={b_sym}",
            ]
            if getattr(self, "market", "auto") == "futures":
                endpoints.reverse()
            for url in endpoints:
                try:
                    resp = requests.get(url, timeout=8)
                    if resp.status_code == 200:
                        data = resp.json()
                        last_price = float(data.get("lastPrice", 0.0))
                        return True, last_price, data
                except Exception:
                    continue
            return False, 0.0, {"error": f"Binance price unavailable for {b_sym}"}
        else:
            target = symbol.strip().upper().replace("/", "").replace("-", "")
            try:
                resp = requests.get(f"{DELTA_TICKER_API}/{target}", timeout=8)
                if resp.status_code == 200:
                    data = resp.json().get("result", {})
                    close_price = float(data.get("close", 0.0) or data.get("mark_price", 0.0))
                    return True, close_price, data
                return False, 0.0, {"error": resp.text[:200]}
            except Exception as e:
                return False, 0.0, {"error": str(e)}

    def place_order(
        self,
        symbol: str,
        direction: str,
        size: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Place live order on exchange."""
        if not self.is_configured:
            return False, "Exchange API is not configured.", {}

        if self.exchange == "delta":
            base_url = "https://cdn.testnet.delta.exchange" if self.testnet else "https://api.india.delta.exchange"
            path = "/v2/orders"
            ts = str(int(time.time()))
            side = "buy" if direction.upper() in ("BUY", "LONG") else "sell"
            payload = {
                "product_symbol": symbol,
                "size": int(size) if "BTC" not in symbol else max(1, int(size)),
                "side": side,
                "order_type": "market_order" if order_type == "market" else "limit_order",
            }
            if order_type != "market" and price:
                payload["limit_price"] = str(price)
            if stop_loss:
                payload["stop_loss_price"] = str(stop_loss)
            if take_profit:
                payload["take_profit_price"] = str(take_profit)

            body_str = json.dumps(payload)
            sig = self._sign_delta_request("POST", path, body_str, ts)
            headers = {
                "api-key": self.api_key,
                "timestamp": ts,
                "signature": sig,
                "Content-Type": "application/json",
            }
            try:
                resp = requests.post(f"{base_url}{path}", headers=headers, data=body_str, timeout=10)
                if resp.status_code in (200, 201):
                    data = resp.json()
                    order_id = str(data.get("result", {}).get("id", f"DELTA_{int(time.time())}"))
                    return True, order_id, data
                return False, f"Delta order rejected (HTTP {resp.status_code}): {resp.text[:200]}", {}
            except Exception as e:
                return False, f"Delta request failed: {e}", {}

        elif self.exchange == "binance":
            ts = int(time.time() * 1000)
            side = "BUY" if direction.upper() in ("BUY", "LONG") else "SELL"
            b_sym = self.format_binance_symbol(symbol)

            params: Dict[str, Any] = {
                "symbol": b_sym,
                "side": side,
                "type": "MARKET" if order_type.lower() == "market" else "LIMIT",
                "timestamp": ts,
                "recvWindow": 5000,
            }
            if b_sym.startswith("BTC"):
                params["quantity"] = f"{float(size):.4f}"
            elif b_sym.startswith("ETH"):
                params["quantity"] = f"{float(size):.3f}"
            else:
                params["quantity"] = f"{float(size):.2f}"

            if order_type.lower() != "market" and price:
                params["timeInForce"] = "GTC"
                params["price"] = f"{float(price):.2f}"

            query_str = urllib.parse.urlencode(params)
            sig = hmac.new(self.api_secret.encode("utf-8"), query_str.encode("utf-8"), hashlib.sha256).hexdigest()
            headers = {
                "X-MBX-APIKEY": self.api_key,
                "User-Agent": "TradingBot/1.0",
            }

            use_futures = getattr(self, "market", "auto") == "futures"
            endpoints = []
            if use_futures:
                endpoints.append(("futures", "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com", "/fapi/v1/order"))
                endpoints.append(("spot", "https://testnet.binance.vision" if self.testnet else "https://api.binance.com", "/api/v3/order"))
            else:
                endpoints.append(("spot", "https://testnet.binance.vision" if self.testnet else "https://api.binance.com", "/api/v3/order"))
                endpoints.append(("futures", "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com", "/fapi/v1/order"))

            last_err = ""
            for mkt_type, base_url, path in endpoints:
                try:
                    resp = requests.post(f"{base_url}{path}?{query_str}&signature={sig}", headers=headers, timeout=10)
                    if resp.status_code in (200, 201):
                        self.market = mkt_type
                        data = resp.json()
                        order_id = str(data.get("orderId", f"BINANCE_{int(time.time())}"))
                        return True, order_id, data
                    last_err = f"Binance {mkt_type.title()} rejected (HTTP {resp.status_code}): {resp.text[:200]}"
                except Exception as e:
                    last_err = f"Binance {mkt_type.title()} order request failed: {e}"

            return False, last_err, {}

        return False, f"Live ordering on {self.exchange} preview.", {}

    def cancel_order(self, symbol: str, order_id: str) -> Tuple[bool, str]:
        """Cancel live order on exchange."""
        if not self.is_configured:
            return False, "Exchange API is not configured."

        if self.exchange == "binance":
            ts = int(time.time() * 1000)
            b_sym = self.format_binance_symbol(symbol)
            params = {
                "symbol": b_sym,
                "orderId": order_id,
                "timestamp": ts,
                "recvWindow": 5000,
            }
            query_str = urllib.parse.urlencode(params)
            sig = hmac.new(self.api_secret.encode("utf-8"), query_str.encode("utf-8"), hashlib.sha256).hexdigest()
            headers = {"X-MBX-APIKEY": self.api_key, "User-Agent": "TradingBot/1.0"}

            use_futures = getattr(self, "market", "auto") == "futures"
            endpoints = []
            if use_futures:
                endpoints.append(("futures", "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com", "/fapi/v1/order"))
                endpoints.append(("spot", "https://testnet.binance.vision" if self.testnet else "https://api.binance.com", "/api/v3/order"))
            else:
                endpoints.append(("spot", "https://testnet.binance.vision" if self.testnet else "https://api.binance.com", "/api/v3/order"))
                endpoints.append(("futures", "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com", "/fapi/v1/order"))

            last_err = ""
            for mkt_type, base_url, path in endpoints:
                try:
                    resp = requests.delete(f"{base_url}{path}?{query_str}&signature={sig}", headers=headers, timeout=10)
                    if resp.status_code == 200:
                        return True, f"Binance {mkt_type.title()} order {order_id} cancelled successfully."
                    last_err = f"Binance {mkt_type.title()} cancel failed (HTTP {resp.status_code}): {resp.text[:200]}"
                except Exception as e:
                    last_err = f"Binance {mkt_type.title()} cancel request failed: {e}"

            return False, last_err

        elif self.exchange == "delta":
            base_url = "https://cdn.testnet.delta.exchange" if self.testnet else "https://api.india.delta.exchange"
            path = "/v2/orders"
            ts = str(int(time.time()))
            payload = {"product_symbol": symbol, "order_id": order_id}
            body_str = json.dumps(payload)
            sig = self._sign_delta_request("DELETE", path, body_str, ts)
            headers = {
                "api-key": self.api_key,
                "timestamp": ts,
                "signature": sig,
                "Content-Type": "application/json",
            }
            try:
                resp = requests.delete(f"{base_url}{path}", headers=headers, data=body_str, timeout=10)
                if resp.status_code in (200, 204):
                    return True, f"Delta order {order_id} cancelled successfully."
                return False, f"Delta cancel failed (HTTP {resp.status_code}): {resp.text[:200]}"
            except Exception as e:
                return False, f"Delta cancel request failed: {e}"

        return False, f"Cancel not supported on {self.exchange}."


# ==================================================================
# 2. MULTI-LAYER INTER-ENGINE DELIBERATION SYSTEM
# ==================================================================

@dataclass
class LayerPerspective:
    layer_id: str
    name: str
    stance: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 to 1.0 normalized
    argument: str  # Primary thesis
    critique_peers: str  # Cross-examination commentary
    key_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliberationOutcome:
    symbol: str
    timestamp: str
    consensus_score: float  # -1.0 to 1.0
    consensus_direction: str  # "LONG", "SHORT", "FLAT"
    verdict: str
    agreement_rate: float  # 0.0 to 1.0
    conviction_multiplier: float  # 0.0 to 1.30
    layers: Dict[str, LayerPerspective] = field(default_factory=dict)
    aligned_layers: List[str] = field(default_factory=list)
    dissenting_layers: List[str] = field(default_factory=list)
    discussion_dialogue: List[str] = field(default_factory=list)
    risk_approval: bool = True
    risk_summary: str = ""
    learning_notes: str = ""
    quality_score: float = 3.5
    mtf_alignment: int = 1
    mtf_summary: str = ""
    rel_vol: float = 1.0


class MultiLayerDeliberationEngine:
    """
    Coordinates collaborative 5-layer inter-engine deliberation with mutual dependence:
    - Layer 1: SMC Structure & Institutional Liquidity (Order blocks, FVGs, swing runs)
    - Layer 2: Technical Momentum & Indicators Pro (RSI, ADX, Supertrend, EMA structure)
    - Layer 3: ITB Machine Learning Engine (Ridge regression trajectory & statistical moments)
    - Layer 4: AI Bot Learning & Regime Expectancy (Adaptive memory, regime win rates & calibration)
    - Layer 5: Risk Guardian Arbiter (Portfolio limits, cross-layer vetoes & conviction sizing)
    
    Rather than isolated voting, each layer cross-examines peers to achieve true consensus.
    """

    def deliberate(self, symbol: str, df: pd.DataFrame, trader: Any) -> DeliberationOutcome:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if df is None or df.empty:
            df = trader._generate_dummy_candles(symbol, count=120)

        curr_price = float(df["close"].iloc[-1])
        atr_s = IndicatorEngine.atr_series(df, 14)
        atr_last = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0
        atr = atr_last if (not math.isnan(atr_last) and atr_last > 0) else max(0.5, curr_price * 0.005)

        # 1. SMC Structure & Institutional Liquidity Layer
        levels = detect_order_blocks_and_fvg(df)
        bob = levels.get("bullish_ob")
        sob = levels.get("bearish_ob")
        swing_hi = levels.get("swing_high") or curr_price
        swing_lo = levels.get("swing_low") or curr_price

        near_demand = False
        if bob:
            near_demand = (curr_price >= bob["bottom"] - atr * 0.5) and (curr_price <= bob["top"] + atr * 1.5)
        near_supply = False
        if sob:
            near_supply = (curr_price <= sob["top"] + atr * 0.5) and (curr_price >= sob["bottom"] - atr * 1.5)

        recent_close_5 = float(df["close"].iloc[-5]) if len(df) >= 5 else curr_price

        if near_demand or (curr_price > swing_hi - atr * 0.5 and curr_price > recent_close_5):
            smc_stance = "LONG"
            smc_score = 0.70
            smc_conf = 0.78
            smc_arg = (
                f"Price bouncing from Institutional Demand Block (${bob['bottom']:,.1f}–${bob['top']:,.1f}); "
                f"clear upward liquidity runway toward swing high ${swing_hi:,.1f}."
                if bob else f"Institutional expansion breaking structural high (${swing_hi:,.1f})."
            )
        elif near_supply or (curr_price < swing_lo + atr * 0.5 and curr_price < recent_close_5):
            smc_stance = "SHORT"
            smc_score = -0.70
            smc_conf = 0.78
            smc_arg = (
                f"Price rejecting Institutional Supply Block (${sob['bottom']:,.1f}–${sob['top']:,.1f}); "
                f"downward liquidity sweep targeting swing low ${swing_lo:,.1f}."
                if sob else f"Institutional structure breakdown below swing low (${swing_lo:,.1f})."
            )
        else:
            smc_stance = "NEUTRAL"
            smc_score = 0.0
            smc_conf = 0.50
            bob_str = f"${bob['bottom']:,.1f}" if bob else "N/A"
            sob_str = f"${sob['top']:,.1f}" if sob else "N/A"
            smc_arg = (
                f"Price equilibrating between Demand ({bob_str}) "
                f"and Supply ({sob_str}); waiting for institutional sweep."
            )

        # 2. Technical Momentum & Indicators Pro Layer
        eng = IndicatorEngine()
        snap = eng.compute(df)
        pro_score = float(np.clip(snap.score / 5.0, -1.0, 1.0))
        if snap.score >= 0.60:
            pro_stance = "LONG"
            pro_conf = min(0.92, 0.50 + abs(snap.score) * 0.08)
        elif snap.score <= -0.60:
            pro_stance = "SHORT"
            pro_conf = min(0.92, 0.50 + abs(snap.score) * 0.08)
        else:
            pro_stance = "NEUTRAL"
            pro_conf = 0.50

        pro_arg = (
            f"Confluence score {snap.score:+.2f}/5.0. RSI at {snap.rsi:.1f}, "
            f"ADX at {snap.adx:.1f} ({'trend accelerating' if snap.adx > 25 else 'ranging consolidation'}), "
            f"Supertrend confirms {pro_stance}."
        )

        # 3. ITB Machine Learning Engine Layer
        strat_itb = (
            getattr(trader, "_strategies_itb", {}).get(symbol)
            or getattr(trader, "_strategy_itb", None)
            or ITBStrategy(symbol=symbol)
        )
        itb_pred = strat_itb.predictor.predict(df, symbol=symbol)
        itb_score = float(np.clip(itb_pred.smoothed_indicator, -1.0, 1.0))
        if itb_score >= 0.08:
            itb_stance = "LONG"
        elif itb_score <= -0.08:
            itb_stance = "SHORT"
        else:
            itb_stance = "NEUTRAL"
        itb_conf = float(np.clip(itb_pred.confidence, 0.35, 0.95))
        itb_arg = (
            f"Ridge regression trajectory slope {itb_score:+.2f} ({itb_pred.zone}). "
            f"Statistical moments confirm {itb_stance} bias with {itb_conf*100:.0f}% predictive confidence."
        )

        # 4. AI Bot Learning & Adaptive Memory Layer
        strat_ai = (
            getattr(trader, "_strategies_ai", {}).get(symbol)
            or getattr(trader, "_strategy_ai", None)
            or AIBotLearning(symbol=symbol)
        )
        ai_regime = "NORMAL"
        try:
            reg = strat_ai.regime(df)
            ai_regime = getattr(reg, "name", str(reg))
        except Exception:
            pass

        ai_mem_stats = getattr(getattr(strat_ai, "mem", None), "stats", None)
        ai_recent_wr = getattr(ai_mem_stats, "recent_winrate", 0.50) if ai_mem_stats else 0.50
        ai_expectancy = getattr(ai_mem_stats, "expectancy", 0.0) if ai_mem_stats else 0.0
        ai_best_qual = getattr(ai_mem_stats, "best_quality", 4.2) if ai_mem_stats else 4.2

        ema20 = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        if curr_price > ema20 > ema50 and snap.rsi > 48:
            ai_stance = "LONG"
            ai_score = 0.75
        elif curr_price < ema20 < ema50 and snap.rsi < 52:
            ai_stance = "SHORT"
            ai_score = -0.75
        else:
            ai_score = float(np.clip((curr_price - ema50) / max(0.001, ema50) * 100.0, -1.0, 1.0))
            ai_stance = "LONG" if ai_score > 0.15 else ("SHORT" if ai_score < -0.15 else "NEUTRAL")

        ai_conf = float(np.clip(ai_recent_wr, 0.40, 0.90))
        ai_arg = (
            f"Market regime classified as {ai_regime}. Historical expectancy is {ai_expectancy:+.2f}R "
            f"(recent WR: {ai_recent_wr*100:.0f}%). Quality gate Q≥{ai_best_qual:.1f} calibrated."
        )

        # Inter-Layer Cross-Examination Critiques
        if smc_stance == pro_stance and smc_stance != "NEUTRAL":
            smc_critique = f"Confirms Technical Momentum: Structural liquidity runway aligns with indicator push."
        elif smc_stance != "NEUTRAL" and pro_stance != "NEUTRAL" and smc_stance != pro_stance:
            smc_critique = f"WARN Technical Momentum: Price is running into opposing institutional block! Reversal risk elevated."
        else:
            smc_critique = f"Observes mid-range structure; allows momentum to lead while watching swing bounds."

        if pro_stance == itb_stance and pro_stance != "NEUTRAL":
            pro_critique = f"Confirms ITB ML: Momentum acceleration perfectly matches statistical regression slope."
        elif pro_stance != "NEUTRAL" and itb_stance != "NEUTRAL" and pro_stance != itb_stance:
            pro_critique = f"WARN ITB ML: Indicator momentum opposes ML slope; flags potential divergence or whip."
        else:
            pro_critique = f"Momentum in neutral equilibrium; cross-referencing AI regime for direction."

        if itb_stance == smc_stance and itb_stance != "NEUTRAL":
            itb_critique = f"Quantitative regression verifies institutional accumulation/distribution zone."
        elif itb_stance != "NEUTRAL" and smc_stance != "NEUTRAL" and itb_stance != smc_stance:
            itb_critique = f"Statistical moments indicate distribution asymmetry contrary to local structure."
        else:
            itb_critique = f"Predictive feature weights indicate range-bound transition; recommends defensive sizing."

        if ai_expectancy > 0.10:
            ai_critique = f"Regime favorable (expectancy {ai_expectancy:+.2f}R). Validates active multi-engine execution."
        else:
            ai_critique = f"Regime choppy or expectancy thin. Demands strict unanimous consensus before entry."

        p_smc = LayerPerspective(
            layer_id="smc_structure",
            name="SMC & Structure",
            stance=smc_stance,
            confidence=round(smc_conf, 2),
            score=round(smc_score, 2),
            argument=smc_arg,
            critique_peers=smc_critique,
            key_metrics={"swing_high": swing_hi, "swing_low": swing_lo, "atr": atr},
        )
        p_pro = LayerPerspective(
            layer_id="technical_momentum",
            name="Technical Momentum (Pro)",
            stance=pro_stance,
            confidence=round(pro_conf, 2),
            score=round(pro_score, 2),
            argument=pro_arg,
            critique_peers=pro_critique,
            key_metrics={"rsi": snap.rsi, "adx": snap.adx, "score": snap.score},
        )
        p_itb = LayerPerspective(
            layer_id="itb_machine_learning",
            name="ITB Machine Learning",
            stance=itb_stance,
            confidence=round(itb_conf, 2),
            score=round(itb_score, 2),
            argument=itb_arg,
            critique_peers=itb_critique,
            key_metrics={"smoothed": itb_score, "zone": itb_pred.zone},
        )
        p_ai = LayerPerspective(
            layer_id="ai_bot_learning",
            name="AI Bot Learning",
            stance=ai_stance,
            confidence=round(ai_conf, 2),
            score=round(ai_score, 2),
            argument=ai_arg,
            critique_peers=ai_critique,
            key_metrics={"regime": ai_regime, "winrate": ai_recent_wr, "expectancy": ai_expectancy},
        )

        layers_map = {
            "smc_structure": p_smc,
            "technical_momentum": p_pro,
            "itb_machine_learning": p_itb,
            "ai_bot_learning": p_ai,
        }

        # 5. Risk Guardian Arbiter & Sizing Arbiter
        cfg = getattr(trader, "config", None)
        w_itb = getattr(cfg, "weight_itb", 0.35)
        w_pro = getattr(cfg, "weight_pro", 0.35)
        w_ai = getattr(cfg, "weight_ai", 0.30)
        w_smc = 0.20
        total_w = w_itb + w_pro + w_ai + w_smc

        composite_score = (
            (w_itb * itb_score) + (w_pro * pro_score) + (w_ai * ai_score) + (w_smc * smc_score)
        ) / total_w
        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        stances = [p_smc.stance, p_pro.stance, p_itb.stance, p_ai.stance]
        long_votes = sum(1 for s in stances if s == "LONG")
        short_votes = sum(1 for s in stances if s == "SHORT")
        neutral_votes = sum(1 for s in stances if s == "NEUTRAL")

        aligned_layers = []
        dissenting_layers = []

        if long_votes > short_votes:
            target_dir = "LONG"
            aligned_layers = [lid for lid, p in layers_map.items() if p.stance == "LONG"]
            dissenting_layers = [lid for lid, p in layers_map.items() if p.stance == "SHORT"]
        elif short_votes > long_votes:
            target_dir = "SHORT"
            aligned_layers = [lid for lid, p in layers_map.items() if p.stance == "SHORT"]
            dissenting_layers = [lid for lid, p in layers_map.items() if p.stance == "LONG"]
        else:
            target_dir = "FLAT"

        conviction_mult = 0.0
        risk_approval = False
        verdict = "NEUTRAL / RANGE ⚪"
        risk_summary = ""

        if long_votes >= 3 and short_votes == 0:
            target_dir = "LONG"
            if long_votes == 4:
                verdict = "STRONG BUY 🟢 (Unanimous 4/4 Synergy)"
                conviction_mult = 1.20
                risk_approval = True
                risk_summary = "Unanimous alignment across all 4 specialist layers. 1.20x conviction bonus authorized."
            else:
                verdict = "BUY 🟢 (Consensus 3/4 Aligned)"
                conviction_mult = 1.00
                risk_approval = True
                risk_summary = "Solid 3-layer consensus with no dissenting opposition. Standard 1.00x sizing approved."
        elif short_votes >= 3 and long_votes == 0:
            target_dir = "SHORT"
            if short_votes == 4:
                verdict = "STRONG SELL 🔴 (Unanimous 4/4 Synergy)"
                conviction_mult = 1.20
                risk_approval = True
                risk_summary = "Unanimous alignment across all 4 specialist layers. 1.20x conviction bonus authorized."
            else:
                verdict = "SELL 🔴 (Consensus 3/4 Aligned)"
                conviction_mult = 1.00
                risk_approval = True
                risk_summary = "Solid 3-layer consensus with no dissenting opposition. Standard 1.00x sizing approved."
        elif long_votes >= 2 and short_votes == 0 and composite_score >= 0.20:
            target_dir = "LONG"
            verdict = "MODERATE BUY 🟢 (Favorable Confluence)"
            conviction_mult = 0.85
            risk_approval = True
            risk_summary = "Moderate confluence without opposition. Defensive 0.85x sizing approved."
        elif short_votes >= 2 and long_votes == 0 and composite_score <= -0.20:
            target_dir = "SHORT"
            verdict = "MODERATE SELL 🔴 (Favorable Confluence)"
            conviction_mult = 0.85
            risk_approval = True
            risk_summary = "Moderate confluence without opposition. Defensive 0.85x sizing approved."
        elif long_votes >= 1 and short_votes >= 1:
            target_dir = "FLAT"
            verdict = "CONFLICT / VETOED ⚠️ (Engines Disagree)"
            conviction_mult = 0.0
            risk_approval = False
            risk_summary = f"Inter-layer conflict detected ({long_votes} Long vs {short_votes} Short). Vetoed by Risk Guardian to protect capital."
        else:
            target_dir = "FLAT"
            verdict = "NEUTRAL / RANGE ⚪ (Awaiting Confluence)"
            conviction_mult = 0.0
            risk_approval = False
            risk_summary = "Market in equilibrium or consolidation. Risk Guardian stands aside."

        dominant_votes = max(long_votes, short_votes, neutral_votes)
        agreement_rate = round(dominant_votes / 4.0, 2)

        # Relative Volume (RVOL) and Volatility Squeeze detection
        vol_s = df["volume"] if "volume" in df.columns else pd.Series(100.0, index=df.index)
        vol_curr = float(vol_s.iloc[-1]) if not vol_s.empty else 100.0
        vol_mean = float(vol_s.tail(20).mean()) if len(vol_s) >= 20 else vol_curr
        rel_vol = round(vol_curr / max(vol_mean, 1e-4), 2)
        in_squeeze = bool(snap.bb_width < 1.5 * atr)

        # Multi-Timeframe (MTF) Trend Structure
        ema9 = float(df["close"].ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(df["close"].ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        span_macro = min(100, max(20, len(df) - 1))
        ema_macro = float(df["close"].ewm(span=span_macro, adjust=False).mean().iloc[-1])

        tf_fast = "LONG" if ema9 > ema21 else "SHORT"
        tf_mid = "LONG" if ema21 > ema50 else "SHORT"
        tf_macro = "LONG" if ema50 > ema_macro else "SHORT"

        # Evaluate Multi-Timeframe Alignment
        mtf_agreed = sum(1 for v in [tf_fast, tf_mid, tf_macro] if v == target_dir) if target_dir != "FLAT" else 1
        mtf_summary = f"{mtf_agreed}/3 MTF aligned (Fast:{tf_fast}, Mid:{tf_mid}, Macro:{tf_macro})"

        # Dynamic Composite Quality Score (0 to 5.0)
        q_mtf = (mtf_agreed / 3.0) * 1.5
        q_vol = 0.8 if rel_vol >= 1.15 else (0.5 if rel_vol >= 0.85 else 0.2)
        q_rsi = 0.6 if (35.0 <= snap.rsi <= 65.0) else (0.4 if (28.0 <= snap.rsi <= 72.0) else 0.15)
        q_adx = 0.6 if snap.adx >= 22.0 else (0.4 if snap.adx >= 16.0 else 0.2)
        q_struct = 0.8 if (near_demand or near_supply) else 0.4
        q_ml = round(itb_conf * 0.7, 2)
        composite_quality = round(min(5.0, q_mtf + q_vol + q_rsi + q_adx + q_struct + q_ml), 2)

        # Dynamic Quality Gate & Synergy Modulation
        if target_dir != "FLAT" and composite_quality < 2.8 and risk_approval:
            verdict = f"VETOED BY QUALITY GATE (Q={composite_quality:.2f} < 2.8) ⚠️"
            risk_approval = False
            conviction_mult = 0.0
            risk_summary = f"Setup rejected: Insufficient quality score ({composite_quality:.2f}/5.0). Filtered to prevent whipsaws."
        elif target_dir != "FLAT" and composite_quality >= 4.0 and mtf_agreed == 3 and risk_approval:
            conviction_mult = min(1.30, conviction_mult + 0.10)
            verdict += f" 🎯 [Quality: {composite_quality:.1f} | MTF: 3/3]"

        dialogue = [
            f"🧱 <b>SMC Structure</b> ({smc_stance}): {smc_arg}\n   ↳ <i>Cross-Critique</i>: {smc_critique}",
            f"📊 <b>Technical Pro</b> ({pro_stance}): {pro_arg}\n   ↳ <i>Cross-Critique</i>: {pro_critique}",
            f"🤖 <b>ITB Machine Learning</b> ({itb_stance}): {itb_arg}\n   ↳ <i>Cross-Critique</i>: {itb_critique}",
            f"🧠 <b>AI Bot Learning</b> ({ai_stance}): {ai_arg}\n   ↳ <i>Cross-Critique</i>: {ai_critique}",
            f"🛡️ <b>Risk Guardian Arbiter</b>: {risk_summary} [{mtf_summary} | Vol×{rel_vol:.2f}]",
        ]

        learning_notes = (
            f"Dynamic Weights: ITB={w_itb:.0%}, Pro={w_pro:.0%}, AI={w_ai:.0%} | "
            f"Quality={composite_quality:.2f}/5.0 | MTF={mtf_summary} | "
            f"Learning Cycles: {getattr(cfg, 'learning_cycles', 0)}"
        )

        return DeliberationOutcome(
            symbol=symbol,
            timestamp=now_str,
            consensus_score=round(composite_score, 3),
            consensus_direction=target_dir,
            verdict=verdict,
            agreement_rate=agreement_rate,
            conviction_multiplier=round(conviction_mult, 2),
            layers=layers_map,
            aligned_layers=aligned_layers,
            dissenting_layers=dissenting_layers,
            discussion_dialogue=dialogue,
            risk_approval=risk_approval,
            risk_summary=risk_summary,
            learning_notes=learning_notes,
            quality_score=composite_quality,
            mtf_alignment=mtf_agreed,
            mtf_summary=mtf_summary,
            rel_vol=rel_vol,
        )


# ==================================================================
# 3. AUTO TRADER CORE ENGINE (FOR BOT & MONITORING)
# ==================================================================


class AutoTrader:
    """
    Automated trading engine with live Delta Exchange market feed,
    strategy signal generation, position lifecycle management, and
    interactive configuration.
    """

    def __init__(self, config: Optional[AutoTradeConfig] = None):
        self.config = config or AutoTradeConfig.load()
        if self.config.live_exchange == "binance":
            init_k = self.config.binance_api_key or self.config.exchange_api_key
            init_s = self.config.binance_api_secret or self.config.exchange_api_secret
        else:
            init_k = self.config.delta_api_key or self.config.exchange_api_key
            init_s = self.config.delta_api_secret or self.config.exchange_api_secret
        self.exchange_client = ExchangeApiClient(
            exchange=self.config.live_exchange,
            api_key=init_k,
            api_secret=init_s,
            market=getattr(self.config, "binance_market", "auto"),
        )
        self.positions: List[AutoTradePosition] = self._load_open_positions()
        self.closed_trades: List[AutoTradePosition] = self._load_trades_history()
        self.alerts: List[TradeLevelAlert] = self._load_alerts()
        self._recent_level_alerts: Dict[str, float] = {}
        self.is_running: bool = False
        self._strategy_pro: Optional[IndicatorsProStrategy] = None
        self._strategies_pro: Dict[str, IndicatorsProStrategy] = {}
        self._strategy_itb: Optional[ITBStrategy] = None
        self._strategy_ai: Optional[AIBotLearning] = None
        self._strategies_itb: Dict[str, ITBStrategy] = {}
        self._strategies_ai: Dict[str, AIBotLearning] = {}
        self._last_retrain_timestamp: float = 0.0
        self._deliberation_engine: MultiLayerDeliberationEngine = MultiLayerDeliberationEngine()
        self._init_strategy()

    @property
    def position(self) -> Optional[AutoTradePosition]:
        """Returns primary active position for backwards compatibility."""
        return self.positions[0] if self.positions else None

    @position.setter
    def position(self, pos: Optional[AutoTradePosition]) -> None:
        """Sets or clears primary position for backwards compatibility."""
        if pos is None:
            self.positions = []
        else:
            if not self.positions:
                self.positions = [pos]
            else:
                self.positions[0] = pos
        self._save_open_positions()

    def _load_open_positions(self) -> List[AutoTradePosition]:
        open_path = getattr(self.config, "open_positions_file", OPEN_POSITIONS_PATH)
        if os.path.exists(open_path):
            try:
                with open(open_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                positions = []
                for item in data:
                    if isinstance(item, dict):
                        filtered = {
                            k: v
                            for k, v in item.items()
                            if k in AutoTradePosition.__dataclass_fields__
                        }
                        positions.append(AutoTradePosition(**filtered))
                logger.info("Loaded %d open positions from %s", len(positions), open_path)
                return positions
            except Exception as exc:
                logger.warning("Could not read open positions %s: %s", open_path, exc)
        return []

    def _save_open_positions(self) -> None:
        open_path = getattr(self.config, "open_positions_file", OPEN_POSITIONS_PATH)
        try:
            tmp_path = f"{open_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in self.positions], f, indent=2)
            os.replace(tmp_path, open_path)
            logger.info("Saved %d open positions to %s", len(self.positions), open_path)
        except Exception as exc:
            logger.warning("Could not save open positions %s: %s", open_path, exc)

    def _load_trades_history(self) -> List[AutoTradePosition]:
        history_path = getattr(self.config, "trades_history_file", TRADES_HISTORY_PATH)
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                trades = []
                for item in data:
                    if isinstance(item, dict):
                        filtered = {
                            k: v
                            for k, v in item.items()
                            if k in AutoTradePosition.__dataclass_fields__
                        }
                        trades.append(AutoTradePosition(**filtered))
                logger.info("Loaded %d historical trades from %s", len(trades), history_path)
                return trades
            except Exception as exc:
                logger.warning("Could not read trades history %s: %s", history_path, exc)
        return []

    def _save_trades_history(self) -> None:
        history_path = getattr(self.config, "trades_history_file", TRADES_HISTORY_PATH)
        try:
            tmp_path = f"{history_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump([asdict(t) for t in self.closed_trades[-100:]], f, indent=2)
            os.replace(tmp_path, history_path)
            logger.info("Saved %d trades to history %s", len(self.closed_trades), history_path)
        except Exception as exc:
            logger.warning("Could not save trades history %s: %s", history_path, exc)

    def _load_alerts(self) -> List[TradeLevelAlert]:
        alerts_path = getattr(self.config, "alerts_file", ALERTS_FILE_PATH)
        if os.path.exists(alerts_path):
            try:
                with open(alerts_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                alerts = []
                for item in data:
                    if isinstance(item, dict):
                        filtered = {
                            k: v
                            for k, v in item.items()
                            if k in TradeLevelAlert.__dataclass_fields__
                        }
                        alerts.append(TradeLevelAlert(**filtered))
                logger.info("Loaded %d alerts from %s", len(alerts), alerts_path)
                return alerts
            except Exception as exc:
                logger.warning("Could not read alerts %s: %s", alerts_path, exc)
        return []

    def _save_alerts(self) -> None:
        alerts_path = getattr(self.config, "alerts_file", ALERTS_FILE_PATH)
        try:
            tmp_path = f"{alerts_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump([asdict(a) for a in self.alerts], f, indent=2)
            os.replace(tmp_path, alerts_path)
            logger.info("Saved %d alerts to %s", len(self.alerts), alerts_path)
        except Exception as exc:
            logger.warning("Could not save alerts %s: %s", alerts_path, exc)

    def _init_strategy(self) -> None:
        self._strategies_pro = {}
        self._strategies_itb = {}
        for sym in self.config.symbols:
            self._strategies_pro[sym] = IndicatorsProStrategy(
                symbol=sym,
                risk_per_trade=self.config.risk_pct,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
                tp_value=self.config.tp_value,
                sl_value=self.config.sl_value,
                lot_size=self.config.lot_size,
                lot_mode=self.config.lot_mode,
            )
            self._strategies_ai[sym] = AIBotLearning(symbol=sym)
            self._strategies_itb[sym] = ITBStrategy(
                symbol=sym,
                risk_per_trade=self.config.risk_pct,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
                tp_value=self.config.tp_value,
                sl_value=self.config.sl_value,
                lot_size=self.config.lot_size,
                lot_mode=self.config.lot_mode,
            )
        self._strategy_pro = self._strategies_pro.get(self.config.symbol) or (
            IndicatorsProStrategy(
                symbol=self.config.symbol,
                risk_per_trade=self.config.risk_pct,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
                tp_value=self.config.tp_value,
                sl_value=self.config.sl_value,
                lot_size=self.config.lot_size,
                lot_mode=self.config.lot_mode,
            )
        )
        self._strategy_ai = self._strategies_ai.get(self.config.symbol) or AIBotLearning(symbol=self.config.symbol)
        self._strategy_itb = self._strategies_itb.get(self.config.symbol) or (
            ITBStrategy(
                symbol=self.config.symbol,
                risk_per_trade=self.config.risk_pct,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
                tp_value=self.config.tp_value,
                sl_value=self.config.sl_value,
                lot_size=self.config.lot_size,
                lot_mode=self.config.lot_mode,
            )
        )
        if RiskManager:
            risk_cfg = RiskConfig(
                base_risk_pct=self.config.risk_pct,
                min_risk_pct=0.05,
                max_risk_pct=5.0,
                max_daily_loss_pct=self.config.max_daily_loss_pct,
                auto_be=getattr(self.config, "auto_breakeven", True),
                rf=getattr(self.config, "risk_factor", 1.0),
            )
            state_p = getattr(self.config, "risk_state_file", RISK_STATE_PATH)
            self.risk_manager = RiskManager(
                equity=self.config.equity,
                symbol=self.config.symbol,
                config=risk_cfg,
                state_file=state_p,
            )
        else:
            self.risk_manager = None

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
        m_lower = mode.lower()
        if m_lower in ("auto", "risk_pct", "risk", "dynamic"):
            self.config.lot_mode = "risk_pct"
            self.config.risk_pct = float(size)
            mode_desc = "Risk % lot sizing"
        else:
            self.config.lot_mode = "fixed"
            mode_desc = "Fixed lot size"

        for strat in self._strategies_pro.values():
            strat.set_lot_size(self.config.lot_size, self.config.lot_mode)
        if self._strategy_pro:
            self._strategy_pro.set_lot_size(self.config.lot_size, self.config.lot_mode)
        for strat_ai in self._strategies_ai.values():
            strat_ai.tp_mode = self.config.tp_mode
        for strat_itb in self._strategies_itb.values():
            strat_itb.set_lot_size(self.config.lot_size, self.config.lot_mode)
        if self._strategy_ai:
            self._strategy_ai.tp_mode = self.config.tp_mode
        if self._strategy_itb:
            self._strategy_itb.set_lot_size(self.config.lot_size, self.config.lot_mode)
        self.config.save()

        return True, f"Lot size updated: {self.config.lot_size} ({mode_desc})"

    def set_auto_lot(self, risk_pct: Optional[float] = None) -> Tuple[bool, str]:
        """
        Activate dynamic Auto Lot Sizing.
        Position size is calculated automatically per trade from account equity, risk %, and stop loss distance.
        """
        if risk_pct is not None:
            if risk_pct <= 0:
                return False, "Error: Risk percentage must be greater than 0."
            self.config.risk_pct = float(risk_pct)
        self.config.lot_mode = "risk_pct"
        for strat in self._strategies_pro.values():
            strat.set_lot_size(self.config.lot_size, self.config.lot_mode)
        if self._strategy_pro:
            self._strategy_pro.set_lot_size(self.config.lot_size, self.config.lot_mode)
        for strat_itb in self._strategies_itb.values():
            strat_itb.set_lot_size(self.config.lot_size, self.config.lot_mode)
        if self._strategy_itb:
            self._strategy_itb.set_lot_size(self.config.lot_size, self.config.lot_mode)
        self.config.save()
        risk_usd = self.config.equity * (self.config.risk_pct / 100.0)
        return (
            True,
            f"Auto lot sizing activated: Risk {self.config.risk_pct}% per trade "
            f"(${risk_usd:,.2f} risk on ${self.config.equity:,.2f} equity)",
        )

    def set_manual_lot(self, size: Optional[float] = None) -> Tuple[bool, str]:
        """
        Activate Manual Fixed Lot Sizing.
        Position size uses user-specified fixed volume per trade.
        """
        if size is not None:
            if size <= 0:
                return False, "Error: Lot size must be greater than 0."
            self.config.lot_size = float(size)
        self.config.lot_mode = "fixed"
        for strat in self._strategies_pro.values():
            strat.set_lot_size(self.config.lot_size, self.config.lot_mode)
        if self._strategy_pro:
            self._strategy_pro.set_lot_size(self.config.lot_size, self.config.lot_mode)
        for strat_itb in self._strategies_itb.values():
            strat_itb.set_lot_size(self.config.lot_size, self.config.lot_mode)
        if self._strategy_itb:
            self._strategy_itb.set_lot_size(self.config.lot_size, self.config.lot_mode)
        self.config.save()
        return (
            True,
            f"Manual lot sizing activated: Fixed {self.config.lot_size} lots per trade",
        )

    def calculate_lot_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        conviction_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate position lot size according to active mode:
        - Manual/Fixed: returns configured fixed lot size (optionally scaled by conviction)
        - Auto/Risk %: dynamically sizes lots from equity, risk %, and stop distance
        """
        symbol_upper = self.normalize_symbol(symbol)
        mult = conviction_multiplier if conviction_multiplier > 0 else 1.0
        is_btc = "BTC" in symbol_upper
        min_lot = 0.001 if is_btc else 0.01
        decimals = 3 if is_btc else 2

        if self.config.lot_mode in ("fixed", "manual"):
            return max(min_lot, round(self.config.lot_size * mult, 4))

        # Auto dynamic risk-based sizing
        if self.risk_manager:
            self.risk_manager.state.equity = self.config.equity
            self.risk_manager.config.base_risk_pct = self.config.risk_pct
            res = self.risk_manager.position_size(
                entry=entry_price,
                stop=stop_loss,
                equity=self.config.equity,
                symbol=symbol_upper,
            )
            if res.lots > 0:
                return max(min_lot, round(res.lots * mult, decimals))

        price_risk = max(0.0001, abs(entry_price - stop_loss))
        risk_budget = self.config.equity * (self.config.risk_pct / 100.0)
        raw_lots = (risk_budget / price_risk) * mult
        lots = max(min_lot, round(raw_lots, decimals))
        return lots

    def get_lot_size_report(self) -> str:
        """Format a rich status card displaying current Auto and Manual lot sizing configuration."""
        is_auto = self.config.lot_mode in ("auto", "risk_pct", "risk", "dynamic")
        mode_badge = "🤖 <b>AUTO</b> (Dynamic Risk)" if is_auto else "👤 <b>MANUAL</b> (Fixed Lots)"
        mode_desc = f"Risk {self.config.risk_pct}% sizing" if is_auto else "Fixed lot size"

        risk_dollars = self.config.equity * (self.config.risk_pct / 100.0)

        # Sample calculations
        sample_btc_lot = self.calculate_lot_size("BTCUSD", entry_price=80000.0, stop_loss=79000.0)
        sample_xau_lot = self.calculate_lot_size("XAUTUSD", entry_price=4100.0, stop_loss=4060.0)

        return (
            f"📦 <b>Current Lot Size Configuration</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Active Mode</b>: {mode_badge}\n"
            f"• <b>Mode</b>: {mode_desc}\n"
            f"• <b>Manual Lot Size</b>: <code>{self.config.lot_size} lots</code>\n"
            f"• <b>Auto Risk Sizing</b>: <code>{self.config.risk_pct}%</code> equity "
            f"(${risk_dollars:,.2f} risk on ${self.config.equity:,.2f} equity)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Sample Auto Sizing Estimates:</b>\n"
            f"• <b>BTCUSD</b> ($1,000 SL dist): <code>{sample_btc_lot} lots</code>\n"
            f"• <b>XAUTUSD</b> ($40 SL dist): <code>{sample_xau_lot} lots</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ <b>Switch & Configure Options:</b>\n"
            f"• <code>/lotsize auto</code> ➔ Enable dynamic auto sizing\n"
            f"• <code>/lotsize auto 2%</code> ➔ Auto sizing with 2% risk\n"
            f"• <code>/lotsize manual 0.05</code> ➔ Enable manual sizing with 0.05 lots\n"
            f"• <code>/lotsize 0.05</code> ➔ Set fixed lot size to 0.05\n"
            f"• <code>/lotsize 1.0</code> ➔ Set fixed lot size to 1.0\n"
            f"• <code>/lotsize risk 2%</code> ➔ Set risk-based sizing to 2% equity"
        )

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

        for strat in self._strategies_pro.values():
            strat.set_tp_sl(
                tp_val=self.config.tp_value,
                sl_val=self.config.sl_value,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
            )
        if self._strategy_pro:
            self._strategy_pro.set_tp_sl(
                tp_val=self.config.tp_value,
                sl_val=self.config.sl_value,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
            )
        for strat_ai in self._strategies_ai.values():
            strat_ai.tp_mode = self.config.tp_mode
        for strat_itb in self._strategies_itb.values():
            strat_itb.set_tp_sl(
                tp_val=self.config.tp_value,
                sl_val=self.config.sl_value,
                tp_mode=self.config.tp_mode,
                sl_mode=self.config.sl_mode,
            )
        if self._strategy_ai:
            self._strategy_ai.tp_mode = self.config.tp_mode
        if self._strategy_itb:
            self._strategy_itb.set_tp_sl(
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

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
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
            "ETHEREUM": "ETHUSD",
            "ETHUSDT": "ETHUSD",
            "SOL": "SOLUSD",
            "SOLANA": "SOLUSD",
            "SOLUSDT": "SOLUSD",
            "XRP": "XRPUSD",
            "RIPPLE": "XRPUSD",
            "XRPUSDT": "XRPUSD",
        }
        return alias_map.get(cleaned, cleaned)

    def set_symbol(self, symbol: str) -> str:
        target = self.normalize_symbol(symbol)
        self.config.symbol = target
        if target not in self.config.symbols:
            self.config.symbols.insert(0, target)
        if getattr(self, "risk_manager", None):
            self.risk_manager.set_symbol(target)
        self._init_strategy()
        self.config.save()
        return f"Auto trade symbol set to: {self.config.symbol} (Active: {', '.join(self.config.symbols)})"

    def set_symbols(self, symbols: List[str]) -> Tuple[bool, str]:
        """Configure which symbols the bot automatically scans and trades."""
        cleaned_list: List[str] = []
        for s in symbols:
            target = self.normalize_symbol(s)
            if target and target not in cleaned_list:
                cleaned_list.append(target)

        if not cleaned_list:
            return False, "Error: No valid symbols provided."

        self.config.symbols = cleaned_list
        if cleaned_list:
            self.config.symbol = cleaned_list[0]
        self._init_strategy()
        self.config.save()
        syms_str = ", ".join(self.config.symbols)
        return True, f"Auto-trading active symbols updated: {syms_str}"

    def add_symbol(self, symbol: str) -> Tuple[bool, str]:
        target = self.normalize_symbol(symbol)
        if target in self.config.symbols:
            return False, f"Symbol {target} is already in the active auto-trade list."
        self.config.symbols.append(target)
        self._init_strategy()
        self.config.save()
        return True, f"Added {target} to auto-trade list. Now trading: {', '.join(self.config.symbols)}"

    def remove_symbol(self, symbol: str) -> Tuple[bool, str]:
        target = self.normalize_symbol(symbol)
        if target not in self.config.symbols:
            return False, f"Symbol {target} is not in the active auto-trade list."
        if len(self.config.symbols) <= 1:
            return False, "Cannot remove the only remaining symbol. Add another symbol first."
        self.config.symbols.remove(target)
        if self.config.symbol == target:
            self.config.symbol = self.config.symbols[0]
        self._init_strategy()
        self.config.save()
        return True, f"Removed {target} from auto-trade list. Now trading: {', '.join(self.config.symbols)}"

    def set_max_positions(
        self,
        max_pos: Optional[int] = None,
        per_symbol: Optional[int] = None,
        total: Optional[int] = None,
    ) -> Tuple[bool, str]:
        actual_max = total if total is not None else (max_pos if max_pos is not None else 5)
        if actual_max < 1 or actual_max > 20:
            return False, "Error: Max positions must be between 1 and 20."
        self.config.max_positions = int(actual_max)
        if per_symbol is not None:
            if per_symbol < 1 or per_symbol > actual_max:
                return False, f"Error: Max positions per symbol must be between 1 and {actual_max}."
            self.config.max_positions_per_symbol = int(per_symbol)
        self.config.save()
        return (
            True,
            f"Position limits updated: Total max {self.config.max_positions} concurrent trades "
            f"(Max {self.config.max_positions_per_symbol} per symbol).",
        )

    def get_positions(self, symbol: Optional[str] = None) -> List[AutoTradePosition]:
        """Return all active positions, optionally filtered by symbol."""
        if not symbol:
            return list(self.positions)
        target = self.normalize_symbol(symbol)
        return [p for p in self.positions if p.symbol == target]

    def set_strategy_type(self, strategy_type: str) -> Tuple[bool, str]:
        st = strategy_type.strip().lower()
        alias_map = {
            "itb": "itb_ml",
            "intelligent": "itb_ml",
            "itb_ml": "itb_ml",
            "ml": "itb_ml",
            "indicators": "indicators_pro",
            "pro": "indicators_pro",
            "indicators_pro": "indicators_pro",
            "ai": "ai_learning",
            "ai_learning": "ai_learning",
            "combined": "combined_ensemble",
            "ensemble": "combined_ensemble",
        }
        resolved = alias_map.get(st, "combined_ensemble")
        self.config.strategy_type = resolved
        self.config.save()
        names = {
            "indicators_pro": "Indicators Pro (Technical Indicators)",
            "itb_ml": "Intelligent Trading Bot (ITB Machine Learning)",
            "ai_learning": "AI Bot Learning (Adaptive Reinforcement)",
            "combined_ensemble": "Combined Ensemble System (All Engines)",
        }
        name = names.get(resolved, resolved)
        return True, f"Strategy switched to: <b>{name}</b> (Combined Ensemble active)"

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

    # ------------------------------------------------------------------
    # Trading Mode & Capital / Funds Management
    # ------------------------------------------------------------------
    def set_trading_mode(self, mode: str) -> Tuple[bool, str]:
        """Switch between Paper Trading (simulated funds) and Live Trading (real exchange API)."""
        target = mode.strip().lower()
        if target in ("paper", "demo", "sim", "virtual"):
            self.config.trading_mode = "paper"
            self.config.save()
            return (
                True,
                f"📄 <b>PAPER TRADING MODE ACTIVATED</b>\n"
                f"• Paper Capital: <code>${self.config.equity:,.2f}</code>\n"
                f"• Simulated execution with real-time market data.\n"
                f"• Zero financial risk to real capital.\n\n"
                f"<i>💡 Use /capital add or /capital set to adjust funds anytime.</i>",
            )
        elif target in ("live", "real", "prod"):
            if not self.exchange_client.is_configured:
                return (
                    False,
                    f"⚠️ <b>CANNOT ACTIVATE LIVE TRADING</b>\n\n"
                    f"Exchange API credentials are not configured.\n"
                    f"To enable live trading with real funds, configure your exchange API:\n"
                    f"• <code>/api set delta &lt;API_KEY&gt; &lt;API_SECRET&gt;</code>\n"
                    f"• or export <code>EXCHANGE_API_KEY</code> and <code>EXCHANGE_API_SECRET</code>\n\n"
                    f"<i>Trading remains safely in PAPER mode.</i>",
                )
            self.config.trading_mode = "live"
            self.config.save()
            return (
                True,
                f"🚨 <b>LIVE TRADING MODE ACTIVATED</b>\n"
                f"• Exchange: <code>{self.config.live_exchange.upper()}</code>\n"
                f"• Real orders will be dispatched to the exchange.\n"
                f"• API Key: <code>{self.exchange_client.mask_key(self.exchange_client.api_key)}</code>\n\n"
                f"<i>⚠️ Monitor open positions carefully with /positions or /close.</i>",
            )
        else:
            return (
                False,
                f"Unknown mode '{mode}'. Use <code>/mode paper</code> or <code>/mode live</code>.",
            )

    def set_paper_capital(self, amount: float) -> Tuple[bool, str]:
        """Set base paper trading capital (e.g. $100)."""
        if amount <= 0:
            return False, "Error: Capital must be greater than $0."
        old = self.config.equity
        self.config.paper_capital = float(amount)
        self.config.equity = float(amount)
        self.config.save()
        return (
            True,
            f"Paper trading capital set to <b>${amount:,.2f}</b> (Previous: ${old:,.2f}).",
        )

    def add_funds(self, amount: float) -> Tuple[bool, str]:
        """Deposit / add funds to current paper capital."""
        if amount <= 0:
            return False, "Error: Deposit amount must be greater than $0."
        old = self.config.equity
        self.config.equity += float(amount)
        self.config.total_deposited += float(amount)
        self.config.save()
        return (
            True,
            f"💰 <b>Funds Added Successfully</b>\n"
            f"• Deposited: <code>+${amount:,.2f}</code>\n"
            f"• Previous Balance: <code>${old:,.2f}</code>\n"
            f"• New Balance: <b>${self.config.equity:,.2f}</b>",
        )

    def reduce_funds(self, amount: float) -> Tuple[bool, str]:
        """Withdraw / reduce funds from current paper capital."""
        if amount <= 0:
            return False, "Error: Reduction amount must be greater than $0."
        if amount > self.config.equity:
            return (
                False,
                f"Error: Cannot reduce ${amount:,.2f}. Current balance is only ${self.config.equity:,.2f}.",
            )
        old = self.config.equity
        self.config.equity -= float(amount)
        self.config.total_withdrawn += float(amount)
        self.config.save()
        return (
            True,
            f"💸 <b>Funds Reduced Successfully</b>\n"
            f"• Withdrawn: <code>-${amount:,.2f}</code>\n"
            f"• Previous Balance: <code>${old:,.2f}</code>\n"
            f"• New Balance: <b>${self.config.equity:,.2f}</b>",
        )

    def reset_funds(self, amount: float = 100.0) -> Tuple[bool, str]:
        """Reset paper trading capital back to default $100 or specified value."""
        old = self.config.equity
        self.config.paper_capital = float(amount)
        self.config.equity = float(amount)
        self.config.total_deposited = 0.0
        self.config.total_withdrawn = 0.0
        self.config.save()
        return (
            True,
            f"🔄 Paper capital reset to <b>${amount:,.2f}</b> (Previous: ${old:,.2f}).",
        )

    def get_capital_report(self) -> str:
        """Detailed capital, deposits, withdrawals, and equity dashboard."""
        mode_icon = "📄 PAPER TRADING" if self.config.trading_mode == "paper" else "🚨 LIVE TRADING"
        mode_color = "🟢" if self.config.trading_mode == "paper" else "🔴"

        unrealized = 0.0
        for p in self.positions:
            df = self.fetch_candles(p.symbol, count=1)
            cp = float(df["close"].iloc[-1]) if not df.empty else p.entry_price
            unrealized += p.current_pnl(cp)

        realized = sum(t.pnl for t in self.closed_trades)
        net_pnl = realized + unrealized
        sign = "+" if net_pnl >= 0 else ""
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
        gain_pct = (
            ((self.config.equity - self.config.paper_capital) / self.config.paper_capital * 100.0)
            if self.config.paper_capital > 0
            else 0.0
        )
        gain_sign = "+" if gain_pct >= 0 else ""
        load_pct = (len(self.positions) / max(1, self.config.max_positions)) * 100.0
        load_bar = make_modern_meter(load_pct, width=8, fill_char="■", empty_char="░")

        return (
            f"💼 <b>TRADING CAPITAL & FUNDS DASHBOARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Trading Mode</b>: {mode_color} <b>{mode_icon}</b>\n"
            f"• <b>Current Equity / Funds</b>: <code>${self.config.equity:,.2f}</code>\n"
            f"• <b>Base Paper Capital</b>: <code>${self.config.paper_capital:,.2f}</code>\n"
            f"• <b>Total Deposited</b>: <code>+${self.config.total_deposited:,.2f}</code>\n"
            f"• <b>Total Withdrawn</b>: <code>-${self.config.total_withdrawn:,.2f}</code>\n"
            f"• <b>Net PnL (Closed)</b>: <code>{'+' if realized >= 0 else ''}${realized:,.2f}</code>\n"
            f"• <b>Unrealized PnL</b>: <code>{'+' if unrealized >= 0 else ''}${unrealized:,.2f}</code>\n"
            f"• <b>Total Return (ROI)</b>: {pnl_emoji} <b>{gain_sign}{gain_pct:.2f}%</b> ({sign}${net_pnl:,.2f})\n"
            f"• <b>Open Capacity</b>: <code>[{load_bar}]</code> <code>{len(self.positions)}/{self.config.max_positions}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Manage Funds:</b>\n"
            f"• <code>/capital set 100</code> — Set capital to $100\n"
            f"• <code>/capital add 50</code> (or <code>/deposit 50</code>) — Add funds\n"
            f"• <code>/capital reduce 25</code> (or <code>/withdraw 25</code>) — Reduce funds\n"
            f"• <code>/capital reset</code> — Reset back to default $100\n"
            f"• <code>/mode [paper|live]</code> — Switch trading mode"
        )

    def set_exchange_api(self, exchange: str, api_key: str, api_secret: str, market: str = "auto") -> Tuple[bool, str]:
        """Configure live exchange API credentials."""
        ex = exchange.strip().lower()
        if ex not in ("delta", "binance"):
            return False, f"Unsupported exchange '{exchange}'. Supported: delta, binance"

        self.config.live_exchange = ex
        self.config.exchange_api_key = api_key.strip()
        self.config.exchange_api_secret = api_secret.strip()
        if ex == "binance":
            self.config.binance_api_key = api_key.strip()
            self.config.binance_api_secret = api_secret.strip()
            if market:
                self.config.binance_market = market.lower()
        elif ex == "delta":
            self.config.delta_api_key = api_key.strip()
            self.config.delta_api_secret = api_secret.strip()

        self.config.save()
        self.exchange_client = ExchangeApiClient(
            exchange=ex,
            api_key=api_key.strip(),
            api_secret=api_secret.strip(),
            market=getattr(self.config, "binance_market", "auto"),
        )
        ok, msg, _ = self.exchange_client.test_connection()
        masked_k = self.exchange_client.mask_key(api_key)
        return (
            True,
            f"✅ <b>Exchange API Configured</b>\n"
            f"• Exchange: <code>{ex.upper()}</code>\n"
            f"• API Key: <code>{masked_k}</code>\n"
            f"• Connection Test: <i>{msg}</i>\n\n"
            f"<i>💡 To trade with this exchange, switch mode with /mode live</i>",
        )

    def clear_exchange_api(self, exchange: Optional[str] = None) -> str:
        """Clear exchange API credentials and revert safely to paper mode."""
        ex = (exchange or self.config.live_exchange).strip().lower()
        if ex == "binance":
            self.config.binance_api_key = ""
            self.config.binance_api_secret = ""
        elif ex == "delta":
            self.config.delta_api_key = ""
            self.config.delta_api_secret = ""
        self.config.exchange_api_key = ""
        self.config.exchange_api_secret = ""
        if self.config.trading_mode == "live":
            self.config.trading_mode = "paper"
        self.config.save()
        self.exchange_client = ExchangeApiClient(
            exchange=self.config.live_exchange,
            market=getattr(self.config, "binance_market", "auto"),
        )
        return f"Exchange API credentials for {ex.upper()} cleared. Mode reverted to PAPER TRADING 📄."

    def switch_exchange(self, exchange: str, market: Optional[str] = None) -> Tuple[bool, str]:
        """Switch active exchange between delta and binance."""
        ex = exchange.strip().lower()
        if ex not in ("delta", "binance"):
            return False, f"Unsupported exchange '{exchange}'. Supported: delta, binance"
        self.config.live_exchange = ex
        if market and ex == "binance":
            self.config.binance_market = market.lower()

        if ex == "binance":
            api_k = self.config.binance_api_key or os.getenv("BINANCE_API_KEY") or self.config.exchange_api_key
            api_s = self.config.binance_api_secret or os.getenv("BINANCE_API_SECRET") or self.config.exchange_api_secret
        else:
            api_k = self.config.delta_api_key or os.getenv("DELTA_API_KEY") or self.config.exchange_api_key
            api_s = self.config.delta_api_secret or os.getenv("DELTA_API_SECRET") or self.config.exchange_api_secret

        self.config.exchange_api_key = api_k
        self.config.exchange_api_secret = api_s
        self.config.save()

        self.exchange_client = ExchangeApiClient(
            exchange=ex,
            api_key=api_k,
            api_secret=api_s,
            market=getattr(self.config, "binance_market", "auto"),
        )
        return (
            True,
            f"🔄 <b>Active Exchange Switched</b>\n"
            f"• Now using: <code>{ex.upper()}</code>\n"
            f"• Status: {self.exchange_client.get_masked_status()['status']}\n\n"
            f"<i>Test connection with /api test or configure with /api set {ex} &lt;KEY&gt; &lt;SECRET&gt;</i>",
        )

    def get_api_status_report(self) -> str:
        """Report live exchange API credentials and connectivity test."""
        st = self.exchange_client.get_masked_status()
        conn_ok, conn_msg, conn_data = self.exchange_client.test_connection()
        conn_str = f"🟢 Connected" if conn_ok else f"🔴 Not Connected ({conn_msg})"

        delta_status = "CONFIGURED 🟢" if (self.config.delta_api_key or (self.config.live_exchange == "delta" and self.config.exchange_api_key)) else "NOT SET 🔴"
        binance_status = "CONFIGURED 🟢" if (self.config.binance_api_key or (self.config.live_exchange == "binance" and self.config.exchange_api_key)) else "NOT SET 🔴"

        return (
            f"🔌 <b>LIVE EXCHANGE API SYSTEM</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Active Exchange</b>: <code>{st['exchange']}</code>\n"
            f"• <b>Active Credentials</b>: {st['status']}\n"
            f"• <b>API Key</b>: <code>{st['api_key']}</code>\n"
            f"• <b>API Secret</b>: <code>{st['api_secret']}</code>\n"
            f"• <b>Connectivity</b>: {conn_str}\n"
            f"• <b>Current Mode</b>: {'📄 PAPER TRADING' if self.config.trading_mode == 'paper' else '🚨 LIVE TRADING'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Saved Profiles:</b>\n"
            f"• <b>Delta Exchange</b>: {delta_status}\n"
            f"• <b>Binance ({self.config.binance_market.upper()})</b>: {binance_status}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Setup & Controls:</b>\n"
            f"• <code>/api set delta &lt;KEY&gt; &lt;SECRET&gt;</code> — Set Delta credentials\n"
            f"• <code>/api set binance &lt;KEY&gt; &lt;SECRET&gt; [spot|futures]</code> — Set Binance credentials\n"
            f"• <code>/api switch binance [spot|futures]</code> (or <code>delta</code>) — Switch active exchange\n"
            f"• <code>/api test</code> — Test exchange connection & query live balance\n"
            f"• <code>/api clear [binance|delta]</code> — Clear credentials & return to paper mode\n"
            f"• <code>/mode live</code> — Switch to live real-order trading"
        )

    # ------------------------------------------------------------------
    # Intelligent Trading Bot (ITB) Machine Learning Engine Methods
    # ------------------------------------------------------------------
    def get_itb_analysis(self, symbol: Optional[str] = None) -> str:
        """Run ITB Feature Engineering & ML Indicator prediction for symbol."""
        target_sym = self.normalize_symbol(symbol) if symbol else self.config.symbol
        df = self.fetch_candles(target_sym, count=120)
        strat = self._strategies_itb.get(target_sym) or self._strategy_itb or ITBStrategy(symbol=target_sym)
        pred = strat.predictor.predict(df, symbol=target_sym)
        return format_itb_card(pred)

    def run_itb_backtest(self, symbol: Optional[str] = None, count: int = 200, threshold: float = 0.12) -> str:
        """Run ITB simulated trading backtest over historic/demo candles."""
        target_sym = self.normalize_symbol(symbol) if symbol else self.config.symbol
        df = self.fetch_candles(target_sym, count=max(60, count))
        strat = self._strategies_itb.get(target_sym) or self._strategy_itb or ITBStrategy(symbol=target_sym)
        perf = ITBBacktester.backtest(df, predictor=strat.predictor, threshold=threshold)
        return format_backtest_report(perf, symbol=target_sym)

    def train_itb_model(self, symbol: Optional[str] = None, count: int = 250) -> str:
        """Fit ITB Ridge Regression ML Predictor weights on candle series."""
        target_sym = self.normalize_symbol(symbol) if symbol else self.config.symbol
        df = self.fetch_candles(target_sym, count=max(80, count))
        strat = self._strategies_itb.get(target_sym) or self._strategy_itb or ITBStrategy(symbol=target_sym)
        train_res = strat.predictor.train(df)
        if "error" in train_res:
            return f"❌ <b>ITB Training Failed:</b> {train_res['error']}"

        weights_fmt = "\n".join(
            f"• <code>{k:<12}</code>: <b>{v:+.4f}</b>"
            for k, v in list(train_res.get("weights", {}).items())[:6]
        )
        return (
            f"🧠 <b>INTELLIGENT TRADING BOT (ITB) MODEL TRAINED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Symbol</b>: <code>{target_sym}</code>\n"
            f"• <b>Samples Fitted</b>: <code>{train_res.get('samples', 0)}</code> candles\n"
            f"• <b>R² Score</b>: <code>{train_res.get('r2_score', 0.0):.4f}</code>\n"
            f"• <b>MAE</b>: <code>{train_res.get('mae', 0.0):.6f}</code>\n"
            f"• <b>Model Bias</b>: <code>{train_res.get('bias', 0.0):+.4f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Top Feature Weights:</b>\n"
            f"{weights_fmt}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 The trained model is now actively powering your ITB signals!</i>\n"
            f"• Query live status: <code>/itb {target_sym}</code>\n"
            f"• Run backtest: <code>/itb backtest {target_sym}</code>"
        )

    # ------------------------------------------------------------------
    # Combined Multi-Model Ensemble System & Multi-Layer Deliberation
    # ------------------------------------------------------------------
    def evaluate_ensemble(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified 5-layer ensemble confluence evaluator:
        Combines SMC Structure + Technical Momentum + ITB Machine Learning +
        AI Bot Learning + Risk Guardian Arbiter.
        Returns detailed deliberation metrics, cross-critiques, and conviction sizing.
        """
        target_sym = self.normalize_symbol(symbol) if symbol else self.config.symbol
        df1 = self.fetch_candles(target_sym, count=120)
        if df1 is None or df1.empty:
            df1 = self._generate_dummy_candles(target_sym, count=120)

        curr_price = float(df1["close"].iloc[-1])
        delib = self._deliberation_engine.deliberate(target_sym, df1, self)

        p_itb = delib.layers.get("itb_machine_learning")
        p_pro = delib.layers.get("technical_momentum")
        p_ai = delib.layers.get("ai_bot_learning")
        p_smc = delib.layers.get("smc_structure")

        itb_score = p_itb.score if p_itb else 0.0
        itb_dir = p_itb.stance if p_itb else "FLAT"
        itb_zone = p_itb.key_metrics.get("zone", "NEUTRAL") if p_itb else "NEUTRAL"
        itb_conf = p_itb.confidence if p_itb else 0.5

        pro_score = p_pro.key_metrics.get("score", 0.0) if p_pro else 0.0
        pro_norm = p_pro.score if p_pro else 0.0
        pro_dir = p_pro.stance if p_pro else "FLAT"
        pro_rsi = p_pro.key_metrics.get("rsi", 50.0) if p_pro else 50.0
        pro_adx = p_pro.key_metrics.get("adx", 20.0) if p_pro else 20.0

        ai_score = p_ai.score if p_ai else 0.0
        ai_dir = p_ai.stance if p_ai else "FLAT"
        ai_regime = p_ai.key_metrics.get("regime", "NORMAL") if p_ai else "NORMAL"

        return {
            "symbol": target_sym,
            "price": curr_price,
            "itb_score": itb_score,
            "itb_zone": itb_zone,
            "itb_dir": itb_dir,
            "itb_confidence": itb_conf,
            "pro_score": pro_score,
            "pro_norm": pro_norm,
            "pro_dir": pro_dir,
            "pro_rsi": pro_rsi,
            "pro_adx": pro_adx,
            "ai_score": ai_score,
            "ai_dir": ai_dir,
            "ai_regime": ai_regime,
            "composite_score": delib.consensus_score,
            "direction": delib.consensus_direction,
            "verdict": delib.verdict,
            "agreement_pct": round(delib.agreement_rate * 100.0),
            "engines_aligned": f"{len(delib.aligned_layers)}/4",
            "deliberation": delib,
            "conviction_multiplier": delib.conviction_multiplier,
            "risk_approval": delib.risk_approval,
            "smc_stance": p_smc.stance if p_smc else "NEUTRAL",
            "smc_arg": p_smc.argument if p_smc else "",
        }

    def generate_deliberation_report(self, symbol: Optional[str] = None) -> str:
        """
        Formats the interactive 5-layer deliberation forum dialogue:
        Presents SMC, Momentum, ITB ML, AI Learning, and Risk Guardian
        debating and cross-examining each other's theses.
        """
        target_sym = self.normalize_symbol(symbol) if symbol else self.config.symbol
        df1 = self.fetch_candles(target_sym, count=120)
        if df1 is None or df1.empty:
            df1 = self._generate_dummy_candles(target_sym, count=120)

        delib = self._deliberation_engine.deliberate(target_sym, df1, self)
        curr_price = float(df1["close"].iloc[-1])

        comp_sign = "+" if delib.consensus_score >= 0 else ""
        meter_pct = max(0.0, min(100.0, (delib.consensus_score + 1.0) / 2.0 * 100.0))
        meter_bar = make_modern_meter(meter_pct, width=10, fill_char="■", empty_char="░")

        status_icon = "✅ APPROVED" if delib.risk_approval else "⚠️ DEFENSIVE / VETOED"
        dialogue_str = "\n\n".join(delib.discussion_dialogue)

        w_itb = getattr(self.config, "weight_itb", 0.35) * 100.0
        w_pro = getattr(self.config, "weight_pro", 0.35) * 100.0
        w_ai = getattr(self.config, "weight_ai", 0.30) * 100.0
        cycles = getattr(self.config, "learning_cycles", 0)
        retrain_t = getattr(self.config, "last_retrain_time", "") or "Continuous background active"

        return (
            f"🗣️ <b>INTER-ENGINE DELIBERATION FORUM: {target_sym}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Market Price</b>: <code>${curr_price:,.2f}</code>\n"
            f"• <b>Consensus Verdict</b>: <b>{delib.verdict}</b>\n"
            f"• <b>Consensus Gauge</b>: <code>[{meter_bar}]</code> ({comp_sign}{delib.consensus_score:.2f})\n"
            f"• <b>Agreement Rate</b>: <code>{delib.agreement_rate * 100:.0f}%</code> ({len(delib.aligned_layers)}/4 layers aligned)\n"
            f"• <b>Conviction Sizing</b>: <code>{delib.conviction_multiplier:.2f}x multiplier</code>\n"
            f"• <b>Risk Clearance</b>: <b>{status_icon}</b>\n"
            f"• <b>Quality Gate & MTF</b>: <code>Q={delib.quality_score:.1f}/5.0</code> ({delib.mtf_summary} | RVOL: {delib.rel_vol:.2f}x)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>💬 INTER-LAYER DEBATE & CROSS-CRITIQUE:</b>\n\n"
            f"{dialogue_str}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 <b>AUTONOMOUS SELF-LEARNING STATUS:</b>\n"
            f"• <b>Dynamic Allocation</b>: ITB: <code>{w_itb:.1f}%</code> | Pro: <code>{w_pro:.1f}%</code> | AI: <code>{w_ai:.1f}%</code>\n"
            f"• <b>Learning Cycles</b>: <code>{cycles}</code> | <b>Last Retrain</b>: <code>{retrain_t}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 All layers cross-examine each other continuously. Use <code>/scan {target_sym}</code> for master scan.</i>"
        )

    def generate_ensemble_report(self, symbol: Optional[str] = None) -> str:
        """Formats the Combined Ensemble confluence card powered by inter-layer deliberation."""
        e = self.evaluate_ensemble(symbol)
        comp = e["composite_score"]
        comp_sign = "+" if comp >= 0 else ""
        meter_pct = max(0.0, min(100.0, (comp + 1.0) / 2.0 * 100.0))
        meter_bar = make_modern_meter(meter_pct, width=10, fill_char="■", empty_char="░")

        itb_sign = "+" if e["itb_score"] >= 0 else ""
        pro_sign = "+" if e["pro_score"] >= 0 else ""
        conv_mult = e.get("conviction_multiplier", 1.0)
        risk_str = "✅ Cleared" if e.get("risk_approval", True) else "⚠️ Vetoed / Neutral"

        delib = e.get("deliberation")
        smc_s = getattr(delib.layers.get("smc_structure"), "stance", "NEUTRAL") if delib else "NEUTRAL"

        w_itb = getattr(self.config, "weight_itb", 0.35) * 100.0
        w_pro = getattr(self.config, "weight_pro", 0.35) * 100.0
        w_ai = getattr(self.config, "weight_ai", 0.30) * 100.0

        q_val = getattr(delib, "quality_score", 3.5) if delib else 3.5
        mtf_summary = getattr(delib, "mtf_summary", "MTF Active") if delib else "MTF Active"
        rel_v = getattr(delib, "rel_vol", 1.0) if delib else 1.0

        return (
            f"🌟 <b>COMBINED ENSEMBLE SYSTEM: {e['symbol']}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Market Price</b>: <code>${e['price']:,.2f}</code>\n"
            f"• <b>Consensus Verdict</b>: <b>{e['verdict']}</b> ({e['agreement_pct']}% agreement)\n"
            f"• <b>Composite Gauge</b>: <code>[{meter_bar}]</code> ({comp_sign}{comp:.2f})\n"
            f"• <b>Quality Gate & MTF</b>: <code>Q={q_val:.1f}/5.0</code> ({mtf_summary} | RVOL: {rel_v:.2f}x)\n"
            f"• <b>Conviction Sizing</b>: <code>{conv_mult:.2f}x</code> | <b>Risk Clearance</b>: <code>{risk_str}</code>\n"
            f"• <b>Dynamic Allocation</b>: ITB: <code>{w_itb:.0f}%</code> | Pro: <code>{w_pro:.0f}%</code> | AI: <code>{w_ai:.0f}%</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Individual Layer Stances (Mutual Deliberation):</b>\n"
            f"1. 🧱 <b>SMC Structure</b>: {smc_s}\n"
            f"2. 📊 <b>Indicators Pro</b>: {pro_sign}{e['pro_score']:.2f}/5.0 (RSI: {e['pro_rsi']:.1f} | ADX: {e['pro_adx']:.1f})\n"
            f"3. 🤖 <b>ITB Machine Learning</b>: {itb_sign}{e['itb_score']:.2f} ({e['itb_zone']})\n"
            f"4. 🧠 <b>AI Bot Learning</b>: {e['ai_dir']} (Regime: {e['ai_regime']})\n"
            f"5. 🛡️ <b>Risk Guardian</b>: {risk_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 All engines discuss each other's theses. Full dialogue: <code>/discussion {e['symbol']}</code></i>"
        )

    def learn_from_trade(
        self,
        pos: AutoTradePosition,
        current_price: float,
        pnl: float,
        reason: str = "MANUAL",
    ) -> None:
        """
        Feedback-driven continuous self-learning loop:
        1. Records closed trade into AI Bot Learning permanent memory with true indicator and execution telemetry.
        2. Dynamically adapts ITB, Pro, and AI ensemble weights via multi-armed bandit credit assignment based on each layer's forecast correctness.
        3. Updates per-engine accuracy metrics (wins, total, win rate).
        4. Triggers immediate online incremental retraining for ITB ML model on the concluded trade's symbol.
        5. Dynamically calibrates RiskManager and regime-adaptive targets.
        """
        won = bool(pnl > 0)
        risk_budget = (self.config.equity * self.config.risk_pct / 100.0) if self.config.risk_pct > 0 else 1.0
        r_multiple = round(pnl / max(0.01, risk_budget), 2)
        setup_name = getattr(pos, "strategy", "Ensemble") or "Ensemble"

        # Calculate actual bars held from entry to exit
        bars_held = 5
        if pos.entry_time and pos.exit_time:
            try:
                t0 = pd.to_datetime(pos.entry_time)
                t1 = pd.to_datetime(pos.exit_time)
                bars_held = max(1, int((t1 - t0).total_seconds() / 60))
            except Exception:
                bars_held = 5

        # 1. Update AI Bot Permanent Memory with authentic trade telemetry
        try:
            strat_ai = self._strategies_ai.get(pos.symbol) or self._strategy_ai
            if strat_ai and hasattr(strat_ai, "mem"):
                df_cur = self.fetch_candles(pos.symbol, count=50)
                regime_name = getattr(pos, "regime", "NORMAL")
                if df_cur is not None and not df_cur.empty:
                    try:
                        reg_obj = strat_ai.regime(df_cur)
                        regime_name = getattr(reg_obj, "name", str(reg_obj)).lower()
                    except Exception:
                        pass

                rec = TradeRecord(
                    id=pos.id,
                    timestamp=pos.exit_time or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    direction=pos.direction,
                    setup=setup_name,
                    entry=pos.entry_price,
                    exit=current_price,
                    pnl=pnl,
                    r_multiple=r_multiple,
                    quality=getattr(pos, "quality_score", (4.8 if won else 3.2)),
                    regime=regime_name,
                    hour=datetime.now(timezone.utc).hour,
                    session=getattr(pos, "session", "active"),
                    rsi=getattr(pos, "entry_rsi", 50.0),
                    adx=getattr(pos, "entry_adx", 25.0),
                    ind_score=getattr(pos, "entry_score", (1.0 if won else -1.0)),
                    ai_conf=getattr(pos, "ai_conf", (0.70 if won else 0.45)),
                    ai_correct=bool((pos.direction == "LONG" and won) or (pos.direction == "SHORT" and won)),
                    rel_vol=getattr(pos, "rel_vol", 1.0),
                    bars_held=bars_held,
                    won=won,
                )
                strat_ai.mem.remember(rec)
                logger.info("AI Bot Memory updated with trade %s (PnL: $%.2f, Won: %s, Regime: %s)", pos.id, pnl, won, regime_name)
        except Exception as e:
            logger.warning("Could not record trade into AI memory: %s", e)

        # 2. Layer-Specific Credit Assignment (Per-Engine Predictive Attribution)
        try:
            w_itb = self.config.weight_itb
            w_pro = self.config.weight_pro
            w_ai = self.config.weight_ai

            votes = getattr(pos, "engine_votes", None)
            if votes and isinstance(votes, dict):
                # Distinct attribution based on each engine's stance
                for eng_key in ("itb", "pro", "ai", "smc"):
                    v = votes.get(eng_key, "NEUTRAL")
                    is_aligned = (v == pos.direction)
                    is_opposed = (v in ("LONG", "SHORT") and v != pos.direction)

                    # Update engine statistics
                    if eng_key in self.config.engine_stats:
                        self.config.engine_stats[eng_key]["total"] += 1
                        if (won and is_aligned) or (not won and is_opposed):
                            self.config.engine_stats[eng_key]["wins"] += 1

                    # Compute engine shift
                    if eng_key in ("itb", "pro", "ai"):
                        if won:
                            if is_aligned:
                                shift = 0.035
                            elif is_opposed:
                                shift = -0.030
                            else:
                                shift = -0.005
                        else:
                            if is_aligned:
                                shift = -0.040
                            elif is_opposed:
                                shift = 0.035
                            else:
                                shift = 0.015

                        if eng_key == "itb":
                            w_itb += shift
                        elif eng_key == "pro":
                            w_pro += shift
                        elif eng_key == "ai":
                            w_ai += shift
            else:
                # Fallback for positions without granular votes
                shift = 0.02 if won else -0.015
                pos_desc = f"{pos.reason} {pos.strategy}"
                if "ITB" in pos_desc or "Ensemble" in pos_desc or "Deliberat" in pos_desc:
                    w_itb += shift
                if "Pro" in pos_desc or "Ensemble" in pos_desc or "Deliberat" in pos_desc:
                    w_pro += shift
                if "AI" in pos_desc or "Ensemble" in pos_desc or "Deliberat" in pos_desc:
                    w_ai += shift

            w_itb = max(0.15, min(0.60, w_itb))
            w_pro = max(0.15, min(0.60, w_pro))
            w_ai = max(0.15, min(0.60, w_ai))

            total_w = w_itb + w_pro + w_ai
            self.config.weight_itb = round(w_itb / total_w, 3)
            self.config.weight_pro = round(w_pro / total_w, 3)
            self.config.weight_ai = round(1.0 - self.config.weight_itb - self.config.weight_pro, 3)

            self.config.learning_cycles += 1
            self.config.save()
            logger.info(
                "Self-learning rebalanced ensemble weights -> ITB: %.1f%%, Pro: %.1f%%, AI: %.1f%% (Cycle #%d)",
                self.config.weight_itb * 100,
                self.config.weight_pro * 100,
                self.config.weight_ai * 100,
                self.config.learning_cycles,
            )
        except Exception as e:
            logger.warning("Could not adapt ensemble weights: %s", e)

        # 3. Strategy PnL tracking
        if pos.symbol in self._strategies_pro:
            self._strategies_pro[pos.symbol].update(pnl)
        elif self._strategy_pro:
            self._strategy_pro.update(pnl)

        if pos.symbol in self._strategies_itb:
            self._strategies_itb[pos.symbol].update(pnl)
        elif self._strategy_itb:
            self._strategy_itb.update(pnl)

        # 4. Immediate On-Trade Incremental Online Retraining of ITB ML Model
        try:
            strat_itb = self._strategies_itb.get(pos.symbol) or self._strategy_itb
            if strat_itb and hasattr(strat_itb, "predictor"):
                df_on_trade = self.fetch_candles(pos.symbol, count=150)
                if df_on_trade is not None and len(df_on_trade) >= 35:
                    train_res = strat_itb.predictor.train(df_on_trade, horizon=10, l2_reg=1.0)
                    if train_res.get("success"):
                        logger.info(
                            "[On-Trade Learn] %s ITB ML model retrained: R²=%.3f, DirAcc=%.1f%% (%d samples)",
                            pos.symbol, train_res.get("r2_score", 0.0),
                            train_res.get("directional_accuracy", 0.0), train_res.get("samples", 0),
                        )
        except Exception as itb_err:
            logger.debug("On-trade ITB retrain skipped: %s", itb_err)

        # 5. Update Practical Risk Manager state
        if getattr(self, "risk_manager", None):
            try:
                initial_sl = pos.initial_stop_loss if pos.initial_stop_loss is not None else pos.stop_loss
                initial_risk = abs(pos.entry_price - initial_sl)
                risk_usd = initial_risk * pos.lot_size if (initial_risk > 0 and pos.lot_size > 0) else 1.0
                r_mult = round(pnl / max(0.01, risk_usd), 2)
                self.risk_manager.update_after_trade(pnl=pnl, r_multiple=r_mult)
            except Exception as re_err:
                logger.warning("Could not update RiskManager: %s", re_err)

    def get_learning_report(self) -> str:
        """
        Generates an extensive diagnostic card detailing:
        - Self-Learned Dynamic Allocation (ITB %, Pro %, AI %)
        - Per-Engine Predictive Accuracy & Win Rates
        - Calibrated Indicator & AI Model Weights
        - Quality Gate & Multi-Timeframe Status
        - Continuous Retraining Telemetry
        """
        w_itb = getattr(self.config, "weight_itb", 0.35) * 100.0
        w_pro = getattr(self.config, "weight_pro", 0.35) * 100.0
        w_ai = getattr(self.config, "weight_ai", 0.30) * 100.0
        cycles = getattr(self.config, "learning_cycles", 0)
        last_retrain = getattr(self.config, "last_retrain_time", "") or "Continuous active"

        # Per-engine accuracy stats
        stats = getattr(self.config, "engine_stats", {})
        def fmt_eng(k, name):
            d = stats.get(k, {"wins": 0, "total": 0})
            w = d.get("wins", 0)
            t = d.get("total", 0)
            pct = (w / t * 100.0) if t > 0 else 0.0
            return f"• <b>{name}</b>: <code>{pct:.1f}%</code> accuracy ({w}/{t} trades)"

        eng_lines = [
            fmt_eng("itb", "🤖 ITB Machine Learning"),
            fmt_eng("pro", "📊 Indicators Pro"),
            fmt_eng("ai", "🧠 AI Bot Memory"),
            fmt_eng("smc", "🧱 SMC Structure"),
        ]

        # AI Bot Calibrated stats
        strat_ai = self._strategy_ai
        mem = getattr(strat_ai, "mem", None) if strat_ai else None
        m_stats = getattr(mem, "stats", None) if mem else None
        best_q = getattr(m_stats, "best_quality", 4.2) if m_stats else 4.2
        best_trail = getattr(m_stats, "best_trail_mult", 2.15) if m_stats else 2.15
        best_ai = getattr(m_stats, "best_ai_conf", 0.64) if m_stats else 0.64
        exp_r = getattr(m_stats, "expectancy", 0.0) if m_stats else 0.0
        rec_wr = getattr(m_stats, "recent_winrate", 0.50) if m_stats else 0.50

        ind_w = getattr(m_stats, "indicator_weights", {}) if m_stats else {}
        rsi_w = ind_w.get("rsi", 1.0)
        adx_w = ind_w.get("adx", 1.0)
        ema_w = ind_w.get("ema", 1.0)

        # Risk Manager status
        rm = getattr(self, "risk_manager", None)
        rf_val = getattr(rm.state, "rf", 1.0) if (rm and hasattr(rm, "state")) else 1.0
        streak = getattr(rm.state, "win_streak", 0) if (rm and hasattr(rm, "state")) else 0

        return (
            f"🧠 <b>AUTONOMOUS MULTI-LAYER LEARNING DASHBOARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Learning Cycles</b>: <code>#{cycles}</code>\n"
            f"• <b>Online Retraining</b>: <code>{last_retrain}</code>\n"
            f"• <b>Risk Factor Multiplier</b>: <code>{rf_val:.2f}x</code> (Win Streak: {streak})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚖️ <b>Dynamic Self-Learned Allocation:</b>\n"
            f"• 🤖 <b>ITB Machine Learning</b>: <b>{w_itb:.1f}%</b>\n"
            f"• 📊 <b>Indicators Pro</b>: <b>{w_pro:.1f}%</b>\n"
            f"• 🧠 <b>AI Bot Memory</b>: <b>{w_ai:.1f}%</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Per-Engine Predictive Accuracy:</b>\n"
            f"{chr(10).join(eng_lines)}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔬 <b>Calibrated Neural & Statistical Parameters:</b>\n"
            f"• <b>Quality Gate Threshold</b>: <code>Q ≥ {best_q:.2f}</code>\n"
            f"• <b>Optimal Trailing Stop</b>: <code>{best_trail:.2f}x ATR</code>\n"
            f"• <b>AI Confidence Cutoff</b>: <code>{best_ai:.2f}</code>\n"
            f"• <b>Recent Expectancy</b>: <code>{exp_r:+.2f}R</code> (WR: {rec_wr*100:.0f}%)\n"
            f"• <b>Indicator Weights</b>: RSI=<code>{rsi_w:.2f}</code> | ADX=<code>{adx_w:.2f}</code> | EMA=<code>{ema_w:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 The system dynamically adjusts weights & models on every trade. Use <code>/discussion</code> to view active inter-engine debate.</i>"
        )

    def auto_learn_step(self) -> List[str]:
        """
        Background autonomous retraining and self-learning loop:
        1. Checks elapsed time since last retraining cycle.
        2. Retrains ITB Ridge Regression models on rolling live candle feeds for all configured symbols.
        3. Persists updated weights, logs progress, and returns notification notes.
        """
        now = time_module.time()
        interval = getattr(self.config, "auto_retrain_interval_seconds", 300)
        if (now - self._last_retrain_timestamp) < interval:
            return []

        self._last_retrain_timestamp = now
        notes = []
        target_symbols = list(self.config.symbols) if self.config.symbols else [self.config.symbol]

        for sym in target_symbols:
            try:
                df = self.fetch_candles(sym, count=150)
                if df is None or len(df) < 35:
                    continue
                strat_itb = self._strategies_itb.get(sym) or self._strategy_itb
                if strat_itb and hasattr(strat_itb, "predictor"):
                    res = strat_itb.predictor.train(df, horizon=10, l2_reg=1.0)
                    if res.get("success"):
                        r2 = res.get("r2_score", 0.0)
                        acc_val = res.get("directional_accuracy", res.get("direction_accuracy", 0.0))
                        acc_pct = acc_val if acc_val > 1.0 else (acc_val * 100.0)
                        logger.info(
                            "[Auto-Learn] Background ITB retrain for %s completed: R²=%.3f, DirAcc=%.1f%% (%d samples)",
                            sym, r2, acc_pct, res.get("samples", 0),
                        )
                        notes.append(f"🧠 Retrained {sym} ML model (R²: {r2:+.2f}, Acc: {acc_pct:.0f}%)")
            except Exception as e:
                logger.debug("Background retraining error for %s: %s", sym, e)

        self.config.last_retrain_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.config.learning_cycles += 1
        self.config.save()
        return notes

    def generate_trade_analysis(self, symbol: str) -> str:
        """Alias to analyze_market for unified master scan."""
        return self.analyze_market(symbol)

    def generate_order_blocks(self, symbol: str) -> str:
        """Generate Smart Money order blocks and fair value gaps report."""
        return get_levels_report(symbol, trader=self)

    def generate_pinpoint_entry(self, symbol: str) -> str:
        """Generate precision pinpoint entry and targets report."""
        plan = generate_pinpoint_plan(symbol, trader=self)
        return format_pinpoint_report(plan)

    # ------------------------------------------------------------------
    # Trade Level Alerts System
    # ------------------------------------------------------------------
    def add_alert(
        self,
        symbol: str,
        target_price: float,
        condition: str = "AUTO",
        note: str = "",
        alert_type: str = "CUSTOM",
        chat_id: Optional[int] = None,
        one_shot: bool = True,
    ) -> Tuple[bool, str, Optional[TradeLevelAlert]]:
        """Register a new trade price level trigger alert."""
        if target_price <= 0:
            return False, "Error: Target price must be greater than 0.", None

        target_sym = self.normalize_symbol(symbol)
        df = self.fetch_candles(target_sym, count=1)
        curr_price = float(df["close"].iloc[-1]) if not df.empty else target_price

        cond = condition.strip().upper()
        if cond in ("AUTO", ""):
            cond = "CROSS_ABOVE" if target_price >= curr_price else "CROSS_BELOW"
        elif cond not in ("CROSS_ABOVE", "CROSS_BELOW", "TOUCH"):
            cond = "CROSS_ABOVE" if target_price >= curr_price else "CROSS_BELOW"

        alert_id = f"ALT_{target_sym[:3]}_{int(time.time() * 1000) % 1000000}"
        alert = TradeLevelAlert(
            id=alert_id,
            symbol=target_sym,
            target_price=round(float(target_price), 2),
            condition=cond,
            alert_type=alert_type,
            note=note or f"Target ${target_price:,.2f}",
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            chat_id=chat_id,
            one_shot=one_shot,
        )
        self.alerts.append(alert)
        self._save_alerts()

        cond_desc = (
            "Crosses Above 🟢"
            if cond == "CROSS_ABOVE"
            else ("Crosses Below 🔴" if cond == "CROSS_BELOW" else "Touches 🎯")
        )
        return (
            True,
            f"🔔 <b>Trade Level Alert Created</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>ID</b>: <code>{alert.id}</code>\n"
            f"• <b>Symbol</b>: <code>{alert.symbol}</code>\n"
            f"• <b>Target Price</b>: <code>${alert.target_price:,.2f}</code>\n"
            f"• <b>Trigger Condition</b>: {cond_desc}\n"
            f"• <b>Current Market</b>: <code>${curr_price:,.2f}</code>\n"
            f"• <b>Note</b>: <i>{alert.note}</i>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>You will receive an instant priority alert when price hits this level.</i>",
            alert,
        )

    def remove_alert(self, alert_id: str) -> Tuple[bool, str]:
        """Remove an alert by ID."""
        target = alert_id.strip().upper()
        for a in list(self.alerts):
            if a.id.upper() == target:
                self.alerts.remove(a)
                self._save_alerts()
                return True, f"Alert <code>{a.id}</code> ({a.symbol} @ ${a.target_price:,.2f}) removed."
        return False, f"Alert <code>{target}</code> not found."

    def clear_alerts(self, symbol: Optional[str] = None) -> int:
        """Clear active alerts, optionally filtered by symbol."""
        if symbol:
            sym_norm = self.normalize_symbol(symbol)
            before = len(self.alerts)
            self.alerts = [a for a in self.alerts if a.symbol != sym_norm]
            cleared = before - len(self.alerts)
        else:
            cleared = len(self.alerts)
            self.alerts = []
        self._save_alerts()
        return cleared

    def get_alerts(self, symbol: Optional[str] = None) -> List[TradeLevelAlert]:
        """Return active untriggered alerts, optionally filtered by symbol."""
        if symbol:
            sym_norm = self.normalize_symbol(symbol)
            return [a for a in self.alerts if a.symbol == sym_norm and not a.triggered]
        return [a for a in self.alerts if not a.triggered]

    def get_alerts_report(self) -> str:
        """Modern dashboard of active trade level triggers."""
        active = [a for a in self.alerts if not a.triggered]
        if not active:
            return (
                "🔔 <b>TRADE LEVEL ALERTS DASHBOARD</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "• <b>Status</b>: No active price triggers set.\n\n"
                "<b>How to set price alerts:</b>\n"
                "• <code>/alert BTC 85000</code> — Alert when BTC hits $85,000\n"
                "• <code>/alert GOLD 4200</code> — Alert when Gold hits $4,200\n"
                "• <code>/alert 82000 below</code> — Alert if current pair drops to $82k\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>💡 Note: The bot also monitors Order Blocks, FVGs, and Position TP/SL targets automatically.</i>"
            )

        lines = [
            f"🔔 <b>ACTIVE TRADE LEVEL ALERTS ({len(active)})</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for a in active:
            df = self.fetch_candles(a.symbol, count=1)
            curr = float(df["close"].iloc[-1]) if not df.empty else a.target_price
            dist = a.target_price - curr
            dist_pct = (dist / curr) * 100.0 if curr > 0 else 0.0
            dir_icon = "🟢" if a.condition == "CROSS_ABOVE" else "🔴"
            lines.append(
                f"• <b>[{a.id}]</b> <code>{a.symbol}</code> @ <b>${a.target_price:,.2f}</b> {dir_icon}\n"
                f"  Trigger: {a.condition} | Current: <code>${curr:,.2f}</code> ({dist_pct:+.2f}% away)\n"
                f"  Note: <i>{a.note}</i>"
            )
        lines.extend(
            [
                "━━━━━━━━━━━━━━━━━━━━━━",
                "<b>Manage Alerts:</b>\n"
                "• <code>/alert del &lt;ID&gt;</code> — Remove specific alert\n"
                "• <code>/alert clear</code> — Wipe all active alerts",
            ]
        )
        return "\n".join(lines)

    
    def generate_mtf_trend_report(self, symbol: str) -> str:
        """Generate a Multi-Timeframe Trend Confluence report."""
        timeframes = [("1m", "1"), ("5m", "5"), ("15m", "15"), ("1h", "60"), ("4h", "240"), ("1D", "1D")]
        
        lines = [f"📊 <b>MTF Trend Scanner: {symbol.upper()}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
        bullish_count = 0
        bearish_count = 0
        total_tf = len(timeframes)
        
        for label, res in timeframes:
            try:
                df = self.fetch_candles(symbol, count=100, resolution=res)
                if df is None or df.empty or len(df) < 50:
                    lines.append(f"• <b>{label}</b>: ⚠️ Insufficient Data")
                    continue
                    
                closes = df['close']
                ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
                rsi = IndicatorEngine.rsi(closes, period=14).iloc[-1]
                current_price = closes.iloc[-1]
                
                if ema20 > ema50 and current_price > ema20 and rsi > 50:
                    trend = "🟢 BULLISH"
                    bullish_count += 1
                elif ema20 < ema50 and current_price < ema20 and rsi < 50:
                    trend = "🔴 BEARISH"
                    bearish_count += 1
                else:
                    trend = "⚪ NEUTRAL"
                    
                lines.append(f"• <b>{label}</b>: {trend} (RSI: {rsi:.1f})")
            except Exception as e:
                logger.error("MTF scan error for %s on %s: %s", symbol, res, e)
                lines.append(f"• <b>{label}</b>: ⚠️ Error")
                
        score = (bullish_count / total_tf) * 100
        b_score = (bearish_count / total_tf) * 100
        
        # Determine overall trend
        if score >= 80:
            overall = "🚀 STRONG UPTREND"
            meter_val = score
        elif score >= 60:
            overall = "📈 UPTREND"
            meter_val = score
        elif b_score >= 80:
            overall = "📉 STRONG DOWNTREND"
            meter_val = 100 - b_score
        elif b_score >= 60:
            overall = "🩸 DOWNTREND"
            meter_val = 100 - b_score
        else:
            overall = "⚖️ CHOPPY / CONSOLIDATING"
            meter_val = 50
            
        meter = make_modern_meter(meter_val, width=12)
        
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"<b>Overall Alignment</b>: {overall}")
        lines.append(f"<b>Bullish vs Bearish</b>: {bullish_count} vs {bearish_count}")
        lines.append(f"<code>[{meter}]</code>")
        return "\n".join(lines)

    def check_trade_level_alerts(
        self, candles_cache: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[str]:
        """
        Evaluates real-time price action against:
        1. Custom user price targets (CROSS_ABOVE, CROSS_BELOW, TOUCH).
        2. Active open positions for TP/SL proximity and breakeven milestones.
        3. Institutional Order Block (OB) demand & supply zone entries.
        4. Fair Value Gap (FVG) imbalance fill levels.
        Returns list of modern Telegram HTML notification strings.
        """
        notifications: List[str] = []
        now_ts = time.time()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cache = candles_cache if candles_cache is not None else {}

        def get_sym_candles(sym: str) -> pd.DataFrame:
            if sym not in cache:
                cache[sym] = self.fetch_candles(sym, count=60)
            return cache[sym]

        # 1. Custom User Price Level Alerts
        alerts_changed = False
        for alert in list(self.alerts):
            if alert.triggered:
                continue
            df = get_sym_candles(alert.symbol)
            if df.empty:
                continue
            candle = df.iloc[-1]
            c_high = float(candle["high"])
            c_low = float(candle["low"])
            c_close = float(candle["close"])

            triggered = False
            trig_note = ""
            if alert.condition == "CROSS_ABOVE" and c_high >= alert.target_price:
                triggered = True
                trig_note = f"Price crossed above target (High: ${c_high:,.2f}) 🟢"
            elif alert.condition == "CROSS_BELOW" and c_low <= alert.target_price:
                triggered = True
                trig_note = f"Price dropped below target (Low: ${c_low:,.2f}) 🔴"
            elif alert.condition in ("TOUCH", "AUTO"):
                if c_low <= alert.target_price <= c_high:
                    triggered = True
                    trig_note = f"Price touched target level (${c_close:,.2f}) 🎯"

            if triggered:
                alert.triggered = True
                alert.triggered_at = now_str
                alerts_changed = True
                msg = (
                    f"🔔 <b>TRADE LEVEL ALERT TRIGGERED!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"• <b>Asset</b>: <code>{alert.symbol}</code>\n"
                    f"• <b>Target Level</b>: <code>${alert.target_price:,.2f}</code>\n"
                    f"• <b>Current Market</b>: <code>${c_close:,.2f}</code>\n"
                    f"• <b>Event</b>: {trig_note}\n"
                    f"• <b>Note</b>: <i>{alert.note}</i>\n"
                    f"• <b>Time</b>: <code>{now_str}</code>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"<i>💡 Quick check: /entry {alert.symbol} | Instant order: /buy or /sell</i>"
                )
                notifications.append(msg)

        if alerts_changed:
            self._save_alerts()

        # 2. Open Position Proximity & Milestone Alerts
        for pos in self.positions:
            df = get_sym_candles(pos.symbol)
            if df.empty:
                continue
            curr = float(df.iloc[-1]["close"])
            risk_dist = abs(pos.entry_price - pos.stop_loss)
            tp_dist = abs(pos.take_profit_1 - pos.entry_price)
            if risk_dist <= 0 or tp_dist <= 0:
                continue

            # a) Take Profit Proximity Warning (Within 15% distance to TP1)
            tp_prox_key = f"tp_prox_{pos.id}"
            if pos.direction == "LONG" and curr >= (pos.take_profit_1 - (tp_dist * 0.15)) and curr < pos.take_profit_1:
                if now_ts - self._recent_level_alerts.get(tp_prox_key, 0) > 1800:
                    self._recent_level_alerts[tp_prox_key] = now_ts
                    pnl = pos.current_pnl(curr)
                    pnl_sign = "+" if pnl >= 0 else ""
                    notifications.append(
                        f"🎯 <b>TARGET REACH WARNING: TP1 NEARBY ({pos.symbol})</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Position ID</b>: <code>{pos.id}</code> ({pos.direction})\n"
                        f"• <b>Entry Price</b>: <code>${pos.entry_price:,.2f}</code>\n"
                        f"• <b>Current Market</b>: <code>${curr:,.2f}</code>\n"
                        f"• <b>Take Profit 1</b>: <code>${pos.take_profit_1:,.2f}</code> (Only ${abs(pos.take_profit_1 - curr):,.2f} away!)\n"
                        f"• <b>Current Unrealized Gain</b>: 🟢 <b>{pnl_sign}${pnl:,.2f}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"<i>💡 Consider locking in profit with /trailing on or exit with /close {pos.id}</i>"
                    )
            elif pos.direction == "SHORT" and curr <= (pos.take_profit_1 + (tp_dist * 0.15)) and curr > pos.take_profit_1:
                if now_ts - self._recent_level_alerts.get(tp_prox_key, 0) > 1800:
                    self._recent_level_alerts[tp_prox_key] = now_ts
                    pnl = pos.current_pnl(curr)
                    pnl_sign = "+" if pnl >= 0 else ""
                    notifications.append(
                        f"🎯 <b>TARGET REACH WARNING: TP1 NEARBY ({pos.symbol})</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Position ID</b>: <code>{pos.id}</code> ({pos.direction})\n"
                        f"• <b>Entry Price</b>: <code>${pos.entry_price:,.2f}</code>\n"
                        f"• <b>Current Market</b>: <code>${curr:,.2f}</code>\n"
                        f"• <b>Take Profit 1</b>: <code>${pos.take_profit_1:,.2f}</code> (Only ${abs(curr - pos.take_profit_1):,.2f} away!)\n"
                        f"• <b>Current Unrealized Gain</b>: 🟢 <b>{pnl_sign}${pnl:,.2f}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"<i>💡 Consider locking in profit with /trailing on or exit with /close {pos.id}</i>"
                    )

            # b) Stop Loss Danger Warning (Within 20% distance to SL)
            sl_prox_key = f"sl_danger_{pos.id}"
            if pos.direction == "LONG" and curr <= (pos.stop_loss + (risk_dist * 0.20)) and curr > pos.stop_loss:
                if now_ts - self._recent_level_alerts.get(sl_prox_key, 0) > 1800:
                    self._recent_level_alerts[sl_prox_key] = now_ts
                    pnl = pos.current_pnl(curr)
                    notifications.append(
                        f"⚠️ <b>RISK ALERT: STOP LOSS PROXIMITY ({pos.symbol})</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Position ID</b>: <code>{pos.id}</code> ({pos.direction})\n"
                        f"• <b>Entry Price</b>: <code>${pos.entry_price:,.2f}</code>\n"
                        f"• <b>Current Market</b>: <code>${curr:,.2f}</code> (Danger Zone 🔴)\n"
                        f"• <b>Invalidation SL</b>: <code>${pos.stop_loss:,.2f}</code> (${abs(curr - pos.stop_loss):,.2f} buffer)\n"
                        f"• <b>Current Drawdown</b>: 🔴 <b>${pnl:,.2f}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"<i>💡 Review position with /position or close manually with /close {pos.id}</i>"
                    )
            elif pos.direction == "SHORT" and curr >= (pos.stop_loss - (risk_dist * 0.20)) and curr < pos.stop_loss:
                if now_ts - self._recent_level_alerts.get(sl_prox_key, 0) > 1800:
                    self._recent_level_alerts[sl_prox_key] = now_ts
                    pnl = pos.current_pnl(curr)
                    notifications.append(
                        f"⚠️ <b>RISK ALERT: STOP LOSS PROXIMITY ({pos.symbol})</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Position ID</b>: <code>{pos.id}</code> ({pos.direction})\n"
                        f"• <b>Entry Price</b>: <code>${pos.entry_price:,.2f}</code>\n"
                        f"• <b>Current Market</b>: <code>${curr:,.2f}</code> (Danger Zone 🔴)\n"
                        f"• <b>Invalidation SL</b>: <code>${pos.stop_loss:,.2f}</code> (${abs(pos.stop_loss - curr):,.2f} buffer)\n"
                        f"• <b>Current Drawdown</b>: 🔴 <b>${pnl:,.2f}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"<i>💡 Review position with /position or close manually with /close {pos.id}</i>"
                    )

            # c) Breakeven Milestone (+1.0R Reached)
            be_key = f"breakeven_milestone_{pos.id}"
            if pos.direction == "LONG" and curr >= (pos.entry_price + risk_dist) and pos.stop_loss < pos.entry_price:
                if now_ts - self._recent_level_alerts.get(be_key, 0) > 3600:
                    self._recent_level_alerts[be_key] = now_ts
                    notifications.append(
                        f"🛡️ <b>BREAKEVEN MILESTONE (+1.0R ACHIEVED): {pos.symbol}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Position ID</b>: <code>{pos.id}</code>\n"
                        f"• <b>Current Gain</b>: 🟢 <b>+${pos.current_pnl(curr):,.2f}</b>\n"
                        f"• <b>Status</b>: Profit is >= 1x initial risk! Risk-free territory.\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"<i>💡 Recommendation: Enable /trailing on to lock in Breakeven stop!</i>"
                    )
            elif pos.direction == "SHORT" and curr <= (pos.entry_price - risk_dist) and pos.stop_loss > pos.entry_price:
                if now_ts - self._recent_level_alerts.get(be_key, 0) > 3600:
                    self._recent_level_alerts[be_key] = now_ts
                    notifications.append(
                        f"🛡️ <b>BREAKEVEN MILESTONE (+1.0R ACHIEVED): {pos.symbol}</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"• <b>Position ID</b>: <code>{pos.id}</code>\n"
                        f"• <b>Current Gain</b>: 🟢 <b>+${pos.current_pnl(curr):,.2f}</b>\n"
                        f"• <b>Status</b>: Profit is >= 1x initial risk! Risk-free territory.\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"<i>💡 Recommendation: Enable /trailing on to lock in Breakeven stop!</i>"
                    )

        # 3. Institutional Order Block (OB) & Fair Value Gap (FVG) Level Alerts
        scan_syms = list(self.config.symbols) if self.config.symbols else [self.config.symbol]
        for sym in scan_syms:
            df = get_sym_candles(sym)
            if len(df) < 15:
                continue
            curr = float(df.iloc[-1]["close"])
            levels = detect_order_blocks_and_fvg(df)

            bob = levels.get("bullish_ob")
            if bob and bob.get("bottom") and bob.get("top"):
                b_lo, b_hi = float(bob["bottom"]), float(bob["top"])
                if b_lo <= curr <= b_hi:
                    ob_key = f"ob_bull_{sym}_{int(b_lo // 10)}"
                    if now_ts - self._recent_level_alerts.get(ob_key, 0) > 1800:
                        self._recent_level_alerts[ob_key] = now_ts
                        notifications.append(
                            f"🧱 <b>INSTITUTIONAL LEVEL REACHED ({sym})</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"• <b>Zone</b>: Bullish Demand Order Block (OB) 🟢\n"
                            f"• <b>Price Range</b>: <code>${b_lo:,.2f} – ${b_hi:,.2f}</code>\n"
                            f"• <b>Current Market</b>: <code>${curr:,.2f}</code>\n"
                            f"• <b>Confluence</b>: Smart Money Accumulation Support Area\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"<i>💡 Generate pinpoint setup: /entry {sym} | Order: /buy</i>"
                        )

            sob = levels.get("bearish_ob")
            if sob and sob.get("bottom") and sob.get("top"):
                s_lo, s_hi = float(sob["bottom"]), float(sob["top"])
                if s_lo <= curr <= s_hi:
                    ob_key = f"ob_bear_{sym}_{int(s_lo // 10)}"
                    if now_ts - self._recent_level_alerts.get(ob_key, 0) > 1800:
                        self._recent_level_alerts[ob_key] = now_ts
                        notifications.append(
                            f"🧱 <b>INSTITUTIONAL LEVEL REACHED ({sym})</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"• <b>Zone</b>: Bearish Supply Order Block (OB) 🔴\n"
                            f"• <b>Price Range</b>: <code>${s_lo:,.2f} – ${s_hi:,.2f}</code>\n"
                            f"• <b>Current Market</b>: <code>${curr:,.2f}</code>\n"
                            f"• <b>Confluence</b>: Smart Money Distribution Resistance Area\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"<i>💡 Generate pinpoint setup: /entry {sym} | Order: /sell</i>"
                        )

            bfvg = levels.get("bullish_fvg")
            if bfvg and bfvg.get("bottom") and bfvg.get("top"):
                f_lo, f_hi = float(bfvg["bottom"]), float(bfvg["top"])
                if f_lo <= curr <= f_hi:
                    fvg_key = f"fvg_bull_{sym}_{int(f_lo // 10)}"
                    if now_ts - self._recent_level_alerts.get(fvg_key, 0) > 1800:
                        self._recent_level_alerts[fvg_key] = now_ts
                        notifications.append(
                            f"⚡ <b>FAIR VALUE GAP (FVG) REACHED ({sym})</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"• <b>Imbalance</b>: Bullish Imbalance Zone 🟢\n"
                            f"• <b>Gap Range</b>: <code>${f_lo:,.2f} – ${f_hi:,.2f}</code>\n"
                            f"• <b>Current Market</b>: <code>${curr:,.2f}</code>\n"
                            f"• <b>Status</b>: Liquidity gap is being filled\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"<i>💡 View structural liquidity: /levels {sym}</i>"
                        )

        return notifications

    def fetch_binance_candles(self, symbol: str, count: int = 120, interval: str = "1m") -> Optional[pd.DataFrame]:
        """Fetch candle series from Binance REST API."""
        b_sym = ExchangeApiClient.format_binance_symbol(symbol)
        tf = interval if interval in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d") else "1m"
        url = f"https://api.binance.com/api/v3/klines?symbol={b_sym}&interval={tf}&limit={min(1000, count)}"
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                raw = resp.json()
                if raw and isinstance(raw, list) and len(raw) > 0:
                    rows = []
                    for c in raw:
                        rows.append({
                            "open": float(c[1]),
                            "high": float(c[2]),
                            "low": float(c[3]),
                            "close": float(c[4]),
                            "volume": float(c[5]),
                            "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                        })
                    df = pd.DataFrame(rows).set_index("timestamp").astype(float)
                    df = df[~df.index.duplicated(keep="last")]
                    df.sort_index(inplace=True)
                    if not df.empty:
                        return df.tail(count)
        except Exception as exc:
            logger.warning("Binance candle fetch error for %s: %s", b_sym, exc)
        return None

    def fetch_candles(self, symbol: str, count: int = 120, resolution: str = "1") -> pd.DataFrame:
        """Fetch candle series from active exchange (Delta or Binance) with dual-exchange failover."""
        # If active exchange is Binance, try Binance first
        if getattr(self.config, "live_exchange", "delta") == "binance":
            b_tf = "1d" if resolution == "1D" else f"{resolution}m" if resolution.isdigit() else "1m"
            b_df = self.fetch_binance_candles(symbol, count=count, interval=b_tf)
            if b_df is not None and not b_df.empty:
                return b_df

        target = self.normalize_symbol(symbol)
        now = int(time.time())
        
        # Determine minute multiplier for 'from_ts' calculation
        mult = 1
        if resolution == "1D":
            mult = 1440
        elif resolution.isdigit():
            mult = int(resolution)
            
        from_ts = now - (count + 30) * 60 * mult
        url = f"{DELTA_CHART_API}?symbol={target}&resolution={resolution}&from={from_ts}&to={now}"

        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                payload = resp.json()
                res = payload.get("result", {})
                if res and "c" in res and len(res["c"]) > 0:
                    t_arr = res.get("t", [])
                    o_arr = res.get("o", [])
                    h_arr = res.get("h", [])
                    l_arr = res.get("l", [])
                    c_arr = res.get("c", [])
                    v_arr = res.get("v", [])
                    n = min(len(t_arr), len(o_arr), len(h_arr), len(l_arr), len(c_arr))
                    if n > 0:
                        v_slice = v_arr[:n] if len(v_arr) >= n else [100.0] * n
                        df = pd.DataFrame(
                            {
                                "open": o_arr[:n],
                                "high": h_arr[:n],
                                "low": l_arr[:n],
                                "close": c_arr[:n],
                                "volume": v_slice,
                            },
                            index=pd.to_datetime(t_arr[:n], unit="s", utc=True),
                        ).astype(float)
                        df = df[~df.index.duplicated(keep="last")]
                        df.sort_index(inplace=True)
                        if not df.empty:
                            return df
        except Exception as exc:
            logger.warning("Delta candle fetch error for %s: %s", target, exc)

        # Fallback to Binance REST API if Delta is unavailable
        b_tf = "1d" if resolution == "1D" else f"{resolution}m" if resolution.isdigit() else "1m"
        b_df = self.fetch_binance_candles(symbol, count=count, interval=b_tf)
        if b_df is not None and not b_df.empty:
            return b_df

        # Fallback synthetic series for offline resilience and tests
        return self._generate_dummy_candles(target, count=count)

    def _generate_dummy_candles(self, target: str, count: int = 120) -> pd.DataFrame:
        """Generate deterministic synthetic candles for offline resilience, testing, and backtesting."""
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=count, freq="1min")
        if "BTC" in target:
            base, scale, spread = 82000.0, 10.0, 15.0
        elif "XAU" in target:
            base, scale, spread = 4135.0, 0.4, 1.0
        elif "ETH" in target:
            base, scale, spread = 2650.0, 0.5, 1.2
        elif "SOL" in target:
            base, scale, spread = 180.0, 0.1, 0.3
        elif "XRP" in target:
            base, scale, spread = 1.85, 0.002, 0.005
        else:
            base, scale, spread = 100.0, 0.05, 0.1

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

    def _close_single_position(
        self,
        pos: AutoTradePosition,
        current_price: Optional[float] = None,
        reason: str = "MANUAL",
    ) -> str:
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

        if pos in self.positions:
            self.positions.remove(pos)
            self._save_open_positions()

        # Continuous Self-Learning: Feedback closed trade into AI memory, adapt weights & strategy PnL
        self.learn_from_trade(pos, current_price=price, pnl=pnl, reason=reason)

        # Live order execution for live trading
        live_note = ""
        if pos.mode == "live" and self.exchange_client.is_configured:
            opp_dir = "SELL" if pos.direction == "LONG" else "BUY"
            try:
                ok_ord, ord_msg, _ = self.exchange_client.place_order(
                    symbol=pos.symbol,
                    direction=opp_dir,
                    size=pos.lot_size,
                    order_type="market",
                )
                if ok_ord:
                    live_note = f"\n• Live Exchange Order: <code>{ord_msg}</code>"
                else:
                    live_note = f"\n• Live Exchange Order Warning: <i>{ord_msg}</i>"
            except Exception as e:
                logger.warning("Could not submit live exit order for %s: %s", pos.id, e)

        sign = "+" if pnl >= 0 else ""
        return (
            f"🔄 Position Closed ({reason})\n"
            f"• ID: <code>{pos.id}</code>\n"
            f"• Pair: {pos.symbol}\n"
            f"• Direction: {pos.direction}\n"
            f"• Entry: {pos.entry_price:.2f} | Exit: {price:.2f}\n"
            f"• Lot Size: {pos.lot_size}\n"
            f"• Realized PnL: {sign}${pnl:.2f}\n"
            f"• New Account Equity: ${self.config.equity:.2f}{live_note}"
        )

    def close_current_position(
        self,
        current_price: Optional[float] = None,
        reason: str = "MANUAL",
        position_id: Optional[str] = None,
    ) -> Optional[str]:
        if not self.positions:
            return None

        target_pos: Optional[AutoTradePosition] = None
        if position_id:
            for p in self.positions:
                if p.id.lower() == position_id.strip().lower():
                    target_pos = p
                    break
            if not target_pos:
                return None
        else:
            target_pos = self.positions[0]

        return self._close_single_position(target_pos, current_price=current_price, reason=reason)

    def close_position_by_id(
        self, pos_id: str, current_price: Optional[float] = None, reason: str = "MANUAL"
    ) -> Optional[str]:
        for pos in list(self.positions):
            if pos.id.lower() == pos_id.strip().lower():
                return self._close_single_position(pos, current_price=current_price, reason=reason)
        return None

    def close_positions_by_symbol(
        self, symbol: str, current_price: Optional[float] = None, reason: str = "MANUAL"
    ) -> List[str]:
        sym_norm = self.normalize_symbol(symbol)
        closed_msgs: List[str] = []
        for pos in list(self.positions):
            if pos.symbol.upper() == sym_norm:
                msg = self._close_single_position(pos, current_price=current_price, reason=reason)
                closed_msgs.append(msg)
        return closed_msgs

    def close_all_positions(
        self, current_prices: Optional[Dict[str, float]] = None, reason: str = "MANUAL"
    ) -> List[str]:
        closed_msgs: List[str] = []
        for pos in list(self.positions):
            cp = current_prices.get(pos.symbol) if current_prices else None
            msg = self._close_single_position(pos, current_price=cp, reason=reason)
            closed_msgs.append(msg)
        return closed_msgs

    def step(self) -> List[str]:
        """
        Runs one evaluation cycle:
        1. Checks all active positions across pairs for TP/SL and Trailing SL triggers.
        2. Enforces daily risk guard if max daily loss is reached.
        3. If auto trade is ON and capacity permits, scans all configured symbols
           (e.g. BTCUSD and XAUTUSD) and executes qualified setups.
        Returns a list of notification strings for any significant events.
        """
        notifications: List[str] = []
        candles_cache: Dict[str, pd.DataFrame] = {}

        # 0. Autonomous Background Retraining & Self-Learning Loop
        if getattr(self.config, "auto_learn_enabled", True):
            self.auto_learn_step()

        def get_candles(sym: str) -> pd.DataFrame:
            if sym not in candles_cache:
                candles_cache[sym] = self.fetch_candles(sym, count=120)
            return candles_cache[sym]

        # 1. Manage Active Positions across all pairs
        for pos in list(self.positions):
            df_pos = get_candles(pos.symbol)
            if df_pos.empty:
                continue
            last_candle = df_pos.iloc[-1]
            curr_high = float(last_candle["high"])
            curr_low = float(last_candle["low"])
            curr_close = float(last_candle["close"])

            pos.highest_price = max(pos.highest_price, curr_high)
            pos.lowest_price = min(pos.lowest_price, curr_low)

            closed_event = False
            if pos.direction == "LONG":
                # Check Take Profit
                if curr_high >= pos.take_profit_1:
                    msg = self._close_single_position(pos, current_price=pos.take_profit_1, reason="TP1")
                    notifications.append(f"🎯 <b>TAKE PROFIT HIT ({pos.symbol})</b>\n{msg}")
                    closed_event = True
                # Check Stop Loss
                elif curr_low <= pos.stop_loss:
                    msg = self._close_single_position(pos, current_price=pos.stop_loss, reason="SL")
                    notifications.append(f"🛑 <b>STOP LOSS HIT ({pos.symbol})</b>\n{msg}")
                    closed_event = True

            elif pos.direction == "SHORT":
                # Check Take Profit
                if curr_low <= pos.take_profit_1:
                    msg = self._close_single_position(pos, current_price=pos.take_profit_1, reason="TP1")
                    notifications.append(f"🎯 <b>TAKE PROFIT HIT ({pos.symbol})</b>\n{msg}")
                    closed_event = True
                # Check Stop Loss
                elif curr_high >= pos.stop_loss:
                    msg = self._close_single_position(pos, current_price=pos.stop_loss, reason="SL")
                    notifications.append(f"🛑 <b>STOP LOSS HIT ({pos.symbol})</b>\n{msg}")
                    closed_event = True

            # Trailing Stop Loss dynamic update if position is still open
            if not closed_event and pos in self.positions and self.config.trailing_sl:
                initial_sl = pos.initial_stop_loss if pos.initial_stop_loss is not None else pos.stop_loss
                risk_amt = abs(pos.entry_price - initial_sl)
                if risk_amt <= 0:
                    risk_amt = max(pos.entry_price * 0.01, 1.0)
                if pos.direction == "LONG" and curr_close > pos.entry_price + risk_amt:
                    trail_target = round(curr_close - risk_amt, 2)
                    if trail_target > pos.stop_loss:
                        old_sl = pos.stop_loss
                        pos.stop_loss = trail_target
                        self._save_open_positions()
                        notifications.append(
                            f"🛡️ <b>Trailing Stop Moved Up ({pos.symbol}):</b> SL updated from {old_sl:.2f} ➔ <b>{pos.stop_loss:.2f}</b>"
                        )
                elif pos.direction == "SHORT" and curr_close < pos.entry_price - risk_amt:
                    trail_target = round(curr_close + risk_amt, 2)
                    if trail_target < pos.stop_loss:
                        old_sl = pos.stop_loss
                        pos.stop_loss = trail_target
                        self._save_open_positions()
                        notifications.append(
                            f"🛡️ <b>Trailing Stop Moved Down ({pos.symbol}):</b> SL updated from {old_sl:.2f} ➔ <b>{pos.stop_loss:.2f}</b>"
                        )

            # Progressive Auto Break-Even (Practical RiskManager)
            if not closed_event and pos in self.positions and getattr(self, "risk_manager", None) and getattr(self.risk_manager.config, "auto_be", True):
                initial_sl = pos.initial_stop_loss if pos.initial_stop_loss is not None else pos.stop_loss
                initial_risk = abs(pos.entry_price - initial_sl)
                if initial_risk > 0:
                    new_be_stop, be_reason = self.risk_manager.auto_breakeven_stop(
                        direction=pos.direction,
                        entry=pos.entry_price,
                        current_stop=pos.stop_loss,
                        highest=pos.highest_price,
                        lowest=pos.lowest_price,
                        initial_risk=initial_risk,
                    )
                    if be_reason:
                        should_update = False
                        if pos.direction == "LONG" and new_be_stop > pos.stop_loss:
                            should_update = True
                        elif pos.direction == "SHORT" and new_be_stop < pos.stop_loss:
                            should_update = True
                        if should_update:
                            old_sl = pos.stop_loss
                            pos.stop_loss = round(new_be_stop, 2)
                            self._save_open_positions()
                            notifications.append(
                                f"🛡️ <b>Auto Break-Even ({pos.symbol}):</b> SL moved from {old_sl:.2f} ➔ <b>{pos.stop_loss:.2f}</b> [{be_reason}]"
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

        # 2. Check for New Entries across configured symbols if auto-trade is ON
        if self.config.enabled and len(self.positions) < self.config.max_positions:
            if getattr(self, "risk_manager", None):
                can_tr, cant_reason = self.risk_manager.can_trade()
                if not can_tr:
                    logger.info("RiskManager paused entry: %s", cant_reason)
                    target_symbols = []
                else:
                    target_symbols = list(self.config.symbols) if self.config.symbols else [self.config.symbol]
            else:
                target_symbols = list(self.config.symbols) if self.config.symbols else [self.config.symbol]
            for sym in target_symbols:
                if len(self.positions) >= self.config.max_positions:
                    break

                sym_positions = [p for p in self.positions if p.symbol == sym]
                if len(sym_positions) >= self.config.max_positions_per_symbol:
                    continue

                df1 = get_candles(sym)
                if df1.empty:
                    continue
                curr_close = float(df1.iloc[-1]["close"])

                if isinstance(df1.index, pd.DatetimeIndex):
                    daily = (
                        df1.resample("1D")
                        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                        .dropna()
                    )
                else:
                    daily = pd.DataFrame()
                if len(daily) < 2:
                    spread_d = 400.0 if "BTC" in sym else 10.0
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

                # Deliberate among all 5 layers with mutual cross-examination & dependency
                delib = self._deliberation_engine.deliberate(sym, df1, self)

                strat_pro = self._strategies_pro.get(sym) or self._strategy_pro
                strat_itb = self._strategies_itb.get(sym) or self._strategy_itb
                strat_ai = self._strategies_ai.get(sym) or self._strategy_ai

                sig_pro = strat_pro.generate_signal(df1, daily) if strat_pro else None
                sig_itb = strat_itb.generate_signal(df1, daily) if strat_itb else None
                sig_ai = strat_ai.generate_signal(df1, daily) if strat_ai else None

                active_sigs = []
                for s, name in [(sig_pro, "Pro"), (sig_itb, "ITB"), (sig_ai, "AI")]:
                    if s is not None and getattr(s.direction, "name", "FLAT") != "FLAT":
                        active_sigs.append((s, name))

                sig = None
                names = [s[1] for s in active_sigs]
                strat_label = f"Ensemble ({'+'.join(names)})" if names else "Ensemble"
                target_dir = "FLAT"

                if active_sigs:
                    # Check for conflicts among active signals
                    directions = set(s[0].direction.name for s in active_sigs)
                    if len(directions) == 1:
                        target_dir = list(directions)[0]
                        # Verify against continuous multi-layer deliberation
                        try:
                            # Reject if multi-layer consensus is strongly opposed (no solo conflicting trades)
                            if (target_dir == "LONG" and delib.consensus_score < -0.15 and not delib.risk_approval) or \
                               (target_dir == "SHORT" and delib.consensus_score > 0.15 and not delib.risk_approval):
                                sig = None
                            else:
                                sig = active_sigs[0][0]
                                if hasattr(sig, "setup"):
                                    sig.setup = strat_label
                        except Exception:
                            sig = active_sigs[0][0]
                    else:
                        # Conflict across signals, stay flat
                        sig = None
                elif delib.risk_approval and delib.consensus_direction != "FLAT" and delib.conviction_multiplier > 0.0:
                    target_dir = delib.consensus_direction
                    strat_label = f"Ensemble (Deliberated x{delib.conviction_multiplier:.2f})"

                if sig is not None:
                    target_dir = sig.direction.name
                    entry = sig.entry
                    stop = sig.stop
                    tp1 = sig.tp1
                    tp2 = sig.tp2
                    setup_str = getattr(sig.setup, "value", str(sig.setup))
                    reason_str = sig.reason
                elif delib.risk_approval and delib.consensus_direction != "FLAT" and delib.conviction_multiplier > 0.0:
                    target_dir = delib.consensus_direction
                    atr_s = IndicatorEngine.atr_series(df1, 14)
                    atr_last = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0
                    atr_val = atr_last if (not math.isnan(atr_last) and atr_last > 0) else max(0.5, curr_close * 0.005)
                    if target_dir == "LONG":
                        entry = curr_close
                        stop = round(entry - 1.5 * atr_val, 2)
                        tp1 = round(entry + 2.0 * atr_val, 2)
                        tp2 = round(entry + 3.5 * atr_val, 2)
                    else:
                        entry = curr_close
                        stop = round(entry + 1.5 * atr_val, 2)
                        tp1 = round(entry - 2.0 * atr_val, 2)
                        tp2 = round(entry - 3.5 * atr_val, 2)
                    setup_str = f"Institutional Flow ({delib.verdict.split()[0]})"
                    reason_str = f"Multi-Layer Consensus ({len(delib.aligned_layers)}/4 layers)"
                else:
                    continue

                # Calculate lot size with dynamic conviction multiplier
                lots = self.calculate_lot_size(
                    symbol=sym,
                    entry_price=entry,
                    stop_loss=stop,
                    conviction_multiplier=delib.conviction_multiplier,
                )

                mode = self.config.trading_mode
                order_id = None
                if mode == "live":
                    ok_ord, ord_msg, _ = self.exchange_client.place_order(
                        symbol=sym,
                        direction=target_dir,
                        size=lots,
                        stop_loss=stop,
                        take_profit=tp1,
                    )
                    if not ok_ord:
                        notifications.append(
                            f"⚠️ <b>LIVE ORDER REJECTED ({sym})</b>: {ord_msg}"
                        )
                        continue
                    order_id = ord_msg

                pos_id = f"TRADE_{sym[:3]}_{int(time.time())}_{len(self.positions) + 1}"

                # Capture granular layer votes and execution telemetry for feedback learning
                engine_votes = {
                    "smc": getattr(delib.layers.get("smc_structure"), "stance", "NEUTRAL"),
                    "pro": getattr(delib.layers.get("technical_momentum"), "stance", "NEUTRAL"),
                    "itb": getattr(delib.layers.get("itb_machine_learning"), "stance", "NEUTRAL"),
                    "ai": getattr(delib.layers.get("ai_bot_learning"), "stance", "NEUTRAL"),
                }
                pro_l = delib.layers.get("technical_momentum")
                itb_l = delib.layers.get("itb_machine_learning")
                ai_l = delib.layers.get("ai_bot_learning")

                entry_rsi = float(pro_l.key_metrics.get("rsi", 50.0)) if pro_l else 50.0
                entry_adx = float(pro_l.key_metrics.get("adx", 25.0)) if pro_l else 25.0
                entry_score = float(pro_l.score) if pro_l else 0.0
                itb_score_val = float(itb_l.score) if itb_l else 0.0
                ai_conf_val = float(ai_l.confidence) if ai_l else 0.50
                regime_val = str(ai_l.key_metrics.get("regime", "NORMAL")) if ai_l else "NORMAL"

                now_hr = datetime.now(timezone.utc).hour
                session_val = "asian" if 0 <= now_hr < 8 else ("london" if 8 <= now_hr < 13 else ("overlap" if 13 <= now_hr < 17 else "ny"))

                new_pos = AutoTradePosition(
                    id=pos_id,
                    symbol=sym,
                    direction=target_dir,
                    entry_price=round(entry, 2),
                    stop_loss=round(stop, 2),
                    take_profit_1=round(tp1, 2),
                    take_profit_2=round(tp2, 2) if tp2 else None,
                    lot_size=round(lots, 4),
                    entry_time=datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    ),
                    strategy=strat_label,
                    reason=reason_str,
                    highest_price=entry,
                    lowest_price=entry,
                    mode=mode,
                    exchange_order_id=order_id,
                    initial_stop_loss=round(stop, 2),
                    engine_votes=engine_votes,
                    entry_rsi=round(entry_rsi, 1),
                    entry_adx=round(entry_adx, 1),
                    entry_score=round(entry_score, 2),
                    ai_conf=round(ai_conf_val, 2),
                    itb_score=round(itb_score_val, 2),
                    regime=regime_val,
                    session=session_val,
                    rel_vol=round(getattr(delib, "rel_vol", 1.0), 2),
                    conviction_mult=round(delib.conviction_multiplier, 2),
                    quality_score=round(getattr(delib, "quality_score", 3.5), 2),
                    mtf_alignment=getattr(delib, "mtf_alignment", 1),
                )
                self.positions.append(new_pos)
                self._save_open_positions()
                mode_tag = " [LIVE 🚨]" if mode == "live" else " [PAPER 📄]"
                notifications.append(
                    f"🚀 <b>AUTO TRADE OPENED ({new_pos.symbol}){mode_tag}</b>\n"
                    f"• ID: <code>{new_pos.id}</code>\n"
                    f"• Mode: <b>{new_pos.mode.upper()}</b>\n"
                    f"• Symbol: {new_pos.symbol}\n"
                    f"• Direction: <b>{new_pos.direction}</b>\n"
                    f"• Entry: {new_pos.entry_price:.2f}\n"
                    f"• Lot Size: {new_pos.lot_size}\n"
                    f"• Take Profit: {new_pos.take_profit_1:.2f}\n"
                    f"• Stop Loss: {new_pos.stop_loss:.2f}\n"
                    f"• Strategy: {new_pos.strategy}\n"
                    f"• Setup: {setup_str}\n"
                    f"• Quality Score: <b>{new_pos.quality_score:.1f}/5.0</b> (MTF: {new_pos.mtf_alignment}/3 | Vol×{new_pos.rel_vol:.1f})\n"
                    f"• Layer Votes: SMC:{engine_votes['smc'][:1]} | Pro:{engine_votes['pro'][:1]} | ITB:{engine_votes['itb'][:1]} | AI:{engine_votes['ai'][:1]}"
                )
        # 3. Check and trigger Trade Level Alerts (Price levels, OB/FVG zones, Proximity)
        level_alerts = self.check_trade_level_alerts(candles_cache=candles_cache)
        notifications.extend(level_alerts)

        return notifications

    def get_status_text(self) -> str:
        status_icon = (
            "🟢 <b>ACTIVE (ON)</b>" if self.config.enabled else "🔴 <b>DISABLED (OFF)</b>"
        )
        lot_mode_str = (
            "Manual Fixed"
            if self.config.lot_mode in ("fixed", "manual")
            else f"Auto Risk {self.config.risk_pct}%"
        )
        symbols_str = ", ".join(self.config.symbols) if self.config.symbols else self.config.symbol
        mode_icon = "📄 PAPER TRADING" if self.config.trading_mode == "paper" else "🚨 LIVE TRADING"
        mode_color = "🟢" if self.config.trading_mode == "paper" else "🔴"

        pos_bar = make_modern_meter((len(self.positions) / max(1, self.config.max_positions)) * 100.0, width=8, fill_char="■", empty_char="░")
        text = [
            "⚡ <b>AUTO TRADING DASHBOARD</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"• <b>Status</b>: {status_icon}",
            f"• <b>Trading Mode</b>: {mode_color} <b>{mode_icon}</b>",
            f"• <b>Active Pairs</b>: <code>{symbols_str}</code>",
            f"• <b>Primary Symbol</b>: <code>{self.config.symbol}</code>",
            f"• <b>Strategy</b>: 🌟 Combined Ensemble",
            f"• <b>Lot Size</b>: {self.config.lot_size} ({lot_mode_str})",
            f"• <b>Max Positions</b>: <code>[{pos_bar}]</code> {len(self.positions)}/{self.config.max_positions} (Max/pair: {self.config.max_positions_per_symbol})",
            f"• <b>Take Profit</b>: {self.config.tp_value} ({self.config.tp_mode.upper()})",
            f"• <b>Stop Loss</b>: {self.config.sl_value} ({self.config.sl_mode.upper()})",
            f"• <b>Trailing Stop</b>: {'🟢 ON' if self.config.trailing_sl else '⚪ OFF'}",
            f"• <b>Daily Risk Guard</b>: {self.config.max_daily_loss_pct}% equity",
            f"• <b>Capital / Equity</b>: <code>${self.config.equity:,.2f}</code>",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if self.positions:
            text.append(f"📊 <b>Active Positions ({len(self.positions)}/{self.config.max_positions}):</b>")
            for pos in self.positions:
                df = self.fetch_candles(pos.symbol, count=3)
                cp = float(df["close"].iloc[-1]) if not df.empty else pos.entry_price
                pnl = pos.current_pnl(cp)
                sign = "+" if pnl >= 0 else ""
                emoji = "🟢" if pnl >= 0 else "🔴"
                text.extend(
                    [
                        f"  • <b>[{pos.id}]</b> {pos.direction} {pos.symbol} @ {pos.entry_price:.2f}",
                        f"    Lot: {pos.lot_size} | TP: {pos.take_profit_1:.2f} | SL: {pos.stop_loss:.2f}",
                        f"    PnL: {emoji} {sign}${pnl:.2f} | Time: {pos.entry_time}",
                    ]
                )
        else:
            text.append("📊 <b>Active Positions</b>: None (0 open)")

        text.append("━━━━━━━━━━━━━━━━━━━━━━")
        total_closed = len(self.closed_trades)
        wins = [t for t in self.closed_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in self.closed_trades)
        win_rate = (len(wins) / total_closed * 100) if total_closed > 0 else 0.0
        win_bar = make_modern_meter(win_rate, width=8, fill_char="█", empty_char="░")

        text.extend(
            [
                "📈 <b>Trade Performance:</b>",
                f"  • Total Closed: {total_closed}",
                f"  • Win Rate: <code>[{win_bar}]</code> <b>{win_rate:.1f}%</b> ({len(wins)}/{total_closed})",
                f"  • Net PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}",
            ]
        )

        return "\n".join(text)

    def get_position_text(self) -> str:
        """Return formatted dashboard text of all open positions or idle status."""
        if not self.positions:
            symbols_str = ", ".join(self.config.symbols) if self.config.symbols else self.config.symbol
            return (
                "💼 <b>Active Positions Dashboard</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "• <b>Status</b>: No active open positions\n"
                f"• <b>Monitoring Pairs</b>: <code>{symbols_str}</code>\n"
                f"• <b>Auto-Trading</b>: {'🟢 ENABLED' if self.config.enabled else '🔴 DISABLED'}\n"
                f"• <b>Capacity</b>: 0/{self.config.max_positions} (Max {self.config.max_positions_per_symbol}/pair)\n"
                f"• <b>Account Equity</b>: <code>${self.config.equity:,.2f}</code>\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>💡 Market is continuously scanned for BTC & Gold high-probability setups.</i>"
            )

        total_unrealized = 0.0
        lines = [
            f"💼 <b>Active Positions ({len(self.positions)}/{self.config.max_positions})</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        for idx, pos in enumerate(self.positions, start=1):
            df = self.fetch_candles(pos.symbol, count=5)
            current_price = float(df["close"].iloc[-1]) if not df.empty else pos.entry_price
            pnl = pos.current_pnl(current_price)
            total_unrealized += pnl
            denom = pos.entry_price if pos.entry_price > 0 else 1.0
            pnl_pct = (
                ((current_price - pos.entry_price) / denom * 100.0)
                if pos.direction == "LONG"
                else ((pos.entry_price - current_price) / denom * 100.0)
            )
            tp_span = abs(pos.take_profit_1 - pos.entry_price)
            if tp_span > 0:
                if pos.direction == "LONG":
                    tp_progress = max(0.0, min(100.0, ((current_price - pos.entry_price) / tp_span) * 100.0))
                else:
                    tp_progress = max(0.0, min(100.0, ((pos.entry_price - current_price) / tp_span) * 100.0))
                tp_meter = f" <code>[{make_modern_meter(tp_progress, width=6, fill_char='█', empty_char='░')}]</code>"
            else:
                tp_meter = ""
            sign = "+" if pnl >= 0 else ""
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            tp2_str = f" | TP2: <code>${pos.take_profit_2:,.2f}</code>" if pos.take_profit_2 else ""

            lines.append(
                f"<b>#{idx} • {pos.symbol} [{pos.id}]</b>\n"
                f"• Direction: <b>{pos.direction}</b> {'🟢' if pos.direction == 'LONG' else '🔴'}\n"
                f"• Entry: <code>${pos.entry_price:,.2f}</code> | Current: <code>${current_price:,.2f}</code>\n"
                f"• PnL: {pnl_emoji} <b>{sign}${pnl:,.2f}</b> ({sign}{pnl_pct:.2f}%)\n"
                f"• Lot: <code>{pos.lot_size}</code> | SL: <code>${pos.stop_loss:,.2f}</code> | TP1: <code>${pos.take_profit_1:,.2f}</code>{tp_meter}{tp2_str}\n"
                f"• Opened: <code>{pos.entry_time}</code>"
            )
            if idx < len(self.positions):
                lines.append("──────────────────────")

        sign_tot = "+" if total_unrealized >= 0 else ""
        tot_emoji = "🟢" if total_unrealized >= 0 else "🔴"
        lines.extend(
            [
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"• <b>Total Unrealized PnL</b>: {tot_emoji} <b>{sign_tot}${total_unrealized:,.2f}</b>",
                f"• <b>Account Equity</b>: <code>${self.config.equity:,.2f}</code>",
                "━━━━━━━━━━━━━━━━━━━━━━",
                "<i>💡 Use <code>/close [ID|SYM|all]</code> or <code>/autotrade close [ID]</code> to exit positions.</i>",
            ]
        )
        return "\n".join(lines)

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
        win_meter = make_modern_meter(win_rate, width=10, fill_char="█", empty_char="░")
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
            f"• <b>Win Rate</b>: <code>[{win_meter}]</code> <b>{win_rate:.1f}%</b>\n"
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
        target = self.normalize_symbol(symbol)
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

        delib_txt = ""
        try:
            ens = self.evaluate_ensemble(target)
            d = ens.get("deliberation")
            q_val = getattr(d, "quality_score", 3.5) if d else 3.5
            mtf_sum = getattr(d, "mtf_summary", "MTF: Active") if d else "MTF: Active"
            r_vol = getattr(d, "rel_vol", 1.0) if d else 1.0
            delib_txt = (
                f"• <b>5-Layer Deliberation</b>: <b>{ens['verdict']}</b> ({ens['agreement_pct']}% agreement)\n"
                f"• <b>Quality Gate & MTF</b>: <code>Q={q_val:.1f}/5.0</code> ({mtf_sum} | RVOL: {r_vol:.2f}x)\n"
            )
        except Exception:
            delib_txt = ""

        return (
            f"📊 <b>TECHNICAL ANALYSIS: {target}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Signal</b>: <b>{rec}</b>\n"
            f"• <b>Confluence Score</b>: <code>{score:+.2f} / 5.0</code>\n"
            f"{delib_txt}"
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
            f"<i>💡 Auto trade: /autotrade on | Pinpoint entry: /entry {target} | Master scan: /scan {target}</i>"
        )

    def open_position_manually(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float] = None,
        lot_size: Optional[float] = None,
        reason: str = "Pinpoint Manual Execution",
        strategy: str = "Pinpoint Strategy",
    ) -> Tuple[bool, str, Optional[AutoTradePosition]]:
        """
        Manually or semi-automatically open a new position with custom or pinpoint parameters.
        Enforces maximum overall positions and per-symbol positions.
        """
        if len(self.positions) >= self.config.max_positions:
            return (
                False,
                f"⚠️ Maximum total concurrent positions limit reached "
                f"({len(self.positions)}/{self.config.max_positions}).\n"
                f"Please close an existing position first using <code>/close [ID]</code> or <code>/close all</code>.",
                None,
            )

        symbol_upper = self.normalize_symbol(symbol)
        sym_positions = [p for p in self.positions if p.symbol == symbol_upper]
        if len(sym_positions) >= self.config.max_positions_per_symbol:
            return (
                False,
                f"⚠️ An active position is already open for <b>{symbol_upper}</b> "
                f"({len(sym_positions)}/{self.config.max_positions_per_symbol}).\n"
                f"Please close the open {symbol_upper} position first with <code>/autotrade close</code> or increase limit with <code>/autotrade max</code>.",
                None,
            )

        if lot_size is None or lot_size <= 0:
            lot_size = self.calculate_lot_size(
                symbol=symbol_upper,
                entry_price=entry_price,
                stop_loss=stop_loss,
                conviction_multiplier=1.0,
            )

        pos_id = f"TRADE_{symbol_upper[:3]}_{int(time.time())}_{len(self.positions) + 1}"
        new_pos = AutoTradePosition(
            id=pos_id,
            symbol=symbol_upper,
            direction=direction.upper(),
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(take_profit_1, 2),
            take_profit_2=round(take_profit_2, 2) if take_profit_2 else None,
            lot_size=round(lot_size, 4),
            entry_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            strategy=strategy,
            reason=reason,
            highest_price=entry_price,
            lowest_price=entry_price,
            mode=self.config.trading_mode,
            initial_stop_loss=round(stop_loss, 2),
        )

        if self.config.trading_mode == "live":
            ok_ord, ord_msg, _ = self.exchange_client.place_order(
                symbol=symbol_upper,
                direction=direction.upper(),
                size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit_1,
            )
            if not ok_ord:
                return False, f"⚠️ Exchange live order rejected: {ord_msg}", None
            new_pos.exchange_order_id = ord_msg

        self.positions.append(new_pos)
        self._save_open_positions()
        tp2_str = (
            f"• <b>Take Profit 2</b>: <code>${new_pos.take_profit_2:,.2f}</code>\n"
            if new_pos.take_profit_2
            else ""
        )
        mode_tag = " [LIVE 🚨]" if new_pos.mode == "live" else " [PAPER 📄]"
        return (
            True,
            f"🚀 <b>POSITION OPENED SUCCESSFULLY{mode_tag}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>ID</b>: <code>{new_pos.id}</code>\n"
            f"• <b>Mode</b>: <b>{new_pos.mode.upper()}</b>\n"
            f"• <b>Symbol</b>: <code>{new_pos.symbol}</code>\n"
            f"• <b>Direction</b>: <b>{new_pos.direction}</b> {'🟢' if new_pos.direction == 'LONG' else '🔴'}\n"
            f"• <b>Entry Price</b>: <code>${new_pos.entry_price:,.2f}</code>\n"
            f"• <b>Lot Size</b>: <code>{new_pos.lot_size}</code>\n"
            f"• <b>Stop Loss</b>: <code>${new_pos.stop_loss:,.2f}</code>\n"
            f"• <b>Take Profit 1</b>: <code>${new_pos.take_profit_1:,.2f}</code>\n"
            f"{tp2_str}"
            f"• <b>Strategy</b>: {new_pos.strategy}\n"
            f"• <b>Reason</b>: {new_pos.reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>💡 Managed automatically. Use /position or /close at any time.</i>",
            new_pos,
        )


_GLOBAL_AUTO_TRADER: Optional[AutoTrader] = None


def get_auto_trader() -> AutoTrader:
    global _GLOBAL_AUTO_TRADER
    if _GLOBAL_AUTO_TRADER is None:
        _GLOBAL_AUTO_TRADER = AutoTrader()
    return _GLOBAL_AUTO_TRADER


# ==================================================================
# 2.5 PINPOINT TRADE ENTRY & TARGET ANALYZER & LIQUIDITY LEVELS
# ==================================================================


@dataclass
class PinpointTradePlan:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    current_price: float
    market_entry: float
    limit_entry: float
    breakout_entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_amount: float
    rr_ratio_tp1: float
    rr_ratio_tp2: float
    rr_ratio_tp3: float
    recommended_lots: float
    order_block_zone: Optional[Tuple[float, float]] = None
    order_block_type: Optional[str] = None
    fvg_zone: Optional[Tuple[float, float]] = None
    fvg_type: Optional[str] = None
    confidence_score: float = 0.0
    atr: float = 0.0
    reason: str = ""
    timestamp: str = ""


def detect_order_blocks_and_fvg(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scans candlestick price action for:
    - Bullish and Bearish Order Blocks (OB)
    - Fair Value Gaps (FVG)
    - Structural swing levels
    """
    result: Dict[str, Any] = {
        "bullish_ob": None,
        "bearish_ob": None,
        "bullish_fvg": None,
        "bearish_fvg": None,
        "all_obs": [],
        "all_fvgs": [],
        "swing_high": None,
        "swing_low": None,
    }
    if len(df) < 5:
        return result

    # 1. Structural Swing High & Low
    recent_window = df.tail(min(30, len(df)))
    result["swing_high"] = float(recent_window["high"].max())
    result["swing_low"] = float(recent_window["low"].min())

    # 2. Fair Value Gaps (FVG 3-candle imbalance)
    for i in range(2, len(df)):
        c_prev2 = df.iloc[i - 2]
        c_curr = df.iloc[i]
        c_p2_high = float(c_prev2["high"])
        c_p2_low = float(c_prev2["low"])
        c_curr_high = float(c_curr["high"])
        c_curr_low = float(c_curr["low"])

        # Bullish FVG
        if c_curr_low > c_p2_high:
            gap = {
                "type": "BULLISH",
                "bottom": c_p2_high,
                "top": c_curr_low,
                "size": c_curr_low - c_p2_high,
                "index": i,
            }
            result["all_fvgs"].append(gap)
            result["bullish_fvg"] = gap

        # Bearish FVG
        elif c_curr_high < c_p2_low:
            gap = {
                "type": "BEARISH",
                "bottom": c_curr_high,
                "top": c_p2_low,
                "size": c_p2_low - c_curr_high,
                "index": i,
            }
            result["all_fvgs"].append(gap)
            result["bearish_fvg"] = gap

    # 3. Order Blocks (OB: last candle prior to displacement)
    for i in range(1, len(df) - 1):
        c_ob = df.iloc[i]
        c_next = df.iloc[i + 1]
        ob_close = float(c_ob["close"])
        ob_open = float(c_ob["open"])
        ob_high = float(c_ob["high"])
        ob_low = float(c_ob["low"])
        next_close = float(c_next["close"])

        # Bullish OB
        if ob_close < ob_open and next_close > ob_high:
            ob = {
                "type": "BULLISH",
                "bottom": ob_low,
                "top": ob_high,
                "index": i,
            }
            result["all_obs"].append(ob)
            result["bullish_ob"] = ob

        # Bearish OB
        elif ob_close > ob_open and next_close < ob_low:
            ob = {
                "type": "BEARISH",
                "bottom": ob_low,
                "top": ob_high,
                "index": i,
            }
            result["all_obs"].append(ob)
            result["bearish_ob"] = ob

    return result


def generate_pinpoint_plan(
    symbol: str,
    direction_override: Optional[str] = None,
    risk_pct: Optional[float] = None,
    trader: Optional[AutoTrader] = None,
) -> PinpointTradePlan:
    """
    Computes a Pinpoint Trade Entry & Target Plan:
    - 3-tier Entry: Market (Now), Optimal Limit (OB / 50% Pullback), Breakout
    - Pinpoint Invalidation Stop Loss: Structural Swing + 1.2x ATR buffer
    - Precision Multi-tier Take Profit: TP1 (1.5R), TP2 (2.6R), TP3 (4.2R Runner)
    - Position Sizing (lots) based on risk budget
    - Institutional Order Blocks & FVGs confluence
    """
    if trader is None:
        trader = get_auto_trader()
    target = trader.normalize_symbol(symbol)
    df = trader.fetch_candles(target, count=120)
    if df.empty:
        df = trader._generate_dummy_candles(target, count=120)

    eng = IndicatorEngine()
    snap = eng.compute(df)
    atr_s = IndicatorEngine.atr_series(df, 14)
    atr_last = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0
    curr_price = float(df["close"].iloc[-1])
    atr = atr_last if (not math.isnan(atr_last) and atr_last > 0) else max(0.5, curr_price * 0.005)

    levels = detect_order_blocks_and_fvg(df)
    swing_high = levels["swing_high"] or (curr_price + atr * 2)
    swing_low = levels["swing_low"] or (curr_price - atr * 2)

    if direction_override:
        ov = direction_override.strip().upper()
        if ov in ("BUY", "LONG", "B"):
            direction = "LONG"
        elif ov in ("SELL", "SHORT", "S"):
            direction = "SHORT"
        else:
            direction = "LONG" if snap.score >= 0 else "SHORT"
    else:
        try:
            ens = trader.evaluate_ensemble(target)
            if ens["direction"] in ("LONG", "SHORT"):
                direction = ens["direction"]
            elif snap.score >= 0.5:
                direction = "LONG"
            elif snap.score <= -0.5:
                direction = "SHORT"
            else:
                direction = "LONG" if snap.supertrend_dir == 1 else "SHORT"
        except Exception:
            if snap.score >= 0.5:
                direction = "LONG"
            elif snap.score <= -0.5:
                direction = "SHORT"
            else:
                direction = "LONG" if snap.supertrend_dir == 1 else "SHORT"

    market_entry = curr_price

    if direction == "LONG":
        raw_sl = swing_low - (1.2 * atr)
        min_sl = curr_price - (0.8 * atr)
        max_dist_sl = curr_price - (3.0 * atr)
        sl = max(max_dist_sl, min(min_sl, raw_sl))
        risk_dist = max(market_entry - sl, atr * 0.5)
        sl = round(market_entry - risk_dist, 2)

        bob = levels["bullish_ob"]
        if bob and bob["top"] < curr_price and bob["top"] > sl:
            limit_entry = round(bob["top"], 2)
        else:
            limit_entry = round(curr_price - (0.5 * atr), 2)

        breakout_entry = round(swing_high + (0.2 * atr), 2)

        tp1 = round(market_entry + (1.5 * risk_dist), 2)
        tp2 = round(market_entry + (2.6 * risk_dist), 2)
        tp3 = round(market_entry + (4.2 * risk_dist), 2)

        ob_zone = (round(bob["bottom"], 2), round(bob["top"], 2)) if bob else None
        ob_type = "Bullish Demand OB" if bob else None
        bfvg = levels["bullish_fvg"]
        fvg_zone = (round(bfvg["bottom"], 2), round(bfvg["top"], 2)) if bfvg else None
        fvg_type = "Bullish FVG Imbalance" if bfvg else None
        reason = f"Confluence score {snap.score:+.2f} with Bullish Momentum"

    else:  # SHORT
        raw_sl = swing_high + (1.2 * atr)
        min_sl = curr_price + (0.8 * atr)
        max_dist_sl = curr_price + (3.0 * atr)
        sl = min(max_dist_sl, max(min_sl, raw_sl))
        risk_dist = max(sl - market_entry, atr * 0.5)
        sl = round(market_entry + risk_dist, 2)

        sob = levels["bearish_ob"]
        if sob and sob["bottom"] > curr_price and sob["bottom"] < sl:
            limit_entry = round(sob["bottom"], 2)
        else:
            limit_entry = round(curr_price + (0.5 * atr), 2)

        breakout_entry = round(swing_low - (0.2 * atr), 2)

        tp1 = round(market_entry - (1.5 * risk_dist), 2)
        tp2 = round(market_entry - (2.6 * risk_dist), 2)
        tp3 = round(market_entry - (4.2 * risk_dist), 2)

        ob_zone = (round(sob["bottom"], 2), round(sob["top"], 2)) if sob else None
        ob_type = "Bearish Supply OB" if sob else None
        sfvg = levels["bearish_fvg"]
        fvg_zone = (round(sfvg["bottom"], 2), round(sfvg["top"], 2)) if sfvg else None
        fvg_type = "Bearish FVG Imbalance" if sfvg else None
        reason = f"Confluence score {snap.score:+.2f} with Bearish Momentum"

    eff_risk_pct = risk_pct if risk_pct and risk_pct > 0 else trader.config.risk_pct
    risk_dollars = trader.config.equity * (eff_risk_pct / 100.0)
    decimals = 3 if "BTC" in target else 2
    min_lot = 0.001 if "BTC" in target else 0.01
    rec_lots = max(min_lot, round(risk_dollars / risk_dist, decimals))

    plan = PinpointTradePlan(
        symbol=target,
        direction=direction,
        current_price=round(curr_price, 2),
        market_entry=round(market_entry, 2),
        limit_entry=round(limit_entry, 2),
        breakout_entry=round(breakout_entry, 2),
        stop_loss=round(sl, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        take_profit_3=round(tp3, 2),
        risk_amount=round(risk_dist, 2),
        rr_ratio_tp1=1.5,
        rr_ratio_tp2=2.6,
        rr_ratio_tp3=4.2,
        recommended_lots=rec_lots,
        order_block_zone=ob_zone,
        order_block_type=ob_type,
        fvg_zone=fvg_zone,
        fvg_type=fvg_type,
        confidence_score=round(snap.score, 2),
        atr=round(atr, 2),
        reason=reason,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    return plan


def format_pinpoint_report(plan: PinpointTradePlan) -> str:
    """Format a PinpointTradePlan into a rich Telegram HTML report card."""
    dir_emoji = "🟢" if plan.direction == "LONG" else "🔴"
    ob_str = (
        f"<code>${plan.order_block_zone[0]:,.2f} – ${plan.order_block_zone[1]:,.2f}</code> ({plan.order_block_type})"
        if plan.order_block_zone
        else "<i>None within immediate range</i>"
    )
    fvg_str = (
        f"<code>${plan.fvg_zone[0]:,.2f} – ${plan.fvg_zone[1]:,.2f}</code> ({plan.fvg_type})"
        if plan.fvg_zone
        else "<i>None within immediate range</i>"
    )

    return (
        f"🎯 <b>PINPOINT TRADE ENTRY & TARGET ANALYZER</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Asset</b>: <code>{plan.symbol}</code>\n"
        f"• <b>Direction Bias</b>: <b>{plan.direction}</b> {dir_emoji}\n"
        f"• <b>Confluence Score</b>: <code>{plan.confidence_score:+.2f} / 5.0</code>\n"
        f"• <b>Current Market Price</b>: <code>${plan.current_price:,.2f}</code>\n"
        f"• <b>Volatility (ATR 14)</b>: <code>${plan.atr:,.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 <b>ENTRY EXECUTION TIERS:</b>\n"
        f"  1️⃣ <b>Market Entry (Now)</b>: <code>${plan.market_entry:,.2f}</code>\n"
        f"  2️⃣ <b>Optimal Limit (Pullback)</b>: <code>${plan.limit_entry:,.2f}</code>\n"
        f"  3️⃣ <b>Momentum Breakout</b>: <code>${plan.breakout_entry:,.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ <b>PINPOINT INVALIDATION (STOP LOSS):</b>\n"
        f"• <b>Stop Loss</b>: <code>${plan.stop_loss:,.2f}</code>\n"
        f"• <b>Risk per Unit</b>: <code>${plan.risk_amount:,.2f}</code> (Swing + 1.2x ATR buffer)\n"
        f"• <b>Recommended Position Size</b>: <code>{plan.recommended_lots} lots</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>PRECISION TAKE PROFIT TARGETS:</b>\n"
        f"• <b>TP1 (Conservative 1.5R)</b>: <code>${plan.take_profit_1:,.2f}</code> (R:R {plan.rr_ratio_tp1:.1f}:1)\n"
        f"• <b>TP2 (Structural 2.6R)</b>: <code>${plan.take_profit_2:,.2f}</code> (R:R {plan.rr_ratio_tp2:.1f}:1)\n"
        f"• <b>TP3 (Runner 4.2R)</b>: <code>${plan.take_profit_3:,.2f}</code> (R:R {plan.rr_ratio_tp3:.1f}:1)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧱 <b>INSTITUTIONAL LIQUIDITY ZONES:</b>\n"
        f"• <b>Key Order Block (OB)</b>: {ob_str}\n"
        f"• <b>Fair Value Gap (FVG)</b>: {fvg_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>💡 Quick Execute: <code>/execute</code> or <code>/buy</code> | Custom lots: <code>/execute {plan.recommended_lots}</code></i>"
    )


def execute_pinpoint_plan(
    plan: PinpointTradePlan,
    lot_override: Optional[float] = None,
    entry_mode: str = "market",
    trader: Optional[AutoTrader] = None,
) -> Tuple[bool, str]:
    """Execute a pinpoint trade plan directly through the AutoTrader engine."""
    if trader is None:
        trader = get_auto_trader()
    if entry_mode == "limit":
        entry_price = plan.limit_entry
    elif entry_mode == "breakout":
        entry_price = plan.breakout_entry
    else:
        entry_price = plan.market_entry

    lots = lot_override if lot_override and lot_override > 0 else plan.recommended_lots
    success, msg, _ = trader.open_position_manually(
        symbol=plan.symbol,
        direction=plan.direction,
        entry_price=entry_price,
        stop_loss=plan.stop_loss,
        take_profit_1=plan.take_profit_1,
        take_profit_2=plan.take_profit_2,
        lot_size=lots,
        reason=f"Pinpoint {plan.direction} ({plan.reason})",
        strategy="Pinpoint Strategy",
    )
    return success, msg


def get_levels_report(symbol: str, trader: Optional[AutoTrader] = None) -> str:
    """
    Generates institutional Smart Money Liquidity report
    showing Order Blocks (OB), Fair Value Gaps (FVG), swing levels, and volume zones.
    """
    if trader is None:
        trader = get_auto_trader()
    target = trader.normalize_symbol(symbol)
    df = trader.fetch_candles(target, count=120)
    if df.empty:
        df = trader._generate_dummy_candles(target, count=120)

    curr_price = float(df["close"].iloc[-1])
    atr_s = IndicatorEngine.atr_series(df, 14)
    atr_last = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0
    atr = atr_last if (not math.isnan(atr_last) and atr_last > 0) else max(0.5, curr_price * 0.005)
    levels = detect_order_blocks_and_fvg(df)

    bob = levels["bullish_ob"]
    bob_txt = (
        f"<code>${bob['bottom']:,.2f} – ${bob['top']:,.2f}</code> (Demand Block)"
        if bob
        else "<i>None identified in local window</i>"
    )

    sob = levels["bearish_ob"]
    sob_txt = (
        f"<code>${sob['bottom']:,.2f} – ${sob['top']:,.2f}</code> (Supply Block)"
        if sob
        else "<i>None identified in local window</i>"
    )

    bfvg = levels["bullish_fvg"]
    bfvg_txt = (
        f"<code>${bfvg['bottom']:,.2f} – ${bfvg['top']:,.2f}</code> (Bullish Gap)"
        if bfvg
        else "<i>None open nearby</i>"
    )
    sfvg = levels["bearish_fvg"]
    sfvg_txt = (
        f"<code>${sfvg['bottom']:,.2f} – ${sfvg['top']:,.2f}</code> (Bearish Gap)"
        if sfvg
        else "<i>None open nearby</i>"
    )

    swing_hi = levels["swing_high"] or curr_price
    swing_lo = levels["swing_low"] or curr_price

    step = 500.0 if "BTC" in target else 25.0
    psy_above = (int(curr_price // step) + 1) * step
    psy_below = int(curr_price // step) * step

    return (
        f"🧱 <b>SMART MONEY & LIQUIDITY LEVELS: {target}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Current Price</b>: <code>${curr_price:,.2f}</code>\n"
        f"• <b>Volatility (ATR)</b>: <code>${atr:,.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📦 <b>ORDER BLOCKS (INSTITUTIONAL FOOTPRINT):</b>\n"
        f"• 🟢 <b>Bullish Demand OB</b>: {bob_txt}\n"
        f"• 🔴 <b>Bearish Supply OB</b>: {sob_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ <b>FAIR VALUE GAPS (FVG IMBALANCES):</b>\n"
        f"• 🟢 <b>Bullish FVG</b>: {bfvg_txt}\n"
        f"• 🔴 <b>Bearish FVG</b>: {sfvg_txt}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 <b>KEY STRUCTURAL LIQUIDITY:</b>\n"
        f"• <b>Recent Swing High (BSL)</b>: <code>${swing_hi:,.2f}</code>\n"
        f"• <b>Recent Swing Low (SSL)</b>: <code>${swing_lo:,.2f}</code>\n"
        f"• <b>Psychological Levels</b>: <code>${psy_below:,.0f}</code> | <code>${psy_above:,.0f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>💡 Generate pinpoint entry: <code>/entry {target}</code></i>"
    )


def calculate_risk_reward(
    entry: float,
    sl: float,
    tp: Optional[float] = None,
    risk_dollars: Optional[float] = None,
    equity: Optional[float] = None,
    symbol: str = "BTCUSD",
) -> str:
    """
    Position Sizing and Risk:Reward ratio calculator for any given trade setup.
    """
    if entry <= 0 or sl <= 0 or entry == sl:
        return "❌ Invalid Entry or Stop Loss. Values must be positive numbers and entry != sl."

    direction = "LONG" if entry > sl else "SHORT"
    risk_dist = abs(entry - sl)
    risk_pct_dist = (risk_dist / entry) * 100.0

    eq = equity if equity and equity > 0 else 10000.0
    r_budget = risk_dollars if risk_dollars and risk_dollars > 0 else (eq * 0.01)

    is_btc = "BTC" in symbol.upper()
    decimals = 3 if is_btc else 2
    min_lot = 0.001 if is_btc else 0.01
    lots_calc = max(min_lot, round(r_budget / risk_dist, decimals))

    lines = [
        "📐 <b>POSITION SIZING & RISK:REWARD CALCULATOR</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"• <b>Asset</b>: <code>{symbol.upper()}</code>",
        f"• <b>Direction</b>: <b>{direction}</b> {'🟢' if direction == 'LONG' else '🔴'}",
        f"• <b>Entry Price</b>: <code>${entry:,.2f}</code>",
        f"• <b>Stop Loss</b>: <code>${sl:,.2f}</code>",
        f"• <b>Risk Distance</b>: <code>${risk_dist:,.2f}</code> ({risk_pct_dist:.2f}%)",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if tp is not None and tp > 0:
        reward_dist = (tp - entry) if direction == "LONG" else (entry - tp)
        if reward_dist > 0:
            rr = reward_dist / risk_dist
            profit_dollars = round(reward_dist * lots_calc, 2)
            lines.extend(
                [
                    f"• <b>Take Profit</b>: <code>${tp:,.2f}</code>",
                    f"• <b>Reward Distance</b>: <code>${reward_dist:,.2f}</code>",
                    f"• <b>Risk:Reward Ratio</b>: <b>{rr:.2f} : 1</b>",
                    f"• <b>Expected Profit</b>: 🟢 <b>+${profit_dollars:,.2f}</b>",
                    f"• <b>Max Risk / Loss</b>: 🔴 <b>-${r_budget:,.2f}</b>",
                    "━━━━━━━━━━━━━━━━━━━━━━",
                ]
            )
        else:
            lines.append(
                "⚠️ <i>Note: Provided Take Profit is on the losing side of entry.</i>\n━━━━━━━━━━━━━━━━━━━━━━"
            )
    else:
        tp1 = entry + (1.5 * risk_dist) if direction == "LONG" else entry - (1.5 * risk_dist)
        tp2 = entry + (2.5 * risk_dist) if direction == "LONG" else entry - (2.5 * risk_dist)
        tp3 = entry + (4.0 * risk_dist) if direction == "LONG" else entry - (4.0 * risk_dist)
        lines.extend(
            [
                "<b>Projected Targets:</b>",
                f"• <b>TP1 (1.5R)</b>: <code>${tp1:,.2f}</code> (+${1.5 * r_budget:,.2f})",
                f"• <b>TP2 (2.5R)</b>: <code>${tp2:,.2f}</code> (+${2.5 * r_budget:,.2f})",
                f"• <b>TP3 (4.0R)</b>: <code>${tp3:,.2f}</code> (+${4.0 * r_budget:,.2f})",
                "━━━━━━━━━━━━━━━━━━━━━━",
            ]
        )

    lines.extend(
        [
            "💼 <b>RECOMMENDED POSITION SIZING:</b>",
            f"• <b>Target Risk Budget</b>: <code>${r_budget:,.2f}</code>",
            f"• <b>Recommended Lot Size</b>: <b>{lots_calc} lots</b>",
            f"• <b>Risk 1.0% ($100 on $10k)</b>: <code>{max(min_lot, round((eq * 0.01) / risk_dist, decimals))} lots</code>",
            f"• <b>Risk 2.0% ($200 on $10k)</b>: <code>{max(min_lot, round((eq * 0.02) / risk_dist, decimals))} lots</code>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"<i>💡 Execute this setup directly: <code>/execute {lots_calc}</code></i>",
        ]
    )

    return "\n".join(lines)


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
    dd = float((np.maximum.accumulate(eq) - eq).max() / peak * 100) if peak > 0 else 0.0
    ret_pct = ((equity / args.equity - 1) * 100) if getattr(args, "equity", 0) > 0 else 0.0
    print("\n" + "=" * 64)
    print("   AI BOT LEARNING - DEMO BACKTEST REPORT")
    print("=" * 64)
    print(f"  trades   : {trades}")
    print(f"  winrate  : {wins/trades*100 if trades else 0:.1f}%")
    print(f"  net pnl  : {pnl_sum:.2f}")
    print(f"  return   : {ret_pct:.2f}%")
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
            if isinstance(df1.index, pd.DatetimeIndex):
                daily = (
                    df1.resample("1D")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                    .dropna()
                )
            else:
                daily = pd.DataFrame()

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


def get_feed(source: str = "delta", symbol: str = "XAUTUSD", exchange: str = "delta"):
    ai_mod = load_module("ai_strat", AI_FILE)
    return ai_mod.MarketDataFeed(symbol=symbol, source=source, exchange_id=exchange)


def run_backtest_itb(args):
    print("=" * 64)
    print("  MODE: BACKTEST - Intelligent Trading Bot (ITB Machine Learning)")
    print("=" * 64)
    feed = get_feed(args.source, args.symbol, args.exchange)
    data = feed.get_multi_tf(limit_1m=max(200, args.bars))
    df = (
        data.get("1m")
        if data and "1m" in data and not data["1m"].empty
        else feed.get_candles(limit=max(200, args.bars))
    )
    res = ITBBacktester.backtest(df, threshold=0.12)
    print(format_backtest_report(res, symbol=args.symbol))


# ==================================================================
# 4. MAIN CLI PARSER & RUNNER
# ==================================================================


def main():
    p = argparse.ArgumentParser(
        description="Auto Trade Runner - runs both strategy files"
    )
    p.add_argument(
        "mode",
        choices=["backtest", "backtest-ai", "backtest-itb", "live", "both"],
        help="backtest | backtest-ai | backtest-itb | live | both",
    )
    p.add_argument(
        "--strategy",
        choices=["ai", "pro", "itb"],
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
    elif args.mode == "backtest-itb":
        run_backtest_itb(args)
    elif args.mode == "live":
        run_live_pro(args) if args.strategy == "pro" else run_live_ai(args)
    elif args.mode == "both":
        run_backtest(args)
        print("\n[Runner] Backtest done. Starting AI live bot...\n")
        run_live_ai(args)


if __name__ == "__main__":
    main()
