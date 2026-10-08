"""
AI BOT LEARNING – Improved Edition
==================================
Self-learning trading bot with permanent memory + real-time data.
Auto-starts live mode when you run the file.

Improvements:
- Multi-TF bias (1m / 5m / 15m / 1H)
- Extra setups: BOS continuation + Rejection candle
- Stronger confluence & quality gate
- Better risk (daily loss limit, cooldown, drawdown cut)
- Smarter permanent learning
- Robust live loop (reconnect, error recovery)
- Optional Telegram alerts
- Auto-start live mode

Memory: ai_bot_learning_memory.json (never forgets)
Educational only. Not financial advice.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
from datetime import datetime, time, timedelta, timezone
import pandas as pd
import numpy as np
import json
import os
import time as time_module
import traceback
import warnings
warnings.filterwarnings("ignore")

MEMORY_FILE = "ai_bot_learning_memory.json"


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class SetupType(Enum):
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BOS = "bos"
    REJECTION = "rejection"


@dataclass
class TradeRecord:
    id: str
    timestamp: str
    direction: str
    setup: str
    entry: float
    exit: float
    pnl: float
    r_multiple: float
    quality: float
    regime: str
    hour: int
    session: str
    rsi: float
    adx: float
    ind_score: float
    ai_conf: float
    ai_correct: bool
    rel_vol: float
    won: bool
    bars_held: int = 0
    trailing_used: float = 2.0


@dataclass
class MemoryStats:
    total_trades: int = 0
    total_wins: int = 0
    total_pnl: float = 0.0
    by_setup: Dict[str, Dict] = field(default_factory=dict)
    by_regime: Dict[str, Dict] = field(default_factory=dict)
    by_hour: Dict[str, Dict] = field(default_factory=dict)
    by_session: Dict[str, Dict] = field(default_factory=dict)
    by_direction: Dict[str, Dict] = field(default_factory=dict)
    disabled_setups: List[str] = field(default_factory=list)
    indicator_weights: Dict[str, float] = field(default_factory=dict)
    ai_weights: Dict[str, float] = field(default_factory=dict)
    ai_calibration: Dict[str, float] = field(default_factory=dict)
    best_quality: float = 4.2
    best_ai_conf: float = 0.64
    best_min_rr: float = 1.9
    best_risk_pct: float = 0.18
    best_trail_mult: float = 2.15
    expectancy: float = 0.0
    recent_winrate: float = 0.5
    recent_expectancy: float = 0.0
    max_dd_seen: float = 0.0
    last_updated: str = ""
    learning_steps: int = 0


@dataclass
class TradeSignal:
    direction: Direction
    entry: float
    stop: float
    tp1: float
    tp2: Optional[float] = None
    setup: SetupType = SetupType.LIQUIDITY_SWEEP
    reason: str = ""
    r_multiple: float = 1.9
    quality: float = 0.0
    regime: str = "range"
    session: str = "other"
    timestamp: Optional[datetime] = None
    atr: float = 0.0
    rel_vol: float = 1.0
    rsi: float = 50.0
    adx: float = 20.0
    ind_score: float = 0.0
    ai_conf: float = 0.5
    ai_dir: Direction = Direction.FLAT
    bias_aligned: int = 0


@dataclass
class Position:
    direction: Direction
    entry: float
    stop: float
    size: float
    tp1: float
    tp2: Optional[float]
    remaining: float
    entry_time: Optional[datetime] = None
    setup: str = ""
    risk: float = 0.0
    highest: float = 0.0
    lowest: float = 0.0
    partials: int = 0
    signal_meta: Dict = field(default_factory=dict)
    trail_mult: float = 2.15


class MarketDataFeed:
    def __init__(self, symbol="XAUUSD", source="demo", exchange_id="binance",
                 timeframe="1m", api_key="", api_secret=""):
        self.symbol = symbol
        self.source = source.lower()
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.api_key = api_key or os.getenv("EXCHANGE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("EXCHANGE_API_SECRET", "")
        self._exchange = None
        self._init()

    def _init(self):
        if self.source in ("ccxt", "binance"):
            try:
                import ccxt
                cls = getattr(ccxt, self.exchange_id if self.source == "ccxt" else "binance")
                self._exchange = cls({"apiKey": self.api_key, "secret": self.api_secret,
                                      "enableRateLimit": True, "options": {"defaultType": "future"}})
                print(f"[Data] CCXT → {self.exchange_id} | {self.symbol}")
            except Exception as e:
                print(f"[Data] CCXT failed ({e}) → demo")
                self.source = "demo"
        elif self.source == "yfinance":
            try:
                import yfinance as yf
                self._yf = yf
                print(f"[Data] yfinance | {self.symbol}")
            except ImportError:
                print("[Data] yfinance missing → demo")
                self.source = "demo"
        elif self.source == "mt5":
            try:
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    raise RuntimeError("MT5 init failed")
                self._mt5 = mt5
                print(f"[Data] MT5 | {self.symbol}")
            except Exception as e:
                print(f"[Data] MT5 failed ({e}) → demo")
                self.source = "demo"
        else:
            self.source = "demo"
            print("[Data] Demo feed")

    def fetch_ohlcv(self, limit=500, timeframe=None) -> pd.DataFrame:
        tf = timeframe or self.timeframe
        try:
            if self.source in ("ccxt", "binance") and self._exchange:
                sym = "XAU/USDT" if self.symbol == "XAUUSD" else self.symbol
                if "/" not in sym and sym.endswith("USD"):
                    sym = sym.replace("USD", "/USDT")
                raw = self._exchange.fetch_ohlcv(sym, timeframe=tf, limit=limit)
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
            if self.source == "yfinance":
                yf_sym = "GC=F" if self.symbol in ("XAUUSD", "GOLD", "XAU") else self.symbol
                if self.symbol == "EURUSD":
                    yf_sym = "EURUSD=X"
                t = self._yf.Ticker(yf_sym)
                period = "7d" if tf == "1m" else "60d"
                df = t.history(period=period, interval={"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"}.get(tf, "1m"))
                if df.empty:
                    return self._demo(limit)
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df = df[["open", "high", "low", "close", "volume"]].tail(limit)
                df.index = pd.to_datetime(df.index, utc=True)
                return df.astype(float)
            if self.source == "mt5" and hasattr(self, "_mt5"):
                mt5_tf = {"1m": 1, "5m": 5, "15m": 15, "1h": 16385}.get(tf, 1)
                rates = self._mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, limit)
                if rates is None:
                    return self._demo(limit)
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df = df.set_index("time").rename(columns={"tick_volume": "volume"})
                return df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as e:
            print(f"[Data] Fetch error: {e}")
        return self._demo(limit)

    def _demo(self, limit=500):
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq="1min")
        p = 2650 + np.cumsum(np.random.randn(limit) * 0.28)
        return pd.DataFrame({
            "open": p, "high": p + np.random.uniform(0.1, 1.6, limit),
            "low": p - np.random.uniform(0.1, 1.6, limit),
            "close": p + np.random.randn(limit) * 0.09,
            "volume": np.random.randint(40, 1400, limit)
        }, index=dates)

    def get_multi_tf(self, limit_1m=600) -> Dict[str, pd.DataFrame]:
        df1 = self.fetch_ohlcv(limit=limit_1m, timeframe="1m")
        if df1.empty:
            return {}
        out = {"1m": df1}
        for tf, rule in [("5m", "5min"), ("15m", "15min"), ("1h", "1h")]:
            try:
                out[tf] = df1.resample(rule).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna()
            except Exception:
                out[tf] = pd.DataFrame()
        try:
            out["daily"] = df1.resample("1D").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
        except Exception:
            out["daily"] = pd.DataFrame()
        return out


class AIBotMemory:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        self.trades: List[Dict] = []
        self.stats = MemoryStats()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.trades = data.get("trades", [])
                s = data.get("stats", {})
                self.stats = MemoryStats(**{k: s.get(k, getattr(MemoryStats(), k)) for k in MemoryStats.__dataclass_fields__})
                print(f"[AI Bot Learning] Memory loaded: {len(self.trades)} trades | steps {self.stats.learning_steps}")
            except Exception as e:
                print(f"[Memory] Load error ({e}) → fresh")
                self._defaults()
        else:
            self._defaults()

    def _defaults(self):
        self.stats.indicator_weights = {"rsi": 1.0, "adx": 1.0, "ema": 1.0, "vol": 1.0}
        self.stats.ai_weights = {"momentum": 1.0, "slope": 1.0, "structure": 1.0}
        self.stats.ai_calibration = {"0.55": 0.50, "0.60": 0.55, "0.65": 0.62, "0.70": 0.70, "0.75": 0.76}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"trades": self.trades[-8000:], "stats": asdict(self.stats)}, f, indent=2)

    def remember(self, rec: TradeRecord):
        self.trades.append(asdict(rec))
        self.stats.total_trades += 1
        if rec.won:
            self.stats.total_wins += 1
        self.stats.total_pnl += rec.pnl
        self.stats.learning_steps += 1

        for key, bucket in [
            (rec.setup, self.stats.by_setup), (rec.regime, self.stats.by_regime),
            (str(rec.hour), self.stats.by_hour), (rec.session, self.stats.by_session),
            (rec.direction, self.stats.by_direction)
        ]:
            if key not in bucket:
                bucket[key] = {"trades": 0, "wins": 0, "pnl": 0.0, "sum_r": 0.0, "avg_r": 0.0}
            b = bucket[key]
            b["trades"] += 1
            if rec.won:
                b["wins"] += 1
            b["pnl"] += rec.pnl
            b["sum_r"] += rec.r_multiple
            b["avg_r"] = b["sum_r"] / b["trades"]

        recent = self.trades[-40:]
        if recent:
            wts = [np.exp(-(len(recent) - i - 1) / 35) for i in range(len(recent))]
            ws = sum(wts) + 1e-9
            self.stats.recent_winrate = sum(w for t, w in zip(recent, wts) if t["won"]) / ws
            self.stats.recent_expectancy = sum(t["r_multiple"] * w for t, w in zip(recent, wts)) / ws

        if self.stats.total_trades > 0:
            wr = self.stats.total_wins / self.stats.total_trades
            aw = np.mean([t["r_multiple"] for t in self.trades if t["won"]]) if self.stats.total_wins else 1.5
            al = np.mean([abs(t["r_multiple"]) for t in self.trades if not t["won"]]) if (self.stats.total_trades - self.stats.total_wins) else 1.0
            self.stats.expectancy = wr * aw - (1 - wr) * al

        self.stats.last_updated = datetime.now().isoformat()
        self._improve(rec)
        self._save()
        print(f"[Learn] #{self.stats.total_trades} Exp:{self.stats.expectancy:.3f}R "
              f"RecentWR:{self.stats.recent_winrate:.0%} Q≥{self.stats.best_quality:.2f}")

    def _improve(self, rec: TradeRecord):
        if self.stats.total_trades >= 12:
            target = 5.0 if self.stats.recent_winrate > 0.58 else 3.7 if self.stats.recent_winrate < 0.42 else 4.3
            self.stats.best_quality += 0.045 * (target - self.stats.best_quality)

        bin_key = f"{round(rec.ai_conf * 20) / 20:.2f}"
        self.stats.ai_calibration[bin_key] = self.stats.ai_calibration.get(bin_key, 0.5) * 0.90 + (1.0 if rec.won else 0.0) * 0.10

        if rec.won and rec.ai_conf > 0.70:
            self.stats.best_ai_conf = min(0.82, self.stats.best_ai_conf + 0.005)
        elif not rec.won and rec.ai_conf < 0.58:
            self.stats.best_ai_conf = max(0.52, self.stats.best_ai_conf - 0.01)

        if rec.won and rec.bars_held > 10:
            self.stats.best_trail_mult = min(2.9, self.stats.best_trail_mult + 0.025)
        elif not rec.won and rec.bars_held < 5:
            self.stats.best_trail_mult = max(1.55, self.stats.best_trail_mult - 0.035)

        if rec.won:
            if rec.rsi < 32 or rec.rsi > 68:
                self.stats.indicator_weights["rsi"] = min(2.0, self.stats.indicator_weights.get("rsi", 1) + 0.03)
            if rec.adx > 25:
                self.stats.indicator_weights["adx"] = min(2.0, self.stats.indicator_weights.get("adx", 1) + 0.03)
            if abs(rec.ind_score) > 2.0:
                self.stats.indicator_weights["ema"] = min(1.8, self.stats.indicator_weights.get("ema", 1) + 0.025)
        else:
            if abs(rec.ind_score) < 0.7:
                self.stats.indicator_weights["ema"] = max(0.5, self.stats.indicator_weights.get("ema", 1) - 0.025)

        if rec.ai_correct:
            self.stats.ai_weights["momentum"] = min(1.7, self.stats.ai_weights.get("momentum", 1) + 0.02)
            self.stats.ai_weights["slope"] = min(1.7, self.stats.ai_weights.get("slope", 1) + 0.02)
        else:
            self.stats.ai_weights["momentum"] = max(0.65, self.stats.ai_weights.get("momentum", 1) - 0.025)

        for setup, data in list(self.stats.by_setup.items()):
            if data["trades"] >= 10:
                wr = data["wins"] / data["trades"]
                if wr < 0.35 and data["avg_r"] < 0.0:
                    if setup not in self.stats.disabled_setups:
                        self.stats.disabled_setups.append(setup)
                        print(f"[Learn] Disabled weak setup: {setup}")
                elif wr > 0.55 and setup in self.stats.disabled_setups:
                    self.stats.disabled_setups.remove(setup)

        if self.stats.recent_expectancy > 0.30:
            self.stats.best_risk_pct = min(0.30, self.stats.best_risk_pct + 0.005)
        elif self.stats.recent_expectancy < 0.05:
            self.stats.best_risk_pct = max(0.08, self.stats.best_risk_pct - 0.01)

        if self.stats.recent_winrate > 0.56:
            self.stats.best_min_rr = min(2.5, self.stats.best_min_rr + 0.025)
        elif self.stats.recent_winrate < 0.40:
            self.stats.best_min_rr = max(1.5, self.stats.best_min_rr - 0.04)

    def edge(self, bucket, key, min_n=5):
        d = bucket.get(key, {})
        return d.get("avg_r", 0.08) if d.get("trades", 0) >= min_n else 0.08

    def is_setup_allowed(self, s):
        return s not in self.stats.disabled_setups

    def calibrated_ai_conf(self, raw):
        return self.stats.ai_calibration.get(f"{round(raw*20)/20:.2f}", raw)

    def report(self):
        wr = self.stats.total_wins / self.stats.total_trades * 100 if self.stats.total_trades else 0
        return (f"Trades:{self.stats.total_trades} WR:{wr:.1f}% Exp:{self.stats.expectancy:.3f}R "
                f"Q≥{self.stats.best_quality:.2f} AI≥{self.stats.best_ai_conf:.2f} Trail×{self.stats.best_trail_mult:.2f}")


def calc_rsi(close, p=14):
    if len(close) < p + 2:
        return 50.0
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/p).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p).mean()
    return float((100 - 100 / (1 + g / (l + 1e-10))).iloc[-1])

def calc_adx(df, p=14):
    if len(df) < p + 5:
        return 20.0
    h, l, c = df["high"], df["low"], df["close"]
    plus = h.diff().clip(lower=0)
    minus = (-l.diff()).clip(lower=0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p).mean()
    pdi = 100 * plus.ewm(alpha=1/p).mean() / (atr + 1e-10)
    mdi = 100 * minus.ewm(alpha=1/p).mean() / (atr + 1e-10)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return float(dx.ewm(alpha=1/p).mean().iloc[-1])

def ema_score(close):
    if len(close) < 55:
        return 0.0
    e9 = close.ewm(span=9).mean().iloc[-1]
    e21 = close.ewm(span=21).mean().iloc[-1]
    e50 = close.ewm(span=50).mean().iloc[-1]
    if e9 > e21 > e50:
        return 1.7
    if e9 < e21 < e50:
        return -1.7
    return 0.0

def detect_trend(df, lb=10):
    if df is None or len(df) < lb:
        return Direction.FLAT
    r = df.iloc[-lb:]
    g = (r["close"] > r["open"]).sum()
    rd = (r["close"] < r["open"]).sum()
    if g >= lb * 0.65 and (r["low"].is_monotonic_increasing or r["high"].is_monotonic_increasing):
        return Direction.LONG
    if rd >= lb * 0.65 and (r["low"].is_monotonic_decreasing or r["high"].is_monotonic_decreasing):
        return Direction.SHORT
    return Direction.FLAT


class AdaptiveAI:
    def __init__(self, mem: AIBotMemory):
        self.mem = mem

    def predict(self, df, horizon=22):
        if len(df) < 60:
            return Direction.FLAT, 0.5, 0.0
        c, h, l = df["close"].values, df["high"].values, df["low"].values
        w = self.mem.stats.ai_weights
        def mom(n):
            return (c[-1] - c[-n-1]) / c[-n-1] if len(c) > n else 0
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = np.mean(tr[-14:])
        slope = np.polyfit(np.arange(25), c[-25:], 1)[0] / c[-1]
        hh = np.mean(h[-10:] > h[-11:-1]) if len(h) > 11 else 0.5
        ll = np.mean(l[-10:] < l[-11:-1]) if len(l) > 11 else 0.5
        score = (mom(5)*3.0*w.get("momentum",1) + mom(15)*2.2*w.get("momentum",1) +
                 slope*4.2*w.get("slope",1) + (hh-ll)*1.4*w.get("structure",1))
        raw = np.tanh(score * 8.5)
        bull = 0.5 + raw * 0.47
        bear = 1 - bull
        if bull > 0.64:
            d, conf = Direction.LONG, bull
        elif bear > 0.64:
            d, conf = Direction.SHORT, bear
        else:
            d, conf = Direction.FLAT, max(bull, bear)
        exp = atr * np.sqrt(horizon/14) * (0.55 + conf * 0.85)
        return d, float(conf), float(exp)


class AIBotLearning:
    def __init__(self, memory_path=MEMORY_FILE):
        self.mem = AIBotMemory(memory_path)
        self.ai = AdaptiveAI(self.mem)
        self.consec_loss = 0
        self.win_streak = 0
        self.daily_trades = 0
        self.daily_risk = 0.0
        self.last_day = None
        self.peak_eq = 10000.0
        self.cooldown_until = None
        print("[AI Bot Learning] " + self.mem.report())

    @property
    def min_q(self): return self.mem.stats.best_quality
    @property
    def min_ai(self): return self.mem.stats.best_ai_conf
    @property
    def min_rr(self): return self.mem.stats.best_min_rr
    @property
    def risk_pct(self): return self.mem.stats.best_risk_pct
    @property
    def trail_mult(self): return self.mem.stats.best_trail_mult

    def atr(self, df, p=14):
        if len(df) < p+1:
            return 0.0
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        return float(np.mean(tr[-p:]))

    def regime(self, df):
        if len(df) < 40:
            return "range"
        c = df["close"].values
        slope = np.polyfit(np.arange(30), c[-30:], 1)[0] / c[-1]
        atrp = self.atr(df) / c[-1]
        if atrp > 0.012:
            return "expanding"
        if atrp < 0.0035:
            return "compressing"
        if slope > 0.0004:
            return "trend_up"
        if slope < -0.0004:
            return "trend_down"
        return "range"

    def session_name(self, ts):
        t = ts.time() if hasattr(ts, "time") else ts
        if time(7, 0) <= t <= time(10, 30):
            return "london_open"
        if time(12, 30) <= t <= time(16, 0):
            return "ny_open"
        if time(8, 0) <= t <= time(16, 30):
            return "london"
        return "other"

    def session_ok(self, ts):
        t = ts.time() if hasattr(ts, "time") else ts
        return (time(7, 0) <= t <= time(16, 45)) or (time(12, 0) <= t <= time(21, 0))

    def multi_tf_bias(self, df1, df5, df15, df1h):
        votes = []
        for df, lb in [(df1, 12), (df5, 10), (df15, 8), (df1h, 6)]:
            if df is not None and len(df) >= lb:
                votes.append(detect_trend(df, lb))
        longs = sum(1 for v in votes if v == Direction.LONG)
        shorts = sum(1 for v in votes if v == Direction.SHORT)
        if longs >= 2 and longs > shorts:
            return Direction.LONG, longs
        if shorts >= 2 and shorts > longs:
            return Direction.SHORT, shorts
        return Direction.FLAT, 0

    def mark_daily(self, daily):
        if len(daily) < 2:
            return {}
        p, c = daily.iloc[-2], daily.iloc[-1]
        return {"PDH": float(p["high"]), "PDL": float(p["low"])}

    def candle_pattern(self, df):
        if len(df) < 3:
            return None, ""
        b, c = df.iloc[-2], df.iloc[-1]
        body = abs(c["close"] - c["open"])
        rng = c["high"] - c["low"] + 1e-9
        if b["close"] < b["open"] and c["close"] > c["open"] and c["close"] > b["open"] and c["open"] < b["close"]:
            return Direction.LONG, "BullEngulf"
        if b["close"] > b["open"] and c["close"] < c["open"] and c["close"] < b["open"] and c["open"] > b["close"]:
            return Direction.SHORT, "BearEngulf"
        if body / rng < 0.32:
            up = c["high"] - max(c["open"], c["close"])
            dn = min(c["open"], c["close"]) - c["low"]
            if dn > body * 2.2 and dn > up * 1.7:
                return Direction.LONG, "BullPin"
            if up > body * 2.2 and up > dn * 1.7:
                return Direction.SHORT, "BearPin"
        return None, ""

    def generate_signal(self, lower, daily, df5=None, df15=None, df1h=None):
        if len(lower) < 80:
            return None
        ts = lower.index[-1]

        if self.cooldown_until and ts < self.cooldown_until:
            return None

        day = ts.date() if hasattr(ts, "date") else ts
        if self.last_day != day:
            self.last_day = day
            self.daily_trades = 0
            self.daily_risk = 0.0

        if self.consec_loss >= 3 or self.daily_trades >= 5 or self.daily_risk >= 1.2:
            return None
        if not self.session_ok(ts):
            return None

        hour_edge = self.mem.edge(self.mem.stats.by_hour, str(getattr(ts, "hour", 12)))
        if hour_edge < -0.12:
            return None
        sess = self.session_name(ts)
        sess_edge = self.mem.edge(self.mem.stats.by_session, sess)
        if sess_edge < -0.10:
            return None

        atr = self.atr(lower)
        if atr == 0:
            return None
        reg = self.regime(lower)
        if reg in ("compressing", "expanding"):
            return None
        if self.mem.edge(self.mem.stats.by_regime, reg) < -0.08:
            return None

        bias_dir, bias_n = self.multi_tf_bias(lower, df5, df15, df1h)
        if bias_dir == Direction.FLAT:
            return None

        r = calc_rsi(lower["close"])
        a = calc_adx(lower)
        e = ema_score(lower["close"])
        w = self.mem.stats.indicator_weights
        ind = (e * w.get("ema", 1) +
               (1.4 if r < 30 else -1.4 if r > 70 else 0) * w.get("rsi", 1) +
               (1.1 if a > 25 else 0) * w.get("adx", 1))

        ai_dir, ai_raw, ai_exp = self.ai.predict(lower)
        ai_conf = self.mem.calibrated_ai_conf(ai_raw)

        levels = self.mark_daily(daily)
        if not levels:
            return None

        curr, prev = lower.iloc[-1], lower.iloc[-2]
        signal = None

        if levels.get("PDH") and float(prev["high"]) > levels["PDH"] and float(curr["close"]) < levels["PDH"] and curr["close"] < curr["open"]:
            if bias_dir == Direction.SHORT and ind < -0.8 and ai_dir == Direction.SHORT and self.mem.is_setup_allowed("liquidity_sweep"):
                entry = float(curr["close"])
                stop = float(prev["high"]) + atr * 0.12
                risk = stop - entry
                if risk > 0:
                    tp1 = entry - risk * max(self.min_rr, 2.0)
                    signal = TradeSignal(Direction.SHORT, entry, stop, tp1, entry - risk * 3.2,
                                         SetupType.LIQUIDITY_SWEEP, "Sweep PDH", max(self.min_rr, 2.0))
        elif levels.get("PDL") and float(prev["low"]) < levels["PDL"] and float(curr["close"]) > levels["PDL"] and curr["close"] > curr["open"]:
            if bias_dir == Direction.LONG and ind > 0.8 and ai_dir == Direction.LONG and self.mem.is_setup_allowed("liquidity_sweep"):
                entry = float(curr["close"])
                stop = float(prev["low"]) - atr * 0.12
                risk = entry - stop
                if risk > 0:
                    tp1 = entry + risk * max(self.min_rr, 2.0)
                    signal = TradeSignal(Direction.LONG, entry, stop, tp1, entry + risk * 3.2,
                                         SetupType.LIQUIDITY_SWEEP, "Sweep PDL", max(self.min_rr, 2.0))

        if signal is None and len(lower) >= 14 and self.mem.is_setup_allowed("bos"):
            highs = lower["high"].iloc[-12:-1]
            lows = lower["low"].iloc[-12:-1]
            if bias_dir == Direction.LONG and float(curr["close"]) > float(highs.max()) and ind > 1.0 and ai_dir == Direction.LONG:
                entry = float(curr["close"])
                stop = float(lows.min()) - atr * 0.1
                risk = entry - stop
                if risk > 0:
                    tp1 = entry + risk * max(self.min_rr, 1.9)
                    signal = TradeSignal(Direction.LONG, entry, stop, tp1, entry + risk * 3.0,
                                         SetupType.BOS, "BOS Long", max(self.min_rr, 1.9))
            elif bias_dir == Direction.SHORT and float(curr["close"]) < float(lows.min()) and ind < -1.0 and ai_dir == Direction.SHORT:
                entry = float(curr["close"])
                stop = float(highs.max()) + atr * 0.1
                risk = stop - entry
                if risk > 0:
                    tp1 = entry - risk * max(self.min_rr, 1.9)
                    signal = TradeSignal(Direction.SHORT, entry, stop, tp1, entry - risk * 3.0,
                                         SetupType.BOS, "BOS Short", max(self.min_rr, 1.9))

        if signal is None and self.mem.is_setup_allowed("rejection"):
            pat_dir, pat_name = self.candle_pattern(lower)
            if pat_dir is not None and pat_dir == bias_dir and ai_dir == bias_dir:
                entry = float(curr["close"])
                if pat_dir == Direction.LONG:
                    stop = float(curr["low"]) - atr * 0.08
                    risk = entry - stop
                    if risk > 0 and ind > 0.5:
                        tp1 = entry + risk * max(self.min_rr, 1.8)
                        signal = TradeSignal(Direction.LONG, entry, stop, tp1, entry + risk * 2.8,
                                             SetupType.REJECTION, f"Rejection {pat_name}", max(self.min_rr, 1.8))
                else:
                    stop = float(curr["high"]) + atr * 0.08
                    risk = stop - entry
                    if risk > 0 and ind < -0.5:
                        tp1 = entry - risk * max(self.min_rr, 1.8)
                        signal = TradeSignal(Direction.SHORT, entry, stop, tp1, entry - risk * 2.8,
                                             SetupType.REJECTION, f"Rejection {pat_name}", max(self.min_rr, 1.8))

        if signal is None:
            return None

        setup_edge = self.mem.edge(self.mem.stats.by_setup, signal.setup.value)
        quality = (2.8 + abs(ind) * 0.55 + ai_conf * 1.5 + setup_edge * 0.9 +
                   sess_edge * 0.45 + hour_edge * 0.35 + bias_n * 0.25)

        if quality < self.min_q or ai_conf < self.min_ai:
            return None

        if ai_exp > 0:
            risk = abs(signal.entry - signal.stop)
            ai_tp = signal.entry + ai_exp * (1 if signal.direction == Direction.LONG else -1)
            if abs(ai_tp - signal.entry) > risk * self.min_rr:
                signal.tp2 = ai_tp

        signal.quality = round(quality, 2)
        signal.regime = reg
        signal.session = sess
        signal.timestamp = ts
        signal.atr = atr
        signal.rsi = r
        signal.adx = a
        signal.ind_score = ind
        signal.ai_conf = ai_conf
        signal.ai_dir = ai_dir
        signal.bias_aligned = bias_n
        signal.reason += f" | Q:{quality:.1f} AI:{ai_conf:.0%} Bias:{bias_n}TF Ind:{ind:.1f} Exp:{self.mem.stats.expectancy:.2f}R"
        return signal

    def manage(self, pos, candle):
        h, l = float(candle["high"]), float(candle["low"])
        pos.highest = max(pos.highest or pos.entry, h)
        pos.lowest = min(pos.lowest or pos.entry, l)
        mult = pos.trail_mult
        if pos.direction == Direction.LONG:
            if pos.partials == 0 and h >= pos.tp1:
                pos.remaining *= 0.5
                pos.stop = pos.entry
                pos.partials = 1
                return pos.tp1, False
            if pos.partials == 1 and pos.tp2 and h >= pos.tp2:
                pos.remaining *= 0.5
                pos.partials = 2
                return pos.tp2, False
            pos.stop = max(pos.stop, pos.highest - pos.risk * mult)
            if l <= pos.stop:
                return pos.stop, True
        else:
            if pos.partials == 0 and l <= pos.tp1:
                pos.remaining *= 0.5
                pos.stop = pos.entry
                pos.partials = 1
                return pos.tp1, False
            if pos.partials == 1 and pos.tp2 and l <= pos.tp2:
                pos.remaining *= 0.5
                pos.partials = 2
                return pos.tp2, False
            pos.stop = min(pos.stop, pos.lowest + pos.risk * mult)
            if h >= pos.stop:
                return pos.stop, True
        return 0.0, False

    def size(self, equity, entry, stop, atr=0):
        risk_pct = self.risk_pct
        if equity < self.peak_eq * 0.93:
            risk_pct *= 0.5
        if equity < self.peak_eq * 0.87:
            risk_pct *= 0.35
        self.peak_eq = max(self.peak_eq, equity)
        if self.mem.stats.recent_expectancy > 0.28:
            risk_pct *= 1.1
        base = equity * (risk_pct / 100)
        if self.win_streak > 0:
            base *= (1.07 ** min(self.win_streak, 2))
        ru = abs(entry - stop)
        if ru == 0:
            return 0.0
        sz = base / ru
        if atr > 0 and ru > atr * 1.3:
            sz *= 0.65
        return sz

    def on_trade_closed(self, pos, exit_price, pnl, bars=0):
        won = pnl > 0
        r = pnl / (pos.risk * pos.size) if pos.risk * pos.size else 0.0
        meta = pos.signal_meta
        ai_correct = meta.get("ai_dir") == pos.direction.name

        rec = TradeRecord(
            id=f"T{self.mem.stats.total_trades+1}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            direction=pos.direction.name,
            setup=pos.setup,
            entry=pos.entry,
            exit=exit_price,
            pnl=pnl,
            r_multiple=r,
            quality=meta.get("quality", 0),
            regime=meta.get("regime", "range"),
            hour=pos.entry_time.hour if pos.entry_time else 12,
            session=meta.get("session", "other"),
            rsi=meta.get("rsi", 50),
            adx=meta.get("adx", 20),
            ind_score=meta.get("ind_score", 0),
            ai_conf=meta.get("ai_conf", 0.5),
            ai_correct=ai_correct,
            rel_vol=meta.get("rel_vol", 1.0),
            won=won,
            bars_held=bars,
            trailing_used=pos.trail_mult
        )
        self.mem.remember(rec)

        if won:
            self.consec_loss = 0
            self.win_streak += 1
        else:
            self.consec_loss += 1
            self.win_streak = 0
            if self.consec_loss >= 3:
                self.cooldown_until = (pos.entry_time or datetime.now()) + timedelta(minutes=75)
                print("[Safety] 3 losses → 75 min cooldown")
        self.daily_trades += 1
        self.daily_risk += self.risk_pct


def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        print(f"[Telegram] {e}")


class LiveAIBot:
    def __init__(self, strategy, symbol="XAUUSD", data_source="demo",
                 exchange_id="binance", poll_seconds=30, equity=10000.0):
        self.strategy = strategy
        self.symbol = symbol
        self.poll_seconds = poll_seconds
        self.equity = equity
        self.feed = MarketDataFeed(symbol=symbol, source=data_source, exchange_id=exchange_id)
        self.position = None
        self.running = False
        self.errors = 0

    def on_signal(self, sig):
        msg = (f"AI BOT LEARNING SIGNAL\n{sig.direction.name} @ {sig.entry:.2f}\n"
               f"SL {sig.stop:.2f} | TP1 {sig.tp1:.2f}\nQ:{sig.quality:.1f} | {sig.reason}")
        print(f"\n{'='*55}\n  {msg}\n{'='*55}")
        send_telegram(msg)
        size = self.strategy.size(self.equity, sig.entry, sig.stop, sig.atr)
        if size > 0:
            self.position = Position(
                sig.direction, sig.entry, sig.stop, size, sig.tp1, sig.tp2,
                size, sig.timestamp, sig.setup.value, abs(sig.entry - sig.stop),
                sig.entry, sig.entry, 0,
                signal_meta={
                    "quality": sig.quality, "regime": sig.regime, "session": sig.session,
                    "rsi": sig.rsi, "adx": sig.adx, "ind_score": sig.ind_score,
                    "ai_conf": sig.ai_conf, "ai_dir": sig.ai_dir.name, "rel_vol": sig.rel_vol
                },
                trail_mult=self.strategy.trail_mult
            )
            print(f"  Position opened | Size {size:.4f}")

    def run(self):
        print(f"\n[Live] AI Bot Learning | {self.symbol} | {self.feed.source} | poll {self.poll_seconds}s")
        print("[Live] Auto-running. Ctrl+C to stop.\n")
        self.running = True
        while self.running:
            try:
                data = self.feed.get_multi_tf(limit_1m=700)
                if not data or "1m" not in data or data["1m"].empty:
                    print("[Live] No data, retry...")
                    time_module.sleep(self.poll_seconds)
                    continue
                df1 = data["1m"]
                df5 = data.get("5m")
                df15 = data.get("15m")
                df1h = data.get("1h")
                daily = data.get("daily", pd.DataFrame())
                if len(daily) < 2:
                    daily = df1.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()

                if self.position is not None:
                    candle = df1.iloc[-1]
                    exit_p, closed = self.strategy.manage(self.position, candle)
                    if exit_p > 0:
                        pnl = ((exit_p - self.position.entry) if self.position.direction == Direction.LONG
                               else (self.position.entry - exit_p)) * self.position.remaining
                        self.equity += pnl
                        self.strategy.on_trade_closed(self.position, exit_p, pnl)
                        send_telegram(f"Closed {self.position.direction.name} | PnL {pnl:.2f} | Eq {self.equity:.2f}")
                        print(f"[Live] Closed | PnL {pnl:.2f} | Equity {self.equity:.2f}")
                        if closed:
                            self.position = None

                if self.position is None and len(daily) >= 2:
                    sig = self.strategy.generate_signal(df1, daily, df5, df15, df1h)
                    if sig:
                        self.on_signal(sig)

                self.errors = 0
                time_module.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                print("\n[Live] Stopped")
                self.running = False
            except Exception as e:
                self.errors += 1
                print(f"[Live] Error ({self.errors}): {e}")
                traceback.print_exc()
                time_module.sleep(min(120, 30 * self.errors))


def auto_start_live():
    print("=" * 60)
    print("  AI BOT LEARNING – AUTO LIVE MODE")
    print("=" * 60)
    SYMBOL = os.getenv("BOT_SYMBOL", "XAUUSD")
    DATA_SOURCE = os.getenv("BOT_DATA_SOURCE", "demo")
    EXCHANGE_ID = os.getenv("BOT_EXCHANGE", "binance")
    POLL = int(os.getenv("BOT_POLL", "30"))
    EQUITY = float(os.getenv("BOT_EQUITY", "10000"))
    strategy = AIBotLearning()
    print(f"\n[Auto] {SYMBOL} | {DATA_SOURCE} | poll {POLL}s | equity {EQUITY}")
    print(f"[Auto] Memory → {MEMORY_FILE}")
    print("[Auto] Starting live now...\n")
    live = LiveAIBot(strategy, SYMBOL, DATA_SOURCE, EXCHANGE_ID, POLL, EQUITY)
    live.run()


if __name__ == "__main__":
    auto_start_live()
