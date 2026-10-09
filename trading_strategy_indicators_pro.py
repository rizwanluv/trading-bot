"""
TOP-10 PRO + ADVANCED INDICATORS STRATEGY
=========================================
Adds high-accuracy technical indicators on top of the professional composite:

INDICATORS ADDED:
1. RSI (14) + Divergence detection
2. EMA 9 / 21 / 50 structure
3. MACD (12,26,9) histogram + signal
4. ADX (14) trend strength filter
5. Bollinger Bands (20,2) squeeze + position
6. Stochastic (14,3,3)
7. VWAP (session) distance
8. Supertrend (ATR based)
9. CCI (20)
10. Volume Oscillator

All indicators are used as CONFIRMATION filters (not primary signals)
to increase accuracy while keeping the core liquidity + structure edge.

Educational only. Not financial advice.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
from datetime import datetime, time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# TYPES
# ============================================================

class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class SetupType(Enum):
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BOS_CONTINUATION = "bos_continuation"
    REJECTION = "rejection"
    INDICATOR_CONFLUENCE = "indicator_confluence"

class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    EXPANDING = "expanding"
    COMPRESSING = "compressing"


@dataclass
class AIPrediction:
    direction: Direction
    confidence: float
    expected_move: float
    horizon: int
    summary: str


@dataclass
class IndicatorSnapshot:
    rsi: float = 50.0
    rsi_div: int = 0          # +1 bullish div, -1 bearish div
    ema9: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    macd_hist: float = 0.0
    macd_signal: float = 0.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    bb_upper: float = 0.0
    bb_mid: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    vwap: float = 0.0
    supertrend_dir: int = 0   # 1 long, -1 short
    cci: float = 0.0
    vol_osc: float = 0.0
    score: float = 0.0        # combined indicator score (-5 to +5)


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
    indicators: Optional[IndicatorSnapshot] = None
    ai: Optional[AIPrediction] = None
    regime: Regime = Regime.RANGE
    timestamp: Optional[datetime] = None
    atr: float = 0.0
    rel_vol: float = 1.0

TradeSignalPro = TradeSignal


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


@dataclass
class TradeResult:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry: float
    exit: float
    pnl: float
    r: float
    setup: str
    quality: float


# ============================================================
# INDICATOR ENGINE
# ============================================================

class IndicatorEngine:
    """Pure pandas/numpy implementation of all key indicators."""

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        line = ema_fast - ema_slow
        sig = line.ewm(span=signal, adjust=False).mean()
        hist = line - sig
        return line, sig, hist

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm = np.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), minus_dm.abs(), 0.0)
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def bollinger(close: pd.Series, period=20, std=2.0):
        mid = close.rolling(period).mean()
        dev = close.rolling(period).std()
        upper = mid + std * dev
        lower = mid - std * dev
        width = (upper - lower) / (mid + 1e-10)
        return upper, mid, lower, width

    @staticmethod
    def stochastic(df: pd.DataFrame, k=14, d=3):
        low_min = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        stoch_d = stoch_k.rolling(d).mean()
        return stoch_k, stoch_d

    @staticmethod
    def vwap(df: pd.DataFrame):
        if "volume" not in df.columns:
            return df["close"]
        tp = (df["high"] + df["low"] + df["close"]) / 3
        return (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)

    @staticmethod
    def supertrend(df: pd.DataFrame, period=10, mult=3.0):
        atr = IndicatorEngine.atr_series(df, period)
        hl2 = (df["high"] + df["low"]) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        direction = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                if direction.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i-1]:
                    lower.iloc[i] = lower.iloc[i-1]
                if direction.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i-1]:
                    upper.iloc[i] = upper.iloc[i-1]
        return direction

    @staticmethod
    def atr_series(df: pd.DataFrame, period=14):
        if df.empty:
            return pd.Series(dtype=float)
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        fallback = (h - l).abs()
        tr = tr.fillna(fallback).fillna(1.0)
        atr_ewm = tr.ewm(alpha=1/period, min_periods=1).mean()
        return atr_ewm.bfill().fillna(1.0)

    @staticmethod
    def cci(df: pd.DataFrame, period=20):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (tp - sma) / (0.015 * mad + 1e-10)

    @staticmethod
    def volume_oscillator(volume: pd.Series, short=5, long=20):
        short_ma = volume.rolling(short).mean()
        long_ma = volume.rolling(long).mean()
        return 100 * (short_ma - long_ma) / (long_ma + 1e-10)

    @staticmethod
    def rsi_divergence(close: pd.Series, rsi: pd.Series, look=20) -> int:
        """Simple divergence: +1 bullish, -1 bearish, 0 none"""
        if len(close) < look + 5:
            return 0
        c = close.iloc[-look:]
        r = rsi.iloc[-look:]
        # Bullish divergence: price lower low, RSI higher low
        if c.iloc[-1] < c.iloc[-5] and r.iloc[-1] > r.iloc[-5] and r.iloc[-1] < 40:
            return 1
        # Bearish divergence
        if c.iloc[-1] > c.iloc[-5] and r.iloc[-1] < r.iloc[-5] and r.iloc[-1] > 60:
            return -1
        return 0

    def compute(self, df: pd.DataFrame) -> IndicatorSnapshot:
        if len(df) < 60:
            return IndicatorSnapshot()
        close = df["close"]
        snap = IndicatorSnapshot()

        # RSI
        rsi_s = self.rsi(close)
        snap.rsi = float(rsi_s.iloc[-1])
        snap.rsi_div = self.rsi_divergence(close, rsi_s)

        # EMAs
        snap.ema9 = float(self.ema(close, 9).iloc[-1])
        snap.ema21 = float(self.ema(close, 21).iloc[-1])
        snap.ema50 = float(self.ema(close, 50).iloc[-1])

        # MACD
        _, sig, hist = self.macd(close)
        snap.macd_hist = float(hist.iloc[-1])
        snap.macd_signal = float(sig.iloc[-1])

        # ADX
        adx_s, pdi, mdi = self.adx(df)
        snap.adx = float(adx_s.iloc[-1])
        snap.plus_di = float(pdi.iloc[-1])
        snap.minus_di = float(mdi.iloc[-1])

        # Bollinger
        up, mid, low, width = self.bollinger(close)
        snap.bb_upper = float(up.iloc[-1])
        snap.bb_mid = float(mid.iloc[-1])
        snap.bb_lower = float(low.iloc[-1])
        snap.bb_width = float(width.iloc[-1])

        # Stochastic
        k, d = self.stochastic(df)
        snap.stoch_k = float(k.iloc[-1])
        snap.stoch_d = float(d.iloc[-1])

        # VWAP
        snap.vwap = float(self.vwap(df).iloc[-1])

        # Supertrend
        st = self.supertrend(df)
        snap.supertrend_dir = int(st.iloc[-1])

        # CCI
        snap.cci = float(self.cci(df).iloc[-1])

        # Volume Oscillator
        if "volume" in df.columns:
            snap.vol_osc = float(self.volume_oscillator(df["volume"]).iloc[-1])

        # Combined score (-5 to +5)
        score = 0.0
        # RSI
        if snap.rsi < 32: score += 1.1
        elif snap.rsi > 68: score -= 1.1
        if snap.rsi_div == 1: score += 1.3
        elif snap.rsi_div == -1: score -= 1.3
        # EMA structure
        if snap.ema9 > snap.ema21 > snap.ema50: score += 1.2
        elif snap.ema9 < snap.ema21 < snap.ema50: score -= 1.2
        # MACD
        if snap.macd_hist > 0: score += 0.7
        else: score -= 0.7
        # ADX + DI
        if snap.adx > 22:
            if snap.plus_di > snap.minus_di: score += 0.9
            else: score -= 0.9
        # Stochastic
        if snap.stoch_k < 22 and snap.stoch_k > snap.stoch_d: score += 0.8
        elif snap.stoch_k > 78 and snap.stoch_k < snap.stoch_d: score -= 0.8
        # Supertrend
        score += snap.supertrend_dir * 0.9
        # CCI
        if snap.cci < -110: score += 0.7
        elif snap.cci > 110: score -= 0.7
        # Volume
        if snap.vol_osc > 15: score += 0.4
        elif snap.vol_osc < -15: score -= 0.3

        snap.score = round(max(min(score, 5.0), -5.0), 2)
        return snap


# ============================================================
# AI PREDICTOR
# ============================================================

class AIPredictor:
    def predict(self, df: pd.DataFrame, horizon: int = 22) -> AIPrediction:
        if len(df) < 60:
            return AIPrediction(Direction.FLAT, 0.0, 0.0, horizon, "No data")
        c = df["close"].values
        h, l, o = df["high"].values, df["low"].values, df["open"].values
        def mom(n): return (c[-1]-c[-n-1])/(c[-n-1] if c[-n-1] != 0 else 1.0) if len(c)>n else 0
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = np.mean(tr[-14:])
        c_last = c[-1] if c[-1] != 0 else 1.0
        slope = np.polyfit(np.arange(25), c[-25:], 1)[0] / c_last
        score = mom(5)*3.1 + mom(15)*2.2 + mom(30)*1.4 + slope*4.2
        raw = np.tanh(score*8.2)
        bull = 0.5 + raw*0.47
        bear = 1-bull
        if bull > 0.64: d, conf = Direction.LONG, bull
        elif bear > 0.64: d, conf = Direction.SHORT, bear
        else: d, conf = Direction.FLAT, max(bull, bear)
        exp = atr * np.sqrt(horizon/14) * (0.55 + conf*0.8)
        return AIPrediction(d, float(conf), float(exp), horizon, f"AI{horizon}c {d.name} {conf:.0%}")


# ============================================================
# MAIN STRATEGY
# ============================================================

class IndicatorsProStrategy:
    def __init__(self, **kw):
        self.risk_pct = kw.get("risk_per_trade", 0.18)
        self.max_daily_risk = kw.get("max_daily_risk", 1.1)
        self.max_trades = kw.get("max_trades_per_day", 4)
        self.max_losses = kw.get("max_consecutive_losses", 2)
        self.min_quality = kw.get("min_quality", 4.5)
        self.min_rr = kw.get("min_rr", 2.0)
        self.min_ind_score = kw.get("min_indicator_score", 1.8)  # absolute value
        self.use_ai = kw.get("use_ai", True)
        self.ai_conf = kw.get("ai_min_conf", 0.65)
        self.use_killzone = kw.get("use_killzone", True)
        self.use_adx_filter = kw.get("use_adx_filter", True)
        self.min_adx = kw.get("min_adx", 18)
        self.symbol = str(kw.get("symbol", "BTCUSD")).upper()
        self.is_crypto = any(c in self.symbol for c in ("BTC", "ETH", "SOL", "XRP"))
        self.tp_mode = kw.get("tp_mode", "rr")
        self.sl_mode = kw.get("sl_mode", "swing")
        self.tp_value = float(kw.get("tp_value", 2.0))
        self.sl_value = float(kw.get("sl_value", 1.0))
        self.lot_size = float(kw.get("lot_size", 0.01))
        self.lot_mode = kw.get("lot_mode", "fixed")

        self.levels = {}
        self.consec_loss = 0
        self.win_streak = 0
        self.daily_trades = 0
        self.daily_risk = 0.0
        self.last_day = None
        self.peak_eq = 10000.0
        self.ind = IndicatorEngine()
        self.ai = AIPredictor()

    def set_tp_sl(self, tp_val: float, sl_val: float, tp_mode: str = "rr", sl_mode: str = "swing") -> None:
        self.tp_value = float(tp_val)
        self.sl_value = float(sl_val)
        self.tp_mode = tp_mode
        self.sl_mode = sl_mode
        if tp_mode == "rr":
            self.min_rr = float(tp_val)

    def set_lot_size(self, lot_size: float, mode: str = "fixed") -> None:
        self.lot_size = max(0.0001, float(lot_size))
        self.lot_mode = mode

    def set_symbol(self, symbol: str) -> None:
        self.symbol = str(symbol).upper()
        self.is_crypto = any(c in self.symbol for c in ("BTC", "ETH", "SOL", "XRP"))

    def calculate_tp_sl(self, df, d: Direction, entry: float) -> Tuple[float, float, Optional[float], float]:
        atr = self.atr(df)
        if self.sl_mode == "pts":
            stop = entry - self.sl_value if d == Direction.LONG else entry + self.sl_value
        elif self.sl_mode == "pct":
            stop = entry * (1 - self.sl_value / 100.0) if d == Direction.LONG else entry * (1 + self.sl_value / 100.0)
        elif self.sl_mode == "atr":
            eff_atr = atr if atr > 0 else (entry * 0.005)
            stop = entry - eff_atr * self.sl_value if d == Direction.LONG else entry + eff_atr * self.sl_value
        else:
            raw_stop = self.swing_stop(df, d, entry)
            risk_raw = abs(entry - raw_stop)
            stop = (entry - risk_raw * self.sl_value) if d == Direction.LONG else (entry + risk_raw * self.sl_value)

        risk = abs(entry - stop)
        if risk <= 0:
            risk = entry * 0.002
            stop = entry - risk if d == Direction.LONG else entry + risk

        if self.tp_mode == "pts":
            tp1 = entry + self.tp_value if d == Direction.LONG else entry - self.tp_value
            tp2 = entry + self.tp_value * 1.5 if d == Direction.LONG else entry - self.tp_value * 1.5
        elif self.tp_mode == "pct":
            tp1 = entry * (1 + self.tp_value / 100.0) if d == Direction.LONG else entry * (1 - self.tp_value / 100.0)
            tp2 = entry * (1 + self.tp_value * 1.5 / 100.0) if d == Direction.LONG else entry * (1 - self.tp_value * 1.5 / 100.0)
        elif self.tp_mode == "atr":
            eff_atr = atr if atr > 0 else (entry * 0.005)
            tp1 = entry + eff_atr * self.tp_value if d == Direction.LONG else entry - eff_atr * self.tp_value
            tp2 = entry + eff_atr * self.tp_value * 1.5 if d == Direction.LONG else entry - eff_atr * self.tp_value * 1.5
        else:
            eff_rr = max(self.tp_value, self.min_rr)
            tp1 = entry + risk * eff_rr if d == Direction.LONG else entry - risk * eff_rr
            tp2 = entry + risk * (eff_rr * 1.5) if d == Direction.LONG else entry - risk * (eff_rr * 1.5)

        return stop, tp1, tp2, risk

    def atr(self, df, p=14):
        if len(df) < p+1: return 0.0
        h,l,c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        return float(np.mean(tr[-p:]))

    def rvol(self, df, n=20):
        if "volume" not in df.columns or len(df)<n+1: return 1.0
        return float(df["volume"].iloc[-1]/(df["volume"].iloc[-n-1:-1].mean()+1e-9))

    def regime(self, df):
        if len(df)<50: return Regime.RANGE
        c = df["close"].values
        c_last = c[-1] if (len(c) > 0 and c[-1] != 0) else 1.0
        slope = np.polyfit(np.arange(30), c[-30:], 1)[0] / c_last
        atrp = self.atr(df) / c_last
        if atrp > 0.013: return Regime.EXPANDING
        if atrp < 0.003: return Regime.COMPRESSING
        if slope > 0.0004: return Regime.TREND_UP
        if slope < -0.0004: return Regime.TREND_DOWN
        return Regime.RANGE

    def killzone(self, ts):
        if not self.use_killzone or self.is_crypto: return True
        t = ts.time() if hasattr(ts,"time") else ts
        if time(7,0)<=t<time(7,12) or time(12,30)<=t<time(12,42): return False
        return (time(7,12)<=t<=time(10,30)) or (time(12,42)<=t<=time(16,0)) or (time(8,0)<=t<=time(16,30))

    def session_ok(self, ts):
        if self.is_crypto: return True
        t = ts.time() if hasattr(ts,"time") else ts
        return (time(7,0)<=t<=time(16,45)) or (time(12,0)<=t<=time(21,0))

    def trend(self, df, lb=12):
        if df is None or len(df)<lb: return Direction.FLAT
        r = df.iloc[-lb:]
        g = (r["close"]>r["open"]).sum()
        rd= (r["close"]<r["open"]).sum()
        if g>=lb*0.65 and (r["low"].is_monotonic_increasing or r["high"].is_monotonic_increasing):
            return Direction.LONG
        if rd>=lb*0.65 and (r["low"].is_monotonic_decreasing or r["high"].is_monotonic_decreasing):
            return Direction.SHORT
        return Direction.FLAT

    def mark_daily(self, daily):
        if len(daily)<2: return
        p,c = daily.iloc[-2], daily.iloc[-1]
        self.levels = {"PDH":float(p["high"]),"PDL":float(p["low"]),"Open":float(c["open"])}

    def swing_stop(self, df, d, entry):
        atr = self.atr(df)
        if d == Direction.LONG:
            return float(df["low"].iloc[-10:].min()) - atr*0.12
        return float(df["high"].iloc[-10:].max()) + atr*0.12

    # ---------- Setups ----------
    def setup_sweep(self, df, ind: IndicatorSnapshot):
        if len(df)<3 or not self.levels: return None
        curr, prev = df.iloc[-1], df.iloc[-2]
        pdh, pdl = self.levels.get("PDH"), self.levels.get("PDL")
        if pdh and float(prev["high"])>pdh and float(curr["close"])<pdh and curr["close"]<curr["open"]:
            if ind.score > -0.5: return None  # need bearish indicators
            entry = float(curr["close"])
            stop, tp1, tp2, risk = self.calculate_tp_sl(df, Direction.SHORT, entry)
            r_mult = round(abs(entry - tp1) / risk, 2) if risk > 0 else self.min_rr
            return TradeSignal(Direction.SHORT, entry, stop, tp1, tp2,
                               setup=SetupType.LIQUIDITY_SWEEP, reason="Sweep PDH + Indicators", r_multiple=r_mult)
        if pdl and float(prev["low"])<pdl and float(curr["close"])>pdl and curr["close"]>curr["open"]:
            if ind.score < 0.5: return None
            entry = float(curr["close"])
            stop, tp1, tp2, risk = self.calculate_tp_sl(df, Direction.LONG, entry)
            r_mult = round(abs(tp1 - entry) / risk, 2) if risk > 0 else self.min_rr
            return TradeSignal(Direction.LONG, entry, stop, tp1, tp2,
                               setup=SetupType.LIQUIDITY_SWEEP, reason="Sweep PDL + Indicators", r_multiple=r_mult)
        return None

    def setup_bos(self, df, trend, ind: IndicatorSnapshot):
        if len(df)<12 or trend==Direction.FLAT: return None
        if trend==Direction.LONG and ind.score < 1.2: return None
        if trend==Direction.SHORT and ind.score > -1.2: return None
        highs = df["high"].iloc[-12:-1]
        lows = df["low"].iloc[-12:-1]
        last = df.iloc[-1]
        if trend==Direction.LONG and float(last["close"]) > float(highs.max()):
            entry = float(last["close"])
            stop, tp1, tp2, risk = self.calculate_tp_sl(df, Direction.LONG, entry)
            r_mult = round(abs(tp1 - entry) / risk, 2) if risk > 0 else self.min_rr
            return TradeSignal(Direction.LONG, entry, stop, tp1, tp2,
                               setup=SetupType.BOS_CONTINUATION, reason="BOS Long + Indicators", r_multiple=r_mult)
        if trend==Direction.SHORT and float(last["close"]) < float(lows.min()):
            entry = float(last["close"])
            stop, tp1, tp2, risk = self.calculate_tp_sl(df, Direction.SHORT, entry)
            r_mult = round(abs(entry - tp1) / risk, 2) if risk > 0 else self.min_rr
            return TradeSignal(Direction.SHORT, entry, stop, tp1, tp2,
                               setup=SetupType.BOS_CONTINUATION, reason="BOS Short + Indicators", r_multiple=r_mult)
        return None

    # ---------- Main ----------
    def generate_signal(self, lower, daily, d1=None, d5=None, d15=None, d1h=None) -> Optional[TradeSignal]:
        if len(lower) < 80: return None
        ts = lower.index[-1]
        day = ts.date() if hasattr(ts,"date") else ts
        if self.last_day != day:
            self.last_day = day
            self.daily_trades = 0
            self.daily_risk = 0.0

        if self.consec_loss >= self.max_losses: return None
        if self.daily_trades >= self.max_trades: return None
        if self.daily_risk >= self.max_daily_risk: return None
        if not self.session_ok(ts) or not self.killzone(ts): return None

        # Indicators
        ind = self.ind.compute(lower)
        if abs(ind.score) < self.min_ind_score:
            return None  # indicators not decisive enough

        if self.use_adx_filter and ind.adx < self.min_adx:
            return None  # no trend strength

        atr = self.atr(lower)
        rvol = self.rvol(lower)
        reg = self.regime(lower)
        if reg in (Regime.COMPRESSING, Regime.EXPANDING): return None

        ai = self.ai.predict(lower)
        self.mark_daily(daily)
        trend = self.trend(lower)

        def accept(sig: TradeSignal) -> Optional[TradeSignal]:
            # Indicator direction must agree
            if sig.direction == Direction.LONG and ind.score < 1.5: return None
            if sig.direction == Direction.SHORT and ind.score > -1.5: return None
            if self.use_ai and (ai.direction != sig.direction or ai.confidence < self.ai_conf):
                return None
            # Quality
            q = 2.0 + abs(ind.score)*0.6
            if ai.direction == sig.direction: q += ai.confidence*1.4
            if rvol >= 1.3: q += 0.5
            if (reg==Regime.TREND_UP and sig.direction==Direction.LONG) or \
               (reg==Regime.TREND_DOWN and sig.direction==Direction.SHORT): q += 0.7
            if ind.rsi_div != 0 and ((ind.rsi_div==1 and sig.direction==Direction.LONG) or
                                     (ind.rsi_div==-1 and sig.direction==Direction.SHORT)): q += 0.8
            if q < self.min_quality: return None

            # AI TP scaling
            if ai.expected_move > 0:
                risk = abs(sig.entry - sig.stop)
                ai_tp = sig.entry + ai.expected_move * (1 if sig.direction==Direction.LONG else -1)
                if abs(ai_tp - sig.entry) > risk * self.min_rr:
                    sig.tp2 = ai_tp

            sig.quality = round(q, 2)
            sig.indicators = ind
            sig.ai = ai
            sig.atr = atr
            sig.rel_vol = rvol
            sig.regime = reg
            sig.timestamp = ts
            sig.reason += f" | IndScore:{ind.score} RSI:{ind.rsi:.0f} ADX:{ind.adx:.0f} | {ai.summary} | Q:{q:.1f}"
            return sig

        for fn in [lambda: self.setup_sweep(lower, ind),
                   lambda: self.setup_bos(lower, trend, ind)]:
            sig = fn()
            if sig:
                res = accept(sig)
                if res: return res
        return None

    def manage(self, pos: Position, candle):
        h, l = float(candle["high"]), float(candle["low"])
        pos.highest = max(pos.highest or pos.entry, h)
        pos.lowest = min(pos.lowest or pos.entry, l)
        if pos.direction == Direction.LONG:
            if pos.partials==0 and h>=pos.tp1:
                pos.remaining *= 0.5
                pos.stop = pos.entry
                pos.partials = 1
                return pos.tp1, False
            if pos.partials==1 and pos.tp2 and h>=pos.tp2:
                pos.remaining *= 0.5
                pos.partials = 2
                return pos.tp2, False
            pos.stop = max(pos.stop, pos.highest - pos.risk*2.0)
            if l <= pos.stop: return pos.stop, True
        else:
            if pos.partials==0 and l<=pos.tp1:
                pos.remaining *= 0.5
                pos.stop = pos.entry
                pos.partials = 1
                return pos.tp1, False
            if pos.partials==1 and pos.tp2 and l<=pos.tp2:
                pos.remaining *= 0.5
                pos.partials = 2
                return pos.tp2, False
            pos.stop = min(pos.stop, pos.lowest + pos.risk*2.0)
            if h >= pos.stop: return pos.stop, True
        return 0.0, False

    def size(self, equity, entry, stop, atr=0):
        if getattr(self, "lot_mode", "fixed") == "fixed":
            return getattr(self, "lot_size", 0.01)
        risk_pct = self.risk_pct
        if equity < self.peak_eq * 0.95: risk_pct *= 0.6
        self.peak_eq = max(self.peak_eq, equity)
        base = equity * (risk_pct/100)
        if self.win_streak > 0: base *= (1.08 ** min(self.win_streak, 2))
        ru = abs(entry-stop)
        if ru==0: return 0.0
        sz = base / ru
        if atr>0 and ru > atr*1.35: sz *= 0.7
        return sz

    def update(self, pnl):
        if pnl > 0:
            self.consec_loss = 0
            self.win_streak += 1
        else:
            self.consec_loss += 1
            self.win_streak = 0
        self.daily_trades += 1
        self.daily_risk += self.risk_pct


# ============================================================
# BACKTESTER + LIVE
# ============================================================

class Backtester:
    def __init__(self, strategy, capital=10000, commission=0.0002, slippage=0.0001):
        self.s = strategy
        self.capital = capital
        self.comm = commission
        self.slip = slippage
        self.eq = capital
        self.trades = []
        self.curve = []
        self.pos = None

    def _slip(self, p, d, entry=True):
        s = p*self.slip
        return (p+s if entry else p-s) if d==Direction.LONG else (p-s if entry else p+s)

    def run(self, df1, df5, df15, df1h, daily, start=300):
        self.eq = self.capital
        self.trades = []
        self.curve = [self.capital]
        self.pos = None
        self.s.consec_loss = 0
        self.s.win_streak = 0
        self.s.peak_eq = self.capital
        for i in range(start, len(df1)-1):
            w1 = df1.iloc[:i+1]
            ts = df1.index[i]
            w5 = df5[df5.index<=ts]
            w15 = df15[df15.index<=ts]
            w1h = df1h[df1h.index<=ts]
            wd = daily[daily.index<=ts]
            candle = df1.iloc[i]
            if self.pos:
                ep, closed = self.s.manage(self.pos, candle)
                if ep > 0:
                    ep = self._slip(ep, self.pos.direction, False)
                    pnl = ((ep-self.pos.entry) if self.pos.direction==Direction.LONG else (self.pos.entry-ep)) * self.pos.remaining
                    pnl -= (self.pos.entry+ep)*self.pos.remaining*self.comm
                    self.eq += pnl
                    r = pnl/(self.pos.risk*self.pos.size) if self.pos.risk*self.pos.size else 0
                    self.trades.append(TradeResult(self.pos.entry_time or ts, ts, self.pos.direction.name,
                                                   self.pos.entry, ep, pnl, r, self.pos.setup, 0))
                    self.s.update(pnl)
                    if closed: self.pos = None
            if self.pos is None and len(wd)>=2:
                sig = self.s.generate_signal(w1, wd, w1, w5 if len(w5) else None,
                                             w15 if len(w15) else None, w1h if len(w1h) else None)
                if sig:
                    entry = self._slip(sig.entry, sig.direction, True)
                    sz = self.s.size(self.eq, entry, sig.stop, sig.atr)
                    if sz > 0:
                        self.pos = Position(sig.direction, entry, sig.stop, sz, sig.tp1, sig.tp2,
                                            sz, ts, sig.setup.value, abs(entry-sig.stop), entry, entry, 0)
            self.curve.append(self.eq)
        return self.stats()

    def stats(self):
        if not self.trades:
            return {"trades":0,"winrate":0,"pf":0,"return":0,"maxdd":0,"avgR":0,"equity":self.eq}
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p>0]
        losses = [p for p in pnls if p<=0]
        rs = [t.r for t in self.trades]
        eq = np.array(self.curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak-eq)/np.maximum(peak,1e-9)
        return {
            "trades": len(self.trades),
            "winrate": round(len(wins)/len(self.trades)*100,2),
            "pf": round(sum(wins)/(abs(sum(losses)) or 1),2),
            "return": round((self.eq/self.capital-1)*100,2),
            "maxdd": round(float(np.max(dd))*100,2),
            "avgR": round(float(np.mean(rs)),2),
            "equity": round(self.eq,2)
        }

    def report(self, s):
        print("\n"+"="*64)
        print("   INDICATORS PRO STRATEGY - BACKTEST REPORT")
        print("="*64)
        for k,v in s.items(): print(f"  {k:<10}: {v}")
        print("="*64)


def make_data(n=7000, symbol="BTCUSD"):
    np.random.seed(42)
    d = pd.date_range("2025-01-01", periods=n, freq="1min")
    is_btc = "BTC" in str(symbol).upper()
    base = 82000.0 if is_btc else 4135.0
    scale = 12.0 if is_btc else 0.4
    spread_min = 5.0 if is_btc else 0.15
    spread_max = 45.0 if is_btc else 2.6
    p = base + np.cumsum(np.random.randn(n) * scale)
    df1 = pd.DataFrame({
        "open": p,
        "high": p + np.random.uniform(spread_min, spread_max, n),
        "low": p - np.random.uniform(spread_min, spread_max, n),
        "close": p + np.random.randn(n) * (scale * 0.3),
        "volume": np.random.randint(40, 2000, n),
    }, index=d)
    df5 = df1.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    df15 = df1.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    df1h = df1.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    daily = df1.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return df1, df5, df15, df1h, daily


if __name__ == "__main__":
    print("Creating data...")
    dfs = make_data(6800)
    print("\n>>> INDICATORS PRO Backtest")
    strat = IndicatorsProStrategy(
        risk_per_trade=0.17,
        min_quality=4.6,
        min_indicator_score=2.0,
        ai_min_conf=0.66,
        min_rr=2.1,
        use_adx_filter=True,
        min_adx=20,
    )
    bt = Backtester(strat, 10000)
    stats = bt.run(*dfs)
    bt.report(stats)
    print("\nIndicators Pro file ready.")