"""
Intelligent Trading Bot (ITB) Engine
Integrated from https://github.com/asavinov/intelligent-trading-bot

Components:
1. Feature Engineering:
   - Rolling trend slope using closed-form linear regression convolution
   - Rolling statistical moments: mean, standard deviation, skewness, kurtosis
   - Relative differences, log returns, momentum, and high/low channel position
2. Machine Learning Predictive Scoring & Indicator Combiner:
   - Adaptive ML regression / classification predicting forward price trajectory
   - Buy score & Sell score combiner into a single normalized Intelligent Indicator in [-1.0, +1.0]
   - Exponential Moving Average (EMA) score smoothing and zone thresholding (BUY ZONE, SELL ZONE, NEUTRAL)
3. Simulated Trade Performance & Backtester:
   - Exact implementation of ITB simulated trade performance tracking long and short transactions
4. ITBStrategy:
   - Integrated strategy for AutoTrader execution based on the Intelligent Indicator
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot.itb_engine")


# =====================================================================
# 1. DATA MODELS & SNAPSHOTS
# =====================================================================

@dataclass
class ITBFeatureSnapshot:
    """Snapshot of statistical and engineered features for a market candle."""
    trend_slope_5: float = 0.0
    trend_slope_10: float = 0.0
    trend_slope_20: float = 0.0
    skewness_15: float = 0.0
    kurtosis_15: float = 0.0
    volatility_std_15: float = 0.0
    zscore_15: float = 0.0
    hl_ratio_15: float = 0.5
    momentum_5: float = 0.0
    momentum_15: float = 0.0
    log_return: float = 0.0


@dataclass
class ITBPredictionResult:
    """Result of Intelligent Trading Bot ML evaluation."""
    symbol: str
    price: float
    buy_score: float
    sell_score: float
    indicator: float  # [-1.0, +1.0]
    smoothed_indicator: float  # [-1.0, +1.0]
    zone: str  # "BUY ZONE", "SELL ZONE", "NEUTRAL"
    confidence: float  # [0.0, 1.0]
    direction: str  # "LONG", "SHORT", "FLAT"
    features: ITBFeatureSnapshot
    model_version: str = "ITB-Ridge-v2.5"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )


# =====================================================================
# 2. FEATURE ENGINEERING ENGINE
# =====================================================================

class ITBFeatureGenerator:
    """
    Computes derived statistical and technical features from raw OHLCV candles
    inspired by asavinov/intelligent-trading-bot/common/gen_features_rolling_agg.py.
    """

    @staticmethod
    def rolling_slope_weights(window: int) -> np.ndarray:
        """
        Calculates 1D convolution weights for rolling linear regression slope:
        slope = sum(weights * y)
        """
        w = max(2, int(window))
        weights = (np.arange(w) - (w - 1) / 2.0) / (w * (w**2 - 1) / 12.0)
        return weights

    @classmethod
    def compute_rolling_slope(cls, series: pd.Series, window: int) -> pd.Series:
        """Computes rolling linear regression slope normalized by price."""
        w = max(2, int(window))
        if len(series) < w:
            return pd.Series(0.0, index=series.index)
        weights = cls.rolling_slope_weights(w)
        # Vectorized convolution using numpy
        vals = series.to_numpy(dtype=float)
        # Fill NaNs with forward fill for stability
        vals = np.nan_to_num(vals, nan=vals[0] if len(vals) > 0 else 0.0)
        conv = np.convolve(vals, weights[::-1], mode="valid")
        # Pad beginning with zeros to match index length
        pad_len = len(series) - len(conv)
        padded = np.pad(conv, (pad_len, 0), mode="edge") if pad_len > 0 else conv
        denom = np.where(vals > 0, vals, 1.0)
        normalized = (padded / denom) * 100.0  # % per candle bar
        return pd.Series(normalized, index=series.index, dtype=float)

    @classmethod
    def generate_features(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generates full matrix of rolling statistical and technical features.
        Returns the enriched dataframe and list of feature column names.
        """
        if df.empty or len(df) < 5:
            return df.copy(), []

        out = df.copy()
        c = out["close"]
        h = out["high"] if "high" in out.columns else c
        l = out["low"] if "low" in out.columns else c
        v = out["volume"] if "volume" in out.columns else pd.Series(100.0, index=c.index)

        feature_cols: List[str] = []

        # 1. Rolling Linear Regression Slopes (Trend Strength)
        for w in (5, 10, 20):
            col_name = f"itb_slope_{w}"
            out[col_name] = cls.compute_rolling_slope(c, w)
            feature_cols.append(col_name)

        # 2. Rolling Statistical Moments (Mean, Std, Skewness, Kurtosis)
        mean_15 = c.rolling(15, min_periods=3).mean()
        std_15 = c.rolling(15, min_periods=3).std().fillna(0.0)
        out["itb_std_15"] = std_15 / np.where(c > 0, c, 1.0) * 100.0
        feature_cols.append("itb_std_15")

        out["itb_zscore_15"] = (c - mean_15) / np.where(std_15 > 1e-8, std_15, 1.0)
        out["itb_zscore_15"] = out["itb_zscore_15"].clip(-3.0, 3.0).fillna(0.0)
        feature_cols.append("itb_zscore_15")

        out["itb_skew_15"] = c.rolling(15, min_periods=5).skew().clip(-3.0, 3.0).fillna(0.0)
        feature_cols.append("itb_skew_15")

        out["itb_kurt_15"] = c.rolling(15, min_periods=5).kurt().clip(-3.0, 3.0).fillna(0.0)
        feature_cols.append("itb_kurt_15")

        # 3. Relative Differences & Momentum
        for w in (1, 5, 15):
            mom_col = f"itb_mom_{w}"
            shift_c = c.shift(w).bfill()
            out[mom_col] = ((c - shift_c) / np.where(shift_c > 0, shift_c, 1.0)) * 100.0
            feature_cols.append(mom_col)

        # Log Returns
        log_ret = np.log(np.maximum(c, 1e-8)) - np.log(np.maximum(c.shift(1).bfill(), 1e-8))
        out["itb_log_return"] = log_ret * 100.0
        feature_cols.append("itb_log_return")

        # 4. High/Low Channel Position (Oscillator)
        low_15 = l.rolling(15, min_periods=3).min()
        high_15 = h.rolling(15, min_periods=3).max()
        hl_span = np.where(high_15 - low_15 > 1e-8, high_15 - low_15, 1.0)
        out["itb_hl_ratio_15"] = ((c - low_15) / hl_span).clip(0.0, 1.0).fillna(0.5)
        feature_cols.append("itb_hl_ratio_15")

        # 5. Volume Momentum
        vol_mean = v.rolling(15, min_periods=3).mean()
        out["itb_vol_ratio"] = (v / np.where(vol_mean > 0, vol_mean, 1.0)).clip(0.1, 5.0).fillna(1.0)
        feature_cols.append("itb_vol_ratio")

        return out, feature_cols

    @classmethod
    def extract_latest_features(cls, df: pd.DataFrame) -> ITBFeatureSnapshot:
        """Extracts the latest row's statistical feature snapshot."""
        if df.empty:
            return ITBFeatureSnapshot()

        out, _ = cls.generate_features(df)
        last = out.iloc[-1]
        return ITBFeatureSnapshot(
            trend_slope_5=float(last.get("itb_slope_5", 0.0)),
            trend_slope_10=float(last.get("itb_slope_10", 0.0)),
            trend_slope_20=float(last.get("itb_slope_20", 0.0)),
            skewness_15=float(last.get("itb_skew_15", 0.0)),
            kurtosis_15=float(last.get("itb_kurt_15", 0.0)),
            volatility_std_15=float(last.get("itb_std_15", 0.0)),
            zscore_15=float(last.get("itb_zscore_15", 0.0)),
            hl_ratio_15=float(last.get("itb_hl_ratio_15", 0.5)),
            momentum_5=float(last.get("itb_mom_5", 0.0)),
            momentum_15=float(last.get("itb_mom_15", 0.0)),
            log_return=float(last.get("itb_log_return", 0.0)),
        )


# =====================================================================
# 3. MACHINE LEARNING PREDICTOR & INDICATOR COMBINER
# =====================================================================

class ITBPredictor:
    """
    Intelligent Trading Bot Machine Learning Engine.
    Predicts directional trend strength and produces normalized
    Intelligent Indicator score in [-1.0, +1.0].
    """

    DEFAULT_WEIGHTS = {
        "itb_slope_5": 2.40,
        "itb_slope_10": 1.85,
        "itb_slope_20": 1.30,
        "itb_mom_1": 1.50,
        "itb_mom_5": 2.20,
        "itb_mom_15": 1.40,
        "itb_log_return": 1.80,
        "itb_zscore_15": 0.95,
        "itb_hl_ratio_15": 1.10,
        "itb_skew_15": 0.45,
        "itb_kurt_15": -0.20,
        "itb_vol_ratio": 0.35,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(weights) if weights else dict(self.DEFAULT_WEIGHTS)
        self.bias: float = 0.0
        self.smoothing_history: List[float] = []
        self.is_trained: bool = False

    def train(
        self,
        df: pd.DataFrame,
        horizon: int = 10,
        threshold: float = 0.001,
        l2_reg: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Trains the predictive model on historic OHLCV candle series using
        regularized ridge regression on future price trajectory targets.
        """
        if len(df) < horizon + 20:
            return {"success": False, "error": f"Insufficient candles ({len(df)} < {horizon + 20})"}

        feat_df, cols = ITBFeatureGenerator.generate_features(df)
        if not cols:
            return {"success": False, "error": "No features generated"}

        # Target forward return over horizon: (close_{t+h} - close_t) / close_t
        c = feat_df["close"].to_numpy(dtype=float)
        h = int(horizon)
        target = np.zeros(len(c) - h)
        for i in range(len(target)):
            target[i] = (c[i + h] - c[i]) / max(c[i], 1e-8)

        # Scale target to roughly [-1.0, +1.0] range
        y = np.tanh(target / max(threshold, 1e-5))

        # Design matrix X
        X_df = feat_df[cols].iloc[:-h].fillna(0.0)
        X = X_df.to_numpy(dtype=float)

        # Add bias column
        N, D = X.shape
        X_bias = np.hstack([X, np.ones((N, 1))])

        # Regularized normal equation: (X^T X + lambda I)^(-1) X^T y
        reg_matrix = l2_reg * np.eye(D + 1)
        reg_matrix[-1, -1] = 0.0  # Do not penalize bias
        try:
            w = np.linalg.solve(X_bias.T @ X_bias + reg_matrix, X_bias.T @ y)
            for idx, col in enumerate(cols):
                self.weights[col] = float(w[idx])
            self.bias = float(w[-1])
            self.is_trained = True

            # Evaluation metrics
            y_pred = X_bias @ w
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))
            dir_acc = np.mean(np.sign(y) == np.sign(y_pred))

            mae = float(np.mean(np.abs(y - y_pred)))

            return {
                "success": True,
                "samples": N,
                "features_count": D,
                "r2_score": round(float(r2), 4),
                "directional_accuracy": round(float(dir_acc) * 100.0, 1),
                "bias": round(self.bias, 4),
                "weights": dict(self.weights),
                "mae": round(mae, 6),
            }
        except Exception as exc:
            logger.warning("ITB train optimization failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def predict(self, df: pd.DataFrame, symbol: str = "BTCUSD") -> ITBPredictionResult:
        """
        Computes forward directional probability, produces the Intelligent Indicator,
        smoothens signal, and identifies market zone.
        """
        if df.empty:
            curr = 0.0
            feat = ITBFeatureSnapshot()
            return ITBPredictionResult(
                symbol=symbol,
                price=curr,
                buy_score=0.5,
                sell_score=0.5,
                indicator=0.0,
                smoothed_indicator=0.0,
                zone="NEUTRAL",
                confidence=0.5,
                direction="FLAT",
                features=feat,
            )

        feat_df, cols = ITBFeatureGenerator.generate_features(df)
        curr = float(df["close"].iloc[-1])
        feat = ITBFeatureGenerator.extract_latest_features(df)

        last_row = feat_df.iloc[-1]
        raw_score = self.bias
        for col, weight in self.weights.items():
            if col in last_row:
                val = float(last_row[col])
                if not math.isnan(val):
                    raw_score += val * weight

        # Logistic squashing
        # buy_score in [0.0, 1.0], sell_score in [0.0, 1.0]
        prob_up = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, raw_score))))
        prob_down = 1.0 - prob_up

        # Combined Intelligent Indicator in [-1.0, +1.0]
        # (matching ITB's combine_scores: buy_score - sell_score)
        indicator = round(prob_up - prob_down, 4)

        # Apply EMA score smoothing (matching ITB's generate_smoothen_scores)
        self.smoothing_history.append(indicator)
        if len(self.smoothing_history) > 10:
            self.smoothing_history.pop(0)

        # Weighted exponential moving average
        weights_arr = np.exp(np.linspace(-1, 0, len(self.smoothing_history)))
        weights_arr /= weights_arr.sum()
        smoothed_indicator = round(float(np.dot(self.smoothing_history, weights_arr)), 4)

        # Identify Signal Zone
        if smoothed_indicator >= 0.10:
            zone = "BUY ZONE"
            direction = "LONG"
            confidence = prob_up
        elif smoothed_indicator <= -0.10:
            zone = "SELL ZONE"
            direction = "SHORT"
            confidence = prob_down
        else:
            zone = "NEUTRAL"
            direction = "FLAT"
            confidence = max(prob_up, prob_down)

        return ITBPredictionResult(
            symbol=symbol,
            price=curr,
            buy_score=round(prob_up, 4),
            sell_score=round(prob_down, 4),
            indicator=indicator,
            smoothed_indicator=smoothed_indicator,
            zone=zone,
            confidence=round(confidence, 4),
            direction=direction,
            features=feat,
        )


# =====================================================================
# 4. SIMULATED TRADE PERFORMANCE & BACKTESTING ENGINE
# =====================================================================

class ITBBacktester:
    """
    Backtesting simulator directly based on
    asavinov/intelligent-trading-bot/common/backtesting.py.
    """

    @staticmethod
    def simulate_trade_performance(
        df: pd.DataFrame,
        buy_signal_col: str,
        sell_signal_col: str,
        price_col: str = "close",
    ) -> Dict[str, Any]:
        """
        Simulates long and short trades sequentially through candles,
        tracking transactions, profit, win rate, and performance per trade.
        """
        if df.empty or len(df) < 5:
            return {"error": "Insufficient candles for simulation"}

        is_buy_mode = True

        long_profit = 0.0
        long_profit_percent = 0.0
        long_transactions = 0
        long_profitable = 0
        longs: List[Tuple[Any, float, float, float, float]] = []

        short_profit = 0.0
        short_profit_percent = 0.0
        short_transactions = 0
        short_profitable = 0
        shorts: List[Tuple[Any, float, float, float, float]] = []

        sub_df = df[[sell_signal_col, buy_signal_col, price_col]].copy()

        for index, sell_sig, buy_sig, price in sub_df.itertuples(name=None):
            if not price or math.isnan(price) or price <= 0:
                continue

            if is_buy_mode:
                if buy_sig:
                    previous_price = shorts[-1][2] if len(shorts) > 0 else 0.0
                    profit = (previous_price - price) if previous_price > 0 else 0.0
                    profit_pct = (100.0 * profit / previous_price) if previous_price > 0 else 0.0
                    if previous_price > 0:
                        short_profit += profit
                        short_profit_percent += profit_pct
                        short_transactions += 1
                        if profit > 0:
                            short_profitable += 1
                    shorts.append((index, previous_price, price, profit, profit_pct))
                    is_buy_mode = False
            else:
                if sell_sig:
                    previous_price = longs[-1][2] if len(longs) > 0 else 0.0
                    profit = (price - previous_price) if previous_price > 0 else 0.0
                    profit_pct = (100.0 * profit / previous_price) if previous_price > 0 else 0.0
                    if previous_price > 0:
                        long_profit += profit
                        long_profit_percent += profit_pct
                        long_transactions += 1
                        if profit > 0:
                            long_profitable += 1
                    longs.append((index, previous_price, price, profit, profit_pct))
                    is_buy_mode = True

        total_tx = long_transactions + short_transactions
        total_profit = long_profit + short_profit
        total_profit_pct = long_profit_percent + short_profit_percent
        total_profitable = long_profitable + short_profitable
        win_rate = (100.0 * total_profitable / total_tx) if total_tx > 0 else 0.0

        return {
            "total_transactions": total_tx,
            "total_profit": round(total_profit, 2),
            "total_profit_percent": round(total_profit_pct, 2),
            "win_rate": round(win_rate, 1),
            "profitable_trades": total_profitable,
            "profit_per_trade": round(total_profit / total_tx, 2) if total_tx > 0 else 0.0,
            "profit_pct_per_trade": round(total_profit_pct / total_tx, 2) if total_tx > 0 else 0.0,
            "long": {
                "transactions": long_transactions,
                "profit": round(long_profit, 2),
                "profit_percent": round(long_profit_percent, 2),
                "win_rate": round(100.0 * long_profitable / long_transactions, 1) if long_transactions > 0 else 0.0,
            },
            "short": {
                "transactions": short_transactions,
                "profit": round(short_profit, 2),
                "profit_percent": round(short_profit_percent, 2),
                "win_rate": round(100.0 * short_profitable / short_transactions, 1) if short_transactions > 0 else 0.0,
            },
        }

    @classmethod
    def run_backtest(
        cls,
        df: pd.DataFrame,
        predictor: Optional[ITBPredictor] = None,
        threshold: float = 0.12,
        price_col: str = "close",
    ) -> Dict[str, Any]:
        """
        Runs full ITB backtest pipeline on OHLCV series.
        Generates features, applies predictive score, binarizes signals,
        and simulates execution performance.
        """
        if df.empty or len(df) < 20:
            return {"error": "Insufficient candle history (minimum 20 candles needed)."}

        pred = predictor or ITBPredictor()
        feat_df, cols = ITBFeatureGenerator.generate_features(df)

        # Generate scores across all rows
        scores = []
        for idx in range(len(feat_df)):
            sub_df = feat_df.iloc[: idx + 1]
            p = pred.predict(sub_df)
            scores.append(p.smoothed_indicator)

        feat_df["itb_score"] = scores
        feat_df["itb_buy_sig"] = feat_df["itb_score"] >= threshold
        feat_df["itb_sell_sig"] = feat_df["itb_score"] <= -threshold

        perf = cls.simulate_trade_performance(
            feat_df,
            buy_signal_col="itb_buy_sig",
            sell_signal_col="itb_sell_sig",
            price_col=price_col,
        )
        perf["candles_evaluated"] = len(df)
        perf["threshold_used"] = threshold
        return perf

    backtest = run_backtest


# =====================================================================
# 5. ITB TRADING STRATEGY (AUTOTRADER COMPATIBLE)
# =====================================================================

class ITBStrategy:
    """
    Intelligent Trading Bot Strategy for automated trading.
    Matches the IndicatorsProStrategy interface used by AutoTrader.
    """

    def __init__(self, **kw):
        self.symbol = str(kw.get("symbol", "BTCUSD")).upper()
        self.lot_size = float(kw.get("lot_size", 0.01))
        self.lot_mode = kw.get("lot_mode", "fixed")
        self.tp_mode = kw.get("tp_mode", "rr")
        self.sl_mode = kw.get("sl_mode", "swing")
        self.tp_value = float(kw.get("tp_value", 2.0))
        self.sl_value = float(kw.get("sl_value", 1.0))
        self.risk_pct = float(kw.get("risk_per_trade", 1.0))
        self.min_threshold = float(kw.get("min_threshold", 0.14))

        self.predictor = ITBPredictor()
        self.total_pnl = 0.0

    def set_tp_sl(self, tp_val: float, sl_val: float, tp_mode: str = "rr", sl_mode: str = "swing") -> None:
        self.tp_value = float(tp_val)
        self.sl_value = float(sl_val)
        self.tp_mode = tp_mode
        self.sl_mode = sl_mode

    def set_lot_size(self, lot_size: float, mode: str = "fixed") -> None:
        self.lot_size = max(0.0001, float(lot_size))
        self.lot_mode = mode

    def set_symbol(self, symbol: str) -> None:
        self.symbol = str(symbol).upper()

    def update(self, pnl: float) -> None:
        self.total_pnl += pnl

    def size(self, equity: float, entry: float, stop: float, atr: float = 0.0) -> float:
        risk_dist = abs(entry - stop)
        if risk_dist <= 0:
            return self.lot_size
        if self.lot_mode == "fixed":
            return self.lot_size
        budget = equity * (self.risk_pct / 100.0)
        is_btc = "BTC" in self.symbol
        dec = 3 if is_btc else 2
        min_l = 0.001 if is_btc else 0.01
        return max(min_l, round(budget / risk_dist, dec))

    def generate_signal(self, df1: pd.DataFrame, daily: Optional[pd.DataFrame] = None) -> Any:
        """
        Generates trading signal from ITB Machine Learning Intelligent Indicator.
        Returns a signal object with direction, entry, stop, tp1, and reason.
        """
        if df1.empty or len(df1) < 15:
            return None

        pred = self.predictor.predict(df1, symbol=self.symbol)
        curr = pred.price

        # ATR calculation for risk stops
        c = df1["close"].to_numpy(dtype=float)
        h = df1["high"].to_numpy(dtype=float)
        l = df1["low"].to_numpy(dtype=float)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else (curr * 0.005)

        # Import Direction and SetupType from trading_strategy_indicators_pro
        try:
            from trading_strategy_indicators_pro import Direction, SetupType, TradeSignal
        except ImportError:
            return None

        thresh = float(self.min_threshold)
        if thresh <= 0:
            trigger_buy = pred.smoothed_indicator >= 0
            trigger_sell = not trigger_buy
        else:
            trigger_buy = pred.smoothed_indicator >= thresh
            trigger_sell = pred.smoothed_indicator <= -thresh

        if trigger_buy:
            # Bullish Buy Zone
            d = Direction.LONG
            risk_amt = max(atr * self.sl_value, curr * 0.003)
            stop = round(curr - risk_amt, 2)
            tp1 = round(curr + (risk_amt * self.tp_value), 2)
            tp2 = round(curr + (risk_amt * (self.tp_value + 1.2)), 2)
            reason = f"ITB ML Indicator +{pred.smoothed_indicator:.2f} (BUY ZONE 🟢)"
            return TradeSignal(
                direction=d,
                entry=curr,
                stop=stop,
                tp1=tp1,
                tp2=tp2,
                atr=atr,
                quality=round(pred.confidence * 10.0, 1),
                setup=SetupType.INDICATOR_CONFLUENCE,
                reason=reason,
            )

        elif trigger_sell:
            # Bearish Sell Zone
            d = Direction.SHORT
            risk_amt = max(atr * self.sl_value, curr * 0.003)
            stop = round(curr + risk_amt, 2)
            tp1 = round(curr - (risk_amt * self.tp_value), 2)
            tp2 = round(curr - (risk_amt * (self.tp_value + 1.2)), 2)
            reason = f"ITB ML Indicator {pred.smoothed_indicator:.2f} (SELL ZONE 🔴)"
            return TradeSignal(
                direction=d,
                entry=curr,
                stop=stop,
                tp1=tp1,
                tp2=tp2,
                atr=atr,
                quality=round(pred.confidence * 10.0, 1),
                setup=SetupType.INDICATOR_CONFLUENCE,
                reason=reason,
            )

        return None


# =====================================================================
# 6. REPORT FORMATTERS
# =====================================================================

def format_itb_card(
    pred: ITBPredictionResult,
    symbol: Optional[str] = None,
    current_price: Optional[float] = None,
) -> str:
    """Formats rich modern HTML card for Telegram /itb command."""
    from auto_trade import make_modern_meter

    sym_upper = (symbol or pred.symbol or "BTCUSD").upper()
    curr_price = float(current_price if current_price is not None else pred.price)
    if "BTC" in sym_upper:
        icon, asset_name = "⚡ ₿", "Bitcoin (BTC)"
    elif "XAU" in sym_upper:
        icon, asset_name = "🥇", "Gold (XAU)"
    elif "ETH" in sym_upper:
        icon, asset_name = "🔷", "Ethereum (ETH)"
    elif "SOL" in sym_upper:
        icon, asset_name = "🟣", "Solana (SOL)"
    elif "XRP" in sym_upper:
        icon, asset_name = "💧", "Ripple (XRP)"
    else:
        icon, asset_name = "📈", sym_upper

    # Meter bar for indicator [-1.0, +1.0] scaled to [0, 100%]
    meter_pct = max(0.0, min(100.0, (pred.smoothed_indicator + 1.0) / 2.0 * 100.0))
    meter_bar = make_modern_meter(meter_pct, width=10, fill_char="■", empty_char="░")

    if pred.zone == "BUY ZONE":
        zone_chip = "🟢 <b>BUY ZONE</b> ↑"
        arrow = "📈 〉〉〉"
    elif pred.zone == "SELL ZONE":
        zone_chip = "🔴 <b>SELL ZONE</b> ↓"
        arrow = "📉 〈〈〈"
    else:
        zone_chip = "⚪ <b>NEUTRAL / RANGE</b>"
        arrow = "⏸️ ───"

    score_sign = "+" if pred.smoothed_indicator >= 0 else ""

    lines = [
        f"{icon} <b>INTELLIGENT TRADING SIGNALS</b> [{zone_chip}]",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"• <b>Asset</b>: <code>{asset_name}</code>",
        f"• <b>Market Price</b>: <code>${curr_price:,.2f}</code>",
        f"• <b>Intelligent Indicator</b>: <b>{score_sign}{pred.smoothed_indicator:.2f}</b>",
        f"• <b>Signal Gauge</b>: <code>[{meter_bar}]</code> ({meter_pct:.0f}%)",
        f"• <b>ML Confidence</b>: <code>{pred.confidence * 100.0:.1f}%</code> (Buy: {pred.buy_score:.2f} | Sell: {pred.sell_score:.2f})",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "📊 <b>ENGINEERED FEATURE MOMENTS:</b>",
        f"• <b>Trend Slope (10m)</b>: <code>{pred.features.trend_slope_10:+.3f}%/bar</code>",
        f"• <b>Skewness (Asymmetry)</b>: <code>{pred.features.skewness_15:+.2f}</code>",
        f"• <b>Kurtosis (Tail Risk)</b>: <code>{pred.features.kurtosis_15:+.2f}</code>",
        f"• <b>Channel Position</b>: <code>{pred.features.hl_ratio_15 * 100.0:.1f}%</code>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"<i>{arrow} {pred.symbol} Indicator: {score_sign}{pred.smoothed_indicator:.2f} {pred.zone}</i>\n"
        f"<i>💡 Actions: /entry {sym_upper} | /itb backtest {sym_upper} | /buy | /sell</i>",
    ]
    return "\n".join(lines)


def format_backtest_report(results: Dict[str, Any], symbol: str = "BTCUSD") -> str:
    """Formats rich backtest performance report."""
    if "error" in results:
        return f"⚠️ <b>Backtest Error:</b> {results['error']}"

    tot_tx = results.get("total_transactions", 0)
    profit = results.get("total_profit", 0.0)
    profit_pct = results.get("total_profit_percent", 0.0)
    win_rate = results.get("win_rate", 0.0)
    ppt = results.get("profit_per_trade", 0.0)
    long_res = results.get("long", {})
    short_res = results.get("short", {})

    from auto_trade import make_modern_meter
    win_bar = make_modern_meter(win_rate, width=10, fill_char="█", empty_char="░")

    pnl_sign = "+" if profit >= 0 else ""
    pnl_emoji = "🟢" if profit >= 0 else "🔴"

    return (
        f"🔬 <b>ITB SIMULATED TRADE PERFORMANCE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Asset</b>: <code>{symbol.upper()}</code>\n"
        f"• <b>Candles Evaluated</b>: <code>{results.get('candles_evaluated', 0)}</code>\n"
        f"• <b>Signal Threshold</b>: <code>±{results.get('threshold_used', 0.12):.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Total Trades</b>: <code>{tot_tx}</code> (Profitable: {results.get('profitable_trades', 0)})\n"
        f"• <b>Win Rate</b>: <code>[{win_bar}]</code> <b>{win_rate:.1f}%</b>\n"
        f"• <b>Total Realized Profit</b>: {pnl_emoji} <b>{pnl_sign}${profit:,.2f}</b> ({pnl_sign}{profit_pct:.1f}%)\n"
        f"• <b>Profit / Trade</b>: <code>{pnl_sign}${ppt:,.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Long Trades</b>: {long_res.get('transactions', 0)} trades | Win: {long_res.get('win_rate', 0)}% | PnL: ${long_res.get('profit', 0):,.2f}\n"
        f"📉 <b>Short Trades</b>: {short_res.get('transactions', 0)} trades | Win: {short_res.get('win_rate', 0)}% | PnL: ${short_res.get('profit', 0):,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>💡 Run live automated ensemble: <code>/autotrade on</code></i>"
    )
