#!/usr/bin/env python3
"""
AUTO TRADE RUNNER
=================
Single entry point that runs BOTH uploaded strategy files:

  1. ai_bot_learning.py                 -> self-learning bot (permanent memory, live-ready)
  2. trading_strategy_indicators_pro.py -> 10-indicator confirmation strategy

MODES
-----
  backtest               Backtest the Indicators-Pro strategy on demo data
  backtest-ai            Quick learning smoke-test of the AI bot (demo data, no memory write)
  live --strategy ai     Live/paper trading with the AI Learning bot
  live --strategy pro    Live/paper trading with the Indicators-Pro strategy
  both                   Backtest first, then start AI live bot

EXAMPLES
--------
  python auto_trade_runner.py backtest --bars 5000
  python auto_trade_runner.py live --strategy pro --symbol XAUUSD --source demo
  python auto_trade_runner.py live --strategy ai --source binance --poll 30
  python auto_trade_runner.py both --bars 3000

ENV VARS (same as original files)
---------------------------------
  BOT_SYMBOL, BOT_DATA_SOURCE, BOT_EXCHANGE, BOT_POLL, BOT_EQUITY,
  EXCHANGE_API_KEY, EXCHANGE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

NOTE: keep this file in the SAME folder as the two strategy files.
Educational only. Not financial advice.
"""

import argparse
import importlib.util
import os
import sys
import time as time_module
import traceback
from datetime import datetime

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AI_FILE = os.path.join(BASE, "ai_bot_learning.py")
PRO_FILE = os.path.join(BASE, "trading_strategy_indicators_pro.py")


# ------------------------------------------------------------------
# Module loading (no changes needed to your original files)
# ------------------------------------------------------------------
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------
# MODE 1 : BACKTEST (Indicators-Pro)
# ------------------------------------------------------------------
def run_backtest(args):
    print("=" * 64)
    print("  MODE: BACKTEST  -  trading_strategy_indicators_pro.py")
    print("=" * 64)
    pro = load_module("pro_strat", PRO_FILE)

    dfs = pro.make_data(args.bars)
    strat = pro.IndicatorsProStrategy(
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


# ------------------------------------------------------------------
# MODE 2 : BACKTEST-AI (smoke test of the learning bot, demo data)
# ------------------------------------------------------------------
def run_backtest_ai(args):
    print("=" * 64)
    print("  MODE: BACKTEST-AI  -  ai_bot_learning.py (demo data, temp memory)")
    print("=" * 64)
    ai = load_module("ai_strat", AI_FILE)
    import tempfile

    tmp_mem = os.path.join(tempfile.gettempdir(), "ai_bot_smoke_memory.json")
    if os.path.exists(tmp_mem):
        os.remove(tmp_mem)

    bot = ai.AIBotLearning(memory_path=tmp_mem)
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
    dfs = pro.make_data(args.bars)

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

    eq = __import__("numpy").array(curve)
    dd = float(
        (__import__("numpy").maximum.accumulate(eq) - eq).max() / peak * 100
    )
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


# ------------------------------------------------------------------
# MODE 3 : LIVE - AI LEARNING BOT  (uses the file's own LiveAIBot)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# MODE 4 : LIVE - INDICATORS PRO  (adds a live loop to the pro file)
# ------------------------------------------------------------------
class LiveProBot:
    def __init__(self, pro, ai_mod, args):
        self.pro = pro
        self.ai_mod = ai_mod
        self.args = args
        self.strat = pro.IndicatorsProStrategy(
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


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
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
            sys.exit(f"[Runner] Missing file: {f}  (keep all 3 files in one folder)")

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
