"""
Automated test suite verifying price alerts, level analysis, and multi-timeframe candle entry analysis.
"""
import os
import unittest
from indicators import TechnicalAnalysis
from market_data import (
    get_ticker,
    get_level_analysis,
    format_level_analysis_message,
    get_candle_entry,
    get_multi_timeframe_entry,
    format_entry_analysis_message,
)
from alerts_manager import AlertManager
from main import _parse_alert_args, _parse_watch_args


class TestTechnicalAnalysis(unittest.TestCase):
    def test_rsi(self):
        # 14 flat closes
        closes = [100.0] * 20
        rsi = TechnicalAnalysis.calc_rsi(closes)
        self.assertEqual(rsi, 50.0)

        # Monotonically increasing closes (strong bull)
        closes = [float(100 + i * 2) for i in range(25)]
        rsi = TechnicalAnalysis.calc_rsi(closes)
        self.assertGreater(rsi, 70.0)

        # Monotonically decreasing closes (strong bear)
        closes = [float(100 - i * 2) for i in range(25)]
        rsi = TechnicalAnalysis.calc_rsi(closes)
        self.assertLess(rsi, 30.0)

    def test_pivots(self):
        high, low, close = 4200.0, 4100.0, 4150.0
        pivots = TechnicalAnalysis.calc_pivot_points(high, low, close)
        expected_p = (4200 + 4100 + 4150) / 3.0
        self.assertAlmostEqual(pivots["pivot"], round(expected_p, 2))
        self.assertGreater(pivots["r1"], pivots["pivot"])
        self.assertLess(pivots["s1"], pivots["pivot"])
        self.assertGreater(pivots["r2"], pivots["r1"])
        self.assertLess(pivots["s2"], pivots["s1"])

    def test_fib_retracements(self):
        fibs = TechnicalAnalysis.calc_fib_retracements(100.0, 0.0)
        self.assertEqual(fibs["100.0%"], 100.0)
        self.assertEqual(fibs["50.0%"], 50.0)
        self.assertEqual(fibs["61.8%"], 61.8)
        self.assertEqual(fibs["0.0%"], 0.0)

    def test_hammer_detection(self):
        # Bullish hammer: open=100, close=102, high=102.5, low=90
        candle = {"open": 100.0, "close": 102.0, "high": 102.5, "low": 90.0}
        pattern = TechnicalAnalysis.detect_candle_pattern(candle)
        self.assertEqual(pattern["sentiment"], "BULLISH")
        self.assertIn("Hammer", pattern["pattern"])

    def test_shooting_star_detection(self):
        # Shooting star: open=100, close=98, high=112, low=97.5
        candle = {"open": 100.0, "close": 98.0, "high": 112.0, "low": 97.5}
        pattern = TechnicalAnalysis.detect_candle_pattern(candle)
        self.assertEqual(pattern["sentiment"], "BEARISH")
        self.assertIn("Shooting Star", pattern["pattern"])

    def test_engulfing_detection(self):
        prev = {"open": 100.0, "close": 95.0, "high": 101.0, "low": 94.0}
        curr = {"open": 94.0, "close": 102.0, "high": 103.0, "low": 93.0}
        pattern = TechnicalAnalysis.detect_candle_pattern(curr, prev)
        self.assertEqual(pattern["pattern"], "Bullish Engulfing")
        self.assertEqual(pattern["sentiment"], "BULLISH")


class TestAlertManager(unittest.TestCase):
    def setUp(self):
        self.store = "test_alert_manager_store.json"
        if os.path.exists(self.store):
            os.remove(self.store)
        self.mgr = AlertManager(self.store)

    def tearDown(self):
        if os.path.exists(self.store):
            os.remove(self.store)

    def test_price_alerts_lifecycle(self):
        chat_id = 999
        # Auto >= condition
        a1 = self.mgr.add_price_alert(chat_id, "XAUTUSD", 4200.0, current_price=4170.0)
        self.assertEqual(a1["condition"], ">=")

        # Auto <= condition
        a2 = self.mgr.add_price_alert(chat_id, "BTCUSD", 80000.0, current_price=85000.0)
        self.assertEqual(a2["condition"], "<=")

        chat_alerts = self.mgr.get_chat_alerts(chat_id)
        self.assertEqual(len(chat_alerts), 2)

        # Trigger check: BTCUSD reaches 79500
        triggered = self.mgr.check_price_alerts({"BTCUSD": 79500.0, "XAUTUSD": 4180.0})
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["symbol"], "BTCUSD")

        # BTC alert removed, XAUTUSD still active
        remaining = self.mgr.get_chat_alerts(chat_id)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["symbol"], "XAUTUSD")

        # Remove alert
        self.assertTrue(self.mgr.remove_alert(chat_id, a1["id"]))
        self.assertEqual(len(self.mgr.get_chat_alerts(chat_id)), 0)

    def test_entry_watchers(self):
        chat_id = 888
        w = self.mgr.add_entry_watcher(chat_id, "XAUTUSD", ["1m", "5m", "15m"])
        self.assertEqual(w["symbol"], "XAUTUSD")
        self.assertEqual(w["timeframes"], ["1m", "5m", "15m"])

        self.mgr.update_watcher_alert_time(chat_id, "XAUTUSD", "5m", 1234567890)
        updated = self.mgr.get_chat_entry_watchers(chat_id)[0]
        self.assertEqual(updated["last_alert_time"]["5m"], 1234567890)

        # Reload from disk
        mgr2 = AlertManager(self.store)
        loaded = mgr2.get_chat_entry_watchers(chat_id)[0]
        self.assertEqual(loaded["last_alert_time"]["5m"], 1234567890)

        # Remove
        removed = self.mgr.remove_entry_watcher(chat_id, "XAUTUSD")
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.mgr.get_chat_entry_watchers(chat_id)), 0)


class TestArgumentParsers(unittest.TestCase):
    def test_alert_args(self):
        sym, pr, cond, err = _parse_alert_args(["4180"])
        self.assertIsNone(err)
        self.assertEqual(sym, "XAUTUSD")
        self.assertEqual(pr, 4180.0)

        sym, pr, cond, err = _parse_alert_args(["BTCUSD", "86000"])
        self.assertIsNone(err)
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(pr, 86000.0)

        sym, pr, cond, err = _parse_alert_args(["BTCUSD", "<=", "85000"])
        self.assertIsNone(err)
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(pr, 85000.0)
        self.assertEqual(cond, "<=")

    def test_watch_args(self):
        sym, tfs = _parse_watch_args([])
        self.assertEqual(sym, "XAUTUSD")
        self.assertEqual(tfs, ["1m", "5m", "15m"])

        sym, tfs = _parse_watch_args(["BTCUSD", "5m,15m"])
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(tfs, ["5m", "15m"])


class TestGautamJhaStrategy(unittest.TestCase):
    def test_gautam_jha_levels_calculation(self):
        daily_candles = [
            {"time": 100, "open": 4150.0, "high": 4180.0, "low": 4140.0, "close": 4160.0},  # yesterday
            {"time": 200, "open": 4160.0, "high": 4190.0, "low": 4155.0, "close": 4185.0},  # today
        ]
        curr_price = 4185.0
        gj = TechnicalAnalysis.calc_gautam_jha_levels(daily_candles, curr_price)
        self.assertEqual(gj["daily_open"], 4160.0)
        self.assertEqual(gj["pdh"], 4180.0)
        self.assertEqual(gj["pdl"], 4140.0)
        self.assertEqual(gj["daily_candle_color"], "GREEN")
        self.assertTrue(gj["pdh_swept"])
        self.assertFalse(gj["pdl_swept"])

    def test_reversal_setup_at_pdh(self):
        gj_levels = {
            "daily_open": 4160.0,
            "pdh": 4180.0,
            "pdl": 4140.0,
            "daily_candle_color": "GREEN",
        }
        # Candle swept PDH (high=4185) and closed back below PDH (close=4175, open=4178)
        prev = {"open": 4170.0, "close": 4178.0, "high": 4179.0, "low": 4168.0}
        curr = {"open": 4178.0, "close": 4175.0, "high": 4185.0, "low": 4173.0}
        setup = TechnicalAnalysis.detect_gautam_jha_setup(curr, prev, gj_levels, atr=2.0)
        self.assertIsNotNone(setup)
        self.assertEqual(setup["direction"], "SHORT")
        self.assertIn("Level Reversal", setup["type"])

    def test_break_and_go_setup(self):
        gj_levels = {
            "daily_open": 4160.0,
            "pdh": 4200.0,
            "pdl": 4140.0,
            "daily_candle_color": "GREEN",
        }
        prev = {"open": 4170.0, "close": 4175.0, "high": 4176.0, "low": 4169.0}
        curr = {"open": 4175.0, "close": 4182.0, "high": 4183.0, "low": 4174.0}
        setup = TechnicalAnalysis.detect_gautam_jha_setup(curr, prev, gj_levels, atr=2.0)
        self.assertIsNotNone(setup)
        self.assertEqual(setup["direction"], "LONG")
        self.assertIn("Break-and-Go", setup["type"])


class TestMarketDataLive(unittest.TestCase):
    def test_live_ticker(self):
        t = get_ticker("XAUTUSD")
        self.assertEqual(t["symbol"], "XAUTUSD")
        self.assertGreater(t["close"], 0)

    def test_live_levels(self):
        levels = get_level_analysis("XAUTUSD")
        self.assertEqual(levels["symbol"], "XAUTUSD")
        self.assertIn("pivots", levels)
        self.assertIn("nearest_resistance", levels)
        self.assertIn("nearest_support", levels)
        self.assertIn("gautam_jha", levels)
        msg = format_level_analysis_message(levels)
        self.assertIn("AUTOMATIC LEVEL ANALYSIS", msg)
        self.assertIn("Gautam Jha Liquidity", msg)

    def test_live_gautam_jha_analysis(self):
        from market_data import get_gautam_jha_analysis, format_gautam_jha_message
        analysis = get_gautam_jha_analysis("XAUTUSD")
        self.assertEqual(analysis["symbol"], "XAUTUSD")
        self.assertIn("gautam_jha", analysis)
        self.assertIn("market_structure", analysis)
        msg = format_gautam_jha_message(analysis)
        self.assertIn("GAUTAM JHA PRICE-ACTION ANALYSIS", msg)
        self.assertIn("Daily Open", msg)
        self.assertIn("Previous Day High", msg)

    def test_live_multi_entry(self):
        multi = get_multi_timeframe_entry("XAUTUSD", ["1m", "5m", "15m"])
        self.assertEqual(multi["symbol"], "XAUTUSD")
        self.assertIn("1m", multi["timeframes"])
        self.assertIn("5m", multi["timeframes"])
        self.assertIn("15m", multi["timeframes"])
        msg = format_entry_analysis_message(multi)
        self.assertIn("CANDLE ENTRY SCANNER", msg)
        self.assertIn("Timeframe: 1M", msg)
        self.assertIn("Timeframe: 5M", msg)
        self.assertIn("Timeframe: 15M", msg)


class TestBotCommands(unittest.TestCase):
    def test_list_command_content(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from main import list_cmd

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_ctx = MagicMock()

        asyncio.run(list_cmd(mock_update, mock_ctx))
        mock_update.message.reply_text.assert_called_once()
        sent_text = mock_update.message.reply_text.call_args[0][0]

        # Verify all essential commands are listed
        expected_cmds = [
            "/price", "/levels", "/analysis", "/gj", "/liquidity",
            "/alert", "/alerts", "/delalert", "/clearalerts",
            "/entry", "/scan", "/watch", "/unwatch", "/watchers",
            "/list", "/help", "/start", "/reset"
        ]
        for cmd in expected_cmds:
            self.assertIn(cmd, sent_text, f"Command {cmd} should be in /list output")


if __name__ == "__main__":
    unittest.main()
