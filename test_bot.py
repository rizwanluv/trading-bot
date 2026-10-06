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
        # Auto-detects Gold when price in Gold range
        sym, pr, cond, err = _parse_alert_args(["4180"])
        self.assertIsNone(err)
        self.assertEqual(sym, "XAUTUSD")
        self.assertEqual(pr, 4180.0)

        # Auto-detects Bitcoin when price >= 15000
        sym, pr, cond, err = _parse_alert_args(["85000"])
        self.assertIsNone(err)
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(pr, 85000.0)

        # Explicit BTC alias
        sym, pr, cond, err = _parse_alert_args(["btc", "86000"])
        self.assertIsNone(err)
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(pr, 86000.0)

        # Explicit Gold alias
        sym, pr, cond, err = _parse_alert_args(["gold", "4200"])
        self.assertIsNone(err)
        self.assertEqual(sym, "XAUTUSD")
        self.assertEqual(pr, 4200.0)

        # Explicit condition
        sym, pr, cond, err = _parse_alert_args(["BTCUSD", "<=", "85000"])
        self.assertIsNone(err)
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(pr, 85000.0)
        self.assertEqual(cond, "<=")

    def test_watch_args(self):
        sym, tfs = _parse_watch_args([])
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(tfs, ["1m", "5m", "15m"])

        sym, tfs = _parse_watch_args(["gold"])
        self.assertEqual(sym, "XAUTUSD")
        self.assertEqual(tfs, ["1m", "5m", "15m"])

        sym, tfs = _parse_watch_args(["BTCUSD", "5m,15m"])
        self.assertEqual(sym, "BTCUSD")
        self.assertEqual(tfs, ["5m", "15m"])


class TestSymbolResolution(unittest.TestCase):
    def test_resolve_symbol(self):
        from market_data import resolve_symbol
        self.assertEqual(resolve_symbol("btc"), "BTCUSD")
        self.assertEqual(resolve_symbol("bitcoin"), "BTCUSD")
        self.assertEqual(resolve_symbol("BTCUSD"), "BTCUSD")
        self.assertEqual(resolve_symbol("BTC/USD"), "BTCUSD")
        self.assertEqual(resolve_symbol("xbt"), "BTCUSD")
        self.assertEqual(resolve_symbol("gold"), "XAUTUSD")
        self.assertEqual(resolve_symbol("xau"), "XAUTUSD")
        self.assertEqual(resolve_symbol("xauusd"), "XAUTUSD")
        self.assertEqual(resolve_symbol("XAU/USD"), "XAUTUSD")
        self.assertEqual(resolve_symbol("goldusd"), "XAUTUSD")
        self.assertEqual(resolve_symbol("paxg"), "XAUTUSD")
        self.assertEqual(resolve_symbol("xautusd"), "XAUTUSD")
        self.assertEqual(resolve_symbol("eth"), "ETHUSD")
        self.assertEqual(resolve_symbol("sol"), "SOLUSD")
        self.assertEqual(resolve_symbol(None), "BTCUSD")

    def test_detect_symbol_from_text(self):
        from main import _detect_symbol_from_text
        self.assertEqual(_detect_symbol_from_text("what is the gold price?"), "XAUTUSD")
        self.assertEqual(_detect_symbol_from_text("is xau forming a reversal?"), "XAUTUSD")
        self.assertEqual(_detect_symbol_from_text("XAUUSD 15m"), "XAUTUSD")
        self.assertEqual(_detect_symbol_from_text("XAU/USD 5m"), "XAUTUSD")
        self.assertEqual(_detect_symbol_from_text("btc looking bullish today"), "BTCUSD")
        self.assertEqual(_detect_symbol_from_text("bitcoin break and go setup"), "BTCUSD")
        self.assertEqual(_detect_symbol_from_text("eth levels"), "ETHUSD")


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
    def test_live_tickers(self):
        t_btc = get_ticker("BTCUSD")
        self.assertEqual(t_btc["symbol"], "BTCUSD")
        self.assertGreater(t_btc["close"], 0)

        t_gold = get_ticker("XAUUSD")  # resolved to XAUTUSD
        self.assertEqual(t_gold["symbol"], "XAUTUSD")
        self.assertGreater(t_gold["close"], 0)

    def test_live_market_overview(self):
        from market_data import get_market_overview
        overview = get_market_overview()
        self.assertIn("LIVE MARKET OVERVIEW", overview)
        self.assertIn("Bitcoin (BTC/USD)", overview)
        self.assertIn("Gold (XAU/USD)", overview)

    def test_live_levels(self):
        levels_btc = get_level_analysis("BTCUSD")
        self.assertEqual(levels_btc["symbol"], "BTCUSD")
        self.assertIn("pivots", levels_btc)
        msg_btc = format_level_analysis_message(levels_btc)
        self.assertIn("AUTOMATIC LEVEL ANALYSIS", msg_btc)

        levels_gold = get_level_analysis("XAUUSD")
        self.assertEqual(levels_gold["symbol"], "XAUTUSD")
        self.assertIn("pivots", levels_gold)
        msg_gold = format_level_analysis_message(levels_gold)
        self.assertIn("Gold (XAU/USD)", msg_gold)
        self.assertIn("Gautam Jha Liquidity", msg_gold)

    def test_live_gautam_jha_analysis(self):
        from market_data import get_gautam_jha_analysis, format_gautam_jha_message
        analysis = get_gautam_jha_analysis("XAUUSD")
        self.assertEqual(analysis["symbol"], "XAUTUSD")
        self.assertIn("gautam_jha", analysis)
        self.assertIn("market_structure", analysis)
        msg = format_gautam_jha_message(analysis)
        self.assertIn("GAUTAM JHA PRICE-ACTION ANALYSIS", msg)
        self.assertIn("Gold (XAU/USD)", msg)
        self.assertIn("Daily Open", msg)
        self.assertIn("Previous Day High", msg)

    def test_live_multi_entry(self):
        multi = get_multi_timeframe_entry("XAUUSD", ["1m", "5m", "15m"])
        self.assertEqual(multi["symbol"], "XAUTUSD")
        self.assertIn("1m", multi["timeframes"])
        self.assertIn("5m", multi["timeframes"])
        self.assertIn("15m", multi["timeframes"])
        msg = format_entry_analysis_message(multi)
        self.assertIn("CANDLE ENTRY SCANNER", msg)
        self.assertIn("Gold (XAU/USD)", msg)
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

        # Verify all essential commands and shortcuts are listed
        expected_cmds = [
            "/btc", "/btclevels", "/btcgj", "/btcentry", "/btcwatch",
            "/gold", "/xau", "/xauusd", "/goldlevels", "/goldgj", "/goldentry", "/goldwatch",
            "/price", "/levels", "/analysis", "/gj", "/liquidity",
            "/alert", "/alerts", "/delalert", "/clearalerts",
            "/entry", "/scan", "/watch", "/unwatch", "/watchers",
            "/autotrade", "/starttrade", "/stoptrade", "/trade", "/positions", "/closeposition", "/balance", "/mode",
            "/alertson", "/alertsoff", "/setkey", "/setsecret", "/setkeys", "/keys",
            "/list", "/help", "/start", "/reset"
        ]
        for cmd in expected_cmds:
            self.assertIn(cmd, sent_text, f"Command {cmd} should be in /list output")

    def test_trading_commands(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from main import (
            keys_cmd,
            mode_cmd,
            autotrade_cmd,
            start_trade_cmd,
            stop_trade_cmd,
            auto_alert_on_cmd,
            auto_alert_off_cmd,
            set_key_cmd,
            set_secret_cmd,
            balance_cmd,
            positions_cmd,
            set_keys_cmd,
            trade_cmd,
            close_all_cmd,
            auto_trader,
            alert_manager,
        )

        mock_update = MagicMock()
        mock_update.effective_chat.id = 555
        mock_update.message.reply_text = AsyncMock()

        # 1. /keys
        mock_ctx = MagicMock()
        asyncio.run(keys_cmd(mock_update, mock_ctx))
        self.assertIn("Delta Exchange API Status", mock_update.message.reply_text.call_args[0][0])

        # 2. /mode
        mock_ctx.args = []
        asyncio.run(mode_cmd(mock_update, mock_ctx))
        self.assertIn("Current Trading Mode", mock_update.message.reply_text.call_args[0][0])

        # 3. /starttrade (Trading Start Automatic ON)
        mock_ctx.args = []
        asyncio.run(start_trade_cmd(mock_update, mock_ctx))
        self.assertTrue(auto_trader.enabled)
        self.assertIn("AUTOMATIC TRADING ENGINE: STARTED", mock_update.message.reply_text.call_args[0][0])

        # 4. /stoptrade (Trading Stop Automatic OFF)
        asyncio.run(stop_trade_cmd(mock_update, mock_ctx))
        self.assertFalse(auto_trader.enabled)
        self.assertIn("AUTOMATIC TRADING ENGINE: STOPPED", mock_update.message.reply_text.call_args[0][0])

        # 5. /alertson (Automatic Alerts ON)
        asyncio.run(auto_alert_on_cmd(mock_update, mock_ctx))
        watchers = alert_manager.get_chat_entry_watchers(555)
        self.assertGreaterEqual(len(watchers), 2)
        self.assertIn("AUTOMATIC ALERTS: TURNED ON", mock_update.message.reply_text.call_args[0][0])

        # 6. /alertsoff (Automatic Alerts OFF)
        asyncio.run(auto_alert_off_cmd(mock_update, mock_ctx))
        watchers_after = alert_manager.get_chat_entry_watchers(555)
        self.assertEqual(len(watchers_after), 0)
        self.assertIn("AUTOMATIC ALERTS: TURNED OFF", mock_update.message.reply_text.call_args[0][0])

        # 7. /setkey and /setsecret
        mock_ctx.args = ["ovRwsM4ZGWkK2JI67yOIiTdu2BWnhg"]
        asyncio.run(set_key_cmd(mock_update, mock_ctx))
        self.assertIn("Delta Exchange API Key Saved", mock_update.message.reply_text.call_args[0][0])

        mock_ctx.args = ["dummy_secret_12345"]
        asyncio.run(set_secret_cmd(mock_update, mock_ctx))
        self.assertIn("Delta Exchange API Secret Saved", mock_update.message.reply_text.call_args[0][0])

        # 8. /balance
        mock_ctx.args = []
        asyncio.run(balance_cmd(mock_update, mock_ctx))
        self.assertIn("Paper Trading Account Balance", mock_update.message.reply_text.call_args[0][0])

        # 9. /trade BTC buy
        mock_ctx.args = ["BTC", "buy", "0.01"]
        asyncio.run(trade_cmd(mock_update, mock_ctx))
        self.assertIn("TRADE EXECUTED", mock_update.message.reply_text.call_args[0][0])

        # 10. /positions
        asyncio.run(positions_cmd(mock_update, mock_ctx))
        self.assertIn("Active Paper Positions", mock_update.message.reply_text.call_args[0][0])

        # 11. /closeall
        asyncio.run(close_all_cmd(mock_update, mock_ctx))
        self.assertIn("Closed", mock_update.message.reply_text.call_args[0][0])



class TestDeltaClient(unittest.TestCase):
    def setUp(self):
        from delta_client import DeltaClient
        self.client = DeltaClient(api_key="test_api_key_12345", api_secret="test_secret_67890")

    def test_client_configuration(self):
        self.assertTrue(self.client.is_configured())
        masked = self.client.get_masked_key()
        self.assertTrue(masked.startswith("te"))
        self.assertTrue(masked.endswith("45"))
        self.assertIn("...", masked)

    def test_product_resolution(self):
        self.assertEqual(self.client.get_product_id("BTCUSD"), 27)
        self.assertEqual(self.client.get_product_id("BTC"), 27)
        self.assertEqual(self.client.get_product_id("XAUTUSD"), 131253)
        self.assertEqual(self.client.get_product_id("GOLD"), 131253)

    def test_signature_generation(self):
        sig, ts = self.client._generate_signature("GET", "/v2/wallet/balances")
        self.assertIsInstance(sig, str)
        self.assertGreater(len(sig), 20)
        self.assertIsInstance(ts, str)
        self.assertTrue(ts.isdigit())

        headers = self.client._get_headers("POST", "/v2/orders", payload={"product_id": 27})
        self.assertEqual(headers["api-key"], "test_api_key_12345")
        self.assertIn("signature", headers)
        self.assertIn("timestamp", headers)
        self.assertEqual(headers["Content-Type"], "application/json")


class TestAutoTrader(unittest.TestCase):
    def setUp(self):
        from auto_trader import AutoTrader
        self.test_store = "test_autotrader_store.json"
        if os.path.exists(self.test_store):
            os.remove(self.test_store)
        self.trader = AutoTrader(store_file=self.test_store, mode="paper", default_size=0.01)

    def tearDown(self):
        if os.path.exists(self.test_store):
            os.remove(self.test_store)

    def test_trader_initial_state(self):
        self.assertEqual(self.trader.mode, "paper")
        self.assertFalse(self.trader.enabled)
        self.assertEqual(self.trader.balance, 10000.0)
        self.assertEqual(len(self.trader.get_open_positions()), 0)

    def test_execute_paper_trade(self):
        # Open Long position on BTCUSD
        pos = self.trader.execute_trade(
            symbol="BTCUSD",
            side="buy",
            size=0.02,
            trade_type="manual",
            sl_price=60000.0,
            tp1_price=70000.0,
            tp2_price=75000.0,
        )
        self.assertEqual(pos["status"], "open")
        self.assertEqual(pos["symbol"], "BTCUSD")
        self.assertEqual(pos["side"], "buy")
        self.assertEqual(pos["size"], 0.02)
        self.assertEqual(pos["sl_price"], 60000.0)
        self.assertEqual(pos["tp1_price"], 70000.0)

        # Verify open positions count
        open_pos = self.trader.get_open_positions()
        self.assertEqual(len(open_pos), 1)
        self.assertEqual(open_pos[0]["id"], pos["position_id"])

    def test_close_paper_trade_with_pnl(self):
        # Execute trade with manual entry
        pos = self.trader.execute_trade(
            symbol="BTCUSD",
            side="buy",
            size=1.0,
            trade_type="manual",
        )
        pid = pos["position_id"]
        entry = pos["entry_price"]

        # Close position at entry + $1000
        close_res = self.trader.close_position(pid, exit_price=entry + 1000.0)
        self.assertEqual(close_res["status"], "closed")
        self.assertEqual(close_res["exit_price"], entry + 1000.0)
        self.assertAlmostEqual(close_res["pnl"], 1000.0)
        self.assertEqual(self.trader.balance, 11000.0)
        self.assertEqual(len(self.trader.get_open_positions()), 0)

    def test_close_all_positions(self):
        self.trader.execute_trade("BTCUSD", "buy", size=0.01)
        self.trader.execute_trade("XAUTUSD", "sell", size=0.05)
        self.assertEqual(len(self.trader.get_open_positions()), 2)

        closed = self.trader.close_all_positions()
        self.assertEqual(len(closed), 2)
        self.assertEqual(len(self.trader.get_open_positions()), 0)

    def test_exit_trigger_monitoring(self):
        # Create position with tight SL and TP
        pos = self.trader.execute_trade(
            symbol="BTCUSD",
            side="buy",
            size=0.1,
            sl_price=50000.0,
            tp1_price=60000.0,
            tp2_price=65000.0,
        )
        pid = pos["position_id"]

        # Modify position entry to simulate price reaching TP1
        for p in self.trader.positions:
            if p["id"] == pid:
                p["tp1_price"] = 1.0  # Mark price is definitely > 1.0, triggering TP
                break

        exits = self.trader.check_open_positions_for_exits()
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["position_id"], pid)
        self.assertIn("TAKE PROFIT", exits[0]["exit_reason"])
        self.assertEqual(len(self.trader.get_open_positions()), 0)

    def test_subscribers_and_mode_toggle(self):
        self.trader.add_subscriber(12345)
        self.assertIn(12345, self.trader.subscribers)
        self.trader.remove_subscriber(12345)
        self.assertNotIn(12345, self.trader.subscribers)

        # Before switching to live, credentials must be set
        self.trader.delta_client.set_credentials("test_key", "test_secret")
        self.trader.set_mode("live")
        self.assertEqual(self.trader.mode, "live")
        self.trader.set_mode("paper")
        self.assertEqual(self.trader.mode, "paper")

        self.trader.set_enabled(True)
        self.assertTrue(self.trader.enabled)
        self.trader.set_enabled(False)
        self.assertFalse(self.trader.enabled)

    def test_account_summary(self):
        summary = self.trader.get_account_summary()
        self.assertEqual(summary["mode"], "paper")
        self.assertEqual(summary["balance"], 10000.0)
        self.assertIn("win_rate_pct", summary)


if __name__ == "__main__":
    unittest.main()

