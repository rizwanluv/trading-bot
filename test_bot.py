"""
Automated test suite verifying price alerts, level analysis, and multi-timeframe candle entry analysis.
"""
import os
import tempfile
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from auto_trader import AutoTrader
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

        # Telegram hard limit: message length must strictly not exceed 4096 characters
        self.assertLessEqual(len(sent_text), 4096, "list_cmd must strictly not exceed Telegram 4096 character limit")

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

        # 7. /setkey and /setsecret (mocked to prevent overwriting real .env)
        from unittest.mock import patch
        with patch("main.save_delta_credentials") as mock_save:
            mock_ctx.args = ["ovRwsM4ZGWkK2JI67yOIiTdu2BWnhg"]
            asyncio.run(set_key_cmd(mock_update, mock_ctx))
            self.assertIn("Delta Exchange API Key Saved", mock_update.message.reply_text.call_args[0][0])
            self.assertTrue(mock_save.called)

            mock_ctx.args = ["dummy_secret_12345"]
            asyncio.run(set_secret_cmd(mock_update, mock_ctx))
            self.assertIn("Delta Exchange API Secret Saved", mock_update.message.reply_text.call_args[0][0])

        # 8. /balance
        mock_ctx.args = []
        asyncio.run(balance_cmd(mock_update, mock_ctx))
        self.assertIn("Paper Trading Account Balance", mock_update.message.reply_text.call_args[0][0])

        # 9. /trade BTC buy
        auto_trader.close_all_positions()
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


class TestChatRoutingAndAliases(unittest.TestCase):
    def test_chat_plain_text_routing(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from main import chat, set_base_url_cmd, delta_client

        mock_update = MagicMock()
        mock_update.effective_chat.id = 777
        mock_update.message.reply_text = AsyncMock()
        mock_ctx = MagicMock()

        # 1. Plain text "keys check"
        mock_update.message.text = "keys check"
        asyncio.run(chat(mock_update, mock_ctx))
        self.assertIn("Delta Exchange API Status", mock_update.message.reply_text.call_args[0][0])

        # 2. Plain text "start trade"
        mock_update.message.text = "start trade"
        asyncio.run(chat(mock_update, mock_ctx))
        self.assertIn("AUTOMATIC TRADING ENGINE: STARTED", mock_update.message.reply_text.call_args[0][0])

        # 3. Plain text "alerts on"
        mock_update.message.text = "alerts on"
        asyncio.run(chat(mock_update, mock_ctx))
        self.assertIn("AUTOMATIC ALERTS: TURNED ON", mock_update.message.reply_text.call_args[0][0])

        # Plain text "list command" and "menu"
        mock_update.message.text = "list command"
        asyncio.run(chat(mock_update, mock_ctx))
        self.assertIn("UNIFIED COMMAND HUBS", mock_update.message.reply_text.call_args[0][0])

        # 4. Pasted API credentials in chat
        with patch("main.save_delta_credentials") as mock_save, patch("main.keys_cmd") as mock_keys:
            mock_update.message.text = "delta new api test_key_12345 api secret test_secret_67890"
            asyncio.run(chat(mock_update, mock_ctx))
            self.assertTrue(mock_save.called)
            self.assertTrue(mock_keys.called)

        # 5. /setbaseurl
        mock_ctx.args = ["global"]
        asyncio.run(set_base_url_cmd(mock_update, mock_ctx))
        self.assertEqual(delta_client.base_url, "https://api.delta.exchange")
        mock_ctx.args = ["india"]
        asyncio.run(set_base_url_cmd(mock_update, mock_ctx))
        self.assertEqual(delta_client.base_url, "https://api.india.delta.exchange")

        # 6. /model and dynamic model switching
        from main import gemini_model_cmd, get_gemini_model
        mock_ctx.args = []
        asyncio.run(gemini_model_cmd(mock_update, mock_ctx))
        self.assertIn("Google Gemini AI Model Configuration", mock_update.message.reply_text.call_args[0][0])

        mock_ctx.args = ["pro"]
        asyncio.run(gemini_model_cmd(mock_update, mock_ctx))
        self.assertEqual(get_gemini_model(), "gemini-2.5-pro")

        mock_update.message.text = "update google model latest model"
        mock_ctx.args = []
        asyncio.run(chat(mock_update, mock_ctx))
        self.assertEqual(get_gemini_model(), "gemini-2.5-flash")
        self.assertIn("Google Gemini Model Updated", mock_update.message.reply_text.call_args[0][0])


class TestSelfLearningEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_store = "test_learning_store.json"
        if os.path.exists(self.tmp_store):
            os.remove(self.tmp_store)

    def tearDown(self):
        if os.path.exists(self.tmp_store):
            os.remove(self.tmp_store)

    def test_setup_metrics_calculations(self):
        from self_learning import SetupMetrics
        m = SetupMetrics("GJ_Reversal_PDH")
        self.assertEqual(m.win_rate, 0.0)
        self.assertEqual(m.profit_factor, 1.0)
        self.assertEqual(m.score, 1.0)

        # Record win
        m.total_trades = 5
        m.wins = 4
        m.losses = 1
        m.gross_profit = 400.0
        m.gross_loss = 100.0
        self.assertEqual(m.win_rate, 80.0)
        self.assertEqual(m.profit_factor, 4.0)
        self.assertGreater(m.score, 1.0)

    def test_learning_engine_recording_and_suppression(self):
        from self_learning import LearningEngine
        engine = LearningEngine(store_file=self.tmp_store)
        engine.settings["min_trades_for_adaptation"] = 3

        # Record 3 consecutive losses on GJ_Break_And_Go
        for i in range(3):
            engine.record_trade_outcome({
                "symbol": "BTCUSD",
                "strategy": "Gautam Jha Break and Go",
                "pnl": -50.0,
                "is_win": False,
                "exit_reason": "Stop Loss Triggered",
            })

        # Check suppression
        setup_m = engine.setups.get("GJ_Break_And_Go")
        self.assertIsNotNone(setup_m)
        self.assertEqual(setup_m.status, "SUPPRESSED")
        self.assertEqual(setup_m.win_rate, 0.0)

        # can_execute should now return False for suppressed setup!
        can_run, reason = engine.can_execute("Gautam Jha Break and Go", "BTCUSD")
        self.assertFalse(can_run)
        self.assertIn("suppressed", reason.lower())

        # Unsuppress setup
        self.assertTrue(engine.unsuppress_setup("Gautam Jha Break and Go"))
        self.assertEqual(engine.setups["GJ_Break_And_Go"].status, "ACTIVE")

    def test_learning_engine_prioritization(self):
        from self_learning import LearningEngine
        engine = LearningEngine(store_file=self.tmp_store)
        engine.settings["min_trades_for_adaptation"] = 3

        # Record 4 wins and 1 loss on GJ_Reversal_PDL
        for i in range(4):
            engine.record_trade_outcome({
                "symbol": "BTCUSD",
                "strategy": "Gautam Jha Reversal at PDL",
                "pnl": 120.0,
                "is_win": True,
                "exit_reason": "Take Profit (TP1 Hit)",
            })
        engine.record_trade_outcome({
            "symbol": "BTCUSD",
            "strategy": "Gautam Jha Reversal at PDL",
            "pnl": -30.0,
            "is_win": False,
            "exit_reason": "Stop Loss Triggered",
        })

        setup_m = engine.setups.get("GJ_Reversal_PDL")
        self.assertIsNotNone(setup_m)
        self.assertEqual(setup_m.status, "PRIORITIZED")
        self.assertGreater(setup_m.weight_multiplier, 1.0)

        params = engine.get_adapted_parameters("Gautam Jha Reversal at PDL", "BTCUSD", default_size=1.0)
        self.assertGreater(params["recommended_size"], 1.0)

    def test_zero_token_report_and_token_capped_ai_insight(self):
        from self_learning import LearningEngine
        engine = LearningEngine(store_file=self.tmp_store)

        engine.record_trade_outcome({
            "symbol": "XAUTUSD",
            "strategy": "15m Pin Bar",
            "pnl": 150.0,
            "is_win": True,
            "exit_reason": "Take Profit (TP2 Hit)",
        })

        # Local report consumes 0 tokens
        summary = engine.get_learning_summary()
        self.assertEqual(summary["tokens_consumed"], 0)
        html = engine.format_html_report()
        self.assertIn("SELF-LEARNING STRATEGY ENGINE", html)
        self.assertIn("0 AI Tokens Used", html)

        # AI insight uses minimal tokens with cache
        mock_ai_called = []
        def mock_caller(prompt, max_tokens=250):
            mock_ai_called.append(prompt)
            self.assertLessEqual(max_tokens, 250)
            return "1. Exploit Pin Bar. 2. London session edge. 3. Respect stops."

        insight1 = engine.generate_ai_insight(mock_caller)
        self.assertIn("Exploit Pin Bar", insight1)
        self.assertEqual(len(mock_ai_called), 1)

        # Second call hits cache (0 additional tokens)
        insight2 = engine.generate_ai_insight(mock_caller)
        self.assertIn("Cached insight", insight2)
        self.assertEqual(len(mock_ai_called), 1)

    def test_autotrader_learning_integration(self):
        from auto_trader import AutoTrader
        from self_learning import LearningEngine

        engine = LearningEngine(store_file=self.tmp_store)
        trader = AutoTrader(store_file="test_autotrade_store.json", learning_engine=engine)

        # Execute and close a trade
        pos = trader.execute_trade("BTCUSD", "BUY", size=0.01)
        closed = trader.close_position(pos["id"], reason="TAKE PROFIT (TP1 Hit)", exit_price=pos["entry_price"] + 500)
        self.assertTrue(closed["is_win"])

        # Check that learning engine recorded it
        self.assertEqual(engine.overall.total_trades, 1)
        self.assertEqual(engine.overall.wins, 1)

        # Clean up test store
        if os.path.exists("test_autotrade_store.json"):
            os.remove("test_autotrade_store.json")

    def test_learn_commands_in_chat(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from main import chat, learn_cmd

        mock_update = MagicMock()
        mock_update.effective_chat.id = 777
        mock_update.message.reply_text = AsyncMock()
        mock_ctx = MagicMock()

        # Plain text "self learning"
        mock_update.message.text = "self learning and self improve its own"
        asyncio.run(chat(mock_update, mock_ctx))
        call_text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("SELF-LEARNING STRATEGY ENGINE", call_text)

        # /learn reset
        mock_ctx.args = ["reset"]
        asyncio.run(learn_cmd(mock_update, mock_ctx))
        call_text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Self-Learning Memory Reset", call_text)


class TestNewsAndOrderbookAndConfluence(unittest.TestCase):
    def test_news_sentiment_lexicon_scoring(self):
        from news_analysis import score_headline, format_news_html_report, get_news_sentiment

        # Bullish headline
        bull = score_headline("Bitcoin breaks out to record high with massive institutional inflows")
        self.assertEqual(bull["label"], "BULLISH")
        self.assertGreater(bull["score"], 0.2)

        # Bearish headline
        bear = score_headline("Crypto flash crash as SEC launches lawsuit and liquidation cascade ensues")
        self.assertEqual(bear["label"], "BEARISH")
        self.assertLess(bear["score"], -0.2)

        # Sentiment data structure
        data = get_news_sentiment("BTCUSD", limit=5)
        self.assertIn(data["sentiment_label"], ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"))
        self.assertIn("articles", data)
        self.assertGreater(len(data["articles"]), 0)

        # HTML formatting
        html = format_news_html_report(data)
        self.assertIn("MARKET NEWS & SENTIMENT ANALYSIS", html)

    def test_orderbook_analysis(self):
        from orderbook_analysis import analyze_orderbook, format_orderbook_html_report

        mock_raw = {
            "symbol": "BTCUSD",
            "buy": [
                {"price": "84000.0", "size": 25000},
                {"price": "83990.0", "size": 3000},
                {"price": "83980.0", "size": 2000},
            ],
            "sell": [
                {"price": "84010.0", "size": 2000},
                {"price": "84020.0", "size": 3000},
                {"price": "84030.0", "size": 2500},
            ],
        }
        res = analyze_orderbook("BTCUSD", raw_data=mock_raw)
        self.assertEqual(res["best_bid"], 84000.0)
        self.assertEqual(res["best_ask"], 84010.0)
        self.assertEqual(res["spread"], 10.0)
        self.assertGreater(res["imbalance_ratio"], 0.2)
        self.assertEqual(res["direction"], "LONG")
        self.assertIn("BULLISH", res["bias"])

        # Liquidity wall detection: the 10000 buy level should be flagged as a buy wall
        self.assertGreater(len(res["bid_walls"]), 0)
        self.assertEqual(res["bid_walls"][0]["price"], 84000.0)

        # HTML formatting
        html = format_orderbook_html_report(res)
        self.assertIn("ORDER BOOK (L2 DEPTH) ANALYSIS", html)
        self.assertIn("Buy Wall", html)

    def test_confluence_engine_evaluation_and_execution(self):
        from confluence_engine import ConfluenceEngine, format_confluence_html_report
        from auto_trader import AutoTrader

        ce = ConfluenceEngine()
        res = ce.evaluate_confluence("BTCUSD")
        self.assertIn("confluence_score", res)
        self.assertIn("layers", res)
        self.assertIn("gautam_jha", res["layers"])
        self.assertIn("candlestick", res["layers"])
        self.assertIn("orderbook", res["layers"])
        self.assertIn("news", res["layers"])
        self.assertIn("self_learning", res["layers"])

        html = format_confluence_html_report(res)
        self.assertIn("MULTI-STRATEGY MASTER CONFLUENCE", html)

        # Test trade execution via AutoTrader
        trader = AutoTrader(store_file="test_autotrade_store_conf.json")
        exec_res = trader.execute_confluence_trade("BTCUSD", force=True)
        self.assertEqual(exec_res["status"], "executed")
        self.assertEqual(len(trader.positions), 1)

        # Clean up
        trader.close_all_positions()
        if os.path.exists("test_autotrade_store_conf.json"):
            os.remove("test_autotrade_store_conf.json")

    def test_news_orderbook_confluence_chat_routing(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from main import chat, news_cmd, orderbook_cmd, confluence_cmd

        mock_update = MagicMock()
        mock_update.effective_chat.id = 777
        mock_update.message.reply_text = AsyncMock()
        mock_ctx = MagicMock()

        # 1. Plain text "news analysis"
        mock_update.message.text = "news analysis"
        asyncio.run(chat(mock_update, mock_ctx))
        call_text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MARKET NEWS & SENTIMENT ANALYSIS", call_text)

        # 2. Plain text "oderbook analysis" (user prompt spelling)
        mock_update.message.text = "oderbook analysis"
        asyncio.run(chat(mock_update, mock_ctx))
        call_text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("ORDER BOOK (L2 DEPTH) ANALYSIS", call_text)

        # 3. Plain text "every strategy combined"
        mock_update.message.text = "every strategy combined"
        asyncio.run(chat(mock_update, mock_ctx))
        call_text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MULTI-STRATEGY MASTER CONFLUENCE", call_text)


class Test18AgentsAndUnifiedHubs(unittest.TestCase):
    def setUp(self):
        from unittest.mock import AsyncMock, MagicMock
        self.mock_update = MagicMock()
        self.mock_update.effective_chat.id = 888
        self.mock_update.message.reply_text = AsyncMock()
        self.mock_ctx = MagicMock()

    def test_18_agents_system_prompt_structure(self):
        from main import SYSTEM_PROMPT_18_AGENTS
        self.assertIn("1. Price Action Core", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("2. Liquidity & Sessions", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("3. Market Context", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("4. News & Sentiment", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("5. Momentum & Strength", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("6. Decision Layer", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("Confluence Score", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("Risk Manager", SYSTEM_PROMPT_18_AGENTS)
        self.assertIn("Final Decision", SYSTEM_PROMPT_18_AGENTS)

    def test_generate_18_agents_analysis_deterministic(self):
        from main import generate_18_agents_analysis
        report = generate_18_agents_analysis("BTCUSD", "15m")
        self.assertIn("18-Agent Institutional Desk Analysis", report)
        self.assertIn("1. Price Action Core", report)
        self.assertIn("2. Liquidity & Sessions", report)
        self.assertIn("3. Market Context", report)
        self.assertIn("4. News & Sentiment", report)
        self.assertIn("5. Momentum & Strength", report)
        self.assertIn("6. Decision Layer", report)
        self.assertIn("Confluence Agent:", report)
        self.assertIn("Risk Manager:", report)
        self.assertIn("Final Decision:", report)
        self.assertIn("Direction:", report)
        self.assertIn("Score:", report)

    def test_format_symbol_hub_overview(self):
        from main import format_symbol_hub_overview
        btc_card = format_symbol_hub_overview("BTCUSD")
        self.assertIn("BITCOIN", btc_card)
        self.assertIn("ALL-IN-ONE HUB", btc_card)
        self.assertIn("Daily Open:", btc_card)
        self.assertIn("Order Book:", btc_card)
        self.assertIn("News Sentiment:", btc_card)
        self.assertIn("Master Confluence:", btc_card)
        self.assertIn("/btc price", btc_card)
        self.assertIn("/btc analyze", btc_card)

        gold_card = format_symbol_hub_overview("XAUTUSD")
        self.assertIn("GOLD", gold_card)
        self.assertIn("ALL-IN-ONE HUB", gold_card)
        self.assertIn("/gold price", gold_card)
        self.assertIn("/gold analyze", gold_card)

    def test_unified_btc_cmd_dispatch(self):
        import asyncio
        from main import btc_cmd

        # 1. /btc with no args -> All-in-one card
        self.mock_ctx.args = []
        asyncio.run(btc_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("BITCOIN", text)
        self.assertIn("ALL-IN-ONE HUB", text)

        # 2. /btc price -> Live ticker
        self.mock_ctx.args = ["price"]
        asyncio.run(btc_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Live Ticker", text)

        # 3. /btc levels -> Levels analysis
        self.mock_ctx.args = ["levels"]
        asyncio.run(btc_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AUTOMATIC LEVEL ANALYSIS", text)

        # 4. /btc gj -> Gautam Jha liquidity
        self.mock_ctx.args = ["gj"]
        asyncio.run(btc_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("GAUTAM JHA PRICE-ACTION ANALYSIS", text)

    def test_unified_gold_cmd_dispatch(self):
        import asyncio
        from main import gold_cmd

        # 1. /gold no args -> Gold all-in-one card
        self.mock_ctx.args = []
        asyncio.run(gold_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("GOLD", text)
        self.assertIn("ALL-IN-ONE HUB", text)

        # 2. /gold news -> Gold news
        self.mock_ctx.args = ["news"]
        asyncio.run(gold_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MARKET NEWS & SENTIMENT ANALYSIS", text)

        # 3. /gold confluence -> Gold confluence
        self.mock_ctx.args = ["confluence"]
        asyncio.run(gold_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MULTI-STRATEGY MASTER CONFLUENCE", text)

    def test_unified_trade_cmd_dispatch(self):
        import asyncio
        from main import trade_cmd, auto_trader

        # 1. /trade no args -> Master dashboard
        self.mock_ctx.args = []
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MASTER TRADING & PORTFOLIO HUB", text)
        self.assertIn("Bot Status:", text)
        self.assertIn("Account Balance:", text)

        # 2. /trade on -> Start auto trading
        self.mock_ctx.args = ["on"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        self.assertTrue(auto_trader.enabled)

        # 3. /trade off -> Stop auto trading
        self.mock_ctx.args = ["off"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        self.assertFalse(auto_trader.enabled)

        # 4. /trade pos -> Open positions
        self.mock_ctx.args = ["pos"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertTrue("Positions" in text or "Position" in text)

        # 5. /trade bal -> Balances
        self.mock_ctx.args = ["bal"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertTrue("Balance" in text or "Balances" in text)

    def test_unified_alert_cmd_dispatch(self):
        import asyncio
        from main import set_alert_cmd

        # 1. /alert no args -> Master alerts hub
        self.mock_ctx.args = []
        asyncio.run(set_alert_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MASTER ALERTS CONTROL CENTER", text)

        # 2. /alert on -> Auto alerts turned on
        self.mock_ctx.args = ["on"]
        asyncio.run(set_alert_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AUTOMATIC ALERTS: TURNED ON", text)

        # 3. /alert off -> Auto alerts turned off
        self.mock_ctx.args = ["off"]
        asyncio.run(set_alert_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AUTOMATIC ALERTS: TURNED OFF", text)

        # 4. /alert btc 85000 -> Price alert
        self.mock_ctx.args = ["btc", "85000"]
        asyncio.run(set_alert_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Price Alert Set", text)
        self.assertIn("85,000", text)

    def test_unified_keys_cmd_dispatch(self):
        import asyncio
        from main import keys_cmd, delta_client

        # 1. /keys no args
        self.mock_ctx.args = []
        asyncio.run(keys_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("MASTER KEYS & BOT CONFIGURATION HUB", text)

        # 2. /keys base india
        self.mock_ctx.args = ["base", "india"]
        asyncio.run(keys_cmd(self.mock_update, self.mock_ctx))
        self.assertIn("india.delta.exchange", delta_client.base_url)

    def test_analyze_cmd_and_chat_routing(self):
        import asyncio
        from main import analyze_cmd, chat

        # 1. Direct /analyze cmd
        self.mock_ctx.args = ["BTCUSD", "15m"]
        asyncio.run(analyze_cmd(self.mock_update, self.mock_ctx))
        reply_calls = [c[0][0] for c in self.mock_update.message.reply_text.call_args_list]
        found_header = any("18 Institutional Agents Analyzing" in c for c in reply_calls)
        found_report = any("18-Agent Institutional Desk Analysis" in c for c in reply_calls)
        self.assertTrue(found_header)
        self.assertTrue(found_report)

        # 2. Plain-text "analysis"
        self.mock_update.message.reply_text.reset_mock()
        self.mock_update.message.text = "analysis"
        asyncio.run(chat(self.mock_update, self.mock_ctx))
        reply_calls = [c[0][0] for c in self.mock_update.message.reply_text.call_args_list]
        self.assertTrue(any("18 Institutional Agents Analyzing" in c for c in reply_calls))


class TestAMDScalpAndMultiTrade(unittest.TestCase):
    """Test suite for AMD Scalp Engine (1m/5m/15m), automated risk management, and multi-trade execution."""

    def setUp(self):
        self.mock_update = MagicMock()
        self.mock_update.effective_chat.id = 12345
        self.mock_update.message.text = ""
        self.mock_update.message.reply_text = AsyncMock()
        self.mock_ctx = MagicMock()
        self.mock_ctx.args = []

    def test_amd_accumulation_detection(self):
        from amd_scalper import detect_accumulation_range

        # Generate 15 candles in a consolidation range [85,000, 85,300]
        candles = []
        for i in range(20):
            candles.append({
                "open": 85100.0,
                "high": 85300.0,
                "low": 85000.0,
                "close": 85150.0,
                "volume": 10.0,
            })

        res = detect_accumulation_range(candles, window=10)
        self.assertIsNotNone(res)
        self.assertEqual(res["range_high"], 85300.0)
        self.assertEqual(res["range_low"], 85000.0)
        self.assertEqual(res["range_eq"], 85150.0)
        self.assertTrue(res["is_compressed"])

    def test_amd_manipulation_sweep(self):
        from amd_scalper import detect_manipulation_sweep

        range_info = {
            "range_high": 85300.0,
            "range_low": 85000.0,
            "range_eq": 85150.0,
            "range_height": 300.0,
        }

        # 5m candle that sweeps below 85000 (e.g. low 84920) but rejects back to 85080
        candles = [
            {"open": 85100.0, "high": 85200.0, "low": 85050.0, "close": 85120.0},
            {"open": 85120.0, "high": 85150.0, "low": 84920.0, "close": 85080.0},
            {"open": 85080.0, "high": 85100.0, "low": 85060.0, "close": 85090.0},
        ]

        manip = detect_manipulation_sweep(candles, range_info, atr_5m=50.0)
        self.assertIsNotNone(manip)
        self.assertEqual(manip["bias"], "BULLISH")
        self.assertEqual(manip["phase"], "MANIPULATION")
        self.assertEqual(manip["sweep_level"], 84920.0)

    def test_amd_1m_distribution_trigger(self):
        from amd_scalper import detect_1m_distribution_trigger

        # 1m candles displaying bullish displacement and breaking recent swing high
        candles_1m = []
        for i in range(12):
            candles_1m.append({
                "open": 85000.0 + i * 5,
                "high": 85020.0 + i * 5,
                "low": 84990.0 + i * 5,
                "close": 85010.0 + i * 5,
            })
        # Add strong green displacement candle breaking above EMA9
        candles_1m.append({
            "open": 85060.0,
            "high": 85140.0,
            "low": 85055.0,
            "close": 85135.0,
        })
        candles_1m.append({
            "open": 85135.0,
            "high": 85150.0,
            "low": 85120.0,
            "close": 85145.0,
        })

        trigger = detect_1m_distribution_trigger(candles_1m, bias="BULLISH", manipulation_sweep_level=84920.0, atr_1m=20.0)
        self.assertIsNotNone(trigger)
        self.assertTrue(trigger["triggered"])
        self.assertEqual(trigger["side"], "buy")
        self.assertGreater(trigger["entry_price"], 85100.0)

    def test_calculate_risk_managed_plan(self):
        from amd_scalper import calculate_risk_managed_plan

        plan = calculate_risk_managed_plan(
            symbol="BTCUSD",
            side="buy",
            entry_price=85100.0,
            invalidation_level=84950.0,
            range_target=85400.0,
            external_target=85700.0,
            atr_1m=25.0,
            balance=10000.0,
            risk_pct=0.015,
        )

        self.assertEqual(plan["side"], "buy")
        self.assertEqual(plan["entry"], 85100.0)
        self.assertLess(plan["sl"], 84950.0)
        self.assertGreaterEqual(plan["tp1"], 85400.0)
        self.assertGreaterEqual(plan["tp2"], 85700.0)
        self.assertGreater(plan["suggested_size"], 0.0)
        self.assertEqual(plan["capital_at_risk"], 150.0)

    def test_auto_trader_multi_positions_and_risk(self):
        from auto_trader import AutoTrader
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        trader = AutoTrader(store_file=temp_path, mode="paper")
        trader.data["positions"] = {}

        # 1. Test set_max_positions and set_risk_per_trade
        self.assertEqual(trader.set_max_positions(5), 5)
        self.assertEqual(trader.set_risk_per_trade(0.02), 0.02)

        # 2. Test calculate_risk_position_size
        size_btc = trader.calculate_risk_position_size("BTCUSD", entry_price=85000.0, sl_price=84500.0)
        self.assertGreater(size_btc, 0.0)

        # 3. Test multi-trade execution on distinct symbols
        pos1 = trader.execute_trade("BTCUSD", "BUY", size=0.01)
        pos2 = trader.execute_trade("XAUTUSD", "SELL", size=0.05)

        self.assertEqual(len(trader.positions), 2)
        summary = trader.get_summary()
        self.assertEqual(summary["open_positions_count"], 2)
        self.assertEqual(summary["max_open_positions"], 5)

        # Clean up
        trader.close_all_positions()
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_amd_commands_and_hub_dispatch(self):
        import asyncio
        from main import amd_cmd, btc_cmd, gold_cmd, trade_cmd, chat

        # 1. /amd
        self.mock_ctx.args = ["BTCUSD"]
        asyncio.run(amd_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AMD SCALP TRADING DESK", text)
        self.assertIn("Multi-Timeframe", text)

        # 2. /btc amd
        self.mock_ctx.args = ["amd"]
        asyncio.run(btc_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AMD SCALP TRADING DESK", text)

        # 3. /gold amd
        self.mock_ctx.args = ["amd"]
        asyncio.run(gold_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AMD SCALP TRADING DESK", text)

        # 4. /trade maxpos 4
        self.mock_ctx.args = ["maxpos", "4"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Max concurrent open positions set to", text)

        # 5. /trade risk 2
        self.mock_ctx.args = ["risk", "2"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Capital risk per trade set to", text)

        # 6. Plain-text "amd scalp" in chat
        self.mock_update.message.reply_text.reset_mock()
        self.mock_update.message.text = "amd scalp"
        asyncio.run(chat(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AMD SCALP TRADING DESK", text)


class TestInstitutionalDeskAndAdvancedRisk(unittest.TestCase):
    def setUp(self):
        self.mock_update = MagicMock()
        self.mock_update.effective_chat.id = 12345678
        self.mock_update.message.reply_text = AsyncMock()
        self.mock_ctx = MagicMock()
        self.mock_ctx.args = []

    def test_market_session_killzones(self):
        from datetime import datetime, timezone
        from market_data import get_market_session

        # 08:30 UTC -> London Open Killzone
        dt_london = datetime(2026, 10, 7, 8, 30, tzinfo=timezone.utc)
        sess_london = get_market_session(dt_london)
        self.assertIn("London Open Killzone", sess_london["session"])
        self.assertTrue(sess_london["is_killzone"])

        # 13:30 UTC -> New York Open Killzone
        dt_ny = datetime(2026, 10, 7, 13, 30, tzinfo=timezone.utc)
        sess_ny = get_market_session(dt_ny)
        self.assertIn("New York Open Killzone", sess_ny["session"])
        self.assertTrue(sess_ny["is_killzone"])

        # 03:00 UTC -> Asian Session
        dt_asia = datetime(2026, 10, 7, 3, 0, tzinfo=timezone.utc)
        sess_asia = get_market_session(dt_asia)
        self.assertIn("Asian Session", sess_asia["session"])
        self.assertFalse(sess_asia["is_killzone"])

    def test_auto_trader_breakeven_protection(self):
        import tempfile
        import os
        from auto_trader import AutoTrader

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        trader = AutoTrader(store_file=temp_path, mode="paper")
        trader.set_auto_breakeven(True)
        self.assertTrue(trader.data["auto_breakeven"])

        # Open Long position
        pos = trader.execute_trade(
            symbol="BTCUSD",
            side="buy",
            size=0.1,
        )
        pid = pos["position_id"]
        entry = pos["entry_price"]
        pos["sl_price"] = round(entry * 0.98, 2)
        pos["sl"] = round(entry * 0.98, 2)
        pos["tp1_price"] = round(entry * 1.015, 2)
        pos["tp1"] = round(entry * 1.015, 2)
        pos["tp2_price"] = round(entry * 1.03, 2)
        pos["tp2"] = round(entry * 1.03, 2)
        trader.save()

        # 1. Price hits TP1 (entry * 1.018) -> moves SL to Breakeven (entry) and keeps position open for TP2 runner
        prices = {"BTCUSD": round(entry * 1.018, 2)}
        exits = trader.check_open_positions_for_exits(prices)
        self.assertEqual(len(exits), 0)  # Remains open for TP2 runner!
        updated_pos = trader.data["positions"][pid]
        self.assertTrue(updated_pos["tp1_hit"])
        self.assertTrue(updated_pos["is_breakeven"])
        self.assertEqual(updated_pos["sl_price"], updated_pos["entry_price"])

        # 2. Price continues to TP2 (entry * 1.035) -> exits with TP2 hit
        prices = {"BTCUSD": round(entry * 1.035, 2)}
        exits = trader.check_open_positions_for_exits(prices)
        self.assertEqual(len(exits), 1)
        self.assertIn("TP2 Hit", exits[0]["exit_reason"])

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_auto_trader_trailing_stop_loss(self):
        import tempfile
        import os
        from auto_trader import AutoTrader

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        trader = AutoTrader(store_file=temp_path, mode="paper")
        trader.set_trailing_sl(True, pct=0.01)  # 1% trailing
        self.assertTrue(trader.data["trailing_sl"])

        # Open Long position
        pos = trader.execute_trade(
            symbol="BTCUSD",
            side="buy",
            size=0.1,
        )
        pid = pos["position_id"]
        entry = pos["entry_price"]
        pos["sl_price"] = round(entry * 0.98, 2)
        pos["sl"] = round(entry * 0.98, 2)
        pos["tp1_price"] = round(entry * 1.10, 2)
        pos["tp1"] = round(entry * 1.10, 2)
        trader.save()

        # Price surges to entry * 1.02 (> 1.2% profit) -> Trailing SL trails at (entry * 1.02) * 0.99
        surge_price = round(entry * 1.02, 2)
        prices = {"BTCUSD": surge_price}
        trader.check_open_positions_for_exits(prices)
        updated_pos = trader.data["positions"][pid]
        self.assertTrue(updated_pos.get("is_trailing"))
        expected_sl = round(surge_price * 0.99, 2)
        self.assertAlmostEqual(updated_pos["sl_price"], expected_sl, delta=0.5)

        # Price pulls back below trailed stop -> position closed
        prices = {"BTCUSD": round(expected_sl - 10.0, 2)}
        exits = trader.check_open_positions_for_exits(prices)
        self.assertEqual(len(exits), 1)
        self.assertIn("STOP LOSS", exits[0]["exit_reason"])

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_institutional_dashboard_card(self):
        from auto_trader import AutoTrader
        trader = AutoTrader(mode="paper")
        card = trader.format_institutional_dashboard()
        self.assertIn("INSTITUTIONAL TRADING DESK DASHBOARD", card)
        self.assertIn("Bot Status:", card)
        self.assertIn("Session:", card)
        self.assertIn("PORTFOLIO & CAPITAL:", card)
        self.assertIn("RISK MANAGEMENT RULES:", card)
        self.assertIn("SELF-LEARNING DESK:", card)

    def test_status_command_and_quick_chat_routes(self):
        import asyncio
        from main import status_cmd, trade_cmd, chat

        # 1. /status command
        self.mock_update.message.reply_text.reset_mock()
        asyncio.run(status_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("INSTITUTIONAL TRADING DESK DASHBOARD", text)

        # 2. /trade status
        self.mock_update.message.reply_text.reset_mock()
        self.mock_ctx.args = ["status"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("INSTITUTIONAL TRADING DESK DASHBOARD", text)

        # 3. /trade be on
        self.mock_update.message.reply_text.reset_mock()
        self.mock_ctx.args = ["be", "on"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Breakeven Stop-Loss ENABLED", text)

        # 4. /trade trail on
        self.mock_update.message.reply_text.reset_mock()
        self.mock_ctx.args = ["trail", "on"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Dynamic Trailing Stop-Loss ENABLED", text)

        # 5. Plain-text "status" in chat
        self.mock_update.message.reply_text.reset_mock()
        self.mock_update.message.text = "status"
        asyncio.run(chat(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("INSTITUTIONAL TRADING DESK DASHBOARD", text)

        # 6. Plain-text "buy btc 0.01" in chat
        from main import auto_trader
        auto_trader.close_all_positions()
        self.mock_update.message.reply_text.reset_mock()
        self.mock_update.message.text = "buy btc 0.01"
        asyncio.run(chat(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TRADE EXECUTED", text)


class TestAutoTradeLotSizeManagement(unittest.TestCase):
    """Test suite for Auto-Trade Lot Size configuration and strict bot execution."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        self.trader = AutoTrader(store_file=self.temp_file.name, default_size=1.0)
        self.mock_update = MagicMock()
        self.mock_update.effective_chat.id = 888
        self.mock_update.message.reply_text = AsyncMock()
        self.mock_ctx = MagicMock()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_lot_size_get_set_global(self):
        """Verify global lot size setter, getter, and persistence."""
        self.assertEqual(self.trader.get_lot_size(), 1.0)
        self.assertEqual(self.trader.data["lot_size_mode"], "custom")

        new_sz = self.trader.set_lot_size(0.05)
        self.assertEqual(new_sz, 0.05)
        self.assertEqual(self.trader.get_lot_size(), 0.05)
        self.assertEqual(self.trader.data["default_size"], 0.05)
        self.assertEqual(self.trader.data["lot_size"], 0.05)

        # Verify disk persistence
        reloaded = AutoTrader(store_file=self.temp_file.name)
        self.assertEqual(reloaded.get_lot_size(), 0.05)
        self.assertEqual(reloaded.data["lot_size_mode"], "custom")

    def test_lot_size_per_symbol_overrides(self):
        """Verify per-symbol lot size overrides and fallbacks."""
        self.trader.set_lot_size(0.05)  # Global size
        self.trader.set_lot_size(0.01, symbol="BTCUSD")  # BTC override
        self.trader.set_lot_size(0.5, symbol="XAUTUSD")  # Gold override

        self.assertEqual(self.trader.get_effective_lot_size("BTCUSD"), 0.01)
        self.assertEqual(self.trader.get_effective_lot_size("XAUTUSD"), 0.5)
        self.assertEqual(self.trader.get_effective_lot_size("ETHUSD"), 0.05)  # Fallback to global

        # Clear override
        cleared = self.trader.remove_symbol_lot_size("BTCUSD")
        self.assertTrue(cleared)
        self.assertEqual(self.trader.get_effective_lot_size("BTCUSD"), 0.05)

    def test_lot_size_mode_toggle(self):
        """Verify switching between custom lot size mode and dynamic risk_pct mode."""
        self.trader.set_lot_size_mode("risk_pct")
        self.assertEqual(self.trader.data["lot_size_mode"], "risk_pct")

        self.trader.set_lot_size_mode("custom")
        self.assertEqual(self.trader.data["lot_size_mode"], "custom")

        with self.assertRaises(ValueError):
            self.trader.set_lot_size_mode("invalid_mode")

    def test_trade_execution_strictly_follows_custom_lot_size(self):
        """Verify that execute_trade and execute_amd_trade strictly follow user's lot size."""
        self.trader.set_lot_size(0.02, symbol="BTCUSD")
        self.trader.close_all_positions()

        # 1. execute_trade without explicit size
        pos = self.trader.execute_trade("BTCUSD", "BUY")
        self.assertEqual(pos["size"], 0.02)
        self.trader.close_all_positions()

        # 2. execute_trade with explicit override should still work
        pos_custom = self.trader.execute_trade("BTCUSD", "BUY", size=0.1)
        self.assertEqual(pos_custom["size"], 0.1)
        self.trader.close_all_positions()

        # 3. execute_amd_trade follows user's configured lot size
        amd_res = self.trader.execute_amd_trade("BTCUSD", force=True)
        self.assertEqual(amd_res["status"], "executed")
        self.assertEqual(amd_res["trade"]["size"], 0.02)
        self.trader.close_all_positions()

    def test_telegram_trade_size_commands(self):
        """Verify /trade size, /size, and /lotsize commands."""
        from main import trade_cmd, size_cmd, chat

        # 1. /trade size without args (dashboard view)
        self.mock_update.message.reply_text.reset_mock()
        self.mock_ctx.args = ["size"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AUTO-TRADE LOT SIZE CONFIGURATION", text)

        # 2. /trade size 0.05 (set global size)
        self.mock_update.message.reply_text.reset_mock()
        self.mock_ctx.args = ["size", "0.05"]
        asyncio.run(trade_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Global Auto-Trade Lot Size set to 0.05", text)

        # 3. /size btc 0.01 (shortcut command with pair override)
        self.mock_update.message.reply_text.reset_mock()
        self.mock_ctx.args = ["btc", "0.01"]
        asyncio.run(size_cmd(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Lot Size for BTCUSD set to 0.01", text)

        # 4. Plain text in chat "set lot size 0.03"
        self.mock_update.message.reply_text.reset_mock()
        self.mock_update.message.text = "set lot size 0.03"
        asyncio.run(chat(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Global Auto-Trade Lot Size set to 0.03", text)

        # 5. Plain text in chat "btc size 0.015"
        self.mock_update.message.reply_text.reset_mock()
        self.mock_update.message.text = "btc size 0.015"
        asyncio.run(chat(self.mock_update, self.mock_ctx))
        text = self.mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Lot Size for BTCUSD set to 0.015", text)


class TestMemoryAndConsolidationEngine(unittest.TestCase):
    """Test suite for autonomous cognitive memory, procedural rule adaptation, and reflection."""

    def setUp(self):
        self.tmp_store = "test_memory_store_tmp.json"
        if os.path.exists(self.tmp_store):
            os.remove(self.tmp_store)

    def tearDown(self):
        if os.path.exists(self.tmp_store):
            os.remove(self.tmp_store)

    def test_memory_extraction_schema_and_validation(self):
        import json
        from memory_manager import MemoryExtraction
        raw_json = json.dumps({
            "new_facts": ["User trades with $10,000 balance", "Prefers London session"],
            "conflicts_to_remove": ["Old risk was 5%"],
            "behavioral_corrections": ["Never risk more than 1.5%", "Always display R:R ratio"]
        })
        extracted = MemoryExtraction.model_validate_json(raw_json)
        self.assertEqual(len(extracted.new_facts), 2)
        self.assertEqual(len(extracted.conflicts_to_remove), 1)
        self.assertEqual(len(extracted.behavioral_corrections), 2)
        self.assertIn("Never risk more than 1.5%", extracted.behavioral_corrections)

        # Also test with markdown code fence wrapping
        from memory_manager import _clean_json_text
        markdown_json = f"```json\n{raw_json}\n```"
        cleaned = _clean_json_text(markdown_json)
        extracted_md = MemoryExtraction.model_validate_json(cleaned)
        self.assertEqual(len(extracted_md.new_facts), 2)

    def test_memory_manager_lifecycle_and_invalidation(self):
        from unittest.mock import patch
        from memory_manager import MemoryManager, MemoryExtraction
        mgr = MemoryManager(store_path=self.tmp_store)
        mgr.reset()

        # Add initial rule and fact
        mgr.add_rule("Old risk was 5%")
        mgr.semantic_facts.append("User is beginner")
        mgr.record_interaction("I am now an advanced trader with $20k balance", "Understood!")
        mgr.record_interaction("Please never risk more than 1.5% and discard old risk rule", "Noted!")

        # Mock reflect_and_consolidate
        with patch("memory_manager.reflect_and_consolidate") as mock_reflect:
            mock_reflect.return_value = MemoryExtraction(
                new_facts=["User has $20k balance"],
                conflicts_to_remove=["Old risk was 5%"],
                behavioral_corrections=["Never risk more than 1.5%"]
            )
            extraction = mgr.consolidate(force=True)
            self.assertEqual(len(extraction.behavioral_corrections), 1)
            # Verify old rule removed and new rule added
            self.assertNotIn("Old risk was 5%", mgr.procedural_rules)
            self.assertIn("Never risk more than 1.5%", mgr.procedural_rules)
            self.assertIn("User has $20k balance", mgr.semantic_facts)

        # Check prompt injection
        injection = mgr.get_system_instructions_injection()
        self.assertIn("DYNAMIC PROCEDURAL MEMORY", injection)
        self.assertIn("Never risk more than 1.5%", injection)

        # Check report format
        report = mgr.format_memory_report()
        self.assertIn("Cognitive Memory & Behavioral Rules", report)
        self.assertIn("Never risk more than 1.5%", report)

    def test_rules_and_reflect_telegram_commands(self):
        from unittest.mock import AsyncMock, MagicMock
        from main import rules_cmd, reflect_cmd, memory_manager, chat
        mock_update = MagicMock()
        mock_update.effective_chat.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_ctx = MagicMock()

        # 1. /rules report
        mock_ctx.args = []
        asyncio.run(rules_cmd(mock_update, mock_ctx))
        text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Cognitive Memory & Behavioral Rules", text)

        # 2. /rules add <rule>
        mock_update.message.reply_text.reset_mock()
        mock_ctx.args = ["add", "Always", "confirm", "15m", "trend"]
        asyncio.run(rules_cmd(mock_update, mock_ctx))
        text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Behavioral Rule Added", text)
        self.assertIn("Always confirm 15m trend", memory_manager.procedural_rules)

        # 3. Plain text chat routing "rules"
        mock_update.message.reply_text.reset_mock()
        mock_update.message.text = "rules"
        asyncio.run(chat(mock_update, mock_ctx))
        text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Cognitive Memory & Behavioral Rules", text)

        # 4. /rules del <rule>
        mock_update.message.reply_text.reset_mock()
        mock_ctx.args = ["del", "Always confirm 15m trend"]
        asyncio.run(rules_cmd(mock_update, mock_ctx))
        text = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Rule Removed", text)


if __name__ == "__main__":
    unittest.main()




