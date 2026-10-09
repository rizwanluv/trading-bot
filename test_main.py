"""
Unit tests for main.py trading bot.
"""
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from types import SimpleNamespace

import main
import auto_trade


class TestTradingBot(unittest.TestCase):
    def setUp(self):
        main.histories.clear()

    def test_load_dotenv(self):
        test_env_path = "/workspace/bright-darwin/.test_env"
        try:
            with open(test_env_path, "w") as f:
                f.write("# comment\nTEST_CUSTOM_VAR=hello_world\nINVALID_LINE\n")
            main.load_dotenv(test_env_path)
            self.assertEqual(os.environ.get("TEST_CUSTOM_VAR"), "hello_world")
        finally:
            if os.path.exists(test_env_path):
                os.remove(test_env_path)
            os.environ.pop("TEST_CUSTOM_VAR", None)

    @patch("requests.get")
    def test_get_price_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "mark_price": "4135.50",
                "close": 4135.0,
                "high": 4150.0,
                "low": 4100.0,
                "open": 4110.0,
            },
        }
        mock_get.return_value = mock_resp

        res = main.get_price("XAUTUSD")
        self.assertIn("XAUTUSD:", res)
        self.assertIn("mark 4135.50", res)
        self.assertIn("last 4135.0", res)

    @patch("requests.get")
    def test_get_price_symbol_alias(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "mark_price": "80000",
                "close": 80000,
                "high": 81000,
                "low": 79000,
                "open": 79500,
            },
        }
        mock_get.return_value = mock_resp

        # 'btc' should map to 'BTCUSD'
        res = main.get_price("btc")
        self.assertIn("BTCUSD:", res)

    @patch("requests.get")
    def test_get_price_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "result": None}
        mock_get.return_value = mock_resp

        res = main.get_price("INVALID")
        self.assertIn("not found on Delta Exchange", res)

    @patch("requests.get")
    def test_get_price_network_error(self, mock_get):
        mock_get.side_effect = Exception("Connection timeout")
        res = main.get_price("XAUTUSD")
        self.assertIn("Price unavailable (Connection timeout)", res)

    def test_call_gemini_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                main.call_gemini([{"role": "user", "parts": [{"text": "hi"}]}])

    @patch("requests.post")
    def test_call_gemini_rest_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gold is bullish above support."}],
                        "role": "model",
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reply = main.call_gemini_rest([{"role": "user", "parts": [{"text": "analyze"}]}])
            self.assertEqual(reply, "Gold is bullish above support.")

    @patch("requests.post")
    def test_call_gemini_rest_fallback_on_404(self, mock_post):
        # 1st call returns 404, 2nd call returns 200
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.json.return_value = {"error": {"message": "Model not found"}}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Fallback model response."}],
                        "role": "model",
                    }
                }
            ]
        }
        mock_post.side_effect = [resp_404, resp_200]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            reply = main.call_gemini_rest([{"role": "user", "parts": [{"text": "test"}]}])
            self.assertEqual(reply, "Fallback model response.")
            self.assertEqual(mock_post.call_count, 2)

    def test_diagnostics(self):
        # Diagnostics should execute without raising exceptions
        result = main.run_diagnostics()
        self.assertIsInstance(result, bool)


class TestAsyncHandlers(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        main.histories.clear()

    async def test_reply_safely_chunking(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock())
        )
        long_message = "A" * 8500
        await main.reply_safely(mock_update, long_message)
        # Should be split into 3 chunks: 4000, 4000, 500
        self.assertEqual(mock_update.message.reply_text.call_count, 3)

    async def test_start_command(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock())
        )
        ctx = SimpleNamespace(args=[])
        await main.start(mock_update, ctx)
        mock_update.message.reply_text.assert_called_once()
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Trading assistant ready", sent)

    async def test_price_command(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock())
        )
        ctx = SimpleNamespace(args=["BTC"])
        with patch("main.get_price", return_value="BTCUSD: mark 80000"):
            await main.price(mock_update, ctx)
            mock_update.message.reply_text.assert_called_once_with("BTCUSD: mark 80000")

    async def test_reset_command(self):
        mock_update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=12345),
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        main.histories[12345] = [{"role": "user", "parts": [{"text": "prev"}]}]
        ctx = SimpleNamespace(args=[])
        await main.reset(mock_update, ctx)
        self.assertNotIn(12345, main.histories)
        mock_update.message.reply_text.assert_called_once_with("Conversation cleared.")

    async def test_chat_success(self):
        mock_update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=999),
            message=SimpleNamespace(
                text="Should I buy gold?",
                reply_text=AsyncMock(),
            ),
        )
        ctx = SimpleNamespace(args=[])
        with patch("main.call_gemini", return_value="Gold looks strong."):
            await main.chat(mock_update, ctx)
            mock_update.message.reply_text.assert_called_once_with("Gold looks strong.")
            # Verify history has user and model turns
            self.assertEqual(len(main.histories[999]), 2)
            self.assertEqual(main.histories[999][0]["role"], "user")
            self.assertEqual(main.histories[999][1]["role"], "model")

    async def test_chat_error_rolls_back_history(self):
        mock_update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=999),
            message=SimpleNamespace(
                text="Should I buy gold?",
                reply_text=AsyncMock(),
            ),
        )
        ctx = SimpleNamespace(args=[])
        with patch("main.call_gemini", side_effect=RuntimeError("API quota exceeded")):
            await main.chat(mock_update, ctx)
            sent = mock_update.message.reply_text.call_args[0][0]
            self.assertIn("Error: API quota exceeded", sent)
            # History should NOT contain the failed user turn
            self.assertEqual(len(main.histories[999]), 0)


class TestAutoTradeModule(unittest.TestCase):
    def setUp(self):
        import auto_trade
        self.tmp_config = "/workspace/bright-darwin/.test_autotrade_config.json"
        self.tmp_hist = "/workspace/bright-darwin/.test_trades_history.json"
        self.config = auto_trade.AutoTradeConfig(
            config_file=self.tmp_config,
            trades_history_file=self.tmp_hist,
        )
        self.trader = auto_trade.AutoTrader(config=self.config)

    def tearDown(self):
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)
        if os.path.exists(self.tmp_hist):
            os.remove(self.tmp_hist)

    def test_config_save_and_load(self):
        import auto_trade
        self.config.enabled = True
        self.config.lot_size = 0.05
        self.config.tp_value = 3.0
        self.config.sl_value = 1.5
        self.config.save()

        loaded = auto_trade.AutoTradeConfig.load(self.tmp_config)
        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.lot_size, 0.05)
        self.assertEqual(loaded.tp_value, 3.0)
        self.assertEqual(loaded.sl_value, 1.5)

    def test_toggle_on_off(self):
        self.assertFalse(self.trader.config.enabled)
        state, msg = self.trader.toggle(True, chat_id=111)
        self.assertTrue(state)
        self.assertTrue(self.trader.config.enabled)
        self.assertEqual(self.trader.config.notify_chat_id, 111)
        self.assertIn("ENABLED", msg)

        state, msg = self.trader.toggle(False)
        self.assertFalse(state)
        self.assertFalse(self.trader.config.enabled)
        self.assertIn("DISABLED", msg)

    def test_set_lot_size(self):
        ok, msg = self.trader.set_lot_size(0.1, mode="fixed")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.lot_size, 0.1)
        self.assertEqual(self.trader.config.lot_mode, "fixed")

        # Risk mode
        ok, msg = self.trader.set_lot_size(2.5, mode="risk_pct")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.lot_size, 2.5)
        self.assertEqual(self.trader.config.lot_mode, "risk_pct")

        # Invalid size <= 0
        ok, msg = self.trader.set_lot_size(0)
        self.assertFalse(ok)
        self.assertIn("greater than 0", msg)

    def test_set_tp_sl(self):
        # RR mode
        ok, msg = self.trader.set_tp_sl(2.5, 1.0, tp_mode="rr", sl_mode="swing")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.tp_value, 2.5)
        self.assertEqual(self.trader.config.sl_value, 1.0)
        self.assertEqual(self.trader.config.tp_mode, "rr")

        # Points mode
        ok, msg = self.trader.set_tp_sl(50, 25, tp_mode="pts", sl_mode="pts")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.tp_mode, "pts")
        self.assertEqual(self.trader.config.sl_mode, "pts")

        # Invalid values
        ok, msg = self.trader.set_tp_sl(-1, 2)
        self.assertFalse(ok)
        ok, msg = self.trader.set_tp_sl(2, 1, tp_mode="invalid")
        self.assertFalse(ok)

    def test_position_pnl_and_close(self):
        import auto_trade
        pos = auto_trade.AutoTradePosition(
            id="T1",
            symbol="XAUTUSD",
            direction="LONG",
            entry_price=4000.0,
            stop_loss=3980.0,
            take_profit_1=4040.0,
            take_profit_2=None,
            lot_size=1.0,
            entry_time="now",
            strategy="Ensemble (Pro)",
            reason="test",
            highest_price=4000.0,
            lowest_price=4000.0,
        )
        self.trader.position = pos

        # Test PnL calculation
        self.assertEqual(pos.current_pnl(4020.0), 20.0)
        self.assertEqual(pos.current_pnl(3990.0), -10.0)

        # Test manual close
        close_msg = self.trader.close_current_position(4030.0, reason="MANUAL")
        self.assertIsNotNone(close_msg)
        self.assertIsNone(self.trader.position)
        self.assertEqual(len(self.trader.closed_trades), 1)
        self.assertEqual(self.trader.closed_trades[0].pnl, 30.0)

    def test_step_tp_and_sl_execution(self):
        import pandas as pd
        import auto_trade

        # 1. Test TP execution
        pos_long = auto_trade.AutoTradePosition(
            id="T_TP",
            symbol="XAUTUSD",
            direction="LONG",
            entry_price=4000.0,
            stop_loss=3950.0,
            take_profit_1=4050.0,
            take_profit_2=None,
            lot_size=1.0,
            entry_time="now",
            strategy="Ensemble (Pro)",
            reason="test",
            highest_price=4000.0,
            lowest_price=4000.0,
        )
        self.trader.position = pos_long

        # Mock candle with high >= TP
        mock_df = pd.DataFrame(
            [{"open": 4020.0, "high": 4060.0, "low": 4010.0, "close": 4055.0, "volume": 100}],
            index=pd.date_range("2026-01-01", periods=1, freq="1min"),
        )
        with patch.object(self.trader, "fetch_candles", return_value=mock_df):
            notes = self.trader.step()
            self.assertEqual(len(notes), 1)
            self.assertIn("TAKE PROFIT HIT", notes[0])
            self.assertIsNone(self.trader.position)

        # 2. Test SL execution
        pos_short = auto_trade.AutoTradePosition(
            id="T_SL",
            symbol="XAUTUSD",
            direction="SHORT",
            entry_price=4000.0,
            stop_loss=4050.0,
            take_profit_1=3950.0,
            take_profit_2=None,
            lot_size=1.0,
            entry_time="now",
            strategy="Ensemble (Pro)",
            reason="test",
            highest_price=4000.0,
            lowest_price=4000.0,
        )
        self.trader.position = pos_short

        # Mock candle with high >= SL
        mock_df_sl = pd.DataFrame(
            [{"open": 4020.0, "high": 4060.0, "low": 4010.0, "close": 4055.0, "volume": 100}],
            index=pd.date_range("2026-01-01", periods=1, freq="1min"),
        )
        with patch.object(self.trader, "fetch_candles", return_value=mock_df_sl):
            notes = self.trader.step()
            self.assertEqual(len(notes), 1)
            self.assertIn("STOP LOSS HIT", notes[0])
            self.assertIsNone(self.trader.position)


class TestAutoTradeTelegramHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_autotrade_command_flow(self):
        mock_update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=777),
            message=SimpleNamespace(reply_text=AsyncMock()),
        )

        # 1. Status query
        ctx = SimpleNamespace(args=[])
        await main.autotrade_command(mock_update, ctx)
        mock_update.message.reply_text.assert_called_once()
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("AUTO TRADING DASHBOARD", sent)

        # 2. Turn ON
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["on"])
        await main.autotrade_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("ENABLED (ON)", sent)

        # 3. Turn OFF
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["off"])
        await main.autotrade_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("DISABLED (OFF)", sent)

        # 4. Close when no position
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["close"])
        await main.autotrade_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("No active position", sent)

    async def test_lotsize_command_flow(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )

        # 1. Query lot size
        ctx = SimpleNamespace(args=[])
        await main.lotsize_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Current Lot Size Configuration", sent)

        # 2. Set numeric lot size
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["0.05"])
        await main.lotsize_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("0.05", sent)

        # 3. Set risk percentage
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["risk", "2%"])
        await main.lotsize_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Risk % lot sizing", sent)

    async def test_tpsl_command_flow(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )

        # 1. Query TP/SL
        ctx = SimpleNamespace(args=[])
        await main.tpsl_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Current Strategy TP / SL Configuration", sent)

        # 2. Set RR
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["2.5", "1.0"])
        await main.tpsl_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Take Profit: 2.5", sent)
        self.assertIn("Stop Loss: 1.0", sent)

        # 3. Set points
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["pts", "60", "30"])
        await main.tpsl_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("mode: pts", sent)

    async def test_help_contains_new_commands(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=[])
        await main.help_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("/btc", sent)
        self.assertIn("/gold", sent)
        self.assertIn("/symbol", sent)
        self.assertIn("/autotrade", sent)
        self.assertIn("/lotsize", sent)
        self.assertIn("/tpsl", sent)

    @patch("requests.get")
    async def test_btc_command(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "mark_price": "82500.00",
                "close": 82510.0,
                "high": 83000.0,
                "low": 81000.0,
                "open": 82000.0,
                "volume": 2500.0,
            },
        }
        mock_get.return_value = mock_resp

        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=[])
        await main.btc_command(mock_update, ctx)
        mock_update.message.reply_text.assert_called_once()
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Bitcoin (BTC)", sent)
        self.assertIn("82,500.00", sent)

    @patch("requests.get")
    async def test_gold_command(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "mark_price": "4135.50",
                "close": 4135.0,
                "high": 4150.0,
                "low": 4100.0,
                "open": 4110.0,
                "volume": 500.0,
            },
        }
        mock_get.return_value = mock_resp

        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=[])
        await main.gold_command(mock_update, ctx)
        mock_update.message.reply_text.assert_called_once()
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Gold (XAU)", sent)
        self.assertIn("4,135.50", sent)

    @patch("requests.get")
    async def test_symbol_command(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "mark_price": "82000.00",
                "close": 82000.0,
                "high": 83000.0,
                "low": 81000.0,
                "open": 81500.0,
                "volume": 1200.0,
            },
        }
        mock_get.return_value = mock_resp

        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )

        # 1. Query current symbol
        ctx = SimpleNamespace(args=[])
        await main.symbol_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Active Auto-Trade Symbol", sent)

        # 2. Switch symbol to btc
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["btc"])
        await main.symbol_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Auto trade symbol set to: BTCUSD", sent)
        self.assertIn("Bitcoin (BTC)", sent)


class TestCryptoAndBtcStrategies(unittest.TestCase):
    def test_indicators_pro_crypto_sessions(self):
        from trading_strategy_indicators_pro import IndicatorsProStrategy, make_data
        from datetime import datetime, time

        # Crypto (BTC) trades 24/7 without forex killzone gating
        strat = IndicatorsProStrategy(symbol="BTCUSD")
        self.assertTrue(strat.is_crypto)
        # Any arbitrary time (e.g. Asian session / midnight) is OK for crypto
        t_asian = datetime(2026, 1, 1, 3, 15)
        self.assertTrue(strat.session_ok(t_asian))
        self.assertTrue(strat.killzone(t_asian))

        # Check symbol update
        strat.set_symbol("XAUTUSD")
        self.assertFalse(strat.is_crypto)

    def test_ai_learning_crypto_support(self):
        import ai_bot_learning
        bot = ai_bot_learning.AIBotLearning(symbol="BTCUSD")
        self.assertTrue(bot.is_crypto)
        from datetime import datetime
        t_weekend = datetime(2026, 1, 3, 4, 0)
        self.assertTrue(bot.session_ok(t_weekend))

    def test_make_data_btc_scaling(self):
        from trading_strategy_indicators_pro import make_data
        dfs = make_data(100, symbol="BTCUSD")
        df1 = dfs[0]
        self.assertGreater(df1["close"].mean(), 50000.0)

    def test_auto_trade_symbol_btc_switching(self):
        import auto_trade
        cfg = auto_trade.AutoTradeConfig(config_file="/workspace/bright-darwin/.test_switch_cfg.json")
        trader = auto_trade.AutoTrader(config=cfg)
        try:
            msg = trader.set_symbol("btc")
            self.assertEqual(trader.config.symbol, "BTCUSD")
            self.assertTrue(trader._strategy_pro.is_crypto)
            self.assertIn("BTCUSD", msg)

            # Test synthetic candles generation for BTC
            with patch("requests.get", side_effect=Exception("offline")):
                candles = trader.fetch_candles("BTCUSD", count=50)
                self.assertEqual(len(candles), 50)
                self.assertGreater(candles["close"].iloc[0], 50000.0)
        finally:
            if os.path.exists(cfg.config_file):
                os.remove(cfg.config_file)

    def test_trailing_stop_in_step(self):
        import auto_trade
        import pandas as pd
        cfg = auto_trade.AutoTradeConfig(
            config_file="/workspace/bright-darwin/.test_trail_cfg.json",
            trailing_sl=True,
            symbol="BTCUSD",
        )
        trader = auto_trade.AutoTrader(config=cfg)
        try:
            pos = auto_trade.AutoTradePosition(
                id="T_TRAIL",
                symbol="BTCUSD",
                direction="LONG",
                entry_price=80000.0,
                stop_loss=79000.0,
                take_profit_1=83000.0,
                take_profit_2=None,
                lot_size=0.1,
                entry_time="now",
                strategy="Ensemble (Pro)",
                reason="test",
                highest_price=80000.0,
                lowest_price=80000.0,
            )
            trader.position = pos

            # Price moves up to 81500 (profit > risk of 1000)
            mock_df = pd.DataFrame(
                [{"open": 81000.0, "high": 81600.0, "low": 80900.0, "close": 81500.0, "volume": 100}],
                index=pd.date_range("2026-01-01", periods=1, freq="1min"),
            )
            with patch.object(trader, "fetch_candles", return_value=mock_df):
                notes = trader.step()
                # Trailing SL should have moved up to 80500 (81500 - 1000)
                self.assertGreater(trader.position.stop_loss, 79000.0)
                self.assertTrue(any("Trailing Stop Moved" in n for n in notes))
        finally:
            if os.path.exists(cfg.config_file):
                os.remove(cfg.config_file)

    def test_daily_risk_guard_in_step(self):
        import auto_trade
        import pandas as pd
        cfg = auto_trade.AutoTradeConfig(
            config_file="/workspace/bright-darwin/.test_risk_cfg.json",
            enabled=True,
            equity=10000.0,
            max_daily_loss_pct=2.0,  # Max loss $200
        )
        trader = auto_trade.AutoTrader(config=cfg)
        try:
            # Simulate a large loss closed today ($250 loss)
            from datetime import datetime, timezone
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d 10:00:00 UTC")
            closed_loss = auto_trade.AutoTradePosition(
                id="T_LOSS",
                symbol="BTCUSD",
                direction="LONG",
                entry_price=80000.0,
                stop_loss=78000.0,
                take_profit_1=84000.0,
                take_profit_2=None,
                lot_size=0.1,
                entry_time="now",
                strategy="Ensemble (Pro)",
                reason="test",
                highest_price=80000.0,
                lowest_price=77500.0,
                exit_price=77500.0,
                exit_time=today_str,
                exit_reason="SL",
                pnl=-250.0,
            )
            trader.closed_trades.append(closed_loss)

            mock_df = pd.DataFrame(
                [{"open": 80000.0, "high": 80100.0, "low": 79900.0, "close": 80000.0, "volume": 100}],
                index=pd.date_range("2026-01-01", periods=1, freq="1min"),
            )
            with patch.object(trader, "fetch_candles", return_value=mock_df):
                notes = trader.step()
                # Should trigger risk guard and disable trading
                self.assertFalse(trader.config.enabled)
                self.assertTrue(any("Risk Guard Triggered" in n for n in notes))
        finally:
            if os.path.exists(cfg.config_file):
                os.remove(cfg.config_file)

    def test_trades_history_persistence(self):
        import auto_trade
        tmp_cfg = "/workspace/bright-darwin/.test_hist_cfg.json"
        tmp_hist = "/workspace/bright-darwin/.test_trades_hist.json"
        cfg = auto_trade.AutoTradeConfig(
            config_file=tmp_cfg,
            trades_history_file=tmp_hist,
        )
        trader = auto_trade.AutoTrader(config=cfg)
        try:
            # Open position and close it
            pos = auto_trade.AutoTradePosition(
                id="T_PERSIST",
                symbol="BTCUSD",
                direction="LONG",
                entry_price=80000.0,
                stop_loss=79000.0,
                take_profit_1=82000.0,
                take_profit_2=None,
                lot_size=0.1,
                entry_time="now",
                strategy="Ensemble (Pro)",
                reason="test",
                highest_price=80000.0,
                lowest_price=80000.0,
            )
            trader.position = pos
            trader.close_current_position(81000.0, reason="TP1")

            # Verify file was written
            self.assertTrue(os.path.exists(tmp_hist))

            # New trader loading same history file
            trader2 = auto_trade.AutoTrader(config=cfg)
            self.assertEqual(len(trader2.closed_trades), 1)
            self.assertEqual(trader2.closed_trades[0].id, "T_PERSIST")
            self.assertEqual(trader2.closed_trades[0].pnl, 100.0)
        finally:
            if os.path.exists(tmp_cfg):
                os.remove(tmp_cfg)
            if os.path.exists(tmp_hist):
                os.remove(tmp_hist)


class TestNewTelegramHandlers(unittest.IsolatedAsyncioTestCase):
    @patch("requests.get")
    async def test_eth_and_sol_commands(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "result": {
                "mark_price": "2750.00",
                "close": 2750.0,
                "high": 2800.0,
                "low": 2700.0,
                "open": 2720.0,
                "volume": 850.0,
            },
        }
        mock_get.return_value = mock_resp

        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=[])

        # Test /eth
        await main.eth_command(mock_update, ctx)
        sent_eth = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("ETHUSD", sent_eth)

        # Test /sol
        mock_update.message.reply_text.reset_mock()
        mock_resp.json.return_value["result"]["mark_price"] = "175.50"
        await main.sol_command(mock_update, ctx)
        sent_sol = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("SOLUSD", sent_sol)

    async def test_analyze_command(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=["BTC"])
        await main.analyze_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TECHNICAL ANALYSIS", sent)
        self.assertIn("RSI", sent)
        self.assertIn("Confluence Score", sent)

    async def test_position_and_pnl_commands(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=[])

        # /position
        await main.position_command(mock_update, ctx)
        sent_pos = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Active Position", sent_pos)

        # /pnl
        mock_update.message.reply_text.reset_mock()
        await main.pnl_command(mock_update, ctx)
        sent_pnl = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Trading Performance & PnL Report", sent_pnl)

    async def test_trailing_and_risk_commands(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )

        # /trailing on
        ctx = SimpleNamespace(args=["on"])
        await main.trailing_command(mock_update, ctx)
        sent_trail = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Trailing Stop Loss is now ENABLED", sent_trail)

        # /risk 4.0
        mock_update.message.reply_text.reset_mock()
        ctx = SimpleNamespace(args=["4.0"])
        await main.risk_command(mock_update, ctx)
        sent_risk = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("4.0%", sent_risk)

    async def test_menu_command(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
        )
        ctx = SimpleNamespace(args=[])
        await main.menu_command(mock_update, ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TRADING BOT COMMAND MENU", sent)
        self.assertIn("/analyze", sent)
        self.assertIn("/trailing", sent)
        self.assertIn("/entry", sent)
        self.assertIn("/levels", sent)
        self.assertIn("/calc", sent)
        self.assertIn("/execute", sent)

    async def test_entry_and_execute_commands(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock(), text="/entry"),
            effective_chat=SimpleNamespace(id=12345),
        )
        ctx = SimpleNamespace(args=["BTCUSD", "LONG"])
        await main.entry_command(mock_update, ctx)
        sent_entry = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("PINPOINT TRADE ENTRY & TARGET ANALYZER", sent_entry)
        self.assertIn(12345, main.latest_trade_plans)

        # /execute command
        mock_update.message.reply_text.reset_mock()
        mock_update.message.text = "/execute 0.02"
        ctx = SimpleNamespace(args=["0.02"])
        with patch.object(main.get_auto_trader(), "open_position_manually") as mock_open:
            mock_open.return_value = (True, "Position Opened", None)
            await main.execute_command(mock_update, ctx)
            sent_exec = mock_update.message.reply_text.call_args[0][0]
            self.assertIn("Position Opened", sent_exec)

    async def test_buy_and_sell_commands(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock(), text="/buy"),
            effective_chat=SimpleNamespace(id=55555),
        )
        ctx = SimpleNamespace(args=[])
        with patch.object(main.get_auto_trader(), "open_position_manually") as mock_open:
            mock_open.return_value = (True, "Bought", None)
            await main.execute_command(mock_update, ctx)
            sent = mock_update.message.reply_text.call_args[0][0]
            self.assertIn("Bought", sent)

    async def test_calc_command(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
            effective_chat=SimpleNamespace(id=12345),
        )
        ctx_no_args = SimpleNamespace(args=[])
        main.latest_trade_plans.pop(12345, None)
        await main.calc_command(mock_update, ctx_no_args)
        sent_help = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Position Sizing & Risk:Reward Calculator", sent_help)

        mock_update.message.reply_text.reset_mock()
        ctx_args = SimpleNamespace(args=["81000", "80000", "83000"])
        await main.calc_command(mock_update, ctx_args)
        sent_calc = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("2.00 : 1", sent_calc)

    async def test_levels_command(self):
        mock_update = SimpleNamespace(
            message=SimpleNamespace(reply_text=AsyncMock()),
            effective_chat=SimpleNamespace(id=12345),
        )
        ctx = SimpleNamespace(args=["BTC"])
        await main.levels_command(mock_update, ctx)
        sent_levels = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("SMART MONEY & LIQUIDITY LEVELS", sent_levels)


class TestPinpointAndLiquidity(unittest.TestCase):
    def setUp(self):
        self.tmp_cfg = "/workspace/bright-darwin/.test_pinpoint_cfg.json"
        self.tmp_hist = "/workspace/bright-darwin/.test_pinpoint_hist.json"
        self.tmp_pos = "/workspace/bright-darwin/.test_pinpoint_pos.json"
        self.config = auto_trade.AutoTradeConfig(
            config_file=self.tmp_cfg,
            trades_history_file=self.tmp_hist,
            open_positions_file=self.tmp_pos,
            equity=10000.0,
            symbol="BTCUSD",
        )
        self.trader = auto_trade.AutoTrader(config=self.config)

    def tearDown(self):
        for p in (self.tmp_cfg, self.tmp_hist, self.tmp_pos):
            if os.path.exists(p):
                os.remove(p)

    def test_detect_order_blocks_and_fvg(self):
        df = self.trader._generate_dummy_candles("BTCUSD", count=60)
        res = auto_trade.detect_order_blocks_and_fvg(df)
        self.assertIn("swing_high", res)
        self.assertIn("swing_low", res)
        self.assertIn("all_obs", res)
        self.assertIn("all_fvgs", res)
        self.assertIsNotNone(res["swing_high"])
        self.assertIsNotNone(res["swing_low"])

    def test_generate_pinpoint_plan_btc(self):
        plan = auto_trade.generate_pinpoint_plan("BTCUSD", trader=self.trader)
        self.assertEqual(plan.symbol, "BTCUSD")
        self.assertIn(plan.direction, ("LONG", "SHORT"))
        self.assertGreater(plan.market_entry, 0)
        self.assertGreater(plan.stop_loss, 0)
        self.assertGreater(plan.take_profit_1, 0)
        self.assertGreater(plan.take_profit_2, 0)
        self.assertGreater(plan.take_profit_3, 0)
        self.assertGreater(plan.recommended_lots, 0)
        self.assertEqual(plan.rr_ratio_tp1, 1.5)
        self.assertEqual(plan.rr_ratio_tp2, 2.6)
        self.assertEqual(plan.rr_ratio_tp3, 4.2)

    def test_generate_pinpoint_plan_direction_override(self):
        long_plan = auto_trade.generate_pinpoint_plan("BTCUSD", direction_override="BUY", trader=self.trader)
        self.assertEqual(long_plan.direction, "LONG")
        self.assertGreater(long_plan.take_profit_1, long_plan.market_entry)
        self.assertLess(long_plan.stop_loss, long_plan.market_entry)

        short_plan = auto_trade.generate_pinpoint_plan("BTCUSD", direction_override="SELL", trader=self.trader)
        self.assertEqual(short_plan.direction, "SHORT")
        self.assertLess(short_plan.take_profit_1, short_plan.market_entry)
        self.assertGreater(short_plan.stop_loss, short_plan.market_entry)

    def test_format_pinpoint_report(self):
        plan = auto_trade.generate_pinpoint_plan("BTCUSD", trader=self.trader)
        rep = auto_trade.format_pinpoint_report(plan)
        self.assertIn("PINPOINT TRADE ENTRY & TARGET ANALYZER", rep)
        self.assertIn("ENTRY EXECUTION TIERS", rep)
        self.assertIn("PINPOINT INVALIDATION (STOP LOSS)", rep)
        self.assertIn("PRECISION TAKE PROFIT TARGETS", rep)
        self.assertIn("INSTITUTIONAL LIQUIDITY ZONES", rep)

    def test_open_position_manually(self):
        ok, msg, pos = self.trader.open_position_manually(
            symbol="BTCUSD",
            direction="LONG",
            entry_price=80000.0,
            stop_loss=79000.0,
            take_profit_1=82000.0,
            take_profit_2=84000.0,
            lot_size=0.1,
        )
        self.assertTrue(ok)
        self.assertIsNotNone(pos)
        self.assertEqual(self.trader.position.direction, "LONG")
        self.assertEqual(self.trader.position.entry_price, 80000.0)

        # Attempting second position while one is open should fail
        ok2, msg2, pos2 = self.trader.open_position_manually(
            symbol="BTCUSD",
            direction="SHORT",
            entry_price=80500.0,
            stop_loss=81500.0,
            take_profit_1=79000.0,
        )
        self.assertFalse(ok2)
        self.assertIn("already open", msg2)

    def test_execute_pinpoint_plan(self):
        plan = auto_trade.generate_pinpoint_plan("BTCUSD", direction_override="LONG", trader=self.trader)
        ok, msg = auto_trade.execute_pinpoint_plan(plan, lot_override=0.05, entry_mode="market", trader=self.trader)
        self.assertTrue(ok)
        self.assertIsNotNone(self.trader.position)
        self.assertEqual(self.trader.position.lot_size, 0.05)
        self.assertEqual(self.trader.position.direction, "LONG")

    def test_get_levels_report(self):
        report = auto_trade.get_levels_report("BTCUSD", trader=self.trader)
        self.assertIn("SMART MONEY & LIQUIDITY LEVELS", report)
        self.assertIn("ORDER BLOCKS", report)
        self.assertIn("FAIR VALUE GAPS", report)
        self.assertIn("KEY STRUCTURAL LIQUIDITY", report)

    def test_calculate_risk_reward(self):
        res_tp = auto_trade.calculate_risk_reward(81000.0, 80000.0, 83000.0, risk_dollars=100.0, equity=10000.0, symbol="BTCUSD")
        self.assertIn("POSITION SIZING & RISK:REWARD CALCULATOR", res_tp)
        self.assertIn("2.00 : 1", res_tp)
        self.assertIn("Expected Profit", res_tp)

        res_no_tp = auto_trade.calculate_risk_reward(80000.0, 81000.0, equity=10000.0, symbol="BTCUSD")
        self.assertIn("Projected Targets", res_no_tp)
        self.assertIn("TP1 (1.5R)", res_no_tp)

        err = auto_trade.calculate_risk_reward(80000.0, 80000.0)
        self.assertIn("Invalid Entry or Stop Loss", err)


class TestMultiAssetMultiPositionAutoTrade(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_cfg = "/workspace/bright-darwin/.test_multi_cfg.json"
        self.tmp_hist = "/workspace/bright-darwin/.test_multi_hist.json"
        self.tmp_pos = "/workspace/bright-darwin/.test_multi_pos.json"
        self.config = auto_trade.AutoTradeConfig(
            config_file=self.tmp_cfg,
            trades_history_file=self.tmp_hist,
            open_positions_file=self.tmp_pos,
            equity=10000.0,
            symbols=["BTCUSD", "XAUTUSD"],
            max_positions=5,
            max_positions_per_symbol=1,
        )
        self.trader = auto_trade.AutoTrader(config=self.config)

    def tearDown(self):
        for p in (self.tmp_cfg, self.tmp_hist, self.tmp_pos):
            if os.path.exists(p):
                os.remove(p)

    def test_default_config_includes_btc_and_gold(self):
        cfg = auto_trade.AutoTradeConfig(symbol="BTCUSD")
        self.assertIn("BTCUSD", cfg.symbols)
        self.assertIn("XAUTUSD", cfg.symbols)

        cfg_gold = auto_trade.AutoTradeConfig(symbol="XAUTUSD")
        self.assertIn("BTCUSD", cfg_gold.symbols)
        self.assertIn("XAUTUSD", cfg_gold.symbols)

    def test_symbol_management_methods(self):
        # Symbol normalization
        self.assertEqual(self.trader.normalize_symbol("btc"), "BTCUSD")
        self.assertEqual(self.trader.normalize_symbol("gold"), "XAUTUSD")
        self.assertEqual(self.trader.normalize_symbol("xau"), "XAUTUSD")
        self.assertEqual(self.trader.normalize_symbol("eth"), "ETHUSD")

        # Set symbols
        ok, msg = self.trader.set_symbols(["BTCUSD", "ETHUSD"])
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.symbols, ["BTCUSD", "ETHUSD"])

        # Add symbol
        ok2, msg2 = self.trader.add_symbol("SOL")
        self.assertTrue(ok2)
        self.assertIn("SOLUSD", self.trader.config.symbols)

        # Remove symbol
        ok3, msg3 = self.trader.remove_symbol("ETH")
        self.assertTrue(ok3)
        self.assertNotIn("ETHUSD", self.trader.config.symbols)

    def test_multiple_concurrent_positions(self):
        # 1. Open BTC position
        ok1, msg1, pos1 = self.trader.open_position_manually(
            symbol="BTCUSD",
            direction="LONG",
            entry_price=82000.0,
            stop_loss=81000.0,
            take_profit_1=84000.0,
            lot_size=0.05,
        )
        self.assertTrue(ok1)
        self.assertEqual(len(self.trader.positions), 1)
        self.assertEqual(self.trader.position.symbol, "BTCUSD")

        # 2. Open Gold position simultaneously
        ok2, msg2, pos2 = self.trader.open_position_manually(
            symbol="XAUTUSD",
            direction="LONG",
            entry_price=4150.0,
            stop_loss=4120.0,
            take_profit_1=4210.0,
            lot_size=0.1,
        )
        self.assertTrue(ok2)
        self.assertEqual(len(self.trader.positions), 2)
        self.assertEqual(self.trader.positions[0].symbol, "BTCUSD")
        self.assertEqual(self.trader.positions[1].symbol, "XAUTUSD")

        # 3. Opening 2nd BTC position when limit is 1 per symbol should fail
        ok3, msg3, pos3 = self.trader.open_position_manually(
            symbol="BTCUSD",
            direction="SHORT",
            entry_price=82500.0,
            stop_loss=83500.0,
            take_profit_1=81000.0,
        )
        self.assertFalse(ok3)
        self.assertIn("already open", msg3)

        # 4. Increasing limit allows 2nd BTC position
        self.trader.set_max_positions(total=5, per_symbol=2)
        ok4, msg4, pos4 = self.trader.open_position_manually(
            symbol="BTCUSD",
            direction="SHORT",
            entry_price=82500.0,
            stop_loss=83500.0,
            take_profit_1=81000.0,
        )
        self.assertTrue(ok4)
        self.assertEqual(len(self.trader.positions), 3)

    def test_closing_positions_by_id_symbol_and_all(self):
        # Open BTC and Gold positions
        self.trader.open_position_manually("BTCUSD", "LONG", 82000.0, 81000.0, 84000.0, lot_size=0.1)
        self.trader.open_position_manually("XAUTUSD", "LONG", 4150.0, 4120.0, 4210.0, lot_size=0.2)
        self.assertEqual(len(self.trader.positions), 2)

        btc_id = self.trader.positions[0].id

        # Close BTC by ID
        res_id = self.trader.close_position_by_id(btc_id, current_price=83000.0)
        self.assertIsNotNone(res_id)
        self.assertIn(btc_id, res_id)
        self.assertEqual(len(self.trader.positions), 1)
        self.assertEqual(self.trader.positions[0].symbol, "XAUTUSD")

        # Close Gold by symbol
        msgs_sym = self.trader.close_positions_by_symbol("gold", current_price=4180.0)
        self.assertEqual(len(msgs_sym), 1)
        self.assertEqual(len(self.trader.positions), 0)

        # Open both again and close all
        self.trader.open_position_manually("BTCUSD", "LONG", 82000.0, 81000.0, 84000.0, lot_size=0.1)
        self.trader.open_position_manually("XAUTUSD", "LONG", 4150.0, 4120.0, 4210.0, lot_size=0.2)
        self.assertEqual(len(self.trader.positions), 2)

        all_msgs = self.trader.close_all_positions()
        self.assertEqual(len(all_msgs), 2)
        self.assertEqual(len(self.trader.positions), 0)
        self.assertGreaterEqual(len(self.trader.closed_trades), 4)

    def test_multi_position_step_tp_sl(self):
        # Set up 1 BTC trade and 1 Gold trade
        self.trader.open_position_manually("BTCUSD", "LONG", 82000.0, 81000.0, 84000.0, lot_size=0.1)
        self.trader.open_position_manually("XAUTUSD", "LONG", 4150.0, 4120.0, 4210.0, lot_size=0.1)
        self.assertEqual(len(self.trader.positions), 2)

        btc_pos = self.trader.positions[0]
        gold_pos = self.trader.positions[1]

        # Mock candles: BTC hits TP (84100), Gold hits SL (4110)
        def mock_fetch(symbol, count=120):
            import pandas as pd
            if "BTC" in symbol:
                return pd.DataFrame([{"open": 82500, "high": 84100, "low": 82000, "close": 84050, "volume": 100}],
                                    index=pd.date_range("2026-01-01", periods=1, freq="1m"))
            else:
                return pd.DataFrame([{"open": 4140, "high": 4145, "low": 4110, "close": 4115, "volume": 100}],
                                    index=pd.date_range("2026-01-01", periods=1, freq="1m"))

        self.trader.fetch_candles = mock_fetch
        notifications = self.trader.step()

        self.assertEqual(len(self.trader.positions), 0)
        self.assertTrue(any("TAKE PROFIT HIT (BTCUSD)" in n for n in notifications))
        self.assertTrue(any("STOP LOSS HIT (XAUTUSD)" in n for n in notifications))

    async def test_symbols_command_telegram(self):
        mock_update = unittest.mock.AsyncMock()
        mock_ctx = unittest.mock.MagicMock()

        # 1. /symbols without args
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.symbols_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Active Auto-Trading Pairs", sent)
        self.assertIn("BTCUSD", sent)
        self.assertIn("XAUTUSD", sent)

        # 2. /symbols both
        mock_ctx.args = ["both"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.symbols_command(mock_update, mock_ctx)
        self.assertEqual(self.trader.config.symbols, ["BTCUSD", "XAUTUSD"])

        # 3. /symbols add eth
        mock_ctx.args = ["add", "eth"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.symbols_command(mock_update, mock_ctx)
        self.assertIn("ETHUSD", self.trader.config.symbols)

    async def test_close_command_telegram(self):
        mock_update = unittest.mock.AsyncMock()
        mock_ctx = unittest.mock.MagicMock()

        # 1. When no positions open
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.close_command(mock_update, mock_ctx)
        sent_empty = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("No active positions to close", sent_empty)

        # 2. Open 2 positions
        self.trader.open_position_manually("BTCUSD", "LONG", 82000.0, 81000.0, 84000.0, lot_size=0.1)
        self.trader.open_position_manually("XAUTUSD", "LONG", 4150.0, 4120.0, 4210.0, lot_size=0.1)

        # /close all
        mock_ctx.args = ["all"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.close_command(mock_update, mock_ctx)
        sent_all = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Closed 2 Position(s)", sent_all)
        self.assertEqual(len(self.trader.positions), 0)

    async def test_positions_dashboard_formatting(self):
        self.trader.open_position_manually("BTCUSD", "LONG", 82000.0, 81000.0, 84000.0, lot_size=0.1)
        self.trader.open_position_manually("XAUTUSD", "LONG", 4150.0, 4120.0, 4210.0, lot_size=0.2)

        dashboard = self.trader.get_position_text()
        self.assertIn("Active Positions (2/5)", dashboard)
        self.assertIn("BTCUSD", dashboard)
        self.assertIn("XAUTUSD", dashboard)
        self.assertIn("Total Unrealized PnL", dashboard)


class TestTradingModeAndCapital(unittest.TestCase):
    def setUp(self):
        self.tmp_cfg = "/workspace/bright-darwin/.test_mode_cfg.json"
        self.tmp_pos = "/workspace/bright-darwin/.test_mode_pos.json"
        self.tmp_hist = "/workspace/bright-darwin/.test_mode_hist.json"
        for p in (self.tmp_cfg, self.tmp_hist, self.tmp_pos):
            if os.path.exists(p):
                os.remove(p)
        self.config = auto_trade.AutoTradeConfig(
            config_file=self.tmp_cfg,
            trades_history_file=self.tmp_hist,
            open_positions_file=self.tmp_pos,
            paper_capital=100.0,
            equity=100.0,
            trading_mode="paper",
        )
        self.trader = auto_trade.AutoTrader(config=self.config)

    def tearDown(self):
        for p in (self.tmp_cfg, self.tmp_hist, self.tmp_pos):
            if os.path.exists(p):
                os.remove(p)

    def test_default_capital_is_100(self):
        cfg = auto_trade.AutoTradeConfig()
        self.assertEqual(cfg.paper_capital, 100.0)
        self.assertEqual(cfg.equity, 100.0)
        self.assertEqual(cfg.trading_mode, "paper")
        self.assertEqual(self.trader.config.trading_mode, "paper")

    def test_set_paper_capital(self):
        ok, msg = self.trader.set_paper_capital(250.0)
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.paper_capital, 250.0)
        self.assertEqual(self.trader.config.equity, 250.0)

        # Invalid amount
        ok_inv, _ = self.trader.set_paper_capital(-50.0)
        self.assertFalse(ok_inv)

    def test_add_funds(self):
        self.trader.set_paper_capital(100.0)
        ok, msg = self.trader.add_funds(50.0)
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.equity, 150.0)
        self.assertEqual(self.trader.config.total_deposited, 50.0)

    def test_reduce_funds(self):
        self.trader.set_paper_capital(100.0)
        ok, msg = self.trader.reduce_funds(30.0)
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.equity, 70.0)
        self.assertEqual(self.trader.config.total_withdrawn, 30.0)

        # Excess reduction error
        ok_fail, msg_fail = self.trader.reduce_funds(200.0)
        self.assertFalse(ok_fail)
        self.assertIn("Cannot reduce", msg_fail)

    def test_reset_funds(self):
        self.trader.set_paper_capital(500.0)
        self.trader.add_funds(100.0)
        ok, msg = self.trader.reset_funds()
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.paper_capital, 100.0)
        self.assertEqual(self.trader.config.equity, 100.0)
        self.assertEqual(self.trader.config.total_deposited, 0.0)
        self.assertEqual(self.trader.config.total_withdrawn, 0.0)

    def test_capital_report(self):
        self.trader.set_paper_capital(100.0)
        self.trader.add_funds(25.0)
        report = self.trader.get_capital_report()
        self.assertIn("TRADING CAPITAL & FUNDS DASHBOARD", report)
        self.assertIn("PAPER TRADING", report)
        self.assertIn("$125.00", report)
        self.assertIn("$100.00", report)

    def test_mode_switching_guard(self):
        # Unconfigured live mode should fail
        self.trader.exchange_client.api_key = ""
        self.trader.exchange_client.api_secret = ""
        ok, msg = self.trader.set_trading_mode("live")
        self.assertFalse(ok)
        self.assertIn("CANNOT ACTIVATE LIVE TRADING", msg)
        self.assertEqual(self.trader.config.trading_mode, "paper")

        # Configured live mode should succeed
        self.trader.exchange_client.api_key = "test_key_123"
        self.trader.exchange_client.api_secret = "test_secret_456"
        ok_live, msg_live = self.trader.set_trading_mode("live")
        self.assertTrue(ok_live)
        self.assertEqual(self.trader.config.trading_mode, "live")
        self.assertIn("LIVE TRADING MODE ACTIVATED", msg_live)

        # Switch back to paper
        ok_paper, msg_paper = self.trader.set_trading_mode("paper")
        self.assertTrue(ok_paper)
        self.assertEqual(self.trader.config.trading_mode, "paper")
        self.assertIn("PAPER TRADING MODE ACTIVATED", msg_paper)

    def test_exchange_api_credentials_and_test(self):
        with patch.object(self.trader.exchange_client, "test_connection", return_value=(True, "Success (Mock)", {"mock": True})):
            ok, msg = self.trader.set_exchange_api("delta", "my_api_key_delta", "my_api_secret_delta")
            self.assertTrue(ok)
            self.assertEqual(self.trader.config.live_exchange, "delta")
            self.assertEqual(self.trader.config.exchange_api_key, "my_api_key_delta")

        report = self.trader.get_api_status_report()
        self.assertIn("LIVE EXCHANGE API SYSTEM", report)
        self.assertIn("DELTA", report)
        self.assertIn("my_a...elta", report)

        clear_msg = self.trader.clear_exchange_api()
        self.assertIn("cleared", clear_msg)
        self.assertEqual(self.trader.config.exchange_api_key, "")
        self.assertEqual(self.trader.config.trading_mode, "paper")


class TestModeCapitalApiTelegramCommands(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_cfg = "/workspace/bright-darwin/.test_cmd_cfg.json"
        self.tmp_pos = "/workspace/bright-darwin/.test_cmd_pos.json"
        self.tmp_hist = "/workspace/bright-darwin/.test_cmd_hist.json"
        for p in (self.tmp_cfg, self.tmp_hist, self.tmp_pos):
            if os.path.exists(p):
                os.remove(p)
        self.config = auto_trade.AutoTradeConfig(
            config_file=self.tmp_cfg,
            trades_history_file=self.tmp_hist,
            open_positions_file=self.tmp_pos,
            paper_capital=100.0,
            equity=100.0,
            trading_mode="paper",
        )
        self.trader = auto_trade.AutoTrader(config=self.config)

    def tearDown(self):
        for p in (self.tmp_cfg, self.tmp_hist, self.tmp_pos):
            if os.path.exists(p):
                os.remove(p)

    async def test_mode_command_telegram(self):
        mock_update = unittest.mock.AsyncMock()
        mock_ctx = unittest.mock.MagicMock()

        # 1. /mode status
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.mode_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TRADING MODE CONFIGURATION", sent)
        self.assertIn("PAPER TRADING", sent)

        # 2. /mode live without keys
        mock_ctx.args = ["live"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.mode_command(mock_update, mock_ctx)
        sent_live_fail = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("CANNOT ACTIVATE LIVE TRADING", sent_live_fail)

        # 3. /mode paper
        mock_ctx.args = ["paper"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.mode_command(mock_update, mock_ctx)
        sent_paper = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("PAPER TRADING MODE ACTIVATED", sent_paper)

    async def test_capital_command_telegram(self):
        mock_update = unittest.mock.AsyncMock()
        mock_ctx = unittest.mock.MagicMock()

        # 1. /capital (view report)
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.capital_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TRADING CAPITAL & FUNDS DASHBOARD", sent)

        # 2. /capital set 100
        mock_ctx.args = ["set", "100"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.capital_command(mock_update, mock_ctx)
        self.assertEqual(self.trader.config.paper_capital, 100.0)

        # 3. /capital 200 (direct number)
        mock_ctx.args = ["200"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.capital_command(mock_update, mock_ctx)
        self.assertEqual(self.trader.config.paper_capital, 200.0)

        # 4. /deposit 50 (via deposit_command)
        mock_ctx.args = ["50"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.deposit_command(mock_update, mock_ctx)
        self.assertEqual(self.trader.config.equity, 250.0)

        # 5. /withdraw 25 (via withdraw_command)
        mock_ctx.args = ["25"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.withdraw_command(mock_update, mock_ctx)
        self.assertEqual(self.trader.config.equity, 225.0)

        # 6. /capital reset
        mock_ctx.args = ["reset"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.capital_command(mock_update, mock_ctx)
        self.assertEqual(self.trader.config.equity, 100.0)

    async def test_api_command_telegram(self):
        mock_update = unittest.mock.AsyncMock()
        mock_ctx = unittest.mock.MagicMock()

        # 1. /api status
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.api_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("LIVE EXCHANGE API SYSTEM", sent)

        # 2. /api set delta key secret
        mock_ctx.args = ["set", "delta", "key123", "sec456"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader), \
             unittest.mock.patch.object(self.trader.exchange_client, "test_connection", return_value=(True, "Connected", {})):
            await main.api_command(mock_update, mock_ctx)
        sent_set = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Exchange API Configured", sent_set)
        self.assertEqual(self.trader.config.exchange_api_key, "key123")

        # 3. /api clear
        mock_ctx.args = ["clear"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.api_command(mock_update, mock_ctx)
        sent_clear = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("cleared", sent_clear)
        self.assertEqual(self.trader.config.exchange_api_key, "")

        # 4. /api test (verifies unpacking fix for tuple get_balance)
        mock_ctx.args = ["test"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader), \
             unittest.mock.patch.object(self.trader.exchange_client, "test_connection", return_value=(True, "Connected OK", {"balance": 5000.0})), \
             unittest.mock.patch.object(self.trader.exchange_client, "get_balance", return_value=(True, 5000.0, "Connected OK")):
            await main.api_command(mock_update, mock_ctx)
        sent_test = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Exchange API Connected Successfully", sent_test)
        self.assertIn("5,000.00", sent_test)

    async def test_start_and_menu_command_categories(self):
        mock_update = unittest.mock.AsyncMock()
        mock_ctx = unittest.mock.MagicMock()

        # Test /start contains categories
        await main.start(mock_update, mock_ctx)
        start_txt = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("1. Market & Live Quotes", start_txt)
        self.assertIn("2. Pinpoint Trade Planning", start_txt)
        self.assertIn("3. Auto-Trading & Execution", start_txt)
        self.assertIn("4. Paper & Live Trading / Capital Management", start_txt)
        self.assertIn("5. Live Exchange API System", start_txt)
        self.assertIn("6. Risk & Strategy Configuration", start_txt)
        self.assertIn("7. Trade Level Alerts & Notifications", start_txt)
        self.assertIn("8. Assistant & Diagnostics", start_txt)
        self.assertIn("/mode", start_txt)
        self.assertIn("/capital", start_txt)
        self.assertIn("/api", start_txt)
        self.assertIn("/alert", start_txt)

        # Test /menu contains categories
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.menu_command(mock_update, mock_ctx)
        menu_txt = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TRADING BOT COMMAND MENU", menu_txt)
        self.assertIn("1. Market & Quotes", menu_txt)
        self.assertIn("2. Pinpoint Trade Planning", menu_txt)
        self.assertIn("3. Auto-Trading & Execution", menu_txt)
        self.assertIn("4. Paper/Live Mode & Capital", menu_txt)
        self.assertIn("5. Live Exchange API System", menu_txt)
        self.assertIn("6. Risk & Strategy", menu_txt)
        self.assertIn("7. Trade Level Alerts", menu_txt)
        self.assertIn("8. Assistant & Settings", menu_txt)


class TestTradeLevelAlerts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import pandas as pd
        self.pd = pd
        self.temp_cfg = "/workspace/bright-darwin/.test_alerts_cfg.json"
        self.temp_pos = "/workspace/bright-darwin/.test_alerts_pos.json"
        self.temp_hist = "/workspace/bright-darwin/.test_alerts_hist.json"
        self.temp_alerts = "/workspace/bright-darwin/.test_alerts_file.json"
        for f in (self.temp_cfg, self.temp_pos, self.temp_hist, self.temp_alerts):
            if os.path.exists(f):
                os.remove(f)

        self.cfg = auto_trade.AutoTradeConfig(
            config_file=self.temp_cfg,
            open_positions_file=self.temp_pos,
            trades_history_file=self.temp_hist,
            alerts_file=self.temp_alerts,
            symbol="BTCUSD",
            symbols=["BTCUSD", "XAUTUSD"],
            enabled=False,
        )
        self.trader = auto_trade.AutoTrader(config=self.cfg)

    def tearDown(self):
        for f in (self.temp_cfg, self.temp_pos, self.temp_hist, self.temp_alerts):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_make_modern_meter(self):
        m0 = auto_trade.make_modern_meter(0, width=10)
        self.assertEqual(m0, "░" * 10)
        m100 = auto_trade.make_modern_meter(100, width=10)
        self.assertEqual(m100, "■" * 10)
        m50 = auto_trade.make_modern_meter(50, width=10)
        self.assertEqual(m50, "■" * 5 + "░" * 5)
        m_neg = auto_trade.make_modern_meter(-10, width=10)
        self.assertEqual(m_neg, "░" * 10)
        m_over = auto_trade.make_modern_meter(150, width=10)
        self.assertEqual(m_over, "■" * 10)

    def test_add_and_get_alerts(self):
        ok, msg, alt1 = self.trader.add_alert("BTCUSD", 85000.0, condition="CROSS_ABOVE", note="Resistance level")
        self.assertTrue(ok)
        self.assertIsNotNone(alt1)
        self.assertIsNotNone(alt1.id)
        self.assertEqual(alt1.symbol, "BTCUSD")
        self.assertEqual(alt1.target_price, 85000.0)
        self.assertEqual(alt1.condition, "CROSS_ABOVE")
        self.assertFalse(alt1.triggered)

        ok2, msg2, alt2 = self.trader.add_alert("XAUTUSD", 4100.0, condition="CROSS_BELOW", note="Support level")
        self.assertTrue(ok2)
        self.assertEqual(len(self.trader.alerts), 2)

        btc_alerts = self.trader.get_alerts("BTCUSD")
        self.assertEqual(len(btc_alerts), 1)
        self.assertEqual(btc_alerts[0].id, alt1.id)

        rep = self.trader.get_alerts_report()
        self.assertIn("ACTIVE TRADE LEVEL ALERTS", rep)
        self.assertIn("BTCUSD", rep)
        self.assertIn("85,000.00", rep)

    def test_remove_and_clear_alerts(self):
        ok1, _, alt1 = self.trader.add_alert("BTCUSD", 85000.0, condition="CROSS_ABOVE")
        ok2, _, alt2 = self.trader.add_alert("XAUTUSD", 4200.0, condition="CROSS_BELOW")

        ok_rem, _ = self.trader.remove_alert(alt1.id)
        self.assertTrue(ok_rem)
        ok_bad, _ = self.trader.remove_alert("NON_EXISTENT")
        self.assertFalse(ok_bad)
        self.assertEqual(len(self.trader.alerts), 1)

        cleared = self.trader.clear_alerts()
        self.assertEqual(cleared, 1)
        self.assertEqual(len(self.trader.alerts), 0)

    def test_check_trade_level_alerts_cross_above(self):
        self.trader.add_alert("BTCUSD", 83000.0, condition="CROSS_ABOVE", note="Breakout")

        # Fake candles with close below target
        df_below = self.pd.DataFrame({
            "open": [82000.0], "high": [82500.0], "low": [81900.0], "close": [82200.0], "volume": [100.0]
        })
        notes = self.trader.check_trade_level_alerts(candles_cache={"BTCUSD": df_below, "XAUTUSD": df_below})
        custom_notes = [n for n in notes if "TRADE LEVEL ALERT TRIGGERED" in n]
        self.assertEqual(len(custom_notes), 0)

        # Fake candles with close above target
        df_above = self.pd.DataFrame({
            "open": [82500.0], "high": [83500.0], "low": [82400.0], "close": [83200.0], "volume": [100.0]
        })
        notes_above = self.trader.check_trade_level_alerts(candles_cache={"BTCUSD": df_above, "XAUTUSD": df_below})
        custom_notes_above = [n for n in notes_above if "TRADE LEVEL ALERT TRIGGERED" in n]
        self.assertEqual(len(custom_notes_above), 1)
        self.assertIn("83,000.00", custom_notes_above[0])

        # Second check should not trigger one_shot alert again
        notes_again = self.trader.check_trade_level_alerts(candles_cache={"BTCUSD": df_above, "XAUTUSD": df_below})
        custom_notes_again = [n for n in notes_again if "TRADE LEVEL ALERT TRIGGERED" in n]
        self.assertEqual(len(custom_notes_again), 0)

    def test_check_trade_level_alerts_tp_proximity(self):
        pos = auto_trade.AutoTradePosition(
            id="TP_TEST_1",
            symbol="BTCUSD",
            direction="LONG",
            entry_price=80000.0,
            stop_loss=78000.0,
            take_profit_1=85000.0,
            take_profit_2=None,
            lot_size=0.1,
            entry_time="2026-10-09 00:00:00 UTC",
            strategy="Ensemble (Pro)",
            reason="Signal",
            highest_price=80000.0,
            lowest_price=80000.0,
        )
        self.trader.positions.append(pos)

        # Price at 84500 is within 15% distance to TP1
        df = self.pd.DataFrame({
            "open": [84400.0], "high": [84600.0], "low": [84300.0], "close": [84500.0], "volume": [100.0]
        })
        notes = self.trader.check_trade_level_alerts(candles_cache={"BTCUSD": df})
        tp_notes = [n for n in notes if "TARGET REACH WARNING" in n or "TP1 NEARBY" in n]
        self.assertEqual(len(tp_notes), 1)
        self.assertIn("85,000.00", tp_notes[0])

    async def test_alert_telegram_command(self):
        mock_update = unittest.mock.AsyncMock()
        mock_update.message.reply_text = unittest.mock.AsyncMock()
        mock_update.effective_chat.id = 123456

        # 1. /alert with no args -> empty list
        mock_ctx = unittest.mock.MagicMock()
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.alert_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("No active price triggers set", sent)

        # 2. /alert BTC 85000 above
        mock_ctx.args = ["BTC", "85000", "above", "TP", "level"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.alert_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Trade Level Alert Set", sent)
        self.assertIn("BTCUSD", sent)
        self.assertIn("85,000.00", sent)

        self.assertEqual(len(self.trader.alerts), 1)
        alert_id = self.trader.alerts[0].id

        # 3. /alert del <id>
        mock_ctx.args = ["del", alert_id]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.alert_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("removed successfully", sent)
        self.assertEqual(len(self.trader.alerts), 0)

        # 4. /alert clear
        self.trader.add_alert("BTCUSD", 90000.0)
        mock_ctx.args = ["clear"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.alert_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Cleared", sent)
        self.assertEqual(len(self.trader.alerts), 0)


class TestITBEngineAndIntegration(unittest.TestCase):
    """Unit tests for Intelligent Trading Bot (ITB) ML engine, AutoTrader integration, and Telegram commands."""

    def setUp(self):
        import pandas as pd
        import numpy as np
        self.pd = pd
        self.np = np

        dates = pd.date_range("2026-01-01", periods=100, freq="1min")
        close = 80000.0 + np.cumsum(np.random.RandomState(42).randn(100) * 15.0)
        self.candles = pd.DataFrame(
            {
                "open": close - 5.0,
                "high": close + 10.0,
                "low": close - 10.0,
                "close": close,
                "volume": 200.0,
            },
            index=dates,
        )

        self.cfg_file = "/workspace/bright-darwin/.test_itb_cfg.json"
        self.pos_file = "/workspace/bright-darwin/.test_itb_pos.json"
        self.history_file = "/workspace/bright-darwin/.test_itb_hist.json"
        self.alerts_file = "/workspace/bright-darwin/.test_itb_alts.json"

        cfg = auto_trade.AutoTradeConfig(
            config_file=self.cfg_file,
            open_positions_file=self.pos_file,
            trades_history_file=self.history_file,
            alerts_file=self.alerts_file,
            trading_mode="paper",
            equity=100.0,
            paper_capital=100.0,
            symbols=["BTCUSD", "XAUTUSD"],
            strategy_type="itb_ml",
        )
        self.trader = auto_trade.AutoTrader(config=cfg)

    def tearDown(self):
        for f in (self.cfg_file, self.pos_file, self.history_file, self.alerts_file):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_itb_features_generation(self):
        import itb_engine
        feat_df, cols = itb_engine.ITBFeatureGenerator.generate_features(self.candles)
        self.assertFalse(feat_df.empty)
        self.assertIn("itb_slope_10", cols)
        self.assertIn("itb_skew_15", cols)
        self.assertIn("itb_kurt_15", cols)
        self.assertIn("itb_hl_ratio_15", cols)
        self.assertIn("itb_log_return", cols)
        self.assertEqual(len(feat_df), len(self.candles))

    def test_itb_predictor_predict_and_train(self):
        import itb_engine
        pred = itb_engine.ITBPredictor()
        result = pred.predict(self.candles, symbol="BTCUSD")
        self.assertIsInstance(result, itb_engine.ITBPredictionResult)
        self.assertIn(result.zone, ("BUY ZONE", "SELL ZONE", "NEUTRAL"))
        self.assertTrue(-1.0 <= result.indicator <= 1.0)
        self.assertTrue(-1.0 <= result.smoothed_indicator <= 1.0)
        self.assertTrue(0.0 <= result.confidence <= 1.0)

        # Test model training
        train_res = pred.train(self.candles)
        self.assertNotIn("error", train_res)
        self.assertIn("weights", train_res)
        self.assertIn("r2_score", train_res)
        self.assertIn("mae", train_res)
        self.assertGreater(train_res["samples"], 20)

    def test_itb_backtester(self):
        import itb_engine
        res = itb_engine.ITBBacktester.backtest(self.candles, threshold=0.05)
        self.assertNotIn("error", res)
        self.assertIn("total_transactions", res)
        self.assertIn("win_rate", res)
        self.assertIn("total_profit", res)
        self.assertIn("long", res)
        self.assertIn("short", res)
        self.assertEqual(res["candles_evaluated"], len(self.candles))

    def test_itb_strategy_lifecycle(self):
        import itb_engine
        strat = itb_engine.ITBStrategy(symbol="BTCUSD", min_threshold=0.01)
        sig = strat.generate_signal(self.candles)
        # Check signal interface
        if sig:
            self.assertIn(sig.direction.name, ("LONG", "SHORT"))
            self.assertGreater(sig.entry, 0)
            self.assertGreater(sig.stop, 0)
            self.assertGreater(sig.tp1, 0)

        # Test size calculation
        lot = strat.size(1000.0, 80000.0, 79000.0)
        self.assertGreater(lot, 0)

        # Update PnL
        strat.update(25.5)
        self.assertEqual(strat.total_pnl, 25.5)

    def test_autotrader_strategy_management(self):
        # 1. Switch strategy
        ok, msg = self.trader.set_strategy_type("itb")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.strategy_type, "itb_ml")
        self.assertIn("Intelligent Trading Bot", msg)

        ok, msg = self.trader.set_strategy_type("indicators")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.strategy_type, "indicators_pro")

        ok, msg = self.trader.set_strategy_type("ai")
        self.assertTrue(ok)
        self.assertEqual(self.trader.config.strategy_type, "ai_learning")

        ok, msg = self.trader.set_strategy_type("invalid_strat")
        self.assertFalse(ok)

        # 2. Trader ITB helper reports
        analysis = self.trader.get_itb_analysis("BTCUSD")
        self.assertIn("INTELLIGENT TRADING SIGNALS", analysis)

        backtest = self.trader.run_itb_backtest("BTCUSD", count=50)
        self.assertIn("ITB SIMULATED TRADE PERFORMANCE", backtest)

        train_card = self.trader.train_itb_model("BTCUSD", count=60)
        self.assertIn("INTELLIGENT TRADING BOT (ITB) MODEL TRAINED", train_card)

    def test_autotrader_step_itb_execution(self):
        self.trader.set_strategy_type("itb_ml")
        self.trader.enable()
        self.trader._strategies_itb["BTCUSD"].min_threshold = -1.0  # Force signal trigger
        notifs = self.trader.step()
        self.assertGreaterEqual(len(notifs), 1)
        if self.trader.positions:
            self.assertEqual(self.trader.positions[0].strategy, "Ensemble (ITB)")

    async def test_itb_and_strategy_telegram_commands(self):
        mock_update = unittest.mock.AsyncMock()
        mock_update.message.reply_text = unittest.mock.AsyncMock()

        # 1. /strategy without args
        mock_ctx = unittest.mock.MagicMock()
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.strategy_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("TRADING STRATEGY CONFIGURATION", sent)

        # 2. /strategy itb
        mock_ctx.args = ["itb"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.strategy_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("Strategy switched to", sent)
        self.assertEqual(self.trader.config.strategy_type, "itb_ml")

        # 3. /itb live analysis
        mock_ctx.args = []
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.itb_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("INTELLIGENT TRADING SIGNALS", sent)

        # 4. /itb backtest BTC 60
        mock_ctx.args = ["backtest", "BTC", "60"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.itb_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("ITB SIMULATED TRADE PERFORMANCE", sent)

        # 5. /itb train BTC 70
        mock_ctx.args = ["train", "BTC", "70"]
        with unittest.mock.patch("main.get_auto_trader", return_value=self.trader):
            await main.itb_command(mock_update, mock_ctx)
        sent = mock_update.message.reply_text.call_args[0][0]
        self.assertIn("INTELLIGENT TRADING BOT (ITB) MODEL TRAINED", sent)


def tearDownModule():
    import glob
    for f in glob.glob("/workspace/bright-darwin/.test_*"):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()


