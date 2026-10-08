"""
Unit tests for main.py trading bot.
"""
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from types import SimpleNamespace

import main


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
        self.config = auto_trade.AutoTradeConfig(config_file=self.tmp_config)
        self.trader = auto_trade.AutoTrader(config=self.config)

    def tearDown(self):
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

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
            strategy="Indicators Pro",
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
            strategy="Indicators Pro",
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
            strategy="Indicators Pro",
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
        self.assertIn("/autotrade", sent)
        self.assertIn("/lotsize", sent)
        self.assertIn("/tpsl", sent)


if __name__ == "__main__":
    unittest.main()

