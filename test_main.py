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


if __name__ == "__main__":
    unittest.main()
