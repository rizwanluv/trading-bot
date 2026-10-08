"""
Unit & Integration Tests for Telegram and Google API Connectors
==============================================================
"""

import unittest
from unittest.mock import patch, MagicMock

import config
from telegram_client import TelegramClient
from google_client import GoogleAIClient


class TestTelegramClient(unittest.TestCase):
    def setUp(self):
        self.client = TelegramClient()

    def test_client_is_configured(self):
        self.assertTrue(self.client.is_configured())

    def test_live_telegram_connection(self):
        """Tests live connection to Telegram API with configured bot token."""
        result = self.client.test_connection()
        self.assertTrue(result["success"])
        self.assertEqual(result["status_code"], 200)
        self.assertIsNotNone(result["bot"])
        self.assertEqual(result["bot"]["username"], "Rizwan9635Bot")

    def test_missing_token_handling(self):
        unconfigured_client = TelegramClient(token="")
        self.assertFalse(unconfigured_client.is_configured())
        res = unconfigured_client.test_connection()
        self.assertFalse(res["success"])
        self.assertIn("not configured", res["error"])

    @patch("requests.post")
    def test_send_message_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 999}}
        mock_post.return_value = mock_resp

        res = self.client.send_message(chat_id="12345678", text="Hello world")
        self.assertTrue(res["success"])
        self.assertEqual(res["message_id"], 999)

    def test_send_message_empty_text(self):
        res = self.client.send_message(chat_id="12345678", text="")
        self.assertFalse(res["success"])
        self.assertIn("empty message", res["error"])


class TestGoogleAIClient(unittest.TestCase):
    def setUp(self):
        self.client = GoogleAIClient()

    def test_client_configuration(self):
        self.assertTrue(self.client.is_configured())
        self.assertEqual(self.client.model, config.GEMINI_MODEL)

    def test_missing_key_behavior(self):
        unconfigured = GoogleAIClient(api_key="")
        self.assertFalse(unconfigured.is_configured())
        res = unconfigured.test_connection()
        self.assertFalse(res["success"])
        self.assertIn("not configured", res["error"])

    @patch("requests.get")
    def test_connection_mock_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "models/gemini-2.5-flash",
            "displayName": "Gemini 2.5 Flash",
            "supportedGenerationMethods": ["generateContent"],
        }
        mock_get.return_value = mock_resp

        res = self.client.test_connection()
        self.assertTrue(res["success"])
        self.assertEqual(res["display_name"], "Gemini 2.5 Flash")

    @patch("requests.get")
    def test_connection_mock_401_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        res = self.client.test_connection()
        self.assertFalse(res["success"])
        self.assertEqual(res["status_code"], 401)
        self.assertIn("Invalid or unauthenticated", res["error"])

    @patch("requests.post")
    def test_generate_content_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Market looks bullish above 90,000."}]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        mock_post.return_value = mock_resp

        res = self.client.generate_content("Analyze BTC")
        self.assertTrue(res["success"])
        self.assertEqual(res["text"], "Market looks bullish above 90,000.")
        self.assertEqual(res["finish_reason"], "STOP")

    @patch("requests.post")
    def test_generate_json_parsing(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "```json\n{\"bias\": \"BULLISH\", \"confidence\": 0.85}\n```"}
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        success, parsed, raw = self.client.generate_json("Give json signal")
        self.assertTrue(success)
        self.assertEqual(parsed["bias"], "BULLISH")
        self.assertEqual(parsed["confidence"], 0.85)


if __name__ == "__main__":
    unittest.main()
