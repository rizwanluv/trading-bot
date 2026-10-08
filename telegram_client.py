"""
Telegram API Connector Module
=============================
Handles communication with the Telegram Bot API using both lightweight HTTP requests
and python-telegram-bot application lifecycle.
"""

import logging
from typing import Dict, Any, Optional, List
import requests

import config

logger = logging.getLogger("telegram_client")


class TelegramClient:
    """
    Client for interacting with Telegram Bot API.
    Provides methods for testing connectivity, sending alerts, and handling bot instances.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = config.TELEGRAM_BOT_TOKEN if token is None else token
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def is_configured(self) -> bool:
        """Checks if a bot token is set."""
        return bool(self.token and len(self.token) > 10)

    def test_connection(self) -> Dict[str, Any]:
        """
        Calls getMe to verify Telegram Bot token validity and fetch bot profile.
        Returns a dictionary with status, bot info, or error message.
        """
        if not self.is_configured():
            return {
                "success": False,
                "status_code": 0,
                "error": "TELEGRAM_BOT_TOKEN is not configured or empty.",
                "bot": None,
            }

        url = f"{self.base_url}/getMe"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if resp.status_code == 200 and data.get("ok"):
                bot_info = data.get("result", {})
                logger.info("Connected to Telegram Bot: @%s (ID: %s)", bot_info.get("username"), bot_info.get("id"))
                return {
                    "success": True,
                    "status_code": resp.status_code,
                    "error": None,
                    "bot": {
                        "id": bot_info.get("id"),
                        "username": bot_info.get("username"),
                        "first_name": bot_info.get("first_name"),
                        "can_join_groups": bot_info.get("can_join_groups", False),
                    },
                }
            else:
                err_msg = data.get("description", "Unknown Telegram API error")
                logger.error("Telegram getMe failed [%s]: %s", resp.status_code, err_msg)
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": err_msg,
                    "bot": None,
                }
        except requests.RequestException as exc:
            logger.error("Network error connecting to Telegram: %s", exc)
            return {
                "success": False,
                "status_code": 0,
                "error": f"Network exception: {str(exc)}",
                "bot": None,
            }

    def send_message(
        self,
        chat_id: Optional[str] = None,
        text: str = "",
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> Dict[str, Any]:
        """
        Sends a message to a Telegram chat using HTTP REST API.
        """
        target_chat = chat_id or config.TELEGRAM_CHAT_ID
        if not target_chat:
            return {"success": False, "error": "No chat_id provided and TELEGRAM_CHAT_ID is not configured."}

        if not text:
            return {"success": False, "error": "Cannot send empty message."}

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

        try:
            resp = requests.post(url, json=payload, timeout=12)
            data = resp.json()
            if resp.status_code == 200 and data.get("ok"):
                return {"success": True, "message_id": data.get("result", {}).get("message_id")}
            else:
                err = data.get("description", f"HTTP {resp.status_code}")
                return {"success": False, "error": err}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_updates(self, offset: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches incoming updates/messages for the bot via getUpdates.
        """
        if not self.is_configured():
            return []

        url = f"{self.base_url}/getUpdates"
        params: Dict[str, Any] = {"limit": limit}
        if offset is not None:
            params["offset"] = offset

        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if resp.status_code == 200 and data.get("ok"):
                return data.get("result", [])
        except Exception as exc:
            logger.error("Failed to fetch updates: %s", exc)

        return []


def print_telegram_status() -> None:
    """Utility function to print Telegram API connection status."""
    client = TelegramClient()
    print("=" * 60)
    print(" TELEGRAM API CONNECTION TEST")
    print("=" * 60)
    res = client.test_connection()
    if res["success"]:
        bot = res["bot"]
        print(f" Status        : [CONNECTED] HTTP {res['status_code']}")
        print(f" Bot Username  : @{bot['username']}")
        print(f" Bot Name      : {bot['first_name']}")
        print(f" Bot ID        : {bot['id']}")
        print(f" Group Support : {bot['can_join_groups']}")
    else:
        print(f" Status        : [FAILED] HTTP {res['status_code']}")
        print(f" Error         : {res['error']}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_telegram_status()
