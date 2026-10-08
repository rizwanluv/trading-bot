"""
Configuration Module for Trading Bot
====================================
Centralized configuration manager supporting environment variables, local .env files,
risk parameters, asset definitions, and diagnostic validation.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("config")


def load_dotenv(filepath: str = ".env") -> Dict[str, str]:
    """
    Parse a local .env file into key-value pairs and inject them into os.environ
    if not already present. Does not require external packages.
    """
    loaded: Dict[str, str] = {}
    if not os.path.exists(filepath):
        return loaded

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                loaded[k] = v
                if k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        logger.warning("Failed reading %s: %s", filepath, exc)

    return loaded


# Load .env file automatically upon import
_ENV_VARS = load_dotenv(".env")


def get_env_var(*keys: str, default: str = "") -> str:
    """
    Retrieve the first matching environment variable from a list of alias keys.
    Supports case-insensitive and underscore/hyphen-agnostic matching.
    """
    for k in keys:
        val = os.getenv(k)
        if val is not None and val.strip():
            return val.strip().strip("'\"")

    # Secondary check: case and punctuation normalization
    normalized_keys = {k.lower().replace("_", "").replace("-", "").replace(" ", "") for k in keys}
    for env_k, env_v in os.environ.items():
        clean_k = env_k.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        if clean_k in normalized_keys and env_v.strip():
            return env_v.strip().strip("'\"")

    return default


def get_env_float(key: str, default: float) -> float:
    """Safely parse float from environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return default


def get_env_int(key: str, default: int) -> int:
    """Safely parse int from environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Safely parse boolean from environment variable."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "t", "on")


# ==============================================================================
# 1. Telegram Bot Settings
# ==============================================================================
TELEGRAM_BOT_TOKEN: str = get_env_var(
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_TOKEN",
    "BOT_TOKEN",
    default=""
)
TELEGRAM_CHAT_ID: str = get_env_var("TELEGRAM_CHAT_ID", "CHAT_ID", default="")
_allowed_raw = get_env_var("TELEGRAM_ALLOWED_USERS", "ALLOWED_USERS", default="")
TELEGRAM_ALLOWED_USERS: List[str] = [u.strip() for u in _allowed_raw.split(",") if u.strip()]


# ==============================================================================
# 2. AI / Gemini Model Settings
# ==============================================================================
GEMINI_API_KEY: str = get_env_var(
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "API_KEY_GEMINI",
    default=""
)
GOOGLE_API_KEY: str = GEMINI_API_KEY
GEMINI_MODEL: str = get_env_var("GEMINI_MODEL", default="gemini-2.5-flash")


# ==============================================================================
# 3. Exchange / Binance Settings
# ==============================================================================
BINANCE_API_KEY: str = get_env_var(
    "BINANCE_API_KEY",
    "BINANCE_KEY",
    "EXCHANGE_API_KEY",
    default=""
)
BINANCE_API_SECRET: str = get_env_var(
    "BINANCE_API_SECRET",
    "BINANCE_SECRET",
    "EXCHANGE_API_SECRET",
    default=""
)
BINANCE_TESTNET: bool = get_env_bool("BINANCE_TESTNET", default=False)

# Trading Mode: "paper" (simulated execution) or "live" (real orders)
TRADING_MODE: str = get_env_var("TRADING_MODE", default="paper").lower()
if TRADING_MODE not in ("paper", "live"):
    TRADING_MODE = "paper"


# ==============================================================================
# 4. Risk & Trade Sizing Parameters
# ==============================================================================
INITIAL_PAPER_BALANCE: float = get_env_float("INITIAL_PAPER_BALANCE", 10000.0)
DEFAULT_RISK_PER_TRADE_PCT: float = get_env_float("DEFAULT_RISK_PER_TRADE_PCT", 0.01)  # 1% per trade
MIN_REWARD_TO_RISK: float = get_env_float("MIN_REWARD_TO_RISK", 1.5)                  # 1.5 R:R minimum
DEFAULT_ORDER_SIZE: float = get_env_float("DEFAULT_ORDER_SIZE", 1.0)
MAX_OPEN_POSITIONS: int = get_env_int("MAX_OPEN_POSITIONS", 3)
MAX_DAILY_DRAWDOWN_PCT: float = get_env_float("MAX_DAILY_DRAWDOWN_PCT", 0.05)         # 5% max daily drawdown


# ==============================================================================
# 5. Asset & Symbol Specifications
# ==============================================================================
SUPPORTED_ASSETS: Dict[str, Dict[str, Any]] = {
    "BTC": {
        "symbol": "BTC",
        "name": "Bitcoin",
        "ccxt_symbol": "BTC/USDT",
        "display_name": "BTC/USDT",
        "price_decimals": 2,
        "lot_step": 0.001,
        "default_sl_pct": 0.015,
        "default_tp_pct": 0.030,
    },
    "ETH": {
        "symbol": "ETH",
        "name": "Ethereum",
        "ccxt_symbol": "ETH/USDT",
        "display_name": "ETH/USDT",
        "price_decimals": 2,
        "lot_step": 0.01,
        "default_sl_pct": 0.020,
        "default_tp_pct": 0.040,
    },
    "XAU": {
        "symbol": "XAU",
        "name": "Gold",
        "ccxt_symbol": "PAXG/USDT",
        "display_name": "XAU/USD (PAXG)",
        "price_decimals": 2,
        "lot_step": 0.01,
        "default_sl_pct": 0.010,
        "default_tp_pct": 0.020,
    },
}

# Alias mapping for user input normalization
ASSET_ALIASES: Dict[str, str] = {
    "BTC": "BTC",
    "BTCUSD": "BTC",
    "BTCUSDT": "BTC",
    "BITCOIN": "BTC",
    "ETH": "ETH",
    "ETHUSD": "ETH",
    "ETHUSDT": "ETH",
    "ETHEREUM": "ETH",
    "XAU": "XAU",
    "GOLD": "XAU",
    "XAUUSD": "XAU",
    "PAXG": "XAU",
    "PAXGUSDT": "XAU",
    "PAXGUSD": "XAU",
}


def resolve_asset(identifier: str) -> Optional[Dict[str, Any]]:
    """
    Resolves any user-entered symbol or alias (e.g. 'btc', 'gold', 'eth/usdt')
    to standard asset metadata.
    """
    if not identifier:
        return None
    cleaned = identifier.strip().upper().replace("/", "").replace("-", "").replace(" ", "")
    asset_key = ASSET_ALIASES.get(cleaned)
    if asset_key and asset_key in SUPPORTED_ASSETS:
        return SUPPORTED_ASSETS[asset_key]
    return None


# ==============================================================================
# 6. Server & Health Check Settings
# ==============================================================================
PORT: str = get_env_var("PORT", default="")
LOG_LEVEL: str = get_env_var("LOG_LEVEL", default="INFO").upper()
DEBUG: bool = get_env_bool("DEBUG", default=False)


# ==============================================================================
# 7. Diagnostic & Validation Helper
# ==============================================================================
def validate_config() -> Dict[str, Any]:
    """
    Validates current configuration, checking for missing or default values.
    Returns a status report dictionary.
    """
    status = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "details": {
            "trading_mode": TRADING_MODE,
            "telegram_configured": bool(TELEGRAM_BOT_TOKEN),
            "gemini_configured": bool(GEMINI_API_KEY),
            "binance_configured": bool(BINANCE_API_KEY and BINANCE_API_SECRET),
            "gemini_model": GEMINI_MODEL,
            "port": PORT or "Disabled",
            "supported_assets": list(SUPPORTED_ASSETS.keys()),
        }
    }

    if not TELEGRAM_BOT_TOKEN:
        status["errors"].append("TELEGRAM_BOT_TOKEN is not set.")
        status["valid"] = False

    if not GEMINI_API_KEY:
        status["warnings"].append("GEMINI_API_KEY is not set. AI analysis features will be disabled.")

    if TRADING_MODE == "live" and (not BINANCE_API_KEY or not BINANCE_API_SECRET):
        status["errors"].append("TRADING_MODE is 'live' but Binance API credentials are not set.")
        status["valid"] = False

    return status


def print_summary() -> None:
    """Prints a formatted summary of the active configuration."""
    diag = validate_config()
    print("=" * 60)
    print(" TRADING BOT CONFIGURATION")
    print("=" * 60)
    print(f" Trading Mode        : {TRADING_MODE.upper()}")
    print(f" Telegram Bot Token  : {'[CONFIGURED]' if TELEGRAM_BOT_TOKEN else '[MISSING]'}")
    print(f" Telegram Chat ID    : {TELEGRAM_CHAT_ID or '[NOT SET]'}")
    print(f" Gemini AI Model     : {GEMINI_MODEL} ({'[CONFIGURED]' if GEMINI_API_KEY else '[MISSING]'})")
    print(f" Binance API Keys    : {'[CONFIGURED]' if BINANCE_API_KEY else '[NOT SET]'}")
    print(f" Paper Balance       : ${INITIAL_PAPER_BALANCE:,.2f}")
    print(f" Risk Per Trade      : {DEFAULT_RISK_PER_TRADE_PCT * 100:.1f}%")
    print(f" Min Reward:Risk     : 1:{MIN_REWARD_TO_RISK:.1f}")
    print(f" Supported Assets    : {', '.join(SUPPORTED_ASSETS.keys())}")
    print(f" Healthcheck Port    : {PORT or '[NONE]'}")
    print("-" * 60)
    if diag["errors"]:
        print(" ERRORS:")
        for err in diag["errors"]:
            print(f"  - {err}")
    if diag["warnings"]:
        print(" WARNINGS:")
        for w in diag["warnings"]:
            print(f"  - {w}")
    if diag["valid"] and not diag["warnings"]:
        print(" STATUS: Ready")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
