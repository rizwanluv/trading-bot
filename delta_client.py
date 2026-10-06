"""
Delta Exchange API Client for authenticated trading and account management.
Supports Delta Exchange India (https://api.india.delta.exchange) and Global (https://api.delta.exchange).
"""
import os
import time
import email.utils
import hmac
import hashlib
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.india.delta.exchange"


def _load_env_fallback():
    """Load credentials from .env if environment variables are not pre-set."""
    if not os.environ.get("DELTA_API_KEY") and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_fallback()


class DeltaClient:
    """Authenticated client for Delta Exchange REST API v2."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        _load_env_fallback()
        self.api_key = api_key or os.environ.get("DELTA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("DELTA_API_SECRET", "")
        self.base_url = (base_url or os.environ.get("DELTA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._products_cache: Dict[str, Dict[str, Any]] = {}
        self._products_cache_time: float = 0.0
        self._server_time_offset: float = 0.0

        # Fallback known product mapping for Delta Exchange India
        self._fallback_products = {
            "BTCUSD": {"id": 27, "symbol": "BTCUSD", "contract_value": 0.001, "tick_size": 0.5},
            "XAUTUSD": {"id": 131253, "symbol": "XAUTUSD", "contract_value": 0.001, "tick_size": 0.01},
            "ETHUSD": {"id": 28, "symbol": "ETHUSD", "contract_value": 0.01, "tick_size": 0.05},
            "SOLUSD": {"id": 69, "symbol": "SOLUSD", "contract_value": 0.1, "tick_size": 0.01},
        }

    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.api_key and self.api_secret)

    def set_credentials(self, api_key: str, api_secret: str, base_url: Optional[str] = None):
        """Update API credentials at runtime."""
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        if base_url:
            self.base_url = base_url.rstrip("/")

    def set_base_url(self, base_url: str):
        """Set Delta Exchange base URL (e.g. India vs Global)."""
        self.base_url = base_url.strip().rstrip("/")
        self._products_cache.clear()
        self.sync_server_time()

    def sync_server_time(self) -> float:
        """Fetch current Delta Exchange server time and update internal clock offset."""
        try:
            r = requests.get(
                f"{self.base_url}/v2/products",
                params={"page_size": 1, "_cb": str(int(time.time()))},
                timeout=5,
            )
            date_header = r.headers.get("date") or r.headers.get("Date")
            if date_header:
                server_ts = email.utils.parsedate_to_datetime(date_header).timestamp()
                diff = server_ts - time.time()
                # Delta allows a ±5-second window. Only apply if drift exceeds 3 seconds.
                if abs(diff) > 3.0:
                    self._server_time_offset = diff
                else:
                    self._server_time_offset = 0.0
        except Exception as e:
            logger.debug(f"Could not synchronize server time: {e}")
        return self._server_time_offset

    def get_masked_key(self) -> str:
        """Return masked API key for safe display in Telegram."""
        if not self.api_key:
            return "Not Set"
        if len(self.api_key) <= 8:
            return f"{self.api_key[:2]}***{self.api_key[-2:]}"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def generate_signature(
        self, method: str, path: str, query_string: str = "", payload: Any = ""
    ) -> Tuple[str, str]:
        """
        Generate HMAC-SHA256 signature according to Delta Exchange specification:
        signature_data = method + timestamp + path + query_string + payload
        Timestamp is automatically synchronized with Delta Exchange server time.
        """
        timestamp = str(int(time.time() + self._server_time_offset))
        if isinstance(payload, (dict, list)):
            payload_str = json.dumps(payload, separators=(",", ":"))
        else:
            payload_str = str(payload or "")

        signature_data = method.upper() + timestamp + path + (query_string or "") + payload_str
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature, timestamp

    def _get_headers(
        self, method: str, path: str, query_string: str = "", payload: Any = ""
    ) -> Dict[str, str]:
        """Build headers with authentication signature."""
        sig, ts = self.generate_signature(method, path, query_string, payload)
        return {
            "api-key": self.api_key,
            "signature": sig,
            "timestamp": ts,
            "Content-Type": "application/json",
            "User-Agent": "GeminiTradingAssistant/2.0",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        auth_required: bool = True,
        timeout: int = 15,
        _is_retry: bool = False,
    ) -> Dict[str, Any]:
        """Execute HTTP request to Delta Exchange API with automatic server clock synchronization."""
        if auth_required and not self.is_configured():
            raise ValueError("Delta API key and secret are not configured.")

        query_string = ""
        if params:
            parts = [f"{k}={v}" for k, v in sorted(params.items()) if v is not None]
            if parts:
                query_string = "?" + "&".join(parts)

        payload_str = ""
        if data is not None:
            payload_str = json.dumps(data, separators=(",", ":"))

        url = f"{self.base_url}{path}{query_string}"

        if auth_required:
            headers = self._get_headers(method, path, query_string, payload_str)
        else:
            headers = {"Content-Type": "application/json"}

        try:
            r = requests.request(
                method=method.upper(),
                url=url,
                headers=headers,
                data=payload_str if payload_str else None,
                timeout=timeout,
            )
            r.raise_for_status()
            res = r.json()
            if isinstance(res, dict) and not res.get("success", True):
                err = res.get("error", "Unknown API error")
                raise RuntimeError(f"Delta API error: {err}")
            return res
        except requests.exceptions.HTTPError as e:
            try:
                err_data = r.json()
                err_dict = err_data.get("error", {})
                # Auto-heal clock drift if server reports expired_signature
                if isinstance(err_dict, dict) and err_dict.get("code") == "expired_signature":
                    server_time = err_dict.get("context", {}).get("server_time")
                    if server_time and not _is_retry:
                        self._server_time_offset = float(server_time) - time.time()
                        return self._request(
                            method=method,
                            path=path,
                            params=params,
                            data=data,
                            auth_required=auth_required,
                            timeout=timeout,
                            _is_retry=True,
                        )
                if isinstance(err_dict, dict):
                    err_msg = err_dict.get("code") or err_dict.get("message") or str(err_dict)
                else:
                    err_msg = str(err_dict or err_data)
            except Exception:
                err_msg = r.text if hasattr(r, "text") else str(e)
            raise RuntimeError(f"Delta API HTTP {r.status_code}: {err_msg}")
        except Exception as e:
            raise RuntimeError(f"Delta API connection error: {e}")

    # ==================== Public Methods ====================

    def get_products(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch all products list and cache locally."""
        now = time.time()
        if not force_refresh and self._products_cache and (now - self._products_cache_time < 1800):
            return list(self._products_cache.values())

        try:
            res = self._request("GET", "/v2/products", auth_required=False)
            items = res.get("result", [])
            for item in items:
                sym = item.get("symbol")
                if sym:
                    self._products_cache[sym] = item
            self._products_cache_time = now
            return items
        except Exception as e:
            logger.warning(f"Failed to fetch products from Delta: {e}. Using fallback.")
            if not self._products_cache:
                self._products_cache = dict(self._fallback_products)
            return list(self._products_cache.values())

    def get_product(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get product details by canonical symbol (e.g. BTCUSD, XAUTUSD)."""
        sym = symbol.upper()
        if not self._products_cache:
            self.get_products()
        if sym in self._products_cache:
            return self._products_cache[sym]
        if sym in self._fallback_products:
            return self._fallback_products[sym]
        # Refresh and retry
        self.get_products(force_refresh=True)
        return self._products_cache.get(sym) or self._fallback_products.get(sym)

    def resolve_product_id(self, symbol: str) -> int:
        """Resolve symbol string to integer product_id."""
        prod = self.get_product(symbol)
        if prod and "id" in prod:
            return int(prod["id"])
        if symbol.upper() in ("BTCUSD", "BTC"):
            return 27
        if symbol.upper() in ("XAUTUSD", "XAUUSD", "GOLD"):
            return 131253
        if symbol.upper() in ("ETHUSD", "ETH"):
            return 28
        if symbol.upper() in ("SOLUSD", "SOL"):
            return 69
        raise ValueError(f"Could not resolve product ID for symbol: {symbol}")

    def get_product_id(self, symbol: str) -> int:
        """Alias for resolve_product_id."""
        return self.resolve_product_id(symbol)

    # Alias for private signature generator
    _generate_signature = generate_signature

    # ==================== Authenticated Methods ====================

    def test_connection(self) -> Dict[str, Any]:
        """Test API credentials by fetching wallet balances safely."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "not_configured",
                "message": "Delta Exchange API Key and Secret are not configured.",
                "base_url": self.base_url,
                "masked_key": self.get_masked_key(),
            }

        try:
            self.sync_server_time()
            res = self._request("GET", "/v2/wallet/balances", auth_required=True)
            balances = res.get("result", [])
            return {
                "success": True,
                "message": "Delta Exchange API credentials successfully validated!",
                "balances": balances,
                "base_url": self.base_url,
                "masked_key": self.get_masked_key(),
            }
        except Exception as e:
            err_str = str(e)
            return {
                "success": False,
                "error": err_str,
                "message": f"Delta Exchange connection test failed: {err_str}",
                "base_url": self.base_url,
                "masked_key": self.get_masked_key(),
            }

    def get_wallet_balances(self) -> List[Dict[str, Any]]:
        """Retrieve account wallet balances."""
        res = self._request("GET", "/v2/wallet/balances", auth_required=True)
        return res.get("result", [])

    def get_positions(self, product_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve open positions with margin and liquidation details."""
        params = {}
        if product_id:
            params["product_id"] = product_id
        res = self._request("GET", "/v2/positions/margined", params=params, auth_required=True)
        positions = res.get("result", [])
        # Filter non-zero positions
        return [p for p in positions if float(p.get("size", 0)) != 0.0]

    def get_open_orders(self, product_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve active working orders."""
        params = {"state": "open"}
        if product_id:
            params["product_id"] = product_id
        res = self._request("GET", "/v2/orders", params=params, auth_required=True)
        return res.get("result", [])

    def place_order(
        self,
        symbol: str,
        size: int,
        side: str,
        order_type: str = "market_order",
        limit_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Place an order on Delta Exchange.
        Supports bracket parameters for automatic Stop-Loss and Take-Profit.
        """
        product_id = self.resolve_product_id(symbol)
        side_clean = side.strip().lower()
        if side_clean not in ("buy", "sell"):
            raise ValueError("Order side must be 'buy' or 'sell'.")

        payload: Dict[str, Any] = {
            "product_id": product_id,
            "size": int(size),
            "side": side_clean,
            "order_type": order_type,
            "reduce_only": bool(reduce_only),
        }

        if limit_price is not None:
            payload["limit_price"] = str(round(limit_price, 2))

        if stop_loss is not None:
            payload["bracket_stop_loss_price"] = str(round(stop_loss, 2))

        if take_profit is not None:
            payload["bracket_take_profit_price"] = str(round(take_profit, 2))

        res = self._request("POST", "/v2/orders", data=payload, auth_required=True)
        return res.get("result", res)

    def cancel_order(self, order_id: int, product_id: int) -> Dict[str, Any]:
        """Cancel a specific order by order_id."""
        payload = {"id": int(order_id), "product_id": int(product_id)}
        res = self._request("DELETE", f"/v2/orders/{order_id}", data=payload, auth_required=True)
        return res.get("result", res)

    def cancel_all_orders(self, product_id: Optional[int] = None) -> Dict[str, Any]:
        """Cancel all open orders, optionally filtered by product_id."""
        payload = {}
        if product_id:
            payload["product_id"] = int(product_id)
        res = self._request("DELETE", "/v2/orders/all", data=payload, auth_required=True)
        return res.get("result", res)

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close open position for symbol by submitting market order in opposite direction."""
        product_id = self.resolve_product_id(symbol)
        positions = self.get_positions(product_id=product_id)
        if not positions:
            raise ValueError(f"No open position found for {symbol}.")

        pos = positions[0]
        size = float(pos.get("size", 0))
        if size == 0:
            raise ValueError(f"Position size is 0 for {symbol}.")

        close_side = "sell" if size > 0 else "buy"
        close_size = int(abs(size))

        return self.place_order(
            symbol=symbol,
            size=close_size,
            side=close_side,
            order_type="market_order",
            reduce_only=True,
        )
