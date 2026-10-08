"""
Google Gemini API Connector Module
==================================
Handles communication with Google's Generative AI (Gemini) API via standard REST endpoints.
Supports model cascades, system instructions, temperature control, JSON extraction,
and connection diagnostic validation.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import requests

import config

logger = logging.getLogger("google_client")

# Candidate models in priority order
DEFAULT_CANDIDATE_MODELS = [
    config.GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


class GoogleAIClient:
    """
    Client for Google Gemini REST API.
    Zero-dependency implementation utilizing standard requests library.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = config.GEMINI_API_KEY if api_key is None else api_key
        self.model = model or config.GEMINI_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def is_configured(self) -> bool:
        """Returns True if an API key is set."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _get_headers(self) -> Dict[str, str]:
        """Build standard headers with API key authentication."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return headers

    def test_connection(self) -> Dict[str, Any]:
        """
        Validates the Google Gemini API key by testing connectivity.
        Returns a dictionary with connectivity details, HTTP status code, and diagnostics.
        """
        if not self.is_configured():
            return {
                "success": False,
                "status_code": 0,
                "error": "GEMINI_API_KEY is not configured in .env file.",
                "hint": "Set GEMINI_API_KEY in your .env file (obtain from https://aistudio.google.com/).",
                "model": self.model,
            }

        url = f"{self.base_url}/models/{self.model}"
        headers = self._get_headers()

        try:
            # Query the model metadata endpoint
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                logger.info("Successfully connected to Google Gemini API: %s", data.get("displayName", self.model))
                return {
                    "success": True,
                    "status_code": 200,
                    "error": None,
                    "model": self.model,
                    "display_name": data.get("displayName", self.model),
                    "supported_generation_methods": data.get("supportedGenerationMethods", []),
                }
            elif resp.status_code == 401:
                return {
                    "success": False,
                    "status_code": 401,
                    "error": "Invalid or unauthenticated GEMINI_API_KEY (HTTP 401).",
                    "hint": "Ensure you have a valid Google AI Studio API key (starts with 'AIzaSy...') from https://aistudio.google.com/",
                    "model": self.model,
                }
            elif resp.status_code == 404:
                return {
                    "success": False,
                    "status_code": 404,
                    "error": f"Model '{self.model}' not found (HTTP 404).",
                    "hint": "Try using 'gemini-1.5-flash' or 'gemini-2.0-flash'.",
                    "model": self.model,
                }
            else:
                err_text = resp.text[:300]
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "error": f"Google API returned HTTP {resp.status_code}: {err_text}",
                    "hint": None,
                    "model": self.model,
                }
        except requests.RequestException as exc:
            return {
                "success": False,
                "status_code": 0,
                "error": f"Network error connecting to Google API: {str(exc)}",
                "hint": "Check your internet connection or proxy settings.",
                "model": self.model,
            }

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        candidate_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a generation request to the Gemini API.
        Automatically falls back through candidate models if the requested model encounters an error.
        """
        if not self.is_configured():
            return {
                "success": False,
                "text": "",
                "error": "GEMINI_API_KEY is not configured.",
                "model_used": None,
            }

        models_to_try = candidate_models or [
            m for i, m in enumerate(DEFAULT_CANDIDATE_MODELS) if m and m not in DEFAULT_CANDIDATE_MODELS[:i]
        ]

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        headers = self._get_headers()
        last_error = "Unknown error"

        for model_name in models_to_try:
            url = f"{self.base_url}/models/{model_name}:generateContent"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        return {
                            "success": True,
                            "text": text,
                            "error": None,
                            "model_used": model_name,
                            "finish_reason": candidates[0].get("finishReason"),
                        }
                    return {
                        "success": False,
                        "text": "",
                        "error": "No candidates returned by Gemini API.",
                        "model_used": model_name,
                    }
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning("Gemini model %s call returned %s", model_name, last_error)
                    if resp.status_code == 401:
                        # 401 is credential failure; no need to retry other models with same key
                        break
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning("Network failure for model %s: %s", model_name, exc)

        return {
            "success": False,
            "text": "",
            "error": last_error,
            "model_used": None,
        }

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Requests a JSON response from Gemini, stripping markdown code fences if present.
        Returns: (success, parsed_dict, raw_or_error_text)
        """
        augmented_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON inside a standard ```json ... ``` codeblock or raw JSON object."
        )

        res = self.generate_content(
            prompt=augmented_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        if not res["success"]:
            return False, None, res["error"]

        text = res["text"].strip()
        cleaned = text

        # Strip markdown ```json codeblock
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            cleaned = match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            return True, parsed, text
        except json.JSONDecodeError as err:
            logger.warning("Failed to parse Gemini JSON: %s. Raw text: %s", err, text[:200])
            return False, None, f"JSONDecodeError: {err}"


def print_google_status() -> None:
    """Utility function to print Google Gemini API connection status."""
    client = GoogleAIClient()
    print("=" * 60)
    print(" GOOGLE GEMINI API CONNECTION TEST")
    print("=" * 60)
    print(f" Target Model : {client.model}")
    print(f" API Key      : {'[CONFIGURED]' if client.is_configured() else '[MISSING]'}")

    res = client.test_connection()
    if res["success"]:
        print(f" Status       : [CONNECTED] HTTP {res['status_code']}")
        print(f" Display Name : {res.get('display_name')}")
    else:
        print(f" Status       : [FAILED] HTTP {res['status_code']}")
        print(f" Error        : {res['error']}")
        if res.get("hint"):
            print(f" Hint         : {res['hint']}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_google_status()
