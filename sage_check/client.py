"""HTTP client for the VLM detection service."""
from __future__ import annotations

import os

import requests


class VLMClient:
    """Thin HTTP client wrapping the VLM detection service (/detect)."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

    @staticmethod
    def from_env() -> "VLMClient":
        url = os.environ.get("SAGE_VLM_URL", "http://localhost:8010")
        timeout = float(os.environ.get("SAGE_VLM_TIMEOUT", "60"))
        return VLMClient(url, timeout)

    def detect(self, image_b64: str, mode: str = "no_reason",
               max_tokens: int = 1024) -> dict:
        """POST /detect and return the parsed JSON response."""
        resp = requests.post(
            f"{self.base_url}/detect",
            json={"image": image_b64, "mode": mode, "max_tokens": max_tokens},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @property
    def available(self) -> bool:
        return bool(self.base_url)
