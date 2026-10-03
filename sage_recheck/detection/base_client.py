"""HTTP service client base."""
from __future__ import annotations

import time

import requests


class ServiceError(Exception):
    pass


class BaseServiceClient:
    def __init__(self, base_url: str, timeout: float = 30.0,
                 max_retries: int = 3, retry_backoff: float = 1.0):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code >= 500 and attempt < self.max_retries:
                    last_exc = ServiceError(
                        f"{url} -> {resp.status_code}: {resp.text[:200]}")
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise ServiceError(
                    f"{url} -> {resp.status_code}: {resp.text[:200]}")
            except requests.ConnectionError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise ServiceError(f"{url} connection error: {exc}") from exc
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise ServiceError(f"{url} timeout: {exc}") from exc
        raise last_exc or ServiceError(f"{url} -> exhausted retries")

    @property
    def available(self) -> bool:
        return bool(self.base_url)
