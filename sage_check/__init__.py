"""SAGE large-model detection client (calls the VLM detection service)."""
from __future__ import annotations

from sage_check.client import VLMClient
from sage_check.result import VLMResult


def check(image_base64: str, mode: str = "no_reason",
          client: VLMClient | None = None) -> VLMResult:
    """Run VLM detection on a single frame.

    Args:
        image_base64: single-frame image encoded as base64 string.
        mode: "no_reason" (code only) or "reason" (code + explanation).
        client: optional pre-configured client; defaults to VLMClient.from_env().
    Returns:
        VLMResult with primary_code, probabilities, explanation, etc.
    """
    c = client or VLMClient.from_env()
    resp = c.detect(image_base64, mode)
    return VLMResult.from_response(resp)


__all__ = ["check", "VLMClient", "VLMResult"]
