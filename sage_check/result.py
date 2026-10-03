"""VLM detection result dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field

from sage_check.constants import CODE_TO_NAME


@dataclass
class VLMResult:
    """Structured result from the VLM detection service."""
    primary_code: str = ""
    primary_name: str = ""
    explanation: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @classmethod
    def from_response(cls, resp: dict) -> "VLMResult":
        """Build from the JSON response of the VLM detection service."""
        code = resp.get("primary_code", "")
        return cls(
            primary_code=code,
            primary_name=CODE_TO_NAME.get(code, code),
            explanation=resp.get("explanation", ""),
            probabilities=resp.get("probabilities", {}),
            raw_output=resp.get("raw_output", ""),
            latency_ms=resp.get("latency_ms", 0.0),
            error=resp.get("error", ""),
        )

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.primary_code)
