"""Final detection result dataclass (veto or fusion decision path)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.veto import VetoOutcome
    from sage_check.result import VLMResult
    from sage_recheck.enums import EvidenceResult


@dataclass
class FinalResult:
    """Final detection result combining VLM + small-model evidence."""
    final_code: str = ""
    final_name: str = ""
    confidence: float = 0.0
    source: str = ""
    evidence: str = ""
    vlm_result: "VLMResult | None" = None
    recheck_results: "dict[str, EvidenceResult] | None" = None
    adjusted_probabilities: dict[str, float] = field(default_factory=dict)
    veto_outcome: "VetoOutcome | None" = None
