"""Evidence verdict enum and result dataclass (claim-conditioned contract).

The strategy layer is a set of *claim-conditioned evidence checkers*: a
strategy is invoked only when the VLM claims the corresponding anomaly
class, and it returns a verdict on the CLAIM itself ("is the frame evidence
compatible with the presence of c?"), never an active anomaly judgment.

Verdict vocabulary (frozen 2026-09-22, docs/veto-algorithm.md):
  consistent    evidence is compatible with the claim    -> no action
  contradicted  evidence is inconsistent with the claim  -> vetoable
  insufficient  no decisive evidence (nothing computed,
                or computed but inconclusive)            -> no action

Mapping from the retired active-detection vocabulary:
  old confirmed   -> consistent    (evidence supports presence)
  old rejected    -> contradicted  (evidence contradicts presence)
  old weak        -> insufficient  (computed but inconclusive; the reason
                                    string preserves the detail)
  old unavailable -> insufficient  (nothing computed)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvidenceVerdict(Enum):
    CONSISTENT = "consistent"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


@dataclass
class EvidenceResult:
    """Result of a claim-conditioned evidence check for one anomaly class.

    raw_evidence: the strategy's primary continuous evidence quantity (e.g.
    shoulder tilt angle in degrees, object-to-wrist distance ratio). The
    veto layer normalizes this per-strategy into a quantile evidence
    strength for primary-demotion gating (docs/veto-algorithm.md §3).
    None when nothing was computed.
    """
    verdict: EvidenceVerdict = EvidenceVerdict.INSUFFICIENT
    evidence: str = ""
    metrics: dict = field(default_factory=dict)
    raw_evidence: float | None = None

    @classmethod
    def consistent(cls, evidence: str = "", raw: float | None = None,
                   **metrics) -> "EvidenceResult":
        return cls(EvidenceVerdict.CONSISTENT, evidence, metrics, raw)

    @classmethod
    def contradicted(cls, evidence: str = "", raw: float | None = None,
                     **metrics) -> "EvidenceResult":
        return cls(EvidenceVerdict.CONTRADICTED, evidence, metrics, raw)

    @classmethod
    def insufficient(cls, evidence: str = "", raw: float | None = None,
                     **metrics) -> "EvidenceResult":
        return cls(EvidenceVerdict.INSUFFICIENT, evidence, metrics, raw)
