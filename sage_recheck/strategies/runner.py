"""StrategyRunner: claim-conditioned evidence checks.

Strategies are invoked ONLY for the anomaly classes the VLM claims
(frozen decision 3, docs/veto-algorithm.md §1): the small model never
initiates an anomaly judgment of its own. Normal is outside the veto
scope — no strategy exists for it.
"""
from __future__ import annotations

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies import PRIMARY_STRATEGIES

_BY_NAME = {s.behavior_name: s for s in PRIMARY_STRATEGIES}


class StrategyRunner:
    def run(self, context: DetectionContext,
            claims: list[str] | None = None) -> dict[str, EvidenceResult]:
        """Check the given anomaly claims (behavior names).

        claims=None runs all registered strategies (used by the
        evidence-collection stage to build calibration data); with claims,
        only those strategies run, mirroring the claim-conditioned
        inference contract.
        """
        names = claims if claims is not None else list(_BY_NAME)
        results: dict[str, EvidenceResult] = {}
        for name in names:
            s = _BY_NAME.get(name)
            if s is None:
                continue  # normal or unknown class: outside veto scope
            try:
                results[name] = s.verify(context)
            except Exception as e:
                results[name] = EvidenceResult.insufficient(f"error: {e}")
        return results
