"""Top-level SAGE pipeline: VLM detection -> claim-conditioned evidence -> veto.

Decision paths (frozen decision 4, docs/veto-algorithm.md §7):
  * ``decision="veto"``   (default): Algorithm 1 via :mod:`sage.veto`.
  * ``decision="fusion"``: log-odds soft fusion (E4 baseline), consuming the
    SAME strategy-layer evidence.

Both paths run strategies ONLY on VLM anomaly claims (claim-conditioned
contract). Constructing the VLM client and detection runner is expensive
(HTTP clients, models on disk), so engines are built once per pipeline —
this is the inference hot path.
"""
from __future__ import annotations

from sage.result import FinalResult

# Multi-label claim threshold (BCE decision boundary, paper §4.2).
CLAIM_THRESHOLD = 0.5


class SAGEPipeline:
    """Durable VLM -> evidence -> veto/fusion pipeline for a stream of frames."""

    def __init__(self, fusion: "FusionEngine | None" = None,
                 tau: float | None = None,
                 ecdf: "ECDFStore | None" = None):
        from sage_check.client import VLMClient
        from sage_recheck.detection.runner import DetectionRunner
        from sage_recheck.strategies.runner import StrategyRunner

        self.checker = VLMClient.from_env()
        self.runner = DetectionRunner.from_env()
        self.strategies = StrategyRunner()
        self.fusion = fusion  # built lazily; only the fusion path needs it
        self.tau = tau        # None disables all vetoes (conservative)
        self.ecdf = ecdf

    def detect(self, image_base64: str, mode: str = "no_reason",
               decision: str = "veto") -> FinalResult:
        from sage.taxonomy import NORMAL_CODE
        from sage.veto import apply_veto

        from sage_check.result import VLMResult

        try:
            resp = self.checker.detect(image_base64, mode)
            vlm = VLMResult.from_response(resp)
        except Exception as e:
            vlm = VLMResult(error=str(e))

        # Claims: VLM-predicted anomaly positives (normal out of scope).
        claims = [code for code, p in (vlm.probabilities or {}).items()
                  if p >= CLAIM_THRESHOLD and code != NORMAL_CODE]
        claim_names = [code_to_name(c) for c in claims]

        context = self.runner.run(image_base64)
        evidence = self.strategies.run(context, claim_names)

        if decision == "fusion":
            if self.fusion is None:
                from sage.fuse import FusionEngine
                self.fusion = FusionEngine()
            return self.fusion.fuse(vlm, evidence)

        outcome = apply_veto(
            vlm.probabilities or {}, vlm.primary_code, evidence,
            tau=self.tau if self.tau is not None else float("inf"),
            ecdf=self.ecdf, threshold=CLAIM_THRESHOLD)
        # tau=inf vetoes nothing (uniform gating never passes); deployment
        # must set the frozen tau.
        return FinalResult(
            final_code=outcome.final_code,
            final_name=outcome.final_name,
            confidence=(vlm.probabilities or {}).get(outcome.final_code, 0.0),
            source="veto" if outcome.vetoed else "vlm",
            evidence="; ".join(
                f"{r.name}:{r.verdict}" + (f"(~e={r.tilde_e:.2f})" if r.tilde_e is not None else "")
                for r in outcome.vetoed) or "no veto",
            vlm_result=vlm,
            recheck_results=evidence,
            veto_outcome=outcome)


def code_to_name(code: str) -> str:
    from sage.taxonomy import CODE_TO_NAME
    return CODE_TO_NAME.get(code, code)


_shared: SAGEPipeline | None = None


def get_pipeline() -> SAGEPipeline:
    """Return the process-wide shared pipeline, constructing it on first use."""
    global _shared
    if _shared is None:
        _shared = SAGEPipeline()
    return _shared
