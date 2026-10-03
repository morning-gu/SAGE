"""Fusion engine: log-odds full-class adjustment using recheck evidence.

Recheck acts as a post-VLM confidence layer.  For each behavior class the
VLM probability is converted to log-odds, shifted by a per-class
log-likelihood ratio derived from the recheck verdict, then converted back
via sigmoid.  The final prediction is argmax of the adjusted per-class
sigmoid probabilities (no renormalisation is needed for argmax).

Only ``confirmed`` and ``rejected`` verdicts carry adjustment weight.
``unavailable`` and ``weak`` are treated as *no evidence* (LR = 0) so that
the VLM distribution is preserved when recheck has no clear signal.

LR weights are loaded from an external JSON file (default:
``checkpoints/fuse_lr_weights.json``) so they can be re-estimated and swapped
without touching code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from sage.taxonomy import BEHAVIOR_CODES, CODE_TO_NAME
from sage_check.result import VLMResult
from sage_recheck.enums import EvidenceResult, EvidenceVerdict
from sage.result import FinalResult

from sage.paths import FUSE_WEIGHTS

# Default path to the external LR-weights JSON file.
DEFAULT_WEIGHTS_PATH = FUSE_WEIGHTS

# Map EvidenceVerdict enum to the string keys used in LR weights.
# E4 baseline path: the soft-fusion engine consumes the SAME strategy-layer
# output as the veto layer (frozen decision 4, docs/veto-algorithm.md §7) —
# consistent -> positive LR, contradicted -> negative LR, insufficient -> 0.
_VERDICT_KEY = {
    EvidenceVerdict.CONSISTENT: "confirmed",
    EvidenceVerdict.CONTRADICTED: "rejected",
    EvidenceVerdict.INSUFFICIENT: "insufficient",
}

_EPS = 1e-7


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_lr_weights(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-class LR weights from a JSON file.

    The JSON file should contain a ``"weights"`` key mapping each behavior
    name to a dict of ``{confirmed, rejected, weak, unavailable}`` float
    values.  A ``"_metadata"`` key is ignored.

    Args:
        path: Path to the weights JSON file.  Defaults to
            ``checkpoints/fuse_lr_weights.json`` under the repo root.
    Returns:
        Dict mapping behavior name to per-verdict LR weights.
    """
    p = Path(path) if path else DEFAULT_WEIGHTS_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["weights"]


class FusionEngine:
    """Log-odds full-class adjustment: recheck as a post-VLM confidence layer.

    For every class *c* the **binary log-odds** is adjusted:

        logit'(c) = logit(p_vlm(c)) + LR(c, verdict_recheck(c))

   Each adjusted log-odds is converted back to a probability via **sigmoid**
    (the inverse of the logit transform).  The final prediction is argmax of
    the 15 per-class sigmoid probabilities; no renormalisation is needed since
    argmax is invariant to any positive scaling.

    Only ``consistent`` / ``contradicted`` verdicts produce non-zero LR
    (weight keys ``confirmed`` / ``rejected``); an ``insufficient`` verdict
    leaves the VLM log-odds unchanged.
    """

    def __init__(
        self,
        weights: dict[str, dict[str, float]] | None = None,
        weights_path: str | Path | None = None,
    ):
        if weights is not None:
            self.weights = weights
        elif weights_path is not None:
            self.weights = load_lr_weights(weights_path)
        else:
            self.weights = load_lr_weights()

    # -- public API ---------------------------------------------------------

    def fuse(self, vlm: VLMResult,
             recheck: dict[str, EvidenceResult]) -> FinalResult:
        if not vlm.ok:
            return self._vlm_error_fallback(vlm, recheck)

        adjusted = self._adjust_probabilities(vlm, recheck)
        final_code = max(adjusted, key=adjusted.get)
        final_name = CODE_TO_NAME.get(final_code, final_code)
        confidence = adjusted[final_code]

        # Did recheck change the prediction relative to VLM?
        source = "vlm" if final_code == vlm.primary_code else "adjusted"

        return FinalResult(
            final_code=final_code,
            final_name=final_name,
            confidence=confidence,
            source=source,
            evidence=self._build_evidence(vlm, recheck, final_name),
            vlm_result=vlm,
            recheck_results=recheck,
            adjusted_probabilities=adjusted,
        )

    # -- core logic ---------------------------------------------------------

    def _adjust_probabilities(
        self, vlm: VLMResult, recheck: dict[str, EvidenceResult]
    ) -> dict[str, float]:
        """Return code-keyed adjusted probabilities via log-odds + sigmoid.

        Each class's VLM probability is converted to binary log-odds,
        shifted by the per-class LR weight, then mapped back through
        sigmoid.  The resulting per-class probabilities are returned as-is
        (no renormalisation); argmax over them is the final prediction.
        """
        probs: dict[str, float] = {}
        for code in BEHAVIOR_CODES:
            name = CODE_TO_NAME[code]
            p = vlm.probabilities.get(code, 0.0)
            p = max(min(p, 1.0 - _EPS), _EPS)
            log_odds = math.log(p / (1.0 - p))

            vr = recheck.get(name)
            key = _VERDICT_KEY.get(vr.verdict, "unavailable") if vr else "unavailable"
            log_odds += self.weights.get(name, {}).get(key, 0.0)

            probs[code] = _sigmoid(log_odds)

        return probs

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_evidence(vlm: VLMResult,
                        recheck: dict[str, EvidenceResult],
                        final_name: str) -> str:
        parts = [f"VLM->{vlm.primary_name}"]
        active = []
        for name, vr in recheck.items():
            if vr.verdict in (EvidenceVerdict.CONSISTENT,
                              EvidenceVerdict.CONTRADICTED):
                active.append(f"{name}:{vr.verdict.value}")
        if active:
            parts.append("recheck[" + ",".join(active) + "]")
        parts.append(f"final->{final_name}")
        return "; ".join(parts)

    @staticmethod
    def _vlm_error_fallback(vlm: VLMResult,
                            recheck: dict[str, EvidenceResult]) -> FinalResult:
        """Degrade to the default state when the VLM service reports an error.

        Claim-conditioned contract: without VLM claims there is nothing for
        the veto layer to check, so the small model cannot supply an
        alternative judgment — the system falls back to normal.
        """
        return FinalResult(
            final_code="nr", final_name="normal", confidence=0.0,
            source="vlm_error",
            evidence=f"VLM error: {vlm.error}; degraded to default state",
            vlm_result=vlm, recheck_results=recheck)
