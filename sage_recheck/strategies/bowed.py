"""Bowed verification -- neck tilt or nose below shoulders."""
from __future__ import annotations

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy


class BowedStrategy(VerificationStrategy):
    behavior_name = "bowed"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person or not person.features:
            return EvidenceResult.insufficient("no person/features")
        feat = person.features
        if not feat.critical_kp_visible:
            return EvidenceResult.insufficient("critical keypoints not visible")
        # raw_evidence: negated neck tilt -- an upright neck strongly
        # contradicts the bowed (head-down) claim.
        raw = -feat.neck_tilt if feat.neck_tilt is not None else None
        if feat.neck_tilt is not None and feat.neck_tilt > 45:
            if self._looks_prone(person):
                return EvidenceResult.insufficient(
                    f"neck_tilt {feat.neck_tilt:.1f} deg > 45 but body horizontal (prone?)",
                    raw=raw)
            return EvidenceResult.consistent(
                f"neck_tilt {feat.neck_tilt:.1f} deg > 45", raw=raw)
        if feat.neck_tilt is not None and feat.neck_tilt > 38:
            return EvidenceResult.insufficient(
                f"neck_tilt {feat.neck_tilt:.1f} deg > 35 (borderline)", raw=raw)
        if feat.nose_below_shoulders and person.head_height is not None:
            hh = person.head_height
            if hh > 0:
                return EvidenceResult.insufficient(
                    f"nose below shoulders (head_h={hh:.1f})", raw=raw)
        if feat.neck_tilt is not None and feat.neck_tilt <= 35:
            return EvidenceResult.contradicted(
                f"neck_tilt {feat.neck_tilt:.1f} deg <= 35", raw=raw)
        return EvidenceResult.insufficient("neck_tilt unavailable")
