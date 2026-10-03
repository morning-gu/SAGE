"""Prone verification -- only rejects (nose above shoulder line).

Cannot confirm: recheck does not detect desk contact, so nose-below-
shoulders alone is insufficient to confirm prone (could be bowed).
Only rejects when nose is clearly above the shoulder line.
"""
from __future__ import annotations

import math

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy


class ProneStrategy(VerificationStrategy):
    behavior_name = "prone"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person or not person.features:
            return EvidenceResult.insufficient("no person/features")
        feat = person.features
        if not feat.critical_kp_visible:
            return EvidenceResult.insufficient("critical keypoints not visible")
        # raw_evidence: nose height above the shoulder line, normalized by
        # shoulder width -- larger = head held clearly upright = stronger
        # contradiction of the prone (face-down) claim.
        raw = None
        kp = person.keypoints
        if kp:
            nose, ls, rs = kp.get("0"), kp.get("5"), kp.get("6")
            if nose and ls and rs:
                sw = math.hypot(ls[0] - rs[0], ls[1] - rs[1]) + 1e-6
                raw = ((ls[1] + rs[1]) / 2 - nose[1]) / sw
        st = feat.shoulder_tilt
        if feat.nose_below_shoulders:
            # Nose below shoulders but cannot confirm desk contact.
            return EvidenceResult.insufficient(
                f"nose below shoulders (head_h={person.head_height})", raw=raw)
        # Nose above shoulder line with moderate neck_tilt.
        if (feat.neck_tilt is not None and feat.neck_tilt > 35
                and st < 20):
            return EvidenceResult.insufficient(
                f"neck_tilt {feat.neck_tilt:.1f} deg > 35 (may be prone)", raw=raw)
        nr = feat.nose_ratio
        if nr is not None and (nr < 0 or nr > 1):
            return EvidenceResult.insufficient(
                f"nose_ratio {nr:.2f} unusual (outside [0,1]), "
                f"keypoints may be unreliable", raw=raw)
        return EvidenceResult.contradicted("nose above shoulder line", raw=raw)
