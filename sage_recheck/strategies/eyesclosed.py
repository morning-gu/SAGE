"""Eyesclosed verification -- Eye Aspect Ratio from 106-point face landmarks."""
from __future__ import annotations

import math

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy

EAR_THRESHOLD = 0.05
EAR_WEAK_THRESHOLD = 0.20

_RIGHT_EYE = {"top": 40, "bottom": 33, "outer": 35, "inner": 39}
_LEFT_EYE = {"top": 94, "bottom": 87, "inner": 89, "outer": 93}


def _compute_ear(lm, eye) -> float:
    top = lm[eye["top"]]
    bot = lm[eye["bottom"]]
    ca = lm[eye["outer"]]
    cb = lm[eye["inner"]]
    v = math.hypot(top[0] - bot[0], top[1] - bot[1])
    h = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) + 1e-6
    return v / h


class EyesclosedStrategy(VerificationStrategy):
    behavior_name = "eyesclosed"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person:
            return EvidenceResult.insufficient("no person")
        face = person.face_data
        if not face or not face.get("detected"):
            return EvidenceResult.insufficient("face not detected")
        lm = face.get("landmarks") or []
        if len(lm) < 106:
            return EvidenceResult.insufficient("insufficient landmarks")
        left = _compute_ear(lm, _LEFT_EYE)
        right = _compute_ear(lm, _RIGHT_EYE)
        eye_max = max(left, right)
        et = {"left_ear": round(left, 4), "right_ear": round(right, 4)}
        # raw_evidence: EAR -- larger = eyes more open = stronger
        # contradiction of the eyesclosed claim.
        raw = eye_max
        if eye_max < EAR_THRESHOLD:
            feat = person.features
            if not feat:
                return EvidenceResult.insufficient(
                    "EAR low but features unavailable", raw=raw, **et)
            nr = feat.nose_ratio
            nt = feat.neck_tilt
            if nr is not None and (nr < 0.2 or nr > 0.8):
                return EvidenceResult.insufficient(
                    f"EAR low but side view (nose_ratio={nr:.2f})",
                    raw=raw, **et)
            if nt is not None and nt > 15:
                return EvidenceResult.insufficient(
                    f"EAR low but head down (neck_tilt={nt:.1f} deg)",
                    raw=raw, **et)
            if feat.nose_below_shoulders:
                return EvidenceResult.insufficient(
                    "EAR low but nose below shoulders", raw=raw, **et)
            if nt is None and nr is None:
                return EvidenceResult.insufficient(
                    "EAR low but head pose unavailable", raw=raw, **et)
            return EvidenceResult.consistent(
                f"EAR left={left:.3f} right={right:.3f}", raw=raw, **et)
        if eye_max < EAR_WEAK_THRESHOLD:
            return EvidenceResult.insufficient(
                f"EAR borderline left={left:.3f} right={right:.3f}", raw=raw, **et)
        return EvidenceResult.contradicted(
            f"eyes open (EAR left={left:.3f} right={right:.3f})", raw=raw, **et)
