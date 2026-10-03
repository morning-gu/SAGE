"""Lookup verification -- nose actually above eye line.

raw_evidence: nose position relative to the eye line, normalized by eye
span -- larger (more positive) = nose clearly below the eyes = stronger
contradiction of the lookup claim; negative = nose above the eye line.
"""
from __future__ import annotations

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy


class LookupStrategy(VerificationStrategy):
    behavior_name = "lookup"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person or not person.features or not person.keypoints:
            return EvidenceResult.insufficient("no person/features")
        feat = person.features
        if not feat.has_both_ears:
            return EvidenceResult.insufficient("ears missing")
        kp = person.keypoints
        if "0" not in kp:
            return EvidenceResult.insufficient("nose missing")
        nose_y = kp["0"][1]
        eye_ys = [kp[k][1] for k in ("1", "2") if k in kp]
        if not eye_ys:
            return EvidenceResult.insufficient("eyes missing")
        eye_avg_y = sum(eye_ys) / len(eye_ys)
        eye_span = abs(kp["1"][0] - kp["2"][0]) if ("1" in kp and "2" in kp) else 0.0
        raw = (nose_y - eye_avg_y) / eye_span if eye_span >= 1 else None
        # Looking up: nose is above eyes (lower Y) AND geometric gate allows it
        if nose_y < eye_avg_y and feat.upper_head_condition:
            return EvidenceResult.consistent("nose above eye line", raw=raw)
        # Moderate lookup: ears dropped below eyes while nose stays near eye line.
        ear_ys = [kp[k][1] for k in ("3", "4") if k in kp]
        if len(eye_ys) == 2 and len(ear_ys) == 2:
            ear_avg_y = sum(ear_ys) / 2
            if eye_span >= 10:
                ear_drop = (ear_avg_y - eye_avg_y) / eye_span
                both_ears_below = all(e > eye_avg_y for e in ear_ys)
                if nose_y < ear_avg_y and both_ears_below and ear_drop > 0.5:
                    # Very strong ear drop -> confirmed; moderate -> weak.
                    if ear_drop > 1.2:
                        return EvidenceResult.consistent(
                            f"strong ear_drop ({ear_drop:.2f})",
                            raw=raw, ear_drop=ear_drop)
                    return EvidenceResult.insufficient(
                        f"ears below eyes (ear_drop={ear_drop:.2f})",
                        raw=raw, ear_drop=ear_drop)
        return EvidenceResult.contradicted("nose below eye line", raw=raw)
