"""Tilt verification -- ear tilt with layered thresholds by nose_ratio."""
from __future__ import annotations

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy


# Shoulder tilt threshold: high shoulder_tilt indicates body rotation
# (typical of prone), not a simple head tilt while sitting upright.
_PRONE_SHOULDER_TILT = 30.0

# Fixed contradiction threshold for the face eye-line fallback (v5): the
# eye-line angle is not corrected for head turn and reads systematically
# larger than the pose ear line, so the nose_ratio-adaptive ear-line bands
# do not transfer.  23 deg matches the mid-band ear-line threshold.
_FALLBACK_TH = 23.0


class TiltStrategy(VerificationStrategy):
    behavior_name = "tilt"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person or not person.features:
            return EvidenceResult.insufficient("no person/features")
        feat = person.features
        # v5: when the pose ear geometry is missing or degenerate but the
        # face landmarks are available, ear_tilt carries the face eye-line
        # measurement (ear_tilt_source == "face_eye_line") and the ear-
        # availability gates are bypassed.
        fallback = getattr(feat, "ear_tilt_source", None) == "face_eye_line"
        if not feat.has_both_ears and not fallback:
            return EvidenceResult.insufficient("both ears missing")
        if feat.two_ear_near and not fallback:
            return EvidenceResult.insufficient("ears too close")
        if feat.ear_tilt_unreliable and not fallback:
            return EvidenceResult.insufficient("ear_tilt unreliable (extreme nose_ratio)")
        nr = feat.nose_ratio or 0.5
        ce = feat.ear_tilt
        if ce < 0:
            # -1 sentinel: ears present but the ear line is degenerate and
            # no face fallback was available -- nothing was measured.
            return EvidenceResult.insufficient(
                "ear line unusable (degenerate ear geometry, no face fallback)")
        # raw_evidence: negated ear tilt -- an upright head (ce small)
        # strongly contradicts the tilt claim.
        raw = -ce
        if ce > 45:
            return EvidenceResult.insufficient(
                f"tilt {ce:.1f} deg > 45 (extreme)", raw=raw, ear_tilt=ce,
                nose_ratio=nr, source=getattr(feat, "ear_tilt_source", None))
        if fallback:
            th = _FALLBACK_TH
        elif 0.35 <= nr <= 0.65:
            th = 23.5
        elif 0.25 <= nr < 0.35 or 0.65 < nr <= 0.75:
            th = 23.0
        else:
            th = 18.0
        if ce > th:
            # Prone gate: horizontal body suggests lying face-down.
            if self._looks_prone(person):
                return EvidenceResult.insufficient(
                    f"tilt {ce:.1f} deg > {th} but body horizontal (prone?)",
                    raw=raw, ear_tilt=ce, nose_ratio=nr, threshold=th)
            # Shoulder tilt gate: high shoulder_tilt indicates body
            # rotation typical of prone, not head tilt while sitting.
            st = feat.shoulder_tilt
            if st is not None and st > _PRONE_SHOULDER_TILT:
                return EvidenceResult.insufficient(
                    f"tilt {ce:.1f} deg > {th} but shoulder_tilt {st:.1f} > {_PRONE_SHOULDER_TILT:.0f} (prone?)",
                    raw=raw, ear_tilt=ce, nose_ratio=nr, threshold=th)
            # Neck tilt gate (P0-2): forward bow looks like tilt but
            # neck_tilt dominates ear_tilt, indicating sagittal motion.
            nt = feat.neck_tilt
            if nt is not None and nt > 35 and ce <= nt:
                return EvidenceResult.insufficient(
                    f"neck_tilt {nt:.1f} > 35 and ear_tilt {ce:.1f} <= neck_tilt "
                    f"(forward bow, not lateral tilt)",
                    raw=raw, ear_tilt=ce, nose_ratio=nr, neck_tilt=nt)
            return EvidenceResult.consistent(
                f"tilt {ce:.1f} deg > {th}", raw=raw, ear_tilt=ce, nose_ratio=nr,
                threshold=th, source=getattr(feat, "ear_tilt_source", None))
        return EvidenceResult.contradicted(
            f"tilt {ce:.1f} deg <= {th}", raw=raw, ear_tilt=ce, nose_ratio=nr,
            threshold=th, source=getattr(feat, "ear_tilt_source", None))
