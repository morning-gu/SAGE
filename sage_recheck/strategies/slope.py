"""Slope verification -- shoulder tilt angle."""
from __future__ import annotations

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy


class SlopeStrategy(VerificationStrategy):
    behavior_name = "slope"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person or not person.features:
            return EvidenceResult.insufficient("no person/features")
        feat = person.features
        if not feat.has_both_shoulders:
            return EvidenceResult.insufficient("both shoulders missing")
        if feat.two_shoulder_near:
            return EvidenceResult.insufficient("shoulders too close")
        st = feat.shoulder_tilt
        if st < 0:
            return EvidenceResult.insufficient("shoulder tilt not computed")
        et = getattr(feat, "ear_tilt", 0.0)
        nt = getattr(feat, "neck_tilt", None)

        # Relative depth difference between shoulders -- large values indicate
        # body rotation (apparent tilt from projection) rather than true slope.
        di = person.depth_info
        left_d = di.get("left_depth") if di else None
        right_d = di.get("right_depth") if di else None
        if left_d is not None and right_d is not None:
            _mean_d = (left_d + right_d) / 2
            rdd = abs(left_d - right_d) / (_mean_d + 1e-6)
        else:
            rdd = 0.0

        extra = {"shoulder_tilt": st, "ear_tilt": et, "neck_tilt": nt,
                 "rel_depth_diff": rdd}
        # raw_evidence: negated shoulder tilt -- level shoulders (st small)
        # strongly contradict the slope (uneven-shoulder) claim.
        raw = -st

        if st > 50:
            return EvidenceResult.insufficient(
                f"shoulder_tilt {st:.1f} deg > 50 (extreme, maybe prone/recline)",
                raw=raw, **extra)

        # Lookup artifact: nose above eye line causes apparent shoulder tilt.
        kp = person.keypoints
        nose_above_eyes = False
        if kp and "0" in kp:
            nose_y = kp["0"][1]
            eye_ys = [kp[k][1] for k in ("1", "2") if k in kp]
            if eye_ys and nose_y < sum(eye_ys) / len(eye_ys):
                nose_above_eyes = True

        # Adaptive confirmed threshold: when the two shoulders differ
        # significantly in depth, the 2-D tilt is likely a projection
        # artifact from body rotation.  Require a stronger shoulder-tilt
        # signal to confirm slope in those cases.
        if rdd < 0.15:
            confirm_thresh = 20
        elif rdd < 0.25:
            confirm_thresh = 28
        else:
            confirm_thresh = 38

        # Artifact exclusions still apply at st > 15.
        if st > 15:
            if nose_above_eyes:
                return EvidenceResult.insufficient(
                    f"shoulder_tilt {st:.1f} deg > 15 but nose above eyes",
                    raw=raw, **extra)
            if feat.nose_below_shoulders:
                return EvidenceResult.insufficient(
                    f"shoulder_tilt {st:.1f} deg > 15 but nose below shoulders (prone?)",
                    raw=raw, **extra)
            # Use corrected ear_tilt (already accounts for head turn and
            # body-slope subtraction).  If still high, head is tilted
            # independently of body -- likely a head-tilt artifact.
            if et > 20:
                return EvidenceResult.insufficient(
                    f"shoulder_tilt {st:.1f} deg > 15 but ear_tilt {et:.1f} > 20",
                    raw=raw, **extra)
            # Head tilted forward/downward -- shoulder tilt is likely a
            # projection artifact from head tilt, not true body slope.
            if nt is not None and nt > 25:
                return EvidenceResult.insufficient(
                    f"shoulder_tilt {st:.1f} deg > 15 but neck_tilt {nt:.1f} > 25",
                    raw=raw, **extra)
            # Extreme body yaw -- shoulder tilt is a projection artifact
            # from rotation, not true slope.  Hard cutoff regardless of st.
            if rdd > 0.35:
                return EvidenceResult.insufficient(
                    f"shoulder_tilt {st:.1f} deg but rel_depth_diff {rdd:.2f} > 0.35 (body yaw)",
                    raw=raw, **extra)

        # Confirmed only when tilt exceeds the depth-adaptive threshold.
        if st > confirm_thresh:
            return EvidenceResult.consistent(
                f"shoulder_tilt {st:.1f} deg > {confirm_thresh} (rdd={rdd:.3f})",
                raw=raw, **extra)

        if st > 10:
            return EvidenceResult.insufficient(
                f"shoulder_tilt {st:.1f} deg borderline (10-{confirm_thresh}, rdd={rdd:.3f})",
                raw=raw, **extra)

        return EvidenceResult.contradicted(
            f"shoulder_tilt {st:.1f} deg <= 10", raw=raw, **extra)
