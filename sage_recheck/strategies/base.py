"""Strategy base classes."""
from __future__ import annotations

from abc import ABC, abstractmethod

from sage_recheck.context import DetectionContext, PersonData

# Keypoint confidence gate shared by pose-based strategies.
_KP_CONF = 0.3


class VerificationStrategy(ABC):
    """Base class for per-behavior geometric verification.

    Subclasses emit a continuous raw_evidence alongside the verdict,
    oriented so that LARGER values indicate a STRONGER contradiction of
    the claim (see sage_recheck.enums.EvidenceResult).  The veto layer
    normalizes it per-strategy into a quantile evidence strength
    (sage.veto §3); None means "nothing computed" and carries no veto
    authority under uniform tau gating.
    """

    @property
    @abstractmethod
    def behavior_name(self) -> str:
        pass

    def verify(self, context: DetectionContext) -> "object":
        raise NotImplementedError

    @staticmethod
    def _select_primary(context: DetectionContext) -> PersonData | None:
        """Select most prominent person: area*0.5 + center*0.3 + bottom*0.2.

        Persons without keypoints are deprioritized — a centered but
        keypoint-less person (e.g. occluded, low confidence) should not
        be selected over a less-centered person with full keypoints.
        """
        if not context.persons:
            return None
        candidates = [p for p in context.persons if p.keypoints]
        if not candidates:
            candidates = context.persons
        if len(candidates) == 1:
            return candidates[0]
        W = max(context.image_width, 1)
        H = max(context.image_height, 1)

        def score(p: PersonData) -> float:
            x1, y1, x2, y2 = p.box[:4]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            area = (w * h) / (W * H)
            center = 1 - abs(cx - W / 2) / (W / 2)
            bottom = cy / H
            return area * 0.5 + center * 0.3 + bottom * 0.2

        return max(candidates, key=score)

    # Prone gate thresholds: horizontal body indicates prone.
    _PRONE_TORSO_ANGLE: float = 45.0
    _PRONE_VGAP: float = 80.0

    @staticmethod
    def _looks_prone(person) -> bool:
        """Detect horizontal body suggesting prone posture."""
        feat = person.features
        if feat and feat.torso_angle is not None:
            if feat.torso_angle >= VerificationStrategy._PRONE_TORSO_ANGLE:
                return True
        kp = person.keypoints
        if kp:
            ls = kp.get("5")
            rs = kp.get("6")
            lh = kp.get("11")
            rh = kp.get("12")
            if (ls and rs and lh and rh
                    and len(ls) >= 3 and len(rs) >= 3
                    and len(lh) >= 3 and len(rh) >= 3):
                shoulders_ok = ls[2] >= 0.3 and rs[2] >= 0.3
                hips_ok = lh[2] >= 0.1 and rh[2] >= 0.1
                if shoulders_ok and hips_ok:
                    sh_cy = (ls[1] + rs[1]) / 2
                    hip_cy = (lh[1] + rh[1]) / 2
                    vgap = hip_cy - sh_cy
                    if vgap < VerificationStrategy._PRONE_VGAP:
                        return True
        return False
