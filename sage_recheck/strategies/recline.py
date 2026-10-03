"""Recline verification -- torso extension (hip-based) or depth ratio.

Optimized 2026-09-26 after the first E3 collection showed the hip path
almost never firing (5% decisive rate): hips are read directly at the
same looser confidence gate (0.1) used by the posture feature extractor,
instead of the strict 0.3 has_both_hips gate, which desk occlusion
almost always failed.

raw_evidence: torso extension ratio vgap / head_height (hip path only) --
larger = more extended, vertical torso = stronger contradiction of the
recline claim. The depth-ratio fallback keeps its verdict but carries no
raw_evidence: without a calibrated strength it cannot veto under uniform
tau gating (sage.veto), it only records the reason.
"""
from __future__ import annotations

from sage_recheck.context import DetectionContext
from sage_recheck.enums import EvidenceResult
from sage_recheck.strategies.base import VerificationStrategy

# Hip confidence gate: matches compute_posture_features (_POSTURE_HIP_CONF).
# Seated students' hips are usually desk-occluded; 0.3 rejects almost all
# of them, leaving the strategy unable to check the VLM's recline claims
# (its single worst class: 74% primary error in E1/E2).
_HIP_CONF = 0.1


class ReclineStrategy(VerificationStrategy):
    behavior_name = "recline"

    def verify(self, context: DetectionContext) -> EvidenceResult:
        person = self._select_primary(context)
        if not person or not person.features:
            return EvidenceResult.insufficient("no person/features")
        feat = person.features
        if not feat.has_both_shoulders or not person.keypoints:
            return EvidenceResult.insufficient("shoulders missing")
        kp = person.keypoints

        # --- Primary: hip-based torso extension ---
        lh, rh = kp.get("11"), kp.get("12")
        if (lh and rh and len(lh) >= 3 and len(rh) >= 3
                and lh[2] >= _HIP_CONF and rh[2] >= _HIP_CONF):
            sh_cy = (kp["5"][1] + kp["6"][1]) / 2
            hip_cy = (lh[1] + rh[1]) / 2
            vgap = hip_cy - sh_cy
            hh = person.head_height
            if not hh or hh <= 0:
                return EvidenceResult.insufficient(
                    f"vgap={vgap:.1f} but no head height for normalization")
            raw = vgap / hh
            if vgap > hh:
                return EvidenceResult.contradicted(
                    f"torso extended (vgap={vgap:.1f}, ratio={raw:.2f})",
                    raw=raw, vgap=vgap)
            return EvidenceResult.insufficient(
                f"torso borderline (vgap={vgap:.1f}, ratio={raw:.2f})",
                raw=raw, vgap=vgap)

        # --- Fallback: depth ratio (verdict-only, no calibrated strength) ---
        di = person.depth_info
        if not di:
            return EvidenceResult.insufficient("hips missing, no depth info")
        nose_d = di.get("nose_depth")
        mid_d = di.get("mid_depth")
        if nose_d is None or mid_d is None or mid_d < 0.01:
            return EvidenceResult.insufficient("depth values missing/invalid")

        ratio = nose_d / mid_d
        if ratio < 1.0:
            return EvidenceResult.contradicted(
                f"depth ratio {ratio:.2f} < 1.0 (not recline)")
        return EvidenceResult.insufficient(
            f"depth ratio {ratio:.2f} (cannot confirm recline)")
