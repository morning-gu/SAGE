"""Tests for the v5 face eye-line fallback (features + tilt strategy).

When the pose model's ear keypoints are missing or degenerate (head-down /
profile frames) but the face detector still sees the face, the 106-point
eye centers provide a conservative head-roll measurement: ear_tilt carries
the eye-line angle, ear_tilt_source == "face_eye_line", and the tilt
strategy bypasses the ear-availability gates with its own fixed threshold.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sage_recheck.context import DetectionContext, PersonData
from sage_recheck.enums import EvidenceVerdict
from sage_recheck.features import compute_head_features, eye_line_tilt, parse_keypoints
from sage_recheck.strategies.tilt import TiltStrategy


def _person(keypoints, head_height=None, features=None, depth_info=None):
    """PersonData with caller-supplied FeatureReport stand-in (or None)."""
    return PersonData(
        box=[200, 100, 400, 500],
        keypoints=keypoints,
        kp_valid={},
        head_height=head_height,
        depth_info=depth_info or {},
        features=features,
    )


def _context(persons, objects=(), width=800, height=600):
    return DetectionContext(
        image_base64="", image_width=width, image_height=height,
        persons=list(persons), objects=list(objects))


def _kp(*pairs):
    """keypoints dict from (id, x, y, conf) tuples."""
    return {i: [x, y, c] for i, x, y, c in pairs}


def _face_lm(re_xy, le_xy):
    """106-pt face landmarks with the two eye groups at the given centers."""
    lm = [[0.0, 0.0]] * 106
    for i in range(33, 43):
        lm[i] = [re_xy[0], re_xy[1]]
    for i in range(87, 97):
        lm[i] = [le_xy[0], le_xy[1]]
    return lm


def _kp_dict(ears_conf=0.0):
    """Frontal-facing pose keypoints; both ears at the given confidence.

    Frontal by the pipeline's conventions: COCO right ear (kp 4) sits left
    of the left ear (kp 3) in image coordinates, eyes above the ear line,
    nose centered between the ears (nose_ratio ~0.5 -> not unreliable).
    """
    coords = {
        "0": (100, 190), "1": (95, 185), "2": (105, 185),
        "3": (110, 200), "4": (90, 200),
        "5": (90, 300), "6": (110, 300),
    }
    kp = {}
    for i in range(17):
        x, y = coords.get(str(i), (100 + i, 400))
        kp[str(i)] = [x, y, ears_conf if str(i) in ("3", "4") else 0.9]
    return kp


class TestEyeLineFallback:
    def test_eye_line_tilt_angle(self):
        # right eye (110, 100), left eye (90, 90) -> roll atan2(10, 20)
        tilt = eye_line_tilt(_face_lm((110, 100), (90, 90)))
        assert tilt == pytest.approx(math.degrees(math.atan2(10, 20)))

    def test_eye_line_tilt_short_landmarks_return_none(self):
        assert eye_line_tilt([[0.0, 0.0]] * 50) is None

    def test_fallback_sets_source_and_clears_unreliable(self):
        kp = parse_keypoints(_kp_dict(ears_conf=0.0))  # ears missing
        head = compute_head_features(
            kp, shoulder_tilt=0.0, infer_shelter=False,
            face_lm=_face_lm((110, 100), (90, 90)))
        assert head["ear_tilt_source"] == "face_eye_line"
        assert head["ear_tilt_unreliable"] is False
        assert head["ear_tilt"] == pytest.approx(math.degrees(math.atan2(10, 20)))

    def test_no_face_keeps_sentinel(self):
        kp = parse_keypoints(_kp_dict(ears_conf=0.0))
        head = compute_head_features(kp, shoulder_tilt=0.0, infer_shelter=False,
                                     face_lm=None)
        assert head["ear_tilt_source"] is None
        assert head["ear_tilt"] == -1.0  # sentinel: nothing was measured

    def test_tilt_insufficient_on_sentinel(self):
        # ears "present" (conf > 0) but degenerate ear geometry and no face
        # fallback: the filler-0.0 bug used to read this as a level head
        feats = SimpleNamespace(has_both_ears=True, two_ear_near=False,
                                ear_tilt_unreliable=False, nose_ratio=0.5,
                                ear_tilt=-1.0, ear_tilt_source=None,
                                has_both_shoulders=True, neck_tilt=None,
                                shoulder_tilt=0.0, torso_angle=None)
        kps = _kp(("5", 300, 250, 0.9), ("6", 360, 250, 0.9),
                  ("11", 310, 400, 0.9), ("12", 350, 400, 0.9))
        person = _person(kps, head_height=100, features=feats)
        r = TiltStrategy().verify(_context([person]))
        assert r.verdict is EvidenceVerdict.INSUFFICIENT
        assert "ear line unusable" in r.evidence

    def test_face_preferred_over_pose_ears(self):
        # v6 face-primary roll: with both sources available the ROI-cropped
        # face eye line wins (the pose ear line misreads head roll on
        # classroom frames); pose ears remain the no-face fallback.
        kp = parse_keypoints(_kp_dict(ears_conf=0.9))
        head = compute_head_features(
            kp, shoulder_tilt=0.0, infer_shelter=False,
            face_lm=_face_lm((110, 100), (90, 90)))
        assert head["ear_tilt_source"] == "face_eye_line"
        assert head["ear_tilt"] == pytest.approx(math.degrees(math.atan2(10, 20)))

    def test_pose_ears_when_no_face(self):
        kp = parse_keypoints(_kp_dict(ears_conf=0.9))
        head = compute_head_features(kp, shoulder_tilt=0.0, infer_shelter=False,
                                     face_lm=None)
        assert head["ear_tilt_source"] == "pose_ears"


class TestTiltStrategyFallback:
    def _feat(self, **over):
        base = dict(has_both_ears=False, two_ear_near=False,
                    ear_tilt_unreliable=False, nose_ratio=0.5,
                    ear_tilt=math.degrees(math.atan2(10, 20)),
                    ear_tilt_source="face_eye_line",
                    has_both_shoulders=True, neck_tilt=None,
                    shoulder_tilt=0.0, torso_angle=None)
        base.update(over)
        return SimpleNamespace(**base)

    def _verify(self, feat):
        kps = _kp(("5", 300, 250, 0.9), ("6", 360, 250, 0.9),
                  ("11", 310, 400, 0.9), ("12", 350, 400, 0.9))
        person = _person(kps, head_height=100, features=feat)
        return TiltStrategy().verify(_context([person]))

    def test_fallback_bypasses_ear_gates_consistent_above_threshold(self):
        # eye-line roll ~26.6 deg > fallback threshold 23 -> consistent
        r = self._verify(self._feat())
        assert r.verdict is EvidenceVerdict.CONSISTENT

    def test_fallback_level_line_contradicts(self):
        r = self._verify(self._feat(ear_tilt=3.0))
        assert r.verdict is EvidenceVerdict.CONTRADICTED
        assert r.raw_evidence == -3.0

    def test_no_fallback_ears_missing_still_insufficient(self):
        r = self._verify(self._feat(ear_tilt_source=None, ear_tilt=0.0))
        assert r.verdict is EvidenceVerdict.INSUFFICIENT
        assert "both ears missing" in r.evidence

    def test_fallback_unreliable_flag_ignored(self):
        # the pose-side unreliability flag must not block the fallback path
        r = self._verify(self._feat(ear_tilt_unreliable=True))
        assert r.verdict is EvidenceVerdict.CONSISTENT
