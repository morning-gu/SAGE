"""Strategy-layer tests on synthetic DetectionContexts.

Exercises the kept strategies through the public verify() interface (no
services, no GPU): LookupStrategy covers the gaze-line geometry,
ReclineStrategy the torso-extension check.  Golden expectations mirror
the verdict bands documented in each strategy module; raw_evidence
assertions pin the contradiction-oriented convention (larger = stronger
contradiction of the claim) that the ECDF layer (sage.veto §3) relies on.
"""
from __future__ import annotations

from types import SimpleNamespace

from sage_recheck.context import DetectionContext, PersonData
from sage_recheck.enums import EvidenceVerdict
from sage_recheck.strategies.lookup import LookupStrategy
from sage_recheck.strategies.recline import ReclineStrategy


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


class TestLookupStrategy:
    def _features(self):
        return SimpleNamespace(has_both_ears=True, upper_head_condition=True)

    def test_nose_above_eyes_consistent_negative_raw(self):
        # nose y=90 above eyes y=100; eye span 40 -> raw = -0.25
        kps = _kp(("0", 300, 90, 0.9), ("1", 280, 100, 0.9),
                  ("2", 320, 100, 0.9), ("3", 270, 105, 0.9),
                  ("4", 330, 105, 0.9))
        r = LookupStrategy().verify(_context([_person(kps, features=self._features())]))
        assert r.verdict is EvidenceVerdict.CONSISTENT
        assert r.raw_evidence == -0.25

    def test_nose_below_eyes_contradicted_positive_raw(self):
        # nose y=115 below eyes y=100 -> raw = +0.375
        kps = _kp(("0", 300, 115, 0.9), ("1", 280, 100, 0.9),
                  ("2", 320, 100, 0.9), ("3", 270, 105, 0.9),
                  ("4", 330, 105, 0.9))
        r = LookupStrategy().verify(_context([_person(kps, features=self._features())]))
        assert r.verdict is EvidenceVerdict.CONTRADICTED
        assert r.raw_evidence == 0.375

    def test_ears_missing_unavailable(self):
        kps = _kp(("0", 300, 90, 0.9), ("1", 280, 100, 0.9),
                  ("2", 320, 100, 0.9))
        feats = SimpleNamespace(has_both_ears=False, upper_head_condition=True)
        r = LookupStrategy().verify(_context([_person(kps, features=feats)]))
        assert r.verdict is EvidenceVerdict.INSUFFICIENT
        assert r.raw_evidence is None


class TestReclineStrategy:
    def _features(self, has_both_shoulders=True, has_both_hips=False):
        return SimpleNamespace(has_both_shoulders=has_both_shoulders,
                               has_both_hips=has_both_hips)

    def test_extended_torso_contradicts_with_ratio(self):
        # shoulders y=250, hips y=400 -> vgap=150; head_height=100 -> raw=1.5
        kps = _kp(("5", 300, 250, 0.9), ("6", 360, 250, 0.9),
                  ("11", 310, 400, 0.2), ("12", 350, 400, 0.2))
        r = ReclineStrategy().verify(
            _context([_person(kps, head_height=100, features=self._features())]))
        assert r.verdict is EvidenceVerdict.CONTRADICTED
        assert r.raw_evidence == 1.5
        assert "torso extended" in r.evidence

    def test_loose_hip_gate_accepted(self):
        # hips at conf 0.15 (below the old 0.3 gate, above the 0.1 floor)
        kps = _kp(("5", 300, 250, 0.9), ("6", 360, 250, 0.9),
                  ("11", 310, 400, 0.15), ("12", 350, 400, 0.15))
        r = ReclineStrategy().verify(
            _context([_person(kps, head_height=100, features=self._features())]))
        assert "hips missing" not in r.evidence  # hip path ran

    def test_compressed_torso_borderline(self):
        # vgap=60 < head_height=100 -> raw=0.6, borderline insufficient
        kps = _kp(("5", 300, 250, 0.9), ("6", 360, 250, 0.9),
                  ("11", 310, 310, 0.9), ("12", 350, 310, 0.9))
        r = ReclineStrategy().verify(
            _context([_person(kps, head_height=100, features=self._features())]))
        assert r.verdict is EvidenceVerdict.INSUFFICIENT
        assert r.raw_evidence == 0.6

    def test_depth_fallback_carries_no_raw(self):
        # no hips, but depth info present -> verdict-only path
        kps = _kp(("5", 300, 250, 0.9), ("6", 360, 250, 0.9))
        depth = {"nose_depth": 0.5, "mid_depth": 1.0}
        r = ReclineStrategy().verify(
            _context([_person(kps, head_height=100, features=self._features(),
                              depth_info=depth)]))
        assert r.verdict is EvidenceVerdict.CONTRADICTED
        assert r.raw_evidence is None  # uncalibrated path: no veto authority
