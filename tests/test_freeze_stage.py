"""Tests for the v5 frozen-threshold freeze stage (scripts/eval_veto.py).

The freeze rule mirrors the tau freeze: for each interpretable-angle
strategy, max_angle = largest ceiling on the grid such that vetoing all
claimed-and-contradicted validation frames with angle <= ceiling has
FVR <= delta.  Pure-function tests on synthetic records.
"""
from __future__ import annotations

from scripts.eval_veto import freeze_angle_thresholds
from sage.taxonomy import NAME_TO_CODE


def _record(cls: str, prob: float, angle: float | None, gt: bool) -> dict:
    """A validation record claiming cls with a contradicted angle verdict."""
    raw = None if angle is None else -angle
    return {
        "probs": {NAME_TO_CODE[cls]: prob, NAME_TO_CODE["normal"]: 0.1},
        "claims": [cls] if prob >= 0.5 else [],
        "gt_labels": {cls: gt},
        "evidence": {cls: {"verdict": "contradicted", "raw_evidence": raw,
                           "evidence": ""}},
    }


def test_freeze_picks_largest_angle_below_first_true_claim():
    # false claims at 0/5/7 deg, true claim at 20.9 deg -> any ceiling
    # below 20.9 has FVR 0; the grid freeze lands on 20 deg.
    recs = ([_record("slope", 0.9, a, False) for a in (0.0, 5.0, 7.0)]
            + [_record("slope", 0.8, 20.9, True)])
    out = freeze_angle_thresholds(recs)
    assert out["slope"]["max_angle"] == 20
    assert out["slope"]["raw_floor"] == -20.0
    assert out["slope"]["n_pool"] == 4
    assert out["slope"]["n_vetoes_at_T"] == 3
    assert out["slope"]["fvr_at_T"] == 0.0


def test_freeze_skips_strategy_with_no_feasible_ceiling():
    # a true claim at 0 deg: even the tightest ceiling kills a true claim
    recs = [_record("tilt", 0.9, 0.0, True),
            _record("tilt", 0.9, 30.0, False)]
    out = freeze_angle_thresholds(recs)
    assert "tilt" not in out


def test_freeze_ignores_unclaimed_and_non_contradicted():
    # frames below the claim threshold or without contradicted evidence
    # must not enter the pool
    recs = ([_record("slope", 0.3, 1.0, False),      # not claimed
             _record("slope", 0.9, None, False),     # no raw
             _record("slope", 0.9, 2.0, False)])     # only this one counts
    out = freeze_angle_thresholds(recs)
    assert out["slope"]["n_pool"] == 1
    assert out["slope"]["max_angle"] == 45  # pool all-false -> grid max


def test_freeze_empty_pool_yields_nothing():
    # lookup is claimed but not an interpretable-angle strategy -> no floor
    assert freeze_angle_thresholds([_record("lookup", 0.9, 3.0, False)
                                    ]) == {}
