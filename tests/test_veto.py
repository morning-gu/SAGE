"""Tests for sage.veto — Algorithm 1, ECDF normalization, and veto metrics.

Pure-function tests on synthetic inputs (no services, no GPU). Mirrors the
frozen protocol constants in docs/veto-algorithm.md.
"""
from __future__ import annotations

import pytest

from sage.veto import (
    DELTA,
    N_MIN,
    ECDFStore,
    aggregate_metrics,
    apply_veto,
    metrics_by_group,
)
from sage_recheck.enums import EvidenceResult, EvidenceVerdict


def _ev(name: str, verdict: EvidenceVerdict,
        raw: float | None = None) -> tuple[str, EvidenceResult]:
    return name, EvidenceResult(verdict=verdict, raw_evidence=raw)


def _probs(**by_name: float) -> dict[str, str]:
    """Name-keyed probabilities -> code-keyed, with normal defaulting to 0."""
    from sage.taxonomy import NAME_TO_CODE
    probs = {NAME_TO_CODE[n]: p for n, p in by_name.items()}
    probs.setdefault(NAME_TO_CODE["normal"], 0.0)
    return probs


def _primary(name: str) -> str:
    from sage.taxonomy import NAME_TO_CODE
    return NAME_TO_CODE[name]


# ---------------------------------------------------------------------------
# ECDFStore
# ---------------------------------------------------------------------------

def test_ecdf_fit_disables_uncalibrated_strategies():
    records = ([{"name": "slope", "raw_evidence": float(i)} for i in range(N_MIN)]
               + [{"name": "tilt", "raw_evidence": 1.0}])  # < N_MIN
    store = ECDFStore.fit(records)
    assert "slope" not in store.disabled
    assert "tilt" in store.disabled
    assert store.apply("tilt", 1.0) is None  # disabled => no veto authority
    assert "n_min" in store.disabled_reasons["tilt"]


def test_ecdf_fit_disables_degenerate_point_mass_pool():
    # The away failure mode (v3 CPU run): one value held 283/284 of the val
    # pool, so its quantile was ~1.0 at every tau and the gate could not
    # discriminate.  The v4 guard disables such pools outright.
    records = ([{"name": "slope", "raw_evidence": float(i)} for i in range(N_MIN)]
               + [{"name": "away", "raw_evidence": 1.0}] * (N_MIN + 10)
               + [{"name": "away", "raw_evidence": 0.5}])
    store = ECDFStore.fit(records)
    assert "slope" not in store.disabled
    assert "away" in store.disabled
    assert "degenerate" in store.disabled_reasons["away"]
    # the point-mass maximum carries no veto authority even though it is
    # the pool maximum
    assert store.apply("away", 1.0) is None


def test_ecdf_keeps_pool_with_small_modal_share():
    # A healthy pool: the modal value is far below the DEGENERATE_MASS share
    # (level shoulders were 7/257 of the val slope pool) -> stays calibrated.
    records = ([{"name": "tilt", "raw_evidence": 0.0}] * 5
               + [{"name": "tilt", "raw_evidence": -float(i)}
                  for i in range(1, N_MIN)])
    store = ECDFStore.fit(records)
    assert "tilt" not in store.disabled
    assert store.apply("tilt", 0.0) is not None


def test_ecdf_quantile_semantics():
    store = ECDFStore.fit([{"name": "slope", "raw_evidence": float(i)}
                           for i in range(100)])
    # Below min -> 0, above max -> 1, mid-order-statistics -> their rank.
    assert store.apply("slope", -1.0) == 0.0
    assert store.apply("slope", 1000.0) == 1.0
    assert store.apply("slope", 49.0) == pytest.approx(50 / 100)
    # 49.5 lies between order statistics 49 (q=0.50) and 50 (q=0.51).
    assert 0.50 < store.apply("slope", 49.5) < 0.51


def test_ecdf_roundtrip_via_json(tmp_path):
    store = ECDFStore.fit(
        [{"name": "slope", "raw_evidence": float(i)} for i in range(N_MIN)]
        + [{"name": "away", "raw_evidence": 1.0}] * (N_MIN + 10))  # degenerate
    path = tmp_path / "ecdf.json"
    store.save(path)
    loaded = ECDFStore.load(path)
    assert loaded.apply("slope", 49.0) == pytest.approx(store.apply("slope", 49.0))
    assert "away" in loaded.disabled
    assert loaded.disabled_reasons["away"] == store.disabled_reasons["away"]


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def _all_insufficient() -> dict[str, EvidenceResult]:
    return {}


def test_non_primary_contradicted_above_tau_vetoes():
    # v3 uniform gating: non-primary vetoes also require ~e >= tau.
    ecdf = ECDFStore.fit([{"name": "phone", "raw_evidence": i / 1000}
                          for i in range(40)])
    out = apply_veto(_probs(normal=0.6, phone=0.7), _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.CONTRADICTED, 0.1)]),
                     tau=0.5, ecdf=ecdf)
    assert [v.name for v in out.vetoed] == ["phone"]
    assert out.final_name == "normal"  # primary untouched
    assert not out.primary_demoted
    assert out.corrected_codes == []


def test_non_primary_contradicted_below_tau_is_kept():
    ecdf = ECDFStore({"phone": [float(i) for i in range(100)]}, set())
    # raw=10 -> tilde ~ 0.11 < tau=0.5 -> no veto under uniform gating
    out = apply_veto(_probs(normal=0.6, phone=0.7), _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.CONTRADICTED, 10.0)]),
                     tau=0.5, ecdf=ecdf)
    assert out.vetoed == []
    assert out.claims == ["phone"]  # still counted as a checked claim


def test_no_ecdf_means_no_veto_authority():
    # Without calibration nothing vetoes (v3: no calibration, no veto).
    out = apply_veto(_probs(normal=0.6, phone=0.7), _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.CONTRADICTED, 0.1)]),
                     tau=0.5, ecdf=None)
    assert out.vetoed == []


def test_uncalibrated_raw_never_vetoes():
    # Contradicted with raw=None (e.g. recline depth fallback): no veto.
    ecdf = ECDFStore.fit([{"name": "phone", "raw_evidence": i / 1000}
                          for i in range(40)])
    out = apply_veto(_probs(normal=0.6, phone=0.7), _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.CONTRADICTED, None)]),
                     tau=0.5, ecdf=ecdf)
    assert out.vetoed == []


def test_claim_without_strategy_excluded_from_claims():
    # v3 registry trim: claims of classes with no evidence entry (phone,
    # toy, snack, ...) are outside the veto scope and the coverage base.
    out = apply_veto(_probs(normal=0.6, phone=0.7, slope=0.8),
                     _primary("normal"),
                     dict([_ev("slope", EvidenceVerdict.INSUFFICIENT)]),
                     tau=0.5)
    assert out.claims == ["slope"]
    assert out.claim_verdicts == {"slope": "insufficient"}


def test_primary_contradicted_below_tau_is_kept():
    ecdf = ECDFStore({"slope": [float(i) for i in range(100)]}, set())
    # raw=10 -> tilde ~ 0.11 < tau=0.5
    out = apply_veto(_probs(normal=0.2, slope=0.9), _primary("slope"),
                     dict([_ev("slope", EvidenceVerdict.CONTRADICTED, 10.0)]),
                     tau=0.5, ecdf=ecdf)
    assert out.vetoed == []
    assert out.final_name == "slope"
    assert not out.primary_demoted


def test_primary_contradicted_above_tau_is_demoted():
    ecdf = ECDFStore({"slope": [float(i) for i in range(100)]}, set())
    # raw=90 -> tilde ~ 0.91 >= tau=0.5
    out = apply_veto(_probs(normal=0.2, slope=0.9, tilt=0.6),
                     _primary("slope"),
                     dict([_ev("slope", EvidenceVerdict.CONTRADICTED, 90.0)]),
                     tau=0.5, ecdf=ecdf)
    assert out.primary_demoted
    assert out.final_name == "tilt"  # argmax over remaining claims
    assert out.demotion_outcome is None  # no gt supplied


def test_primary_demoted_to_normal_when_no_claims_remain():
    ecdf = ECDFStore({"slope": [float(i) for i in range(100)]}, set())
    out = apply_veto(_probs(normal=0.2, slope=0.9), _primary("slope"),
                     dict([_ev("slope", EvidenceVerdict.CONTRADICTED, 90.0)]),
                     tau=0.5, ecdf=ecdf)
    assert out.primary_demoted
    assert out.final_name == "normal"  # default-state fallback (Stage-1 rule)


def test_normal_is_outside_veto_scope():
    # Even a contradicted verdict on normal cannot appear: strategies for
    # normal do not exist and claims exclude it by construction.
    out = apply_veto(_probs(normal=0.9), _primary("normal"), {}, tau=0.5)
    assert out.claims == []
    assert out.vetoed == []


def test_insufficient_and_consistent_never_veto():
    out = apply_veto(_probs(normal=0.5, phone=0.8, toy=0.8),
                     _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.INSUFFICIENT),
                           _ev("toy", EvidenceVerdict.CONSISTENT, 0.5)]),
                     tau=0.5)
    assert out.vetoed == []
    assert out.decisive == 1  # only the consistent claim counted decisive


def test_gt_annotation_marks_record_correctness():
    ecdf = ECDFStore.fit([{"name": "phone", "raw_evidence": i / 1000}
                          for i in range(40)])
    out = apply_veto(_probs(normal=0.6, phone=0.7),
                     _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.CONTRADICTED, 0.2)]),
                     tau=0.5, ecdf=ecdf,
                     gt_labels={"phone": False})  # phone truly absent
    assert out.vetoed[0].correct is True


# ---------------------------------------------------------------------------
# v5 frozen-threshold authority
# ---------------------------------------------------------------------------

def test_frozen_floor_replaces_quantile_gate():
    # slope-style case: raw = -7.0 (7 deg, contradicted).  The all-frames
    # ECDF maps it to a low quantile (blocked under pure tau gating), but
    # the frozen floor -12.0 (max_angle 12 deg) grants veto authority.
    ecdf = ECDFStore.fit([{"name": "slope", "raw_evidence": -float(i)}
                          for i in range(40)])  # tilde(-7.0) < 0.5
    ev = dict([_ev("slope", EvidenceVerdict.CONTRADICTED, -7.0)])
    blocked = apply_veto(_probs(normal=0.6, slope=0.7), _primary("normal"),
                         ev, tau=0.95, ecdf=ecdf)
    assert blocked.vetoed == []  # quantile gate alone blocks

    out = apply_veto(_probs(normal=0.6, slope=0.7), _primary("normal"),
                     ev, tau=0.95, ecdf=ecdf, raw_floors={"slope": -12.0})
    assert len(out.vetoed) == 1
    assert out.vetoed[0].gate == "frozen_threshold"
    assert out.vetoed[0].correct is None  # no gt given


def test_frozen_floor_requires_numeric_raw_above_floor():
    ev_none = dict([_ev("slope", EvidenceVerdict.CONTRADICTED, None)])
    out = apply_veto(_probs(normal=0.6, slope=0.7), _primary("normal"),
                     ev_none, tau=0.95, ecdf=None, raw_floors={"slope": -12.0})
    assert out.vetoed == []  # no raw -> no authority even with a floor

    ev_below = dict([_ev("slope", EvidenceVerdict.CONTRADICTED, -20.0)])
    out = apply_veto(_probs(normal=0.6, slope=0.7), _primary("normal"),
                     ev_below, tau=0.95, ecdf=None, raw_floors={"slope": -12.0})
    assert out.vetoed == []  # angle 20 deg > 12 deg ceiling


def test_no_floor_keeps_quantile_gate():
    # strategies absent from raw_floors are untouched by the v5 path
    ecdf = ECDFStore.fit([{"name": "phone", "raw_evidence": i / 1000}
                          for i in range(40)])
    out = apply_veto(_probs(normal=0.6, phone=0.7), _primary("normal"),
                     dict([_ev("phone", EvidenceVerdict.CONTRADICTED, 0.02)]),
                     tau=0.95, ecdf=ecdf, raw_floors={"slope": -12.0})
    assert out.vetoed == []  # tilde(0.02)=0.525 < 0.95 -> quantile gate blocks


def test_frozen_floor_can_demote_primary():
    out = apply_veto(_probs(normal=0.3, slope=0.7), _primary("slope"),
                     dict([_ev("slope", EvidenceVerdict.CONTRADICTED, -2.0)]),
                     tau=0.95, ecdf=None, raw_floors={"slope": -12.0},
                     gt_labels={"slope": False}, gt_primary="normal")
    assert out.primary_demoted is True
    assert out.final_name == "normal"
    assert out.demotion_outcome == "improved"
    assert out.vetoed[0].gate == "frozen_threshold"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _wide_ecdf() -> ECDFStore:
    """ECDF whose support lies far below 0.1, so raw=0.1 maps to tilde 1.0."""
    return ECDFStore.fit([{"name": n, "raw_evidence": i / 1000}
                          for n in ("phone", "toy", "snack", "slope")
                          for i in range(40)])


def _frame(probs_claims: dict[str, float], primary: str,
           evidence: list[tuple[str, EvidenceVerdict, float | None]],
           gt: dict[str, bool], gt_primary: str):
    return apply_veto(_probs(**probs_claims), _primary(primary),
                      dict(_ev(n, v, r) for n, v, r in evidence),
                      tau=0.5, ecdf=_wide_ecdf(),
                      gt_labels=gt, gt_primary=gt_primary)


def test_aggregate_metrics_vp_fvr_tpk_coverage():
    # Frame 1: phone claimed + contradicted; phone truly absent -> correct veto.
    o1 = _frame({"normal": 0.6, "phone": 0.8}, "normal",
                [("phone", EvidenceVerdict.CONTRADICTED, 0.1)],
                {"phone": False}, "normal")
    # Frame 2: toy claimed + contradicted; toy truly present -> false veto.
    o2 = _frame({"normal": 0.6, "toy": 0.8}, "normal",
                [("toy", EvidenceVerdict.CONTRADICTED, 0.1)],
                {"toy": True}, "normal")
    # Frame 3: snack claimed, verdict insufficient -> claim without evidence.
    o3 = _frame({"normal": 0.6, "snack": 0.8}, "normal",
                [("snack", EvidenceVerdict.INSUFFICIENT, None)],
                {"snack": False}, "normal")

    m = aggregate_metrics([o1, o2, o3],
                          [{"phone": False}, {"toy": True}, {"snack": False}])
    assert m["n_vetoes"] == 2
    assert m["n_vetoes_correct"] == 1
    assert m["veto_precision"] == pytest.approx(0.5)
    assert m["false_veto_rate"] == pytest.approx(0.5)
    # TPK: 1 false veto over 1 true anomaly positive.
    assert m["true_positive_kill_rate"] == pytest.approx(1.0)
    # Coverage: 2 decisive of 3 claims.
    assert m["coverage"] == pytest.approx(2 / 3)


def test_metrics_by_group_decomposition():
    # phone is semantic-cued; slope is geometric-cued.
    o1 = _frame({"normal": 0.6, "phone": 0.8, "slope": 0.8}, "normal",
                [("phone", EvidenceVerdict.CONTRADICTED, 0.1),
                 ("slope", EvidenceVerdict.CONTRADICTED, 0.1)],
                {"phone": False, "slope": False}, "normal")
    gts = [{"phone": False, "slope": False}]
    groups = metrics_by_group([o1], gts)
    assert groups["semantic"]["n_vetoes"] == 1
    assert groups["geometric"]["n_vetoes"] == 1
    assert groups["semantic"]["veto_precision"] == 1.0
    assert groups["geometric"]["veto_precision"] == 1.0


def test_frozen_constants_match_preregistration():
    assert DELTA == 0.05
    assert N_MIN == 30
