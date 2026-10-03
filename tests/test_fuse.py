"""Tests for sage.fuse.FusionEngine — the paper's core fusion contribution.

The engine is a pure function of (VLM probabilities, recheck verdicts,
weights); tests inject synthetic weights so no disk or service is needed.
"""
from __future__ import annotations

import json

import pytest

from sage.fuse import FusionEngine, load_lr_weights
from sage.taxonomy import BEHAVIOR_CODES, CODE_TO_NAME, NAME_TO_CODE
from sage_check.result import VLMResult
from sage_recheck.enums import EvidenceResult, EvidenceVerdict

# Synthetic LR weights: confirmed boosts, rejected suppresses (log-odds shift).
WEIGHTS = {
    name: {"confirmed": 0.0, "rejected": 0.0, "weak": 0.0, "unavailable": 0.0}
    for name in CODE_TO_NAME.values()
}
WEIGHTS["phone"]["confirmed"] = 0.5
WEIGHTS["phone"]["rejected"] = -2.0
WEIGHTS["toy"]["confirmed"] = 2.0


def _vlm(probs_by_name: dict[str, float], primary: str) -> VLMResult:
    code_probs = {NAME_TO_CODE[n]: p for n, p in probs_by_name.items()}
    code_probs.setdefault(NAME_TO_CODE["normal"], 0.0)
    return VLMResult(primary_code=NAME_TO_CODE[primary],
                     primary_name=primary, probabilities=code_probs)


def _vr(name: str, verdict: EvidenceVerdict,
        evidence: str = "") -> tuple[str, EvidenceResult]:
    return name, EvidenceResult(verdict=verdict, evidence=evidence)


def _no_evidence() -> dict[str, EvidenceResult]:
    return {n: EvidenceResult(verdict=EvidenceVerdict.INSUFFICIENT)
            for n in CODE_TO_NAME.values()}


def test_no_evidence_preserves_vlm_distribution():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = _vlm({"normal": 0.6, "phone": 0.3}, "normal")
    out = engine.fuse(vlm, _no_evidence())
    for code, p in vlm.probabilities.items():
        assert out.adjusted_probabilities[code] == pytest.approx(p, abs=1e-6)
    assert out.source == "vlm"
    assert out.final_name == "normal"


def test_confirmed_boosts_probability():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = _vlm({"normal": 0.6, "phone": 0.3}, "normal")
    recheck = dict(_no_evidence())
    name, r = _vr("phone", EvidenceVerdict.CONSISTENT)
    recheck[name] = r
    out = engine.fuse(vlm, recheck)
    boosted = out.adjusted_probabilities[NAME_TO_CODE["phone"]]
    assert boosted > vlm.probabilities[NAME_TO_CODE["phone"]]
    assert out.source == "vlm"  # boost alone did not flip the argmax


def test_confirmed_can_flip_argmax():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = _vlm({"normal": 0.45, "toy": 0.40}, "normal")
    recheck = dict(_no_evidence())
    name, r = _vr("toy", EvidenceVerdict.CONSISTENT)
    recheck[name] = r
    out = engine.fuse(vlm, recheck)
    assert out.final_name == "toy"
    assert out.source == "adjusted"
    assert "toy:consistent" in out.evidence


def test_rejected_suppresses_probability():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = _vlm({"normal": 0.5, "phone": 0.45}, "normal")
    recheck = dict(_no_evidence())
    name, r = _vr("phone", EvidenceVerdict.CONTRADICTED)
    recheck[name] = r
    out = engine.fuse(vlm, recheck)
    assert out.adjusted_probabilities[NAME_TO_CODE["phone"]] < 0.45
    assert out.final_name == "normal"


def test_weak_verdict_is_no_evidence():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = _vlm({"normal": 0.6, "phone": 0.3}, "normal")
    recheck = dict(_no_evidence())
    name, r = _vr("phone", EvidenceVerdict.INSUFFICIENT)
    recheck[name] = r
    out = engine.fuse(vlm, recheck)
    assert out.adjusted_probabilities[NAME_TO_CODE["phone"]] == pytest.approx(
        vlm.probabilities[NAME_TO_CODE["phone"]], abs=1e-6)


def test_vlm_error_degrades_to_normal_even_with_consistent_evidence():
    # Claim-conditioned contract: without VLM claims the small model cannot
    # supply an alternative judgment — the system degrades to normal.
    engine = FusionEngine(weights=WEIGHTS)
    vlm = VLMResult(error="service down")
    recheck = dict(_no_evidence())
    name, r = _vr("toy", EvidenceVerdict.CONSISTENT)
    recheck[name] = r
    out = engine.fuse(vlm, recheck)
    assert out.source == "vlm_error"
    assert out.final_name == "normal"


def test_vlm_error_without_alternative_degrades_to_normal():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = VLMResult(error="service down")
    out = engine.fuse(vlm, _no_evidence())
    assert out.source == "vlm_error"
    assert out.final_name == "normal"
    assert out.confidence == 0.0


def test_weights_loaded_from_json_file(tmp_path):
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({"weights": WEIGHTS}), encoding="utf-8")
    engine = FusionEngine(weights_path=weights_file)
    assert engine.weights == WEIGHTS
    assert load_lr_weights(weights_file) == WEIGHTS


def test_fusion_output_is_wellformed():
    engine = FusionEngine(weights=WEIGHTS)
    vlm = _vlm({"normal": 0.6, "phone": 0.3}, "normal")
    out = engine.fuse(vlm, _no_evidence())
    assert set(out.adjusted_probabilities) == set(BEHAVIOR_CODES)
    assert all(0.0 <= p <= 1.0 for p in out.adjusted_probabilities.values())
    assert out.final_name == CODE_TO_NAME[out.final_code]
