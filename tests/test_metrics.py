"""Anchor regression tests for sage.metrics — pins the paper's numbers.

Anchors (2026-09-21 revision):
  - all-positive degenerate macro-F1 == 0.12902 (paper: 0.129; the D0/D1-CE
    failure mode where every label is predicted positive for every sample)
  - McNemar n01=4/n10=0  -> p = 0.125   (D1 vs D2, paper §5.2)
  - McNemar n01=18/n10=12 -> p = 0.3616 (D1-CE vs D1 primary-label counts)
  - McNemar n01=95/n10=7  -> p < 0.001  (D0 vs D1)
  - bootstrap p-value resolution floor: 1/n_boot
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
from sage.metrics import (
    align_predictions,
    compute_ece,
    load_predictions,
    macro_f1,
    mcnemar_exact,
    paired_bootstrap,
    per_class_prf,
    threshold_curve,
)

# test-split per-class supports (N=287)
TEST_SUPPORTS = [104, 15, 7, 14, 12, 13, 13, 22, 14, 15, 13, 17, 16, 14, 28]
N_TEST = 287


def _all_positive_gt_preds():
    gt = np.zeros((N_TEST, 15), dtype=np.float32)
    for c, s in enumerate(TEST_SUPPORTS):
        gt[:s, c] = 1.0
    # keep row count consistent: labels spread over first rows only, that is
    # fine for the degenerate-case computation (preds all positive).
    preds = np.ones_like(gt)
    return gt, preds


def test_degenerate_all_positive_macro_f1_anchor():
    """Every sample predicted positive for all 15 classes.

    Per class: p = s/N, r = 1, f1 = 2s/(N+s); macro mean must equal the
    paper's 0.129 (computed 0.12902).
    """
    gt, preds = _all_positive_gt_preds()
    val = macro_f1(gt, preds)
    expected = float(np.mean([2 * s / (N_TEST + s) for s in TEST_SUPPORTS]))
    assert val == pytest.approx(expected, abs=1e-6)
    assert round(val, 3) == 0.129 or abs(val - 0.12902) < 5e-4


def test_degenerate_anchor_against_real_predictions():
    """Cross-check the synthetic anchor against the real D1-CE fixture."""
    fixture = REPO / "tests/fixtures/predictions_d1_ce_bce.json"
    if not fixture.exists():
        pytest.skip("historical fixture not available")
    ids, gt, probs = load_predictions(fixture)
    preds = np.ones_like(gt)
    val = macro_f1(gt, preds)
    assert 0.1290 <= val <= 0.1291, f"anchor drift: {val}"


def test_mcnemar_anchor_0125():
    out = mcnemar_exact(4, 0)
    assert out["p_exact"] == pytest.approx(0.125)
    assert out["method"] == "exact_binomial"


def test_mcnemar_anchor_03616():
    out = mcnemar_exact(18, 12)
    assert out["p_exact"] == pytest.approx(0.3616, abs=5e-5)


def test_mcnemar_anchor_significant():
    out = mcnemar_exact(95, 7)
    p = out.get("p_chi2", out.get("p_exact"))
    assert p < 0.001
    assert out["method"] == "chi2_continuity"  # n_disc=102 >= 25


def test_per_class_prf_basics():
    yt = np.array([1, 1, 0, 0, 1])
    yp = np.array([1, 0, 1, 0, 1])
    p, r, f1 = per_class_prf(yt, yp)
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_per_class_prf_empty_edge():
    assert per_class_prf([0, 0], [1, 0]) == (0.0, 0.0, 0.0)
    assert per_class_prf([1, 1], [0, 0]) == (0.0, 0.0, 0.0)


def test_ece_perfect_and_worst():
    rng = np.random.default_rng(0)
    labels = (rng.random((50, 15)) > 0.9).astype(np.float32)
    ece = compute_ece(labels.astype(float), labels)  # probs == labels
    assert ece < 0.01
    ece_bad = compute_ece((1.0 - labels), labels)  # completely wrong confidence
    assert ece_bad > 0.5


def test_paired_bootstrap_identical():
    rng = np.random.default_rng(1)
    gt = (rng.random((60, 15)) > 0.9).astype(np.float32)
    preds = gt.copy()
    out = paired_bootstrap(gt, preds, preds, n_boot=200, seed=42)
    assert out["delta"] == 0.0
    assert out["p_value"] == 1.0


def test_bootstrap_resolution_floor():
    """p is a multiple of 1/(2*n_boot) at best; never below it for one-sided
    evidence.  10000 boots -> reports must not claim finer than 0.0001."""
    rng = np.random.default_rng(2)
    gt = (rng.random((80, 15)) > 0.9).astype(np.float32)
    better = ((gt > 0.5) | (rng.random(gt.shape) < 0.02)).astype(np.float32)
    out = paired_bootstrap(gt, better.astype(np.float32), gt, n_boot=10000, seed=42)
    assert out["p_value"] >= 1.0 / 10000 or out["p_value"] == 0.0
    # resolution: with 10k boots, nonzero p is a multiple of 1e-4 (up to fp)
    if out["p_value"] > 0:
        assert abs(out["p_value"] * 10000 - round(out["p_value"] * 10000)) < 1e-6


def test_threshold_curve_shapes():
    rng = np.random.default_rng(3)
    gt = (rng.random((40, 15)) > 0.9).astype(np.float32)
    probs = rng.random((40, 15))
    rows = threshold_curve(gt, probs)
    assert [r["threshold"] for r in rows][:3] == [0.30, 0.35, 0.40]
    assert all(0.0 <= r["macro_f1"] <= 1.0 for r in rows)


def test_load_predictions_roundtrip(tmp_path):
    preds = [{
        "sample_id": "s1",
        "gt_labels": {"normal": 1, "away": 0},
        "probs": {"normal": 0.9, "away": 0.1},
    }]
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(preds), encoding="utf-8")
    ids, gt, probs = load_predictions(p)
    assert ids == ["s1"]
    assert gt[0, 0] == pytest.approx(1.0) and gt[0, 1] == pytest.approx(0.0)
    assert probs[0, 0] == pytest.approx(0.9)


def test_align_predictions_inner_join():
    gt = np.eye(3, dtype=np.float32)
    ids_a = ["a", "b", "c"]
    probs_b = np.array([[0.1], [0.2], [0.3]])
    ids_b = ["c", "a"]  # 'b' missing in B
    g2, pa, pb = align_predictions(ids_a, gt, gt, ids_b, probs_b)
    assert g2.shape[0] == 2
    assert pb[0, 0] == pytest.approx(0.2)   # a (B row 1)
    assert pb[1, 0] == pytest.approx(0.1)   # c (B row 0), aligned to A order
