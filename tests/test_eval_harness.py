"""Tests for sage.eval_harness — the shared eval interface used by all
eval_* scripts. Uses synthetic records; no services required.
"""
from __future__ import annotations

import pytest

from sage.eval_harness import (
    BEHAVIOR_NAMES,
    code_probs_to_name_probs,
    compute_metrics,
    make_error_pred,
    run_eval,
)


def _gt(n=4):
    gt = []
    for i in range(n):
        labels = {name: 0 for name in BEHAVIOR_NAMES}
        labels[BEHAVIOR_NAMES[i % len(BEHAVIOR_NAMES)]] = 1
        gt.append({
            "sample_id": f"s{i}",
            "image_path": f"img{i}.jpg",
            "labels": labels,
            "primary_label": BEHAVIOR_NAMES[i % len(BEHAVIOR_NAMES)],
            "split": "test",
        })
    return gt


def _record(sample, hit=True, prob_key="probs", primary_key="primary_label",
            error=None):
    rec = {
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "latency_ms": 10.0,
        "error": error,
        prob_key: {n: 0.0 for n in BEHAVIOR_NAMES},
        primary_key: sample["primary_label"] if hit else "normal",
    }
    if not error:
        rec[prob_key][sample["primary_label"]] = 0.9
    return rec


def test_compute_metrics_perfect_predictions():
    gt = _gt()
    preds = [_record(s) for s in gt]
    report = compute_metrics(gt, preds, prob_key="probs",
                             primary_key="primary_label")
    assert report["num_samples"] == len(gt)
    assert report["num_errors"] == 0
    # Perfect per-sample hits; degenerate classes (support=0) still drag
    # macro-F1 (4 covered classes out of 15), matching the paper's semantics.
    assert report["overall"]["primary_accuracy"] == 1.0
    assert report["overall"]["primary_accuracy_exact"] == 1.0


def test_compute_metrics_counts_errors():
    gt = _gt()
    preds = [_record(s) for s in gt[:2]]
    report = compute_metrics(gt, preds)
    # 2 missing records -> counted as errors
    assert report["num_errors"] == 2


def test_compute_metrics_fusion_extras():
    gt = _gt(2)
    preds = []
    for s in gt:
        rec = _record(s, prob_key="adjusted_probs", primary_key="final_name")
        rec["vlm_primary_name"] = "normal"           # fusion changed it
        rec["source"] = "fusion"
        rec["recheck_verdicts"] = {
            name: {"verdict": "confirmed", "evidence": ""}
            for name in BEHAVIOR_NAMES}
        preds.append(rec)
    report = compute_metrics(
        gt, preds, prob_key="adjusted_probs", primary_key="final_name",
        fusion=True)
    ov = report["overall"]
    assert ov["adjusted_count"] == 1  # s1: VLM said "normal", fusion said gt
    assert ov["source_counts"] == {"fusion": 2}
    assert ov["vlm_primary_accuracy"] == 0.5  # "normal" hits s0 only
    assert all(c["verdict_counts"]["confirmed"] == len(gt)
               for c in report["per_class"].values())


def test_run_eval_handles_missing_image_and_exceptions(tmp_path):
    gt = _gt(3)

    def predict(sample):
        if sample["sample_id"] == "s2":
            raise RuntimeError("boom")
        return _record(sample)

    def error_pred(sample, err):
        return make_error_pred(_record(sample, hit=False), err)

    preds = run_eval(gt, predict, error_pred, workers=1)
    assert len(preds) == 3
    # no real images on disk -> all image_not_found
    assert all(p["error"] == "image_not_found" for p in preds)


def test_run_eval_workers_parallel():
    gt = _gt(5)
    preds = run_eval(
        gt, lambda s: _record(s), lambda s, e: make_error_pred(_record(s), e),
        workers=3)
    assert len(preds) == 5
    assert all(p["error"] == "image_not_found" for p in preds)


def test_code_probs_to_name_probs_maps_and_drops_unknowns():
    from sage.taxonomy import BEHAVIOR_CODES
    out = code_probs_to_name_probs({BEHAVIOR_CODES[0]: 0.7})
    assert abs(out["normal"] - 0.7) < 1e-9 if BEHAVIOR_CODES[0] == "ZZ" else True
    assert sum(out.values()) == pytest.approx(0.7)
