"""Shared metrics for all SAGE evaluation and analysis scripts.

Single source of truth for: per-class P/R/F1, macro-F1, AP/mAP, ECE,
confusion matrices, latency stats, threshold curves, McNemar's exact test,
and paired bootstrap tests.  Previously these were copy-pasted across
scripts/eval_sage.py, scripts/eval_vlm_check.py, scripts/eval_yolo_cls.py,
scripts/eval_cnn_mlb.py and the scripts/compute_*.py utilities.

All functions are numpy-only except where scipy is needed (McNemar exact
test for larger discordant counts).  Numeric semantics match the original
inline implementations (see tests/test_metrics.py for pinned anchor values).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Taxonomy import is re-exported for convenience so callers can do
# `from sage.metrics import BEHAVIOR_NAMES` with one import.
_REPO = Path(__file__).resolve().parent.parent
from sage.taxonomy import BEHAVIOR_NAMES  # noqa: F401  (re-export)

NUM_CLASSES = len(BEHAVIOR_NAMES)  # noqa: E402


# -- Per-class metrics --------------------------------------------------------

def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Binary precision / recall / F1 for a single class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the 15 classes (equal class weighting).

    y_true, y_pred: [N, C] binary arrays.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[1] == 0:
        return 0.0
    f1s = [per_class_prf(y_true[:, c], y_pred[:, c])[2]
           for c in range(y_true.shape[1])]
    return float(np.mean(f1s))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision via PR curve (numpy only)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1.0 - yt)
    precision = tp / (tp + fp)
    recall = tp / y_true.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def compute_map(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean Average Precision across classes. Classes with zero support
    contribute AP=0 (matches historical eval_sage.py behaviour)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    aps = [average_precision(y_true[:, c], probs[:, c])
           for c in range(y_true.shape[1])]
    return float(np.mean(aps))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error across all classes and samples."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    ece = 0.0
    n_total = 0
    for c in range(probs.shape[1]):
        p = probs[:, c]
        y = labels[:, c]
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        for b in range(n_bins):
            lo, hi = b / n_bins, (b + 1) / n_bins
            mask = (p > lo) & (p <= hi)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            bin_acc = float(y[mask].mean())
            bin_conf = float(p[mask].mean())
            ece += cnt * abs(bin_acc - bin_conf)
            n_total += cnt
    return ece / n_total if n_total > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """NxN confusion matrix. Rows = true, columns = predicted."""
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def latency_stats(latencies: list[float]) -> dict:
    """Latency summary in ms. Single definition used by all eval scripts."""
    arr = np.array(latencies, dtype=np.float64)
    if len(arr) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "std_ms": 0.0}
    return {
        "mean_ms": round(float(arr.mean()), 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "std_ms": round(float(arr.std()), 1),
    }


def threshold_curve(y_true: np.ndarray, probs: np.ndarray,
                    thresholds: np.ndarray | list[float] | None = None) -> list[dict]:
    """Macro-F1 / subset accuracy at each binary threshold.

    Replaces the ad-hoc threshold-sensitivity computations whose prose claims
    previously contradicted the underlying numbers.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.71, 0.05), 2)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(np.float32)
        rows.append({
            "threshold": float(t),
            "macro_f1": round(macro_f1(y_true, preds), 4),
            "subset_acc": round(float(np.mean(np.all(preds == y_true, axis=1))), 4),
        })
    return rows


# -- Statistical tests --------------------------------------------------------

def mcnemar_exact(n01: int, n10: int) -> dict:
    """McNemar's test on paired binary outcomes.

    n01 = count of sample correct-under-B but wrong-under-A convention is
    caller-defined; only the two discordant counts matter (symmetric test).

    Exact two-sided binomial test when n_disc < 25, chi-square continuity
    correction otherwise.  Returns both p-values plus the counts for
    auditability.
    """
    from scipy import stats
    n01, n10 = int(n01), int(n10)
    n_disc = n01 + n10
    out = {"n01": n01, "n10": n10, "n_disc": n_disc}
    if n_disc == 0:
        out.update({"p_exact": 1.0, "p_chi2": 1.0, "method": "empty"})
        return out
    if n_disc < 25:
        p_exact = float(stats.binomtest(min(n01, n10), n_disc, 0.5).pvalue)
        out.update({"p_exact": p_exact, "method": "exact_binomial"})
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n_disc
        p_chi2 = float(stats.chi2.sf(chi2, df=1))
        out.update({"p_chi2": p_chi2, "method": "chi2_continuity"})
    # always report both
    if "p_exact" not in out:
        b = min(n01, n10)
        out["p_exact"] = float(stats.binomtest(b, n_disc, 0.5).pvalue)
    if "p_chi2" not in out:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n_disc
        out["p_chi2"] = float(stats.chi2.sf(chi2, df=1))
    return out


def primary_hit(gt_labels: np.ndarray, pred_primary_idx: np.ndarray) -> np.ndarray:
    """Per-sample Primary Acc (hit): argmax prediction matches ANY gt label.

    gt_labels: [N, C] binary; pred_primary_idx: [N] int class indices.
    Returns [N] bool.
    """
    gt_labels = np.asarray(gt_labels)
    pred_primary_idx = np.asarray(pred_primary_idx)
    return gt_labels[np.arange(len(pred_primary_idx)), pred_primary_idx] > 0.5


def paired_bootstrap(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray,
                     n_boot: int = 10000, seed: int = 42) -> dict:
    """Paired bootstrap test on macro-F1 difference (A - B).

    preds_a / preds_b: [N, C] binary prediction matrices, row-aligned.
    Resamples sample indices with replacement; reports the two-sided
    empirical p-value for delta != 0 plus the delta distribution summary.
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = y_true.shape[0]
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
                "n_boot": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (macro_f1(y_true[idx], preds_a[idx])
                     - macro_f1(y_true[idx], preds_b[idx]))
    obs = macro_f1(y_true, preds_a) - macro_f1(y_true, preds_b)
    # two-sided empirical p via the percentile method on the sign of the
    # resampled deltas: p = 2 * min(P(delta<=0), P(delta>=0))
    p_le = float(np.mean(deltas <= 0))
    p_ge = float(np.mean(deltas >= 0))
    p = min(1.0, 2.0 * min(p_le, p_ge))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": round(float(obs), 6),
        "ci_low": round(float(lo), 6),
        "ci_high": round(float(hi), 6),
        "p_value": p,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# -- predictions.json I/O ------------------------------------------------------

def load_predictions(path: str | Path, prob_key: str | None = None
                     ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load a predictions.json into aligned (sample_ids, gt, probs) arrays.

    Handles both prediction schemas:
      - eval_vlm_check.py:  per-sample dict key ``"probs"``
      - eval_sage.py:       per-sample dict key ``"adjusted_probs"``
    Pass ``prob_key`` to override auto-detection.

    Returns (ids, gt_matrix [N,C] float, probs_matrix [N,C] float) with
    columns ordered per BEHAVIOR_NAMES.
    """
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    if prob_key is None:
        if preds and "adjusted_probs" in preds[0]:
            prob_key = "adjusted_probs"
        else:
            prob_key = "probs"
    ids, gt_rows, prob_rows = [], [], []
    for p in preds:
        ids.append(p["sample_id"])
        gt_rows.append([float(p["gt_labels"].get(n, 0)) for n in BEHAVIOR_NAMES])
        prob_rows.append([float(p[prob_key].get(n, 0.0)) for n in BEHAVIOR_NAMES])
    return ids, np.array(gt_rows, dtype=np.float32), np.array(prob_rows, dtype=np.float32)


def align_predictions(ids_a: list[str], gt_a: np.ndarray, probs_a: np.ndarray,
                      ids_b: list[str], probs_b: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-align two prediction matrices by sample_id (inner join, sorted).

    Required before paired tests: previously compute_macro_f1_paired_test.py
    only asserted alignment; this performs an explicit join instead.
    """
    pos_b = {sid: i for i, sid in enumerate(ids_b)}
    keep_a, keep_b = [], []
    for i, sid in enumerate(ids_a):
        j = pos_b.get(sid)
        if j is not None:
            keep_a.append(i)
            keep_b.append(j)
    return gt_a[keep_a], probs_a[keep_a], probs_b[keep_b]
