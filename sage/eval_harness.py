"""Shared evaluation harness.

Single implementation of everything the eval_* scripts used to copy-paste:
ground-truth loading, image encoding, batch execution with progress reporting,
metric computation (P/R/F1/AP/ECE/mAP/subset-acc/hamming/primary-acc/
confusion matrices), report printing, and artifact saving.

Scripts keep only what is genuinely theirs: argparse, the per-sample predict
adapter (how one image is turned into a prediction record), and any
script-specific extras (e.g. failure-image annotation).

Record schema (adapter contract):
  Required by compute_metrics:
    sample_id          str
    probs | adjusted_probs   name-keyed probability dict  (via prob_key)
    primary_label | final_name  str                     (via primary_key)
    latency_ms         float
    error              str | None
  Optional (fusion=True reads them, present when present in records):
    vlm_primary_name   str   — pre-fusion VLM prediction
    recheck_verdicts   dict  — name -> {"verdict": ..., "evidence": ...}
    source             str   — fusion source label
"""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sage.data import load_annotations
from sage.metrics import (
    BEHAVIOR_NAMES,
    NUM_CLASSES,
    average_precision,
    compute_ece,
    confusion_matrix,
    latency_stats,
    per_class_prf,
)
from sage.taxonomy import CODE_TO_NAME

# Re-exports so scripts can `from sage.eval_harness import ...` with one import.
NAME_TO_INDEX = {name: i for i, name in enumerate(BEHAVIOR_NAMES)}
VERDICTS = ("confirmed", "rejected", "weak", "unavailable")

# Ground-truth loading is sage.data.load_annotations (default-fill + split filter).
load_ground_truth = load_annotations


# -- Small shared helpers -----------------------------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def code_probs_to_name_probs(code_probs: dict) -> dict:
    """Convert code-keyed probs to behavior-name-keyed probs."""
    out = {name: 0.0 for name in BEHAVIOR_NAMES}
    for code, prob in code_probs.items():
        name = CODE_TO_NAME.get(code, code)
        if name in out:
            out[name] = float(prob)
    return out


def resolve_image_path(sample: dict, image_root: str | None = None) -> str:
    img_path = sample["image_path"]
    if image_root and not Path(img_path).is_absolute():
        return str(Path(image_root) / img_path)
    return img_path


def make_error_pred(record: dict, err: str) -> dict:
    """Return *record* shaped like a prediction but marked as an error.

    *record* is a template record (pre-filled defaults); the caller builds it
    so the harness stays agnostic of the record schema.
    """
    out = dict(record)
    out["error"] = err
    return out


# -- Batch execution ----------------------------------------------------------

def run_eval(
    gt: list[dict],
    predict,
    error_pred,
    *,
    image_root: str | None = None,
    workers: int = 1,
    limit: int = 0,
    summary_key: str = "final_name",
    summary_fn=None,
) -> list[dict]:
    """Run *predict* over ground-truth samples with progress reporting.

    predict(sample) -> prediction record   (all path/mode/client context is
                            captured by the caller's closure)
    error_pred(sample, err) -> record      (used for image-not-found and for
                            exceptions escaping predict)
    summary_fn(pred) -> str overrides the default progress line format.
    """
    samples = gt[:limit] if limit > 0 else gt
    n = len(samples)
    predictions: list[dict | None] = [None] * n

    def _summary(pred: dict) -> str:
        if summary_fn is not None:
            return summary_fn(pred)
        err = " ERR:" + pred["error"] if pred["error"] else ""
        src = " via " + pred["source"] if pred.get("source") else ""
        return "{} -> {}{} ({}ms){}".format(
            pred["sample_id"], pred.get(summary_key, "?"), src,
            pred.get("latency_ms", 0.0), err)

    if workers <= 1:
        for i, sample in enumerate(samples):
            pred = _predict_safe(sample, predict, error_pred, image_root)
            predictions[i] = pred
            if (i + 1) % 10 == 0 or i == n - 1:
                print("  [{}/{}] {}".format(i + 1, n, _summary(pred)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_predict_safe, s, predict, error_pred, image_root): i
                for i, s in enumerate(samples)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                predictions[idx] = future.result()
                done += 1
                if done % 10 == 0 or done == n:
                    print("  [{}/{}]".format(done, n))

    errors = sum(1 for p in predictions if p and p.get("error"))
    print("[eval] {} samples, {} errors".format(n, errors))
    return predictions  # type: ignore[return-value]


def _predict_safe(sample: dict, predict, error_pred, image_root: str | None) -> dict:
    # Resolve image_path up front so prediction records always carry the
    # resolved path, regardless of --image-root.
    sample = {**sample, "image_path": resolve_image_path(sample, image_root)}
    if not Path(sample["image_path"]).exists():
        return error_pred(sample, "image_not_found")
    try:
        return predict(sample)
    except Exception as e:  # noqa: BLE001  (one bad image must not kill the batch)
        return error_pred(sample, str(e))


# -- Metrics ------------------------------------------------------------------

def compute_metrics(
    gt: list[dict],
    predictions: list[dict],
    *,
    prob_key: str = "probs",
    primary_key: str = "primary_label",
    threshold: float = 0.5,
    ece_bins: int = 10,
    fusion: bool = False,
) -> dict:
    """Compute all metrics from ground truth and prediction records.

    prob_key / primary_key select the record fields holding probabilities and
    the dominant prediction (eval_vlm_check flavor: probs/primary_label;
    eval_sage fusion flavor: adjusted_probs/final_name).

    fusion=True additionally reports VLM-only accuracy (vlm_primary_name),
    recheck verdict distributions, fusion source counts, and the count of
    fusion-adjusted predictions.
    """
    pred_map = {p["sample_id"]: p for p in predictions}

    labels = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    probs = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    gt_primary = np.zeros(len(gt), dtype=np.int32)
    pred_primary = np.zeros(len(gt), dtype=np.int32)
    vlm_primary = np.zeros(len(gt), dtype=np.int32)
    latencies: list[float] = []
    n_errors = 0
    source_counts: dict[str, int] = {}

    for i, sample in enumerate(gt):
        sid = sample["sample_id"]
        labels[i] = [sample["labels"].get(n, 0) for n in BEHAVIOR_NAMES]
        gt_primary[i] = NAME_TO_INDEX.get(sample.get("primary_label", "normal"), 0)

        pred = pred_map.get(sid)
        if pred is None:
            n_errors += 1
            pred_primary[i] = 0
            vlm_primary[i] = 0
            continue
        if pred.get("error"):
            n_errors += 1
        probs[i] = [pred[prob_key].get(n, 0.0) for n in BEHAVIOR_NAMES]
        pred_primary[i] = NAME_TO_INDEX.get(pred.get(primary_key, "normal"), 0)
        vlm_primary[i] = NAME_TO_INDEX.get(pred.get("vlm_primary_name", "normal"), 0)
        if pred.get("latency_ms"):
            latencies.append(pred["latency_ms"])
        if fusion:
            src = pred.get("source", "error")
            source_counts[src] = source_counts.get(src, 0) + 1

    preds_binary = (probs >= threshold).astype(np.float32)

    # Per-class P/R/F1/AP + recheck verdict counts + FP/FN lists
    per_class: dict[str, dict] = {}
    f1s, aps = [], []
    for c, name in enumerate(BEHAVIOR_NAMES):
        yt = labels[:, c]
        yp = preds_binary[:, c]
        ys = probs[:, c]
        support = int(yt.sum())
        if support == 0:
            p = r = f1 = ap = 0.0
        else:
            p, r, f1 = per_class_prf(yt, yp)
            ap = average_precision(yt, ys)

        vcounts = {v: 0 for v in VERDICTS}
        fp_samples: list[str] = []
        fn_samples: list[str] = []
        for i, sample in enumerate(gt):
            sid = sample["sample_id"]
            if fusion:
                pred = pred_map.get(sid)
                if pred:
                    rv = pred.get("recheck_verdicts", {}).get(name, {}).get(
                        "verdict", "unavailable")
                    if rv not in vcounts:
                        rv = "unavailable"
                    vcounts[rv] += 1
            if yt[i] == 0 and yp[i] == 1:
                fp_samples.append(sid)
            elif yt[i] == 1 and yp[i] == 0:
                fn_samples.append(sid)

        per_class[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "ap": round(ap, 4),
            "support": support,
            "verdict_counts": vcounts,
            "fp": len(fp_samples),
            "fn": len(fn_samples),
            "fp_samples": fp_samples,
            "fn_samples": fn_samples,
        }
        f1s.append(f1)
        aps.append(ap)

    macro_f1 = float(np.mean(f1s))
    mAP = float(np.mean(aps))
    ece = compute_ece(probs, labels, ece_bins)
    subset_acc = float(np.mean(np.all(preds_binary == labels, axis=1)))
    hamming = float(np.mean(preds_binary != labels))
    # Primary accuracy (multi-label hit): prediction matches ANY GT label.
    primary_hit = labels[np.arange(len(gt)), pred_primary] > 0.5
    primary_acc = float(np.mean(primary_hit))
    primary_acc_exact = float(np.mean(pred_primary == gt_primary))
    cm = confusion_matrix(gt_primary, pred_primary, NUM_CLASSES)
    # Hit-aware CM: effective true = predicted label when it hit any valid
    # ground-truth label (diagonal), else gt_primary (off-diagonal miss).
    effective_true = np.where(primary_hit, pred_primary, gt_primary)
    cm_hit = confusion_matrix(effective_true, pred_primary, NUM_CLASSES)

    overall: dict = {
        "macro_f1": round(macro_f1, 4),
        "mAP": round(mAP, 4),
        "ECE": round(ece, 4),
        "subset_accuracy": round(subset_acc, 4),
        "hamming_loss": round(hamming, 4),
        "primary_accuracy": round(primary_acc, 4),
        "primary_accuracy_exact": round(primary_acc_exact, 4),
    }
    if fusion:
        # VLM-only primary accuracy (before fusion adjustment)
        vlm_hit = labels[np.arange(len(gt)), vlm_primary] > 0.5
        overall["vlm_primary_accuracy"] = round(float(np.mean(vlm_hit)), 4)
        overall["vlm_primary_accuracy_exact"] = round(
            float(np.mean(vlm_primary == gt_primary)), 4)
        overall["adjusted_count"] = int(np.sum(pred_primary != vlm_primary))
        overall["source_counts"] = source_counts

    return {
        "num_samples": len(gt),
        "num_errors": n_errors,
        "threshold": threshold,
        "ece_bins": ece_bins,
        "overall": overall,
        "per_class": per_class,
        "latency": latency_stats(latencies),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_hit": cm_hit.tolist(),
        "confusion_labels": BEHAVIOR_NAMES,
    }


# -- Report / artifacts -------------------------------------------------------

def print_report(report: dict, title: str, *, fusion: bool = False,
                 width: int = 80) -> None:
    ov = report["overall"]
    print()
    print("=" * width)
    print(title)
    print("Samples={}, Errors={}, Threshold={}".format(
        report["num_samples"], report["num_errors"], report["threshold"]))
    if fusion:
        sc = ov.get("source_counts", {})
        print("Sources: {}".format(", ".join(
            "{}={}".format(k, v) for k, v in sorted(sc.items()))))
        print("Fusion adjusted {} predictions (VLM->final changed)".format(
            ov["adjusted_count"]))
    print("=" * width)
    print()

    if fusion:
        hdr = ("{:<12s} {:>7s} {:>7s} {:>7s} {:>7s} {:>6s} {:>4s} {:>4s} "
               "{:>4s} {:>4s} {:>4s} {:>4s}").format(
            "Class", "P", "R", "F1", "AP", "Sup", "Cnf", "Rej", "Wk", "Unv",
            "FP", "FN")
        row = "{:<12s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:>6d} {:>4d} {:>4d} {:>4d} {:>4d} {:>4d} {:>4d}"
    else:
        hdr = "{:<12s} {:>7s} {:>7s} {:>7s} {:>7s} {:>6s} {:>4s} {:>4s}".format(
            "Class", "P", "R", "F1", "AP", "Sup", "FP", "FN")
        row = "{:<12s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:>6d} {:>4d} {:>4d}"
    print(hdr)
    print("-" * len(hdr))
    for name in BEHAVIOR_NAMES:
        c = report["per_class"][name]
        if fusion:
            vc = c["verdict_counts"]
            print(row.format(
                name, c["precision"], c["recall"], c["f1"], c["ap"],
                c["support"], vc.get("confirmed", 0), vc.get("rejected", 0),
                vc.get("weak", 0), vc.get("unavailable", 0), c["fp"], c["fn"]))
        else:
            print(row.format(
                name, c["precision"], c["recall"], c["f1"], c["ap"],
                c["support"], c["fp"], c["fn"]))
    print()
    label = "Macro-F1:           " if fusion else "Macro-F1:         "
    print("{}{:.4f}".format(label, ov["macro_f1"]))
    print("{}{:.4f}".format("mAP:                " if fusion else "mAP:              ", ov["mAP"]))
    print("{}{:.4f}".format("ECE:                " if fusion else "ECE:              ", ov["ECE"]))
    print("{}{:.4f}".format("Subset-Acc:         " if fusion else "Subset-Acc:       ", ov["subset_accuracy"]))
    print("{}{:.4f}".format("Hamming-Loss:       " if fusion else "Hamming-Loss:     ", ov["hamming_loss"]))
    print("{}{:.4f}  (hit: any valid label)".format(
        "Primary-Acc:        " if fusion else "Primary-Acc:      ",
        ov["primary_accuracy"]))
    print("{}{:.4f}  (strict dominant match)".format(
        "Primary-Exact:      " if fusion else "Primary-Exact:    ",
        ov["primary_accuracy_exact"]))
    if fusion:
        print("Primary-Acc:        {:.4f}  (VLM-only, before fusion)".format(
            ov["vlm_primary_accuracy"]))
        print("Primary-Exact:      {:.4f}  (VLM-only, before fusion)".format(
            ov["vlm_primary_accuracy_exact"]))
    lat = report.get("latency", {})
    if lat.get("mean_ms", 0) > 0:
        print()
        print("Latency: mean={:.0f}ms, p50={:.0f}ms, p95={:.0f}ms".format(
            lat["mean_ms"], lat["p50_ms"], lat["p95_ms"]))

    # FP / FN lists
    print()
    print("=" * width)
    print("False Positives (pred=1 but GT=0) -- high risk:")
    any_fp = False
    for name in BEHAVIOR_NAMES:
        c = report["per_class"][name]
        if c["fp"]:
            any_fp = True
            print("  {} ({}): {}".format(name, c["fp"], ", ".join(c["fp_samples"])))
    if not any_fp:
        print("  (none)")

    print()
    print("False Negatives (pred=0 but GT=1) -- missed behaviors:")
    any_fn = False
    for name in BEHAVIOR_NAMES:
        c = report["per_class"][name]
        if c["fn"]:
            any_fn = True
            print("  {} ({}): {}".format(name, c["fn"], ", ".join(c["fn_samples"])))
    if not any_fn:
        print("  (none)")
    print()


def save_results(predictions: list[dict], report: dict, output_dir: Path,
                 *, save_npy: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print("[eval] saved {} predictions -> {}".format(len(predictions), pred_path))

    rep_path = output_dir / "eval_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[eval] report saved -> {}".format(rep_path))

    if save_npy:
        np.save(output_dir / "confusion_matrix.npy",
                np.array(report["confusion_matrix"], dtype=np.int32))
        np.save(output_dir / "confusion_matrix_hit.npy",
                np.array(report["confusion_matrix_hit"], dtype=np.int32))
