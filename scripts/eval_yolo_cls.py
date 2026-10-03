"""Evaluate YOLOv8s-cls on SAGE test set with multi-label metrics.

Runs inference, extracts 15-dim softmax probabilities, computes
macro-F1 / mAP / ECE aligned with ``eval_sage.py``, and reports
multi-label capability analysis showing softmax's inability to
predict co-occurring labels.

Usage:
    python scripts/eval_yolo_cls.py --model checkpoints/yolo_cls/train/weights/best.pt
    python scripts/eval_yolo_cls.py --model best.pt --gt data/sage_eval/annotations.json --split test
"""
from __future__ import annotations

import argparse
from sage.paths import GT_ANNOTATIONS
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Shared taxonomy + metric implementations (single source of truth).
from sage.metrics import (  # noqa: E402
    BEHAVIOR_NAMES,
    NUM_CLASSES,
    per_class_prf,
)

NAME_TO_INDEX = {name: i for i, name in enumerate(BEHAVIOR_NAMES)}

# Reuse metric functions from eval_sage.py (numpy-only, no sage dependency).
from sage.eval_harness import compute_metrics, save_results  # noqa: E402


# -- Inference ---------------------------------------------------------------

def run_inference(model_path: str, gt: list[dict], image_root: Path) -> list[dict]:
    """Run YOLOv8s-cls inference on test images, return prediction records
    in the same format that ``sage.eval_harness.compute_metrics`` expects.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_names = model.names  # {0: 'away', 1: 'blocked', ...} (alphabetical)

    yolo_classes = set(class_names.values())
    expected = set(BEHAVIOR_NAMES)
    if yolo_classes != expected:
        missing = expected - yolo_classes
        extra = yolo_classes - expected
        print(f"[eval] WARN class mismatch: missing={missing}, extra={extra}")

    predictions: list[dict] = []
    n = len(gt)

    for i, sample in enumerate(gt):
        img_path = sample["image_path"]
        if not Path(img_path).is_absolute():
            img_path = str(image_root / img_path)

        gt_labels = dict(sample["labels"])
        gt_primary = sample.get("primary_label", "normal")

        try:
            t0 = time.perf_counter()
            results = model(img_path, verbose=False)
            ms = round((time.perf_counter() - t0) * 1000, 1)

            result = results[0]
            probs = result.probs.data.cpu().numpy()  # [C] softmax probs

            yolo_probs = {}
            for idx in range(len(probs)):
                name = class_names.get(idx, str(idx))
                yolo_probs[name] = float(probs[idx])
            adjusted_probs = {name: yolo_probs.get(name, 0.0) for name in BEHAVIOR_NAMES}

            top1_idx = int(result.probs.top1)
            final_name = class_names.get(top1_idx, "normal")
            if final_name not in BEHAVIOR_NAMES:
                final_name = "normal"

            pred = {
                "sample_id": sample["sample_id"],
                "image_path": img_path,
                "gt_labels": gt_labels,
                "gt_primary": gt_primary,
                "final_code": "",
                "final_name": final_name,
                "confidence": round(float(probs[top1_idx]), 4),
                "source": "yolo_cls",
                "evidence": "",
                "adjusted_probs": adjusted_probs,
                "vlm_primary_code": "",
                "vlm_primary_name": final_name,
                "recheck_verdicts": {
                    name: {"verdict": "unavailable", "evidence": ""}
                    for name in BEHAVIOR_NAMES
                },
                "latency_ms": ms,
                "error": None,
            }
        except Exception as e:
            pred = {
                "sample_id": sample["sample_id"],
                "image_path": img_path,
                "gt_labels": gt_labels,
                "gt_primary": gt_primary,
                "final_code": "",
                "final_name": "normal",
                "confidence": 0.0,
                "source": "error",
                "evidence": "",
                "adjusted_probs": {n: 0.0 for n in BEHAVIOR_NAMES},
                "vlm_primary_code": "",
                "vlm_primary_name": "normal",
                "recheck_verdicts": {},
                "latency_ms": 0.0,
                "error": str(e),
            }

        predictions.append(pred)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {sample['sample_id']} -> {pred['final_name']}"
                  f" ({pred['latency_ms']}ms)")

    return predictions


# -- Multi-label analysis ----------------------------------------------------

def multi_label_analysis(gt: list[dict], predictions: list[dict]) -> dict:
    """Analyze YOLO's multi-label capability on multi-label samples."""
    pred_map = {p["sample_id"]: p for p in predictions}

    probs = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    labels = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)

    for i, sample in enumerate(gt):
        pred = pred_map.get(sample["sample_id"])
        if pred and not pred.get("error"):
            probs[i] = [pred["adjusted_probs"].get(n, 0.0) for n in BEHAVIOR_NAMES]
        labels[i] = [sample["labels"].get(n, 0) for n in BEHAVIOR_NAMES]

    preds_binary = (probs >= 0.5).astype(np.float32)

    multi_idx = [i for i in range(len(gt)) if labels[i].sum() > 1]
    single_idx = [i for i in range(len(gt)) if labels[i].sum() == 1]

    analysis: dict = {
        "single_label_count": len(single_idx),
        "multi_label_count": len(multi_idx),
    }

    ml_details = []
    for idx in multi_idx:
        sample = gt[idx]
        gt_labels = [BEHAVIOR_NAMES[j] for j in range(NUM_CLASSES) if labels[idx, j] == 1]
        pred_labels = [BEHAVIOR_NAMES[j] for j in range(NUM_CLASSES) if preds_binary[idx, j] == 1]
        hits = set(gt_labels) & set(pred_labels)

        top2_idx = np.argsort(-probs[idx])[:2]
        top2_names = [BEHAVIOR_NAMES[j] for j in top2_idx]
        top2_hits = len(set(gt_labels) & set(top2_names))

        ml_details.append({
            "sample_id": sample["sample_id"],
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "hits": len(hits),
            "gt_label_count": len(gt_labels),
            "top2": top2_names,
            "top2_hits": top2_hits,
            "max_prob": round(float(probs[idx].max()), 4),
            "second_prob": round(float(np.sort(probs[idx])[-2]), 4),
        })
    analysis["multi_label_details"] = ml_details

    for label_name, indices in [("single_label", single_idx), ("multi_label", multi_idx)]:
        if not indices:
            continue
        sub = probs[indices]
        sorted_probs = np.sort(sub, axis=1)[:, ::-1]
        analysis[f"{label_name}_max_prob_mean"] = round(float(sorted_probs[:, 0].mean()), 4)
        if sorted_probs.shape[1] > 1:
            analysis[f"{label_name}_second_prob_mean"] = round(float(sorted_probs[:, 1].mean()), 4)
        else:
            analysis[f"{label_name}_second_prob_mean"] = 0.0

    if multi_idx:
        ml_labels = labels[multi_idx]
        ml_preds = preds_binary[multi_idx]
        ml_f1s = []
        for c in range(NUM_CLASSES):
            if ml_labels[:, c].sum() == 0:
                continue
            ml_f1s.append(per_class_prf(ml_labels[:, c], ml_preds[:, c])[2])
        analysis["multi_label_macro_f1"] = round(float(np.mean(ml_f1s)), 4) if ml_f1s else 0.0
    else:
        analysis["multi_label_macro_f1"] = 0.0

    return analysis


# -- Report ------------------------------------------------------------------

def print_yolo_report(report: dict, ml_analysis: dict) -> None:
    ov = report["overall"]
    print()
    print("=" * 90)
    print("YOLOv8s-cls Baseline Evaluation (softmax classification head)")
    print(f"Samples={report['num_samples']}, Errors={report['num_errors']}, "
          f"Threshold={report['threshold']}")
    print("=" * 90)
    print()

    hdr = "{:<12s} {:>7s} {:>7s} {:>7s} {:>7s} {:>6s} {:>4s} {:>4s}".format(
        "Class", "P", "R", "F1", "AP", "Sup", "FP", "FN")
    print(hdr)
    print("-" * len(hdr))
    for name in BEHAVIOR_NAMES:
        c = report["per_class"][name]
        print("{:<12s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:>6d} {:>4d} {:>4d}".format(
            name, c["precision"], c["recall"], c["f1"], c["ap"],
            c["support"], c["fp"], c["fn"]))

    print()
    print(f"Macro-F1:            {ov['macro_f1']:.4f}")
    print(f"mAP:                 {ov['mAP']:.4f}")
    print(f"ECE:                 {ov['ECE']:.4f}")
    print(f"Subset-Acc:          {ov['subset_accuracy']:.4f}")
    print(f"Hamming-Loss:        {ov['hamming_loss']:.4f}")
    print(f"Primary-Acc (hit):   {ov['primary_accuracy']:.4f}")
    print(f"Primary-Acc (exact): {ov['primary_accuracy_exact']:.4f}")

    lat = report.get("latency", {})
    if lat.get("mean_ms", 0) > 0:
        print(f"\nLatency: mean={lat['mean_ms']:.0f}ms, "
              f"p50={lat['p50_ms']:.0f}ms, p95={lat['p95_ms']:.0f}ms")

    print()
    print("=" * 90)
    print("Multi-Label Capability Analysis")
    print("=" * 90)
    print(f"Single-label samples: {ml_analysis['single_label_count']}")
    print(f"Multi-label samples:  {ml_analysis['multi_label_count']}")
    print(f"Multi-label macro-F1: {ml_analysis.get('multi_label_macro_f1', 0.0):.4f}")

    if "single_label_max_prob_mean" in ml_analysis:
        print(f"\nSoftmax probability distribution:")
        print(f"  Single-label: max={ml_analysis['single_label_max_prob_mean']:.3f}, "
              f"second={ml_analysis['single_label_second_prob_mean']:.3f}")
        print(f"  Multi-label:  max={ml_analysis['multi_label_max_prob_mean']:.3f}, "
              f"second={ml_analysis['multi_label_second_prob_mean']:.3f}")

    if ml_analysis["multi_label_details"]:
        print(f"\nPer-sample multi-label breakdown:")
        for d in ml_analysis["multi_label_details"]:
            print(f"  {d['sample_id']}: GT={d['gt_labels']}, Pred={d['pred_labels']}, "
                  f"Hits={d['hits']}/{d['gt_label_count']}, "
                  f"Top2={d['top2']} ({d['top2_hits']} hits), "
                  f"max={d['max_prob']:.3f}, 2nd={d['second_prob']:.3f}")
    print()


# -- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Path to trained YOLOv8s-cls best.pt")
    ap.add_argument("--gt", default=str(GT_ANNOTATIONS),
                    help="Ground truth annotations JSON")
    ap.add_argument("--split", default="test",
                    help="Split to evaluate (default: test)")
    ap.add_argument("--output", default=str(REPO / "results/yolo_cls_eval"),
                    help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binary decision threshold for P/R/F1")
    ap.add_argument("--image-root", default="",
                    help="Root dir for resolving relative image paths")
    a = ap.parse_args()

    image_root = Path(a.image_root) if a.image_root else REPO

    with open(a.gt, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        for name in BEHAVIOR_NAMES:
            s["labels"].setdefault(name, 0)
    gt = [s for s in data if s.get("split") == a.split]
    print(f"[eval] {a.split} split: {len(gt)} samples from {a.gt}")
    print(f"[eval] model: {a.model}")

    predictions = run_inference(a.model, gt, image_root)

    report = compute_metrics(
        gt, predictions, prob_key="adjusted_probs",
        primary_key="final_name", threshold=a.threshold, fusion=True)
    report["model"] = "yolov8s-cls"
    report["split"] = a.split

    ml_analysis = multi_label_analysis(gt, predictions)
    report["multi_label_analysis"] = ml_analysis

    print_yolo_report(report, ml_analysis)
    save_results(predictions, report, Path(a.output))

    print(f"[eval] results saved -> {a.output}")


if __name__ == "__main__":
    main()
