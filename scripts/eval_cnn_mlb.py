"""Evaluate ResNet50 + sigmoid + BCE multi-label classifier on SAGE test set.

This is the CNN multi-label baseline (D-cnn) evaluation script.  It mirrors
``eval_yolo_cls.py`` in structure and reuses the metric functions from
``eval_sage.py`` so that results are directly comparable with D0/D1/D1-CE/
D-yolo/D-sup/D2.

Key difference from YOLO baseline: this model outputs 15 **independent**
 sigmoid probabilities (BCE-trained), so multiple labels can be predicted
 simultaneously.  The multi-label analysis section quantifies this capability.

Usage:
    python scripts/eval_cnn_mlb.py --model results/cnn_mlb/best.pt
    python scripts/eval_cnn_mlb.py --model best.pt --gt data/sage_eval/annotations.json --split test
"""
from __future__ import annotations

import argparse
from sage.paths import GT_ANNOTATIONS
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
# Shared taxonomy + metric implementations (single source of truth).
from sage.metrics import (  # noqa: E402
    BEHAVIOR_NAMES,
    NUM_CLASSES,
    per_class_prf,
)

NAME_TO_INDEX = {name: i for i, name in enumerate(BEHAVIOR_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Shared harness (batch metrics + artifact saving; single source of truth).
from sage.eval_harness import compute_metrics, save_results  # noqa: E402


# -- Model -------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load ResNet50 with trained weights from checkpoint."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# -- Inference ----------------------------------------------------------------

def run_inference(model_path: str, gt: list[dict], image_root: Path,
                  device: torch.device, batch_size: int = 32) -> list[dict]:
    """Run ResNet50 inference on test images, return prediction records
    in the same format that ``sage.eval_harness.compute_metrics`` expects.
    """
    model = load_model(model_path, device)
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    predictions: list[dict] = []
    n = len(gt)

    for i, sample in enumerate(gt):
        img_path = sample["image_path"]
        if not Path(img_path).is_absolute():
            img_path = str(image_root / img_path)

        gt_labels = dict(sample["labels"])
        gt_primary = sample.get("primary_label", "normal")

        try:
            img = Image.open(img_path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            ms = round((time.perf_counter() - t0) * 1000, 1)

            adjusted_probs = {
                BEHAVIOR_NAMES[j]: round(float(probs[j]), 4)
                for j in range(NUM_CLASSES)
            }
            top1_idx = int(np.argmax(probs))
            final_name = BEHAVIOR_NAMES[top1_idx]

            pred = {
                "sample_id": sample["sample_id"],
                "image_path": img_path,
                "gt_labels": gt_labels,
                "gt_primary": gt_primary,
                "final_code": "",
                "final_name": final_name,
                "confidence": round(float(probs[top1_idx]), 4),
                "source": "cnn_mlb",
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
    """Analyze CNN multi-label capability on multi-label samples.

    Mirrors the analysis in ``eval_yolo_cls.py`` but for sigmoid outputs.
    Unlike YOLO's softmax, sigmoid allows multiple labels above threshold.
    """
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

def print_cnn_report(report: dict, ml_analysis: dict) -> None:
    ov = report["overall"]
    print()
    print("=" * 90)
    print("ResNet50 + sigmoid + BCE Multi-Label Baseline (D-cnn)")
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
    print("Multi-Label Capability Analysis (sigmoid, threshold=0.5)")
    print("=" * 90)
    print(f"Single-label samples: {ml_analysis['single_label_count']}")
    print(f"Multi-label samples:  {ml_analysis['multi_label_count']}")
    print(f"Multi-label macro-F1: {ml_analysis.get('multi_label_macro_f1', 0.0):.4f}")

    if "single_label_max_prob_mean" in ml_analysis:
        print(f"\nSigmoid probability distribution:")
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
                    help="Path to trained CNN checkpoint (.pt)")
    ap.add_argument("--gt", default=str(GT_ANNOTATIONS),
                    help="Ground truth annotations JSON")
    ap.add_argument("--split", default="test",
                    help="Split to evaluate (default: test)")
    ap.add_argument("--output", default=str(REPO / "results/cnn_mlb_eval"),
                    help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binary decision threshold for P/R/F1")
    ap.add_argument("--image-root", default="",
                    help="Root dir for resolving relative image paths")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Inference batch size (1 for per-sample latency)")
    a = ap.parse_args()

    image_root = Path(a.image_root) if a.image_root else REPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(a.gt, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        for name in BEHAVIOR_NAMES:
            s["labels"].setdefault(name, 0)
    gt = [s for s in data if s.get("split") == a.split]
    print(f"[eval] {a.split} split: {len(gt)} samples from {a.gt}")
    print(f"[eval] model: {a.model}")
    print(f"[eval] device: {device}")

    predictions = run_inference(a.model, gt, image_root, device, a.batch_size)

    report = compute_metrics(
        gt, predictions, prob_key="adjusted_probs",
        primary_key="final_name", threshold=a.threshold, fusion=True)
    report["model"] = "resnet50-sigmoid-bce"
    report["split"] = a.split

    ml_analysis = multi_label_analysis(gt, predictions)
    report["multi_label_analysis"] = ml_analysis

    print_cnn_report(report, ml_analysis)
    save_results(predictions, report, Path(a.output))

    print(f"[eval] results saved -> {a.output}")


if __name__ == "__main__":
    main()
