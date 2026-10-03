"""Evaluate YOLOv8s backbone + sigmoid-BCE multi-label head on SAGE test set.

Companion to scripts/train_yolo_bce.py — rebuilds the exact architecture via
sage.models.build_yolo_bce and loads the state_dict that sage.ml_common.fit
saves to best.pt ({"model_state_dict": ...}; NOT an ultralytics checkpoint,
so eval_yolo_cls.py cannot load it).

Preprocessing matches training exactly (fairness protocol):
build_transform(imgsz, train=False, normalize="scale01") from sage.data —
[0,1] inputs, no ImageNet statistics.

Usage:
    python scripts/eval_yolo_bce.py --model results/experiments/<exp>/train/best.pt \
        --imgsz 224
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
from sage.data import (  # noqa: E402
    BEHAVIOR_NAMES,
    NUM_CLASSES,
    build_transform,
)
from sage.models import build_yolo_bce  # noqa: E402
from sage.paths import GT_ANNOTATIONS  # noqa: E402
from sage.eval_harness import compute_metrics, save_results  # noqa: E402


def load_model(ckpt_path: str, device: torch.device,
               weights: str = "models/detectors/yolov8s-cls.pt") -> torch.nn.Module:
    """Rebuild the yolo_bce architecture and load the trained state_dict."""
    model = build_yolo_bce(weights=weights, num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_inference(model_path: str, gt: list[dict], image_root: Path,
                  device: torch.device, imgsz: int = 224) -> list[dict]:
    """Run inference; prediction records match sage.eval_harness.compute_metrics."""
    model = load_model(model_path, device)
    tf = build_transform(imgsz=imgsz, train=False, normalize="scale01")

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
            top1_idx = int(np.argmax(probs))
            predictions.append({
                "sample_id": sample["sample_id"],
                "image_path": img_path,
                "gt_labels": gt_labels,
                "gt_primary": gt_primary,
                "final_code": "",
                "final_name": BEHAVIOR_NAMES[top1_idx],
                "confidence": round(float(probs[top1_idx]), 4),
                "source": "yolo_bce",
                "evidence": "",
                "adjusted_probs": {BEHAVIOR_NAMES[j]: round(float(probs[j]), 4)
                                   for j in range(NUM_CLASSES)},
                "vlm_primary_code": "",
                "vlm_primary_name": BEHAVIOR_NAMES[top1_idx],
                "recheck_verdicts": {},
                "latency_ms": ms,
                "error": None,
            })
        except Exception as e:
            predictions.append({
                "sample_id": sample["sample_id"],
                "image_path": img_path,
                "gt_labels": gt_labels,
                "gt_primary": gt_primary,
                "final_code": "",
                "final_name": "normal",
                "confidence": 0.0,
                "source": "error",
                "evidence": "",
                "adjusted_probs": {n_: 0.0 for n_ in BEHAVIOR_NAMES},
                "vlm_primary_code": "",
                "vlm_primary_name": "normal",
                "recheck_verdicts": {},
                "latency_ms": 0.0,
                "error": str(e),
            })
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {sample['sample_id']} -> "
                  f"{predictions[-1]['final_name']}")
    return predictions


def multi_label_analysis(gt: list[dict], predictions: list[dict],
                         threshold: float) -> dict:
    """Compact sigmoid multi-label capability summary."""
    pred_map = {p["sample_id"]: p for p in predictions}
    probs = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    labels = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    for i, sample in enumerate(gt):
        pred = pred_map.get(sample["sample_id"])
        if pred and not pred.get("error"):
            probs[i] = [pred["adjusted_probs"].get(n_, 0.0) for n_ in BEHAVIOR_NAMES]
        labels[i] = [sample["labels"].get(n_, 0) for n_ in BEHAVIOR_NAMES]
    preds_binary = (probs >= threshold).astype(np.float32)
    multi_idx = [i for i in range(len(gt)) if labels[i].sum() > 1]
    analysis: dict = {
        "single_label_count": len(gt) - len(multi_idx),
        "multi_label_count": len(multi_idx),
        "mean_pred_positives_per_sample": round(float(preds_binary.sum(1).mean()), 3),
    }
    if multi_idx:
        hits = (preds_binary[multi_idx] * labels[multi_idx]).sum(1)
        analysis["multi_label_partial_hits_mean"] = round(float(hits.mean()), 3)
        analysis["multi_label_gt_labels_mean"] = round(
            float(labels[multi_idx].sum(1).mean()), 3)
    return analysis


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Path to trained yolo_bce checkpoint (state_dict .pt)")
    ap.add_argument("--gt", default=str(GT_ANNOTATIONS),
                    help="Ground truth annotations JSON")
    ap.add_argument("--split", default="test",
                    help="Split to evaluate (default: test)")
    ap.add_argument("--output", default=str(REPO / "results/yolo_bce_eval"),
                    help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binary decision threshold for P/R/F1")
    ap.add_argument("--imgsz", type=int, default=224,
                    help="Eval image size — MUST match the training grid imgsz")
    ap.add_argument("--weights", default="models/detectors/yolov8s-cls.pt",
                    help="YOLO-cls pretrained weights (architecture template)")
    ap.add_argument("--image-root", default="",
                    help="Root dir for resolving relative image paths")
    a = ap.parse_args()

    image_root = Path(a.image_root) if a.image_root else REPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(a.gt, encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        for name in BEHAVIOR_NAMES:
            s["labels"].setdefault(name, 0)
    gt = [s for s in data if s.get("split") == a.split]
    print(f"[eval] {a.split} split: {len(gt)} samples from {a.gt}")
    print(f"[eval] model: {a.model}  imgsz={a.imgsz}  device: {device}")

    predictions = run_inference(a.model, gt, image_root, device, imgsz=a.imgsz)

    report = compute_metrics(
        gt, predictions, prob_key="adjusted_probs",
        primary_key="final_name", threshold=a.threshold, fusion=True)
    report["model"] = "yolov8s-backbone-sigmoid-bce"
    report["split"] = a.split
    report["imgsz"] = a.imgsz
    report["multi_label_analysis"] = multi_label_analysis(gt, predictions,
                                                          a.threshold)
    save_results(predictions, report, Path(a.output))
    print(f"[eval] results saved -> {a.output}")


if __name__ == "__main__":
    main()
