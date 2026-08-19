"""
SAGE Evaluation Script (based on sage_infer.py local VLM inference).

Loads a model (base or LoRA) via sage_infer.LocalModel, runs on images from a
ground-truth file, and computes multi-label metrics.

Usage:
  python scripts/sage_eval.py --model-path Qwen/Qwen3.5-4B --gt data/sage_eval/eval_gt.json --output results/
  python scripts/sage_eval.py --model-path Qwen/Qwen3.5-4B --lora-path ./lora --gt data/sage_eval/eval_gt.json --split test
  python scripts/sage_eval.py --model-path Qwen/Qwen3.5-4B --gt data/sage_eval/eval_gt.json --mode reason --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# -- Constants ---------------------------------------------------------------

BEHAVIOR_NAMES = [
    "normal", "away", "blocked", "toy", "phone", "snack", "eyesclosed",
    "prone", "bowed", "chinrest", "tilt", "turn", "slope", "recline", "lookup",
]

NAME_TO_INDEX = {name: i for i, name in enumerate(BEHAVIOR_NAMES)}
NUM_CLASSES = len(BEHAVIOR_NAMES)

CODE_TO_NAME = {
    "nr": "normal", "aw": "away", "bl": "blocked", "ty": "toy",
    "ph": "phone", "sn": "snack", "ec": "eyesclosed", "pr": "prone",
    "bw": "bowed", "cr": "chinrest", "tl": "tilt", "tn": "turn",
    "sl": "slope", "rc": "recline", "lu": "lookup",
}


# -- Metrics (numpy-only, no sklearn) ---------------------------------------

def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Binary precision / recall / F1 for a single class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


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


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error across all classes and samples."""
    ece = 0.0
    n_total = 0
    for c in range(probs.shape[1]):
        p = probs[:, c]
        y = labels[:, c]
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
    arr = np.array(latencies, dtype=np.float64)
    if len(arr) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "std_ms": 0.0}
    return {
        "mean_ms": round(float(arr.mean()), 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "std_ms": round(float(arr.std()), 1),
    }


# -- Data I/O ----------------------------------------------------------------

def load_ground_truth(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        for name in BEHAVIOR_NAMES:
            s["labels"].setdefault(name, 0)
    return data


def code_probs_to_name_probs(code_probs: dict) -> dict:
    """Convert sage_infer.py 2-letter-code probs to behavior-name probs."""
    out = {name: 0.0 for name in BEHAVIOR_NAMES}
    for code, prob in code_probs.items():
        name = CODE_TO_NAME.get(code, code)
        if name in out:
            out[name] = float(prob)
    return out


# -- Inference ----------------------------------------------------------------

def _error_pred(sid: str, err: str) -> dict:
    return {
        "sample_id": sid,
        "primary_label": "normal",
        "probs": {name: 0.0 for name in BEHAVIOR_NAMES},
        "explanation": "",
        "latency_ms": 0.0,
        "raw_output": "",
        "error": err,
    }


def run_inference(
    gt: list[dict],
    model_path: str,
    output_dir: Path,
    lora_path: str | None = None,
    device: str = "cuda",
    dtype: str = "bf16",
    download_source: str = "auto",
    no_reason: bool = False,
    image_root: str | None = None,
) -> list[dict]:
    """Run sage_infer.py LocalModel on all ground-truth images."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from sage_infer import LocalModel, parse_output, render_prompt  # type: ignore

    prompt = render_prompt(no_reason=no_reason)
    mode = "no_reason" if no_reason else "reason"
    print(f"[eval] mode={mode}, model={model_path}")
    print(f"[eval] {len(gt)} samples")

    model = LocalModel(model_path, device, dtype, lora_path, download_source)
    predictions: list[dict] = []
    errors = 0

    for idx, sample in enumerate(gt):
        sid = sample["sample_id"]
        img_path = sample["image_path"]
        if image_root and not Path(img_path).is_absolute():
            img_path = str(Path(image_root) / img_path)

        if not Path(img_path).exists():
            print(f"  [{idx}] image not found: {img_path}")
            predictions.append(_error_pred(sid, "image_not_found"))
            errors += 1
            continue

        try:
            from PIL import Image
            import io
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)

            t0 = time.perf_counter()
            raw_text, probs = model.infer(
                image_bytes=buf.getvalue(),
                system_prompt=prompt,
                max_tokens=1 if no_reason else 1024,
                no_reason=no_reason,
            )
            ms = round((time.perf_counter() - t0) * 1000, 1)

            r = parse_output(raw_text)
            r.latency_ms = ms
            if probs:
                r.probs = probs
            else:
                r.probs[r.primary_code] = 1.0

            name_probs = code_probs_to_name_probs(r.probs)
            predictions.append({
                "sample_id": sid,
                "primary_label": CODE_TO_NAME.get(r.primary_code, r.primary_code),
                "probs": name_probs,
                "explanation": r.explanation,
                "latency_ms": r.latency_ms,
                "raw_output": raw_text,
            })
        except Exception as e:
            print(f"  [{idx}] inference error ({sid}): {e}")
            predictions.append(_error_pred(sid, str(e)))
            errors += 1

        if (idx + 1) % 20 == 0:
            print(f"  processed {idx + 1}/{len(gt)}, errors={errors}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"[eval] saved {len(predictions)} predictions -> {pred_path}")
    return predictions


# -- Metrics Computation ------------------------------------------------------

def compute_metrics(
    gt: list[dict],
    predictions: list[dict],
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict:
    """Compute all metrics from ground truth and predictions."""
    pred_map = {p["sample_id"]: p for p in predictions}

    labels = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    probs = np.zeros((len(gt), NUM_CLASSES), dtype=np.float32)
    gt_primary = np.zeros(len(gt), dtype=np.int32)
    pred_primary = np.zeros(len(gt), dtype=np.int32)
    latencies: list[float] = []
    n_errors = 0

    for i, sample in enumerate(gt):
        sid = sample["sample_id"]
        labels[i] = [sample["labels"].get(n, 0) for n in BEHAVIOR_NAMES]
        gt_primary[i] = NAME_TO_INDEX.get(sample["primary_label"], 0)

        pred = pred_map.get(sid)
        if pred is None:
            n_errors += 1
            pred_primary[i] = 0
            continue
        if "error" in pred:
            n_errors += 1
        probs[i] = [pred["probs"].get(n, 0.0) for n in BEHAVIOR_NAMES]
        pred_primary[i] = NAME_TO_INDEX.get(pred.get("primary_label", "normal"), 0)
        if pred.get("latency_ms"):
            latencies.append(pred["latency_ms"])

    preds_binary = (probs >= threshold).astype(np.float32)

    # Per-class P/R/F1/AP
    per_class = {}
    f1s, aps = [], []
    for c, name in enumerate(BEHAVIOR_NAMES):
        yt = labels[:, c]
        yp = preds_binary[:, c]
        ys = probs[:, c]
        support = int(yt.sum())
        if support == 0:
            p = r = f1 = ap = 0.0
        else:
            p, r, f1 = _prf(yt, yp)
            ap = average_precision(yt, ys)
        per_class[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "ap": round(ap, 4),
            "support": support,
        }
        f1s.append(f1)
        aps.append(ap)

    macro_f1 = float(np.mean(f1s))
    mAP = float(np.mean(aps))
    ece = compute_ece(probs, labels, ece_bins)
    subset_acc = float(np.mean(np.all(preds_binary == labels, axis=1)))
    hamming = float(np.mean(preds_binary != labels))
    primary_acc = float(np.mean(pred_primary == gt_primary))
    cm = confusion_matrix(gt_primary, pred_primary, NUM_CLASSES)

    return {
        "num_samples": len(gt),
        "num_errors": n_errors,
        "threshold": threshold,
        "ece_bins": ece_bins,
        "overall": {
            "macro_f1": round(macro_f1, 4),
            "mAP": round(mAP, 4),
            "ECE": round(ece, 4),
            "subset_accuracy": round(subset_acc, 4),
            "hamming_loss": round(hamming, 4),
            "primary_accuracy": round(primary_acc, 4),
        },
        "per_class": per_class,
        "latency": latency_stats(latencies),
        "confusion_matrix": cm.tolist(),
        "confusion_labels": BEHAVIOR_NAMES,
    }


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report to stdout."""
    ov = report["overall"]
    print()
    print("=" * 70)
    print(f"Samples={report['num_samples']}, Errors={report['num_errors']}, "
          f"Threshold={report['threshold']}")
    print("=" * 70)
    print()
    print(f"{'Class':<12s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AP':>7s} {'Sup':>6s}")
    print("-" * 50)
    for name in BEHAVIOR_NAMES:
        c = report["per_class"][name]
        print(f"{name:<12s} {c['precision']:7.3f} {c['recall']:7.3f} "
              f"{c['f1']:7.3f} {c['ap']:7.3f} {c['support']:>6d}")
    print()
    print(f"{'Macro-F1':<12s} {ov['macro_f1']:.4f}")
    print(f"{'mAP':<12s} {ov['mAP']:.4f}")
    print(f"{'ECE':<12s} {ov['ECE']:.4f}")
    print(f"{'Subset-Acc':<12s} {ov['subset_accuracy']:.4f}")
    print(f"{'Hamming-Loss':<12s} {ov['hamming_loss']:.4f}")
    print(f"{'Primary-Acc':<12s} {ov['primary_accuracy']:.4f}")
    lat = report.get("latency", {})
    if lat.get("mean_ms", 0) > 0:
        print(f"\nLatency: mean={lat['mean_ms']:.0f}ms, "
              f"p50={lat['p50_ms']:.0f}ms, p95={lat['p95_ms']:.0f}ms")
    print()


def save_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[eval] report saved -> {path}")
    np.save(output_dir / "confusion_matrix.npy",
            np.array(report["confusion_matrix"], dtype=np.int32))


# -- Main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="SAGE evaluation (live inference via sage_infer)")
    p.add_argument("--gt", required=True, help="Ground truth JSON file path")
    p.add_argument("--model-path", required=True,
                   help="Base model path or HuggingFace/ModelScope ID")
    p.add_argument("--lora-path", default="", help="LoRA adapter path (optional)")
    p.add_argument("--mode", choices=["no_reason", "reason"], default="no_reason",
                   help="no_reason: code only; reason: code + explanation")
    p.add_argument("--split", default="",
                   help="Filter GT by split field (e.g. 'test'). Empty = all samples.")
    p.add_argument("--output", default="results/sage_eval",
                   help="Output directory for predictions and report")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary decision threshold for P/R/F1")
    p.add_argument("--download-source", default="auto",
                   choices=["auto", "modelscope", "huggingface"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--image-root", default="",
                   help="Root dir for resolving relative image_path in GT")
    a = p.parse_args()

    no_reason = (a.mode == "no_reason")
    output_dir = Path(a.output)
    gt = load_ground_truth(a.gt)
    if a.split:
        gt = [s for s in gt if s.get("split") == a.split]
        print(f"[eval] filtered to split='{a.split}': {len(gt)} samples")
    print(f"[eval] loaded {len(gt)} ground-truth samples from {a.gt}")

    predictions = run_inference(
        gt=gt, model_path=a.model_path, output_dir=output_dir,
        lora_path=a.lora_path or None, device=a.device, dtype=a.dtype,
        download_source=a.download_source, no_reason=no_reason,
        image_root=a.image_root or None,
    )

    report = compute_metrics(gt, predictions, a.threshold)
    report["model"] = a.model_path
    report["mode"] = a.mode

    print_report(report)
    save_report(report, output_dir)


if __name__ == "__main__":
    main()
