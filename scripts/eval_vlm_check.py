"""SAGE VLM Check Evaluation Script.

Calls the VLM detection service (HTTP) on images from a ground-truth
annotation file and computes per-class and overall metrics. The model is
loaded by the service, not by this script. Batch execution, metrics, report
printing, and artifact saving live in sage.eval_harness; this script only
defines the VLM predict adapter and failure-image annotation.

Usage:
    uv run python scripts/eval_vlm_check.py --gt data/sage_eval/annotations.json --split test
    uv run python scripts/eval_vlm_check.py --gt data/sage_eval/annotations.json --vlm-url http://localhost:8010 --split test --output results/vlm_eval/
    uv run python scripts/eval_vlm_check.py --gt data/sage_eval/annotations.json --limit 10
    uv run python scripts/eval_vlm_check.py --gt data/sage_eval/annotations.json --split test --annotate-failures results/vlm_fail/

Environment variables (read by VLMClient.from_env):
    SAGE_VLM_URL      default http://localhost:8010
    SAGE_VLM_TIMEOUT  default 60
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sage_check.client import VLMClient
from sage.taxonomy import CODE_TO_NAME
from sage.eval_harness import (
    BEHAVIOR_NAMES,
    code_probs_to_name_probs,
    compute_metrics,
    encode_image,
    load_ground_truth,
    make_error_pred,
    print_report,
    run_eval,
    save_results,
)


def _vlm_record_template(sid: str, img_path: str, gt_labels: dict,
                         gt_primary: str) -> dict:
    return {
        "sample_id": sid,
        "image_path": img_path,
        "gt_labels": gt_labels,
        "gt_primary": gt_primary,
        "primary_label": "normal",
        "probs": {name: 0.0 for name in BEHAVIOR_NAMES},
        "explanation": "",
        "latency_ms": 0.0,
        "raw_output": "",
        "error": None,
    }


def _error_pred(sample: dict, err: str) -> dict:
    return make_error_pred(_vlm_record_template(
        sample["sample_id"], sample["image_path"],
        dict(sample["labels"]), sample.get("primary_label", "")), err)


def predict(sample: dict, client, mode: str,
            primary_source: str = "generated") -> dict:
    """Run VLM inference on one sample via the detection service.

    primary_source selects the primary-decision mode under evaluation (E2):
      generated — the autoregressive primary-label token (default; the C2
                  dual-channel decision), or
      argmax    — argmax over the auxiliary first-token sigmoid probabilities
                  (decided client-side; no server change needed).
    """
    sid = sample["sample_id"]
    img_path = sample["image_path"]
    gt_labels = dict(sample["labels"])
    gt_primary = sample.get("primary_label", "")

    image_b64 = encode_image(img_path)
    resp = client.detect(image_b64, mode)

    if resp.get("error"):
        return _error_pred(sample, resp["error"])

    code_probs = resp.get("probabilities", {})
    primary_code = resp.get("primary_code", "")
    if not code_probs:
        code_probs = {primary_code: 1.0}
    if primary_source == "argmax":
        # E2 decision-mode ablation: override the generated primary with the
        # argmax of the auxiliary multi-label channel.
        primary_code = max(code_probs, key=lambda c: code_probs.get(c, 0.0))

    return {
        "sample_id": sid,
        "image_path": img_path,
        "gt_labels": gt_labels,
        "gt_primary": gt_primary,
        "primary_label": CODE_TO_NAME.get(primary_code, primary_code),
        "probs": code_probs_to_name_probs(code_probs),
        "explanation": resp.get("explanation", ""),
        "latency_ms": resp.get("latency_ms", 0.0),
        "raw_output": resp.get("raw_output", ""),
        "error": None,
    }


def annotate_failures(failure_ids: set, predictions: list[dict],
                      output_dir: Path) -> None:
    """Save failure images with prediction vs GT text overlay."""
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_map = {p["sample_id"]: p for p in predictions}

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for sid in sorted(failure_ids):
        pred = pred_map.get(sid)
        if not pred:
            continue
        img_path = pred["image_path"]
        if not Path(img_path).exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            gt_labels = [n for n in BEHAVIOR_NAMES if pred["gt_labels"].get(n) == 1]
            pred_label = pred.get("primary_label", "?")
            top_prob = max(pred["probs"].values()) if pred["probs"] else 0.0

            lines = [
                "ID: {}".format(sid),
                "GT: {}".format(", ".join(gt_labels) or "none"),
                "Pred: {} (p={:.2f})".format(pred_label, top_prob),
            ]
            if pred.get("error"):
                lines.append("ERR: {}".format(pred["error"]))

            y = 4
            for line in lines:
                draw.rectangle([(0, y - 2), (360, y + 22)], fill=(0, 0, 0))
                draw.text((6, y), line, fill=(255, 255, 0), font=font)
                y += 24

            out_path = output_dir / (sid + "_annotated.jpg")
            img.save(str(out_path), format="JPEG", quality=90)
        except Exception as e:
            print("  annotate error ({}): {}".format(sid, e))


def main():
    p = argparse.ArgumentParser(
        description="SAGE VLM check evaluation (via VLM detection service)")
    p.add_argument("--gt", required=True, help="Ground truth JSON file path")
    p.add_argument("--vlm-url", default="",
                   help="VLM service URL (default: SAGE_VLM_URL or http://localhost:8010)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="HTTP timeout seconds")
    p.add_argument("--mode", choices=["no_reason", "reason"], default="no_reason",
                   help="no_reason: code only; reason: code + explanation")
    p.add_argument("--primary-source", choices=["generated", "argmax"],
                   default="generated",
                   help="primary-decision mode under evaluation (E2): "
                        "generated = autoregressive primary token (default); "
                        "argmax = argmax over the auxiliary multi-label "
                        "sigmoid probabilities (client-side override)")
    p.add_argument("--split", default="",
                   help="Filter by split (test/val/train). Empty = all.")
    p.add_argument("--output", default="results/vlm_eval",
                   help="Output directory")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary decision threshold for P/R/F1")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N samples (0 = all)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (1 = sequential)")
    p.add_argument("--image-root", default="",
                   help="Root dir for resolving relative image paths")
    p.add_argument("--annotate-failures", default="",
                   help="Save annotated images for FP/FN cases to this dir")
    a = p.parse_args()

    gt = load_ground_truth(a.gt, a.split)
    if a.split:
        print("[eval] filtered to split={}: {} samples".format(a.split, len(gt)))
    print("[eval] loaded {} ground-truth samples from {}".format(len(gt), a.gt))

    # Configure VLM detection service client (model lives in the service)
    if a.vlm_url:
        os.environ["SAGE_VLM_URL"] = a.vlm_url
    os.environ["SAGE_VLM_TIMEOUT"] = str(a.timeout)

    client = VLMClient.from_env()
    print("[eval] mode={}, vlm_url={}".format(a.mode, client.base_url))

    predictions = run_eval(
        gt, lambda s: predict(s, client, a.mode, a.primary_source), _error_pred,
        image_root=a.image_root or None,
        workers=a.workers, limit=a.limit,
        summary_key="primary_label")

    report = compute_metrics(
        gt, predictions,
        prob_key="probs", primary_key="primary_label",
        threshold=a.threshold)
    report["vlm_url"] = client.base_url
    report["mode"] = a.mode
    report["primary_source"] = a.primary_source
    report["split"] = a.split or "all"

    print_report(report, "SAGE VLM Check Evaluation")
    save_results(predictions, report, Path(a.output))

    # Total-request failure means infrastructure trouble (e.g. the detection
    # server died), not a model result.  A garbage all-error report must not
    # count as "done" for the runner's idempotence — exit non-zero so the
    # runner marks the experiment failed (and removes the partial report).
    if gt and int(report.get("num_errors", 0)) == len(gt):
        print("[eval] ALL {} requests failed (service unreachable or crashed) "
              "— exiting 2".format(len(gt)))
        sys.exit(2)

    if a.annotate_failures:
        fail_ids = set()
        for name in BEHAVIOR_NAMES:
            c = report["per_class"][name]
            fail_ids.update(c.get("fp_samples", []))
            fail_ids.update(c.get("fn_samples", []))
        print("\n[eval] annotating {} failure images...".format(len(fail_ids)))
        annotate_failures(fail_ids, predictions, Path(a.annotate_failures))


if __name__ == "__main__":
    main()
