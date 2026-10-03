#!/usr/bin/env python3
"""Compute calibration metrics from SAGE prediction files.

Thin shim over sage.metrics: the ECE implementation lives in
sage.metrics.compute_ece (identical binning semantics: per-class, 10 equal
bins on (0, 1], clipped probabilities, slot-count weighted).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sage.metrics import compute_ece, load_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="tests/fixtures/predictions_d1_ce_bce.json",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--extreme-threshold", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    ids, gt, probs = load_predictions(args.predictions)

    # Slot-wise (per sample x per class) statistics, same flattening the
    # original inline implementation used.
    slot_count = int(gt.size)
    targets = gt.astype(np.float64).ravel()
    probabilities = probs.astype(np.float64).ravel()
    positive_count = int(gt.sum())
    negative_count = slot_count - positive_count
    probability_brier = float(np.mean((probabilities - targets) ** 2))
    binary_brier = float(np.mean(
        ((probabilities >= args.threshold).astype(np.float64) - targets) ** 2))
    extreme_count = int(np.sum(
        (probabilities <= args.extreme_threshold)
        | (probabilities >= 1.0 - args.extreme_threshold)))

    result = {
        "num_samples": len(ids),
        "num_label_slots": slot_count,
        "positive_label_slots": positive_count,
        "negative_label_slots": negative_count,
        "threshold": args.threshold,
        "extreme_threshold": args.extreme_threshold,
        "extreme_rate": extreme_count / slot_count,
        "probability_brier": probability_brier,
        "binary_brier_or_hamming_loss": binary_brier,
        "ece": compute_ece(probs, gt, args.bins),
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
