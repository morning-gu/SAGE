#!/usr/bin/env python3
"""Compute a bootstrap confidence interval for macro-F1.

Thin shim over sage.metrics: the per-class F1 / macro-F1 math lives in
sage.metrics; this script only handles the CLI, resampling, and output.
Prediction files are loaded via sage.metrics.load_predictions, which
auto-detects the ``probs`` / ``adjusted_probs`` schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sage.metrics import load_predictions, macro_f1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="tests/fixtures/predictions_d1_ce_bce.json",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output")
    args = parser.parse_args()

    ids, gt, probs = load_predictions(args.predictions)
    preds = (probs >= args.threshold).astype(np.float32)
    n = len(ids)

    rng = np.random.default_rng(args.seed)
    bootstrap_values = np.empty(args.repeats, dtype=np.float64)
    for i in range(args.repeats):
        idx = rng.integers(0, n, size=n)
        bootstrap_values[i] = macro_f1(gt[idx], preds[idx])

    alpha = (1.0 - args.confidence) / 2.0
    result = {
        "num_samples": n,
        "threshold": args.threshold,
        "repeats": args.repeats,
        "seed": args.seed,
        "confidence_level": args.confidence,
        "observed_macro_f1": macro_f1(gt, preds),
        "bootstrap_ci": [
            float(np.percentile(bootstrap_values, alpha * 100.0)),
            float(np.percentile(bootstrap_values, (1.0 - alpha) * 100.0)),
        ],
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
