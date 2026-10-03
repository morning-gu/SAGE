"""Bootstrap paired test for macro-F1 difference between two models.

Thin shim over sage.metrics (macro_f1, load_predictions, align_predictions).
The two prediction files are row-aligned by sample_id via an explicit join
(previously the script only asserted that row orders matched).  The
resampling loop mirrors sage.metrics.paired_bootstrap but also reports the
standard deviation of the delta distribution, matching the historical CLI
output format.

Usage:
  python scripts/compute_macro_f1_paired_test.py \
      --a tests/fixtures/predictions_d1_ce_bce.json \
      --b tests/fixtures/predictions_d1_ce_bce.json \
      --n-boot 10000 --seed 42
"""
import argparse

import numpy as np

from sage.metrics import (  # noqa: E402
    align_predictions,
    load_predictions,
    macro_f1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids_a, gt, probs_a = load_predictions(args.a)
    ids_b, _, probs_b = load_predictions(args.b)
    gt, probs_a, probs_b = align_predictions(
        ids_a, gt, probs_a, ids_b, probs_b)

    preds_a = (probs_a >= 0.5).astype(np.float32)
    preds_b = (probs_b >= 0.5).astype(np.float32)
    ma = macro_f1(gt, preds_a)
    mb = macro_f1(gt, preds_b)

    # Paired bootstrap on the macro-F1 difference (B - A); same procedure
    # as sage.metrics.paired_bootstrap with preds_a/preds_b swapped so the
    # reported delta keeps the historical sign convention.
    rng = np.random.default_rng(args.seed)
    n = gt.shape[0]
    deltas = np.empty(args.n_boot, dtype=np.float64)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (macro_f1(gt[idx], preds_b[idx])
                     - macro_f1(gt[idx], preds_a[idx]))
    p_le = float(np.mean(deltas <= 0))
    p_ge = float(np.mean(deltas >= 0))
    pv = min(1.0, 2.0 * min(p_le, p_ge))
    md = float(np.mean(deltas))
    sd = float(np.std(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    print(f"A macro-F1={ma:.4f}, B macro-F1={mb:.4f}, Δ(B-A)={mb - ma:+.4f}")
    print(f"Bootstrap (n={args.n_boot}, seed={args.seed}): "
          f"Δ={md:+.4f}±{sd:.4f}, 95%CI=[{lo:+.4f},{hi:+.4f}], p={pv:.6f}")


if __name__ == "__main__":
    main()
