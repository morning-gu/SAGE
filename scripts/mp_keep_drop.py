"""MP keep/drop preregistered decision (docs/ablation_protocol.md §4).

Frozen rule (2026-09-21, anti-HARKing — must not be modified after seeing
the val numbers):

  Per seed pair (42/43/44): paired bootstrap of the macro-F1 difference
  D1(BCE+MP) - D1(BCE-only) on the VAL split (10000 resamples, seed=42,
  two-sided percentile), via sage.metrics.paired_bootstrap.

  KEEP : >= 2/3 seeds' 95% CI excludes 0  AND  pooled (row-concatenated
         three-seed) delta macro-F1 >= +1pp
  DROP : otherwise — including "inconclusive" (mixed seed directions);
         no post-hoc seed picking.  MP stays as an implementation detail
         in code + appendix with the ablation numbers reported.

Usage:
  python scripts/mp_keep_drop.py \
    --mp-glob 'results/experiments/d1_bce_mp_s*_val/predictions.json' \
    --nomp-glob 'results/experiments/d1_bce_nomp_s*_val/predictions.json' \
    --out results/mp_keep_drop_decision.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sage.metrics import paired_bootstrap

N_BOOT = 10000
BOOT_SEED = 42
POOLED_MIN_DELTA = 0.01  # +1pp


def _matrix(pred_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """([N,15] binary predictions at thr 0.5, [N,15] gt, sample_ids)."""
    preds = json.loads(Path(pred_path).read_text())
    classes = sorted({c for p in preds for c in p["gt_labels"]})
    y = np.array([[int(bool(p["gt_labels"][c])) for c in classes] for p in preds])
    yh = np.array([[int((p.get("probs") or {}).get(c, 0.0) >= 0.5)
                    for c in classes] for p in preds])
    return y, yh, [p["sample_id"] for p in preds]


def _load_pair(mp_glob: str, nomp_glob: str, seed: int):
    def find(glob: str) -> str:
        hits = sorted(Path().glob(glob.replace("s*", f"s{seed}*")))
        if not hits:
            raise SystemExit(f"missing predictions for seed {seed}: {glob}")
        return str(hits[0])

    y_m, p_m, ids_m = _matrix(find(mp_glob))
    y_n, p_n, ids_n = _matrix(find(nomp_glob))
    if ids_m != ids_n:
        raise SystemExit(f"sample order mismatch for seed {seed}")
    if not np.array_equal(y_m, y_n):
        raise SystemExit(f"gt mismatch between mp/nomp for seed {seed}")
    return y_m, p_m, p_n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mp-glob", default="results/experiments/d1_bce_mp_s*_val/predictions.json")
    ap.add_argument("--nomp-glob", default="results/experiments/d1_bce_nomp_s*_val/predictions.json")
    ap.add_argument("--out", default="results/mp_keep_drop_decision.json")
    a = ap.parse_args()

    per_seed, pooled_y, pooled_m, pooled_n = {}, [], [], []
    for seed in (42, 43, 44):
        y, pm, pn = _load_pair(a.mp_glob, a.nomp_glob, seed)
        res = paired_bootstrap(y, pm, pn, n_boot=N_BOOT, seed=BOOT_SEED)
        per_seed[seed] = res
        pooled_y.append(y)
        pooled_m.append(pm)
        pooled_n.append(pn)
        ci_excl = (res["ci_low"] > 0) or (res["ci_high"] < 0)
        print(f"seed {seed}: dF1={res['delta']:+.4f} "
              f"95%CI=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] "
              f"p={res['p_value']:.4f} CI-excl-0={ci_excl}")

    pooled = paired_bootstrap(np.vstack(pooled_y), np.vstack(pooled_m),
                              np.vstack(pooled_n), n_boot=N_BOOT,
                              seed=BOOT_SEED)
    ci_excl_n = sum(1 for r in per_seed.values()
                    if r["ci_low"] > 0 or r["ci_high"] < 0)
    decision = "KEEP" if (ci_excl_n >= 2 and pooled["delta"] >= POOLED_MIN_DELTA) else "DROP"
    print(f"\npooled: dF1={pooled['delta']:+.4f} "
          f"95%CI=[{pooled['ci_low']:+.4f}, {pooled['ci_high']:+.4f}] "
          f"p={pooled['p_value']:.4f}")
    print(f"seeds with CI excluding 0: {ci_excl_n}/3")
    print(f"\n*** PREREGISTERED DECISION: {decision} MP "
          f"(rule: >=2/3 CI-excl-0 AND pooled dF1 >= +1pp; "
          f"inconclusive -> DROP) ***")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "rule": "docs/ablation_protocol.md §4 (frozen 2026-09-21)",
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "pooled": pooled,
        "seeds_ci_excl_zero": ci_excl_n,
        "decision": decision,
    }, indent=2))
    print(f"decision written to {a.out}")


if __name__ == "__main__":
    main()
