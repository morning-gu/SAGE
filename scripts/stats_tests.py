#!/usr/bin/env python3
"""Paper-grade statistical tests across SAGE experiment directories.

One entry point producing markdown tables for:
  - mcnemar      exact McNemar's test on per-sample Primary Acc (hit)
  - bootstrap    paired bootstrap test on macro-F1 (binarized at 0.5)
  - seed-summary mean +/- std of headline metrics across seed runs

Usage:
  python scripts/stats_tests.py --dir results/experiments \
    --pairs d1_bce_mp_s42:d1_bce_nomp_s42 d1_bce_mp_pooled:exthead_ce-lora_s42 \
    --tests mcnemar,bootstrap,seed-summary --n-boot 10000 --seed 42

Experiment directories contain predictions.json (+ optional eval_report.json).
``*_pooled`` pseudo-ids are supported: if `d1_bce_mp_pooled` has no directory,
the predictions of `d1_bce_mp_s<seed>` for every seed present on BOTH sides
of the pair are aligned per seed and stacked, and the tests run on the
stacked matrices.

Pure CPU (numpy/scipy only).  Markdown output goes to stdout and to
`stats_report.md` written next to --dir (override with --out).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
from sage.metrics import (  # noqa: E402
    align_predictions,
    compute_ece,
    compute_map,
    load_predictions,
    macro_f1,
    mcnemar_exact,
    paired_bootstrap,
    primary_hit,
)

HEADLINE_FIELDS = ("macro_f1", "primary_accuracy", "mAP", "ECE")
FIELD_LABELS = {
    "macro_f1": "Macro-F1",
    "primary_accuracy": "Primary Acc(hit)",
    "mAP": "mAP",
    "ECE": "ECE",
}


# -- Experiment loading -------------------------------------------------------

def load_exp(base_dir: Path, exp_id: str):
    """Load one experiment's predictions. Returns (ids, gt, probs)."""
    pred_path = base_dir / exp_id / "predictions.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.json not found: {pred_path}")
    return load_predictions(pred_path)


def load_report(base_dir: Path, exp_id: str) -> dict | None:
    rep_path = base_dir / exp_id / "eval_report.json"
    if not rep_path.exists():
        return None
    with open(rep_path, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_from_predictions(ids, gt, probs) -> dict:
    """Fallback headline metrics computed from raw predictions."""
    preds = (probs >= 0.5).astype(np.float32)
    primary_idx = np.argmax(probs, axis=1)
    return {
        "macro_f1": float(macro_f1(gt, preds)),
        "primary_accuracy": float(np.mean(primary_hit(gt, primary_idx))),
        "mAP": float(compute_map(gt, probs)),
        "ECE": float(compute_ece(probs, gt)),
    }


def headline_metrics(base_dir: Path, exp_id: str) -> dict:
    """Headline metrics for one experiment: eval_report.json fields when
    present, otherwise computed from predictions.json."""
    report = load_report(base_dir, exp_id)
    if report is not None:
        ov = report.get("overall", {})
        vals = {f: ov.get(f) for f in HEADLINE_FIELDS}
        if all(v is not None for v in vals.values()):
            return {f: float(v) for f, v in vals.items()}
        computed = metrics_from_predictions(*load_exp(base_dir, exp_id))
        return {f: float(vals[f]) if vals[f] is not None else computed[f]
                for f in HEADLINE_FIELDS}
    return metrics_from_predictions(*load_exp(base_dir, exp_id))


# -- Pair loading (with *_pooled support) -------------------------------------

def expand_side(base_dir: Path, side: str) -> dict:
    """Resolve one side of a pair to {seed_or_None: (ids, gt, probs)}.

    - Normal exp id: single entry keyed None.
    - `<base>_pooled` with no directory: every seed run `<base>_s<seed>`
      found in base_dir, keyed by int seed.
    """
    if (base_dir / side).is_dir():
        return {None: load_exp(base_dir, side)}
    if side.endswith("_pooled"):
        base = side[: -len("_pooled")]
        pat = re.compile(rf"^{re.escape(base)}_s(\d+)$")
        out = {}
        for d in sorted(base_dir.iterdir()):
            m = pat.match(d.name)
            if m and (d / "predictions.json").exists():
                out[int(m.group(1))] = load_exp(base_dir, d.name)
        if out:
            return out
        raise FileNotFoundError(
            f"No pooled source runs found for '{side}' in {base_dir} "
            f"(looked for {base}_s<seed>/predictions.json)")
    raise FileNotFoundError(
        f"Experiment directory not found for '{side}' in {base_dir}")


def load_pair(base_dir: Path, side_a: str, side_b: str):
    """Load + row-align both sides of a pair.

    For pooled sides, each seed present on both sides is aligned
    independently (align_predictions) and the aligned blocks are stacked,
    so row order is always valid for paired tests.

    Returns (gt, probs_a, probs_b, seeds_used).
    """
    ea = expand_side(base_dir, side_a)
    eb = expand_side(base_dir, side_b)
    if None in ea and None in eb:
        seeds = [None]
    elif None in ea:
        seeds = sorted(k for k in eb if k is not None)
    elif None in eb:
        seeds = sorted(k for k in ea if k is not None)
    else:
        seeds = sorted(set(ea) & set(eb))
    if not seeds:
        raise ValueError(
            f"No common seeds between '{side_a}' and '{side_b}': "
            f"{sorted(k for k in ea if k is not None)} vs "
            f"{sorted(k for k in eb if k is not None)}")

    gts, pas, pbs = [], [], []
    for s in seeds:
        ids_a, gt_a, probs_a = ea[s]
        ids_b, _, probs_b = eb[s]
        gt_al, pa_al, pb_al = align_predictions(
            ids_a, gt_a, probs_a, ids_b, probs_b)
        gts.append(gt_al)
        pas.append(pa_al)
        pbs.append(pb_al)
    gt = np.concatenate(gts, axis=0)
    probs_a = np.concatenate(pas, axis=0)
    probs_b = np.concatenate(pbs, axis=0)
    seed_label = ",".join(str(s) for s in seeds) if seeds[0] is not None else "single"
    return gt, probs_a, probs_b, seed_label


# -- Tests --------------------------------------------------------------------

def run_mcnemar(base_dir: Path, side_a: str, side_b: str) -> dict:
    gt, probs_a, probs_b, seeds = load_pair(base_dir, side_a, side_b)
    hits_a = primary_hit(gt, np.argmax(probs_a, axis=1))
    hits_b = primary_hit(gt, np.argmax(probs_b, axis=1))
    n01 = int(np.sum(~hits_a & hits_b))   # A wrong, B correct
    n10 = int(np.sum(hits_a & ~hits_b))   # A correct, B wrong
    res = mcnemar_exact(n01, n10)
    res.update({
        "acc_a": float(np.mean(hits_a)),
        "acc_b": float(np.mean(hits_b)),
        "n": int(len(gt)),
        "seeds": seeds,
    })
    return res


def run_bootstrap(base_dir: Path, side_a: str, side_b: str,
                  n_boot: int, seed: int) -> dict:
    gt, probs_a, probs_b, seeds = load_pair(base_dir, side_a, side_b)
    preds_a = (probs_a >= 0.5).astype(np.float32)
    preds_b = (probs_b >= 0.5).astype(np.float32)
    res = paired_bootstrap(gt, preds_a, preds_b, n_boot=n_boot, seed=seed)
    res.update({
        "n": int(len(gt)),
        "seeds": seeds,
        "macro_f1_a": float(macro_f1(gt, preds_a)),
        "macro_f1_b": float(macro_f1(gt, preds_b)),
    })
    return res


def seed_summary(base_dir: Path) -> list[dict]:
    """Group `*_s<seed>` experiment dirs by base id; mean +/- std per metric."""
    pat = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")
    groups: dict[str, list[tuple[int, str]]] = {}
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {base_dir}")
    for d in sorted(base_dir.iterdir()):
        m = pat.match(d.name)
        if m and (d / "predictions.json").exists():
            groups.setdefault(m.group("base"), []).append(
                (int(m.group("seed")), d.name))
    rows = []
    for base in sorted(groups):
        runs = sorted(groups[base])
        per_seed = {f: [] for f in HEADLINE_FIELDS}
        for _, exp_id in runs:
            met = headline_metrics(base_dir, exp_id)
            for f in HEADLINE_FIELDS:
                per_seed[f].append(met[f])
        row = {
            "exp": base,
            "seeds": [s for s, _ in runs],
            "n": len(runs),
        }
        for f in HEADLINE_FIELDS:
            arr = np.array(per_seed[f], dtype=np.float64)
            row[f] = (float(arr.mean()), float(arr.std()))
        rows.append(row)
    return rows


# -- Markdown rendering -------------------------------------------------------

def fmt_p(p: float) -> str:
    if p < 0.0001:
        return "{:.2e}".format(p)
    return "{:.4f}".format(p)


def render_mcnemar(results: list[dict]) -> list[str]:
    lines = [
        "## McNemar's exact test (Primary Acc, hit definition)",
        "",
        "| Pair | n | Primary Acc A | Primary Acc B | n01 (A-wrong/B-correct) "
        "| n10 (A-correct/B-wrong) | n_disc | method | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        p = r.get("p_exact") if "exact" in r.get("method", "") else r.get("p_chi2")
        if p is None:
            p = min(r.get("p_exact", 1.0), r.get("p_chi2", 1.0))
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {} | {} | {} | {} | {} |".format(
                r["pair"], r["n"], r["acc_a"], r["acc_b"],
                r["n01"], r["n10"], r["n_disc"], r["method"], fmt_p(p)))
    lines.append("")
    lines.append("n01/n10 are the two discordant counts; the exact binomial "
                 "test is used for n_disc < 25, chi-square with continuity "
                 "correction otherwise (sage.metrics.mcnemar_exact).")
    lines.append("")
    return lines


def render_bootstrap(results: list[dict]) -> list[str]:
    lines = [
        "## Paired bootstrap test (macro-F1, threshold 0.5, delta = A - B)",
        "",
        "| Pair | n | Macro-F1 A | Macro-F1 B | delta | 95% CI | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:+.4f} | [{:+.4f}, {:+.4f}] | {} |".format(
                r["pair"], r["n"], r["macro_f1_a"], r["macro_f1_b"],
                r["delta"], r["ci_low"], r["ci_high"], fmt_p(r["p_value"])))
    lines.append("")
    lines.append("Empirical two-sided p-value from {} resamples "
                 "(seed={}); CI is the 2.5/97.5 percentile of the resampled "
                 "delta distribution.".format(
                     results[0]["n_boot"] if results else 0,
                     results[0]["seed"] if results else 0))
    lines.append("")
    return lines


def render_seed_summary(rows: list[dict]) -> list[str]:
    lines = [
        "## Seed summary (mean +/- std across seeds)",
        "",
        "| Exp | seeds | " + " | ".join(FIELD_LABELS[f] for f in HEADLINE_FIELDS) + " |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cells = [", ".join(str(s) for s in r["seeds"])]
        for f in HEADLINE_FIELDS:
            mean, std = r[f]
            cells.append("{:.4f} +/- {:.4f}".format(mean, std))
        lines.append("| {} | {} ({}) | {} |".format(
            r["exp"], " | ".join(cells[:1]), r["n"], " | ".join(cells[1:])))
    lines.append("")
    return lines


# -- Main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Paper-grade statistical tests across experiment dirs")
    p.add_argument("--dir", required=True,
                   help="Directory containing experiment subdirectories")
    p.add_argument("--pairs", nargs="*", default=[],
                   help="Pairs as a:b (a, b = exp ids; *_pooled pseudo-ids "
                        "allowed)")
    p.add_argument("--tests", default="mcnemar,bootstrap,seed-summary",
                   help="Comma-separated subset of: mcnemar,bootstrap,"
                        "seed-summary")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="",
                   help="Output markdown path (default: stats_report.md "
                        "next to --dir)")
    args = p.parse_args()

    base_dir = Path(args.dir)
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    for t in tests:
        if t not in ("mcnemar", "bootstrap", "seed-summary"):
            p.error(f"unknown test '{t}'")

    lines = [
        "# SAGE Statistical Tests",
        "",
        "- Results dir: `{}`".format(base_dir),
        "- Tests: {}".format(", ".join(tests)),
        "- Bootstrap: n={}, seed={}".format(args.n_boot, args.seed),
        "",
    ]

    if "mcnemar" in tests:
        results = []
        for pair in args.pairs:
            a, _, b = pair.partition(":")
            if not b:
                print(f"[stats] WARN skipping malformed pair '{pair}' "
                      "(expected a:b)", file=sys.stderr)
                continue
            try:
                r = run_mcnemar(base_dir, a, b)
            except (FileNotFoundError, ValueError) as e:
                print(f"[stats] WARN mcnemar {pair}: {e}", file=sys.stderr)
                continue
            r["pair"] = pair
            results.append(r)
        if results:
            lines += render_mcnemar(results)
        else:
            lines += ["## McNemar's exact test", "", "(no pairs resolved)", ""]

    if "bootstrap" in tests:
        results = []
        for pair in args.pairs:
            a, _, b = pair.partition(":")
            if not b:
                continue
            try:
                r = run_bootstrap(base_dir, a, b, args.n_boot, args.seed)
            except (FileNotFoundError, ValueError) as e:
                print(f"[stats] WARN bootstrap {pair}: {e}", file=sys.stderr)
                continue
            r["pair"] = pair
            results.append(r)
        if results:
            lines += render_bootstrap(results)
        else:
            lines += ["## Paired bootstrap test", "", "(no pairs resolved)", ""]

    if "seed-summary" in tests:
        try:
            rows = seed_summary(base_dir)
        except FileNotFoundError as e:
            print(f"[stats] WARN seed-summary: {e}", file=sys.stderr)
            rows = []
        if rows:
            lines += render_seed_summary(rows)
        else:
            lines += ["## Seed summary", "", "(no `*_s<seed>` runs found)", ""]

    md = "\n".join(lines)
    print(md)

    out_path = Path(args.out) if args.out else base_dir.parent / "stats_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[stats] report saved -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
