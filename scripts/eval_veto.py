"""Claim-conditioned veto evaluation — collect / fit / eval stages.

Implements the preregistered two-stage protocol (docs/veto-algorithm.md §6):

  collect  run VLM + claim-conditioned strategies on a split; dump per-frame
           claims and evidence records (requires detection services up)
  fit      fit per-strategy validation ECDFs, sweep tau over the frozen
           grid, freeze tau at the largest value with FVR <= delta (0.05)
  freeze   (v5) freeze per-strategy raw floors for the interpretable-angle
           strategies (slope/tilt) on validation CLAIMS -- requires records
           with VLM probabilities; same FVR <= delta rule as the tau freeze
  eval     single frozen-tau pass: VP / FVR / TPK / coverage overall and by
           semantic-/geometric-cued group, per-veto case records, summary

Usage:
  python scripts/eval_veto.py --stage collect --gt data/sage_eval/annotations.json --split val --out results/veto_eval/val.jsonl
  python scripts/eval_veto.py --stage fit     --records results/veto_eval/val.jsonl --out-dir results/veto_eval/
  python scripts/eval_veto.py --stage eval    --records results/veto_eval/test.jsonl --ecdf results/veto_eval/ecdf.json --tau-from results/veto_eval/tau.json --out-dir results/veto_eval/

A single union parser (no subcommands) so the experiments registry can drive
every stage through one `--stage <name>` flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sage.taxonomy import BEHAVIOR_CODES, CODE_TO_NAME, NORMAL_CODE  # noqa: E402
from sage.veto import (  # noqa: E402
    DELTA,
    TAU_GRID,
    ECDFStore,
    aggregate_metrics,
    apply_veto,
    metrics_by_group,
)

RECORD_VERSION = 2  # evidence-record schema (claim-conditioned vocabulary)

# v5 frozen-threshold strategies: raw_evidence = negated angle in degrees,
# directly interpretable, with the contradiction region (a level head)
# being the MAJORITY region of the all-frames pool -- the quantile gate
# cannot discriminate there, so their veto authority is a per-strategy
# raw floor frozen on validation claims instead.
ANGLE_VETO_STRATEGIES = ("slope", "tilt")
ANGLE_GRID_DEG = list(range(0, 46))  # candidate veto ceilings, 0..45 deg


def freeze_angle_thresholds(recs: list[dict],
                            strategies: tuple[str, ...] = ANGLE_VETO_STRATEGIES,
                            threshold: float = 0.5) -> dict:
    """Freeze per-strategy raw floors on validation claims (v5).

    Preregistered rule (mirrors the tau freeze, docs/veto-algorithm.md §4):
    for strategy s, consider validation frames where the VLM claimed s
    (prob >= threshold) and the evidence verdict is 'contradicted' with a
    numeric raw; angle = -raw.  max_angle T* = largest angle on the grid
    such that vetoing every claimed-and-contradicted frame with
    angle <= T* has FVR <= DELTA on those validation claims.  The raw
    floor is -T* (contradiction-oriented raw = negated angle).  Strategies
    with no qualifying T* get no threshold authority.

    Requires records WITH VLM probabilities (the claims define the pool);
    evidence-only validation records cannot support freezing.
    """
    from sage.taxonomy import NAME_TO_CODE

    out: dict = {}
    for cls in strategies:
        code = NAME_TO_CODE.get(cls, cls)
        pool: list[tuple[float, bool]] = []  # (angle, veto_would_be_correct)
        for r in recs:
            p = (r.get("probs") or {}).get(code)
            if p is None or p < threshold:
                continue
            ev = (r.get("evidence") or {}).get(cls) or {}
            raw = ev.get("raw_evidence")
            if ev.get("verdict") != "contradicted" or raw is None:
                continue
            angle = -raw
            pool.append((angle, not r["gt_labels"].get(cls, False)))
        if not pool:
            continue
        best = None
        for T in ANGLE_GRID_DEG:
            v = [ok for ang, ok in pool if ang <= T]
            if not v:
                continue
            fvr = 1.0 - sum(v) / len(v)
            if fvr <= DELTA:
                best = (T, fvr, len(v))
        if best is None:
            continue
        T, fvr, n_v = best
        out[cls] = {"max_angle": T, "raw_floor": -float(T),
                    "n_pool": len(pool), "n_vetoes_at_T": n_v,
                    "fvr_at_T": fvr, "delta": DELTA}
    return out


def cmd_freeze(a: argparse.Namespace) -> None:
    recs = _load_records(a.records)
    if not any(r.get("probs") for r in recs):
        raise SystemExit(
            "[freeze] validation records carry no VLM probabilities -- "
            "evidence-only records cannot define the claim pool. Re-run "
            "collect with the VLM server (or --predictions).")
    thresholds = freeze_angle_thresholds(recs, threshold=a.threshold)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    if not thresholds:
        print("[freeze] no strategy qualified for threshold authority")
    for cls, spec in thresholds.items():
        print(f"[freeze] {cls}: max_angle={spec['max_angle']}deg "
              f"(raw_floor={spec['raw_floor']}), vetoes={spec['n_vetoes_at_T']}"
              f"/{spec['n_pool']} claimed-contradicted, FVR={spec['fvr_at_T']:.3f}")
    return {"stage": "freeze", "records": a.records, **thresholds}


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(a: argparse.Namespace) -> None:
    import os

    from sage_check.client import VLMClient
    from sage_recheck.detection.runner import DetectionRunner
    from sage_recheck.strategies.runner import StrategyRunner

    from sage.eval_harness import encode_image, load_ground_truth

    if a.vlm_url:
        os.environ["SAGE_VLM_URL"] = a.vlm_url
    client = None
    preds = {}
    if a.predictions:
        # Deterministic CPU path (spec §6 --predictions input): reuse a
        # recorded eval_vlm_check predictions.json instead of calling the
        # VLM server.  probs are name-keyed there; records keep the
        # server's code-keyed convention.
        from sage.taxonomy import NAME_TO_CODE
        for r in json.loads(Path(a.predictions).read_text()):
            preds[r["sample_id"]] = {
                "probs": {NAME_TO_CODE[n]: float(p)
                          for n, p in (r.get("probs") or {}).items()},
                "primary_code": NAME_TO_CODE.get(
                    r.get("primary_label", ""), ""),
            }
        print(f"[collect] using recorded predictions: {a.predictions} "
              f"({len(preds)} samples)")
    elif os.environ.get("SAGE_VLM_URL") or a.vlm_url:
        client = VLMClient.from_env()
    else:
        # Evidence-only collection (no VLM output): valid for the fit
        # stage, whose ECDF pools depend only on raw_evidence.  probs
        # stay empty; apply_veto on such records would see no claims.
        print("[collect] evidence-only mode: no VLM server, no predictions "
              "-> probs/primary left empty (fit-stage calibration pool)")
    det_runner = DetectionRunner.from_env()
    strategies = StrategyRunner()

    gt = load_ground_truth(a.gt, a.split)
    if a.predictions:
        gt = [s for s in gt if s["sample_id"] in preds]
    print(f"[collect] {len(gt)} samples, split={a.split or 'all'}")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_err = 0

    def process(sample: dict) -> dict:
        try:
            img_b64 = encode_image(sample["image_path"])
            if client is not None:
                resp = client.detect(img_b64, a.mode)
                probs = resp.get("probabilities", {})
                primary = resp.get("primary_code", "")
                err = resp.get("error")
            else:
                rec_p = preds.get(sample["sample_id"], {})
                probs = rec_p.get("probs", {})
                primary = rec_p.get("primary_code", "")
                err = None
            claims = [c for c in BEHAVIOR_CODES
                      if c != NORMAL_CODE and probs.get(c, 0.0) >= a.threshold]
            claim_names = [CODE_TO_NAME[c] for c in claims]
            context = det_runner.run(img_b64)
            # Run ALL strategies (not just the claimed ones): the fit
            # stage needs each strategy's raw_evidence over the full
            # validation split to clear the n_min=30 calibration floor.
            # apply_veto consumes only the claimed classes' entries.
            evidence = strategies.run(context)
            return {
                "schema": RECORD_VERSION,
                "sample_id": sample["sample_id"],
                "gt_labels": {k: bool(v) for k, v in sample["labels"].items()},
                "gt_primary": sample.get("primary_label", ""),
                "probs": probs,
                "primary_code": primary,
                "claims": claim_names,
                "evidence": {
                    name: {
                        "verdict": r.verdict.value,
                        "raw_evidence": r.raw_evidence,
                        "evidence": r.evidence,
                    } for name, r in evidence.items()
                },
                "error": err,
            }
        except Exception as e:  # keep collection going; record the failure
            return {"schema": RECORD_VERSION,
                    "sample_id": sample["sample_id"],
                    "error": str(e)}

    with open(out_path, "w") as f:
        if a.workers <= 1:
            for i, sample in enumerate(gt):
                rec = process(sample)
                if rec.get("error"):
                    n_err += 1
                f.write(json.dumps(rec) + "\n")
                if (i + 1) % 50 == 0:
                    print(f"[collect] {i + 1}/{len(gt)}", flush=True)
        else:
            # Order-preserving thread pool (same pattern as run_eval): the
            # per-frame work is HTTP-bound (VLM + 4 detection services), so
            # threads overlap the latency while records keep sample order.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            records: list[dict | None] = [None] * len(gt)
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                future_to_idx = {pool.submit(process, s): i
                                 for i, s in enumerate(gt)}
                done = 0
                for future in as_completed(future_to_idx):
                    records[future_to_idx[future]] = future.result()
                    done += 1
                    if done % 50 == 0 or done == len(gt):
                        print(f"[collect] {done}/{len(gt)}", flush=True)
            for rec in records:
                if rec is None:  # unreachable: future.result() re-raises
                    raise RuntimeError("missing record")
                if rec.get("error"):
                    n_err += 1
                f.write(json.dumps(rec) + "\n")
    print(f"[collect] wrote {out_path} ({n_err} errored frames)")
    # Fail fast on a dead service: an all-error (or mostly-error) collection
    # must not exit 0, otherwise the runner keeps a garbage eval_report.json
    # that permanently blocks idempotent re-runs.
    if n_err * 2 >= len(gt):
        print(f"[collect] FATAL: {n_err}/{len(gt)} frames errored — a "
              f"detection service is down or unreachable", file=sys.stderr)
        raise SystemExit(2)
    return {"stage": "collect", "split": a.split, "records": str(out_path),
            "n_samples": len(gt), "n_errored": n_err}


def _load_records(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or r.get("schema") != RECORD_VERSION:
                continue
            recs.append(r)
    return recs


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def cmd_fit(a: argparse.Namespace) -> None:
    recs = _load_records(a.records)
    print(f"[fit] {len(recs)} frames")
    ecdf = ECDFStore.fit([
        {"name": name, "raw_evidence": ev.get("raw_evidence")}
        for r in recs for name, ev in r["evidence"].items()
    ])
    print(f"[fit] calibrated strategies: {sorted(ecdf._sorted)}")
    for name in sorted(ecdf.disabled):
        print(f"[fit] disabled: {name} -- {ecdf.disabled_reasons.get(name, '?')}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ecdf.save(out_dir / "ecdf.json")

    sweep = []
    for tau in TAU_GRID:
        outcomes, gts = [], []
        for r in recs:
            evidence = _evidence_from_record(r)
            gt = r["gt_labels"]
            o = apply_veto(r["probs"], r["primary_code"], evidence,
                           tau=tau, ecdf=ecdf, gt_labels=gt)
            outcomes.append(o)
            gts.append(gt)
        m = aggregate_metrics(outcomes, gts)
        sweep.append({"tau": tau, **m})
        print(f"[fit] tau={tau:.2f} vetoes={m['n_vetoes']:3d} "
              f"VP={_fmt(m['veto_precision'])} FVR={_fmt(m['false_veto_rate'])} "
              f"coverage={_fmt(m['coverage'])}")

    # Freeze rule: largest tau on the grid with FVR <= delta.
    feasible = [s for s in sweep
                if s["false_veto_rate"] is not None
                and s["false_veto_rate"] <= DELTA]
    if not feasible:
        print(f"[fit] WARNING: no tau on the grid achieves FVR <= {DELTA}; "
              f"freezing tau={TAU_GRID[-1]} (no vetoes pass gating)")
        frozen = {"tau": TAU_GRID[-1], "delta": DELTA, "feasible": False}
    else:
        best = max(feasible, key=lambda s: s["tau"])
        print(f"[fit] frozen tau={best['tau']:.2f} (FVR={best['false_veto_rate']:.4f})")
        frozen = {"tau": best["tau"], "delta": DELTA, "feasible": True}
    (out_dir / "tau.json").write_text(json.dumps(frozen, indent=2))
    (out_dir / "tau_sweep.json").write_text(json.dumps(sweep, indent=2))
    return {"stage": "fit", "records": a.records, **frozen}


def _fmt(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "  n/a"


def _evidence_from_record(r: dict) -> dict:
    """Rebuild name-keyed EvidenceResult objects from a collected record."""
    from sage_recheck.enums import EvidenceResult, EvidenceVerdict

    return {
        name: EvidenceResult(
            verdict=EvidenceVerdict(ev["verdict"]),
            evidence=ev.get("evidence", ""),
            raw_evidence=ev.get("raw_evidence"),
        ) for name, ev in r["evidence"].items()
    }


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(a: argparse.Namespace) -> None:
    recs = _load_records(a.records)
    ecdf = ECDFStore.load(a.ecdf)
    tau = json.loads(Path(a.tau_from).read_text())["tau"]
    raw_floors = None
    if a.thresholds:
        raw_floors = {name: spec["raw_floor"] for name, spec
                      in json.loads(Path(a.thresholds).read_text()).items()}
        print(f"[eval] frozen raw floors (v5): {raw_floors}")
    print(f"[eval] {len(recs)} frames, frozen tau={tau}")

    outcomes, gts = [], []
    for r in recs:
        o = apply_veto(r["probs"], r["primary_code"],
                       _evidence_from_record(r), tau=tau, ecdf=ecdf,
                       raw_floors=raw_floors,
                       gt_labels=r["gt_labels"], gt_primary=r["gt_primary"])
        outcomes.append(o)
        gts.append(r["gt_labels"])

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "tau": tau,
        "delta": DELTA,
        "overall": aggregate_metrics(outcomes, gts),
        "by_group": metrics_by_group(outcomes, gts),
        "end_to_end": _end_to_end(recs, outcomes, threshold=a.threshold),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    with open(out_dir / "cases.jsonl", "w") as f:
        for r, o in zip(recs, outcomes):
            for v in o.vetoed:
                f.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "class": v.name,
                    "verdict": v.verdict,
                    "raw_evidence": v.raw_evidence,
                    "tilde_e": v.tilde_e,
                    "was_primary": v.was_primary,
                    "correct": v.correct,
                    "gate": v.gate,
                }) + "\n")

    _write_summary(report, out_dir / "summary.md")
    _print_report(report)
    return report


def _end_to_end(recs: list[dict], outcomes: list, threshold: float) -> dict:
    """VLM-only vs post-veto primary accuracy on the same frames."""
    vlm_hit = veto_hit = n = 0
    for r, o in zip(recs, outcomes):
        if r.get("error"):
            continue
        n += 1
        gt_primary = r["gt_primary"]
        vlm_primary = CODE_TO_NAME.get(r["primary_code"], r["primary_code"])
        vlm_hit += int(vlm_primary == gt_primary)
        veto_hit += int(o.final_name == gt_primary)
    return {"n": n, "vlm_primary_acc": vlm_hit / n if n else None,
            "veto_primary_acc": veto_hit / n if n else None}


def _write_summary(report: dict, path: Path) -> None:
    lines = ["# Veto evaluation summary", "",
             f"- frozen tau = {report['tau']} (delta = {report['delta']})", "",
             "| group | VP | FVR | TPK | coverage | vetoes | demotions |",
             "|---|---|---|---|---|---|---|"]
    for label, m in [("overall", report["overall"]),
                     ("semantic", report["by_group"]["semantic"]),
                     ("geometric", report["by_group"]["geometric"])]:
        lines.append(
            f"| {label} | {_fmt(m['veto_precision'])} | {_fmt(m['false_veto_rate'])} "
            f"| {_fmt(m['true_positive_kill_rate'])} | {_fmt(m['coverage'])} "
            f"| {m['n_vetoes']} | {m['n_primary_demotions']} |")
    e2e = report["end_to_end"]
    lines += ["", f"- VLM-only primary acc: {_fmt(e2e['vlm_primary_acc'])}",
              f"- post-veto primary acc: {_fmt(e2e['veto_primary_acc'])}"]
    path.write_text("\n".join(lines) + "\n")


def _print_report(report: dict) -> None:
    for label, m in [("overall", report["overall"])] + \
            list(report["by_group"].items()):
        print(f"[eval] {label:9s} VP={_fmt(m['veto_precision'])} "
              f"FVR={_fmt(m['false_veto_rate'])} TPK={_fmt(m['true_positive_kill_rate'])} "
              f"coverage={_fmt(m['coverage'])} vetoes={m['n_vetoes']}")
    print(f"[eval] end-to-end: {_print_e2e(report['end_to_end'])}")


def _print_e2e(e2e: dict) -> str:
    return (f"vlm={_fmt(e2e['vlm_primary_acc'])} "
            f"veto={_fmt(e2e['veto_primary_acc'])} (n={e2e['n']})")


def main() -> None:
    # Single union parser: every flag is defined here, --stage selects the
    # command. This keeps the experiments registry able to merge its common
    # eval args (gt/split/...) into any stage's command line.
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["collect", "fit", "freeze", "eval"],
                   required=True)
    # collect
    p.add_argument("--gt", default="data/sage_eval/annotations.json")
    p.add_argument("--split", default="")
    p.add_argument("--out", default="")
    p.add_argument("--vlm-url", default="")
    p.add_argument("--predictions", default="",
                   help="recorded eval_vlm_check predictions.json; when set, "
                        "the VLM server is not called (deterministic CPU path, "
                        "spec §6 --predictions input)")
    p.add_argument("--mode", choices=["no_reason", "reason"], default="no_reason")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=1,
                   help="collect-stage parallel workers (1 = sequential); "
                        "per-frame work is HTTP-bound (VLM + detection "
                        "services), so 4 overlaps the latency")
    # fit
    p.add_argument("--records", default="")
    p.add_argument("--out-dir", default="")
    # eval
    p.add_argument("--ecdf", default="")
    p.add_argument("--tau-from", default="")
    p.add_argument("--thresholds", default="",
                   help="optional thresholds.json (from --stage freeze) with "
                        "per-strategy raw floors; strategies listed there "
                        "gate on the frozen floor instead of the quantile")
    # runner-managed: where eval_report.json (idempotence marker) is written
    p.add_argument("--output", default="",
                   help="experiment results dir; a stage report is written "
                        "to eval_report.json here when given")
    a = p.parse_args()

    if a.stage == "collect" and not a.out:
        p.error("--stage collect requires --out")
    if a.stage in ("fit", "freeze", "eval") and not a.records:
        p.error(f"--stage {a.stage} requires --records")
    if a.stage in ("fit", "freeze", "eval") and not a.out_dir:
        p.error(f"--stage {a.stage} requires --out-dir")
    if a.stage == "eval" and (not a.ecdf or not a.tau_from):
        p.error("--stage eval requires --ecdf and --tau-from")

    report = {"collect": cmd_collect, "fit": cmd_fit, "freeze": cmd_freeze,
              "eval": cmd_eval}[a.stage](a)
    if a.output:
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_report.json").write_text(
            json.dumps({"script": "scripts/eval_veto.py", **report},
                       ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
