"""Export anonymized, non-image-derived review artifacts.

Replays the frozen veto eval per seed and dumps:
  results/released/veto_decisions_s<seed>.jsonl  -- per-decision veto log (verdict,
          raw evidence, quantile, gate, correctness) == the auditable record
          the paper claims (Section 5.3).
  results/released/contradicted_decisions_s<seed>.jsonl -- all contradicted
          claims (fired and withheld), the veto candidates the gate saw.
  results/released/consistent_decisions_s<seed>.jsonl -- all consistent
          verdicts on claimed classes, the checkers' cleared-claim record
          whose precision the paper measures (Section 5.3, consistent side).
  results/released/perframe_primary_s<seed>.json -- per-frame primary correctness for
          the VLM-only and post-veto systems, letting third parties recompute
          the pooled / frame-level McNemar tests and accuracy deltas.

No image data, no features, no annotations beyond the per-frame primary
ground-truth string and correctness booleans.

Usage: PYTHONPATH=. python scripts/export_review_artifacts.py
Validates reconstructed accuracies against the frozen metrics.json before
writing; aborts a seed on mismatch.
"""

import argparse
import json
from pathlib import Path

from scripts.eval_veto import _evidence_from_record  # noqa: E402
from sage.veto import apply_veto, CODE_TO_NAME, ECDFStore  # noqa: E402

SEEDS = {"42": "", "43": "_s43", "44": "_s44"}


def export_seed(seed: str, suffix: str, out_dir: Path) -> dict:
    base = Path(f"results/veto_eval{suffix}")
    recs = [json.loads(l) for l in (base / "test.jsonl").read_text().splitlines() if l.strip()]
    ecdf = ECDFStore.load(base / "ecdf.json")
    tau = json.loads((base / "tau.json").read_text())["tau"]
    floors = {k: v["raw_floor"] for k, v in
              json.loads((base / "thresholds.json").read_text()).items()}

    rows, decisions = [], []
    for r in recs:
        if r.get("error"):
            continue
        o = apply_veto(r["probs"], r["primary_code"],
                       _evidence_from_record(r), tau=tau, ecdf=ecdf,
                       raw_floors=floors,
                       gt_labels=r["gt_labels"], gt_primary=r["gt_primary"])
        vlm_primary = CODE_TO_NAME.get(r["primary_code"], r["primary_code"])
        rows.append({
            "sample_id": r["sample_id"],
            "vlm_primary": vlm_primary,
            "veto_primary": o.final_name,
            "gt_primary": r["gt_primary"],
            "vlm_correct": vlm_primary == r["gt_primary"],
            "veto_correct": o.final_name == r["gt_primary"],
            "primary_demoted": o.primary_demoted,
            "demotion_outcome": o.demotion_outcome,
        })
        for v in o.vetoed:
            decisions.append({
                "sample_id": r["sample_id"], "class": v.name,
                "verdict": v.verdict, "raw_evidence": v.raw_evidence,
                "tilde_e": v.tilde_e, "was_primary": v.was_primary,
                "correct": v.correct, "gate": v.gate,
            })

    frozen = json.loads((base / "metrics.json").read_text())["end_to_end"]
    n = len(rows)
    vlm_acc = sum(x["vlm_correct"] for x in rows) / n
    veto_acc = sum(x["veto_correct"] for x in rows) / n
    ok = (n == frozen["n"]
          and abs(vlm_acc - frozen["vlm_primary_acc"]) < 1e-9
          and abs(veto_acc - frozen["veto_primary_acc"]) < 1e-9)
    if not ok:
        raise SystemExit(f"seed {seed}: reconstruction mismatch "
                         f"({vlm_acc:.4f}/{veto_acc:.4f} vs frozen "
                         f"{frozen['vlm_primary_acc']:.4f}/{frozen['veto_primary_acc']:.4f})")

    (out_dir / f"veto_decisions_s{seed}.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions) + "\n")
    (out_dir / f"perframe_primary_s{seed}.json").write_text(
        json.dumps({"seed": int(seed), "tau": tau, "raw_floors": floors,
                    "n_frames": n, "vlm_primary_acc": vlm_acc,
                    "veto_primary_acc": veto_acc, "frames": rows}, indent=1))
    return {"seed": seed, "frames": n, "decisions": len(decisions),
            "vlm_acc": round(vlm_acc, 4), "veto_acc": round(veto_acc, 4)}


def export_contradicted(seed: str, suffix: str, out_dir: Path) -> dict:
    """All contradicted *claims* per seed, fired and withheld (round-9 N3a).

    Restricted to classes the VLM claimed (the veto layer's scope); the
    collect stage runs every strategy on every frame, so unclaimed-class
    verdicts are outside the mechanism's decision surface.
    """
    base = Path(f"results/veto_eval{suffix}")
    floors = {k: v["raw_floor"] for k, v in
              json.loads((base / "thresholds.json").read_text()).items()}
    rows = []
    for line in (base / "test.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        claimed = set(r.get("claims", []))
        for name, ev in r.get("evidence", {}).items():
            if ev.get("verdict") != "contradicted" or name not in claimed:
                continue
            raw = ev.get("raw_evidence")
            if name in floors and raw is not None:
                gate, vetoed = "frozen_threshold", bool(raw >= floors[name])
            else:
                gate, vetoed = "quantile_tau", False
            rows.append({"sample_id": r["sample_id"], "class": name,
                         "verdict": "contradicted", "raw_evidence": raw,
                         "gate": gate, "vetoed": vetoed,
                         "claim_is_true": bool(r["gt_labels"].get(name, False))})
    path = out_dir / f"contradicted_decisions_s{seed}.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    from collections import Counter
    vt = Counter((x["vetoed"], x["claim_is_true"]) for x in rows)
    return {"seed": seed, "contradicted": len(rows),
            "by_gate": dict(Counter(x["gate"] for x in rows)),
            "vetoed_true_false": {f"vetoed={k[0]}/true={k[1]}": v
                                  for k, v in vt.items()}}


def export_consistent(seed: str, suffix: str, out_dir: Path) -> dict:
    """All consistent *claims* per seed (the checkers' cleared-claim record).

    Same filter discipline as export_contradicted: restricted to classes the
    VLM claimed.  Consistent verdicts never reach the authority gate, so the
    rows carry no gate/vetoed fields; claim_is_true is the consistent-side
    correctness whose pooled precision the paper reports.
    """
    base = Path(f"results/veto_eval{suffix}")
    rows = []
    for line in (base / "test.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        claimed = set(r.get("claims", []))
        primary = CODE_TO_NAME.get(r["primary_code"], r["primary_code"])
        for name, ev in r.get("evidence", {}).items():
            if ev.get("verdict") != "consistent" or name not in claimed:
                continue
            rows.append({"sample_id": r["sample_id"], "class": name,
                         "verdict": "consistent",
                         "raw_evidence": ev.get("raw_evidence"),
                         "was_primary": name == primary,
                         "claim_is_true": bool(r["gt_labels"].get(name, False))})
    path = out_dir / f"consistent_decisions_s{seed}.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    n_true = sum(x["claim_is_true"] for x in rows)
    return {"seed": seed, "consistent": len(rows),
            "true": n_true, "precision": round(n_true / len(rows), 4) if rows else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/released")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed, suffix in SEEDS.items():
        print(f"[export] seed {seed}:", json.dumps(export_seed(seed, suffix, out)))
        print(f"[export] contradicted {seed}:", json.dumps(
            export_contradicted(seed, suffix, out)))
        print(f"[export] consistent {seed}:", json.dumps(
            export_consistent(seed, suffix, out)))
    e4 = json.loads(Path("results/e4_fusion_comparison.json").read_text())
    (out / "e4_perframe_s42.json").write_text(json.dumps(
        {"n": e4["n"],
         "accuracies": {k: e4[k]["primary_acc"]
                        for k in ("none", "veto", "fusion", "fusion_ro")},
         "mcnemar": {k: e4[k] for k in e4 if k.startswith("mcnemar")},
         "per_frame": e4["per_frame"]}, indent=1))
    print("[export] e4 per-frame written")


if __name__ == "__main__":
    main()
