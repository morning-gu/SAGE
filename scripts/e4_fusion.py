"""E4: veto vs soft-fusion vs none on the SAME strategy-layer output.

Frozen decision 4 (docs/veto-algorithm.md §7): the soft-fusion baseline
consumes the SAME evidence the veto layer consumes — consistent ->
positive LR, contradicted -> negative LR, insufficient -> 0 (no evidence).

LR weights come from ``checkpoints/fuse_lr_weights.json`` — regenerated from
the current-generation strategy outputs on the validation records (the
same split every other constant is frozen on; no test leakage).  The
pre-2026-09-28 file held train-split weights from the superseded v4
strategy generation and was not reproducible; it has been replaced.

Arms (identical inputs, per-frame):
  none         VLM primary stands
  veto         Algorithm 1 with the frozen tau + raw floors (cmd_eval)
  fusion       log-odds LR adjustment, argmax over adjusted probs
  fusion_ro    removal-only fusion: adjust, then only REMOVE claims whose
               adjusted probability drops below threshold -- the no-add
               contract isolated from the veto mechanism
  fusion_T*    temperature-tuned LR (T selected on validation accuracy)

Outputs results/e4_fusion_comparison.json + a printed table.

Outputs results/e4_fusion_comparison.json + a printed table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sage.taxonomy import BEHAVIOR_CODES, CODE_TO_NAME, NORMAL_CODE
from sage.veto import ECDFStore, apply_veto
from scripts.eval_veto import _evidence_from_record

LAPLACE_ALPHA = 0.5
EPS = 1e-7


def estimate_lr(val_records: list[dict]) -> dict[str, dict[str, float]]:
    """LR(c, v) = log[P(v | gt=c) / P(v | gt!=c)] on val frames.

    Faithful to the LR-weights metadata formula, with Laplace smoothing:
    P(v | gt=c) = (cell.pos + a) / (n_pos_c + 2a) over the n_pos_c val
    frames whose gt contains c (multi-label).  A class with no decisive
    val cells gets LR 0 (no evidence — never invent weight from smoothing).
    """
    n = len(val_records)
    n_pos = {name: sum(1 for r in val_records if r["gt_labels"].get(name))
             for name in {CODE_TO_NAME[c] for c in BEHAVIOR_CODES}}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for r in val_records:
        for name, ev in r["evidence"].items():
            v = ev.get("verdict")
            if v not in ("consistent", "contradicted"):
                continue  # insufficient: no evidence by design
            gt_pos = bool(r["gt_labels"].get(name, False))
            c = counts.setdefault(name, {}).setdefault(
                v, {"pos": 0, "neg": 0})
            c["pos" if gt_pos else "neg"] += 1
    weights: dict[str, dict[str, float]] = {}
    for name, n_pos_c in n_pos.items():
        n_neg_c = n - n_pos_c
        entry: dict[str, float] = {}
        for v in ("consistent", "contradicted"):
            cell = counts.get(name, {}).get(v, {"pos": 0, "neg": 0})
            if cell["pos"] + cell["neg"] == 0:
                entry["confirmed" if v == "consistent" else "rejected"] = 0.0
                continue
            p_pos = (cell["pos"] + LAPLACE_ALPHA) / (n_pos_c + 2 * LAPLACE_ALPHA)
            p_neg = (cell["neg"] + LAPLACE_ALPHA) / (n_neg_c + 2 * LAPLACE_ALPHA)
            entry["confirmed" if v == "consistent" else "rejected"] = round(
                math.log(p_pos / p_neg), 4)
        weights[name] = entry
    return weights


def fuse_primary(rec: dict, engine) -> str:
    from sage_check.result import VLMResult

    vlm = VLMResult(primary_code=rec["primary_code"],
                    probabilities=dict(rec["probs"]))
    final = engine.fuse(vlm, _evidence_from_record(rec))
    return final.final_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-records", default="results/veto_eval/val.jsonl")
    ap.add_argument("--test-records", default="results/veto_eval/test.jsonl")
    ap.add_argument("--veto-dir", default="results/veto_eval")
    ap.add_argument("--out", default="results/e4_fusion_comparison.json")
    ap.add_argument("--surgery-out", default="results/e4_surgeries.json")
    a = ap.parse_args()

    val = [json.loads(l) for l in open(a.val_records)]
    test = [json.loads(l) for l in open(a.test_records)]
    ecdf = ECDFStore.load(f"{a.veto_dir}/ecdf.json")
    tau = json.load(open(f"{a.veto_dir}/tau.json"))["tau"]
    floors = {k: v["raw_floor"] for k, v in
              json.load(open(f"{a.veto_dir}/thresholds.json")).items()}

    # --- arms ------------------------------------------------------------
    from sage.fuse import FusionEngine

    engine = FusionEngine()  # default weights = val-estimated (regenerated)

    def veto_primary(rec: dict) -> str:
        o = apply_veto(rec["probs"], rec["primary_code"],
                       _evidence_from_record(rec), tau=tau, ecdf=ecdf,
                       raw_floors=floors, gt_labels=rec["gt_labels"],
                       gt_primary=rec["gt_primary"])
        return o.final_name

    def fusion_ro_primary(rec: dict) -> str:
        """Removal-only fusion: adjusted probabilities may only REMOVE
        claims; the primary can demote to a surviving claim or normal."""
        from sage_check.result import VLMResult
        vlm = VLMResult(primary_code=rec["primary_code"],
                        probabilities=dict(rec["probs"]))
        adj = engine.fuse(vlm, _evidence_from_record(rec)).adjusted_probabilities
        orig = {c for c, p in rec["probs"].items() if p >= 0.5 and c != NORMAL_CODE}
        kept = {c for c in orig if adj.get(c, 0.0) >= 0.5}
        if rec["primary_code"] in kept:
            return CODE_TO_NAME.get(rec["primary_code"], rec["primary_code"])
        if kept:
            return CODE_TO_NAME.get(max(kept, key=lambda c: rec["probs"].get(c, 0.0)),
                                    "")
        return "normal"

    # temperature: one global scale on all LRs, tuned on validation accuracy
    val = [json.loads(l) for l in open(a.val_records)]
    def fused_acc(records, T):
        from sage_check.result import VLMResult
        adj_engine = FusionEngine(weights={c: {k: v * T for k, v in e.items()}
                                           for c, e in engine.weights.items()})
        hits = 0
        for r in records:
            vlm = VLMResult(primary_code=r["primary_code"],
                            probabilities=dict(r["probs"]))
            hits += int(adj_engine.fuse(vlm, _evidence_from_record(r)).final_name
                        == r["gt_primary"])
        return hits / len(records)

    best_T, best_val = 1.0, fused_acc(val, 1.0)
    for T in (0.5, 0.75, 1.5, 2.0):
        acc = fused_acc(val, T)
        if acc > best_val:
            best_T, best_val = T, acc
    engine = FusionEngine(weights={c: {k: v * best_T for k, v in e.items()}
                                   for c, e in engine.weights.items()})
    print(f"[e4] temperature selected on validation: T={best_T} "
          f"(val acc {best_val:.4f})")

    arms = {
        "none": lambda r: CODE_TO_NAME.get(r["primary_code"], r["primary_code"]),
        "veto": veto_primary,
        "fusion": lambda r: fuse_primary(r, engine),
        "fusion_ro": fusion_ro_primary,
    }

    results: dict = {"n": len(test),
                     "lr_weights": "checkpoints/fuse_lr_weights.json (val-estimated)"}
    per_arm_correct: dict[str, list[int]] = {}
    for label, fn in arms.items():
        correct = [int(fn(r) == r["gt_primary"]) for r in test]
        per_arm_correct[label] = correct
        results[label] = {"primary_acc": sum(correct) / len(correct)}
        print(f"{label:16s} primary_acc = {sum(correct) / len(correct):.4f}")

    # McNemar: veto vs fusion_val_lr (the preregistered comparison)
    def mcnemar(ca: list[int], cb: list[int]) -> dict:
        b = sum(1 for x, y in zip(ca, cb) if x and not y)
        c = sum(1 for x, y in zip(ca, cb) if not x and y)
        from math import comb
        n = b + c
        p = 1.0 if n == 0 else min(
            1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)
        return {"b_fix": b, "c_break": c, "p": p}

    results["mcnemar_veto_vs_fusion"] = mcnemar(
        per_arm_correct["veto"], per_arm_correct["fusion"])
    results["mcnemar_veto_vs_none"] = mcnemar(
        per_arm_correct["veto"], per_arm_correct["none"])
    results["mcnemar_fusion_vs_none"] = mcnemar(
        per_arm_correct["fusion"], per_arm_correct["none"])
    results["mcnemar_veto_vs_fusion_ro"] = mcnemar(
        per_arm_correct["veto"], per_arm_correct["fusion_ro"])
    results["temperature"] = best_T

    # Per-frame correctness dump (round-9 N3a): lets third parties recompute
    # every pairwise McNemar and the surgery counts from results/released files.
    results["per_frame"] = [
        {"sample_id": r["sample_id"],
         **{label: int(per_arm_correct[label][i]) for label in arms}}
        for i, r in enumerate(test)]

    # --- label surgery: final multi-label set vs the VLM's claims ---------
    # Surgery = labels added to / removed from the VLM's multi-label set
    # (the set the paper's Table 4 reports as +add/-rem), archived per frame
    # so the counts are reproducible without re-running the arms.
    def veto_claim_set(rec: dict) -> tuple[set[str], set[str]]:
        o = apply_veto(rec["probs"], rec["primary_code"],
                       _evidence_from_record(rec), tau=tau, ecdf=ecdf,
                       raw_floors=floors, gt_labels=rec["gt_labels"],
                       gt_primary=rec["gt_primary"])
        # Surgery is measured against the VLM's FULL claim set (same
        # definition as the fusion arms), not o.claims (veto-scope only).
        # corrected_codes is code-keyed; unify to names.
        orig = {CODE_TO_NAME.get(c, c) for c, p in rec["probs"].items()
                if p >= 0.5 and c != NORMAL_CODE}
        return orig, {CODE_TO_NAME.get(c, c)
                      for c in o.corrected_codes}

    def fused_claim_set(rec: dict) -> tuple[set[str], set[str]]:
        from sage_check.result import VLMResult
        vlm = VLMResult(primary_code=rec["primary_code"],
                        probabilities=dict(rec["probs"]))
        adj = engine.fuse(vlm, _evidence_from_record(rec)).adjusted_probabilities
        orig = {c for c, p in rec["probs"].items()
                if p >= 0.5 and c != NORMAL_CODE}
        final = {c for c, p in adj.items() if p >= 0.5 and c != NORMAL_CODE}
        return orig, final

    def fused_ro_claim_set(rec: dict) -> tuple[set[str], set[str]]:
        from sage_check.result import VLMResult
        vlm = VLMResult(primary_code=rec["primary_code"],
                        probabilities=dict(rec["probs"]))
        adj = engine.fuse(vlm, _evidence_from_record(rec)).adjusted_probabilities
        orig = {c for c, p in rec["probs"].items()
                if p >= 0.5 and c != NORMAL_CODE}
        final = {c for c in orig if adj.get(c, 0.0) >= 0.5}
        return orig, final

    surgery_arms = {
        "veto": veto_claim_set,
        "fusion": fused_claim_set,
        "fusion_ro": fused_ro_claim_set,
    }
    surgeries: dict = {}
    for label, fn in surgery_arms.items():
        frames, n_add, n_rem = [], 0, 0
        for r in test:
            orig, final = fn(r)
            added, removed = sorted(final - orig), sorted(orig - final)
            n_add += len(added)
            n_rem += len(removed)
            if added or removed:
                frames.append({"sample_id": r["sample_id"],
                               "orig": sorted(orig), "final": sorted(final),
                               "added": added, "removed": removed})
        surgeries[label] = {"add": n_add, "rem": n_rem, "frames": frames}
        print(f"surgery {label:16s} +{n_add}/-{n_rem}")

    Path(a.surgery_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.surgery_out).write_text(json.dumps(
        {"n": len(test), "temperature": best_T, "surgeries": surgeries},
        indent=2))
    print(f"\nsurgery details written to {a.surgery_out}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {a.out}")
    for k in ("mcnemar_veto_vs_fusion", "mcnemar_veto_vs_none",
              "mcnemar_fusion_vs_none", "mcnemar_veto_vs_fusion_ro"):
        print(f"{k}: {results[k]}")


if __name__ == "__main__":
    main()
