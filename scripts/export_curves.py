#!/usr/bin/env python
"""Export training curves from a ms-swift / HF Trainer run directory.

Reads whichever of these exist under --run-dir:
  - logging.jsonl          ms-swift 4.4.2 per-log records
  - trainer_state.json     HF Trainer standard format (log_history)
  - loss_components.jsonl  written by train_sage_vlm.py's LossComponentsCallback

and writes <run-dir>/curves.json:
  {steps[], train_loss[], eval_loss[], eval_token_acc[], loss_components[]}

Field names are tolerant of swift/HF variants:
  train loss:  loss | train_loss
  eval loss:   eval_loss
  token acc:   eval_token_acc | token_acc
  step:        global_step | step   (falls back to record index)

A markdown summary table (mean train loss per epoch) is printed to stdout.
Works without ms-swift installed (trainer_state.json fallback path).

Usage:
  python scripts/export_curves.py --run-dir checkpoints/experiments/d1_bce_mp_s42
  python scripts/export_curves.py --run-dir <dir> --out /tmp/curves.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def read_jsonl(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def load_log_records(run_dir: Path) -> list[dict]:
    """Merge logging.jsonl (swift) and trainer_state.json log_history (HF)."""
    recs: list[dict] = []
    logging_jsonl = run_dir / "logging.jsonl"
    if logging_jsonl.exists():
        recs.extend(read_jsonl(logging_jsonl))
    trainer_state = run_dir / "trainer_state.json"
    if trainer_state.exists():
        try:
            state = json.loads(trainer_state.read_text(encoding="utf-8"))
            recs.extend(state.get("log_history") or [])
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] cannot parse {trainer_state}: {e}")
    return recs


def _record_kept(rec: dict) -> bool:
    """Same filter as extract_curves so curve arrays stay position-aligned
    with the records returned here."""
    is_eval = "eval_loss" in rec
    loss = _first(rec, ["loss", "train_loss"]) if not is_eval else None
    tacc = _first(rec, ["eval_token_acc", "token_acc"]) if is_eval else None
    return loss is not None or rec.get("eval_loss") is not None or tacc is not None


def extract_curves(records: list[dict]) -> tuple[dict, list[dict]]:
    steps, train_loss, eval_loss, eval_token_acc = [], [], [], []
    kept = []
    for i, rec in enumerate(records):
        is_eval = "eval_loss" in rec
        loss = _first(rec, ["loss", "train_loss"]) if not is_eval else None
        eloss = rec.get("eval_loss") if is_eval else None
        tacc = _first(rec, ["eval_token_acc", "token_acc"]) if is_eval else None
        if loss is None and eloss is None and tacc is None:
            continue  # e.g. lr-scheduler-only records
        step = _first(rec, ["global_step", "step"])
        if step is None:
            step = _first(rec, ["epoch"])
        if step is None:
            step = i + 1
        steps.append(step)
        train_loss.append(loss)
        eval_loss.append(eloss)
        eval_token_acc.append(tacc)
        kept.append(rec)
    curves = {"steps": steps, "train_loss": train_loss, "eval_loss": eval_loss,
              "eval_token_acc": eval_token_acc}
    return curves, kept


def load_loss_components(run_dir: Path) -> list[dict]:
    path = run_dir / "loss_components.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def markdown_summary(records: list[dict], curves: dict,
                     components: list[dict]) -> str:
    """Markdown table with mean train loss per epoch (HF/swift records carry a
    fractional `epoch` field; falls back to a per-step table when absent)."""
    fmt = lambda v: ("-" if v is None else f"{v:.4f}")  # noqa: E731

    # position-aligned with the curve arrays by construction
    paired = [r for r in records if _record_kept(r)]
    epochs = [_first(r, ["epoch"]) for r in paired]
    if all(e is not None for e in epochs) and epochs:
        by_epoch: dict[int, dict] = {}
        for e, rec, step, tl, el, ta in zip(
                epochs, paired, curves["steps"], curves["train_loss"],
                curves["eval_loss"], curves["eval_token_acc"]):
            slot = by_epoch.setdefault(int(e), {"tl": [], "el": [], "ta": [], "step": step})
            if tl is not None:
                slot["tl"].append(tl)
            if el is not None:
                slot["el"].append(el)
            if ta is not None:
                slot["ta"].append(ta)
        comps = {}
        for c in components:
            st = c.get("step")
            if st is None:
                continue
            ep = None
            for e, r, step in zip(epochs, paired, curves["steps"]):
                if step == st:
                    ep = int(e)
                    break
            if ep is not None:
                comps[ep] = c
        lines = [
            "| epoch | train_loss(mean) | eval_loss | eval_token_acc "
            "| loss_ce(mean) | loss_bce(mean) | loss_mp(mean) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ep in sorted(by_epoch):
            s = by_epoch[ep]
            mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
            c = comps.get(ep, {})
            lines.append(
                f"| {ep} | {fmt(mean(s['tl']))} | {fmt(mean(s['el']))} "
                f"| {fmt(mean(s['ta']))} | {fmt(c.get('loss_ce'))} "
                f"| {fmt(c.get('loss_bce'))} | {fmt(c.get('loss_mp'))} |")
        return "\n".join(lines)

    # fallback: per-step table
    comps_by_step = {c.get("step"): c for c in components}
    lines = [
        "| step | train_loss | eval_loss | eval_token_acc | loss_ce | loss_bce | loss_mp |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, step in enumerate(curves["steps"]):
        comp = comps_by_step.get(step, {})
        lines.append(
            f"| {step} | {fmt(curves['train_loss'][i])} "
            f"| {fmt(curves['eval_loss'][i])} | {fmt(curves['eval_token_acc'][i])} "
            f"| {fmt(comp.get('loss_ce'))} | {fmt(comp.get('loss_bce'))} "
            f"| {fmt(comp.get('loss_mp'))} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="checkpoint output dir (logging.jsonl / "
                         "trainer_state.json / loss_components.jsonl)")
    ap.add_argument("--out", default="",
                    help="output curves.json path (default: <run-dir>/curves.json)")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    if not run_dir.is_dir():
        print(f"[err] run dir not found: {run_dir}")
        raise SystemExit(1)

    records = load_log_records(run_dir)
    if not records:
        print(f"[err] no logging.jsonl or trainer_state.json with log_history "
              f"under {run_dir}")
        raise SystemExit(1)
    curves, kept_records = extract_curves(records)
    components = load_loss_components(run_dir)

    out_path = Path(a.out) if a.out else run_dir / "curves.json"
    payload = dict(curves, loss_components=components)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_train = sum(v is not None for v in curves["train_loss"])
    n_eval = sum(v is not None for v in curves["eval_loss"])
    print(f"[export] {len(records)} log records -> {out_path} "
          f"(train points: {n_train}, eval points: {n_eval}, "
          f"loss_components: {len(components)})")
    print("\n" + markdown_summary(kept_records, curves, components))


if __name__ == "__main__":
    main()
