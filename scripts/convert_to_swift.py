"""
Convert SAGE annotations -> ms-swift training jsonl.

Reads an annotations JSON file (e.g. data/sage_eval/annotations_v3.json) that
carries sample_id, image_path, multi-label `labels`, primary_label, and a
`split` field (train/val/test). No dynamic val splitting -- the split field
determines the output bucket.

Each record:
  messages : system (rendered via sage_infer.render_prompt) + user(<image>text)
             + assistant (2-letter code, optionally + <explanation>, no JSON)
  images   : absolute image path
  primary_code : 2-letter behavior code
  label_vector : 15-dim multi-label 0/1 vector (BEHAVIOR_CODES order) -- the BCE
                 target for scripts/sage_hybrid_loss.py

The assistant target matches scripts/sage_infer.py EXACTLY (same SYSTEM_TEMPLATE,
same 2-letter codes, same <explanation> layout), so a model trained here plugs
straight into sage_infer.py / sage_eval.py with no inference changes.

Usage:
  python scripts/convert_to_swift.py --annotations data/sage_eval/annotations_v3.json
  python scripts/convert_to_swift.py --annotations data/sage_eval/annotations_v3.json --mode reason
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sage_infer import BEHAVIOR_CODES, CODE_TO_NAME, render_prompt  # noqa: E402

NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}

# Templated explanations used as fallback when samples lack an `explanation` field.
# To generate per-sample explanations via OpenAI vision API, run:
#   python scripts/generate_explanations.py --annotations data/sage_eval/annotations_v3.json
# Keep in sync with sage_infer.py SYSTEM_TEMPLATE (single source of truth).
LABEL_EXPLANATION = {
    "normal":   "The student sits upright with gaze on study materials.",
    "away":     "The student is not in seat; nobody is in frame or not at desk.",
    "blocked":  "The body is occluded beyond 50 percent or the face is occluded such that behavior cannot be determined.",
    "toy":      "The student is playing with non-study toys, excluding study-dependent items.",
    "phone":    "The student is holding an electronic device with face clearly facing the screen.",
    "snack":    "The student is holding food, handling snack packaging, or eating.",
    "eyesclosed":   "The student's eyes are closed.",
    "prone":    "The upper body is slumped onto the desk, head resting on the desk or arms.",
    "bowed":    "The head is clearly lowered or face down, not touching the desk.",
    "chinrest": "Both hands with elbows on the desk support the chin or cheek; a single hand does not count.",
    "tilt":     "The head leans noticeably left or right.",
    "turn":     "The head is noticeably turned to one side.",
    "slope":    "The shoulders are noticeably not level.",
    "recline":  "The body leans back against the chair.",
    "lookup":   "The face is clearly upward, chin raised.",
}


def resolve_image(sample):
    """Resolve the sample's image_path to an absolute path."""
    rec = sample.get("image_path")
    if not rec:
        return ""
    p = Path(rec)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p)


def build_target(primary: str, mode: str, explanation: str = "") -> str:
    """Build the assistant target string matching sage_infer.py's prompt layout."""
    code = NAME_TO_CODE.get(primary)
    if code is None:
        raise ValueError(f"unknown primary label: {primary!r}")
    if mode == "no_reason":
        return code
    expl = explanation or LABEL_EXPLANATION.get(primary, f"The behavior is {primary}.")
    return f"{code}\n<explanation>{expl}</explanation>"


def label_vector_for(sample: dict) -> list[int]:
    labels = sample.get("labels", {})
    return [int(labels.get(CODE_TO_NAME[c], 0)) for c in BEHAVIOR_CODES]


def to_record(sample, sys_prompt, image_abs, mode):
    """Build a training record."""
    explanation = sample.get("explanation", "")
    target = build_target(sample["primary_label"], mode, explanation)
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",
             "content": "<image>Analyze the student's behavior and provide your classification."},
            {"role": "assistant", "content": target},
        ],
        "images": [image_abs],
        "primary_code": NAME_TO_CODE.get(sample["primary_label"], ""),
        "label_vector": label_vector_for(sample),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Convert SAGE annotations to ms-swift jsonl format.")
    ap.add_argument("--annotations", required=True,
                    help="Input annotations JSON file (must carry a 'split' field)")
    ap.add_argument("--mode", choices=["no_reason", "reason"], default="no_reason",
                    help="no_reason: code only; reason: code + explanation")
    ap.add_argument("--out", default="data/swift", help="Output directory for jsonl files")
    a = ap.parse_args()

    sys_prompt = render_prompt(no_reason=(a.mode == "no_reason"))

    with open(a.annotations, encoding="utf-8") as f:
        samples = json.load(f)
    # Bucket by the record's own `split` field (train/val/test).
    buckets = {"train": [], "val": [], "test": []}
    no_split = []
    for s in samples:
        sp = s.get("split")
        if sp in buckets:
            buckets[sp].append(s)
        else:
            no_split.append(s)
    if no_split:
        print(f"[warn] {len(no_split)} samples have no `split` field; assigning to train")
        buckets["train"].extend(no_split)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, missing, bad_label = {}, 0, 0
    for split, items in buckets.items():
        records = []
        for s in items:
            if s.get("primary_label") not in NAME_TO_CODE:
                bad_label += 1
                continue
            img_abs = resolve_image(s)
            if not Path(img_abs).exists():
                missing += 1
                continue
            rec = to_record(s, sys_prompt, img_abs, a.mode)
            records.append(rec)
        p = out / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(records)
        print(f"[{split}] {len(records):>4d} -> {p}")

    total = sum(counts.values())
    print(f"\nmode={a.mode}  total={total}  split={counts}  "
          f"missing_images={missing}  bad_labels={bad_label}")
    if a.mode != "no_reason":
        has_expl = sum(1 for s in samples if s.get("explanation"))
        if has_expl:
            print(f"[info] {has_expl} samples have per-sample explanations")
        else:
            print("[warn] reason mode uses TEMPLATED explanations (annotations have none).")
            print("       Run scripts/generate_explanations.py to generate per-sample explanations.")
    print("\nTrain with ms-swift + hybrid loss:")
    print("  python scripts/sage_hybrid_loss.py --model Qwen/Qwen3.5-4B \\")
    print("    --dataset data/swift/train.jsonl --output-dir checkpoints/swift_lora")
    print("  python scripts/sage_eval.py --model-path <merged> --gt <gt> --output <out>")


if __name__ == "__main__":
    main()
