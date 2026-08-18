"""
Convert SAGE annotations -> ms-swift training jsonl (YuFeng-XGuard-Reason style).

Reads a SINGLE source -- data/sage_eval/annotations.json -- which already carries
sample_id, image_path, multi-label `labels`, primary_label, and a `split` field
(train/test). No dependency on the old data/{train,val,test}.json split files.

Each record:
  messages : system (rendered via sage_infer.render_prompt) + user(<image>text)
             + assistant (2-letter code, optionally + <explanation>, no JSON)
  images   : absolute image path
  primary_code : 2-letter behavior code (for label-token id lookup at train/infer)
  label_vector : 15-dim multi-label 0/1 vector (BEHAVIOR_CODES order) -- the BCE
                 target for scripts/sage_hybrid_loss.py; carries the co-occurring
                 labels that CE on the single primary code would otherwise suppress.

The assistant target matches scripts/sage_infer.py EXACTLY (same SYSTEM_TEMPLATE,
same 2-letter codes, same <explanation> layout), so a model trained here plugs
straight into sage_infer.py / sage_eval.py with no inference changes.

Usage:
  python scripts/convert_to_swift.py
  python scripts/convert_to_swift.py --mode reason_last
  python scripts/convert_to_swift.py --val-ratio 0.1
  python scripts/convert_to_swift.py --mode reason_last --annotations data/sage_eval/annotations_with_explanations.json
  python scripts/convert_to_swift.py --mode reason_last --mismatch-policy template
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sage_infer import BEHAVIOR_CODES, CODE_TO_NAME, render_prompt  # noqa: E402

NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}

# Templated explanations used as fallback when samples lack an `explanation` field.
# To generate per-sample explanations via OpenAI vision API, run:
#   python scripts/generate_explanations.py
# Then pass --annotations data/sage_eval/annotations_with_explanations.json
LABEL_EXPLANATION = {
    "normal":   "The student sits upright with gaze on study materials.",
    "away":     "The student is not in seat; nobody is in frame or not at desk.",
    "blocked":  "The frame is occluded beyond 50 percent.",
    "toy":      "The student is holding non-study items and playing.",
    "phone":    "The student is holding a phone or tablet with gaze on the screen.",
    "snack":    "The student is holding food near the mouth.",
    "sleepy":   "The student's eyes are closed.",
    "prone":    "The upper body leans forward over 45 degrees, head near the desk.",
    "bowed":    "The head is down, nose below the shoulder line.",
    "chinrest": "Both hands support the chin or cheek.",
    "tilt":     "The head leans noticeably left or right.",
    "turn":     "The head is turned to the side.",
    "slope":    "The shoulders are noticeably not level.",
    "recline":  "The body leans back against the chair.",
    "lookup":   "The head tilts noticeably back.",
}

THINK_BLOCK = "\x3cthink\x3e\n\n\x3c/think\x3e\n\n"


def _group_by_date(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[s["sample_id"].split("_")[0]].append(s)
    return groups


def split_by_date(samples, seed, train_r, val_r):
    """Anti-leakage date-prefix split (groups whole sessions together)."""
    groups = _group_by_date(samples)
    dates = sorted(groups)
    random.seed(seed)
    random.shuffle(dates)
    n = len(dates)
    nt = max(1, int(n * train_r))
    nv = max(1, int(n * val_r))
    out = {"train": [], "val": [], "test": []}
    for i, d in enumerate(dates):
        key = "train" if i < nt else ("val" if i < nt + nv else "test")
        out[key].extend(groups[d])
    return out


def carve_val_from_train(train_samples, seed, val_r):
    """Carve a val set out of the train bucket by random sample split.

    Session sizes in this dataset are highly skewed (one date holds ~300 of 749
    train samples), so date-level splitting either under- or over-shoots badly.
    The val set here is only a training monitor; the real held-out set is the
    `test` bucket from annotations.json, so a sample-level random carve is the
    pragmatic choice and yields a val of the requested size.
    """
    shuffled = list(train_samples)
    random.seed(seed)
    random.shuffle(shuffled)
    nv = max(1, int(len(shuffled) * val_r))
    return shuffled[nv:], shuffled[:nv]


def resolve_image(sample, images_dir, image_ext):
    """Prefer the record's image_path; resolve absolute; fall back to images_dir."""
    rec = sample.get("image_path")
    if rec:
        p = Path(rec)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            return str(p)
    # fallback by sample_id
    for cand in (images_dir / f"{sample['sample_id']}{image_ext}",):
        if cand.exists():
            return str(cand.resolve())
    # last resort: return the preferred path even if missing (caller counts it)
    return str((Path(rec) if rec else images_dir / f"{sample['sample_id']}{image_ext}").resolve())


def build_target(primary: str, mode: str, think_block: bool, explanation: str = "") -> str:
    code = NAME_TO_CODE.get(primary)
    if code is None:
        raise ValueError(f"unknown primary label: {primary!r}")
    prefix = THINK_BLOCK if think_block else ""
    if mode == "no_reason":
        body = code
    else:
        expl = explanation or LABEL_EXPLANATION.get(primary, f"The behavior is {primary}.")
        body = (f"<explanation>{expl}</explanation>\n{code}" if mode == "reason_first"
                else f"{code}\n<explanation>{expl}</explanation>")
    return prefix + body


def label_vector_for(sample: dict) -> list[int]:
    labels = sample.get("labels", {})
    return [int(labels.get(CODE_TO_NAME[c], 0)) for c in BEHAVIOR_CODES]


def to_record(sample, sys_prompt, image_abs, mode, think_block,
              mismatch_policy="template"):
    """Build a training record. Returns None when mismatch_policy='skip' and
    the sample's model classification didn't match ground truth."""
    """Build a training record. Returns None when mismatch_policy='skip' and
    the sample's model classification didn't match ground truth (any
    verification tag ending in 'mismatch': redirect_mismatch, forced_mismatch, etc.)"""
    verification = sample.get("verification", "")
    explanation = sample.get("explanation", "")
    if verification and verification.endswith("mismatch"):
        if mismatch_policy == "skip":
            return None
        elif mismatch_policy != "keep":
            explanation = ""  # fall back to LABEL_EXPLANATION template
    target = build_target(sample["primary_label"], mode, think_block, explanation)
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
        description="Convert SAGE annotations to ms-swift jsonl (YuFeng style).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="data/sage_eval/annotations.json")
    ap.add_argument("--images-dir", default="data/sage_eval/images")
    ap.add_argument("--image-ext", default=".jpg")
    ap.add_argument("--out", default="data/swift")
    ap.add_argument("--mode", choices=["no_reason", "reason_first", "reason_last"],
                    default="no_reason")
    ap.add_argument("--think-block", action="store_true",
                    help="prepend empty think block to the target")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="carve val from the train bucket by date (0 = keep split as-is)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mismatch-policy", choices=["template", "skip", "keep"],
                    default="template",
                    help="how to handle model-GT classification mismatches: "
                         "template=use template explanation (default), "
                         "skip=exclude from training data, "
                         "keep=use model explanation anyway")
    a = ap.parse_args()

    sys_prompt = render_prompt(
        reason_first=(a.mode == "reason_first"), policy=None,
        no_reason=(a.mode == "no_reason"))

    samples = json.load(open(a.annotations, encoding="utf-8"))
    images_dir = Path(a.images_dir)

    # Bucket by the record's own `split` field (train/test/val).
    buckets = {"train": [], "val": [], "test": []}
    no_split = []
    for s in samples:
        sp = s.get("split")
        if sp in buckets:
            buckets[sp].append(s)
        else:
            no_split.append(s)
    if no_split:
        print(f"[warn] {len(no_split)} samples have no `split` field; date fallback")
        for k, v in split_by_date(no_split, a.seed, 0.7, 0.15).items():
            buckets[k].extend(v)

    # Carve val from train if the source has none (annotations.json is train/test).
    if a.val_ratio > 0 and not buckets["val"] and buckets["train"]:
        buckets["train"], buckets["val"] = carve_val_from_train(
            buckets["train"], a.seed, a.val_ratio)
        print(f"[info] carved val (ratio={a.val_ratio}) from train (random sample split)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, missing, bad_label = {}, 0, 0
    skipped = 0
    for split, items in buckets.items():
        records = []
        for s in items:
            if s.get("primary_label") not in NAME_TO_CODE:
                bad_label += 1
                continue
            img_abs = resolve_image(s, images_dir, a.image_ext)
            if not Path(img_abs).exists():
                missing += 1
                continue
            rec = to_record(s, sys_prompt, img_abs, a.mode, a.think_block,
                            a.mismatch_policy)
            if rec is None:
                skipped += 1
                continue
            records.append(rec)
        p = out / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(records)
        print(f"[{split}] {len(records):>4d} -> {p}")

    total = sum(counts.values())
    print(f"\nmode={a.mode}  think_block={a.think_block}  seed={a.seed}  val_ratio={a.val_ratio}  mismatch_policy={a.mismatch_policy}")
    print(f"total={total}  split={counts}  missing_images={missing}  bad_labels={bad_label}  skipped={skipped}")
    if a.mode != "no_reason":
        has_expl = sum(1 for s in samples if s.get("explanation"))
        if has_expl:
            from collections import Counter
            tags = Counter(s.get("verification", "") for s in samples if s.get("verification"))
            print(f"[info] verification tags: {dict(tags)}")
            print(f"       policy={a.mismatch_policy}, skipped={skipped}")
        else:
            print("[warn] reason modes use TEMPLATED explanations (annotations have none).")
            print("       Run scripts/generate_explanations.py to generate per-sample explanations.")
    print("\nTrain with ms-swift + hybrid loss:")
    print("  python scripts/sage_hybrid_loss.py --model Qwen/Qwen3.5-4B \\")
    print("    --dataset data/swift/train.jsonl --val-dataset data/swift/val.jsonl")
    print("  python scripts/sage_eval.py --model-path <merged> --gt <gt> --output <out> --no-reason")


if __name__ == "__main__":
    main()
