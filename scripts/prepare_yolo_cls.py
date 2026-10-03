"""Prepare a YOLOv8 classification dataset from SAGE annotations.

Organizes images by ``primary_label`` into train/val class folders for
ultralytics YOLOv8-cls training.  The test split is not copied into folders
(because test images are multi-label and cannot belong to a single class
folder); instead a manifest is saved for the evaluation script.

Usage:
    python scripts/prepare_yolo_cls.py
    python scripts/prepare_yolo_cls.py --gt data/sage_eval/annotations.json --output data/yolo_cls
"""
from __future__ import annotations

import argparse
from sage.paths import GT_ANNOTATIONS
import json
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BEHAVIOR_NAMES = [
    "normal", "away", "blocked", "toy", "phone", "snack", "eyesclosed",
    "prone", "bowed", "chinrest", "tilt", "turn", "slope", "recline", "lookup",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(GT_ANNOTATIONS),
                    help="Ground truth annotations JSON")
    ap.add_argument("--output", default=str(REPO / "data/yolo_cls"),
                    help="Output dataset root directory")
    ap.add_argument("--image-root", default="",
                    help="Root dir for resolving relative image paths (default: repo root)")
    a = ap.parse_args()

    image_root = Path(a.image_root) if a.image_root else REPO
    output = Path(a.output)

    with open(a.gt, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[prepare] loaded {len(data)} samples from {a.gt}")

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in data:
        sp = s.get("split", "")
        if sp in splits:
            splits[sp].append(s)

    test_manifest: list[dict] = []
    copy_stats: Counter = Counter()
    missing = 0

    for split_name in ("train", "val"):
        for s in splits[split_name]:
            primary = s.get("primary_label", "normal")
            if primary not in BEHAVIOR_NAMES:
                primary = "normal"
            img_path = s["image_path"]
            if not Path(img_path).is_absolute():
                img_path = str(image_root / img_path)
            if not Path(img_path).exists():
                missing += 1
                continue
            dst_dir = output / split_name / primary
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_dir / f"{s['sample_id']}.jpg")
            copy_stats[f"{split_name}/{primary}"] += 1

    for s in splits["test"]:
        img_path = s["image_path"]
        if not Path(img_path).is_absolute():
            img_path = str(image_root / img_path)
        test_manifest.append({
            "sample_id": s["sample_id"],
            "image_path": img_path,
            "labels": dict(s["labels"]),
            "primary_label": s.get("primary_label", "normal"),
        })

    manifest_path = output / "test_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2, ensure_ascii=False)

    print(f"[prepare] copied {sum(copy_stats.values())} images, {missing} missing")
    for split_name in ("train", "val"):
        split_dir = output / split_name
        if not split_dir.exists():
            continue
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        counts = {c: len(list((split_dir / c).iterdir())) for c in classes}
        total = sum(counts.values())
        print(f"  {split_name}: {total} images, {len(classes)} classes")
        for c in sorted(counts, key=lambda x: -counts[x]):
            print(f"    {c}: {counts[c]}")
    print(f"[prepare] test manifest: {len(test_manifest)} samples -> {manifest_path}")
    print("[prepare] done")


if __name__ == "__main__":
    main()
