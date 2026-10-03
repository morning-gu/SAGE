"""Train YOLOv8s classification head on SAGE dataset.

Uses the ultralytics classification API.  Images are organized by
``primary_label`` (see ``prepare_yolo_cls.py``).  Hyperparameters are
aligned with the VLM LoRA baseline where possible: epochs=5, seed=42.

Usage:
    python scripts/train_yolo_cls.py
    python scripts/train_yolo_cls.py --epochs 10 --batch 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "data/yolo_cls"),
                    help="Dataset root (train/ and val/ class folders)")
    ap.add_argument("--output", default=str(REPO / "checkpoints/yolo_cls"),
                    help="Output directory for checkpoints and logs")
    ap.add_argument("--model", default="models/detectors/yolov8s-cls.pt",
                    help="Pretrained model weights (default: models/detectors/yolov8s-cls.pt)")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Training epochs (default: 5, aligned with VLM LoRA)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size (default: 16, aligned with VLM grad_accum)")
    ap.add_argument("--imgsz", type=int, default=224,
                    help="Input image size (default: 224, YOLO-cls standard)")
    ap.add_argument("--lr0", type=float, default=0.01,
                    help="Initial learning rate (default: 0.01, YOLO cosine)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42, aligned with VLM)")
    a = ap.parse_args()

    from ultralytics import YOLO

    print(f"[train] model={a.model}  data={a.data}")
    print(f"[train] epochs={a.epochs} batch={a.batch} imgsz={a.imgsz} lr0={a.lr0} seed={a.seed}")
    print(f"[train] output={a.output}")

    model = YOLO(a.model)

    # Print model summary for the paper
    info = model.info()
    print(f"[train] model info: {info}")

    model.train(
        data=a.data,
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        lr0=a.lr0,
        seed=a.seed,
        project=a.output,
        name="train",
        exist_ok=True,
    )

    best = Path(a.output) / "train" / "weights" / "best.pt"
    print(f"\n[train] training complete, best weights -> {best}")

    # Built-in validation on val split
    if best.exists():
        m = YOLO(str(best))
        metrics = m.val(data=a.data, split="val")
        print(f"[train] val top-1 accuracy: {float(metrics.top1):.4f}")
        print(f"[train] val top-5 accuracy: {float(metrics.top5):.4f}")
    else:
        print(f"[train] WARN: best.pt not found at {best}")


if __name__ == "__main__":
    main()
