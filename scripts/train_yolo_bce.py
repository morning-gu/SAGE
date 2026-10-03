"""Train a YOLOv8s backbone + sigmoid + BCE multi-label head on SAGE dataset.

Missing-baseline control requested by peer review (DA C-1 follow-up): the
YOLOv8s-cls baseline uses a softmax head that is mathematically incapable of
multi-label output.  This baseline keeps the YOLOv8s backbone but replaces
the classification head with Linear(1280, 15) + BCEWithLogitsLoss, isolating
"sigmoid+BCE multi-label formulation" from "VLM vs CNN backbone" factors.

Backbone: YOLOv8s-cls feature extractor (layers before the Classify head plus
          the head's own Conv(->1280) + AdaptiveAvgPool2d, 1280-d pooled
          output), fully fine-tuned. Built by sage.models.build_yolo_bce —
          the factory is shared with scripts/eval_yolo_bce.py so eval can
          rebuild the exact architecture from the saved state_dict.
Loss:     BCEWithLogitsLoss.
Head:     Linear(1280, 15).

NOTE ultralytics internals: the backbone is extracted from
``YOLO("models/detectors/yolov8s-cls.pt").model.model``.  This is not a
stable public API — pin ultralytics and treat extraction failures as
non-blocking (the yolobce_grid experiments may fail gracefully without
affecting the main line).

Data pipeline / epoch loop / early-stopping are shared with the other
sigmoid baselines via sage/data.py + sage/ml_common.py (fairness protocol:
docs/ablation_protocol.md).  Inputs are normalized to [0,1] (YOLO's own
preprocessing), NOT ImageNet statistics.

Usage:
    python scripts/train_yolo_bce.py --epochs 25 --lr 1e-3 --imgsz 224
"""
from __future__ import annotations

import argparse
from sage.paths import GT_ANNOTATIONS
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
from sage.data import BEHAVIOR_NAMES, NUM_CLASSES, load_annotations, make_loaders  # noqa: E402
from sage.ml_common import fit, set_seed, torch_device  # noqa: E402
from sage.models import build_yolo_bce  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(GT_ANNOTATIONS),
                    help="Ground truth annotations JSON")
    ap.add_argument("--output", default=str(REPO / "results/yolo_bce"),
                    help="Output directory for checkpoints and logs")
    ap.add_argument("--weights", default="models/detectors/yolov8s-cls.pt",
                    help="YOLO-cls pretrained weights")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=0,
                    help="Early-stop patience on val macro-F1 (0 = off)")
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device(torch_device())
    print(f"[train] device={device}")
    print(f"[train] gt={a.gt}  output={a.output}")
    print(f"[train] epochs={a.epochs} batch={a.batch} lr={a.lr} seed={a.seed} "
          f"imgsz={a.imgsz} patience={a.patience}")

    set_seed(a.seed)

    try:
        model = build_yolo_bce(weights=a.weights, num_classes=NUM_CLASSES)
        in_features = model[-1].in_features
    except Exception as e:
        print(f"[err] YOLO backbone extraction failed (non-blocking, see "
              f"docs/ablation_protocol.md risk section): {e}")
        sys.exit(2)

    model = model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] YOLOv8s backbone ({in_features}-d) + Linear({in_features},{NUM_CLASSES})"
          f"  trainable={n_trainable:,}")

    train_anns = load_annotations(a.gt, split="train")
    val_anns = load_annotations(a.gt, split="val")
    print(f"[train] train samples: {len(train_anns)}  val samples: {len(val_anns)}")

    train_loader, val_loader = make_loaders(
        train_anns, val_anns, REPO, imgsz=a.imgsz, batch=a.batch,
        num_workers=a.num_workers, normalize="scale01")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr,
                                 weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.epochs)

    fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=a.epochs, output_dir=a.output, device=device,
        patience=a.patience,
        config={
            "model": "yolov8s-cls-backbone+sigmoid-bce",
            "weights": a.weights,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "lr": a.lr,
            "weight_decay": a.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "epochs": a.epochs,
            "batch": a.batch,
            "seed": a.seed,
            "imgsz": a.imgsz,
            "patience": a.patience,
            "num_classes": NUM_CLASSES,
            "classes": BEHAVIOR_NAMES,
        })


if __name__ == "__main__":
    main()
