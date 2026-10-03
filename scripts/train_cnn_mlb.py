"""Train a ResNet50 + sigmoid + BCE multi-label classifier on SAGE dataset.

This is the CNN multi-label baseline (D-cnn).  It uses the same 2014
annotated samples, same train/val/test splits.  Unlike YOLO's softmax
(mutually exclusive), this model outputs 15 independent sigmoid
probabilities, enabling true multi-label prediction.

Backbone: ResNet50 (ImageNet pre-trained, fine-tuned fully).
Loss:      BCEWithLogitsLoss (numerically stable sigmoid + BCE).
Head:      Linear(2048, 15) replacing the original 1000-class FC.

Data pipeline / epoch loop / early-stopping are shared with the other
sigmoid baselines via sage/data.py + sage/ml_common.py (fairness protocol:
docs/ablation_protocol.md).

Usage:
    python scripts/train_cnn_mlb.py
    python scripts/train_cnn_mlb.py --epochs 10 --batch 32 --lr 1e-3
    python scripts/train_cnn_mlb.py --gt data/sage_eval/annotations.json
"""
from __future__ import annotations

import argparse
from sage.paths import GT_ANNOTATIONS
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
from sage.data import (  # noqa: E402
    BEHAVIOR_NAMES,
    NUM_CLASSES,
    load_annotations,
    make_loaders,
)
from sage.ml_common import fit, set_seed, torch_device  # noqa: E402


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """ResNet50 with 15-output sigmoid head (logits, sigmoid applied by loss)."""
    import torch.nn as nn
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(GT_ANNOTATIONS),
                    help="Ground truth annotations JSON")
    ap.add_argument("--output", default=str(REPO / "results/cnn_mlb"),
                    help="Output directory for checkpoints and logs")
    ap.add_argument("--epochs", type=int, default=5,
                    help="Training epochs (default: 5, aligned with VLM LoRA and YOLO)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size (default: 16, aligned with YOLO)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default: 1e-3, Adam)")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="Weight decay (default: 1e-4)")
    ap.add_argument("--imgsz", type=int, default=224,
                    help="Input image size (default: 224; grid also tries 448)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42, aligned with VLM and YOLO)")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers")
    ap.add_argument("--patience", type=int, default=0,
                    help="Early-stop patience on val macro-F1 (0 = off)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Disable ImageNet pre-training (scratch)")
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device(torch_device())
    print(f"[train] device={device}")
    print(f"[train] gt={a.gt}  output={a.output}")
    print(f"[train] epochs={a.epochs} batch={a.batch} lr={a.lr} seed={a.seed} "
          f"imgsz={a.imgsz} patience={a.patience}")

    set_seed(a.seed)

    train_anns = load_annotations(a.gt, split="train")
    val_anns = load_annotations(a.gt, split="val")
    print(f"[train] train samples: {len(train_anns)}")
    print(f"[train] val samples: {len(val_anns)}")

    train_loader, val_loader = make_loaders(
        train_anns, val_anns, REPO, imgsz=a.imgsz, batch=a.batch,
        num_workers=a.num_workers, normalize="imagenet")

    model = build_model(pretrained=not a.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] ResNet50: {n_params:,} params ({n_trainable:,} trainable)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr,
                                 weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.epochs)

    fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=a.epochs, output_dir=a.output, device=device,
        patience=a.patience,
        config={
            "model": "resnet50",
            "pretrained": not a.no_pretrained,
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
