"""Shared training-loop skeleton for the sigmoid multi-label baselines.

Used by scripts/train_cnn_mlb.py (ResNet50), scripts/train_yolo_bce.py
(YOLOv8s backbone + sigmoid/BCE head) and scripts/train_ext_head.py
(external linear head on VLM features) so that all three share the exact
same epoch loop, early-stopping rule and logging format — a requirement of
the fairness protocol (docs/ablation_protocol.md).

Metrics come from sage/metrics.py (no duplicated implementations).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from sage.metrics import macro_f1, primary_hit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _epoch_stats(preds: np.ndarray, gts: np.ndarray) -> tuple[float, float]:
    """(macro-F1@0.5, primary_acc_hit) over concatenated epoch predictions."""
    f1 = macro_f1(gts, (preds >= 0.5).astype(np.float32))
    pred_primary = np.argmax(preds, axis=1)
    hit = primary_hit(gts, pred_primary)
    return f1, float(np.mean(hit)) if len(hit) else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int,
                    log_prefix: str = "[train]"):
    import torch

    model.train()
    total_loss, n_batches = 0.0, 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    f1, acc = _epoch_stats(np.concatenate(all_preds), np.concatenate(all_labels))
    print(f"  {log_prefix} [epoch {epoch}] train loss={avg_loss:.4f}  macro-F1={f1:.4f}")
    return avg_loss, f1


def evaluate(model, loader, criterion, device, log_prefix: str = "[train]"):
    import torch

    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_labels)
    f1, acc = _epoch_stats(preds, gts)
    print(f"  {log_prefix} [val] loss={avg_loss:.4f}  macro-F1={f1:.4f}  "
          f"primary_acc(hit)={acc:.4f}")
    return avg_loss, f1, acc, preds, gts


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs: int, output_dir: str | Path, device,
        patience: int = 0, config: dict | None = None,
        log_prefix: str = "[train]") -> dict:
    """Full training loop with best-checkpoint tracking and optional early stop.

    Early stopping (fairness protocol): monitors val macro-F1, patience 0 = off.
    Saves best.pt (highest val macro-F1) and final.pt, plus train_log.json
    (config + per-epoch history).  Returns the history list.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict] = []
    val_f1 = 0.0
    val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, log_prefix)
        val_loss, val_f1, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, log_prefix)
        if scheduler is not None:
            scheduler.step()
        lr_now = scheduler.get_last_lr()[0] if scheduler is not None else None

        history.append({
            "epoch": epoch,
            "seconds": round(time.time() - t0, 1),
            "train_loss": round(train_loss, 4),
            "train_macro_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_primary_acc": round(val_acc, 4),
            "lr": lr_now,
        })

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            epochs_no_improve = 0
            ckpt_path = output_dir / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "val_primary_acc": val_acc,
            }, ckpt_path)
            print(f"  {log_prefix} -> new best macro-F1={val_f1:.4f}, saved -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  {log_prefix} early stop at epoch {epoch} "
                      f"(no val macro-F1 improvement for {patience} epochs)")
                break

    final_path = output_dir / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_macro_f1": val_f1,
        "val_primary_acc": val_acc,
    }, final_path)

    log_path = output_dir / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config or {},
            "best_val_macro_f1": round(best_f1, 4),
            "best_epoch": best_epoch,
            "early_stopped": epoch < epochs,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  {log_prefix} done. best val macro-F1={best_f1:.4f} (epoch {best_epoch})")
    print(f"  {log_prefix} best -> {output_dir / 'best.pt'}  log -> {log_path}")
    return history


def torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
