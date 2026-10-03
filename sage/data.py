"""Shared data pipeline for the sigmoid multi-label baselines.

Single source of truth for loading the SAGE annotations (2014 samples,
15-class multi-label, stratified 70/15/15 splits) and building image
datasets/transforms.  Previously inlined in scripts/train_cnn_mlb.py;
now shared with scripts/train_yolo_bce.py so all sigmoid baselines train
on the identical data pipeline (fairness protocol, docs/ablation_protocol.md).

Taxonomy is imported from sage_check/constants.py (single source);
metrics live in sage/metrics.py.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
from sage.taxonomy import BEHAVIOR_NAMES  # noqa: F401  (re-export)

NUM_CLASSES = len(BEHAVIOR_NAMES)  # noqa: E402
NAME_TO_INDEX = {n: i for i, n in enumerate(BEHAVIOR_NAMES)}  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_annotations(path: str | Path, split: str = "") -> list[dict]:
    """Load annotations-style JSON, default-fill missing labels, filter split."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        for name in BEHAVIOR_NAMES:
            s["labels"].setdefault(name, 0)
    if split:
        data = [s for s in data if s.get("split") == split]
    return data


def label_vector(sample: dict) -> list[float]:
    """15-dim 0/1 vector in BEHAVIOR_NAMES order."""
    return [float(sample["labels"].get(n, 0)) for n in BEHAVIOR_NAMES]


def build_transform(imgsz: int = 224, train: bool = True,
                    normalize: str = "imagenet"):
    """Build a torchvision transform.

    train:   Resize(imgsz+32) + RandomCrop(imgsz) + HorizontalFlip
    eval:    Resize(imgsz+32) + CenterCrop(imgsz)
    (at imgsz=224 this reproduces the historical Resize(256)+Crop(224) recipe)

    normalize: "imagenet" (CNN backbones pretrained on ImageNet) or
               "scale01" (YOLOv8 backbones, trained on /255 inputs).
    """
    from torchvision import transforms

    if normalize == "imagenet":
        norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    elif normalize == "scale01":
        norm = transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    else:
        raise ValueError(f"unknown normalize mode: {normalize}")
    pad = imgsz + 32
    if train:
        return transforms.Compose([
            transforms.Resize(pad),
            transforms.RandomCrop(imgsz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(pad),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        norm,
    ])


class SAGEMLDataset:
    """Multi-label dataset over annotations records.

    Each item yields (image_tensor, label_vector[15], sample_id).
    Kept as a plain class (not inheriting torch Dataset) so importing this
    module never requires torch — heavy imports happen in __init__.
    """

    def __init__(self, annotations: list[dict], image_root: Path | str,
                 transform=None):
        from torch.utils.data import Dataset as _TorchDataset  # noqa: F401
        from PIL import Image  # noqa: F401  (fail fast if PIL missing)

        self.samples = annotations
        self.image_root = Path(image_root)
        self.transform = transform
        self._Image = Image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = s["image_path"]
        if not Path(img_path).is_absolute():
            img_path = str(self.image_root / img_path)
        img = self._Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        import torch
        labels = torch.tensor(label_vector(s), dtype=torch.float32)
        return img, labels, s["sample_id"]


def make_loaders(train_anns: list[dict], val_anns: list[dict],
                 image_root: Path | str, imgsz: int = 224, batch: int = 16,
                 num_workers: int = 4, normalize: str = "imagenet"):
    """Train/val DataLoader pair with the shared transform recipe."""
    from torch.utils.data import DataLoader

    train_ds = SAGEMLDataset(train_anns, image_root,
                             build_transform(imgsz, train=True, normalize=normalize))
    val_ds = SAGEMLDataset(val_anns, image_root,
                           build_transform(imgsz, train=False, normalize=normalize))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
