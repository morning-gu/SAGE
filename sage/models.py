"""Model factories shared by training and eval scripts.

Single source of truth for baseline architectures so the checkpoint saved by
a train script can always be rebuilt and loaded by its eval counterpart.
"""
from __future__ import annotations

from sage.metrics import NUM_CLASSES  # noqa: E402


def build_yolo_bce(weights: str = "models/detectors/yolov8s-cls.pt",
                   num_classes: int = NUM_CLASSES):
    """YOLOv8s-cls backbone + fresh sigmoid-BCE multi-label head.

    Backbone = everything before the Classify head PLUS the Classify head's
    own Conv(c1->1280) + AdaptiveAvgPool2d: in ultralytics the pooling lives
    inside Classify.forward, so layers[:-1] alone would return an unpooled
    (B, 512, 7, 7) feature map (the 57344x7 matmul crash). The pretrained
    conv weights are kept; only the final Linear(1280, num_classes) is fresh.

    Returns the full nn.Sequential(backbone, head). Raises RuntimeError with
    an actionable message if the ultralytics internal layout changed.
    """
    import torch.nn as nn
    from ultralytics import YOLO

    inner = getattr(YOLO(weights).model, "model", None)
    if inner is None or not isinstance(inner, nn.Sequential):
        raise RuntimeError(
            "ultralytics internal layout changed: YOLO(...).model.model is "
            "not an nn.Sequential. Pin the ultralytics version and update "
            "sage.models.build_yolo_bce().")
    layers = list(inner.children())
    head = layers[-1]
    if head.__class__.__name__ != "Classify":
        raise RuntimeError(
            f"expected last module to be Classify, got {head.__class__.__name__};"
            " update sage.models.build_yolo_bce() for this ultralytics version.")
    conv = getattr(head, "conv", None)
    pool = getattr(head, "pool", None)
    linear = getattr(head, "linear", None)
    if conv is None or pool is None or linear is None \
            or not hasattr(linear, "in_features"):
        raise RuntimeError("Classify head lacks .conv/.pool/.linear; update "
                           "sage.models.build_yolo_bce().")
    backbone = nn.Sequential(*layers[:-1], conv, pool, nn.Flatten())
    return nn.Sequential(backbone, nn.Linear(linear.in_features, num_classes))
