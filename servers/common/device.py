"""Inference device selection (cuda when available, else cpu)."""
from __future__ import annotations

import os


def resolve_device() -> str:
    """Return the inference device.

    Honors an explicit ``SAGE_SERVER_DEVICE`` override; otherwise auto-selects
    ``cuda`` when a CUDA-capable torch is available and falls back to ``cpu``.
    """
    explicit = os.environ.get("SAGE_SERVER_DEVICE")
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
