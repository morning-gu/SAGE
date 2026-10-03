"""Canonical filesystem paths and service URLs — one place, no repetition.

Every default that used to be hardcoded in individual script argparse
defaults (7+ copies of the annotations path, the weights path, ports, ...)
comes from here.  All paths are absolute, resolved from the repo root.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# -- Data --------------------------------------------------------------------
DATA_DIR = REPO / "data"
GT_ANNOTATIONS = DATA_DIR / "sage_eval" / "annotations.json"

# -- Weights / artifacts -----------------------------------------------------
CHECKPOINTS_DIR = REPO / "checkpoints"
FUSE_WEIGHTS = CHECKPOINTS_DIR / "fuse_lr_weights.json"
MODELS_DIR = REPO / "models"
DETECTORS_DIR = MODELS_DIR / "detectors"
HF_MODELS_DIR = MODELS_DIR / "hf"

CHECKPOINTS_ROOT = REPO / "checkpoints" / "experiments"
RESULTS_ROOT = REPO / "results" / "experiments"

# -- Registry ----------------------------------------------------------------
CONFIG_DIR = REPO / "configs" / "experiments"

# -- Services ----------------------------------------------------------------
DEFAULT_VLM_URL = "http://localhost:8010"
