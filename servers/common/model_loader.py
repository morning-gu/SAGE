"""Model download/loading utility (ModelScope or HuggingFace).

All downloaded models land under ``SAGE_MODELS_CACHE_DIR`` (default ``./models``)
so the cache stays centralized instead of scattered across framework defaults.
Models already placed in that cache dir are reused as-is to avoid re-downloads.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CACHE_DIR = os.environ.get("SAGE_MODELS_CACHE_DIR", "./models")


def _exists(path: str) -> bool:
    return Path(path).is_dir() or os.path.exists(path)


def resolve_model_path(model_id: str, cache_dir: str | None = None) -> str:
    """Return a local path for *model_id*, reusing the cache or downloading.

    Resolution order:
      1. *model_id* itself is an existing local file/dir -> returned as-is.
      2. ``<cache_dir>/<model_id>`` exists -> returned (reuses a previously
         downloaded/copied model, preventing repeated downloads).
      3. *model_id* is a repo id (contains ``/``) -> downloaded into
         *cache_dir* (default ``SAGE_MODELS_CACHE_DIR`` or ``./models``),
         preferring ModelScope and falling back to HuggingFace.
      4. A bare name without ``/`` -> returned unchanged so the downstream
         framework (e.g. ultralytics) can resolve it itself.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    if _exists(model_id):
        return model_id
    cached = os.path.join(cache_dir, model_id)
    if _exists(cached):
        print(f"[model_loader] reuse cached {model_id} -> {cached}")
        return cached
    if "/" not in model_id:
        return model_id
    print(f"[model_loader] downloading {model_id} -> {cache_dir}")
    try:
        from modelscope import snapshot_download
        return snapshot_download(model_id, cache_dir=cache_dir)
    except ImportError:
        pass
    except Exception as e:
        print(f"[model_loader] ModelScope failed ({e}); trying HuggingFace")
    from huggingface_hub import snapshot_download as hf_download
    return hf_download(model_id, cache_dir=cache_dir)
