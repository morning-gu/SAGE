"""SAGE fusion layer: combines VLM detection + small-model re-check.

This package init stays deliberately light: ``sage.taxonomy`` is the domain
core imported by every other layer (including sage_check), so importing it
must not pull in the fusion/check dependencies.  The heavier names
(``FusionEngine``, ``FinalResult``, ``SAGEPipeline``) are resolved lazily on
first attribute access.
"""
from __future__ import annotations

__all__ = ["detect", "FinalResult", "FusionEngine", "SAGEPipeline"]


def detect(image_base64: str, mode: str = "no_reason",
           fusion=None):
    """Fuse large-model detection and small-model re-check for a single frame.

    Uses a process-wide shared pipeline (engines built once, not per frame).
    Args:
        image_base64: single-frame image encoded as base64 string.
        mode: "no_reason" (code only) or "reason" (code + explanation).
        fusion: optional pre-configured FusionEngine; a custom engine gets
            its own throwaway pipeline, the shared one is untouched.
    Returns:
        FinalResult with final_code, confidence, source, and evidence.
    """
    from sage.pipeline import SAGEPipeline, get_pipeline

    if fusion is not None:
        return SAGEPipeline(fusion=fusion).detect(image_base64, mode)
    return get_pipeline().detect(image_base64, mode)


def __getattr__(name: str):
    if name == "FusionEngine":
        from sage.fuse import FusionEngine
        return FusionEngine
    if name == "FinalResult":
        from sage.result import FinalResult
        return FinalResult
    if name == "SAGEPipeline":
        from sage.pipeline import SAGEPipeline
        return SAGEPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
