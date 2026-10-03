"""DetectionRunner: topological-sort extractors, run, assemble context."""
from __future__ import annotations

import os
from typing import Any

from sage_recheck.detection.assembler import ContextAssembler
from sage_recheck.detection.base import DataExtractor
from sage_recheck.detection.clients import (
    DepthClient,
    FaceLandmarkClient,
    HumanDetectClient,
    PoseClient,
)
from sage_recheck.detection.extractors import (
    DepthExtractor,
    FaceLandmarkExtractor,
    HumanDetectExtractor,
    ObjectDetectExtractor,
    PoseExtractor,
)
from sage_recheck.context import DetectionContext
from sage_recheck.features import FeatureEngine


class DetectionRunner:
    def __init__(self, extractors: list[DataExtractor],
                 feature_engine: FeatureEngine | None = None):
        self.extractors = {e.name: e for e in extractors}
        self.feature_engine = feature_engine or FeatureEngine()
        self.assembler = ContextAssembler(self.feature_engine)
        self._order = self._topo_sort()

    def _topo_sort(self) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str):
            if name in visited or name not in self.extractors:
                return
            visited.add(name)
            for dep in self.extractors[name].depends_on:
                visit(dep)
            order.append(name)

        for name in self.extractors:
            visit(name)
        return order

    def run(self, image_b64: str) -> DetectionContext:
        raw_data: dict[str, Any] = {}
        errors: list[str] = []
        for name in self._order:
            ext = self.extractors[name]
            prior = {dep: raw_data[dep] for dep in ext.depends_on
                     if dep in raw_data}
            try:
                raw_data[name] = ext.extract(image_b64, prior)
            except Exception as e:
                raw_data[name] = None
                errors.append(f"{name}: {e}")
        return self.assembler.assemble(image_b64, raw_data, errors)

    @staticmethod
    def from_env() -> "DetectionRunner":
        timeout = float(os.environ.get("SAGE_RECHECK_TIMEOUT", "30"))
        hd_url = os.environ.get("SAGE_HUMAN_DETECT_URL", "http://localhost:8001")
        pose_url = os.environ.get("SAGE_POSE_URL", "http://localhost:8002")
        depth_url = os.environ.get("SAGE_DEPTH_URL", "http://localhost:8003")
        face_url = os.environ.get("SAGE_FACE_URL", "http://localhost:8005")
        # Object detection reuses the human-detect server (方案 A);
        # override only when switching to a dedicated server.
        object_url = os.environ.get("SAGE_OBJECT_DETECT_URL", hd_url)
        extractors: list[DataExtractor] = [
            HumanDetectExtractor(HumanDetectClient(hd_url, timeout)),
            PoseExtractor(PoseClient(pose_url, timeout)),
            ObjectDetectExtractor(HumanDetectClient(object_url, timeout)),
            FaceLandmarkExtractor(FaceLandmarkClient(face_url, timeout)),
        ]
        if depth_url:
            extractors.append(DepthExtractor(DepthClient(depth_url, timeout)))
        return DetectionRunner(extractors, FeatureEngine())
