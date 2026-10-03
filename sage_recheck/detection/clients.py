"""Concrete HTTP clients for the built-in services."""
from __future__ import annotations

from sage_recheck.detection.base_client import BaseServiceClient


class HumanDetectClient(BaseServiceClient):
    def detect(self, image_b64: str) -> dict:
        """Returns {"boxes": [[x1,y1,x2,y2,score],...], "width": W, "height": H}."""
        return self._post("/detect", {"image": image_b64})

    def detect_objects(self, image_b64: str, classes: list[int] | None = None) -> dict:
        """Returns {"objects": [{"box": [...], "class_id": int, "class_name": str}, ...], ...}."""
        return self._post("/detect_objects", {"image": image_b64, "classes": classes or []})


class PoseClient(BaseServiceClient):
    def extract(self, image_b64: str, boxes: list[list[float]]) -> dict:
        """Returns {"persons": [{"box": [...], "keypoints": {id: [x,y,score]}}, ...]}."""
        return self._post("/pose", {"image": image_b64, "boxes": boxes})


class DepthClient(BaseServiceClient):
    def get_depth(self, image_b64: str, points: list[list[float]]) -> dict:
        """Returns {"depths": [float, ...]} aligned with *points*."""
        return self._post("/depth", {"image": image_b64, "points": points})


class FaceLandmarkClient(BaseServiceClient):
    def extract(self, image_b64: str, boxes: list[list[float]]) -> dict:
        """Returns {"persons": [{"detected": bool, "landmarks": [[x, y], ...] (106 pts)}, ...]}."""
        return self._post("/face", {"image": image_b64, "boxes": boxes})
