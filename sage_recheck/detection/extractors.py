"""Built-in data extractors wrapping the service clients."""
from __future__ import annotations

from typing import Any

from sage_recheck.detection.base import DataExtractor
from sage_recheck.detection.clients import (
    DepthClient,
    FaceLandmarkClient,
    HumanDetectClient,
    PoseClient,
)

# COCO class IDs relevant to item-verification behaviors.
# phone: cell phone (67)
# toy: teddy bear (77)
# snack: banana(46) apple(47) sandwich(48) orange(49) hot dog(52) pizza(53)
#        donut(54) cake(55) bottle(39) cup(41) bowl(45)
OBJECT_CLASS_MAP: dict[str, list[int]] = {
    "phone": [67],
    "toy": [77],
    "snack": [46, 47, 48, 49, 52, 53, 54, 55, 39, 41, 45],
}
ALL_OBJECT_CLASSES = sorted(set(c for cls in OBJECT_CLASS_MAP.values() for c in cls))


class HumanDetectExtractor(DataExtractor):
    name = "human_detect"

    def __init__(self, client: HumanDetectClient):
        self.client = client

    def extract(self, image_b64: str, prior: dict[str, Any]) -> dict:
        return self.client.detect(image_b64)


class ObjectDetectExtractor(DataExtractor):
    name = "object_detect"

    def __init__(self, client: HumanDetectClient):
        self.client = client

    def extract(self, image_b64: str, prior: dict[str, Any]) -> dict:
        return self.client.detect_objects(image_b64, ALL_OBJECT_CLASSES)


class PoseExtractor(DataExtractor):
    name = "pose"
    depends_on = ["human_detect"]

    def __init__(self, client: PoseClient):
        self.client = client

    def extract(self, image_b64: str, prior: dict[str, Any]) -> dict:
        hd = prior.get("human_detect") or {}
        boxes = [b[:4] for b in hd.get("boxes", []) if len(b) >= 4]
        if not boxes:
            return {"persons": []}
        return self.client.extract(image_b64, boxes)


class FaceLandmarkExtractor(DataExtractor):
    name = "face_landmark"
    depends_on = ["human_detect"]

    def __init__(self, client: FaceLandmarkClient):
        self.client = client

    def extract(self, image_b64: str, prior: dict[str, Any]) -> dict:
        hd = prior.get("human_detect") or {}
        boxes = [b[:4] for b in hd.get("boxes", []) if len(b) >= 4]
        if not boxes:
            return {"persons": []}
        return self.client.extract(image_b64, boxes)


class DepthExtractor(DataExtractor):
    name = "depth"
    depends_on = ["pose"]

    def __init__(self, client: DepthClient):
        self.client = client

    def extract(self, image_b64: str, prior: dict[str, Any]) -> list:
        persons = (prior.get("pose") or {}).get("persons", [])
        results = []
        for person in persons:
            kp = person.get("keypoints") or {}
            points = self._extract_targets(kp)
            if not points:
                results.append(None)
                continue
            try:
                resp = self.client.get_depth(image_b64, points)
                results.append({"points": points, "depths": resp.get("depths", [])})
            except Exception:
                results.append(None)
        return results

    @staticmethod
    def _extract_targets(kp_dict: dict) -> list[list[float]]:
        pts = []
        if "5" in kp_dict:
            pts.append([kp_dict["5"][0], kp_dict["5"][1]])
        if "6" in kp_dict:
            pts.append([kp_dict["6"][0], kp_dict["6"][1]])
        if "5" in kp_dict and "6" in kp_dict:
            pts.append([(kp_dict["5"][0] + kp_dict["6"][0]) / 2,
                        (kp_dict["5"][1] + kp_dict["6"][1]) / 2])
        if "0" in kp_dict:
            pts.append([kp_dict["0"][0], kp_dict["0"][1]])
        return pts

    @staticmethod
    def map_depth(raw: dict | None, kp_dict: dict | None) -> dict:
        if not raw or not kp_dict:
            return {}
        depths = raw.get("depths", [])
        info = {}
        i = 0
        if "5" in kp_dict and i < len(depths):
            info["left_depth"] = depths[i]; i += 1
        if "6" in kp_dict and i < len(depths):
            info["right_depth"] = depths[i]; i += 1
        if "5" in kp_dict and "6" in kp_dict and i < len(depths):
            info["mid_depth"] = depths[i]; i += 1
        if "0" in kp_dict and i < len(depths):
            info["nose_depth"] = depths[i]
        return info
