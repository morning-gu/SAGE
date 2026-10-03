"""Assembles raw extractor outputs into DetectionContext with PersonData list."""
from __future__ import annotations

from typing import Any

from sage_recheck.context import DetectedObject, DetectionContext, PersonData
from sage_recheck.features import FeatureEngine, calc_head_height, get_valid_keypoints
from sage_recheck.detection.extractors import DepthExtractor


class ContextAssembler:
    def __init__(self, feature_engine: FeatureEngine):
        self.fe = feature_engine

    def assemble(self, image_b64: str, raw_data: dict[str, Any],
                 errors: list[str]) -> DetectionContext:
        hd = raw_data.get("human_detect") or {}
        pose = (raw_data.get("pose") or {}).get("persons", [])
        depth = raw_data.get("depth") or []
        face = (raw_data.get("face_landmark") or {}).get("persons", [])
        boxes = hd.get("boxes", [])
        width = hd.get("width", 0)
        height = hd.get("height", 0)

        obj_raw = raw_data.get("object_detect") or {}
        objects = [
            DetectedObject(
                box=o.get("box", []),
                class_id=o.get("class_id", 0),
                class_name=o.get("class_name", ""),
            )
            for o in obj_raw.get("objects", [])
        ]

        persons: list[PersonData] = []
        for i, box in enumerate(boxes):
            kp_dict = pose[i].get("keypoints") if i < len(pose) else None
            kp_valid = get_valid_keypoints(kp_dict) if kp_dict else {}
            head_h = calc_head_height(kp_valid) if kp_valid else None
            raw_depth = depth[i] if i < len(depth) else None
            depth_info = DepthExtractor.map_depth(raw_depth, kp_dict)
            face_data = face[i] if i < len(face) else None
            # v5: detected face landmarks feed the eye-line fallback for the
            # head-roll angle when the pose ear geometry is missing/unusable.
            face_lm = (face_data.get("landmarks") or []
                       if face_data and face_data.get("detected") else [])
            features = (self.fe.compute(kp_dict, kp_valid, head_h, depth_info,
                                        face_lm=face_lm or None)
                        if kp_dict else None)
            persons.append(PersonData(
                box=box,
                keypoints=kp_dict,
                kp_valid=kp_valid,
                head_height=head_h,
                depth_info=depth_info,
                features=features,
                face_data=face_data,
            ))
        return DetectionContext(
            image_base64=image_b64,
            image_width=width,
            image_height=height,
            persons=persons,
            objects=objects,
            raw_data=raw_data,
            errors=errors,
        )
