"""Skeleton keypoint service (YOLO-pose, port 8002).

Env vars:
   SAGE_POSE_MODEL    model id or path (default: models/detectors/yolov8x-pose.pt)
   SAGE_POSE_PORT     listen port (default: 8002)
   SAGE_SERVER_DEVICE  inference device (default: auto, cuda if available else cpu)
   SAGE_MODELS_CACHE_DIR  download cache dir (default: ./models)
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from servers.common.image_utils import decode_base64_image
from servers.common.device import resolve_device
from servers.common.model_loader import resolve_model_path
from servers.common.protocol import PoseRequest

app = FastAPI(title="SAGE Pose Estimation")
_model = None
_device = "cpu"


@app.on_event("startup")
def _load():
    global _model, _device
    from ultralytics import YOLO
    model_id = os.environ.get("SAGE_POSE_MODEL", "models/detectors/yolov8x-pose.pt")
    _device = resolve_device()
    model_path = resolve_model_path(model_id)
    # YOLO() ctor has no device kwarg on current ultralytics; pass it per-inference.
    _model = YOLO(model_path)
    print(f"[pose] loaded {model_id} -> {model_path} on {_device}")


@app.post("/pose")
def pose(req: PoseRequest) -> dict:
    img = decode_base64_image(req.image)
    persons = []
    if req.boxes:
        for box in req.boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                persons.append({"box": box[:4], "keypoints": {}})
                continue
            crop = img.crop((x1, y1, x2, y2))
            results = _model(crop, verbose=False, device=_device)
            kp = _extract_kp(results, offset=(x1, y1))
            persons.append({"box": box[:4], "keypoints": kp})
    else:
        results = _model(img, verbose=False, device=_device)
        for r in results:
            if r.keypoints is None or len(r.keypoints) == 0:
                continue
            if r.keypoints.data.shape[0] == 0:
                continue
            kpts = r.keypoints.data[0]
            kp = {}
            for i in range(min(17, kpts.shape[0])):
                x, y, s = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
                if s > 0:
                    kp[str(i)] = [x, y, s]
            if r.boxes is not None and len(r.boxes) > 0 and len(r.keypoints) > 0:
                xyxy = r.boxes.xyxy[0].tolist()
                persons.append({"box": xyxy, "keypoints": kp})
            else:
                persons.append({"box": [], "keypoints": kp})
    return {"persons": persons}


def _extract_kp(results, offset=(0, 0)) -> dict:
    ox, oy = offset
    kp = {}
    for r in results:
        if r.keypoints is None or len(r.keypoints) == 0:
            continue
        if r.keypoints.data.shape[0] == 0:
            continue
        kpts = r.keypoints.data[0]
        for i in range(min(17, kpts.shape[0])):
            x, y, s = float(kpts[i, 0]) + ox, float(kpts[i, 1]) + oy, float(kpts[i, 2])
            if s > 0:
                kp[str(i)] = [x, y, s]
    return kp


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAGE_POSE_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
