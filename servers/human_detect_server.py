"""Human detection service (YOLO, port 8001).

Also serves object detection via /detect_objects reusing the same YOLO model.

Env vars:
   SAGE_HUMAN_DETECT_MODEL  model id or path (default: models/detectors/yolov8n.pt)
   SAGE_HUMAN_DETECT_PORT   listen port (default: 8001)
   SAGE_SERVER_DEVICE       inference device (default: auto, cuda if available else cpu)
   SAGE_MODELS_CACHE_DIR    download cache dir (default: ./models)
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from servers.common.image_utils import decode_base64_image
from servers.common.device import resolve_device
from servers.common.model_loader import resolve_model_path
from servers.common.protocol import DetectRequest, DetectObjectsRequest

app = FastAPI(title="SAGE Human Detection")
_model = None
_device = "cpu"


@app.on_event("startup")
def _load():
    global _model, _device
    from ultralytics import YOLO
    model_id = os.environ.get("SAGE_HUMAN_DETECT_MODEL", "models/detectors/yolov8n.pt")
    _device = resolve_device()
    model_path = resolve_model_path(model_id)
    # YOLO() ctor has no device kwarg on current ultralytics; pass it per-inference.
    _model = YOLO(model_path)
    print(f"[human_detect] loaded {model_id} -> {model_path} on {_device}")


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    img = decode_base64_image(req.image)
    results = _model(img, classes=[0], verbose=False, device=_device)
    boxes = []
    for r in results:
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            boxes.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])])
    return {"boxes": boxes, "width": img.width, "height": img.height}


@app.post("/detect_objects")
def detect_objects(req: DetectObjectsRequest) -> dict:
    """Detect objects of specified COCO classes, reusing the same YOLO model."""
    img = decode_base64_image(req.image)
    classes = req.classes if req.classes else None
    results = _model(img, classes=classes, verbose=False, device=_device)
    objects = []
    for r in results:
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            objects.append({
                "box": [xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])],
                "class_id": cls_id,
                "class_name": _model.names.get(cls_id, ""),
            })
    return {"objects": objects, "width": img.width, "height": img.height}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAGE_HUMAN_DETECT_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
