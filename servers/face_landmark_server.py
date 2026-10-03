"""Face landmark service (InsightFace SCRFD + 106-point landmark, port 8005).

Detects faces using SCRFD and returns 106-point 2D landmarks per person.
No business logic (EAR, eye-closed, etc.) is computed here -- the
sage_recheck layer derives metrics from the returned landmarks.

Env vars:
   SAGE_FACE_MODEL         model pack name (default: buffalo_l)
   SAGE_FACE_PORT          listen port (default: 8005)
   SAGE_SERVER_DEVICE      inference device (default: auto, cuda if available else cpu)
   SAGE_MODELS_CACHE_DIR   download cache dir (default: ./models)
"""
from __future__ import annotations

import math
import os
import sys
import threading

import numpy as np

from fastapi import FastAPI

from servers.common.device import resolve_device
from servers.common.image_utils import decode_base64_image
from servers.common.model_loader import DEFAULT_CACHE_DIR
from servers.common.protocol import FaceLandmarkRequest

app = FastAPI(title="SAGE Face Landmark")
_app_lock = threading.Lock()
_face_app = None


@app.on_event("startup")
def _load():
    global _face_app
    from insightface.app import FaceAnalysis
    model_name = os.environ.get("SAGE_FACE_MODEL", "buffalo_l")
    # Land insightface packs under the shared SAGE_MODELS_CACHE_DIR (default
    # ./models) like the other small models, instead of insightface's own
    # ~/.insightface/models, so the cache stays centralized and reusable.
    insightface_root = os.path.join(DEFAULT_CACHE_DIR, "detectors", "insightface")
    device = resolve_device()
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    _face_app = FaceAnalysis(
        name=model_name,
        root=insightface_root,
        providers=providers,
        allowed_modules=["detection", "landmark_2d_106"],
    )
    _face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    model_dir = os.path.join(insightface_root, model_name)
    print(f"[face_landmark] loaded {model_name} -> {model_dir} on {device}")


# ROI cropping (v6): the detector runs at det_size=(640, 640).  On the
# 1280x720 classroom frames that halves every face (a 60px face becomes
# 30px) and landmark noise lands at ~1px ~ 2-3 deg of roll -- enough to
# drown the tilt true/false separation gap.  Instead of full-frame
# detection, each requested person box is cropped with margin and resized
# so the face fills the detector input; landmarks are mapped back to
# full-image coordinates.
_CROP_MAX_DIM = 640      # detector input cap per crop
_CROP_MARGIN = 0.35      # expand the person box by this fraction per side


def _crop_for_box(img, box):
    """Expanded person crop + params to map crop coords back to full image."""
    W, H = img.size
    bx1, by1, bx2, by2 = box
    mw = (bx2 - bx1) * _CROP_MARGIN
    mh = (by2 - by1) * _CROP_MARGIN
    ox = max(0, int(bx1 - mw))
    oy = max(0, int(by1 - mh))
    ox2 = min(W, int(bx2 + mw) + 1)
    oy2 = min(H, int(by2 + mh) + 1)
    crop = img.crop((ox, oy, ox2, oy2))
    cw, ch = crop.size
    scale = _CROP_MAX_DIM / max(cw, ch)
    if scale != 1.0:
        crop = crop.resize((max(1, round(cw * scale)),
                            max(1, round(ch * scale))))
    return crop, ox, oy, scale


@app.post("/face")
def face(req: FaceLandmarkRequest) -> dict:
    img = decode_base64_image(req.image)

    persons = []
    for box in req.boxes:
        bx1, by1, bx2, by2 = [int(round(v)) for v in box[:4]]
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        box_diag = math.hypot(bx2 - bx1, by2 - by1)

        crop, ox, oy, scale = _crop_for_box(img, box)
        rgb = np.array(crop.convert("RGB"))
        crop_bgr = rgb[:, :, ::-1].copy()

        with _app_lock:
            detected_faces = _face_app.get(crop_bgr)

        best = None
        best_dist = float("inf")
        for f in detected_faces:
            fx1, fy1, fx2, fy2 = [float(v) for v in f.bbox[:4]]
            fcx = (fx1 + fx2) / 2 / scale + ox
            fcy = (fy1 + fy2) / 2 / scale + oy
            d = math.hypot(fcx - bcx, fcy - bcy)
            if d < best_dist:
                best_dist = d
                best = f

        if best is None or best_dist > box_diag * 0.5:
            print(f"[face_landmark][dbg] MISS best_dist={best_dist:.1f} "
                  f"thresh={box_diag * 0.5:.1f}", flush=True)
            persons.append({"detected": False, "landmarks": []})
            continue

        lm = getattr(best, "landmark_2d_106", None)
        if lm is None:
            print("[face_landmark][dbg] landmarks_2d_106 missing", flush=True)
            persons.append({"detected": False, "landmarks": []})
            continue

        # map crop-space landmarks back to full-image coordinates
        landmarks = [[round(float(p[0]) / scale + ox, 1),
                      round(float(p[1]) / scale + oy, 1)] for p in lm]
        persons.append({"detected": True, "landmarks": landmarks})

    return {"persons": persons}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAGE_FACE_PORT", "8005"))
    uvicorn.run(app, host="0.0.0.0", port=port)
