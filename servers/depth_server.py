"""Depth estimation service (Depth Anything V2, port 8003).

Env vars:
   SAGE_DEPTH_MODEL    model id or path
   SAGE_DEPTH_PORT     listen port (default: 8003)
   SAGE_SERVER_DEVICE  inference device (default: auto, cuda if available else cpu)
   SAGE_MODELS_CACHE_DIR  download cache dir (default: ./models)
"""
from __future__ import annotations

import os

import numpy as np

from fastapi import FastAPI

from servers.common.device import resolve_device
from servers.common.image_utils import decode_base64_image
from servers.common.model_loader import resolve_model_path
from servers.common.protocol import DepthRequest

app = FastAPI(title="SAGE Depth Estimation")
_pipe = None


@app.on_event("startup")
def _load():
    global _pipe
    from transformers import pipeline
    model_id = os.environ.get(
        "SAGE_DEPTH_MODEL",
        "models/hf/depth-anything-v2-small")
    device = resolve_device()
    model_path = resolve_model_path(model_id)
    _pipe = pipeline(task="depth-estimation", model=model_path, device=device)
    print(f"[depth] loaded {model_id} -> {model_path} on {device}")


@app.post("/depth")
def depth(req: DepthRequest) -> dict:
    img = decode_base64_image(req.image)
    result = _pipe(img)
    raw = result.get("predicted_depth")
    if raw is not None:
        dm = raw.cpu().numpy() if hasattr(raw, "cpu") else np.array(raw)
        dm = dm.squeeze()
    else:
        pil_depth = result.get("depth")
        dm = np.array(pil_depth, dtype=np.float32)
    dmin, dmax = float(dm.min()), float(dm.max())
    if dmax - dmin > 1e-6:
        dm = (dm - dmin) / (dmax - dmin)
    # predicted_depth: higher=farther; we want higher=closer -> invert
    dm = 1.0 - dm
    H, W = dm.shape
    sx = W / max(img.width, 1)
    sy = H / max(img.height, 1)
    depths = []
    for px, py in req.points:
        mx = max(0, min(W - 1, int(round(px * sx))))
        my = max(0, min(H - 1, int(round(py * sy))))
        depths.append(float(dm[my, mx]))
    return {"depths": depths}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SAGE_DEPTH_PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
