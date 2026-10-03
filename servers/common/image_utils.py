"""Base64 image decoding utility."""
from __future__ import annotations

import base64
import io

from PIL import Image


def decode_base64_image(image_b64: str) -> Image.Image:
    """Decode a base64 string into a PIL RGB Image."""
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
