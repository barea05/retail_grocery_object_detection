from __future__ import annotations

import base64
import io
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import PIL.Image as Image


SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


def make_request_id(prefix: str = "req") -> str:
    return f"{prefix}_{secrets.token_hex(8)}_{int(time.time())}"


def decode_base64_image(image_base64: str) -> Image.Image:
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def encode_pil_to_base64_jpeg(image: Image.Image, quality: int = 85) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    # CLIP and OpenCV pipelines commonly use numpy arrays.
    # We keep it RGB here; OpenCV color conversions are handled where needed.
    return np.asarray(image)


def resize_to_max_dim(image: Image.Image, max_dim: int = 1024) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
    w, h = image.size
    if max(w, h) <= max_dim:
        return image, (w, h), (w, h)

    scale = max_dim / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, (w, h), (new_w, new_h)


def clamp_int(x: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(x))))


def normalize_bboxes_xyxy(bboxes: Iterable[Iterable[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for b in bboxes:
        x1, y1, x2, y2 = b
        out.append([float(x1), float(y1), float(x2), float(y2)])
    return out


def hsv_color(i: int) -> Tuple[int, int, int]:
    # Generate distinct bright colors deterministically.
    # i is a cluster index.
    import colorsys

    h = (i * 0.61803398875) % 1.0  # golden ratio spacing
    s = 0.75
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def ensure_env_defaults() -> None:
    # Keep defaults local; no hardcoded filesystem paths.
    os.environ.setdefault("DETECTOR_PORT", "5001")
    os.environ.setdefault("GROUPER_PORT", "5002")
    os.environ.setdefault("MAIN_PORT", "5000")

