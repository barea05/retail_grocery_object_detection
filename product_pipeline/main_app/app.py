from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from requests import RequestException
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont

from product_pipeline.common import decode_base64_image, encode_pil_to_base64_jpeg, hsv_color, make_request_id, ensure_env_defaults


def pick_font(size: int = 14) -> ImageFont.ImageFont:
    # Use a bundled PIL default font; avoids filesystem dependency.
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def draw_visualization(
    image: Image.Image,
    objects: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    font = pick_font()

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        # Color boxes by detection class so different classes use different colors.
        class_name = str(obj.get("class_name", "unknown"))
        class_id = int(obj.get("class_id", -1))
        class_key = f"{class_id}:{class_name}"
        class_hash = int(hashlib.md5(class_key.encode("utf-8")).hexdigest()[:8], 16)
        r, g, b = hsv_color(class_hash)
        color = (r, g, b)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = class_name
        # Label background for readability.
        tx, ty = x1, max(0, y1 - 18)
        draw.rectangle([tx, ty, tx + 90, ty + 18], fill=color)
        draw.text((tx + 4, ty + 3), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, format="JPEG", quality=90)


def create_app() -> Flask:
    ensure_env_defaults()

    base_dir = Path(__file__).resolve().parent.parent  # product_pipeline/
    main_dir = Path(__file__).resolve().parent  # main_app/
    templates_dir = main_dir / "templates"
    static_dir = main_dir / "static"

    app = Flask(__name__, template_folder=str(templates_dir), static_folder=str(static_dir))
    app.config["JSON_SORT_KEYS"] = False

    visual_dir = base_dir / "outputs" / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index() -> Any:
        return render_template("index.html")

    @app.post("/api/infer")
    def infer() -> Any:
        # Accept multipart upload or raw JSON base64 payload.
        req_id = make_request_id("infer")

        if "image" in request.files:
            file = request.files["image"]
            img = Image.open(file.stream)
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_b64 = encode_pil_to_base64_jpeg(img)
            original_filename = file.filename
        else:
            payload_json = request.get_json(force=True)
            image_b64 = payload_json["image_base64"]
            original_filename = payload_json.get("filename", "uploaded")

        det_url = f"http://127.0.0.1:{os.getenv('DETECTOR_PORT','5001')}/detect"
        group_url = f"http://127.0.0.1:{os.getenv('GROUPER_PORT','5002')}/group"

        # Detection microservice
        det_payload = {
            "image_base64": image_b64,
            "min_confidence": float(request.form.get("min_confidence", 0.15)) if request.form else 0.15,
            "max_objects": int(request.form.get("max_objects", 120)) if request.form else 120,
            "imgsz": int(request.form.get("imgsz", 960)) if request.form else 960,
            "use_tiling": True,
            "tile_size": int(request.form.get("tile_size", 640)) if request.form else 640,
            "tile_overlap": float(request.form.get("tile_overlap", 0.25)) if request.form else 0.25,
            "nms_iou": float(request.form.get("nms_iou", 0.5)) if request.form else 0.5,
        }
        det_resp = requests.post(det_url, json=det_payload, timeout=int(os.getenv("HTTP_TIMEOUT_S", "120")))
        det_resp.raise_for_status()
        det_data = det_resp.json()

        detections = det_data.get("detections", [])

        # Grouping microservice
        group_payload = {
            "image_base64": image_b64,
            "detections": detections,
            "dbscan_eps": float(request.form.get("dbscan_eps", 0.18)) if request.form else 0.18,
            "dbscan_min_samples": int(request.form.get("dbscan_min_samples", 1)) if request.form else 1,
            "agglo_distance_threshold": float(request.form.get("agglo_distance_threshold", 0.22)) if request.form else 0.22,
            "ocr_conf_threshold": float(request.form.get("ocr_conf_threshold", 0.25)) if request.form else 0.25,
            "max_objects": int(request.form.get("max_objects", 120)) if request.form else 120,
        }
        try:
            group_resp = requests.post(group_url, json=group_payload, timeout=int(os.getenv("HTTP_TIMEOUT_S", "600")))
            group_resp.raise_for_status()
            group_data = group_resp.json()
            objects = group_data.get("objects", [])
        except RequestException:
            # Degrade gracefully when grouping service is slow (e.g., first-time OCR model download).
            # Keep pipeline alive by assigning all detections to a default group.
            objects = []
            for det in detections:
                objects.append(
                    {
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "group_index": 0,
                        "group_id": "brand_0",
                        "shape_tag": "unknown",
                        "ocr_text": "",
                        "ocr_text_key": "",
                        "ocr_confidence": 0.0,
                    }
                )

        # Visualization
        img_for_viz = decode_base64_image(image_b64)
        vis_name = f"{req_id}.jpg"
        out_path = visual_dir / vis_name
        draw_visualization(img_for_viz, objects, out_path)

        vis_url = f"/visualizations/{vis_name}"
        return jsonify(
            {
                "request_id": req_id,
                "source_filename": original_filename,
                "image_size": det_data.get("image_size", {}),
                "objects": objects,
                "visualization": {"url": vis_url, "file": vis_name},
            }
        )

    @app.get("/visualizations/<path:filename>")
    def get_visualization(filename: str) -> Any:
        return send_from_directory(str(visual_dir), filename)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    port = int(os.getenv("MAIN_PORT", "5000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)

