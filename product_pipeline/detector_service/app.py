from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from product_pipeline.common import decode_base64_image, resize_to_max_dim


def create_app() -> Flask:
    app = Flask(__name__)

    app.config["JSON_SORT_KEYS"] = False

    # Lazily load model to keep startup fast when just importing.
    model = {"obj": None}

    def load_model() -> Any:
        if model["obj"] is not None:
            return model["obj"]

        # Local, lightweight detection model.
        # YOLOv8n is fast and widely used; it is trained on COCO but works as a proxy for "product" regions.
        try:
            from ultralytics import YOLO

            weights = os.getenv("DETECTOR_WEIGHTS", "yolov8n.pt")
            model["obj"] = YOLO(weights)
            return model["obj"]
        except Exception:
            # Fallback: allow the demo to run without heavy model dependencies.
            model["obj"] = None
            return None

    @app.post("/detect")
    def detect() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True)
        image_base64: str = payload["image_base64"]
        min_confidence: float = float(payload.get("min_confidence", 0.25))
        max_objects: int = int(payload.get("max_objects", 50))

        img = decode_base64_image(image_base64)
        resized, orig_size, resized_size = resize_to_max_dim(img, max_dim=int(payload.get("max_dim", 1024)))

        model_obj = load_model()

        if model_obj is None:
            # Grid fallback: split image into candidate product regions.
            w, h = orig_size
            cols = 3
            rows = 3
            # Adjust density to obey max_objects.
            while (cols * rows) > max_objects and cols > 1:
                cols -= 1
            while (cols * rows) > max_objects and rows > 1:
                rows -= 1

            detections: List[Dict[str, Any]] = []
            cell_w = w / float(cols)
            cell_h = h / float(rows)
            margin = 0.06  # relative margin inside each cell
            for r in range(rows):
                for c in range(cols):
                    x1 = c * cell_w + cell_w * margin
                    y1 = r * cell_h + cell_h * margin
                    x2 = (c + 1) * cell_w - cell_w * margin
                    y2 = (r + 1) * cell_h - cell_h * margin
                    detections.append(
                        {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": 0.1,
                            "class_id": -1,
                            "class_name": "product",
                        }
                    )
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            return jsonify({"image_size": {"width": w, "height": h}, "detections": detections[:max_objects]})

        # Ultralytics handles resizing internally; results are in resized image coordinates.
        # We'll trust xyxy returned as absolute coords for the input image.
        try:
            use_tiling = bool(payload.get("use_tiling", True))
            if use_tiling:
                from torchvision.ops import nms
                import torch

                tile_size = int(payload.get("tile_size", 640))
                tile_overlap = float(payload.get("tile_overlap", 0.25))
                stride = max(64, int(round(tile_size * (1.0 - tile_overlap))))

                width, height = resized.size
                all_boxes: List[List[float]] = []
                all_scores: List[float] = []
                all_classes: List[int] = []

                # Slide over the image to recover small shelf products.
                for y in range(0, max(1, height), stride):
                    for x in range(0, max(1, width), stride):
                        x2 = min(width, x + tile_size)
                        y2 = min(height, y + tile_size)
                        if (x2 - x) < 64 or (y2 - y) < 64:
                            continue

                        tile = resized.crop((x, y, x2, y2))
                        tile_results = model_obj.predict(
                            tile,
                            imgsz=int(payload.get("imgsz", 960)),
                            conf=min_confidence,
                            max_det=max_objects,
                            verbose=False,
                        )
                        if not tile_results:
                            continue
                        tr = tile_results[0]
                        tb = getattr(tr, "boxes", None)
                        if tb is None or tb.xyxy is None:
                            continue

                        txy = tb.xyxy.cpu().numpy()
                        tsc = tb.conf.cpu().numpy()
                        tcl = tb.cls.cpu().numpy().astype(int)

                        for i in range(len(txy)):
                            bx1, by1, bx2, by2 = txy[i].tolist()
                            all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                            all_scores.append(float(tsc[i]))
                            all_classes.append(int(tcl[i]))

                if not all_boxes:
                    results = []
                else:
                    # Class-aware NMS by offsetting boxes per class.
                    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
                    scores_t = torch.tensor(all_scores, dtype=torch.float32)
                    classes_t = torch.tensor(all_classes, dtype=torch.float32)
                    offsets = classes_t.view(-1, 1) * 4096.0
                    keep = nms(boxes_t + offsets.repeat(1, 4), scores_t, iou_threshold=float(payload.get("nms_iou", 0.5)))
                    keep = keep[:max_objects].cpu().numpy().tolist()

                    class_names = getattr(model_obj, "names", {}) or {}
                    detections: List[Dict[str, Any]] = []
                    scale_x = orig_size[0] / float(resized_size[0])
                    scale_y = orig_size[1] / float(resized_size[1])
                    for k in keep:
                        x1, y1, x2, y2 = all_boxes[k]
                        cls_id = all_classes[k]
                        detections.append(
                            {
                                "bbox": [float(x1 * scale_x), float(y1 * scale_y), float(x2 * scale_x), float(y2 * scale_y)],
                                "confidence": float(all_scores[k]),
                                "class_id": int(cls_id),
                                "class_name": str(class_names.get(cls_id, cls_id)),
                            }
                        )

                    detections.sort(key=lambda d: d["confidence"], reverse=True)
                    return jsonify({"image_size": {"width": orig_size[0], "height": orig_size[1]}, "detections": detections[:max_objects]})
            else:
                results = model_obj.predict(
                    resized,
                    imgsz=int(payload.get("imgsz", 960)),
                    conf=min_confidence,
                    max_det=max_objects,
                    verbose=False,
                )
        except Exception:
            # If inference fails, also fall back to grid boxes.
            model_obj = None
            w, h = orig_size
            cols = 3
            rows = 3
            while (cols * rows) > max_objects and cols > 1:
                cols -= 1
            while (cols * rows) > max_objects and rows > 1:
                rows -= 1

            detections: List[Dict[str, Any]] = []
            cell_w = w / float(cols)
            cell_h = h / float(rows)
            margin = 0.06
            for r in range(rows):
                for c in range(cols):
                    x1 = c * cell_w + cell_w * margin
                    y1 = r * cell_h + cell_h * margin
                    x2 = (c + 1) * cell_w - cell_w * margin
                    y2 = (r + 1) * cell_h - cell_h * margin
                    detections.append(
                        {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": 0.1,
                            "class_id": -1,
                            "class_name": "product",
                        }
                    )
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            return jsonify({"image_size": {"width": w, "height": h}, "detections": detections[:max_objects]})

        if not results:
            return jsonify({"image_size": {"width": orig_size[0], "height": orig_size[1]}, "detections": []})

        res = results[0]
        detections: List[Dict[str, Any]] = []
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return jsonify({"image_size": {"width": orig_size[0], "height": orig_size[1]}, "detections": []})

        xys = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        clss = boxes.cls.cpu().numpy().astype(int).tolist()

        class_names = getattr(model_obj, "names", {}) or {}

        # Map coords from resized back to original if we resized.
        scale_x = orig_size[0] / float(resized_size[0])
        scale_y = orig_size[1] / float(resized_size[1])

        for i, (x1, y1, x2, y2) in enumerate(xys):
            x1o = x1 * scale_x
            x2o = x2 * scale_x
            y1o = y1 * scale_y
            y2o = y2 * scale_y
            cls_id = clss[i]
            detections.append(
                {
                    "bbox": [float(x1o), float(y1o), float(x2o), float(y2o)],
                    "confidence": float(confs[i]),
                    "class_id": int(cls_id),
                    "class_name": str(class_names.get(cls_id, cls_id)),
                }
            )

        # Final cap by confidence to control payload sizes.
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        detections = detections[:max_objects]

        return jsonify({"image_size": {"width": orig_size[0], "height": orig_size[1]}, "detections": detections})

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    port = int(os.getenv("DETECTOR_PORT", "5001"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)

