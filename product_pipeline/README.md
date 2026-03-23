# Product Brand Grouping Pipeline (Flask + Microservices)

This folder contains an end-to-end demo that:
1. Accepts an image via a Flask web UI / API
2. Detects multiple product-like objects on shelves (Detector microservice)
3. Groups detections into brand-like groups (Grouping microservice)
4. Returns JSON with bounding boxes + a stable `group_id`
5. Saves a color-coded visualization image to disk

## Architecture

- `main_app` (Flask): upload UI + orchestrator
- `detector_service` (Flask): YOLO object detection
- `grouping_service` (Flask): CLIP embedding + DBSCAN clustering

## Model choices (open-source)

- Detection: `ultralytics/yolov8n.pt` (fast default YOLOv8)
- Grouping: `open_clip_torch` with CLIP `ViT-B-32` embeddings

## JSON Formats

### Main API: `POST /api/infer`

Accepts:
- `multipart/form-data` with fields:
  - `image`: the uploaded image file
  - `min_confidence` (optional, default `0.25`)
  - `max_objects` (optional, default `50`)
  - `imgsz` (optional, default `640`)
  - `dbscan_eps` (optional, default `0.25`)
  - `dbscan_min_samples` (optional, default `2`)
- OR JSON with:
  - `image_base64`: base64-encoded image bytes
  - `filename` (optional)

Response JSON:
```json
{
  "request_id": "infer_<...>",
  "source_filename": "uploaded.jpg",
  "image_size": {"width": 1024, "height": 768},
  "objects": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.62,
      "class_id": 39,
      "class_name": "bottle",
      "group_index": 0,
      "group_id": "brand_0"
    }
  ],
  "visualization": {
    "url": "/visualizations/<request_id>.jpg",
    "file": "<request_id>.jpg"
  }
}
```

### Detector microservice: `POST /detect`

Request JSON:
```json
{
  "image_base64": "<...>",
  "min_confidence": 0.25,
  "max_objects": 50,
  "imgsz": 640,
  "max_dim": 1024
}
```

Response JSON:
```json
{
  "image_size": {"width": 1024, "height": 768},
  "detections": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.62, "class_id": 39, "class_name": "bottle"}
  ]
}
```

### Grouping microservice: `POST /group`

Request JSON:
```json
{
  "image_base64": "<...>",
  "detections": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.62, "class_id": 39, "class_name": "bottle"}
  ],
  "dbscan_eps": 0.25,
  "dbscan_min_samples": 2,
  "max_objects": 50
}
```

Response JSON:
```json
{
  "objects": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.62,
      "class_id": 39,
      "class_name": "bottle",
      "group_index": 0,
      "group_id": "brand_0"
    }
  ],
  "groups": [
    {
      "group_index": 0,
      "group_id": "brand_0",
      "member_indices": [0, 3],
      "color": {"r": 255, "g": 120, "b": 10}
    }
  ]
}
```

## Visualization output

The main app saves images here:
`inflect_project/product_pipeline/outputs/visualizations/<request_id>.jpg`

Boxes are color-coded by `group_index` and labeled with `group_id`.

## Setup & Run (end-to-end)

### 1. Create a virtual environment + install dependencies

From this folder:

```bash
cd /home/dhivakar/AI_project/inflect_project/product_pipeline
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The first run will download model weights from the internet:
- YOLO `yolov8n.pt`
- CLIP weights for the chosen `CLIP_PRETRAINED` value

If the heavy model dependencies/weights are not available, the services include lightweight fallbacks so the demo (and visualizations) can still run end-to-end.

### 2. Start the three microservices (3 terminals)

Terminal A (detector):
```bash
cd /home/dhivakar/AI_project/inflect_project
. ./product_pipeline/.venv/bin/activate
python3 -m product_pipeline.detector_service.app
```

Terminal B (grouping):
```bash
cd /home/dhivakar/AI_project/inflect_project
. ./product_pipeline/.venv/bin/activate
python3 -m product_pipeline.grouping_service.app
```

Terminal C (main UI + orchestrator):
```bash
cd /home/dhivakar/AI_project/inflect_project
. ./product_pipeline/.venv/bin/activate
python3 -m product_pipeline.main_app.app
```

### 3. Use the UI

Open:
- `http://localhost:5000`

Upload an image and click `Infer`.

## Environment Variables (optional)

- `DETECTOR_PORT` (default `5001`)
- `GROUPER_PORT` (default `5002`)
- `MAIN_PORT` (default `5000`)
- `DETECTOR_WEIGHTS` (default `yolov8n.pt`)
- `CLIP_ARCH` (default `ViT-B-32`)
- `CLIP_PRETRAINED` (default `laion2b_s34b_b79k`)
- `CLIP_DEVICE` (default `cpu`)

