# End-to-end solution write-up

## Goal

Build an AI pipeline that:
1. Exposes a simple Flask-based API/Web UI.
2. Runs product detection on shelf images.
3. Groups detected products into brand-like groups and assigns a stable `group_id`.
4. Returns inference results as JSON.
5. Saves color-coded visualization images to disk.

## Data flow (pipeline)

When the user uploads an image to the main service (`main_app`):

1. **Detector microservice** (`detector_service`)
   - Receives the uploaded image as base64.
   - Runs an object detector (YOLOv8n).
   - Returns a list of detections with:
     - bounding box (`bbox`: `[x1, y1, x2, y2]`)
     - confidence
     - class label

2. **Grouping microservice** (`grouping_service`)
   - Receives the original image (base64) + the detector’s bounding boxes.
   - Crops each detected object region from the image.
   - Embeds each crop using a CLIP image encoder (`open_clip_torch`).
   - Clusters embeddings using DBSCAN with a cosine distance metric.
   - Produces:
     - `group_id` for each object (`brand_<k>`)
     - `group_index` and membership lists

3. **Main service visualization**
   - Draws detection boxes on the original image.
   - Colors each box based on the assigned `group_index`.
   - Saves the result to `outputs/visualizations/<request_id>.jpg`.
   - Returns JSON including the visualization URL.

## Why this works for “brand grouping”

There is no ground-truth brand label in the assignment, and importing a retail-SKU-specific model would be brittle. Instead, this solution groups by **visual similarity**:
- CLIP embeddings capture semantic information from the object crop.
- DBSCAN groups clusters even when the number of brands is unknown.

The output is a `group_id` per cluster. On real shelf images, clusters often correspond to visually consistent product packaging (frequently aligning with brand families).

## Latency & scalability considerations

1. **Microservices keep model loads out of request path**
   - Each microservice caches the model in memory (lazy load, then reuse).

2. **Payload size control**
   - The main service and detector both enforce `max_objects` caps.

 3. **Deterministic visualization**
   - Colors are derived from `group_index`, so repeated runs are consistent for a given clustering result.

In practice, you can scale horizontally:
- Add multiple detector replicas behind a load balancer.
- Add multiple grouping replicas (grouping can be CPU-bound on cosine clustering).

## Alternative approaches considered

1. **OCR-based brand extraction**
   - Pros: can directly identify text on labels.
   - Cons: OCR is error-prone on small fonts/angles; more fragile and often slower.

2. **Supervised SKU/brand classifier**
   - Pros: potentially accurate.
   - Cons: requires labeled brand data and training; not feasible in an offline assignment setup.

This solution’s clustering approach is model-agnostic and works with the provided requirement (“any technique of your choice”).

## How to run

See `product_pipeline/README.md` in the same folder.

