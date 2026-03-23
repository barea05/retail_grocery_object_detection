from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from product_pipeline.common import decode_base64_image, hsv_color


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    clip_model = {"model": None, "preprocess": None}
    ocr_reader = {"obj": None}

    def load_clip() -> Tuple[Any, Any]:
        if clip_model["model"] is not None:
            return clip_model["model"], clip_model["preprocess"]

        try:
            import open_clip

            arch = os.getenv("CLIP_ARCH", "ViT-B-32")
            pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            model.eval()
            clip_model["model"] = model
            clip_model["preprocess"] = preprocess
            return model, preprocess
        except Exception:
            # Fallback: keep model as None; embed via simple image features.
            clip_model["model"] = None
            clip_model["preprocess"] = None
            return None, None

    def load_ocr() -> Any:
        if ocr_reader["obj"] is not None:
            return ocr_reader["obj"]
        try:
            import easyocr

            # CPU by default for portability.
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            ocr_reader["obj"] = reader
            return reader
        except Exception:
            ocr_reader["obj"] = None
            return None

    def normalize_text(text: str) -> str:
        # Keep letters/numbers only and collapse spaces.
        clean = re.sub(r"[^A-Za-z0-9 ]+", " ", text.upper())
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return ""
        # Reduce noisy OCR to a stable token key.
        tokens = [t for t in clean.split(" ") if len(t) >= 3]
        if not tokens:
            return ""
        # Keep top 2 longest tokens to preserve brand names.
        tokens = sorted(tokens, key=len, reverse=True)[:2]
        return " ".join(sorted(tokens))

    def extract_text_features(pil_crops: List[Any]) -> List[Dict[str, Any]]:
        reader = load_ocr()
        out: List[Dict[str, Any]] = []
        for crop in pil_crops:
            if reader is None:
                out.append({"text": "", "text_key": "", "text_confidence": 0.0})
                continue
            arr = np.asarray(crop.convert("RGB"))
            try:
                results = reader.readtext(arr, detail=1, paragraph=False)
            except Exception:
                results = []

            if not results:
                out.append({"text": "", "text_key": "", "text_confidence": 0.0})
                continue

            # Merge the best few OCR outputs from this crop.
            parts = []
            confs = []
            for r in results[:3]:
                txt = str(r[1]).strip()
                conf = float(r[2])
                if txt:
                    parts.append(txt)
                    confs.append(conf)

            merged = " ".join(parts).strip()
            key = normalize_text(merged)
            conf = float(np.mean(confs)) if confs else 0.0
            out.append({"text": merged, "text_key": key, "text_confidence": conf})
        return out

    def extract_shape_features(pil_crops: List[Any]) -> Tuple[np.ndarray, List[str]]:
        feats: List[np.ndarray] = []
        tags: List[str] = []

        for crop in pil_crops:
            rgb = np.asarray(crop.convert("RGB"))
            h, w = rgb.shape[:2]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # Fallback simple geometry-only features.
                aspect = float(w) / float(max(h, 1))
                feat = np.array([aspect, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                feats.append(feat)
                tags.append("unknown")
                continue

            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            peri = float(cv2.arcLength(c, True))
            x, y, bw_box, bh_box = cv2.boundingRect(c)
            rect_area = float(max(bw_box * bh_box, 1))
            aspect = float(bw_box) / float(max(bh_box, 1))
            extent = area / rect_area
            hull = cv2.convexHull(c)
            hull_area = float(max(cv2.contourArea(hull), 1e-6))
            solidity = area / hull_area
            area_ratio = area / float(max(h * w, 1))
            peri_ratio = peri / float(max((w + h), 1))

            # Hu moments are shape-sensitive and scale/rotation invariant.
            hu = cv2.HuMoments(cv2.moments(c)).flatten()
            hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-8)
            hu = hu[:3]

            approx = cv2.approxPolyDP(c, 0.02 * max(peri, 1e-6), True)
            vertices = len(approx)
            if vertices <= 4:
                tag = "box_like"
            elif vertices <= 8:
                tag = "rounded"
            else:
                tag = "irregular"

            feat = np.array(
                [aspect, extent, solidity, area_ratio, peri_ratio, float(hu[0]), float(hu[1]), float(hu[2])],
                dtype=np.float32,
            )
            feats.append(feat)
            tags.append(tag)

        arr = np.stack(feats, axis=0)
        # Normalize across crops for stable fusion with visual embedding.
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-6
        arr = (arr - mean) / std
        return arr, tags

    def embed_crops(pil_crops: List[Any]) -> np.ndarray:
        import PIL.Image as PILImage

        model, preprocess = load_clip()
        if model is None or preprocess is None:
            # Lightweight embedding fallback (no torch/CLIP):
            # Extract a compact RGB histogram feature per crop.
            feats_list: List[np.ndarray] = []
            for crop in pil_crops:
                if crop.mode != "RGB":
                    crop = crop.convert("RGB")
                small = crop.resize((64, 64), PILImage.Resampling.BILINEAR)
                arr = np.asarray(small).astype(np.float32) / 255.0  # H,W,3
                # Histogram per channel (8 bins) -> 24 dims.
                hist = []
                for ch in range(3):
                    h, _ = np.histogram(arr[:, :, ch], bins=8, range=(0.0, 1.0), density=True)
                    hist.append(h)
                feat = np.concatenate(hist, axis=0)
                feat = feat / (np.linalg.norm(feat) + 1e-9)
                feats_list.append(feat)
            return np.stack(feats_list, axis=0)

        import torch

        device = os.getenv("CLIP_DEVICE", "cpu")
        model.to(device)

        with torch.no_grad():
            batch = torch.stack([preprocess(crop) for crop in pil_crops]).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.detach().cpu().numpy()

    @app.post("/group")
    def group() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True)
        image_base64: str = payload["image_base64"]
        detections: List[Dict[str, Any]] = payload["detections"]

        min_crops: int = int(payload.get("min_crops", 1))
        max_objects: int = int(payload.get("max_objects", 50))

        # Cropping happens on the same coordinate system as returned by detector.
        img = decode_base64_image(image_base64)
        img_w, img_h = img.size

        detections = detections[:max_objects]
        if len(detections) < min_crops:
            return jsonify({"objects": [], "groups": []})

        crops: List[Any] = []
        kept_indices: List[int] = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            # Clamp to image boundaries.
            x1i = max(0, int(round(x1)))
            y1i = max(0, int(round(y1)))
            x2i = min(img_w - 1, int(round(x2)))
            y2i = min(img_h - 1, int(round(y2)))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop = img.crop((x1i, y1i, x2i, y2i))
            crops.append(crop)
            kept_indices.append(idx)

        if not crops:
            return jsonify({"objects": [], "groups": []})

        eps: float = float(payload.get("dbscan_eps", 0.18))
        min_samples: int = int(payload.get("dbscan_min_samples", 1))
        agglo_threshold: float = float(payload.get("agglo_distance_threshold", 0.22))

        visual_embeddings = embed_crops(crops)
        shape_embeddings, shape_tags = extract_shape_features(crops)
        text_features = extract_text_features(crops)

        # Hybrid representation: visual + shape.
        # OCR is used as a strong post-clustering rule.
        embeddings = np.concatenate(
            [visual_embeddings * float(payload.get("visual_weight", 0.75)), shape_embeddings * float(payload.get("shape_weight", 0.25))],
            axis=1,
        )

        # DBSCAN with cosine distance.
        dist_mat = cosine_distances(embeddings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(dist_mat).tolist()

        # Collect cluster -> members
        clusters: Dict[int, List[int]] = {}
        for member_pos, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(member_pos)

        # If DBSCAN fails to find clusters, fallback to simple grouping by single-link heuristic.
        # Here we fallback to KMeans-like behavior without importing extra libs:
        # create 1 cluster when only one point exists, else use "nearest neighbor by cosine sim" into up to 3 groups.
        unique_positive = [k for k in clusters.keys() if k >= 0]

        # If DBSCAN merges too much (or finds almost nothing), switch to agglomerative clustering
        # to get finer-grained brand groups.
        if len(crops) >= 4 and len(unique_positive) <= 2:
            try:
                agg = AgglomerativeClustering(
                    n_clusters=None,
                    metric="cosine",
                    linkage="average",
                    distance_threshold=agglo_threshold,
                )
                labels = agg.fit_predict(embeddings).tolist()
                clusters = {}
                for member_pos, lab in enumerate(labels):
                    clusters.setdefault(lab, []).append(member_pos)
                unique_positive = [k for k in clusters.keys() if k >= 0]
            except Exception:
                pass
        if not unique_positive and len(crops) > 1:
            # Assign each point to group 0 by default, then split outliers by max distance.
            # This is a pragmatic fallback to keep output stable.
            labels = [0 for _ in labels]
            unique_positive = [0]
            clusters = {0: list(range(len(crops)))}

        # Prepare centroids for cluster IDs.
        # Also handle DBSCAN outliers (-1): assign them to nearest positive cluster centroid.
        positive_centroids: Dict[int, np.ndarray] = {}
        for lab in unique_positive:
            idxs = clusters.get(lab, [])
            if idxs:
                positive_centroids[lab] = embeddings[idxs].mean(axis=0)

        def assign_outlier(emb: np.ndarray) -> int:
            if not positive_centroids:
                return 0
            best_lab = None
            best_score = -1.0
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            for lab, cent in positive_centroids.items():
                cent_norm = cent / (np.linalg.norm(cent) + 1e-9)
                score = float(emb_norm @ cent_norm)  # cosine similarity
                if score > best_score:
                    best_score = score
                    best_lab = lab
            return int(best_lab) if best_lab is not None else 0

        # Final group indices are dense 0..k-1 with human-friendly IDs.
        positive_sorted = sorted(unique_positive)
        lab_to_dense: Dict[int, int] = {lab: i for i, lab in enumerate(positive_sorted)}

        final_dense_labels: List[int] = []
        for member_pos, lab in enumerate(labels):
            if lab >= 0 and lab in lab_to_dense:
                final_dense_labels.append(lab_to_dense[lab])
            else:
                out_lab = assign_outlier(embeddings[member_pos])
                final_dense_labels.append(lab_to_dense.get(out_lab, 0))

        # OCR-guided regrouping:
        # if the same normalized text appears across multiple products, force them into one group.
        # This aligns grouping more closely with brand text on packaging.
        text_counts: Dict[str, int] = {}
        for t in text_features:
            key = t["text_key"]
            if key:
                text_counts[key] = text_counts.get(key, 0) + 1

        text_group_map: Dict[str, int] = {}
        next_group = (max(final_dense_labels) + 1) if final_dense_labels else 0
        for i, t in enumerate(text_features):
            key = t["text_key"]
            conf = float(t["text_confidence"])
            if key and conf >= float(payload.get("ocr_conf_threshold", 0.25)) and text_counts.get(key, 0) >= 2:
                if key not in text_group_map:
                    text_group_map[key] = next_group
                    next_group += 1
                final_dense_labels[i] = text_group_map[key]

        # Re-map to dense labels 0..k-1.
        uniq = sorted(set(final_dense_labels))
        remap = {old: new for new, old in enumerate(uniq)}
        final_dense_labels = [remap[x] for x in final_dense_labels]

        # Final grouping signature:
        # 1) OCR text key when reliable
        # 2) else class + shape + cluster bucket
        # This reduces over-merging where many products were ending up in same group.
        signature_to_group: Dict[str, int] = {}
        next_gid = 0
        objects: List[Dict[str, Any]] = []
        group_members: Dict[int, List[int]] = {}

        for obj_pos, dense_lab in enumerate(final_dense_labels):
            det_idx = kept_indices[obj_pos]
            det = detections[det_idx]
            text_info = text_features[obj_pos]
            class_name = str(det.get("class_name", "unknown"))
            shape_tag = shape_tags[obj_pos]
            text_key = text_info["text_key"]
            text_conf = float(text_info["text_confidence"])

            if text_key and text_conf >= float(payload.get("ocr_conf_threshold", 0.25)):
                signature = f"text:{text_key}"
            else:
                signature = f"class:{class_name}|shape:{shape_tag}|cluster:{int(dense_lab)}"

            if signature not in signature_to_group:
                signature_to_group[signature] = next_gid
                next_gid += 1
            gid = signature_to_group[signature]

            group_members.setdefault(gid, []).append(det_idx)
            objects.append(
                {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "class_id": det["class_id"],
                    "class_name": class_name,
                    "group_index": int(gid),
                    "group_id": f"brand_{gid}",
                    "shape_tag": shape_tag,
                    "ocr_text": text_info["text"],
                    "ocr_text_key": text_key,
                    "ocr_confidence": text_conf,
                }
            )

        groups_out: List[Dict[str, Any]] = []
        for dense_lab, member_det_indices in sorted(group_members.items(), key=lambda x: x[0]):
            groups_out.append(
                {
                    "group_index": int(dense_lab),
                    "group_id": f"brand_{dense_lab}",
                    "member_indices": [int(i) for i in member_det_indices],
                    "color": {"r": hsv_color(dense_lab)[0], "g": hsv_color(dense_lab)[1], "b": hsv_color(dense_lab)[2]},
                }
            )

        return jsonify({"objects": objects, "groups": groups_out})

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    port = int(os.getenv("GROUPER_PORT", "5002"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)

