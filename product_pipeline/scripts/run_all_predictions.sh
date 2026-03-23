#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dhivakar/AI_project/inflect_project"
SCRIPT_DIR="$ROOT_DIR/product_pipeline/scripts"

echo "[1/3] Restarting all services..."
bash "$SCRIPT_DIR/restart_all.sh"

echo "[2/3] Running batch predictions on sample_images_extracted/sample_images..."
python3 - <<'PY'
from pathlib import Path
import requests

base = Path('/home/dhivakar/AI_project/inflect_project')
img_dir = base / 'sample_images_extracted' / 'sample_images'

images = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
print(f"[info] images_found={len(images)}")

for p in images:
    with p.open('rb') as f:
        files = {'image': (p.name, f, 'image/jpeg')}
        data = {
            'min_confidence': '0.15',
            'max_objects': '120',
            'imgsz': '960',
            'dbscan_eps': '0.18',
            'dbscan_min_samples': '1',
            'agglo_distance_threshold': '0.22',
            'ocr_conf_threshold': '0.25',
        }
        r = requests.post('http://127.0.0.1:5000/api/infer', files=files, data=data, timeout=900)
        r.raise_for_status()
        j = r.json()
        print(f"[ok] {p.name} -> {j['visualization']['file']}")
PY

echo "[3/3] Done."
echo "Check outputs in:"
echo "  $ROOT_DIR/product_pipeline/outputs/visualizations"

