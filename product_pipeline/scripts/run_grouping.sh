#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dhivakar/AI_project/inflect_project"
VENV_PATH="$ROOT_DIR/product_pipeline/.venv/bin/activate"

source "$VENV_PATH"
cd "$ROOT_DIR"

exec python3 -m product_pipeline.grouping_service.app

