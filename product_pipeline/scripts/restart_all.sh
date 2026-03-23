#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dhivakar/AI_project/inflect_project"
SCRIPT_DIR="$ROOT_DIR/product_pipeline/scripts"
LOG_DIR="$ROOT_DIR/product_pipeline/logs"
PID_DIR="$ROOT_DIR/product_pipeline/logs"

mkdir -p "$LOG_DIR"

echo "[info] Stopping existing services (if running)..."
pkill -f "python3 -m product_pipeline.detector_service.app" || true
pkill -f "python3 -m product_pipeline.grouping_service.app" || true
pkill -f "python3 -m product_pipeline.main_app.app" || true
sleep 1

echo "[info] Starting detector service..."
nohup "$SCRIPT_DIR/run_detector.sh" > "$LOG_DIR/detector.log" 2>&1 &
echo $! > "$PID_DIR/detector.pid"

echo "[info] Starting grouping service..."
nohup "$SCRIPT_DIR/run_grouping.sh" > "$LOG_DIR/grouping.log" 2>&1 &
echo $! > "$PID_DIR/grouping.pid"

echo "[info] Starting main app..."
nohup "$SCRIPT_DIR/run_main.sh" > "$LOG_DIR/main.log" 2>&1 &
echo $! > "$PID_DIR/main.pid"

sleep 2

echo "[ok] Restart triggered for all three services."
echo "  detector pid: $(cat "$PID_DIR/detector.pid")"
echo "  grouping pid: $(cat "$PID_DIR/grouping.pid")"
echo "  main pid:     $(cat "$PID_DIR/main.pid")"
echo
echo "Logs:"
echo "  $LOG_DIR/detector.log"
echo "  $LOG_DIR/grouping.log"
echo "  $LOG_DIR/main.log"

