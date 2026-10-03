#!/usr/bin/env bash
# Start all small model inference servers (Linux).
# Linux counterpart of start_servers.ps1.
set -euo pipefail

# Resolve workspace root (parent of this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: '$PYTHON' not found in PATH. Set PYTHON to your interpreter." >&2
    exit 1
fi

LOG_DIR="$WORKSPACE/servers/logs"
mkdir -p "$LOG_DIR"

echo "Starting SAGE small model servers..."

start_server() {
    local name="$1" module="$2" port="$3"
    nohup "$PYTHON" -m "$module" >"$LOG_DIR/$name.log" 2>&1 &
    echo "  $name -> :$port  (pid $!, log servers/logs/$name.log)"
}

start_server human_detect servers.human_detect_server 8001
start_server pose        servers.pose_server        8002
start_server depth       servers.depth_server       8003
start_server face_landmark servers.face_landmark_server 8005

echo "All servers started. Stop with: pkill -f 'servers.*_server'"
