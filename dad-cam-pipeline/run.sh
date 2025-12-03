#!/bin/bash
# Dad Cam Pipeline - Run Script
# ==============================
#
# Usage:
#   ./run.sh /path/to/source /path/to/output [options]
#
# Options are passed directly to dad_cam_pipeline.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

# Check for required tools
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi

if ! command -v exiftool &> /dev/null; then
    echo "Error: exiftool not found. Install with: brew install exiftool"
    exit 1
fi

# Run pipeline
python "$SCRIPT_DIR/dad_cam_pipeline.py" "$@"
