#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 --img_json <path> [--output_dir <path>] [other args]"
    exit 1
fi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
python -m genlit.inference --mode multi "$@"
