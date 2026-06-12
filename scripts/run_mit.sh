#!/usr/bin/env bash
# MIT mode runs on the MIT Multi-Illumination test set (theta+phi control only,
# 2 conditioning channels). The MIT model is not designed for in-the-wild relighting.
# Requires --mit_data_root pointing at the MIT MI dataset root.
set -euo pipefail
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 --mit_data_root <path> [other args]"
    echo "  --mit_data_root: root of MIT Multi-Illumination dataset (https://projects.csail.mit.edu/illumination/)"
    exit 1
fi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
python -m genlit.inference --mode mit "$@"
