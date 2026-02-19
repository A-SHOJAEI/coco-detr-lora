#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper matching the project plan phrasing.
# This runs the experiment list defined in configs/coco2017_full.yaml.

CONFIG="${CONFIG:-configs/coco2017_full.yaml}"

make setup
make data CONFIG="$CONFIG"
make train CONFIG="$CONFIG"
make eval CONFIG="$CONFIG"
make report
