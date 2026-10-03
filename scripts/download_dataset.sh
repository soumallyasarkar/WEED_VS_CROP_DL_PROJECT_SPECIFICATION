#!/usr/bin/env bash
set -e

echo "=== Downloading and setting up CropAndWeed Dataset ==="
python scripts/download_dataset.py "$@"
