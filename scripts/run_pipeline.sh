#!/usr/bin/env bash
# Run full pipeline 02 -> 03 -> 04 -> 05 -> 06 (no comments in execution path)
set -e
cd "$(dirname "$0")/.."
export TARGET_SEGMENTS="${TARGET_SEGMENTS:-5000}"

echo "=== 02 Extract ==="
python scripts/02_extract_relevant_segments.py

echo "=== 03 Normalize ==="
python scripts/03_normalize_base_segments.py

echo "=== 04 Standard + Contextualized ==="
python scripts/04_make_standard_and_contextualized.py

echo "=== 05 Build indices ==="
bash scripts/05_build_indices.sh

echo "=== 06 Evaluate ==="
python scripts/06_evaluate.py

echo "Done."
