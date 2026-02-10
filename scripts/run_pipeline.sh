#!/usr/bin/env bash
# Run full pipeline: mine_distractors -> extract_segments -> make_corpora -> build_indices -> evaluate
# mine_distractors builds fair mini-corpus IDs (positives + BM25 hard negatives); extract then pulls from tar.
set -e
cd "$(dirname "$0")/.."
export TARGET_SEGMENTS="${TARGET_SEGMENTS:-5000}"

echo "=== mine_distractors (positives + BM25 hard negatives) ==="
python scripts/mine_distractors.py

echo "=== extract_segments ==="
python scripts/extract_segments.py

echo "=== make_corpora ==="
python scripts/make_corpora.py

echo "=== build_indices ==="
bash scripts/build_indices.sh

echo "=== evaluate ==="
python scripts/evaluate.py

echo "Done."
