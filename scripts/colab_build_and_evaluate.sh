#!/usr/bin/env bash
# Colab or local: run only build_indices + evaluate.
# Assumes data/ has corpus_standard.jsonl, corpus_contextualized.jsonl, qrels_filtered.txt, topics_filtered.txt.
# On Colab: no conda needed if you already ran pip install pyserini torch transformers faiss-cpu tqdm.
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
DATA_DIR="${ROOT_DIR}/data"
INDEX_DIR="${ROOT_DIR}/indices"
THREADS="${THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# Use cuda on Colab when no conda (DEVICE not set); else respect env (e.g. mps locally)
if [ -z "${DEVICE:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  export DEVICE="${DEVICE:-cuda}"
fi
DEVICE="${DEVICE:-mps}"

for f in "$DATA_DIR/corpus_standard.jsonl" "$DATA_DIR/corpus_contextualized.jsonl" "$DATA_DIR/qrels_filtered.txt" "$DATA_DIR/topics_filtered.txt"; do
  if [ ! -f "$f" ]; then
    echo "Missing: $f (run mine_distractors → extract_segments → make_corpora locally, or add data/ to repo)"
    exit 1
  fi
done

# Optional: activate conda if available (local runs)
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate pyserini_env 2>/dev/null || true
fi

mkdir -p "$INDEX_DIR"
STD_CORPUS="$DATA_DIR/corpus_standard.jsonl"
CTX_CORPUS="$DATA_DIR/corpus_contextualized.jsonl"

build_bm25() {
  local corpus_file="$1"
  local index_path="$2"
  local corpus_dir="${index_path}/corpus"
  mkdir -p "$corpus_dir"
  cp "$corpus_file" "$corpus_dir/corpus.jsonl"
  python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$corpus_dir" \
    --index "$index_path" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "$THREADS" \
    --storePositions --storeDocvectors --storeRaw
}

build_dense() {
  local corpus_file="$1"
  local out_dir="$2"
  local index_dir="${out_dir}/index"
  mkdir -p "$index_dir"
  python "$ROOT_DIR/scripts/encode_contriever.py" \
    --corpus "$corpus_file" \
    --output "$index_dir" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
}

echo "=== Building indices ==="
echo "Building BM25: standard"
build_bm25 "$STD_CORPUS" "$INDEX_DIR/bm25_standard"
echo "Building BM25: contextualized"
build_bm25 "$CTX_CORPUS" "$INDEX_DIR/bm25_contextualized"
echo "Building dense: standard"
build_dense "$STD_CORPUS" "$INDEX_DIR/dense_standard"
echo "Building dense: contextualized"
build_dense "$CTX_CORPUS" "$INDEX_DIR/dense_contextualized"
echo "=== Evaluate ==="
python "$ROOT_DIR/scripts/evaluate.py"
echo "Done. Runs in $ROOT_DIR/runs/"
