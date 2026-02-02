#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pyserini_env

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
INDEX_DIR="${ROOT_DIR}/indices"

THREADS="${THREADS:-8}"
ENCODER="${ENCODER:-facebook/dpr-ctx_encoder-multiset-base}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-mps}"

STD_CORPUS="${DATA_DIR}/corpus_standard.jsonl"
CTX_CORPUS="${DATA_DIR}/corpus_contextualized.jsonl"

mkdir -p "${INDEX_DIR}"

build_bm25() {
  local corpus_file="$1"
  local index_path="$2"
  local corpus_dir="${index_path}/corpus"

  mkdir -p "${corpus_dir}"
  ln -sf "${corpus_file}" "${corpus_dir}/corpus.jsonl"
  python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${corpus_dir}" \
    --index "${index_path}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${THREADS}" \
    --storePositions \
    --storeDocvectors \
    --storeRaw
}

build_dense() {
  local corpus_file="$1"
  local out_dir="$2"
  local embed_dir="${out_dir}/embeddings"
  local index_dir="${out_dir}/index"
  local corpus_dir="${out_dir}/corpus"
  local text_corpus_dir="${out_dir}/corpus_text"

  mkdir -p "${embed_dir}" "${index_dir}" "${corpus_dir}" "${text_corpus_dir}"
  ln -sf "${corpus_file}" "${corpus_dir}/corpus.jsonl"
  python - <<'PY' "${corpus_file}" "${text_corpus_dir}/corpus.jsonl"
import json
import sys

inp = sys.argv[1]
out = sys.argv[2]

with open(inp, "r", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
    for line in f_in:
        rec = json.loads(line)
        contents = rec.get("contents", "")
        f_out.write(json.dumps({"id": rec.get("id", ""), "text": contents}, ensure_ascii=False) + "\n")
PY

  python -m pyserini.encode \
    input --corpus "${text_corpus_dir}" \
          --fields text \
    output --embeddings "${embed_dir}" \
    encoder --encoder "${ENCODER}" \
            --fields text \
            --batch "${BATCH_SIZE}" \
            --device "${DEVICE}"

  python -m pyserini.index.faiss \
    --input "${embed_dir}" \
    --output "${index_dir}"
}

echo "Building BM25 index: original segments"
build_bm25 "${STD_CORPUS}" "${INDEX_DIR}/bm25_standard"

echo "Building dense index: original segments"
build_dense "${STD_CORPUS}" "${INDEX_DIR}/dense_standard"

echo "Building BM25 index: context-augmented segments"
build_bm25 "${CTX_CORPUS}" "${INDEX_DIR}/bm25_contextualized"

echo "Building dense index: context-augmented segments"
build_dense "${CTX_CORPUS}" "${INDEX_DIR}/dense_contextualized"

echo "Done. Indices are under: ${INDEX_DIR}"
