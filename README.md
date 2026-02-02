# RAG Context Indexing

Experiments comparing standard vs. contextualized segment retrieval on MS MARCO V2.1 for the TREC 2024 RAG Track.

## Overview

This project evaluates whether adding document context (title, headings) to segments improves retrieval effectiveness. We compare:
- **Standard**: Raw segment text only
- **Contextualized**: Segment text + title/section metadata

## Project Structure

```
rag-context-indexing/
├── data/                   # Corpus and evaluation files
│   ├── msmarco_v2.1_doc_segmented.tar   # Source corpus
│   ├── qrels.umbrela.rag24.test.txt     # Original qrels
│   ├── topics.rag24.test.txt            # Original topics
│   ├── qrels_filtered.txt               # Filtered qrels (subset)
│   ├── topics_filtered.txt              # Filtered topics (subset)
│   ├── corpus_standard.jsonl            # Standard corpus
│   └── corpus_contextualized.jsonl      # Contextualized corpus
├── indices/                # Built search indices
│   ├── bm25_standard/
│   ├── bm25_contextualized/
│   ├── dense_standard/
│   └── dense_contextualized/
├── runs/                   # Retrieval run files (TREC format)
├── scripts/                # Pipeline scripts
└── results/                # Analysis outputs
```

## Pipeline

### 1. Data Processing

```bash
# Parse qrels and extract relevant doc/segment IDs (optional, for analysis)
python scripts/01_parse_qrels.py

# Select topics until ~5000 docs, extract segments, create filtered qrels/topics
TARGET_DOCS=5000 python scripts/02_extract_relevant_segments.py

# Normalize segments to standard schema
python scripts/03_normalize_base_segments.py

# Create standard and contextualized corpus variants
python scripts/04_make_standard_and_contextualized.py
```

**Corpus formats:**

- **Standard**: `{"id": "...", "contents": "<segment text>"}`
- **Contextualized**: `{"id": "...", "contents": "<segment text>\n\n[Context: Title: ... | Section: ...]"}`

### 2. Indexing

```bash
# Build BM25 (Lucene) and Dense (DPR + FAISS) indices
bash scripts/05_build_indices.sh
```

Environment variables:
- `THREADS`: Indexing threads (default: 8)
- `DEVICE`: Encoding device - `cpu` or `mps` (default: mps)
- `BATCH_SIZE`: Dense encoding batch size (default: 64)

### 3. Retrieval & Evaluation

```bash
# Run retrieval on all indices and evaluate
python scripts/06_evaluate.py
```

This script:
1. Loads filtered topics
2. Runs BM25 and Dense retrieval (top-100)
3. Saves run files to `runs/`
4. Evaluates using `pyserini.eval.trec_eval`

**Manual evaluation:**
```bash
python -m pyserini.eval.trec_eval -c -M 100 \
  -m ndcg_cut.10 -m map -m recip_rank -m recall.100 \
  data/qrels_filtered.txt runs/bm25_standard.txt
```

## Results

| Method | NDCG@10 | MRR | MAP | Recall@100 |
|--------|---------|-----|-----|------------|
| BM25 Standard | **0.4899** | **0.8693** | **0.2760** | **0.4623** |
| BM25 Contextualized | 0.4194 | 0.7883 | 0.2055 | 0.3982 |
| Dense (DPR) Standard | 0.3345 | 0.6899 | 0.1610 | 0.3413 |
| Dense (DPR) Contextualized | 0.3128 | 0.6525 | 0.1500 | 0.3292 |

**Key findings:**
- Standard segments slightly outperform contextualized (~10-15% gap)
- BM25 outperforms DPR on this subset
- Context format matters: putting segment text first is critical for both BM25 and DPR

## Conclusion

Adding document context (title, section headings) to segments does not improve retrieval effectiveness on this dataset. The contextualized approach performs ~10-15% worse than standard segments across all metrics.

**Why context didn't help:**
- The title/section metadata doesn't add useful retrieval signal for these queries
- Extra terms dilute BM25's term matching on the core segment content
- DPR encoders were not trained on this context format

**Lesson learned - format matters:**
Early experiments showed contextualized performing ~50% worse due to a formatting issue where segment text was placed *after* the context. This caused:
1. BM25 to over-weight context terms
2. DPR to truncate the actual segment (512 token limit)

After fixing the format (segment first, short context suffix), the gap narrowed to ~10-15%. This highlights that *how* you add context is as important as *what* context you add.

## Requirements

- Python 3.10+
- Pyserini (`pip install pyserini`)
- Java 21+ (for Lucene indexing)
- PyTorch (for dense encoding)

```bash
conda create -n pyserini_env python=3.11
conda activate pyserini_env
pip install pyserini torch faiss-cpu
```

## References

- [TREC 2024 RAG Track](https://trec-rag.github.io/)
- [MS MARCO V2.1](https://microsoft.github.io/msmarco/)
- [Pyserini](https://github.com/castorini/pyserini)
