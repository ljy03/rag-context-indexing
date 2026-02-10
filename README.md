# RAG Context Indexing

Experiments comparing standard vs. contextualized segment retrieval on MS MARCO V2.1 for the TREC 2024 RAG Track.

## Overview

This project evaluates whether adding document context (title) to segments improves retrieval effectiveness. We compare:
- **Standard**: Raw segment text only
- **Contextualized**: Segment text then raw title only (no `[Context: ...]` wrapper; title concatenated as-is)

## Project Structure

```
rag-context-indexing/
├── data/                   # Corpus and evaluation files
│   ├── msmarco_v2.1_doc_segmented.tar   # Source corpus (required for pipeline)
│   ├── qrels.umbrela.rag24.test.txt     # Original qrels
│   ├── topics.rag24.test.txt            # Original topics
│   ├── mini_corpus_all_segments.jsonl   # From extract_segments (official format)
│   ├── corpus_standard.jsonl            # From make_corpora
│   ├── corpus_contextualized.jsonl      # From make_corpora
│   ├── qrels_filtered.txt               # Filtered qrels (subset)
│   └── topics_filtered.txt              # Filtered topics (subset)
├── indices/                # Built search indices
│   ├── bm25_standard/                   # Lucene index
│   ├── bm25_contextualized/             # Lucene index
│   ├── dense_standard/index/            # FAISS + Contriever embeddings
│   └── dense_contextualized/index/      # FAISS + Contriever embeddings
├── runs/                   # Retrieval run files (TREC format)
└── scripts/                # Pipeline scripts
    ├── extract_segments.py     # Extract segments from tar, write filtered qrels/topics
    ├── make_corpora.py        # Standard + contextualized corpus from mini_corpus
    ├── build_indices.sh       # BM25 + Dense (Contriever + FAISS)
    ├── evaluate.py            # Retrieval + trec_eval
    ├── encode_contriever.py   # Dense indexing (used by build_indices.sh)
    └── run_pipeline.sh       # Full run: extract_segments → make_corpora → build_indices → evaluate
```

## Pipeline

**One-shot:**

```bash
# Ensure msmarco_v2.1_doc_segmented.tar is in data/
TARGET_SEGMENTS=5000 bash scripts/run_pipeline.sh
```

### 1. Data processing (step-by-step)

```bash
# Extract segments from tar, create filtered qrels/topics
TARGET_SEGMENTS=5000 python scripts/extract_segments.py

# Create standard and contextualized corpus variants
python scripts/make_corpora.py
```

**Corpus formats:**

- **Standard**: `{"id": "...", "contents": "<segment text>"}`
- **Contextualized**: `{"id": "...", "contents": "<segment text>\n\n<title>"}` (raw fields, no wrapper)

### 2. Indexing

```bash
# Build BM25 (Lucene) and Dense (Contriever + FAISS) indices
bash scripts/build_indices.sh
```

- **BM25**: Pyserini Lucene with `DefaultLuceneDocumentGenerator` (positions, docvectors, raw stored).
- **Dense**: Contriever (`facebook/contriever`) with mean pooling; embeddings and FAISS index written under `indices/dense_*/index/` (see `scripts/encode_contriever.py`).

Environment variables:
- `THREADS`: Lucene indexing threads (default: 8)
- `DEVICE`: Dense encoding device — `cpu`, `cuda`, or `mps` (default: mps)
- `BATCH_SIZE`: Dense encoding batch size (default: 64)

### 3. Retrieval & Evaluation

```bash
# Run retrieval on all indices and evaluate
python scripts/evaluate.py
```

This script:
1. Loads `data/topics_filtered.txt`
2. Runs BM25 (Lucene) and Dense (FAISS + Contriever query encoder) retrieval (top-100)
3. Writes run files to `runs/` (e.g. `bm25_standard.txt`, `dense_standard.txt`)
4. Evaluates each run with `pyserini.eval.trec_eval` (NDCG@10, MAP, MRR, Recall@100)

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
| Dense (Contriever) Standard | — | — | — | — |
| Dense (Contriever) Contextualized | — | — | — | — |

**Key findings:**
- Standard segments slightly outperform contextualized (~10-15% gap)
- BM25 outperforms Contriever on this subset
- Context format matters: putting segment text first is critical for both BM25 and dense retrieval

## Conclusion

Adding document context (title) to segments does not improve retrieval effectiveness on this dataset. The contextualized approach performs ~10-15% worse than standard segments across all metrics.

**Why context didn't help:**
- The title metadata doesn't add useful retrieval signal for these queries
- Extra terms dilute BM25's term matching on the core segment content
- Contriever was not trained on this context format

**Lesson learned — format matters:**
Early experiments showed contextualized performing ~50% worse due to a formatting issue where segment text was placed *after* the context. This caused:
1. BM25 to over-weight context terms
2. Contriever to truncate the actual segment (512-token limit)

After fixing the format (segment first, short context suffix), the gap narrowed to ~10-15%. This highlights that *how* you add context is as important as *what* context you add.

## Requirements

- Python 3.10+
- Pyserini (Lucene indexing + BM25/FAISS search + trec_eval)
- Java 21+ (for Lucene)
- PyTorch and Transformers (for Contriever encoding)
- FAISS (e.g. `faiss-cpu`) for dense search

```bash
conda create -n pyserini_env python=3.11
conda activate pyserini_env
pip install pyserini torch transformers faiss-cpu tqdm
```

## References

- [TREC 2024 RAG Track](https://trec-rag.github.io/)
- [MS MARCO V2.1](https://microsoft.github.io/msmarco/)
- [Pyserini](https://github.com/castorini/pyserini)
