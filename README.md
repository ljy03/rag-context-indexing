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
│   ├── qrels.umbrela.rag24.test.txt    # Original qrels
│   ├── topics.rag24.test.txt           # Original topics
│   ├── mini_ids.txt                    # From mine_distractors (positives + distractors)
│   ├── slice_corpus.jsonl              # BM25 slice for mining distractors (optional)
│   ├── mini_corpus_all_segments.jsonl  # From extract_segments (official format)
│   ├── corpus_standard.jsonl           # From make_corpora
│   ├── corpus_contextualized.jsonl     # From make_corpora
│   ├── qrels_filtered.txt              # Filtered qrels (subset)
│   └── topics_filtered.txt             # Filtered topics (subset)
├── indices/                # Built search indices
│   ├── bm25_slice/                      # Lucene index for distractor mining
│   ├── bm25_standard/                   # Lucene index (mini corpus)
│   ├── bm25_contextualized/             # Lucene index (mini corpus)
│   ├── dense_standard/index/           # FAISS + Contriever embeddings
│   └── dense_contextualized/index/     # FAISS + Contriever embeddings
├── runs/                   # Retrieval run files (TREC format)
└── scripts/
    ├── mine_distractors.py    # Fair mini-corpus IDs: P_PER_TOPIC positives + BM25 hard negatives
    ├── extract_segments.py   # Extract segments from tar using mini_ids.txt; filtered qrels/topics
    ├── make_corpora.py       # Standard + contextualized corpus from mini_corpus
    ├── build_indices.sh      # BM25 + Dense (Contriever + FAISS)
    ├── evaluate.py           # Retrieval + trec_eval
    ├── encode_contriever.py  # Dense indexing (used by build_indices.sh)
    └── run_pipeline.sh       # Full: mine_distractors → extract → make_corpora → build_indices → evaluate
```

## Pipeline

**One-shot (full pipeline):**

```bash
# Requires msmarco_v2.1_doc_segmented.tar in data/
bash scripts/run_pipeline.sh
```

For a target of ~5k segment IDs (positives + distractors):

```bash
P_PER_TOPIC=10 D_PER_TOPIC=8 MAX_DISTRACTORS=2000 BM25_K=2000 MAX_PER_DOC_DISTRACTORS=2 bash scripts/run_pipeline.sh
```

### 1. Data processing (step-by-step)

**Step 1 — Mine distractors (fair mini-corpus IDs)**  
Builds `data/mini_ids.txt`: per-topic positives (capped with `P_PER_TOPIC`, deterministic sampling) plus BM25-mined hard negatives.

```bash
# Default: P_PER_TOPIC=10, MAX_DISTRACTORS=5000, D_PER_TOPIC=100, etc.
python scripts/mine_distractors.py

# ~5k total IDs example:
P_PER_TOPIC=10 D_PER_TOPIC=8 MAX_DISTRACTORS=2000 python scripts/mine_distractors.py
```

**Step 2 — Extract segments**  
Extracts segments listed in `mini_ids.txt` from the tar; writes `mini_corpus_all_segments.jsonl`, `qrels_filtered.txt`, `topics_filtered.txt`.

```bash
python scripts/extract_segments.py
```

**Step 3 — Build corpus variants**

```bash
python scripts/make_corpora.py
```

- **Standard**: `{"id": "...", "contents": "<segment text>"}`
- **Contextualized**: `{"id": "...", "contents": "<segment text>\n\n<title>"}` (raw, no wrapper)

### 2. Indexing

```bash
bash scripts/build_indices.sh
```

- **BM25**: Pyserini Lucene (`DefaultLuceneDocumentGenerator`; positions, docvectors, raw stored).
- **Dense**: Contriever (`facebook/contriever`) mean pooling; FAISS index under `indices/dense_*/index/`.

Env: `THREADS`, `DEVICE` (e.g. `cuda`, `mps`), `BATCH_SIZE`.

### 3. Retrieval & Evaluation

```bash
python scripts/evaluate.py
```

Runs BM25 and Dense retrieval (top-100, doc-level diversification), writes runs to `runs/`, then runs `pyserini.eval.trec_eval` (NDCG@10, MAP, MRR, Recall@100).

**Manual evaluation:**
```bash
python -m pyserini.eval.trec_eval -c -M 100 \
  -m ndcg_cut.10 -m map -m recip_rank -m recall.100 \
  data/qrels_filtered.txt runs/bm25_standard.txt
```

## Results

Mini-corpus setup: **301 topics**, **~8k segments** (positives + BM25 hard negatives), filtered qrels/topics.

| Method | NDCG@10 | MRR | MAP | Recall@100 |
|--------|---------|-----|-----|------------|
| BM25 Standard | 0.6986 | 0.9321 | 0.7299 | 0.8891 |
| BM25 Contextualized | **0.7219** | **0.9373** | **0.7524** | **0.9060** |
| Dense (Contriever) Standard | 0.6797 | 0.9379 | 0.7002 | 0.9070 |
| Dense (Contriever) Contextualized | 0.6985 | 0.9364 | 0.7186 | **0.9192** |

**Findings (mini-corpus):**
- Contextualized is competitive or slightly better: higher NDCG@10 and MAP for BM25 and Dense; best Recall@100 with Dense Contextualized.
- All four setups achieve high MRR (~0.93) and Recall@100 (~0.89–0.92).
- Adding title after the segment (segment-first format) does not hurt and can help on this subset.

## Conclusion

On this fair mini-corpus (per-topic capped positives + BM25 hard negatives), **adding document title as context (segment then title) is competitive or slightly beneficial** for retrieval. Standard and contextualized both work well; contextualized gives small gains in NDCG@10, MAP, and Recall@100.

**Format matters:** Segment text must come first; appending a short title suffix avoids diluting the main content and keeps Contriever within effective context length.

## Requirements

- Python 3.10+
- Pyserini (Lucene + BM25/FAISS + trec_eval)
- Java 21+ (Lucene)
- PyTorch, Transformers (Contriever), FAISS (e.g. `faiss-cpu`)

```bash
conda create -n pyserini_env python=3.11
conda activate pyserini_env
pip install pyserini torch transformers faiss-cpu tqdm
```

## References

- [TREC 2024 RAG Track](https://trec-rag.github.io/)
- [MS MARCO V2.1](https://microsoft.github.io/msmarco/)
- [Pyserini](https://github.com/castorini/pyserini)
