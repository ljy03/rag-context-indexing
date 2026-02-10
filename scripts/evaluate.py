#!/usr/bin/env python3
"""
Run retrieval on all indices and evaluate using pyserini.eval.trec_eval.
"""
import subprocess
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "indices"
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

TOPICS_PATH = DATA_DIR / "topics_filtered.txt"
QRELS_PATH = DATA_DIR / "qrels_filtered.txt"

TOP_K = 100
# Document-level diversification: at most this many segments per parent doc per query (1 or 2)
MAX_PER_DOC = 2


def parent_docid(segid: str) -> str:
    """Segment id is like msmarco_v2.1_doc_29_677149#3_1637632; parent doc is before '#'."""
    return segid.split("#", 1)[0] if "#" in segid else segid


# ============ Load topics ============
topics = {}
with TOPICS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            topics[parts[0]] = parts[1]

print(f"Loaded {len(topics)} topics")

# ============ BM25 Retrieval ============
def run_bm25(index_path: Path, run_name: str):
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_path))
    searcher.set_bm25(k1=0.9, b=0.4)

    run_file = RUNS_DIR / f"{run_name}.txt"
    with run_file.open("w", encoding="utf-8") as f:
        for qid, query in topics.items():
            hits = searcher.search(query, k=TOP_K * 5)
            seen = {}
            rank = 0
            for hit in hits:
                doc = parent_docid(hit.docid)
                if seen.get(doc, 0) >= MAX_PER_DOC:
                    continue
                seen[doc] = seen.get(doc, 0) + 1
                rank += 1
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {run_name}\n")
                if rank >= TOP_K:
                    break

    print(f"Wrote: {run_file}")
    return run_file


# ============ Dense Retrieval (Contriever: mean pooling + L2-normalize, same as index) ============
def run_dense(index_path: Path, run_name: str):
    import sys
    from pyserini.search.faiss import FaissSearcher

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from encode_contriever import ContrieverQueryEncoder

    query_encoder = ContrieverQueryEncoder(model_name="facebook/contriever")
    searcher = FaissSearcher(str(index_path), query_encoder)
    print(f"  FAISS index dimension: {searcher.dimension}")
    first_query = next(iter(topics.values()))
    emb_q = query_encoder.encode(first_query)
    print(f"  Query emb shape: {getattr(emb_q, 'shape', None)}, len(emb_q)={len(emb_q)}")
    print(f"  q_norm (should be ~1.0 for IndexFlatIP/cosine): {np.linalg.norm(emb_q):.6f}")
    assert len(emb_q) == searcher.dimension, (
        f"Query encoder must return 1D for single query: len(emb_q)={len(emb_q)} != {searcher.dimension}"
    )

    run_file = RUNS_DIR / f"{run_name}.txt"
    with run_file.open("w", encoding="utf-8") as f:
        for qid, query in topics.items():
            hits = searcher.search(query, k=TOP_K * 5)
            seen = {}
            rank = 0
            for hit in hits:
                doc = parent_docid(hit.docid)
                if seen.get(doc, 0) >= MAX_PER_DOC:
                    continue
                seen[doc] = seen.get(doc, 0) + 1
                rank += 1
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {run_name}\n")
                if rank >= TOP_K:
                    break

    print(f"Wrote: {run_file}")
    return run_file


# ============ Evaluate with pyserini.eval.trec_eval ============
def evaluate(run_file: Path):
    print(f"\n=== {run_file.stem} ===")
    if not run_file.exists() or run_file.stat().st_size == 0:
        print("(skip: run file empty or missing)")
        return
    result = subprocess.run(
        [
            "python", "-m", "pyserini.eval.trec_eval",
            "-c", "-M", "100",
            "-m", "ndcg_cut.10",
            "-m", "map",
            "-m", "recip_rank",
            "-m", "recall.100",
            str(QRELS_PATH),
            str(run_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")


# ============ Main ============
if __name__ == "__main__":
    if (INDEX_DIR / "bm25_standard").exists():
        run = run_bm25(INDEX_DIR / "bm25_standard", "bm25_standard")
        evaluate(run)

    if (INDEX_DIR / "bm25_contextualized").exists():
        run = run_bm25(INDEX_DIR / "bm25_contextualized", "bm25_contextualized")
        evaluate(run)

    dense_std_index = INDEX_DIR / "dense_standard" / "index"
    if dense_std_index.exists():
        run = run_dense(dense_std_index, "dense_standard")
        evaluate(run)

    dense_ctx_index = INDEX_DIR / "dense_contextualized" / "index"
    if dense_ctx_index.exists():
        run = run_dense(dense_ctx_index, "dense_contextualized")
        evaluate(run)

    print("\n✓ All runs saved to:", RUNS_DIR)
