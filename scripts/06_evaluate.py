#!/usr/bin/env python3
"""
Run retrieval on all indices and evaluate using pyserini.eval.trec_eval.
"""
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "indices"
RUNS_DIR = ROOT_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

TOPICS_PATH = DATA_DIR / "topics_filtered.txt"
QRELS_PATH = DATA_DIR / "qrels_filtered.txt"

TOP_K = 100

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
            hits = searcher.search(query, k=TOP_K)
            for rank, hit in enumerate(hits, 1):
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {run_name}\n")
    
    print(f"Wrote: {run_file}")
    return run_file

# ============ Dense Retrieval (Contriever: facebook/contriever with mean pooling) ============
def run_dense(index_path: Path, encoder: str, run_name: str):
    from pyserini.search.faiss import FaissSearcher
    from pyserini.encode import AutoQueryEncoder
    
    # Contriever uses same model for query and document (AutoQueryEncoder with facebook/contriever)
    query_encoder = AutoQueryEncoder(encoder)
    searcher = FaissSearcher(str(index_path), query_encoder)
    
    run_file = RUNS_DIR / f"{run_name}.txt"
    with run_file.open("w", encoding="utf-8") as f:
        for qid, query in topics.items():
            hits = searcher.search(query, k=TOP_K)
            for rank, hit in enumerate(hits, 1):
                f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {run_name}\n")
    
    print(f"Wrote: {run_file}")
    return run_file

# ============ Evaluate with pyserini.eval.trec_eval ============
def evaluate(run_file: Path):
    print(f"\n=== {run_file.stem} ===")
    if not run_file.exists() or run_file.stat().st_size == 0:
        print("(skip: run file empty or missing)")
        return
    # Run pyserini's trec_eval
    result = subprocess.run(
        [
            "python", "-m", "pyserini.eval.trec_eval",
            "-c", "-M", "100",
            "-m", "ndcg_cut.10",
            "-m", "map",
            "-m", "recip_rank",
            "-m", "recall.100",
            str(QRELS_PATH),
            str(run_file)
        ],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")

# ============ Main ============
if __name__ == "__main__":
    # BM25 Standard
    if (INDEX_DIR / "bm25_standard").exists():
        run = run_bm25(INDEX_DIR / "bm25_standard", "bm25_standard")
        evaluate(run)
    
    # BM25 Contextualized
    if (INDEX_DIR / "bm25_contextualized").exists():
        run = run_bm25(INDEX_DIR / "bm25_contextualized", "bm25_contextualized")
        evaluate(run)
    
    # Dense Standard (Contriever)
    dense_std_index = INDEX_DIR / "dense_standard" / "index"
    if dense_std_index.exists():
        run = run_dense(
            dense_std_index,
            "facebook/contriever",
            "dense_standard"
        )
        evaluate(run)
    
    # Dense Contextualized (Contriever)
    dense_ctx_index = INDEX_DIR / "dense_contextualized" / "index"
    if dense_ctx_index.exists():
        run = run_dense(
            dense_ctx_index,
            "facebook/contriever",
            "dense_contextualized"
        )
        evaluate(run)
    
    print("\n✓ All runs saved to:", RUNS_DIR)
