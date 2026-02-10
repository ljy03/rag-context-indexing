#!/usr/bin/env python3
"""
Build a fair mini-corpus: positives (rel>0) + BM25 hard negatives per topic.

Strategy:
  A) From qrels: P(qid) = relevant segment IDs per topic.
  B) Build BM25 on a slice of the segmented corpus; for each topic run BM25 top K,
     take up to D_PER_TOPIC distractors (not in P, max MAX_PER_DOC per parent doc).
  C) MiniCorpus IDs = union of all P(qid) ∪ all D(qid).
  D) Write mini_ids.txt for extract_segments.py to use.

Requires: qrels, topics, and msmarco_v2.1_doc_segmented.tar (to build slice).
"""
import json
import os
import random
import subprocess
import sys
import tarfile
import gzip
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "indices"

CORPUS_TAR = DATA_DIR / "msmarco_v2.1_doc_segmented.tar"
QRELS_PATH = DATA_DIR / "qrels.umbrela.rag24.test.txt"
TOPICS_PATH = DATA_DIR / "topics.rag24.test.txt"
SLICE_CORPUS = DATA_DIR / "slice_corpus.jsonl"
MINI_IDS = DATA_DIR / "mini_ids.txt"
BM25_SLICE_INDEX = INDEX_DIR / "bm25_slice"

# Corpus size caps and distractor controls (override with env vars):
#   P_PER_TOPIC       max positive segment IDs per topic; sampled if over (default 10)
#   MAX_DISTRACTORS   cap on total unique distractors (default 5000)
#   D_PER_TOPIC       max distractors per topic (default 100)
#   BM25_K            BM25 overfetch depth per topic (default 2000)
#   MAX_PER_DOC_DISTRACTORS  max segments per parent doc when picking distractors (default 2)
#   SLICE_SIZE        segments in slice corpus for BM25 (default 100000)
SLICE_SIZE = int(os.getenv("SLICE_SIZE", "100000"))
P_PER_TOPIC = int(os.getenv("P_PER_TOPIC", "10"))
MAX_DISTRACTORS = int(os.getenv("MAX_DISTRACTORS", "5000"))
D_PER_TOPIC = int(os.getenv("D_PER_TOPIC", "100"))
BM25_K = int(os.getenv("BM25_K", "2000"))
MAX_PER_DOC = int(os.getenv("MAX_PER_DOC_DISTRACTORS", "2"))


def parent_docid(segid: str) -> str:
    return segid.split("#", 1)[0] if "#" in segid else segid


def segment_id_from_record(obj: dict) -> str:
    v = obj.get("docid")
    if isinstance(v, str) and v.strip():
        return v.strip()
    for k in ("id", "segment_id"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main():
    if not CORPUS_TAR.exists():
        print(f"Error: corpus not found: {CORPUS_TAR}", file=sys.stderr)
        sys.exit(1)
    if not QRELS_PATH.exists() or not TOPICS_PATH.exists():
        print("Error: need qrels and topics (e.g. qrels.umbrela.rag24.test.txt, topics.rag24.test.txt)", file=sys.stderr)
        sys.exit(1)

    # --- Step A: P(qid) = relevant segment IDs per topic (full); then cap per topic with P_PER_TOPIC ---
    positives_per_topic = {}
    with QRELS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic, _, segid, rel = parts[0], parts[1], parts[2], parts[3]
            if int(rel) <= 0:
                continue
            if topic not in positives_per_topic:
                positives_per_topic[topic] = set()
            positives_per_topic[topic].add(segid)

    # --- Load topics ---
    topic_to_query = {}
    with TOPICS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                topic_to_query[parts[0]] = parts[1]
    topics_to_mine = sorted(t for t in topic_to_query if t in positives_per_topic)
    print(f"Topics with positives: {len(topics_to_mine)}")

    # Cap positives per topic: sample P_PER_TOPIC per topic (deterministic, balanced)
    random.seed(0)
    all_positives = set()
    for topic in topics_to_mine:
        pos_list = sorted(positives_per_topic[topic])
        if len(pos_list) > P_PER_TOPIC:
            pos_list = random.sample(pos_list, P_PER_TOPIC)
        all_positives.update(pos_list)
    print(f"Positives: {len(all_positives)} (P_PER_TOPIC={P_PER_TOPIC}) across {len(topics_to_mine)} topics")
    print(f"Distractor settings: MAX_DISTRACTORS={MAX_DISTRACTORS}, D_PER_TOPIC={D_PER_TOPIC}, BM25_K={BM25_K}, MAX_PER_DOC={MAX_PER_DOC}")

    # --- Step B (slice): Build slice corpus from tar if missing ---
    # Note: slice is built from first SLICE_SIZE segments in tar order; many qrels positives
    # may not appear in this slice, so BM25 negatives are an approximation (can be easier/off-distribution).
    if not SLICE_CORPUS.exists() or os.path.getsize(SLICE_CORPUS) == 0:
        print(f"Building slice corpus ({SLICE_SIZE} segments) from tar...")
        n = 0
        printed_keys = False
        with tarfile.open(CORPUS_TAR, "r") as tf, SLICE_CORPUS.open("w", encoding="utf-8") as out:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".json.gz"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                with gzip.open(f, "rt", encoding="utf-8") as gzf:
                    for line in gzf:
                        if n >= SLICE_SIZE:
                            break
                        obj = json.loads(line)
                        if not printed_keys:
                            print(f"First corpus record keys: {list(obj.keys())}")
                            printed_keys = True
                        segid = segment_id_from_record(obj)
                        seg_text = obj.get("segment") or obj.get("contents") or obj.get("text") or ""
                        if isinstance(seg_text, list):
                            seg_text = "\n".join(str(x) for x in seg_text)
                        if not segid or not str(seg_text).strip():
                            continue
                        out.write(json.dumps({"id": segid, "contents": seg_text.strip()}, ensure_ascii=False) + "\n")
                        n += 1
                if n >= SLICE_SIZE:
                    break
        print(f"Wrote slice: {SLICE_CORPUS} ({n} segments)")
    else:
        print(f"Using existing slice: {SLICE_CORPUS}")

    # --- Build BM25 index on slice ---
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = BM25_SLICE_INDEX / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    index_target = BM25_SLICE_INDEX
    # Pyserini writes Lucene index under index/; avoid treating corpus/ or leftovers as ready
    index_ready = (index_target / "index").exists()
    if not index_ready:
        try:
            (corpus_dir / "corpus.jsonl").unlink(missing_ok=True)
        except Exception:
            pass
        try:
            (corpus_dir / "corpus.jsonl").symlink_to(SLICE_CORPUS.resolve())
        except OSError:
            import shutil
            shutil.copy(SLICE_CORPUS, corpus_dir / "corpus.jsonl")
        print("Building BM25 index on slice...")
        subprocess.run(
            [
                sys.executable, "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", str(corpus_dir),
                "--index", str(index_target),
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", os.getenv("THREADS", "8"),
                "--storePositions", "--storeDocvectors", "--storeRaw",
            ],
            check=True,
        )
    else:
        print(f"Using existing BM25 slice index: {BM25_SLICE_INDEX}")

    # --- Mine distractors per topic ---
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(BM25_SLICE_INDEX))
    searcher.set_bm25(k1=0.9, b=0.4)

    all_distractors = set()
    for topic in topics_to_mine:
        if len(all_distractors) >= MAX_DISTRACTORS:
            break
        query = topic_to_query.get(topic, "")
        if not query:
            continue
        P_q = positives_per_topic.get(topic, set())
        hits = searcher.search(query, k=BM25_K)
        doc_count = {}
        added_this_topic = 0
        for hit in hits:
            if len(all_distractors) >= MAX_DISTRACTORS or added_this_topic >= D_PER_TOPIC:
                break
            if hit.docid in P_q or hit.docid in all_positives:
                continue
            doc = parent_docid(hit.docid)
            if doc_count.get(doc, 0) >= MAX_PER_DOC:
                continue
            all_distractors.add(hit.docid)
            doc_count[doc] = doc_count.get(doc, 0) + 1
            added_this_topic += 1

    wanted_ids = all_positives | all_distractors
    print(f"Distractors: {len(all_distractors)} unique; mini corpus size: {len(wanted_ids)}")

    # --- Write mini_ids.txt for extract_segments.py ---
    with MINI_IDS.open("w", encoding="utf-8") as f:
        for segid in sorted(wanted_ids):
            f.write(segid + "\n")
    print(f"Wrote: {MINI_IDS}")


if __name__ == "__main__":
    main()
