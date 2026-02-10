#!/usr/bin/env python3
"""
Extract segments from the tar into the mini corpus and write filtered qrels/topics.

If data/mini_ids.txt exists (from mine_distractors.py): use those segment IDs
(positives + BM25 hard negatives). Otherwise: take first TARGET_SEGMENTS relevant
segment IDs from qrels (rel > 0).

- Filtered qrels: only (topic, segid, rel) with rel > 0 and segid in the mini corpus.
- Mini-corpus is either fair (P + distractors) or positives-only depending on input.
"""
import os
import sys
import tarfile
import gzip
import json
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

CORPUS_TAR = DATA_DIR / "msmarco_v2.1_doc_segmented.tar"
QRELS_PATH = DATA_DIR / "qrels.umbrela.rag24.test.txt"
TOPICS_PATH = DATA_DIR / "topics.rag24.test.txt"
MINI_IDS = DATA_DIR / "mini_ids.txt"
OUT_MINI = DATA_DIR / "mini_corpus_all_segments.jsonl"
OUT_FILTERED_QRELS = DATA_DIR / "qrels_filtered.txt"
OUT_FILTERED_TOPICS = DATA_DIR / "topics_filtered.txt"

if not CORPUS_TAR.exists():
    print(f"Error: corpus not found: {CORPUS_TAR}", file=sys.stderr)
    print("Download the MS MARCO V2.1 segmented corpus (~25 GB) and place it in data/:", file=sys.stderr)
    sys.exit(1)

target_segments = int(os.getenv("TARGET_SEGMENTS", "5000"))


def segment_id_from_record(obj: dict) -> str:
    v = obj.get("docid")
    if isinstance(v, str) and v.strip():
        return v.strip()
    for k in ("id", "segment_id"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ============ Step 1: Wanted segment IDs (from mini_ids.txt or from qrels) ============
if MINI_IDS.exists():
    wanted_segids = set()
    with MINI_IDS.open("r", encoding="utf-8") as f:
        for line in f:
            segid = line.strip()
            if segid:
                wanted_segids.add(segid)
    print(f"Wanted segment IDs (from {MINI_IDS}): {len(wanted_segids)}")
else:
    wanted_segids = set()
    with QRELS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic, q0, segid, rel = parts[0], parts[1], parts[2], parts[3]
            if int(rel) > 0 and segid not in wanted_segids:
                wanted_segids.add(segid)
                if len(wanted_segids) >= target_segments:
                    break
    print(f"Target relevant segment IDs (from qrels): {len(wanted_segids)}")

# ============ Step 2: Extract from tar; one record per segment id (no duplicates) ============
segids_in_corpus = set()
with tarfile.open(CORPUS_TAR, "r") as tf, OUT_MINI.open("w", encoding="utf-8") as out:
    for member in tqdm(tf, desc="Tar files", unit="file"):
        if not member.isfile() or not member.name.endswith(".json.gz"):
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        with gzip.open(f, "rt", encoding="utf-8") as gzf:
            for line in gzf:
                if len(segids_in_corpus) >= len(wanted_segids):
                    break
                obj = json.loads(line)
                segid = segment_id_from_record(obj)
                if segid and segid in wanted_segids and segid not in segids_in_corpus:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    segids_in_corpus.add(segid)
        if len(segids_in_corpus) >= len(wanted_segids):
            break

print(f"Kept segments: {len(segids_in_corpus)} unique")
print(f"Wrote: {OUT_MINI}")

# ============ Step 3: Filtered qrels — only rel > 0 and segid in corpus ============
selected_topics = set()
with QRELS_PATH.open("r", encoding="utf-8") as f_in, OUT_FILTERED_QRELS.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        topic, q0, segid, rel = parts[0], parts[1], parts[2], parts[3]
        if segid in segids_in_corpus and int(rel) > 0:
            f_out.write(f"{topic} {q0} {segid} {rel}\n")
            selected_topics.add(topic)
print(f"Wrote: {OUT_FILTERED_QRELS} (topics with ≥1 relevant segment in corpus: {len(selected_topics)})")

# ============ Step 4: Filtered topics ============
with TOPICS_PATH.open("r", encoding="utf-8") as f_in, OUT_FILTERED_TOPICS.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        parts = line.strip().split("\t", 1)
        if parts and parts[0] in selected_topics:
            f_out.write(line)
print(f"Wrote: {OUT_FILTERED_TOPICS}")
