#!/usr/bin/env python3
"""
Take up to TARGET_SEGMENTS relevant segment IDs from qrels (rel > 0), extract those
segments from the tar, then write filtered qrels/topics only for segments that are
actually in the mini corpus (so qrels never reference segments outside the corpus).
"""
import os
import tarfile
import gzip
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

CORPUS_TAR = DATA_DIR / "msmarco_v2.1_doc_segmented.tar"
QRELS_PATH = DATA_DIR / "qrels.umbrela.rag24.test.txt"
TOPICS_PATH = DATA_DIR / "topics.rag24.test.txt"
OUT_MINI = DATA_DIR / "mini_corpus_all_segments.jsonl"
OUT_FILTERED_QRELS = DATA_DIR / "qrels_filtered.txt"
OUT_FILTERED_TOPICS = DATA_DIR / "topics_filtered.txt"

target_segments = int(os.getenv("TARGET_SEGMENTS", "5000"))


def segment_id_from_record(obj: dict) -> str:
    for k in ("id", "segment_id", "docid"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ============ Step 1: From qrels, take up to target_segments relevant segment IDs ============
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
print(f"Target relevant segment IDs: {len(wanted_segids)}")

# ============ Step 2: Extract from tar; track which segment IDs we actually put in the corpus ============
segids_in_corpus = set()
kept = 0
with tarfile.open(CORPUS_TAR, "r") as tf, OUT_MINI.open("w", encoding="utf-8") as out:
    for member in tf:
        if not member.isfile() or not member.name.endswith(".json.gz"):
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        with gzip.open(f, "rt", encoding="utf-8") as gzf:
            for line in gzf:
                obj = json.loads(line)
                segid = segment_id_from_record(obj)
                if segid and segid in wanted_segids:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    segids_in_corpus.add(segid)
                    kept += 1
                if kept >= target_segments:
                    break
        if kept >= target_segments:
            break

print(f"Kept segments: {kept} (in corpus: {len(segids_in_corpus)} unique)")
print(f"Wrote: {OUT_MINI}")

# ============ Step 3: Filtered qrels — only rows whose segid is actually in the mini corpus ============
selected_topics = set()
with QRELS_PATH.open("r", encoding="utf-8") as f_in, OUT_FILTERED_QRELS.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        topic, q0, segid, rel = parts[0], parts[1], parts[2], parts[3]
        if segid in segids_in_corpus:
            f_out.write(f"{topic} {q0} {segid} {rel}\n")
            selected_topics.add(topic)
print(f"Wrote: {OUT_FILTERED_QRELS} (topics with ≥1 segment in corpus: {len(selected_topics)})")

# ============ Step 4: Filtered topics ============
with TOPICS_PATH.open("r", encoding="utf-8") as f_in, OUT_FILTERED_TOPICS.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        parts = line.strip().split("\t", 1)
        if parts and parts[0] in selected_topics:
            f_out.write(line)
print(f"Wrote: {OUT_FILTERED_TOPICS}")
