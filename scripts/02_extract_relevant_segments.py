#!/usr/bin/env python3
"""
Select topics from qrels until we reach ~TARGET_DOCS documents.
Guarantees 100% relevant doc coverage for each selected topic.
"""
import os
import tarfile, gzip, json
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

CORPUS_TAR = DATA_DIR / "msmarco_v2.1_doc_segmented.tar"
QRELS_PATH = DATA_DIR / "qrels.umbrela.rag24.test.txt"
TOPICS_PATH = DATA_DIR / "topics.rag24.test.txt"
OUT_MINI = DATA_DIR / "mini_corpus_all_segments.jsonl"
OUT_FILTERED_QRELS = DATA_DIR / "qrels_filtered.txt"
OUT_FILTERED_TOPICS = DATA_DIR / "topics_filtered.txt"

target_docs = int(os.getenv("TARGET_DOCS", "5000"))

# ============ Step 1: Parse qrels, group by topic ============
qrels_by_topic = defaultdict(list)  # topic -> [(q0, segid, rel), ...]
docs_by_topic = defaultdict(set)    # topic -> set of parent doc_ids (rel > 0)

with QRELS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        topic, q0, segid, rel = parts[0], parts[1], parts[2], parts[3]
        qrels_by_topic[topic].append((q0, segid, rel))
        if int(rel) > 0:
            docs_by_topic[topic].add(segid.split("#", 1)[0])

all_topics = sorted(qrels_by_topic.keys())
print(f"Total topics in qrels: {len(all_topics)}")

# ============ Step 2: Select topics until we reach target_docs ============
selected_topics = []
selected_docs = set()

for topic in all_topics:
    topic_docs = docs_by_topic[topic]
    new_docs = topic_docs - selected_docs
    
    # Add this topic
    selected_topics.append(topic)
    selected_docs.update(topic_docs)
    
    if len(selected_docs) >= target_docs:
        break

print(f"Selected topics: {len(selected_topics)}")
print(f"Selected docs: {len(selected_docs)}")

# ============ Step 3: Write filtered qrels and topics ============
selected_topics_set = set(selected_topics)

with OUT_FILTERED_QRELS.open("w", encoding="utf-8") as f:
    for topic in selected_topics:
        for q0, segid, rel in qrels_by_topic[topic]:
            f.write(f"{topic} {q0} {segid} {rel}\n")
print(f"Wrote: {OUT_FILTERED_QRELS}")

# Filter topics file (format: topic_id<TAB>query_text)
with TOPICS_PATH.open("r", encoding="utf-8") as f_in, \
     OUT_FILTERED_TOPICS.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        parts = line.strip().split("\t", 1)
        if parts and parts[0] in selected_topics_set:
            f_out.write(line)
print(f"Wrote: {OUT_FILTERED_TOPICS}")

# ============ Step 4: Extract segments from tar ============
kept = 0

def parent_docid_from_record(obj: dict) -> str:
    for k in ("id", "segment_id", "docid"):
        v = obj.get(k)
        if isinstance(v, str) and "#" in v:
            return v.split("#", 1)[0]
    for k in ("doc_id", "document_id", "parent_docid"):
        v = obj.get(k)
        if isinstance(v, str) and v.startswith("msmarco"):
            return v
    return ""

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
                docid = parent_docid_from_record(obj)
                if docid and docid in selected_docs:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1

print(f"Kept segments: {kept}")
print(f"Wrote: {OUT_MINI}")
