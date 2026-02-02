#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
QRELS = DATA_DIR / "qrels.umbrela.rag24.test.txt"

OUT_REL_SEGS = DATA_DIR / "relevant_segment_ids.txt"       # full seg ids (keep #...)
OUT_REL_DOCS = DATA_DIR / "relevant_doc_ids.txt"           # parent doc ids (strip #...)
OUT_TOPIC_TO_REL_SEGS = DATA_DIR / "topic_to_relsegs.json" # for analysis/debugging

topic_to_relsegs = defaultdict(set)
rel_segs = set()
rel_docs = set()

with QRELS.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        topic, _, segid, rel = line.split()
        if int(rel) > 0:
            topic_to_relsegs[topic].add(segid)
            rel_segs.add(segid)
            rel_docs.add(segid.split("#", 1)[0])

# write outputs
OUT_REL_SEGS.write_text("\n".join(sorted(rel_segs)) + "\n", encoding="utf-8")
OUT_REL_DOCS.write_text("\n".join(sorted(rel_docs)) + "\n", encoding="utf-8")
OUT_TOPIC_TO_REL_SEGS.write_text(
    json.dumps({t: sorted(list(s)) for t, s in topic_to_relsegs.items()}, indent=2),
    encoding="utf-8",
)

print("Topics with >=1 relevant segment:", len(topic_to_relsegs))
print("Unique relevant segments:", len(rel_segs))
print("Unique docs touched by relevance:", len(rel_docs))
print("Wrote:", OUT_REL_SEGS)
print("Wrote:", OUT_REL_DOCS)
print("Wrote:", OUT_TOPIC_TO_REL_SEGS)
